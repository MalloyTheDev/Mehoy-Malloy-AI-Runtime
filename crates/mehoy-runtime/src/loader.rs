//! Turning a registered artifact into a live instance.
//!
//! # Startup is a transaction
//!
//! Every stage before the instance exists can fail, and a failure at any of them
//! must leave nothing behind: no running worker, no credential file, no half-built
//! instance. The stages are prepare, spawn, verify, commit, and only the last one
//! publishes anything.
//!
//! The hardest case is a failure *after* the backend is healthy and authenticated,
//! because at that point a process is running and holding memory. That path is
//! exercised deliberately rather than assumed, through a fault the tests can inject.
//!
//! # What readiness means here
//!
//! An instance reaching [`InstanceState::BackendReady`] means the backend loaded
//! the artifact and is accepting authenticated requests. It does not mean any
//! particular kind of request will succeed. Capability claims are tracked
//! separately and start out mostly unknown.

use std::fmt;

use mehoy_backend_llama::{
    BackendChannel, BackendError, BackendIdentity, LlamaCppBackend, LlamaCppWorkerSpec,
    ModelDescriptor, RunningBackend,
};
use mehoy_core::cancel::{
    CancellationCause, CancellationStrategy, RequestBudget, RequestHandle, RequestState,
    UnloadBudget,
};
use mehoy_core::id::{IdAllocator, RequestId, WorkerId};
use mehoy_core::inference::{
    self, EmbedRequest, EmbeddingResult, GenerateTextRequest, GenerationResult, GenerationStream,
};
use mehoy_core::worker::Deadlines;
use mehoy_registry::{ArtifactId, ArtifactRegistry, ArtifactState, ModelArtifact, RegistryError};

use crate::capability::{ModelCapability, indicated_by_metadata};
use crate::instance::{InstanceId, ModelInstance};

/// Why an artifact could not be brought up.
#[derive(Debug)]
pub enum LoadError {
    /// The identifier is not registered.
    ///
    /// Only registered artifacts can be started. Accepting a bare path here would
    /// bypass the validation and provenance registration exists to establish.
    UnknownArtifact { id: ArtifactId },
    /// The file behind the artifact is gone.
    ArtifactMissing { id: ArtifactId, path: String },
    /// The file behind the artifact is not the one that was registered.
    ArtifactChanged { id: ArtifactId, reason: String },
    /// The backend will not run this artifact.
    Incompatible { reason: String },
    /// Starting the backend failed.
    Backend(BackendError),
    /// The registry could not be consulted.
    Registry(RegistryError),
    /// The instance could not be constructed after the backend was already up.
    ///
    /// The backend is stopped before this is returned.
    Instance { reason: String },
    /// The instance is not in a state that can serve requests.
    NotUsable { state: String },
    /// The work was stopped after being admitted.
    ///
    /// Distinct from a refusal: this request was accepted, started, and then told
    /// to stop, most often because the instance serving it was unloaded.
    Stopped { cause: CancellationCause },
    /// The request itself cannot be honoured.
    ///
    /// Refused before anything reaches a backend, so an unusable parameter reads as
    /// a caller error rather than as an engine failure.
    InvalidRequest { detail: String },
    /// The backend refused or could not serve the request.
    ///
    /// Carries the backend's own account rather than flattening it, so an
    /// unsupported mode, a refused credential, and a rejected input stay
    /// distinguishable.
    Inference { detail: String },
    /// The backend answered, but with something that cannot be used.
    UnusableResult { detail: String },
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownArtifact { id } => write!(f, "no artifact registered with id {id}"),
            Self::ArtifactMissing { id, path } => {
                write!(f, "artifact {id} is registered but {path} no longer exists")
            }
            Self::ArtifactChanged { id, reason } => write!(
                f,
                "artifact {id} has changed since it was registered ({reason}); \
                 re-register it rather than loading something that was never inspected"
            ),
            Self::Incompatible { reason } => {
                write!(f, "the backend cannot run this artifact: {reason}")
            }
            Self::Backend(err) => write!(f, "{err}"),
            Self::Registry(err) => write!(f, "{err}"),
            Self::Instance { reason } => {
                write!(
                    f,
                    "the backend started but the instance could not be created: {reason}"
                )
            }
            Self::NotUsable { state } => {
                write!(f, "the instance cannot serve requests while it is {state}")
            }
            Self::InvalidRequest { detail } => {
                write!(f, "the request cannot be honoured: {detail}")
            }
            Self::Stopped { cause } => write!(f, "the request was stopped: {cause}"),
            Self::Inference { detail } => write!(f, "{detail}"),
            Self::UnusableResult { detail } => {
                write!(
                    f,
                    "the backend returned a result that cannot be used: {detail}"
                )
            }
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Backend(err) => Some(err),
            Self::Registry(err) => Some(err),
            _ => None,
        }
    }
}

impl From<RegistryError> for LoadError {
    fn from(err: RegistryError) -> Self {
        Self::Registry(err)
    }
}

/// A request the runtime has accepted, and the stream that observes it.
///
/// The identifier is separate from the stream on purpose. Cancelling needs only
/// the identifier, so a caller can hand the stream elsewhere, or drop it, and
/// still stop the work.
#[derive(Debug)]
pub struct GenerationRequest {
    pub request_id: RequestId,
    pub stream: GenerationStream,
}

/// What asking to stop a request achieved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelOutcome {
    /// The request was running and has now been told to stop.
    ///
    /// Not a claim that it has stopped. The stream reports that when it happens.
    Requested,
    /// The request was already stopping when this call arrived.
    AlreadyStopping { cause: CancellationCause },
    /// The request had already ended.
    AlreadyFinished { state: RequestState },
}

impl fmt::Display for CancelOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Requested => f.write_str("the request has been asked to stop"),
            Self::AlreadyStopping { cause } => {
                write!(f, "the request was already stopping ({cause})")
            }
            Self::AlreadyFinished { state } => write!(f, "the request had already {state}"),
        }
    }
}

/// Why a request could not be cancelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelError {
    /// This instance has no record of the request.
    ///
    /// Either it never existed here, or it ended long enough ago to have been
    /// forgotten. The two are not distinguished, because the runtime does not keep
    /// a permanent history of every request it has served.
    Unknown { request_id: RequestId },
}

impl fmt::Display for CancelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown { request_id } => {
                write!(f, "no request {request_id} is running on this instance")
            }
        }
    }
}

impl std::error::Error for CancelError {}

/// Waits for every handle to reach a terminal state, returning how many did not.
///
/// Polled rather than awaited on a signal because the handles come from anywhere
/// and a request may end through a path this function knows nothing about. The
/// interval is short relative to any sensible drain budget.
async fn drain(handles: &[RequestHandle], budget: std::time::Duration) -> usize {
    let deadline = std::time::Instant::now() + budget;
    loop {
        let unsettled = handles
            .iter()
            .filter(|handle| !handle.is_terminal())
            .count();
        if unsettled == 0 {
            return 0;
        }
        if std::time::Instant::now() >= deadline {
            return unsettled;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
}

/// A running instance together with the backend serving it.
#[derive(Debug)]
pub struct LoadedModel {
    instance: ModelInstance,
    /// Where to reach the backend.
    ///
    /// Held separately from the worker so that starting a request never has to
    /// take the lifecycle lock for longer than the admission decision itself.
    channel: BackendChannel,
    identity: Option<BackendIdentity>,
    /// Identifies each inference request this instance serves.
    ///
    /// Owned here because nothing above the loader allocates one yet. When a
    /// request pipeline exists, identity will arrive with the request instead.
    requests: IdAllocator,
    /// Admission, in-flight work, and the worker, under one lock.
    ///
    /// One lock rather than three because the decision that matters is atomic:
    /// whether this instance is still admitting work, and if so registering the
    /// request, must happen together. Checking and then registering separately
    /// admits the classic race in which a request is accepted moments after the
    /// instance stopped accepting any.
    lifecycle: std::sync::Mutex<Lifecycle>,
}

/// Everything an instance's lifetime turns on.
#[derive(Debug)]
struct Lifecycle {
    phase: LifecyclePhase,
    active: std::collections::HashMap<RequestId, RequestHandle>,
    /// Taken out when unloading begins, so a second attempt finds it gone.
    worker: Option<RunningBackend>,
}

/// Whether an instance is still serving.
///
/// Distinct from [`InstanceState`], which describes whether the backend came up.
/// This describes whether the runtime is still willing to put work on it, and it
/// is the authority for that decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecyclePhase {
    /// Accepting requests.
    Serving,
    /// Not accepting requests; existing ones are being stopped.
    Unloading,
    /// The worker is gone and its resources are released.
    Unloaded,
    /// Teardown did not complete, so what remains is unknown.
    ///
    /// Deliberately not reported as unloaded. Something may still be running, and
    /// saying otherwise would be the more dangerous of the two wrong answers.
    UnloadFailed,
}

impl fmt::Display for LifecyclePhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Serving => "serving",
            Self::Unloading => "unloading",
            Self::Unloaded => "unloaded",
            Self::UnloadFailed => "failed to unload",
        })
    }
}

/// What unloading did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnloadOutcome {
    /// Every outstanding request settled before the worker was stopped.
    Drained { cancelled: usize },
    /// The drain budget expired, so teardown went ahead anyway.
    ///
    /// Not a failure. Terminating the worker is what ends the work for certain,
    /// and it is the reason the budget can be allowed to expire at all.
    Escalated { cancelled: usize, unsettled: usize },
    /// Another unload was already in progress.
    AlreadyUnloading,
    /// This instance had already been unloaded.
    AlreadyUnloaded,
}

impl fmt::Display for UnloadOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Drained { cancelled } => {
                write!(f, "unloaded after {cancelled} request(s) settled")
            }
            Self::Escalated {
                cancelled,
                unsettled,
            } => write!(
                f,
                "unloaded with {unsettled} of {cancelled} request(s) still unsettled"
            ),
            Self::AlreadyUnloading => f.write_str("an unload was already in progress"),
            Self::AlreadyUnloaded => f.write_str("the instance was already unloaded"),
        }
    }
}

impl LoadedModel {
    #[must_use]
    pub fn instance(&self) -> &ModelInstance {
        &self.instance
    }

    pub fn instance_mut(&mut self) -> &mut ModelInstance {
        &mut self.instance
    }

    /// The private channel to the backend serving this model.
    #[must_use]
    pub const fn channel(&self) -> &BackendChannel {
        &self.channel
    }

    /// Whether this instance is still serving, being torn down, or gone.
    #[must_use]
    pub fn phase(&self) -> LifecyclePhase {
        self.lifecycle().phase
    }

    /// Admits one unit of backend work, or refuses it.
    ///
    /// Every task class goes through this, and nothing may reach a backend without
    /// it. A task that checked readiness for itself and then called the backend
    /// would be invisible to unloading, so it would neither be refused once the
    /// instance stopped serving nor stopped when the instance was torn down. That
    /// is a per-task lifecycle rather than a runtime one, and it gets less correct
    /// with every task class added.
    ///
    /// The check and the registration are one operation under one lock, so a
    /// request cannot be admitted moments after admission closed.
    fn admit(&self) -> Result<RequestHandle, LoadError> {
        let mut lifecycle = self.lifecycle();
        if lifecycle.phase != LifecyclePhase::Serving {
            return Err(LoadError::NotUsable {
                state: lifecycle.phase.to_string(),
            });
        }
        if !self.instance.state().is_usable() {
            return Err(LoadError::NotUsable {
                state: self.instance.state().to_string(),
            });
        }
        lifecycle
            .active
            .retain(|_, existing| !existing.is_terminal());
        let handle = RequestHandle::new(self.requests.request());
        lifecycle.active.insert(handle.id(), handle.clone());
        Ok(handle)
    }

    /// Performs an embedding request against this instance.
    ///
    /// Structural validation happens before anything is returned: the number of
    /// vectors must match the number of inputs, every vector must be non-empty with
    /// finite components, and the dimensions must agree. A vector containing a
    /// non-finite value is not a lesser problem than an error response, because it
    /// silently poisons every distance computed from it.
    ///
    /// # Errors
    ///
    /// Returns [`LoadError::NotUsable`] when the instance is not ready,
    /// [`LoadError::Inference`] when the backend refuses, and
    /// [`LoadError::UnusableResult`] when the response is structurally wrong.
    pub async fn embed(&self, request: &EmbedRequest) -> Result<EmbeddingResult, LoadError> {
        let handle = self.admit()?;
        handle.mark_running();

        let sent = tokio::select! {
            biased;
            cause = handle.token().cancelled() => {
                handle.finish(RequestState::Cancelled);
                return Err(LoadError::Stopped { cause });
            }
            sent = mehoy_backend_llama::embed(&self.channel, request) => sent,
        };

        let result = match sent {
            Ok(result) => result,
            Err(source) => {
                handle.finish(RequestState::Failed);
                return Err(LoadError::Inference {
                    detail: source.to_string(),
                });
            }
        };

        if let Err(defect) = inference::validate(request, &result) {
            handle.finish(RequestState::Failed);
            return Err(LoadError::UnusableResult {
                detail: defect.to_string(),
            });
        }
        handle.finish(RequestState::Completed);

        Ok(result)
    }

    /// Performs an embedding request and, if it succeeds, records that this backend
    /// demonstrated the capability.
    ///
    /// The only route by which [`ModelCapability::Embeddings`] becomes verified.
    /// Nothing derived from metadata reaches it, and a failed request leaves the
    /// capability exactly where it was rather than downgrading it: a transient
    /// refusal is not evidence of absence.
    ///
    /// # Errors
    ///
    /// Returns whatever [`LoadedModel::embed`] returns. The capability is unchanged
    /// on failure.
    pub async fn verify_embeddings(
        &mut self,
        request: &EmbedRequest,
    ) -> Result<EmbeddingResult, LoadError> {
        let result = self.embed(request).await?;
        let backend = self.identity.clone();
        self.instance
            .capabilities_mut()
            .verify(ModelCapability::Embeddings, backend);
        Ok(result)
    }

    /// Continues text using this instance.
    ///
    /// Parameters are validated before anything is sent, so an unusable value is a
    /// clear refusal rather than a backend error, and the result is validated before
    /// it is returned: a successful status carrying no content is not generation.
    ///
    /// # Errors
    ///
    /// Returns [`LoadError::NotUsable`] when the instance is not ready,
    /// [`LoadError::InvalidRequest`] when a parameter cannot be honoured,
    /// [`LoadError::Inference`] when the backend refuses, and
    /// [`LoadError::UnusableResult`] when the response is not usable.
    pub async fn generate_text(
        &self,
        request: &GenerateTextRequest,
    ) -> Result<GenerationResult, LoadError> {
        // Validated before admission, so an unusable request never occupies one.
        request
            .parameters
            .validate()
            .map_err(|invalid| LoadError::InvalidRequest {
                detail: invalid.to_string(),
            })?;

        let handle = self.admit()?;
        handle.mark_running();

        let sent = tokio::select! {
            biased;
            cause = handle.token().cancelled() => {
                handle.finish(RequestState::Cancelled);
                return Err(LoadError::Stopped { cause });
            }
            sent = mehoy_backend_llama::generate(&self.channel, request) => sent,
        };

        let result = match sent {
            Ok(result) => result,
            Err(source) => {
                handle.finish(RequestState::Failed);
                return Err(LoadError::Inference {
                    detail: source.to_string(),
                });
            }
        };

        if let Err(defect) = inference::validate_generation(&result) {
            handle.finish(RequestState::Failed);
            return Err(LoadError::UnusableResult {
                detail: defect.to_string(),
            });
        }
        handle.finish(RequestState::Completed);

        Ok(result)
    }

    /// Generates text and, if it succeeds, records that this backend demonstrated
    /// the capability.
    ///
    /// The only route by which [`ModelCapability::TextGeneration`] becomes verified,
    /// mirroring embeddings exactly. A successful status is not enough: the result
    /// must actually carry content, so a backend returning an empty success cannot
    /// verify anything.
    ///
    /// # Errors
    ///
    /// Returns whatever [`LoadedModel::generate_text`] returns. The capability is
    /// unchanged on failure, because a refusal is not evidence of absence.
    pub async fn verify_text_generation(
        &mut self,
        request: &GenerateTextRequest,
    ) -> Result<GenerationResult, LoadError> {
        let result = self.generate_text(request).await?;
        let backend = self.identity.clone();
        self.instance
            .capabilities_mut()
            .verify(ModelCapability::TextGeneration, backend);
        Ok(result)
    }

    /// Begins continuing text, delivering it incrementally.
    ///
    /// Returns as soon as the runtime owns the request. It does not wait for the
    /// backend, which on a measured engine can spend more than ten seconds reading
    /// a large prompt before answering at all. Waiting for that would leave the
    /// caller holding nothing to cancel during exactly the period when cancelling
    /// matters most. ADR-0009 records the reasoning.
    ///
    /// Because nothing is awaited here, a backend refusal is not reported by this
    /// call. It arrives through the stream, alongside every other way the request
    /// can fail.
    ///
    /// # Errors
    ///
    /// Returns [`LoadError::NotUsable`] when the instance is not ready and
    /// [`LoadError::InvalidRequest`] when a parameter cannot be honoured. Both are
    /// decided before a request is accepted, so neither creates one.
    pub fn generate_stream(
        &self,
        request: &GenerateTextRequest,
    ) -> Result<GenerationRequest, LoadError> {
        self.generate_stream_within(request, RequestBudget::default())
    }

    /// The same, with an explicit budget.
    ///
    /// Separate so the default is one value in one place rather than a constant
    /// repeated at call sites, and so a test can choose a budget it can actually
    /// wait for.
    ///
    /// # Errors
    ///
    /// As [`LoadedModel::generate_stream`].
    pub fn generate_stream_within(
        &self,
        request: &GenerateTextRequest,
        budget: RequestBudget,
    ) -> Result<GenerationRequest, LoadError> {
        // Validated before admission, so an unusable request never occupies one.
        request
            .parameters
            .validate()
            .map_err(|invalid| LoadError::InvalidRequest {
                detail: invalid.to_string(),
            })?;

        let handle = self.admit()?;
        let request_id = handle.id();

        let stream = mehoy_backend_llama::start_generation(&self.channel, request, handle, budget);

        Ok(GenerationRequest { request_id, stream })
    }

    /// Asks a request to stop.
    ///
    /// Returns once the request has been told to stop, which is not the same as
    /// its having stopped. The backend may keep working afterwards, and on the
    /// measured engine it does when it is still reading its input. The request's
    /// stream reports [`inference::GenerationEvent::Cancelled`] when execution has
    /// actually ended.
    ///
    /// Addressed by identity rather than by stream, so a request can be stopped by
    /// something that is not reading it, and so dropping a stream does not silently
    /// mean the same thing.
    ///
    /// # Errors
    ///
    /// Returns [`CancelError::Unknown`] when this instance has no such request.
    /// A request that has already ended is not an error; it is reported as such.
    pub fn cancel(&self, request_id: RequestId) -> Result<CancelOutcome, CancelError> {
        let handle = {
            let mut lifecycle = self.lifecycle();
            lifecycle.active.retain(|_, handle| !handle.is_terminal());
            lifecycle.active.get(&request_id).cloned()
        };

        let Some(handle) = handle else {
            return Err(CancelError::Unknown { request_id });
        };

        if handle.is_terminal() {
            return Ok(CancelOutcome::AlreadyFinished {
                state: handle.state(),
            });
        }

        if handle.request_cancellation(CancellationCause::User) {
            Ok(CancelOutcome::Requested)
        } else if handle.is_terminal() {
            Ok(CancelOutcome::AlreadyFinished {
                state: handle.state(),
            })
        } else {
            Ok(CancelOutcome::AlreadyStopping {
                cause: handle
                    .cancellation_cause()
                    .unwrap_or(CancellationCause::User),
            })
        }
    }

    /// How many accepted requests have not yet been seen to end.
    ///
    /// Ended requests are dropped as they are noticed rather than tracked forever,
    /// so this is a live count and not a total.
    #[must_use]
    pub fn active_requests(&self) -> usize {
        let mut lifecycle = self.lifecycle();
        lifecycle.active.retain(|_, handle| !handle.is_terminal());
        lifecycle.active.len()
    }

    /// The state of one request, if this instance still knows about it.
    #[must_use]
    pub fn request_state(&self, request_id: RequestId) -> Option<RequestState> {
        self.lifecycle()
            .active
            .get(&request_id)
            .map(RequestHandle::state)
    }

    /// How this instance's backend is able to stop work.
    #[must_use]
    pub fn cancellation_strategy(&self) -> CancellationStrategy {
        mehoy_backend_llama::cancellation_strategy()
    }

    fn lifecycle(&self) -> std::sync::MutexGuard<'_, Lifecycle> {
        // Never held across an await. Every slow step of unloading happens with
        // this released, which is what keeps teardown from blocking every other
        // caller for the length of a drain deadline.
        self.lifecycle
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Records what a drained stream demonstrated, promoting the capability if it
    /// earned it.
    ///
    /// Returns whether the capability is now verified by this stream.
    ///
    /// A stream earns it only by reaching a terminal completion after delivering at
    /// least one non-empty delta. An opened stream proves the backend accepted a
    /// request, and a first delta proves it began answering, but neither survives
    /// the backend dying immediately afterwards. Evidence a later event could
    /// contradict is not evidence, and a cancelled request is not evidence either:
    /// it was stopped before the backend said it had finished.
    ///
    /// A stream that failed leaves the capability untouched rather than clearing
    /// it. A refusal is not proof of absence, and this is a record of what has been
    /// demonstrated, not a health check: whether the instance is currently well is
    /// [`ModelInstance::state`].
    pub fn record_generation_stream(&mut self, stream: &GenerationStream) -> bool {
        if !stream.demonstrated_generation() {
            return false;
        }
        let backend = self.identity.clone();
        self.instance
            .capabilities_mut()
            .verify(ModelCapability::TextGeneration, backend);
        true
    }

    /// Stops the backend and destroys the instance.
    ///
    /// # Errors
    ///
    /// Returns an error when the backend had to be killed rather than stopping
    /// politely. The backend is stopped either way.
    pub async fn unload(&self) -> Result<UnloadOutcome, LoadError> {
        self.unload_within(UnloadBudget::default()).await
    }

    /// The same, with an explicit drain budget.
    ///
    /// # Errors
    ///
    /// As [`LoadedModel::unload`].
    pub async fn unload_within(&self, budget: UnloadBudget) -> Result<UnloadOutcome, LoadError> {
        // Phase one, holding the lock: close admission and take ownership of what
        // has to be cleaned up. Nothing slow happens here.
        let (mut worker, outstanding) = {
            let mut lifecycle = self.lifecycle();
            match lifecycle.phase {
                LifecyclePhase::Unloading => return Ok(UnloadOutcome::AlreadyUnloading),
                LifecyclePhase::Unloaded => return Ok(UnloadOutcome::AlreadyUnloaded),
                LifecyclePhase::UnloadFailed | LifecyclePhase::Serving => {}
            }
            lifecycle.phase = LifecyclePhase::Unloading;
            let outstanding: Vec<RequestHandle> = lifecycle
                .active
                .values()
                .filter(|handle| !handle.is_terminal())
                .cloned()
                .collect();
            (lifecycle.worker.take(), outstanding)
        };

        let Some(worker) = worker.take() else {
            // Admission was closed by a previous attempt that then failed. There
            // is no worker left to stop, so finish the bookkeeping.
            self.finish(LifecyclePhase::Unloaded);
            return Ok(UnloadOutcome::AlreadyUnloaded);
        };

        // Phase two, with the lock released: stopping requests, waiting for them,
        // and stopping a process are all slow, and none of them may block another
        // caller for the length of a drain deadline.
        let cancelled = outstanding.len();
        for handle in &outstanding {
            // Through the same mechanism as any other cancellation, so a request
            // has one way to end rather than two that could disagree.
            handle.request_cancellation(CancellationCause::InstanceUnloading);
        }
        let unsettled = drain(&outstanding, budget.drain).await;

        let mut worker = worker;
        let stopped = worker.stop().await;

        // Phase three: record what happened.
        let phase = if stopped.is_ok() {
            LifecyclePhase::Unloaded
        } else {
            LifecyclePhase::UnloadFailed
        };
        self.finish(phase);
        stopped.map_err(LoadError::Backend)?;

        Ok(if unsettled == 0 {
            UnloadOutcome::Drained { cancelled }
        } else {
            UnloadOutcome::Escalated {
                cancelled,
                unsettled,
            }
        })
    }

    fn finish(&self, phase: LifecyclePhase) {
        let mut lifecycle = self.lifecycle();
        lifecycle.phase = phase;
        // Requests do not outlive the instance that was serving them.
        lifecycle.active.clear();
    }
}

/// Brings registered artifacts up on a backend.
#[derive(Debug, Clone)]
pub struct ModelLoader {
    backend: LlamaCppBackend,
}

impl ModelLoader {
    #[must_use]
    pub fn new(backend: LlamaCppBackend) -> Self {
        Self { backend }
    }

    /// Starts a registered artifact and returns the resulting instance.
    ///
    /// # Errors
    ///
    /// Returns [`LoadError`] for an unknown, missing, changed, or incompatible
    /// artifact, or when the backend fails to start. No worker, credential file, or
    /// instance survives any failure.
    pub async fn load(
        &self,
        registry: &ArtifactRegistry,
        id: &ArtifactId,
        worker_id: WorkerId,
        deadlines: Deadlines,
    ) -> Result<LoadedModel, LoadError> {
        // Prepare. Nothing has been started, so every refusal here is free.
        let artifact = self.resolve(registry, id)?;
        let descriptor = describe(&artifact);

        let compatibility = mehoy_backend_llama::compatibility::assess(&descriptor);
        if !compatibility.permits_start() {
            return Err(LoadError::Incompatible {
                reason: compatibility.to_string(),
            });
        }

        // Spawn and verify. The backend crate owns the credential, the private
        // channel, and the authenticated readiness check; a failure inside it
        // already leaves nothing running.
        let spec = LlamaCppWorkerSpec {
            model_path: artifact.path.clone(),
            // The same description the preflight judged. What the backend does with
            // it, including how it chooses to start, is the backend's business.
            descriptor,
            context_size: None,
            gpu_layers: None,
            deadlines,
        };
        let backend = self
            .backend
            .start(worker_id, &spec)
            .await
            .map_err(LoadError::Backend)?;

        // Commit. From here a process is running and holding memory, so every
        // failure must stop it rather than leaking it.
        match self.commit(&artifact, backend) {
            Ok(loaded) => Ok(loaded),
            Err(rollback) => {
                let (reason, mut backend) = *rollback;
                let _ = backend.stop().await;
                Err(LoadError::Instance { reason })
            }
        }
    }

    /// Resolves an artifact and confirms it is still the one that was inspected.
    fn resolve(
        &self,
        registry: &ArtifactRegistry,
        id: &ArtifactId,
    ) -> Result<ModelArtifact, LoadError> {
        let artifact = registry
            .get(id)?
            .ok_or_else(|| LoadError::UnknownArtifact { id: id.clone() })?;

        // Checked immediately before use rather than trusted from registration
        // time. A file can be replaced between the two, and loading something that
        // was never inspected would defeat registration entirely.
        match mehoy_registry::detect_drift(&artifact) {
            ArtifactState::Unchanged => Ok(artifact),
            ArtifactState::Missing => Err(LoadError::ArtifactMissing {
                id: id.clone(),
                path: artifact.path.display().to_string(),
            }),
            ArtifactState::Changed { reason } => Err(LoadError::ArtifactChanged {
                id: id.clone(),
                reason,
            }),
        }
    }

    /// Builds the instance.
    ///
    /// Hands the backend back on failure so the caller can stop it. Boxed because
    /// a running backend is large and this sits on the hot path of a `Result`.
    fn commit(
        &self,
        artifact: &ModelArtifact,
        backend: RunningBackend,
    ) -> Result<LoadedModel, Box<(String, RunningBackend)>> {
        if let Err(reason) = injected_commit_fault() {
            return Err(Box::new((reason, backend)));
        }

        let instance_id = match InstanceId::generate() {
            Ok(id) => id,
            Err(err) => return Err(Box::new((err.to_string(), backend))),
        };

        let capabilities = indicated_by_metadata(
            artifact.metadata.architecture.as_deref(),
            artifact.metadata.embedding_length,
            artifact.metadata.has_chat_template(),
        );

        let instance = ModelInstance::new(
            instance_id,
            artifact.id.clone(),
            backend.identity().cloned(),
            capabilities,
        );

        Ok(LoadedModel {
            instance,
            channel: backend.channel().clone(),
            identity: backend.identity().cloned(),
            requests: IdAllocator::default(),
            lifecycle: std::sync::Mutex::new(Lifecycle {
                phase: LifecyclePhase::Serving,
                active: std::collections::HashMap::new(),
                worker: Some(backend),
            }),
        })
    }
}

/// Translates a registered artifact into what the backend needs to judge it.
///
/// The join lives here because this is the only layer that legitimately knows both
/// the registry and the backend. Neither of those crates references the other.
fn describe(artifact: &ModelArtifact) -> ModelDescriptor {
    ModelDescriptor {
        architecture: artifact.metadata.architecture.clone(),
        context_length: artifact.metadata.context_length,
        embedding_length: artifact.metadata.embedding_length,
        tokenizer_model: artifact.metadata.tokenizer_model.clone(),
        has_chat_template: artifact.metadata.has_chat_template(),
    }
}

/// A failure injected after the backend is healthy, so the rollback can be tested.
///
/// The path it exercises is otherwise unreachable from a test: everything else
/// that can fail does so before a process exists.
///
/// The flag is process-wide, so tests that arm it must not run concurrently with
/// other tests that load a model, or one will consume the other's injected failure.
#[cfg(feature = "fault-injection")]
mod fault {
    use std::sync::atomic::{AtomicBool, Ordering};

    static FAIL_COMMIT: AtomicBool = AtomicBool::new(false);

    /// Makes the next instance construction fail.
    pub fn fail_next_commit() {
        FAIL_COMMIT.store(true, Ordering::SeqCst);
    }

    /// Clears any pending injected failure.
    pub fn clear() {
        FAIL_COMMIT.store(false, Ordering::SeqCst);
    }

    pub(super) fn take() -> bool {
        FAIL_COMMIT.swap(false, Ordering::SeqCst)
    }
}

#[cfg(feature = "fault-injection")]
pub use fault::{clear as clear_injected_faults, fail_next_commit};

#[cfg(feature = "fault-injection")]
fn injected_commit_fault() -> Result<(), String> {
    if fault::take() {
        return Err("injected commit failure".to_owned());
    }
    Ok(())
}

#[cfg(not(feature = "fault-injection"))]
fn injected_commit_fault() -> Result<(), String> {
    Ok(())
}

/// Re-exported so callers can name the verdict without depending on the backend
/// crate directly.
pub use mehoy_backend_llama::Compatibility as BackendCompatibility;
