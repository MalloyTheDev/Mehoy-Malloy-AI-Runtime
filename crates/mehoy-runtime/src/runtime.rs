//! The runtime's ownership of live models.
//!
//! # Why this layer exists
//!
//! Before it, a caller held the loaded model itself. That made the runtime a
//! library for building instances rather than a runtime that has them: nothing
//! could be addressed by name, nothing could be asked about from elsewhere, and
//! the only thing keeping a model alive was whoever happened to hold the value.
//!
//! Here the runtime keeps them and hands out identifiers. Everything else
//! follows: a request names an instance rather than carrying one, teardown can be
//! asked for by something that is not using the model, and the number of resident
//! models becomes a property the runtime can state rather than an accident of how
//! many values a caller is holding.
//!
//! # Two locks, deliberately
//!
//! The registry lock resolves an identifier to an instance and is released
//! immediately. The instance's own lifecycle lock decides whether the request is
//! admitted. Merging them into one lock would mean holding the registry across
//! inference, cancellation, draining and process teardown, which would make every
//! other caller wait on whatever one instance happened to be doing.
//!
//! Keeping them separate does not weaken the guarantee, because the race that
//! matters is per instance. A request and an unload that both resolve the same
//! identifier then meet at that instance's lock, and whichever arrives first
//! wins: either the request is registered and unloading finds it, or admission
//! has closed and the request is refused. There is no interleaving in which a
//! request reaches a backend that is being torn down.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::PoisonError;

use mehoy_backend_llama::LlamaCppBackend;
use mehoy_core::cancel::{CancellationStrategy, RequestBudget, RequestState, UnloadBudget};
use mehoy_core::id::{RequestId, WorkerId};
use mehoy_core::inference::{
    EmbedRequest, EmbeddingResult, GenerateTextRequest, GenerationResult, GenerationStream,
};
use mehoy_core::worker::Deadlines;
use mehoy_registry::{ArtifactId, ArtifactRegistry};

use crate::instance::{InstanceId, ModelInstance};
use crate::loader::{
    CancelError, CancelOutcome, GenerationRequest, LifecyclePhase, LoadError, LoadedModel,
    ModelLoader, UnloadOutcome,
};

/// How many models may be resident at once.
///
/// One, stated rather than implied. The registry is keyed by identifier and could
/// hold more, so without this the limit would be an accident of nothing having
/// been written yet. Lifting it means deciding what happens when two models
/// compete for one accelerator, which is resource accounting and does not exist.
pub const RESIDENT_CAPACITY: usize = 1;

/// Why a runtime operation could not be carried out.
#[derive(Debug)]
pub enum RuntimeError {
    /// No instance with that identifier is resident.
    ///
    /// Either it never existed, or it was unloaded. The two are not
    /// distinguished: the runtime keeps no record of instances it no longer has,
    /// and an identifier is not reused, so a stale one can never reach a later
    /// model.
    UnknownInstance { instance: InstanceId },
    /// Another model is already resident.
    ResidentCapacityReached { capacity: usize },
    /// The operation itself failed.
    Instance(LoadError),
    /// The request could not be cancelled.
    Cancel(CancelError),
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownInstance { instance } => {
                write!(f, "no instance {instance} is resident")
            }
            Self::ResidentCapacityReached { capacity } => write!(
                f,
                "this runtime holds at most {capacity} resident model(s); unload one first"
            ),
            Self::Instance(err) => write!(f, "{err}"),
            Self::Cancel(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for RuntimeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Instance(err) => Some(err),
            Self::Cancel(err) => Some(err),
            _ => None,
        }
    }
}

impl From<LoadError> for RuntimeError {
    fn from(err: LoadError) -> Self {
        Self::Instance(err)
    }
}

impl From<CancelError> for RuntimeError {
    fn from(err: CancelError) -> Self {
        Self::Cancel(err)
    }
}

/// Owns every live model.
///
/// Models are addressed by [`InstanceId`] and never handed out. A caller cannot
/// obtain the instance itself, which is what keeps every operation on it going
/// through this type and therefore through admission.
#[derive(Debug)]
pub struct Runtime {
    loader: ModelLoader,
    resident: Mutex<HashMap<InstanceId, Arc<LoadedModel>>>,
}

impl Runtime {
    #[must_use]
    pub fn new(backend: LlamaCppBackend) -> Self {
        Self {
            loader: ModelLoader::new(backend),
            resident: Mutex::new(HashMap::new()),
        }
    }

    /// Brings a registered artifact up and keeps it.
    ///
    /// Returns an identifier rather than the model. The runtime holds the model
    /// for as long as it is resident, so nothing outside can keep one alive, use
    /// one after unloading it, or reach its backend directly.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::ResidentCapacityReached`] when a model is already
    /// resident, and [`RuntimeError::Instance`] when the artifact cannot be
    /// brought up.
    pub async fn load(
        &self,
        registry: &ArtifactRegistry,
        artifact: &ArtifactId,
        worker: WorkerId,
        deadlines: Deadlines,
    ) -> Result<InstanceId, RuntimeError> {
        // Checked before the work rather than after it. Starting a multi-gigabyte
        // load and then refusing to keep it would waste the expensive part.
        self.admit_resident()?;

        let loaded = self
            .loader
            .load(registry, artifact, worker, deadlines)
            .await?;
        let id = loaded.instance_id();

        let mut resident = self.resident();
        resident.insert(id.clone(), Arc::new(loaded));
        Ok(id)
    }

    /// What is known about a resident instance.
    ///
    /// A snapshot, so a caller can read an instance's identity, artifact, backend
    /// and capability evidence without being handed the instance itself. The copy
    /// does not change as the instance does.
    #[must_use]
    pub fn instance(&self, id: &InstanceId) -> Option<ModelInstance> {
        self.resolve(id).map(|model| model.instance_snapshot())
    }

    /// Every resident instance, in no particular order.
    #[must_use]
    pub fn resident_instances(&self) -> Vec<InstanceId> {
        self.resident().keys().cloned().collect()
    }

    /// Whether a resident instance is still serving.
    #[must_use]
    pub fn phase(&self, id: &InstanceId) -> Option<LifecyclePhase> {
        self.resolve(id).map(|model| model.phase())
    }

    /// How many requests the named instance has accepted and not seen end.
    #[must_use]
    pub fn active_requests(&self, id: &InstanceId) -> Option<usize> {
        self.resolve(id).map(|model| model.active_requests())
    }

    /// The state of one request, if its instance still knows about it.
    #[must_use]
    pub fn request_state(&self, id: &InstanceId, request: RequestId) -> Option<RequestState> {
        self.resolve(id)
            .and_then(|model| model.request_state(request))
    }

    /// Where the named instance's backend listens.
    ///
    /// The address alone, never the channel, which carries the credential. Enough
    /// to check that a worker is gone after its instance has been forgotten.
    #[must_use]
    pub fn backend_address(&self, id: &InstanceId) -> Option<std::net::SocketAddr> {
        self.resolve(id).map(|model| model.backend_address())
    }

    /// Whether anything still answers on the named instance's backend address.
    ///
    /// Reports what teardown achieved without exposing the channel, which
    /// carries the backend's credential.
    pub async fn backend_reachable(&self, id: &InstanceId) -> Option<bool> {
        let model = self.resolve(id)?;
        Some(model.backend_reachable().await)
    }

    /// How the named instance's backend is able to stop work.
    #[must_use]
    pub fn cancellation_strategy(&self, id: &InstanceId) -> Option<CancellationStrategy> {
        self.resolve(id).map(|model| model.cancellation_strategy())
    }

    /// Embeds through the named instance.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::UnknownInstance`] before any backend is contacted
    /// when the identifier is not resident, and otherwise whatever the instance
    /// reports.
    pub async fn embed(
        &self,
        id: &InstanceId,
        request: &EmbedRequest,
    ) -> Result<EmbeddingResult, RuntimeError> {
        let model = self.require(id)?;
        Ok(model.embed(request).await?)
    }

    /// Embeds, and records the capability if it succeeds.
    ///
    /// # Errors
    ///
    /// As [`Runtime::embed`].
    pub async fn verify_embeddings(
        &self,
        id: &InstanceId,
        request: &EmbedRequest,
    ) -> Result<EmbeddingResult, RuntimeError> {
        let model = self.require(id)?;
        Ok(model.verify_embeddings(request).await?)
    }

    /// Generates a whole response through the named instance.
    ///
    /// # Errors
    ///
    /// As [`Runtime::embed`].
    pub async fn generate_text(
        &self,
        id: &InstanceId,
        request: &GenerateTextRequest,
    ) -> Result<GenerationResult, RuntimeError> {
        let model = self.require(id)?;
        Ok(model.generate_text(request).await?)
    }

    /// Generates a whole response, and records the capability if it succeeds.
    ///
    /// # Errors
    ///
    /// As [`Runtime::embed`].
    pub async fn verify_text_generation(
        &self,
        id: &InstanceId,
        request: &GenerateTextRequest,
    ) -> Result<GenerationResult, RuntimeError> {
        let model = self.require(id)?;
        Ok(model.verify_text_generation(request).await?)
    }

    /// Begins a streaming generation through the named instance.
    ///
    /// # Errors
    ///
    /// As [`Runtime::embed`].
    pub fn generate_stream(
        &self,
        id: &InstanceId,
        request: &GenerateTextRequest,
    ) -> Result<GenerationRequest, RuntimeError> {
        self.generate_stream_within(id, request, RequestBudget::default())
    }

    /// The same, with an explicit budget.
    ///
    /// # Errors
    ///
    /// As [`Runtime::embed`].
    pub fn generate_stream_within(
        &self,
        id: &InstanceId,
        request: &GenerateTextRequest,
        budget: RequestBudget,
    ) -> Result<GenerationRequest, RuntimeError> {
        let model = self.require(id)?;
        Ok(model.generate_stream_within(request, budget)?)
    }

    /// Records what a drained stream demonstrated.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::UnknownInstance`] when the identifier is not
    /// resident.
    pub fn record_generation_stream(
        &self,
        id: &InstanceId,
        stream: &GenerationStream,
    ) -> Result<bool, RuntimeError> {
        Ok(self.require(id)?.record_generation_stream(stream))
    }

    /// Asks a request on the named instance to stop.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::UnknownInstance`] when the instance is not
    /// resident, and [`RuntimeError::Cancel`] when it has no such request.
    pub fn cancel(
        &self,
        id: &InstanceId,
        request: RequestId,
    ) -> Result<CancelOutcome, RuntimeError> {
        Ok(self.require(id)?.cancel(request)?)
    }

    /// Takes a resident instance down and forgets it.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::UnknownInstance`] when the identifier is not
    /// resident, and [`RuntimeError::Instance`] when teardown fails.
    pub async fn unload(&self, id: &InstanceId) -> Result<UnloadOutcome, RuntimeError> {
        self.unload_within(id, UnloadBudget::default()).await
    }

    /// The same, with an explicit drain budget.
    ///
    /// # Errors
    ///
    /// As [`Runtime::unload`].
    pub async fn unload_within(
        &self,
        id: &InstanceId,
        budget: UnloadBudget,
    ) -> Result<UnloadOutcome, RuntimeError> {
        // Held only long enough to resolve. Teardown cancels requests, waits for
        // them, and stops a process, none of which may block another caller from
        // reaching a different instance.
        let model = self.require(id)?;

        // The entry stays while this runs. Removing it first would make an
        // instance that is being torn down indistinguishable from one that never
        // existed, so a second attempt could not be told it was already in
        // progress and an observer could not see it stopping.
        let outcome = model.unload_within(budget).await;

        match &outcome {
            Ok(_) if model.phase() == LifecyclePhase::Unloaded => {
                self.resident().remove(id);
            }
            // A teardown that did not finish keeps its entry, so what is left can
            // still be named and asked about.
            _ => {}
        }

        Ok(outcome?)
    }

    /// Resolves an identifier, releasing the registry lock immediately.
    fn resolve(&self, id: &InstanceId) -> Option<Arc<LoadedModel>> {
        self.resident().get(id).map(Arc::clone)
    }

    fn require(&self, id: &InstanceId) -> Result<Arc<LoadedModel>, RuntimeError> {
        self.resolve(id)
            .ok_or_else(|| RuntimeError::UnknownInstance {
                instance: id.clone(),
            })
    }

    /// Refuses a load that would exceed what this runtime will hold.
    fn admit_resident(&self) -> Result<(), RuntimeError> {
        let resident = self.resident();
        if resident.len() >= RESIDENT_CAPACITY {
            return Err(RuntimeError::ResidentCapacityReached {
                capacity: RESIDENT_CAPACITY,
            });
        }
        Ok(())
    }

    fn resident(&self) -> MutexGuard<'_, HashMap<InstanceId, Arc<LoadedModel>>> {
        // Never held across an await. Every slow operation happens after the
        // guard has been dropped, which is what keeps one instance's teardown
        // from blocking access to another.
        self.resident.lock().unwrap_or_else(PoisonError::into_inner)
    }
}
