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
//! The ledger lock resolves an identifier to an instance and is released
//! immediately. The instance's own lifecycle lock decides whether the request is
//! admitted. Merging them into one lock would mean holding the ledger across
//! inference, cancellation, draining and process teardown, which would make every
//! other caller wait on whatever one instance happened to be doing.
//!
//! Keeping them separate does not weaken the guarantee, because the race that
//! matters is per instance. A request and an unload that both resolve the same
//! identifier then meet at that instance's lock, and whichever arrives first
//! wins: either the request is registered and unloading finds it, or admission
//! has closed and the request is refused. There is no interleaving in which a
//! request reaches a backend that is being torn down.
//!
//! # A limit is claimed, not merely checked
//!
//! The one race the ledger lock does own is between two loads. Checking a limit
//! and then spending a minute bringing a model up would let two concurrent loads
//! both pass a check that only one of them can satisfy, and the second would
//! discover this only after the expensive part was already done. So a load takes
//! a reservation under the ledger lock before it starts and holds it until the
//! model is either resident or gone. A load in flight is accounted for exactly as
//! a resident model is, because the memory is being taken either way.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
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
use crate::resources::{Refusal, ResidentLimits, ResidentUsage, ResourceFootprint};

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
    /// This runtime already holds as many models as it will.
    ///
    /// Counts loads in flight, so a second caller is refused while the first is
    /// still bringing its model up rather than after.
    ResidentCapacityReached { capacity: usize },
    /// The model's accounted weights do not fit this runtime's byte budget.
    ///
    /// Every figure is declared rather than measured: see
    /// [`crate::resources`]. A refusal here means the stated policy would be
    /// exceeded, not that the hardware would be.
    ResourcesExhausted {
        requested: u64,
        committed: u64,
        limit: u64,
    },
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
            Self::ResourcesExhausted {
                requested,
                committed,
                limit,
            } => {
                if requested > limit {
                    write!(
                        f,
                        "this model is accounted at {requested} byte(s) of weights, more than the \
                         whole {limit} byte budget, so it will not fit here however much is \
                         unloaded first"
                    )
                } else {
                    write!(
                        f,
                        "this model is accounted at {requested} byte(s) of weights and \
                         {committed} of {limit} are already committed; unload something first"
                    )
                }
            }
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

impl RuntimeError {
    /// Translates a policy refusal into this layer's own vocabulary.
    ///
    /// Exhaustive on purpose: a new limit forces this to be updated rather than
    /// silently collapsing into an existing message.
    fn from_refusal(refusal: Refusal) -> Self {
        match refusal {
            Refusal::Instances { capacity } => Self::ResidentCapacityReached { capacity },
            Refusal::WeightBytes {
                requested,
                committed,
                limit,
            } => Self::ResourcesExhausted {
                requested,
                committed,
                limit,
            },
        }
    }
}

/// Identifies one load in flight.
///
/// Internal and short lived: it exists only so a reservation can be released
/// again, and it is gone once the load has either produced a model or failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ReservationId(u64);

/// A model this runtime holds, and what it is accounted for.
///
/// One entry rather than two maps, so releasing a model and releasing what it
/// occupied cannot come apart. There is no code path that forgets one and not the
/// other, because there is only one thing to forget.
#[derive(Debug)]
struct Resident {
    model: Arc<LoadedModel>,
    footprint: ResourceFootprint,
}

/// Everything committed against this runtime's limits.
#[derive(Debug, Default)]
struct Ledger {
    /// Loads in flight, which have claimed a limit but have no model yet.
    pending: HashMap<ReservationId, ResourceFootprint>,
    resident: HashMap<InstanceId, Resident>,
    next_reservation: u64,
}

impl Ledger {
    /// What is committed right now, loads in flight included.
    fn committed(&self) -> ResidentUsage {
        let weight_bytes = self
            .pending
            .values()
            .chain(self.resident.values().map(|entry| &entry.footprint))
            // Saturating, which errs towards refusing. A total that saturated
            // would have to come from exabytes of containers, and reporting the
            // maximum is the safe direction to be wrong in.
            .fold(0u64, |total, footprint| {
                total.saturating_add(footprint.weight_bytes)
            });
        ResidentUsage {
            instances: self.pending.len().saturating_add(self.resident.len()),
            loading: self.pending.len(),
            weight_bytes,
        }
    }

    fn allocate(&mut self) -> ReservationId {
        self.next_reservation = self.next_reservation.wrapping_add(1);
        ReservationId(self.next_reservation)
    }
}

/// A claim on this runtime's limits, held for the whole of one load.
///
/// Released when it is dropped, which is what makes every way a load can fail
/// release it: an early return, an error from the backend, or a panic unwinding
/// through the load. The alternative, releasing explicitly on each failure path,
/// is one path away from leaking a limit that nothing will ever give back.
struct Reserved<'a> {
    runtime: &'a Runtime,
    id: ReservationId,
    /// Set once the claim has been turned into a resident model, so dropping the
    /// guard afterwards does not release what is now held for real.
    settled: bool,
}

impl Reserved<'_> {
    /// Turns the claim into a resident model and reports its identifier.
    ///
    /// # Errors
    ///
    /// Hands the model back, with an account of why, when the runtime cannot take
    /// it. The caller is then the only party able to stop the worker it just
    /// started.
    fn fulfil(mut self, model: LoadedModel) -> Result<InstanceId, Box<(String, LoadedModel)>> {
        let runtime = self.runtime;
        let id = model.instance_id();
        let mut ledger = runtime.ledger();

        let Some(footprint) = ledger.pending.remove(&self.id) else {
            // Unreachable: this consumes the reservation, so it is settled once.
            self.settled = true;
            return Err(Box::new((
                "the reservation this load was admitted under had already been released".to_owned(),
                model,
            )));
        };
        self.settled = true;

        match ledger.resident.entry(id.clone()) {
            Entry::Vacant(slot) => {
                slot.insert(Resident {
                    model: Arc::new(model),
                    footprint,
                });
                Ok(id)
            }
            // Unreachable in practice: an identifier is 96 random bits. Refused
            // rather than inserted anyway, because inserting would drop a live
            // model whose worker nothing would then stop.
            Entry::Occupied(_) => Err(Box::new((
                format!("instance {id} is already resident"),
                model,
            ))),
        }
    }
}

impl Drop for Reserved<'_> {
    fn drop(&mut self) {
        if !self.settled {
            self.runtime.ledger().pending.remove(&self.id);
        }
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
    /// Stated once, when the runtime is built, and never changed afterwards.
    ///
    /// A limit that could move while loads were being judged against it would
    /// make a refusal unexplainable, so changing what this runtime will hold
    /// means building one that holds it.
    limits: ResidentLimits,
    ledger: Mutex<Ledger>,
}

impl Runtime {
    /// A runtime with the default limits.
    ///
    /// See [`ResidentLimits::default`]: one model, no byte budget.
    #[must_use]
    pub fn new(backend: LlamaCppBackend) -> Self {
        Self::with_limits(backend, ResidentLimits::default())
    }

    /// A runtime that will hold what `limits` says.
    #[must_use]
    pub fn with_limits(backend: LlamaCppBackend, limits: ResidentLimits) -> Self {
        Self {
            loader: ModelLoader::new(backend),
            limits,
            ledger: Mutex::new(Ledger::default()),
        }
    }

    /// What this runtime will hold.
    #[must_use]
    pub fn limits(&self) -> ResidentLimits {
        self.limits
    }

    /// What this runtime currently holds against those limits.
    ///
    /// Includes loads in flight, so the figures are what the next load will be
    /// judged against rather than a count of finished work.
    #[must_use]
    pub fn usage(&self) -> ResidentUsage {
        self.ledger().committed()
    }

    /// What one resident model is accounted as occupying.
    ///
    /// The figure a load was admitted under. It does not change afterwards:
    /// nothing re-measures a running model, and a number that drifted would make
    /// the accounting impossible to reason about.
    #[must_use]
    pub fn footprint(&self, id: &InstanceId) -> Option<ResourceFootprint> {
        self.ledger().resident.get(id).map(|entry| entry.footprint)
    }

    /// Brings a registered artifact up and keeps it.
    ///
    /// Returns an identifier rather than the model. The runtime holds the model
    /// for as long as it is resident, so nothing outside can keep one alive, use
    /// one after unloading it, or reach its backend directly.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError::ResidentCapacityReached`] or
    /// [`RuntimeError::ResourcesExhausted`] when this runtime will not hold the
    /// model, and [`RuntimeError::Instance`] when the artifact cannot be brought
    /// up. A refused or failed load leaves nothing committed.
    pub async fn load(
        &self,
        registry: &ArtifactRegistry,
        artifact: &ArtifactId,
        worker: WorkerId,
        deadlines: Deadlines,
    ) -> Result<InstanceId, RuntimeError> {
        // Sized before the limit is claimed, so an artifact this runtime has
        // never heard of is refused without disturbing the accounting at all.
        let footprint = declared_footprint(registry, artifact)?;

        // Claimed before the work rather than checked before it. Starting a
        // multi-gigabyte load and then refusing to keep it would waste the
        // expensive part, and checking without claiming would let a second
        // concurrent load pass the same check.
        let reserved = self.reserve(footprint)?;

        let loaded = self
            .loader
            .load(registry, artifact, worker, deadlines)
            .await?;

        match reserved.fulfil(loaded) {
            Ok(id) => Ok(id),
            Err(rejected) => {
                let (reason, model) = *rejected;
                // A model the runtime will not take is a model nothing else can
                // reach, and its worker is holding memory. Stopping it here is the
                // only opportunity anything has to do so.
                let reason = match model.unload_within(UnloadBudget::default()).await {
                    Ok(_) => reason,
                    Err(err) => format!(
                        "{reason}; the backend could not be stopped afterwards and may still be \
                         running: {err}"
                    ),
                };
                Err(RuntimeError::Instance(LoadError::Instance { reason }))
            }
        }
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
        self.ledger().resident.keys().cloned().collect()
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
                // Removing the entry releases what the model was accounted for,
                // because the two are one entry.
                self.ledger().resident.remove(id);
            }
            // A teardown that did not finish keeps its entry, so what is left can
            // still be named and asked about, and stays accounted for. A worker
            // that could not be stopped may still be holding its memory, and
            // handing that budget to another model would be the more dangerous of
            // the two wrong answers.
            _ => {}
        }

        Ok(outcome?)
    }

    /// Resolves an identifier, releasing the ledger lock immediately.
    fn resolve(&self, id: &InstanceId) -> Option<Arc<LoadedModel>> {
        self.ledger()
            .resident
            .get(id)
            .map(|entry| Arc::clone(&entry.model))
    }

    fn require(&self, id: &InstanceId) -> Result<Arc<LoadedModel>, RuntimeError> {
        self.resolve(id)
            .ok_or_else(|| RuntimeError::UnknownInstance {
                instance: id.clone(),
            })
    }

    /// Claims room for one model, or refuses it.
    ///
    /// Deciding and recording happen under one lock, so two concurrent loads
    /// cannot both be told there is room for one.
    fn reserve(&self, footprint: ResourceFootprint) -> Result<Reserved<'_>, RuntimeError> {
        let mut ledger = self.ledger();
        self.limits
            .admits(ledger.committed(), footprint)
            .map_err(RuntimeError::from_refusal)?;
        let id = ledger.allocate();
        ledger.pending.insert(id, footprint);
        Ok(Reserved {
            runtime: self,
            id,
            settled: false,
        })
    }

    fn ledger(&self) -> MutexGuard<'_, Ledger> {
        // Never held across an await. Every slow operation happens after the
        // guard has been dropped, which is what keeps one instance's teardown
        // from blocking access to another.
        self.ledger.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// What a registered artifact is accounted as needing.
///
/// Read here rather than taken from the loader, because the limit has to be
/// claimed before the expensive work and the loader does not return until that
/// work is done. The loader resolves the artifact again and confirms it has not
/// drifted immediately before starting a worker, which is where that check
/// belongs: a file can be replaced between the two, and the one that matters is
/// the one taken immediately before use.
fn declared_footprint(
    registry: &ArtifactRegistry,
    artifact: &ArtifactId,
) -> Result<ResourceFootprint, LoadError> {
    let record = registry
        .get(artifact)?
        .ok_or_else(|| LoadError::UnknownArtifact {
            id: artifact.clone(),
        })?;
    Ok(ResourceFootprint::from_container_size(record.size_bytes))
}
