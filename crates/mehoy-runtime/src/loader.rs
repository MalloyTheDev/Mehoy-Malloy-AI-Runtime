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
    BackendError, LlamaCppBackend, LlamaCppWorkerSpec, ModelDescriptor, RunningBackend,
};
use mehoy_core::id::WorkerId;
use mehoy_core::inference::{
    self, EmbedRequest, EmbeddingResult, GenerateTextRequest, GenerationResult,
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

/// A running instance together with the backend serving it.
#[derive(Debug)]
pub struct LoadedModel {
    instance: ModelInstance,
    backend: RunningBackend,
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
    pub fn backend(&self) -> &RunningBackend {
        &self.backend
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
        if !self.instance.state().is_usable() {
            return Err(LoadError::NotUsable {
                state: self.instance.state().to_string(),
            });
        }

        let result = mehoy_backend_llama::embed(self.backend.channel(), request)
            .await
            .map_err(|source| LoadError::Inference {
                detail: source.to_string(),
            })?;

        inference::validate(request, &result).map_err(|defect| LoadError::UnusableResult {
            detail: defect.to_string(),
        })?;

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
        let backend = self.backend.identity().cloned();
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
        if !self.instance.state().is_usable() {
            return Err(LoadError::NotUsable {
                state: self.instance.state().to_string(),
            });
        }

        request
            .parameters
            .validate()
            .map_err(|invalid| LoadError::InvalidRequest {
                detail: invalid.to_string(),
            })?;

        let result = mehoy_backend_llama::generate(self.backend.channel(), request)
            .await
            .map_err(|source| LoadError::Inference {
                detail: source.to_string(),
            })?;

        inference::validate_generation(&result).map_err(|defect| LoadError::UnusableResult {
            detail: defect.to_string(),
        })?;

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
        let backend = self.backend.identity().cloned();
        self.instance
            .capabilities_mut()
            .verify(ModelCapability::TextGeneration, backend);
        Ok(result)
    }

    /// Stops the backend and destroys the instance.
    ///
    /// # Errors
    ///
    /// Returns an error when the backend had to be killed rather than stopping
    /// politely. The backend is stopped either way.
    pub async fn unload(mut self) -> Result<(), LoadError> {
        self.instance
            .set_state(crate::instance::InstanceState::Stopping);
        self.backend.stop().await.map_err(LoadError::Backend)
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

        Ok(LoadedModel { instance, backend })
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
