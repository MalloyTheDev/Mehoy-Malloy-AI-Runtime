//! Live model instances.
//!
//! An instance is deliberately not persisted. Artifacts survive a restart because
//! they are facts about files on disk; an instance is a resource reservation held
//! by a process, and a restarted runtime that read instances from storage would be
//! claiming that processes which died with it are still resident.
//!
//! An instance references its artifact by identifier rather than copying it, so
//! there is exactly one owner of what an artifact is.

use std::fmt;

use mehoy_backend_llama::BackendIdentity;
use mehoy_registry::ArtifactId;

use crate::capability::ModelCapabilities;

/// An instance's identifier.
///
/// Unique within one runtime process and meaningless outside it, which is the
/// point: the thing it names cannot outlive the process either.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InstanceId(String);

impl InstanceId {
    /// Generates an identifier.
    ///
    /// # Errors
    ///
    /// Returns an error when the system randomness source is unavailable.
    pub fn generate() -> std::io::Result<Self> {
        let mut bytes = [0u8; 12];
        getrandom::fill(&mut bytes).map_err(|err| {
            std::io::Error::other(format!("system randomness unavailable: {err}"))
        })?;
        let encoded: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
        Ok(Self(encoded))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for InstanceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Why an instance failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstanceFailure {
    pub reason: String,
}

/// Where an instance is in its life.
///
/// `Failed` is terminal with no edge back into `Starting`, matching the worker
/// state machine. Recovering from a failure is an explicit act, not something the
/// runtime does quietly on a model's behalf.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstanceState {
    Starting,
    /// The backend loaded the artifact and is accepting authenticated requests.
    ///
    /// This says the model is resident and reachable. It does not say that any
    /// particular kind of request will succeed, which is what
    /// [`crate::capability`] exists to track separately.
    BackendReady,
    Stopping,
    Failed(InstanceFailure),
}

impl InstanceState {
    /// Whether the instance can serve requests.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        matches!(self, Self::BackendReady)
    }

    /// Whether no further transition happens without outside intervention.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Failed(_))
    }

    /// Whether moving to `next` is allowed.
    #[must_use]
    pub fn can_transition_to(&self, next: &Self) -> bool {
        matches!(
            (self, next),
            (Self::Starting, Self::BackendReady | Self::Failed(_))
                | (Self::BackendReady, Self::Stopping | Self::Failed(_))
                | (Self::Stopping, Self::Failed(_))
        )
    }
}

impl fmt::Display for InstanceState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Starting => f.write_str("starting"),
            Self::BackendReady => f.write_str("backend ready"),
            Self::Stopping => f.write_str("stopping"),
            Self::Failed(failure) => write!(f, "failed: {}", failure.reason),
        }
    }
}

/// A model resident in a backend.
#[derive(Debug, Clone)]
pub struct ModelInstance {
    id: InstanceId,
    artifact_id: ArtifactId,
    backend: Option<BackendIdentity>,
    state: InstanceState,
    capabilities: ModelCapabilities,
}

impl ModelInstance {
    #[must_use]
    pub(crate) fn new(
        id: InstanceId,
        artifact_id: ArtifactId,
        backend: Option<BackendIdentity>,
        capabilities: ModelCapabilities,
    ) -> Self {
        Self {
            id,
            artifact_id,
            backend,
            state: InstanceState::BackendReady,
            capabilities,
        }
    }

    #[must_use]
    pub fn id(&self) -> &InstanceId {
        &self.id
    }

    /// The artifact this instance was loaded from.
    ///
    /// A reference rather than a copy: the registry owns what an artifact is.
    #[must_use]
    pub fn artifact_id(&self) -> &ArtifactId {
        &self.artifact_id
    }

    /// Which backend build is running it.
    #[must_use]
    pub fn backend(&self) -> Option<&BackendIdentity> {
        self.backend.as_ref()
    }

    #[must_use]
    pub fn state(&self) -> &InstanceState {
        &self.state
    }

    /// What is known about this model's capabilities, and how strongly.
    #[must_use]
    pub fn capabilities(&self) -> &ModelCapabilities {
        &self.capabilities
    }

    /// Mutable capabilities, for a probe that demonstrates one.
    pub fn capabilities_mut(&mut self) -> &mut ModelCapabilities {
        &mut self.capabilities
    }

    pub(crate) fn set_state(&mut self, next: InstanceState) {
        self.state = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn failure() -> InstanceState {
        InstanceState::Failed(InstanceFailure {
            reason: "test".to_owned(),
        })
    }

    #[test]
    fn identifiers_are_unique() {
        let first = InstanceId::generate().expect("generates");
        let second = InstanceId::generate().expect("generates");
        assert_ne!(first, second);
    }

    #[test]
    fn only_backend_ready_is_usable() {
        assert!(InstanceState::BackendReady.is_usable());
        assert!(!InstanceState::Starting.is_usable());
        assert!(!InstanceState::Stopping.is_usable());
        assert!(!failure().is_usable());
    }

    #[test]
    fn a_failed_instance_never_restarts_itself() {
        // Matches the worker state machine: recovery is explicit.
        assert!(failure().is_terminal());
        assert!(!failure().can_transition_to(&InstanceState::Starting));
        assert!(!failure().can_transition_to(&InstanceState::BackendReady));
    }

    #[test]
    fn the_happy_path_is_permitted() {
        assert!(InstanceState::Starting.can_transition_to(&InstanceState::BackendReady));
        assert!(InstanceState::BackendReady.can_transition_to(&InstanceState::Stopping));
    }

    #[test]
    fn readiness_cannot_be_reached_without_starting() {
        assert!(!InstanceState::Stopping.can_transition_to(&InstanceState::BackendReady));
    }
}
