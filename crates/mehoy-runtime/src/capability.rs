//! What a model can do, and how strongly that is known.
//!
//! A capability claim is only as good as the check that produced it, so every
//! claim carries the evidence behind it. Three sources exist, and they are not
//! interchangeable:
//!
//! - **declared**: the artifact's own metadata says something. It is a statement
//!   by whoever packaged the file, not a demonstration.
//! - **backend reported**: the loaded backend says it offers something. Better,
//!   because a running backend has actually read the model, but still a claim.
//! - **probed**: a real request of that kind was made and produced a usable
//!   answer. This is the only evidence that demonstrates rather than asserts.
//!
//! Only a probe can mark a capability [`Verified`]. Anything else is at most
//! [`Indicated`]. In particular, an architecture name never verifies anything: it
//! describes what the tensors are, not what the runtime will let you ask of them.
//!
//! [`Verified`]: CapabilityState::Verified
//! [`Indicated`]: CapabilityState::Indicated

use std::collections::BTreeMap;
use std::fmt;

/// A thing a model might be able to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ModelCapability {
    TextGeneration,
    Embeddings,
    Vision,
    ToolCalling,
    StructuredOutput,
}

impl ModelCapability {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TextGeneration => "text-generation",
            Self::Embeddings => "embeddings",
            Self::Vision => "vision",
            Self::ToolCalling => "tool-calling",
            Self::StructuredOutput => "structured-output",
        }
    }
}

impl fmt::Display for ModelCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Where a capability claim came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityEvidence {
    /// The artifact's metadata suggests it.
    Declared,
    /// A running backend reports offering it.
    BackendReported,
    /// A request of this kind was actually made and answered.
    Probed,
}

impl CapabilityEvidence {
    /// Whether this evidence demonstrates the capability rather than asserting it.
    #[must_use]
    pub fn is_demonstration(self) -> bool {
        matches!(self, Self::Probed)
    }
}

impl fmt::Display for CapabilityEvidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Declared => "declared by the artifact",
            Self::BackendReported => "reported by the backend",
            Self::Probed => "demonstrated by a request",
        })
    }
}

/// How strongly a capability is known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityState {
    /// Nothing establishes it either way.
    Unknown,
    /// Something suggests it, but it has not been demonstrated.
    Indicated { by: CapabilityEvidence },
    /// It has been demonstrated.
    Verified,
}

impl CapabilityState {
    /// Whether the capability has actually been demonstrated.
    ///
    /// Callers that must not guess should branch on this rather than on whether a
    /// capability is merely present in the map.
    #[must_use]
    pub fn is_verified(self) -> bool {
        matches!(self, Self::Verified)
    }
}

impl fmt::Display for CapabilityState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown => f.write_str("unknown"),
            Self::Indicated { by } => write!(f, "indicated ({by})"),
            Self::Verified => f.write_str("verified"),
        }
    }
}

/// What is known about one model's capabilities.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelCapabilities {
    entries: BTreeMap<ModelCapability, CapabilityState>,
}

impl ModelCapabilities {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records that something suggests a capability.
    ///
    /// Never strengthens an existing verified claim: a demonstration outranks any
    /// later assertion.
    pub fn indicate(&mut self, capability: ModelCapability, by: CapabilityEvidence) {
        let entry = self
            .entries
            .entry(capability)
            .or_insert(CapabilityState::Unknown);
        if !entry.is_verified() {
            *entry = CapabilityState::Indicated { by };
        }
    }

    /// Records that a capability was demonstrated.
    ///
    /// Only reachable from an actual probe. Nothing derived from metadata may call
    /// this, which is why it takes no evidence argument: there is only one kind of
    /// evidence that justifies it.
    pub fn verify(&mut self, capability: ModelCapability) {
        self.entries.insert(capability, CapabilityState::Verified);
    }

    /// How strongly a capability is known.
    #[must_use]
    pub fn state(&self, capability: ModelCapability) -> CapabilityState {
        self.entries
            .get(&capability)
            .copied()
            .unwrap_or(CapabilityState::Unknown)
    }

    /// Whether a capability has been demonstrated.
    #[must_use]
    pub fn is_verified(&self, capability: ModelCapability) -> bool {
        self.state(capability).is_verified()
    }

    /// Every capability with something known about it.
    pub fn entries(&self) -> impl Iterator<Item = (ModelCapability, CapabilityState)> + '_ {
        self.entries.iter().map(|(name, state)| (*name, *state))
    }
}

/// Derives what an artifact's own metadata suggests.
///
/// Everything here is at most [`CapabilityState::Indicated`]. Packaging metadata is
/// a claim by whoever produced the file, and a runtime that promoted it to a
/// guarantee would be repeating that claim as its own.
#[must_use]
pub fn indicated_by_metadata(
    architecture: Option<&str>,
    embedding_length: Option<u64>,
    has_chat_template: bool,
) -> ModelCapabilities {
    let mut capabilities = ModelCapabilities::new();

    // A chat template describes how to format a conversation for this model. That
    // is a strong hint about intent and no evidence at all that the backend will
    // serve one.
    if has_chat_template {
        capabilities.indicate(
            ModelCapability::TextGeneration,
            CapabilityEvidence::Declared,
        );
    }

    // An embedding dimension is present on plenty of models that are not embedding
    // models, so on its own it indicates nothing. Only an architecture that exists
    // to produce embeddings is treated as a hint, and only as a hint.
    if let Some(architecture) = architecture
        && architecture.contains("bert")
        && embedding_length.is_some()
    {
        capabilities.indicate(ModelCapability::Embeddings, CapabilityEvidence::Declared);
    }

    capabilities
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_is_known_until_something_says_otherwise() {
        let capabilities = ModelCapabilities::new();
        assert_eq!(
            capabilities.state(ModelCapability::TextGeneration),
            CapabilityState::Unknown
        );
        assert!(!capabilities.is_verified(ModelCapability::Embeddings));
    }

    #[test]
    fn metadata_can_only_indicate_never_verify() {
        // The distinction this whole module exists for. A packaged claim is not a
        // demonstration, and must never be reported as one.
        let capabilities = indicated_by_metadata(Some("llama"), Some(4096), true);
        assert_eq!(
            capabilities.state(ModelCapability::TextGeneration),
            CapabilityState::Indicated {
                by: CapabilityEvidence::Declared
            }
        );
        assert!(!capabilities.is_verified(ModelCapability::TextGeneration));
    }

    #[test]
    fn an_embedding_model_does_not_indicate_text_generation() {
        // The reason an embedding model was chosen as the first real artifact: it
        // must not silently acquire a generation claim.
        let capabilities = indicated_by_metadata(Some("nomic-bert"), Some(768), false);
        assert_eq!(
            capabilities.state(ModelCapability::TextGeneration),
            CapabilityState::Unknown
        );
        assert_eq!(
            capabilities.state(ModelCapability::Embeddings),
            CapabilityState::Indicated {
                by: CapabilityEvidence::Declared
            }
        );
    }

    #[test]
    fn an_architecture_name_alone_verifies_nothing() {
        let capabilities = indicated_by_metadata(Some("nomic-bert"), Some(768), false);
        for capability in [
            ModelCapability::TextGeneration,
            ModelCapability::Embeddings,
            ModelCapability::Vision,
            ModelCapability::ToolCalling,
            ModelCapability::StructuredOutput,
        ] {
            assert!(
                !capabilities.is_verified(capability),
                "{capability} must not be verified by metadata alone"
            );
        }
    }

    #[test]
    fn a_demonstration_outranks_a_later_assertion() {
        let mut capabilities = ModelCapabilities::new();
        capabilities.verify(ModelCapability::Embeddings);
        capabilities.indicate(ModelCapability::Embeddings, CapabilityEvidence::Declared);
        assert!(
            capabilities.is_verified(ModelCapability::Embeddings),
            "an assertion must not downgrade a demonstration"
        );
    }

    #[test]
    fn only_probing_counts_as_a_demonstration() {
        assert!(CapabilityEvidence::Probed.is_demonstration());
        assert!(!CapabilityEvidence::Declared.is_demonstration());
        assert!(!CapabilityEvidence::BackendReported.is_demonstration());
    }

    #[test]
    fn an_embedding_dimension_alone_indicates_nothing() {
        // Plenty of generative models declare one.
        let capabilities = indicated_by_metadata(Some("llama"), Some(4096), false);
        assert_eq!(
            capabilities.state(ModelCapability::Embeddings),
            CapabilityState::Unknown
        );
    }
}
