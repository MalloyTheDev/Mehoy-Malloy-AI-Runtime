//! Whether this backend can run a given artifact, decided before spawning it.
//!
//! A container format is not a capability declaration. "It is GGUF, therefore
//! llama.cpp can run it" is wrong often enough to matter: the format carries
//! embedding models, multimodal projectors, and architectures a given build does
//! not implement, all in the same container.
//!
//! The preflight is deliberately allowed to answer [`Compatibility::Unknown`].
//! Metadata is frequently insufficient to decide, and inventing certainty from a
//! partial signal is worse than admitting the limit: an incorrect `Unsupported`
//! refuses a model that would have worked, and an incorrect `Supported` turns a
//! clear preflight refusal into a confusing runtime failure.

use std::fmt;

/// The facts about an artifact this backend uses to make a preflight decision.
///
/// A plain description rather than the registry's own type, so this crate does not
/// depend on the registry. The two are joined a layer above, which is where
/// knowledge of both belongs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelDescriptor {
    pub architecture: Option<String>,
    pub context_length: Option<u64>,
    pub embedding_length: Option<u64>,
    pub tokenizer_model: Option<String>,
    pub has_chat_template: bool,
}

/// Whether the backend expects to be able to run an artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Compatibility {
    /// The backend is expected to run this artifact.
    Supported,
    /// The backend is known not to run this artifact. Do not spawn.
    Unsupported { reason: String },
    /// The available evidence does not decide it.
    ///
    /// Spawning is still allowed. The backend itself is the authority, and its
    /// refusal is a clear, evidenced failure; a guess here would be neither.
    Unknown { reason: String },
}

impl Compatibility {
    /// Whether starting the backend is worth attempting.
    #[must_use]
    pub fn permits_start(&self) -> bool {
        !matches!(self, Self::Unsupported { .. })
    }
}

impl fmt::Display for Compatibility {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Supported => f.write_str("supported"),
            Self::Unsupported { reason } => write!(f, "unsupported: {reason}"),
            Self::Unknown { reason } => write!(f, "undetermined: {reason}"),
        }
    }
}

/// How the backend should be started for a given artifact.
///
/// Decided here rather than by a caller, because the flag that selects it is this
/// engine's vocabulary. A runtime that had to know the flag exists would be
/// carrying backend detail it has no business holding, and a second backend would
/// need the caller changed rather than only the adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaunchMode {
    /// The engine's ordinary mode.
    General,
    /// Dedicated embedding mode.
    ///
    /// Measured on build 9010: without it the embeddings endpoint answers `501`
    /// with `not_supported_error`, so an embedding model started in the general
    /// mode loads successfully and then refuses every request it exists to serve.
    Embedding,
}

/// Chooses how to start the backend for an artifact.
///
/// A heuristic, and deliberately a conservative one: choosing embedding mode for a
/// generative model would disable generation, so only an architecture that exists
/// to produce embeddings and carries no conversation template qualifies. A wrong
/// answer surfaces as a clear refusal from the engine rather than as silence.
#[must_use]
pub fn launch_mode(descriptor: &ModelDescriptor) -> LaunchMode {
    let embedding_oriented = descriptor
        .architecture
        .as_deref()
        .is_some_and(|architecture| {
            EMBEDDING_ORIENTED
                .iter()
                .any(|family| architecture.contains(family))
        });

    if embedding_oriented && !descriptor.has_chat_template {
        LaunchMode::Embedding
    } else {
        LaunchMode::General
    }
}

/// Architecture families that exist to produce embeddings.
const EMBEDDING_ORIENTED: [&str; 1] = ["bert"];

/// Architectures this backend is known not to serve as a standalone model.
///
/// A multimodal projector is a companion to another model rather than something
/// that can be loaded on its own, so starting a backend for one would always fail.
/// This is the one case the evidence decides cleanly.
const NOT_STANDALONE: [&str; 1] = ["clip"];

/// Decides whether to attempt a start.
///
/// Note what this does not do: it does not infer generation capability from the
/// architecture name. An architecture tells you what the tensors are, not what the
/// backend will let you ask of them.
#[must_use]
pub fn assess(descriptor: &ModelDescriptor) -> Compatibility {
    let Some(architecture) = descriptor.architecture.as_deref() else {
        return Compatibility::Unknown {
            reason: "the container declares no architecture".to_owned(),
        };
    };

    if NOT_STANDALONE.contains(&architecture) {
        return Compatibility::Unsupported {
            reason: format!(
                "{architecture} is a companion projector rather than a standalone model"
            ),
        };
    }

    if descriptor.tokenizer_model.is_none() {
        return Compatibility::Unknown {
            reason: format!(
                "{architecture} declares no tokenizer, so whether it can be served alone \
                 is undetermined"
            ),
        };
    }

    // Everything else is left to the backend. Maintaining a list of architectures
    // this build supports would be wrong within a release of upstream adding one.
    Compatibility::Unknown {
        reason: format!("{architecture} is not known to be unsupported; the backend decides"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(architecture: &str) -> ModelDescriptor {
        ModelDescriptor {
            architecture: Some(architecture.to_owned()),
            tokenizer_model: Some("gpt2".to_owned()),
            ..ModelDescriptor::default()
        }
    }

    #[test]
    fn a_projector_is_refused_before_anything_is_spawned() {
        // Loading one alone always fails, so spending a process start to discover
        // that is waste.
        let verdict = assess(&descriptor("clip"));
        match &verdict {
            Compatibility::Unsupported { reason } => {
                assert!(reason.contains("projector"), "{reason}")
            }
            other => panic!("expected Unsupported, got {other}"),
        }
        assert!(!verdict.permits_start());
    }

    #[test]
    fn an_unfamiliar_architecture_is_undetermined_rather_than_refused() {
        // Refusing an unknown architecture would break the moment upstream adds
        // one, which happens often.
        let verdict = assess(&descriptor("some-future-architecture"));
        assert!(
            matches!(verdict, Compatibility::Unknown { .. }),
            "expected Unknown, got {verdict}"
        );
        assert!(
            verdict.permits_start(),
            "an undetermined verdict must still allow a start"
        );
    }

    #[test]
    fn an_embedding_architecture_is_not_refused() {
        // An embedding model loads perfectly well. Whether it can generate text is
        // a different question this preflight deliberately does not answer.
        let verdict = assess(&descriptor("nomic-bert"));
        assert!(verdict.permits_start());
    }

    #[test]
    fn a_missing_architecture_is_undetermined() {
        let verdict = assess(&ModelDescriptor::default());
        match &verdict {
            Compatibility::Unknown { reason } => {
                assert!(reason.contains("architecture"), "{reason}")
            }
            other => panic!("expected Unknown, got {other}"),
        }
        assert!(verdict.permits_start());
    }

    #[test]
    fn a_chat_template_is_not_treated_as_a_capability_claim() {
        // Carrying a chat template says how to format a conversation, not that the
        // backend will serve one.
        let with_template = ModelDescriptor {
            has_chat_template: true,
            ..descriptor("llama")
        };
        let without = descriptor("llama");
        assert_eq!(
            assess(&with_template).permits_start(),
            assess(&without).permits_start()
        );
    }
}
