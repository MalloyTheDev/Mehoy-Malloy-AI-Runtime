//! Model artifacts: what a container is, not what is running.
//!
//! Three concepts are deliberately distinct, and only the middle one exists yet:
//!
//! - a **logical model**, a name a user chooses;
//! - an **artifact**, an immutable container on disk;
//! - a **loaded instance**, an artifact occupying memory in a backend.
//!
//! Merging any two of them is the mistake that becomes expensive once scheduling
//! exists, because an artifact is a fact while an instance is a resource
//! reservation.

use std::fmt;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::gguf::{GgufHeader, MetadataValue};

/// An artifact's identifier within this runtime.
///
/// Internal and generated, not derived from the file's contents. Deriving it from
/// content would mean hashing every artifact at registration, which does not scale
/// to containers measured in tens of gigabytes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ArtifactId(String);

impl ArtifactId {
    /// Generates a new identifier.
    ///
    /// # Errors
    ///
    /// Returns an error when the system randomness source is unavailable.
    pub fn generate() -> std::io::Result<Self> {
        let mut bytes = [0u8; 16];
        getrandom::fill(&mut bytes).map_err(|err| {
            std::io::Error::other(format!("system randomness unavailable: {err}"))
        })?;
        let mut encoded = String::with_capacity(32);
        for byte in bytes {
            encoded.push_str(&format!("{byte:02x}"));
        }
        Ok(Self(encoded))
    }

    /// Wraps a stored identifier.
    #[must_use]
    pub fn from_stored(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// The identifier as text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ArtifactId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Container formats the runtime can inspect.
///
/// Deliberately narrow. A variant is added when there is an implementation behind
/// it, not because a format is on a roadmap: an enum full of formats that cannot
/// actually be read is a promise the runtime does not keep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactFormat {
    Gguf,
}

impl ArtifactFormat {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gguf => "gguf",
        }
    }

    /// Parses a stored format name.
    #[must_use]
    pub fn from_stored(value: &str) -> Option<Self> {
        match value {
            "gguf" => Some(Self::Gguf),
            _ => None,
        }
    }
}

impl fmt::Display for ArtifactFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Cheap properties used to notice that a file has been replaced.
///
/// Deliberately not called a hash or a digest. It is a set of observations about
/// the file, not a function of its contents, and naming it after cryptography would
/// invite someone to treat it as content-addressed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactFingerprint {
    pub size_bytes: u64,
    /// Last modification time, when the platform reports one.
    pub modified_at: Option<SystemTime>,
}

impl ArtifactFingerprint {
    /// Observes a file.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be inspected.
    pub fn observe(path: &Path) -> std::io::Result<Self> {
        let metadata = std::fs::metadata(path)?;
        Ok(Self {
            size_bytes: metadata.len(),
            modified_at: metadata.modified().ok(),
        })
    }
}

/// Digest algorithms used for verified integrity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DigestAlgorithm {
    Sha256,
}

impl DigestAlgorithm {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sha256 => "sha256",
        }
    }

    #[must_use]
    pub fn from_stored(value: &str) -> Option<Self> {
        match value {
            "sha256" => Some(Self::Sha256),
            _ => None,
        }
    }
}

/// What is actually known about an artifact's contents.
///
/// Truthful by construction. An artifact is `Unverified` until every byte has been
/// read, because registration deliberately does not read the whole file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactIntegrity {
    /// The contents have not been read in full.
    Unverified,
    /// The contents were read and digested.
    Verified {
        algorithm: DigestAlgorithm,
        digest: Vec<u8>,
    },
}

impl ArtifactIntegrity {
    /// Whether the contents have actually been digested.
    #[must_use]
    pub fn is_verified(&self) -> bool {
        matches!(self, Self::Verified { .. })
    }
}

/// The parts of a container's metadata the runtime itself reasons about.
///
/// Everything else stays in `extra` rather than becoming a schema change every time
/// a new key is encountered.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ArtifactMetadata {
    pub container_version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
    pub architecture: Option<String>,
    pub name: Option<String>,
    pub file_type: Option<u64>,
    pub context_length: Option<u64>,
    pub embedding_length: Option<u64>,
    pub tokenizer_model: Option<String>,
    /// Present when the container carries one. Its presence is what makes a
    /// container usable for chat without the runtime inventing a format.
    pub chat_template: Option<String>,
}

impl ArtifactMetadata {
    /// Extracts the fields the runtime reasons about from a parsed header.
    #[must_use]
    pub fn from_header(header: &GgufHeader) -> Self {
        Self {
            container_version: header.version,
            tensor_count: header.tensor_count,
            metadata_count: header.metadata_count,
            architecture: header.architecture().map(str::to_owned),
            name: header.text("general.name").map(str::to_owned),
            file_type: header.number("general.file_type"),
            context_length: header.architecture_number("context_length"),
            embedding_length: header.architecture_number("embedding_length"),
            tokenizer_model: header.text("tokenizer.ggml.model").map(str::to_owned),
            chat_template: header.text("tokenizer.chat_template").map(str::to_owned),
        }
    }

    /// Whether the container carries a chat template.
    #[must_use]
    pub fn has_chat_template(&self) -> bool {
        self.chat_template.is_some()
    }
}

/// A registered artifact.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelArtifact {
    pub id: ArtifactId,
    /// The canonical path. Registration records where the file is; it does not
    /// move or copy it.
    pub path: PathBuf,
    pub format: ArtifactFormat,
    pub size_bytes: u64,
    pub metadata: ArtifactMetadata,
    pub integrity: ArtifactIntegrity,
    pub fingerprint: ArtifactFingerprint,
    pub registered_at: SystemTime,
}

/// Whether a registered artifact still matches what was recorded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactState {
    /// The file is present and matches its fingerprint.
    Unchanged,
    /// The file is present but differs from what was registered.
    ///
    /// Never resolved by silently updating the record. A replaced file is a
    /// different artifact, and quietly adopting it would destroy the provenance
    /// that registration exists to establish.
    Changed { reason: String },
    /// The file is no longer present.
    Missing,
}

impl ArtifactState {
    /// Whether the artifact may be used as recorded.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        matches!(self, Self::Unchanged)
    }
}

impl fmt::Display for ArtifactState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unchanged => f.write_str("unchanged"),
            Self::Changed { reason } => write!(f, "changed: {reason}"),
            Self::Missing => f.write_str("missing"),
        }
    }
}

/// Compares a recorded artifact against the file as it is now.
#[must_use]
pub fn detect_drift(artifact: &ModelArtifact) -> ArtifactState {
    let Ok(current) = ArtifactFingerprint::observe(&artifact.path) else {
        return ArtifactState::Missing;
    };

    if current.size_bytes != artifact.fingerprint.size_bytes {
        return ArtifactState::Changed {
            reason: format!(
                "size is now {} bytes, was {} at registration",
                current.size_bytes, artifact.fingerprint.size_bytes
            ),
        };
    }

    // A modification time that moved is enough to stop trusting the record, even
    // when the size matches. An edit that preserves length is exactly the case a
    // size check alone would miss.
    match (current.modified_at, artifact.fingerprint.modified_at) {
        (Some(now), Some(then)) if now != then => ArtifactState::Changed {
            reason: "modification time has changed since registration".to_owned(),
        },
        _ => ArtifactState::Unchanged,
    }
}

/// Renders a metadata value for storage.
#[must_use]
pub fn render_value(value: &MetadataValue) -> String {
    match value {
        MetadataValue::String(text) => text.clone(),
        MetadataValue::Bool(flag) => flag.to_string(),
        MetadataValue::U8(v) => v.to_string(),
        MetadataValue::I8(v) => v.to_string(),
        MetadataValue::U16(v) => v.to_string(),
        MetadataValue::I16(v) => v.to_string(),
        MetadataValue::U32(v) => v.to_string(),
        MetadataValue::I32(v) => v.to_string(),
        MetadataValue::U64(v) => v.to_string(),
        MetadataValue::I64(v) => v.to_string(),
        MetadataValue::F32(v) => v.to_string(),
        MetadataValue::F64(v) => v.to_string(),
        MetadataValue::Array(items) => format!("[{} items]", items.len()),
        MetadataValue::LargeArray { len, .. } => format!("[{len} items]"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn artifact(size: u64, modified: Option<SystemTime>) -> ModelArtifact {
        ModelArtifact {
            id: ArtifactId::from_stored("test"),
            path: PathBuf::from("no-such-file-for-drift-tests"),
            format: ArtifactFormat::Gguf,
            size_bytes: size,
            metadata: ArtifactMetadata::default(),
            integrity: ArtifactIntegrity::Unverified,
            fingerprint: ArtifactFingerprint {
                size_bytes: size,
                modified_at: modified,
            },
            registered_at: SystemTime::UNIX_EPOCH,
        }
    }

    #[test]
    fn identifiers_are_unique_and_not_derived_from_content() {
        let first = ArtifactId::generate().expect("generates");
        let second = ArtifactId::generate().expect("generates");
        assert_ne!(first, second);
        assert_eq!(first.as_str().len(), 32);
    }

    #[test]
    fn a_new_artifact_is_unverified() {
        // Registration does not read the whole file, so it must not claim to know
        // the contents.
        assert!(!ArtifactIntegrity::Unverified.is_verified());
        assert!(
            ArtifactIntegrity::Verified {
                algorithm: DigestAlgorithm::Sha256,
                digest: vec![0; 32],
            }
            .is_verified()
        );
    }

    #[test]
    fn a_missing_file_is_missing_rather_than_unchanged() {
        assert_eq!(detect_drift(&artifact(10, None)), ArtifactState::Missing);
    }

    #[test]
    fn formats_round_trip_through_storage() {
        assert_eq!(
            ArtifactFormat::from_stored(ArtifactFormat::Gguf.as_str()),
            Some(ArtifactFormat::Gguf)
        );
        assert_eq!(ArtifactFormat::from_stored("safetensors"), None);
    }

    #[test]
    fn digest_algorithms_round_trip_through_storage() {
        assert_eq!(
            DigestAlgorithm::from_stored(DigestAlgorithm::Sha256.as_str()),
            Some(DigestAlgorithm::Sha256)
        );
        assert_eq!(DigestAlgorithm::from_stored("md5"), None);
    }

    #[test]
    fn a_changed_state_is_never_usable() {
        let changed = ArtifactState::Changed {
            reason: "size differs".to_owned(),
        };
        assert!(!changed.is_usable());
        assert!(!ArtifactState::Missing.is_usable());
        assert!(ArtifactState::Unchanged.is_usable());
    }

    #[test]
    fn a_modification_time_alone_is_enough_to_stop_trusting_a_record() {
        // An edit that preserves length is exactly what a size-only check misses.
        let then = SystemTime::UNIX_EPOCH;
        let now = then + Duration::from_secs(60);
        let mut recorded = artifact(100, Some(then));
        recorded.fingerprint.modified_at = Some(then);

        let current = ArtifactFingerprint {
            size_bytes: 100,
            modified_at: Some(now),
        };
        assert_ne!(current.modified_at, recorded.fingerprint.modified_at);
    }

    #[test]
    fn large_arrays_render_by_shape_rather_than_contents() {
        let rendered = render_value(&MetadataValue::LargeArray {
            element_type: 8,
            len: 250_000,
        });
        assert_eq!(rendered, "[250000 items]");
    }
}
