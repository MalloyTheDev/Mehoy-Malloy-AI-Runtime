//! Model artifact inspection and registration.
//!
//! The runtime reads model containers itself and delegates only execution. That
//! boundary is why this crate has no dependency on any backend: knowing what an
//! artifact is must not require being able to run it, and an artifact inspected
//! here can later be routed to a different backend without being re-understood.
//!
//! # Register, not import
//!
//! Registering records where a file is and what it contains. It does not copy,
//! move, or modify the file. Taking ownership of an artifact into a managed store
//! is a separate operation with different consequences, and does not exist yet.

pub mod artifact;
pub mod gguf;
pub mod store;

pub use artifact::{
    ArtifactFingerprint, ArtifactFormat, ArtifactId, ArtifactIntegrity, ArtifactMetadata,
    ArtifactState, DigestAlgorithm, ModelArtifact, detect_drift,
};
pub use gguf::{GgufError, GgufHeader, MetadataValue};
pub use store::{ArtifactRegistry, Registration, RegistryError};
