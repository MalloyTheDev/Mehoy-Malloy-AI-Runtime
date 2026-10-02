//! Joins registered artifacts to execution backends and owns live model instances.
//!
//! This is the only layer that legitimately knows both the registry and a backend.
//! The registry has no dependency on any backend, and no backend depends on the
//! registry, so the translation between them lives here rather than weakening
//! either boundary.
//!
//! # What this layer establishes
//!
//! That a registered artifact still exists and is unchanged, that a backend is
//! willing to attempt it, that the backend loaded it and accepts authenticated
//! requests, and that exactly one live instance results.
//!
//! # What it deliberately does not
//!
//! Claim that any particular kind of request will succeed. A model being resident
//! and a model being able to answer are different facts, tracked separately by
//! [`capability`].
//!
//! # Who owns a live model
//!
//! [`Runtime`] does. A caller receives an [`instance::InstanceId`] and addresses
//! the model by it; the model itself is never handed out. That is what keeps
//! every operation going through admission, and therefore visible to teardown.
//! The loader that builds instances is deliberately not public, so there is one
//! way to have a live model rather than two.

pub mod capability;
pub mod instance;
mod loader;
pub mod runtime;

pub use capability::{CapabilityEvidence, CapabilityState, ModelCapabilities, ModelCapability};
pub use instance::{InstanceFailure, InstanceId, InstanceState, ModelInstance};
pub use loader::{
    CancelError, CancelOutcome, GenerationRequest, LifecyclePhase, LoadError, UnloadOutcome,
};
pub use runtime::{RESIDENT_CAPACITY, Runtime, RuntimeError};

#[cfg(feature = "fault-injection")]
pub use loader::{clear_injected_faults, fail_next_commit};
