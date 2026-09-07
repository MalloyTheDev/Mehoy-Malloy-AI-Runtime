//! The Mehoy runtime daemon.
//!
//! This crate is the control plane described by ADR-0001: the daemon is the
//! system, and every client is equal. It listens on the local endpoint defined by
//! ADR-0004 and serves the minimal protocol defined by ADR-0003.
//!
//! No inference, model registry, or worker supervision exists yet. What is proven
//! here is the part that must be right before any of that arrives: that the
//! endpoint is created securely, that a second daemon cannot quietly take it over,
//! that a stale endpoint is recovered rather than blindly deleted, and that a
//! clean shutdown leaves nothing behind.

pub mod server;
pub mod service;

pub use server::{DRAIN_DEADLINE, serve};
pub use service::{RUNTIME_NAME, RUNTIME_VERSION, runtime_info};
