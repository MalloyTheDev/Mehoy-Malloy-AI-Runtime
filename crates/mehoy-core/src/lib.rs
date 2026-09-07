//! Shared runtime types and the local endpoint transport for the Mehoy runtime.
//!
//! The event and identifier types are not exposed on the wire yet. They exist now
//! because the separation they encode is expensive to introduce later: see
//! ADR-0003, which records why per-request inference events and global runtime
//! events are kept apart rather than merged into a single event enum.
//!
//! The transport implements ADR-0004: a per-user local endpoint, unreachable from
//! the network by construction, with access control delegated to the operating
//! system rather than reimplemented here.

pub mod event;
pub mod id;
pub mod inference;
pub mod transport;
pub mod worker;

pub use event::{ExitCause, InferenceEvent, RuntimeEvent};
pub use id::{IdAllocator, ModelId, RequestId, WorkerId};
pub use inference::{EmbedRequest, Embedding, EmbeddingResult, Task, TaskId};
pub use transport::{ClientStream, Endpoint, EndpointAddress, EndpointError, Stream};
pub use worker::{ProcessWorker, WorkerHandle, WorkerSpec, WorkerState};
