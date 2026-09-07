//! Runtime events, split by delivery shape rather than by subject.
//!
//! ADR-0003 records why these are two types and not one. [`InferenceEvent`] is
//! per-request, high frequency, ordered within a request, and meaningless once
//! that request ends. [`RuntimeEvent`] is process-global, infrequent, and matters
//! to subscribers with no request in flight. A single channel carrying both would
//! either over-serve the token path or under-serve the lifecycle path.
//!
//! Neither type is exposed on the wire yet.

use crate::id::{ModelId, RequestId, WorkerId};

/// An event belonging to exactly one inference request.
///
/// Events for a given request are ordered. `Completed` and `Failed` are terminal:
/// no further event for that [`RequestId`] follows either one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferenceEvent {
    /// The request has been admitted and generation is beginning.
    Started { request_id: RequestId },
    /// One generated token. `index` counts from zero within the request.
    Token {
        request_id: RequestId,
        index: u64,
        text: String,
    },
    /// Generation finished normally.
    Completed {
        request_id: RequestId,
        token_count: u64,
    },
    /// Generation ended without completing.
    Failed {
        request_id: RequestId,
        cause: String,
    },
}

impl InferenceEvent {
    /// The request this event belongs to.
    #[must_use]
    pub fn request_id(&self) -> RequestId {
        match self {
            Self::Started { request_id }
            | Self::Token { request_id, .. }
            | Self::Completed { request_id, .. }
            | Self::Failed { request_id, .. } => *request_id,
        }
    }

    /// Whether no further event for this request will follow.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed { .. } | Self::Failed { .. })
    }
}

/// Why a supervised worker process stopped running.
///
/// ADR-0002 requires that an unexpected exit is reported rather than silently
/// restarted, so the cause is carried explicitly instead of being flattened into
/// a boolean.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExitCause {
    /// The worker stopped because the runtime asked it to.
    Requested,
    /// The worker exited on its own with the given status code, if one was reported.
    Unexpected { code: Option<i32> },
    /// The worker failed to become ready within its startup deadline.
    StartupTimeout,
    /// The worker was forcibly terminated after missing its shutdown deadline.
    Killed,
}

impl ExitCause {
    /// Whether this exit was the result of a deliberate runtime action.
    #[must_use]
    pub fn is_expected(&self) -> bool {
        matches!(self, Self::Requested)
    }
}

/// A process-global event, not tied to any single request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeEvent {
    /// A worker process was spawned and reached a ready state.
    WorkerStarted { worker_id: WorkerId },
    /// A worker process is no longer running.
    WorkerExited {
        worker_id: WorkerId,
        cause: ExitCause,
    },
    /// A model became resident and is available for requests.
    ModelLoaded {
        model_id: ModelId,
        worker_id: WorkerId,
    },
    /// A model is no longer resident.
    ModelUnloaded { model_id: ModelId },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req() -> RequestId {
        RequestId::from_raw(42)
    }

    #[test]
    fn every_inference_event_reports_its_request() {
        let events = [
            InferenceEvent::Started { request_id: req() },
            InferenceEvent::Token {
                request_id: req(),
                index: 0,
                text: "hello".into(),
            },
            InferenceEvent::Completed {
                request_id: req(),
                token_count: 1,
            },
            InferenceEvent::Failed {
                request_id: req(),
                cause: "backend gone".into(),
            },
        ];
        for event in &events {
            assert_eq!(event.request_id(), req());
        }
    }

    #[test]
    fn only_completed_and_failed_are_terminal() {
        assert!(!InferenceEvent::Started { request_id: req() }.is_terminal());
        assert!(
            !InferenceEvent::Token {
                request_id: req(),
                index: 0,
                text: String::new(),
            }
            .is_terminal()
        );
        assert!(
            InferenceEvent::Completed {
                request_id: req(),
                token_count: 0,
            }
            .is_terminal()
        );
        assert!(
            InferenceEvent::Failed {
                request_id: req(),
                cause: String::new(),
            }
            .is_terminal()
        );
    }

    #[test]
    fn only_requested_exit_is_expected() {
        assert!(ExitCause::Requested.is_expected());
        assert!(!ExitCause::Unexpected { code: Some(1) }.is_expected());
        assert!(!ExitCause::StartupTimeout.is_expected());
        assert!(!ExitCause::Killed.is_expected());
    }
}
