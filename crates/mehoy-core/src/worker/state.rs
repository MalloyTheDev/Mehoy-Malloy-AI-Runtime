//! The worker state machine.
//!
//! ```text
//!   Absent
//!     |
//!     v
//!   Starting ----- startup failure -----> Failed
//!     |
//!     v
//!   Ready
//!     |\
//!     | \--- unexpected exit ------------> Failed
//!     |
//!     v
//!   Draining
//!     |
//!     v
//!   Stopping
//!     |
//!     v
//!   Absent
//! ```
//!
//! `Failed` is terminal and deliberately has no edge back to `Starting`. ADR-0002
//! requires that an unexpected exit is reported and recovered from explicitly. An
//! automatic restart here would hide defects at exactly the point in the project
//! where they most need to be visible, and there is no observability yet to notice
//! a worker quietly restarting in a loop.

use std::fmt;

/// Where a supervised worker is in its life.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkerState {
    /// No process exists.
    Absent,
    /// A process exists but has not reported readiness.
    Starting,
    /// The process reported readiness and may be used.
    Ready,
    /// The process was asked to stop and is finishing outstanding work.
    Draining,
    /// The process is being stopped.
    Stopping,
    /// The process ended in a way that requires explicit recovery.
    Failed,
}

impl WorkerState {
    /// Whether the worker can serve work.
    #[must_use]
    pub fn is_usable(self) -> bool {
        matches!(self, Self::Ready)
    }

    /// Whether no further transition will happen without outside intervention.
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Failed)
    }

    /// Whether moving to `next` is allowed.
    #[must_use]
    pub fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Absent, Self::Starting)
                | (Self::Starting, Self::Ready | Self::Failed)
                | (Self::Ready, Self::Draining | Self::Failed)
                | (Self::Draining, Self::Stopping | Self::Failed)
                | (Self::Stopping, Self::Absent | Self::Failed)
        )
    }

    /// Moves to `next`, rejecting transitions the machine does not allow.
    ///
    /// # Errors
    ///
    /// Returns [`WorkerStateError`] when the transition is not permitted.
    pub fn transition_to(self, next: Self) -> Result<Self, WorkerStateError> {
        if self.can_transition_to(next) {
            Ok(next)
        } else {
            Err(WorkerStateError {
                from: self,
                to: next,
            })
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Absent => "absent",
            Self::Starting => "starting",
            Self::Ready => "ready",
            Self::Draining => "draining",
            Self::Stopping => "stopping",
            Self::Failed => "failed",
        }
    }
}

impl fmt::Display for WorkerState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A transition the state machine does not permit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerStateError {
    pub from: WorkerState,
    pub to: WorkerState,
}

impl fmt::Display for WorkerStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "a worker cannot go from {} to {}", self.from, self.to)
    }
}

impl std::error::Error for WorkerStateError {}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [WorkerState; 6] = [
        WorkerState::Absent,
        WorkerState::Starting,
        WorkerState::Ready,
        WorkerState::Draining,
        WorkerState::Stopping,
        WorkerState::Failed,
    ];

    #[test]
    fn the_happy_path_is_permitted_end_to_end() {
        let mut state = WorkerState::Absent;
        for next in [
            WorkerState::Starting,
            WorkerState::Ready,
            WorkerState::Draining,
            WorkerState::Stopping,
            WorkerState::Absent,
        ] {
            state = state.transition_to(next).expect("permitted transition");
        }
        assert_eq!(state, WorkerState::Absent);
    }

    #[test]
    fn failed_is_terminal_and_has_no_way_back() {
        assert!(WorkerState::Failed.is_terminal());
        for next in ALL {
            assert!(
                !WorkerState::Failed.can_transition_to(next),
                "failed must not transition to {next}"
            );
        }
    }

    #[test]
    fn a_worker_never_restarts_itself() {
        // ADR-0002 requires explicit recovery rather than automatic restart. A
        // direct edge from Failed back into Starting would be that restart.
        assert!(!WorkerState::Failed.can_transition_to(WorkerState::Starting));
        assert!(!WorkerState::Failed.can_transition_to(WorkerState::Ready));
    }

    #[test]
    fn only_ready_is_usable() {
        for state in ALL {
            assert_eq!(
                state.is_usable(),
                state == WorkerState::Ready,
                "{state} usability"
            );
        }
    }

    #[test]
    fn a_worker_cannot_become_ready_without_starting() {
        assert!(!WorkerState::Absent.can_transition_to(WorkerState::Ready));
        let err = WorkerState::Absent
            .transition_to(WorkerState::Ready)
            .expect_err("must be rejected");
        assert_eq!(err.from, WorkerState::Absent);
        assert_eq!(err.to, WorkerState::Ready);
    }

    #[test]
    fn shutdown_cannot_skip_draining() {
        assert!(!WorkerState::Ready.can_transition_to(WorkerState::Stopping));
        assert!(!WorkerState::Ready.can_transition_to(WorkerState::Absent));
    }

    #[test]
    fn every_state_except_absent_and_failed_can_fail() {
        // A worker can die at any point once a process exists. Absent has no
        // process to lose, and Failed is already there.
        for state in [
            WorkerState::Starting,
            WorkerState::Ready,
            WorkerState::Draining,
            WorkerState::Stopping,
        ] {
            assert!(
                state.can_transition_to(WorkerState::Failed),
                "{state} must be able to fail"
            );
        }
        assert!(!WorkerState::Absent.can_transition_to(WorkerState::Failed));
    }

    #[test]
    fn rejected_transitions_report_both_ends() {
        let err = WorkerState::Starting
            .transition_to(WorkerState::Draining)
            .expect_err("must be rejected");
        let message = err.to_string();
        assert!(message.contains("starting"), "{message}");
        assert!(message.contains("draining"), "{message}");
    }
}
