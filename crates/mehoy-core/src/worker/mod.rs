//! Backend worker lifecycle.
//!
//! ADR-0002 puts execution engines in separate processes and makes the runtime
//! responsible for their lifetime. This module answers one question and no others:
//! can the runtime reliably own the lifetime of another process?
//!
//! There is deliberately nothing here about models, generation, prompts, or any
//! particular engine. A worker is a process that starts, becomes ready, runs, and
//! stops. What it does in between is not this layer's concern, and adding that
//! vocabulary now would be the speculative design ADR-0003 rules out.
//!
//! The supervision contract ADR-0002 left open is specified here as deadlines and
//! an explicit state machine rather than as prose.

pub mod process;
pub mod state;

pub use process::{ProcessWorker, WorkerHandle};
pub use state::{WorkerState, WorkerStateError};

use std::fmt;
use std::path::PathBuf;
use std::time::Duration;

use crate::event::ExitCause;
use crate::id::WorkerId;

/// How a worker is started.
///
/// A readiness probe is required rather than optional. ADR-0002 records that a
/// running process proves only that `exec` succeeded, which is not readiness.
#[derive(Debug, Clone)]
pub struct WorkerSpec {
    pub id: WorkerId,
    /// Program to run.
    pub program: PathBuf,
    /// Arguments passed to the program.
    pub args: Vec<String>,
    /// Deadlines governing the worker's lifetime.
    pub deadlines: Deadlines,
}

/// The three deadlines a supervised worker is held to.
///
/// These are deliberately separate numbers. ADR-0002 records why a single
/// universal timeout is wrong: legitimate durations differ by orders of magnitude
/// between phases, so collapsing them either kills healthy work or waits forever
/// on dead work.
///
/// Note what is absent. There is no request or generation deadline here, because
/// this layer has no concept of a request, and because a large model doing a long
/// prefill and a small model answering interactively cannot share one number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Deadlines {
    /// How long a worker may take to report readiness before it is treated as
    /// failed to start.
    pub startup: Duration,
    /// How long a worker is given to stop after being asked politely.
    pub shutdown: Duration,
    /// How long a readiness probe may take before it counts as unanswered.
    pub health: Duration,
}

impl Default for Deadlines {
    fn default() -> Self {
        Self {
            startup: Duration::from_secs(30),
            shutdown: Duration::from_secs(10),
            health: Duration::from_secs(5),
        }
    }
}

/// What a worker reported when it became ready.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerReady {
    /// The line the worker emitted to announce readiness. Opaque at this layer.
    pub announcement: String,
}

/// How a worker's life ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerExit {
    pub cause: ExitCause,
}

/// Why a supervision operation failed.
#[derive(Debug)]
pub enum WorkerError {
    /// The process could not be started at all.
    Spawn {
        program: PathBuf,
        source: std::io::Error,
    },
    /// The worker did not report readiness within its startup deadline.
    StartupTimeout { waited: Duration },
    /// The worker exited before reporting readiness.
    ExitedDuringStartup { code: Option<i32> },
    /// The worker ignored a polite stop and had to be killed.
    ///
    /// This is reported rather than swallowed. A worker that has to be killed is
    /// a fact an operator needs, not an implementation detail.
    ShutdownTimeout { waited: Duration },
    /// The operation is not valid for the worker's current state.
    InvalidState(WorkerStateError),
    /// An underlying operating system failure.
    Io(std::io::Error),
}

impl fmt::Display for WorkerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spawn { program, source } => {
                write!(f, "cannot start worker {}: {source}", program.display())
            }
            Self::StartupTimeout { waited } => write!(
                f,
                "worker did not become ready within {}s",
                waited.as_secs_f32()
            ),
            Self::ExitedDuringStartup { code } => write!(
                f,
                "worker exited before becoming ready (status {})",
                code.map_or_else(|| "unknown".to_owned(), |code| code.to_string())
            ),
            Self::ShutdownTimeout { waited } => write!(
                f,
                "worker ignored shutdown for {}s and was terminated",
                waited.as_secs_f32()
            ),
            Self::InvalidState(err) => write!(f, "{err}"),
            Self::Io(err) => write!(f, "worker supervision failure: {err}"),
        }
    }
}

impl std::error::Error for WorkerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Spawn { source, .. } | Self::Io(source) => Some(source),
            _ => None,
        }
    }
}

impl From<WorkerStateError> for WorkerError {
    fn from(err: WorkerStateError) -> Self {
        Self::InvalidState(err)
    }
}

impl From<std::io::Error> for WorkerError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}
