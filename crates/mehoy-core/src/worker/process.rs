//! Supervising a worker as a child process.
//!
//! # Readiness
//!
//! A worker announces readiness by printing [`READY_MARKER`] on standard output.
//! ADR-0002 requires a real readiness condition: a live process proves only that
//! `exec` succeeded, which says nothing about whether the worker can do anything.
//!
//! This line protocol is provisional and exists so the supervision machinery can be
//! proven against a controllable worker. A real engine will need its own probe.
//!
//! # Stopping
//!
//! A polite stop is a `shutdown` line on the worker's standard input, which works
//! identically on every platform. Windows has no signal to send a specific process
//! for this purpose, and inventing a platform split for a stop request would put
//! the difference in the wrong place.
//!
//! A worker that ignores the request past its deadline is killed, and that is
//! reported rather than swallowed.
//!
//! # Orphans
//!
//! No worker may outlive the daemon that owns it. A leaked worker holds
//! accelerator memory indefinitely with nothing left to reclaim it.
//!
//! Windows has no parent-death signal, so a child survives an abruptly terminated
//! parent by default. The mechanism used here is a Job Object with
//! kill-on-job-close: the daemon holds the job handle, and when the daemon dies by
//! any means the handle closes and the operating system terminates everything in
//! the job.
//!
//! Linux uses a parent-death signal, requested by the child between fork and exec.
//! Other Unix targets have no equivalent and fall back to killing the process
//! group on explicit shutdown, which does not cover abrupt daemon death.

use std::process::Stdio;
use std::time::Instant;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdout, Command};

use super::{Deadlines, WorkerError, WorkerExit, WorkerReady, WorkerSpec, WorkerState};
use crate::event::ExitCause;
use crate::id::WorkerId;

/// The line a worker prints on standard output to announce readiness.
pub const READY_MARKER: &str = "MEHOY-WORKER-READY";

/// The line written to a worker's standard input to request a polite stop.
pub const SHUTDOWN_COMMAND: &str = "shutdown";

#[cfg(windows)]
mod job {
    //! Windows Job Object holding a worker, with kill-on-job-close.

    use std::io;

    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
        SetInformationJobObject,
    };

    /// A job whose closure kills every process still assigned to it.
    ///
    /// The handle is held for as long as the worker should live. If the owning
    /// process dies for any reason, including being terminated without warning,
    /// the operating system closes the handle and reaps the job's processes.
    #[derive(Debug)]
    pub struct Job {
        handle: HANDLE,
    }

    // Safety: the handle is an opaque kernel object reference. It is not
    // dereferenced, is used only through thread-safe Win32 calls, and is closed
    // exactly once in `Drop`.
    unsafe impl Send for Job {}
    unsafe impl Sync for Job {}

    impl Job {
        /// Creates an unnamed job configured to kill its processes when closed.
        pub fn create() -> io::Result<Self> {
            // Safety: an unnamed job with default security attributes.
            let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
            if handle.is_null() {
                return Err(io::Error::last_os_error());
            }
            let job = Self { handle };

            let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION =
                // Safety: the structure is plain data with no invalid bit patterns,
                // and every field the call reads is set below.
                unsafe { std::mem::zeroed() };
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

            // Safety: `limits` matches the information class being set, and its
            // size is passed alongside it.
            let ok = unsafe {
                SetInformationJobObject(
                    job.handle,
                    JobObjectExtendedLimitInformation,
                    std::ptr::from_ref(&limits).cast(),
                    u32::try_from(size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>())
                        .expect("job limit structure size fits in u32"),
                )
            };
            if ok == 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(job)
        }

        /// Puts a running process into this job.
        pub fn assign(&self, process: HANDLE) -> io::Result<()> {
            // Safety: both handles are valid and owned by the caller.
            let ok = unsafe { AssignProcessToJobObject(self.handle, process) };
            if ok == 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        }
    }

    impl Drop for Job {
        fn drop(&mut self) {
            // Safety: created by `CreateJobObjectW` and closed once.
            unsafe { CloseHandle(self.handle) };
        }
    }
}

/// A running worker and everything needed to supervise it.
#[derive(Debug)]
pub struct WorkerHandle {
    id: WorkerId,
    state: WorkerState,
    child: Option<Child>,
    stdout: Option<Lines<BufReader<ChildStdout>>>,
    deadlines: Deadlines,
    /// Held for the worker's lifetime so that losing the daemon reaps the worker.
    #[cfg(windows)]
    _job: job::Job,
}

impl WorkerHandle {
    /// The worker's identifier.
    #[must_use]
    pub fn id(&self) -> WorkerId {
        self.id
    }

    /// Where the worker is in its lifecycle.
    #[must_use]
    pub fn state(&self) -> WorkerState {
        self.state
    }

    /// The operating system process identifier, while a process exists.
    #[must_use]
    pub fn pid(&self) -> Option<u32> {
        self.child.as_ref().and_then(Child::id)
    }
}

/// Supervises workers that are child processes.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProcessWorker;

impl ProcessWorker {
    /// Starts a worker.
    ///
    /// The process is placed under the runtime's lifetime control before this
    /// returns, so a worker is never briefly outside supervision.
    ///
    /// # Errors
    ///
    /// Returns [`WorkerError::Spawn`] when the process cannot be started.
    pub async fn spawn(&self, spec: WorkerSpec) -> Result<WorkerHandle, WorkerError> {
        let state = WorkerState::Absent.transition_to(WorkerState::Starting)?;

        #[cfg(windows)]
        let job = job::Job::create().map_err(WorkerError::Io)?;

        let mut command = Command::new(&spec.program);
        command
            .args(&spec.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);

        #[cfg(unix)]
        configure_unix_lifetime(&mut command);

        let mut child = command.spawn().map_err(|source| WorkerError::Spawn {
            program: spec.program.clone(),
            source,
        })?;

        #[cfg(windows)]
        {
            // Assigning immediately after spawn leaves a very small interval in
            // which the child is running but unassigned. Closing it entirely
            // requires starting the process suspended, which is not yet done.
            let handle = child
                .raw_handle()
                .ok_or_else(|| WorkerError::Io(std::io::Error::other("worker has no handle")))?;
            job.assign(handle.cast()).map_err(|err| {
                // A worker that cannot be brought under lifetime control must not
                // be left running: that is precisely the orphan being prevented.
                let _ = child.start_kill();
                WorkerError::Io(err)
            })?;
        }

        let stdout = child
            .stdout
            .take()
            .map(|out| BufReader::new(out).lines())
            .ok_or_else(|| WorkerError::Io(std::io::Error::other("worker stdout unavailable")))?;

        Ok(WorkerHandle {
            id: spec.id,
            state,
            child: Some(child),
            stdout: Some(stdout),
            deadlines: spec.deadlines,
            #[cfg(windows)]
            _job: job,
        })
    }

    /// Waits for the worker to announce readiness.
    ///
    /// Readiness is an announcement from the worker, not the mere existence of a
    /// process. A worker that exits during startup is reported as such rather than
    /// waiting out the full deadline for something that can no longer happen.
    ///
    /// # Errors
    ///
    /// Returns [`WorkerError::StartupTimeout`] when the deadline passes and
    /// [`WorkerError::ExitedDuringStartup`] when the process ends first.
    pub async fn wait_ready(&self, handle: &mut WorkerHandle) -> Result<WorkerReady, WorkerError> {
        let started = Instant::now();
        let budget = handle.deadlines.startup;
        let mut lines = handle
            .stdout
            .take()
            .ok_or_else(|| WorkerError::Io(std::io::Error::other("worker stdout already taken")))?;
        let child = handle
            .child
            .as_mut()
            .ok_or_else(|| WorkerError::Io(std::io::Error::other("worker has no process")))?;

        let outcome = tokio::time::timeout(budget, async {
            loop {
                tokio::select! {
                    line = lines.next_line() => match line {
                        Ok(Some(line)) if line.trim_start().starts_with(READY_MARKER) => {
                            return Ok(WorkerReady { announcement: line });
                        }
                        // Any other output is the worker's own business.
                        Ok(Some(_)) => continue,
                        // Standard output closed without a readiness line. Wait for
                        // the exit status rather than guessing at a cause.
                        Ok(None) => {
                            let status = child.wait().await.map_err(WorkerError::Io)?;
                            return Err(WorkerError::ExitedDuringStartup { code: status.code() });
                        }
                        Err(err) => return Err(WorkerError::Io(err)),
                    },
                    status = child.wait() => {
                        let status = status.map_err(WorkerError::Io)?;
                        return Err(WorkerError::ExitedDuringStartup { code: status.code() });
                    }
                }
            }
        })
        .await;

        handle.stdout = Some(lines);

        match outcome {
            Ok(Ok(ready)) => {
                handle.state = handle.state.transition_to(WorkerState::Ready)?;
                Ok(ready)
            }
            Ok(Err(err)) => {
                handle.state = handle.state.transition_to(WorkerState::Failed)?;
                Err(err)
            }
            Err(_elapsed) => {
                handle.state = handle.state.transition_to(WorkerState::Failed)?;
                // A worker that never became ready must not be left running.
                let _ = self.terminate(handle).await;
                Err(WorkerError::StartupTimeout {
                    waited: started.elapsed().max(budget),
                })
            }
        }
    }

    /// Asks the worker to stop, then kills it if it does not.
    ///
    /// # Errors
    ///
    /// Returns [`WorkerError::ShutdownTimeout`] when the worker had to be killed.
    /// The worker is stopped either way; the error reports how.
    pub async fn shutdown(&self, handle: &mut WorkerHandle) -> Result<WorkerExit, WorkerError> {
        if handle.state == WorkerState::Ready {
            handle.state = handle.state.transition_to(WorkerState::Draining)?;
        }
        if handle.state == WorkerState::Draining {
            handle.state = handle.state.transition_to(WorkerState::Stopping)?;
        }

        let budget = handle.deadlines.shutdown;
        let Some(child) = handle.child.as_mut() else {
            return Ok(WorkerExit {
                cause: ExitCause::Requested,
            });
        };

        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin
                .write_all(format!("{SHUTDOWN_COMMAND}\n").as_bytes())
                .await;
            let _ = stdin.flush().await;
            // Dropping stdin closes the worker's input, which is a second, weaker
            // stop signal for a worker that reads rather than parses.
            drop(stdin);
        }

        match tokio::time::timeout(budget, child.wait()).await {
            Ok(Ok(_status)) => {
                handle.child = None;
                handle.state = if handle.state == WorkerState::Stopping {
                    handle.state.transition_to(WorkerState::Absent)?
                } else {
                    handle.state
                };
                Ok(WorkerExit {
                    cause: ExitCause::Requested,
                })
            }
            Ok(Err(err)) => Err(WorkerError::Io(err)),
            Err(_elapsed) => {
                self.terminate(handle).await?;
                Err(WorkerError::ShutdownTimeout { waited: budget })
            }
        }
    }

    /// Waits for the worker to exit on its own.
    ///
    /// # Errors
    ///
    /// Returns [`WorkerError::Io`] when the exit status cannot be collected.
    pub async fn wait(&self, handle: &mut WorkerHandle) -> Result<WorkerExit, WorkerError> {
        let Some(child) = handle.child.as_mut() else {
            return Ok(WorkerExit {
                cause: ExitCause::Requested,
            });
        };
        let status = child.wait().await.map_err(WorkerError::Io)?;
        handle.child = None;
        // An exit that was not requested is a failure, and is reported with its
        // status rather than being flattened into a boolean.
        if handle.state.can_transition_to(WorkerState::Failed) {
            handle.state = handle.state.transition_to(WorkerState::Failed)?;
        }
        Ok(WorkerExit {
            cause: ExitCause::Unexpected {
                code: status.code(),
            },
        })
    }

    /// Kills the worker without asking.
    async fn terminate(&self, handle: &mut WorkerHandle) -> Result<(), WorkerError> {
        if let Some(child) = handle.child.as_mut() {
            let _ = child.start_kill();
            let _ = child.wait().await;
            handle.child = None;
        }
        Ok(())
    }
}

/// Requests that the child be reaped if this process dies.
///
/// Linux offers a parent-death signal, requested by the child itself between fork
/// and exec. Other Unix targets have no equivalent, so they get a process group,
/// which allows an explicit group kill but does not survive abrupt parent death.
#[cfg(unix)]
fn configure_unix_lifetime(command: &mut Command) {
    // A new process group so the worker and anything it starts can be signalled
    // together.
    command.process_group(0);

    #[cfg(target_os = "linux")]
    {
        use std::io;
        // Safety: this closure runs in the forked child before `exec`. `prctl` is
        // async-signal-safe, takes no allocations, and touches no shared state,
        // which is what this context requires.
        unsafe {
            command.pre_exec(|| {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) == -1 {
                    return Err(io::Error::last_os_error());
                }
                // The parent may already have died between fork and here, in which
                // case the death signal will never arrive and this child would be
                // an orphan from birth.
                if libc::getppid() == 1 {
                    libc::_exit(1);
                }
                Ok(())
            });
        }
    }
}

/// Deadlines used when a spec does not state its own.
#[must_use]
pub fn default_deadlines() -> Deadlines {
    Deadlines::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ready_marker_is_matched_at_the_start_of_a_line() {
        assert!(READY_MARKER.starts_with("MEHOY"));
        let announcement = format!("{READY_MARKER} port=1234");
        assert!(announcement.starts_with(READY_MARKER));
    }

    #[test]
    fn default_deadlines_are_distinct_numbers() {
        // ADR-0002 keeps these separate deliberately. If they ever collapse into
        // one value, the reason for the separation has been lost.
        let deadlines = default_deadlines();
        assert!(deadlines.startup > deadlines.health);
        assert!(deadlines.shutdown > std::time::Duration::ZERO);
        assert_ne!(deadlines.startup, deadlines.shutdown);
    }
}
