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

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::broadcast;

use super::log::{LogHandle, LogStream};
use super::{
    Deadlines, StopProtocol, WorkerError, WorkerExit, WorkerReady, WorkerSpec, WorkerState,
};
use crate::event::ExitCause;
use crate::id::WorkerId;

/// The line a worker prints on standard output to announce readiness.
pub const READY_MARKER: &str = "MEHOY-WORKER-READY";

/// The line written to a worker's standard input to request a polite stop.
pub const SHUTDOWN_COMMAND: &str = "shutdown";

/// How often a readiness probe is consulted.
const PROBE_INTERVAL: std::time::Duration = std::time::Duration::from_millis(50);

/// How many captured lines the live feed buffers for a watching probe.
const LINE_FEED_CAPACITY: usize = 256;

/// How long to wait for a worker's stream readers to finish after it exits.
const OUTPUT_FLUSH_BUDGET: std::time::Duration = std::time::Duration::from_secs(2);

/// What a readiness probe observed.
#[derive(Debug)]
pub enum ProbeOutcome {
    /// Not ready yet. Keep waiting until the startup deadline.
    NotYet,
    /// Ready, with whatever the backend reported.
    Ready(WorkerReady),
}

/// Continuously reads one of a worker's streams into its log and live feed.
///
/// A stream that is never read fills its pipe and blocks the worker, which is
/// indistinguishable from a hang, so this runs for the worker's whole life rather
/// than only during startup.
fn drain<R>(
    reader: R,
    stream: LogStream,
    log: LogHandle,
    feed: broadcast::Sender<super::log::LogLine>,
) -> tokio::task::JoinHandle<()>
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut lines = BufReader::new(reader).lines();
        while let Ok(Some(text)) = lines.next_line().await {
            log.record(stream, text.clone());
            // No subscribers is the normal case and not an error.
            let _ = feed.send(super::log::LogLine {
                stream,
                at: Instant::now(),
                text,
            });
        }
    })
}

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
    /// Bounded capture of everything the worker has printed.
    log: LogHandle,
    /// Live feed of captured lines, for a readiness probe that watches output.
    lines: broadcast::Sender<super::log::LogLine>,
    /// The tasks reading the worker's streams. Awaited after the process exits so
    /// its final output is captured rather than lost to a race with the exit.
    drains: Vec<tokio::task::JoinHandle<()>>,
    deadlines: Deadlines,
    stop: StopProtocol,
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

    /// The worker's captured output.
    ///
    /// Retained after the process exits, so a failure report can include the
    /// evidence the worker printed on its way out.
    #[must_use]
    pub fn log(&self) -> &LogHandle {
        &self.log
    }

    /// The deadlines this worker is held to.
    #[must_use]
    pub fn deadlines(&self) -> Deadlines {
        self.deadlines
    }

    /// Marks the worker failed after an external readiness decision.
    ///
    /// Used by a backend that runs its own readiness protocol and needs the shared
    /// state machine to agree with what it observed.
    ///
    /// # Errors
    ///
    /// Returns an error when the worker cannot move to `Failed` from its state.
    pub fn mark_failed(&mut self) -> Result<(), WorkerError> {
        self.state = self.state.transition_to(WorkerState::Failed)?;
        Ok(())
    }

    /// Waits for the stream readers to finish after the process has exited.
    ///
    /// The worker's pipes close when it exits, so the readers end on their own
    /// shortly afterwards. Without waiting for them, a caller can read the capture
    /// before the final lines have been recorded, which loses exactly the output a
    /// failing worker prints on its way out.
    ///
    /// Bounded, because a reader blocked on a pipe inherited by some other process
    /// must not hold up the caller.
    pub async fn flush_output(&mut self) {
        let drains = std::mem::take(&mut self.drains);
        for drain in drains {
            let _ = tokio::time::timeout(OUTPUT_FLUSH_BUDGET, drain).await;
        }
    }

    /// Marks the worker ready after an external readiness decision.
    ///
    /// # Errors
    ///
    /// Returns an error when the worker cannot move to `Ready` from its state.
    pub fn mark_ready(&mut self) -> Result<(), WorkerError> {
        self.state = self.state.transition_to(WorkerState::Ready)?;
        Ok(())
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
            .stderr(Stdio::piped())
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

        // Both streams are drained continuously. A worker whose output is never
        // read will eventually block writing to a full pipe, which looks exactly
        // like a hang and would be misdiagnosed as one.
        let log = LogHandle::new(spec.capture_lines);
        let (lines, _) = broadcast::channel(LINE_FEED_CAPACITY);

        let mut drains = Vec::new();
        if let Some(out) = child.stdout.take() {
            drains.push(drain(out, LogStream::Stdout, log.clone(), lines.clone()));
        }
        if let Some(err) = child.stderr.take() {
            drains.push(drain(err, LogStream::Stderr, log.clone(), lines.clone()));
        }

        Ok(WorkerHandle {
            id: spec.id,
            state,
            child: Some(child),
            log,
            lines,
            drains,
            deadlines: spec.deadlines,
            stop: spec.stop,
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
        // The probe is called repeatedly and each call returns an owned future, so
        // the receiver is shared rather than borrowed across calls.
        let feed = std::sync::Arc::new(tokio::sync::Mutex::new(handle.lines.subscribe()));
        self.wait_ready_with(handle, move || {
            let feed = std::sync::Arc::clone(&feed);
            async move {
                let mut feed = feed.lock().await;
                loop {
                    match feed.try_recv() {
                        Ok(line) if line.text.trim_start().starts_with(READY_MARKER) => {
                            return Ok(ProbeOutcome::Ready(WorkerReady {
                                announcement: line.text,
                            }));
                        }
                        // Some other output line: keep looking at what is buffered.
                        Ok(_) => {}
                        // Falling behind loses diagnostics only. The readiness line
                        // may still arrive, so this keeps waiting rather than
                        // failing the startup.
                        Err(broadcast::error::TryRecvError::Lagged(_)) => {}
                        Err(
                            broadcast::error::TryRecvError::Empty
                            | broadcast::error::TryRecvError::Closed,
                        ) => return Ok(ProbeOutcome::NotYet),
                    }
                }
            }
        })
        .await
    }

    /// Waits for readiness as decided by a caller-supplied probe.
    ///
    /// The generic supervisor deliberately does not know how any particular backend
    /// reports readiness. It owns the parts that are the same for every backend:
    /// the startup deadline, noticing that the process died, and keeping the state
    /// machine honest. The probe owns what readiness actually means.
    ///
    /// The probe is polled on an interval. Returning `NotYet` means keep waiting;
    /// an error from the probe is a terminal startup failure.
    ///
    /// # Errors
    ///
    /// Returns [`WorkerError::StartupTimeout`] when the deadline passes and
    /// [`WorkerError::ExitedDuringStartup`] when the process ends first.
    pub async fn wait_ready_with<F, Fut>(
        &self,
        handle: &mut WorkerHandle,
        mut probe: F,
    ) -> Result<WorkerReady, WorkerError>
    where
        F: FnMut() -> Fut + Send,
        Fut: Future<Output = Result<ProbeOutcome, WorkerError>> + Send,
    {
        let started = Instant::now();
        let budget = handle.deadlines.startup;
        let child = handle
            .child
            .as_mut()
            .ok_or_else(|| WorkerError::Io(std::io::Error::other("worker has no process")))?;

        let outcome = tokio::time::timeout(budget, async {
            loop {
                // A dead process can never become ready, so this is checked before
                // waiting out the rest of the deadline for something impossible.
                if let Some(status) = child.try_wait().map_err(WorkerError::Io)? {
                    return Err(WorkerError::ExitedDuringStartup {
                        code: status.code(),
                    });
                }
                match probe().await? {
                    ProbeOutcome::Ready(ready) => return Ok(ready),
                    ProbeOutcome::NotYet => {}
                }
                // Deliberately `try_wait` and a sleep rather than awaiting `wait`.
                // Tokio's `Child::wait` closes the child's standard input before
                // waiting, so polling it here would send the worker an end of input
                // during startup. A worker that stops on end of input would then
                // exit while merely being watched, and the polite shutdown protocol
                // would never actually be exercised.
                tokio::time::sleep(PROBE_INTERVAL).await;
            }
        })
        .await;

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

        // A worker with no graceful stop is terminated directly. Sending a request
        // it does not implement and then waiting out the deadline would delay every
        // shutdown and report a timeout that describes the protocol rather than the
        // worker.
        if handle.stop == StopProtocol::Terminate {
            self.terminate(handle).await?;
            handle.state = if handle.state == WorkerState::Stopping {
                handle.state.transition_to(WorkerState::Absent)?
            } else {
                handle.state
            };
            return Ok(WorkerExit {
                cause: ExitCause::Requested,
            });
        }

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
                handle.flush_output().await;
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
        handle.flush_output().await;
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
        let mut refused = None;
        if let Some(child) = handle.child.as_mut() {
            // A worker that has already exited needs no killing, and signalling one
            // that has also been reaped fails. Asking first keeps that ordinary case
            // from being reported as a refusal to stop.
            let already_gone = matches!(child.try_wait(), Ok(Some(_)));
            if !already_gone {
                // Otherwise the outcome is reported. Discarding it lets a worker
                // that could not be killed be described as stopped, which is the
                // more dangerous of the two wrong answers: the caller believes the
                // accelerator has been released.
                if let Err(err) = child.start_kill() {
                    // It may simply have exited in the interval. That is not a
                    // failure to stop it, so the claim is checked before it is made.
                    if !matches!(child.try_wait(), Ok(Some(_))) {
                        refused = Some(err);
                    }
                }
            }
            // The wait is never reported. Losing a race to reap a process that has
            // already exited is ordinary, and says nothing about whether it is gone.
            let _ = child.wait().await;
            handle.child = None;
        }
        handle.flush_output().await;
        match refused {
            Some(err) => Err(WorkerError::Io(err)),
            None => Ok(()),
        }
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
