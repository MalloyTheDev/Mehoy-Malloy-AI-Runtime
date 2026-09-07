//! Supervision tests against a controllable stand-in worker.
//!
//! These answer the question M1.8 through M1.10 exist to answer: can the runtime
//! reliably own the lifetime of another process? Every edge of the supervision
//! contract from ADR-0002 is exercised here against a worker that fails on demand,
//! before any real engine, model container, or accelerator is involved.

use std::path::PathBuf;
use std::time::Duration;

use mehoy_core::event::ExitCause;
use mehoy_core::id::IdAllocator;
use mehoy_core::worker::{Deadlines, ProcessWorker, WorkerError, WorkerSpec, WorkerState};

/// Locates an example binary built alongside this test.
///
/// The test binary's own location varies between cargo versions and layouts, so
/// rather than assuming a fixed relative path this walks up from the executable
/// looking for the `examples` directory. Assuming `deps/..` works on one layout and
/// silently fails on another.
fn example_binary(name: &str) -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let file = format!("{name}{}", std::env::consts::EXE_SUFFIX);
    for ancestor in exe.ancestors().take(6) {
        let candidate = ancestor.join("examples").join(&file);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// The dummy worker binary.
fn dummy_worker() -> PathBuf {
    example_binary("dummy_worker").unwrap_or_else(|| {
        panic!("the dummy_worker example was not built; it is required by these tests")
    })
}

fn spec(args: &[&str], deadlines: Deadlines) -> WorkerSpec {
    static IDS: std::sync::LazyLock<IdAllocator> = std::sync::LazyLock::new(IdAllocator::new);
    WorkerSpec {
        id: IDS.worker(),
        program: dummy_worker(),
        args: args.iter().map(|arg| (*arg).to_owned()).collect(),
        deadlines,
        capture_lines: mehoy_core::worker::log::DEFAULT_CAPTURE_LINES,
        stop: mehoy_core::worker::StopProtocol::ShutdownLine,
    }
}

fn quick() -> Deadlines {
    // Short enough that a failing test does not stall the suite, long enough that
    // an ordinary process start is not mistaken for a timeout.
    Deadlines {
        startup: Duration::from_secs(5),
        shutdown: Duration::from_secs(2),
        health: Duration::from_secs(1),
    }
}

#[tokio::test]
async fn a_worker_starts_becomes_ready_and_stops_politely() {
    let supervisor = ProcessWorker;
    let mut handle = supervisor
        .spawn(spec(&["--ready"], quick()))
        .await
        .expect("worker starts");
    assert_eq!(handle.state(), WorkerState::Starting);

    let ready = supervisor
        .wait_ready(&mut handle)
        .await
        .expect("worker becomes ready");
    assert!(ready.announcement.contains("MEHOY-WORKER-READY"));
    assert_eq!(handle.state(), WorkerState::Ready);
    assert!(handle.state().is_usable());

    let exit = supervisor
        .shutdown(&mut handle)
        .await
        .expect("worker stops politely");
    assert_eq!(exit.cause, ExitCause::Requested);
    assert!(exit.cause.is_expected());
    assert_eq!(handle.state(), WorkerState::Absent);
}

#[tokio::test]
async fn readiness_waits_for_the_announcement_not_the_process() {
    // A running process is not readiness. The worker delays its announcement, and
    // the supervisor must wait for the announcement rather than returning as soon
    // as the process exists.
    let supervisor = ProcessWorker;
    let mut handle = supervisor
        .spawn(spec(&["--ready", "--startup-delay", "300"], quick()))
        .await
        .expect("worker starts");

    let began = std::time::Instant::now();
    supervisor
        .wait_ready(&mut handle)
        .await
        .expect("worker becomes ready");
    assert!(
        began.elapsed() >= Duration::from_millis(250),
        "returned in {:?}, which is before the worker announced readiness",
        began.elapsed()
    );

    let _ = supervisor.shutdown(&mut handle).await;
}

#[tokio::test]
async fn a_worker_that_never_becomes_ready_hits_its_startup_deadline() {
    let supervisor = ProcessWorker;
    let deadlines = Deadlines {
        startup: Duration::from_millis(400),
        ..quick()
    };
    let mut handle = supervisor
        .spawn(spec(&["--never-ready"], deadlines))
        .await
        .expect("worker starts");

    let err = supervisor
        .wait_ready(&mut handle)
        .await
        .expect_err("must time out");
    assert!(
        matches!(err, WorkerError::StartupTimeout { .. }),
        "expected a startup timeout, got {err}"
    );
    assert_eq!(handle.state(), WorkerState::Failed);
    // A worker that never became ready must not be left running.
    assert_eq!(handle.pid(), None, "worker was left alive after timing out");
}

#[tokio::test]
async fn a_worker_that_exits_during_startup_is_reported_with_its_status() {
    let supervisor = ProcessWorker;
    let mut handle = supervisor
        .spawn(spec(&["--exit-immediately"], quick()))
        .await
        .expect("worker starts");

    let err = supervisor
        .wait_ready(&mut handle)
        .await
        .expect_err("must fail");
    match err {
        WorkerError::ExitedDuringStartup { code } => assert_eq!(code, Some(7)),
        other => panic!("expected ExitedDuringStartup, got {other}"),
    }
    assert_eq!(handle.state(), WorkerState::Failed);
}

#[tokio::test]
async fn an_unexpected_exit_after_ready_is_a_failure_carrying_its_code() {
    // ADR-0002 requires a crashed worker to be reported rather than silently
    // restarted, and the cause to be carried rather than flattened.
    let supervisor = ProcessWorker;
    // The delay must comfortably exceed the readiness poll interval. At 50ms the
    // worker could die before readiness was ever observed, which reports an exit
    // during startup rather than the post-ready crash this test is about. That is
    // correct runtime behaviour (a dead process is not ready) but the wrong thing
    // to be testing here, and it raced differently on Linux than on Windows.
    let mut handle = supervisor
        .spawn(spec(&["--crash-after-ready", "1500"], quick()))
        .await
        .expect("worker starts");
    supervisor
        .wait_ready(&mut handle)
        .await
        .expect("worker becomes ready");

    let exit = supervisor.wait(&mut handle).await.expect("exit observed");
    assert_eq!(exit.cause, ExitCause::Unexpected { code: Some(9) });
    assert!(!exit.cause.is_expected());
    assert_eq!(handle.state(), WorkerState::Failed);
    assert!(
        handle.state().is_terminal(),
        "a crashed worker must require explicit recovery, not restart itself"
    );
}

#[tokio::test]
async fn a_worker_that_ignores_shutdown_is_killed_and_that_is_reported() {
    let supervisor = ProcessWorker;
    let deadlines = Deadlines {
        shutdown: Duration::from_millis(400),
        ..quick()
    };
    let mut handle = supervisor
        .spawn(spec(&["--ignore-shutdown"], deadlines))
        .await
        .expect("worker starts");
    supervisor
        .wait_ready(&mut handle)
        .await
        .expect("worker becomes ready");

    let err = supervisor
        .shutdown(&mut handle)
        .await
        .expect_err("must report that it had to be killed");
    assert!(
        matches!(err, WorkerError::ShutdownTimeout { .. }),
        "expected a shutdown timeout, got {err}"
    );
    // Killed either way. The error says how it stopped, not whether it stopped.
    assert_eq!(handle.pid(), None, "worker survived shutdown");
}

#[tokio::test]
async fn a_worker_that_stops_reading_input_is_still_stopped() {
    // Distinct from ignoring the request: this worker never even reads it, so a
    // stop protocol that relied on acknowledgement would hang forever.
    let supervisor = ProcessWorker;
    let deadlines = Deadlines {
        shutdown: Duration::from_millis(400),
        ..quick()
    };
    let mut handle = supervisor
        .spawn(spec(&["--hang-after-ready"], deadlines))
        .await
        .expect("worker starts");
    supervisor
        .wait_ready(&mut handle)
        .await
        .expect("worker becomes ready");

    let err = supervisor
        .shutdown(&mut handle)
        .await
        .expect_err("a hung worker cannot stop politely");
    assert!(
        matches!(err, WorkerError::ShutdownTimeout { .. }),
        "expected a shutdown timeout, got {err}"
    );
    assert_eq!(handle.pid(), None, "hung worker survived shutdown");
}

#[tokio::test]
async fn spawning_a_program_that_does_not_exist_fails_clearly() {
    let supervisor = ProcessWorker;
    let mut spec = spec(&["--ready"], quick());
    spec.program = PathBuf::from("mehoy-no-such-worker-binary");

    let err = supervisor.spawn(spec).await.expect_err("must fail");
    assert!(
        matches!(err, WorkerError::Spawn { .. }),
        "expected a spawn failure, got {err}"
    );
}

#[tokio::test]
async fn dropping_a_handle_does_not_leave_the_worker_running() {
    let supervisor = ProcessWorker;
    let mut handle = supervisor
        .spawn(spec(&["--ignore-shutdown"], quick()))
        .await
        .expect("worker starts");
    supervisor
        .wait_ready(&mut handle)
        .await
        .expect("worker becomes ready");
    let pid = handle.pid().expect("worker has a pid");

    drop(handle);

    // The child is killed on drop. Give the operating system a moment to reap it.
    let gone = wait_for_process_exit(pid, Duration::from_secs(10)).await;
    assert!(gone, "worker {pid} survived its handle being dropped");
}

/// Polls until the process is gone or the budget expires.
async fn wait_for_process_exit(pid: u32, budget: Duration) -> bool {
    let deadline = std::time::Instant::now() + budget;
    while std::time::Instant::now() < deadline {
        if !process_is_alive(pid) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    !process_is_alive(pid)
}

#[cfg(windows)]
fn process_is_alive(pid: u32) -> bool {
    use std::process::Command;
    // `tasklist` filters exactly, and prints an informational line rather than a
    // row when nothing matches.
    let output = Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/NH", "/FO", "CSV"])
        .output();
    match output {
        Ok(output) => {
            let text = String::from_utf8_lossy(&output.stdout);
            text.contains(&format!("\"{pid}\""))
        }
        Err(_) => false,
    }
}

#[cfg(unix)]
fn process_is_alive(pid: u32) -> bool {
    // Signal zero performs the permission and existence checks without sending
    // anything.
    // Safety: `kill` with signal zero has no side effects on the target.
    let alive = unsafe { libc::kill(pid as libc::pid_t, 0) };
    alive == 0
}

#[tokio::test]
async fn a_polite_stop_is_actually_delivered_not_merely_an_end_of_input() {
    // Regression guard. Tokio's `Child::wait` closes the child's standard input
    // before waiting. A readiness loop that awaited `wait` therefore sent the
    // worker an end of input while merely watching it, so a worker that stops on
    // closed input would exit on its own and the shutdown protocol would appear to
    // work without ever being exercised.
    //
    // The worker reports why it stopped, so this distinguishes the two.
    let supervisor = ProcessWorker;
    let mut handle = supervisor
        .spawn(spec(&["--ready"], quick()))
        .await
        .expect("worker starts");
    supervisor
        .wait_ready(&mut handle)
        .await
        .expect("worker becomes ready");

    let exit = supervisor
        .shutdown(&mut handle)
        .await
        .expect("worker stops politely");
    assert_eq!(exit.cause, ExitCause::Requested);

    let output = handle.log().render();
    assert!(
        output.contains("MEHOY-WORKER-STOPPING reason=command"),
        "worker did not report receiving the stop command; it likely exited \
         because its input was closed. Captured output:\n{output}"
    );
}

#[tokio::test]
async fn a_workers_output_is_captured_for_diagnostics() {
    // Backend failure evidence appears in output before the process exits, so it
    // must survive the exit rather than being discarded with the process.
    let supervisor = ProcessWorker;
    let mut handle = supervisor
        .spawn(spec(&["--exit-immediately"], quick()))
        .await
        .expect("worker starts");
    let _ = supervisor.wait_ready(&mut handle).await;

    // The worker printed nothing here, but the capture must still be usable rather
    // than absent, and must not panic after the process is gone.
    let _ = handle.log().render();
    assert_eq!(handle.log().dropped(), 0);
}

#[tokio::test]
async fn readiness_output_is_retained_after_the_worker_stops() {
    let supervisor = ProcessWorker;
    let mut handle = supervisor
        .spawn(spec(&["--ready"], quick()))
        .await
        .expect("worker starts");
    supervisor
        .wait_ready(&mut handle)
        .await
        .expect("worker becomes ready");
    let _ = supervisor.shutdown(&mut handle).await;

    let output = handle.log().render();
    assert!(
        output.contains("MEHOY-WORKER-READY"),
        "the readiness announcement should still be in the capture:\n{output}"
    );
}

#[tokio::test]
async fn a_worker_with_no_stop_protocol_is_terminated_without_waiting() {
    // Some programs implement no graceful stop. Sending a request they do not
    // understand and then waiting out the deadline delays every shutdown and
    // reports a timeout that describes the protocol rather than the worker.
    //
    // Termination is the agreed mechanism for such a worker, so it is neither slow
    // nor a failure.
    use mehoy_core::worker::StopProtocol;

    let supervisor = ProcessWorker;
    let mut spec = spec(&["--ignore-shutdown"], quick());
    spec.stop = StopProtocol::Terminate;

    let mut handle = supervisor.spawn(spec).await.expect("worker starts");
    supervisor
        .wait_ready(&mut handle)
        .await
        .expect("worker becomes ready");

    let began = std::time::Instant::now();
    let exit = supervisor
        .shutdown(&mut handle)
        .await
        .expect("termination is the defined stop, not a failure");
    let elapsed = began.elapsed();

    assert_eq!(exit.cause, ExitCause::Requested);
    assert_eq!(handle.pid(), None, "the worker survived termination");
    assert!(
        elapsed < handle.deadlines().shutdown,
        "termination took {elapsed:?}, which suggests the shutdown deadline was \
         spent waiting for a protocol this worker does not implement"
    );
}

#[tokio::test]
async fn a_worker_that_does_implement_the_stop_protocol_still_reports_ignoring_it() {
    // The other half: choosing termination for one worker must not stop the
    // supervisor from reporting a worker that was asked politely and refused.
    use mehoy_core::worker::StopProtocol;

    let supervisor = ProcessWorker;
    let deadlines = Deadlines {
        shutdown: Duration::from_millis(400),
        ..quick()
    };
    let mut spec = spec(&["--ignore-shutdown"], deadlines);
    spec.stop = StopProtocol::ShutdownLine;

    let mut handle = supervisor.spawn(spec).await.expect("worker starts");
    supervisor
        .wait_ready(&mut handle)
        .await
        .expect("worker becomes ready");

    let err = supervisor
        .shutdown(&mut handle)
        .await
        .expect_err("ignoring a protocol the worker implements is still a fault");
    assert!(
        matches!(err, WorkerError::ShutdownTimeout { .. }),
        "expected a shutdown timeout, got {err}"
    );
}
