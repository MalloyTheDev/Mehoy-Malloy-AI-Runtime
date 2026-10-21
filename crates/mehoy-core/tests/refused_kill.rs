//! What happens when the system will not let the supervisor kill a worker.
//!
//! This is the one branch of supervision that reports "the worker could not be
//! stopped", and everything above it treats that as the answer it must not get
//! wrong: a caller reads it as "the accelerator may still be held". It was
//! unreachable from a test until a refusal could be injected, because an
//! operating system that declines to signal a live process is not a condition a
//! test can arrange.
//!
//! These tests live in their own target rather than beside the other supervision
//! tests because the injected refusal is a process-wide flag. A separate test
//! binary is a separate process, so arming it here cannot reach into a test that
//! is legitimately stopping a worker.

use std::path::PathBuf;
use std::time::Duration;

use mehoy_core::id::IdAllocator;
use mehoy_core::worker::{
    Deadlines, ProcessWorker, StopProtocol, WorkerError, WorkerSpec, WorkerState,
};

/// Locates an example binary built alongside this test.
///
/// The test binary's own location varies between cargo versions and layouts, so
/// rather than assuming a fixed relative path this walks up from the executable
/// looking for the `examples` directory.
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

fn dummy_worker() -> PathBuf {
    example_binary("dummy_worker").unwrap_or_else(|| {
        panic!("the dummy_worker example was not built; it is required by these tests")
    })
}

/// A worker the supervisor will terminate rather than ask politely.
///
/// `StopProtocol::Terminate` is what the llama backend uses, and it goes straight
/// to the kill, so the only wait in play is the one under test.
fn spec(args: &[&str], deadlines: Deadlines) -> WorkerSpec {
    static IDS: std::sync::LazyLock<IdAllocator> = std::sync::LazyLock::new(IdAllocator::new);
    WorkerSpec {
        id: IDS.worker(),
        program: dummy_worker(),
        args: args.iter().map(|arg| (*arg).to_owned()).collect(),
        deadlines,
        capture_lines: mehoy_core::worker::log::DEFAULT_CAPTURE_LINES,
        stop: StopProtocol::Terminate,
    }
}

fn deadlines(shutdown: Duration) -> Deadlines {
    Deadlines {
        startup: Duration::from_secs(5),
        shutdown,
        health: Duration::from_secs(1),
    }
}

/// Serialises the two tests, because the injected refusal is one flag.
async fn exclusive() -> tokio::sync::MutexGuard<'static, ()> {
    static LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    LOCK.lock().await
}

#[tokio::test]
async fn a_refused_kill_is_reported_rather_than_waited_out() {
    let _exclusive = exclusive().await;
    let supervisor = ProcessWorker;
    let budget = Duration::from_millis(400);
    // Hangs rather than obeying, because `Child::wait` drops the child's stdin
    // before waiting, and a worker that stops on end of input would exit for that
    // reason instead of staying alive to be reported as unkillable.
    let mut handle = supervisor
        .spawn(spec(&["--hang-after-ready"], deadlines(budget)))
        .await
        .expect("worker starts");
    supervisor
        .wait_ready(&mut handle)
        .await
        .expect("worker becomes ready");

    mehoy_core::refuse_next_kill();
    // Bounded here as well as in the code under test. Without the fix this call
    // never returns, and a test that hangs reports nothing: it has to fail.
    let outcome = tokio::time::timeout(Duration::from_secs(20), supervisor.shutdown(&mut handle))
        .await
        .expect("a refused kill must be reported rather than waited out indefinitely");
    mehoy_core::clear_injected_faults();

    match outcome {
        Err(WorkerError::Io(err)) => assert!(
            err.to_string().contains("injected"),
            "reported something other than the refusal: {err}"
        ),
        Err(other) => panic!("expected the refusal to be reported, got {other}"),
        Ok(exit) => panic!("a worker that could not be killed was reported as stopped: {exit:?}"),
    }

    assert_eq!(
        handle.state(),
        WorkerState::Stopping,
        "the supervisor claimed a worker it could not kill had gone"
    );
    assert!(
        handle.pid().is_some(),
        "the handle discarded a process that is still running, so nothing can try again"
    );

    // The worker was left for a later attempt, and a later attempt works. This is
    // the half that makes keeping the child worthwhile rather than tidy.
    let exit = supervisor
        .shutdown(&mut handle)
        .await
        .expect("the second attempt stops it");
    assert!(exit.cause.is_expected());
    assert_eq!(handle.state(), WorkerState::Absent);
    assert!(handle.pid().is_none());
}

#[tokio::test]
async fn a_refused_kill_whose_worker_ends_anyway_is_not_reported() {
    // A refusal describes one moment. If the worker ends within the budget then
    // the accelerator has been released, which is what the caller was actually
    // asking about, so reporting a failure would be the wrong answer.
    let _exclusive = exclusive().await;
    let supervisor = ProcessWorker;
    let mut handle = supervisor
        .spawn(spec(
            &["--ready", "--crash-after-ready", "600"],
            deadlines(Duration::from_secs(5)),
        ))
        .await
        .expect("worker starts");
    supervisor
        .wait_ready(&mut handle)
        .await
        .expect("worker becomes ready");

    mehoy_core::refuse_next_kill();
    let outcome = tokio::time::timeout(Duration::from_secs(20), supervisor.shutdown(&mut handle))
        .await
        .expect("the bounded wait must end well inside this");
    // Cleared unconditionally: if the worker had already exited the refusal was
    // never consumed, and a flag left armed would reach the other test.
    mehoy_core::clear_injected_faults();

    let exit = outcome.expect("a worker that ended is not a worker that could not be stopped");
    assert!(exit.cause.is_expected());
    assert_eq!(handle.state(), WorkerState::Absent);
    assert!(
        handle.pid().is_none(),
        "the worker ended but the handle still holds it"
    );
}
