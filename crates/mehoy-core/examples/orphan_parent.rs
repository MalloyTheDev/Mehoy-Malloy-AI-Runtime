//! A stand-in owner process for the orphan-cleanup test.
//!
//! It starts a supervised worker, reports that worker's process id on standard
//! output, and then waits forever. The test kills this process without warning and
//! checks that the worker dies with it.
//!
//! This has to be a separate process. The invariant under test is that a worker
//! does not survive the abrupt death of its owner, and a test running in-process
//! cannot kill itself and then observe the result.

use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

use mehoy_core::id::IdAllocator;
use mehoy_core::worker::{Deadlines, ProcessWorker, WorkerSpec};

fn dummy_worker() -> PathBuf {
    let mut path = std::env::current_exe().expect("example has a path");
    path.pop();
    path.push(format!("dummy_worker{}", std::env::consts::EXE_SUFFIX));
    path
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let ids = IdAllocator::new();
    let supervisor = ProcessWorker;

    let spec = WorkerSpec {
        id: ids.worker(),
        program: dummy_worker(),
        // Ignores a polite stop, so if it dies it is because the operating system
        // reaped it rather than because it chose to cooperate.
        args: vec!["--ignore-shutdown".to_owned()],
        deadlines: Deadlines {
            startup: Duration::from_secs(20),
            shutdown: Duration::from_secs(2),
            health: Duration::from_secs(2),
        },
        capture_lines: mehoy_core::worker::log::DEFAULT_CAPTURE_LINES,
    };

    let mut handle = match supervisor.spawn(spec).await {
        Ok(handle) => handle,
        Err(err) => {
            eprintln!("orphan_parent: cannot start worker: {err}");
            std::process::exit(1);
        }
    };

    if let Err(err) = supervisor.wait_ready(&mut handle).await {
        eprintln!("orphan_parent: worker never became ready: {err}");
        std::process::exit(1);
    }

    let Some(pid) = handle.pid() else {
        eprintln!("orphan_parent: worker has no process id");
        std::process::exit(1);
    };

    let mut stdout = std::io::stdout();
    let _ = writeln!(stdout, "WORKER_PID {pid}");
    let _ = stdout.flush();

    // Hold the handle, and with it the job object, until this process is killed.
    loop {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
