//! Owner process for the real-backend orphan test.
//!
//! Starts an actual `llama-server` under supervision, reports its process id, and
//! waits to be killed. The point is to prove the lifetime mechanism against the
//! real native executable, which loads accelerator libraries and spawns its own
//! threads, rather than only against a small stand-in.
//!
//! No model is loaded. The backend is started in its model-less mode, which stays
//! listening, because this test is about process lifetime and not about inference.

use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

use mehoy_backend_llama::{BackendChannel, EXECUTABLE_ENV};
use mehoy_core::id::IdAllocator;
use mehoy_core::worker::log::DEFAULT_CAPTURE_LINES;
use mehoy_core::worker::{Deadlines, ProcessWorker, WorkerSpec};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let Some(executable) = std::env::var_os(EXECUTABLE_ENV).map(PathBuf::from) else {
        eprintln!("llama_orphan_parent: {EXECUTABLE_ENV} is not set");
        std::process::exit(2);
    };

    let channel = match BackendChannel::reserve() {
        Ok(channel) => channel,
        Err(err) => {
            eprintln!("llama_orphan_parent: cannot reserve a channel: {err}");
            std::process::exit(1);
        }
    };

    let ids = IdAllocator::new();
    let spec = WorkerSpec {
        id: ids.worker(),
        program: executable,
        // Model-less mode: stays listening, loads nothing. Still loopback only,
        // still behind a secret, still no web interface.
        args: vec![
            "--host".to_owned(),
            channel.host(),
            "--port".to_owned(),
            channel.port().to_string(),
            "--api-key".to_owned(),
            channel.secret().expose().to_owned(),
            "--no-webui".to_owned(),
        ],
        deadlines: Deadlines {
            startup: Duration::from_secs(30),
            shutdown: Duration::from_secs(5),
            health: Duration::from_secs(5),
        },
        capture_lines: DEFAULT_CAPTURE_LINES,
    };

    let supervisor = ProcessWorker;
    let handle = match supervisor.spawn(spec).await {
        Ok(handle) => handle,
        Err(err) => {
            eprintln!("llama_orphan_parent: cannot start backend: {err}");
            std::process::exit(1);
        }
    };

    let Some(pid) = handle.pid() else {
        eprintln!("llama_orphan_parent: backend has no process id");
        std::process::exit(1);
    };

    // Give the executable time to finish loading its accelerator libraries, so the
    // test kills the owner while the backend is genuinely up rather than mid-start.
    tokio::time::sleep(Duration::from_secs(2)).await;

    let mut stdout = std::io::stdout();
    let _ = writeln!(stdout, "WORKER_PID {pid}");
    let _ = stdout.flush();

    // Hold the handle, and with it the lifetime mechanism, until killed.
    loop {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
