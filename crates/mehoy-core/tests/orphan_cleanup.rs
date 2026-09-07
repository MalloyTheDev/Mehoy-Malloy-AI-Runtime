//! The orphan invariant.
//!
//! > No runtime-owned worker survives the loss of the process that owns it.
//!
//! This is tested against an abrupt kill, not a graceful shutdown. Graceful paths
//! are covered elsewhere and prove nothing here: the failure being prevented is a
//! worker left holding accelerator memory after its owner died without running any
//! cleanup code at all.
//!
//! On Windows the mechanism is a Job Object with kill-on-job-close, because there
//! is no parent-death signal and a child otherwise outlives a terminated parent
//! indefinitely.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Locates an example binary built alongside this test.
fn example(name: &str) -> PathBuf {
    let mut path = std::env::current_exe().expect("test executable has a path");
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.push("examples");
    path.push(format!("{name}{}", std::env::consts::EXE_SUFFIX));
    assert!(
        path.exists(),
        "example {name} not built at {}",
        path.display()
    );
    path
}

#[cfg(windows)]
fn process_is_alive(pid: u32) -> bool {
    let output = Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/NH", "/FO", "CSV"])
        .output();
    output
        .is_ok_and(|output| String::from_utf8_lossy(&output.stdout).contains(&format!("\"{pid}\"")))
}

#[cfg(unix)]
fn process_is_alive(pid: u32) -> bool {
    // Safety: signal zero performs existence and permission checks only.
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}

fn wait_for_exit(pid: u32, budget: Duration) -> bool {
    let deadline = Instant::now() + budget;
    while Instant::now() < deadline {
        if !process_is_alive(pid) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    !process_is_alive(pid)
}

#[test]
fn a_worker_does_not_survive_the_abrupt_death_of_its_owner() {
    // 1. Start an owner process, which starts a worker.
    let mut owner = Command::new(example("orphan_parent"))
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("owner process starts");

    // 2. Learn the worker's process id from the owner.
    let stdout = owner.stdout.take().expect("owner stdout is piped");
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .expect("owner reports the worker process id");
    let worker_pid: u32 = line
        .strip_prefix("WORKER_PID ")
        .unwrap_or_else(|| panic!("unexpected owner output: {line:?}"))
        .trim()
        .parse()
        .expect("worker process id is a number");

    // 3. Confirm the worker really is running before proving anything about it
    //    dying, so a test that never started a worker cannot pass trivially.
    assert!(
        process_is_alive(worker_pid),
        "worker {worker_pid} was not running before the owner was killed"
    );

    // 4. Kill the owner with no chance to clean up. This is TerminateProcess on
    //    Windows and SIGKILL on Unix: no unwinding, no destructors, no shutdown.
    owner.kill().expect("owner is killed");
    let owner_pid = owner.id();
    owner.wait().expect("owner is reaped");
    assert!(
        wait_for_exit(owner_pid, Duration::from_secs(10)),
        "owner {owner_pid} did not actually die"
    );

    // 5. The worker must go with it.
    assert!(
        wait_for_exit(worker_pid, Duration::from_secs(20)),
        "worker {worker_pid} survived the abrupt death of its owner, \
         which is the orphan this design exists to prevent"
    );
}

#[test]
fn the_owner_is_genuinely_holding_the_worker_open() {
    // Guards against the previous test passing for the wrong reason. If the worker
    // exited on its own promptly, the orphan test would pass without the cleanup
    // mechanism doing anything. Here the owner is left alive and the worker is
    // expected to stay alive with it.
    let mut owner = Command::new(example("orphan_parent"))
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("owner process starts");

    let stdout = owner.stdout.take().expect("owner stdout is piped");
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    reader.read_line(&mut line).expect("owner reports the pid");
    let worker_pid: u32 = line
        .strip_prefix("WORKER_PID ")
        .unwrap_or_else(|| panic!("unexpected owner output: {line:?}"))
        .trim()
        .parse()
        .expect("worker process id is a number");

    std::thread::sleep(Duration::from_millis(750));
    let still_alive = process_is_alive(worker_pid);

    let _ = owner.kill();
    let _ = owner.wait();
    let _ = wait_for_exit(worker_pid, Duration::from_secs(20));

    assert!(
        still_alive,
        "worker {worker_pid} exited on its own, so the orphan test above would \
         pass whether or not cleanup works"
    );
}
