//! Integration tests against a real `llama-server`.
//!
//! These need an actual backend executable, named by `MEHOY_LLAMA_SERVER`. When it
//! is absent they skip rather than fail, so the suite still passes on a machine
//! without one, and each skip says so out loud rather than reporting a pass that
//! proved nothing.
//!
//! None of these load a real model. They cover the parts that can be established
//! without one: identity, refusal to start, failure reporting, and lifetime
//! ownership. Proving a model loads and generates needs a real model file and is a
//! later milestone.

use std::path::PathBuf;
use std::time::Duration;

use mehoy_backend_llama::{
    BackendError, EXECUTABLE_ENV, GpuLayerPolicy, LlamaCppBackend, LlamaCppWorkerSpec, Readiness,
};
use mehoy_core::id::IdAllocator;
use mehoy_core::worker::{Deadlines, WorkerError};

/// Resolves the configured backend, or explains why the test is being skipped.
fn backend() -> Option<LlamaCppBackend> {
    match LlamaCppBackend::from_env() {
        Ok(backend) => Some(backend),
        Err(err) => {
            eprintln!("SKIPPED: no real backend available ({err}). Set {EXECUTABLE_ENV} to run.");
            None
        }
    }
}

fn worker_id() -> mehoy_core::id::WorkerId {
    static IDS: std::sync::LazyLock<IdAllocator> = std::sync::LazyLock::new(IdAllocator::new);
    IDS.worker()
}

#[tokio::test]
async fn the_backend_build_is_identified() {
    let Some(backend) = backend() else { return };

    let identity = backend.identify().await.expect("backend reports a version");

    assert!(
        identity.build.is_some(),
        "no build number parsed from banner:\n{}",
        identity.banner
    );
    assert!(
        !identity.banner.is_empty(),
        "the banner must be kept verbatim even if parsing finds nothing"
    );
    // A failure report has to be able to name the exact build, since upstream
    // behaviour differs between revisions.
    let described = identity.to_string();
    assert!(described.contains("llama.cpp"), "{described}");
    assert!(
        described.contains(identity.build.as_deref().unwrap_or("?")),
        "{described}"
    );
    eprintln!("identified backend: {described}");
}

#[tokio::test]
async fn a_configured_path_that_is_not_a_file_is_refused() {
    let err = LlamaCppBackend::new(PathBuf::from("no-such-llama-server-anywhere"))
        .expect_err("must be refused");
    assert!(
        matches!(err, BackendError::Unavailable { .. }),
        "expected Unavailable, got {err}"
    );
}

#[tokio::test]
async fn an_invalid_model_produces_a_terminal_failure_with_evidence() {
    // The most common real failure: the backend starts, cannot load what it was
    // pointed at, and exits. The report must carry the backend's own output,
    // because the exit status alone never explains it.
    let Some(backend) = backend() else { return };

    let spec = LlamaCppWorkerSpec {
        gpu_layers: Some(GpuLayerPolicy::CpuOnly),
        deadlines: Deadlines {
            startup: Duration::from_secs(25),
            shutdown: Duration::from_secs(5),
            health: Duration::from_secs(2),
        },
        ..LlamaCppWorkerSpec::new("definitely-not-a-model.gguf")
    };

    let err = backend
        .start(worker_id(), &spec)
        .await
        .expect_err("loading a nonexistent model must fail");

    match err {
        BackendError::Worker(failure) => {
            let (source, identity, output) = (failure.source, failure.identity, failure.output);
            assert!(
                matches!(
                    source,
                    WorkerError::ExitedDuringStartup { .. } | WorkerError::StartupTimeout { .. }
                ),
                "expected a startup failure, got {source}"
            );
            assert!(
                identity.is_some(),
                "a failure report must name the backend build"
            );
            assert!(
                !output.trim().is_empty(),
                "a failure report must carry the backend's own output"
            );
            eprintln!("backend failure evidence:\n{output}");
        }
        other => panic!("expected a worker failure, got {other}"),
    }
}

#[tokio::test]
async fn the_backend_is_never_told_to_bind_a_routable_address() {
    // Asserted against the real executable's command line, so a future change to
    // argument construction cannot quietly expose the backend.
    let Some(backend) = backend() else { return };
    let channel = mehoy_backend_llama::BackendChannel::reserve().expect("channel reserves");
    let file = mehoy_backend_llama::SecretFile::create(
        &std::env::temp_dir().join("mehoy-backend-test"),
        channel.secret(),
    )
    .expect("secret file is creatable");
    let args = backend.command_line(&LlamaCppWorkerSpec::new("model.gguf"), &channel, &file);

    assert!(channel.is_loopback());
    let host_index = args
        .iter()
        .position(|arg| arg == "--host")
        .expect("--host is passed");
    assert_eq!(args[host_index + 1], "127.0.0.1");
    assert!(args.contains(&"--api-key-file".to_owned()));
    assert!(args.contains(&"--no-webui".to_owned()));
    assert!(
        !args.iter().any(|arg| arg == channel.secret().expose()),
        "the credential must never appear in the process arguments, which other          local accounts can read"
    );
}

#[tokio::test]
async fn readiness_is_backend_ready_and_not_inference_verified() {
    // A guard on the claim rather than on behaviour. Nothing in this milestone may
    // return InferenceVerified, because no generation has been performed.
    assert_ne!(Readiness::BackendReady, Readiness::InferenceVerified);
}

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

#[cfg(windows)]
fn process_is_alive(pid: u32) -> bool {
    let output = std::process::Command::new("tasklist")
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
    let deadline = std::time::Instant::now() + budget;
    while std::time::Instant::now() < deadline {
        if !process_is_alive(pid) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    !process_is_alive(pid)
}

#[test]
fn a_real_backend_does_not_survive_the_abrupt_death_of_its_owner() {
    // The stand-in worker already proves the mechanism. This proves it against the
    // actual native executable, which loads accelerator libraries and starts its
    // own threads, so a mechanism that only worked for simple children would show
    // up here.
    if backend().is_none() {
        return;
    }
    let Some(parent) = example_binary("llama_orphan_parent") else {
        eprintln!("SKIPPED: the llama_orphan_parent example was not built");
        return;
    };

    let mut owner = std::process::Command::new(&parent)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .expect("owner process starts");

    let stdout = owner.stdout.take().expect("owner stdout is piped");
    let mut reader = std::io::BufReader::new(stdout);
    let mut line = String::new();
    std::io::BufRead::read_line(&mut reader, &mut line).expect("owner reports the backend pid");
    let backend_pid: u32 = line
        .strip_prefix("WORKER_PID ")
        .unwrap_or_else(|| panic!("unexpected owner output: {line:?}"))
        .trim()
        .parse()
        .expect("backend process id is a number");

    assert!(
        process_is_alive(backend_pid),
        "the backend was not running before the owner was killed, so this test \
         would pass without proving anything"
    );

    owner.kill().expect("owner is killed");
    owner.wait().expect("owner is reaped");

    assert!(
        wait_for_exit(backend_pid, Duration::from_secs(30)),
        "llama-server {backend_pid} survived the abrupt death of its owner, \
         leaving a backend holding accelerator memory with nothing to reclaim it"
    );
}

/// Starts a model-less backend on a private channel and runs `body` against it.
///
/// Model-less mode stays listening without loading anything, which is enough to
/// exercise the credential boundary.
async fn with_running_backend<F, Fut>(body: F)
where
    F: FnOnce(mehoy_backend_llama::BackendChannel) -> Fut,
    Fut: Future<Output = ()>,
{
    let Some(backend) = backend() else { return };
    let channel = mehoy_backend_llama::BackendChannel::reserve().expect("channel reserves");

    // The credential is delivered by file, the same way the production start path
    // does it. Testing a different delivery mechanism would validate a path the
    // runtime does not use.
    let secret_file = mehoy_backend_llama::SecretFile::create(
        &std::env::temp_dir().join("mehoy-backend-test"),
        channel.secret(),
    )
    .expect("secret file is creatable");

    let spec = mehoy_core::worker::WorkerSpec {
        id: worker_id(),
        program: backend.executable().to_path_buf(),
        args: vec![
            "--host".to_owned(),
            channel.host(),
            "--port".to_owned(),
            channel.port().to_string(),
            "--api-key-file".to_owned(),
            secret_file.path().display().to_string(),
            "--no-webui".to_owned(),
        ],
        deadlines: Deadlines {
            startup: Duration::from_secs(30),
            shutdown: Duration::from_secs(5),
            health: Duration::from_secs(5),
        },
        capture_lines: mehoy_core::worker::log::DEFAULT_CAPTURE_LINES,
    };

    let supervisor = mehoy_core::worker::ProcessWorker;
    let mut handle = supervisor.spawn(spec).await.expect("backend starts");

    // Wait for the listener rather than assuming it is immediate.
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let mut listening = false;
    while std::time::Instant::now() < deadline {
        if matches!(
            mehoy_backend_llama::health::probe(&channel).await,
            Ok(mehoy_backend_llama::StartupPhase::BackendReady)
        ) {
            listening = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(
        listening,
        "backend never started listening; output:\n{}",
        handle.log().render()
    );

    body(channel).await;

    let _ = supervisor.shutdown(&mut handle).await;
}

#[tokio::test]
async fn the_backend_credential_is_enforced_not_merely_configured() {
    // The backend logs that it registered an API key, which proves configuration
    // and nothing about enforcement. This proves enforcement, against a path
    // measured to require a credential.
    //
    // The health endpoint is deliberately not used: it answers 200 with no
    // credential at all on this build, so an enforcement test written against it
    // would pass while proving nothing.
    with_running_backend(|channel| async move {
        use mehoy_backend_llama::health;

        let unprotected = health::status_without_credential(&channel, health::HEALTH_PATH)
            .await
            .expect("health answers");
        assert_eq!(
            unprotected.as_u16(),
            200,
            "health is expected to be unauthenticated on this build; if that has \
             changed, the credential test should move to it"
        );

        let without = health::status_without_credential(&channel, health::PROPS_PATH)
            .await
            .expect("the protected path answers");
        assert_eq!(
            without.as_u16(),
            401,
            "the path used for the credential check must actually require one, \
             otherwise this test proves nothing"
        );

        let accepted = health::probe_credential(&channel)
            .await
            .expect("the credential check completes");
        assert_eq!(
            accepted,
            mehoy_backend_llama::CredentialState::Accepted,
            "the backend refused the runtime's own credential"
        );
    })
    .await;
}

#[tokio::test]
async fn a_wrong_credential_is_refused_by_the_backend() {
    // The other half of enforcement: the correct credential being accepted is only
    // meaningful if an incorrect one is refused.
    with_running_backend(|channel| async move {
        use mehoy_backend_llama::{BackendChannel, ChannelSecret, CredentialState, health};

        let wrong = BackendChannel::at(
            channel.address(),
            ChannelSecret::generate().expect("generates a different secret"),
        );
        assert_ne!(wrong.secret().expose(), channel.secret().expose());

        let refused = health::probe_credential(&wrong)
            .await
            .expect("the credential check completes");
        match refused {
            CredentialState::Refused { status } => assert_eq!(status, 401),
            CredentialState::Accepted => {
                panic!("the backend accepted a credential it was never given")
            }
        }
    })
    .await;
}
