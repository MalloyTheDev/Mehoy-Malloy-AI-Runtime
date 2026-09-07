//! Acceptance tests for artifact to instance.
//!
//! Failure paths are tested first and hardest, because the property that matters
//! most is not that a good artifact loads but that a bad one leaves nothing behind.
//! A leaked backend holds accelerator memory with nothing left to reclaim it.
//!
//! Tests needing a real backend or a real container skip when one is absent, and
//! say so, rather than passing while proving nothing.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use mehoy_backend_llama::{EXECUTABLE_ENV, LlamaCppBackend};
use mehoy_core::id::IdAllocator;
use mehoy_core::worker::Deadlines;
use mehoy_registry::{ArtifactId, ArtifactRegistry};
use mehoy_runtime::{InstanceState, LoadError, ModelCapability, ModelLoader};

fn worker_id() -> mehoy_core::id::WorkerId {
    static IDS: std::sync::LazyLock<IdAllocator> = std::sync::LazyLock::new(IdAllocator::new);
    IDS.worker()
}

fn deadlines() -> Deadlines {
    Deadlines {
        startup: Duration::from_secs(60),
        shutdown: Duration::from_secs(10),
        health: Duration::from_secs(5),
    }
}

fn loader() -> Option<ModelLoader> {
    match LlamaCppBackend::from_env() {
        Ok(backend) => Some(ModelLoader::new(backend)),
        Err(err) => {
            eprintln!("SKIPPED: no backend available ({err}). Set {EXECUTABLE_ENV} to run.");
            None
        }
    }
}

/// A syntactically valid container that no backend can actually load.
///
/// Enough for the paths that must refuse before a process is ever started.
fn synthetic_container() -> Vec<u8> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&3u64.to_le_bytes());
    let mut entry = |key: &str, value: &str| {
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key.as_bytes());
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    };
    entry("general.architecture", "llama");
    entry("general.name", "synthetic");
    entry("tokenizer.ggml.model", "gpt2");
    bytes
}

fn write_file(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.join(name);
    let mut file = std::fs::File::create(&path).expect("test file is creatable");
    file.write_all(bytes).expect("writable");
    file.sync_all().expect("flushed");
    path
}

/// Real containers on this machine, if any.
fn real_containers() -> Vec<PathBuf> {
    let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) else {
        return Vec::new();
    };
    let mut found = Vec::new();
    for root in [
        PathBuf::from(&home).join(".lmstudio/.internal/bundled-models"),
        PathBuf::from(&home).join(".lmstudio/models"),
    ] {
        collect(&root, &mut found, 0);
    }
    found
}

fn collect(dir: &Path, found: &mut Vec<PathBuf>, depth: usize) {
    if depth > 4 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, found, depth + 1);
        } else if path.extension().is_some_and(|ext| ext == "gguf") {
            found.push(path);
        }
    }
}

/// The smallest real container available, to keep load times honest.
fn smallest_real_container() -> Option<PathBuf> {
    real_containers()
        .into_iter()
        .filter(|path| {
            // Projectors are refused by the preflight, which is a different test.
            !path
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("mmproj"))
        })
        .min_by_key(|path| std::fs::metadata(path).map(|m| m.len()).unwrap_or(u64::MAX))
}

#[cfg(windows)]
fn backend_process_count() -> usize {
    std::process::Command::new("tasklist")
        .args(["/FI", "IMAGENAME eq llama-server.exe", "/NH", "/FO", "CSV"])
        .output()
        .map_or(0, |out| {
            String::from_utf8_lossy(&out.stdout)
                .lines()
                .filter(|line| line.contains("llama-server"))
                .count()
        })
}

#[cfg(unix)]
fn backend_process_count() -> usize {
    std::process::Command::new("pgrep")
        .args(["-c", "llama-server"])
        .output()
        .map_or(0, |out| {
            String::from_utf8_lossy(&out.stdout)
                .trim()
                .parse()
                .unwrap_or(0)
        })
}

/// Serialises every test that starts a backend.
///
/// Two reasons, both of which produced real flakes. The process count these tests
/// assert on is global to the machine, so a concurrent backend makes it meaningless.
/// And the injected commit fault is a process-wide flag, so a concurrent test could
/// consume a failure another test armed.
///
/// Tests that merely refuse before spawning do not need this: the error variant
/// already proves no backend was started, which is a stronger and steadier
/// assertion than counting processes.
async fn exclusive() -> tokio::sync::MutexGuard<'static, ()> {
    // An async-aware lock, because it is held across awaits. A blocking guard held
    // over an await can stall the runtime it is running on.
    static LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    LOCK.lock().await
}

// ---------------------------------------------------------------- failure paths

#[tokio::test]
async fn an_unknown_artifact_is_refused() {
    let Some(loader) = loader() else { return };
    let registry = ArtifactRegistry::open_in_memory().expect("registry opens");

    let err = loader
        .load(
            &registry,
            &ArtifactId::from_stored("no-such-artifact"),
            worker_id(),
            deadlines(),
        )
        .await
        .expect_err("an unregistered id must be refused");

    // The variant is the proof that nothing was spawned: it is returned before the
    // backend is ever asked to start. A global process count would be a weaker
    // assertion and a flaky one, since other tests start backends concurrently.
    assert!(
        matches!(err, LoadError::UnknownArtifact { .. }),
        "expected UnknownArtifact, got {err}"
    );
}

#[tokio::test]
async fn an_artifact_whose_file_was_removed_is_refused() {
    let Some(loader) = loader() else { return };
    let dir = tempfile::tempdir().expect("temp dir");
    let path = write_file(dir.path(), "model.gguf", &synthetic_container());
    let mut registry = ArtifactRegistry::open_in_memory().expect("registry opens");
    let id = registry
        .register(&path)
        .expect("registers")
        .artifact()
        .id
        .clone();

    std::fs::remove_file(&path).expect("removes");

    let err = loader
        .load(&registry, &id, worker_id(), deadlines())
        .await
        .expect_err("a missing file must be refused");

    assert!(
        matches!(err, LoadError::ArtifactMissing { .. }),
        "expected ArtifactMissing, got {err}"
    );
}

#[tokio::test]
async fn a_drifted_artifact_is_refused_rather_than_loaded() {
    // The registry recorded what was inspected. Loading a file that has since been
    // replaced would run something nobody validated.
    let Some(loader) = loader() else { return };
    let dir = tempfile::tempdir().expect("temp dir");
    let path = write_file(dir.path(), "model.gguf", &synthetic_container());
    let mut registry = ArtifactRegistry::open_in_memory().expect("registry opens");
    let id = registry
        .register(&path)
        .expect("registers")
        .artifact()
        .id
        .clone();

    let mut replacement = synthetic_container();
    replacement.extend_from_slice(b"different length now");
    write_file(dir.path(), "model.gguf", &replacement);

    let err = loader
        .load(&registry, &id, worker_id(), deadlines())
        .await
        .expect_err("a drifted artifact must be refused");

    match err {
        LoadError::ArtifactChanged { reason, .. } => assert!(reason.contains("size"), "{reason}"),
        other => panic!("expected ArtifactChanged, got {other}"),
    }
}

#[tokio::test]
async fn an_incompatible_artifact_is_refused_before_spawning() {
    // A projector cannot be served alone, and the preflight knows it, so no process
    // start should be spent discovering that.
    let Some(loader) = loader() else { return };
    let projector = real_containers().into_iter().find(|path| {
        path.file_name()
            .is_some_and(|name| name.to_string_lossy().starts_with("mmproj"))
    });
    let Some(projector) = projector else {
        eprintln!("SKIPPED: no projector container available on this machine");
        return;
    };

    let mut registry = ArtifactRegistry::open_in_memory().expect("registry opens");
    let id = registry
        .register(&projector)
        .expect("registers")
        .artifact()
        .id
        .clone();

    let err = loader
        .load(&registry, &id, worker_id(), deadlines())
        .await
        .expect_err("a projector must be refused");

    match err {
        LoadError::Incompatible { reason } => assert!(reason.contains("projector"), "{reason}"),
        other => panic!("expected Incompatible, got {other}"),
    }
}

#[tokio::test]
async fn a_backend_that_cannot_load_the_artifact_leaves_nothing_running() {
    let _exclusive = exclusive().await;
    // The synthetic container is structurally valid GGUF with no tensors, so the
    // backend starts and then fails to load it. That is the realistic failure.
    let Some(loader) = loader() else { return };
    let dir = tempfile::tempdir().expect("temp dir");
    let path = write_file(dir.path(), "model.gguf", &synthetic_container());
    let mut registry = ArtifactRegistry::open_in_memory().expect("registry opens");
    let id = registry
        .register(&path)
        .expect("registers")
        .artifact()
        .id
        .clone();
    let before = backend_process_count();

    let err = loader
        .load(
            &registry,
            &id,
            worker_id(),
            Deadlines {
                startup: Duration::from_secs(30),
                ..deadlines()
            },
        )
        .await
        .expect_err("an unloadable container must fail");

    assert!(
        matches!(err, LoadError::Backend(_)),
        "expected a backend failure, got {err}"
    );

    // Give the operating system a moment to reap the failed process.
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(
        backend_process_count(),
        before,
        "a failed load left a backend process running"
    );
}

#[tokio::test]
async fn a_failure_after_readiness_still_tears_the_backend_down() {
    let _exclusive = exclusive().await;
    // The hardest path. Everything else fails before a process exists; this one
    // fails while a backend is healthy, authenticated, and holding memory. Without
    // an explicit rollback it would be leaked.
    let Some(loader) = loader() else { return };
    let Some(container) = smallest_real_container() else {
        eprintln!("SKIPPED: no real container available to reach readiness");
        return;
    };

    let mut registry = ArtifactRegistry::open_in_memory().expect("registry opens");
    let id = registry
        .register(&container)
        .expect("registers")
        .artifact()
        .id
        .clone();

    let before = backend_process_count();
    mehoy_runtime::fail_next_commit();

    let err = loader
        .load(&registry, &id, worker_id(), deadlines())
        .await
        .expect_err("the injected failure must surface");
    mehoy_runtime::clear_injected_faults();

    match err {
        LoadError::Instance { reason } => assert!(reason.contains("injected"), "{reason}"),
        other => panic!("expected an instance failure, got {other}"),
    }

    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(
        backend_process_count(),
        before,
        "a failure after readiness leaked a running backend"
    );
}

// ------------------------------------------------------------------ happy path

#[tokio::test]
async fn a_registered_artifact_loads_and_yields_one_instance() {
    let _exclusive = exclusive().await;
    let Some(loader) = loader() else { return };
    let Some(container) = smallest_real_container() else {
        eprintln!("SKIPPED: no real container available on this machine");
        return;
    };

    let mut registry = ArtifactRegistry::open_in_memory().expect("registry opens");
    let artifact = registry
        .register(&container)
        .expect("registers")
        .artifact()
        .clone();
    let before = backend_process_count();

    let loaded = loader
        .load(&registry, &artifact.id, worker_id(), deadlines())
        .await
        .unwrap_or_else(|err| panic!("{} did not load: {err}", container.display()));

    let instance = loaded.instance();
    assert_eq!(instance.state(), &InstanceState::BackendReady);
    assert!(instance.state().is_usable());

    // The instance references its artifact rather than duplicating ownership of it.
    assert_eq!(instance.artifact_id(), &artifact.id);

    // The backend build is recorded, so a later failure can name it.
    assert!(
        instance.backend().is_some(),
        "the instance should record which backend build is running it"
    );

    eprintln!(
        "loaded {} as instance {} on {}",
        container.file_name().unwrap_or_default().to_string_lossy(),
        instance.id(),
        instance
            .backend()
            .map_or_else(|| "?".to_owned(), ToString::to_string)
    );

    loaded.unload().await.expect("unloads");
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(
        backend_process_count(),
        before,
        "unloading must reap the backend"
    );
}

#[tokio::test]
async fn readiness_does_not_claim_any_capability() {
    let _exclusive = exclusive().await;
    // The reason an embedding model was chosen first. Reaching BackendReady means
    // the artifact loaded and the backend answers authenticated requests. It must
    // not quietly become a claim that generation works.
    let Some(loader) = loader() else { return };
    let Some(container) = smallest_real_container() else {
        eprintln!("SKIPPED: no real container available on this machine");
        return;
    };

    let mut registry = ArtifactRegistry::open_in_memory().expect("registry opens");
    let id = registry
        .register(&container)
        .expect("registers")
        .artifact()
        .id
        .clone();

    let loaded = loader
        .load(&registry, &id, worker_id(), deadlines())
        .await
        .expect("loads");

    let capabilities = loaded.instance().capabilities();
    for capability in [
        ModelCapability::TextGeneration,
        ModelCapability::Embeddings,
        ModelCapability::Vision,
        ModelCapability::ToolCalling,
        ModelCapability::StructuredOutput,
    ] {
        assert!(
            !capabilities.is_verified(capability),
            "{capability} was reported verified, but nothing has been demonstrated; \
             loading is not the same as working"
        );
    }

    loaded.unload().await.expect("unloads");
}
