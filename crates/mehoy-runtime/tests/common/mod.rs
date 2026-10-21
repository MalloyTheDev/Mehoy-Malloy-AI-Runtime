//! Shared scaffolding for tests that need a real backend and a real container.
//!
//! Every test here depends on machine-local state: an executable named by an
//! environment variable, and model containers that may or may not be present. That
//! is deliberate, because the questions these tests answer cannot be answered
//! against a stand-in. The cost is that each one has to be able to skip cleanly,
//! and the helpers below exist so skipping looks the same everywhere rather than
//! being reinvented per file.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::time::Duration;

use mehoy_backend_llama::compatibility::{self, LaunchMode};
use mehoy_backend_llama::{EXECUTABLE_ENV, LlamaCppBackend, ModelDescriptor};
use mehoy_core::id::{IdAllocator, WorkerId};
use mehoy_core::worker::Deadlines;
use mehoy_registry::{ArtifactRegistry, ModelArtifact};
use mehoy_runtime::{ResidentLimits, Runtime};

pub fn worker_id() -> WorkerId {
    static IDS: std::sync::LazyLock<IdAllocator> = std::sync::LazyLock::new(IdAllocator::new);
    IDS.worker()
}

/// Serialises tests that start a backend.
///
/// A multi-gigabyte model is loaded per test, and running two at once means two
/// resident copies competing for the same accelerator.
pub async fn exclusive() -> tokio::sync::MutexGuard<'static, ()> {
    static LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    LOCK.lock().await
}

pub fn deadlines() -> Deadlines {
    Deadlines {
        // A multi-gigabyte container takes real time to reach the accelerator.
        startup: Duration::from_secs(180),
        shutdown: Duration::from_secs(15),
        health: Duration::from_secs(5),
    }
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

pub fn containers() -> Vec<PathBuf> {
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

/// Registers every container found and returns them classified by what their own
/// metadata says, not by their filename.
pub fn survey() -> Option<(ArtifactRegistry, Vec<ModelArtifact>)> {
    let mut registry = ArtifactRegistry::open_in_memory().expect("registry opens");
    let mut artifacts = Vec::new();
    for path in containers() {
        if let Ok(registration) = registry.register(&path) {
            artifacts.push(registration.artifact().clone());
        }
    }
    if artifacts.is_empty() {
        eprintln!("SKIPPED: no containers found on this machine");
        return None;
    }
    Some((registry, artifacts))
}

pub fn describe(artifact: &ModelArtifact) -> ModelDescriptor {
    ModelDescriptor {
        architecture: artifact.metadata.architecture.clone(),
        context_length: artifact.metadata.context_length,
        embedding_length: artifact.metadata.embedding_length,
        tokenizer_model: artifact.metadata.tokenizer_model.clone(),
        has_chat_template: artifact.metadata.has_chat_template(),
    }
}

/// A container the backend would start in its ordinary, non-embedding mode.
pub fn generative_artifact(artifacts: &[ModelArtifact]) -> Option<&ModelArtifact> {
    artifacts.iter().find(|artifact| {
        let descriptor = describe(artifact);
        compatibility::assess(&descriptor).permits_start()
            && compatibility::launch_mode(&descriptor) == LaunchMode::General
            && artifact.metadata.has_chat_template()
    })
}

pub fn embedding_artifact(artifacts: &[ModelArtifact]) -> Option<&ModelArtifact> {
    artifacts
        .iter()
        .find(|artifact| compatibility::launch_mode(&describe(artifact)) == LaunchMode::Embedding)
}

pub fn runtime() -> Option<Runtime> {
    runtime_with_limits(ResidentLimits::default())
}

/// A runtime that will hold what `limits` says, or an explained skip.
pub fn runtime_with_limits(limits: ResidentLimits) -> Option<Runtime> {
    match LlamaCppBackend::from_env() {
        Ok(backend) => Some(Runtime::with_limits(backend, limits)),
        Err(err) => {
            eprintln!("SKIPPED: no backend available ({err}). Set {EXECUTABLE_ENV} to run.");
            None
        }
    }
}

/// Loads the one generative container on this machine, or explains the skip.
///
/// Returns the runtime alongside the identifier, because the runtime owns the
/// model: letting it drop would take the instance with it.
pub async fn loaded_generative() -> Option<(Runtime, ArtifactRegistry, mehoy_runtime::InstanceId)> {
    let runtime = runtime()?;
    let (registry, artifacts) = survey()?;
    let Some(artifact) = generative_artifact(&artifacts) else {
        eprintln!("SKIPPED: no text-generative container found on this machine");
        return None;
    };
    let id = runtime
        .load(&registry, &artifact.id, worker_id(), deadlines())
        .await
        .expect("loads");
    Some((runtime, registry, id))
}

// ---------------------------------------------------------------------------
// Testing the runtime without an engine or a container
//
// Everything above needs a real `llama-server` and a multi-gigabyte container,
// which is why most of what this runtime does was only ever exercised on one
// machine. The two helpers below remove both requirements: a stand-in executable
// that answers the adapter's requests with shaped responses, and a synthetic
// container built byte by byte.
//
// What they cannot do is verify a capability. Only performing a task does that,
// and a stand-in performs nothing, so no test built on these may promote one.
// ---------------------------------------------------------------------------

/// Locates an example binary built alongside this test.
///
/// The test binary's own location varies between cargo versions and layouts, so
/// this walks up looking for the `examples` directory rather than assuming a
/// fixed relative path.
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

/// A runtime whose backend is the stand-in rather than an engine.
///
/// Panics rather than skipping when the stand-in is missing. Unlike a container
/// or an installed engine, it is built by `cargo test` from this workspace, so
/// its absence is a broken build and not a machine that lacks something.
///
/// One trap worth knowing: `cargo test --test stand_in` restricts the build to
/// that one target and does not build examples, so it can run against whatever
/// copy of the helper is already on disk. `cargo test -p mehoy-runtime` and
/// `cargo test --workspace` both build it. A stale helper usually announces
/// itself, since an unrecognised behaviour name makes it exit with a message,
/// but not if the behaviour being tested used to exist and no longer does.
pub fn stand_in_runtime() -> Runtime {
    stand_in_runtime_with_limits(ResidentLimits::default())
}

/// The same, with explicit limits.
pub fn stand_in_runtime_with_limits(limits: ResidentLimits) -> Runtime {
    let executable = example_binary("stand_in_backend").unwrap_or_else(|| {
        panic!("the stand_in_backend example was not built; `cargo test` builds it")
    });
    let backend = LlamaCppBackend::new(executable).expect("the stand-in is an existing file");
    Runtime::with_limits(backend, limits)
}

/// Deadlines for a backend that starts immediately.
///
/// Short, so a test that is going to fail does so quickly. The long deadlines
/// above exist for a real engine moving gigabytes onto an accelerator.
pub fn stand_in_deadlines() -> Deadlines {
    Deadlines {
        startup: Duration::from_secs(10),
        shutdown: Duration::from_secs(5),
        health: Duration::from_secs(2),
    }
}

/// A synthetic GGUF container, written to a temporary directory and registered.
///
/// Held together so the directory outlives the registration: dropping it removes
/// the file, and the loader checks for drift immediately before every load.
pub struct Fixture {
    pub registry: ArtifactRegistry,
    pub artifact: mehoy_registry::ArtifactId,
    pub size_bytes: u64,
    /// Kept alive, not used. The file must outlive every load of it.
    _dir: tempfile::TempDir,
}

/// Registers a synthetic container.
///
/// `behaviour` is passed to the stand-in through the file name, which is the one
/// channel a test controls: the runtime builds the command line, and an
/// environment variable would be process-wide and so shared between concurrent
/// tests. Pass `None` for a container that simply serves.
pub fn fixture(behaviour: Option<&str>) -> Fixture {
    static NAMES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let unique = NAMES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let name = match behaviour {
        Some(behaviour) => format!("fixture-{unique}__{behaviour}.gguf"),
        None => format!("fixture-{unique}.gguf"),
    };

    let dir = tempfile::tempdir().expect("a temporary directory");
    let path = dir.path().join(&name);
    let bytes = container();
    std::fs::write(&path, &bytes).expect("the fixture is writable");

    let mut registry = ArtifactRegistry::open_in_memory().expect("registry opens");
    let artifact = registry
        .register(&path)
        .expect("a well-formed container registers")
        .artifact()
        .id
        .clone();

    Fixture {
        registry,
        artifact,
        size_bytes: bytes.len() as u64,
        _dir: dir,
    }
}

/// A well-formed container carrying the fields the runtime reasons about.
///
/// Built here rather than shared with the registry's own fixtures, which are
/// private to that crate's tests. It declares an architecture and a tokenizer,
/// which is what the compatibility preflight requires before it will let a
/// backend be started, and a chat template so it reads as generative.
fn container() -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    let u32 = |out: &mut Vec<u8>, value: u32| out.extend_from_slice(&value.to_le_bytes());
    let u64 = |out: &mut Vec<u8>, value: u64| out.extend_from_slice(&value.to_le_bytes());
    let string = |out: &mut Vec<u8>, value: &str| {
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value.as_bytes());
    };

    u32(&mut out, 3);
    u64(&mut out, 0);
    u64(&mut out, 6);

    let text = |out: &mut Vec<u8>, key: &str, value: &str| {
        string(out, key);
        u32(out, 8);
        string(out, value);
    };
    text(&mut out, "general.architecture", "llama");
    text(&mut out, "general.name", "stand-in fixture");
    text(&mut out, "tokenizer.ggml.model", "gpt2");
    text(&mut out, "tokenizer.chat_template", "{{ messages }}");

    let number = |out: &mut Vec<u8>, key: &str, value: u32| {
        string(out, key);
        u32(out, 4);
        u32(out, value);
    };
    number(&mut out, "llama.context_length", 4096);
    number(&mut out, "llama.embedding_length", 8);

    out
}
