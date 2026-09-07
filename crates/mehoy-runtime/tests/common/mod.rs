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
use mehoy_runtime::ModelLoader;

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

pub fn loader() -> Option<ModelLoader> {
    match LlamaCppBackend::from_env() {
        Ok(backend) => Some(ModelLoader::new(backend)),
        Err(err) => {
            eprintln!("SKIPPED: no backend available ({err}). Set {EXECUTABLE_ENV} to run.");
            None
        }
    }
}
