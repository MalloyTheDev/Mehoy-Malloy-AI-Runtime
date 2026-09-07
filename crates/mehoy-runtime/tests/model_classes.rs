//! Two different model classes through one runtime path.
//!
//! This is the empirical test of ADR-0006 rather than its restatement. An embedding
//! model and a text-generative model are loaded through the same public call, with
//! the same transaction, and the runtime is not permitted to branch on which it is.
//! Whatever differs between them is the backend's business.
//!
//! What is deliberately not proven here: that either model can actually perform its
//! task. Loading is lifecycle. Performing is a later slice, and conflating the two
//! is the mistake this whole design exists to avoid.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mehoy_backend_llama::compatibility::{self, LaunchMode};
use mehoy_backend_llama::{EXECUTABLE_ENV, LlamaCppBackend, ModelDescriptor};
use mehoy_core::id::IdAllocator;
use mehoy_core::worker::Deadlines;
use mehoy_registry::{ArtifactRegistry, ModelArtifact};
use mehoy_runtime::{InstanceState, ModelCapability, ModelLoader};

fn worker_id() -> mehoy_core::id::WorkerId {
    static IDS: std::sync::LazyLock<IdAllocator> = std::sync::LazyLock::new(IdAllocator::new);
    IDS.worker()
}

/// Serialises tests that start a backend.
async fn exclusive() -> tokio::sync::MutexGuard<'static, ()> {
    static LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    LOCK.lock().await
}

fn deadlines() -> Deadlines {
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

fn containers() -> Vec<PathBuf> {
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
fn survey() -> Option<(ArtifactRegistry, Vec<ModelArtifact>)> {
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

fn describe(artifact: &ModelArtifact) -> ModelDescriptor {
    ModelDescriptor {
        architecture: artifact.metadata.architecture.clone(),
        context_length: artifact.metadata.context_length,
        embedding_length: artifact.metadata.embedding_length,
        tokenizer_model: artifact.metadata.tokenizer_model.clone(),
        has_chat_template: artifact.metadata.has_chat_template(),
    }
}

/// A container the backend would start in its ordinary, non-embedding mode.
fn generative_artifact(artifacts: &[ModelArtifact]) -> Option<&ModelArtifact> {
    artifacts.iter().find(|artifact| {
        let descriptor = describe(artifact);
        compatibility::assess(&descriptor).permits_start()
            && compatibility::launch_mode(&descriptor) == LaunchMode::General
            && artifact.metadata.has_chat_template()
    })
}

fn embedding_artifact(artifacts: &[ModelArtifact]) -> Option<&ModelArtifact> {
    artifacts
        .iter()
        .find(|artifact| compatibility::launch_mode(&describe(artifact)) == LaunchMode::Embedding)
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

#[test]
fn the_preflight_separates_the_two_classes_without_the_runtime_asking() {
    // Decided from the artifact's own metadata, before any process exists, by the
    // backend rather than by a caller who would have to know what a mode is.
    let Some((_registry, artifacts)) = survey() else {
        return;
    };

    let mut saw_general = false;
    let mut saw_embedding = false;
    for artifact in &artifacts {
        let descriptor = describe(artifact);
        let mode = compatibility::launch_mode(&descriptor);
        eprintln!(
            "{}: architecture {:?}, chat template {}, launch mode {mode:?}",
            artifact
                .path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            artifact.metadata.architecture.as_deref().unwrap_or("?"),
            artifact.metadata.has_chat_template(),
        );
        match mode {
            LaunchMode::General => saw_general = true,
            LaunchMode::Embedding => saw_embedding = true,
        }
    }

    assert!(
        saw_general,
        "no container on this machine would start in the ordinary mode"
    );
    if !saw_embedding {
        eprintln!("note: no embedding-oriented container present, so that half is unproven here");
    }
}

#[tokio::test]
async fn a_generative_artifact_loads_through_the_same_path() {
    let _exclusive = exclusive().await;
    let Some(loader) = loader() else { return };
    let Some((registry, artifacts)) = survey() else {
        return;
    };
    let Some(artifact) = generative_artifact(&artifacts) else {
        eprintln!("SKIPPED: no text-generative container found on this machine");
        return;
    };

    eprintln!(
        "loading {} ({} MB, architecture {:?})",
        artifact
            .path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy(),
        artifact.size_bytes / (1024 * 1024),
        artifact.metadata.architecture.as_deref().unwrap_or("?"),
    );

    let began = Instant::now();
    let loaded = loader
        .load(&registry, &artifact.id, worker_id(), deadlines())
        .await
        .unwrap_or_else(|err| panic!("{} did not load: {err}", artifact.path.display()));
    eprintln!("reached backend ready in {:?}", began.elapsed());

    let instance = loaded.instance();
    assert_eq!(instance.state(), &InstanceState::BackendReady);
    assert_eq!(instance.artifact_id(), &artifact.id);
    assert!(instance.backend().is_some());

    // Readiness means the artifact loaded and authenticated requests are accepted.
    // It says nothing about generation, exactly as it said nothing about embedding.
    assert!(
        !instance
            .capabilities()
            .is_verified(ModelCapability::TextGeneration),
        "loading a generative model must not verify generation; only generating may"
    );
    assert!(
        !instance
            .capabilities()
            .is_verified(ModelCapability::Embeddings),
        "a generative model must not acquire an embedding claim by association"
    );

    // The artifact's own metadata may suggest generation. A suggestion is all it is.
    let generation = instance
        .capabilities()
        .state(ModelCapability::TextGeneration);
    eprintln!("text generation is currently {generation}");

    loaded.unload().await.expect("unloads");
}

#[tokio::test]
async fn both_model_classes_travel_the_same_public_call() {
    // The architectural claim of ADR-0006, tested rather than asserted. If the
    // runtime had grown separate entry points for the two classes, this test could
    // not be written the way it is: one loader, one call, one transaction.
    let _exclusive = exclusive().await;
    let Some(loader) = loader() else { return };
    let Some((registry, artifacts)) = survey() else {
        return;
    };

    let Some(embedding) = embedding_artifact(&artifacts) else {
        eprintln!("SKIPPED: no embedding container present");
        return;
    };
    let Some(generative) = generative_artifact(&artifacts) else {
        eprintln!("SKIPPED: no generative container present");
        return;
    };
    assert_ne!(
        embedding.id, generative.id,
        "the two classes must be different artifacts for this to prove anything"
    );

    // One call, twice, differing only in which artifact is named.
    for artifact in [embedding, generative] {
        let loaded = loader
            .load(&registry, &artifact.id, worker_id(), deadlines())
            .await
            .unwrap_or_else(|err| panic!("{} did not load: {err}", artifact.path.display()));

        assert_eq!(loaded.instance().state(), &InstanceState::BackendReady);
        eprintln!(
            "{:?} reached backend ready through the same call",
            artifact.metadata.architecture.as_deref().unwrap_or("?")
        );

        // Neither class arrives with a verified capability. Both start out having
        // proven nothing beyond being resident.
        for capability in [
            ModelCapability::TextGeneration,
            ModelCapability::Embeddings,
            ModelCapability::Vision,
            ModelCapability::ToolCalling,
            ModelCapability::StructuredOutput,
        ] {
            assert!(
                !loaded.instance().capabilities().is_verified(capability),
                "{capability} was verified merely by loading"
            );
        }

        loaded.unload().await.expect("unloads");
    }
}

#[tokio::test]
async fn a_reloaded_generative_instance_inherits_nothing() {
    let _exclusive = exclusive().await;
    let Some(loader) = loader() else { return };
    let Some((registry, artifacts)) = survey() else {
        return;
    };
    let Some(artifact) = generative_artifact(&artifacts) else {
        eprintln!("SKIPPED: no text-generative container found on this machine");
        return;
    };

    let first = loader
        .load(&registry, &artifact.id, worker_id(), deadlines())
        .await
        .expect("loads");
    let first_id = first.instance().id().clone();
    first.unload().await.expect("unloads");

    let second = loader
        .load(&registry, &artifact.id, worker_id(), deadlines())
        .await
        .expect("loads again");

    assert_ne!(
        second.instance().id(),
        &first_id,
        "a reload must produce a new instance rather than resurrecting the old one"
    );
    assert!(
        !second
            .instance()
            .capabilities()
            .is_verified(ModelCapability::TextGeneration),
        "a fresh instance inherited a verification it never demonstrated"
    );
    second.unload().await.expect("unloads");
}
