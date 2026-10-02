//! Proving an embedding capability by performing one.
//!
//! The point of this slice is not that embeddings work. It is that the runtime
//! contract holds for a model that generates nothing: a non-generative artifact
//! loads, serves a real authenticated request, and only then earns a capability
//! claim. If the runtime had text-generation assumptions in it, this is where they
//! would show.
//!
//! Nothing here judges model quality. Vector values move with quantisation,
//! pooling, backend build, and floating point, so a similarity threshold would be a
//! flaky gate that measures the model rather than the runtime.

use std::path::{Path, PathBuf};
use std::time::Duration;

use mehoy_backend_llama::{EXECUTABLE_ENV, LlamaCppBackend};
use mehoy_core::id::IdAllocator;
use mehoy_core::inference::EmbedRequest;
use mehoy_core::worker::Deadlines;
use mehoy_registry::ArtifactRegistry;
use mehoy_runtime::{InstanceId, ModelCapability, Runtime, RuntimeError};

/// The model card requires a task prefix; without one the vectors are not what the
/// model was trained to produce.
const QUERY_PREFIX: &str = "search_query: ";

fn worker_id() -> mehoy_core::id::WorkerId {
    static IDS: std::sync::LazyLock<IdAllocator> = std::sync::LazyLock::new(IdAllocator::new);
    IDS.worker()
}

/// Serialises tests that start a backend, since several assert on machine state.
async fn exclusive() -> tokio::sync::MutexGuard<'static, ()> {
    static LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
    LOCK.lock().await
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

/// An embedding-oriented container, if one is present on this machine.
fn embedding_container() -> Option<PathBuf> {
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))?;
    let mut found = Vec::new();
    for root in [
        PathBuf::from(&home).join(".lmstudio/.internal/bundled-models"),
        PathBuf::from(&home).join(".lmstudio/models"),
    ] {
        collect(&root, &mut found, 0);
    }
    found.into_iter().find(|path| {
        path.file_name()
            .is_some_and(|name| name.to_string_lossy().contains("embed"))
    })
}

/// Loads an embedding model, or explains why the test is skipped.
async fn with_embedding_model<F, Fut>(body: F)
where
    F: FnOnce(Runtime, InstanceId) -> Fut,
    Fut: Future<Output = (Runtime, InstanceId)>,
{
    let Ok(backend) = LlamaCppBackend::from_env() else {
        eprintln!("SKIPPED: no backend available. Set {EXECUTABLE_ENV} to run.");
        return;
    };
    let Some(container) = embedding_container() else {
        eprintln!("SKIPPED: no embedding container found on this machine");
        return;
    };

    let mut registry = ArtifactRegistry::open_in_memory().expect("registry opens");
    let id = registry
        .register(&container)
        .expect("registers")
        .artifact()
        .id
        .clone();

    let runtime = Runtime::new(backend);
    let instance = runtime
        .load(
            &registry,
            &id,
            worker_id(),
            Deadlines {
                startup: Duration::from_secs(90),
                shutdown: Duration::from_secs(10),
                health: Duration::from_secs(5),
            },
        )
        .await
        .unwrap_or_else(|err| panic!("{} did not load: {err}", container.display()));

    let (runtime, instance) = body(runtime, instance).await;
    // A body may have unloaded already, which is not a failure to clean up.
    match runtime.unload(&instance).await {
        Ok(_) | Err(RuntimeError::UnknownInstance { .. }) => {}
        Err(err) => panic!("the instance did not unload: {err}"),
    }
}

#[tokio::test]
async fn an_embedding_request_returns_a_usable_vector() {
    let _exclusive = exclusive().await;
    with_embedding_model(|runtime, id| async move {
        let request = EmbedRequest::single(format!("{QUERY_PREFIX}What is virtual memory?"));
        let result = runtime
            .embed(&id, &request)
            .await
            .expect("the model embeds");

        assert_eq!(result.embeddings.len(), 1, "one input, one vector");
        let embedding = result.for_input(0).expect("attributable to input 0");

        assert!(embedding.dimension() > 0, "the vector must not be empty");
        assert!(
            embedding.is_finite(),
            "a non-finite component silently poisons every distance computed from it"
        );

        // The engine documents these as Euclidean-normalised, which is a far
        // stronger check than merely receiving some floats.
        let norm = embedding.l2_norm();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "expected a unit vector, got a norm of {norm}"
        );

        eprintln!(
            "embedded one input: dimension {}, norm {norm:.6}",
            embedding.dimension()
        );
        (runtime, id)
    })
    .await;
}

#[tokio::test]
async fn the_dimension_is_stable_across_requests() {
    // Discovered from the first result rather than asserted against a constant, so
    // the test stays about the runtime rather than about this particular model.
    let _exclusive = exclusive().await;
    with_embedding_model(|runtime, id| async move {
        let first = runtime
            .embed(&id, &EmbedRequest::single(format!("{QUERY_PREFIX}first")))
            .await
            .expect("embeds");
        let expected = first.dimension().expect("a consistent dimension");

        for round in 0..3 {
            let again = runtime
                .embed(
                    &id,
                    &EmbedRequest::single(format!("{QUERY_PREFIX}round {round}")),
                )
                .await
                .expect("embeds");
            assert_eq!(
                again.dimension(),
                Some(expected),
                "dimension changed between requests to one loaded instance"
            );
        }
        (runtime, id)
    })
    .await;
}

#[tokio::test]
async fn a_batch_returns_one_independently_indexed_vector_per_input() {
    let _exclusive = exclusive().await;
    with_embedding_model(|runtime, id| async move {
        let request = EmbedRequest::batch([
            format!("{QUERY_PREFIX}What is virtual memory?"),
            format!("{QUERY_PREFIX}How do I bake a cake?"),
        ]);
        let result = runtime.embed(&id, &request).await.expect("embeds a batch");

        assert_eq!(result.embeddings.len(), 2);
        let first = result.for_input(0).expect("input 0 answered");
        let second = result.for_input(1).expect("input 1 answered");
        assert_eq!(first.dimension(), second.dimension());

        // Different inputs must not produce the same vector. This is a
        // non-degeneracy check, not a quality measurement: no similarity threshold
        // is asserted, because those move with quantisation and backend build.
        assert_ne!(
            first.vector, second.vector,
            "two unrelated inputs produced an identical vector, which suggests the \
             input was not reaching the model"
        );
        (runtime, id)
    })
    .await;
}

#[tokio::test]
async fn the_same_input_embeds_consistently() {
    let _exclusive = exclusive().await;
    with_embedding_model(|runtime, id| async move {
        let input = format!("{QUERY_PREFIX}What is virtual memory?");
        let first = runtime
            .embed(&id, &EmbedRequest::single(input.clone()))
            .await
            .expect("embeds");
        let second = runtime
            .embed(&id, &EmbedRequest::single(input))
            .await
            .expect("embeds again");

        let a = first.for_input(0).expect("answered");
        let b = second.for_input(0).expect("answered");
        assert_eq!(a.dimension(), b.dimension());

        // Equal or extremely close. Exact equality is not required, because batching
        // and scheduling can legitimately change the last bits.
        let drift: f32 = a
            .vector
            .iter()
            .zip(&b.vector)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        assert!(
            drift < 1e-3,
            "the same input embedded differently, largest component drift {drift}"
        );
        (runtime, id)
    })
    .await;
}

#[tokio::test]
async fn embeddings_are_only_verified_by_performing_one() {
    // The claim this whole slice exists to establish.
    let _exclusive = exclusive().await;
    with_embedding_model(|runtime, id| async move {
        assert!(
            !runtime
                .instance(&id)
                .expect("resident")
                .capabilities()
                .is_verified(ModelCapability::Embeddings),
            "loading must not verify a capability; only performing one may"
        );

        runtime
            .verify_embeddings(
                &id,
                &EmbedRequest::single(format!("{QUERY_PREFIX}What is virtual memory?")),
            )
            .await
            .expect("the model embeds");

        assert!(
            runtime
                .instance(&id)
                .expect("resident")
                .capabilities()
                .is_verified(ModelCapability::Embeddings),
            "a successful embedding should have verified the capability"
        );

        // Verification belongs to this live instance on this backend build, and is
        // recorded with the build that demonstrated it.
        let state = runtime
            .instance(&id)
            .expect("resident")
            .capabilities()
            .state(ModelCapability::Embeddings);
        eprintln!("embeddings capability is now {state}");

        // Nothing else was verified by association.
        for other in [
            ModelCapability::TextGeneration,
            ModelCapability::Vision,
            ModelCapability::ToolCalling,
            ModelCapability::StructuredOutput,
        ] {
            assert!(
                !runtime
                    .instance(&id)
                    .expect("resident")
                    .capabilities()
                    .is_verified(other),
                "{other} was verified without being demonstrated"
            );
        }
        (runtime, id)
    })
    .await;
}

#[tokio::test]
async fn verification_does_not_survive_the_instance() {
    // A demonstration is a fact about a live instance on a particular backend, not a
    // permanent property of the artifact. A fresh load starts unverified.
    let _exclusive = exclusive().await;
    let Ok(backend) = LlamaCppBackend::from_env() else {
        eprintln!("SKIPPED: no backend available. Set {EXECUTABLE_ENV} to run.");
        return;
    };
    let Some(container) = embedding_container() else {
        eprintln!("SKIPPED: no embedding container found on this machine");
        return;
    };

    let mut registry = ArtifactRegistry::open_in_memory().expect("registry opens");
    let id = registry
        .register(&container)
        .expect("registers")
        .artifact()
        .id
        .clone();
    let runtime = Runtime::new(backend);
    let deadlines = Deadlines {
        startup: Duration::from_secs(90),
        shutdown: Duration::from_secs(10),
        health: Duration::from_secs(5),
    };

    let first = runtime
        .load(&registry, &id, worker_id(), deadlines)
        .await
        .expect("loads");
    runtime
        .verify_embeddings(
            &first,
            &EmbedRequest::single(format!("{QUERY_PREFIX}hello")),
        )
        .await
        .expect("embeds");
    assert!(
        runtime
            .instance(&first)
            .expect("resident")
            .capabilities()
            .is_verified(ModelCapability::Embeddings)
    );
    runtime.unload(&first).await.expect("unloads");

    let second = runtime
        .load(&registry, &id, worker_id(), deadlines)
        .await
        .expect("loads again");
    assert!(
        !runtime
            .instance(&second)
            .expect("resident")
            .capabilities()
            .is_verified(ModelCapability::Embeddings),
        "a new instance inherited a verification it never demonstrated"
    );
    runtime.unload(&second).await.expect("unloads");
}

#[tokio::test]
async fn a_request_to_an_instance_that_is_gone_is_refused() {
    // Nothing is sent to a backend for an instance the runtime no longer holds,
    // so the caller gets a clear refusal rather than a transport error from a
    // socket nobody is listening on. The identifier is not reused, so a stale one
    // can never reach a later model either.
    let _exclusive = exclusive().await;
    with_embedding_model(|runtime, id| async move {
        runtime.unload(&id).await.expect("unloads");

        let err = runtime
            .embed(&id, &EmbedRequest::single("x"))
            .await
            .expect_err("an unloaded instance must not serve requests");
        assert!(
            matches!(err, RuntimeError::UnknownInstance { .. }),
            "expected UnknownInstance, got {err}"
        );
        (runtime, id)
    })
    .await;
}
