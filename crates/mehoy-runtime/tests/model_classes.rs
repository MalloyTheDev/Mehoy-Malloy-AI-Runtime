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

use std::time::Instant;

use mehoy_backend_llama::compatibility::{self, LaunchMode};
use mehoy_runtime::{InstanceState, LoadError, ModelCapability};

mod common;
use common::{
    deadlines, describe, embedding_artifact, exclusive, generative_artifact, loader, survey,
    worker_id,
};

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

// --------------------------------------------------------------- text generation

/// A deliberately dull continuation prompt with a small budget.
///
/// The test proves machinery, not intelligence. Whether the model produces a good
/// answer is a question about the model; whether the runtime can carry a request to
/// a backend and a result back is the question here.
fn generation_request() -> mehoy_core::inference::GenerateTextRequest {
    use mehoy_core::inference::{GenerateTextRequest, GenerationParameters};
    GenerateTextRequest::continuation(
        "Complete this sentence with one word: The opposite of hot is",
    )
    .with_parameters(GenerationParameters {
        max_output_tokens: Some(8),
        temperature: Some(0.0),
        seed: Some(42),
        stop: Vec::new(),
    })
}

#[tokio::test]
async fn a_generative_model_actually_generates() {
    use mehoy_core::inference::FinishReason;

    let _exclusive = exclusive().await;
    let Some(loader) = loader() else { return };
    let Some((registry, artifacts)) = survey() else {
        return;
    };
    let Some(artifact) = generative_artifact(&artifacts) else {
        eprintln!("SKIPPED: no text-generative container found on this machine");
        return;
    };

    let loaded = loader
        .load(&registry, &artifact.id, worker_id(), deadlines())
        .await
        .expect("loads");

    let result = loaded
        .generate_text(&generation_request())
        .await
        .expect("the model generates");

    // Machinery assertions only.
    assert!(!result.text.is_empty(), "generation produced no content");
    assert!(
        matches!(
            result.finish_reason,
            FinishReason::Stop | FinishReason::Length
        ),
        "unrecognised finish reason {}",
        result.finish_reason
    );
    if let Some(usage) = result.usage {
        assert!(
            usage.output_tokens > 0,
            "content returned but no output tokens"
        );
        assert!(
            usage.input_tokens > 0,
            "a prompt was sent but none was counted"
        );
        assert!(
            usage.output_tokens <= 8,
            "produced {} tokens against a budget of 8",
            usage.output_tokens
        );
    }

    eprintln!(
        "generated {:?} (finish: {}, usage: {:?})",
        result.text, result.finish_reason, result.usage
    );

    loaded.unload().await.expect("unloads");
}

#[tokio::test]
async fn generation_is_only_verified_by_generating() {
    let _exclusive = exclusive().await;
    let Some(loader) = loader() else { return };
    let Some((registry, artifacts)) = survey() else {
        return;
    };
    let Some(artifact) = generative_artifact(&artifacts) else {
        eprintln!("SKIPPED: no text-generative container found on this machine");
        return;
    };

    let mut loaded = loader
        .load(&registry, &artifact.id, worker_id(), deadlines())
        .await
        .expect("loads");

    assert!(
        !loaded
            .instance()
            .capabilities()
            .is_verified(ModelCapability::TextGeneration),
        "loading must not verify generation"
    );

    loaded
        .verify_text_generation(&generation_request())
        .await
        .expect("generates");

    assert!(
        loaded
            .instance()
            .capabilities()
            .is_verified(ModelCapability::TextGeneration),
        "a successful generation should have verified the capability"
    );
    eprintln!(
        "text generation is now {}",
        loaded
            .instance()
            .capabilities()
            .state(ModelCapability::TextGeneration)
    );

    // Nothing else was verified by association. In particular a generative model
    // does not acquire an embedding claim.
    for other in [
        ModelCapability::Embeddings,
        ModelCapability::Vision,
        ModelCapability::ToolCalling,
        ModelCapability::StructuredOutput,
    ] {
        assert!(
            !loaded.instance().capabilities().is_verified(other),
            "{other} was verified without being demonstrated"
        );
    }

    loaded.unload().await.expect("unloads");
}

#[tokio::test]
async fn an_unusable_parameter_is_refused_before_reaching_a_backend() {
    use mehoy_core::inference::{GenerateTextRequest, GenerationParameters};

    let _exclusive = exclusive().await;
    let Some(loader) = loader() else { return };
    let Some((registry, artifacts)) = survey() else {
        return;
    };
    let Some(artifact) = generative_artifact(&artifacts) else {
        eprintln!("SKIPPED: no text-generative container found on this machine");
        return;
    };

    let loaded = loader
        .load(&registry, &artifact.id, worker_id(), deadlines())
        .await
        .expect("loads");

    // Rejected rather than clamped: a caller who sends a negative temperature gets
    // an error naming their mistake, not output produced from a number they never
    // chose.
    let request =
        GenerateTextRequest::continuation("hello").with_parameters(GenerationParameters {
            temperature: Some(-1.0),
            ..GenerationParameters::default()
        });
    let err = loaded
        .generate_text(&request)
        .await
        .expect_err("a negative temperature must be refused");
    assert!(
        matches!(err, LoadError::InvalidRequest { .. }),
        "expected InvalidRequest, got {err}"
    );

    // The instance is unharmed and still serves a valid request.
    loaded
        .generate_text(&generation_request())
        .await
        .expect("a valid request still works after a refused one");

    loaded.unload().await.expect("unloads");
}

#[tokio::test]
async fn a_reloaded_instance_has_not_generated_anything() {
    let _exclusive = exclusive().await;
    let Some(loader) = loader() else { return };
    let Some((registry, artifacts)) = survey() else {
        return;
    };
    let Some(artifact) = generative_artifact(&artifacts) else {
        eprintln!("SKIPPED: no text-generative container found on this machine");
        return;
    };

    let mut first = loader
        .load(&registry, &artifact.id, worker_id(), deadlines())
        .await
        .expect("loads");
    first
        .verify_text_generation(&generation_request())
        .await
        .expect("generates");
    assert!(
        first
            .instance()
            .capabilities()
            .is_verified(ModelCapability::TextGeneration)
    );
    first.unload().await.expect("unloads");

    let second = loader
        .load(&registry, &artifact.id, worker_id(), deadlines())
        .await
        .expect("loads again");
    assert!(
        !second
            .instance()
            .capabilities()
            .is_verified(ModelCapability::TextGeneration),
        "a new instance inherited a generation it never performed"
    );
    second.unload().await.expect("unloads");
}

// ----------------------------------------------------------- streaming generation

/// Drains a stream, asserting the sequence rather than merely the contents.
///
/// The shape is the contract: exactly one `Started` first, then deltas, then
/// exactly one `Completed`. Checking only that each kind appeared would pass on a
/// stream that emitted them in any order.
async fn drain_asserting_shape(
    stream: &mut mehoy_core::inference::GenerationStream,
) -> (String, mehoy_core::inference::GenerationSummary) {
    use mehoy_core::inference::GenerationEvent;

    let request_id = stream.request_id();
    let mut text = String::new();
    let mut summary = None;
    let mut position = 0usize;

    while let Some(item) = stream.next().await {
        let event = item.expect("the stream must not fail");
        assert_eq!(
            event.request_id(),
            request_id,
            "event {position} belongs to another request"
        );
        match event {
            GenerationEvent::Started { .. } => {
                assert_eq!(position, 0, "Started appeared at position {position}");
            }
            GenerationEvent::TextDelta { text: delta, .. } => {
                assert_ne!(position, 0, "a delta preceded Started");
                assert!(summary.is_none(), "a delta followed the completion");
                assert!(!delta.is_empty(), "an empty delta reached the consumer");
                text.push_str(&delta);
            }
            GenerationEvent::Completed {
                summary: reported, ..
            } => {
                assert!(summary.is_none(), "Completed appeared twice");
                summary = Some(reported);
            }
            GenerationEvent::Cancelled { cause, .. } => {
                panic!("nothing cancelled this request, yet it reported {cause}")
            }
        }
        position += 1;
    }

    (text, summary.expect("the stream must complete"))
}

#[tokio::test]
async fn a_generative_model_actually_streams() {
    use mehoy_core::inference::FinishReason;

    let _exclusive = exclusive().await;
    let Some(loader) = loader() else { return };
    let Some((registry, artifacts)) = survey() else {
        return;
    };
    let Some(artifact) = generative_artifact(&artifacts) else {
        eprintln!("SKIPPED: no text-generative container found on this machine");
        return;
    };

    let loaded = loader
        .load(&registry, &artifact.id, worker_id(), deadlines())
        .await
        .expect("loads");

    let mut stream = loaded
        .generate_stream(&generation_request())
        .expect("the request is accepted")
        .stream;

    let (text, summary) = drain_asserting_shape(&mut stream).await;

    assert!(!text.is_empty(), "the stream produced no content");
    assert!(
        matches!(
            summary.finish_reason,
            FinishReason::Stop | FinishReason::Length
        ),
        "unrecognised finish reason {}",
        summary.finish_reason
    );
    if let Some(usage) = summary.usage {
        assert!(
            usage.output_tokens > 0,
            "content streamed but no output tokens"
        );
        assert!(
            usage.output_tokens <= 8,
            "produced {} tokens against a budget of 8",
            usage.output_tokens
        );
    }
    assert!(stream.next().await.is_none(), "an event followed Completed");

    eprintln!(
        "streamed {text:?} (finish: {}, usage: {:?})",
        summary.finish_reason, summary.usage
    );

    loaded.unload().await.expect("unloads");
}

#[tokio::test]
async fn a_stream_is_evidence_only_once_it_completes() {
    use mehoy_core::inference::GenerationEvent;

    let _exclusive = exclusive().await;
    let Some(loader) = loader() else { return };
    let Some((registry, artifacts)) = survey() else {
        return;
    };
    let Some(artifact) = generative_artifact(&artifacts) else {
        eprintln!("SKIPPED: no text-generative container found on this machine");
        return;
    };

    let mut loaded = loader
        .load(&registry, &artifact.id, worker_id(), deadlines())
        .await
        .expect("loads");

    assert!(
        !loaded
            .instance()
            .capabilities()
            .is_verified(ModelCapability::TextGeneration),
        "loading must not verify generation"
    );

    let mut stream = loaded
        .generate_stream(&generation_request())
        .expect("the request is accepted")
        .stream;

    // An open stream proves the backend accepted a request, nothing more.
    assert!(!stream.demonstrated_generation());

    let mut deltas = 0usize;
    while let Some(item) = stream.next().await {
        let event = item.expect("the stream must not fail");
        if let GenerationEvent::TextDelta { .. } = event {
            deltas += 1;
            // The backend could still die before completing, so content on its own
            // is not yet a demonstrated capability.
            assert!(
                !stream.demonstrated_generation(),
                "a delta alone verified the capability"
            );
        }
    }

    assert!(deltas > 0, "the model streamed no content");
    assert!(stream.demonstrated_generation());

    assert!(
        !loaded
            .instance()
            .capabilities()
            .is_verified(ModelCapability::TextGeneration),
        "streaming must not verify anything until the outcome is recorded"
    );

    assert!(loaded.record_generation_stream(&stream));
    assert!(
        loaded
            .instance()
            .capabilities()
            .is_verified(ModelCapability::TextGeneration),
        "a completed stream should have verified the capability"
    );
    eprintln!(
        "text generation verified by streaming: {}",
        loaded
            .instance()
            .capabilities()
            .state(ModelCapability::TextGeneration)
    );

    // Nothing was verified by association.
    for other in [
        ModelCapability::Embeddings,
        ModelCapability::Vision,
        ModelCapability::ToolCalling,
        ModelCapability::StructuredOutput,
    ] {
        assert!(
            !loaded.instance().capabilities().is_verified(other),
            "{other} was verified without being demonstrated"
        );
    }

    loaded.unload().await.expect("unloads");
}

#[tokio::test]
async fn both_delivery_modes_serve_the_same_request() {
    use mehoy_core::inference::FinishReason;

    let _exclusive = exclusive().await;
    let Some(loader) = loader() else { return };
    let Some((registry, artifacts)) = survey() else {
        return;
    };
    let Some(artifact) = generative_artifact(&artifacts) else {
        eprintln!("SKIPPED: no text-generative container found on this machine");
        return;
    };

    let loaded = loader
        .load(&registry, &artifact.id, worker_id(), deadlines())
        .await
        .expect("loads");

    // The same request type, the same parameters, one instance. Streaming is a
    // delivery choice rather than a different kind of work.
    let whole = loaded
        .generate_text(&generation_request())
        .await
        .expect("the model generates");

    let mut stream = loaded
        .generate_stream(&generation_request())
        .expect("the request is accepted")
        .stream;
    let streamed = stream
        .collect()
        .await
        .expect("the stream completes")
        .completed()
        .cloned()
        .expect("a completion, since nothing cancelled this request");

    for (label, result) in [("whole", &whole), ("streamed", &streamed)] {
        assert!(!result.text.is_empty(), "{label} produced no content");
        assert!(
            matches!(
                result.finish_reason,
                FinishReason::Stop | FinishReason::Length
            ),
            "{label} reported an unrecognised finish reason {}",
            result.finish_reason
        );
    }

    // Sampling is not asserted to be reproducible across requests, which is
    // ADR-0008's position on seeds. What is asserted is that both paths carry real
    // content and a normalised outcome.
    eprintln!(
        "whole {:?} (finish {}) / streamed {:?} (finish {})",
        whole.text, whole.finish_reason, streamed.text, streamed.finish_reason
    );

    loaded.unload().await.expect("unloads");
}

#[tokio::test]
async fn a_stream_is_refused_before_reaching_a_backend_when_a_parameter_is_unusable() {
    use mehoy_core::inference::{GenerateTextRequest, GenerationParameters};

    let _exclusive = exclusive().await;
    let Some(loader) = loader() else { return };
    let Some((registry, artifacts)) = survey() else {
        return;
    };
    let Some(artifact) = generative_artifact(&artifacts) else {
        eprintln!("SKIPPED: no text-generative container found on this machine");
        return;
    };

    let loaded = loader
        .load(&registry, &artifact.id, worker_id(), deadlines())
        .await
        .expect("loads");

    // Streaming must not become a way around validation.
    let request = GenerateTextRequest::continuation("x").with_parameters(GenerationParameters {
        temperature: Some(-1.0),
        ..GenerationParameters::default()
    });

    match loaded.generate_stream(&request) {
        Err(LoadError::InvalidRequest { detail }) => {
            assert!(
                detail.contains("temperature"),
                "unexpected detail: {detail}"
            );
        }
        Err(other) => panic!("expected a refused request, got {other}"),
        Ok(_) => panic!("an unusable temperature opened a stream"),
    }

    loaded.unload().await.expect("unloads");
}
