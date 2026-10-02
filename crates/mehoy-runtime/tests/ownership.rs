//! The runtime owning its instances.
//!
//! The claim is that no live model exists outside runtime ownership. Most of it
//! is enforced by the types: the loader is private and nothing hands out a model,
//! so a caller cannot obtain one to use behind the runtime's back. What the types
//! cannot state is tested here: that an identifier is the only way in, that a
//! stale one never reaches a later model, and that a request racing a teardown
//! has exactly two legal outcomes.

use std::sync::Arc;
use std::time::Duration;

use mehoy_core::inference::{EmbedRequest, GenerateTextRequest, GenerationParameters};
use mehoy_registry::ArtifactRegistry;
use mehoy_runtime::{
    InstanceId, LifecyclePhase, LoadError, ModelCapability, RESIDENT_CAPACITY, Runtime,
    RuntimeError,
};

mod common;
use common::{deadlines, exclusive, generative_artifact, runtime, survey, worker_id};

fn short_request() -> GenerateTextRequest {
    GenerateTextRequest::continuation("The opposite of hot is").with_parameters(
        GenerationParameters {
            max_output_tokens: Some(8),
            temperature: Some(0.0),
            seed: Some(1),
            stop: Vec::new(),
        },
    )
}

fn long_request() -> GenerateTextRequest {
    GenerateTextRequest::continuation(
        "Write a very long numbered list of every English word you know, one per line.",
    )
    .with_parameters(GenerationParameters {
        max_output_tokens: Some(4096),
        temperature: Some(0.0),
        seed: Some(1),
        stop: Vec::new(),
    })
}

/// A runtime holding one generative model, with the registry that described it.
async fn resident() -> Option<(
    Runtime,
    ArtifactRegistry,
    mehoy_registry::ArtifactId,
    InstanceId,
)> {
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
    let artifact_id = artifact.id.clone();
    Some((runtime, registry, artifact_id, id))
}

#[tokio::test]
async fn loading_yields_an_identifier_and_the_runtime_keeps_the_model() {
    let _exclusive = exclusive().await;
    let Some((runtime, _registry, artifact, id)) = resident().await else {
        return;
    };

    assert_eq!(runtime.resident_instances(), vec![id.clone()]);
    let snapshot = runtime.instance(&id).expect("resident");
    assert_eq!(snapshot.id(), &id);
    assert_eq!(snapshot.artifact_id(), &artifact);
    assert_eq!(runtime.phase(&id), Some(LifecyclePhase::Serving));

    runtime.unload(&id).await.expect("unloads");
}

#[tokio::test]
async fn a_snapshot_does_not_change_as_the_instance_does() {
    // What a caller receives describes what was known when it was taken. If it
    // tracked the instance, it would be the instance, which is the thing the
    // runtime declines to hand out.
    let _exclusive = exclusive().await;
    let Some((runtime, _registry, _artifact, id)) = resident().await else {
        return;
    };

    let before = runtime.instance(&id).expect("resident");
    assert!(
        !before
            .capabilities()
            .is_verified(ModelCapability::TextGeneration)
    );

    runtime
        .verify_text_generation(&id, &short_request())
        .await
        .expect("generates");

    assert!(
        !before
            .capabilities()
            .is_verified(ModelCapability::TextGeneration),
        "an earlier snapshot changed underneath its holder"
    );
    assert!(
        runtime
            .instance(&id)
            .expect("resident")
            .capabilities()
            .is_verified(ModelCapability::TextGeneration),
        "a later snapshot did not see the verification"
    );

    runtime.unload(&id).await.expect("unloads");
}

#[tokio::test]
async fn every_task_refuses_an_unknown_instance_before_reaching_a_backend() {
    let _exclusive = exclusive().await;
    let Some(runtime) = runtime() else { return };
    let absent = InstanceId::generate().expect("an identifier");

    assert!(matches!(
        runtime.embed(&absent, &EmbedRequest::single("x")).await,
        Err(RuntimeError::UnknownInstance { .. })
    ));
    assert!(matches!(
        runtime.generate_text(&absent, &short_request()).await,
        Err(RuntimeError::UnknownInstance { .. })
    ));
    assert!(matches!(
        runtime.generate_stream(&absent, &short_request()),
        Err(RuntimeError::UnknownInstance { .. })
    ));
    assert!(matches!(
        runtime.cancel(&absent, mehoy_core::id::RequestId::from_raw(1)),
        Err(RuntimeError::UnknownInstance { .. })
    ));
    assert!(matches!(
        runtime.unload(&absent).await,
        Err(RuntimeError::UnknownInstance { .. })
    ));

    assert!(runtime.instance(&absent).is_none());
    assert_eq!(runtime.active_requests(&absent), None);
    assert_eq!(runtime.phase(&absent), None);
}

#[tokio::test]
async fn a_second_resident_model_is_refused_rather_than_loaded() {
    // Stated rather than implied. The registry is keyed by identifier and would
    // happily hold two, so without this the limit would be an accident of nothing
    // having been written yet, and the second multi-gigabyte load would simply
    // happen.
    let _exclusive = exclusive().await;
    let Some((runtime, registry, artifact, id)) = resident().await else {
        return;
    };

    match runtime
        .load(&registry, &artifact, worker_id(), deadlines())
        .await
    {
        Err(RuntimeError::ResidentCapacityReached { capacity }) => {
            assert_eq!(capacity, RESIDENT_CAPACITY);
        }
        Err(other) => panic!("expected a capacity refusal, got {other}"),
        Ok(_) => panic!("a second model was loaded while one was already resident"),
    }

    assert_eq!(
        runtime.resident_instances().len(),
        1,
        "the refused load still changed what is resident"
    );
    runtime.unload(&id).await.expect("unloads");
}

#[tokio::test]
async fn an_instance_remains_addressable_while_it_is_unloading() {
    // Removing the entry when teardown starts would make an instance being torn
    // down indistinguishable from one that never existed, so nothing could be
    // told that an unload was already in progress.
    let _exclusive = exclusive().await;
    let Some((runtime, _registry, _artifact, id)) = resident().await else {
        return;
    };
    let runtime = Arc::new(runtime);

    let started = runtime
        .generate_stream(&id, &long_request())
        .expect("the request is accepted");
    let _stream = started.stream;

    let shared = Arc::clone(&runtime);
    let instance = id.clone();
    let tearing_down = tokio::spawn(async move { shared.unload(&instance).await });

    // While that runs the instance must still resolve, and must say it is going.
    let mut saw_unloading = false;
    for _ in 0..200 {
        match runtime.phase(&id) {
            Some(LifecyclePhase::Unloading) => {
                saw_unloading = true;
                break;
            }
            Some(LifecyclePhase::Unloaded) | None => break,
            _ => tokio::time::sleep(Duration::from_millis(5)).await,
        }
    }

    tearing_down
        .await
        .expect("the task did not panic")
        .expect("unloads");

    assert!(
        runtime.instance(&id).is_none(),
        "a finished unload left its entry behind"
    );
    if !saw_unloading {
        eprintln!("teardown completed too quickly to observe the intermediate phase");
    }
}

#[tokio::test]
async fn a_stale_identifier_can_never_address_the_reloaded_model() {
    // Identifiers are not reused, so the question is not merely improbable but
    // settled. A caller holding one from before a reload addresses nothing.
    let _exclusive = exclusive().await;
    let Some((runtime, registry, artifact, first)) = resident().await else {
        return;
    };

    runtime.unload(&first).await.expect("unloads");

    let second = runtime
        .load(&registry, &artifact, worker_id(), deadlines())
        .await
        .expect("reloads");
    assert_ne!(second, first, "a reload reused an identifier");

    assert!(matches!(
        runtime.generate_text(&first, &short_request()).await,
        Err(RuntimeError::UnknownInstance { .. })
    ));
    runtime
        .generate_text(&second, &short_request())
        .await
        .expect("the live instance serves");

    // The artifact is untouched by any of this.
    assert!(
        registry
            .get(&artifact)
            .expect("the registry is readable")
            .is_some(),
        "unloading an instance deregistered its artifact"
    );

    runtime.unload(&second).await.expect("unloads");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_request_racing_a_teardown_is_served_or_refused_and_never_stranded() {
    // The registry lock resolves the identifier and is released; the instance's
    // own lock decides admission. A request can therefore be admitted and then
    // owned by the teardown, or refused because admission closed, or refused
    // because the instance is gone. What it must never do is reach a backend that
    // is being destroyed, which shows up as a transport error.
    let _exclusive = exclusive().await;
    let Some((runtime, _registry, _artifact, id)) = resident().await else {
        return;
    };
    let runtime = Arc::new(runtime);

    let mut racers = Vec::new();
    for _ in 0..8 {
        let shared = Arc::clone(&runtime);
        let instance = id.clone();
        racers.push(tokio::spawn(async move {
            let mut refused = 0usize;
            for _ in 0..6 {
                match shared.generate_text(&instance, &short_request()).await {
                    Ok(_) => {}
                    Err(RuntimeError::UnknownInstance { .. })
                    | Err(RuntimeError::Instance(LoadError::NotUsable { .. })) => refused += 1,
                    Err(RuntimeError::Instance(LoadError::Stopped { .. })) => refused += 1,
                    Err(RuntimeError::Instance(LoadError::Inference { detail })) => {
                        panic!("a request reached a backend being torn down: {detail}")
                    }
                    Err(other) => panic!("unexpected failure: {other}"),
                }
            }
            refused
        }));
    }

    tokio::time::sleep(Duration::from_millis(150)).await;
    runtime.unload(&id).await.expect("unloads");

    let mut refused = 0usize;
    for racer in racers {
        refused += racer.await.expect("the racer did not panic");
    }

    assert!(
        refused > 0,
        "nothing was refused, so the race never actually happened"
    );
    assert!(
        runtime.resident_instances().is_empty(),
        "the instance survived its own teardown"
    );
}
