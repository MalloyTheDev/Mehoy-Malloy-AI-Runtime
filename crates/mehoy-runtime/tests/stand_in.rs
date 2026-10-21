//! The runtime's own behaviour, proven without an engine or a container.
//!
//! Everything here used to need a real `llama-server` and a multi-gigabyte GGUF
//! on disk, so admission, the resource ledger, instance ownership and unload
//! draining were exercised on one machine and skipped everywhere else. On the
//! Linux run they skipped too, which meant what Linux proved about this runtime
//! was that it compiled.
//!
//! These run on any machine. The backend is a stand-in executable built by
//! `cargo test`, and the container is twelve metadata entries written to a
//! temporary file.
//!
//! # The line these tests do not cross
//!
//! No test here asserts a capability. The stand-in returns responses with the
//! right shape and no meaning, and ADR-0005 and the capability ladder both insist
//! that only performing a task demonstrates one. A stand-in that let a capability
//! reach `Verified` would be a lie in exactly the place this project is most
//! careful, so verification stays with the real-backend gate.

use std::sync::Arc;
use std::time::Duration;

use mehoy_core::inference::{EmbedRequest, GenerateTextRequest, GenerationParameters};
use mehoy_runtime::{
    DEFAULT_RESIDENT_CAPACITY, FootprintBasis, InstanceId, LifecyclePhase, LoadError,
    ModelCapability, ResidentLimits, ResidentUsage, RuntimeError,
};

mod common;
use common::{
    fixture, stand_in_deadlines, stand_in_runtime, stand_in_runtime_with_limits, worker_id,
};

fn short_request() -> GenerateTextRequest {
    GenerateTextRequest::continuation("anything at all").with_parameters(GenerationParameters {
        max_output_tokens: Some(8),
        temperature: Some(0.0),
        seed: Some(1),
        stop: Vec::new(),
    })
}

#[tokio::test]
async fn a_model_loads_serves_and_unloads_with_nothing_installed() {
    // The whole loop, on a machine with no engine and no model. This is the test
    // whose absence meant the Linux gate proved only that the crate compiled.
    let runtime = stand_in_runtime();
    let held = fixture(None);

    assert_eq!(runtime.usage(), ResidentUsage::default());

    let id = runtime
        .load(
            &held.registry,
            &held.artifact,
            worker_id(),
            stand_in_deadlines(),
        )
        .await
        .expect("the stand-in comes up and authenticates");

    let snapshot = runtime.instance(&id).expect("resident");
    assert_eq!(snapshot.artifact_id(), &held.artifact);
    assert_eq!(runtime.phase(&id), Some(LifecyclePhase::Serving));

    let footprint = runtime.footprint(&id).expect("accounted");
    assert_eq!(footprint.weight_bytes, held.size_bytes);
    assert_eq!(footprint.basis, FootprintBasis::ContainerSize);

    // Both task classes reach the backend and come back with usable shapes.
    let embedded = runtime
        .embed(&id, &EmbedRequest::single("anything"))
        .await
        .expect("the stand-in answers an embedding");
    assert_eq!(embedded.embeddings.len(), 1);
    assert!(!embedded.embeddings[0].vector.is_empty());

    let generated = runtime
        .generate_text(&id, &short_request())
        .await
        .expect("the stand-in answers a completion");
    assert!(!generated.text.is_empty());

    // The line this file does not cross, asserted rather than trusted.
    assert!(
        !snapshot
            .capabilities()
            .is_verified(ModelCapability::TextGeneration),
        "a stand-in response promoted a capability"
    );

    runtime.unload(&id).await.expect("unloads");
    assert_eq!(runtime.usage(), ResidentUsage::default());
    assert!(runtime.instance(&id).is_none());
}

#[tokio::test]
async fn a_second_resident_model_is_refused_rather_than_loaded() {
    // The deterministic counterpart of the capacity test in the real-backend
    // gate. Two synthetic containers cost nothing, so the limit is exercised
    // everywhere rather than only where two engines could be afforded.
    let runtime = stand_in_runtime();
    let first = fixture(None);
    let second = fixture(None);

    let id = runtime
        .load(
            &first.registry,
            &first.artifact,
            worker_id(),
            stand_in_deadlines(),
        )
        .await
        .expect("loads");

    match runtime
        .load(
            &second.registry,
            &second.artifact,
            worker_id(),
            stand_in_deadlines(),
        )
        .await
    {
        Err(RuntimeError::ResidentCapacityReached { capacity }) => {
            assert_eq!(capacity, DEFAULT_RESIDENT_CAPACITY);
        }
        Err(other) => panic!("expected a capacity refusal, got {other}"),
        Ok(_) => panic!("a second model was loaded while one was already resident"),
    }

    assert_eq!(runtime.resident_instances().len(), 1);
    runtime.unload(&id).await.expect("unloads");
}

#[tokio::test]
async fn two_models_can_be_resident_when_the_limit_allows_it() {
    // Nothing above the limit had ever actually held two models, because holding
    // two real ones means two multi-gigabyte loads. The ledger's arithmetic for
    // more than one was therefore only unit-tested against invented figures.
    let runtime = stand_in_runtime_with_limits(ResidentLimits::instances(2));
    let first = fixture(None);
    let second = fixture(None);

    let one = runtime
        .load(
            &first.registry,
            &first.artifact,
            worker_id(),
            stand_in_deadlines(),
        )
        .await
        .expect("the first loads");
    let two = runtime
        .load(
            &second.registry,
            &second.artifact,
            worker_id(),
            stand_in_deadlines(),
        )
        .await
        .expect("the second loads");

    assert_ne!(one, two);
    assert_eq!(runtime.resident_instances().len(), 2);
    let usage = runtime.usage();
    assert_eq!(usage.instances, 2);
    assert_eq!(
        usage.weight_bytes,
        first.size_bytes + second.size_bytes,
        "two residents were not accounted as the sum of what they hold"
    );

    // Each has its own backend, so one is unaffected by the other going.
    assert_ne!(
        runtime.backend_address(&one),
        runtime.backend_address(&two),
        "two instances shared a backend address"
    );
    runtime.unload(&one).await.expect("the first unloads");
    runtime
        .generate_text(&two, &short_request())
        .await
        .expect("the survivor still serves");

    runtime.unload(&two).await.expect("the second unloads");
    assert_eq!(runtime.usage(), ResidentUsage::default());
}

#[tokio::test]
async fn a_stale_identifier_can_never_address_the_reloaded_model() {
    let runtime = stand_in_runtime();
    let held = fixture(None);

    let first = runtime
        .load(
            &held.registry,
            &held.artifact,
            worker_id(),
            stand_in_deadlines(),
        )
        .await
        .expect("loads");
    runtime.unload(&first).await.expect("unloads");

    let second = runtime
        .load(
            &held.registry,
            &held.artifact,
            worker_id(),
            stand_in_deadlines(),
        )
        .await
        .expect("reloads");
    assert_ne!(second, first);

    assert!(matches!(
        runtime.generate_text(&first, &short_request()).await,
        Err(RuntimeError::UnknownInstance { .. })
    ));
    runtime
        .generate_text(&second, &short_request())
        .await
        .expect("the live instance serves");

    runtime.unload(&second).await.expect("unloads");
}

#[tokio::test]
async fn a_backend_that_refuses_the_credential_is_not_reported_as_ready() {
    // Readiness means the backend accepts authenticated requests, not that it is
    // listening. A backend that answers its health route and rejects the
    // credential must fail to start rather than become a resident model that
    // refuses everything.
    let runtime = stand_in_runtime();
    let held = fixture(Some("refuse-credential"));

    let outcome = runtime
        .load(
            &held.registry,
            &held.artifact,
            worker_id(),
            stand_in_deadlines(),
        )
        .await;

    match outcome {
        Err(RuntimeError::Instance(LoadError::Backend(_))) => {}
        Err(other) => panic!("expected the backend to be refused, got {other}"),
        Ok(_) => panic!("a backend that rejects the credential was reported as ready"),
    }
    assert!(runtime.resident_instances().is_empty());
    assert_eq!(
        runtime.usage(),
        ResidentUsage::default(),
        "a failed load kept its claim on the limit"
    );
}

#[tokio::test]
async fn a_backend_that_never_binds_hits_its_startup_deadline() {
    let runtime = stand_in_runtime();
    let held = fixture(Some("never-ready"));

    let began = std::time::Instant::now();
    let outcome = runtime
        .load(
            &held.registry,
            &held.artifact,
            worker_id(),
            mehoy_core::worker::Deadlines {
                startup: Duration::from_millis(600),
                ..stand_in_deadlines()
            },
        )
        .await;

    assert!(
        matches!(outcome, Err(RuntimeError::Instance(LoadError::Backend(_)))),
        "expected a startup failure"
    );
    assert!(
        began.elapsed() < Duration::from_secs(10),
        "the startup deadline was not what ended the load, it took {:?}",
        began.elapsed()
    );
    assert_eq!(runtime.usage(), ResidentUsage::default());
}

#[tokio::test]
async fn a_container_that_vanished_is_refused_before_a_backend_starts() {
    // Drift detection, which the real-backend gate covers by deleting a
    // multi-gigabyte file. Here the file costs nothing.
    let runtime = stand_in_runtime();
    let held = fixture(None);
    let id = runtime
        .load(
            &held.registry,
            &held.artifact,
            worker_id(),
            stand_in_deadlines(),
        )
        .await
        .expect("loads once");
    runtime.unload(&id).await.expect("unloads");

    let record = held
        .registry
        .get(&held.artifact)
        .expect("readable")
        .expect("registered");
    std::fs::remove_file(&record.path).expect("the fixture is removable");

    match runtime
        .load(
            &held.registry,
            &held.artifact,
            worker_id(),
            stand_in_deadlines(),
        )
        .await
    {
        Err(RuntimeError::Instance(LoadError::ArtifactMissing { .. })) => {}
        Err(other) => panic!("expected a missing artifact, got {other}"),
        Ok(_) => panic!("a container that is gone was loaded"),
    }
    assert_eq!(
        runtime.usage(),
        ResidentUsage::default(),
        "a refused load kept its claim"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_request_racing_a_teardown_is_served_or_refused_and_never_stranded() {
    // The same claim the real-backend gate makes, at a cost that lets it run
    // everywhere and with far more racers than two engines would allow.
    let runtime = Arc::new(stand_in_runtime());
    let held = fixture(None);
    let id = runtime
        .load(
            &held.registry,
            &held.artifact,
            worker_id(),
            stand_in_deadlines(),
        )
        .await
        .expect("loads");

    // Bounded by time rather than by a request count. The stand-in answers in
    // microseconds, so a fixed number of requests all finish before the teardown
    // starts and the race the test is named after never happens.
    let until = tokio::time::Instant::now() + Duration::from_millis(400);
    let mut racers = Vec::new();
    for _ in 0..8 {
        let shared = Arc::clone(&runtime);
        let instance: InstanceId = id.clone();
        racers.push(tokio::spawn(async move {
            let mut refused = 0usize;
            while tokio::time::Instant::now() < until {
                match shared.generate_text(&instance, &short_request()).await {
                    Ok(_) => {}
                    Err(RuntimeError::UnknownInstance { .. })
                    | Err(RuntimeError::Instance(LoadError::NotUsable { .. }))
                    | Err(RuntimeError::Instance(LoadError::Stopped { .. })) => refused += 1,
                    Err(RuntimeError::Instance(LoadError::Inference { detail })) => {
                        panic!("a request reached a backend being torn down: {detail}")
                    }
                    Err(other) => panic!("unexpected failure: {other}"),
                }
            }
            refused
        }));
    }

    // Partway through, so requests are still arriving after admission closes.
    tokio::time::sleep(Duration::from_millis(100)).await;
    runtime.unload(&id).await.expect("unloads");

    let mut refused = 0usize;
    for racer in racers {
        refused += racer.await.expect("the racer did not panic");
    }
    assert!(
        refused > 0,
        "nothing was refused, so the race never happened"
    );
    assert!(runtime.resident_instances().is_empty());
    assert_eq!(runtime.usage(), ResidentUsage::default());
}

#[tokio::test]
async fn a_backend_that_abandons_its_instance_stops_answering() {
    // The stand-in mode that #18 needs, exercised here so that whoever writes
    // that test is debugging the runtime rather than this helper.
    //
    // It asserts only what is true today: the backend's address stops answering.
    // What the runtime ought to conclude from that is the subject of #18, and
    // pinning the current behaviour would be pinning the defect.
    let runtime = stand_in_runtime();
    let held = fixture(Some("exit-after-serving-50"));

    let id = runtime
        .load(
            &held.registry,
            &held.artifact,
            worker_id(),
            stand_in_deadlines(),
        )
        .await
        .expect("the stand-in loads before it goes");
    assert_eq!(runtime.phase(&id), Some(LifecyclePhase::Serving));

    // The request is what triggers the backend to abandon the instance, so the
    // sequence is deterministic: loaded, used, then gone.
    runtime
        .generate_text(&id, &short_request())
        .await
        .expect("the request that triggers the exit is answered first");

    let mut gone = false;
    for _ in 0..150 {
        if runtime.backend_reachable(&id).await == Some(false) {
            gone = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        gone,
        "the stand-in was asked to abandon its instance and kept answering"
    );

    // Teardown still works. Stopping a process that has already exited is not a
    // failure, which is what the already-gone check in the supervisor is for.
    runtime.unload(&id).await.expect("unloads");
    assert_eq!(runtime.usage(), ResidentUsage::default());
}
