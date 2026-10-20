//! What the runtime will hold, and what it accounts a model as occupying.
//!
//! The policy itself needs no backend and is unit-tested in the library, where it
//! runs on every commit. What needs a real load is everything that joins the
//! policy to one: that room is claimed before the expensive work rather than
//! merely checked, that the claim comes back however the load ends, and that the
//! figure a model was admitted under is the figure it keeps until it is unloaded.

use std::sync::Arc;
use std::time::Duration;

use mehoy_registry::{ArtifactId, ArtifactRegistry};
use mehoy_runtime::{
    DEFAULT_RESIDENT_CAPACITY, FootprintBasis, LoadError, ResidentLimits, ResidentUsage,
    RuntimeError,
};

mod common;
use common::{
    deadlines, exclusive, generative_artifact, runtime, runtime_with_limits, survey, worker_id,
};

#[tokio::test]
async fn a_resident_model_is_accounted_at_its_container_size() {
    let _exclusive = exclusive().await;
    let Some(runtime) = runtime() else { return };
    let Some((registry, artifacts)) = survey() else {
        return;
    };
    let Some(artifact) = generative_artifact(&artifacts) else {
        eprintln!("SKIPPED: no text-generative container found on this machine");
        return;
    };

    assert_eq!(
        runtime.usage(),
        ResidentUsage::default(),
        "a fresh runtime already had something committed"
    );

    let id = runtime
        .load(&registry, &artifact.id, worker_id(), deadlines())
        .await
        .expect("loads");

    let footprint = runtime
        .footprint(&id)
        .expect("a resident model is accounted");
    assert_eq!(footprint.weight_bytes, artifact.size_bytes);
    assert_eq!(
        footprint.basis,
        FootprintBasis::ContainerSize,
        "the figure claimed a basis it does not have"
    );

    let usage = runtime.usage();
    assert_eq!(usage.instances, 1);
    assert_eq!(
        usage.loading, 0,
        "a finished load was still counted as loading"
    );
    assert_eq!(usage.weight_bytes, artifact.size_bytes);

    runtime.unload(&id).await.expect("unloads");

    assert_eq!(
        runtime.usage(),
        ResidentUsage::default(),
        "unloading did not release what the model was accounted for"
    );
    assert!(
        runtime.footprint(&id).is_none(),
        "a model that is gone is still accounted for"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_load_in_flight_is_already_accounted_for() {
    // The reason room is claimed rather than checked. A load holds its claim for
    // the whole of the expensive part, so what the next caller is judged against
    // includes work that has started and not finished.
    let _exclusive = exclusive().await;
    let Some(runtime) = runtime() else { return };
    let Some((registry, artifacts)) = survey() else {
        return;
    };
    let Some(artifact) = generative_artifact(&artifacts) else {
        eprintln!("SKIPPED: no text-generative container found on this machine");
        return;
    };
    let runtime = Arc::new(runtime);

    let observer = {
        let shared = Arc::clone(&runtime);
        tokio::spawn(async move {
            for _ in 0..4000 {
                let usage = shared.usage();
                if usage.loading == 1 {
                    assert_eq!(
                        usage.instances, 1,
                        "a load in flight was not counted as held"
                    );
                    assert!(
                        usage.weight_bytes > 0,
                        "a load in flight was counted but not sized"
                    );
                    return true;
                }
                if !shared.resident_instances().is_empty() {
                    // The load finished without ever being seen in flight.
                    return false;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            false
        })
    };

    let id = runtime
        .load(&registry, &artifact.id, worker_id(), deadlines())
        .await
        .expect("loads");

    assert!(
        observer.await.expect("the observer did not panic"),
        "a load in flight was never visible as committed"
    );
    assert_eq!(runtime.usage().loading, 0);

    runtime.unload(&id).await.expect("unloads");
}

#[tokio::test]
async fn two_concurrent_loads_cannot_both_be_admitted() {
    // Checking a limit and then spending a minute on a load would let both of
    // these pass the same check, and both would finish: the limit would hold only
    // for callers who happened to arrive one at a time.
    //
    // Joined rather than spawned. A registry holds a SQLite connection, which is
    // not shareable, so a load future cannot be sent to another task at all; what
    // it can do is interleave with another load at each of its await points,
    // which is the whole of the race.
    let _exclusive = exclusive().await;
    let Some(runtime) = runtime() else { return };
    let Some((registry, artifacts)) = survey() else {
        return;
    };
    let Some(artifact) = generative_artifact(&artifacts) else {
        eprintln!("SKIPPED: no text-generative container found on this machine");
        return;
    };

    // A second handle on the same container, because each load needs its own
    // registry. What is being raced is the runtime's capacity, not identity.
    let mut second = ArtifactRegistry::open_in_memory().expect("registry opens");
    let second_id = second
        .register(&artifact.path)
        .expect("the same container registers again")
        .artifact()
        .id
        .clone();

    let (first, other) = tokio::join!(
        runtime.load(&registry, &artifact.id, worker_id(), deadlines()),
        runtime.load(&second, &second_id, worker_id(), deadlines()),
    );

    let mut loaded = Vec::new();
    let mut refused = Vec::new();
    for outcome in [first, other] {
        match outcome {
            Ok(id) => loaded.push(id),
            Err(err) => refused.push(err),
        }
    }

    assert_eq!(
        loaded.len(),
        1,
        "{} concurrent loads were admitted where one was allowed",
        loaded.len()
    );
    assert_eq!(refused.len(), 1);
    match &refused[0] {
        RuntimeError::ResidentCapacityReached { capacity } => {
            assert_eq!(*capacity, DEFAULT_RESIDENT_CAPACITY);
        }
        other => panic!("expected a capacity refusal, got {other}"),
    }

    assert_eq!(runtime.resident_instances().len(), 1);
    let usage = runtime.usage();
    assert_eq!(usage.instances, 1);
    assert_eq!(usage.loading, 0, "a reservation outlived its load");

    runtime.unload(&loaded[0]).await.expect("unloads");
    assert_eq!(runtime.usage(), ResidentUsage::default());
}

#[tokio::test]
async fn a_model_too_large_for_the_budget_is_refused_by_its_declared_size() {
    let _exclusive = exclusive().await;
    let Some((registry, artifacts)) = survey() else {
        return;
    };
    let Some(artifact) = generative_artifact(&artifacts) else {
        eprintln!("SKIPPED: no text-generative container found on this machine");
        return;
    };

    // One byte short of what the container declares, so the refusal cannot be
    // explained by anything other than the accounting.
    let budget = artifact.size_bytes.saturating_sub(1);
    let Some(runtime) =
        runtime_with_limits(ResidentLimits::instances(4).with_weight_budget(budget))
    else {
        return;
    };

    match runtime
        .load(&registry, &artifact.id, worker_id(), deadlines())
        .await
    {
        Err(RuntimeError::ResourcesExhausted {
            requested,
            committed,
            limit,
        }) => {
            assert_eq!(
                requested, artifact.size_bytes,
                "the refusal was not sized from the artifact's declared footprint"
            );
            assert_eq!(committed, 0);
            assert_eq!(limit, budget);
        }
        Err(other) => panic!("expected a budget refusal, got {other}"),
        Ok(_) => panic!("a model larger than the whole budget was loaded"),
    }

    assert_eq!(
        runtime.usage(),
        ResidentUsage::default(),
        "a refused load left something committed"
    );
    assert!(runtime.resident_instances().is_empty());
}

#[tokio::test]
async fn a_failed_load_gives_back_what_it_claimed() {
    // The failure is injected after the backend is healthy, which is the latest
    // point a load can fail and therefore the one where a claim is most likely to
    // be stranded. A reservation nothing releases would make this runtime refuse
    // every later load for the rest of the process.
    let _exclusive = exclusive().await;
    let Some(runtime) = runtime() else { return };
    let Some((registry, artifacts)) = survey() else {
        return;
    };
    let Some(artifact) = generative_artifact(&artifacts) else {
        eprintln!("SKIPPED: no text-generative container found on this machine");
        return;
    };

    mehoy_runtime::fail_next_commit();
    let err = runtime
        .load(&registry, &artifact.id, worker_id(), deadlines())
        .await
        .expect_err("the injected failure must surface");
    mehoy_runtime::clear_injected_faults();

    assert!(
        matches!(err, RuntimeError::Instance(LoadError::Instance { .. })),
        "{err}"
    );
    assert_eq!(
        runtime.usage(),
        ResidentUsage::default(),
        "a failed load kept its claim"
    );
    assert!(runtime.resident_instances().is_empty());

    // The claim was genuinely released rather than merely reported as released.
    let id = runtime
        .load(&registry, &artifact.id, worker_id(), deadlines())
        .await
        .expect("a released claim can be taken again");
    runtime.unload(&id).await.expect("unloads");
}

#[tokio::test]
async fn an_unregistered_artifact_is_refused_without_touching_the_accounting() {
    // Sizing a load happens before room is claimed, so an artifact this runtime
    // has never heard of is turned away without disturbing what is held.
    let _exclusive = exclusive().await;
    let Some(runtime) = runtime() else { return };
    let registry = ArtifactRegistry::open_in_memory().expect("registry opens");
    let absent = ArtifactId::generate().expect("an identifier");

    match runtime
        .load(&registry, &absent, worker_id(), deadlines())
        .await
    {
        Err(RuntimeError::Instance(LoadError::UnknownArtifact { id })) => assert_eq!(id, absent),
        Err(other) => panic!("expected an unknown artifact, got {other}"),
        Ok(_) => panic!("an unregistered artifact was loaded"),
    }

    assert_eq!(runtime.usage(), ResidentUsage::default());
}
