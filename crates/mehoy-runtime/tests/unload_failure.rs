//! A teardown that failed, and what the runtime may conclude from it.
//!
//! ADR-0010 decision 6 says an unload that fails keeps both its ledger entry and
//! its accounted footprint, because a worker that could not be stopped may still
//! be holding its memory. That was an argument from construction until a refused
//! kill could be injected, and the argument turned out to be only half true: the
//! first failure kept everything, and a retry quietly released it.
//!
//! One thing to understand before reading the assertions. On this platform the
//! worker does in fact die, because the supervisor's handle owns the job object
//! and closing it reaps the process. That is not the point. The runtime has no
//! evidence of it, and these tests pin what the runtime is entitled to conclude
//! from what it knows, which is nothing.
//!
//! Separate target, because the injected refusal is a process-wide flag and a
//! separate test binary is a separate process.

use mehoy_registry::{ArtifactId, ArtifactRegistry};
use mehoy_runtime::{
    InstanceId, LifecyclePhase, LoadError, ResidentUsage, Runtime, RuntimeError, UnloadOutcome,
};

mod common;
use common::{deadlines, exclusive, generative_artifact, runtime, survey, worker_id};

/// A runtime holding one generative model, with what it was loaded from.
async fn resident() -> Option<(Runtime, ArtifactRegistry, ArtifactId, InstanceId)> {
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
    Some((runtime, registry, artifact.id.clone(), id))
}

/// Unloads with the kill refused, and reports what the runtime said.
async fn unload_with_a_refused_kill(runtime: &Runtime, id: &InstanceId) {
    mehoy_core::refuse_next_kill();
    let outcome = runtime.unload(id).await;
    // Cleared unconditionally. If the worker had already gone the refusal was
    // never consumed, and a flag left armed would reach the next test.
    mehoy_core::clear_injected_faults();

    match outcome {
        Err(RuntimeError::Instance(LoadError::Backend(err))) => assert!(
            err.to_string().contains("injected"),
            "the teardown failed for some other reason: {err}"
        ),
        Err(other) => panic!("expected the refusal to surface, got {other}"),
        Ok(outcome) => {
            panic!("a teardown that could not stop its worker was reported as {outcome}")
        }
    }
}

#[tokio::test]
async fn a_failed_unload_keeps_its_entry_and_its_accounting() {
    let _exclusive = exclusive().await;
    let Some((runtime, _registry, _artifact, id)) = resident().await else {
        return;
    };

    let footprint = runtime
        .footprint(&id)
        .expect("a resident model is accounted");
    let committed = runtime.usage();
    assert_eq!(committed.instances, 1);
    assert!(committed.weight_bytes > 0);

    unload_with_a_refused_kill(&runtime, &id).await;

    assert_eq!(
        runtime.phase(&id),
        Some(LifecyclePhase::UnloadFailed),
        "a teardown that failed did not say so"
    );
    assert!(
        runtime.instance(&id).is_some(),
        "an instance whose teardown failed stopped being addressable, so nothing can be told about it"
    );
    assert_eq!(
        runtime.footprint(&id),
        Some(footprint),
        "the footprint changed, or was released, for a worker nothing has seen stop"
    );
    assert_eq!(
        runtime.usage(),
        committed,
        "a failed teardown released what it was accounted for"
    );
    assert_eq!(runtime.resident_instances(), vec![id]);
}

#[tokio::test]
async fn a_retry_after_a_failed_unload_does_not_release_the_accounting() {
    // The regression. A retry used to find the worker already taken, conclude the
    // instance was unloaded, and hand its budget back.
    let _exclusive = exclusive().await;
    let Some((runtime, registry, artifact, id)) = resident().await else {
        return;
    };

    let footprint = runtime
        .footprint(&id)
        .expect("a resident model is accounted");
    let committed = runtime.usage();

    unload_with_a_refused_kill(&runtime, &id).await;

    let retried = runtime
        .unload(&id)
        .await
        .expect("a retry reports rather than failing again");
    assert_eq!(
        retried,
        UnloadOutcome::AlreadyFailed,
        "a retry claimed something a retry cannot establish"
    );

    assert_eq!(
        runtime.phase(&id),
        Some(LifecyclePhase::UnloadFailed),
        "a retry moved an instance out of the only phase that describes it"
    );
    assert_eq!(
        runtime.footprint(&id),
        Some(footprint),
        "a retry released the footprint"
    );
    assert_eq!(
        runtime.usage(),
        committed,
        "a retry released the budget for a worker nothing has seen stop"
    );
    assert!(runtime.instance(&id).is_some());

    // The budget is genuinely still held, not merely still reported. A load that
    // needs the room is refused.
    match runtime
        .load(&registry, &artifact, worker_id(), deadlines())
        .await
    {
        Err(RuntimeError::ResidentCapacityReached { .. }) => {}
        Err(other) => panic!("expected the retained budget to refuse a load, got {other}"),
        Ok(_) => panic!("a load took room that a failed teardown had not released"),
    }

    assert_ne!(
        runtime.usage(),
        ResidentUsage::default(),
        "nothing is committed, so the refusal above came from somewhere else"
    );
}

#[tokio::test]
async fn repeated_retries_do_not_accumulate_or_drift() {
    // Idempotence, stated. Whatever a retry reports, it has to keep reporting it:
    // a second answer that differs from the first would mean the phase is being
    // used as scratch space.
    let _exclusive = exclusive().await;
    let Some((runtime, _registry, _artifact, id)) = resident().await else {
        return;
    };
    let committed = runtime.usage();

    unload_with_a_refused_kill(&runtime, &id).await;

    for attempt in 0..3 {
        let outcome = runtime.unload(&id).await.expect("a retry does not fail");
        assert_eq!(
            outcome,
            UnloadOutcome::AlreadyFailed,
            "retry {attempt} reported differently from the one before it"
        );
        assert_eq!(
            runtime.usage(),
            committed,
            "retry {attempt} changed what is held"
        );
        assert_eq!(runtime.phase(&id), Some(LifecyclePhase::UnloadFailed));
    }
}
