//! Taking an instance down.
//!
//! The invariant these tests exist for is narrow and load-bearing: once an
//! instance begins unloading, no further request may become active on it. Every
//! other guarantee here follows from that one, and a runtime that got it wrong
//! would accept work onto a model it was in the middle of destroying.
//!
//! The second concern is ordering. Unloading closes admission, stops what is
//! already running, waits a bounded time for it to settle, and only then takes
//! the worker away. Killing the worker first would also stop the work, and would
//! do it by removing the thing every other request depends on.

use std::time::Duration;

use mehoy_core::cancel::{CancellationCause, RequestState, UnloadBudget};
use mehoy_core::inference::{
    GenerateTextRequest, GenerationEvent, GenerationParameters, GenerationStream,
};
use mehoy_runtime::{
    InstanceId, LifecyclePhase, LoadError, ModelCapability, Runtime, RuntimeError, UnloadOutcome,
};

mod common;
use common::{
    deadlines, exclusive, generative_artifact, loaded_generative, runtime, survey, worker_id,
};

/// Loads the generative container and keeps the runtime that owns it.
///
/// The runtime is returned alongside the identifier because it holds the model:
/// dropping it would take the instance down with it.
async fn loaded() -> Option<(Runtime, InstanceId)> {
    loaded_generative()
        .await
        .map(|(runtime, _registry, id)| (runtime, id))
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

/// Drains a stream and reports its terminal event.
async fn terminal(stream: &mut GenerationStream) -> Option<GenerationEvent> {
    let mut last = None;
    while let Some(item) = stream.next().await {
        if let Ok(event @ (GenerationEvent::Completed { .. } | GenerationEvent::Cancelled { .. })) =
            item
        {
            last = Some(event);
        }
    }
    last
}

#[tokio::test]
async fn unloading_an_idle_instance_tears_it_down() {
    let _exclusive = exclusive().await;
    let Some((runtime, id)) = loaded().await else {
        return;
    };

    assert_eq!(
        runtime.phase(&id).expect("resident"),
        LifecyclePhase::Serving
    );
    assert!(
        runtime.backend_reachable(&id).await.unwrap_or(false),
        "the backend should be up"
    );

    let address = runtime.backend_address(&id).expect("resident");

    assert_eq!(
        runtime.unload(&id).await.expect("unloads"),
        UnloadOutcome::Drained { cancelled: 0 }
    );
    assert!(
        runtime.instance(&id).is_none(),
        "a completed unload left the instance addressable"
    );
    assert!(
        tokio::net::TcpStream::connect(address).await.is_err(),
        "something is still listening after the worker was stopped"
    );
}

#[tokio::test]
async fn unloading_twice_is_reported_rather_than_repeated() {
    let _exclusive = exclusive().await;
    let Some((runtime, id)) = loaded().await else {
        return;
    };

    assert!(matches!(
        runtime.unload(&id).await.expect("unloads"),
        UnloadOutcome::Drained { .. }
    ));
    // The instance is forgotten once it is down, so a second attempt addresses
    // nothing rather than reporting on something that no longer exists.
    assert!(matches!(
        runtime.unload(&id).await,
        Err(RuntimeError::UnknownInstance { .. })
    ));
}

#[tokio::test]
async fn no_request_is_admitted_once_unloading_has_begun() {
    // The invariant. A request accepted after this point would be put onto a model
    // that is being destroyed.
    let _exclusive = exclusive().await;
    let Some((runtime, id)) = loaded().await else {
        return;
    };

    runtime.unload(&id).await.expect("unloads");

    match runtime.generate_stream(&id, &short_request()) {
        // Either answer is a refusal. Which one depends on whether teardown has
        // finished and forgotten the instance, and both keep the work away from a
        // backend that is going or gone.
        Err(RuntimeError::UnknownInstance { .. })
        | Err(RuntimeError::Instance(LoadError::NotUsable { .. })) => {}
        Err(other) => panic!("expected a refusal, got {other}"),
        Ok(_) => panic!("a request was admitted onto an unloaded instance"),
    }
    assert_eq!(runtime.active_requests(&id), None);
}

#[tokio::test]
async fn a_request_in_flight_is_stopped_by_unloading_and_says_why() {
    // Unloading does not end requests by a second mechanism. It uses the same one
    // everything else uses, and the cause records what actually happened.
    let _exclusive = exclusive().await;
    let Some((runtime, id)) = loaded().await else {
        return;
    };

    let started = runtime
        .generate_stream(&id, &long_request())
        .expect("the request is accepted");
    let request_id = started.request_id;
    let mut stream = started.stream;

    // Let it get going so this is a genuinely in-flight request.
    while let Some(Ok(event)) = stream.next().await {
        if matches!(event, GenerationEvent::TextDelta { .. }) {
            break;
        }
    }
    assert_eq!(runtime.active_requests(&id).unwrap_or(0), 1);

    let outcome = runtime.unload(&id).await.expect("unloads");
    assert!(
        matches!(outcome, UnloadOutcome::Drained { cancelled: 1 }),
        "expected one request to have been drained, got {outcome}"
    );

    match terminal(&mut stream).await {
        Some(GenerationEvent::Cancelled { cause, .. }) => {
            assert_eq!(
                cause,
                CancellationCause::InstanceUnloading,
                "unloading must not be reported as somebody cancelling"
            );
            assert_ne!(cause, CancellationCause::User);
        }
        other => panic!("expected a cancellation, got {other:?}"),
    }
    let _ = request_id;
    assert!(
        runtime.instance(&id).is_none(),
        "a completed unload left the instance addressable"
    );
}

#[tokio::test]
async fn unloading_a_request_that_is_already_stopping_does_not_corrupt_its_outcome() {
    // Both are asking for the same thing at once. The request must still reach
    // exactly one terminal state, and must keep the reason it was first given.
    let _exclusive = exclusive().await;
    let Some((runtime, id)) = loaded().await else {
        return;
    };

    let started = runtime
        .generate_stream(&id, &long_request())
        .expect("the request is accepted");
    let mut stream = started.stream;
    while let Some(Ok(event)) = stream.next().await {
        if matches!(event, GenerationEvent::TextDelta { .. }) {
            break;
        }
    }

    runtime.cancel(&id, started.request_id).expect("cancels");
    assert_eq!(
        runtime.request_state(&id, started.request_id),
        Some(RequestState::Cancelling)
    );

    runtime.unload(&id).await.expect("unloads");

    let mut terminals = 0usize;
    let mut cause = None;
    while let Some(item) = stream.next().await {
        if let Ok(GenerationEvent::Cancelled {
            cause: reported, ..
        }) = item
        {
            terminals += 1;
            cause = Some(reported);
        }
        if let Ok(GenerationEvent::Completed { .. }) = item {
            terminals += 1;
        }
    }
    assert_eq!(terminals, 1, "the request reported more than one outcome");
    assert_eq!(
        cause,
        Some(CancellationCause::User),
        "the first reason must survive: unloading arrived second"
    );
}

#[tokio::test]
async fn unloading_reports_when_it_stopped_waiting() {
    // A drain budget that expires is not a failure. It means the instance stopped
    // waiting and went ahead, which is honest only if it says so.
    let _exclusive = exclusive().await;
    let Some((runtime, id)) = loaded().await else {
        return;
    };

    let started = runtime
        .generate_stream(&id, &long_request())
        .expect("the request is accepted");
    // Deliberately not read, and given no time to settle.
    let _stream = started.stream;
    let address = runtime.backend_address(&id).expect("resident");

    let outcome = runtime
        .unload_within(
            &id,
            UnloadBudget {
                drain: Duration::ZERO,
            },
        )
        .await
        .expect("unloads");

    match outcome {
        UnloadOutcome::Escalated {
            cancelled,
            unsettled,
        } => {
            assert_eq!(cancelled, 1);
            assert_eq!(unsettled, 1);
        }
        // Settling instantly is legitimate rather than flaky.
        UnloadOutcome::Drained { cancelled } => assert_eq!(cancelled, 1),
        other => panic!("unexpected outcome {other}"),
    }

    assert!(runtime.instance(&id).is_none());
    assert!(
        tokio::net::TcpStream::connect(address).await.is_err(),
        "the worker outlived an escalated unload"
    );
}

#[tokio::test]
async fn an_abandoned_request_does_not_hold_up_unloading() {
    let _exclusive = exclusive().await;
    let Some((runtime, id)) = loaded().await else {
        return;
    };

    let started = runtime
        .generate_stream(&id, &long_request())
        .expect("the request is accepted");
    drop(started.stream);

    // The request ends on its own, because nothing is reading it.
    for _ in 0..100 {
        if runtime.active_requests(&id).unwrap_or(0) == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    assert!(matches!(
        runtime.unload(&id).await.expect("unloads"),
        UnloadOutcome::Drained { .. }
    ));
    assert!(runtime.instance(&id).is_none());
}

#[tokio::test]
async fn nothing_survives_an_unload_except_the_artifact() {
    // What must be gone, and what must not be. Unloading destroys an instance and
    // everything it had demonstrated; it does not deregister the artifact, which
    // is a record of a file on disk rather than of a running thing.
    let _exclusive = exclusive().await;
    let Some(runtime) = runtime() else {
        return;
    };
    let Some((registry, artifacts)) = survey() else {
        return;
    };
    let Some(artifact) = generative_artifact(&artifacts) else {
        eprintln!("SKIPPED: no text-generative container found on this machine");
        return;
    };

    let first_id = runtime
        .load(&registry, &artifact.id, worker_id(), deadlines())
        .await
        .expect("loads");

    let started = runtime
        .generate_stream(&first_id, &short_request())
        .expect("the request is accepted");
    let mut stream = started.stream;
    assert!(matches!(
        terminal(&mut stream).await,
        Some(GenerationEvent::Completed { .. })
    ));
    assert!(
        runtime
            .record_generation_stream(&first_id, &stream)
            .expect("resident")
    );
    assert!(
        runtime
            .instance(&first_id)
            .expect("resident")
            .capabilities()
            .is_verified(ModelCapability::TextGeneration),
        "the setup needs a verified capability to prove it does not survive"
    );

    runtime.unload(&first_id).await.expect("unloads");
    assert_eq!(
        runtime.active_requests(&first_id),
        None,
        "an unloaded instance is still resident"
    );

    // The artifact is untouched.
    assert!(
        registry
            .get(&artifact.id)
            .expect("the registry is readable")
            .is_some(),
        "unloading an instance deregistered its artifact"
    );

    // A reload is a different instance that has demonstrated nothing.
    let second_id = runtime
        .load(&registry, &artifact.id, worker_id(), deadlines())
        .await
        .expect("reloads");
    assert_ne!(
        second_id, first_id,
        "a reload reused the identifier of an instance that no longer exists"
    );
    assert!(
        !runtime
            .instance(&second_id)
            .expect("resident")
            .capabilities()
            .is_verified(ModelCapability::TextGeneration),
        "a fresh instance inherited a capability it never demonstrated"
    );

    runtime.unload(&second_id).await.expect("unloads");
}

#[tokio::test]
async fn no_task_class_is_admitted_once_unloading_has_begun() {
    // The invariant is about the runtime, not about one call. A task that reaches
    // the backend through its own check rather than through admission is invisible
    // to unloading, and would be sent to a worker that is being destroyed.
    let _exclusive = exclusive().await;
    let Some((runtime, id)) = loaded().await else {
        return;
    };

    runtime.unload(&id).await.expect("unloads");

    // Streaming.
    assert!(
        matches!(
            runtime.generate_stream(&id, &short_request()),
            Err(RuntimeError::UnknownInstance { .. })
                | Err(RuntimeError::Instance(LoadError::NotUsable { .. }))
        ),
        "streaming was admitted onto an unloaded instance"
    );

    // Whole-response generation.
    match runtime.generate_text(&id, &short_request()).await {
        Err(RuntimeError::UnknownInstance { .. })
        | Err(RuntimeError::Instance(LoadError::NotUsable { .. })) => {}
        other => {
            panic!("whole-response generation was admitted onto an unloaded instance: {other:?}")
        }
    }

    // Embedding. This instance is not an embedding model, which is the point:
    // admission must refuse before the task ever reaches a backend.
    match runtime
        .embed(&id, &mehoy_core::inference::EmbedRequest::single("x"))
        .await
    {
        Err(RuntimeError::UnknownInstance { .. })
        | Err(RuntimeError::Instance(LoadError::NotUsable { .. })) => {}
        other => panic!("embedding was admitted onto an unloaded instance: {other:?}"),
    }

    assert_eq!(runtime.active_requests(&id), None);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn requests_racing_an_unload_are_admitted_or_refused_but_never_stranded() {
    // The race the admission lock exists for. Either a request is admitted before
    // admission closes, in which case unloading owns it and stops it, or it is
    // refused. What must never happen is a request slipping past the check and
    // reaching a worker that is being destroyed, which shows up as a transport
    // error from a socket nobody is listening on.
    let _exclusive = exclusive().await;
    let Some((runtime, id)) = loaded().await else {
        return;
    };
    let runtime = std::sync::Arc::new(runtime);

    let mut racers = Vec::new();
    for _ in 0..8 {
        let shared = std::sync::Arc::clone(&runtime);
        let instance = id.clone();
        racers.push(tokio::spawn(async move {
            let mut verdicts = Vec::new();
            for _ in 0..6 {
                verdicts.push(
                    match shared.generate_text(&instance, &short_request()).await {
                        Ok(_) => "served",
                        // Either the instance has gone, or it was still serving and
                        // admission took the request. Both are legal; reaching a dying
                        // backend is not.
                        Err(RuntimeError::UnknownInstance { .. })
                        | Err(RuntimeError::Instance(LoadError::NotUsable { .. })) => "refused",
                        Err(RuntimeError::Instance(LoadError::Stopped { .. })) => "stopped",
                        Err(RuntimeError::Instance(LoadError::Inference { detail })) => {
                            panic!("a request reached a dying backend: {detail}")
                        }
                        Err(other) => panic!("unexpected failure: {other}"),
                    },
                );
            }
            verdicts
        }));
    }

    // Let some of them get through before pulling the instance out from under them.
    tokio::time::sleep(Duration::from_millis(150)).await;
    let outcome = runtime.unload(&id).await.expect("unloads");

    let mut served = 0usize;
    let mut refused = 0usize;
    let mut stopped = 0usize;
    for racer in racers {
        for verdict in racer.await.expect("the racer did not panic") {
            match verdict {
                "served" => served += 1,
                "refused" => refused += 1,
                "stopped" => stopped += 1,
                other => panic!("unknown verdict {other}"),
            }
        }
    }

    eprintln!("{served} served, {stopped} stopped, {refused} refused; unload {outcome}");
    assert!(
        refused > 0,
        "nothing was refused, so the race never actually happened"
    );
    assert!(
        runtime.instance(&id).is_none(),
        "the instance survived its own teardown"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_admitted_whole_response_request_is_visible_to_unloading() {
    // The other half: a request that got in before admission closed must be in the
    // set unloading cancels, not merely left to fail when its backend disappears.
    let _exclusive = exclusive().await;
    let Some((runtime, id)) = loaded().await else {
        return;
    };
    let runtime = std::sync::Arc::new(runtime);

    let shared = std::sync::Arc::clone(&runtime);
    let instance = id.clone();
    let in_flight =
        tokio::spawn(async move { shared.generate_text(&instance, &long_request()).await });

    // Wait until the runtime is actually holding the request.
    let mut admitted = false;
    for _ in 0..200 {
        if runtime.active_requests(&id).unwrap_or(0) == 1 {
            admitted = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(admitted, "a whole-response request was never registered");

    let outcome = runtime.unload(&id).await.expect("unloads");
    assert!(
        matches!(
            outcome,
            UnloadOutcome::Drained { cancelled: 1 } | UnloadOutcome::Escalated { cancelled: 1, .. }
        ),
        "unloading did not see the in-flight request: {outcome}"
    );

    match in_flight.await.expect("the request task did not panic") {
        Err(RuntimeError::Instance(LoadError::Stopped { cause })) => {
            assert_eq!(cause, CancellationCause::InstanceUnloading);
        }
        other => panic!("expected the request to be stopped, got {other:?}"),
    }
}
