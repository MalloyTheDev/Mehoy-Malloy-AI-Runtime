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
use mehoy_runtime::{LifecyclePhase, LoadError, LoadedModel, ModelCapability, UnloadOutcome};

mod common;
use common::{deadlines, exclusive, generative_artifact, loader, survey, worker_id};

async fn loaded() -> Option<LoadedModel> {
    let loader = loader()?;
    let (registry, artifacts) = survey()?;
    let artifact = generative_artifact(&artifacts).or_else(|| {
        eprintln!("SKIPPED: no text-generative container found on this machine");
        None
    })?;
    Some(
        loader
            .load(&registry, &artifact.id, worker_id(), deadlines())
            .await
            .expect("loads"),
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

/// Whether anything still answers on the backend's address.
async fn backend_reachable(model: &LoadedModel) -> bool {
    tokio::net::TcpStream::connect(model.channel().address())
        .await
        .is_ok()
}

#[tokio::test]
async fn unloading_an_idle_instance_tears_it_down() {
    let _exclusive = exclusive().await;
    let Some(loaded) = loaded().await else { return };

    assert_eq!(loaded.phase(), LifecyclePhase::Serving);
    assert!(backend_reachable(&loaded).await, "the backend should be up");

    assert_eq!(
        loaded.unload().await.expect("unloads"),
        UnloadOutcome::Drained { cancelled: 0 }
    );
    assert_eq!(loaded.phase(), LifecyclePhase::Unloaded);
    assert!(
        !backend_reachable(&loaded).await,
        "something is still listening after the worker was stopped"
    );
}

#[tokio::test]
async fn unloading_twice_is_reported_rather_than_repeated() {
    let _exclusive = exclusive().await;
    let Some(loaded) = loaded().await else { return };

    assert!(matches!(
        loaded.unload().await.expect("unloads"),
        UnloadOutcome::Drained { .. }
    ));
    assert_eq!(
        loaded.unload().await.expect("second unload"),
        UnloadOutcome::AlreadyUnloaded
    );
    assert_eq!(loaded.phase(), LifecyclePhase::Unloaded);
}

#[tokio::test]
async fn no_request_is_admitted_once_unloading_has_begun() {
    // The invariant. A request accepted after this point would be put onto a model
    // that is being destroyed.
    let _exclusive = exclusive().await;
    let Some(loaded) = loaded().await else { return };

    loaded.unload().await.expect("unloads");

    match loaded.generate_stream(&short_request()) {
        Err(LoadError::NotUsable { state }) => {
            assert!(
                state.contains("unload"),
                "the refusal should say the instance is gone, got {state}"
            );
        }
        Err(other) => panic!("expected a refusal, got {other}"),
        Ok(_) => panic!("a request was admitted onto an unloaded instance"),
    }
    assert_eq!(loaded.active_requests(), 0);
}

#[tokio::test]
async fn a_request_in_flight_is_stopped_by_unloading_and_says_why() {
    // Unloading does not end requests by a second mechanism. It uses the same one
    // everything else uses, and the cause records what actually happened.
    let _exclusive = exclusive().await;
    let Some(loaded) = loaded().await else { return };

    let started = loaded
        .generate_stream(&long_request())
        .expect("the request is accepted");
    let request_id = started.request_id;
    let mut stream = started.stream;

    // Let it get going so this is a genuinely in-flight request.
    while let Some(Ok(event)) = stream.next().await {
        if matches!(event, GenerationEvent::TextDelta { .. }) {
            break;
        }
    }
    assert_eq!(loaded.active_requests(), 1);

    let outcome = loaded.unload().await.expect("unloads");
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
    assert_eq!(loaded.phase(), LifecyclePhase::Unloaded);
}

#[tokio::test]
async fn unloading_a_request_that_is_already_stopping_does_not_corrupt_its_outcome() {
    // Both are asking for the same thing at once. The request must still reach
    // exactly one terminal state, and must keep the reason it was first given.
    let _exclusive = exclusive().await;
    let Some(loaded) = loaded().await else { return };

    let started = loaded
        .generate_stream(&long_request())
        .expect("the request is accepted");
    let mut stream = started.stream;
    while let Some(Ok(event)) = stream.next().await {
        if matches!(event, GenerationEvent::TextDelta { .. }) {
            break;
        }
    }

    loaded.cancel(started.request_id).expect("cancels");
    assert_eq!(
        loaded.request_state(started.request_id),
        Some(RequestState::Cancelling)
    );

    loaded.unload().await.expect("unloads");

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
    let Some(loaded) = loaded().await else { return };

    let started = loaded
        .generate_stream(&long_request())
        .expect("the request is accepted");
    // Deliberately not read, and given no time to settle.
    let _stream = started.stream;

    let outcome = loaded
        .unload_within(UnloadBudget {
            drain: Duration::ZERO,
        })
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

    assert_eq!(loaded.phase(), LifecyclePhase::Unloaded);
    assert!(
        !backend_reachable(&loaded).await,
        "the worker outlived an escalated unload"
    );
}

#[tokio::test]
async fn an_abandoned_request_does_not_hold_up_unloading() {
    let _exclusive = exclusive().await;
    let Some(loaded) = loaded().await else { return };

    let started = loaded
        .generate_stream(&long_request())
        .expect("the request is accepted");
    drop(started.stream);

    // The request ends on its own, because nothing is reading it.
    for _ in 0..100 {
        if loaded.active_requests() == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    assert!(matches!(
        loaded.unload().await.expect("unloads"),
        UnloadOutcome::Drained { .. }
    ));
    assert_eq!(loaded.phase(), LifecyclePhase::Unloaded);
}

#[tokio::test]
async fn nothing_survives_an_unload_except_the_artifact() {
    // What must be gone, and what must not be. Unloading destroys an instance and
    // everything it had demonstrated; it does not deregister the artifact, which
    // is a record of a file on disk rather than of a running thing.
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
    let first_id = first.instance().id().clone();

    let started = first
        .generate_stream(&short_request())
        .expect("the request is accepted");
    let mut stream = started.stream;
    assert!(matches!(
        terminal(&mut stream).await,
        Some(GenerationEvent::Completed { .. })
    ));
    assert!(first.record_generation_stream(&stream));
    assert!(
        first
            .instance()
            .capabilities()
            .is_verified(ModelCapability::TextGeneration),
        "the setup needs a verified capability to prove it does not survive"
    );

    first.unload().await.expect("unloads");
    assert_eq!(first.active_requests(), 0, "requests outlived the instance");

    // The artifact is untouched.
    assert!(
        registry
            .get(&artifact.id)
            .expect("the registry is readable")
            .is_some(),
        "unloading an instance deregistered its artifact"
    );

    // A reload is a different instance that has demonstrated nothing.
    let second = loader
        .load(&registry, &artifact.id, worker_id(), deadlines())
        .await
        .expect("reloads");
    assert_ne!(
        second.instance().id(),
        &first_id,
        "a reload reused the identifier of an instance that no longer exists"
    );
    assert!(
        !second
            .instance()
            .capabilities()
            .is_verified(ModelCapability::TextGeneration),
        "a fresh instance inherited a capability it never demonstrated"
    );

    second.unload().await.expect("unloads");
}

#[tokio::test]
async fn no_task_class_is_admitted_once_unloading_has_begun() {
    // The invariant is about the runtime, not about one call. A task that reaches
    // the backend through its own check rather than through admission is invisible
    // to unloading, and would be sent to a worker that is being destroyed.
    let _exclusive = exclusive().await;
    let Some(loaded) = loaded().await else { return };

    loaded.unload().await.expect("unloads");

    // Streaming.
    assert!(
        matches!(
            loaded.generate_stream(&short_request()),
            Err(LoadError::NotUsable { .. })
        ),
        "streaming was admitted onto an unloaded instance"
    );

    // Whole-response generation.
    match loaded.generate_text(&short_request()).await {
        Err(LoadError::NotUsable { .. }) => {}
        other => {
            panic!("whole-response generation was admitted onto an unloaded instance: {other:?}")
        }
    }

    // Embedding. This instance is not an embedding model, which is the point:
    // admission must refuse before the task ever reaches a backend.
    match loaded
        .embed(&mehoy_core::inference::EmbedRequest::single("x"))
        .await
    {
        Err(LoadError::NotUsable { .. }) => {}
        other => panic!("embedding was admitted onto an unloaded instance: {other:?}"),
    }

    assert_eq!(loaded.active_requests(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn requests_racing_an_unload_are_admitted_or_refused_but_never_stranded() {
    // The race the admission lock exists for. Either a request is admitted before
    // admission closes, in which case unloading owns it and stops it, or it is
    // refused. What must never happen is a request slipping past the check and
    // reaching a worker that is being destroyed, which shows up as a transport
    // error from a socket nobody is listening on.
    let _exclusive = exclusive().await;
    let Some(loaded) = loaded().await else { return };
    let loaded = std::sync::Arc::new(loaded);

    let mut racers = Vec::new();
    for _ in 0..8 {
        let model = std::sync::Arc::clone(&loaded);
        racers.push(tokio::spawn(async move {
            let mut verdicts = Vec::new();
            for _ in 0..6 {
                verdicts.push(match model.generate_text(&short_request()).await {
                    Ok(_) => "served",
                    Err(LoadError::NotUsable { .. }) => "refused",
                    Err(LoadError::Stopped { .. }) => "stopped",
                    Err(LoadError::Inference { detail }) => {
                        panic!("a request reached a dying backend: {detail}")
                    }
                    Err(other) => panic!("unexpected failure: {other}"),
                });
            }
            verdicts
        }));
    }

    // Let some of them get through before pulling the instance out from under them.
    tokio::time::sleep(Duration::from_millis(150)).await;
    let outcome = loaded.unload().await.expect("unloads");

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
    assert_eq!(loaded.phase(), LifecyclePhase::Unloaded);
    assert_eq!(loaded.active_requests(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_admitted_whole_response_request_is_visible_to_unloading() {
    // The other half: a request that got in before admission closed must be in the
    // set unloading cancels, not merely left to fail when its backend disappears.
    let _exclusive = exclusive().await;
    let Some(loaded) = loaded().await else { return };
    let loaded = std::sync::Arc::new(loaded);

    let model = std::sync::Arc::clone(&loaded);
    let in_flight = tokio::spawn(async move { model.generate_text(&long_request()).await });

    // Wait until the runtime is actually holding the request.
    let mut admitted = false;
    for _ in 0..200 {
        if loaded.active_requests() == 1 {
            admitted = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(admitted, "a whole-response request was never registered");

    let outcome = loaded.unload().await.expect("unloads");
    assert!(
        matches!(
            outcome,
            UnloadOutcome::Drained { cancelled: 1 } | UnloadOutcome::Escalated { cancelled: 1, .. }
        ),
        "unloading did not see the in-flight request: {outcome}"
    );

    match in_flight.await.expect("the request task did not panic") {
        Err(LoadError::Stopped { cause }) => {
            assert_eq!(cause, CancellationCause::InstanceUnloading);
        }
        other => panic!("expected the request to be stopped, got {other:?}"),
    }
}
