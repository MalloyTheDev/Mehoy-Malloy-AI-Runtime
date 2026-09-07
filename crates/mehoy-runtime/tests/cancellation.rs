//! Stopping a request, addressed by identity.
//!
//! The claim being tested is stronger than "the consumer stopped receiving
//! output". A runtime can always stop showing someone results. What matters is
//! whether the backend stopped doing the work, and whether the model instance
//! survived having one of its requests cancelled.
//!
//! The second half of that is what these tests are really for. Terminating the
//! worker would stop any request reliably and would also evict a resident model,
//! so a cancellation that quietly became a teardown would pass every test that
//! only watched one request.

use std::time::{Duration, Instant};

use mehoy_core::cancel::{CancellationCause, CancellationStrategy, RequestBudget, RequestState};
use mehoy_core::id::RequestId;
use mehoy_core::inference::{
    GenerateTextRequest, GenerationEvent, GenerationParameters, GenerationStream,
};
use mehoy_runtime::{CancelError, CancelOutcome, LoadedModel};

mod common;
use common::{deadlines, exclusive, generative_artifact, loader, survey, worker_id};

/// Loads the generative model, or explains why the test is skipping.
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

/// A request long enough that it cannot finish before being cancelled.
fn long_request() -> GenerateTextRequest {
    GenerateTextRequest::continuation(
        "Write a very long numbered list of every English word you know, one per line, \
         starting at 1 and continuing without stopping.",
    )
    .with_parameters(GenerationParameters {
        max_output_tokens: Some(4096),
        temperature: Some(0.0),
        seed: Some(1),
        stop: Vec::new(),
    })
}

/// A request small enough to finish quickly.
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

/// Reads until the model has produced real content.
async fn read_until_content(stream: &mut GenerationStream) -> bool {
    while let Some(item) = stream.next().await {
        match item {
            Ok(GenerationEvent::TextDelta { .. }) => return true,
            Ok(_) => {}
            Err(error) => panic!("the stream failed before producing anything: {error}"),
        }
    }
    false
}

/// Drains a stream and reports how it ended.
async fn terminal(stream: &mut GenerationStream) -> Option<GenerationEvent> {
    let mut last = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(event @ (GenerationEvent::Completed { .. } | GenerationEvent::Cancelled { .. })) => {
                last = Some(event);
            }
            Ok(_) => {}
            Err(error) => panic!("the stream failed: {error}"),
        }
    }
    last
}

#[tokio::test]
async fn cancelling_an_unknown_request_says_so() {
    let _exclusive = exclusive().await;
    let Some(loaded) = loaded().await else { return };

    match loaded.cancel(RequestId::from_raw(9999)) {
        Err(CancelError::Unknown { request_id }) => {
            assert_eq!(request_id, RequestId::from_raw(9999));
        }
        other => panic!("expected an unknown request, got {other:?}"),
    }

    loaded.unload().await.expect("unloads");
}

#[tokio::test]
async fn a_request_is_cancellable_the_instant_it_is_accepted() {
    // Accepting a request does not wait for the backend, so there is a window in
    // which the request exists and the backend has not been contacted. It must be
    // stoppable in that window, since on a large prompt that window is most of the
    // request.
    let _exclusive = exclusive().await;
    let Some(loaded) = loaded().await else { return };

    let started = loaded
        .generate_stream(&long_request())
        .expect("the request is accepted");
    assert_eq!(loaded.active_requests(), 1);

    assert_eq!(
        loaded.cancel(started.request_id),
        Ok(CancelOutcome::Requested)
    );
    assert_eq!(
        loaded.request_state(started.request_id),
        Some(RequestState::Cancelling),
        "asking is not the same as having stopped"
    );

    let mut stream = started.stream;
    match terminal(&mut stream).await {
        Some(GenerationEvent::Cancelled { cause, .. }) => {
            assert_eq!(cause, CancellationCause::User);
        }
        other => panic!("expected a cancellation, got {other:?}"),
    }
    assert_eq!(
        loaded.request_state(started.request_id),
        Some(RequestState::Cancelled)
    );

    loaded.unload().await.expect("unloads");
}

#[tokio::test]
async fn cancelling_twice_is_reported_without_being_an_error() {
    let _exclusive = exclusive().await;
    let Some(loaded) = loaded().await else { return };

    let started = loaded
        .generate_stream(&long_request())
        .expect("the request is accepted");

    assert_eq!(
        loaded.cancel(started.request_id),
        Ok(CancelOutcome::Requested)
    );
    match loaded.cancel(started.request_id) {
        Ok(CancelOutcome::AlreadyStopping { cause }) => {
            assert_eq!(cause, CancellationCause::User);
        }
        // The request may already have finished stopping between the two calls,
        // which is a legitimate outcome rather than a flaky one.
        Ok(CancelOutcome::AlreadyFinished { state }) => {
            assert_eq!(state, RequestState::Cancelled);
        }
        other => panic!("expected an already-stopping report, got {other:?}"),
    }

    let mut stream = started.stream;
    let _ = terminal(&mut stream).await;
    loaded.unload().await.expect("unloads");
}

#[tokio::test]
async fn cancelling_a_finished_request_reports_that_it_finished() {
    let _exclusive = exclusive().await;
    let Some(loaded) = loaded().await else { return };

    let started = loaded
        .generate_stream(&short_request())
        .expect("the request is accepted");
    let request_id = started.request_id;
    let mut stream = started.stream;
    assert!(matches!(
        terminal(&mut stream).await,
        Some(GenerationEvent::Completed { .. })
    ));

    match loaded.cancel(request_id) {
        Ok(CancelOutcome::AlreadyFinished { state }) => {
            assert_eq!(state, RequestState::Completed);
        }
        // Forgotten rather than remembered is also correct: the runtime keeps no
        // permanent history of every request it has served.
        Err(CancelError::Unknown { .. }) => {}
        other => panic!("expected a finished report, got {other:?}"),
    }

    loaded.unload().await.expect("unloads");
}

#[tokio::test]
async fn dropping_a_stream_is_not_reported_as_a_user_cancellation() {
    // ADR-0009. If a destructor meant the same thing as asking to cancel, an
    // ordinary refactor could look like a deliberate stop in the record. The
    // request does end, because nothing can receive its output, and it says so in
    // its own terms.
    let _exclusive = exclusive().await;
    let Some(loaded) = loaded().await else { return };

    let started = loaded
        .generate_stream(&long_request())
        .expect("the request is accepted");
    let request_id = started.request_id;
    drop(started.stream);

    let mut ended = false;
    for _ in 0..100 {
        match loaded.request_state(request_id) {
            Some(state) if state.is_terminal() => {
                ended = true;
                break;
            }
            _ => tokio::time::sleep(Duration::from_millis(50)).await,
        }
    }
    assert!(
        ended,
        "an abandoned request never ended, so the runtime would count it forever"
    );
    assert_eq!(
        loaded.active_requests(),
        0,
        "an abandoned request is still being tracked"
    );

    loaded.unload().await.expect("unloads");
}

#[tokio::test]
async fn a_cancelled_request_leaves_the_model_usable() {
    // The test that matters most. Terminating the worker would cancel a request
    // and would also evict the model, and nothing about watching one request would
    // reveal the difference.
    let _exclusive = exclusive().await;
    let Some(mut loaded) = loaded().await else {
        return;
    };

    let first = loaded
        .generate_stream(&long_request())
        .expect("the first request is accepted");
    let mut first_stream = first.stream;
    assert!(
        read_until_content(&mut first_stream).await,
        "the model produced nothing to interrupt"
    );

    let cancelled_at = Instant::now();
    assert_eq!(
        loaded.cancel(first.request_id),
        Ok(CancelOutcome::Requested)
    );
    match terminal(&mut first_stream).await {
        Some(GenerationEvent::Cancelled { cause, .. }) => {
            assert_eq!(cause, CancellationCause::User);
            eprintln!("cancelled after {:?}", cancelled_at.elapsed());
        }
        other => panic!("expected a cancellation, got {other:?}"),
    }
    assert!(
        !first_stream.demonstrated_generation(),
        "a cancelled request is not evidence that generation works"
    );

    // The instance must still be alive and serving.
    assert!(
        loaded.instance().state().is_usable(),
        "cancelling a request left the instance unusable"
    );

    let second = loaded
        .generate_stream(&short_request())
        .expect("the second request is accepted");
    let mut second_stream = second.stream;
    match terminal(&mut second_stream).await {
        Some(GenerationEvent::Completed { .. }) => {}
        other => panic!("the model could not serve a second request: {other:?}"),
    }
    assert!(
        loaded.record_generation_stream(&second_stream),
        "the second request should have demonstrated generation"
    );

    eprintln!(
        "cancelled one request and completed the next on the same instance, stopping {}",
        loaded.cancellation_strategy()
    );
    assert_eq!(
        loaded.cancellation_strategy(),
        CancellationStrategy::ConnectionAbort
    );

    loaded.unload().await.expect("unloads");
}

#[tokio::test]
async fn finished_requests_are_not_tracked_forever() {
    let _exclusive = exclusive().await;
    let Some(loaded) = loaded().await else { return };

    for _ in 0..3 {
        let started = loaded
            .generate_stream(&short_request())
            .expect("the request is accepted");
        let mut stream = started.stream;
        let _ = terminal(&mut stream).await;
    }

    assert_eq!(
        loaded.active_requests(),
        0,
        "requests that ended are still being tracked"
    );

    loaded.unload().await.expect("unloads");
}

#[tokio::test]
async fn a_silent_backend_stops_the_request_with_its_own_cause() {
    // The idle budget and an explicit cancellation share one stopping mechanism,
    // and must not share one reported reason.
    let _exclusive = exclusive().await;
    let Some(loaded) = loaded().await else { return };

    // Short enough to expire while the backend is still reading a large prompt.
    let budget = RequestBudget {
        stream_idle: Duration::from_millis(250),
    };
    let filler = "The quick brown fox jumps over the lazy dog. ".repeat(1500);
    let slow = GenerateTextRequest::continuation(format!("{filler}\n\nSummarise the above."))
        .with_parameters(GenerationParameters {
            max_output_tokens: Some(4096),
            temperature: Some(0.0),
            seed: Some(1),
            stop: Vec::new(),
        });

    let started = loaded
        .generate_stream_within(&slow, budget)
        .expect("the request is accepted");
    let mut stream = started.stream;

    match terminal(&mut stream).await {
        Some(GenerationEvent::Cancelled { cause, .. }) => {
            assert_eq!(
                cause,
                CancellationCause::StreamIdleTimeout,
                "an expired budget must not be reported as somebody cancelling"
            );
        }
        other => panic!("expected an idle timeout, got {other:?}"),
    }

    loaded.unload().await.expect("unloads");
}
