//! Streaming generation against a backend that can be made to misbehave.
//!
//! A real engine produces well-formed streams almost all of the time, which makes
//! it the wrong instrument for testing what happens when one is malformed,
//! truncated, or read too slowly. These tests stand up a server that speaks
//! exactly the bytes each case needs, so every failure path is deterministic and
//! runs in milliseconds without loading a model.
//!
//! The happy path is still verified against a real model elsewhere. This file
//! covers what a real model will not reliably do on request.

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use mehoy_backend_llama::{BackendChannel, ChannelSecret, start_generation};
use mehoy_core::cancel::{CancellationCause, RequestBudget, RequestHandle};
use mehoy_core::id::RequestId;
use mehoy_core::inference::{
    FinishReason, GenerateTextRequest, GenerationEvent, GenerationStream, GenerationStreamError,
};

/// A response the fake backend will send.
enum Script {
    /// Send these byte runs in order, then close the connection.
    ///
    /// Runs are written separately so a test can place a split exactly where it
    /// wants one, including in the middle of an event.
    Chunks(Vec<Vec<u8>>),
    /// Answer with this status and body instead of a stream.
    Status(u16, &'static str),
    /// Accept the connection and never answer at all, not even headers.
    NoHeaders,
    /// Answer with headers and then say nothing, forever.
    Silent,
    /// Send these runs, then hold the connection open without sending anything.
    ChunksThenSilence(Vec<Vec<u8>>),
    /// Say nothing, and record when the peer goes away.
    SilentUntilClosed { closed: Arc<AtomicBool> },
    /// Send events until the client stops reading, counting what got through.
    Flood {
        events: usize,
        written: Arc<AtomicUsize>,
        completed: Arc<AtomicBool>,
    },
}

/// Starts a backend that plays one script for one connection.
async fn backend(script: Script) -> BackendChannel {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .await
        .expect("a loopback port");
    let address: SocketAddr = listener.local_addr().expect("an address");

    tokio::spawn(async move {
        let Ok((socket, _)) = listener.accept().await else {
            return;
        };
        let (mut inbound, mut outbound) = socket.into_split();

        // Drained continuously so the client's request write always completes.
        tokio::spawn(async move {
            let mut scratch = [0u8; 4096];
            while inbound.read(&mut scratch).await.unwrap_or(0) > 0 {}
        });

        match script {
            Script::Status(status, body) => {
                let response = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                    body.len()
                );
                let _ = outbound.write_all(response.as_bytes()).await;
            }
            Script::Chunks(chunks) => {
                let _ = outbound.write_all(headers().as_bytes()).await;
                for chunk in chunks {
                    if outbound.write_all(&chunk).await.is_err() {
                        return;
                    }
                    let _ = outbound.flush().await;
                }
            }
            Script::NoHeaders => {
                std::future::pending::<()>().await;
            }
            Script::Silent => {
                let _ = outbound.write_all(headers().as_bytes()).await;
                let _ = outbound.flush().await;
                std::future::pending::<()>().await;
            }
            Script::ChunksThenSilence(chunks) => {
                let _ = outbound.write_all(headers().as_bytes()).await;
                for chunk in chunks {
                    if outbound.write_all(&chunk).await.is_err() {
                        return;
                    }
                    let _ = outbound.flush().await;
                }
                std::future::pending::<()>().await;
            }
            Script::SilentUntilClosed { closed } => {
                let _ = outbound.write_all(headers().as_bytes()).await;
                let _ = outbound.flush().await;
                // Writing to a socket whose peer has gone eventually fails, which
                // is how this observes the cancellation taking effect.
                loop {
                    tokio::time::sleep(Duration::from_millis(25)).await;
                    if outbound.write_all(b": ping\n\n").await.is_err() {
                        closed.store(true, Ordering::SeqCst);
                        return;
                    }
                }
            }
            Script::Flood {
                events,
                written,
                completed,
            } => {
                let _ = outbound.write_all(headers().as_bytes()).await;
                // Large enough that the operating system's socket buffers cannot
                // absorb the whole flood, so backpressure has to come from the
                // runtime rather than from the kernel being generous.
                let filler = "x".repeat(8192);
                for index in 0..events {
                    let event = format!(
                        "data: {{\"choices\":[{{\"text\":\"{filler}\",\"finish_reason\":null}}]}}\n\n"
                    );
                    if outbound.write_all(event.as_bytes()).await.is_err() {
                        return;
                    }
                    written.store(index + 1, Ordering::SeqCst);
                }
                completed.store(true, Ordering::SeqCst);
            }
        }
    });

    BackendChannel::at(address, ChannelSecret::generate().expect("a secret"))
}

fn headers() -> String {
    // No length and no chunked encoding, so the body ends when the connection
    // closes. That is what makes a truncated stream expressible.
    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n".to_owned()
}

fn event(text: &str, finish: Option<&str>) -> Vec<u8> {
    let finish = finish.map_or_else(|| "null".to_owned(), |value| format!("\"{value}\""));
    format!("data: {{\"choices\":[{{\"text\":\"{text}\",\"finish_reason\":{finish}}}]}}\n\n")
        .into_bytes()
}

fn terminal_with_usage(text: &str) -> Vec<u8> {
    format!(
        "data: {{\"choices\":[{{\"text\":\"{text}\",\"finish_reason\":\"stop\"}}],\
\"usage\":{{\"prompt_tokens\":4,\"completion_tokens\":3}}}}\n\n"
    )
    .into_bytes()
}

async fn open(script: Script) -> GenerationStream {
    open_within(script, RequestBudget::default()).await
}

/// Starts a generation against the fake backend.
///
/// Returns a stream rather than a result: starting a request no longer waits for
/// the backend, so a refusal arrives through the stream like every other failure.
async fn open_within(script: Script, budget: RequestBudget) -> GenerationStream {
    let channel = backend(script).await;
    let request = GenerateTextRequest::continuation("hello");
    let handle = RequestHandle::new(RequestId::from_raw(7));
    start_generation(&channel, &request, handle, budget)
}

/// Drains a stream, returning the events and the outcome.
async fn drain(
    stream: &mut GenerationStream,
) -> (Vec<GenerationEvent>, Option<GenerationStreamError>) {
    let mut events = Vec::new();
    let mut failure = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(event) => events.push(event),
            Err(error) => {
                failure = Some(error);
                break;
            }
        }
    }
    (events, failure)
}

/// Asserts the sequence is Started, then deltas, then Completed, mechanically.
///
/// Checking that each kind appeared somewhere would pass on a stream that emitted
/// them in any order, which is the defect worth catching.
fn assert_shape(events: &[GenerationEvent], expected_text: &str) -> FinishReason {
    let mut kinds = events.iter();

    let first = kinds.next().expect("a stream always starts");
    assert!(
        matches!(first, GenerationEvent::Started { .. }),
        "the first event must be Started, got {first:?}"
    );

    let mut text = String::new();
    let mut finish = None;
    for (position, event) in kinds.enumerate() {
        match event {
            GenerationEvent::Started { .. } => {
                panic!("Started appeared again at position {}", position + 1)
            }
            GenerationEvent::TextDelta { text: delta, .. } => {
                assert!(
                    finish.is_none(),
                    "a delta followed the completion at position {}",
                    position + 1
                );
                assert!(!delta.is_empty(), "an empty delta reached the consumer");
                text.push_str(delta);
            }
            GenerationEvent::Completed { summary, .. } => {
                assert!(
                    finish.is_none(),
                    "Completed appeared twice at position {}",
                    position + 1
                );
                finish = Some(summary.finish_reason.clone());
            }
            GenerationEvent::Cancelled { cause, .. } => {
                panic!(
                    "unexpected cancellation ({cause}) at position {}",
                    position + 1
                )
            }
        }
    }

    assert_eq!(text, expected_text);
    finish.expect("the stream must have completed")
}

#[tokio::test]
async fn a_normal_stream_starts_delivers_and_completes_in_that_order() {
    let mut stream = open(Script::Chunks(vec![
        event("Hel", None),
        event("lo", None),
        terminal_with_usage(""),
        b"data: [DONE]\n\n".to_vec(),
    ]))
    .await;

    let (events, failure) = drain(&mut stream).await;
    assert!(failure.is_none(), "unexpected failure: {failure:?}");
    assert_eq!(assert_shape(&events, "Hello"), FinishReason::Stop);
}

#[tokio::test]
async fn every_event_carries_the_request_it_belongs_to() {
    let mut stream = open(Script::Chunks(vec![
        event("hi", None),
        terminal_with_usage(""),
        b"data: [DONE]\n\n".to_vec(),
    ]))
    .await;

    let (events, _) = drain(&mut stream).await;
    assert!(!events.is_empty());
    for event in &events {
        assert_eq!(event.request_id(), RequestId::from_raw(7));
    }
}

#[tokio::test]
async fn usage_and_finish_reason_are_normalised_from_the_terminal_event() {
    let mut stream = open(Script::Chunks(vec![
        event("hi", None),
        terminal_with_usage(""),
        b"data: [DONE]\n\n".to_vec(),
    ]))
    .await;

    let (events, _) = drain(&mut stream).await;
    let GenerationEvent::Completed { summary, .. } = events.last().expect("a last event") else {
        panic!("expected a completion, got {:?}", events.last());
    };
    assert_eq!(summary.finish_reason, FinishReason::Stop);
    let usage = summary.usage.expect("usage was reported");
    assert_eq!(usage.input_tokens, 4);
    assert_eq!(usage.output_tokens, 3);
}

#[tokio::test]
async fn an_event_split_across_writes_arrives_as_one_delta() {
    // The transport-level version of the decoder's unit test: a split placed in
    // the middle of a JSON payload must not produce two deltas or a parse failure.
    let whole = event("together", None);
    let (head, tail) = whole.split_at(20);
    let mut stream = open(Script::Chunks(vec![
        head.to_vec(),
        tail.to_vec(),
        terminal_with_usage(""),
        b"data: [DONE]\n\n".to_vec(),
    ]))
    .await;

    let (events, failure) = drain(&mut stream).await;
    assert!(failure.is_none(), "unexpected failure: {failure:?}");
    let deltas: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            GenerationEvent::TextDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(deltas, vec!["together"]);
}

#[tokio::test]
async fn multibyte_text_split_across_writes_is_never_delivered_broken() {
    let whole = event("\u{4f60}\u{597d}\u{1f600}", None);
    for split in 1..whole.len() {
        let mut stream = open(Script::Chunks(vec![
            whole[..split].to_vec(),
            whole[split..].to_vec(),
            terminal_with_usage(""),
            b"data: [DONE]\n\n".to_vec(),
        ]))
        .await;

        let (events, failure) = drain(&mut stream).await;
        assert!(failure.is_none(), "split {split} failed: {failure:?}");
        assert_eq!(
            assert_shape(&events, "\u{4f60}\u{597d}\u{1f600}"),
            FinishReason::Stop,
            "split at {split}"
        );
    }
}

#[tokio::test]
async fn an_empty_delta_never_reaches_the_consumer() {
    // The terminal event of a real stream usually carries no text. Forwarding it
    // would make every consumer filter events that mean nothing.
    let mut stream = open(Script::Chunks(vec![
        event("", None),
        event("real", None),
        event("", None),
        terminal_with_usage(""),
        b"data: [DONE]\n\n".to_vec(),
    ]))
    .await;

    let (events, _) = drain(&mut stream).await;
    assert_eq!(assert_shape(&events, "real"), FinishReason::Stop);
    assert_eq!(events.len(), 3, "expected Started, one delta, Completed");
}

#[tokio::test]
async fn a_malformed_event_fails_the_stream_rather_than_completing_it() {
    let mut stream = open(Script::Chunks(vec![
        event("good", None),
        b"data: {\"choices\": [ this is not json\n\n".to_vec(),
        terminal_with_usage(""),
        b"data: [DONE]\n\n".to_vec(),
    ]))
    .await;

    let (events, failure) = drain(&mut stream).await;
    assert!(
        matches!(failure, Some(GenerationStreamError::MalformedEvent { .. })),
        "expected a malformed event, got {failure:?}"
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, GenerationEvent::Completed { .. })),
        "a completion was fabricated from a broken stream"
    );
    assert!(!stream.demonstrated_generation());
}

#[tokio::test]
async fn a_truncated_stream_is_an_error_and_not_a_completion() {
    // The backend produced content and then vanished. A consumer that only checked
    // for the end of the stream would treat this as a finished answer.
    let mut stream = open(Script::Chunks(vec![event("partial answer", None)])).await;

    let (events, failure) = drain(&mut stream).await;
    assert!(
        matches!(failure, Some(GenerationStreamError::UnexpectedEnd { .. })),
        "expected an unexpected end, got {failure:?}"
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, GenerationEvent::Completed { .. })),
        "a truncated stream must not complete"
    );
    assert!(!stream.demonstrated_generation());
}

#[tokio::test]
async fn a_stream_cut_inside_an_event_is_an_error() {
    let whole = event("started", None);
    let mut stream = open(Script::Chunks(vec![whole[..15].to_vec()])).await;

    let (_, failure) = drain(&mut stream).await;
    match failure {
        Some(GenerationStreamError::UnexpectedEnd { detail }) => {
            assert!(detail.contains("part way"), "unexpected detail: {detail}");
        }
        other => panic!("expected an unexpected end, got {other:?}"),
    }
}

#[tokio::test]
async fn a_done_marker_without_a_finish_reason_is_an_error() {
    // The engine said it was finished but never said why. Inventing a reason would
    // report a truncation as an ordinary stop.
    let mut stream = open(Script::Chunks(vec![
        event("text", None),
        b"data: [DONE]\n\n".to_vec(),
    ]))
    .await;

    let (_, failure) = drain(&mut stream).await;
    assert!(
        matches!(failure, Some(GenerationStreamError::UnexpectedEnd { .. })),
        "expected an unexpected end, got {failure:?}"
    );
}

#[tokio::test]
async fn a_refused_request_fails_through_its_stream() {
    // Starting a request no longer waits for the backend, so a refusal cannot be
    // reported by the starting call. It has to reach the caller the same way every
    // other failure does.
    let mut stream = open(Script::Status(401, "{}")).await;
    let (events, failure) = drain(&mut stream).await;

    match failure {
        Some(GenerationStreamError::Refused { detail }) => {
            assert!(detail.contains("credential"), "unexpected detail: {detail}");
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    assert_eq!(
        events.len(),
        1,
        "a refused request should still have been accepted, and nothing more"
    );
    assert!(matches!(events[0], GenerationEvent::Started { .. }));
    assert!(!stream.demonstrated_generation());
}

#[tokio::test]
async fn a_request_is_accepted_before_the_backend_is_reached() {
    // The property ADR-0009 turns on: the request exists, and can be observed and
    // stopped, before anything has been sent anywhere.
    let mut stream = open(Script::Chunks(vec![
        event("hi", None),
        terminal_with_usage(""),
        b"data: [DONE]\n\n".to_vec(),
    ]))
    .await;

    assert_eq!(
        stream.next().await,
        Some(Ok(GenerationEvent::Started {
            request_id: RequestId::from_raw(7)
        })),
        "acceptance is the first thing a consumer sees"
    );
}

#[tokio::test]
async fn nothing_is_yielded_after_the_stream_ends() {
    let mut stream = open(Script::Chunks(vec![
        event("hi", None),
        terminal_with_usage(""),
        b"data: [DONE]\n\n".to_vec(),
    ]))
    .await;

    let (events, _) = drain(&mut stream).await;
    assert!(matches!(
        events.last(),
        Some(GenerationEvent::Completed { .. })
    ));
    assert!(stream.next().await.is_none(), "an event followed Completed");
    assert!(stream.next().await.is_none(), "the stream restarted");
}

#[tokio::test]
async fn a_stream_only_counts_as_evidence_once_it_completes() {
    let mut stream = open(Script::Chunks(vec![
        event("hi", None),
        terminal_with_usage(""),
        b"data: [DONE]\n\n".to_vec(),
    ]))
    .await;

    assert!(
        !stream.demonstrated_generation(),
        "an unread stream has demonstrated nothing"
    );

    let mut completed = false;
    while let Some(Ok(event)) = stream.next().await {
        match event {
            GenerationEvent::TextDelta { .. } => assert!(
                !stream.demonstrated_generation(),
                "a delta alone must not count: the backend may still fail"
            ),
            GenerationEvent::Completed { .. } => completed = true,
            GenerationEvent::Started { .. } => {}
            GenerationEvent::Cancelled { cause, .. } => panic!("unexpected cancellation: {cause}"),
        }
    }

    assert!(completed);
    assert!(stream.demonstrated_generation());
}

#[tokio::test]
async fn a_completion_carrying_no_content_demonstrates_nothing() {
    let mut stream = open(Script::Chunks(vec![
        terminal_with_usage(""),
        b"data: [DONE]\n\n".to_vec(),
    ]))
    .await;

    let (events, failure) = drain(&mut stream).await;
    assert!(failure.is_none(), "unexpected failure: {failure:?}");
    assert!(matches!(
        events.last(),
        Some(GenerationEvent::Completed { .. })
    ));
    assert!(
        !stream.demonstrated_generation(),
        "a completion with no text is not demonstrated generation"
    );
}

#[tokio::test]
async fn a_consumer_that_stops_reading_stops_the_backend() {
    // The property that matters is bounded memory: a slow consumer must slow the
    // producer rather than let this process buffer everything the backend can send.
    let written = Arc::new(AtomicUsize::new(0));
    let completed = Arc::new(AtomicBool::new(false));
    const EVENTS: usize = 2000;

    let mut stream = open(Script::Flood {
        events: EVENTS,
        written: Arc::clone(&written),
        completed: Arc::clone(&completed),
    })
    .await;

    // Take a couple of events, then stop.
    assert!(matches!(
        stream.next().await,
        Some(Ok(GenerationEvent::Started { .. }))
    ));
    assert!(matches!(
        stream.next().await,
        Some(Ok(GenerationEvent::TextDelta { .. }))
    ));

    tokio::time::sleep(std::time::Duration::from_millis(750)).await;

    assert!(
        !completed.load(Ordering::SeqCst),
        "the backend sent all {EVENTS} events while the consumer was idle"
    );
    let sent = written.load(Ordering::SeqCst);
    assert!(
        sent < EVENTS / 2,
        "backpressure did not hold: {sent} of {EVENTS} events were written"
    );
}

#[tokio::test]
async fn abandoning_a_stream_releases_the_backend() {
    let written = Arc::new(AtomicUsize::new(0));
    let completed = Arc::new(AtomicBool::new(false));

    let mut stream = open(Script::Flood {
        events: 2000,
        written: Arc::clone(&written),
        completed: Arc::clone(&completed),
    })
    .await;

    assert!(stream.next().await.is_some());
    drop(stream);

    // The reader notices at its next event and lets the connection go, which the
    // fake backend observes as a failed write.
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    assert!(
        !completed.load(Ordering::SeqCst),
        "the backend ran to completion after its consumer went away"
    );
}

// ------------------------------------------------------------------ cancellation

/// Starts a generation whose handle the test keeps, so it can stop it.
async fn open_cancellable(
    script: Script,
    budget: RequestBudget,
) -> (RequestHandle, GenerationStream) {
    let channel = backend(script).await;
    let request = GenerateTextRequest::continuation("hello");
    let handle = RequestHandle::new(RequestId::from_raw(11));
    let stream = start_generation(&channel, &request, handle.clone(), budget);
    (handle, stream)
}

#[tokio::test]
async fn a_request_cancelled_before_anything_arrives_ends_as_cancelled() {
    // The hardest case on the measured engine: a large prompt means the backend
    // sends nothing for a long time, so this is the window where cancellation is
    // most valuable and least observable.
    let (handle, mut stream) = open_cancellable(Script::Silent, RequestBudget::default()).await;

    assert_eq!(
        stream.next().await,
        Some(Ok(GenerationEvent::Started {
            request_id: RequestId::from_raw(11)
        }))
    );
    assert!(handle.request_cancellation(CancellationCause::User));

    let (events, failure) = drain(&mut stream).await;
    assert!(
        failure.is_none(),
        "cancelling is not a failure: {failure:?}"
    );
    match events.last() {
        Some(GenerationEvent::Cancelled { cause, .. }) => {
            assert_eq!(*cause, CancellationCause::User);
        }
        other => panic!("expected a cancellation, got {other:?}"),
    }
    assert!(!stream.demonstrated_generation());
}

#[tokio::test]
async fn a_request_cancelled_after_content_keeps_what_arrived() {
    let (handle, mut stream) = open_cancellable(
        Script::ChunksThenSilence(vec![event("partial", None)]),
        RequestBudget::default(),
    )
    .await;

    // Wait for real content, then stop.
    let mut text = String::new();
    while let Some(Ok(item)) = stream.next().await {
        if let GenerationEvent::TextDelta { text: delta, .. } = item {
            text.push_str(&delta);
            break;
        }
    }
    assert_eq!(text, "partial");
    handle.request_cancellation(CancellationCause::User);

    let (events, failure) = drain(&mut stream).await;
    assert!(failure.is_none(), "unexpected failure: {failure:?}");
    assert!(matches!(
        events.last(),
        Some(GenerationEvent::Cancelled { .. })
    ));
    assert!(
        !stream.demonstrated_generation(),
        "a cancelled request is not evidence that generation works"
    );
}

#[tokio::test]
async fn a_silent_backend_is_stopped_by_the_idle_budget() {
    // Issue 6: the liveness bound is now reachable from a test because the budget
    // is a parameter rather than a constant.
    let (_handle, mut stream) = open_cancellable(
        Script::ChunksThenSilence(vec![event("one", None)]),
        RequestBudget {
            stream_idle: Duration::from_millis(300),
        },
    )
    .await;

    let (events, failure) = drain(&mut stream).await;
    assert!(
        failure.is_none(),
        "an idle backend is stopped, not a failure: {failure:?}"
    );
    match events.last() {
        Some(GenerationEvent::Cancelled { cause, .. }) => {
            assert_eq!(
                *cause,
                CancellationCause::StreamIdleTimeout,
                "the cause must say it timed out rather than that someone asked"
            );
        }
        other => panic!("expected a cancellation, got {other:?}"),
    }
    assert!(
        events
            .iter()
            .any(|event| matches!(event, GenerationEvent::TextDelta { .. })),
        "the delta that did arrive should still have been delivered"
    );
}

#[tokio::test]
async fn a_user_cancellation_beats_a_later_idle_timeout() {
    // Both use the same stopping machinery, and the cause must still say which
    // one actually stopped it.
    let (handle, mut stream) = open_cancellable(
        Script::Silent,
        RequestBudget {
            stream_idle: Duration::from_secs(30),
        },
    )
    .await;

    handle.request_cancellation(CancellationCause::User);
    let (events, _) = drain(&mut stream).await;
    match events.last() {
        Some(GenerationEvent::Cancelled { cause, .. }) => {
            assert_eq!(*cause, CancellationCause::User);
        }
        other => panic!("expected a cancellation, got {other:?}"),
    }
}

#[tokio::test]
async fn cancelling_a_request_that_already_completed_changes_nothing() {
    let (handle, mut stream) = open_cancellable(
        Script::Chunks(vec![
            event("done", None),
            terminal_with_usage(""),
            b"data: [DONE]\n\n".to_vec(),
        ]),
        RequestBudget::default(),
    )
    .await;

    let (events, _) = drain(&mut stream).await;
    assert!(matches!(
        events.last(),
        Some(GenerationEvent::Completed { .. })
    ));
    assert!(
        !handle.request_cancellation(CancellationCause::User),
        "a finished request cannot be cancelled"
    );
    assert_eq!(handle.state(), mehoy_core::cancel::RequestState::Completed);
    assert!(
        stream.next().await.is_none(),
        "an event followed the terminal one"
    );
}

#[tokio::test]
async fn a_cancelled_request_releases_its_connection() {
    // The mechanism this backend actually has: closing the request's transport.
    // If the connection stayed open, the engine would keep working.
    let closed = Arc::new(AtomicBool::new(false));
    let channel = backend(Script::SilentUntilClosed {
        closed: Arc::clone(&closed),
    })
    .await;
    let request = GenerateTextRequest::continuation("hello");
    let handle = RequestHandle::new(RequestId::from_raw(12));
    let mut stream = start_generation(&channel, &request, handle.clone(), RequestBudget::default());

    assert!(matches!(
        stream.next().await,
        Some(Ok(GenerationEvent::Started { .. }))
    ));
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        !closed.load(Ordering::SeqCst),
        "the connection closed early"
    );

    handle.request_cancellation(CancellationCause::User);
    let (events, _) = drain(&mut stream).await;
    assert!(matches!(
        events.last(),
        Some(GenerationEvent::Cancelled { .. })
    ));

    // The fake backend notices its peer has gone.
    for _ in 0..50 {
        if closed.load(Ordering::SeqCst) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("the request's connection was never closed, so the backend was never told to stop");
}

#[tokio::test]
async fn a_request_cancelled_before_the_backend_answers_at_all_ends_as_cancelled() {
    // Cancelling before any response header exists. On the measured engine this is
    // the whole input-processing window, which for a large prompt is most of the
    // request.
    let (handle, mut stream) = open_cancellable(Script::NoHeaders, RequestBudget::default()).await;

    assert!(matches!(
        stream.next().await,
        Some(Ok(GenerationEvent::Started { .. }))
    ));
    assert_eq!(handle.state(), mehoy_core::cancel::RequestState::Starting);
    handle.request_cancellation(CancellationCause::User);

    let (events, failure) = drain(&mut stream).await;
    assert!(
        failure.is_none(),
        "cancelling is not a failure: {failure:?}"
    );
    assert!(matches!(
        events.last(),
        Some(GenerationEvent::Cancelled {
            cause: CancellationCause::User,
            ..
        })
    ));
    assert_eq!(handle.state(), mehoy_core::cancel::RequestState::Cancelled);
}

#[tokio::test]
async fn a_backend_that_dies_while_cancelling_still_reports_a_cancellation() {
    // Both things happen at once. Whichever the execution notices first must be
    // the only terminal outcome, and a failure must never be dressed up as a
    // clean cancellation nor the reverse.
    let (handle, mut stream) = open_cancellable(
        Script::Chunks(vec![event("some", None)]),
        RequestBudget::default(),
    )
    .await;

    handle.request_cancellation(CancellationCause::User);
    let (events, failure) = drain(&mut stream).await;

    let terminals = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                GenerationEvent::Completed { .. } | GenerationEvent::Cancelled { .. }
            )
        })
        .count();
    assert!(
        terminals <= 1,
        "a request reported {terminals} terminal events"
    );
    assert!(
        terminals == 1 || failure.is_some(),
        "the request ended with neither a terminal event nor a failure"
    );
    assert!(
        handle.is_terminal(),
        "the request never reached a terminal state"
    );
}

#[tokio::test]
async fn dropping_the_stream_is_not_a_cancellation() {
    // ADR-0009: a consumer that stops reading has not asked for the work to stop.
    // The request must not silently report itself cancelled because a value went
    // out of scope.
    let (handle, stream) = open_cancellable(Script::Silent, RequestBudget::default()).await;

    drop(stream);
    tokio::time::sleep(Duration::from_millis(200)).await;

    assert_eq!(
        handle.cancellation_cause(),
        None,
        "dropping a stream must not record a cancellation cause"
    );
    assert_ne!(
        handle.state(),
        mehoy_core::cancel::RequestState::Cancelled,
        "dropping a stream must not mark the request cancelled"
    );
}

#[tokio::test]
async fn a_request_reports_exactly_one_terminal_outcome_when_cancel_races_completion() {
    // The cancellation and the last event are deliberately made to arrive at
    // roughly the same moment, repeatedly, so the winner varies between runs.
    for attempt in 0..25 {
        let (handle, mut stream) = open_cancellable(
            Script::Chunks(vec![
                event("racing", None),
                terminal_with_usage(""),
                b"data: [DONE]

"
                .to_vec(),
            ]),
            RequestBudget::default(),
        )
        .await;

        if attempt % 2 == 0 {
            tokio::task::yield_now().await;
        }
        handle.request_cancellation(CancellationCause::User);

        let (events, failure) = drain(&mut stream).await;
        let terminals: Vec<_> = events
            .iter()
            .filter(|event| {
                matches!(
                    event,
                    GenerationEvent::Completed { .. } | GenerationEvent::Cancelled { .. }
                )
            })
            .collect();
        assert!(
            terminals.len() <= 1,
            "attempt {attempt} produced {} terminal events: {terminals:?}",
            terminals.len()
        );
        assert!(
            terminals.len() == 1 || failure.is_some(),
            "attempt {attempt} ended with no outcome at all"
        );
        assert!(
            handle.is_terminal(),
            "attempt {attempt} left a live request"
        );
    }
}
