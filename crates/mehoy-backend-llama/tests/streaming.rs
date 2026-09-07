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

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use mehoy_backend_llama::{BackendChannel, ChannelSecret, stream};
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

async fn open(script: Script) -> Result<GenerationStream, GenerationStreamError> {
    let channel = backend(script).await;
    let request = GenerateTextRequest::continuation("hello");
    stream(&channel, &request, RequestId::from_raw(7)).await
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
    .await
    .expect("the stream opens");

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
    .await
    .expect("the stream opens");

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
    .await
    .expect("the stream opens");

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
    .await
    .expect("the stream opens");

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
        .await
        .expect("the stream opens");

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
    .await
    .expect("the stream opens");

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
    .await
    .expect("the stream opens");

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
    let mut stream = open(Script::Chunks(vec![event("partial answer", None)]))
        .await
        .expect("the stream opens");

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
    let mut stream = open(Script::Chunks(vec![whole[..15].to_vec()]))
        .await
        .expect("the stream opens");

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
    .await
    .expect("the stream opens");

    let (_, failure) = drain(&mut stream).await;
    assert!(
        matches!(failure, Some(GenerationStreamError::UnexpectedEnd { .. })),
        "expected an unexpected end, got {failure:?}"
    );
}

#[tokio::test]
async fn a_refused_request_never_becomes_a_stream() {
    let opened = open(Script::Status(401, "{}")).await;
    match opened {
        Err(GenerationStreamError::Refused { detail }) => {
            assert!(detail.contains("credential"), "unexpected detail: {detail}");
        }
        Err(other) => panic!("expected a refusal, got {other:?}"),
        Ok(_) => panic!("a refused request must not produce a stream"),
    }
}

#[tokio::test]
async fn nothing_is_yielded_after_the_stream_ends() {
    let mut stream = open(Script::Chunks(vec![
        event("hi", None),
        terminal_with_usage(""),
        b"data: [DONE]\n\n".to_vec(),
    ]))
    .await
    .expect("the stream opens");

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
    .await
    .expect("the stream opens");

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
    .await
    .expect("the stream opens");

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
    .await
    .expect("the stream opens");

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
    .await
    .expect("the stream opens");

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
