//! What this engine build actually does when a generation is abandoned.
//!
//! These tests answer a question rather than guarding a behaviour, and they are
//! ignored by default because each one loads a multi-gigabyte model and then
//! deliberately waits. Run them explicitly:
//!
//! ```text
//! cargo test -p mehoy-backend-llama --test cancellation_characterisation -- --ignored --nocapture --test-threads=1
//! ```
//!
//! # Why this exists before any cancellation is implemented
//!
//! The tempting assumption is that dropping the client stream stops the work. It
//! may not. A client that stops reading, an aborted request, and a request the
//! engine was explicitly told to stop are three different events, and an engine is
//! free to treat all of them as "keep decoding into a buffer nobody will read".
//!
//! If that is what happens, then a runtime that reports a request as cancelled
//! because its stream closed is reporting something false: the accelerator is
//! still busy and the slot is still occupied. Building cancellation on top of that
//! assumption would produce a feature that looks correct in every test that only
//! observes the client.
//!
//! So the observation here is deliberately taken from the engine's own view of its
//! slots rather than from the client's. `/slots` is used only as test
//! instrumentation and must not become part of this runtime's interface: it is one
//! engine's monitoring endpoint, not a portable concept.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Request, StatusCode, header};
use hyper_util::rt::TokioIo;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use mehoy_backend_llama::{
    BackendChannel, LlamaCppBackend, LlamaCppWorkerSpec, ModelDescriptor, RunningBackend, stream,
};
use mehoy_core::id::{IdAllocator, RequestId, WorkerId};
use mehoy_core::inference::{
    GenerateTextRequest, GenerationEvent, GenerationParameters, GenerationStream,
};
use mehoy_core::worker::Deadlines;

fn worker_id() -> WorkerId {
    static IDS: std::sync::LazyLock<IdAllocator> = std::sync::LazyLock::new(IdAllocator::new);
    IDS.worker()
}

/// Finds a container large enough to be a generative model.
///
/// Chosen by size rather than by parsing metadata, because this file is about the
/// engine's behaviour and not about artifact classification, which is tested
/// elsewhere.
fn generative_container() -> Option<PathBuf> {
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))?;
    let mut best: Option<(u64, PathBuf)> = None;
    let mut stack = vec![
        (
            PathBuf::from(&home).join(".lmstudio/.internal/bundled-models"),
            0usize,
        ),
        (PathBuf::from(&home).join(".lmstudio/models"), 0usize),
    ];
    while let Some((dir, depth)) = stack.pop() {
        if depth > 4 {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push((path, depth + 1));
            } else if path.extension().is_some_and(|ext| ext == "gguf") {
                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                // Comfortably larger than an embedding model, comfortably smaller
                // than nothing on this machine.
                if size > 2_000_000_000 && best.as_ref().is_none_or(|(best, _)| size > *best) {
                    best = Some((size, path));
                }
            }
        }
    }
    best.map(|(_, path)| path)
}

/// Stops a backend, reporting rather than hiding a failure to do so.
async fn stop(mut backend: RunningBackend) {
    if let Err(err) = backend.stop().await {
        eprintln!("WARNING: the backend did not stop cleanly: {err}");
    }
}

async fn running_backend() -> Option<RunningBackend> {
    let backend = match LlamaCppBackend::from_env() {
        Ok(backend) => backend,
        Err(err) => {
            eprintln!("SKIPPED: no backend available ({err})");
            return None;
        }
    };
    let Some(model) = generative_container() else {
        eprintln!("SKIPPED: no generative container found on this machine");
        return None;
    };
    eprintln!("using {}", model.display());

    let spec = LlamaCppWorkerSpec {
        descriptor: ModelDescriptor {
            has_chat_template: true,
            ..ModelDescriptor::default()
        },
        deadlines: Deadlines {
            startup: Duration::from_secs(180),
            shutdown: Duration::from_secs(15),
            health: Duration::from_secs(5),
        },
        ..LlamaCppWorkerSpec::new(model)
    };

    Some(
        backend
            .start(worker_id(), &spec)
            .await
            .expect("the backend starts"),
    )
}

/// Sends a plain request to the engine and returns its status and body.
async fn call(channel: &BackendChannel, method: &str, path: &str) -> (StatusCode, String) {
    call_raw(channel.address(), channel.secret().expose(), method, path).await
}

/// The same, addressed by primitives so an observer can outlive a borrow.
async fn call_raw(
    address: std::net::SocketAddr,
    secret: &str,
    method: &str,
    path: &str,
) -> (StatusCode, String) {
    let transport = TcpStream::connect(address).await.expect("connects");
    let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(transport))
        .await
        .expect("handshake");
    let pump = tokio::spawn(async move {
        let _ = connection.await;
    });

    let request = Request::builder()
        .method(method)
        .uri(path)
        .header(header::HOST, address.ip().to_string())
        .header(header::AUTHORIZATION, format!("Bearer {secret}"))
        .body(Full::new(Bytes::new()))
        .expect("builds");

    let response = sender.send_request(request).await.expect("sends");
    let status = response.status();
    let body = response
        .into_body()
        .collect()
        .await
        .expect("reads")
        .to_bytes();
    pump.abort();
    (status, String::from_utf8_lossy(&body).into_owned())
}

/// Whether the engine reports any slot as still working.
async fn any_slot_busy(channel: &BackendChannel) -> Option<bool> {
    any_slot_busy_raw(channel.address(), channel.secret().expose()).await
}

async fn any_slot_busy_raw(address: std::net::SocketAddr, secret: &str) -> Option<bool> {
    let (status, body) = call_raw(address, secret, "GET", "/slots").await;
    if !status.is_success() {
        return None;
    }
    // Deliberately parsed by substring rather than by a typed model. This is one
    // engine's monitoring shape, and giving it a type here would be the first step
    // toward it becoming something this runtime depends on.
    Some(body.contains("\"is_processing\":true") || body.contains("\"is_processing\": true"))
}

// Note on waiting for the engine to become busy: a request is not dispatched to a
// slot the instant it is sent, and dispatch was measured between 68 ms and 375 ms
// depending on prompt size. Experiments therefore poll until the engine reports
// itself working rather than sleeping a guessed interval, which is why that wait
// is written inline alongside the request future each one needs to hold open.

/// Waits for every slot to report idle, returning how long that took.
async fn wait_for_idle(channel: &BackendChannel, budget: Duration) -> Option<Duration> {
    let started = Instant::now();
    while started.elapsed() < budget {
        match any_slot_busy(channel).await {
            Some(false) => return Some(started.elapsed()),
            Some(true) => tokio::time::sleep(Duration::from_millis(200)).await,
            None => return None,
        }
    }
    None
}

/// A generation long enough that it cannot finish while being observed.
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

/// Reads a stream until it has produced real content.
async fn read_until_first_delta(stream: &mut GenerationStream) -> bool {
    while let Some(item) = stream.next().await {
        match item {
            Ok(GenerationEvent::TextDelta { .. }) => return true,
            Ok(_) => {}
            Err(error) => {
                eprintln!("  stream failed before any content: {error}");
                return false;
            }
        }
    }
    false
}

#[tokio::test]
#[ignore = "loads a multi-gigabyte model and deliberately waits"]
async fn what_routes_this_build_exposes() {
    let Some(backend) = running_backend().await else {
        return;
    };
    let channel = backend.channel();

    for (method, path) in [
        ("GET", "/slots"),
        ("GET", "/props"),
        ("POST", "/v1/stream"),
        ("DELETE", "/v1/stream?conv_id=probe"),
        ("DELETE", "/slots/0"),
    ] {
        let (status, body) = call(channel, method, path).await;
        let excerpt: String = body.chars().take(160).collect();
        eprintln!("{method} {path} -> {status}  {excerpt}");
    }

    stop(backend).await;
}

#[tokio::test]
#[ignore = "loads a multi-gigabyte model and deliberately waits"]
async fn a_completed_generation_frees_its_slot() {
    // The control. Without this, an idle slot after cancellation proves nothing,
    // because it might simply be what this endpoint always reports.
    let Some(backend) = running_backend().await else {
        return;
    };
    let channel = backend.channel();

    assert_eq!(
        any_slot_busy(channel).await,
        Some(false),
        "a freshly started backend should be idle"
    );

    let short = GenerateTextRequest::continuation("The opposite of hot is").with_parameters(
        GenerationParameters {
            max_output_tokens: Some(8),
            temperature: Some(0.0),
            seed: Some(1),
            stop: Vec::new(),
        },
    );
    let mut stream = stream(channel, &short, RequestId::from_raw(1))
        .await
        .expect("opens");
    let result = stream.collect().await.expect("completes");
    eprintln!("control generated {:?}", result.text);

    let idle = wait_for_idle(channel, Duration::from_secs(20)).await;
    eprintln!("slot idle after natural completion: {idle:?}");
    assert!(idle.is_some(), "the slot never went idle after completing");

    stop(backend).await;
}

#[tokio::test]
#[ignore = "loads a multi-gigabyte model and deliberately waits"]
async fn dropping_the_stream_may_not_stop_the_work() {
    // The question M1.15 turns on. If this reports that the slot stays busy, then
    // the runtime cannot honestly call a dropped stream a cancellation.
    let Some(backend) = running_backend().await else {
        return;
    };
    let channel = backend.channel();

    let mut generation = stream(channel, &long_request(), RequestId::from_raw(2))
        .await
        .expect("opens");
    assert!(
        read_until_first_delta(&mut generation).await,
        "the model produced no content to interrupt"
    );
    assert_eq!(
        any_slot_busy(channel).await,
        Some(true),
        "the engine should be working while generating"
    );

    let dropped_at = Instant::now();
    drop(generation);
    eprintln!("stream dropped");

    match wait_for_idle(channel, Duration::from_secs(30)).await {
        Some(elapsed) => eprintln!(
            "OBSERVED: work stopped {:?} after the stream was dropped",
            elapsed
        ),
        None => eprintln!(
            "OBSERVED: still working {:?} after the stream was dropped; \
             a dropped stream does NOT stop this engine",
            dropped_at.elapsed()
        ),
    }

    stop(backend).await;
}

#[tokio::test]
#[ignore = "loads a multi-gigabyte model and deliberately waits"]
async fn cutting_the_connection_abruptly_may_not_stop_the_work() {
    // A harsher version of the same question: not a polite drop through hyper, but
    // the socket disappearing, which is what a crashed client looks like.
    let Some(backend) = running_backend().await else {
        return;
    };
    let channel = backend.channel();

    let body = "{\"prompt\":\"Write a very long numbered list of every English word you know.\",\
\"n_predict\":4096,\"temperature\":0,\"stream\":true}";
    let request = format!(
        "POST /v1/completions HTTP/1.1\r\nHost: {}\r\nAuthorization: Bearer {}\r\n\
Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        channel.host(),
        channel.secret().expose(),
        body.len()
    );

    let mut socket = TcpStream::connect(channel.address())
        .await
        .expect("connects");
    socket
        .write_all(request.as_bytes())
        .await
        .expect("sends the request");

    // Read until the engine is demonstrably producing tokens.
    let mut scratch = [0u8; 4096];
    let mut seen = 0usize;
    while seen < 200 {
        match socket.read(&mut scratch).await {
            Ok(0) | Err(_) => break,
            Ok(read) => seen += read,
        }
    }
    assert!(seen > 0, "the engine sent nothing to interrupt");
    assert_eq!(any_slot_busy(channel).await, Some(true));

    let cut_at = Instant::now();
    drop(socket);
    eprintln!("connection cut after {seen} bytes");

    match wait_for_idle(channel, Duration::from_secs(30)).await {
        Some(elapsed) => eprintln!("OBSERVED: work stopped {elapsed:?} after the socket closed"),
        None => eprintln!(
            "OBSERVED: still working {:?} after the socket closed; \
             an abrupt disconnect does NOT stop this engine",
            cut_at.elapsed()
        ),
    }

    stop(backend).await;
}

#[tokio::test]
#[ignore = "loads a multi-gigabyte model and deliberately waits"]
async fn how_long_the_long_request_runs_when_nobody_interrupts_it() {
    // Without this, the interruption experiments prove nothing. If the long
    // request happens to finish in a couple of hundred milliseconds, then a slot
    // going idle shortly after a disconnect is ordinary completion wearing the
    // costume of cancellation.
    let Some(backend) = running_backend().await else {
        return;
    };
    let channel = backend.channel();

    let started = Instant::now();
    let mut generation = stream(channel, &long_request(), RequestId::from_raw(9))
        .await
        .expect("opens");

    let mut deltas = 0usize;
    let mut characters = 0usize;
    let mut first_delta = None;
    while let Some(item) = generation.next().await {
        match item {
            Ok(GenerationEvent::TextDelta { text, .. }) => {
                if first_delta.is_none() {
                    first_delta = Some(started.elapsed());
                }
                deltas += 1;
                characters += text.len();
            }
            Ok(GenerationEvent::Completed { summary, .. }) => {
                eprintln!(
                    "BASELINE: completed naturally, finish {}",
                    summary.finish_reason
                );
            }
            Ok(GenerationEvent::Started { .. }) => {}
            Err(error) => panic!("the baseline generation failed: {error}"),
        }
    }

    eprintln!(
        "BASELINE: {deltas} deltas, {characters} characters, first delta at {first_delta:?}, \
         total {:?}",
        started.elapsed()
    );

    stop(backend).await;
}

/// A prompt long enough that reading it is itself measurable work.
///
/// Cancelling while the engine is still consuming the prompt is a different engine
/// state from cancelling while it is emitting tokens, and upstream has had defects
/// specific to the first.
fn long_prompt_request() -> GenerateTextRequest {
    let filler = "The quick brown fox jumps over the lazy dog. ".repeat(4000);
    GenerateTextRequest::continuation(format!("{filler}\n\nSummarise the text above at length."))
        .with_parameters(GenerationParameters {
            max_output_tokens: Some(4096),
            temperature: Some(0.0),
            seed: Some(1),
            stop: Vec::new(),
        })
}

#[tokio::test]
#[ignore = "loads a multi-gigabyte model and deliberately waits"]
async fn how_long_a_long_prompt_takes_before_its_first_token() {
    // Establishes that there is a prefill window worth cancelling inside.
    let Some(backend) = running_backend().await else {
        return;
    };
    let channel = backend.channel();

    let started = Instant::now();
    let mut generation = stream(channel, &long_prompt_request(), RequestId::from_raw(10))
        .await
        .expect("opens");
    let produced = read_until_first_delta(&mut generation).await;
    eprintln!(
        "BASELINE: first delta after {:?} (produced: {produced})",
        started.elapsed()
    );
    drop(generation);

    stop(backend).await;
}

#[tokio::test]
#[ignore = "loads a multi-gigabyte model and deliberately waits"]
async fn abandoning_a_request_during_prefill() {
    // For this engine the response does not begin until the prompt has been read,
    // so there is no stream to drop while prefill is happening. The only thing that
    // exists during that window is the in-flight request, and abandoning it is the
    // only cancellation available. This is the case upstream has had defects in.
    let Some(backend) = running_backend().await else {
        return;
    };
    let channel = backend.channel();

    let request = long_prompt_request();
    // Boxed rather than stack-pinned on purpose: pin! keeps the future in a hidden
    // local that lives to the end of the scope, so dropping the handle would not
    // actually abandon the request, and this test would prove nothing.
    let mut pending = Box::pin(stream(channel, &request, RequestId::from_raw(11)));

    // Poll for the engine picking the work up while the request is still open.
    let started = Instant::now();
    let mut became_busy = None;
    let opened = loop {
        tokio::select! {
            outcome = &mut pending => break Some(outcome),
            () = tokio::time::sleep(Duration::from_millis(100)) => {
                if became_busy.is_none() && any_slot_busy(channel).await == Some(true) {
                    became_busy = Some(started.elapsed());
                    break None;
                }
                if started.elapsed() > Duration::from_secs(30) {
                    break None;
                }
            }
        }
    };

    assert!(
        opened.is_none(),
        "the response arrived before prefill could be observed, so nothing was interrupted"
    );
    let became_busy = became_busy.expect("the engine never picked the request up");
    eprintln!("prefill observably underway after {became_busy:?}");

    let abandoned_at = Instant::now();
    drop(pending);
    eprintln!("request abandoned mid-prefill, before any response");

    match wait_for_idle(channel, Duration::from_secs(60)).await {
        Some(elapsed) => {
            eprintln!("OBSERVED: work stopped {elapsed:?} after abandoning the request in prefill")
        }
        None => eprintln!(
            "OBSERVED: still working {:?} after abandoning the request in prefill;              this engine does NOT observe cancellation before the first token",
            abandoned_at.elapsed()
        ),
    }

    stop(backend).await;
}

/// Sends a JSON body and returns the raw response, bypassing this crate's adapter.
///
/// Used where the adapter itself must not be part of the observation.
async fn post_json(channel: &BackendChannel, path: &str, body: String) -> (StatusCode, String) {
    let transport = TcpStream::connect(channel.address())
        .await
        .expect("connects");
    let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(transport))
        .await
        .expect("handshake");
    let pump = tokio::spawn(async move {
        let _ = connection.await;
    });

    let request = Request::builder()
        .method("POST")
        .uri(path)
        .header(header::HOST, channel.host())
        .header(header::CONTENT_TYPE, "application/json")
        .header(
            header::AUTHORIZATION,
            format!("Bearer {}", channel.secret().expose()),
        )
        .body(Full::new(Bytes::from(body)))
        .expect("builds");

    let response = sender.send_request(request).await.expect("sends");
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("reads")
        .to_bytes();
    pump.abort();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

#[tokio::test]
#[ignore = "loads a multi-gigabyte model and deliberately waits"]
async fn diagnose_the_long_prompt() {
    let Some(backend) = running_backend().await else {
        return;
    };
    let channel = backend.channel();

    // Watch the engine's own view for the whole request rather than sampling once.
    let watched = channel.address();
    let secret = channel.secret().expose().to_owned();
    let timeline = tokio::spawn(async move {
        let started = Instant::now();
        let mut seen: Vec<(u128, bool)> = Vec::new();
        for _ in 0..160 {
            if let Some(busy) = any_slot_busy_raw(watched, &secret).await {
                let at = started.elapsed().as_millis();
                if seen.last().map(|(_, last)| *last) != Some(busy) {
                    seen.push((at, busy));
                }
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        seen
    });

    let filler = "The quick brown fox jumps over the lazy dog. ".repeat(4000);
    let prompt = format!("{filler}\n\nSummarise the text above at length.");
    eprintln!("DIAGNOSTIC: prompt is {} characters", prompt.len());

    let body = serde_json::json!({
        "prompt": prompt,
        "n_predict": 64,
        "temperature": 0,
        "stream": false,
    })
    .to_string();

    let started = Instant::now();
    let (status, raw) = post_json(channel, "/v1/completions", body).await;
    eprintln!("DIAGNOSTIC: status {status} after {:?}", started.elapsed());
    eprintln!(
        "DIAGNOSTIC: raw response {}",
        raw.chars().take(700).collect::<String>()
    );

    match timeline.await {
        Ok(seen) => eprintln!("DIAGNOSTIC: slot busy transitions (ms, busy) {seen:?}"),
        Err(err) => eprintln!("DIAGNOSTIC: the observer did not finish: {err}"),
    }

    eprintln!(
        "DIAGNOSTIC: engine said:\n{}",
        backend
            .output()
            .lines()
            .rev()
            .take(25)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n")
    );

    stop(backend).await;
}

/// A prompt with a real prefill window that is also followed by real generation.
///
/// The earlier long prompt provokes an immediate end-of-sequence token, which
/// makes "stopped at the end of prefill" and "ran the whole request" the same
/// observation. This one generates for several seconds after prefill, so the two
/// can be told apart.
fn medium_prompt_generative_request(marker: &str) -> GenerateTextRequest {
    // The marker makes each prompt distinct. This engine reuses cached prompt
    // prefixes between requests, so repeating a prompt skips prefill entirely, and
    // an experiment that depends on prefill actually happening would silently
    // measure the token-generation path instead.
    let filler = format!("The {marker} brown fox jumps over the lazy dog. ").repeat(1500);
    GenerateTextRequest::continuation(format!(
        "{filler}\n\nWrite a very long numbered list of every English word you know, \
         one per line, starting at 1 and continuing without stopping."
    ))
    .with_parameters(GenerationParameters {
        max_output_tokens: Some(4096),
        temperature: Some(0.0),
        seed: Some(1),
        stop: Vec::new(),
    })
}

#[tokio::test]
#[ignore = "loads a multi-gigabyte model and deliberately waits"]
async fn how_late_a_prefill_cancellation_is_actually_honoured() {
    // Bounds the worst case. If abandoning during prefill costs the whole request
    // rather than the rest of prefill, then a cancelled request holds its slot for
    // as long as an uncancelled one, and cancellation buys nothing at all here.
    let Some(backend) = running_backend().await else {
        return;
    };
    let channel = backend.channel();

    // Baseline: the same request, undisturbed.
    let baseline_start = Instant::now();
    let mut undisturbed = stream(
        channel,
        &medium_prompt_generative_request("quick"),
        RequestId::from_raw(20),
    )
    .await
    .expect("opens");
    let opened_after = baseline_start.elapsed();
    let mut deltas = 0usize;
    while let Some(item) = undisturbed.next().await {
        if let Ok(GenerationEvent::TextDelta { .. }) = item {
            deltas += 1;
        }
    }
    let baseline_total = baseline_start.elapsed();
    assert!(deltas > 0, "the comparison needs a request that generates");
    eprintln!(
        "BASELINE: prefill ended (response began) at {opened_after:?}, \
         {deltas} deltas, whole request {baseline_total:?}"
    );
    assert!(
        wait_for_idle(channel, Duration::from_secs(30))
            .await
            .is_some(),
        "the slot never settled before the second half of the test"
    );

    // Now the same request, abandoned as soon as the engine picks it up.
    let request = medium_prompt_generative_request("nimble");
    let mut pending = Box::pin(stream(channel, &request, RequestId::from_raw(21)));
    let started = Instant::now();
    let mut became_busy = None;
    let opened = loop {
        tokio::select! {
            outcome = &mut pending => break Some(outcome),
            () = tokio::time::sleep(Duration::from_millis(50)) => {
                if any_slot_busy(channel).await == Some(true) {
                    became_busy = Some(started.elapsed());
                    break None;
                }
                if started.elapsed() > Duration::from_secs(30) {
                    break None;
                }
            }
        }
    };
    assert!(
        opened.is_none(),
        "prefill finished before it could be interrupted"
    );
    eprintln!("prefill observably underway after {became_busy:?}");

    let abandoned_at = Instant::now();
    drop(pending);

    match wait_for_idle(channel, Duration::from_secs(120)).await {
        Some(elapsed) => {
            let held = abandoned_at.elapsed();
            eprintln!(
                "OBSERVED: slot freed {elapsed:?} after abandoning during prefill. \
                 Prefill alone was {opened_after:?}; the whole request was {baseline_total:?}."
            );
            if held < baseline_total.mul_f32(0.6) {
                eprintln!(
                    "INTERPRETATION: the cancellation was honoured at the end of prefill, \
                     so the cost is bounded by prefill rather than by the whole request"
                );
            } else {
                eprintln!(
                    "INTERPRETATION: the cancelled request held its slot for about as long \
                     as an undisturbed one, so cancellation during prefill buys nothing"
                );
            }
        }
        None => eprintln!("OBSERVED: the slot never freed within 120s"),
    }

    stop(backend).await;
}
