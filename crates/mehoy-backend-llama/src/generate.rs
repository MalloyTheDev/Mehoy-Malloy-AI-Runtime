//! The text generation adapter.
//!
//! Translates the runtime's own request and result types to and from this engine's
//! wire format. That format stays below the backend boundary, so nothing above here
//! knows which engine is running or what it calls its controls.
//!
//! # Measured behaviour, llama.cpp build 9010
//!
//! A completion request returns a `choices` array whose first entry carries the
//! generated `text` and a `finish_reason` of `length` or `stop`, alongside a `usage`
//! object with `prompt_tokens` and `completion_tokens`.
//!
//! A streaming request returns `text/event-stream`. Each event carries the same
//! `choices` shape with an incremental `text` and a null `finish_reason`, the last
//! content event carries the real `finish_reason` and a `usage` object, and the
//! engine then sends `data: [DONE]`. Streaming and non-streaming were observed to
//! produce identical text and identical usage counts for the same prompt at
//! temperature zero with a fixed seed.
//!
//! None of that is assumed anywhere below. The terminal event is whatever reports
//! a finish reason, `usage` is optional, and the `[DONE]` marker is accepted but
//! not required, because a second engine will differ in exactly these details.
//!
//! # Parameters
//!
//! Only the portable set from ADR-0008 is sent, and only when the caller set it.
//! An unset parameter is omitted entirely rather than filled with a default, so the
//! engine applies its own and the caller's silence stays distinguishable from a
//! value that happens to match today's default.

use std::fmt;
use std::time::Duration;

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Request, StatusCode, header};
use hyper_util::rt::TokioIo;
use serde::{Deserialize, Serialize};
use tokio::net::TcpStream;

use mehoy_core::cancel::{
    CancellationCause, CancellationStrategy, CancellationToken, RequestBudget, RequestHandle,
};
use mehoy_core::inference::{
    FinishReason, GenerateTextRequest, GenerationResult, GenerationSink, GenerationStream,
    GenerationStreamError, GenerationSummary, GenerationUsage, generation_stream,
};

use crate::channel::BackendChannel;
use crate::sse::SseDecoder;

/// The engine's completion path.
pub const COMPLETIONS_PATH: &str = "/v1/completions";

/// How long one generation request may take.
///
/// Generation is unbounded work in a way an embedding is not, so this is generous.
/// It is a transport ceiling, not a substitute for a caller's own token budget.
const REQUEST_BUDGET: std::time::Duration = std::time::Duration::from_secs(300);

/// Why a generation request failed.
#[derive(Debug)]
pub enum GenerateError {
    /// The backend was not started in a mode that generates text.
    NotSupported { detail: String },
    /// The backend refused the credential.
    Unauthorised { status: u16 },
    /// The backend answered with a failure status.
    Rejected { status: u16, detail: String },
    /// The backend's response could not be understood.
    MalformedResponse { detail: String },
    /// The request could not be delivered.
    Transport { detail: String },
}

impl fmt::Display for GenerateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotSupported { detail } => {
                write!(f, "the backend does not generate text as started: {detail}")
            }
            Self::Unauthorised { status } => {
                write!(f, "the backend refused the runtime's credential ({status})")
            }
            Self::Rejected { status, detail } => {
                write!(f, "the backend rejected the request ({status}): {detail}")
            }
            Self::MalformedResponse { detail } => {
                write!(
                    f,
                    "the backend's generation response was not usable: {detail}"
                )
            }
            Self::Transport { detail } => write!(f, "the generation request failed: {detail}"),
        }
    }
}

impl std::error::Error for GenerateError {}

/// The wire request.
///
/// Every optional field is skipped when unset, which is how ADR-0008's distinction
/// between omission and default is actually delivered to the engine.
#[derive(Serialize)]
struct WireRequest<'a> {
    prompt: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    n_predict: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    #[serde(skip_serializing_if = "<[String]>::is_empty")]
    stop: &'a [String],
    /// Omitted when false, so a non-streaming request is byte-identical to
    /// what it was before streaming existed.
    #[serde(skip_serializing_if = "is_unset")]
    stream: bool,
}

fn is_unset(streaming: &bool) -> bool {
    !*streaming
}

#[derive(Deserialize)]
struct WireResponse {
    #[serde(default)]
    choices: Vec<WireChoice>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
struct WireChoice {
    #[serde(default)]
    text: String,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct WireUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
}

#[derive(Deserialize)]
struct WireError {
    #[serde(default)]
    error: WireErrorBody,
}

#[derive(Deserialize, Default)]
struct WireErrorBody {
    #[serde(default)]
    message: String,
    #[serde(default, rename = "type")]
    kind: String,
}

/// Maps the engine's finish reason onto the runtime's.
///
/// An unrecognised reason is carried through rather than forced into a known one,
/// because reporting an unfamiliar stop as an ordinary one loses exactly the
/// information a caller would need.
fn finish_reason_for(reported: Option<&str>) -> FinishReason {
    match reported {
        Some("stop") => FinishReason::Stop,
        Some("length") => FinishReason::Length,
        Some(other) => FinishReason::Other(other.to_owned()),
        None => FinishReason::Other("unreported".to_owned()),
    }
}

/// Sends a generation request to a running backend.
///
/// # Errors
///
/// Returns [`GenerateError`] describing which layer refused.
pub async fn generate(
    channel: &BackendChannel,
    request: &GenerateTextRequest,
) -> Result<GenerationResult, GenerateError> {
    let parameters = &request.parameters;
    let body = serde_json::to_vec(&WireRequest {
        prompt: request.input.as_str(),
        n_predict: parameters.max_output_tokens,
        temperature: parameters.temperature,
        seed: parameters.seed,
        stop: &parameters.stop,
        stream: false,
    })
    .map_err(|err| GenerateError::MalformedResponse {
        detail: format!("cannot encode the request: {err}"),
    })?;

    let (status, bytes) = tokio::time::timeout(REQUEST_BUDGET, post(channel, body))
        .await
        .map_err(|_| GenerateError::Transport {
            detail: format!("no response within {}s", REQUEST_BUDGET.as_secs()),
        })??;

    if !status.is_success() {
        return Err(classify(status, &bytes));
    }

    let wire: WireResponse =
        serde_json::from_slice(&bytes).map_err(|err| GenerateError::MalformedResponse {
            detail: err.to_string(),
        })?;

    let choice =
        wire.choices
            .into_iter()
            .next()
            .ok_or_else(|| GenerateError::MalformedResponse {
                detail: "the response contained no choices".to_owned(),
            })?;

    Ok(GenerationResult {
        text: choice.text,
        finish_reason: finish_reason_for(choice.finish_reason.as_deref()),
        usage: wire.usage.map(|usage| GenerationUsage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
        }),
    })
}

fn classify(status: StatusCode, bytes: &[u8]) -> GenerateError {
    let parsed: Option<WireError> = serde_json::from_slice(bytes).ok();
    let (message, kind) = parsed.map_or_else(
        || (String::from_utf8_lossy(bytes).into_owned(), String::new()),
        |body| (body.error.message, body.error.kind),
    );

    match status {
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => GenerateError::Unauthorised {
            status: status.as_u16(),
        },
        StatusCode::NOT_IMPLEMENTED => GenerateError::NotSupported { detail: message },
        _ if kind == "not_supported_error" => GenerateError::NotSupported { detail: message },
        _ => GenerateError::Rejected {
            status: status.as_u16(),
            detail: message,
        },
    }
}

async fn post(
    channel: &BackendChannel,
    body: Vec<u8>,
) -> Result<(StatusCode, Bytes), GenerateError> {
    let stream =
        TcpStream::connect(channel.address())
            .await
            .map_err(|err| GenerateError::Transport {
                detail: err.to_string(),
            })?;

    let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
        .await
        .map_err(|err| GenerateError::Transport {
            detail: err.to_string(),
        })?;
    let pump = tokio::spawn(async move {
        let _ = connection.await;
    });

    let request = Request::builder()
        .method("POST")
        .uri(COMPLETIONS_PATH)
        .header(header::HOST, channel.host())
        .header(header::CONTENT_TYPE, "application/json")
        .header(
            header::AUTHORIZATION,
            format!("Bearer {}", channel.secret().expose()),
        )
        .body(Full::new(Bytes::from(body)))
        .map_err(|err| GenerateError::Transport {
            detail: err.to_string(),
        })?;

    let response = sender
        .send_request(request)
        .await
        .map_err(|err| GenerateError::Transport {
            detail: err.to_string(),
        })?;
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .map_err(|err| GenerateError::Transport {
            detail: err.to_string(),
        })?
        .to_bytes();
    pump.abort();

    Ok((status, bytes))
}

/// The marker this engine sends after its final event.
const DONE_MARKER: &str = "[DONE]";

/// How this engine can be made to stop work.
///
/// Build 9010 exposes no cancellation route: `DELETE /v1/stream` and every
/// neighbouring path answer 404. Closing the request's transport is the only
/// signal it responds to, and it does respond to it, which was measured rather
/// than assumed. See `docs/research/llama-cpp-cancellation-b9010.md`.
///
/// This is why every generation gets its own connection rather than sharing a
/// pooled one. If closing a connection is how a request is stopped, then a
/// connection shared between requests cannot be closed to stop one of them.
#[must_use]
pub const fn cancellation_strategy() -> CancellationStrategy {
    CancellationStrategy::ConnectionAbort
}

/// Everything the execution task needs, owned.
///
/// The task outlives the call that started it, so it cannot borrow the channel.
struct Job {
    address: std::net::SocketAddr,
    host: String,
    secret: String,
    body: Result<Vec<u8>, String>,
}

/// Starts a streaming generation and returns immediately.
///
/// Returns as soon as the runtime owns the request, which is what
/// [`mehoy_core::inference::GenerationEvent::Started`] now claims. Connecting, sending, waiting out
/// input processing, and reading the response all happen behind the returned
/// stream.
///
/// This deliberately does not wait for the backend to answer. On the measured
/// engine a large prompt delays the first response header by over ten seconds,
/// and a call that waited for it would offer no way to stop the request during
/// precisely the period when stopping it matters most. ADR-0009 records the
/// reasoning.
///
/// Cancelling `cancel` closes this request's transport and ends the stream with
/// [`mehoy_core::inference::GenerationEvent::Cancelled`] once execution has actually
/// stopped. Because
/// the connection belongs to this request alone, that cannot disturb another
/// request or the worker.
pub fn start_generation(
    channel: &BackendChannel,
    request: &GenerateTextRequest,
    handle: RequestHandle,
    budget: RequestBudget,
) -> GenerationStream {
    let parameters = &request.parameters;
    let body = serde_json::to_vec(&WireRequest {
        prompt: request.input.as_str(),
        n_predict: parameters.max_output_tokens,
        temperature: parameters.temperature,
        seed: parameters.seed,
        stop: &parameters.stop,
        stream: true,
    })
    .map_err(|err| format!("cannot encode the request: {err}"));

    let job = Job {
        address: channel.address(),
        host: channel.host(),
        secret: channel.secret().expose().to_owned(),
        body,
    };

    let cancel = handle.token().clone();
    let (sink, stream) = generation_stream(handle);
    tokio::spawn(execute(job, sink, cancel, budget));
    stream
}

/// Runs `work` unless cancellation arrives first or the budget runs out.
///
/// Biased towards cancellation, so a request cancelled before its work has made
/// progress stops deterministically rather than depending on which future the
/// scheduler happened to poll.
///
/// The idle budget covers waiting for the backend to say anything at all, not
/// only the gaps between things it has said. On the measured engine the longest
/// silence in a request is the one before the first response header, while the
/// prompt is being read, so a budget that started only once the response had begun
/// would leave the longest wait unbounded and a backend that never answered at all
/// would hang forever.
async fn within<F>(
    cancel: &CancellationToken,
    budget: Duration,
    work: F,
) -> Result<F::Output, CancellationCause>
where
    F: std::future::Future,
{
    tokio::select! {
        biased;
        cause = cancel.cancelled() => Err(cause),
        outcome = tokio::time::timeout(budget, work) => match outcome {
            Ok(value) => Ok(value),
            Err(_) => {
                // Silence becomes a cancellation with its own cause rather than a
                // separate stopping mechanism.
                cancel.cancel(CancellationCause::StreamIdleTimeout);
                Err(cancel
                    .cause()
                    .unwrap_or(CancellationCause::StreamIdleTimeout))
            }
        },
    }
}

/// Carries out one generation, ending the sink exactly once.
async fn execute(job: Job, sink: GenerationSink, cancel: CancellationToken, budget: RequestBudget) {
    let body = match job.body {
        Ok(body) => body,
        Err(detail) => {
            sink.fail(GenerationStreamError::Transport { detail }).await;
            return;
        }
    };

    // A connection per request, never pooled, because closing it is how this
    // backend is told to stop.
    let transport = match within(&cancel, budget.stream_idle, TcpStream::connect(job.address)).await
    {
        Err(cause) => return sink.cancel(cause).await,
        Ok(Err(err)) => {
            return sink
                .fail(GenerationStreamError::Transport {
                    detail: err.to_string(),
                })
                .await;
        }
        Ok(Ok(transport)) => transport,
    };

    let handshake = hyper::client::conn::http1::handshake(TokioIo::new(transport));
    let (mut sender, connection) = match within(&cancel, budget.stream_idle, handshake).await {
        Err(cause) => return sink.cancel(cause).await,
        Ok(Err(err)) => {
            return sink
                .fail(GenerationStreamError::Transport {
                    detail: err.to_string(),
                })
                .await;
        }
        Ok(Ok(parts)) => parts,
    };
    let pump = tokio::spawn(async move {
        let _ = connection.await;
    });

    let request = Request::builder()
        .method("POST")
        .uri(COMPLETIONS_PATH)
        .header(header::HOST, job.host)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ACCEPT, "text/event-stream")
        .header(header::AUTHORIZATION, format!("Bearer {}", job.secret))
        .body(Full::new(Bytes::from(body)));
    let request = match request {
        Ok(request) => request,
        Err(err) => {
            pump.abort();
            return sink
                .fail(GenerationStreamError::Transport {
                    detail: err.to_string(),
                })
                .await;
        }
    };

    // The long wait. For a large prompt the backend reads all of it before
    // answering, so this is where a cancellation most often lands.
    let response = match within(&cancel, budget.stream_idle, sender.send_request(request)).await {
        Err(cause) => {
            pump.abort();
            return sink.cancel(cause).await;
        }
        Ok(Err(err)) => {
            pump.abort();
            return sink
                .fail(GenerationStreamError::Transport {
                    detail: err.to_string(),
                })
                .await;
        }
        Ok(Ok(response)) => response,
    };
    sink.handle().mark_running();

    let status = response.status();
    if !status.is_success() {
        let collected = within(&cancel, budget.stream_idle, response.into_body().collect()).await;
        pump.abort();
        return match collected {
            Err(cause) => sink.cancel(cause).await,
            Ok(outcome) => {
                let detail = outcome.map_or_else(
                    |err| err.to_string(),
                    |body| classify(status, &body.to_bytes()).to_string(),
                );
                sink.fail(GenerationStreamError::Refused { detail }).await
            }
        };
    }

    read(response.into_body(), sink, cancel, budget, pump).await;
}

/// Drives the response body until the generation ends.
///
/// Every exit path ends the sink exactly once, so a consumer always learns an
/// outcome rather than watching the stream simply stop. The one exception is a
/// consumer that has gone away, which has nothing left to be told.
async fn read(
    mut body: hyper::body::Incoming,
    mut sink: GenerationSink,
    cancel: CancellationToken,
    budget: RequestBudget,
    pump: tokio::task::JoinHandle<()>,
) {
    let mut decoder = SseDecoder::default();
    let mut terminal: Option<GenerationSummary> = None;

    let outcome: Result<EndOfBody, Interruption> = 'body: loop {
        let waited = tokio::select! {
            biased;
            cause = cancel.cancelled() => break 'body Err(Interruption::Cancelled(cause)),
            // Noticed here rather than only when the next send fails, because a
            // backend that has gone quiet might not give us another send for
            // minutes, and the request would look alive that whole time.
            () = sink.closed() => {
                break 'body Err(Interruption::Cancelled(CancellationCause::ConsumerGone));
            }
            frame = tokio::time::timeout(budget.stream_idle, body.frame()) => frame,
        };

        let frame = match waited {
            // Silence is indistinguishable from a hang, so it becomes a
            // cancellation with its own cause rather than a separate mechanism.
            Err(_) => {
                cancel.cancel(CancellationCause::StreamIdleTimeout);
                break 'body Err(Interruption::Cancelled(
                    cancel
                        .cause()
                        .unwrap_or(CancellationCause::StreamIdleTimeout),
                ));
            }
            Ok(None) => break 'body Ok(EndOfBody::Closed),
            Ok(Some(Err(err))) => {
                break 'body Err(Interruption::Failed(GenerationStreamError::Transport {
                    detail: err.to_string(),
                }));
            }
            Ok(Some(Ok(frame))) => frame,
        };

        let Some(chunk) = frame.data_ref() else {
            continue;
        };

        let payloads = match decoder.push(chunk) {
            Ok(payloads) => payloads,
            Err(detail) => {
                break 'body Err(Interruption::Failed(
                    GenerationStreamError::MalformedEvent { detail },
                ));
            }
        };

        for payload in payloads {
            if payload.trim() == DONE_MARKER {
                break 'body Ok(EndOfBody::Marked);
            }

            let parsed: WireResponse = match serde_json::from_str(&payload) {
                Ok(parsed) => parsed,
                Err(err) => {
                    break 'body Err(Interruption::Failed(
                        GenerationStreamError::MalformedEvent {
                            detail: describe(&err, &payload),
                        },
                    ));
                }
            };

            let Some(choice) = parsed.choices.into_iter().next() else {
                continue;
            };

            // Recorded before the text is handed over, so a terminal event that
            // also carries content still reports why generation stopped.
            if let Some(reported) = choice.finish_reason.as_deref() {
                terminal = Some(GenerationSummary {
                    finish_reason: finish_reason_for(Some(reported)),
                    usage: parsed.usage.map(|usage| GenerationUsage {
                        input_tokens: usage.prompt_tokens,
                        output_tokens: usage.completion_tokens,
                    }),
                });
            }

            if !sink.delta(choice.text).await {
                // Nothing is reading any more. Returning here without recording an
                // outcome would leave the request looking alive forever while its
                // connection closed underneath, so the runtime would report a
                // request as running that the backend had already stopped.
                break 'body Err(Interruption::Cancelled(CancellationCause::ConsumerGone));
            }
        }
    };

    // Dropping the body and stopping the connection closes this request's socket,
    // which is what actually tells the backend to stop.
    drop(body);
    pump.abort();

    match outcome {
        Err(Interruption::Cancelled(cause)) => sink.cancel(cause).await,
        Err(Interruption::Failed(error)) => sink.fail(error).await,
        Ok(end) => match terminal {
            Some(summary) => sink.complete(summary).await,
            None => {
                let detail = match end {
                    EndOfBody::Marked => {
                        "the backend ended the stream without reporting why generation stopped"
                            .to_owned()
                    }
                    EndOfBody::Closed if decoder.has_partial_event() => {
                        "the backend closed the connection part way through an event".to_owned()
                    }
                    EndOfBody::Closed => {
                        "the backend closed the connection before completing".to_owned()
                    }
                };
                sink.fail(GenerationStreamError::UnexpectedEnd { detail })
                    .await;
            }
        },
    }
}

/// Why reading stopped before the response ended on its own.
enum Interruption {
    Cancelled(CancellationCause),
    Failed(GenerationStreamError),
}

/// How the response body stopped producing events.
enum EndOfBody {
    /// The engine sent its end-of-stream marker.
    Marked,
    /// The body ended without one.
    Closed,
}

/// Describes an unparseable event without quoting all of it.
///
/// The payload is backend output of unbounded size, and the useful part for
/// diagnosis is its beginning.
fn describe(error: &serde_json::Error, payload: &str) -> String {
    const SHOWN: usize = 120;
    let mut excerpt: String = payload.chars().take(SHOWN).collect();
    if excerpt.len() < payload.len() {
        excerpt.push_str("...");
    }
    format!("{error}, in: {excerpt}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use mehoy_core::inference::GenerationParameters;

    fn encode(parameters: GenerationParameters) -> serde_json::Value {
        let request = GenerateTextRequest::continuation("hello").with_parameters(parameters);
        let wire = WireRequest {
            prompt: request.input.as_str(),
            n_predict: request.parameters.max_output_tokens,
            temperature: request.parameters.temperature,
            seed: request.parameters.seed,
            stop: &request.parameters.stop,
            stream: false,
        };
        serde_json::to_value(&wire).expect("encodes")
    }

    #[test]
    fn unset_parameters_are_omitted_rather_than_defaulted() {
        // ADR-0008 turns on this. Sending a value the caller did not choose would
        // freeze one engine's current default into the runtime's behaviour.
        let encoded = encode(GenerationParameters::default());
        let object = encoded.as_object().expect("an object");
        assert!(object.contains_key("prompt"));
        for omitted in ["n_predict", "temperature", "seed", "stop"] {
            assert!(
                !object.contains_key(omitted),
                "{omitted} should have been omitted, got {encoded}"
            );
        }
    }

    #[test]
    fn set_parameters_are_sent() {
        let encoded = encode(GenerationParameters {
            max_output_tokens: Some(16),
            temperature: Some(0.0),
            seed: Some(42),
            stop: vec!["\n\n".to_owned()],
        });
        assert_eq!(encoded["n_predict"], 16);
        assert_eq!(encoded["temperature"], 0.0);
        assert_eq!(encoded["seed"], 42);
        assert_eq!(encoded["stop"][0], "\n\n");
    }

    #[test]
    fn a_zero_temperature_is_sent_rather_than_treated_as_unset() {
        // Zero is a meaningful request for the most deterministic sampling
        // available, and is not the same as having no opinion.
        let encoded = encode(GenerationParameters {
            temperature: Some(0.0),
            ..GenerationParameters::default()
        });
        assert!(
            encoded
                .as_object()
                .expect("object")
                .contains_key("temperature"),
            "an explicit zero must reach the backend"
        );
    }

    #[test]
    fn the_measured_response_shape_parses() {
        // Captured from build 9010.
        let body = r#"{"choices":[{"text":" cold.","index":0,"logprobs":null,"finish_reason":"length"}],"usage":{"completion_tokens":8,"prompt_tokens":13,"total_tokens":21}}"#;
        let wire: WireResponse = serde_json::from_str(body).expect("parses");
        assert_eq!(wire.choices[0].text, " cold.");
        assert_eq!(wire.choices[0].finish_reason.as_deref(), Some("length"));
        let usage = wire.usage.expect("usage present");
        assert_eq!(usage.completion_tokens, 8);
        assert_eq!(usage.prompt_tokens, 13);
    }

    #[test]
    fn finish_reasons_are_normalised_without_losing_unfamiliar_ones() {
        assert_eq!(finish_reason_for(Some("stop")), FinishReason::Stop);
        assert_eq!(finish_reason_for(Some("length")), FinishReason::Length);
        assert_eq!(
            finish_reason_for(Some("tool_calls")),
            FinishReason::Other("tool_calls".to_owned())
        );
        assert_eq!(
            finish_reason_for(None),
            FinishReason::Other("unreported".to_owned())
        );
    }

    #[test]
    fn a_refused_credential_stays_its_own_error() {
        match classify(StatusCode::UNAUTHORIZED, b"{}") {
            GenerateError::Unauthorised { status } => assert_eq!(status, 401),
            other => panic!("expected Unauthorised, got {other}"),
        }
    }

    #[test]
    fn a_response_with_no_choices_is_malformed_rather_than_empty_output() {
        // Distinguishing these matters: an empty choices array is a broken response,
        // whereas an empty string in a choice is a model that generated nothing.
        let wire: WireResponse = serde_json::from_str(r#"{"choices":[]}"#).expect("parses");
        assert!(wire.choices.is_empty());
    }
}
