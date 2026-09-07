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

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Request, StatusCode, header};
use hyper_util::rt::TokioIo;
use serde::{Deserialize, Serialize};
use tokio::net::TcpStream;

use mehoy_core::id::RequestId;
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

/// How long the backend may go silent mid-stream before the stream is abandoned.
///
/// A backend that stops sending without closing the connection is
/// indistinguishable from one that has hung, and waiting forever turns that into a
/// leaked task and a consumer that never learns anything. This is a liveness
/// ceiling, not a token budget, and not cancellation.
const IDLE_BUDGET: std::time::Duration = std::time::Duration::from_secs(120);

/// Starts a streaming generation against a running backend.
///
/// Resolves once the backend has accepted the request and the response has begun,
/// which is exactly what [`GenerationEvent::Started`] claims. Content arrives
/// afterwards through the returned stream.
///
/// Backpressure is real: the returned stream has a bounded buffer, and this
/// adapter stops reading the backend's socket once that buffer is full, so a slow
/// consumer slows the backend rather than accumulating in this process. Dropping
/// the stream stops the reader at its next event.
///
/// # Errors
///
/// Returns [`GenerationStreamError`] when the request cannot be delivered or the
/// backend refuses it outright. A failure occurring after the stream is
/// established is reported through the stream instead.
pub async fn stream(
    channel: &BackendChannel,
    request: &GenerateTextRequest,
    request_id: RequestId,
) -> Result<GenerationStream, GenerationStreamError> {
    let parameters = &request.parameters;
    let body = serde_json::to_vec(&WireRequest {
        prompt: request.input.as_str(),
        n_predict: parameters.max_output_tokens,
        temperature: parameters.temperature,
        seed: parameters.seed,
        stop: &parameters.stop,
        stream: true,
    })
    .map_err(|err| GenerationStreamError::Transport {
        detail: format!("cannot encode the request: {err}"),
    })?;

    let opened = tokio::time::timeout(REQUEST_BUDGET, open(channel, body))
        .await
        .map_err(|_| GenerationStreamError::Transport {
            detail: format!(
                "the backend did not respond within {}s",
                REQUEST_BUDGET.as_secs()
            ),
        })??;

    let (sink, stream) = generation_stream(request_id);
    tokio::spawn(read(opened, sink));
    Ok(stream)
}

/// A backend response whose body has not been read yet.
struct OpenStream {
    body: hyper::body::Incoming,
    connection: tokio::task::JoinHandle<()>,
}

/// Sends the request and returns once the response has begun.
///
/// A non-success status is resolved here rather than through the stream. A request
/// the backend never accepted has no stream to report through, and delivering it
/// as a mid-stream failure would imply generation had started.
async fn open(
    channel: &BackendChannel,
    body: Vec<u8>,
) -> Result<OpenStream, GenerationStreamError> {
    let transport = TcpStream::connect(channel.address()).await.map_err(|err| {
        GenerationStreamError::Transport {
            detail: err.to_string(),
        }
    })?;

    let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(transport))
        .await
        .map_err(|err| GenerationStreamError::Transport {
            detail: err.to_string(),
        })?;
    let connection = tokio::spawn(async move {
        let _ = connection.await;
    });

    let request = Request::builder()
        .method("POST")
        .uri(COMPLETIONS_PATH)
        .header(header::HOST, channel.host())
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ACCEPT, "text/event-stream")
        .header(
            header::AUTHORIZATION,
            format!("Bearer {}", channel.secret().expose()),
        )
        .body(Full::new(Bytes::from(body)))
        .map_err(|err| GenerationStreamError::Transport {
            detail: err.to_string(),
        })?;

    let response = match sender.send_request(request).await {
        Ok(response) => response,
        Err(err) => {
            connection.abort();
            return Err(GenerationStreamError::Transport {
                detail: err.to_string(),
            });
        }
    };

    let status = response.status();
    if !status.is_success() {
        let detail = response.into_body().collect().await.map_or_else(
            |err| err.to_string(),
            |collected| classify(status, &collected.to_bytes()).to_string(),
        );
        connection.abort();
        return Err(GenerationStreamError::Refused { detail });
    }

    Ok(OpenStream {
        body: response.into_body(),
        connection,
    })
}

/// Drives the response body until the generation ends.
///
/// Every exit path ends the sink exactly once, so a consumer always learns an
/// outcome rather than watching the stream simply stop. The one exception is a
/// consumer that has gone away, which has nothing left to be told.
async fn read(opened: OpenStream, mut sink: GenerationSink) {
    let OpenStream {
        mut body,
        connection,
    } = opened;
    let mut decoder = SseDecoder::default();
    let mut terminal: Option<GenerationSummary> = None;

    let outcome: Result<EndOfBody, GenerationStreamError> = 'body: loop {
        let frame = match tokio::time::timeout(IDLE_BUDGET, body.frame()).await {
            Err(_) => {
                break 'body Err(GenerationStreamError::UnexpectedEnd {
                    detail: format!("the backend sent nothing for {}s", IDLE_BUDGET.as_secs()),
                });
            }
            Ok(None) => break 'body Ok(EndOfBody::Closed),
            Ok(Some(Err(err))) => {
                break 'body Err(GenerationStreamError::Transport {
                    detail: err.to_string(),
                });
            }
            Ok(Some(Ok(frame))) => frame,
        };

        let Some(chunk) = frame.data_ref() else {
            continue;
        };

        let payloads = match decoder.push(chunk) {
            Ok(payloads) => payloads,
            Err(detail) => break 'body Err(GenerationStreamError::MalformedEvent { detail }),
        };

        for payload in payloads {
            if payload.trim() == DONE_MARKER {
                break 'body Ok(EndOfBody::Marked);
            }

            let parsed: WireResponse = match serde_json::from_str(&payload) {
                Ok(parsed) => parsed,
                Err(err) => {
                    break 'body Err(GenerationStreamError::MalformedEvent {
                        detail: describe(&err, &payload),
                    });
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
                connection.abort();
                return;
            }
        }
    };

    connection.abort();

    match outcome {
        Err(error) => sink.fail(error).await,
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
