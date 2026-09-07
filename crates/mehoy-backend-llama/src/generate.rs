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

use mehoy_core::inference::{FinishReason, GenerateTextRequest, GenerationResult, GenerationUsage};

use crate::channel::BackendChannel;

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
