//! The embedding adapter.
//!
//! Translates the runtime's own request and result types to and from this
//! engine's wire format, which is an OpenAI-shaped `POST /v1/embeddings`. That
//! shape stays below the backend boundary: nothing above this module knows it
//! exists, so a second backend speaking something else needs no changes above.
//!
//! # Measured behaviour, llama.cpp build 9010
//!
//! - Without the embedding launch mode the endpoint answers `501` with
//!   `type: not_supported_error`, which is why the launch mode is decided by this
//!   crate rather than by a caller who would have to know the flag exists.
//! - With it, one input returns one vector of 768 components for the model tested,
//!   with a Euclidean norm of exactly 1.0, matching the documented normalisation.

use std::fmt;

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Request, StatusCode, header};
use hyper_util::rt::TokioIo;
use serde::{Deserialize, Serialize};
use tokio::net::TcpStream;

use mehoy_core::inference::{EmbedRequest, Embedding, EmbeddingResult};

use crate::channel::BackendChannel;

/// The engine's embedding path.
pub const EMBEDDINGS_PATH: &str = "/v1/embeddings";

/// How long one embedding request may take.
const REQUEST_BUDGET: std::time::Duration = std::time::Duration::from_secs(60);

/// Why an embedding request failed.
///
/// Deliberately specific. "The model failed" would collapse a backend that was
/// never started in embedding mode, a rejected credential, and a malformed
/// response into one message that tells an operator nothing about which to fix.
#[derive(Debug)]
pub enum EmbedError {
    /// The backend was not started in a mode that serves embeddings.
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

impl fmt::Display for EmbedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotSupported { detail } => write!(
                f,
                "the backend does not serve embeddings as started: {detail}"
            ),
            Self::Unauthorised { status } => {
                write!(f, "the backend refused the runtime's credential ({status})")
            }
            Self::Rejected { status, detail } => {
                write!(f, "the backend rejected the request ({status}): {detail}")
            }
            Self::MalformedResponse { detail } => {
                write!(
                    f,
                    "the backend's embedding response was not usable: {detail}"
                )
            }
            Self::Transport { detail } => write!(f, "the embedding request failed: {detail}"),
        }
    }
}

impl std::error::Error for EmbedError {}

#[derive(Serialize)]
struct WireRequest<'a> {
    input: &'a [String],
}

#[derive(Deserialize)]
struct WireResponse {
    #[serde(default)]
    data: Vec<WireEmbedding>,
}

#[derive(Deserialize)]
struct WireEmbedding {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    embedding: Vec<f32>,
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

/// Sends an embedding request to a running backend.
///
/// # Errors
///
/// Returns [`EmbedError`] describing which layer refused: the launch mode, the
/// credential, the request, the response shape, or the transport.
pub async fn embed(
    channel: &BackendChannel,
    request: &EmbedRequest,
) -> Result<EmbeddingResult, EmbedError> {
    let body = serde_json::to_vec(&WireRequest {
        input: &request.inputs,
    })
    .map_err(|err| EmbedError::MalformedResponse {
        detail: format!("cannot encode the request: {err}"),
    })?;

    let (status, bytes) = tokio::time::timeout(REQUEST_BUDGET, post(channel, body))
        .await
        .map_err(|_| EmbedError::Transport {
            detail: format!("no response within {}s", REQUEST_BUDGET.as_secs()),
        })??;

    if !status.is_success() {
        return Err(classify(status, &bytes));
    }

    let wire: WireResponse =
        serde_json::from_slice(&bytes).map_err(|err| EmbedError::MalformedResponse {
            detail: err.to_string(),
        })?;

    Ok(EmbeddingResult {
        embeddings: wire
            .data
            .into_iter()
            .map(|item| Embedding {
                index: item.index,
                vector: item.embedding,
            })
            .collect(),
    })
}

/// Turns a failure status into the most specific error the body supports.
fn classify(status: StatusCode, bytes: &[u8]) -> EmbedError {
    let parsed: Option<WireError> = serde_json::from_slice(bytes).ok();
    let (message, kind) = parsed.map_or_else(
        || (String::from_utf8_lossy(bytes).into_owned(), String::new()),
        |body| (body.error.message, body.error.kind),
    );

    match status {
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => EmbedError::Unauthorised {
            status: status.as_u16(),
        },
        // The engine reports an unsupported mode with its own status and type, so
        // this is distinguishable from a request the model simply could not serve.
        StatusCode::NOT_IMPLEMENTED => EmbedError::NotSupported { detail: message },
        _ if kind == "not_supported_error" => EmbedError::NotSupported { detail: message },
        _ => EmbedError::Rejected {
            status: status.as_u16(),
            detail: message,
        },
    }
}

async fn post(channel: &BackendChannel, body: Vec<u8>) -> Result<(StatusCode, Bytes), EmbedError> {
    let stream =
        TcpStream::connect(channel.address())
            .await
            .map_err(|err| EmbedError::Transport {
                detail: err.to_string(),
            })?;

    let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
        .await
        .map_err(|err| EmbedError::Transport {
            detail: err.to_string(),
        })?;
    let pump = tokio::spawn(async move {
        let _ = connection.await;
    });

    let request = Request::builder()
        .method("POST")
        .uri(EMBEDDINGS_PATH)
        .header(header::HOST, channel.host())
        .header(header::CONTENT_TYPE, "application/json")
        .header(
            header::AUTHORIZATION,
            format!("Bearer {}", channel.secret().expose()),
        )
        .body(Full::new(Bytes::from(body)))
        .map_err(|err| EmbedError::Transport {
            detail: err.to_string(),
        })?;

    let response = sender
        .send_request(request)
        .await
        .map_err(|err| EmbedError::Transport {
            detail: err.to_string(),
        })?;
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .map_err(|err| EmbedError::Transport {
            detail: err.to_string(),
        })?
        .to_bytes();
    pump.abort();

    Ok((status, bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unsupported_mode_is_distinguished_from_a_rejected_request() {
        // Measured shape from build 9010 when started without the embedding mode.
        let body = br#"{"error":{"code":501,"message":"This server does not support embeddings. Start it with `--embeddings`","type":"not_supported_error"}}"#;
        match classify(StatusCode::NOT_IMPLEMENTED, body) {
            EmbedError::NotSupported { detail } => {
                assert!(detail.contains("does not support embeddings"), "{detail}");
            }
            other => panic!("expected NotSupported, got {other}"),
        }
    }

    #[test]
    fn a_refused_credential_is_its_own_error() {
        // Collapsing this into a generic failure would send an operator looking at
        // the model when the problem is the credential.
        for status in [StatusCode::UNAUTHORIZED, StatusCode::FORBIDDEN] {
            match classify(status, b"{}") {
                EmbedError::Unauthorised { status: reported } => {
                    assert_eq!(reported, status.as_u16());
                }
                other => panic!("expected Unauthorised for {status}, got {other}"),
            }
        }
    }

    #[test]
    fn the_not_supported_type_is_honoured_even_on_another_status() {
        match classify(
            StatusCode::BAD_REQUEST,
            br#"{"error":{"message":"nope","type":"not_supported_error"}}"#,
        ) {
            EmbedError::NotSupported { .. } => {}
            other => panic!("expected NotSupported, got {other}"),
        }
    }

    #[test]
    fn an_ordinary_failure_keeps_its_status_and_message() {
        match classify(
            StatusCode::BAD_REQUEST,
            br#"{"error":{"message":"input too long","type":"invalid_request_error"}}"#,
        ) {
            EmbedError::Rejected { status, detail } => {
                assert_eq!(status, 400);
                assert_eq!(detail, "input too long");
            }
            other => panic!("expected Rejected, got {other}"),
        }
    }

    #[test]
    fn a_non_json_body_still_produces_a_usable_message() {
        match classify(StatusCode::INTERNAL_SERVER_ERROR, b"upstream exploded") {
            EmbedError::Rejected { detail, .. } => assert_eq!(detail, "upstream exploded"),
            other => panic!("expected Rejected, got {other}"),
        }
    }

    #[test]
    fn a_response_carries_indices_through_from_the_wire() {
        let wire: WireResponse = serde_json::from_str(
            r#"{"object":"list","data":[{"index":1,"embedding":[0.1,0.2]},{"index":0,"embedding":[0.3,0.4]}]}"#,
        )
        .expect("parses");
        assert_eq!(wire.data.len(), 2);
        assert_eq!(wire.data[0].index, 1);
        assert_eq!(wire.data[1].index, 0);
    }
}
