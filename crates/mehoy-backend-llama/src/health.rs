//! Backend readiness, and what it is honestly evidence of.
//!
//! # Two different claims
//!
//! [`BackendReady`] means the process is alive, its endpoint answers, and it
//! reports a serving state. That is all.
//!
//! It does **not** mean the backend can generate correctly. There are reported
//! cases upstream where health reported success while the backend had been left
//! unusable by a failed initialisation, so treating a healthy response as proof of
//! working inference would overstate the evidence.
//!
//! Proving generation works requires performing one, which is a later milestone.
//! The two claims are kept as separate states so that nothing accidentally
//! promotes the weaker one.
//!
//! [`BackendReady`]: Readiness::BackendReady

use std::fmt;
use std::time::Duration;

use http_body_util::{BodyExt, Empty};
use hyper::body::Bytes;
use hyper::{Request, StatusCode, header};
use hyper_util::rt::TokioIo;
use tokio::net::TcpStream;

use crate::channel::BackendChannel;

/// The health path exposed by the backend.
///
/// Measured behaviour on llama.cpp build 9010: this endpoint is **not**
/// authenticated. It answers `200` with no credential, with a wrong credential, and
/// with the correct one. It reports whether the backend is serving and nothing
/// about whether the runtime can actually talk to it.
pub const HEALTH_PATH: &str = "/health";

/// A path that is genuinely credential-protected.
///
/// Measured on the same build: `401` without a credential, `200` with the correct
/// one. Because [`HEALTH_PATH`] ignores credentials entirely, readiness cannot be
/// established from it alone. A misconfigured secret would otherwise produce a
/// backend that reports itself ready and then rejects the first real request.
pub const PROPS_PATH: &str = "/props";

/// How long a single health request may take.
const REQUEST_BUDGET: Duration = Duration::from_secs(5);

/// What has actually been established about a backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Readiness {
    /// The process is alive, its endpoint answers, and it reports a serving state.
    ///
    /// Not evidence that generation works.
    BackendReady,
    /// A generation has actually been performed and produced output.
    ///
    /// Nothing sets this yet. It exists so that the weaker claim above is never
    /// mistaken for it.
    InferenceVerified,
}

/// Whether the runtime's credential is actually accepted by the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CredentialState {
    /// The backend accepted the runtime's credential.
    Accepted,
    /// The backend refused it. Waiting will not help.
    Refused { status: u16 },
}

/// Where a backend is in its startup.
///
/// These are diagnostic states derived from what the endpoint reports, not from
/// reading the backend's log output. Logs are evidence; readiness is a protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartupPhase {
    /// The process exists but nothing is listening yet.
    WaitingForEndpoint,
    /// The endpoint answers but reports that it is still coming up.
    LoadingModel,
    /// The endpoint reports a serving state.
    BackendReady,
}

impl fmt::Display for StartupPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::WaitingForEndpoint => "waiting for endpoint",
            Self::LoadingModel => "loading model",
            Self::BackendReady => "backend ready",
        })
    }
}

/// Why a health check could not be interpreted.
#[derive(Debug)]
pub enum HealthError {
    /// The backend answered with a status that means it will not become ready.
    Rejected { status: StatusCode },
    /// The request could not be completed.
    Transport(String),
}

impl fmt::Display for HealthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rejected { status } => {
                write!(f, "backend health returned {status}, which is terminal")
            }
            Self::Transport(detail) => write!(f, "backend health request failed: {detail}"),
        }
    }
}

impl std::error::Error for HealthError {}

/// Maps a health status code to a startup phase.
///
/// `503` is the documented "still loading" signal rather than a failure, so it must
/// not be treated as one: doing so would abandon every backend that takes more than
/// an instant to load a model.
///
/// `401` and `403` mean the secret was rejected, which will not improve by waiting
/// and is reported immediately.
///
/// # Errors
///
/// Returns [`HealthError::Rejected`] for a status that will not become ready.
pub fn phase_for_status(status: StatusCode) -> Result<StartupPhase, HealthError> {
    match status {
        StatusCode::OK => Ok(StartupPhase::BackendReady),
        StatusCode::SERVICE_UNAVAILABLE => Ok(StartupPhase::LoadingModel),
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => Err(HealthError::Rejected { status }),
        // Anything else is unexpected. Waiting is the safer reading during startup,
        // and the startup deadline still bounds it.
        _ => Ok(StartupPhase::LoadingModel),
    }
}

/// Asks the backend how it is doing.
///
/// A connection failure is reported as [`StartupPhase::WaitingForEndpoint`] rather
/// than an error, because during startup it simply means the listener is not up
/// yet.
///
/// # Errors
///
/// Returns [`HealthError`] when the backend answers in a way that will not become
/// ready.
pub async fn probe(channel: &BackendChannel) -> Result<StartupPhase, HealthError> {
    let attempt = tokio::time::timeout(REQUEST_BUDGET, request(channel)).await;
    match attempt {
        // Not listening yet is the normal early state, not a failure.
        Ok(Err(HealthError::Transport(_))) | Err(_) => Ok(StartupPhase::WaitingForEndpoint),
        Ok(other) => other,
    }
}

/// Confirms the backend actually enforces, and accepts, the runtime's credential.
///
/// This is the difference between a credential being configured and a credential
/// being enforced. It is checked against a path that is known to require one.
///
/// # Errors
///
/// Returns [`HealthError::Transport`] when the request cannot be completed.
pub async fn probe_credential(channel: &BackendChannel) -> Result<CredentialState, HealthError> {
    let status = tokio::time::timeout(REQUEST_BUDGET, status_of(channel, PROPS_PATH, true))
        .await
        .map_err(|_| HealthError::Transport("credential check timed out".to_owned()))??;

    Ok(if status.is_success() {
        CredentialState::Accepted
    } else {
        CredentialState::Refused {
            status: status.as_u16(),
        }
    })
}

/// Asks a path what it answers without any credential.
///
/// Used to establish that a path is protected at all, so a test cannot pass by
/// checking an endpoint that never required a credential.
///
/// # Errors
///
/// Returns [`HealthError::Transport`] when the request cannot be completed.
pub async fn status_without_credential(
    channel: &BackendChannel,
    path: &str,
) -> Result<StatusCode, HealthError> {
    tokio::time::timeout(REQUEST_BUDGET, status_of(channel, path, false))
        .await
        .map_err(|_| HealthError::Transport("request timed out".to_owned()))?
}

async fn request(channel: &BackendChannel) -> Result<StartupPhase, HealthError> {
    phase_for_status(status_of(channel, HEALTH_PATH, true).await?)
}

/// Performs one request and returns its status.
async fn status_of(
    channel: &BackendChannel,
    path: &str,
    with_credential: bool,
) -> Result<StatusCode, HealthError> {
    let stream = TcpStream::connect(channel.address())
        .await
        .map_err(|err| HealthError::Transport(err.to_string()))?;

    let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
        .await
        .map_err(|err| HealthError::Transport(err.to_string()))?;
    let pump = tokio::spawn(async move {
        let _ = connection.await;
    });

    let mut builder = Request::builder()
        .method("GET")
        .uri(path)
        .header(header::HOST, channel.host());
    if with_credential {
        builder = builder.header(
            header::AUTHORIZATION,
            format!("Bearer {}", channel.secret().expose()),
        );
    }
    let request = builder
        .body(Empty::<Bytes>::new())
        .map_err(|err| HealthError::Transport(err.to_string()))?;

    let response = sender
        .send_request(request)
        .await
        .map_err(|err| HealthError::Transport(err.to_string()))?;
    let status = response.status();
    // The body is drained so the connection ends cleanly rather than being reset.
    let _ = response.into_body().collect().await;
    pump.abort();

    Ok(status)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_serving_backend_is_ready() {
        assert_eq!(
            phase_for_status(StatusCode::OK).expect("ok is not terminal"),
            StartupPhase::BackendReady
        );
    }

    #[test]
    fn service_unavailable_means_still_loading_not_failed() {
        // Treating 503 as a failure would abandon every backend that takes more
        // than an instant to load a model.
        assert_eq!(
            phase_for_status(StatusCode::SERVICE_UNAVAILABLE).expect("503 is not terminal"),
            StartupPhase::LoadingModel
        );
    }

    #[test]
    fn a_rejected_secret_is_terminal_rather_than_retried() {
        for status in [StatusCode::UNAUTHORIZED, StatusCode::FORBIDDEN] {
            let err = phase_for_status(status).expect_err("must be terminal");
            assert!(
                matches!(err, HealthError::Rejected { .. }),
                "{status} should be terminal, got {err}"
            );
        }
    }

    #[test]
    fn backend_ready_is_not_inference_verified() {
        // These must stay distinct. Health reporting success is not evidence that
        // generation works, and conflating them would overstate what is known.
        assert_ne!(Readiness::BackendReady, Readiness::InferenceVerified);
    }

    #[test]
    fn an_unexpected_status_keeps_waiting_within_the_deadline() {
        assert_eq!(
            phase_for_status(StatusCode::NOT_FOUND).expect("not terminal"),
            StartupPhase::LoadingModel
        );
    }
}
