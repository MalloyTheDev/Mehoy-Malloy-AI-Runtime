//! Client for the Mehoy runtime daemon's local endpoint.
//!
//! ADR-0001 makes the daemon the public surface and the command-line interface
//! just one client of it. This module holds the client so that the binary, the
//! tests, and any future client share one implementation rather than three.

use std::fmt;

use http_body_util::{BodyExt, Empty};
use hyper::body::Bytes;
use hyper::{Request, StatusCode, header};
use hyper_util::rt::TokioIo;

use mehoy_core::transport::{EndpointAddress, EndpointError, connect};
use mehoy_protocol::{ErrorBody, Health, PATH_HEALTH, PATH_RUNTIME, RuntimeInfo};

/// Authority sent in the `Host` header.
///
/// The transport has no hostname, but HTTP/1.1 requires the header, so a fixed
/// placeholder is used. It is never resolved.
const LOCAL_AUTHORITY: &str = "mehoyd.local";

/// Why a request to the daemon did not produce an answer.
#[derive(Debug)]
pub enum ClientError {
    /// The endpoint could not be reached.
    Transport(EndpointError),
    /// The connection failed or the response was not valid HTTP.
    Protocol(String),
    /// The daemon answered, but with a failure.
    Daemon {
        status: StatusCode,
        code: String,
        message: String,
    },
    /// The daemon answered with a body this client could not parse.
    Malformed(String),
}

impl fmt::Display for ClientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport(err) => write!(f, "{err}"),
            Self::Protocol(detail) => write!(f, "connection failed: {detail}"),
            Self::Daemon {
                status,
                code,
                message,
            } => write!(f, "daemon returned {status} ({code}): {message}"),
            Self::Malformed(detail) => write!(f, "unreadable response: {detail}"),
        }
    }
}

impl std::error::Error for ClientError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Transport(err) => Some(err),
            _ => None,
        }
    }
}

impl From<EndpointError> for ClientError {
    fn from(err: EndpointError) -> Self {
        Self::Transport(err)
    }
}

/// A client bound to one endpoint.
///
/// Each request opens its own connection. Connection reuse is deliberately absent
/// until there is a workload that benefits from it.
#[derive(Debug, Clone)]
pub struct Client {
    address: EndpointAddress,
}

impl Client {
    /// Creates a client for a specific endpoint.
    #[must_use]
    pub fn new(address: EndpointAddress) -> Self {
        Self { address }
    }

    /// Creates a client for the endpoint the daemon would use by default.
    ///
    /// # Errors
    ///
    /// Returns an error when no default endpoint can be determined.
    pub fn resolve() -> Result<Self, ClientError> {
        Ok(Self::new(EndpointAddress::resolve()?))
    }

    /// The endpoint this client talks to.
    #[must_use]
    pub fn address(&self) -> &EndpointAddress {
        &self.address
    }

    /// Fetches the daemon's serving state.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Transport`] when no daemon is listening.
    pub async fn health(&self) -> Result<Health, ClientError> {
        self.get(PATH_HEALTH).await
    }

    /// Fetches the daemon's identity and protocol version.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Transport`] when no daemon is listening.
    pub async fn runtime(&self) -> Result<RuntimeInfo, ClientError> {
        self.get(PATH_RUNTIME).await
    }

    async fn get<T: serde::de::DeserializeOwned>(&self, path: &str) -> Result<T, ClientError> {
        let bytes = self.raw_get(path).await?;
        serde_json::from_slice(&bytes).map_err(|err| ClientError::Malformed(err.to_string()))
    }

    /// Performs a GET and returns the raw body, after mapping any daemon-reported
    /// failure into [`ClientError::Daemon`].
    ///
    /// # Errors
    ///
    /// Returns an error when the request cannot be sent or the daemon reports a
    /// non-success status.
    pub async fn raw_get(&self, path: &str) -> Result<Bytes, ClientError> {
        let stream = connect(&self.address).await?;
        let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
            .await
            .map_err(|err| ClientError::Protocol(err.to_string()))?;

        // The connection task drives the transport while the request is in flight
        // and finishes on its own once the response is complete.
        let pump = tokio::spawn(async move {
            let _ = connection.await;
        });

        let request = Request::builder()
            .method("GET")
            .uri(path)
            .header(header::HOST, LOCAL_AUTHORITY)
            .body(Empty::<Bytes>::new())
            .map_err(|err| ClientError::Protocol(err.to_string()))?;

        let response = sender
            .send_request(request)
            .await
            .map_err(|err| ClientError::Protocol(err.to_string()))?;

        let status = response.status();
        let bytes = response
            .into_body()
            .collect()
            .await
            .map_err(|err| ClientError::Protocol(err.to_string()))?
            .to_bytes();

        pump.abort();

        if status.is_success() {
            return Ok(bytes);
        }

        let (code, message) = serde_json::from_slice::<ErrorBody>(&bytes).map_or_else(
            |_| {
                (
                    "unknown".to_owned(),
                    String::from_utf8_lossy(&bytes).into_owned(),
                )
            },
            |body| (body.error.code, body.error.message),
        );
        Err(ClientError::Daemon {
            status,
            code,
            message,
        })
    }
}
