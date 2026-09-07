//! The local endpoint the daemon listens on and clients connect to.
//!
//! ADR-0004 fixes the shape of this transport:
//!
//! - a Unix domain socket on Unix, a named pipe on Windows;
//! - per-user, never system-wide;
//! - access control delegated to filesystem permissions or the pipe's access
//!   control list, not reimplemented in the daemon;
//! - permissions established at creation, so there is no window during which the
//!   endpoint exists but is not yet restricted;
//! - no network listener of any kind.
//!
//! The platform modules present the same small API, so callers do not branch on
//! the operating system.

use std::env;
use std::fmt;
use std::io;

#[cfg_attr(unix, path = "unix.rs")]
#[cfg_attr(windows, path = "windows.rs")]
mod platform;
pub mod trace;

pub use platform::{ClientStream, Endpoint, Stream, connect};
pub use trace::{EndpointEvent, ErrorFacts, Stage, TRACE_ENV, set_observer};

/// Whether the test-only race amplifier is compiled in.
///
/// Exposed so a stress test can assert it is actually enabled, rather than
/// silently losing its amplification if the feature stops being propagated.
pub const RACE_AMPLIFIER_ENABLED: bool = cfg!(feature = "race-amplifier");

/// Environment variable that overrides the default endpoint location.
pub const ENDPOINT_ENV: &str = "MEHOY_ENDPOINT";

/// Where the daemon listens.
///
/// On Unix this is a filesystem path. On Windows it is a named pipe name. Both
/// are held as strings so callers need not branch on platform to display one.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EndpointAddress(String);

impl EndpointAddress {
    /// Wraps a platform-native endpoint location.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// The platform-native location.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Resolves the endpoint to use.
    ///
    /// Uses [`ENDPOINT_ENV`] when it is set to a non-empty value, and the
    /// per-user platform default otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error when the platform default cannot be determined, for
    /// example when no suitable per-user runtime directory exists.
    pub fn resolve() -> Result<Self, EndpointError> {
        match env::var(ENDPOINT_ENV) {
            Ok(value) if !value.trim().is_empty() => Ok(Self::new(value)),
            _ => platform::default_address(),
        }
    }
}

impl fmt::Display for EndpointAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Why an endpoint could not be established or reached.
#[derive(Debug)]
pub enum EndpointError {
    /// Another daemon is already listening on this endpoint.
    AlreadyRunning { address: EndpointAddress },
    /// No daemon is listening on this endpoint.
    NotRunning { address: EndpointAddress },
    /// The directory holding the endpoint is not safe to use.
    ///
    /// Refusing here is deliberate. Creating a socket inside a directory that
    /// other accounts can write to would let them replace or observe it.
    InsecureDirectory { path: String, reason: String },
    /// Something occupies the endpoint path that is not an endpoint this runtime
    /// created, so removing it would be unsafe.
    UnexpectedOccupant { path: String, reason: String },
    /// The per-user default location could not be determined.
    NoDefaultLocation { reason: String },
    /// An underlying operating system failure.
    Io(io::Error),
}

impl fmt::Display for EndpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyRunning { address } => {
                write!(f, "a daemon is already listening on {address}")
            }
            Self::NotRunning { address } => {
                write!(f, "no daemon is listening on {address}")
            }
            Self::InsecureDirectory { path, reason } => {
                write!(f, "refusing to use endpoint directory {path}: {reason}")
            }
            Self::UnexpectedOccupant { path, reason } => {
                write!(f, "refusing to replace {path}: {reason}")
            }
            Self::NoDefaultLocation { reason } => {
                write!(f, "cannot determine a default endpoint location: {reason}")
            }
            Self::Io(err) => write!(f, "endpoint transport failure: {err}"),
        }
    }
}

impl std::error::Error for EndpointError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for EndpointError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn address_round_trips_through_display() {
        let addr = EndpointAddress::new("some-endpoint");
        assert_eq!(addr.as_str(), "some-endpoint");
        assert_eq!(addr.to_string(), "some-endpoint");
    }

    #[test]
    fn default_address_is_available_on_this_platform() {
        let addr = platform::default_address().expect("platform provides a default");
        assert!(
            !addr.as_str().is_empty(),
            "default endpoint address must not be empty"
        );
    }

    #[test]
    fn resolved_default_is_per_user() {
        // The default must not be a fixed system-wide location, or two accounts on
        // one machine would share a daemon. ADR-0004 requires per-user endpoints.
        let addr = platform::default_address().expect("platform provides a default");
        let generic = ["/tmp/mehoyd.sock", r"\\.\pipe\mehoyd"];
        assert!(
            !generic.contains(&addr.as_str()),
            "default endpoint {addr} is not user-specific"
        );
    }
}
