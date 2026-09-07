//! The private channel between the runtime and a backend process.
//!
//! Starting a backend that serves HTTP creates a second endpoint, outside the
//! local transport boundary ADR-0004 established. That endpoint exposes
//! generation and model operations, so leaving it as an unauthenticated listener
//! on a predictable port would put a side door next to the front door the rest of
//! the runtime is careful about.
//!
//! Three properties are therefore required of every backend channel:
//!
//! - it listens on loopback only, never a routable address;
//! - its port is chosen at run time rather than fixed, so it is not sitting at a
//!   guessable location;
//! - it requires a per-worker secret that no client ever sees.
//!
//! This does not defend against code already running as the same operating system
//! user. ADR-0004 states that boundary and nothing here widens it. What it prevents
//! is the runtime accidentally publishing an unauthenticated backend to every local
//! process for as long as a model is loaded.

use std::fmt;
use std::io;
use std::net::{Ipv4Addr, SocketAddr, TcpListener};

/// Bytes of entropy in a channel secret.
const SECRET_BYTES: usize = 32;

/// A per-worker shared secret.
///
/// Deliberately opaque: it does not implement `Display`, and its `Debug` redacts
/// the value, so it cannot reach a log or an error message by accident.
#[derive(Clone, PartialEq, Eq)]
pub struct ChannelSecret(String);

impl ChannelSecret {
    /// Generates a secret from the operating system's randomness source.
    ///
    /// # Errors
    ///
    /// Returns an error when the system randomness source is unavailable. This is
    /// deliberately not softened into a fallback: a predictable secret would be
    /// worse than refusing to start, because it would look like protection while
    /// providing none.
    pub fn generate() -> io::Result<Self> {
        let mut bytes = [0u8; SECRET_BYTES];
        getrandom::fill(&mut bytes)
            .map_err(|err| io::Error::other(format!("system randomness unavailable: {err}")))?;
        let mut encoded = String::with_capacity(SECRET_BYTES * 2);
        for byte in bytes {
            encoded.push_str(&format!("{byte:02x}"));
        }
        Ok(Self(encoded))
    }

    /// The secret value, for handing to the backend and to requests made to it.
    ///
    /// Named so that a call site reads as a deliberate disclosure.
    #[must_use]
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ChannelSecret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ChannelSecret(<redacted>)")
    }
}

/// Where a backend listens, and what is needed to talk to it.
#[derive(Debug, Clone)]
pub struct BackendChannel {
    address: SocketAddr,
    secret: ChannelSecret,
}

impl BackendChannel {
    /// Reserves a loopback port and generates a secret for a new backend.
    ///
    /// The port is discovered by binding one and releasing it, so there is a brief
    /// interval in which another process could take it. The backend fails to start
    /// if that happens, which is a visible startup failure rather than a silent
    /// mis-binding. Passing a listening socket to the backend directly would close
    /// the gap, but the backend takes a port number rather than a socket.
    ///
    /// # Errors
    ///
    /// Returns an error when no loopback port can be reserved or the system
    /// randomness source is unavailable.
    pub fn reserve() -> io::Result<Self> {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
        let address = listener.local_addr()?;
        drop(listener);
        Ok(Self {
            address,
            secret: ChannelSecret::generate()?,
        })
    }

    /// Builds a channel for an already-known address, for tests.
    #[must_use]
    pub fn at(address: SocketAddr, secret: ChannelSecret) -> Self {
        Self { address, secret }
    }

    /// The loopback address the backend listens on.
    #[must_use]
    pub fn address(&self) -> SocketAddr {
        self.address
    }

    /// The host to pass to the backend.
    #[must_use]
    pub fn host(&self) -> String {
        self.address.ip().to_string()
    }

    /// The port to pass to the backend.
    #[must_use]
    pub fn port(&self) -> u16 {
        self.address.port()
    }

    /// The secret required to talk to the backend.
    #[must_use]
    pub fn secret(&self) -> &ChannelSecret {
        &self.secret
    }

    /// Whether this channel is confined to loopback.
    ///
    /// Checked rather than assumed, because an address reaching the backend from
    /// off-host is the failure this whole module exists to prevent.
    #[must_use]
    pub fn is_loopback(&self) -> bool {
        self.address.ip().is_loopback()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reserved_channel_is_loopback_with_a_nonzero_port() {
        let channel = BackendChannel::reserve().expect("a loopback port is reservable");
        assert!(
            channel.is_loopback(),
            "backend must not be reachable off-host"
        );
        assert_ne!(channel.port(), 0, "a concrete port must be chosen");
        assert_eq!(channel.host(), "127.0.0.1");
    }

    #[test]
    fn ports_are_not_fixed_between_channels() {
        let first = BackendChannel::reserve().expect("reserves");
        let second = BackendChannel::reserve().expect("reserves");
        assert_ne!(
            first.port(),
            second.port(),
            "a fixed port would put the backend at a guessable location"
        );
    }

    #[test]
    fn secrets_are_unique_and_long_enough() {
        let first = ChannelSecret::generate().expect("generates");
        let second = ChannelSecret::generate().expect("generates");
        assert_ne!(first.expose(), second.expose());
        assert_eq!(first.expose().len(), SECRET_BYTES * 2);
        assert!(first.expose().chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn a_secret_cannot_reach_a_log_by_accident() {
        // A secret rendered by an error or a debug line would defeat the point of
        // having one, so the redaction is asserted rather than assumed.
        let secret = ChannelSecret::generate().expect("generates");
        let rendered = format!("{secret:?}");
        assert!(rendered.contains("redacted"), "{rendered}");
        assert!(
            !rendered.contains(secret.expose()),
            "the secret leaked through Debug"
        );
    }

    #[test]
    fn a_channel_debug_does_not_leak_its_secret() {
        let channel = BackendChannel::reserve().expect("reserves");
        let rendered = format!("{channel:?}");
        assert!(
            !rendered.contains(channel.secret().expose()),
            "the secret leaked through the channel's Debug: {rendered}"
        );
    }
}
