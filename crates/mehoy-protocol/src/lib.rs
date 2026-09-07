//! Wire types for the Mehoy runtime's local HTTP surface.
//!
//! # Stability
//!
//! This protocol is not a compatibility commitment. ADR-0003 records that the
//! protocol is defined first but revised under the pressure of the first working
//! implementation. It becomes a contract when a release says it does, not before.
//!
//! # Surfaces
//!
//! ADR-0004 splits the daemon into a control surface and an inference surface,
//! with control reachable only over the local transport. Neither surface has any
//! operations yet. The endpoints defined here are runtime metadata, which belongs
//! to neither: they report what is listening and what it speaks. Control and
//! inference operations get their own modules when they exist, rather than empty
//! modules reserved in advance.

use serde::{Deserialize, Serialize};

/// Major version of the protocol described by this crate.
///
/// Incremented for changes that existing clients cannot ignore.
pub const PROTOCOL_MAJOR: u32 = 1;

/// Minor version of the protocol described by this crate.
///
/// Incremented for additions that older clients can safely ignore.
pub const PROTOCOL_MINOR: u32 = 0;

/// Liveness probe. Answers only whether the daemon is serving.
pub const PATH_HEALTH: &str = "/health";

/// Runtime metadata, including the protocol version the daemon speaks.
pub const PATH_RUNTIME: &str = "/v1/runtime";

/// The protocol version a daemon speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolVersion {
    pub major: u32,
    pub minor: u32,
}

impl ProtocolVersion {
    /// The version implemented by this crate.
    pub const CURRENT: Self = Self {
        major: PROTOCOL_MAJOR,
        minor: PROTOCOL_MINOR,
    };

    /// Whether a client built against `self` can talk to a daemon speaking `daemon`.
    ///
    /// Major versions must match exactly. The daemon's minor version may exceed the
    /// client's, because minor additions are ignorable; the reverse is not usable,
    /// because the client may rely on something the daemon does not implement.
    #[must_use]
    pub fn is_compatible_with(self, daemon: Self) -> bool {
        self.major == daemon.major && daemon.minor >= self.minor
    }
}

impl Default for ProtocolVersion {
    fn default() -> Self {
        Self::CURRENT
    }
}

/// Identity of the running daemon.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeIdentity {
    pub name: String,
    pub version: String,
}

/// Response body for [`PATH_RUNTIME`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeInfo {
    pub protocol: ProtocolVersion,
    pub runtime: RuntimeIdentity,
}

/// Serving state reported by [`PATH_HEALTH`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthStatus {
    /// The daemon is accepting requests.
    Ok,
    /// The daemon is shutting down and will not accept new work.
    ShuttingDown,
}

/// Response body for [`PATH_HEALTH`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Health {
    pub status: HealthStatus,
}

/// Machine-readable failure returned for any request the daemon does not serve.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorBody {
    pub error: ErrorDetail,
}

/// The failure itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorDetail {
    /// Stable, machine-matchable identifier. Clients should branch on this, not on
    /// `message`, which is prose and may change.
    pub code: String,
    pub message: String,
}

impl ErrorBody {
    /// Builds an error body from a stable code and a human-readable message.
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            error: ErrorDetail {
                code: code.into(),
                message: message.into(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_info_matches_the_documented_shape() {
        let info = RuntimeInfo {
            protocol: ProtocolVersion::CURRENT,
            runtime: RuntimeIdentity {
                name: "mehoyd".into(),
                version: "0.1.0".into(),
            },
        };
        let json = serde_json::to_value(&info).expect("serializes");
        let expected = serde_json::json!({
            "protocol": { "major": 1, "minor": 0 },
            "runtime": { "name": "mehoyd", "version": "0.1.0" }
        });
        assert_eq!(json, expected);
    }

    #[test]
    fn health_status_serializes_snake_case() {
        let json = serde_json::to_value(Health {
            status: HealthStatus::Ok,
        })
        .expect("serializes");
        assert_eq!(json, serde_json::json!({ "status": "ok" }));

        let json = serde_json::to_value(Health {
            status: HealthStatus::ShuttingDown,
        })
        .expect("serializes");
        assert_eq!(json, serde_json::json!({ "status": "shutting_down" }));
    }

    #[test]
    fn same_version_is_compatible() {
        assert!(ProtocolVersion::CURRENT.is_compatible_with(ProtocolVersion::CURRENT));
    }

    #[test]
    fn newer_daemon_minor_is_compatible_but_older_is_not() {
        let client = ProtocolVersion { major: 1, minor: 2 };
        assert!(client.is_compatible_with(ProtocolVersion { major: 1, minor: 3 }));
        assert!(client.is_compatible_with(ProtocolVersion { major: 1, minor: 2 }));
        assert!(!client.is_compatible_with(ProtocolVersion { major: 1, minor: 1 }));
    }

    #[test]
    fn differing_major_is_never_compatible() {
        let client = ProtocolVersion { major: 1, minor: 0 };
        assert!(!client.is_compatible_with(ProtocolVersion { major: 2, minor: 0 }));
        assert!(!client.is_compatible_with(ProtocolVersion { major: 0, minor: 0 }));
    }

    #[test]
    fn error_body_round_trips() {
        let body = ErrorBody::new("not_found", "no such path");
        let json = serde_json::to_string(&body).expect("serializes");
        let back: ErrorBody = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(body, back);
        assert_eq!(back.error.code, "not_found");
    }
}
