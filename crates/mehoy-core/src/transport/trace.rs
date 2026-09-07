//! Endpoint lifecycle tracing.
//!
//! This exists to make the next occurrence of an intermittent endpoint failure
//! explain itself, rather than to be summoned by hand. It records the transitions
//! an endpoint goes through and, on failure, the facts that `io::ErrorKind` throws
//! away.
//!
//! The raw operating system error is the point. Windows named pipes report several
//! distinct conditions that flatten into indistinguishable `ErrorKind` values, and
//! telling them apart afterwards is impossible without the original code.
//!
//! Tracing is off unless an observer is installed or `MEHOY_TRACE_ENDPOINT` is set
//! to a non-empty value, in which case events are written to standard error.

use std::io;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

/// Environment variable that turns on stderr tracing.
pub const TRACE_ENV: &str = "MEHOY_TRACE_ENDPOINT";

/// A point in an endpoint instance's life.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// A listening instance was created.
    InstanceCreated,
    /// The endpoint began waiting for a client on an instance.
    ConnectWaitStarted,
    /// A client connected.
    ConnectCompleted,
    /// Creation of the replacement instance began.
    ReplacementCreateStarted,
    /// The replacement instance exists, so the endpoint is listening again.
    ReplacementCreated,
    /// The connected instance was handed to a connection task.
    ConnectionHandedOff,
    /// The listening endpoint was torn down.
    EndpointClosed,
    /// An accept attempt failed.
    AcceptFailed,
}

impl Stage {
    fn as_str(self) -> &'static str {
        match self {
            Self::InstanceCreated => "instance_created",
            Self::ConnectWaitStarted => "connect_wait_started",
            Self::ConnectCompleted => "connect_completed",
            Self::ReplacementCreateStarted => "replacement_create_started",
            Self::ReplacementCreated => "replacement_created",
            Self::ConnectionHandedOff => "connection_handed_off",
            Self::EndpointClosed => "endpoint_closed",
            Self::AcceptFailed => "accept_failed",
        }
    }
}

/// Failure detail that `io::ErrorKind` alone does not preserve.
#[derive(Debug, Clone)]
pub struct ErrorFacts {
    pub kind: io::ErrorKind,
    /// The operating system's own error number. On Windows this distinguishes
    /// conditions that share an `ErrorKind`, such as a pipe whose instances are
    /// all busy from a pipe that does not exist.
    pub raw_os_error: Option<i32>,
    pub message: String,
}

impl ErrorFacts {
    #[must_use]
    pub fn from_io(err: &io::Error) -> Self {
        Self {
            kind: err.kind(),
            raw_os_error: err.raw_os_error(),
            message: err.to_string(),
        }
    }
}

/// One recorded transition.
#[derive(Debug, Clone)]
pub struct EndpointEvent {
    pub stage: Stage,
    /// Which endpoint this concerns. Events from different endpoints share one
    /// observer, so without this they cannot be told apart.
    pub address: Arc<str>,
    /// Which listening instance this concerns, counted from the endpoint's first.
    pub generation: u64,
    /// Whether a listening instance existed when this was recorded. A failure with
    /// no listener present is the state that makes an endpoint unreachable.
    pub listener_present: bool,
    pub error: Option<ErrorFacts>,
    pub at: Instant,
}

impl EndpointEvent {
    pub(crate) fn new(
        stage: Stage,
        address: &Arc<str>,
        generation: u64,
        listener_present: bool,
    ) -> Self {
        Self {
            stage,
            address: Arc::clone(address),
            generation,
            listener_present,
            error: None,
            at: Instant::now(),
        }
    }

    pub(crate) fn with_error(mut self, err: &io::Error) -> Self {
        self.error = Some(ErrorFacts::from_io(err));
        self
    }
}

impl std::fmt::Display for EndpointEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} address={} generation={} listener_present={}",
            self.stage.as_str(),
            self.address,
            self.generation,
            self.listener_present
        )?;
        if let Some(error) = &self.error {
            write!(
                f,
                " kind={:?} raw_os_error={} message={:?}",
                error.kind,
                error
                    .raw_os_error
                    .map_or_else(|| "none".to_owned(), |code| code.to_string()),
                error.message
            )?;
        }
        Ok(())
    }
}

/// A boxed endpoint-event observer.
pub type Observer = Box<dyn Fn(&EndpointEvent) + Send + Sync>;

static OBSERVER: OnceLock<Observer> = OnceLock::new();

/// Installs a process-wide observer for endpoint events.
///
/// # Errors
///
/// Returns the supplied observer unchanged when one is already installed. An
/// observer can be set once, which keeps this from becoming a coordination
/// problem between components.
pub fn set_observer<F>(observer: F) -> Result<(), Observer>
where
    F: Fn(&EndpointEvent) + Send + Sync + 'static,
{
    OBSERVER.set(Box::new(observer))
}

fn stderr_tracing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os(TRACE_ENV).is_some_and(|value| !value.is_empty() && value != "0")
    })
}

pub(crate) fn emit(event: &EndpointEvent) {
    if let Some(observer) = OBSERVER.get() {
        observer(event);
    }
    if stderr_tracing_enabled() {
        eprintln!("mehoy endpoint: {event}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_events_preserve_the_raw_operating_system_error() {
        let err = io::Error::from_raw_os_error(231);
        let address: Arc<str> = Arc::from("test-endpoint");
        let event = EndpointEvent::new(Stage::AcceptFailed, &address, 3, false).with_error(&err);

        let facts = event.error.as_ref().expect("error recorded");
        assert_eq!(facts.raw_os_error, Some(231));

        let rendered = event.to_string();
        assert!(rendered.contains("raw_os_error=231"), "{rendered}");
        assert!(rendered.contains("listener_present=false"), "{rendered}");
        assert!(rendered.contains("generation=3"), "{rendered}");
        assert!(rendered.contains("address=test-endpoint"), "{rendered}");
    }

    #[test]
    fn success_events_render_without_an_error_section() {
        let address: Arc<str> = Arc::from("test-endpoint");
        let event = EndpointEvent::new(Stage::ReplacementCreated, &address, 1, true);
        let rendered = event.to_string();
        assert!(rendered.starts_with("replacement_created"), "{rendered}");
        assert!(!rendered.contains("raw_os_error"), "{rendered}");
    }
}
