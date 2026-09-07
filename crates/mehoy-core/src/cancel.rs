//! Stopping work that is already running.
//!
//! Cancellation is addressed to a request rather than to whoever is watching it,
//! for the reason recorded in ADR-0009: a consumer may stop reading without
//! wanting the work stopped, and a request may need stopping when nobody is
//! watching at all.
//!
//! Nothing here decides *how* a backend stops. This is the signal and the reason
//! for it. What a particular engine can actually do about that signal is the
//! backend's own property, because engines differ in whether they can stop one
//! request, only a whole connection, or nothing short of being killed.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use crate::id::RequestId;

/// Why a request is being stopped.
///
/// The mechanism is the same in every case, and the reason is not. A caller that
/// gave up and a request that outstayed its budget are different events, and a
/// consumer that cannot tell them apart cannot report either honestly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CancellationCause {
    /// Something asked for this request to stop.
    User,
    /// The backend stopped producing output for longer than the request allowed.
    StreamIdleTimeout,
    /// Nothing is reading the output any more.
    ///
    /// Distinct from [`CancellationCause::User`] on purpose. Nobody asked for this
    /// request to stop; the only thing that could observe it went away, and with a
    /// backend stopped by closing its transport there is no way to keep the work
    /// running that does not also hold a connection open to discard the answer.
    ///
    /// Recorded rather than left implicit so that a failed channel send is not
    /// what decides a public semantic.
    ConsumerGone,
    /// The instance serving the request is being unloaded.
    ///
    /// Unloading does not end requests by its own separate mechanism. It stops
    /// them the same way anything else does, so there is one way for a request to
    /// end rather than two that could disagree, and this cause is what preserves
    /// why it happened.
    InstanceUnloading,
}

impl fmt::Display for CancellationCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::User => f.write_str("cancelled"),
            Self::StreamIdleTimeout => f.write_str("idle for longer than allowed"),
            Self::ConsumerGone => f.write_str("nothing was reading the output"),
            Self::InstanceUnloading => f.write_str("the model was being unloaded"),
        }
    }
}

/// How long a single request may take.
///
/// Only the limit that is implemented appears here. A total deadline and a
/// dispatch deadline are both plausible and neither exists, so neither is
/// represented: a field that is accepted and ignored is worse than an absent one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestBudget {
    /// How long the backend may produce nothing before the request is stopped.
    ///
    /// Counted from sending the request, not from the first thing that comes
    /// back. A backend that reads a large input before answering is silent for
    /// that whole period, and it is the longest silence in a request, so a bound
    /// that ignored it would leave the worst case unbounded.
    ///
    /// A liveness bound, not a limit on how long generation may run: a backend
    /// steadily producing output is never stopped by it.
    pub stream_idle: Duration,
}

impl RequestBudget {
    /// The default idle allowance.
    ///
    /// Generous, because input processing happens before any output and can take
    /// tens of seconds for a large prompt on a slow accelerator. Stopping a
    /// request that is merely slow would be worse than waiting.
    pub const DEFAULT_STREAM_IDLE: Duration = Duration::from_secs(120);
}

impl Default for RequestBudget {
    fn default() -> Self {
        Self {
            stream_idle: Self::DEFAULT_STREAM_IDLE,
        }
    }
}

/// How long unloading waits for work to settle before it stops waiting.
///
/// Deliberately not the same value, or the same field, as a request's idle
/// allowance. That one asks whether a single request is still alive; this one asks
/// how long an instance being torn down should let its outstanding work finish
/// first. Sharing one duration between them would mean tuning one and silently
/// changing the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnloadBudget {
    /// How long active requests are given to reach a terminal state.
    ///
    /// Expiring is not a claim that the backend is still working, nor that it has
    /// stopped. It means the instance stopped waiting and escalated to terminating
    /// the worker, which is the one thing that does end the work for certain.
    pub drain: Duration,
}

impl UnloadBudget {
    /// The default drain allowance.
    ///
    /// Long enough for a request already producing output to notice and stop,
    /// short enough that unloading is not held hostage by a backend that has
    /// decided to finish reading a very large prompt first.
    pub const DEFAULT_DRAIN: Duration = Duration::from_secs(10);
}

impl Default for UnloadBudget {
    fn default() -> Self {
        Self {
            drain: Self::DEFAULT_DRAIN,
        }
    }
}

/// Where a request has got to.
///
/// `Cancelling` exists because cancellation is not instantaneous. A backend may
/// keep working for some time after being told to stop, and a state machine that
/// jumps straight to `Cancelled` would claim the work had ended when it had not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RequestState {
    /// Accepted, not yet dispatched to a backend.
    Starting,
    /// Dispatched, and the backend has it.
    Running,
    /// Stopping was requested; the backend has not finished stopping.
    Cancelling,
    /// Ended by producing a complete result.
    Completed,
    /// Ended by being stopped.
    Cancelled,
    /// Ended by failing.
    Failed,
}

impl RequestState {
    /// Whether no further transition can occur.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Cancelled | Self::Failed)
    }
}

impl fmt::Display for RequestState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Starting => "starting",
            Self::Running => "running",
            Self::Cancelling => "cancelling",
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
            Self::Failed => "failed",
        })
    }
}

/// What a backend is able to stop.
///
/// Engines differ, and a runtime that assumes the most capable option will
/// misreport the others. Only the strategy that a backend in this workspace
/// actually implements is represented: further variants belong here when a
/// backend arrives that offers them, not in anticipation of one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CancellationStrategy {
    /// Stopping a request means closing the transport carrying it.
    ///
    /// Requires that each request own its connection, since closing a shared one
    /// would stop unrelated work.
    ConnectionAbort,
}

impl fmt::Display for CancellationStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConnectionAbort => f.write_str("by closing the request's connection"),
        }
    }
}

/// A one-way signal that a request should stop.
///
/// Cloneable and cheap, so the thing that cancels and the thing that is cancelled
/// need not know about each other. Once set it never clears: a request that has
/// been told to stop is not later un-told, and a task that missed the signal while
/// busy still sees it when it next looks.
#[derive(Debug, Clone)]
pub struct CancellationToken {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    // The sender is held here rather than handed out, so it outlives every
    // receiver and `changed()` can never fail because the producer went away.
    signal: tokio::sync::watch::Sender<Option<CancellationCause>>,
}

impl CancellationToken {
    #[must_use]
    pub fn new() -> Self {
        let (signal, _) = tokio::sync::watch::channel(None);
        Self {
            inner: Arc::new(Inner { signal }),
        }
    }

    /// Requests that the work stop, returning whether this call was the one that
    /// did it.
    ///
    /// The first cause wins. A second cancellation is not an error and does not
    /// overwrite the first, so a caller cancelling a request that a timeout has
    /// already claimed does not rewrite why it stopped.
    pub fn cancel(&self, cause: CancellationCause) -> bool {
        let mut claimed = false;
        self.inner.signal.send_if_modified(|slot| {
            if slot.is_none() {
                *slot = Some(cause);
                claimed = true;
                true
            } else {
                false
            }
        });
        claimed
    }

    /// Why the work was stopped, if it was.
    #[must_use]
    pub fn cause(&self) -> Option<CancellationCause> {
        *self.inner.signal.borrow()
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cause().is_some()
    }

    /// Resolves when cancellation is requested, and never otherwise.
    ///
    /// Safe to use as a branch of a select: it resolves at most once per
    /// cancellation and holds no state that a cancelled branch could lose.
    pub async fn cancelled(&self) -> CancellationCause {
        let mut watcher = self.inner.signal.subscribe();
        loop {
            if let Some(cause) = *watcher.borrow_and_update() {
                return cause;
            }
            if watcher.changed().await.is_err() {
                // Unreachable while `self` is alive, since `self` holds the
                // sender. Waiting forever is still the honest answer to "tell me
                // when this is cancelled" when nothing can ever cancel it.
                std::future::pending::<()>().await;
            }
        }
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

/// One in-flight request, observable from anywhere that holds a clone.
///
/// The runtime keeps one to be able to stop the request, the execution task keeps
/// one to report where it has got to, and the consumer's stream keeps one so the
/// two views cannot disagree. Cloning is cheap and every clone sees the same
/// state.
///
/// This is what ADR-0009 means by the runtime owning requests rather than
/// streams: the handle exists from the moment a request is accepted, before any
/// backend has been contacted, and it outlives whoever happens to hold the stream.
#[derive(Debug, Clone)]
pub struct RequestHandle {
    id: RequestId,
    state: Arc<std::sync::Mutex<RequestState>>,
    cancel: CancellationToken,
}

impl RequestHandle {
    #[must_use]
    pub fn new(id: RequestId) -> Self {
        Self {
            id,
            state: Arc::new(std::sync::Mutex::new(RequestState::Starting)),
            cancel: CancellationToken::new(),
        }
    }

    #[must_use]
    pub const fn id(&self) -> RequestId {
        self.id
    }

    /// The signal the execution task watches.
    #[must_use]
    pub const fn token(&self) -> &CancellationToken {
        &self.cancel
    }

    #[must_use]
    pub fn state(&self) -> RequestState {
        *self.read()
    }

    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.state().is_terminal()
    }

    /// Records that the backend has taken the request.
    ///
    /// Ignored once the request is stopping or has ended, so a late report cannot
    /// move a finished request back to running.
    pub fn mark_running(&self) {
        let mut state = self.read();
        if *state == RequestState::Starting {
            *state = RequestState::Running;
        }
    }

    /// Asks for the request to stop, returning whether this call was the one that
    /// started it stopping.
    ///
    /// Moves to [`RequestState::Cancelling`] rather than to `Cancelled`, because
    /// asking is not the same as having stopped. The terminal state is recorded
    /// when execution actually ends.
    pub fn request_cancellation(&self, cause: CancellationCause) -> bool {
        let mut state = self.read();
        if state.is_terminal() {
            return false;
        }
        let claimed = self.cancel.cancel(cause);
        if claimed {
            *state = RequestState::Cancelling;
        }
        claimed
    }

    /// Records the outcome. The first terminal state wins.
    ///
    /// A completion racing a cancellation is decided here, once, so a request can
    /// never report two outcomes.
    pub fn finish(&self, outcome: RequestState) -> bool {
        debug_assert!(outcome.is_terminal(), "{outcome} is not a terminal state");
        let mut state = self.read();
        if state.is_terminal() {
            return false;
        }
        *state = outcome;
        true
    }

    /// Why the request was asked to stop, if it was.
    #[must_use]
    pub fn cancellation_cause(&self) -> Option<CancellationCause> {
        self.cancel.cause()
    }

    fn read(&self) -> std::sync::MutexGuard<'_, RequestState> {
        // Poisoning would mean a panic while holding this lock, which cannot
        // happen: nothing here awaits or calls out while it is held.
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_request_is_starting() {
        let handle = RequestHandle::new(RequestId::from_raw(1));
        assert_eq!(handle.state(), RequestState::Starting);
        assert!(!handle.is_terminal());
    }

    #[test]
    fn cancelling_moves_to_cancelling_and_not_to_cancelled() {
        // Asking is not the same as having stopped. The backend may keep working
        // for seconds afterwards.
        let handle = RequestHandle::new(RequestId::from_raw(1));
        assert!(handle.request_cancellation(CancellationCause::User));
        assert_eq!(handle.state(), RequestState::Cancelling);
        assert!(!handle.is_terminal());
    }

    #[test]
    fn cancelling_twice_is_claimed_once() {
        let handle = RequestHandle::new(RequestId::from_raw(1));
        assert!(handle.request_cancellation(CancellationCause::User));
        assert!(!handle.request_cancellation(CancellationCause::User));
    }

    #[test]
    fn a_finished_request_cannot_be_cancelled() {
        let handle = RequestHandle::new(RequestId::from_raw(1));
        assert!(handle.finish(RequestState::Completed));
        assert!(!handle.request_cancellation(CancellationCause::User));
        assert_eq!(handle.state(), RequestState::Completed);
    }

    #[test]
    fn only_the_first_terminal_state_is_recorded() {
        // The race that matters: a completion arriving as a cancellation lands.
        let handle = RequestHandle::new(RequestId::from_raw(1));
        assert!(handle.finish(RequestState::Completed));
        assert!(!handle.finish(RequestState::Cancelled));
        assert_eq!(handle.state(), RequestState::Completed);
    }

    #[test]
    fn a_cancelled_request_can_still_finish_as_cancelled() {
        let handle = RequestHandle::new(RequestId::from_raw(1));
        handle.request_cancellation(CancellationCause::User);
        assert!(handle.finish(RequestState::Cancelled));
        assert_eq!(handle.state(), RequestState::Cancelled);
        assert_eq!(handle.cancellation_cause(), Some(CancellationCause::User));
    }

    #[test]
    fn a_cancelled_request_that_completes_first_reports_completed() {
        // Cancellation requested, but the backend finished before it took effect.
        let handle = RequestHandle::new(RequestId::from_raw(1));
        handle.request_cancellation(CancellationCause::User);
        assert!(handle.finish(RequestState::Completed));
        assert_eq!(handle.state(), RequestState::Completed);
    }

    #[test]
    fn running_cannot_move_a_stopping_request_backwards() {
        let handle = RequestHandle::new(RequestId::from_raw(1));
        handle.request_cancellation(CancellationCause::User);
        handle.mark_running();
        assert_eq!(handle.state(), RequestState::Cancelling);
    }

    #[test]
    fn every_clone_sees_the_same_request() {
        let handle = RequestHandle::new(RequestId::from_raw(7));
        let other = handle.clone();
        other.mark_running();
        assert_eq!(handle.state(), RequestState::Running);
        assert_eq!(other.id(), RequestId::from_raw(7));
    }

    #[test]
    fn a_fresh_token_is_not_cancelled() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
        assert_eq!(token.cause(), None);
    }

    #[test]
    fn the_first_cause_wins() {
        // A timeout and a caller can arrive together. Whichever is first decides
        // why the request stopped, and the other does not rewrite history.
        let token = CancellationToken::new();
        assert!(token.cancel(CancellationCause::StreamIdleTimeout));
        assert!(!token.cancel(CancellationCause::User));
        assert_eq!(token.cause(), Some(CancellationCause::StreamIdleTimeout));
    }

    #[test]
    fn cancelling_is_visible_through_every_clone() {
        let token = CancellationToken::new();
        let other = token.clone();
        assert!(other.cancel(CancellationCause::User));
        assert!(token.is_cancelled());
    }

    #[tokio::test]
    async fn waiting_resolves_for_a_token_cancelled_beforehand() {
        let token = CancellationToken::new();
        token.cancel(CancellationCause::User);
        assert_eq!(token.cancelled().await, CancellationCause::User);
    }

    #[tokio::test]
    async fn waiting_resolves_when_cancellation_arrives_later() {
        let token = CancellationToken::new();
        let waiter = token.clone();
        let handle = tokio::spawn(async move { waiter.cancelled().await });
        tokio::task::yield_now().await;
        token.cancel(CancellationCause::StreamIdleTimeout);
        assert_eq!(
            handle.await.expect("the waiter finishes"),
            CancellationCause::StreamIdleTimeout
        );
    }

    #[tokio::test]
    async fn waiting_never_resolves_without_cancellation() {
        let token = CancellationToken::new();
        let outcome = tokio::time::timeout(Duration::from_millis(50), token.cancelled()).await;
        assert!(outcome.is_err(), "an uncancelled token must not resolve");
    }

    #[test]
    fn only_ended_states_are_terminal() {
        for state in [
            RequestState::Starting,
            RequestState::Running,
            RequestState::Cancelling,
        ] {
            assert!(!state.is_terminal(), "{state} should not be terminal");
        }
        for state in [
            RequestState::Completed,
            RequestState::Cancelled,
            RequestState::Failed,
        ] {
            assert!(state.is_terminal(), "{state} should be terminal");
        }
    }
}
