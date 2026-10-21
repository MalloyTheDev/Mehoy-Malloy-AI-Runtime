//! Transport stress gate.
//!
//! This is deliberately not a deterministic regression test, and it must not be
//! described as one. It does not reproduce the intermittent endpoint failure
//! recorded as a known issue, and it is not known to fail against any previous
//! implementation. What it does is hammer the endpoint hard enough, and in enough
//! shapes, that a listener-handoff defect has a reasonable chance of surfacing
//! rather than hiding behind the tidy timing of an ordinary test.
//!
//! Every iteration asserts the properties that must hold continuously:
//!
//! - the daemon keeps serving;
//! - the endpoint stays reachable;
//! - new connections keep succeeding;
//! - no accept ever observes a missing listener.
//!
//! Endpoint tracing is captured for the whole run so a failure reports the raw
//! operating system error and the listener state at the moment it happened,
//! instead of a flattened error kind.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use mehoy_cli::Client;
use mehoy_core::transport::{Endpoint, EndpointAddress, EndpointEvent, Stage};
use mehoy_protocol::HealthStatus;

/// Accept failures and missing-listener observations seen during a run.
static ANOMALIES: OnceLock<Arc<Mutex<Vec<String>>>> = OnceLock::new();

/// Every event, kept per address.
///
/// The anomaly list above is one stage across every address, which answers the
/// question these tests ask continuously. It cannot answer the other one: why a
/// single endpoint misbehaved. That needs the whole lifecycle of one address, in
/// order, because the interesting sequence is when its instances were created,
/// when it was closed, and what the next bind found still listening.
static TRACES: OnceLock<Mutex<HashMap<String, VecDeque<String>>>> = OnceLock::new();

/// How much of one address's history is kept.
///
/// A storm test produces thousands of events and only the tail explains a
/// failure, so this is bounded for the same reason the worker log is.
const TRACE_DEPTH: usize = 400;

fn traces() -> &'static Mutex<HashMap<String, VecDeque<String>>> {
    TRACES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn record(event: &EndpointEvent) {
    if let Ok(mut guard) = traces().lock() {
        let history = guard.entry(event.address.to_string()).or_default();
        if history.len() == TRACE_DEPTH {
            history.pop_front();
        }
        history.push_back(event.to_string());
    }
}

/// One address's recorded history, oldest first and indented for a panic message.
fn trace_for(address: &EndpointAddress) -> String {
    let Ok(guard) = traces().lock() else {
        return "    (the trace lock was poisoned)".to_owned();
    };
    match guard.get(address.as_str()) {
        Some(history) if !history.is_empty() => history
            .iter()
            .map(|event| format!("    {event}"))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => "    (nothing was recorded for this address)".to_owned(),
    }
}

fn anomalies() -> Arc<Mutex<Vec<String>>> {
    Arc::clone(ANOMALIES.get_or_init(|| Arc::new(Mutex::new(Vec::new()))))
}

/// Installs the endpoint observer once per test binary.
fn install_observer() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        let sink = anomalies();
        // A second observer cannot be installed, which is fine: the first one wins
        // and this binary only installs this one.
        let _ = mehoy_core::transport::set_observer(move |event: &EndpointEvent| {
            // Everything, keyed by address, so a failure on one endpoint can be
            // explained rather than merely reported.
            record(event);

            // Only accept-time problems count as anomalies. A failed *bind* is an
            // expected, recoverable step while a previous endpoint finishes
            // tearing down, and other tests bind concurrently through the same
            // process-wide observer.
            let interesting = (event.stage == Stage::AcceptFailed) && !event.listener_present;
            if interesting {
                if let Ok(mut guard) = sink.lock() {
                    guard.push(event.to_string());
                }
            }
        });
    });
}

fn drain_anomalies() -> Vec<String> {
    anomalies()
        .lock()
        .map(|mut guard| std::mem::take(&mut *guard))
        .unwrap_or_default()
}

fn unique_address(label: &str) -> EndpointAddress {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = format!(
        "mehoyd-stress-{}-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed),
        label
    );

    #[cfg(windows)]
    {
        EndpointAddress::new(format!(r"\\.\pipe\{unique}"))
    }
    #[cfg(unix)]
    {
        let dir = std::env::temp_dir().join(format!("mehoy-stress-{}", std::process::id()));
        EndpointAddress::new(
            dir.join(format!("{unique}.sock"))
                .to_string_lossy()
                .into_owned(),
        )
    }
}

struct RunningDaemon {
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    joined: Option<tokio::task::JoinHandle<std::io::Result<()>>>,
    alive: Arc<AtomicUsize>,
}

impl RunningDaemon {
    async fn start(address: &EndpointAddress) -> Self {
        // Not `expect`. A bind that fails here is the symptom of the known
        // intermittent, and an `expect` reports the outcome while discarding the
        // only record of how the address got into that state.
        let endpoint = Endpoint::bind(address).await.unwrap_or_else(|err| {
            panic!(
                "daemon could not bind its endpoint: {err}\n\
                 trace for {address}:\n{}\n\
                 anomalies: {:#?}",
                trace_for(address),
                drain_anomalies()
            )
        });
        let (stop, stop_rx) = tokio::sync::oneshot::channel::<()>();
        let alive = Arc::new(AtomicUsize::new(1));
        let flag = Arc::clone(&alive);
        let joined = tokio::spawn(async move {
            let result = mehoy_daemon::serve(endpoint, async {
                let _ = stop_rx.await;
            })
            .await;
            flag.store(0, Ordering::SeqCst);
            result
        });
        Self {
            stop: Some(stop),
            joined: Some(joined),
            alive,
        }
    }

    fn is_serving(&self) -> bool {
        self.alive.load(Ordering::SeqCst) == 1
    }

    async fn shutdown(mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(joined) = self.joined.take() {
            tokio::time::timeout(Duration::from_secs(20), joined)
                .await
                .expect("daemon shuts down within the deadline")
                .expect("daemon task does not panic")
                .expect("daemon exits without error");
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sequential_connection_storm() {
    install_observer();
    let address = unique_address("sequential-storm");
    let daemon = RunningDaemon::start(&address).await;
    let client = Client::new(address.clone());

    for attempt in 0..300u32 {
        let health = client.health().await.unwrap_or_else(|err| {
            panic!(
                "request {attempt} failed: {err}\nendpoint anomalies: {:#?}",
                drain_anomalies()
            )
        });
        assert_eq!(health.status, HealthStatus::Ok);
        assert!(
            daemon.is_serving(),
            "daemon stopped serving after request {attempt}; anomalies: {:#?}",
            drain_anomalies()
        );
    }

    daemon.shutdown().await;
    let anomalies = drain_anomalies();
    assert!(anomalies.is_empty(), "endpoint anomalies: {anomalies:#?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_bursts() {
    install_observer();
    let address = unique_address("bursts");
    let daemon = RunningDaemon::start(&address).await;
    let client = Client::new(address.clone());

    for burst in 0..25u32 {
        let mut tasks = Vec::new();
        for _ in 0..12 {
            let client = client.clone();
            tasks.push(tokio::spawn(async move { client.health().await }));
        }
        for (index, task) in tasks.into_iter().enumerate() {
            let health = task
                .await
                .expect("client task does not panic")
                .unwrap_or_else(|err| {
                    panic!(
                        "burst {burst} client {index} failed: {err}\nanomalies: {:#?}",
                        drain_anomalies()
                    )
                });
            assert_eq!(health.status, HealthStatus::Ok);
        }
        assert!(
            daemon.is_serving(),
            "daemon stopped serving in burst {burst}"
        );
    }

    daemon.shutdown().await;
    let anomalies = drain_anomalies();
    assert!(anomalies.is_empty(), "endpoint anomalies: {anomalies:#?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn clients_that_vanish_mid_handshake_do_not_break_the_endpoint() {
    // A client that opens the transport and disappears without completing an HTTP
    // exchange leaves the daemon holding a connected instance that will never be
    // used. The endpoint must keep serving everyone else.
    install_observer();
    let address = unique_address("abandon");
    let daemon = RunningDaemon::start(&address).await;
    let client = Client::new(address.clone());

    for round in 0..40u32 {
        {
            // Connect at the transport layer and drop immediately, sending nothing.
            let stream = mehoy_core::transport::connect(&address).await;
            drop(stream);
        }

        let health = client.health().await.unwrap_or_else(|err| {
            panic!(
                "round {round} failed after an abandoned connection: {err}\nanomalies: {:#?}",
                drain_anomalies()
            )
        });
        assert_eq!(health.status, HealthStatus::Ok);
        assert!(
            daemon.is_serving(),
            "daemon stopped serving in round {round}"
        );
    }

    daemon.shutdown().await;
    let anomalies = drain_anomalies();
    assert!(anomalies.is_empty(), "endpoint anomalies: {anomalies:#?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelled_requests_do_not_break_the_endpoint() {
    // Requests abandoned part-way cancel the client future at an arbitrary point,
    // which is the pattern that exposed the earlier accept cancellation defect.
    install_observer();
    let address = unique_address("cancel");
    let daemon = RunningDaemon::start(&address).await;
    let client = Client::new(address.clone());

    for round in 0..40u32 {
        let cancelling = client.clone();
        let task = tokio::spawn(async move { cancelling.health().await });
        // Abort at an unpredictable point in the exchange.
        tokio::task::yield_now().await;
        task.abort();
        let _ = task.await;

        let health = client.health().await.unwrap_or_else(|err| {
            panic!(
                "round {round} failed after a cancelled request: {err}\nanomalies: {:#?}",
                drain_anomalies()
            )
        });
        assert_eq!(health.status, HealthStatus::Ok);
        assert!(
            daemon.is_serving(),
            "daemon stopped serving in round {round}"
        );
    }

    daemon.shutdown().await;
    let anomalies = drain_anomalies();
    assert!(anomalies.is_empty(), "endpoint anomalies: {anomalies:#?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn repeated_daemon_lifecycles_reuse_the_endpoint() {
    // Start and stop a daemon on the same address repeatedly. A shutdown that fails
    // to release the endpoint shows up here as a bind failure on the next round.
    install_observer();
    let address = unique_address("lifecycle");

    for round in 0..30u32 {
        let daemon = RunningDaemon::start(&address).await;
        let client = Client::new(address.clone());
        for _ in 0..3 {
            let health = client.health().await.unwrap_or_else(|err| {
                panic!(
                    "round {round} failed: {err}\nanomalies: {:#?}",
                    drain_anomalies()
                )
            });
            assert_eq!(health.status, HealthStatus::Ok);
        }
        daemon.shutdown().await;
    }

    let anomalies = drain_anomalies();
    assert!(anomalies.is_empty(), "endpoint anomalies: {anomalies:#?}");
}

/// The amplifier is only useful if it is compiled into the transport these tests
/// exercise. It was originally gated on `cfg(test)`, which is false when
/// mehoy-core is built as a dependency of this test binary, so it silently did
/// nothing while the tests appeared to cover it.
///
/// This is a compile-time check rather than a test: if the feature stops being
/// propagated, this binary fails to build instead of quietly running the stress
/// tests above without any amplification.
const _: () = assert!(
    mehoy_core::transport::RACE_AMPLIFIER_ENABLED,
    "the race-amplifier feature is not enabled for the transport under test, \
     so the stress tests in this file would run without amplification"
);

#[tokio::test]
async fn an_address_keeps_a_trace_of_its_own_lifecycle() {
    // The instrumentation the known intermittent needs, asserted rather than
    // assumed. Its first recurrence explained nothing, because the observer
    // recorded one stage across every address and the bind failure reported an
    // outcome while discarding the only record of how the address got there.
    //
    // What makes the trace worth having is the ordering question: whether the
    // previous endpoint was closed before the next bind looked at the name.
    // Neither end of that comparison existed before.
    install_observer();
    let address = unique_address("trace");

    let daemon = RunningDaemon::start(&address).await;
    let client = Client::new(address.clone());
    assert_eq!(
        client.health().await.expect("serves").status,
        HealthStatus::Ok
    );
    daemon.shutdown().await;

    let trace = trace_for(&address);
    assert!(
        trace.contains("instance_created"),
        "no listening instance was recorded for {address}:
{trace}"
    );
    // Handed off rather than completed. A socket has no per-connection listening
    // instance, so the stages naming one are emitted on Windows only, and this is
    // the marker both platforms use for a connection that reached a handler.
    assert!(
        trace.contains("connection_handed_off"),
        "the connection that served the request was not recorded for {address}:
{trace}"
    );
    assert!(
        trace.contains("endpoint_closed"),
        "the endpoint closing was not recorded for {address}, which is the half of          the comparison a failed rebind needs:
{trace}"
    );
}
