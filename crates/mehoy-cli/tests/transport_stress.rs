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

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use mehoy_cli::Client;
use mehoy_core::transport::{Endpoint, EndpointAddress, EndpointEvent, Stage};
use mehoy_protocol::HealthStatus;

/// Accept failures and missing-listener observations seen during a run.
static ANOMALIES: OnceLock<Arc<Mutex<Vec<String>>>> = OnceLock::new();

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
            // Only accept-time problems matter here. A failed *bind* is an
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
        let endpoint = Endpoint::bind(address)
            .await
            .expect("daemon binds its endpoint");
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
