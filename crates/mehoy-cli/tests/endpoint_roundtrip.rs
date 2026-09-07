//! End-to-end tests over a real endpoint.
//!
//! These cover the acceptance path for this milestone: the daemon binds a
//! per-user endpoint, a client connects over it, the daemon answers, and a clean
//! shutdown leaves nothing behind. They also cover the two recovery behaviours
//! ADR-0004 requires, namely that a second daemon is refused and that a stale
//! endpoint is reclaimed rather than blindly deleted.
//!
//! Every test uses a unique endpoint so they can run concurrently without
//! colliding, and so a failing run cannot disturb a real daemon.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use mehoy_cli::{Client, ClientError};
use mehoy_core::transport::{Endpoint, EndpointAddress, EndpointError};
use mehoy_protocol::{HealthStatus, ProtocolVersion};

/// Builds an endpoint address unique to this process and call.
fn unique_address(label: &str) -> EndpointAddress {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let unique = format!(
        "mehoyd-test-{}-{}-{}",
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
        let dir = std::env::temp_dir().join(format!("mehoy-test-{}", std::process::id()));
        EndpointAddress::new(
            dir.join(format!("{unique}.sock"))
                .to_string_lossy()
                .into_owned(),
        )
    }
}

/// Creates a test endpoint directory with the same privacy the runtime requires.
///
/// A plain `create_dir_all` uses the process umask, which typically yields a
/// directory readable by everyone. The transport refuses such a directory, and
/// correctly so, which means a test fixture that creates one is testing its own
/// mistake rather than the behaviour under test.
#[cfg(unix)]
fn create_private_dir(path: &std::path::Path) {
    use std::os::unix::fs::DirBuilderExt;
    if path.exists() {
        return;
    }
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(path)
        .expect("test directory is creatable");
}

/// Runs a daemon on `address` for the duration of `body`, then shuts it down and
/// waits for it to finish.
async fn with_daemon<F, Fut, T>(address: &EndpointAddress, body: F) -> T
where
    F: FnOnce(Client) -> Fut,
    Fut: Future<Output = T>,
{
    let endpoint = Endpoint::bind(address)
        .await
        .expect("daemon binds its endpoint");
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let served = tokio::spawn(async move {
        mehoy_daemon::serve(endpoint, async {
            let _ = stop_rx.await;
        })
        .await
    });

    let outcome = body(Client::new(address.clone())).await;

    let _ = stop_tx.send(());
    let result = tokio::time::timeout(Duration::from_secs(10), served)
        .await
        .expect("daemon shuts down within the deadline")
        .expect("daemon task does not panic");
    result.expect("daemon exits without error");

    outcome
}

#[tokio::test]
async fn health_round_trips_over_the_endpoint() {
    let address = unique_address("health");
    let health = with_daemon(&address, |client| async move {
        client.health().await.expect("daemon answers health")
    })
    .await;
    assert_eq!(health.status, HealthStatus::Ok);
}

#[tokio::test]
async fn runtime_reports_a_compatible_protocol() {
    let address = unique_address("runtime");
    let info = with_daemon(&address, |client| async move {
        client.runtime().await.expect("daemon answers runtime")
    })
    .await;

    assert_eq!(info.runtime.name, "mehoyd");
    assert_eq!(info.runtime.version, env!("CARGO_PKG_VERSION"));
    assert!(
        ProtocolVersion::CURRENT.is_compatible_with(info.protocol),
        "daemon protocol {:?} is not compatible with this client",
        info.protocol
    );
}

#[tokio::test]
async fn many_sequential_requests_are_served() {
    // The Windows endpoint must create a replacement pipe instance after each
    // accept. If it did not, only the first request would succeed.
    let address = unique_address("sequential");
    with_daemon(&address, |client| async move {
        for attempt in 0..5 {
            let health = client
                .health()
                .await
                .unwrap_or_else(|err| panic!("request {attempt} failed: {err}"));
            assert_eq!(health.status, HealthStatus::Ok);
        }
    })
    .await;
}

#[tokio::test]
async fn unknown_route_is_reported_as_a_daemon_error() {
    let address = unique_address("unknown-route");
    let err = with_daemon(&address, |client| async move {
        client
            .raw_get("/definitely-not-a-route")
            .await
            .expect_err("unknown route must fail")
    })
    .await;

    match err {
        ClientError::Daemon { status, code, .. } => {
            assert_eq!(status, hyper::StatusCode::NOT_FOUND);
            assert_eq!(code, "not_found");
        }
        other => panic!("expected a daemon error, got {other}"),
    }
}

#[tokio::test]
async fn a_second_daemon_is_refused() {
    let address = unique_address("duplicate");
    let _first = Endpoint::bind(&address).await.expect("first daemon binds");

    match Endpoint::bind(&address).await {
        Err(EndpointError::AlreadyRunning { address: reported }) => {
            assert_eq!(reported, address);
        }
        Err(other) => panic!("expected AlreadyRunning, got {other}"),
        Ok(_) => panic!("a second daemon must not be able to bind the same endpoint"),
    }
}

#[tokio::test]
async fn client_reports_not_running_when_no_daemon_is_listening() {
    let address = unique_address("absent");
    let client = Client::new(address.clone());
    match client.health().await {
        Err(ClientError::Transport(EndpointError::NotRunning { address: reported })) => {
            assert_eq!(reported, address);
        }
        Err(other) => panic!("expected NotRunning, got {other}"),
        Ok(_) => panic!("no daemon is listening, so this must not succeed"),
    }
}

#[tokio::test]
async fn endpoint_is_reusable_after_a_clean_shutdown() {
    // A clean shutdown must leave nothing that blocks the next start. On Unix
    // that means the socket file is gone; on Windows the pipe instance is closed.
    let address = unique_address("reuse");
    with_daemon(&address, |client| async move {
        client.health().await.expect("first daemon answers");
    })
    .await;

    with_daemon(&address, |client| async move {
        client
            .health()
            .await
            .expect("endpoint is reusable by a second daemon");
    })
    .await;
}

#[cfg(unix)]
#[tokio::test]
async fn a_stale_socket_is_reclaimed_rather_than_blocking_startup() {
    use std::os::unix::net::UnixListener;

    let address = unique_address("stale");
    let path = std::path::PathBuf::from(address.as_str());
    create_private_dir(path.parent().expect("address has a parent"));

    // Bind and drop a plain listener without removing the file, which is exactly
    // what an unclean daemon exit leaves behind.
    {
        let listener = UnixListener::bind(&path).expect("stale socket is created");
        drop(listener);
    }
    assert!(path.exists(), "test setup must leave a socket behind");

    with_daemon(&address, |client| async move {
        client
            .health()
            .await
            .expect("daemon starts despite the stale socket");
    })
    .await;

    assert!(
        !path.exists(),
        "a clean shutdown must remove the socket it created"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn a_regular_file_at_the_endpoint_path_is_never_deleted() {
    // Existence is not liveness, but neither is it permission to delete. A
    // non-socket occupant must be reported, not removed.
    let address = unique_address("occupied");
    let path = std::path::PathBuf::from(address.as_str());
    create_private_dir(path.parent().expect("address has a parent"));
    std::fs::write(&path, b"not a socket").expect("occupant file is written");

    match Endpoint::bind(&address).await {
        Err(EndpointError::UnexpectedOccupant { .. }) => {}
        Err(other) => panic!("expected UnexpectedOccupant, got {other}"),
        Ok(_) => panic!("binding over a regular file must be refused"),
    }

    assert_eq!(
        std::fs::read(&path).expect("occupant survives"),
        b"not a socket",
        "the occupying file must not be modified or removed"
    );
    let _ = std::fs::remove_file(&path);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn survives_repeated_connections_on_a_multi_thread_runtime() {
    // The daemon binary runs on a multi-threaded runtime while the tests above
    // run on a current-thread one, so this covers the configuration the binary
    // actually uses.
    //
    // This is not a regression guard for the intermittent startup failure
    // recorded as a known issue: that failure was observed only in the built
    // binary with an out-of-process client, and it does not reproduce here.
    let address = unique_address("multi-thread");
    with_daemon(&address, |client| async move {
        for attempt in 0..10 {
            let health = client
                .health()
                .await
                .unwrap_or_else(|err| panic!("request {attempt} failed: {err}"));
            assert_eq!(health.status, HealthStatus::Ok);
        }
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn serves_concurrent_clients() {
    // Several clients in flight at once must all be served, which requires a
    // replacement instance to exist while an earlier one is still connected.
    let address = unique_address("concurrent");
    with_daemon(&address, |client| async move {
        let mut tasks = Vec::new();
        for _ in 0..8 {
            let client = client.clone();
            tasks.push(tokio::spawn(async move { client.health().await }));
        }
        for (index, task) in tasks.into_iter().enumerate() {
            let health = task
                .await
                .expect("client task does not panic")
                .unwrap_or_else(|err| panic!("concurrent client {index} failed: {err}"));
            assert_eq!(health.status, HealthStatus::Ok);
        }
    })
    .await;
}

#[cfg(unix)]
#[tokio::test]
async fn a_world_accessible_endpoint_directory_is_refused() {
    // ADR-0004 requires the endpoint to be private to its user. A directory other
    // accounts can enter would let them reach or replace the socket, so it is
    // refused rather than used.
    //
    // This is not hypothetical: an ordinary `create_dir_all` produces mode 0755
    // under a typical umask, which is exactly this case.
    use std::os::unix::fs::PermissionsExt;

    let address = unique_address("insecure-dir");
    let path = std::path::PathBuf::from(address.as_str());
    let dir = path.parent().expect("address has a parent").to_path_buf();
    // The shared parent must be created privately first. A plain create_dir_all on
    // the subdirectory would create the parent with the process umask, poisoning it
    // for every other test that uses the same parent.
    create_private_dir(&dir);
    let open_dir = dir.join("world-readable");
    std::fs::create_dir_all(&open_dir).expect("test directory is creatable");
    std::fs::set_permissions(&open_dir, std::fs::Permissions::from_mode(0o755))
        .expect("permissions are settable");

    let inside = EndpointAddress::new(open_dir.join("mehoyd.sock").to_string_lossy().into_owned());
    match Endpoint::bind(&inside).await {
        Err(EndpointError::InsecureDirectory { reason, .. }) => {
            assert!(reason.contains("beyond the owner"), "{reason}");
        }
        Err(other) => panic!("expected InsecureDirectory, got {other}"),
        Ok(_) => panic!("a world-accessible endpoint directory must be refused"),
    }
}

#[cfg(unix)]
#[tokio::test]
async fn a_directory_created_by_the_runtime_is_private() {
    // The directory must be private at creation, not fixed afterwards. A mode
    // applied after the fact is a window, however brief.
    use std::os::unix::fs::PermissionsExt;

    let address = unique_address("created-dir");
    let path = std::path::PathBuf::from(address.as_str());
    let dir = path.parent().expect("address has a parent").to_path_buf();
    create_private_dir(&dir);
    let fresh = dir.join("runtime-created");
    let inside = EndpointAddress::new(fresh.join("mehoyd.sock").to_string_lossy().into_owned());

    let endpoint = Endpoint::bind(&inside).await.expect("binds");
    let mode = std::fs::metadata(&fresh)
        .expect("directory exists")
        .permissions()
        .mode()
        & 0o777;
    drop(endpoint);

    assert_eq!(
        mode, 0o700,
        "the endpoint directory should be owner-only, got {mode:04o}"
    );
}
