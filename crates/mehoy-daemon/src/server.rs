//! The accept loop and shutdown sequence.

use std::convert::Infallible;
use std::future::Future;
use std::io;
use std::time::Duration;

use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use tokio::task::JoinSet;

use mehoy_core::transport::Endpoint;

use crate::service;

/// How long in-flight connections are given to finish once shutdown begins.
///
/// This is a shutdown deadline, not a request timeout. ADR-0002 records why the
/// two are kept apart: legitimate request durations vary enormously by model and
/// prompt, so no single number can serve both purposes.
pub const DRAIN_DEADLINE: Duration = Duration::from_secs(5);

/// Accepts connections until `shutdown` completes, then drains and returns.
///
/// The endpoint is dropped before returning, which on Unix removes the socket
/// file. A clean shutdown therefore leaves nothing behind for the next start to
/// clean up.
///
/// # Errors
///
/// Returns an error only when accepting fails in a way that cannot be retried.
/// Failures affecting a single connection are isolated to that connection.
pub async fn serve<S>(mut endpoint: Endpoint, shutdown: S) -> io::Result<()>
where
    S: Future<Output = ()> + Send,
{
    let mut connections = JoinSet::new();
    tokio::pin!(shutdown);

    loop {
        tokio::select! {
            () = &mut shutdown => break,
            accepted = endpoint.accept() => {
                match accepted {
                    Ok(stream) => {
                        connections.spawn(serve_connection(stream));
                    }
                    Err(err) if is_transient(&err) => {
                        // One client went away mid-handshake. That is not a reason
                        // to stop serving everyone else.
                        continue;
                    }
                    Err(err) => return Err(err),
                }
            }
            // Reap finished connections so the set does not grow without bound
            // over a long-lived daemon.
            Some(_) = connections.join_next(), if !connections.is_empty() => {}
        }
    }

    drain(&mut connections).await;
    drop(endpoint);
    Ok(())
}

/// Waits for outstanding connections, then abandons any that overrun.
///
/// No request served today is long-lived. When token streaming lands, this needs
/// to become a graceful per-connection shutdown so an in-flight stream is closed
/// with a terminal event rather than dropped.
async fn drain(connections: &mut JoinSet<()>) {
    if connections.is_empty() {
        return;
    }
    let deadline = tokio::time::sleep(DRAIN_DEADLINE);
    tokio::pin!(deadline);
    loop {
        tokio::select! {
            () = &mut deadline => {
                connections.abort_all();
                while connections.join_next().await.is_some() {}
                return;
            }
            result = connections.join_next() => {
                if result.is_none() {
                    return;
                }
            }
        }
    }
}

/// Whether an accept failure affects only the connection being accepted.
fn is_transient(err: &io::Error) -> bool {
    matches!(
        err.kind(),
        io::ErrorKind::ConnectionAborted
            | io::ErrorKind::ConnectionReset
            | io::ErrorKind::Interrupted
            | io::ErrorKind::BrokenPipe
    )
}

async fn serve_connection(stream: mehoy_core::transport::Stream) {
    let service = service_fn(|req: hyper::Request<hyper::body::Incoming>| async move {
        Ok::<_, Infallible>(service::route(req.method(), req.uri().path()))
    });

    // A failed connection is that client's problem, not the daemon's. There is no
    // logging layer yet, so the error is dropped rather than silently swallowed
    // by an empty handler that looks deliberate.
    let _ = http1::Builder::new()
        .serve_connection(TokioIo::new(stream), service)
        .await;
}
