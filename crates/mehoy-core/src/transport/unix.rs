//! Unix domain socket endpoint.
//!
//! Access control comes from the directory holding the socket, not from the
//! socket's own mode. The directory is created with mode `0700` before the socket
//! is bound inside it, so the endpoint is never reachable by another account, not
//! even briefly. This is why there is no `chmod` after `bind` anywhere in this
//! file: a permission fix applied after creation is a window, however short.

use std::fs::{self, DirBuilder};
use std::io;
use std::os::unix::fs::{DirBuilderExt, FileTypeExt, MetadataExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::net::{UnixListener, UnixStream};

use super::trace::{self, EndpointEvent, Stage};
use super::{EndpointAddress, EndpointError};

/// Connected socket carrying HTTP between a client and the daemon.
pub type Stream = UnixStream;

/// Connected socket on the client side. The same type on Unix; the platforms
/// differ on Windows, where server and client pipe handles are distinct types.
pub type ClientStream = UnixStream;

/// Mode for the directory containing the socket: owner access only.
const DIR_MODE: u32 = 0o700;

/// Bits that must not be set on the endpoint directory.
const FORBIDDEN_DIR_BITS: u32 = 0o077;

fn effective_uid() -> u32 {
    // Safety: `geteuid` reads process state, takes no arguments, and cannot fail.
    unsafe { libc::geteuid() }
}

/// The per-user default socket location.
///
/// Prefers `XDG_RUNTIME_DIR`, which the operating system already creates as a
/// private per-user directory. Falls back to a uid-qualified directory under
/// `TMPDIR`, which is why the ownership and mode checks below are not optional:
/// a shared temporary directory is exactly where an endpoint could be planted by
/// another account.
pub(super) fn default_address() -> Result<EndpointAddress, EndpointError> {
    let base = match std::env::var_os("XDG_RUNTIME_DIR") {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir).join("mehoy"),
        _ => {
            let tmp = std::env::var_os("TMPDIR")
                .filter(|value| !value.is_empty())
                .map_or_else(|| PathBuf::from("/tmp"), PathBuf::from);
            tmp.join(format!("mehoy-{}", effective_uid()))
        }
    };
    let path = base.join("mehoyd.sock");
    path.to_str()
        .map(EndpointAddress::new)
        .ok_or_else(|| EndpointError::NoDefaultLocation {
            reason: format!("path {} is not valid UTF-8", path.display()),
        })
}

/// Creates the endpoint directory if absent, and verifies it if present.
fn ensure_private_directory(dir: &Path) -> Result<(), EndpointError> {
    match fs::symlink_metadata(dir) {
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            // `mode` is applied by `mkdir` itself, so the directory is never
            // world-accessible even momentarily.
            DirBuilder::new()
                .recursive(true)
                .mode(DIR_MODE)
                .create(dir)
                .map_err(EndpointError::Io)
        }
        Err(err) => Err(EndpointError::Io(err)),
        Ok(meta) => {
            if meta.file_type().is_symlink() {
                return Err(EndpointError::InsecureDirectory {
                    path: dir.display().to_string(),
                    reason: "path is a symbolic link".into(),
                });
            }
            if !meta.is_dir() {
                return Err(EndpointError::InsecureDirectory {
                    path: dir.display().to_string(),
                    reason: "path is not a directory".into(),
                });
            }
            if meta.uid() != effective_uid() {
                return Err(EndpointError::InsecureDirectory {
                    path: dir.display().to_string(),
                    reason: format!(
                        "owned by uid {}, expected uid {}",
                        meta.uid(),
                        effective_uid()
                    ),
                });
            }
            let mode = meta.mode() & 0o777;
            if mode & FORBIDDEN_DIR_BITS != 0 {
                return Err(EndpointError::InsecureDirectory {
                    path: dir.display().to_string(),
                    reason: format!("mode {mode:04o} grants access beyond the owner"),
                });
            }
            Ok(())
        }
    }
}

/// Decides what to do about anything already at the socket path.
///
/// The recovery rule from ADR-0004 is that existence is not liveness. A leftover
/// socket must not block startup, and must not be assumed dead either. The only
/// reliable probe is to attempt a connection:
///
/// - the path is absent, so bind;
/// - a connection succeeds, so a daemon owns it and must not be disturbed;
/// - a connection is refused, so the socket is stale and may be removed once it
///   is confirmed to be a socket we own.
///
/// Unconditionally removing whatever sits at the path would be both a correctness
/// bug, since it would evict a live daemon, and a security bug, since it would
/// delete an arbitrary file chosen by whoever created it.
fn clear_stale_socket(path: &Path, address: &EndpointAddress) -> Result<(), EndpointError> {
    match std::os::unix::net::UnixStream::connect(path) {
        Ok(_) => Err(EndpointError::AlreadyRunning {
            address: address.clone(),
        }),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::ConnectionRefused => {
            let meta = fs::symlink_metadata(path).map_err(EndpointError::Io)?;
            if !meta.file_type().is_socket() {
                return Err(EndpointError::UnexpectedOccupant {
                    path: path.display().to_string(),
                    reason: "existing file is not a socket".into(),
                });
            }
            if meta.uid() != effective_uid() {
                return Err(EndpointError::UnexpectedOccupant {
                    path: path.display().to_string(),
                    reason: format!("socket is owned by uid {}", meta.uid()),
                });
            }
            fs::remove_file(path).map_err(EndpointError::Io)
        }
        Err(err) => Err(EndpointError::Io(err)),
    }
}

/// A bound listening endpoint.
///
/// Removes the socket file when dropped, so a clean shutdown leaves nothing
/// behind.
#[derive(Debug)]
pub struct Endpoint {
    listener: UnixListener,
    address: EndpointAddress,
    path: PathBuf,
    /// Shared copy of the address for trace events.
    trace_address: Arc<str>,
}

impl Endpoint {
    /// Binds the endpoint, preparing and validating its directory first.
    ///
    /// # Errors
    ///
    /// Returns [`EndpointError::AlreadyRunning`] when a daemon already holds the
    /// endpoint, [`EndpointError::InsecureDirectory`] when the containing
    /// directory is not private to this user, and
    /// [`EndpointError::UnexpectedOccupant`] when the path holds something that
    /// must not be removed.
    pub async fn bind(address: &EndpointAddress) -> Result<Self, EndpointError> {
        let path = PathBuf::from(address.as_str());
        let dir = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .ok_or_else(|| EndpointError::NoDefaultLocation {
                reason: format!("endpoint path {} has no parent directory", path.display()),
            })?;

        let trace_address: Arc<str> = Arc::from(address.as_str());

        ensure_private_directory(dir)?;
        clear_stale_socket(&path, address)?;

        let listener = UnixListener::bind(&path).map_err(|err| {
            trace::emit(
                &EndpointEvent::new(Stage::InstanceCreated, &trace_address, 0, false)
                    .with_error(&err),
            );
            if err.kind() == io::ErrorKind::AddrInUse {
                EndpointError::AlreadyRunning {
                    address: address.clone(),
                }
            } else {
                EndpointError::Io(err)
            }
        })?;
        trace::emit(&EndpointEvent::new(
            Stage::InstanceCreated,
            &trace_address,
            0,
            true,
        ));

        Ok(Self {
            listener,
            address: address.clone(),
            path,
            trace_address,
        })
    }

    /// Waits for the next client connection.
    ///
    /// # Errors
    ///
    /// Returns any error reported while accepting.
    pub async fn accept(&mut self) -> io::Result<Stream> {
        trace::emit(&EndpointEvent::new(
            Stage::ConnectWaitStarted,
            &self.trace_address,
            0,
            true,
        ));
        let (stream, _addr) = self.listener.accept().await.inspect_err(|err| {
            trace::emit(
                &EndpointEvent::new(Stage::AcceptFailed, &self.trace_address, 0, true)
                    .with_error(err),
            );
        })?;
        trace::emit(&EndpointEvent::new(
            Stage::ConnectionHandedOff,
            &self.trace_address,
            0,
            true,
        ));
        Ok(stream)
    }

    /// The address this endpoint is listening on.
    #[must_use]
    pub fn address(&self) -> &EndpointAddress {
        &self.address
    }
}

impl Drop for Endpoint {
    fn drop(&mut self) {
        trace::emit(&EndpointEvent::new(
            Stage::EndpointClosed,
            &self.trace_address,
            0,
            false,
        ));
        // Best effort. A failure here leaves a stale socket, which the next
        // startup detects and clears rather than tripping over.
        let _ = fs::remove_file(&self.path);
    }
}

/// Connects to a running daemon.
///
/// # Errors
///
/// Returns [`EndpointError::NotRunning`] when nothing is listening.
pub async fn connect(address: &EndpointAddress) -> Result<Stream, EndpointError> {
    match UnixStream::connect(address.as_str()).await {
        Ok(stream) => Ok(stream),
        Err(err)
            if matches!(
                err.kind(),
                io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused
            ) =>
        {
            Err(EndpointError::NotRunning {
                address: address.clone(),
            })
        }
        Err(err) => Err(EndpointError::Io(err)),
    }
}
