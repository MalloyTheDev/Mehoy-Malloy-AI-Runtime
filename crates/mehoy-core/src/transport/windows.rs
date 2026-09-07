//! Windows named pipe endpoint.
//!
//! Two properties from ADR-0004 are enforced here by the security descriptor the
//! pipe is created with, not by later checks:
//!
//! - only the account that created the pipe may open it;
//! - the `NETWORK` principal is denied, so the pipe cannot be reached remotely
//!   over SMB as `\\host\pipe\name`.
//!
//! The descriptor is supplied at creation, so there is no interval during which
//! the pipe exists with default permissions.
//!
//! `first_pipe_instance` is set on the first instance only. That makes a second
//! daemon fail immediately and unambiguously rather than silently joining the
//! same pipe name and stealing connections.

use std::ffi::c_void;
use std::io;
use std::ptr;

use tokio::net::windows::named_pipe::{ClientOptions, NamedPipeServer, ServerOptions};

use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_ACCESS_DENIED, ERROR_PIPE_BUSY, HANDLE, HLOCAL, LocalFree,
};
use windows_sys::Win32::Security::Authorization::{
    ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
};
use windows_sys::Win32::Security::{
    GetTokenInformation, PSECURITY_DESCRIPTOR, SECURITY_ATTRIBUTES, TOKEN_QUERY, TOKEN_USER,
    TokenUser,
};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

use super::{EndpointAddress, EndpointError};

/// Connected pipe carrying HTTP between a client and the daemon.
pub type Stream = NamedPipeServer;

/// Buffer sizes advertised when creating a pipe instance. These are hints to the
/// system, not limits on message size.
const PIPE_BUFFER_BYTES: u32 = 64 * 1024;

fn wide_to_string(ptr: *const u16) -> String {
    if ptr.is_null() {
        return String::new();
    }
    let mut len = 0usize;
    // Safety: the caller guarantees `ptr` is a valid null-terminated wide string,
    // which is the documented output of the Win32 conversion functions used here.
    unsafe {
        while *ptr.add(len) != 0 {
            len += 1;
        }
        String::from_utf16_lossy(std::slice::from_raw_parts(ptr, len))
    }
}

fn to_wide_null(value: &str) -> Vec<u16> {
    value.encode_utf16().chain(std::iter::once(0)).collect()
}

/// Returns the current user's security identifier in string form.
fn current_user_sid() -> io::Result<String> {
    // Safety: each call below is checked, handles are closed on every path, and
    // the buffer passed to the second `GetTokenInformation` is sized by the first.
    unsafe {
        let mut token: HANDLE = ptr::null_mut();
        if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &raw mut token) == 0 {
            return Err(io::Error::last_os_error());
        }

        let mut needed: u32 = 0;
        // Expected to fail with ERROR_INSUFFICIENT_BUFFER while reporting the size.
        GetTokenInformation(token, TokenUser, ptr::null_mut(), 0, &raw mut needed);
        if needed == 0 {
            let err = io::Error::last_os_error();
            CloseHandle(token);
            return Err(err);
        }

        let mut buffer = vec![0u8; needed as usize];
        let ok = GetTokenInformation(
            token,
            TokenUser,
            buffer.as_mut_ptr().cast::<c_void>(),
            needed,
            &raw mut needed,
        );
        CloseHandle(token);
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }

        let token_user = buffer.as_ptr().cast::<TOKEN_USER>();
        let mut sid_string: *mut u16 = ptr::null_mut();
        if ConvertSidToStringSidW((*token_user).User.Sid, &raw mut sid_string) == 0 {
            return Err(io::Error::last_os_error());
        }
        let sid = wide_to_string(sid_string);
        LocalFree(sid_string.cast::<c_void>() as HLOCAL);
        Ok(sid)
    }
}

/// Builds the SDDL string for the pipe.
///
/// The deny entry precedes the allow entry because Windows evaluates access
/// control entries in order, and a deny must be reached first to take effect.
fn security_descriptor_definition(sid: &str) -> String {
    format!("D:P(D;;GA;;;NU)(A;;GA;;;{sid})")
}

/// Owns a security descriptor allocated by the Win32 conversion function.
struct SecurityDescriptor {
    raw: PSECURITY_DESCRIPTOR,
}

impl SecurityDescriptor {
    fn from_sddl(sddl: &str) -> io::Result<Self> {
        let wide = to_wide_null(sddl);
        let mut raw: PSECURITY_DESCRIPTOR = ptr::null_mut();
        // Safety: `wide` is a valid null-terminated wide string that outlives the
        // call, and `raw` receives an allocation released in `Drop`.
        let ok = unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                wide.as_ptr(),
                SDDL_REVISION_1,
                &raw mut raw,
                ptr::null_mut(),
            )
        };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self { raw })
    }

    fn attributes(&self) -> SECURITY_ATTRIBUTES {
        SECURITY_ATTRIBUTES {
            nLength: u32::try_from(size_of::<SECURITY_ATTRIBUTES>())
                .expect("SECURITY_ATTRIBUTES size fits in u32"),
            lpSecurityDescriptor: self.raw,
            bInheritHandle: 0,
        }
    }
}

impl Drop for SecurityDescriptor {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // Safety: `raw` was allocated by the Win32 conversion function, which
            // documents `LocalFree` as the matching release.
            unsafe { LocalFree(self.raw.cast::<c_void>() as HLOCAL) };
        }
    }
}

/// The per-user default pipe name.
///
/// The name is qualified by the account's security identifier so two accounts on
/// one machine never contend for a single pipe.
pub(super) fn default_address() -> Result<EndpointAddress, EndpointError> {
    let sid = current_user_sid().map_err(|err| EndpointError::NoDefaultLocation {
        reason: format!("cannot read the current user's identity: {err}"),
    })?;
    Ok(EndpointAddress::new(format!(r"\\.\pipe\mehoyd-{sid}")))
}

/// Creates one pipe instance carrying the restrictive descriptor.
///
/// The descriptor is built here and released before returning, rather than being
/// held by [`Endpoint`]. A raw descriptor pointer stored across an await point
/// would make the endpoint `!Send`, and asserting `Send` for it by hand would be
/// an unsafe promise made only to satisfy the scheduler. Reparsing a short SDDL
/// string per connection is not a cost worth that trade.
fn create_instance(name: &str, sddl: &str, first: bool) -> io::Result<NamedPipeServer> {
    let descriptor = SecurityDescriptor::from_sddl(sddl)?;
    let mut attributes = descriptor.attributes();
    let attributes_ptr: *mut c_void = (&raw mut attributes).cast();

    // Safety: `attributes_ptr` points at a fully initialised SECURITY_ATTRIBUTES
    // whose descriptor is alive for the whole call, which is what the API requires.
    unsafe {
        ServerOptions::new()
            .first_pipe_instance(first)
            .reject_remote_clients(true)
            .in_buffer_size(PIPE_BUFFER_BYTES)
            .out_buffer_size(PIPE_BUFFER_BYTES)
            .create_with_security_attributes_raw(name, attributes_ptr)
    }
}

/// A listening endpoint.
///
/// A named pipe has no filesystem presence, so nothing needs removing on
/// shutdown: the pipe disappears when the last handle closes.
pub struct Endpoint {
    address: EndpointAddress,
    /// The descriptor definition, rebuilt for each instance. Holding the string
    /// rather than the parsed descriptor keeps this type `Send`.
    sddl: String,
    /// The instance waiting for the next client. One is always held so that a
    /// client connecting between two accepts is queued rather than refused.
    pending: Option<NamedPipeServer>,
}

impl std::fmt::Debug for Endpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Endpoint")
            .field("address", &self.address)
            .finish_non_exhaustive()
    }
}

impl Endpoint {
    /// Creates the pipe with a descriptor restricting it to the current user.
    ///
    /// # Errors
    ///
    /// Returns [`EndpointError::AlreadyRunning`] when another daemon holds the
    /// pipe name, and [`EndpointError::Io`] for any other failure.
    pub fn bind(address: &EndpointAddress) -> Result<Self, EndpointError> {
        let sid = current_user_sid().map_err(EndpointError::Io)?;
        let sddl = security_descriptor_definition(&sid);

        let pending = create_instance(address.as_str(), &sddl, true).map_err(|err| {
            if err.raw_os_error() == Some(ERROR_ACCESS_DENIED as i32) {
                EndpointError::AlreadyRunning {
                    address: address.clone(),
                }
            } else {
                EndpointError::Io(err)
            }
        })?;

        Ok(Self {
            address: address.clone(),
            sddl,
            pending: Some(pending),
        })
    }

    /// Waits for the next client connection.
    ///
    /// This future is cancel-safe, which is a requirement rather than a nicety:
    /// callers select over it alongside shutdown and task-reaping branches, so it
    /// is dropped part-way through routinely. The pending instance is therefore
    /// awaited through a shared borrow and only taken once a client has actually
    /// connected. Taking it first would destroy the instance on every cancelled
    /// accept, leaving the pipe name with no instance and making the daemon
    /// unreachable after the first connection completed.
    ///
    /// # Errors
    ///
    /// Returns any error reported while waiting for or preparing an instance.
    pub async fn accept(&mut self) -> io::Result<Stream> {
        let pending = self.pending.as_ref().ok_or_else(|| {
            io::Error::other("named pipe endpoint has no pending instance".to_string())
        })?;

        // Cancelling here leaves `self.pending` untouched and still listening.
        pending.connect().await?;

        // Create the replacement before consuming the connected instance. Taking
        // first and replacing afterwards would leave the endpoint with no instance
        // if creation failed, silently unlistening the daemon while it kept
        // running. Ordering it this way makes that failure recoverable.
        let next = create_instance(self.address.as_str(), &self.sddl, false).map_err(|err| {
            io::Error::new(
                err.kind(),
                format!("cannot create a replacement pipe instance: {err}"),
            )
        })?;

        Ok(self
            .pending
            .replace(next)
            .expect("pending instance was present immediately above"))
    }

    /// The address this endpoint is listening on.
    #[must_use]
    pub fn address(&self) -> &EndpointAddress {
        &self.address
    }
}

/// Connected pipe on the client side.
pub type ClientStream = tokio::net::windows::named_pipe::NamedPipeClient;

/// Connects to a running daemon.
///
/// # Errors
///
/// Returns [`EndpointError::NotRunning`] when no daemon holds the pipe.
pub async fn connect(address: &EndpointAddress) -> Result<ClientStream, EndpointError> {
    loop {
        match ClientOptions::new().open(address.as_str()) {
            Ok(client) => return Ok(client),
            Err(err) if err.raw_os_error() == Some(ERROR_PIPE_BUSY as i32) => {
                // Every instance is currently serving a client. The daemon creates
                // a replacement as soon as one is taken, so this resolves shortly.
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                return Err(EndpointError::NotRunning {
                    address: address.clone(),
                });
            }
            Err(err) => return Err(EndpointError::Io(err)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_denies_network_before_allowing_the_user() {
        let sddl = security_descriptor_definition("S-1-5-21-1-2-3-1001");
        let deny = sddl
            .find("(D;;GA;;;NU)")
            .expect("network deny entry present");
        let allow = sddl
            .find("(A;;GA;;;S-1-5-21-1-2-3-1001)")
            .expect("user allow entry present");
        assert!(deny < allow, "deny entry must precede allow entry: {sddl}");
        assert!(sddl.starts_with("D:P"), "DACL must be protected: {sddl}");
    }

    #[test]
    fn descriptor_parses() {
        let sid = current_user_sid().expect("current user has a sid");
        SecurityDescriptor::from_sddl(&security_descriptor_definition(&sid))
            .expect("descriptor is valid SDDL");
    }

    #[tokio::test]
    async fn additional_instances_can_be_created() {
        // The accept loop replaces each consumed instance. If a replacement cannot
        // be created, the endpoint stops listening after one connection.
        let name = format!(r"\\.\pipe\mehoyd-instancetest-{}", std::process::id());
        let sid = current_user_sid().expect("current user has a sid");
        let sddl = security_descriptor_definition(&sid);

        let _first = create_instance(&name, &sddl, true).expect("first instance");
        let second = create_instance(&name, &sddl, false);
        assert!(
            second.is_ok(),
            "replacement instance failed: {:?}",
            second.err()
        );
    }

    #[test]
    fn default_address_is_pipe_qualified_by_sid() {
        let addr = default_address().expect("default is available");
        assert!(addr.as_str().starts_with(r"\\.\pipe\mehoyd-"));
        assert!(addr.as_str().contains("S-1-"), "expected a sid: {addr}");
    }
}
