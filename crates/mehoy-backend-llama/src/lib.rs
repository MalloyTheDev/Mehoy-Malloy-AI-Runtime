//! llama.cpp execution backend.
//!
//! This crate exists as a separate crate rather than a module so the boundary is
//! enforced by the compiler: `mehoy-core` cannot reference anything here, so
//! llama.cpp's flags, endpoints, and vocabulary cannot leak into the generic
//! runtime. The generic layer knows spawn, ready, shutdown, wait, and failure.
//! This layer knows `--model`, `--host`, `--port`, and `/health`.
//!
//! # Scope
//!
//! Starting, identifying, supervising, and stopping a backend process. There is no
//! model registry, scheduling, provider routing, or compatibility endpoint here,
//! and none belongs here.
//!
//! # What is not done yet
//!
//! Installing or downloading a backend. Execution and installation are separate
//! responsibilities, and llama.cpp's build variants differ enough per accelerator
//! that distribution is its own subsystem. The executable is supplied explicitly.

pub mod channel;
pub mod compatibility;
pub mod health;
pub mod identity;

pub use channel::{BackendChannel, ChannelSecret, SecretFile};
pub use compatibility::{Compatibility, ModelDescriptor};
pub use health::{CredentialState, Readiness, StartupPhase};
pub use identity::{BackendFamily, BackendIdentity};

use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Stdio;

use mehoy_core::id::WorkerId;
use mehoy_core::worker::log::DEFAULT_CAPTURE_LINES;
use mehoy_core::worker::{
    Deadlines, ProbeOutcome, ProcessWorker, StopProtocol, WorkerError, WorkerHandle, WorkerReady,
    WorkerSpec,
};

/// Environment variable naming the backend executable.
pub const EXECUTABLE_ENV: &str = "MEHOY_LLAMA_SERVER";

/// Where per-worker credential files are written.
///
/// The system temporary directory is per-user on the platforms targeted here, and
/// the file itself is created owner-only regardless.
fn secret_directory() -> PathBuf {
    std::env::temp_dir().join("mehoy-backend")
}

/// How many layers to place on an accelerator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuLayerPolicy {
    /// Run entirely on the processor.
    CpuOnly,
    /// Place a specific number of layers on the accelerator.
    Layers(u32),
}

/// What to start a llama.cpp worker with.
///
/// Only backend concerns appear here. The generic worker layer never sees these
/// fields.
#[derive(Debug, Clone)]
pub struct LlamaCppWorkerSpec {
    pub model_path: PathBuf,
    pub context_size: Option<u32>,
    pub gpu_layers: Option<GpuLayerPolicy>,
    pub deadlines: Deadlines,
}

impl LlamaCppWorkerSpec {
    /// A specification for a model with default settings.
    #[must_use]
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            context_size: None,
            gpu_layers: None,
            deadlines: Deadlines::default(),
        }
    }
}

/// Why a backend could not be used.
#[derive(Debug)]
pub enum BackendError {
    /// No backend executable is configured or the configured one is absent.
    Unavailable { reason: String },
    /// The backend executable could not be interrogated for its identity.
    Unidentifiable { executable: PathBuf, reason: String },
    /// A private channel for the backend could not be established.
    Channel(std::io::Error),
    /// The backend rejected the secret it was given.
    Rejected { detail: String },
    /// Supervision reported a failure.
    ///
    /// Boxed because it carries the backend's captured output, which is large and
    /// would otherwise bloat every `Result` in this crate.
    Worker(Box<WorkerFailure>),
}

/// A supervision failure together with the evidence needed to act on it.
///
/// The output is carried because the reason a backend failed to start is almost
/// always in what it printed rather than in its exit status, and the identity is
/// carried because upstream behaviour differs between builds.
#[derive(Debug)]
pub struct WorkerFailure {
    pub source: WorkerError,
    pub identity: Option<BackendIdentity>,
    pub output: String,
}

impl BackendError {
    fn worker(source: WorkerError, identity: Option<BackendIdentity>, output: String) -> Self {
        Self::Worker(Box::new(WorkerFailure {
            source,
            identity,
            output,
        }))
    }
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable { reason } => write!(f, "no llama.cpp backend available: {reason}"),
            Self::Unidentifiable { executable, reason } => write!(
                f,
                "cannot identify llama.cpp backend {}: {reason}",
                executable.display()
            ),
            Self::Channel(err) => write!(f, "cannot establish a private backend channel: {err}"),
            Self::Rejected { detail } => {
                write!(f, "backend rejected its own credentials: {detail}")
            }
            Self::Worker(failure) => {
                write!(f, "{}", failure.source)?;
                if let Some(identity) = &failure.identity {
                    write!(f, "\nbackend: {identity}")?;
                }
                if !failure.output.trim().is_empty() {
                    write!(f, "\nbackend output:\n{}", failure.output)?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for BackendError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Channel(err) => Some(err),
            Self::Worker(failure) => Some(&failure.source),
            _ => None,
        }
    }
}

/// A configured llama.cpp backend.
#[derive(Debug, Clone)]
pub struct LlamaCppBackend {
    executable: PathBuf,
}

impl LlamaCppBackend {
    /// Uses a specific executable.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::Unavailable`] when the path does not name an
    /// existing file. Checking now turns a confusing spawn failure later into a
    /// clear configuration error.
    pub fn new(executable: impl Into<PathBuf>) -> Result<Self, BackendError> {
        let executable = executable.into();
        if !executable.is_file() {
            return Err(BackendError::Unavailable {
                reason: format!("{} is not an existing file", executable.display()),
            });
        }
        Ok(Self { executable })
    }

    /// Uses the executable named by [`EXECUTABLE_ENV`].
    ///
    /// Resolution is deliberately explicit rather than searching `PATH`. "Whatever
    /// happens to be installed" is not a contract a runtime can reason about, and
    /// upstream behaviour differs enough between builds that the exact executable
    /// matters.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::Unavailable`] when the variable is unset or names a
    /// path that does not exist.
    pub fn from_env() -> Result<Self, BackendError> {
        let configured =
            std::env::var_os(EXECUTABLE_ENV).ok_or_else(|| BackendError::Unavailable {
                reason: format!("{EXECUTABLE_ENV} is not set"),
            })?;
        if configured.is_empty() {
            return Err(BackendError::Unavailable {
                reason: format!("{EXECUTABLE_ENV} is empty"),
            });
        }
        Self::new(PathBuf::from(configured))
    }

    /// The executable in use.
    #[must_use]
    pub fn executable(&self) -> &Path {
        &self.executable
    }

    /// Interrogates the executable for its build identity.
    ///
    /// The version banner is written to standard error, so that is the stream read.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::Unidentifiable`] when the executable cannot be run.
    pub async fn identify(&self) -> Result<BackendIdentity, BackendError> {
        let output = tokio::process::Command::new(&self.executable)
            .arg("--version")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|err| BackendError::Unidentifiable {
                executable: self.executable.clone(),
                reason: err.to_string(),
            })?;

        // The banner goes to standard error. Standard output is checked too so a
        // future build that moves it does not silently yield an empty identity.
        let mut banner = String::from_utf8_lossy(&output.stderr).into_owned();
        if !banner.contains("version:") {
            let out = String::from_utf8_lossy(&output.stdout);
            if out.contains("version:") {
                banner = out.into_owned();
            }
        }

        Ok(identity::parse_banner(&self.executable, &banner))
    }

    /// Builds the command line for a worker.
    ///
    /// Separated from spawning so the security-relevant flags can be asserted
    /// directly, without starting a process or loading a model.
    #[must_use]
    pub fn command_line(
        &self,
        spec: &LlamaCppWorkerSpec,
        channel: &BackendChannel,
        secret_file: &SecretFile,
    ) -> Vec<String> {
        let mut args = vec![
            "--model".to_owned(),
            spec.model_path.display().to_string(),
            // Loopback only. The backend must never be reachable off-host.
            "--host".to_owned(),
            channel.host(),
            "--port".to_owned(),
            channel.port().to_string(),
            // The credential is passed by file, never as an argument. Process
            // arguments are readable by other local accounts, which would put the
            // secret in reach of every user on the machine.
            "--api-key-file".to_owned(),
            secret_file.path().display().to_string(),
            // The backend is private runtime plumbing, not a user-facing surface.
            "--no-webui".to_owned(),
        ];
        if let Some(context) = spec.context_size {
            args.push("--ctx-size".to_owned());
            args.push(context.to_string());
        }
        match spec.gpu_layers {
            Some(GpuLayerPolicy::CpuOnly) => {
                args.push("--n-gpu-layers".to_owned());
                args.push("0".to_owned());
            }
            Some(GpuLayerPolicy::Layers(count)) => {
                args.push("--n-gpu-layers".to_owned());
                args.push(count.to_string());
            }
            None => {}
        }
        args
    }

    /// Starts a backend and waits for it to report a serving state.
    ///
    /// The returned readiness is [`Readiness::BackendReady`] and nothing stronger.
    /// It does not establish that generation works.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::Worker`] carrying the backend's captured output when
    /// startup fails, since the reason is almost always in what the backend printed
    /// rather than in its exit status.
    pub async fn start(
        &self,
        id: WorkerId,
        spec: &LlamaCppWorkerSpec,
    ) -> Result<RunningBackend, BackendError> {
        let identity = self.identify().await.ok();
        let channel = BackendChannel::reserve().map_err(BackendError::Channel)?;
        debug_assert!(channel.is_loopback(), "backend channel must be loopback");

        // Kept alive until the backend has started and read it, then removed.
        let secret_file = SecretFile::create(&secret_directory(), channel.secret())
            .map_err(BackendError::Channel)?;

        let supervisor = ProcessWorker;
        let worker_spec = WorkerSpec {
            id,
            program: self.executable.clone(),
            args: self.command_line(spec, &channel, &secret_file),
            deadlines: spec.deadlines,
            capture_lines: DEFAULT_CAPTURE_LINES,
            // The backend implements no graceful stop protocol: a stop request on
            // its standard input is simply ignored. Measured on build 9010, where it
            // continued serving until its shutdown deadline expired. Termination is
            // therefore the defined way to stop it, not a fallback.
            stop: StopProtocol::Terminate,
        };

        let mut handle = supervisor
            .spawn(worker_spec)
            .await
            .map_err(|source| BackendError::worker(source, identity.clone(), String::new()))?;

        let probe_channel = channel.clone();
        let ready = supervisor
            .wait_ready_with(&mut handle, move || {
                let channel = probe_channel.clone();
                async move {
                    match health::probe(&channel).await {
                        Ok(StartupPhase::BackendReady) => Ok(ProbeOutcome::Ready(WorkerReady {
                            announcement: format!("{}", StartupPhase::BackendReady),
                        })),
                        Ok(_) => Ok(ProbeOutcome::NotYet),
                        Err(err) => Err(WorkerError::Io(std::io::Error::other(err.to_string()))),
                    }
                }
            })
            .await;

        match ready {
            Ok(_) => {
                // The health endpoint is not authenticated on the builds measured, so
                // reaching it proves the backend is serving and nothing about whether
                // the runtime can actually talk to it. Without this check a
                // misconfigured secret would produce a backend reported as ready that
                // rejects the first real request. ADR-0005 requires a refused
                // credential to be terminal, which is only possible if it is tested.
                match health::probe_credential(&channel).await {
                    Ok(CredentialState::Accepted) => Ok(RunningBackend {
                        handle,
                        channel,
                        identity,
                        readiness: Readiness::BackendReady,
                    }),
                    Ok(CredentialState::Refused { status }) => {
                        let _ = ProcessWorker.shutdown(&mut handle).await;
                        Err(BackendError::Rejected {
                            detail: format!(
                                "the backend refused the runtime's own credential with status                                  {status}; it is serving but unusable"
                            ),
                        })
                    }
                    Err(err) => {
                        let _ = ProcessWorker.shutdown(&mut handle).await;
                        Err(BackendError::Rejected {
                            detail: format!(
                                "cannot confirm the backend accepts its credential: {err}"
                            ),
                        })
                    }
                }
            }
            Err(source) => {
                handle.flush_output().await;
                let output = handle.log().render();
                Err(BackendError::worker(source, identity, output))
            }
        }
    }
}

/// A started backend.
#[derive(Debug)]
pub struct RunningBackend {
    handle: WorkerHandle,
    channel: BackendChannel,
    identity: Option<BackendIdentity>,
    readiness: Readiness,
}

impl RunningBackend {
    /// The supervised worker, for lifecycle operations.
    #[must_use]
    pub fn handle(&self) -> &WorkerHandle {
        &self.handle
    }

    /// Mutable access to the supervised worker.
    pub fn handle_mut(&mut self) -> &mut WorkerHandle {
        &mut self.handle
    }

    /// The private channel to this backend.
    ///
    /// Never given to a client. The client's contract is the runtime's own local
    /// endpoint, and the backend's address is an implementation detail behind it.
    #[must_use]
    pub fn channel(&self) -> &BackendChannel {
        &self.channel
    }

    /// What is known about the backend build.
    #[must_use]
    pub fn identity(&self) -> Option<&BackendIdentity> {
        self.identity.as_ref()
    }

    /// What has actually been established.
    #[must_use]
    pub fn readiness(&self) -> Readiness {
        self.readiness
    }

    /// The backend's captured output.
    #[must_use]
    pub fn output(&self) -> String {
        self.handle.log().render()
    }

    /// Stops the backend.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::Worker`] when the backend had to be killed.
    pub async fn stop(&mut self) -> Result<(), BackendError> {
        let supervisor = ProcessWorker;
        supervisor
            .shutdown(&mut self.handle)
            .await
            .map(|_| ())
            .map_err(|source| {
                BackendError::worker(source, self.identity.clone(), self.handle.log().render())
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn channel() -> BackendChannel {
        BackendChannel::reserve().expect("reserves")
    }

    fn secret_file(channel: &BackendChannel) -> SecretFile {
        SecretFile::create(
            &std::env::temp_dir().join("mehoy-backend-test"),
            channel.secret(),
        )
        .expect("secret file is creatable")
    }

    #[test]
    fn a_missing_executable_is_reported_before_anything_is_spawned() {
        let err = LlamaCppBackend::new("definitely-not-a-real-llama-server")
            .expect_err("must be rejected");
        assert!(
            matches!(err, BackendError::Unavailable { .. }),
            "expected Unavailable, got {err}"
        );
    }

    #[test]
    fn the_command_line_binds_loopback_only() {
        let backend = LlamaCppBackend {
            executable: PathBuf::from("llama-server"),
        };
        let channel = channel();
        let file = secret_file(&channel);
        let args = backend.command_line(&LlamaCppWorkerSpec::new("model.gguf"), &channel, &file);

        let host = args
            .iter()
            .position(|arg| arg == "--host")
            .map(|index| args[index + 1].clone())
            .expect("--host is passed");
        assert_eq!(
            host, "127.0.0.1",
            "the backend must not bind a routable address"
        );
        assert!(
            !args.contains(&"0.0.0.0".to_owned()),
            "a wildcard bind would expose the backend off-host: {args:?}"
        );
    }

    #[test]
    fn the_command_line_requires_a_secret_and_disables_the_web_interface() {
        let backend = LlamaCppBackend {
            executable: PathBuf::from("llama-server"),
        };
        let channel = channel();
        let file = secret_file(&channel);
        let args = backend.command_line(&LlamaCppWorkerSpec::new("model.gguf"), &channel, &file);

        // The credential is passed by file. Process arguments are readable by other
        // local accounts, so a secret placed there would be visible machine-wide.
        assert!(
            args.contains(&"--api-key-file".to_owned()),
            "the credential must be passed by file: {args:?}"
        );
        assert!(
            !args.iter().any(|arg| arg == channel.secret().expose()),
            "the credential leaked into the process arguments"
        );
        assert!(
            args.contains(&"--no-webui".to_owned()),
            "the backend is private plumbing, not a user surface: {args:?}"
        );
    }

    #[test]
    fn the_port_is_not_a_fixed_well_known_one() {
        let backend = LlamaCppBackend {
            executable: PathBuf::from("llama-server"),
        };
        let channel = channel();
        let file = secret_file(&channel);
        let args = backend.command_line(&LlamaCppWorkerSpec::new("model.gguf"), &channel, &file);
        let port: u16 = args
            .iter()
            .position(|arg| arg == "--port")
            .map(|index| args[index + 1].parse().expect("port is numeric"))
            .expect("--port is passed");
        assert_ne!(port, 0);
        assert_ne!(port, 8080, "the backend must not sit on the default port");
    }

    #[test]
    fn optional_settings_are_omitted_rather_than_defaulted() {
        // Passing a guessed context size or layer count would silently override the
        // model's own defaults.
        let backend = LlamaCppBackend {
            executable: PathBuf::from("llama-server"),
        };
        let channel = channel();
        let file = secret_file(&channel);
        let args = backend.command_line(&LlamaCppWorkerSpec::new("model.gguf"), &channel, &file);
        assert!(!args.contains(&"--ctx-size".to_owned()));
        assert!(!args.contains(&"--n-gpu-layers".to_owned()));
    }

    #[test]
    fn optional_settings_are_passed_when_given() {
        let backend = LlamaCppBackend {
            executable: PathBuf::from("llama-server"),
        };
        let spec = LlamaCppWorkerSpec {
            context_size: Some(4096),
            gpu_layers: Some(GpuLayerPolicy::Layers(24)),
            ..LlamaCppWorkerSpec::new("model.gguf")
        };
        let channel = channel();
        let file = secret_file(&channel);
        let args = backend.command_line(&spec, &channel, &file);
        let value_after = |flag: &str| {
            args.iter()
                .position(|arg| arg == flag)
                .map(|index| args[index + 1].clone())
        };
        assert_eq!(value_after("--ctx-size").as_deref(), Some("4096"));
        assert_eq!(value_after("--n-gpu-layers").as_deref(), Some("24"));
    }

    #[test]
    fn cpu_only_is_expressed_as_zero_layers() {
        let backend = LlamaCppBackend {
            executable: PathBuf::from("llama-server"),
        };
        let spec = LlamaCppWorkerSpec {
            gpu_layers: Some(GpuLayerPolicy::CpuOnly),
            ..LlamaCppWorkerSpec::new("model.gguf")
        };
        let channel = channel();
        let file = secret_file(&channel);
        let args = backend.command_line(&spec, &channel, &file);
        let index = args
            .iter()
            .position(|arg| arg == "--n-gpu-layers")
            .expect("layer count is passed");
        assert_eq!(args[index + 1], "0");
    }

    #[test]
    fn an_unset_environment_variable_is_reported_clearly() {
        // Not `PATH` discovery: an unset variable is a configuration error rather
        // than a reason to run whatever happens to be installed.
        unsafe { std::env::remove_var(EXECUTABLE_ENV) };
        let err = LlamaCppBackend::from_env().expect_err("must be unavailable");
        match err {
            BackendError::Unavailable { reason } => {
                assert!(reason.contains(EXECUTABLE_ENV), "{reason}");
            }
            other => panic!("expected Unavailable, got {other}"),
        }
    }
}
