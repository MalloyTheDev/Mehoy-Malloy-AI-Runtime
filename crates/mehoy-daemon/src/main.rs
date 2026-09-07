//! `mehoyd`, the Mehoy runtime daemon.

use std::process::ExitCode;

use mehoy_core::transport::{ENDPOINT_ENV, Endpoint, EndpointAddress, EndpointError};
use mehoy_daemon::{RUNTIME_VERSION, serve};

const USAGE: &str = "\
mehoyd, the Mehoy runtime daemon

Usage:
  mehoyd [options]

Options:
  --endpoint <address>  Listen on this endpoint instead of the per-user default
  -h, --help            Print this message
  -V, --version         Print the version

Environment:
  MEHOY_ENDPOINT        Same as --endpoint, used when the flag is absent
";

#[derive(Debug, PartialEq, Eq)]
enum Invocation {
    Serve { endpoint: Option<String> },
    Help,
    Version,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Invocation, String> {
    let mut endpoint = None;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(Invocation::Help),
            "-V" | "--version" => return Ok(Invocation::Version),
            "--endpoint" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--endpoint requires an address".to_owned())?;
                endpoint = Some(value);
            }
            other => return Err(format!("unrecognised argument: {other}")),
        }
    }
    Ok(Invocation::Serve { endpoint })
}

fn main() -> ExitCode {
    match parse_args(std::env::args().skip(1)) {
        Ok(Invocation::Help) => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(Invocation::Version) => {
            println!("mehoyd {RUNTIME_VERSION}");
            ExitCode::SUCCESS
        }
        Ok(Invocation::Serve { endpoint }) => run(endpoint),
        Err(message) => {
            eprintln!("mehoyd: {message}");
            eprint!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn run(endpoint: Option<String>) -> ExitCode {
    let address = match endpoint {
        Some(value) => EndpointAddress::new(value),
        None => match EndpointAddress::resolve() {
            Ok(address) => address,
            Err(err) => {
                eprintln!("mehoyd: {err}");
                return ExitCode::FAILURE;
            }
        },
    };

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            eprintln!("mehoyd: cannot start the async runtime: {err}");
            return ExitCode::FAILURE;
        }
    };

    runtime.block_on(async move {
        let endpoint = match Endpoint::bind(&address) {
            Ok(endpoint) => endpoint,
            Err(err @ EndpointError::AlreadyRunning { .. }) => {
                eprintln!("mehoyd: {err}");
                return ExitCode::from(3);
            }
            Err(err) => {
                eprintln!("mehoyd: {err}");
                eprintln!("mehoyd: set {ENDPOINT_ENV} to choose a different endpoint");
                return ExitCode::FAILURE;
            }
        };

        println!(
            "mehoyd {RUNTIME_VERSION} listening on {}",
            endpoint.address()
        );

        match serve(endpoint, shutdown_signal()).await {
            Ok(()) => {
                println!("mehoyd: shut down");
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("mehoyd: {err}");
                ExitCode::FAILURE
            }
        }
    })
}

/// Resolves when the process is asked to stop.
///
/// Unix additionally honours `SIGTERM`, which is how service managers and
/// container runtimes request a stop. Treating only Ctrl-C as a stop signal would
/// leave the daemon to be killed instead of shut down, and a kill skips endpoint
/// cleanup.
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut terminate = match signal(SignalKind::terminate()) {
            Ok(stream) => stream,
            Err(err) => {
                eprintln!("mehoyd: cannot listen for SIGTERM: {err}");
                let _ = tokio::signal::ctrl_c().await;
                return;
            }
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = terminate.recv() => {}
        }
    }

    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn no_arguments_serves_on_the_default_endpoint() {
        let parsed = parse_args(args(&[])).expect("parses");
        assert!(matches!(parsed, Invocation::Serve { endpoint: None }));
    }

    #[test]
    fn endpoint_flag_is_captured() {
        let parsed = parse_args(args(&["--endpoint", "custom"])).expect("parses");
        match parsed {
            Invocation::Serve { endpoint } => assert_eq!(endpoint.as_deref(), Some("custom")),
            _ => panic!("expected a serve invocation"),
        }
    }

    #[test]
    fn endpoint_flag_without_a_value_is_an_error() {
        let err = parse_args(args(&["--endpoint"])).expect_err("must fail");
        assert!(err.contains("requires an address"), "{err}");
    }

    #[test]
    fn unknown_arguments_are_rejected_rather_than_ignored() {
        let err = parse_args(args(&["--wat"])).expect_err("must fail");
        assert!(err.contains("--wat"), "{err}");
    }

    #[test]
    fn help_and_version_short_circuit() {
        assert!(matches!(
            parse_args(args(&["--help"])).expect("parses"),
            Invocation::Help
        ));
        assert!(matches!(
            parse_args(args(&["-V"])).expect("parses"),
            Invocation::Version
        ));
    }
}
