//! `mehoy`, the command-line client for the Mehoy runtime daemon.

use std::process::ExitCode;

use mehoy_cli::{Client, ClientError};
use mehoy_core::transport::{ENDPOINT_ENV, EndpointAddress, EndpointError};
use mehoy_protocol::{HealthStatus, ProtocolVersion};

const USAGE: &str = "\
mehoy, the Mehoy runtime client

Usage:
  mehoy [options] <command>

Commands:
  health     Report whether a daemon is serving
  runtime    Report the daemon's identity and protocol version

Options:
  --endpoint <address>  Talk to this endpoint instead of the per-user default
  -h, --help            Print this message
  -V, --version         Print the version

Environment:
  MEHOY_ENDPOINT        Same as --endpoint, used when the flag is absent

Exit codes:
  0  success
  1  the request failed
  2  the command line was invalid
  3  no daemon is listening
";

/// Exit code reserved for a daemon that is not running, so scripts can tell that
/// apart from a request that failed for another reason.
const EXIT_NOT_RUNNING: u8 = 3;

#[derive(Debug, PartialEq, Eq)]
enum Command {
    Health,
    Runtime,
}

#[derive(Debug, PartialEq, Eq)]
enum Invocation {
    Run {
        command: Command,
        endpoint: Option<String>,
    },
    Help,
    Version,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Invocation, String> {
    let mut endpoint = None;
    let mut command = None;
    let mut args = args.into_iter();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(Invocation::Help),
            "-V" | "--version" => return Ok(Invocation::Version),
            "--endpoint" => {
                endpoint = Some(
                    args.next()
                        .ok_or_else(|| "--endpoint requires an address".to_owned())?,
                );
            }
            "health" if command.is_none() => command = Some(Command::Health),
            "runtime" if command.is_none() => command = Some(Command::Runtime),
            other if other.starts_with('-') => {
                return Err(format!("unrecognised option: {other}"));
            }
            other => return Err(format!("unrecognised command: {other}")),
        }
    }

    command
        .map(|command| Invocation::Run { command, endpoint })
        .ok_or_else(|| "no command given".to_owned())
}

fn main() -> ExitCode {
    match parse_args(std::env::args().skip(1)) {
        Ok(Invocation::Help) => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(Invocation::Version) => {
            println!("mehoy {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Ok(Invocation::Run { command, endpoint }) => run(&command, endpoint),
        Err(message) => {
            eprintln!("mehoy: {message}");
            eprint!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn run(command: &Command, endpoint: Option<String>) -> ExitCode {
    let client = match endpoint {
        Some(value) => Client::new(EndpointAddress::new(value)),
        None => match Client::resolve() {
            Ok(client) => client,
            Err(err) => {
                eprintln!("mehoy: {err}");
                return ExitCode::FAILURE;
            }
        },
    };

    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            eprintln!("mehoy: cannot start the async runtime: {err}");
            return ExitCode::FAILURE;
        }
    };

    runtime.block_on(async move {
        let result = match command {
            Command::Health => client.health().await.map(|health| match health.status {
                HealthStatus::Ok => "ok".to_owned(),
                HealthStatus::ShuttingDown => "shutting down".to_owned(),
            }),
            Command::Runtime => client.runtime().await.map(|info| {
                let client_version = ProtocolVersion::CURRENT;
                let compatibility = if client_version.is_compatible_with(info.protocol) {
                    ""
                } else {
                    "  (incompatible with this client)"
                };
                format!(
                    "{} {}\nprotocol {}.{}{}",
                    info.runtime.name,
                    info.runtime.version,
                    info.protocol.major,
                    info.protocol.minor,
                    compatibility
                )
            }),
        };

        match result {
            Ok(output) => {
                println!("{output}");
                ExitCode::SUCCESS
            }
            Err(err @ ClientError::Transport(EndpointError::NotRunning { .. })) => {
                eprintln!("mehoy: {err}");
                eprintln!("mehoy: start one with `mehoyd`, or set {ENDPOINT_ENV}");
                ExitCode::from(EXIT_NOT_RUNNING)
            }
            Err(err) => {
                eprintln!("mehoy: {err}");
                ExitCode::FAILURE
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn commands_parse() {
        assert_eq!(
            parse_args(args(&["health"])).expect("parses"),
            Invocation::Run {
                command: Command::Health,
                endpoint: None
            }
        );
        assert_eq!(
            parse_args(args(&["runtime"])).expect("parses"),
            Invocation::Run {
                command: Command::Runtime,
                endpoint: None
            }
        );
    }

    #[test]
    fn endpoint_may_precede_or_follow_the_command() {
        let before = parse_args(args(&["--endpoint", "x", "health"])).expect("parses");
        let after = parse_args(args(&["health", "--endpoint", "x"])).expect("parses");
        assert_eq!(before, after);
    }

    #[test]
    fn missing_command_is_an_error() {
        let err = parse_args(args(&[])).expect_err("must fail");
        assert!(err.contains("no command"), "{err}");
    }

    #[test]
    fn unknown_command_and_option_are_distinguished() {
        let command = parse_args(args(&["frobnicate"])).expect_err("must fail");
        assert!(command.contains("unrecognised command"), "{command}");
        let option = parse_args(args(&["--frobnicate"])).expect_err("must fail");
        assert!(option.contains("unrecognised option"), "{option}");
    }

    #[test]
    fn second_command_is_rejected_rather_than_silently_ignored() {
        let err = parse_args(args(&["health", "runtime"])).expect_err("must fail");
        assert!(err.contains("runtime"), "{err}");
    }

    #[test]
    fn help_and_version_short_circuit() {
        assert_eq!(
            parse_args(args(&["health", "--help"])).expect("parses"),
            Invocation::Help
        );
        assert_eq!(
            parse_args(args(&["-V"])).expect("parses"),
            Invocation::Version
        );
    }
}
