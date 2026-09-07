//! A controllable stand-in for a real inference worker.
//!
//! Supervision has to be proven before an actual engine is introduced. Debugging a
//! state machine, a GPU driver, a model container, and a process protocol at the
//! same time is how a lifecycle problem gets misattributed to the engine.
//!
//! This program does nothing useful. It exists to fail in specific, requested ways
//! so that every edge of the supervision contract can be exercised deterministically:
//!
//! ```text
//! --ready                 announce readiness and stay alive
//! --startup-delay <ms>    wait before announcing readiness
//! --never-ready           stay alive and never announce readiness
//! --exit-immediately      exit before announcing anything
//! --crash-after-ready <ms>  announce readiness, then exit non-zero
//! --ignore-shutdown       announce readiness and ignore the stop request
//! --hang-after-ready      announce readiness and stop reading input entirely
//! ```
//!
//! This is an example rather than a crate so that `cargo test` builds it
//! automatically: the supervision tests execute this binary, and a helper they
//! cannot rely on being built is a gate that passes only by luck.

use std::io::{BufRead, Write};
use std::process::ExitCode;
use std::time::Duration;

use mehoy_core::worker::process::{READY_MARKER, SHUTDOWN_COMMAND};

/// Printed on the way out, saying why. Lets a test distinguish a delivered stop
/// request from an exit caused by input simply being closed.
const STOPPING_MARKER: &str = "MEHOY-WORKER-STOPPING";

const USAGE: &str = "\
mehoy-dummy-worker, a test stand-in for a supervised worker

Options:
  --ready                    Announce readiness and wait for a stop request
  --startup-delay <ms>       Delay before announcing readiness
  --never-ready              Never announce readiness
  --exit-immediately         Exit without announcing readiness
  --crash-after-ready <ms>   Announce readiness, then exit non-zero after a delay
  --ignore-shutdown          Announce readiness, then ignore the stop request
  --hang-after-ready         Announce readiness, then stop reading input
  -h, --help                 Print this message
";

/// How the worker should behave once it has announced readiness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Behaviour {
    /// Stop politely when asked.
    Obey,
    /// Never announce readiness.
    NeverReady,
    /// Exit before announcing anything.
    ExitImmediately,
    /// Announce readiness, then exit non-zero after a delay.
    CrashAfterReady { after: Duration },
    /// Announce readiness, then ignore stop requests.
    IgnoreShutdown,
    /// Announce readiness, then stop reading input entirely.
    HangAfterReady,
}

#[derive(Debug)]
struct Config {
    behaviour: Behaviour,
    startup_delay: Duration,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Option<Config>, String> {
    let mut behaviour = Behaviour::Obey;
    let mut startup_delay = Duration::ZERO;
    let mut args = args.into_iter();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(None),
            "--ready" => behaviour = Behaviour::Obey,
            "--never-ready" => behaviour = Behaviour::NeverReady,
            "--exit-immediately" => behaviour = Behaviour::ExitImmediately,
            "--ignore-shutdown" => behaviour = Behaviour::IgnoreShutdown,
            "--hang-after-ready" => behaviour = Behaviour::HangAfterReady,
            "--crash-after-ready" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--crash-after-ready requires milliseconds".to_owned())?;
                behaviour = Behaviour::CrashAfterReady {
                    after: parse_millis(&value)?,
                };
            }
            "--startup-delay" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--startup-delay requires milliseconds".to_owned())?;
                startup_delay = parse_millis(&value)?;
            }
            other => return Err(format!("unrecognised argument: {other}")),
        }
    }

    Ok(Some(Config {
        behaviour,
        startup_delay,
    }))
}

fn parse_millis(value: &str) -> Result<Duration, String> {
    value
        .parse::<u64>()
        .map(Duration::from_millis)
        .map_err(|_| format!("expected a whole number of milliseconds, got {value:?}"))
}

fn announce_ready() {
    let mut stdout = std::io::stdout();
    // Readiness is only meaningful once the supervisor can actually see it, so
    // this must be flushed rather than left in a buffer.
    let _ = writeln!(stdout, "{READY_MARKER} pid={}", std::process::id());
    let _ = stdout.flush();
}

fn main() -> ExitCode {
    let config = match parse_args(std::env::args().skip(1)) {
        Ok(Some(config)) => config,
        Ok(None) => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(message) => {
            eprintln!("mehoy-dummy-worker: {message}");
            eprint!("{USAGE}");
            return ExitCode::from(2);
        }
    };

    if config.behaviour == Behaviour::ExitImmediately {
        return ExitCode::from(7);
    }

    if !config.startup_delay.is_zero() {
        std::thread::sleep(config.startup_delay);
    }

    if config.behaviour == Behaviour::NeverReady {
        // Alive, silent, and useless: the case a startup deadline exists for.
        loop {
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    announce_ready();

    match config.behaviour {
        Behaviour::CrashAfterReady { after } => {
            std::thread::sleep(after);
            ExitCode::from(9)
        }
        Behaviour::HangAfterReady => {
            // Never reads input again, so a stop request is never even seen.
            loop {
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        Behaviour::IgnoreShutdown => {
            // Reads and discards input, so the stop request is seen and refused.
            let stdin = std::io::stdin();
            for line in stdin.lock().lines() {
                if line.is_err() {
                    break;
                }
            }
            loop {
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        Behaviour::Obey => {
            let stdin = std::io::stdin();
            for line in stdin.lock().lines() {
                match line {
                    Ok(line) if line.trim() == SHUTDOWN_COMMAND => {
                        // Announced so a test can tell a real stop request apart
                        // from an exit caused merely by input being closed.
                        let mut stdout = std::io::stdout();
                        let _ = writeln!(stdout, "{STOPPING_MARKER} reason=command");
                        let _ = stdout.flush();
                        return ExitCode::SUCCESS;
                    }
                    Ok(_) => {}
                    // Input closed, which is the weaker stop signal.
                    Err(_) => return ExitCode::SUCCESS,
                }
            }
            let mut stdout = std::io::stdout();
            let _ = writeln!(stdout, "{STOPPING_MARKER} reason=input-closed");
            let _ = stdout.flush();
            ExitCode::SUCCESS
        }
        Behaviour::NeverReady | Behaviour::ExitImmediately => unreachable!("handled above"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn default_behaviour_obeys_a_stop_request() {
        let config = parse_args(args(&["--ready"]))
            .expect("parses")
            .expect("not help");
        assert_eq!(config.behaviour, Behaviour::Obey);
        assert!(config.startup_delay.is_zero());
    }

    #[test]
    fn startup_delay_is_parsed_in_milliseconds() {
        let config = parse_args(args(&["--ready", "--startup-delay", "250"]))
            .expect("parses")
            .expect("not help");
        assert_eq!(config.startup_delay, Duration::from_millis(250));
    }

    #[test]
    fn crash_after_ready_carries_its_delay() {
        let config = parse_args(args(&["--crash-after-ready", "10"]))
            .expect("parses")
            .expect("not help");
        assert_eq!(
            config.behaviour,
            Behaviour::CrashAfterReady {
                after: Duration::from_millis(10)
            }
        );
    }

    #[test]
    fn every_failure_mode_is_reachable_from_the_command_line() {
        let cases = [
            ("--never-ready", Behaviour::NeverReady),
            ("--exit-immediately", Behaviour::ExitImmediately),
            ("--ignore-shutdown", Behaviour::IgnoreShutdown),
            ("--hang-after-ready", Behaviour::HangAfterReady),
        ];
        for (flag, expected) in cases {
            let config = parse_args(args(&[flag]))
                .expect("parses")
                .expect("not help");
            assert_eq!(config.behaviour, expected, "flag {flag}");
        }
    }

    #[test]
    fn bad_durations_are_rejected_rather_than_defaulted() {
        let err = parse_args(args(&["--startup-delay", "soon"])).expect_err("must fail");
        assert!(err.contains("milliseconds"), "{err}");
    }

    #[test]
    fn unknown_arguments_are_rejected() {
        let err = parse_args(args(&["--explode"])).expect_err("must fail");
        assert!(err.contains("--explode"), "{err}");
    }
}
