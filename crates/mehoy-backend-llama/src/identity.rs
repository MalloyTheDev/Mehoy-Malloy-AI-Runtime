//! Identifying which build of the backend is actually running.
//!
//! "llama.cpp failed" is not a useful failure report. The project moves quickly,
//! and behaviour differs between revisions, so a report has to say which build was
//! involved or it cannot be acted on or reproduced.
//!
//! The version banner is written to standard error, not standard output, and the
//! process exits successfully. Reading the wrong stream yields nothing at all,
//! which would look like an unidentifiable backend rather than a mistake here.

use std::fmt;
use std::path::{Path, PathBuf};

/// Which backend family a worker belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendFamily {
    LlamaCpp,
}

impl fmt::Display for BackendFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::LlamaCpp => "llama.cpp",
        })
    }
}

/// Everything known about the backend executable in use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendIdentity {
    pub family: BackendFamily,
    /// The executable that was interrogated.
    pub executable: PathBuf,
    /// Build number, when the banner reported one.
    pub build: Option<String>,
    /// Source revision, when the banner reported one.
    pub commit: Option<String>,
    /// The banner as printed, kept verbatim so nothing is lost to parsing.
    pub banner: String,
}

impl fmt::Display for BackendIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.family)?;
        if let Some(build) = &self.build {
            write!(f, " build {build}")?;
        }
        if let Some(commit) = &self.commit {
            write!(f, " ({commit})")?;
        }
        write!(f, " at {}", self.executable.display())
    }
}

/// Extracts identity from a version banner.
///
/// The expected shape is a line such as `version: 9010 (d05fe1d7d)`. Anything that
/// does not match leaves the structured fields empty rather than failing: an
/// unparsed banner is still evidence, and refusing to run against an unfamiliar
/// build would make this brittle against the exact upstream churn it exists to
/// record.
#[must_use]
pub fn parse_banner(executable: &Path, banner: &str) -> BackendIdentity {
    let mut build = None;
    let mut commit = None;

    for line in banner.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("version:") else {
            continue;
        };
        let rest = rest.trim();
        let (number, remainder) = rest
            .split_once(' ')
            .map_or((rest, ""), |(number, remainder)| (number, remainder));
        if !number.is_empty() {
            build = Some(number.to_owned());
        }
        commit = remainder
            .trim()
            .strip_prefix('(')
            .and_then(|value| value.strip_suffix(')'))
            .filter(|value| !value.is_empty())
            .map(str::to_owned);
        break;
    }

    BackendIdentity {
        family: BackendFamily::LlamaCpp,
        executable: executable.to_path_buf(),
        build,
        commit,
        banner: banner.trim().to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exe() -> PathBuf {
        PathBuf::from("llama-server")
    }

    #[test]
    fn parses_a_real_banner() {
        // Captured from llama-server on this machine. Backend loader lines precede
        // the version line and must not confuse the parser.
        let banner = "load_backend: loaded RPC backend from ggml-rpc.dll\n\
                      load_backend: loaded Vulkan backend from ggml-vulkan.dll\n\
                      version: 9010 (d05fe1d7d)\n\
                      built with Clang 19.1.5 for Windows x86_64";
        let identity = parse_banner(&exe(), banner);
        assert_eq!(identity.build.as_deref(), Some("9010"));
        assert_eq!(identity.commit.as_deref(), Some("d05fe1d7d"));
        assert_eq!(identity.family, BackendFamily::LlamaCpp);
    }

    #[test]
    fn keeps_the_banner_verbatim() {
        let banner = "version: 9010 (d05fe1d7d)\nbuilt with Clang";
        let identity = parse_banner(&exe(), banner);
        assert!(identity.banner.contains("built with Clang"));
    }

    #[test]
    fn an_unrecognised_banner_is_still_recorded() {
        // Refusing to run against an unfamiliar build would be brittle against the
        // upstream churn this exists to record.
        let identity = parse_banner(&exe(), "some future format");
        assert_eq!(identity.build, None);
        assert_eq!(identity.commit, None);
        assert_eq!(identity.banner, "some future format");
    }

    #[test]
    fn a_version_without_a_commit_still_yields_a_build() {
        let identity = parse_banner(&exe(), "version: 1234");
        assert_eq!(identity.build.as_deref(), Some("1234"));
        assert_eq!(identity.commit, None);
    }

    #[test]
    fn display_names_the_build_and_executable() {
        let identity = parse_banner(&exe(), "version: 9010 (d05fe1d7d)");
        let shown = identity.to_string();
        assert!(shown.contains("llama.cpp"), "{shown}");
        assert!(shown.contains("9010"), "{shown}");
        assert!(shown.contains("d05fe1d7d"), "{shown}");
        assert!(shown.contains("llama-server"), "{shown}");
    }
}
