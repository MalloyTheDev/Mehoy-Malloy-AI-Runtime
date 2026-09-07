//! Bounded capture of a worker's output.
//!
//! A backend's standard error is where the useful failure evidence appears first:
//! accelerator initialisation failures, unsupported quantisations, driver load
//! errors, and allocation failures usually show up there before the process exits.
//! Discarding it means every backend failure reports only that something failed.
//!
//! The capture is bounded. A noisy or looping backend must not be able to consume
//! the daemon's memory, so this keeps the most recent lines and drops the rest. The
//! most recent lines are the ones that matter for a failure report.
//!
//! Output is diagnostic evidence, not protocol. Readiness is decided by a probe,
//! not by matching strings in here.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Which of a worker's streams a line came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogStream {
    Stdout,
    Stderr,
}

impl std::fmt::Display for LogStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Stdout => "stdout",
            Self::Stderr => "stderr",
        })
    }
}

/// One captured line.
#[derive(Debug, Clone)]
pub struct LogLine {
    pub stream: LogStream,
    pub at: Instant,
    pub text: String,
}

impl std::fmt::Display for LogLine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.stream, self.text)
    }
}

/// How many lines to retain when a spec does not say.
pub const DEFAULT_CAPTURE_LINES: usize = 512;

/// A fixed-capacity record of a worker's most recent output.
#[derive(Debug)]
pub struct WorkerLog {
    capacity: usize,
    lines: VecDeque<LogLine>,
    /// Counts everything ever recorded, including lines already dropped, so a
    /// failure report can say that earlier output was discarded rather than
    /// implying the capture is complete.
    total: u64,
}

impl WorkerLog {
    /// Creates a log retaining at most `capacity` lines.
    ///
    /// A capacity of zero is raised to one so the buffer always reports something.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            capacity,
            lines: VecDeque::with_capacity(capacity),
            total: 0,
        }
    }

    fn push(&mut self, line: LogLine) {
        if self.lines.len() == self.capacity {
            self.lines.pop_front();
        }
        self.lines.push_back(line);
        self.total = self.total.saturating_add(1);
    }

    /// The retained lines, oldest first.
    #[must_use]
    pub fn lines(&self) -> Vec<LogLine> {
        self.lines.iter().cloned().collect()
    }

    /// Retained lines from one stream, oldest first.
    #[must_use]
    pub fn lines_from(&self, stream: LogStream) -> Vec<LogLine> {
        self.lines
            .iter()
            .filter(|line| line.stream == stream)
            .cloned()
            .collect()
    }

    /// How many lines were recorded in total, including any since dropped.
    #[must_use]
    pub fn total_recorded(&self) -> u64 {
        self.total
    }

    /// How many lines were dropped because the buffer was full.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.total.saturating_sub(self.lines.len() as u64)
    }

    /// Renders the retained output for a failure report.
    #[must_use]
    pub fn render(&self) -> String {
        let mut out = String::new();
        if self.dropped() > 0 {
            out.push_str(&format!(
                "... {} earlier line(s) dropped ...\n",
                self.dropped()
            ));
        }
        for line in &self.lines {
            out.push_str(&line.to_string());
            out.push('\n');
        }
        out
    }
}

/// A shared handle to a worker's captured output.
#[derive(Debug, Clone)]
pub struct LogHandle {
    inner: Arc<Mutex<WorkerLog>>,
}

impl LogHandle {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(WorkerLog::with_capacity(capacity))),
        }
    }

    pub(crate) fn record(&self, stream: LogStream, text: String) {
        if let Ok(mut log) = self.inner.lock() {
            log.push(LogLine {
                stream,
                at: Instant::now(),
                text,
            });
        }
    }

    /// Retained lines, oldest first.
    #[must_use]
    pub fn lines(&self) -> Vec<LogLine> {
        self.inner.lock().map(|log| log.lines()).unwrap_or_default()
    }

    /// Retained lines from one stream.
    #[must_use]
    pub fn lines_from(&self, stream: LogStream) -> Vec<LogLine> {
        self.inner
            .lock()
            .map(|log| log.lines_from(stream))
            .unwrap_or_default()
    }

    /// Renders the retained output for a failure report.
    #[must_use]
    pub fn render(&self) -> String {
        self.inner
            .lock()
            .map(|log| log.render())
            .unwrap_or_else(|_| "<worker log unavailable>".to_owned())
    }

    /// How many lines were dropped because the buffer was full.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.inner.lock().map(|log| log.dropped()).unwrap_or(0)
    }
}

impl Default for LogHandle {
    fn default() -> Self {
        Self::new(DEFAULT_CAPTURE_LINES)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retains_only_the_most_recent_lines() {
        let log = LogHandle::new(3);
        for index in 0..10 {
            log.record(LogStream::Stdout, format!("line {index}"));
        }
        let lines = log.lines();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0].text, "line 7");
        assert_eq!(lines[2].text, "line 9");
    }

    #[test]
    fn a_noisy_worker_cannot_grow_the_buffer() {
        let log = LogHandle::new(4);
        for index in 0..100_000 {
            log.record(LogStream::Stderr, format!("noise {index}"));
        }
        assert_eq!(log.lines().len(), 4);
        assert_eq!(log.dropped(), 99_996);
    }

    #[test]
    fn dropped_lines_are_reported_rather_than_hidden() {
        let log = LogHandle::new(2);
        for index in 0..5 {
            log.record(LogStream::Stdout, format!("line {index}"));
        }
        let rendered = log.render();
        assert!(rendered.contains("3 earlier line(s) dropped"), "{rendered}");
        assert!(rendered.contains("line 4"), "{rendered}");
    }

    #[test]
    fn streams_are_distinguishable() {
        let log = LogHandle::new(10);
        log.record(LogStream::Stdout, "out".to_owned());
        log.record(LogStream::Stderr, "err".to_owned());
        assert_eq!(log.lines_from(LogStream::Stderr).len(), 1);
        assert_eq!(log.lines_from(LogStream::Stderr)[0].text, "err");
        assert_eq!(log.lines_from(LogStream::Stdout)[0].text, "out");
    }

    #[test]
    fn zero_capacity_still_records_something() {
        let log = LogHandle::new(0);
        log.record(LogStream::Stdout, "only".to_owned());
        assert_eq!(log.lines().len(), 1);
    }
}
