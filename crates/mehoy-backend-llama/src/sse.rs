//! Server-sent event framing.
//!
//! Exists so that nothing above this module ever sees a network chunk. A backend
//! writes events, but a socket delivers arbitrary byte runs, and the two have no
//! relationship: a single JSON payload can arrive across three reads, split in the
//! middle of a key, a value, or a multi-byte character. Treating a read as an event
//! would produce truncated JSON and invalid text under ordinary transport
//! behaviour, and only under load, which is the worst way to find out.
//!
//! Line splitting is done on bytes rather than on decoded text, which is safe
//! because a UTF-8 continuation byte is never 0x0A or 0x0D. A field value is
//! decoded only once its line is complete, so a character split across reads is
//! held in the buffer rather than being decoded from a partial sequence.
//!
//! Only the subset that matters here is implemented: comments, field lines, the
//! `data` field, and blank-line dispatch. Event types, identifiers, and reconnect
//! hints are parsed as fields and discarded, because nothing in this runtime uses
//! them.

/// Accumulates bytes and yields the payload of each complete event.
#[derive(Debug, Default)]
pub(crate) struct SseDecoder {
    buffer: Vec<u8>,
    data: String,
    has_data: bool,
}

impl SseDecoder {
    /// Feeds a chunk of arbitrary bytes and returns any events it completed.
    ///
    /// Returns an empty vector when the chunk did not finish an event, which is the
    /// normal case rather than an exceptional one.
    ///
    /// # Errors
    ///
    /// Returns a description when a line is not valid UTF-8. A server-sent event
    /// stream is defined as UTF-8, so this is a broken backend rather than
    /// something to recover from by substituting replacement characters.
    pub(crate) fn push(&mut self, chunk: &[u8]) -> Result<Vec<String>, String> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();

        while let Some(line) = self.take_line()? {
            if let Some(event) = self.consume(&line)? {
                events.push(event);
            }
        }

        Ok(events)
    }

    /// Whether anything is still held that never formed a complete event.
    ///
    /// A stream that ends here delivered a partial event, which is a truncation
    /// rather than a clean end.
    pub(crate) fn has_partial_event(&self) -> bool {
        !self.buffer.is_empty() || self.has_data
    }

    /// Removes one complete line, if the buffer holds one.
    ///
    /// A trailing carriage return is not treated as a line ending until the next
    /// byte is known, because a CRLF pair split across two reads would otherwise be
    /// seen as a line ending followed by a blank line, dispatching an event early.
    fn take_line(&mut self) -> Result<Option<Vec<u8>>, String> {
        let Some(index) = self.buffer.iter().position(|&b| b == b'\n' || b == b'\r') else {
            return Ok(None);
        };

        let consumed = if self.buffer[index] == b'\r' {
            match self.buffer.get(index + 1) {
                None => return Ok(None),
                Some(&b'\n') => index + 2,
                Some(_) => index + 1,
            }
        } else {
            index + 1
        };

        let line = self.buffer[..index].to_vec();
        self.buffer.drain(..consumed);
        Ok(Some(line))
    }

    /// Applies one line, returning a payload when the line dispatched an event.
    fn consume(&mut self, line: &[u8]) -> Result<Option<String>, String> {
        if line.is_empty() {
            if !self.has_data {
                return Ok(None);
            }
            self.has_data = false;
            return Ok(Some(std::mem::take(&mut self.data)));
        }

        if line[0] == b':' {
            return Ok(None);
        }

        let (field, value) = match line.iter().position(|&b| b == b':') {
            Some(colon) => {
                let raw = &line[colon + 1..];
                let value = raw.strip_prefix(b" ").unwrap_or(raw);
                (&line[..colon], value)
            }
            None => (line, &[][..]),
        };

        if field != b"data" {
            return Ok(None);
        }

        let decoded = std::str::from_utf8(value)
            .map_err(|err| format!("a data line was not valid UTF-8: {err}"))?;
        if self.has_data {
            self.data.push('\n');
        }
        self.data.push_str(decoded);
        self.has_data = true;
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drain(decoder: &mut SseDecoder, chunk: &[u8]) -> Vec<String> {
        decoder.push(chunk).expect("well formed")
    }

    #[test]
    fn a_whole_event_in_one_chunk_is_returned() {
        let mut decoder = SseDecoder::default();
        assert_eq!(drain(&mut decoder, b"data: hello\n\n"), vec!["hello"]);
    }

    #[test]
    fn an_event_split_across_chunks_is_only_returned_once_complete() {
        // The specific failure this guards: emitting truncated JSON because a read
        // ended in the middle of a payload.
        let mut decoder = SseDecoder::default();
        assert!(drain(&mut decoder, b"data: {\"choices\":[{\"te").is_empty());
        assert!(drain(&mut decoder, b"xt\":\"hel").is_empty());
        assert_eq!(
            drain(&mut decoder, b"lo\"}]}\n\n"),
            vec![r#"{"choices":[{"text":"hello"}]}"#]
        );
    }

    #[test]
    fn splitting_at_every_byte_boundary_yields_the_same_events() {
        // Transport may split anywhere, so correctness must not depend on where.
        // Both line endings are covered, and events carry more than one data line,
        // because mishandling a split line ending dispatches an event early, which
        // is invisible when no further line follows in that event.
        let cases: [(&str, Vec<&str>); 4] = [
            (
                "data: one\n\ndata: two\n\ndata: [DONE]\n\n",
                vec!["one", "two", "[DONE]"],
            ),
            (
                "data: one\r\n\r\ndata: two\r\n\r\ndata: [DONE]\r\n\r\n",
                vec!["one", "two", "[DONE]"],
            ),
            (
                "data: a\r\ndata: b\r\n\r\ndata: c\r\ndata: d\r\n\r\n",
                vec!["a\nb", "c\nd"],
            ),
            (
                ": ping\r\nevent: message\r\ndata: a\r\ndata: b\r\n\r\n",
                vec!["a\nb"],
            ),
        ];

        for (text, expected) in cases {
            let stream = text.as_bytes();
            for split in 0..=stream.len() {
                let mut decoder = SseDecoder::default();
                let mut seen = Vec::new();
                seen.extend(drain(&mut decoder, &stream[..split]));
                seen.extend(drain(&mut decoder, &stream[split..]));
                assert_eq!(seen, expected, "split at {split} of {text:?}");
            }
        }
    }

    #[test]
    fn a_line_ending_split_across_chunks_never_dispatches_early() {
        // The specific defect: treating a trailing carriage return as a complete
        // ending turns the following line feed into a blank line, which ends the
        // event before its remaining data lines have been read.
        let mut decoder = SseDecoder::default();
        assert!(drain(&mut decoder, b"data: first\r").is_empty());
        assert_eq!(
            drain(&mut decoder, b"\ndata: second\r\n\r\n"),
            vec!["first\nsecond"]
        );
    }

    #[test]
    fn a_multibyte_character_split_across_chunks_survives() {
        // The decoder holds the incomplete sequence rather than decoding it, so no
        // caller ever receives half of a character.
        let payload = "data: \u{1f600} \u{4f60}\u{597d} \u{e9}\n\n".as_bytes();
        for split in 0..payload.len() {
            let mut decoder = SseDecoder::default();
            let mut seen = Vec::new();
            seen.extend(drain(&mut decoder, &payload[..split]));
            seen.extend(drain(&mut decoder, &payload[split..]));
            assert_eq!(
                seen,
                vec!["\u{1f600} \u{4f60}\u{597d} \u{e9}"],
                "split at {split}"
            );
        }
    }

    #[test]
    fn carriage_return_line_feed_pairs_are_one_ending() {
        let mut decoder = SseDecoder::default();
        assert_eq!(drain(&mut decoder, b"data: hello\r\n\r\n"), vec!["hello"]);
    }

    #[test]
    fn a_carriage_return_split_from_its_line_feed_is_not_two_endings() {
        // Splitting a CRLF pair would otherwise look like a line ending followed by
        // a blank line, dispatching the event one line early.
        let mut decoder = SseDecoder::default();
        assert!(drain(&mut decoder, b"data: hello\r").is_empty());
        assert_eq!(drain(&mut decoder, b"\n\r\n"), vec!["hello"]);
    }

    #[test]
    fn a_bare_carriage_return_ends_a_line() {
        let mut decoder = SseDecoder::default();
        assert_eq!(drain(&mut decoder, b"data: hello\r\rnext"), vec!["hello"]);
    }

    #[test]
    fn comments_and_unrelated_fields_are_ignored() {
        let mut decoder = SseDecoder::default();
        let chunk = b": keep-alive\nevent: message\nid: 7\ndata: payload\nretry: 10\n\n";
        assert_eq!(drain(&mut decoder, chunk), vec!["payload"]);
    }

    #[test]
    fn multiple_data_lines_join_with_a_newline() {
        let mut decoder = SseDecoder::default();
        assert_eq!(drain(&mut decoder, b"data: a\ndata: b\n\n"), vec!["a\nb"]);
    }

    #[test]
    fn only_one_leading_space_is_stripped_from_a_value() {
        let mut decoder = SseDecoder::default();
        assert_eq!(drain(&mut decoder, b"data:  spaced\n\n"), vec![" spaced"]);
    }

    #[test]
    fn a_blank_line_without_data_dispatches_nothing() {
        // Keep-alive traffic must not manufacture an empty event.
        let mut decoder = SseDecoder::default();
        assert!(drain(&mut decoder, b"\n\n: ping\n\n").is_empty());
    }

    #[test]
    fn an_invalid_byte_sequence_is_reported_rather_than_replaced() {
        let mut decoder = SseDecoder::default();
        let error = decoder
            .push(b"data: \xff\xfe\n\n")
            .expect_err("invalid UTF-8 must be refused");
        assert!(error.contains("UTF-8"), "unexpected message: {error}");
    }

    #[test]
    fn an_unterminated_event_is_visible_as_partial() {
        let mut decoder = SseDecoder::default();
        assert!(drain(&mut decoder, b"data: truncated").is_empty());
        assert!(decoder.has_partial_event());
    }

    #[test]
    fn a_fully_consumed_stream_holds_nothing() {
        let mut decoder = SseDecoder::default();
        drain(&mut decoder, b"data: done\n\n");
        assert!(!decoder.has_partial_event());
    }
}
