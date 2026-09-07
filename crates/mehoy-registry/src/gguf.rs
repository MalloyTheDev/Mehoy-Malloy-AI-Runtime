//! A GGUF header and metadata parser.
//!
//! The runtime reads model containers itself rather than asking a backend what a
//! file is. Delegating that would make every fact about a model dependent on
//! whichever engine happened to be installed, and would make routing the same
//! artifact to a second backend impossible. The ownership split is that the runtime
//! understands the artifact and the backend executes it.
//!
//! # This parses untrusted input
//!
//! A model container is attacker-influenced binary data full of length prefixes and
//! counts. Every one of them is a chance to allocate an enormous buffer, read past
//! the end of the file, or overflow an offset calculation. So:
//!
//! - the magic is checked; a `.gguf` extension is not evidence of anything;
//! - the version must be one that is actually supported, and is rejected by name
//!   rather than best-effort parsed;
//! - every count and length is bounded before it is used to allocate or seek;
//! - all offset arithmetic is checked;
//! - a total budget bounds how much of the file the header may consume, so a
//!   container claiming millions of entries cannot make this read gigabytes.
//!
//! Only a bounded prefix is read. Tensor data is never touched, so inspecting a
//! 70 GB artifact costs the same as inspecting a small one.

use std::collections::BTreeMap;
use std::fmt;
use std::io::{self, Read};

/// `GGUF` in little-endian byte order.
const MAGIC: [u8; 4] = *b"GGUF";

/// Versions this parser understands.
///
/// An unsupported version is refused rather than guessed at, because a layout
/// change is exactly the case where best-effort parsing produces confident nonsense.
const SUPPORTED_VERSIONS: [u32; 2] = [2, 3];

/// Largest number of metadata entries accepted.
const MAX_METADATA_ENTRIES: u64 = 1 << 20;

/// Largest tensor count accepted.
const MAX_TENSOR_COUNT: u64 = 1 << 24;

/// Largest single string accepted, which comfortably covers a chat template.
const MAX_STRING_BYTES: u64 = 16 << 20;

/// Largest array length accepted.
const MAX_ARRAY_LEN: u64 = 1 << 24;

/// Total bytes the header and metadata may consume.
const MAX_HEADER_BYTES: u64 = 64 << 20;

/// Arrays longer than this are summarised rather than retained, so a tokenizer
/// vocabulary does not become several hundred thousand owned strings in memory.
const MAX_RETAINED_ARRAY_LEN: usize = 64;

/// A typed GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    /// An array too long to retain, recorded by shape instead of contents.
    LargeArray {
        element_type: u32,
        len: u64,
    },
}

impl MetadataValue {
    /// The value as a string, when it is one.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    /// The value as an unsigned integer, when it is an integral type that fits.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(v) => Some(u64::from(v)),
            Self::U16(v) => Some(u64::from(v)),
            Self::U32(v) => Some(u64::from(v)),
            Self::U64(v) => Some(v),
            Self::I8(v) => u64::try_from(v).ok(),
            Self::I16(v) => u64::try_from(v).ok(),
            Self::I32(v) => u64::try_from(v).ok(),
            Self::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }
}

/// Why a container could not be read as GGUF.
#[derive(Debug)]
pub enum GgufError {
    /// The file does not begin with the GGUF magic.
    NotGguf { found: [u8; 4] },
    /// The container declares a version this parser does not implement.
    UnsupportedVersion { version: u32 },
    /// A declared count or length exceeds what this parser will accept.
    ///
    /// Refusing is deliberate. A container claiming an implausible size is either
    /// corrupt or hostile, and neither is worth allocating for.
    Implausible {
        what: &'static str,
        value: u64,
        limit: u64,
    },
    /// The file ended before the declared structure did.
    Truncated { what: &'static str },
    /// An offset or length calculation overflowed.
    Overflow { what: &'static str },
    /// A metadata entry declared a type this parser does not know.
    UnknownValueType { type_id: u32 },
    /// A string was not valid UTF-8.
    InvalidText { key: String },
    /// Reading the file failed.
    Io(io::Error),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotGguf { found } => write!(
                f,
                "not a GGUF container: expected magic {:?}, found {:?}",
                String::from_utf8_lossy(&MAGIC),
                String::from_utf8_lossy(found)
            ),
            Self::UnsupportedVersion { version } => write!(
                f,
                "GGUF version {version} is not supported (supported: {SUPPORTED_VERSIONS:?})"
            ),
            Self::Implausible { what, value, limit } => {
                write!(f, "GGUF {what} of {value} exceeds the limit of {limit}")
            }
            Self::Truncated { what } => write!(f, "GGUF file ended while reading {what}"),
            Self::Overflow { what } => write!(f, "GGUF {what} overflowed"),
            Self::UnknownValueType { type_id } => {
                write!(f, "GGUF metadata value type {type_id} is not recognised")
            }
            Self::InvalidText { key } => {
                write!(f, "GGUF metadata key {key:?} is not valid UTF-8")
            }
            Self::Io(err) => write!(f, "cannot read GGUF container: {err}"),
        }
    }
}

impl std::error::Error for GgufError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for GgufError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

/// What the parser extracted.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
    pub metadata: BTreeMap<String, MetadataValue>,
}

impl GgufHeader {
    /// A metadata value by key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.get(key)
    }

    /// A metadata value as text.
    #[must_use]
    pub fn text(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(MetadataValue::as_str)
    }

    /// A metadata value as an unsigned integer.
    #[must_use]
    pub fn number(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(MetadataValue::as_u64)
    }

    /// The model architecture, which names the family of the remaining keys.
    #[must_use]
    pub fn architecture(&self) -> Option<&str> {
        self.text("general.architecture")
    }

    /// A key qualified by this container's architecture.
    ///
    /// Dimensions are stored under architecture-specific keys such as
    /// `llama.context_length`, so they cannot be read without first knowing the
    /// architecture.
    #[must_use]
    pub fn architecture_number(&self, suffix: &str) -> Option<u64> {
        let architecture = self.architecture()?;
        self.number(&format!("{architecture}.{suffix}"))
    }
}

/// A reader that refuses to consume more than a fixed budget.
struct Budgeted<R> {
    inner: R,
    remaining: u64,
}

impl<R: Read> Budgeted<R> {
    fn new(inner: R, budget: u64) -> Self {
        Self {
            inner,
            remaining: budget,
        }
    }

    fn take(&mut self, amount: u64, what: &'static str) -> Result<(), GgufError> {
        self.remaining = self
            .remaining
            .checked_sub(amount)
            .ok_or(GgufError::Implausible {
                what,
                value: amount,
                limit: self.remaining,
            })?;
        Ok(())
    }

    fn read_exact_or_truncated(
        &mut self,
        buf: &mut [u8],
        what: &'static str,
    ) -> Result<(), GgufError> {
        match self.inner.read_exact(buf) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => {
                Err(GgufError::Truncated { what })
            }
            Err(err) => Err(GgufError::Io(err)),
        }
    }

    fn u8(&mut self, what: &'static str) -> Result<u8, GgufError> {
        self.take(1, what)?;
        let mut buf = [0u8; 1];
        self.read_exact_or_truncated(&mut buf, what)?;
        Ok(buf[0])
    }

    fn u16(&mut self, what: &'static str) -> Result<u16, GgufError> {
        self.take(2, what)?;
        let mut buf = [0u8; 2];
        self.read_exact_or_truncated(&mut buf, what)?;
        Ok(u16::from_le_bytes(buf))
    }

    fn u32(&mut self, what: &'static str) -> Result<u32, GgufError> {
        self.take(4, what)?;
        let mut buf = [0u8; 4];
        self.read_exact_or_truncated(&mut buf, what)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn u64(&mut self, what: &'static str) -> Result<u64, GgufError> {
        self.take(8, what)?;
        let mut buf = [0u8; 8];
        self.read_exact_or_truncated(&mut buf, what)?;
        Ok(u64::from_le_bytes(buf))
    }

    /// Reads a length-prefixed string.
    ///
    /// The length is bounded before any allocation, so a container claiming a
    /// multi-terabyte string is refused rather than attempted.
    fn string(&mut self, what: &'static str) -> Result<String, GgufError> {
        let len = self.u64(what)?;
        if len > MAX_STRING_BYTES {
            return Err(GgufError::Implausible {
                what: "string length",
                value: len,
                limit: MAX_STRING_BYTES,
            });
        }
        self.take(len, "string contents")?;
        let capacity = usize::try_from(len).map_err(|_| GgufError::Overflow {
            what: "string length",
        })?;
        let mut buf = vec![0u8; capacity];
        self.read_exact_or_truncated(&mut buf, what)?;
        String::from_utf8(buf).map_err(|_| GgufError::InvalidText {
            key: what.to_owned(),
        })
    }
}

fn read_value<R: Read>(reader: &mut Budgeted<R>, type_id: u32) -> Result<MetadataValue, GgufError> {
    Ok(match type_id {
        0 => MetadataValue::U8(reader.u8("uint8")?),
        1 => MetadataValue::I8(reader.u8("int8")? as i8),
        2 => MetadataValue::U16(reader.u16("uint16")?),
        3 => MetadataValue::I16(reader.u16("int16")? as i16),
        4 => MetadataValue::U32(reader.u32("uint32")?),
        5 => MetadataValue::I32(reader.u32("int32")? as i32),
        6 => MetadataValue::F32(f32::from_bits(reader.u32("float32")?)),
        7 => MetadataValue::Bool(reader.u8("bool")? != 0),
        8 => MetadataValue::String(reader.string("string")?),
        9 => return read_array(reader),
        10 => MetadataValue::U64(reader.u64("uint64")?),
        11 => MetadataValue::I64(reader.u64("int64")? as i64),
        12 => MetadataValue::F64(f64::from_bits(reader.u64("float64")?)),
        other => return Err(GgufError::UnknownValueType { type_id: other }),
    })
}

fn read_array<R: Read>(reader: &mut Budgeted<R>) -> Result<MetadataValue, GgufError> {
    let element_type = reader.u32("array element type")?;
    let len = reader.u64("array length")?;
    if len > MAX_ARRAY_LEN {
        return Err(GgufError::Implausible {
            what: "array length",
            value: len,
            limit: MAX_ARRAY_LEN,
        });
    }

    // A tokenizer vocabulary is a legitimate array of hundreds of thousands of
    // strings. It still has to be read to reach the entries after it, but keeping
    // every element would turn inspection into a large allocation for no benefit.
    let retain = usize::try_from(len).unwrap_or(usize::MAX) <= MAX_RETAINED_ARRAY_LEN;
    let mut retained = Vec::new();
    for _ in 0..len {
        let value = read_value(reader, element_type)?;
        if retain {
            retained.push(value);
        }
    }

    Ok(if retain {
        MetadataValue::Array(retained)
    } else {
        MetadataValue::LargeArray { element_type, len }
    })
}

/// Parses a GGUF header and its metadata from a stream.
///
/// Only the header is consumed. Tensor data is never read.
///
/// # Errors
///
/// Returns [`GgufError`] when the container is not GGUF, declares an unsupported
/// version, is truncated, or declares sizes this parser refuses to honour.
pub fn parse<R: Read>(source: R) -> Result<GgufHeader, GgufError> {
    let mut reader = Budgeted::new(source, MAX_HEADER_BYTES);

    let mut magic = [0u8; 4];
    reader.take(4, "magic")?;
    reader.read_exact_or_truncated(&mut magic, "magic")?;
    if magic != MAGIC {
        return Err(GgufError::NotGguf { found: magic });
    }

    let version = reader.u32("version")?;
    if !SUPPORTED_VERSIONS.contains(&version) {
        return Err(GgufError::UnsupportedVersion { version });
    }

    let tensor_count = reader.u64("tensor count")?;
    if tensor_count > MAX_TENSOR_COUNT {
        return Err(GgufError::Implausible {
            what: "tensor count",
            value: tensor_count,
            limit: MAX_TENSOR_COUNT,
        });
    }

    let metadata_count = reader.u64("metadata count")?;
    if metadata_count > MAX_METADATA_ENTRIES {
        return Err(GgufError::Implausible {
            what: "metadata count",
            value: metadata_count,
            limit: MAX_METADATA_ENTRIES,
        });
    }

    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_count {
        let key = reader.string("metadata key")?;
        let type_id = reader.u32("metadata value type")?;
        let value = read_value(&mut reader, type_id)?;
        metadata.insert(key, value);
    }

    Ok(GgufHeader {
        version,
        tensor_count,
        metadata_count,
        metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds GGUF byte streams, including deliberately malformed ones.
    #[derive(Default)]
    pub(crate) struct Builder {
        bytes: Vec<u8>,
    }

    impl Builder {
        fn new() -> Self {
            Self::default()
        }

        fn raw(mut self, bytes: &[u8]) -> Self {
            self.bytes.extend_from_slice(bytes);
            self
        }

        fn magic(self) -> Self {
            self.raw(&MAGIC)
        }

        fn u32(self, value: u32) -> Self {
            self.raw(&value.to_le_bytes())
        }

        fn u64(self, value: u64) -> Self {
            self.raw(&value.to_le_bytes())
        }

        fn string(self, value: &str) -> Self {
            self.u64(value.len() as u64).raw(value.as_bytes())
        }

        /// A key with a string value.
        fn text_entry(self, key: &str, value: &str) -> Self {
            self.string(key).u32(8).string(value)
        }

        /// A key with a uint32 value.
        fn number_entry(self, key: &str, value: u32) -> Self {
            self.string(key).u32(4).u32(value)
        }

        fn build(self) -> Vec<u8> {
            self.bytes
        }
    }

    fn minimal() -> Vec<u8> {
        Builder::new()
            .magic()
            .u32(3)
            .u64(0)
            .u64(3)
            .text_entry("general.architecture", "llama")
            .text_entry("general.name", "test model")
            .number_entry("llama.context_length", 4096)
            .build()
    }

    #[test]
    fn parses_a_well_formed_container() {
        let header = parse(minimal().as_slice()).expect("parses");
        assert_eq!(header.version, 3);
        assert_eq!(header.metadata_count, 3);
        assert_eq!(header.architecture(), Some("llama"));
        assert_eq!(header.text("general.name"), Some("test model"));
        assert_eq!(header.architecture_number("context_length"), Some(4096));
    }

    #[test]
    fn a_file_named_gguf_is_not_evidence_that_it_is_gguf() {
        let err = parse(b"this is a text file".as_slice()).expect_err("must be rejected");
        assert!(
            matches!(err, GgufError::NotGguf { .. }),
            "expected NotGguf, got {err}"
        );
    }

    #[test]
    fn an_unsupported_version_is_refused_by_name() {
        // Best-effort parsing of an unknown layout produces confident nonsense.
        let bytes = Builder::new().magic().u32(99).u64(0).u64(0).build();
        match parse(bytes.as_slice()).expect_err("must be rejected") {
            GgufError::UnsupportedVersion { version } => assert_eq!(version, 99),
            other => panic!("expected UnsupportedVersion, got {other}"),
        }
    }

    #[test]
    fn version_two_is_accepted() {
        let bytes = Builder::new().magic().u32(2).u64(0).u64(0).build();
        assert_eq!(parse(bytes.as_slice()).expect("parses").version, 2);
    }

    #[test]
    fn a_truncated_header_is_rejected() {
        let full = minimal();
        for cut in [4, 8, 12, 20, 30, 45] {
            let err = parse(&full[..cut.min(full.len())]).expect_err("must be rejected");
            assert!(
                matches!(err, GgufError::Truncated { .. } | GgufError::NotGguf { .. }),
                "cutting at {cut} gave {err}"
            );
        }
    }

    #[test]
    fn an_implausible_metadata_count_is_refused_before_allocating() {
        let bytes = Builder::new().magic().u32(3).u64(0).u64(u64::MAX).build();
        match parse(bytes.as_slice()).expect_err("must be rejected") {
            GgufError::Implausible { what, .. } => assert_eq!(what, "metadata count"),
            other => panic!("expected Implausible, got {other}"),
        }
    }

    #[test]
    fn an_implausible_tensor_count_is_refused() {
        let bytes = Builder::new().magic().u32(3).u64(u64::MAX).u64(0).build();
        match parse(bytes.as_slice()).expect_err("must be rejected") {
            GgufError::Implausible { what, .. } => assert_eq!(what, "tensor count"),
            other => panic!("expected Implausible, got {other}"),
        }
    }

    #[test]
    fn an_enormous_string_length_is_refused_rather_than_allocated() {
        // The declared length is far beyond the file. A parser that allocated first
        // would try to reserve terabytes before discovering the truncation.
        let bytes = Builder::new()
            .magic()
            .u32(3)
            .u64(0)
            .u64(1)
            .u64(u64::MAX)
            .build();
        match parse(bytes.as_slice()).expect_err("must be rejected") {
            GgufError::Implausible { what, .. } => assert_eq!(what, "string length"),
            other => panic!("expected Implausible, got {other}"),
        }
    }

    #[test]
    fn an_enormous_array_length_is_refused() {
        let bytes = Builder::new()
            .magic()
            .u32(3)
            .u64(0)
            .u64(1)
            .string("tokenizer.ggml.tokens")
            .u32(9)
            .u32(8)
            .u64(u64::MAX)
            .build();
        match parse(bytes.as_slice()).expect_err("must be rejected") {
            GgufError::Implausible { what, .. } => assert_eq!(what, "array length"),
            other => panic!("expected Implausible, got {other}"),
        }
    }

    #[test]
    fn an_unknown_value_type_is_reported_rather_than_skipped() {
        let bytes = Builder::new()
            .magic()
            .u32(3)
            .u64(0)
            .u64(1)
            .string("weird")
            .u32(4242)
            .build();
        match parse(bytes.as_slice()).expect_err("must be rejected") {
            GgufError::UnknownValueType { type_id } => assert_eq!(type_id, 4242),
            other => panic!("expected UnknownValueType, got {other}"),
        }
    }

    #[test]
    fn the_header_budget_bounds_total_reading() {
        // Many entries, each individually plausible, must still not be able to make
        // the parser read an unbounded amount.
        let mut builder = Builder::new().magic().u32(3).u64(0).u64(100_000);
        for index in 0..100_000u32 {
            builder = builder.text_entry(&format!("key{index}"), &"x".repeat(1024));
        }
        let bytes = builder.build();
        let err = parse(bytes.as_slice()).expect_err("must hit the budget");
        assert!(
            matches!(
                err,
                GgufError::Implausible { .. } | GgufError::Truncated { .. }
            ),
            "expected a bounded refusal, got {err}"
        );
    }

    #[test]
    fn a_long_array_is_summarised_rather_than_retained() {
        let count = (MAX_RETAINED_ARRAY_LEN + 10) as u64;
        let mut builder = Builder::new()
            .magic()
            .u32(3)
            .u64(0)
            .u64(1)
            .string("tokenizer.ggml.tokens")
            .u32(9)
            .u32(8)
            .u64(count);
        for index in 0..count {
            builder = builder.string(&format!("t{index}"));
        }
        let header = parse(builder.build().as_slice()).expect("parses");
        match header.get("tokenizer.ggml.tokens").expect("present") {
            MetadataValue::LargeArray { len, element_type } => {
                assert_eq!(*len, count);
                assert_eq!(*element_type, 8);
            }
            other => panic!("expected a summarised array, got {other:?}"),
        }
    }

    #[test]
    fn a_short_array_is_retained() {
        let bytes = Builder::new()
            .magic()
            .u32(3)
            .u64(0)
            .u64(1)
            .string("small")
            .u32(9)
            .u32(4)
            .u64(2)
            .u32(7)
            .u32(9)
            .build();
        let header = parse(bytes.as_slice()).expect("parses");
        assert_eq!(
            header.get("small"),
            Some(&MetadataValue::Array(vec![
                MetadataValue::U32(7),
                MetadataValue::U32(9)
            ]))
        );
    }

    #[test]
    fn architecture_qualified_keys_need_the_architecture() {
        // Without general.architecture there is no way to know which prefix the
        // dimension keys use, and guessing one would be wrong for most models.
        let bytes = Builder::new()
            .magic()
            .u32(3)
            .u64(0)
            .u64(1)
            .number_entry("llama.context_length", 2048)
            .build();
        let header = parse(bytes.as_slice()).expect("parses");
        assert_eq!(header.architecture(), None);
        assert_eq!(header.architecture_number("context_length"), None);
        // The raw key is still available.
        assert_eq!(header.number("llama.context_length"), Some(2048));
    }
}
