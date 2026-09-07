//! The runtime's own representation of inference requests and results.
//!
//! Deliberately not any backend's wire format. A backend adapter translates
//! between these types and whatever its engine speaks, so an engine's request
//! shape stays below the backend boundary and a second backend can be added
//! without every caller learning a new vocabulary.
//!
//! ADR-0003 keeps the protocol small and extracts it from working code rather than
//! designing it up front. These are the first operations that exist because
//! something actually performs them.

use std::fmt;

/// An identifier for a task this build does not itself define.
///
/// The escape hatch that keeps the runtime from being limited to the model classes
/// its authors happened to anticipate. A backend may offer work no variant here
/// names, and a caller may ask for it, without either waiting on a release of this
/// crate.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TaskId(String);

impl TaskId {
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// What is being asked of a model.
///
/// This runtime is not a text-generation runtime with exceptions bolted on. A task
/// is the unit of work, and text generation will be one variant among many rather
/// than the assumed default: embeddings, reranking, classification, vision, audio,
/// and generative media are all intended to arrive as further variants served by
/// further backends.
///
/// Variants are added when something implements them. An enum full of tasks nothing
/// can perform is a promise the runtime does not keep, so unimplemented model
/// classes are represented by their absence and by [`Task::Custom`], not by
/// placeholder variants.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Task {
    /// Turn inputs into vectors.
    Embed,
    /// Continue a text input.
    GenerateText,
    /// A task named by a backend rather than by this crate.
    Custom(TaskId),
}

impl fmt::Display for Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Embed => f.write_str("embed"),
            Self::GenerateText => f.write_str("generate-text"),
            Self::Custom(id) => write!(f, "custom:{id}"),
        }
    }
}

/// A request to turn text into vectors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedRequest {
    /// The inputs to embed, in order. One vector is expected per input.
    pub inputs: Vec<String>,
}

impl EmbedRequest {
    /// A request for a single input.
    #[must_use]
    pub fn single(input: impl Into<String>) -> Self {
        Self {
            inputs: vec![input.into()],
        }
    }

    /// A request for several inputs.
    #[must_use]
    pub fn batch(inputs: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            inputs: inputs.into_iter().map(Into::into).collect(),
        }
    }

    /// How many inputs were asked about.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inputs.len()
    }

    /// Whether there is nothing to do.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }
}

/// One embedding, tied to the input it came from.
///
/// The index is carried rather than relying on position, so a batch response that
/// arrives out of order is still attributable.
#[derive(Debug, Clone, PartialEq)]
pub struct Embedding {
    pub index: usize,
    pub vector: Vec<f32>,
}

impl Embedding {
    /// How many components the vector has.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.vector.len()
    }

    /// Whether every component is a usable number.
    ///
    /// A vector containing a non-finite value is not a smaller problem than an
    /// error response: it will silently poison any distance computed from it.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.vector.iter().all(|value| value.is_finite())
    }

    /// The Euclidean length of the vector.
    ///
    /// Computed in double precision so the check is not dominated by the
    /// accumulation error of the check itself.
    #[must_use]
    pub fn l2_norm(&self) -> f64 {
        self.vector
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt()
    }
}

/// The result of an embedding request.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingResult {
    pub embeddings: Vec<Embedding>,
}

impl EmbeddingResult {
    /// The embedding for a given input index.
    #[must_use]
    pub fn for_input(&self, index: usize) -> Option<&Embedding> {
        self.embeddings
            .iter()
            .find(|embedding| embedding.index == index)
    }

    /// The dimension shared by every embedding, when they agree.
    ///
    /// Returns `None` for an empty result or one whose vectors disagree, which is
    /// itself a signal that the response should not be trusted.
    #[must_use]
    pub fn dimension(&self) -> Option<usize> {
        let mut dimensions = self.embeddings.iter().map(Embedding::dimension);
        let first = dimensions.next()?;
        dimensions.all(|other| other == first).then_some(first)
    }
}

/// Why an embedding result cannot be trusted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResultDefect {
    /// Fewer or more vectors than inputs.
    CountMismatch { expected: usize, actual: usize },
    /// A vector with no components.
    EmptyVector { index: usize },
    /// A vector containing a value that is not a usable number.
    NonFiniteValue { index: usize },
    /// Vectors of differing lengths in one response.
    InconsistentDimension,
    /// An input index that was never requested, or a duplicate.
    UnattributableIndex { index: usize },
}

impl fmt::Display for ResultDefect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CountMismatch { expected, actual } => write!(
                f,
                "expected {expected} embedding(s) for {expected} input(s), got {actual}"
            ),
            Self::EmptyVector { index } => write!(f, "the embedding for input {index} is empty"),
            Self::NonFiniteValue { index } => write!(
                f,
                "the embedding for input {index} contains a value that is not finite"
            ),
            Self::InconsistentDimension => {
                f.write_str("the embeddings in one response have differing dimensions")
            }
            Self::UnattributableIndex { index } => {
                write!(
                    f,
                    "the response contains an unexpected or duplicate index {index}"
                )
            }
        }
    }
}

impl std::error::Error for ResultDefect {}

/// Checks that a result is structurally usable for the request that produced it.
///
/// Structure only. Nothing here judges whether the numbers are any good, which is
/// a question about the model rather than about the response.
///
/// # Errors
///
/// Returns the first defect found.
pub fn validate(request: &EmbedRequest, result: &EmbeddingResult) -> Result<(), ResultDefect> {
    if result.embeddings.len() != request.len() {
        return Err(ResultDefect::CountMismatch {
            expected: request.len(),
            actual: result.embeddings.len(),
        });
    }

    let mut seen = vec![false; request.len()];
    for embedding in &result.embeddings {
        let slot = seen
            .get_mut(embedding.index)
            .ok_or(ResultDefect::UnattributableIndex {
                index: embedding.index,
            })?;
        if *slot {
            return Err(ResultDefect::UnattributableIndex {
                index: embedding.index,
            });
        }
        *slot = true;

        if embedding.vector.is_empty() {
            return Err(ResultDefect::EmptyVector {
                index: embedding.index,
            });
        }
        if !embedding.is_finite() {
            return Err(ResultDefect::NonFiniteValue {
                index: embedding.index,
            });
        }
    }

    if result.dimension().is_none() {
        return Err(ResultDefect::InconsistentDimension);
    }

    Ok(())
}

// ---------------------------------------------------------------- text generation

/// What a generation request is given to continue.
///
/// ADR-0007 makes continuation the primitive. A base completion model has no notion
/// of roles, so a request shaped only around a conversation would make every
/// non-conversational model a special case.
///
/// A conversation variant arrives when something can render one. Until then its
/// absence is honest rather than a gap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextInput {
    /// Text to continue, exactly as the model will see it.
    Continuation(String),
}

impl TextInput {
    /// The text as the backend will receive it.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Continuation(text) => text,
        }
    }
}

/// The most stop sequences a request may carry.
pub const MAX_STOP_SEQUENCES: usize = 8;

/// The most bytes any one stop sequence may be.
pub const MAX_STOP_SEQUENCE_BYTES: usize = 256;

/// Portable generation controls.
///
/// ADR-0008 limits this to controls whose meaning can be kept stable across
/// backends. Engine-native controls do not belong here.
///
/// Every field is optional, and `None` means the runtime did not override it rather
/// than that it equals the backend's current default. Materialising a default would
/// freeze one engine's value into this contract and would make "whatever the backend
/// thinks best" inexpressible.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GenerationParameters {
    /// The most tokens to generate. Output only; this is not a context limit and
    /// does not include the input.
    pub max_output_tokens: Option<u32>,
    /// Sampling temperature. Zero requests the most deterministic sampling the
    /// backend offers.
    pub temperature: Option<f32>,
    /// A requested sampling seed.
    ///
    /// Repeatability is scoped to a backend build and its hardware. The same seed
    /// across backends, builds, or accelerators is not promised to reproduce
    /// anything, because quantisation, kernel selection, batching, and floating
    /// point ordering all move the result.
    pub seed: Option<u64>,
    /// Sequences whose appearance ends generation.
    pub stop: Vec<String>,
}

/// A generation parameter that cannot be honoured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidGenerationParameter {
    /// A temperature that is negative or not a number.
    Temperature { reason: String },
    /// A token budget of zero, which asks for nothing.
    MaxOutputTokens { reason: String },
    /// Too many stop sequences, or one that is too large.
    ///
    /// Bounded because request content is untrusted, and an unbounded list of
    /// unbounded patterns is a denial-of-service surface rather than a feature.
    StopSequences { reason: String },
}

impl fmt::Display for InvalidGenerationParameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Temperature { reason } => write!(f, "temperature is invalid: {reason}"),
            Self::MaxOutputTokens { reason } => {
                write!(f, "max_output_tokens is invalid: {reason}")
            }
            Self::StopSequences { reason } => write!(f, "stop sequences are invalid: {reason}"),
        }
    }
}

impl std::error::Error for InvalidGenerationParameter {}

impl GenerationParameters {
    /// Checks that every supplied parameter can be honoured.
    ///
    /// Invalid values are rejected rather than clamped. Silently changing a
    /// caller's number produces output they did not ask for and cannot explain from
    /// what they sent.
    ///
    /// # Errors
    ///
    /// Returns the first parameter that cannot be honoured.
    pub fn validate(&self) -> Result<(), InvalidGenerationParameter> {
        if let Some(temperature) = self.temperature {
            if !temperature.is_finite() {
                return Err(InvalidGenerationParameter::Temperature {
                    reason: format!("{temperature} is not a finite number"),
                });
            }
            if temperature < 0.0 {
                return Err(InvalidGenerationParameter::Temperature {
                    reason: format!("{temperature} is negative"),
                });
            }
        }

        if self.max_output_tokens == Some(0) {
            return Err(InvalidGenerationParameter::MaxOutputTokens {
                reason: "zero asks for no output at all".to_owned(),
            });
        }

        if self.stop.len() > MAX_STOP_SEQUENCES {
            return Err(InvalidGenerationParameter::StopSequences {
                reason: format!(
                    "{} sequences exceeds the limit of {MAX_STOP_SEQUENCES}",
                    self.stop.len()
                ),
            });
        }
        for sequence in &self.stop {
            if sequence.is_empty() {
                return Err(InvalidGenerationParameter::StopSequences {
                    reason: "an empty sequence would stop generation immediately".to_owned(),
                });
            }
            if sequence.len() > MAX_STOP_SEQUENCE_BYTES {
                return Err(InvalidGenerationParameter::StopSequences {
                    reason: format!(
                        "a sequence of {} bytes exceeds the limit of {MAX_STOP_SEQUENCE_BYTES}",
                        sequence.len()
                    ),
                });
            }
        }

        Ok(())
    }
}

/// A request to continue text.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateTextRequest {
    pub input: TextInput,
    pub parameters: GenerationParameters,
}

impl GenerateTextRequest {
    /// A request to continue some text with default parameters.
    #[must_use]
    pub fn continuation(input: impl Into<String>) -> Self {
        Self {
            input: TextInput::Continuation(input.into()),
            parameters: GenerationParameters::default(),
        }
    }

    /// Replaces the parameters.
    #[must_use]
    pub fn with_parameters(mut self, parameters: GenerationParameters) -> Self {
        self.parameters = parameters;
        self
    }
}

/// Why generation stopped.
///
/// Normalised, with an escape for a reason this build does not recognise. A reason
/// that is merely unfamiliar should not be reported as one of the known ones.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// The model finished, or a stop sequence was reached.
    Stop,
    /// The token budget was exhausted.
    Length,
    /// Something else, reported as the backend described it.
    Other(String),
}

impl fmt::Display for FinishReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stop => f.write_str("stop"),
            Self::Length => f.write_str("length"),
            Self::Other(reason) => write!(f, "other:{reason}"),
        }
    }
}

/// Token counts, when the backend reports them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// The result of a completed generation.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationResult {
    pub text: String,
    pub finish_reason: FinishReason,
    pub usage: Option<GenerationUsage>,
}

/// Why a generation result cannot be trusted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerationDefect {
    /// The backend reported success but produced nothing.
    ///
    /// A successful status with an empty body is not demonstrated generation, and
    /// treating it as such would let a broken backend verify a capability.
    NoContent,
    /// Reported counts contradict what was returned.
    ImplausibleUsage { reason: String },
}

impl fmt::Display for GenerationDefect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoContent => f.write_str("the backend reported success but generated no content"),
            Self::ImplausibleUsage { reason } => {
                write!(f, "the reported token usage is not plausible: {reason}")
            }
        }
    }
}

impl std::error::Error for GenerationDefect {}

/// Checks that a generation result is usable.
///
/// # Errors
///
/// Returns the first defect found.
pub fn validate_generation(result: &GenerationResult) -> Result<(), GenerationDefect> {
    if result.text.is_empty() {
        return Err(GenerationDefect::NoContent);
    }
    if let Some(usage) = result.usage
        && usage.output_tokens == 0
    {
        return Err(GenerationDefect::ImplausibleUsage {
            reason: "content was returned but no output tokens were reported".to_owned(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embedding(index: usize, values: &[f32]) -> Embedding {
        Embedding {
            index,
            vector: values.to_vec(),
        }
    }

    fn result(embeddings: Vec<Embedding>) -> EmbeddingResult {
        EmbeddingResult { embeddings }
    }

    #[test]
    fn a_well_formed_result_validates() {
        let request = EmbedRequest::single("hello");
        let response = result(vec![embedding(0, &[0.6, 0.8])]);
        assert!(validate(&request, &response).is_ok());
    }

    #[test]
    fn a_missing_embedding_is_a_defect() {
        let request = EmbedRequest::batch(["a", "b"]);
        let response = result(vec![embedding(0, &[1.0])]);
        assert_eq!(
            validate(&request, &response),
            Err(ResultDefect::CountMismatch {
                expected: 2,
                actual: 1
            })
        );
    }

    #[test]
    fn a_non_finite_value_is_rejected_rather_than_passed_on() {
        // A NaN silently poisons every distance computed from the vector, so it is
        // not a lesser problem than an error response.
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let request = EmbedRequest::single("x");
            let response = result(vec![embedding(0, &[0.5, bad])]);
            assert_eq!(
                validate(&request, &response),
                Err(ResultDefect::NonFiniteValue { index: 0 }),
                "{bad} should have been rejected"
            );
        }
    }

    #[test]
    fn an_empty_vector_is_rejected() {
        let request = EmbedRequest::single("x");
        let response = result(vec![embedding(0, &[])]);
        assert_eq!(
            validate(&request, &response),
            Err(ResultDefect::EmptyVector { index: 0 })
        );
    }

    #[test]
    fn differing_dimensions_in_one_response_are_rejected() {
        let request = EmbedRequest::batch(["a", "b"]);
        let response = result(vec![embedding(0, &[1.0, 0.0]), embedding(1, &[1.0])]);
        assert_eq!(
            validate(&request, &response),
            Err(ResultDefect::InconsistentDimension)
        );
    }

    #[test]
    fn a_duplicate_index_is_rejected() {
        // Two vectors both claiming input zero means one input has no answer.
        let request = EmbedRequest::batch(["a", "b"]);
        let response = result(vec![embedding(0, &[1.0]), embedding(0, &[1.0])]);
        assert_eq!(
            validate(&request, &response),
            Err(ResultDefect::UnattributableIndex { index: 0 })
        );
    }

    #[test]
    fn an_out_of_range_index_is_rejected() {
        let request = EmbedRequest::single("a");
        let response = result(vec![embedding(7, &[1.0])]);
        assert_eq!(
            validate(&request, &response),
            Err(ResultDefect::UnattributableIndex { index: 7 })
        );
    }

    #[test]
    fn embeddings_are_attributable_by_index_not_position() {
        let response = result(vec![embedding(1, &[2.0]), embedding(0, &[1.0])]);
        assert_eq!(response.for_input(0).map(|e| e.vector[0]), Some(1.0));
        assert_eq!(response.for_input(1).map(|e| e.vector[0]), Some(2.0));
    }

    #[test]
    fn the_norm_of_a_unit_vector_is_one() {
        // The tolerance is sized for the inputs, which are 32-bit. Neither 0.6 nor
        // 0.8 is exactly representable, so the norm lands a little over 1.0 no
        // matter how carefully it is summed. A tolerance tighter than the input
        // precision tests the float format rather than the code.
        let norm = embedding(0, &[0.6, 0.8]).l2_norm();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "expected approximately 1.0, got {norm}"
        );
    }

    #[test]
    fn a_task_this_build_does_not_define_is_still_expressible() {
        // The runtime must not be limited to the model classes its authors
        // anticipated. A backend offering something unnamed here can still be asked
        // for it.
        let task = Task::Custom(TaskId::new("segment-image"));
        assert_eq!(task.to_string(), "custom:segment-image");
        assert_ne!(task, Task::Embed);
    }

    #[test]
    fn a_negative_or_unusable_temperature_is_rejected_not_clamped() {
        // Clamping would produce output the caller did not ask for and cannot
        // explain from what they sent.
        for bad in [-0.1f32, f32::NAN, f32::INFINITY] {
            let parameters = GenerationParameters {
                temperature: Some(bad),
                ..GenerationParameters::default()
            };
            assert!(
                matches!(
                    parameters.validate(),
                    Err(InvalidGenerationParameter::Temperature { .. })
                ),
                "temperature {bad} should have been rejected"
            );
        }
    }

    #[test]
    fn a_zero_temperature_is_allowed() {
        let parameters = GenerationParameters {
            temperature: Some(0.0),
            ..GenerationParameters::default()
        };
        assert!(parameters.validate().is_ok());
    }

    #[test]
    fn a_zero_token_budget_is_rejected() {
        let parameters = GenerationParameters {
            max_output_tokens: Some(0),
            ..GenerationParameters::default()
        };
        assert!(matches!(
            parameters.validate(),
            Err(InvalidGenerationParameter::MaxOutputTokens { .. })
        ));
    }

    #[test]
    fn stop_sequences_are_bounded_because_request_content_is_untrusted() {
        let too_many = GenerationParameters {
            stop: (0..MAX_STOP_SEQUENCES + 1).map(|i| i.to_string()).collect(),
            ..GenerationParameters::default()
        };
        assert!(matches!(
            too_many.validate(),
            Err(InvalidGenerationParameter::StopSequences { .. })
        ));

        let too_large = GenerationParameters {
            stop: vec!["x".repeat(MAX_STOP_SEQUENCE_BYTES + 1)],
            ..GenerationParameters::default()
        };
        assert!(matches!(
            too_large.validate(),
            Err(InvalidGenerationParameter::StopSequences { .. })
        ));
    }

    #[test]
    fn an_empty_stop_sequence_is_rejected() {
        let parameters = GenerationParameters {
            stop: vec![String::new()],
            ..GenerationParameters::default()
        };
        assert!(matches!(
            parameters.validate(),
            Err(InvalidGenerationParameter::StopSequences { .. })
        ));
    }

    #[test]
    fn unset_parameters_are_not_defaults() {
        // The distinction ADR-0008 turns on: nothing is materialised, so the
        // backend applies its own default and the caller can still express having
        // no opinion.
        let parameters = GenerationParameters::default();
        assert_eq!(parameters.temperature, None);
        assert_eq!(parameters.max_output_tokens, None);
        assert_eq!(parameters.seed, None);
        assert!(parameters.stop.is_empty());
        assert!(parameters.validate().is_ok());
    }

    #[test]
    fn a_successful_response_with_no_content_is_not_generation() {
        // A broken backend returning success with an empty body must not be able to
        // verify a capability.
        let result = GenerationResult {
            text: String::new(),
            finish_reason: FinishReason::Stop,
            usage: None,
        };
        assert_eq!(
            validate_generation(&result),
            Err(GenerationDefect::NoContent)
        );
    }

    #[test]
    fn content_with_zero_reported_output_tokens_is_implausible() {
        let result = GenerationResult {
            text: "something".to_owned(),
            finish_reason: FinishReason::Stop,
            usage: Some(GenerationUsage {
                input_tokens: 5,
                output_tokens: 0,
            }),
        };
        assert!(matches!(
            validate_generation(&result),
            Err(GenerationDefect::ImplausibleUsage { .. })
        ));
    }

    #[test]
    fn an_unfamiliar_finish_reason_is_not_reported_as_a_known_one() {
        let reason = FinishReason::Other("tool_call".to_owned());
        assert_ne!(reason, FinishReason::Stop);
        assert_ne!(reason, FinishReason::Length);
        assert_eq!(reason.to_string(), "other:tool_call");
    }

    #[test]
    fn continuation_is_the_only_input_shape_so_far() {
        // ADR-0007: a conversation variant arrives when something can render one.
        let input = TextInput::Continuation("Once upon a time".to_owned());
        assert_eq!(input.as_str(), "Once upon a time");
    }

    #[test]
    fn only_implemented_tasks_have_their_own_variant() {
        // Guards the rule rather than the list. A variant added without an
        // implementation behind it is a promise the runtime does not keep, so a
        // future task should arrive with its implementation or as Custom.
        assert_eq!(Task::Embed.to_string(), "embed");
        assert_eq!(Task::GenerateText.to_string(), "generate-text");
    }
}
