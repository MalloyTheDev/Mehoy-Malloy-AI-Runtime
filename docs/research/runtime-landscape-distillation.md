# Runtime Landscape Distillation

Status: research findings. Nothing in this document is implemented.

Three of the four items under "Decision implications" have since been decided. See
[docs/adr](../adr/README.md) for the records. This document is deliberately not revised to
match them; it stands as the evidence that informed them.

Evidence labels follow the project claim discipline: `[VERIFIED]`, `[OBSERVED]`,
`[USER-PROVIDED]`, `[ASSUMPTION]`, `[INFERENCE]`, `[UNVERIFIED]`.

## Research question

Does the proposed positioning for this project hold up under scrutiny, and what must be
decided before implementation begins?

The proposed positioning is a local-first, backend-agnostic AI model runtime that owns the
inference-serving and model-platform layers, exposes one API and one model registry, and
sits below agent applications rather than containing them.

Two sub-questions follow from that:

1. Is the layer boundary correct, and where does it leak?
2. Which decisions are expensive to reverse and therefore block a first commit of code?

## Sources and limitations

### S1: external research summary, supplied by the project owner

Covers Ollama, llama.cpp, vLLM, SGLang, TensorRT-LLM, TGI, ONNX Runtime GenAI,
OpenVINO GenAI, MLX-LM, the GGUF and Safetensors container formats, and chat-template
handling. Label: `[USER-PROVIDED]`.

Limitations:

- The summary cites upstream repositories and documentation, but those citations were
  collected through a browsing tool and condensed secondhand. Primary sources were not read
  during this pass. Every behavioral claim about an upstream project is therefore
  `[UNVERIFIED]` here.
- The cited projects change quickly. Claims about defaults, environment variables, scheduler
  behavior, and supported backends are version-sensitive, and the summary does not pin a
  revision or date to any of them. Anything acted on should be re-checked against a specific
  upstream commit.
- The summary argues a conclusion. It is a useful and largely well-reasoned argument, but it
  is advocacy for a design, not a neutral survey, and it should be read that way.

### S2: this repository

Inspected directly during this pass. Label: `[OBSERVED]`.

### Not consulted

No benchmarks were run. No upstream source was read. No hardware profile for the target
machine was available. No competitive or licensing review was performed beyond what is
recorded below.

## Evidence table

| ID | Claim | Source | Label | Why it matters |
| --- | --- | --- | --- | --- |
| E1 | The runtime stack separates into compute backend, execution engine, serving runtime, model platform, agent runtime, and product | S1 | `[INFERENCE]` | This decomposition matches how the named projects actually divide responsibility, and it is the basis for the whole scope decision |
| E2 | Ollama is not a thin wrapper over llama.cpp: it pins a llama.cpp revision, patches it, and owns scheduling, registry, and GGUF handling above it | S1 | `[UNVERIFIED]` | If true, an independent platform layer is viable without writing kernels. This is the single most load-bearing claim in the source and is worth verifying first |
| E3 | Inference splits into a compute-heavy prefill phase and a memory-bound decode phase, with a KV cache between them | S1 | `[VERIFIED]` | Standard transformer inference behavior. Drives every scheduling and memory decision |
| E4 | Context length is not free: KV cache size grows with context and consumes accelerator memory | S1 | `[VERIFIED]` | Makes memory planning a first-class runtime concern rather than a tuning afterthought |
| E5 | High-throughput engines converge on continuous batching, paged KV cache, and prefix caching | S1 | `[UNVERIFIED]` | Establishes what a serving layer is expected to provide, and what is inherited free by delegating to an existing engine |
| E6 | Prefix reuse is especially valuable for agent workloads that share long system prompts and tool definitions | S1 | `[INFERENCE]` | Plausible and directionally right, but the size of the win is workload-specific and unmeasured here |
| E7 | Chat models require model-specific control tokens, and a wrong template degrades output quality | S1 | `[VERIFIED]` | Template rendering is a correctness requirement, not a convenience. It belongs in the runtime |
| E8 | GGUF is designed for single-file deployment with memory mapping and typed metadata | S1 | `[UNVERIFIED]` | Supports GGUF as the tier-one format for a zero-configuration local experience |
| E9 | Safetensors avoids arbitrary code execution during tensor loading, unlike pickle-based formats | S1 | `[VERIFIED]` | A genuine security property, and a reason to prefer it over legacy checkpoint formats |
| E10 | Safetensors alone is insufficient to run a model: architecture, tokenizer, template, and generation config are still required | S1 | `[VERIFIED]` | Explains why Safetensors support is materially more work than GGUF support |
| E11 | Supervising the execution engine as a separate process isolates the control plane from engine crashes, hangs, and ABI churn | S1 | `[INFERENCE]` | Strong architectural argument, and the most consequential structural recommendation in the source |
| E12 | Serving systems are measured with TTFT, inter-token latency, time per output token, queue latency, and KV-cache utilization | S1 | `[VERIFIED]` | These are the standard metrics. Instrumenting them early is cheap; retrofitting them is not |
| E13 | Ollama's local API requires no authentication by default | S1 | `[UNVERIFIED]` | Cited as a posture to deliberately diverge from. Worth confirming before repeating the claim publicly |
| R1 | The repository is public and has no LICENSE file | S2 | `[OBSERVED]` | Without a license, the default is exclusive copyright. Outside contribution and downstream use are legally blocked despite the repository being visible |
| R2 | Both existing commits are authored under a name that does not match the identity required by the project's working rules, and both are already pushed | S2 | `[OBSERVED]` | Correcting published history requires a force push, which project rules prohibit without explicit authorization |
| R3 | The repository contains no source tree, no build configuration, and no committed language choice | S2 | `[OBSERVED]` | Every implementation decision below is still fully open and cheap to make |

## Alternatives and tradeoffs

### A. How the execution engine is hosted

| Option | Gains | Costs |
| --- | --- | --- |
| Supervised subprocess | Crash and hang isolation, engine upgrades decoupled from the control plane, engine swappable without relinking | Process lifecycle management, IPC overhead, port and zombie handling, harder debugging across the boundary |
| In-process bindings | Lowest latency, direct memory access, simpler deployment as a single binary | An engine fault takes down the whole runtime, and engine ABI churn becomes the control plane's problem |

The source recommends the subprocess model and cites two independent systems that adopt a
similar split. That reasoning is sound `[INFERENCE]`. The cost it understates is that the
supervision contract itself becomes the hard part: health checking, startup timeout, crash
versus hang discrimination, request draining, port allocation, orphan cleanup when the
control plane dies, and the fate of in-flight streams when a worker disappears. None of that
is specified in the source.

### B. Sequencing: protocol first or implementation first

| Option | Gains | Costs |
| --- | --- | --- |
| Define the wire protocol and internal representation first | Stable contract early, parallel work possible, cleaner public surface | High risk of designing an abstraction that fits no real backend, because no backend has yet constrained it |
| Build one backend end to end, then extract the protocol | The representation is derived from real requirements, and dead abstractions never get written | The public surface churns early, and the first extraction costs a refactor |

The source lists protocol definition as step one. That ordering is questionable
`[INFERENCE]`: an internal representation designed before any backend has exercised it tends
to encode guesses. Building one vertical slice first and extracting the contract afterward
is the lower-risk path, and it aligns with the project rule against speculative
infrastructure.

### C. First execution backend

| Option | Fit | Cost |
| --- | --- | --- |
| llama.cpp | Broad hardware coverage, single-file GGUF models, strong local and mixed CPU/GPU story | Lower peak throughput than GPU-specialized servers |
| vLLM or SGLang | Much higher throughput and concurrency on datacenter GPUs | Heavier dependency, GPU-oriented, poor fit for a local-first first release |
| ONNX Runtime GenAI | Broad platform reach beyond language models | The generation API is described upstream as preview, which is a weak foundation for a first backend |

llama.cpp is the defensible first choice for a local-first product `[INFERENCE]`.

### D. License

The repository is public with no license `[OBSERVED]`. Options are permissive
(Apache-2.0 adds an explicit patent grant; MIT does not), dual permissive as is conventional
in some ecosystems, or a deliberate source-available or proprietary posture. This is on the
project's own list of decisions that require explicit sign-off, and it is materially harder
to change after outside contributions arrive.

A related constraint the source does not raise: distributing a bundled execution-engine
binary makes this project a redistributor, and GPU vendor toolkits carry their own
redistribution terms. That interacts with both the license choice and the packaging design.

## Findings

- **F1 `[INFERENCE]` The layer boundary is correct and is the strongest part of the source.**
  Owning the serving-runtime and model-platform layers, delegating execution to existing
  engines, and keeping agent applications above the runtime is a coherent split. The rule
  that model, backend, provider, and application stay distinct is the right organizing
  constraint.

- **F2 `[INFERENCE]` The proposed build order is a multi-year roadmap presented as a
  sequence.** The early items alone constitute a substantial project. Designing now for
  additional backends, multi-machine workers, and a hosted service is speculative
  infrastructure of exactly the kind the project rules prohibit. Reserving namespace is
  cheap; building the abstraction is not.

- **F3 `[INFERENCE]` Automatic capability detection is the highest-risk proposal in the
  source.** Container metadata can supply architecture, quantization, context length, and the
  chat template. It does not reliably establish whether a model was trained for tool calling
  or structured output; upstream detection of those is substantially heuristic
  `[UNVERIFIED]`. A user-facing capability display that presents heuristics as determinations
  will be confidently wrong for some models. Capabilities should carry provenance and admit
  an explicit unknown state rather than being modeled as booleans.

- **F4 `[INFERENCE]` Memory estimation is the actual difficulty in scheduling, and the
  source treats it as a list item.** A fit decision needs weights, KV cache sized from layer
  count, key/value head count, head dimension, context length, and element width, plus
  compute buffers, plus framebuffer and other-process usage. Errors in either direction are
  costly: optimistic estimates crash, conservative ones waste the device. This needs a
  conservative default and a measured-feedback loop designed in from the start.

- **F5 `[INFERENCE]` The supervision contract is missing and is where the difficulty of the
  recommended architecture actually lives.** The source cites an upstream engine-startup
  timeout defect without drawing the design lesson from it.

- **F6 `[INFERENCE]` Treating model artifacts as untrusted needs concrete controls.**
  Parsing a container format means parsing attacker-influenced binary data with
  length-prefixed fields, which requires bounds validation, allocation ceilings, and a
  metadata size cap enforced before allocation. Separately, chat templates are executable
  templates, so rendering one is code execution over untrusted input and needs a restricted
  environment, a time limit, and an output size cap. The source states the principle without
  either control.

- **F7 `[INFERENCE]` Generalizing to non-text modalities now is premature.** Text generation,
  embeddings, and reranking share enough request shape to unify. Image, video, and audio
  generation differ in streaming semantics, response shape, and resource profile. Avoiding a
  chat-shaped protocol lock-in is worthwhile; building the general abstraction before the
  first modality works is not.

- **F8 `[OBSERVED]` The repository is public with no license,** which leaves it under default
  exclusive copyright and blocks the contribution model that the stated ambition implies.

- **F9 `[OBSERVED]` Published commits do not satisfy the project's own identity rule,** and
  the correction requires a prohibited operation. The repository-local identity has since
  been configured, so subsequent commits comply.

- **F10 `[ASSUMPTION]` No target hardware profile is recorded anywhere in the project.**
  Accelerator memory, host memory, and platform determine which models are reachable and
  which scheduling problems are real. Several design arguments above cannot be resolved
  without it.

## Decision implications

Blocking before implementation starts, because each is expensive to reverse:

1. License, and the packaging posture for redistributed engine binaries.
2. Implementation language and the public interface surface.
3. Whether the first slice targets a supervised subprocess or in-process bindings.
4. Whether the protocol is defined up front or extracted from a working slice.

Not blocking, and safe to decide during implementation:

- Choice of first execution backend, given local-first framing points clearly at one option.
- Metric names and telemetry transport, provided the measurement points are instrumented
  from the first slice.
- Whether Safetensors support lands in the first or a later release.

## Remaining gaps

- No upstream claim in this document has been checked against primary sources. E2, E5, E8,
  and E13 in particular are load-bearing and unverified.
- No target hardware profile exists, so no memory or throughput budget can be stated.
- No performance target has been set, so there is no way to tell whether a backend choice is
  adequate.
- No threat model exists, so the security controls in F6 are stated as categories rather
  than requirements.
- No decision record mechanism exists in the repository yet.

## Deliberately not carried over

- Advocacy framing, product positioning language, and comparative marketing claims from the
  source.
- Specific upstream defaults, environment-variable names, and flag behavior, all of which are
  version-sensitive and were not verified.
- The proposed multi-stage roadmap as a commitment. It is recorded above as an option under
  evaluation, not as a plan.

## Handoff

```yaml
handoff:
  from_skill: masterbuilder-researcher
  to_skill: masterbuilder-architect
  project_phase: pre-implementation
  objective: >
    Convert these findings into decision records covering license, language,
    engine hosting model, and protocol sequencing.
  scope: >
    Architecture decisions only. No implementation, no product scope changes.
  inputs_used:
    - external research summary supplied by the project owner
    - direct inspection of this repository
  decisions:
    - none taken; this pass produces findings only
  artifacts_changed:
    - docs/research/runtime-landscape-distillation.md (added)
  evidence: see evidence table above
  tests_or_runtime_checks: none applicable to a research pass
  risks_and_blockers:
    - public repository with no license (F8)
    - capability detection presented as determinate when it is heuristic (F3)
    - memory estimation underspecified (F4)
    - supervision contract unspecified (F5)
  assumptions:
    - target hardware profile is unknown (F10)
  not_run:
    - no upstream source was read
    - no benchmark was executed
    - no threat model was produced
  next_action: >
    Record the four blocking decisions listed under Decision implications,
    then define the first vertical slice.
```
