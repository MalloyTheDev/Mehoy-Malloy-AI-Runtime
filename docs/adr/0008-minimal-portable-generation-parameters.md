# 8. Only portable generation parameters are normalised

- Status: Accepted
- Date: 2026-09-07
- Related: [ADR-0006](0006-general-model-runtime-not-an-llm-runtime.md),
  [ADR-0007](0007-chat-is-not-the-inference-primitive.md)

## Context

Sampling controls are the part of a generation interface that grows without limit.
One engine alone offers top-k, top-p, min-p, typical-p, repetition and frequency and
presence penalties, mirostat, XTC, dynamic temperature, sampler ordering, grammars,
logit bias, and speculative decoding, and the list changes between releases.

Every parameter the runtime normalises becomes a promise: that it means the same
thing on every backend, and that it will keep meaning that. Most of these cannot
carry such a promise, because they either do not exist elsewhere or exist with
different semantics. Normalising them anyway produces a structure of several dozen
backend-specific knobs pretending to be universal, which is worse than not having
them: a caller cannot tell which are portable and which are not.

## Decision

### 1. Only controls with stable cross-backend meaning are normalised

The initial set is exactly four:

| Parameter | Meaning |
| --- | --- |
| `max_output_tokens` | The most tokens the runtime may generate for this request. Output only |
| `temperature` | Sampling temperature, where zero means the most deterministic sampling the backend offers |
| `seed` | A requested sampling seed |
| `stop` | Text sequences whose appearance ends generation |

Nothing else is portable enough to promise yet. A parameter joins this set when a
second backend exists to compare it against, not because one engine has it.

### 2. Semantics are stated, not inherited

`max_output_tokens` bounds generated output. It is not a context limit, and it does
not include the input. An engine that expresses the same idea differently is the
adapter's problem.

`temperature` must be finite and not negative. An invalid value is rejected rather
than clamped: silently changing a caller's number produces output they did not ask
for and cannot explain.

`stop` is bounded in count and in size. Request content is untrusted, and an
unbounded list of unbounded patterns is a denial-of-service surface rather than a
feature.

### 3. A seed promises repeatability within a backend, not determinism

The same seed on the same backend build and hardware may reasonably reproduce the
same output. The same seed across backends, builds, or accelerators does not, and
this runtime does not claim otherwise. Quantisation, kernel selection, batching, and
floating point ordering all move the result.

Recording this now matters because a seed parameter reads like a determinism
guarantee, and someone will eventually depend on it as one.

### 4. Omission is not a default

A parameter left unset means the runtime did not override it, not that it equals
whatever the backend currently defaults to. The adapter passes nothing and the
backend applies its own default.

This distinction is load-bearing. Materialising defaults into the request would
freeze one engine's current values into the runtime's contract, and a caller could
no longer express "whatever this backend thinks best" as distinct from a number that
happens to match today's default.

### 5. Backend-specific controls stay out of the portable namespace

Engine-native controls will eventually be reachable through an explicitly
backend-scoped mechanism, so an advanced caller can use them while remaining aware
that they are not portable. That mechanism does not exist yet and is not needed to
generate text.

## Consequences

### Gained

- A small interface whose every member can be honoured by a future backend.
- Compatibility testing across backends is tractable, because there are four things
  to compare rather than forty.
- The line between portable and engine-specific is visible in the type rather than
  in documentation nobody reads.

### Accepted costs

- Fewer controls than the current backend offers, so some behaviour reachable
  through the engine directly is not reachable through the runtime yet.
- An extension mechanism will be needed before advanced use is possible.
- Each future addition requires deciding whether it is genuinely portable, which is
  slower than adding it because one engine has it.

### Deliberately not decided

Where a conversation template is rendered, and how its correctness is established.
Raw continuation needs no template, so that decision blocks the first conversational
input rather than the first generation.

## Alternatives considered

| Alternative | Why not |
| --- | --- |
| Mirror the current backend's parameters | Freezes one engine's surface into the runtime contract, and every one of those names becomes a compatibility obligation for backends that have no equivalent |
| Pass an untyped map straight through | Abandons portability entirely and makes every caller backend-specific, which is the coupling the backend boundary exists to prevent |
| Normalise now, prune later | Removing a parameter is a breaking change; not adding one is not. The asymmetry favours starting small |
