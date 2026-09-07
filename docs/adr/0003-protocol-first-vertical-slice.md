# 3. Protocol defined first, validated through one minimal vertical slice

- Status: Accepted
- Date: 2026-09-07
- Related: [ADR-0001](0001-rust-daemon-first-runtime.md),
  [ADR-0002](0002-out-of-process-inference-workers.md),
  [runtime landscape distillation](../research/runtime-landscape-distillation.md)

## Context

Two failure modes bracket this decision.

Deriving the public interface from whatever the first execution engine supports permanently
leaks that engine into the architecture, and the project becomes a wrapper around it. That
defeats the backend-agnostic positioning entirely.

Designing a comprehensive protocol before anything runs produces abstractions shaped by
guesses rather than by real constraints, and violates the project rule against speculative
infrastructure. The research pass flagged an up-front full protocol design as the weaker of
the two orderings.

Both extremes are rejected.

## Decision

Define the smallest backend-neutral protocol sufficient for one complete operation, then
implement exactly that, then revise the protocol under the pressure the implementation
reveals.

### The minimal operation

```
  discover model -> register model -> load model -> submit generation
    -> stream tokens -> cancel request -> unload model
```

Nothing outside this path is designed in the first pass.

### Three protocol families, kept separate

A single combined type system is explicitly rejected. Three seams exist and they have
different audiences, different stability obligations, and different rates of change.

**Client protocol**, between applications and `mehoyd`. This is the compatibility surface
and the one that must not churn casually. Covers listing models, loading, generating,
cancelling, unloading, and querying runtime status.

**Backend protocol**, between `mehoyd` and a worker. Internal, and free to change with the
runtime. Covers capability reporting, load, generate, abort, health, and shutdown.

**Event protocol**, carrying asynchronous state changes: model loading and ready, request
queued, prefill started, token, request completed, request cancelled, worker crashed, model
unloaded.

### Note on the event protocol

Those events are not homogeneous, and the design should not treat them as one stream
`[INFERENCE]`. Token events are per-request, high-frequency, ordered within a request, and
useless once the request ends. Worker-crashed and model-unloaded events are global,
low-frequency, and matter to subscribers with no request in flight. Delivering both through
one channel with one set of guarantees means either over-engineering the token path or
under-serving the lifecycle path. Separating per-request streams from a runtime event
channel is the safer starting shape.

### Model identity is defined now, not later

Models are not identified by file path. Two distinct concepts are separated from the start,
because merging them is the mistake that becomes expensive once scheduling exists.

**Artifact**, the immutable thing on disk: identifier, content hash, format, architecture,
quantization, source, local path, metadata.

**Instance**, a loaded occurrence of an artifact: instance identifier, artifact reference,
backend, device placement, context capacity, reserved memory, active request count,
lifecycle state.

One artifact may have zero, one, or several instances. An artifact is a fact; an instance is
a resource reservation.

One open question follows immediately and should be settled during the slice: hashing a
multi-gigabyte artifact on every import is not free, so the design must decide what the
content hash covers, whether it is computed eagerly or lazily, and what cheaper identity is
used before a hash exists `[INFERENCE]`. Single-file containers make this tractable;
multi-file formats will make it harder.

## Consequences

- The protocol exists before the implementation, so the first engine cannot silently define
  the architecture.
- The protocol is small enough that revising it after the first slice is cheap, which is the
  entire point of validating it against a working path.
- The protocol will be revised. It is not a compatibility commitment until a release says it
  is, and that should be stated wherever the protocol is published.
- Work outside the seven operations is out of scope for the first slice, including batching,
  multi-model residency policy, remote providers, non-text modalities, and any second
  backend.

## First milestone

The slice is deliberately narrow. Import a local model artifact, then run a single-turn
generation against it and stream the result.

Underneath, that path exercises daemon lifecycle, the client transport, model identity, the
registry, backend selection, worker supervision, health checking, streaming, cancellation,
error propagation, process cleanup, and basic resource inspection. It looks trivial from the
outside and proves nearly every foundation.

Explicitly excluded from this milestone: memory-aware admission and scheduling, multiple
resident models, compatibility endpoints for other vendors' wire formats, and integration
with any downstream application.

## Deliberate non-goal: no downstream integration yet

No existing application is integrated during this work. The runtime must be proven through a
generic client first.

The boundary test is stated in advance: a downstream agent application should be able to
replace its local model path with this runtime without the runtime learning anything about
repositories, coding tools, permission gates, or version control. If the runtime has to grow
knowledge of any of those to make the integration work, the boundary is in the wrong place
and the integration has become a design dependency rather than a client.

## Alternatives considered

| Alternative | Why not |
| --- | --- |
| Implement the first backend, then expose what it supports | Permanently couples the public interface to one engine. The stated positioning does not survive it |
| Design the full protocol before implementing | Encodes guesses as contracts. The research pass identified this as the higher-risk ordering, and it conflicts with the project rule against speculative infrastructure |
| One unified protocol type system | Merges three seams with different audiences, stability obligations, and change rates. Forces internal churn to surface as public breakage |
