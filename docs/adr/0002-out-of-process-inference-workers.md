# 2. Inference engines run as supervised out-of-process workers

- Status: Accepted
- Date: 2026-09-07
- Related: [ADR-0001](0001-rust-daemon-first-runtime.md),
  [runtime landscape distillation](../research/runtime-landscape-distillation.md)

## Context

The runtime delegates model execution to existing engines rather than implementing kernels.
The first such engine is expected to be llama.cpp, with others following.

Execution engines in this space fail in ways ordinary application code does not: accelerator
out-of-memory conditions, driver faults, hangs that never return, and abrupt aborts. They
also change quickly, and their application binary interfaces are not stable across releases
`[UNVERIFIED, tracked in #1]`.

Two hosting models are available. Either the engine is linked into the control plane and
called directly, or it runs as a separate process the control plane supervises.

## Decision

Execution engines run as separate processes. `mehoyd` spawns, supervises, health-checks,
terminates, and restarts them. No engine is linked into the control plane in the first
implementation.

```
   mehoyd
     |
     |  spawn / supervise / health-check / terminate / restart
     v
  worker process (llama.cpp today, others later)
```

An in-process backend may be introduced later, for a specific engine, only where measurement
shows the process boundary is materially costly. That is explicitly not a first-release
concern.

## Consequences

### Gained

- An engine crash, hang, or out-of-memory kill removes a worker, not the runtime. Registry,
  scheduler, and in-flight bookkeeping survive.
- Engines are upgradable independently of the control plane, and upstream interface churn
  does not propagate into the Rust build.
- Backends become genuinely interchangeable, because the seam is a process boundary rather
  than a linkage decision.

### Accepted costs

- Every request crosses a process boundary, which costs serialization and a copy. Token
  streaming is the highest-frequency path and therefore the one to measure first. This cost
  is accepted deliberately and is not to be optimized speculatively.
- Debugging spans processes.
- The supervision logic becomes a real component with its own failure modes, rather than a
  function call.

### The supervision contract must be specified before implementation

Choosing this architecture moves the difficulty into supervision. The following are
undecided and must be settled as part of the first slice, not discovered during it:

- **Startup**: how a worker is determined to be ready, and the timeout after which a
  non-responsive start is treated as a failure.
- **Liveness**: how a hang is distinguished from slow legitimate work. A crashed process is
  easy to detect; a wedged one is not.
- **Port and endpoint allocation**: how a worker's endpoint is assigned without racing
  another worker, and bound so it is not reachable off-host.
- **Draining**: what happens to queued and in-flight requests when a worker is asked to stop.
- **In-flight streams on crash**: what a client observes when the worker serving its stream
  disappears mid-generation. This must be a defined protocol event, not a dropped connection.
- **Restart policy**: whether, how often, and under what backoff a worker is restarted, and
  when repeated failure is surfaced instead of retried.
- **Orphan cleanup**: what happens to worker processes when `mehoyd` itself dies.

That last point has a platform-specific answer worth recording now. On Windows there is no
equivalent of a parent-death signal, so a child outlives an abruptly terminated parent by
default. The native mechanism is a Job Object with kill-on-job-close semantics, which
guarantees children are reaped when the daemon's handle closes. Unix targets need their own
approach, typically process groups. Without this, an unclean daemon exit leaks worker
processes that continue to hold accelerator memory `[INFERENCE]`.

### The first backend partly shapes the backend protocol

The intended first engine exposes an HTTP server. If workers are addressed over HTTP,
that engine's request and response shapes will exert pull on the backend protocol. ADR-0003
addresses this directly; the mitigation is that the backend protocol is defined as
backend-neutral first and then implemented, rather than derived from whatever the first
engine happens to expose.

## Alternatives considered

| Alternative | Why not |
| --- | --- |
| Link the engine in-process | Lowest latency and a single binary, but any engine fault takes the runtime with it, and upstream interface churn becomes a build problem. Wrong tradeoff for a system whose value is supervision and lifecycle management |
| Both, selected at build time | Doubles the surface to test and reason about before either path has been proven. Available later if measurement justifies it |
