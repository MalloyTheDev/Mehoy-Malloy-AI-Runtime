# 4. Local transport is HTTP over a Unix socket or named pipe, with control and inference on separate surfaces

- Status: Accepted
- Date: 2026-09-07
- Related: [ADR-0001](0001-rust-daemon-first-runtime.md),
  [ADR-0003](0003-protocol-first-vertical-slice.md),
  [runtime landscape distillation](../research/runtime-landscape-distillation.md)
- Resolves: the transport decision left open by ADR-0001

## Context

ADR-0001 established that the daemon is the public surface but deliberately did not choose a
transport, because the choice is simultaneously a public API decision, a wire protocol
decision, and a security model decision.

Two facts constrain it.

**The daemon is privileged in practice.** It holds resident models, can consume most of the
machine's accelerator memory, reads model files from disk, and decides what gets loaded and
evicted. Reaching the daemon means controlling those resources, regardless of what the
operation is nominally called.

**Not all operations carry the same privilege.** Requesting a generation and evicting a
resident model are different acts. The research pass recorded that at least one comparable
runtime exposes an unauthenticated local API `[UNVERIFIED, tracked in #1]`, and flagged it as
a posture to diverge from deliberately rather than inherit by default.

A loopback TCP socket does not distinguish callers. Any process in any session on the machine
can connect to it, and there is no portable peer-credential check that behaves the same way
across Linux, macOS, and Windows.

## Decision

### 1. Transport

Clients reach `mehoyd` over HTTP carried on a local-only transport:

- a Unix domain socket on Linux and macOS;
- a named pipe on Windows.

No TCP listener exists in the first implementation, on either surface.

### 2. Two surfaces, not one

**Control surface.** Model import and registry, load, unload, worker lifecycle,
configuration, runtime status. Local transport only. No configuration option exposes it over
a network socket. This is a structural property, not a policy setting.

**Inference surface.** Generation, token streaming, cancellation. Local transport by default.
It may gain an opt-in network listener in a later record, behind explicit configuration,
authentication, and transport security. That decision is not taken here.

```
   local clients
        |
        v
  +-------------------------------+
  |  Unix socket / named pipe     |
  +---------------+---------------+
                  |
        +---------+---------+
        |                   |
   control surface     inference surface
   (local only,        (local now, network
    always)             possible later)
```

### 3. Authorization is delegated to the operating system

Access control for the local transport is filesystem permissions on the socket, and the pipe
access-control list on Windows. The daemon does not implement its own authentication scheme
for local callers. Writing one would mean reproducing, less well, a check the operating
system already performs.

### 4. The daemon is per-user, not system-wide

The socket or pipe is created in a per-user location with owner-only access. A shared machine
does not get one daemon that every account can drive.

## Consequences

### Gained

- Authorization comes from the operating system rather than from code that must be written,
  reviewed, and kept correct.
- The runtime is unreachable off-host by construction. There is no network listener to
  misconfigure, so accidental exposure is not a failure mode that exists.
- HTTP semantics are retained, so server-sent events cover token streaming, status codes
  behave as clients expect, and a vendor-compatible endpoint stays cheap to add later.
- The privilege split is structural. Exposing inference over a network in future cannot
  accidentally expose control, because control is not on that surface at all.

### Accepted costs

- Two transport implementations rather than one.
- Third-party HTTP clients need transport support. This is widely available but not
  universal, and it is less uniform for named pipes than for Unix sockets.
- Debugging is marginally less convenient than a TCP port, though `curl` supports Unix
  sockets directly.
- The socket or pipe location becomes part of the client contract, and needs a documented
  default plus an override.

### To be specified during the first slice

- **Default location per platform**, per-user rather than system-wide, and the environment
  variable or configuration key that overrides it.
- **Creation permissions**: the socket created owner-only, and the pipe created with an
  access-control list restricted to the creating user. These must be set at creation, not
  applied afterward, so no window exists where the endpoint is open.
- **Stale endpoint handling**: a socket file left behind by an unclean exit must neither
  block startup nor be mistaken for a live daemon. The liveness check must not be "the file
  exists".

### Explicitly deferred

- Whether the inference surface ever gains a network listener, and what would authenticate it.
- Whether peer-credential inspection is used for anything beyond the operating system's own
  check, for example to attribute requests to a caller in telemetry.

## Alternatives considered

| Alternative | Why not |
| --- | --- |
| Loopback HTTP on `127.0.0.1` | Reachable by any local process in any session until a token layer is built, and Windows offers no peer-credential check for TCP. It also makes "is the daemon running" and "can this process control my accelerator" the same question |
| Raw framed protocol over socket or pipe | Same operating-system access control and a smaller surface, but gives up `curl` debugging and the inexpensive path to a vendor-compatible endpoint, in exchange for a latency saving that is negligible next to token generation cost |
| One combined surface | Simpler to implement, but any caller able to request a generation could also evict resident models and repoint the registry. Merging the privilege levels is the specific pattern this record exists to avoid |
| System-wide daemon | Would let any account on a shared machine drive another user's models and read the registry |
