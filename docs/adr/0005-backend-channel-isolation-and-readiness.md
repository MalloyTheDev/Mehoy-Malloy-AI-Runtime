# 5. Backend channels are private, and backend readiness is not inference

- Status: Accepted
- Date: 2026-09-07
- Related: [ADR-0002](0002-out-of-process-inference-workers.md),
  [ADR-0004](0004-local-transport-and-surface-split.md)

## Context

ADR-0002 puts execution engines in separate processes. The first such engine
serves HTTP. Starting it therefore creates a second network endpoint, one that
ADR-0004 never considered, because ADR-0004 was about the runtime's own surface.

That endpoint is not a minor detail. It exposes generation and model operations
for as long as a model is loaded. Started carelessly it would be an
unauthenticated listener on a predictable local port: a side door standing open
next to a front door the runtime is otherwise careful about. The research pass
recorded an existing runtime's unauthenticated local API as a posture to diverge
from deliberately, and accidentally reproducing it one layer down would be the
same mistake wearing a different hat.

Separately, the backend reports its own health, and it is tempting to treat a
healthy response as proof the backend works. It is not. There are reported upstream
cases where health reported success while a failed initialisation had left the
backend unusable. A runtime that equates the two would tell an operator a model is
working when it cannot generate a token.

## Decision

### 1. A backend channel is private runtime plumbing, not an API

Every backend process is started with:

- **loopback only.** The backend is never told to bind a routable address.
- **a port chosen at run time.** No fixed or well-known port, so the endpoint is
  not sitting at a guessable location for the life of a model.
- **a per-worker secret**, required on every request, generated from the operating
  system's randomness source. Generation failure is a startup failure rather than
  a fallback to something predictable, because a predictable secret looks like
  protection while providing none.
- **its user-facing interface disabled.** The backend is plumbing; it does not get
  to present a surface of its own.

No client ever learns a backend's address. The client's contract is the runtime's
local endpoint from ADR-0004, and the backend sits behind it.

```
   client
     |
  local endpoint (ADR-0004)
     |
   mehoyd
     |
  loopback + per-worker secret
     |
   backend process
```

### 2. The secret is opaque by construction

The secret type does not implement `Display`, and its `Debug` redacts the value.
A credential that can reach a log or an error message through ordinary formatting
will eventually do so, and the type system is a better guarantee than a convention.

### 3. Backend readiness and inference are separate claims

Two distinct states exist, and the weaker one may never be promoted to the
stronger:

- **Backend ready.** The process is alive, its endpoint answers, and it reports a
  serving state. That is all it means.
- **Inference verified.** A generation has actually been performed and produced
  output.

Nothing establishes the second yet. It exists now so that nothing accidentally
reports the first as though it were the second.

### 4. Readiness is a protocol, not log parsing

Startup progresses through observable states derived from the backend's endpoint:
waiting for the endpoint to appear, the endpoint reporting it is still coming up,
and the endpoint reporting a serving state. A "still loading" response is a normal
intermediate state and not a failure; a rejected credential is terminal and is
reported immediately rather than retried until a deadline.

Backend output is captured as evidence, and is not consulted to decide readiness.
Logs change without notice; an endpoint contract is a contract.

## Consequences

### Gained

- A loaded model is not accompanied by an open, unauthenticated local service.
- Failure reports name the exact backend build and carry the backend's own output,
  which is where the reason for a failed start almost always is.
- The difference between "it started" and "it works" is visible in the type system
  rather than left to whoever reads the status.

### Accepted costs

- The port is discovered by binding one and releasing it, so another process could
  take it in between. The backend then fails to start, which is a visible startup
  failure rather than a silent mis-binding. Handing the backend a listening socket
  directly would close the gap, but it takes a port number rather than a socket.
- A secret is passed on a command line, so it is visible to anything that can list
  process arguments on the machine.

### What this does not defend against

Code already running as the same operating system user. ADR-0004 states that
boundary and this record does not widen it. What is prevented is the runtime
publishing an unauthenticated backend to every local process by accident.

## Alternatives considered

| Alternative | Why not |
| --- | --- |
| Fixed backend port, no credential | Simplest, and exactly the pattern this record exists to avoid. Any local process could drive the backend for as long as a model is loaded |
| A Unix socket for the backend channel | Better than loopback where available, since the operating system enforces access. Worth adopting on Unix later; not done now because it would split the backend channel across platforms before either has run in anger |
| Treat a healthy response as proof of working inference | Overstates the evidence. Health has been observed reporting success on a backend left unusable by a failed initialisation |
| Decide readiness by matching the backend's log output | Couples the runtime to strings that change without notice, and mistakes diagnostics for protocol |
