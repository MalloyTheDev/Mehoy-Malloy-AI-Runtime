# 1. Rust control plane with a daemon-first public surface

- Status: Accepted
- Date: 2026-09-07
- Supersedes: nothing
- Related: [runtime landscape distillation](../research/runtime-landscape-distillation.md)

## Context

The project is positioned as a local-first, backend-agnostic model runtime that owns the
inference-serving and model-platform layers and sits below agent applications rather than
containing them.

That positioning imposes concrete requirements on the control plane:

- supervise long-running child processes and survive their failure;
- account for accelerator and host memory before admitting work;
- serve concurrent streaming requests with cancellation;
- run on Windows, Linux, and macOS;
- hold models resident across client invocations.

More than one client surface is anticipated: a command-line interface, language SDKs, a
desktop application, and other tools. If the command-line interface were the runtime, each
of those would have to shell out to it or reimplement it, and no model could stay resident
between invocations.

The project owner reports existing Rust work `[USER-PROVIDED]`. This is a preference and
familiarity input, not independently verified and not a technical constraint on its own.

## Decision

1. The control plane is written in Rust.

2. The daemon is the system. A long-running process, `mehoyd`, owns configuration, the model
   registry, scheduling, worker supervision, and the public API.

3. The command-line interface, `mehoy`, is one client among several and carries no privileged
   position. It must be replaceable without changing the runtime.

4. The public surface of this project is the daemon's protocol, not the command-line
   interface. Command-line output formatting is not a compatibility surface.

```
  CLI      SDKs     Desktop UI     other applications
   |         |          |                 |
   +---------+----------+-----------------+
                        |
                 Mehoy protocol
                        |
                        v
                     mehoyd
```

## Consequences

### Accepted costs

- A daemon carries lifecycle burden the command-line interface would not: installation,
  autostart or on-demand start, upgrade, and shutdown semantics on three platforms.
- Client and daemon can drift in version. Once SDKs exist, protocol compatibility becomes a
  real obligation rather than an internal detail.
- Debugging spans a process boundary from the first day.

### Gained

- Models stay resident across invocations, which is the whole point of a runtime rather than
  a launcher.
- Every client surface is equal, so the desktop application and the agent harness are not
  second-class relative to the command-line interface.
- The command-line interface can be rewritten or replaced without touching the runtime.

### Deferred, and blocking before the first slice

**The local transport and its security model are not decided by this record.**

Candidate transports are loopback HTTP, a Unix domain socket, a Windows named pipe, or a
structured RPC layered over any of those.

The choice is security-relevant, not merely ergonomic:

- The primary development host is Windows `[OBSERVED]`. `AF_UNIX` exists on current Windows
  builds, but tooling and library support is less uniform than on Unix. Named pipes are the
  native Windows mechanism and carry access-control lists.
- A loopback TCP socket is reachable by any process running on the machine. Without
  authentication, any local process could load models, submit prompts, and read responses.
  There is no portable peer-credential check that works the same way across the three target
  platforms.
- Filesystem-permission-based transports (Unix sockets, named pipes) get an access-control
  story from the operating system, at the cost of being less convenient for clients that
  already speak HTTP.

Under the project rules, security models require explicit sign-off. This decision is
therefore recorded as open, and a first slice should not bind the transport by accident.

## Alternatives considered

| Alternative | Why not |
| --- | --- |
| Command-line interface as the runtime | Cannot hold models resident between invocations, which defeats the purpose. Forecloses the desktop and SDK surfaces, or forces them to screen-scrape a command-line tool |
| Go control plane | Viable, and demonstrably adequate for this class of system. Rejected on the owner's stated familiarity and on Rust's stronger guarantees when parsing untrusted binary model containers |
| C++ control plane | Closest to the execution engines, but gives up memory-safety guarantees exactly where untrusted input is parsed |
| Python control plane | Fast to write, weakest fit for long-lived process supervision, resource accounting, and single-binary distribution |
