# Mehoy Malloy AI Runtime

A local-first, backend-agnostic AI model runtime.

## Status

Early. There is no inference yet.

What exists is the runtime skeleton: a daemon that listens on a per-user local
endpoint, a client that talks to it, and the security properties that endpoint has to
get right before anything else is built on top of it.

**Implemented**

- `mehoyd`, a daemon listening on a per-user local endpoint
- `mehoy`, a command-line client
- Unix domain socket transport on Unix, named pipe transport on Windows
- `GET /health` and `GET /v1/runtime`
- Supervision of worker processes: readiness probing, startup and shutdown
  deadlines, an explicit state machine, and reporting rather than restarting a
  crashed worker
- Orphan cleanup, so no worker survives the abrupt death of the process that owns
  it. Verified on Windows via a Job Object, against both a stand-in worker and a
  real `llama-server`; the Linux mechanism is implemented but unverified
- A llama.cpp backend that locates and identifies an executable, starts it under
  supervision on a private loopback channel behind a per-worker secret, maps its
  health endpoint to a readiness state, and captures its output for diagnostics

**Not implemented**

- Model import, registry, or loading
- Inference of any kind, and therefore no token streaming or cancellation
- Backend installation or download. The executable is configured explicitly
- Memory-aware scheduling
- Any vendor-compatible endpoint

**Known limitations**

- Unix code compiles and is type-checked, but has never been executed. Only the
  Windows paths have been run. This covers the socket transport and the parent-death
  orphan mechanism. See the open issues.
- The protocol is not a compatibility commitment. It is expected to change.
- The worker readiness and shutdown line protocol is provisional. It exists so
  supervision can be proven against a controllable worker; the llama.cpp backend
  uses its health endpoint instead.
- Backend readiness means the process answers and reports a serving state. It is
  **not** evidence that generation works. No model has been loaded and no
  generation has been performed, so nothing has established that yet.

Architecture decisions are recorded in [docs/adr](docs/adr/README.md). The research
behind them is in
[docs/research](docs/research/runtime-landscape-distillation.md).

## Build

```bash
cargo build --workspace
```

## Run

Start the daemon:

```bash
cargo run --bin mehoyd
```

It prints the endpoint it is listening on. In another shell:

```bash
cargo run --bin mehoy -- health
```

```bash
cargo run --bin mehoy -- runtime
```

Both accept `--endpoint <address>` to target a specific endpoint, and both read
`MEHOY_ENDPOINT` when that flag is absent. The daemon stops on Ctrl-C, and on `SIGTERM`
where that exists.

`mehoy` exits `0` on success, `1` when the request failed, `2` when the command line was
invalid, and `3` when no daemon is listening.

## Validation

```bash
cargo fmt --all --check && cargo clippy --workspace --all-targets && cargo test --workspace
```

`cargo test --workspace` builds the example binaries the supervision tests execute,
so it is sufficient on its own. Running a single test target with `--test` does not
build them.

Tests that need a real llama.cpp backend are skipped unless one is configured, and
say so when they skip:

```bash
MEHOY_LLAMA_SERVER=/path/to/llama-server cargo test --workspace
```

To trace endpoint lifecycle transitions, including raw operating system errors:

```bash
MEHOY_TRACE_ENDPOINT=1 cargo run --bin mehoyd
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

The SPDX expression for this project is `MIT OR Apache-2.0`.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed
as above, without any additional terms or conditions.
