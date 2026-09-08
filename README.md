# Mehoy Malloy AI Runtime

A local-first, backend-agnostic AI model runtime.

Not a language-model runtime. Text generation is intended to be one task among many
rather than the assumed default, so embeddings, reranking, classification, vision,
audio, and generative media can arrive through further backends without a redesign.
See [ADR-0006](docs/adr/0006-general-model-runtime-not-an-llm-runtime.md).

## Status

Early, and it performs real inference.

One complete lifecycle works end to end: a GGUF container is registered, loaded onto
a supervised backend, asked to embed or to generate, streamed incrementally,
cancelled by identity, and unloaded so that nothing survives it. One backend, one
artifact format, one model at a time.

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
  real `llama-server`, and on Linux via `prctl(PR_SET_PDEATHSIG)`. macOS and the
  BSDs have no equivalent mechanism and are not covered
- A llama.cpp backend that locates and identifies an executable, starts it under
  supervision on a private loopback channel behind a per-worker secret, maps its
  health endpoint to a readiness state, and captures its output for diagnostics
- A GGUF reader and a durable artifact registry. Containers are inspected by the
  runtime itself rather than by a backend, validated as untrusted input, and
  recorded without being copied or moved
- Loading a registered artifact into a backend, producing a live in-memory model
  instance. Startup is transactional: a failure at any stage, including after the
  backend is healthy, leaves no worker, credential, or instance behind
- Embeddings, end to end: a real authenticated request through the backend adapter,
  structurally validated, promoting the capability from indicated to verified only
  on success
- Loading a text-generative model through the same call, transaction, and capability
  rules as an embedding model, with the backend choosing its own launch mode from
  the artifact's metadata
- Text generation from a raw continuation, with a deliberately small portable
  parameter set, verified the same way embeddings are: by performing one
- Incremental delivery of a generation as runtime-native events, with the engine's
  event framing and wire format confined to the backend adapter, and bounded
  buffering so a slow consumer slows the backend rather than this process
- Cancelling a request by identity, independently of who is reading its output,
  with an idle budget that shares the same stopping machinery while reporting its
  own distinct cause
- Unloading an instance: admission closes first, outstanding requests are stopped
  through that same machinery, they are given a bounded time to settle, and only
  then is the worker taken away

**Not implemented**

- Observing that a backend has actually stopped. The runtime knows it released the
  request's transport; whether the engine then stops promptly is a property of the
  engine, measured per build rather than guaranteed here
- Concurrency limits and queueing. Requests have identity, which is what those
  would need, and neither exists
- Conversation input. Continuation is the primitive; a conversation shape arrives
  when something can render a template for it
- Any model class beyond text embeddings and text generation. Nothing prevents them;
  nothing implements them yet
- Taking ownership of an artifact into a managed store. Registration records where
  a file is; it never copies or moves it
- Backend installation or download. The executable is configured explicitly
- Memory-aware scheduling
- Any vendor-compatible endpoint

**Known limitations**

- The Unix code paths run on Linux and are unverified everywhere else. The socket
  transport and the parent-death orphan mechanism are exercised there as part of
  the default gate. macOS and the BSDs have no equivalent to the mechanism used on
  Linux, so on those platforms a worker outliving an abruptly killed owner is a
  real possibility rather than a theoretical one. See the open issues.
- Real inference is verified on Windows only, because that is where the backend
  executable on this machine runs. The runtime's own behaviour, including every
  streaming and cancellation failure path, is verified on both platforms.
- The protocol is not a compatibility commitment. It is expected to change.
- The worker readiness and shutdown line protocol is provisional. It exists so
  supervision can be proven against a controllable worker; the llama.cpp backend
  uses its health endpoint instead.
- Backend readiness means the backend loaded the artifact and accepts authenticated
  requests. It is **not** evidence that any particular kind of request works. Only
  performing a task verifies it, and a verification belongs to the live instance and
  the backend build that produced it, not permanently to the artifact.

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

The Unix code paths are additionally type-checked from Windows. The crates named
are the ones holding platform-specific code; the rest are omitted because the
registry's bundled SQLite needs a Linux C compiler and everything depending on it
would pull that in:

```bash
cargo clippy -p mehoy-core -p mehoy-backend-llama -p mehoy-protocol --all-targets --target x86_64-unknown-linux-gnu
```

This is a compile check, not a substitute for running the suite on Linux.

Dependencies are checked for known advisories and for licence and source policy.
The policy lives in `deny.toml`; without it the licence check has an empty
allow-list, so it rejects everything and reports nothing useful:

```bash
cargo audit && cargo deny check
```

`cargo test --workspace` builds the example binaries the supervision tests execute,
so it is sufficient on its own. Running a single test target with `--test` does not
build them.

### Two gates

The default gate and the real-backend gate answer different questions, and only the
first is expected to pass on every machine and every commit.

| | Default gate | Real-backend gate |
| --- | --- | --- |
| Command | `cargo test --workspace` | see below |
| Needs a backend executable | no | yes |
| Needs a multi-gigabyte model | no | yes |
| Deterministic | yes | no, it measures a running engine |
| Portable | yes | no, results are specific to one build |
| Runtime | seconds | minutes |
| Expected on every commit | yes | no |

The default gate covers this runtime's own behaviour, including every failure path
of streaming and cancellation, against servers that misbehave on request. Nothing
in it loads a model.

The real-backend gate covers what a particular engine build actually does: that a
model loads and generates, and that stopping a request stops the work rather than
merely stopping the output. Those are properties of that engine rather than of this
code, which is why they are opt-in. The test targets are not built without their
feature, so they cost the default gate nothing.

```bash
MEHOY_LLAMA_SERVER=/path/to/llama-server cargo test --workspace \
    --features mehoy-backend-llama/real-backend-tests,mehoy-runtime/real-backend-tests
```

Those tests still skip, and say so, when the executable or a suitable model is
absent. Run the cancellation characterisation on its own to see what it measured:

```bash
MEHOY_LLAMA_SERVER=/path/to/llama-server cargo test -p mehoy-backend-llama \
    --features real-backend-tests --test cancellation_characterisation -- --nocapture
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
