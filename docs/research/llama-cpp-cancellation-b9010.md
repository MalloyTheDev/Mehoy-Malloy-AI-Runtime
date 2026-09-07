# Cancellation behaviour of llama.cpp build 9010

Measured against `b9010-d05fe1d7d`, Vulkan, `gemma-4-E4B-it-Q4_K_M.gguf`, Windows,
131072 context per slot, several slots. Reproduced by
`crates/mehoy-backend-llama/tests/cancellation_characterisation.rs`, which is
ignored by default and run explicitly.

Claim labels follow the project convention. Everything below is `[VERIFIED]`
unless marked otherwise, meaning it was observed on this machine against this
build. None of it is a claim about llama.cpp in general, and none of it should be
relied on for a different build without re-running the harness.

## Why this was measured before implementing cancellation

The obvious assumption is that abandoning the client stream stops the work. If
that assumption is wrong, a runtime that reports a request as cancelled when its
stream closes reports something false: the accelerator is still busy and the slot
is still held. Such a mistake passes every test that only observes the client, so
the observations here are taken from the engine's own view of its slots.

## What exists

| Route | Result |
| --- | --- |
| `GET /slots` | 200, array of `{id, n_ctx, speculative, is_processing}` |
| `GET /props` | 200 |
| `POST /v1/stream` | 404 |
| `DELETE /v1/stream?conv_id=...` | 404 |
| `DELETE /slots/0` | 404 |

`[VERIFIED]` This build exposes **no explicit per-request cancellation route**.
The resumable-stream cancellation interface present in later upstream work is not
here. The only cancellation signal available is closing the connection.

`--slots` is enabled by default in this build, so `/slots` is available without
extra flags.

## Cancelling while tokens are being generated

`[VERIFIED]` Closing the connection during token generation stops the work
promptly.

| Action | Slot idle after |
| --- | --- |
| Dropping the runtime's stream | 234 ms |
| Cutting the raw TCP socket | 220 ms |

The control matters: the same request run to completion takes 7.38 s and produces
740 deltas, with the first at 48 ms. The interruption happened just after the
first delta, leaving roughly 7.3 s of work outstanding, and the slot went idle in
about a fifth of a second. A separate control confirms a naturally completed
request also frees its slot, so an idle slot is not simply what this endpoint
always reports.

## Cancelling while the prompt is still being read

`[VERIFIED]` Cancellation during prompt processing is **not observed until prompt
processing finishes**.

| Prompt | Prefill | Abandoned at | Slot freed after | Whole request |
| --- | --- | --- | --- | --- |
| 40011 tokens | 10.44 s | 375 ms | 10.14 s | n/a, ends at prefill |
| ~15000 tokens | 3.11 s | 235 ms | 3.13 s | 9.99 s |

In both cases the slot was held for approximately the full remaining prefill. The
second row is the informative one: the cancelled request stopped at 3.13 s rather
than running the full 9.99 s, so the cancellation was pending and was applied at
the end of prefill. Cancelling during prefill therefore saves the generation
phase but not the prefill.

`[INFERENCE]` The engine appears to check for a departed client between prompt
processing and token generation, and not within the prompt-processing loop.

## The response does not begin until prefill ends

`[VERIFIED]` For a streaming request with a 40011-token prompt, the call that
awaits response headers took **10.55 s** to return, and by then the slot was
already idle.

This has a direct consequence for the runtime's own design. `GenerationEvent::Started`
is defined as "the runtime accepted the request and established the backend
stream". For this engine, establishing the stream cannot happen until prefill is
over, so:

- There is no stream in existence during prefill, and therefore nothing to drop.
  The only cancellation available in that window is abandoning the in-flight
  request, which means **the runtime must own a request before its stream
  exists**.
- The interval between `Started` and the first delta is not time to first token
  for a large prompt. Prefill happens before `Started`, not after it. Any latency
  measurement built on that interval would omit the most expensive part.

## Traps encountered while measuring

Recorded because each one produced a plausible but wrong reading.

**Prompt caching silently removes prefill.** Running the same prompt twice makes
the second request skip prompt processing, so an experiment that depends on
prefill happening instead measures the token-generation path. This produced a
reading of 224 ms for prefill cancellation, which looked like prompt cancellation
and was not. Every prompt in the harness is now made distinct.

**A slot is not busy the instant a request is sent.** Sampling `/slots` at a fixed
150 ms found every slot idle during a request that was genuinely in flight;
dispatch happened at 235 to 375 ms depending on prompt size. Experiments wait for
the engine to report itself busy rather than sleeping a guessed interval.

**An empty completion is not necessarily a failure.** A degenerate 40011-token
prompt returned `finish_reason: "stop"` with `completion_tokens: 1` and empty
text: the model emitted an end-of-sequence token immediately. Untruncated
(`truncated = 0`), fully consumed. A client that filters empty deltas presents
this as a completion with no content, which is correct and not a defect.

**A stack-pinned future is not dropped when its handle is.** `std::pin::pin!`
keeps the future in a hidden local that lives to the end of the scope, so dropping
the handle does not abandon the request and the experiment measures nothing. The
harness boxes these futures.

## What this means for cancellation in this runtime

`[INFERENCE]` Drawn from the measurements above, not yet implemented.

1. Connection closure is a real cancellation signal on this build, so cancellation
   need not destroy the model instance in the common case. That matters, because
   a resident model will eventually serve more than one request.
2. Cancellation latency is bounded by the phase the request is in. During
   generation it is sub-second. During prefill it is the remaining prefill, which
   was over ten seconds for a 40000-token prompt.
3. A backend cannot be assumed to stop work on request. The runtime should be able
   to express that a backend honours cancellation per request, only by connection,
   only by terminating the worker, or not at all, rather than assuming the first.
4. Because the response is withheld until prefill ends, request identity and
   cancellation must exist before the stream does.

## Measured again after the runtime took ownership of requests

Re-run once cancellation was addressed to a request rather than to a stream, which
is the arrangement ADR-0009 describes. The findings are unchanged, and one gap is
now visible as a number rather than as a prediction.

| | |
| --- | --- |
| Stream reports `Cancelled` after | 289 us |
| Engine work actually stops after | 10.17 s |

`[VERIFIED]` Cancelling during input processing releases the transport at once and
the engine keeps working until it has finished reading the prompt. The terminal
event therefore means the runtime has stopped, not that the engine has.

`[VERIFIED]` The bound is unchanged: cancelling 311 ms into a 3.11 s input phase
freed the slot after 3.18 s, against 10.22 s for the same request undisturbed. The
cost of cancelling during input processing is the remainder of that phase, not the
whole request.
