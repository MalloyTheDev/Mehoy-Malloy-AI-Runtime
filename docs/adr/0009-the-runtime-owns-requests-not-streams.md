# 9. The runtime owns requests, and a stream is only how one is observed

- Status: Accepted
- Date: 2026-09-07
- Related: [ADR-0007](0007-chat-is-not-the-inference-primitive.md),
  [ADR-0008](0008-minimal-portable-generation-parameters.md)
- Amends: the meaning of `GenerationEvent::Started` established when streaming was
  implemented

## Context

Streaming was built on the assumption that a request and its stream begin at the
same moment. Measuring the backend showed that they do not.

For a 40011-token prompt on llama.cpp build 9010, the call that awaits response
headers took 10.55 seconds to return, because that engine does not begin its
response until it has finished reading the prompt. The measurements are recorded
in [the cancellation characterisation](../research/llama-cpp-cancellation-b9010.md).

```
 request sent
      |
      |<------------ 10.4 s of real inference ------------>|
      |                                                    |
      v                                                    v
 backend begins work                              response headers arrive
```

Two consequences follow, and neither is specific to this engine. Any backend that
processes input before answering will behave the same way.

First, a runtime whose request lifetime starts when the response starts is blind
for the entire input-processing phase. It cannot report the request as running, it
cannot time it, and it cannot stop it, because as far as it is concerned the
request does not exist yet.

Second, the interval between the first public event and the first output is not
time to first output. It excludes the most expensive part. A latency measurement
built on it would look excellent precisely when the system was slowest.

## Decision

### 1. A request exists from the moment the runtime accepts it

`GenerationEvent::Started` means the runtime accepted the request for execution
and assigned it an identifier. It no longer means the backend stream is
established.

That earlier definition was chosen deliberately, to keep acceptance separate from
first output. The intent was right and the boundary was in the wrong place: it sat
after the backend had already been working for as long as ten seconds.

```
 Started
    |
    |<---------- time to first output ---------->|
    |                                            |
    |   dispatch -> input processing -> decode   |
    |                                            v
    +--------------------------------------- TextDelta
```

Establishing the backend transport becomes internal detail. It is a reasonable
thing to trace, and it is not an event in the inference contract.

### 2. Starting a generation does not wait for the backend

The call that begins a streaming generation returns as soon as the runtime has
taken ownership of the request. It allocates the identifier, creates the event
channel, emits `Started`, and hands back a handle. Connecting, sending, waiting
out input processing, and reading the response all happen behind that handle.

A call that blocks until the backend answers cannot offer cancellation during the
period when cancellation is most valuable, because it has not returned anything to
cancel.

### 3. Cancellation is addressed to the request, not to the stream

Cancellation takes a request identifier. It does not depend on who holds the
stream, or on whether anyone still does.

Dropping a stream is not cancellation. A consumer may stop reading for reasons
that have nothing to do with wanting the work stopped: a view is hidden, a task
hands the stream to another, a client reconnects. Letting a value's destructor
define a public semantic would make the meaning of cancellation an accident of how
callers happen to structure their code.

Abandoned work must still be bounded, but that is a separate obligation met by a
request budget, not by redefining what a drop means.

A dropped stream does still end the request, and that is stated rather than left
to fall out of the implementation. With a backend stopped by closing its
transport, keeping the work alive would mean holding a connection open purely to
discard the answer, occupying a slot nobody is waiting on. So the request ends,
and it ends with its own cause: nothing was reading it. That is not the same
statement as somebody having cancelled it, and a consumer can tell the two apart.

The alternative worth naming is detaching, where the work continues without an
observer and can later be reattached to. That needs a backend able to stop one
request without closing a shared transport, and somewhere for the output to go in
the meantime. Neither exists, so neither is pretended.

What is deliberately avoided either way is a channel send failing and thereby
deciding what cancellation means.

### 4. Cancellation has a requested phase and a stopped phase

Cancelling is not instantaneous, and pretending otherwise would misreport it. On
the measured backend, a request cancelled 235 ms into a 3.1 second input-processing
phase held its slot until that phase ended, then stopped instead of generating.

```
 Running --> cancel requested --> Cancelling --> transport released --> Cancelled
```

`Cancelled` is emitted when this runtime has finished executing the request and
released its transport, not when the cancellation was accepted.

It is not a claim that the backend has stopped, because the runtime cannot see
that. On the measured engine the two are far apart: cancelling during input
processing produced `Cancelled` in under a millisecond while the engine kept
working for a further ten seconds. What the runtime knows directly is that
cancellation was requested and that the transport was torn down. That the
teardown does eventually stop the work was established by measuring the engine,
not by observing it at runtime, and it is a property of that engine rather than a
guarantee of this interface.

Three things are therefore distinct, and only the first two are runtime facts:

| | Known by |
| --- | --- |
| Cancellation requested | The runtime, immediately |
| Transport released | The runtime, when execution ends |
| Backend work stopped | Measurement against a particular engine build |

### 5. Exactly one terminal outcome

A request ends as completed, cancelled, or failed. Never two of them, and never
none. A cancellation racing a natural completion resolves to one of them, decided
once, and the loser is discarded.

### 6. What a backend can stop is a property of the backend

Backends differ in whether they can stop one request, only a whole connection, or
nothing at all short of being killed. That is described per backend rather than
assumed, so a backend that cannot honour cancellation cannot silently appear to.

Only the strategy that exists is represented. The measured backend offers no
cancellation endpoint, so closing the request's transport is the mechanism, and
the runtime knows that is what it is doing.

### 7. Cancelling one request does not destroy the model

Terminating the worker would cancel a request, and would also evict a resident
model that other requests are using or will use. It stays an escalation for a
backend that will not stop, not the ordinary path.

## Consequences

### Gained

- Request lifetime covers the whole of execution, including input processing, so
  a request can be observed and stopped throughout.
- The interval from `Started` to first output is genuinely time to first output.
- Cancellation, idle timeouts, and future deadlines share one mechanism while
  keeping distinct causes.
- A backend that cannot really stop work cannot be misreported as one that can.

### Accepted costs

- A public semantic changed after being recorded. This record exists because the
  original was contradicted by measurement.
- Starting a generation gains a layer: a handle whose work happens elsewhere, and
  errors that once surfaced from the initial call now arrive through the stream.
- The runtime keeps state for every in-flight request, which is bookkeeping it
  previously avoided.

### Not decided

Concurrency limits and queueing. The runtime now identifies requests, which is a
prerequisite for both, but neither is implemented and nothing here assumes them.

## Alternatives considered

| Alternative | Why not |
| --- | --- |
| Keep `Started` at transport establishment and add an earlier event | Two events for one thing, with the more useful one absent from the contract. The problem is not a missing event, it is that the recorded one marks the wrong moment |
| Treat dropping the stream as cancellation | Rust destructors would define a public inference semantic, so an ordinary refactor could silently cancel work. It also cannot express cancelling a request whose stream nobody holds |
| Report `Cancelled` when cancellation is accepted | Claims an observation the runtime has not made. The measured backend keeps working for seconds afterwards, and a terminal event that arrives before the work stops is exactly the false report this design is trying to avoid |
| Terminate the worker to cancel | Reliable and far too broad. It evicts a resident model to stop one request, which stops being defensible the moment a model serves more than one |
