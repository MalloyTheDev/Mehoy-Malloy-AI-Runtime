# 10. Resource accounting is declared, not measured

- Status: Accepted
- Date: 2026-10-20
- Related: [ADR-0005](0005-backend-channel-isolation-and-readiness.md),
  [ADR-0006](0006-general-model-runtime-not-an-llm-runtime.md)

## Context

The runtime owns its live models, and until now it held at most one because a
constant said so. The constant was honest about why: deciding between two models
competing for one accelerator is resource accounting, and there was none.

Making the limit configurable without accounting would be worse than the
constant. An operator could raise it to four and the runtime would agree, having
no idea whether four of anything fit. The limit would read as a capability
statement while being a formality.

So accounting has to come with the configuration. The difficulty is that the
runtime can only honestly account for what it knows, and it knows very little:

```
 what the runtime knows                 what it would need to know
 ----------------------                 --------------------------
 the container is 5368709120 bytes      how much memory this host has
 it was registered at that size         how much of it is free
 the file has not changed since         how much the backend will offload
                                        what the key/value cache will cost
                                        what the compute buffers will cost
```

The left column comes from a file the runtime inspected. The right column
requires discovering the hardware, which is separate work that does not exist.

There is a real precedent for this shape of problem. Capability claims faced it
first: metadata can suggest that a model embeds, but only performing an embedding
demonstrates it, and the two are recorded differently so neither can be mistaken
for the other. Resources are the same problem with different nouns.

## Decision

### 1. Every figure records how it was arrived at

A model's accounted footprint carries a basis alongside its number.
`FootprintBasis::ContainerSize` says the figure is the size of the container on
disk as recorded at registration, which is a floor on what must be resident
somewhere and not a total: it excludes the key/value cache, the backend's compute
buffers, and the backend process itself, and it says nothing about whether the
bytes land in host memory or on an accelerator.

A bare byte count would be read as a measurement. Measured figures, when they
exist, will join this as further variants rather than replacing the number in
place, so a reader will always be able to tell which kind they have.

### 2. Limits are stated by the operator, not derived from the machine

`ResidentLimits` carries how many models may be resident and, optionally, how
many bytes of weights. Both are stated when the runtime is built and never change
afterwards, because a limit that moved while loads were being judged against it
would make a refusal unexplainable.

The byte budget defaults to absent. Picking a number for a machine nothing has
measured would be inventing a limit, and a limit nobody chose is worse than one
that is openly missing. The instance count defaults to one, for the same reason
it was one before.

### 3. The accounting enforces a policy, and claims nothing about hardware

A load that fits is a load that fits what somebody said this process may hold. It
is not evidence that the hardware can take it, and an out-of-memory failure from
a backend is not an accounting bug. This is the boundary the basis in decision 1
exists to keep visible.

### 4. Room is claimed before the expensive work, and held across it

A load reserves its footprint under the ledger lock before it starts, and holds
the reservation until the model is either resident or gone. A load in flight is
counted exactly as a resident model is.

```
 check, then load                        claim, then load
 ----------------                        ----------------
 A: room for one? yes                    A: claim the only slot
 B: room for one? yes                    B: no slot; refused now
 A: loads for 40 s                       A: loads for 40 s
 B: loads for 40 s                       B: already told, 40 s ago
 both resident, limit was one            one resident, limit held
```

Checking and then loading is not a weaker version of this; it is a limit that
holds only for callers who happen to arrive one at a time. The reservation is
released by being dropped, so every way a load can fail releases it, including
one added later by somebody who does not know this paragraph exists.

### 5. What a model is accounted at does not change while it is resident

The figure a load was admitted under is the figure it keeps. Nothing re-measures
a running model, and a number that drifted would make the accounting impossible
to reason about: a model could become unaffordable after the fact, with no action
having caused it.

### 6. A teardown that did not finish stays accounted for

An unload that completes releases the model's reservation, because the model and
its footprint are one ledger entry and there is no path that forgets one without
the other. An unload that fails keeps both. A worker that could not be stopped
may still be holding its memory, and handing that budget to another model is the
more dangerous of the two wrong answers.

## Consequences

### Gained

- The resident limit is a configuration rather than a recompile, without becoming
  a claim the runtime cannot support.
- The limit holds under concurrency. Two loads arriving together are no longer
  both admitted.
- A refusal says which limit was reached and what was already committed, so an
  operator can tell "unload something" from "this will never fit here".
- The accounting policy is a pure function of declared figures, so it is tested
  deterministically on every commit rather than only where a model happens to
  exist.

### Accepted costs

- The figures are weak. A container size tells an operator less than they want,
  and the basis field exists to keep them from forgetting that.
- The runtime reads the registry twice per load: once to size the reservation,
  once inside the loader to resolve the artifact and confirm it has not drifted.
  Collapsing them would mean either sizing from a record read earlier than the
  drift check, or claiming room after the load, and both are worse.
- A pending reservation is observable state that did not exist before.

### Not decided

Whether a model fits the hardware. That needs the right column of the table in
the context above, which means discovering memory, accelerators and their
current occupancy. Until it exists, nothing here should be read as answering it.

Also not decided: what happens when several resident models compete. This record
makes more than one expressible; it does not make it good, and eviction,
per-instance concurrency and scheduling are all still absent.

## Alternatives considered

| Alternative | Why not |
| --- | --- |
| Leave the limit a constant until hardware discovery exists | The constant was already load-bearing in the wrong way: it made single residency look like a decision about this machine rather than a decision about nothing. Separating the policy from the measurement now means measurement can arrive as a new basis rather than as a redesign |
| Estimate the real footprint from the container, context length and quantisation | An estimate dressed as a figure. It would be wrong by a margin nobody could state, and being approximately right is what makes it dangerous: an operator would plan with it |
| Account in a unit the operator cares about, such as a fraction of device memory | Requires knowing device memory, which is the thing that is missing. A fraction of an unknown is not a limit |
| Claim room after the load instead of before | Spends the expensive part before deciding whether to keep it, and leaves the race open for the whole duration of the load |
| Hold the ledger lock across the load | Closes the race and makes every other caller wait on one load, including callers asking about a different model entirely |
| Release the reservation explicitly on each failure path | Correct until somebody adds a path. The failure mode is a limit nothing gives back, which refuses every later load for the life of the process |
