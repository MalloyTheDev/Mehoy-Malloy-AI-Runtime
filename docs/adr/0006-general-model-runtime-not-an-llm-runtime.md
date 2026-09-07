# 6. A general model runtime, not a language-model runtime

- Status: Accepted
- Date: 2026-09-07
- Related: [ADR-0002](0002-out-of-process-inference-workers.md),
  [ADR-0003](0003-protocol-first-vertical-slice.md),
  [ADR-0005](0005-backend-channel-isolation-and-readiness.md)

## Context

The first execution backend runs language models, and the first artifacts to hand
are container files that mostly hold them. That combination exerts a steady pull
toward a design where generating text is what the runtime does and everything else
is an exception bolted on afterwards.

That pull is worth naming, because it is only cheap to resist now. Once a request
type, a capability model, or a public interface assumes generated text, every model
class that is not a chat model becomes a special case, and the runtime becomes a
language-model server with adapters rather than what it is meant to be.

The intended range is much wider than one engine serves: language models, text and
multimodal embeddings, rerankers and classifiers, vision models, speech recognition
and synthesis, generative image, audio and video models, forecasting and anomaly
detection, and models a user trained themselves. Most of those will never run
through the current backend.

## Decision

### 1. The runtime is not specialised to any model class

No design choice may assume that an artifact generates text, that a model is
conversational, or that a given engine can execute an artifact. The current backend
is the first of several, not the shape of the system.

### 2. Work is expressed as a task

The unit of work is a task and its input, not a text prompt. Text generation will be
one task among many rather than the default with exceptions around it.

Task variants are added when something implements them. An enumeration full of tasks
nothing can perform is a promise the runtime does not keep, so unimplemented model
classes are represented by their absence, and by an escape hatch for a task named by
a backend rather than by the runtime.

That escape hatch is not decoration. It is the mechanism by which a backend can
offer work this codebase never anticipated without waiting for a release, which is
the difference between a runtime that supports the model classes its authors
imagined and one that supports the model classes that exist.

### 3. Six concepts stay separate

| Concept | What it is |
| --- | --- |
| Artifact | An immutable container on disk |
| Architecture | What the artifact's tensors are |
| Backend | Something that can execute an artifact |
| Task | Work that can be asked of a model |
| Capability | Whether a task can actually be performed, and on what evidence |
| Instance | A model resident in a backend right now |

Collapsing any pair of these is how a general runtime turns into a specialised one.
"GGUF, therefore a chat model" merges the first two with the fifth; "loaded,
therefore working" merges the last two.

### 4. Effective capability is an intersection

What can actually be done is the intersection of what the artifact supports, what
the chosen backend implements, how the backend was configured, and what has been
demonstrated. A vision-language artifact served by a text-only backend is a
text-only instance, and the runtime must be able to say so.

### 5. Evidence is ranked, and only execution verifies

Four rungs, in increasing strength: nothing known; the artifact's metadata suggests
it; a backend reports it can serve it; the runtime performed it successfully.

Only the last is a demonstration. A verification also records which backend build
produced it, because an artifact supporting a task and a particular build having
performed it are different claims, and a different backend inherits neither.

### 6. The runtime defines contracts rather than implementing models

The runtime does not, and will not, natively understand every neural network
architecture, accelerator, or container format. It defines the contracts that
backends plug into, in the way an operating system defines a driver interface rather
than implementing every device.

Someone with a model the runtime has never seen should eventually be able to supply
an adapter rather than wait for this codebase to learn their architecture.

## Consequences

### Gained

- A model class that arrives later plugs into the same runtime rather than forcing a
  redesign.
- A user can say "I have a model" rather than "I have a language model", and the
  runtime's questions are what the artifact is, what tasks it can perform, and which
  installed backend can execute it.
- The difference between what a model claims and what it has been shown to do stays
  legible.

### Accepted costs

- More indirection than a runtime built around one model class would need, and more
  ceremony for the language-model case specifically.
- A task enumeration that grows over time, so callers must handle tasks they do not
  recognise rather than assuming a closed set.
- Capability answers are often "not known", which is less convenient than a
  confident guess and considerably more truthful.

### Deliberately not done

This record does not add model classes, task variants, or backends. It constrains
how they arrive. Implementing a class before anything can execute it would be the
speculative infrastructure ADR-0003 rules out.

## How this was tested rather than asserted

The first artifact loaded end to end was deliberately an embedding model rather than
a chat model. It generates nothing, so any assumption that a loaded model produces
text would have surfaced as a failure rather than passing unnoticed. Reaching a
serving state verified no capability, and only performing a real embedding request
promoted one.

## Alternatives considered

| Alternative | Why not |
| --- | --- |
| Build for language models now, generalise later | The assumptions leak into the request type, the capability model, and the public interface, and every one of those is expensive to change once anything depends on it |
| Enumerate every intended model class up front | Most variants would have no implementation, which promises capability the runtime does not have, and the list would still be wrong within a year |
| Let each backend define its own request types | Callers would learn a new vocabulary per backend, which is the coupling the backend boundary exists to prevent |
