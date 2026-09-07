# 7. Chat is an application abstraction, not the inference primitive

- Status: Accepted
- Date: 2026-09-07
- Related: [ADR-0006](0006-general-model-runtime-not-an-llm-runtime.md),
  [ADR-0003](0003-protocol-first-vertical-slice.md)

## Context

ADR-0006 establishes that this is a general model runtime rather than a
language-model one. That record guards against assuming every artifact generates
text. It does not guard against the narrower and likelier mistake waiting inside
text generation itself.

Five things are routinely treated as one:

| | |
| --- | --- |
| A text-generative model | Produces a continuation of some input |
| An instruction-tuned model | Has been trained to follow instructions |
| A chat template | Describes how to format a conversation for a particular model |
| A conversation | A sequence of turns attributed to participants |
| An agent | Something that acts using a model |

A base completion model supports the first and none of the rest. Given
`Once upon a time` it produces a continuation; it has no notion of a system role, a
user role, or an assistant role, and no template describing them.

If the runtime's generation request is a list of chat messages, every model that is
not conversational becomes a special case, and the runtime has quietly become a chat
server. The pull toward this is strong because the models most people reach for
first are exactly the conversational ones.

## Decision

### 1. Continuation is the primitive; conversation is a shape of input

The generation request takes an input that can be a raw prompt or a conversation.
It does not take a list of chat messages as its only form.

```
                 generate text
                       |
          +------------+------------+
          |                         |
      raw prompt              conversation
          |                         |
          |                  prompt renderer
          |                         |
          +------------+------------+
                       |
                 backend input
```

A base completion model, an instruction-tuned model, a chat model, and a code
completion model all reach the backend through the same operation. Only the second
branch involves a template.

### 2. Template discovery and template execution are separate

The runtime already reads the chat template an artifact carries. Knowing a template
is present is not the same as being able to render it correctly, and a malformed or
unsupported template ruins output quality while the model itself loads perfectly.

These are therefore distinct facts, and the second follows the same evidence rule as
every other capability claim: what the artifact carries can only indicate, and only
producing correct output verifies.

### 3. Model-specific control tokens never reach the generic runtime

The runtime deals in conversations. Whatever a particular model uses to mark a turn
boundary is the concern of whatever renders the template, which is below the backend
boundary. A control token appearing in a runtime type is the signal that this
decision has been violated.

### 4. Chat lives above the runtime

Conversation management, message history, tool loops, and agent behaviour are
application concerns built on this runtime, not services it provides. A future
compatibility surface for someone else's chat API is an adapter that translates from
these types, not a shape the runtime bends itself into.

## Consequences

### Gained

- A base completion model is a first-class citizen rather than an awkward case.
- Compatibility adapters translate outward from the runtime's own types, rather than
  the runtime being defined by whichever external API was implemented first.
- The template, which is a common source of silently degraded output, is a tracked
  fact rather than an invisible assumption.

### Accepted costs

- The common case, a chat model answering a conversation, carries slightly more
  ceremony than a design that assumed only that.
- Two input shapes to handle rather than one.

### What this does not decide

The generation request's parameters, the rendering strategy, and where the renderer
lives are all open. This record constrains the shape of the operation, not its
details.

## Alternatives considered

| Alternative | Why not |
| --- | --- |
| Take a list of chat messages as the only input | Every non-conversational model becomes a special case, and the runtime becomes a chat server with exceptions. This is the specific outcome ADR-0006 exists to prevent, arriving one level down |
| Take only a raw prompt and make callers render templates | Pushes model-specific control tokens into every application, which is precisely the coupling the backend boundary exists to prevent |
| Decide later, when a generative model is actually served | The shape of the first request type is what everything downstream is written against, and changing it afterwards means changing every caller |
