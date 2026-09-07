# Architecture Decision Records

Each record captures one decision, the context that forced it, and the consequences accepted
by taking it. Records are immutable once accepted. A decision that changes is superseded by a
new record rather than edited in place, so the reasoning behind a past choice stays readable.

Status values: Proposed, Accepted, Superseded, Deprecated.

Evidence labels follow the project claim discipline used elsewhere in `docs/`:
`[VERIFIED]`, `[OBSERVED]`, `[USER-PROVIDED]`, `[ASSUMPTION]`, `[INFERENCE]`, `[UNVERIFIED]`.

## Index

| ADR | Title | Status |
| --- | --- | --- |
| [0001](0001-rust-daemon-first-runtime.md) | Rust control plane with a daemon-first public surface | Accepted |
| [0002](0002-out-of-process-inference-workers.md) | Inference engines run as supervised out-of-process workers | Accepted |
| [0003](0003-protocol-first-vertical-slice.md) | Protocol defined first, validated through one minimal vertical slice | Accepted |
| [0004](0004-local-transport-and-surface-split.md) | Local transport is HTTP over a Unix socket or named pipe, with control and inference on separate surfaces | Accepted |

## Open decisions

Recorded here so they are not lost between records.

| Decision | Blocking | Raised in |
| --- | --- | --- |
| Worker supervision contract: readiness, hang detection, draining, restart policy, orphan cleanup | Yes, part of the first slice | [0002](0002-out-of-process-inference-workers.md) |
| Content-hash scope and eager or lazy computation for model artifacts | Yes, part of the first slice | [0003](0003-protocol-first-vertical-slice.md) |
| Socket and pipe default locations, creation permissions, and stale-endpoint handling | Yes, part of the first slice | [0004](0004-local-transport-and-surface-split.md) |
| Whether the inference surface ever gains a network listener, and what authenticates it | No, deferred by decision | [0004](0004-local-transport-and-surface-split.md) |
| Target hardware profile, which no document currently records | Needed before memory-aware scheduling | [research distillation](../research/runtime-landscape-distillation.md) |

## Resolved since being raised

| Decision | Resolved by |
| --- | --- |
| Local client transport and its authentication model, deferred by ADR-0001 | [0004](0004-local-transport-and-surface-split.md) |
