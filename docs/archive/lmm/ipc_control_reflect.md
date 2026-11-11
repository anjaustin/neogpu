# REFLECT: IPC Control Surface (NeoGPU)

## Core Insight
Treat IPC as a *transport* for the existing `NODE_SYSTEM` contract, not as a second API. Preserve canonical semantics by routing all operations through the message fabric, while adding a bounded "result bus" to make correlation reliable.

## Key Resolutions

### Tension: Direct Reads vs Canonical Semantics
Resolution: queries go through `OP_QUERY_*` and return `OP_RESULT` payloads. This keeps one truth source (the system node + fabric). For purely diagnostic reads that do not exist as ops, add ops rather than bypassing.

### Tension: Concurrency vs Simplicity
Resolution: add a small bounded result mailbox/bus that records recent `OP_RESULT` messages (and payload bytes) with their `cid`. IPC server threads block waiting for matches with timeouts.

### Tension: Tooling QoS vs Engine QoS
Resolution: clamp IPC-submitted ops to `CHAN_RT` (already supported for system ops). Apply strict budgets/limits: fixed maximum pending requests per connection and bounded result bus depth.

## Design Constraints to Keep

- **Local-only P0**: Unix domain socket; no TCP; permissions must be restrictive.
- **Whitelist ops**: do not expose arbitrary message injection.
- **Bounded memory**: fixed-size ring buffers, not dynamic queues.
- **Short lock hold times**: IPC thread should not hold `HSSystem` lock while doing socket I/O.
- **Deterministic outcomes**: failures must be explicit (timeout, queue full, invalid request).

## Assumptions to Challenge

1) "One connection is enough". Not necessarily; multiple tools may connect. But P0 can be single-client with clean refusal semantics.
2) "The system node outbox is sufficient". It routes through `hs_step()`; we need a separate bus that survives non-tool traffic.
3) "Binary is always best". Text is easier to debug, but binary is easier to make deterministic and less error-prone for fixed payloads.

## What Makes This a Step-Change

Without IPC, the system control surface is only a library API. With IPC, NeoGPU becomes inspectable and tunable as a running service. That enables real industry workflows: watchdogs, profilers, capture triggers, and automated regression harnesses.
