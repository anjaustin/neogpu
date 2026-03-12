# REFLECT: Channelized Fabric

## Core Insight
Channelization is not about adding queues; it’s about making guarantees explicit.

Right now the system has a unified fabric, but it relies on implicit conventions ("don’t let telemetry spam", "RT shouldn’t be blocked", "blocking send means pristine"). A channel model turns these into enforceable contracts: drain order, budgets, capture inclusion, and backpressure behavior.

## What Changed In My Understanding

- The hardest part is not implementing channels, it’s specifying deterministic interleaving and budgets.
- A channel field is less important than a channel *spec* (what’s allowed to block, drop, or coalesce).
- The first version should be minimal: two channels solve most practical pain (RT + RENDER).

## Resolved Tensions

### Simplicity vs Control
Resolution: ship a 2-channel scheduler first; extend only when a new policy class is required.

### Determinism vs Throughput
Resolution: determinism is maintained by a fixed drain order and fixed budgets. Throughput is recovered via coalescing and batching, not via nondeterministic scheduling.

## Challenged Assumptions

1) "We need a global total order": we don’t. We need deterministic scheduling and per-channel order, plus clear replay semantics.

2) "Blocking everywhere gives pristine": blocking in telemetry is dangerous. Pristine should apply to critical channels only.
