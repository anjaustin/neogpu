# NODES: Channelized Fabric (What Matters)

## Node 1: Channel count is a policy surface
Observation: every new channel is a new set of guarantees.
Why it matters: keep channels few (start with RT + RENDER).

## Node 2: Determinism lives in the step-thread scheduler
Observation: with multiple producers, global total order is already not guaranteed.
Why it matters: replay must define a deterministic drain order and budget.

## Node 3: Backpressure is not a boolean
Observation: different message classes need different behavior (block, drop, coalesce).
Why it matters: make policy per-channel, not global.

## Node 4: Telemetry must never be allowed to break the system
Observation: error/trace/ack storms can become load-amplifiers.
Why it matters: telemetry is best-effort, always droppable, always bounded.

## Node 5: RT starvation vs render starvation
Observation: if RT always drains first and never yields, render can starve.
Why it matters: scheduler budgets must exist and be deterministic.

## Node 6: Capture should be channel-scoped
Observation: not all messages belong in a capture.
Why it matters: render-only captures are smaller and more stable; input capture is optional.

## Node 7: Coalescing is the hidden superpower
Observation: many messages are "state setters" where only the last value matters.
Why it matters: coalescing in the step thread can reduce load without changing API.

## Node 8: Channelization should be compatible with current queues
Observation: NeoGPU already has SPSC lanes + MPSC fallback.
Why it matters: replicate lanes per channel rather than inventing a new transport.

## Node 9: ABI size vs clarity
Observation: adding a `channel` field grows the header.
Why it matters: opcode->channel LUT keeps ABI stable, but makes messages less self-describing.

Tension A: Simplicity vs control
Tension B: Determinism vs throughput
