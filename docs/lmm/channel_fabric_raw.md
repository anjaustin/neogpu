# RAW: Channelized Fabric Concept

The “channelized fabric” idea feels like the natural next step now that NeoGPU has a real communications spine (submit queues, step-thread routing, capture/replay, structured errors). Right now the fabric is *functionally* unified but *policy-less*: everything competes in the same transport, and we rely on global-ish behaviors ("block_on_full") or best-effort drops.

Channelization would turn this into a first-class scheduler: explicit lanes with explicit semantics. That’s attractive because it solves the "RT vs bulk" tension cleanly (audio/input shouldn’t wait behind render spam; telemetry shouldn’t steal capacity from either).

At the same time, it’s easy to over-engineer: too many channels, too many policies, subtle ordering invariants, and replay determinism gets harder if you don’t keep the drain schedule deterministic.

Gut feeling: keep it brutally small and policy-driven (2 channels first), and make the step-thread scheduler the single source of determinism.

What I’m excited about:
- You get a crisp story for "pristine" behavior: which channels block, which drop, and what is allowed to be lossy.
- Capture becomes meaningful: render-only capture by default, optional input capture for deterministic gameplay.
- Backend evolution becomes simpler: render list becomes one of the channel outputs.

What I’m worried about:
- Losing the mental model of a single ordered stream (even if we never truly had it across producers).
- Starvation bugs if RT drains aggressively.
- People will try to make telemetry reliable and accidentally create deadlocks.

Open questions:
- Do we encode channel as a `Message` field, or as an opcode->channel table? (field is explicit; table keeps ABI smaller.)
- Do we allow channel policies to be changed at runtime, or do we hardcode for determinism?
- How do we define replay interleaving: strict channel order per step, or recorded interleaving in capture?

Naive approach to avoid:
- Creating 8+ channels and a complex scheduler before we have a proven RT+Render split.
