# NeoGPU Channelized Fabric Spec (P0)

**Project:** NeoGPU  
**Scope:** Transport/fabric layer semantics for multi-domain workloads  
**Status:** P0 (active)  
**Date:** 2026-03-12

## 1) Goal

Channelize the message transport into a single fabric layer with explicit lanes and policies so the system is:

- **Pristine for critical work**: no dropped render/RT messages under sustained load (blocking semantics).
- **Latency-aware**: RT messages are not starved by render spam.
- **Deterministic**: step-thread scheduling defines interleaving; capture/replay remains reproducible.
- **Bounded**: fixed-size queues; telemetry is best-effort and never blocks.

## 2) Definitions

### 2.1 Channels

Channels are policy classes, not features.

- `CHAN_RT`: real-time control (input, audio control, timing-critical signals)
- `CHAN_RENDER`: render state + draw commands
- `CHAN_TELEM`: diagnostics/telemetry (errors, traces, queue-full notices)

### 2.2 Transport Topology

For each channel:

- Per-producer SPSC lanes (fast path): `producers[channel][producer_id]`
- Fallback MPSC submit queue (overflow/unregistered producers): `submit[channel]`

Only the **step thread** applies messages:

- Drain channel queues in a deterministic schedule.
- Route into node inboxes.
- Assign `tick` at apply time.
- Perform validation/logging/capture/render-recording.

## 3) Message Semantics

### 3.1 Message header

Add a `channel` field to `Message`.

- `channel` is the channel the producer is submitting to.
- If `channel` is 0 and the opcode has a default, it may be auto-assigned by the sender helper.

### 3.2 Ordering

- FIFO ordering is guaranteed per producer lane within a channel.
- A total order across producers is *not* guaranteed.
- Inter-channel ordering is defined by the step-thread schedule (below).

### 3.3 Backpressure policy

- `CHAN_RT`: blocking (no drops) for critical ops.
- `CHAN_RENDER`: blocking (no drops) for render ops.
- `CHAN_TELEM`: non-blocking, droppable.

Blocking behavior is implemented as:

- if enqueue fails and the channel is blocking, the producer waits until `hs_step()` drains and wakes.
- the step thread must never block.

## 4) Step Thread Schedule

Drain order is fixed:

1) `CHAN_RT`
2) `CHAN_RENDER`
3) `CHAN_TELEM`

Each channel drain is budgeted to prevent starvation. Default budgets (tunable constants):

- `RT_BUDGET = 4096` messages per step
- `RENDER_BUDGET = 16384` messages per step
- `TELEM_BUDGET = 1024` messages per step

Budgets are deterministic and do not depend on wall time.

Implementation: budgets are stored in `HSSystem.chan_budget[]` and can be updated at runtime via `hs_set_channel_budget()`.

## 5) Capture/Replay

- Default capture includes `CHAN_RENDER` only.
- Optional capture profiles may include a subset of `CHAN_RT`.
- `CHAN_TELEM` is excluded by default.

Implementation: capture filtering is controlled by `HSSystem.record_mask` and `hs_set_record_mask()`.

Replay uses the same channel schedule.

## 6) Default Channel Mapping (P0)

Default channel assignment by opcode:

- Render path: `OP_CLEAR`, `OP_CLEAR_DS`, `OP_DRAW`, `OP_DRAW_INSTANCE`, `OP_DRAW_TEXT`, `OP_SHOW_TEXTURE`, and render state ops -> `CHAN_RENDER`
- Frame markers: `OP_FRAME_BEGIN`, `OP_FRAME_END`, `OP_PRESENT` -> `CHAN_RENDER`
- RT/control path: `OP_ACK`, `OP_RESULT`, `OP_ASYNC_DONE`, and other control/interactive ops -> `CHAN_RT`
- Fences: `OP_FENCE` -> `CHAN_RT` (emits `OP_RESULT`)
- Telemetry: `OP_ERROR_EX`, `OP_ERROR`, `OP_TRACE`, `OP_QUEUE_FULL` -> `CHAN_TELEM`

## 6.1) System Inbox Pristinity (P0)

To keep RT results/ACKs usable under telemetry spam, the system may drop non-RT messages targeting `NODE_SYSTEM` when the system inbox is near full, preserving a small reserve for RT.

Notes:

- `OP_ACK`/`OP_RESULT` are treated as RT to keep request/response usable under render load.
- Telemetry must never block the system.

## 7) Success Criteria

- Under sustained multi-producer load, `CHAN_RT` + `CHAN_RENDER` operate with **0 send failures** when configured as blocking.
- RT traffic continues to drain under render spam (bounded latency due to budgets).
- Capture/replay remains deterministic and stable across runs.
- Telemetry drops are measurable and bounded.

## 8) Red-Team Checklist

- > `HS_MAX_PRODUCERS` threads send concurrently (must fall back safely).
- Payload-bearing producers under contention.
- Telemetry spam does not deadlock or block RT/render.
- Blocking send shutdown is clean (wake senders on stop).
