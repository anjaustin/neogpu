# NeoGPU Communications Fabric - Product Requirements Document

**Project:** NeoGPU  
**Focus:** Extend the message-passing fabric to support robust engine-scale communication (results, tooling, async, backpressure)  
**Status:** Active (partially implemented)  
**Version:** 0.1  
**Date:** 2026-03-12

## 1. Overview

NeoGPU’s message layer is the engine spine: it routes commands, enables capture/replay, and separates API intent from subsystem execution. As engine features expand (mesh/text/audio/input, async I/O), the fabric needs stronger semantics than “fire-and-forget messages with ad-hoc payloads”.

This PRD defines a set of communication features that improve correctness, determinism, observability, and portability without sacrificing the project’s performance goals.

Channelization spec: `docs/CHANNEL_FABRIC.md`.

## 2. Goals

- Provide a standard mechanism for acknowledgements and results (success/failure + optional data).
- Make errors structured and machine-checkable (not only strings).
- Support deterministic capture/replay as an artifact (serialize to/from disk).
- Integrate async completions into the same message fabric.
- Add backpressure/flow control primitives to avoid silent queue drops.
- Enable introspection for debugging and profiling.

P0 status note:

- Channelized transport (RT/RENDER/TELEM), capture I/O, structured errors, async done, and blocking backpressure are implemented.
- Frame markers (`OP_FRAME_BEGIN/END`, `OP_PRESENT`), capture filtering (`record_mask`), per-channel budgets, and apply-time fences (`OP_FENCE` -> `OP_RESULT`) are implemented.

## 3. Non-Goals

- Replacing the node architecture.
- Introducing a heavyweight RPC system.
- Implementing the renderer backend itself (covered elsewhere).

## 4. Requirements

### 4.1 Acknowledgements and Results (Priority: HIGH)

**Problem:** API calls currently don’t have a uniform way to confirm application or report per-op failure.

**Requirements:**

- Add standard response messages emitted by nodes:
  - `OP_ACK`: indicates a message was accepted/applied.
  - `OP_RESULT`: returns structured data (e.g., resource handle/id, statistics).
- Correlate responses with requests:
  - Add a `u32 correlation_id` (preferred) or use `tick` + `from` as a best-effort key.
- Ensure responses are recordable/replayable.

Additional P0 requirement now implemented:

- Provide a fence primitive for tooling:
  - `OP_FENCE` to `NODE_SYSTEM` emits an `OP_RESULT` (flags=`OP_FENCE`) with a small payload that includes the apply-time tick.

### 4.2 Structured Error Model (Priority: HIGH)

**Problem:** `OP_ERROR`/`OP_TRACE` are string payloads; tests and tooling can’t reliably act on them.

**Requirements:**

- Define an error struct payload:
  - `code` (enum), `op`, `to`, `detail` (optional small string)
- Add `OP_ERROR_EX` (structured) while keeping string ops for human logs.
- Update validation and replay to optionally emit structured errors instead of printing directly.

### 4.3 Capture Artifact I/O (Priority: HIGH)

**Problem:** In-memory capture exists (`HSCapture`), but can’t be persisted for regression tests and replay tooling.

**Requirements:**

- Serialize/deserialize captures:
  - `hs_capture_write(FILE*)`, `hs_capture_read(FILE*)`
- Include versioning:
  - ABI version/hash, payload block size, endianness marker.
- Provide a CLI/demo mode to:
  - record one frame to a file
  - replay from file

### 4.4 Async Completion as Messages (Priority: MEDIUM)

**Problem:** Background work exists (`hs_async`) but results are not consistently modeled as messages.

**Requirements:**

- Introduce an async completion opcode family:
  - `OP_ASYNC_DONE` with a small structured payload (task id, status, optional payload ref)
- Ensure async completion messages follow the same validation and capture rules.

### 4.5 Backpressure and Flow Control (Priority: MEDIUM)

**Problem:** When queues fill, producers have limited visibility; failures can be silent or late.

**Requirements:**

- Provide explicit queue-full signals:
  - `hs_send()` returns false already; add optional `OP_QUEUE_FULL` to `NODE_SYSTEM` with context.
- Add counters and watermarks:
  - per-node inbox/outbox high-water marks
  - drop counters by opcode/node

### 4.6 Priority Lanes (Priority: LOW)

**Problem:** Real-time domains (audio/input) should not be blocked by bulk work (uploads).

**Requirements:**

- Add either:
  - multiple queues (priority tiers), or
  - a priority field in `Message` and multi-pass processing order.

P0 implementation: multiple queues per channel (`CHAN_RT`, `CHAN_RENDER`, `CHAN_TELEM`) with deterministic per-channel budgets.

See also: `docs/lmm/channel_fabric_synthesize.md` (LMM synthesis for channelized fabric).

### 4.7 Introspection and Query Ops (Priority: LOW)

**Problem:** Debugging and profiling need a way to query internal state.

**Requirements:**

- Add query opcodes to `NODE_SYSTEM`:
  - queue counts, node stats, last error, tick, capture status.
- Provide a stable small result payload format.

P0 implementation now includes:

- `OP_QUERY_STATS` and `OP_QUERY_FABRIC` (both emit `OP_RESULT` with fixed 64B payloads).
- `OP_SET_RECORD_MASK`, `OP_SET_CHAN_BUDGET`, `OP_SET_BLOCK_POLICY` control ops for safe runtime tuning.

## 5. Milestones

### Milestone A: Results + Errors
- [ ] Define correlation id strategy
- [ ] Implement `OP_ACK` / `OP_RESULT`
- [ ] Implement structured error payload + `OP_ERROR_EX`
- [ ] Add tests for correlation + structured error decoding

### Milestone B: Capture I/O
- [ ] Implement capture serialization format + versioning
- [ ] Add record-to-file + replay-from-file demo path
- [ ] Add regression test that replays a stored capture

### Milestone C: Async + Flow Control
- [ ] Implement async completion messages (`OP_ASYNC_DONE`)
- [ ] Add queue-full events + counters

### Milestone D: Priority + Queries
- [ ] Add priority handling
- [ ] Add system query opcodes + result payloads

## 6. Success Criteria

- [ ] Every externally observable failure can be represented as a structured error.
- [ ] Calls can request/receive acknowledgements/results deterministically.
- [ ] Captures can be persisted and replayed across runs, with version checks.
- [ ] Async completions arrive through the same message fabric.
- [ ] Backpressure is visible (counters/events), not silent.

## 7. Out of Scope

- Large payload transport beyond the existing payload block system (covered by Core Hardening PRD).
- High-level engine feature PRDs (mesh/text/audio/input) beyond the messaging implications.
