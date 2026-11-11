# NeoGPU Semantic Message Classes

**Status:** Design target  
**Date:** 2026-03-14

## Purpose

Define message classes based on semantic behavior rather than only opcode or channel. The goal is to let the runtime route different kinds of work according to what they actually need:

- overwriteability
- durability
- ordering
- payload cost
- drop tolerance

This document does not change the current ABI by itself. It defines the structural path plan for future implementation work.

## Why This Exists

Current profiling shows a mismatch between message semantics and routing cost.

Examples:

- `OP_SET_SHADER` is an overwriteable state update
- `OP_FENCE` is a durable control message
- `OP_DRAW_TEXT` is a payload-bearing render command
- `OP_TRACE` is best-effort telemetry

Today, these can still pay broadly similar routing/synchronization costs even though they do not need the same guarantees.

## Message Class Model

### Class A: Overwriteable State

These messages update the latest desired state. Intermediate values may be replaceable if a newer value supersedes them before apply.

Properties:

- latest value matters more than full history
- can support coalescing or overwrite semantics
- still ordered relative to later dependent commands at apply boundaries
- usually no ACK requirement unless explicitly requested

Typical examples:

- `OP_SET_SHADER`
- `OP_SET_PARAM`
- `OP_SET_GLOBAL`
- `OP_SET_CAMERA`
- `OP_CULL`
- `OP_BLEND`
- `OP_ALPHA`
- `OP_DEPTH`
- `OP_COLOR_MASK`
- `OP_CLIP`
- `OP_STENCIL`
- `OP_STENCIL_FUNC`
- `OP_DEPTH_COMPARE`
- `OP_SET_TARGET`
- `OP_SHOW_TEXTURE`
- `OP_TEXTURE_FILTER`
- `OP_TEXTURE_WRAP`
- `OP_SET_CHANNEL`

### Class B: Durable Control

These messages must not be silently overwritten or dropped because they represent explicit control flow, synchronization, or query intent.

Properties:

- durable
- correlation-sensitive when `cid != 0`
- often routed through `NODE_SYSTEM`
- must preserve request/response semantics

Typical examples:

- `OP_QUERY_STATS`
- `OP_QUERY_FABRIC`
- `OP_SET_RECORD_MASK`
- `OP_SET_CHAN_BUDGET`
- `OP_SET_BLOCK_POLICY`
- `OP_FENCE`
- `OP_STOP`
- `OP_ACK`
- `OP_RESULT`
- `OP_ASYNC_DONE`

### Class C: Payload-Bearing Commands

These messages carry data that should be applied as actual command history, not as overwriteable state.

Properties:

- payload copy/allocation cost matters
- history usually matters
- must preserve ordering within the relevant execution stream
- may need dedicated payload handling or batching in future work

Typical examples:

- `OP_LOAD_BUFFER`
- `OP_LOAD_TEXTURE`
- `OP_DRAW_INSTANCE`
- `OP_DRAW_TEXT`
- `OP_CLEAR`
- `OP_CLEAR_DS`
- ML load/forward/generate requests

### Class D: Observable Stream Commands

These messages are commands whose full stream matters to visible output and should not be collapsed unless a stronger contract is defined.

Properties:

- order matters strongly
- history matters strongly
- canonical part of the graphics proof surface

Typical examples:

- `OP_DRAW`
- `OP_FRAME_BEGIN`
- `OP_FRAME_END`
- `OP_PRESENT`

### Class E: Best-Effort Telemetry

These messages are observability traffic. They must never block critical work.

Properties:

- droppable
- lossy by design
- should remain clearly separated from durable control

Typical examples:

- `OP_ERROR`
- `OP_TRACE`
- `OP_ERROR_EX`
- `OP_QUEUE_FULL`

## Current Mapping To Channels

Channels remain useful as policy lanes:

- `CHAN_RT` -> primarily durable control
- `CHAN_RENDER` -> primarily overwriteable state + render command stream
- `CHAN_TELEM` -> best-effort telemetry

Important distinction:

- channel is transport policy
- class is semantic behavior

They are related, but not identical.

## Structural Path Plan

### Phase 1: Documentation And Classification

- keep the current ABI intact
- classify existing opcodes into semantic classes
- make future performance work target classes explicitly

### Phase 2: One Structural Fast Path For Class A

Prototype a structural path for overwriteable state updates that is separate from the general inbox queue path.

Candidate shape:

- per-node state mailbox or state shadow buffer
- latest-wins overwrite semantics for selected ops
- explicit apply boundary in `hs_step()` before durable commands are drained

Important constraints:

- no effect on messages carrying `HS_MSGF_ACK`
- no effect on durable control
- no effect on canonical render command stream ordering

### Phase 3: Dedicated Handling For Payload Cost

For payload-bearing commands:

- preserve durable command ordering
- explore dedicated payload staging or batching only after Class A semantics are explicit

### Phase 4: Keep Telemetry Cheap And Honest

- preserve droppable semantics for telemetry
- avoid contaminating durable/system paths with telemetry pressure

## Prototype Recommendation

The first prototype target should be:

- unacked overwriteable render state updates in Class A

Reason:

- this is where semantic mismatch is clearest
- it avoids touching durable tooling/system contracts first
- it can be validated against visible graphics demos and existing tooling queries

## Non-Goals

This plan does not currently propose:

- dropping or coalescing draw commands
- changing `OP_RESULT` / `OP_ACK` semantics
- making ML operationally first-class yet
- adding more heuristic fast paths inside the current universal route

## Acceptance Criteria For Future Implementation

A future implementation should only be accepted if:

- semantics are explicit in docs before code lands
- visible render behavior remains correct
- system query/control paths remain durable
- telemetry remains best-effort and non-blocking
- benchmark/profile data shows improvement without hidden correctness loss
