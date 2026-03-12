# NeoGPU Message ABI (Current)

This document describes the message formats used by the NeoGPU message layer as implemented today.

The goal is to make the existing implicit ABI explicit so it can be validated, captured, replayed, and extended safely.

## Message Header

`Message` is defined in `include/hs_core.h`:

- `to` / `from`: node ids
- `op`: opcode (`OpCode`)
- `flags`: opcode-specific small field (MSB is reserved for `HS_MSGF_ACK`)
- `cid`: correlation id for request/response-style messages (0 means "none")
- `tick`: assigned when routed/applied (step thread)
- `payload_idx`: either an immediate value (for some ops) OR an index into the payload ring (for payload-bearing ops)
- `payload_len`: payload length in bytes (0 for immediate-only ops)
- `channel`: fabric channel (0 means "use default mapping")

## Payload Storage

- Payload bytes live in `HSSystem.payloads` as fixed-size 64B blocks (`HS_PAYLOAD_SIZE`).
- For payload-bearing ops, `payload_idx` is an index into the payload ring.
- Payloads larger than 64B are currently truncated by send helpers.

Implementation notes:

- Prefer `hs_payload_alloc_and_copy()` for allocating/copying payload bytes.
- Prefer `include/hs_msg.h` pack/unpack helpers for opcode payload layouts.

### Capture/Replay Payloads

The core also supports a self-contained capture format (`HSCapture`) that deep-copies payload blocks and can replay deterministically without relying on the transient payload ring. In a capture, payload-bearing messages can be rewritten so `payload_idx` refers to the capture payload slot.

Capture persistence is documented in `docs/CAPTURE_FORMAT.md`.

## Nodes

Current node ids (see `include/hs_nodes.h`):

- `NODE_CPU` (0): message producer id (not a registered node)
- `NODE_SHADER` (1)
- `NODE_BUFFER` (2)
- `NODE_TEXTURE` (3)
- `NODE_OUTPUT` (4)
- `NODE_SOUND` (5)
- `NODE_SYSTEM` (6) (reserved for system ops)

## Opcode Table

In the table below, "Immediate" means the opcode uses `payload_idx` as its argument and has `payload_len = 0`.

### Shader node ops (`to = NODE_SHADER`)

- `OP_SET_SHADER` (Immediate)
  - Immediate: `payload_idx = shader_id`
- `OP_SET_PARAM` (Payload, 20B)
  - Payload layout: `[u32 param_idx][f32 x][f32 y][f32 z][f32 w]`
- `OP_SET_GLOBAL` (Payload, 64B)
  - Payload layout: `[f32 m[16]]`
  - Header: `flags = global_idx`
- `OP_SET_CAMERA` (Payload, 64B)
  - Payload layout: `[f32 m[16]]`
- `OP_CULL` (Immediate)
  - Immediate: `payload_idx = mode`
- `OP_BLEND` (Payload, 2B)
  - Payload layout: `[u8 src][u8 dst]`
- `OP_ALPHA` (Immediate)
  - Immediate: `payload_idx = 0|1`
- `OP_DEPTH` (Immediate)
  - Immediate: `payload_idx = 0|1`
- `OP_COLOR_MASK` (Immediate)
  - Immediate: `payload_idx = mask`
- `OP_CLIP` (Payload, 8B)
  - Payload layout: `[u16 x][u16 y][u16 w][u16 h]`
- `OP_STENCIL` (Payload, 4B)
  - Payload layout: `[u8 op][u8 fail][u8 pass][u8 front]`
- `OP_STENCIL_FUNC` (Payload, 4B)
  - Payload layout: `[u8 compare][u8 ref][u8 read_mask][u8 write_mask]`
- `OP_DEPTH_COMPARE` (Payload, 2B)
  - Payload layout: `[u8 compare][u8 write]`

### Buffer node ops (`to = NODE_BUFFER`)

- `OP_LOAD_BUFFER` (Immediate)
  - Immediate: `payload_idx = buffer_index`
- `OP_DRAW` (Immediate)
  - Immediate: `payload_idx = buffer_index`
- `OP_DRAW_INSTANCE` (Payload, 4B)
  - Payload layout: `[u8 buffer][u8 instance_buffer][u8 count_lo][u8 count_hi]`
- `OP_DRAW_TEXT` (Payload, 1..64B)
  - Payload layout: C string, null-terminated, truncated to fit 64B.

### Texture node ops (`to = NODE_TEXTURE`)

- `OP_LOAD_TEXTURE` (Immediate)
  - Immediate: `payload_idx = texture_index`
- `OP_SET_TARGET` (Payload, 2B)
  - Payload layout: `[u8 texture][u8 depth_texture]`
- `OP_SHOW_TEXTURE` (Immediate)
  - Immediate: `payload_idx = texture_index`
- `OP_TEXTURE_FILTER` (Immediate, packed)
  - Immediate: `payload_idx = (tex_idx & 0x0F) | ((linear?1:0) << 4)`
- `OP_TEXTURE_WRAP` (Immediate, packed)
  - Immediate: `payload_idx = (tex_idx & 0x0F) | ((repeat?1:0) << 4)`

### Output node ops (`to = NODE_OUTPUT`)

- `OP_CLEAR` (Payload, 16B)
  - Payload layout: `[f32 r][f32 g][f32 b][f32 a]` (a `vec4`)
- `OP_CLEAR_DS` (Payload, 8B)
  - Payload layout: `[f32 depth][u8 stencil][u8 pad0][u8 pad1][u8 pad2]`

### Sound node ops (`to = NODE_SOUND`)

- `OP_SET_CHANNEL` (Payload, 2B)
  - Payload layout: `[u8 channel][u8 shader]`

### System ops (`to = NODE_SYSTEM`)

- `OP_ACK` (Immediate)
  - Header: `cid = request cid`, `from = node that applied the request`
  - Header: `payload_idx = original opcode`, `flags = status (0=ok, 1=unhandled)`
- `OP_RESULT` (Payload, 0..64B)
  - Header: `cid = request cid`, `from = node producing the result`
  - Payload: opcode-specific small result blob

- `OP_ERROR` (Payload, 1..64B)
  - Payload layout: C string, null-terminated
- `OP_TRACE` (Payload, 1..64B)
  - Payload layout: C string, null-terminated
- `OP_ERROR_EX` (Payload, 52B)
  - Payload layout: `[u32 code][u8 op][u8 to][u8 from][u8 stage][u32 cid][u32 arg0][u32 arg1][char msg[32]]`
- `OP_QUEUE_FULL` (Immediate)
  - Header: `flags = destination node id`, `payload_idx = opcode that could not be enqueued`
- `OP_ASYNC_DONE` (Payload, 8B)
  - Payload layout: `[u32 task_id][u8 type][u8 slot][u8 success][u8 reserved]`
- `OP_STOP` (Immediate)
  - No payload

- `OP_FRAME_BEGIN` (Immediate)
  - Frame marker for render recording; resets the render list when applied
  - No payload
- `OP_FRAME_END` (Immediate)
  - Frame marker for render recording
  - No payload
- `OP_PRESENT` (Immediate)
  - Frame marker for render recording (present/swap boundary)
  - No payload

- `OP_FENCE` (Immediate)
  - Header: `cid = correlation id`
  - Header: `flags = target channel` (`CHAN_RENDER` recommended)
  - When applied, the system emits `OP_RESULT` with:
    - Header: `flags = OP_FENCE`
    - Payload (8B): `[u32 tick][u8 channel][u8 rsv][u16 rsv]`

  Note: this is an *apply-time* fence for the step thread ("messages drained and routed"). It is not a GPU hardware fence.

## Notes

- The ABI as described here is the "as implemented" behavior.

Note: with the multi-producer submit queue enabled, `tick` is assigned when the step thread routes the message into a node inbox (i.e. when it is applied), not when the producer enqueues it.
- The preferred direction is to evolve this into a "Message ABI v1" with explicit validation rules and stable capture/replay.

## Render Recording

When `HSSystem.render_list` is set (by `hs_gpu_init`), the step thread records render-relevant ops into an `HSRenderList` at apply time.

Currently recorded ops include:

- state: `OP_CULL`, `OP_BLEND`, `OP_ALPHA`, `OP_DEPTH`, `OP_DEPTH_COMPARE`, `OP_COLOR_MASK`, `OP_CLIP`, `OP_TEXTURE_FILTER`, `OP_TEXTURE_WRAP`
- commands: `OP_CLEAR`, `OP_CLEAR_DS`, `OP_DRAW`, `OP_DRAW_INSTANCE`, `OP_DRAW_TEXT`, `OP_SHOW_TEXTURE`
- frame markers: `OP_FRAME_BEGIN`, `OP_FRAME_END`, `OP_PRESENT`

The minimal GLES executor is implemented in `src/hs_backend_gles.c`.

## Capture Filtering

The message log used for capture can be channel-filtered:

- `HSSystem.record_mask` is a bitmask of channels to record.
- Default is render-only (`CHAN_RENDER`).
- Set it at runtime with `hs_set_record_mask()`.

## Message-Driven Buffers (Current)

The GLES backend can optionally source vertex data from `HSGpu.memory` buffer banks. The current minimal convention is:

- buffer bank `N` contains interleaved floats: `x, y, r, g, b` per vertex
- vertex count is derived from `HSBuffer.length / (5 * 4)`

## Validation

`hs_validate_message()` (see `include/hs_core.h`, implemented in `src/hs_core.c`) enforces:

- Destination routing: each opcode has an expected `to` node (except `OP_NOOP`, which may target any registered node).
- Payload length rules: `none`, `fixed`, or `string/range`.
- Payload index range checks for payload-bearing ops.
- Null termination for string payload ops (`OP_DRAW_TEXT`, `OP_ERROR`, `OP_TRACE`).
