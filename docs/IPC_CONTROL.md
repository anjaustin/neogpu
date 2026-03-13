# IPC Control Surface (P0)

This document specifies the P0 IPC control surface for NeoGPU.

Purpose: allow external tools to query/tune a running NeoGPU instance using the existing `NODE_SYSTEM` query/control contract.

Source: derived from the Lincoln Manifold pass in `docs/lmm/ipc_control_*.md`.

## Scope

- Local machine only (Linux/Pi): Unix domain socket.
- Requests are limited to `NODE_SYSTEM` query/control/fence ops.
- Responses return raw `OP_RESULT` payload bytes.

## Allowed Ops (Whitelist)

- `OP_QUERY_STATS` -> `OP_RESULT(flags=OP_QUERY_STATS, payload_len=64)`
- `OP_QUERY_FABRIC` -> `OP_RESULT(flags=OP_QUERY_FABRIC, payload_len=64)`
- `OP_SET_RECORD_MASK` (payload 4B) -> `OP_RESULT(flags=OP_SET_RECORD_MASK, payload_len=8)`
- `OP_SET_CHAN_BUDGET` (payload 8B) -> `OP_RESULT(flags=OP_SET_CHAN_BUDGET, payload_len=8)`
- `OP_SET_BLOCK_POLICY` (payload 2B) -> `OP_RESULT(flags=OP_SET_BLOCK_POLICY, payload_len=8)`
- `OP_FENCE` -> `OP_RESULT(flags=OP_FENCE, payload_len=8)`

Channel semantics: server forces these requests to `CHAN_RT` regardless of client input.

## Wire Protocol (NGIP v1)

All integers are little-endian.

Frame header (16 bytes):

```text
u32 magic   = 0x4E474950  ('NGIP')
u16 version = 1
u16 type    = 1=request, 2=response
u32 len     = bytes following this header
u32 cid     = correlation id
```

Request body:

```text
u8  op
u8  flags
u16 reserved1
u32 payload_len
u8  payload[payload_len]
```

`flags` is mapped to `Message.flags` for immediate ops that use it (e.g. `OP_FENCE` uses `flags` as the target channel).

Response body:

```text
u32 status    (0=OK, nonzero=error)
u32 result_op (echo)
u32 result_len
u8  result[result_len]
```

Error status codes:

- 1: bad_magic
- 2: bad_version
- 3: bad_type
- 4: bad_len
- 5: unsupported_op
- 6: bad_payload
- 7: enqueue_failed
- 8: timeout
- 9: internal_error

## Result Correlation (Toolbus)

`OP_RESULT` messages to `NODE_SYSTEM` are copied into a bounded internal ring buffer ("toolbus") so the IPC server can match results by `cid`.

- Ring size: 256 entries (configurable via `HS_TOOLBUS_SIZE`)
- Each entry stores: `cid`, `seq`, `result_op`, `payload_len`, `payload[64]`
- Toolbus is cleared on `hs_clear()`
- Dropped counter: `HSSystem.dropped_toolbus`

If the toolbus is full, toolbus copies are dropped (counter increment). IPC requests may time out.

## Red-Team Requirements

- Reject malformed frames and oversize payloads.
- Bound memory use and reject flooding.
- Timeouts are mandatory: never hang a client.
- IPC server must not hold `HSSystem` lock while doing socket I/O.
