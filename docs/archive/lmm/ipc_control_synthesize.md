# SYNTHESIZE: IPC Control Surface (NeoGPU)

Goal: allow external tools to query/tune a running NeoGPU instance via IPC, without linking into the process, while preserving RT/RENDER pristinity and deterministic behavior.

## Output: P0 Spec + Plan

### 1) IPC Scope (P0)

- Transport: Unix domain socket (AF_UNIX, SOCK_STREAM), local machine only.
- Server lives inside the NeoGPU process (library-side), in a dedicated thread.
- Client is a standalone binary (`neogpu_tool`).
- Allowed operations (whitelist):
  - Queries: `OP_QUERY_STATS`, `OP_QUERY_FABRIC`
  - Setters: `OP_SET_RECORD_MASK`, `OP_SET_CHAN_BUDGET`, `OP_SET_BLOCK_POLICY`
  - Fence: `OP_FENCE`
- Channel semantics: server forces all requests to the correct channel (same clamp rules as the runtime: queries/setters/fence -> `CHAN_RT`).

Non-goals (P0): remote network access, authentication, arbitrary message injection, multi-node control, large streaming telemetry.

### 2) Wire Protocol (versioned, length-prefixed)

All integers little-endian.

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

Notes:
- Requests map to `Message {to=NODE_SYSTEM, from=NODE_CPU, op=op, cid=cid, channel=CHAN_RT}`.
- Payload bytes become message payload via `hs_send_with_payload`.
- For immediate ops, `payload_len` must be 0.

Response body:

```text
u32 status         (0=OK, nonzero=error)
u32 result_op      (echo: which op the result corresponds to)
u32 result_len
u8  result[result_len]
```

Success responses return the raw `OP_RESULT` payload bytes (for queries + setters + fence). Clients decode using `include/hs_msg.h` layouts.

Error status codes (P0):
- 1: bad_magic
- 2: bad_version
- 3: bad_type
- 4: bad_len
- 5: unsupported_op
- 6: bad_payload
- 7: enqueue_failed
- 8: timeout
- 9: internal_error

### 3) Result Bus (required for correlation)

Add a bounded internal bus that records recent `OP_RESULT` messages (and their payload bytes) for tooling correlation.

P0 behavior:
- The bus is a fixed-size ring buffer `HS_TOOLBUS_SIZE`.
- Each entry stores: `cid`, `result_op` (the `OP_RESULT.flags` value), `payload_len`, `payload_bytes[]`.
- On each `OP_RESULT` delivered to `NODE_SYSTEM`, record a copy to the bus and signal a condition variable.
- IPC server waits for `(cid, result_op)` match with timeout.

Design constraints:
- Bounded memory and bounded per-request wait time.
- Toolbus write must not block RT; if the bus is full, increment a counter and drop toolbus copies (IPC request then times out).

### 4) Backpressure and QoS

- Per-connection cap: limit outstanding requests (P0: 1 in-flight per connection, sequential request/response).
- Server rejects frames larger than `HS_PAYLOAD_SIZE` (or a small multiple if we later support bigger payloads).
- Timeouts are explicit; do not hang the client.
- Server must never hold `HSSystem` lock while performing socket I/O.

### 5) `neogpu_tool` CLI (P0)

Commands:
- `neogpu_tool query-stats`
- `neogpu_tool query-fabric`
- `neogpu_tool set-record-mask <mask>`
- `neogpu_tool set-budget <rt|render|telem> <budget>`
- `neogpu_tool set-block <rt|render|telem> <0|1>`
- `neogpu_tool fence <rt|render|telem> <cid>`
- `neogpu_tool watch --stats --fabric --period-ms N`

Decoding:
- Use the same payload layouts as `include/hs_msg.h`.

### 6) Red-Team / Falsification Plan

- Malformed frames: wrong magic/version/type/len, truncated bodies, oversized payload_len.
- Flood: rapid-fire queries while TELEM spam occurs; verify RT results still return or time out without deadlock.
- Disconnect mid-frame and mid-response.
- Toolbus overflow: ensure bounded behavior (drop counter increments, server returns timeout/error).

### 7) Success Criteria

- Can start server and attach `neogpu_tool` to a running instance.
- Query/set/fence all work with correct `OP_RESULT` payloads.
- Under TELEM spam, query results still progress (or fail predictably with timeouts).
- No new hangs in `neogpu_demo` falsification suite.
- Benchmark overhead for 100Hz stats polling is negligible (documented).

## Implementation Sequence (when we start coding)

1) Add `docs/IPC_CONTROL.md` derived from this synthesis.
2) Implement toolbus (ring buffer + condvar) and wire `OP_RESULT` recording.
3) Implement IPC server module (`hs_ipc_*`) with strict parsing + whitelist.
4) Implement standalone `tools/neogpu_tool.c` client.
5) Add red-team tests (malformed frames, flood, disconnect) and bench.
