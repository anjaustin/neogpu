# Synthesis: Deploying LMM Output Into an Execution Plan

## Goal
Turn the current fast message layer into a stable, debuggable, replayable command stream that can support real backends and engine features without ABI thrash.

## Key Decisions

1) Keep `Message` small and queues fixed-size for performance, but make every opcode's contract explicit.

2) Standardize payload semantics:
- "Immediate" fields remain allowed, but only when the opcode spec says so.
- Payload bytes are typed via packed structs and helper functions.

3) Make capture/replay self-contained by deep-copying payload bytes into a capture buffer.

4) Add a backend boundary: nodes decode to state/intent; backends execute platform calls.

## Implementation Spec (Milestone 0: Core Hardening)

### 1) Write the ABI spec
- Add `docs/MESSAGE_ABI.md` listing every opcode with:
  - destination node
  - field semantics (`flags`, immediate uses)
  - payload schema and length bounds
  - validation rules (ranges, enums)

### 2) Add typed pack/unpack helpers
- Add a header (name TBD) that defines packed payload structs per opcode.
- Provide `hs_msg_pack_*` / `hs_msg_unpack_*` helpers so producers/consumers share a single source of truth.

### 3) Unify payload allocation and validation
- Create one allocator/copy API used everywhere.
- Add debug-only validation on send/receive:
  - opcode known
  - payload_len within bounds
  - destination node exists
  - indices within range

### 4) Make recording self-contained
- Define a capture format in memory:
  - message headers
  - payload byte storage (deep-copied)
  - ABI version/hash
- Replay validates before executing.

### 5) Backend execution boundary
- Define `include/hs_backend.h` interface.
- Implement a GLES backend that consumes decoded state/intent.
- Keep `hs_graphics.h` as a platform helper, but move "what to draw" decisions to the backend layer.

### 6) Fix system op routing
- Decide whether to add a system node or keep system logging in `HSSystem`.
- Ensure `error/trace/stop` are observable and replayable.

## Success Criteria

- [ ] Every opcode has an explicit payload schema and validator coverage.
- [ ] Capture/replay does not depend on transient payload ring contents.
- [ ] Malformed messages fail loudly in debug builds.
- [ ] A backend can execute message-driven draws without changing the message ABI.
- [ ] Message benchmark stays within an acceptable regression budget (target: <= 5%).
