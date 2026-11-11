# NODES: IPC Control Surface (NeoGPU)

## Node 1: We Already Have a System API, But It's In-Process
`NODE_SYSTEM` supports query/control ops and returns fixed payloads via `OP_RESULT`. `neogpu_demo --tool` exercises it, but cannot attach to a running instance.

## Node 2: Wire Format Must Be Stable (Not `struct Message`)
The internal ABI is strict and may change; capture even validates struct sizes. IPC needs explicit endianness and field sizes.

## Node 3: Tooling Must Not Degrade RT/RENDER
Tool traffic should be RT (or clamped to RT) but still must not starve render or introduce blocking/jitter. TELEM remains droppable.

## Node 4: Result Correlation Requires a Real Mailbox
Current `SystemState` only retains "last result". Concurrency (multiple outstanding tool requests) requires buffering + matching by `cid` (+ expected op).

## Node 5: Minimal Attack Surface
P0 should reject arbitrary message injection. Allowed ops should be a strict whitelist of `NODE_SYSTEM` query/set/fence operations.

## Node 6: Server Thread vs Apply Thread
Server likely runs in its own thread (socket accept/read/write). It must enqueue messages into the fabric and wait for results without calling `hs_step()`.

## Node 7: Backpressure Policy for IPC
If tool floods, we need bounded memory and predictable failure: reject/timeout rather than unbounded buffering.

## Node 8: Timeouts Are a First-Class Outcome
Requests can fail due to overload, stopped system, or missing results. Protocol should include explicit timeout/error codes.

## Node 9: Reuse Existing `OP_RESULT` Payloads
Queries already have fixed 64B payload layouts. IPC can return those bytes unchanged, and clients decode using the same layouts.

## Node 10: Permissions/Scope (Local First)
Unix domain socket with restrictive filesystem permissions is a good P0 stance. Avoid network exposure.

## Node 11: Determinism vs Convenience
Reading stats "directly" under lock is easy but may drift from message-driven semantics; routing everything through `NODE_SYSTEM` preserves one semantic path.

Tensions:
- Node 11 vs performance: direct reads are cheap; message-based path is canonical.
- Node 3 vs Node 6: adding a new thread can introduce contention; we must keep lock hold times tiny.
- Node 4 vs Node 7: result buffering must exist, but it must be bounded.
