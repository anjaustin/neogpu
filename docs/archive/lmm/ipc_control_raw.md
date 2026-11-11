# RAW: IPC Control Surface (NeoGPU)

I think the next big leap is to make NeoGPU introspection/control usable against a running instance without linking into the process. Right now we have a solid `NODE_SYSTEM` query/control surface (`OP_QUERY_STATS`, `OP_QUERY_FABRIC`, `OP_SET_*`) and a demo-only CLI (`./neogpu_demo --tool ...`) that exercises it in-process. That is great, but it does not solve the actual “tooling” problem: attach to a live system, query, tune, capture.

My instinct is to add a small IPC server built into the runtime (Linux/Pi target -> Unix domain socket). Then ship a standalone client binary that speaks a stable wire protocol. The tool can run on the same box, read the stats, adjust budgets/record mask, and maybe trigger capture.

What makes this scary:
- The current `Message` ABI is intentionally strict and capture format checks struct sizes. That is good internally, but it is bad for a wire format. If we just dump `struct Message` over a socket, we risk padding/endianness/ABI drift.
- Query/control ops currently return results as `OP_RESULT` messages that are processed by `system_node_process`, which only stores “last result”. That is fine for tests but not for multiple concurrent tool requests. We need a real result mailbox.
- The system must remain pristine for RT and render. Tooling must not be able to block the render loop or degrade RT reliability. Also TELEM spam and tool spam must not starve RT.

Constraints / environment assumptions:
- Primary platform is Linux on Pi4; Unix sockets are available.
- We want deterministic behavior; “budgeted drain schedule” exists. Tool ops should probably be RT.
- Security posture: local tooling, not remote. No auth for P0, but strict permissions (socket file mode) and minimal allowed ops.

Questions:
- Do we want the IPC server to run inside the apply thread, or as a separate thread that enqueues messages? If separate thread, how do we wait for results cleanly?
- Should the wire protocol be binary or text (JSON)? Binary seems better for performance and determinism, but text is easier to debug.
- Should the tool have direct “get stats” requests that bypass the message fabric (reading `HSSystem` under lock), or should everything go through `NODE_SYSTEM` messages? Bypassing might be faster but risks divergent semantics.

Naive approach:
- “Just open a socket and memcpy Message in/out.” That will absolutely break later.

What I think we want:
- A tiny, versioned, length-prefixed binary protocol.
- Requests are strictly scoped to `NODE_SYSTEM` ops (queries + setters + fence). No arbitrary message injection.
- Server reads request, converts to a `Message` (RT channel) + optional payload, submits via existing `hs_send(_with_payload)`.
- Server waits for matching `OP_RESULT` with the same `cid`, and returns result payload to client.

But that requires a “result bus” that preserves multiple results and payload bytes. Right now we only have `SystemState.last_result_*` and payload idx. That is shared and will be racy under concurrency.

I also need to keep the codebase clean: ideally this is its own module (`hs_ipc.c` + `include/hs_ipc.h`) and can be compiled out if desired.

Risks:
- IPC becomes a side-channel that makes the engine nondeterministic or adds jitter.
- Backpressure: tool client could flood and cause memory growth if we buffer too much.
- Timeouts: if a request waits for a result, but the system is stopped or overloaded, we must fail quickly.

Success looks like:
- Start server with a path.
- `neogpu_tool query-stats` prints the same decoded stats as in-process tool.
- `neogpu_tool set-budget render 8192` returns old/new and the change is reflected immediately.
- Under TELEM spam + tool spam, RT queries still get answers (or predictable timeouts) without stalling the engine.
