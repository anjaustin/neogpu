# Nodes of Interest: NeoGPU Messaging Layer

## Node 1: The core loop is good
Observation: `hs_step()` processes node inboxes, flushes outboxes, and ticks forward. It is simple and predictable.
Why it matters: this is a solid foundation for determinism and performance.

## Node 2: The message ABI is implicit
Observation: per-op payload layouts are encoded ad-hoc in producers and assumed in consumers.
Why it matters: scaling op count and producers increases breakage risk.

## Node 3: `payload_idx` has two meanings
Observation: it sometimes means "immediate value" and sometimes means "index into payload ring".
Why it matters: validation and tooling cannot reliably interpret messages.

## Node 4: Payload lifetime is not captured
Observation: recording logs message headers; payload data is not guaranteed stable for replay.
Why it matters: replay is only conditionally correct.

## Node 5: 64B payload cap is both a feature and a constraint
Observation: fixed payload blocks keep things fast and cache-friendly.
Why it matters: some commands naturally exceed 64B and will tempt hacks.

## Node 6: Validation is missing
Observation: nodes check lengths opportunistically (or not at all), but there is no central validator.
Why it matters: malformed messages become silent state corruption.

## Node 7: System ops routing is inconsistent
Observation: system ops exist (`OP_ERROR`, `OP_TRACE`, `OP_STOP`), but the routing semantics are not clearly defined.
Why it matters: it weakens observability and replay usefulness.

## Node 8: Nodes currently represent "state trackers"
Observation: `hs_nodes.c` mostly updates state and prints debug logs; it does not yet execute real GLES draws from messages.
Why it matters: we need an execution boundary (backend) to turn state into platform calls.

## Node 9: Outbox path exists but is underused
Observation: nodes have outboxes and the system forwards them, but most current nodes do not emit derived messages.
Why it matters: could be leveraged for acknowledgements/events, but only if ABI is clear.

## Tension A: Speed vs Schema
Tradeoff: keeping the ABI informal is fast now; formalizing schemas reduces future thrash.

## Tension B: Fixed payload blocks vs expressive commands
Tradeoff: 64B blocks are fast and simple; richer commands want larger structures.
