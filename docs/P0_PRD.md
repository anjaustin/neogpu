# P0: Messaging Layer Atomics and Thread Safety

**Project:** NeoGPU  
**Scope:** Core message fabric correctness under concurrency  
**Status:** Active (P0)  
**Date:** 2026-03-12

## Goal

Make the messaging layer safe and predictable when used from multiple threads, or fail fast with explicit rules. This P0 focuses on correctness first; performance tuning comes after.

## Current Snapshot (Quick-Glance Table)

| Area | Current Mechanism | Atomic/Thread-Safe? | Assumption Today | What Breaks If Violated | P0 Fix |
|---|---|---:|---|---|---|
| `MessageQueue` (`mq_push/mq_pop`) | Ring buffer `head/tail/count` | No | Single-thread per queue | Data races; lost/duplicated messages | Serialize queue ops under a system lock (re-entrant) |
| `hs_send()` | Validate -> enqueue -> log/render | No | All sends on one thread | Races on queues/payload/log/render | Serialize under system lock; re-entrant safe |
| `hs_step()` | Process nodes; drain outboxes | No | Single-thread dispatch | Push/pop races; inconsistent state | Serialize under system lock; re-entrant safe |
| Payload allocation | `payload_head` bump + copy | No | Single producer | Payload overwrite/corruption | Serialize under system lock |
| Recording log | append on enqueue success | No | Single writer | Corrupt log, overflow logic | Serialize under system lock |
| Render list | record in `hs_send()` | No | Single writer | Non-deterministic render cmds | Serialize under system lock |
| Validation | schema checks + payload reads | Not guaranteed | payload stable | false negatives/positives | Serialize under system lock |
| Error telemetry | best-effort enqueue to system inbox | No | system inbox not full | dropped telemetry | Track dropped counters; keep best-effort eventing |
| Backpressure | `OP_QUEUE_FULL` events | No | system inbox available | drop events | Track counters; still emit events best-effort |
| Async task queue | `pthread_mutex` + `sem_t` | Yes | worker touches only async internals | - | Keep |
| `HSAsync.running` | `volatile bool` | No | relies on OS primitives | UB in strict model | Replace with C11 atomic or guarded access |
| Async->messaging | main thread emits `OP_ASYNC_DONE` | Yes (by design) | worker never calls `hs_send` | races otherwise | Document + enforce (optional) |
| Capture/replay | single-thread replay | No | replay not concurrent | state corruption | Serialize under system lock; document rule |

## Requirements

1) **Thread safety option**: the system must be safe if `hs_send()` is called from threads other than the one running `hs_step()`.
2) **No deadlocks**: `hs_step()` may call node code that calls back into `hs_send()` (acks/errors). Locking must be re-entrant.
3) **Telemetry integrity**: failures to emit error/backpressure messages must be visible via counters.
4) **Falsification tests**: add a multi-thread stress test to ensure the above holds (no crashes, no deadlock).

## Deliverables

- A re-entrant `HSSystem` lock guarding `hs_send/hs_step/hs_replay/hs_clear/payload alloc`.
- Atomic fix for `HSAsync.running`.
- Dropped-telemetry counters.
- Tests proving multi-thread safety and re-entrancy.

## Implementation Notes (Now)

- The recursive system lock lives in `include/hs_core.h` / `src/hs_core.c` (`hs_lock/hs_unlock`).
- Multi-thread falsification test lives in `src/main.c` ("Thread Safety Tests").
- Dropped-telemetry counters live in `HSSystem` (`dropped_error_ex`, `dropped_queue_full`).
- `HSAsync.running` is implemented as `atomic_bool`.
