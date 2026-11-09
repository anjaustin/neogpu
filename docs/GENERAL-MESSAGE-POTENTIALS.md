# General-message redesign opportunities

Status: opportunities and falsifiable hypotheses, **not implemented performance
claims**. Current measured transport work is recorded in
[the bedrock audit](audit/BEDROCK_RESULTS.md),
[P0 results](audit/BEDROCK_P0.md), and [red-team review](audit/BEDROCK_REDTEAM.md).

The participant fabric publishes bounded, copied draw contributions using
acquire/release cursors. It retains generation-checked ownership, producer
serialization and locked retirement. It is not a replacement for arbitrary
shader/resource/control messages. The existing general `HSSystem` transport
already uses acquire/release SPSC queues with an MPSC fallback: its gains cannot
be inferred from eliminating the old participant consumer mutexes.

## Opportunities to investigate

| Opportunity | Potential payoff | Required contract and red-team checks |
|---|---|---|
| Complete render transactions | Prevent partial state/draw visibility; amortize admission and dispatch across related commands | Explicit commit boundary, validation before publication, ordered application, rejection without partial effects |
| Explicit ownership of state and resources | Keep asynchronous contributors from overwriting each other's inputs | Define shader/state scope, owned upload data, immutable snapshots or lifetime-managed references, generation-safe reclamation |
| Transaction-aware backpressure | Bound work and memory without falling back into a different ordering domain | Capacity in bytes and transactions, FULL/BUSY semantics, cancellation, saturation and producer-churn tests |
| Defined control/render ordering | Make fences and acknowledgements describe actual application or completion | Define acceptance, application, GPU completion and presentation separately; preserve RT channel service; test overtaking and starvation |
| Transaction capture/replay | Reproduce committed work with its resource dependencies | Versioned format, owned payloads, deterministic state, malformed-input rejection, compatibility with existing HSCAP1 captures |
| Batched validation/copy/service | Reduce repeated per-command work | Compare equivalent valid work, NEON/compiler-optimized baselines and pinned queues; count exact delivered work rather than admissions |
| Resource retirement and recovery | Reclaim resources only when no consumer/backend still uses them | Distinguish queue drain from GPU completion; test stalled producers, stale handles, failed uploads and teardown |

None of these payoffs has a measured speedup or memory-saving guarantee yet.
Bigger transactions can increase buffering latency, copy volume and failure
recovery cost. A single universal ordering domain could also compromise QoS.

## Preserve before extending

* Keep message opcode/wire/capture compatibility unless an explicit migration
  is approved. Do not silently reinterpret payload lifetimes or acknowledgements.
* Preserve full valid command coverage: shader, camera, texture, upload, state,
  draw, query, input and control work cannot be replaced by a draw-only adapter.
* Do not weaken lifecycle guarantees to reproduce a static SPSC benchmark win.
* Never infer GPU/display FPS from CPU transport throughput. Compare actual
  equivalent pixels and completion/presentation boundaries separately.

## Staged evaluation

1. Integrate the already-qualified participant path into normal `HSGpu` frame
   service while preserving legacy APIs. Measure idle-path and active-path cost.
2. Inventory actual application command sequences and ownership dependencies;
   record the missing transactional semantics before implementing them.
3. Build diagnostic transaction adapters and independent correctness observers.
   Test invalid commits, saturation, ordering, mixed channels and reclamation
   under sanitizers and deterministic adversarial schedules.
4. Compare fresh equivalent-work baselines/reference queues, instructions,
   cycles, retries, sampled latency and measured layout/allocation costs. Preserve
   failed experiments; independent timing differences are not additive budgets.
5. Verify GLES pixels and backend completion behavior. Promote only demonstrated
   improvements with explicit compatibility/latency tradeoffs.

The default frame-runtime integration is a bounded step toward these questions,
not an implementation of this general-message redesign.
