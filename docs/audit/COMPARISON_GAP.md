# Why the one-producer / one-draw gap is large

The committed comparison measured the **complete-frame participant protocol**,
not the old payload-free HSMessage path. Reference/native median rates were
4.314M/0.864M completed frames/s. This diagnosis changes no production source.

## Evidence

### Original-binary sampling

`perf record -e cycles -F 999` sampled the existing optimized comparison binary
with one producer, one draw and 1,000,000 frames on each path. Both completed
exact delivery. Native had **378,009 rejected/retried admission attempts**;
reference had **1,005** in these particular profiled executions. These are not
the five-pair benchmark medians and profiled times are not substituted for them.

The native profile includes `pthread_mutex_trylock`, `pthread_mutex_unlock`,
`sched_yield` and low-level libc atomic helpers. Disassembly of sampled libc
addresses shows compare/exchange / exclusive-load-store helper instructions.
The reference profile instead prominently samples its queue-service adapter
and common data/application work. Profiles include both producer and consumer;
kernel symbols were restricted and many kernel PCs remain unresolved. They do
not furnish an exact percentage decomposition of the fivefold throughput gap.

### Structural call counting

A separate linker-wrapped, count-only execution completed **10,000/10,000**
frames on both paths. It counted the main/consumer thread's explicit mutex calls:

| Calls | Native | Reference |
|---|---:|---:|
| Slot-zero trylocks, including one initial join | 15,265 | 0 |
| Trylocks on the seven other slots | 106,848 | 0 |
| Failed trylocks | 1,910 | 0 |
| Unlocks | 120,203 | 0 |

This exactly matches 15,264 service scans × eight slots, plus the join. The
consumer cannot read `active` safely outside synchronization under the current
concurrent join/leave contract, so inactive slots are locked too. Several scans
make no progress when the active queue is empty or busy.

The count wrapper changes timing and contention; its counts describe that
execution. They are not instrumentation-free per-frame cost estimates.

## Mechanism

The native producer and consumer **compete for the same per-slot mutex and
protected queue metadata** on admission/application. Trylock avoids sleeping
inside the API, but does not make the queue lock-free or remove synchronization
cost. Failed admissions cause the common driver to yield and retry. Scanning
all eight slots adds more synchronization work even for a one-participant run.

Rigtorp is specialized to one writer and one reader: each owns its queue index,
the indices are cache-line separated, opposite indices are cached, and it uses
acquire/release publication without a shared producer/consumer mutex. That is
well suited to this static SPSC workload. Both adapters still perform draw
schema/sequence validation and retained-copy/application checks; the reference
did not win by dropping that work.

The measured evidence points to synchronization/retry overhead as a major
implementation differentiator, not graphics cost: no GPU work occurs in this
CPU test. It does **not** prove that ownership/lifecycle guarantees inherently
cost fivefold, or that all of the gap belongs to one mutex operation. API dispatch,
metadata layout, observation and copying remain in the measured paths too.

## Failed ablation retained, not promoted to an optimization

`tests/test_comparison_gap.py` makes generated source copies under a fresh
`tests/evidence-*` directory. It preserves storage sizes, queue capacity and
frame work, but visits only slot zero in the service loops of both adapters.
That is **invalid for multiple participants** and cannot replace production code.

The million-frame native diagnostic hit its **90-second SIGALRM** and exited
`-14`; the collection is explicitly marked incomplete. No speedup or rate was
accepted from that failed run. A smaller 4,096-frame profiled execution completed
all frames but made **180,956 admission retries** in **329.787444 ms**, with
mutex/atomic and yield costs again prominent in its profile.

Removing seven probes did not isolate them as a simple fix. Rapidly revisiting
the same shared lock creates a pathological contention pattern in this
diagnostic. A production optimization would need a different service/admission
mechanism plus the full ownership, fairness, retirement and concurrency tests;
not merely a shorter loop or a skipped synchronization check.

## Rendering result is a different boundary

The near-direct-GLES rendering experiment reused already-applied retained views;
it excluded new producer admissions. Therefore its ~0.6% observed median total
frame difference does not contradict the transport gap. It is not a measure of
producer-to-screen latency under continuous updates.

## Artifacts and reproduction

- Original profiles: `tests/evidence-gap-{native,reference}.perf.data`,
  corresponding `*-report.log`, `*.jsonl`, and perf diagnostic logs.
- Call counts: `tests/evidence-gap-lock-{native,reference}-counts.json` and
  associated raw application results. `tests/test_comparison_lock_counts.c`
  is linked with the original optimized objects using linker wraps for
  `pthread_mutex_trylock` and `pthread_mutex_unlock`; do not benchmark rates
  from that instrumented binary.
- Failed and smaller ablations: `tests/evidence-gap-ablation/` and
  `tests/evidence-gap-ablation-small*`. The incomplete manifest retains its
  source copies, compile commands, exit status, timeout and partial collection.
- `tests/evidence-gap-libc-atomic-disassembly.log` records the sampled helper
  instruction region. Production `src/` and `include/` remain unchanged.

The comparison work was committed as `6e9d899`; the push was again rejected
because the earlier workflow commit requires a token with `workflow` scope.
No workflow removal, authentication change, history rewrite or speculative
production optimization was performed.
