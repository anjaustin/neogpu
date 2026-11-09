# Architecture evidence contract

Baseline: Pi commit `4bd8221`. CI is deferred. No runtime fixes precede the
initial architectural measurements.

## Success and invariants

Success means reproducible evidence distinguishing supported guarantees from
counterexamples, not a predetermined winning benchmark. Preserve the existing
ABI, all source files, delivery checks, and previous acceptance gates. Separate
in-process admission/application, IPC, command construction, GPU execution, and
presentation. A CPU queue result is not graphics throughput.

## Phase 1: deterministic architectural probes

Each probe emits a JSON record. Process failure means the experiment could not
be executed as specified; a recorded false hypothesis is a valid finding, not
a passing architectural guarantee. All accepted messages must be accounted for.

| Hypothesis | Experiment / falsification |
|---|---|
| Per-producer FIFO survives fallback | Fill a producer's SPSC lane, admit fallback work, free one lane slot, admit newer work. Inspect actual handler order. |
| Bounded budgets give producer fairness | Two producers preload work; drain one small-budget step and count applications per producer. This tests a step, not sustained starvation. |
| Departed producers reclaim registration | Sequentially create/join more threads than lane capacity; inspect registration and fallback counters and verify delivery. |
| Stalled fallback reservation cannot block peers | Controlled queue-state fixture: reserve the head without publication, enqueue behind it, also enqueue in another lane/channel. Observe progress before/after publication. This models a stalled reservation, not an actual process crash or cancellation test. |
| Independent frame sequences compose coherently | Interleave A begin/draw and B begin/draw/end/A end; inspect the decoded render artifact and owner-tagged draws. This is command-list evidence, not pixel evidence. |

No claim of process crash isolation follows from an in-process thread test.
Frame coherence requires an ownership/transaction definition; retaining both
independent draw contributions is a necessary initial criterion, not sufficient
proof of coherent resource/state snapshots.

## Phase 2: latency and isolation

Measure RT application latency with an unloaded control, background render
traffic, and separately a deliberately slow synchronous handler. Record admission
attempt/start/end/application timestamps, loss, retries, per-producer service,
p50/p95/p99/max, and the workload duration. Distinguish RT-first drain order from
preemption of already-running handlers. Use a monotonic clock; do not subtract
unmeasured overhead or infer deadlines from percentiles. Repeat on the same Pi.

## Phase 3: external performance comparisons

Pin an established optimized queue baseline to a full source revision and retain
license/source provenance. Match message size, payload copying, producer count,
queue capacity, consumer work, batching, instrumentation, compiler flags, CPU
affinity, and timing boundaries. Report mismatches explicitly. Include recording
and render-list construction as separate workloads. Do not use a naive mutex
queue as the sole competitor. Do not compare published desktop queue rates or
GPU draw rates to NeoGPU CPU messages/s.

For graphics compare equivalent raw GLES and NeoGPU scene/resource/state/draw
work on the same framebuffer, verify output pixels, and distinguish CPU enqueue,
GPU completion, and presentation timing. bgfx is relevant prior art but not a
numerically comparable baseline until equivalent work is implemented and run.

## Reproducibility and limits

Retain raw JSON/JSONL, source hashes, baseline revision, compiler/flags, kernel,
CPU/governor/affinity, thermal/throttle observations where available, run order,
and exit status. Repeat measurements; report all samples and variability. No
novelty, superiority, hard-real-time, recovery, or independent-process guarantee
is established merely by completing a test. Counterexamples precede remediation
proposals. Prior test/sanitizer gates must remain green after any eventual fix.
