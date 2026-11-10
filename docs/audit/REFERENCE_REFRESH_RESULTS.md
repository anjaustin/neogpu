# Reference refresh: committed default path

Runtime revision: `c9307e1` (production runtime unchanged from `3aaab60`).
Measurement contracts, references, failure preservation and reproduction tools
are described in [REFERENCE_REFRESH.md](REFERENCE_REFRESH.md).

## Findings

- The new concurrent real-GLES comparison completed **120 measured processes,
  112,640 contributions and 30,760 rendered frames**. All contributions were
  observed in FIFO order in completed render frames. Final 800×480 RGBA bytes
  matched across native/reference paths, capture modes and repetitions; nonblank
  lane pixels and complete context/fd release were checked. Four separate actual-
  GLES ASan/UBSan/leak qualification executions passed.
- Across its 12 workload/capture cells, the ratio of NeoGPU/reference median
  contribution rates ranged **0.9927–1.0106**. These are point measurements, not
  statistical equivalence, a confidence bound, or a general engine speedup.
- Forced-rebuild retained rendering measured **461.038 µs direct GLES** versus
  **463.120 µs NeoGPU**, a difference of **2.083 µs / 0.452%** between aggregate
  medians. Ranges overlap; this is not a hard overhead bound.
- The optimized CPU transport comparison remains workload-dependent: NeoGPU
  medians ranged **71.5%–112.8%** of the pinned static SPSC reference. This is a
  different workload boundary from rendering.
- Complete cleanup still costs substantially more than the leaking parent path.
  No rejected state cache or shutdown candidate was installed for these tests.

## Concurrent producer → real GLES completion

Five alternating pairs per cell, 256 immutable contributions per producer,
one/64 draws, 1/2/8 producers, capture off/on. Same actual GLES executor on both
paths; reference replaces only admission/service/composition. Every render frame
ends with `glFinish`, so work completion is serialized. The consumer is pinned
to CPU 0 and producers to CPUs 1–3. Driver-thread affinities are separately
recorded. This is neither scheduler-unrestricted asynchronous rendering nor
physical scanout latency. Stable geometry is verified by pixels; sequence
integrity is verified against consumed views, not visibly encoded animation.

Capture-on medians (rates are **contributions/s, not FPS**; latency is median
of per-process p99 first-attempt-to-completed-frame, including queue/retry delay):

| Producers | Draws/contribution | NeoGPU rate | Reference rate | NeoGPU p99 ms | Reference p99 ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 2,826.78 | 2,831.73 | 2.330 | 2.288 |
| 1 | 64 | 1,281.61 | 1,282.56 | 5.024 | 4.922 |
| 2 | 1 | 5,377.63 | 5,370.97 | 2.405 | 2.442 |
| 2 | 64 | 1,657.45 | 1,656.01 | 7.599 | 7.619 |
| 8 | 1 | 17,216.83 | 17,059.64 | 3.148 | 2.969 |
| 8 | 64 | 2,069.85 | 2,072.85 | 23.830 | 23.904 |

For eight producers/64 draws/capture on, median timed aggregate process CPU was
**3.103 s native / 3.109 s reference**. Whole-process wall medians were
**1.299/1.302 s**, including startup/output/cleanup; peak RSS medians were
**84,880/84,828 KiB**. This sums CPU across threads, not GPU execution time or
energy. Per-process major-fault medians were zero in all capture-on cells.
Involuntary context-switch medians in this dense oversubscribed case were
**813,420/842,534**. The saturated producer policy calls `sched_yield` on full
queues, so these are properties of this explicit benchmark policy as well as
the runtime, not expected costs for every application.

Admission may return after consumption; completed latency is measured from the
first attempt, and admission timestamps are read after producer join. Observer
clocks/cache traffic, full draw-schema/FIFO validation, retries and worker joins
are included for both paths. Pixel readbacks and output files are outside warm
timing. Whole-process timings additionally include startup, verification/output,
cleanup and sudo/time overhead. Peak RSS is not allocated GPU memory.

Full off/on results retain rate ranges, paired ratios, frame counts, retries,
admission/completion p50/p95/p99/max, submission/completion intervals, aggregate
process CPU, startup/cold/shutdown costs, RSS, faults and context switches in the
derived E2E report. A six-panel contact sheet was visually inspected: native and
reference have matching nonblank red/green triangles in all intended lane
positions for one, two and eight producers.

## Optimized CPU transport/application

Seven alternating pairs per cell, 200,000 contributions per producer. Both
implement four usable queue slots per lane, validate/copy immutable frames,
observe canonical applied content and per-lane FIFO, and run the same sampled
latency observer. The pinned reference is an optimized SPSC adapter, not a naive
mutex queue and not an implementation of NeoGPU's dynamic lifecycle contract.
All **84,000,000 accepted contributions** were applied across 112 timed runs.

| Producers | Draws | NeoGPU M contributions/s | Reference M contributions/s | NeoGPU/reference | NeoGPU p99 µs | Reference p99 µs |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 3.175 | 4.441 | 0.715 | 0.760 | 0.759 |
| 1 | 64 | 1.011 | 1.093 | 0.925 | 1.908 | 2.667 |
| 2 | 1 | 5.012 | 5.304 | 0.945 | 1.833 | 2.333 |
| 2 | 64 | 1.136 | 1.128 | 1.007 | 9.500 | 10.111 |
| 4 | 1 | 4.771 | 5.042 | 0.946 | 2.611 | 2.944 |
| 4 | 64 | 1.109 | 1.109 | 1.000 | 20.537 | 21.278 |
| 8 | 1 | 4.501 | 3.990 | 1.128 | 16.723 | 14.852 |
| 8 | 64 | 0.996 | 1.062 | 0.938 | 45.352 | 42.908 |

Per-run ranges are retained in the comparison report. These latency p99s are
sampled once per 128th contribution and cannot be pooled with the all-
contribution real-GLES completion distribution.

Eight-producer/one-draw NeoGPU has the higher throughput median but worse
sampled p99 latency (16.723 versus 14.852 µs). A rate win is not a tail-latency
win. Small dense-workload point differences and overlapping ranges must not
be generalized as an algorithmic advantage.

## Direct GLES versus retained executor

After an explicit `make -B tests/test_comparison_graphics`, three fresh
collections contained 15 alternating 600-frame blocks per path: **18,000 timed
frames**. Both paths share one context/surface, equivalent optimized shaders,
preloaded geometry, two draws and actual swap. Contributions are admitted before
timing; this measures retained manual composition plus stock execution, not
concurrent producer delivery or the automatic default participant API.

Median of block medians, microseconds:

| Boundary | Direct GLES | NeoGPU |
|---|---:|---:|
| Begin → pre-swap submission | 77.250 | 79.648 |
| Begin → `glFinish` completion | 418.260 | 420.380 |
| `glFinish` wait | 339.324 | 339.361 |
| Real swap call | 42.019 | 41.703 |
| Whole instrumented frame | 461.038 | 463.120 |

Whole-frame block medians ranged **458.815–463.325 µs** direct and
**461.130–467.398 µs** NeoGPU. Block p99s ranged **502.981–848.611 µs** and
**503.704–557.945 µs**; tails and ranges are retained, not discarded. These
metrics' independently aggregated medians need not add. All framebuffer bytes
and nonblank interior pixels matched; shaders were checked equal. The earlier
incremental-build repeat measured 461.825/465.926 µs and is retained separately,
not pooled with the forced-rebuild authoritative result. Neither produces a
panel-FPS or graphics-engine superiority claim.

## Default frame integration versus frozen manual plumbing

Three pairs, 200,000 contributions per producer. **84 measured executions**
applied **2,400,000 legacy scenes**, **26,400,000 native contributions**, and
**26,400,000 reference contributions**. These units are reported separately.
Both revisions use the same validating mock backend here, not GPU execution.

Native medians, millions of validated applied contributions/s:

| Producers | Draws | Frozen `7d1889d` manual plumbing | Current default integration |
|---:|---:|---:|---:|
| 1 | 1 | 2.887 | 2.728 |
| 1 | 64 | 0.787 | 0.761 |
| 2 | 1 | 4.063 | 4.083 |
| 2 | 64 | 0.828 | 0.782 |
| 8 | 1 | 3.078 | 3.124 |
| 8 | 64 | 0.729 | 0.732 |

Legacy-scene medians were 1.128/1.129 M scenes/s (one draw) and
0.1296/0.1293 M scenes/s (64 draws), frozen/current respectively. Integration
does not show a universal throughput improvement over manual plumbing. The
full report includes the current static-reference path, retries, backend calls
per contribution, initialization and memory layout.

## Seven-scene rendering and complete lifecycle

Exact committed `2bf623a` versus `c9307e1`, with no common source repair or cache
injection. Seven pairs, seven scenes and capture off/on produced **196 timed
processes / 100,352 timed frames**. Fourteen separate scene/capture audits
verified RGBA equivalence and matching GL-call counts. Current contexts were
released in **98/98** timed processes; the parent in **0/98**. Separate complete-
cleanup actual-GLES sanitizer qualifications passed (both `3aaab60` and current
have the same production runtime); they are excluded from speed results.

Capture-on median submission costs, microseconds/frame:

| Scene | Parent | Current |
|---|---:|---:|
| single | 62.056 | 61.963 |
| same | 226.139 | 224.703 |
| alternate | 407.399 | 397.574 |
| dirty | 96.481 | 96.398 |
| mixed | 290.213 | 292.416 |
| fallback | 69.278 | 69.713 |
| retained | 221.361 | 220.731 |

Submission is not completed GPU work. The full report also records normal
presentation plus a single separately recorded block-end `glFinish`, process
CPU, capture off/on, startup and teardown, with per-pair ratios and ranges.
These asynchronous blocks must not be pooled with per-frame-finish E2E timing.
No general rendering speedup is established. In the single-draw capture-on cell,
parent/current shutdown medians were **1.456/35.884 ms wall** and
**1.497/6.511 ms process CPU**: the shorter parent path did not release its EGL
context and is not equivalent complete cleanup.

## Separate whole-process profiles and environment

All four perf and four syscall-summary processes exited 0. Perf's `branches`
event reported **not supported**, so no branch-miss rate is derived. Other raw
counter rows, including cycles/instructions/cache events/task-clock, are kept.
For the dense eight-producer/64-draw/capture-on sample, native/reference whole-
process task-clock was **3,324.45/3,282.50 ms**, cycles **5.670/5.583 billion**,
and instructions **4.416/4.389 billion**. These are separate single diagnostic
executions covering startup, warmup, measured work, validation/output and
teardown, not gated steady-state causal attribution or throughput samples.

In the dense syscall probes, `sched_yield` accounted for **70.58%/71.21%** of
accumulated traced syscall time, followed by futex **18.39%/17.91%** and ioctl
**8.95%/8.77%**. There were **80,213/80,010** yield calls under tracing. This
identifies backpressure/scheduling as a diagnostic target in this saturated
policy; strace changes scheduling and the summed thread syscall times can exceed
wall time. It does not mean 70% of untraced application CPU or wall time is
proven recoverable. No new optimization follows from these counters alone.

Observed boundary temperatures across the refreshed collections ranged
**34.076–57.452 °C**, CPU frequency snapshots **600–1,800 MHz**, with unchanged
`ondemand` governors. Throttle queries failed (status 255); no absence-of-
throttling claim is made. These are boundary observations with cached-record
duplicates, not continuous telemetry. The actual-GLES process had two threads
after warmup, with observed masks **[0]** and **[0,1,2,3]**; producer affinity
was subsequently pinned as specified. No assumed all-threads-on-one-core model
is used. Pi wall timestamps differ from the host and are preserved literally;
duration/latency calculations use monotonic clocks.

## Verification and provenance

Headless regression, all-source warning-as-error checks, standard ASan/UBSan/
leak gates, CPU reference/native ASan/UBSan/leak and TSan qualifications, actual-
GLES qualification, hardware captures and standard `make benchmark` passed.
The original failed report attempt and failed E2E qualification remain separate
from accepted measurements; see the measurement contract for their causes.
The production runtime was not modified. New diagnostics and results remain
uncommitted; the previously requested shutdown commit is `c9307e1`.

The actual-data reducer accepted its isolated positive copy and rejected
**12 corrupted collections**, including mismatched raw/cached timing, live
resources, wrong invocation, duplicated sequence, missing frame, wrong pixels'
hash, source/binary hash mismatches, missing run, invalid pair, zero elapsed
time and fabricated latency. A separate explicitly synthetic validator accepted
its positive control and rejected **12** malformed inputs. These controls are
not counted as hardware benchmark samples.

## Artifact map

- `tests/evidence-reference-c9307e1-refresh/`: orchestration, committed-source
  archive, resource reports, preserved failed/resumed wrapper attempts, standard
  benchmark output, authoritative frame/lifecycle derived reports.
- Sibling `-transport`, `-frames`, `-lifecycle`, `-gles-sanitizers`, `-hardware`
  and initial `-graphics-{1..3}` directories: all raw collections and builds.
- `tests/evidence-e2e-c9307e1/`: preserved failed initial qualification; no
  accepted timed measurements.
- `tests/evidence-e2e-c9307e1-bounded/`: frozen sources/reference/license,
  binaries, four sanitizer qualifications, 120 raw process outputs, RGBA frames,
  per-process GNU-time reports and manifest.
- `tests/evidence-reference-c9307e1-final/`: final E2E/comparison reductions,
  forced graphics build hash, environment summary and final gate invocations.
- Sibling `-graphics-{1..3}`, `-profiles` and `-e2e-controls`: authoritative
  retained-render collections, raw perf/strace output, isolated reducer controls.
- `tests/evidence-reference-c9307e1.tar.gz` and `.sha256`: complete local archive
  and checksum, including failed attempts and source/binary provenance.
