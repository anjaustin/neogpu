# Initial architectural evidence: Pi baseline 4bd8221

## Outcome

The existing high-throughput message path is **not yet a composable graphics
participant protocol**. Reproducible counterexamples establish missing ordering,
fairness, reclamation, fallback isolation, and frame-contribution guarantees.
These are new architectural requirements, not failures of the previous bounded
remediation acceptance contract. No runtime source was changed for this work.

This is an initial architecture/latency result, **not a competitor performance
comparison or a novelty claim**. External queue and end-to-end graphics baselines
from the plan remain unimplemented and unmeasured. No superiority is established.

## Conditions and reproducibility

- Runtime baseline: `4bd8221d1c5521f1c2d3820f6deb2ee8545ea997`.
- Raspberry Pi, four aarch64 CPUs; GCC Ubuntu 15.2.0-16ubuntu1.
- Kernel: `7.0.0-1023-raspi`, PREEMPT_DYNAMIC.
- Default Makefile optimized flags: `-O3 -march=armv8-a+simd
  -mtune=cortex-a72 -ffast-math -funroll-loops -Wall -Wextra -DNDEBUG`.
- Allowed CPUs 0–3, no per-thread pinning; `ondemand` governor. At collection
  boundaries the reported frequency was 1,800,000 kHz. Temperatures at overall
  start/end were 37.972/41.868 C; these are snapshots, not continuous monitoring.
- `vcgencmd get_throttled` failed because `/dev/vcio` was unavailable. No
  no-throttling assertion follows from these runs. No device nodes were created.
- Pi wall-clock UTC differs from the harness date. Durations and sample timestamps
  use `CLOCK_MONOTONIC`; the manifest preserves the machine's UTC literally.

Run from the repository root on the Pi:

```sh
make evidence-probes
python3 tests/test_architecture_evidence.py --output tests/evidence-NEW-RUN
make evidence-sanitize
make check-sources
make test
```

The output directory must not already exist. Raw samples, actual build recipe,
source SHA-256s, runtime revision, exit statuses, affinity and before/after sysfs
snapshots are in `tests/evidence-architecture-final/`. A copy is retained in the
local `neogpu-evidence/tests/` snapshot. Generated evidence is ignored by git,
not deleted; no commit or push was performed.

Manifest SHA-256:
`db3dae820ec8503beab326f1dd541d7a6d65803dc6809436ec376681ed360a88`.

The runner recomputes every reported percentile from raw timestamps and checks
sample counts and background delivery. Exit zero means an executable experiment,
not that a tested hypothesis holds. The deterministic probes have positive
controls for FIFO without fallback and two draws inside one frame.

## Deterministic counterexamples

All three final repetitions produced byte-identical seven-record probe output.
Twenty additional executions of the same probe source also matched the initial
collection byte-for-byte. No admission/application loss occurred in the counted
transport probes.

| Question | Observed result | Supported conclusion |
|---|---|---|
| Same-producer FIFO across fallback? | 258 admitted/applied; final IDs **257, 256**; one inversion | Later SPSC work can overtake earlier fallback work. FIFO within a lane is not FIFO across both paths. |
| Per-step producer fairness? | With 8 queued messages each and budget 4, producer 0 applied **4**, producer 1 **0**; eventually all 16 applied | Ascending lane scans do not guarantee service to each participant per step. This finite experiment does not by itself establish indefinite starvation. |
| Lane reclamation after departure? | 24 sequential create/join threads, capacity 8; registration counter **24**, fallback admissions **16**, all 24 applied | Thread exit does not automatically return a registered lane. Joining/leaving still delivers via fallback, but registration is not dynamically reclaimed. |
| Does a stalled fallback writer isolate peers? | Unpublished fallback head: RT and independent SPSC messages applied; published fallback peer did not. Publishing the head allowed all four fixture messages to apply | Head-of-line blocking exists inside fallback; it does not block every lane/channel. This is a controlled reservation-state fixture, not an actual crash, killed thread, or IPC failure test. |
| Do independent frame sequences retain both contributions? | A begin/draw, B begin/draw/end, A end: decoded list retained A draws **0**, B draws **1** | Frame begin resets a shared list. Legal interleaving alone discards the earlier contribution. Distinct resource indices identify contributions; `cid` is not an implemented ownership protocol. |

The frame stream is deliberately prescribed through public message admission,
not dependent on accidental thread scheduling. Its counterexample is a necessary
command-artifact failure, not framebuffer/pixel evidence. A single begin with
both draws retains both, which is the positive control. No independent process
or independently owned resource snapshot protocol has been demonstrated.

## RT application latency

Each regular workload schedules 1,000 RT samples at 1 ms intervals, with one
consumer. The unloaded case has no background producer; flood adds one continuously
submitting render-channel producer. Run 2 reverses unloaded/flood order.

Messages are payload-free `OP_SET_SHADER` routed to an instrumented observer in
place of the stock shader handler. The observer timestamps RT applications and
counts render applications. Recording, decoded render-list construction, GPU and
IPC are absent. Consequently the background counts are **not** stock shader
throughput, draw rates, or a new NeoGPU performance headline.

Latency below is actual first admission **attempt to application**, including
admission retries and scheduling after that attempt, with no overhead subtraction.
Admission-return and intended scheduling timestamps are retained separately.
Application can in general precede admission return; no negative latency is
clamped or hidden. Scheduled-to-application latency can be recomputed from raw
records; the table does not include pre-attempt timer wakeup delay.

Nearest-rank percentiles; all values in **microseconds**:

| Workload | Run | p50 | p95 | p99 | Maximum |
|---|---:|---:|---:|---:|---:|
| Unloaded | 1 | 0.871 | 1.092 | 1.259 | 13.444 |
| Unloaded | 2 | 0.814 | 1.055 | 1.111 | 3.296 |
| Unloaded | 3 | 0.833 | 1.074 | 1.167 | 13.797 |
| Render flood | 1 | 466.352 | 2328.407 | 2600.630 | 2712.129 |
| Render flood | 2 | 60.944 | 1008.092 | 1375.259 | 1584.815 |
| Render flood | 3 | 674.796 | 2486.611 | 2653.648 | 2752.704 |

Each row applied **1,000/1,000 RT messages**, with zero RT admission retries.
The three flood runs applied exactly **8,365,431**, **10,344,542**, and
**7,613,956** background admissions respectively; no loss. Whole-run completion
intervals were **1002.581592**, **999.205777**, **1003.082074 ms**. Raw samples
show substantial run-to-run variability; no unmeasured causal attribution or
hard-real-time deadline guarantee is inferred.

### Slow synchronous handler

One render-channel handler sets an entry flag, then sleeps until a monotonic
deadline 5 ms later. The RT producer waits for entry and submits one message.
RT attempt-to-application latencies were **5049.296**, **5069.667**, and
**5091.648 µs**. Actual handler spans were **5067.222**, **5083.388**, and
**5084.889 µs**. These are three individual observations, not a meaningful tail
percentile distribution. The first producer can be scheduled after the handler
starts; the spans need not equal request latency.

RT-first drain order is not preemption: a running synchronous handler holds the
consumer. This experiment is deliberately blocking, not representative shader
work; it falsifies isolation from an uncooperative handler.

## Verification of the new harnesses

- `make evidence-sanitize`: all four instrumented executions (both harnesses under
  ASan/UBSan/leak detection and TSan) exited zero; total wall time **70.69 s**.
  Instrumented timing is not used in the performance tables.
- `make check-sources`: warning-as-error syntax gate passed, including both new
  C files.
- `make test`: **141 headless assertions**, **6,827 P0 assertions**, and normal/
  malformed Unix/TCP CLI contracts passed. Both CLI batches returned **100/100**.
- No runtime source/header edits, new CI work, graphics initialization, physical
  input grabs, commits, or pushes were made in this evidence batch.

## Next architectural acceptance boundary

Before claiming composable participants, define and implement explicit contracts:

1. Ordered admission/application across overflow paths, or explicit rejection
   rather than silently weakening order.
2. Bounded fair service between participants as well as channels.
3. Explicit participant registration, quiescent deregistration, and generation-
   checked resource ownership; process recovery needs a separate transport model.
4. Immutable participant submissions with a commit boundary and retained last
   complete contribution, rather than independent resets of a global frame list.
5. Handler execution limits/cooperative work segmentation; hard preemption is
   not supplied by queue priority.

Those are design requirements, not implemented fixes. Queue and graphics
comparisons remain necessary after the workload and guarantee boundaries are
matched; results here do not warrant a winner claim.
