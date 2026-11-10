# Committed-path reference refresh

Production revision: `c9307e16666b7c29a24c28bd2d9136a87ac41d71` (runtime
identical to `3aaab60`). Shutdown diagnostics are committed; rejected candidates
are not installed. Reference queue remains unmodified, SHA-pinned
`rigtorp/SPSCQueue` revision `c8e9008dc4639220cc9bcf4c84abd2ff0d344aa5`.

## Measurement contract, before execution

Success is a reproducible characterization, not a required speedup. Preserve
accepted/applied counts, per-lane FIFO, immutable copied input, ordered draws,
actual nonblank RGBA equality, capture defaults, and complete cleanup. Do not
change governors, driver settings, credentials or security settings. Keep all
failed/incomplete collections. Run performance stages serially, not competing
with each other. Instrumented qualifications are not throughput samples.

Collect optimized transport (1/2/4/8 producers, 1/64 draws, seven alternating
pairs), actual default HSGpu frame plumbing versus frozen `7d1889d` manual
plumbing and optimized reference (1/2/8 producers), retained direct GLES
(three collections), and seven-scene committed GLES versus `2bf623a` with
capture off/on (seven pairs, 512 frames). Repeat headless, ASan/UBSan/leak,
hardware, and actual-GLES sanitizer checks separately. Record startup, cold
frame, warm submission/presentation/completion, journal cost, shutdown wall and
CPU where supported, whole-process resource usage, retries, latency samples,
source provenance and thermal/frequency observations. Sampling boundaries are
not continuous hardware telemetry and cannot establish absence of throttling.

Unknowns: current repeatability/tails, remaining static-reference gap, retained
executor overhead, and producer-to-completed-GLES behavior. Existing CPU frame
tests use a validating mock backend; existing direct-GLES tests pre-admit
contributions and use manual composition. Neither is a concurrent
producer-to-display benchmark. Add a bounded actual-GLES integration diagnostic
to cover that gap, validate its reference equivalence before accepting timings,
and distinguish GPU completion via `glFinish` from physical panel scanout.

The static SPSC reference does not implement NeoGPU's dynamic lifecycle and
recovery contract. Direct GLES is a matched small scene, not a graphics engine.
Do not combine these workload units into one headline speedup or claim an
engine-wide comparison. All new measurements remain uncommitted until reviewed.

### Preserved harness failures

The first orchestration attempt completed transport/default-frame/direct-GLES
collections, then failed the comparison reducer's output-name guard. A resume
retained that failed command and reused successful recorded invocations. A
second review found wrapper command logs colliding with two derived report
filenames; those reports were recomputed from unchanged raw collections under
distinct `evidence-*.json` names. The wrapper is corrected for fresh runs.

The first actual-GLES qualification (`evidence-e2e-c9307e1`) failed its FIFO-to-
render assertion before collecting timed samples. The harness incorrectly called
`hs_gpu_process`, which intentionally drains multiple participant updates, before
end-frame's bounded service. The corrected harness pumps only the CLEAR message
queue with `hs_step(&gpu.system)`; automatic checked end-frame remains the sole
native participant service/composition boundary. The original early assertion
exit also skipped EGL cleanup and produced a leak report; preserve it, do not
attribute it to normal production shutdown or suppress the clean-rerun gate.

The initial direct-GLES collections used the normal up-to-date Make target.
Preserve them as repeatability data, but perform an explicit `make -B` rebuild,
record the resulting graphics binary hash, and use three fresh collections as
the authoritative retained-render comparison. This removes dependence on
incremental object provenance rather than assuming it from Make's exit status.

## Actual-GLES diagnostic boundary

`tests/test_gpu_gles_e2e.c` reuses the existing producer/schema/FIFO observer
and pinned reference adapter. The native path calls the stock checked end-frame
API, including automatic service/composition. The reference path performs
equivalent external static service/composition, then invokes the same actual
GLES executor. Both issue one actual CLEAR message per render frame and use
identical pre-uploaded vertex data and immutable per-lane draws. Both verify
every composed command, every applied sequence, and every completed contribution.
One final readback per process is outside timing and checks a nonblank interior
pixel for each lane; all bytes must match across paths/capture modes/repetitions.

Run 1/2/8 producers, 1/64 draws per contribution, capture off/on, five alternating
pairs with 256 contributions per producer. Before timing, qualify sparse and
dense paths with actual-GLES ASan/UBSan/leak runs. Consumer is pinned to CPU 0;
producers use CPUs 1–3, so eight deliberately oversubscribe.
Driver-thread affinity is recorded after warmup; context creation follows
consumer pinning, so this is not the scheduler-unrestricted direct-GLES test.
Per-contribution clock/cache observer traffic and ordered-command validation are included for
both paths. Each measured render frame ends with `glFinish`; first-attempt
latency includes retry/queue delay, service, composition and GPU completion.
Admission-return timestamps are read only after producer join because return
may follow consumption. Geometry is stable rather than visibly sequence-coded;
sequence integrity is checked on the actual consumed retained views.

Report contribution rate, rendered-frame counts, retries, admission and completed
latency p50/p95/p99/max, frame submission/completion, process CPU, startup/cold
frame and complete teardown. GNU time records whole-process CPU, peak RSS,
context switches and faults separately; these include sudo, verification/output
and cleanup, and peak RSS is not allocated GPU memory. Independent reducer
checks frozen hashes, raw concordance and exact invocation. Synthetic validator
controls are explicitly labeled and never included as hardware evidence.

Separate whole-process `perf stat` and syscall-summary probes may identify
remaining costs, but include startup/verification/output/cleanup and must not be
mixed with uninstrumented benchmark rates. Preserve unsupported/forbidden
counter errors; do not change permissions or security settings to obtain them.
