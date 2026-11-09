# Same-Pi comparison contract

Baseline runtime commit: `44090bb`. No runtime optimization precedes these
experiments. Success is reproducible equivalent-work evidence, including losing
results; no superiority is assumed. Existing delivery/ownership tests must pass.

## CPU frame transport/application

Reference: unmodified MIT-licensed `rigtorp/SPSCQueue`, pinned to
`c8e9008dc4639220cc9bcf4c84abd2ff0d344aa5`. Fetch only the header and license,
retain their SHA-256s and copies under tests/evidence-*; never use a moving branch.

Both adapters use the actual HSParticipantFrame/HSRenderCmd layouts, eight lane
positions, four usable queued frames per lane, one producer per active lane,
identical schema/sequence checks and canonical draw contents, copied ownership,
FIFO application into retained views, and one visit per lane per consumer step.
No fallback, overwrite, GPU, recording, composition, or dynamic lifecycle is in
this timed workload. Rigtorp supplies optimized SPSC transport, not participant
lifecycles: this is its static-participant adapter versus the **full** NeoGPU
participant admission/service path, not equivalent implementations of recovery.

Vary active producers 1/2/4/8 and frame sizes 1/64 draws. Time fixed completed
work, not admissions: 100,000 frames per producer, including input construction,
validation, retained copies, common correctness observation, sampled latency,
stop/join. Check every applied sequence and draw. Sample first-attempt-to-apply
latency every 128th frame, including admission retry delay. The consumer timestamp
is taken after checking that frame in the common observer, not inside either
queue implementation; include that observation delay rather than guessing it
away. Retain raw samples.

Use identical shared driver/indirect adapter calls, -O3 Cortex-A72 flags, no LTO,
GCC/G++ from the same toolchain, and explicit consumer CPU 0 / producers CPUs
1–3 (round-robin, oversubscribing at 4/8 producers). Warm each configuration,
then collect five alternating A/B pairs. Record all samples, medians and ratios
of medians, environment, source hashes, elapsed intervals and rejected attempts.
Verification overhead remains included and is disclosed, not estimated away.

## Graphics execution

Compare equivalent two-triangle scenes through (a) direct GLES and (b) retained
participant composition plus the existing GLES executor. Match framebuffer,
shader sources, vertex layout/colors, VBO contents, GL state and draw count. The
direct path hoists redundant program/attribute-enable calls out of the draw loop;
it is not deliberately saddled with the executor's per-command dispatch costs.
Cache the current EGL display/surface outside timing. Use one
context and swap path, preload/warm resources, and alternate order in each pair.
The initial 120-frame blocks qualify correctness; final measurements use
600-frame blocks, five pairs per collection, and three separate collections.
Retain the initial collection rather than silently replacing it with longer runs.
The direct path is a real graphics baseline, not an empty loop or CPU-only mock.

Verify full-frame RGBA equality outside timed runs and retain captures. For each
path record CPU prepare/submit duration up to the pre-swap boundary, explicit
glFinish completion interval, actual swap/presentation interval, and total
display-path duration. GPU-completion timing is a **CPU wall interval around
glFinish**, not a hardware GPU timer. Do not label presentation rate physical
panel refresh. Exclude readbacks from timing. No independent producer traffic or
resource updates occur in this retained-scene graphics experiment; CPU queue
results are not implied by it.

## Verification and reporting

Keep runtime source unchanged. Test adapters/driver with ASan/UBSan/leaks and
TSan before accepting rates; instrumented numbers are not performance evidence.
Retain source revisions/hashes, exact compiler commands, affinity, kernel,
frequency/governor/thermal snapshots, throttle-query failures, raw JSONL, and
exit statuses. Use monotonic timestamps, report variability and limitations, and
do not infer novelty or general engine superiority from these two workloads.
