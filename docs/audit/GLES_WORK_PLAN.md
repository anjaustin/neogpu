# GLES state setup and capture cost: measurement contract

Baseline: committed `2bf623a`. Success means reducing actual renderer CPU work
on equivalent valid scenes without changing pixels, command order, capture
contents/defaults, dirty-buffer uploads or externally visible state boundaries.
No speedup, energy saving or display-FPS gain is assumed.

## Scope and invariants

* Profile legacy message CLEAR/DRAW scenes resembling the capture demo, dense
  repeated/alternating-bank draws, dirty uploads, texture/draw transitions,
  fallback geometry, and normal retained-participant rendering separately.
  These are specified workload probes, not a claim to cover every application.
* Compare capture on/off using the existing API. Keep recording defaults and
  owned journal semantics unchanged. Bound each block below journal capacity;
  report cold first-use allocation separately from warmed steady-state timing.
* Diagnostic candidate only: cache draw program/attribute/buffer setup within
  one backend execute call. Reset on every non-draw command and every frame.
  Never assume ownership of GL state across externally accessible frame calls.
  Dirty upload helpers can bind buffers; their effect must remain correct.
* Do not reorder, merge or omit draws. Do not change texture semantics,
  presentation, synchronization, queues or resource ownership in this experiment.
* Preserve the unchanged compiler-optimized baseline. Count actual GL calls in
  separate instrumented qualification binaries, not the accepted timed runs.
  No naive-copy baseline or CPU arithmetic substitute is involved.

## Uncertainty and acceptance evidence

The driver may already reject redundant state cheaply; extra cache branches can
regress small/alternating workloads. GPU/swap pacing may hide CPU savings. The
capture journal can fill and disable itself, so all timed frames must retain the
requested recording state and exact message count. A per-frame journal reset
would be a different workload; use bounded per-block logs instead.

Require frame RGBA equality against the frozen backend, expected scene pixels,
dirty/fallback/pipeline-transition tests, external GL-state perturbation between
frames, exact draw/upload counts, capture count/payload verification, and GLES
error checks. Timing uses repeated alternating-order blocks after warmup; retain
per-frame submit/swap intervals, end-block completion waits, cold-frame timing,
source snapshots/hashes, commands, compiler and environment metadata. Keep
forced-completion diagnostic timing distinct from normal pipelined presentation.

Qualify under ASan/UBSan and regressions before promotion; distinguish driver
leaks/unsupported sanitizer instrumentation from runtime findings. Retain rejected
candidates and failed attempts. Promote only demonstrated improvements with no
unaccepted workload regression; otherwise leave production unchanged. No new
commit/push, CI, credentials, driver or frequency changes are part of this task.

## Qualification finding before final timing

The initial strict-build attempt rejected an audit-only unused helper; no timing
was accepted from it. The corrected pilot completed 84 timed blocks and matching
pixels/counts, with process CPU time added to the common harness. A stronger
audit now includes a no-clear frame after external GL state perturbation.

LeakSanitizer then stopped the first final collection on the frozen backend,
before any timed blocks: graphics initialization retains its selected DRM
connector, and finish destroys the surface/context without unbinding or
terminating/releasing EGL. A common diagnostic repair frees the connector and
unbinds, terminates and releases EGL on successful shutdown. Both compared
variants receive identical repair bytes; the original header is preserved.
Thus the new baseline is explicitly `2bf623a + common teardown repair`, not
misrepresented as unchanged committed source. Leak detection remains enabled;
the failed collection is retained. No cache optimization has been promoted.

The source transformation initially rejected a whitespace mismatch before
building the repaired header; that failed attempt is also preserved. Exact
transformations now have a local preflight. Three deliberate unsafe caches were
then rejected on actual GLES: persistence across calls failed the no-clear
perturbation, omitted texture-transition invalidation produced different RGBA,
and omitted buffer-transition setup failed expected pixels. Build failures do
not count as successful mutation rejection; each mutant compiled and linked.

The earlier standard regression output also measures `sizeof(Payload)=128` and
the lazy journal request as 8,388,608 bytes at full configured log capacity. That
is requested allocation capacity, not measured resident memory. Its default
enablement remains unchanged; disabling capture is a measured comparison, not
a silently installed product policy.

The completed five-pair broad-cache run passed all 14 actual GLES
ASan/UBSan/leak qualifications after the common teardown repair. Its dense clean
draw CPU results improved, but the dirty/capture-off workload regressed in four
of five pairs. It is not promoted. A second, narrower diagnostic retains the
original setup order after uploads and caches only clean draw runs. No further
candidate will be accepted by simply ignoring that workload.

Warmup caveat discovered during review: the 16 render warmup frames initialize
GL resources, but a longer recording block writes additional journal slots.
Capture-on timings therefore include journal page first-touch as well as normal
bookkeeping; they are not isolated fully resident journal-copy costs. This
matches a bounded growing capture, and is reported explicitly rather than
subtracting or estimating the residency effect.

Final disposition and full raw-concordant numbers are recorded in
[GLES_WORK_RESULTS.md](GLES_WORK_RESULTS.md). The narrowed cache also failed the
workload gate and was not promoted. Only the independently qualified graphics
ownership repair is installed; standard regression, sanitizer and hardware
entry points passed, including the new headless GL-state observer. No capture
default or production draw-setup change was retained.

Subsequent user approval covers committing the repair/tests/documentation and
benchmarking the committed default path against its parent. The direct
`--production` comparison does not inject either rejected cache or a shared
repair into the parent; initialization, shutdown and EGL-release observations
are now separate metrics. This supersedes the original no-commit diagnostic
scope, not the no-unaccepted-regression gate.
