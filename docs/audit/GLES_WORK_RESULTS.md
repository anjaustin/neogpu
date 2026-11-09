# GLES setup and default-capture profiling results

## Outcome

* **Promoted repair:** release the selected DRM connector and unbind/terminate/
  release EGL during successful graphics shutdown. The unchanged baseline's
  first real-GLES LeakSanitizer qualification reported **11,732 bytes in 65
  allocations**. All fourteen scene/variant ASan/UBSan/leak qualifications passed
  with the repair; leak checking was not disabled or suppressed.
* **Not promoted:** two frame-local GL-state cache candidates. They reduce CPU
  work in dense clean draw runs, but do not satisfy the no-unaccepted-workload-
  regression gate. The second candidate's single-draw/capture-off completed-frame
  interval regressed in all five pairs. Production `src/hs_backend_gles.c`
  remains unchanged; no draw coalescing, reordering or state caching is installed.
* **Default capture is unchanged.** Its measured cost varies by scene. The
  mixed triangle/texture workload shows a repeatable cost; capture is not shown
  to be the dominant cost everywhere. Disabling it silently is not a fix.

Contract and intermediate findings: [GLES_WORK_PLAN.md](GLES_WORK_PLAN.md).

## Method and provenance

Frozen source is `2bf623a5f40060dbd345744fb0702306f6896cdc`. Both variants in
the final comparisons receive the **same teardown repair**. Thus the baseline
is explicitly `2bf623a + common teardown repair`; only the diagnostic candidate
changes draw-state setup. The original header and all compared headers/sources
are snapshotted. Builds use the same optimized Cortex-A72 compiler flags, not a
naive C baseline or a substituted renderer.

Seven specified workload probes cover the capture demo's single triangle,
64 repeated same-bank draws, 64 alternating-bank draws, two dirty/resized-bank
draws, four triangle/texture/triangle groups, two fallback draws, and two retained
participants contributing 32 draws each. They do not cover every product app,
shader, resource-upload pattern or display configuration.

The corrected pilot completed 84 timed runs. Each of the two final collections
completed **140 runs**, five alternating-order pairs per scene/capture cell,
384 frames per run: **53,760 timed frames per collection**. Each final
collection issued 1,167,360 legacy messages, plus retained participant draws.
Recording stays in its requested state throughout every block and capture-on
journals have exact expected messages, draw-bank indices and owned CLEAR payloads.
No journal overflow or silent recording shutoff is accepted.

GL call counting and framebuffer reads occur in separate audit binaries, not
timed binaries. Timing retains every frame's submit/swap/return intervals, a
single separately reported end-block `glFinish`, and aggregate process CPU time
(all process threads). There is no per-frame forced `glFinish` in these timing
runs. None of these measurements is physical display FPS or GPU-cycle profiling.
Source/binary hashes, invocation arguments and raw-summary agreement are checked
by `tests/test_gles_work_report.py`; twelve synthetic controls exercise failures.

Rendering has 16 warmup frames. **Longer capture blocks reach new journal slots**,
so capture-on includes journal page first-touch, not only fully resident copy
cost. Log reset is outside each measured block. First rendered-frame timing is
recorded separately, but includes geometry upload and driver first-use work:
it is not an isolated measurement of journal allocation. No overhead subtraction
or additive causal partition is claimed.

## Default capture cost

The following are baseline results from `tests/evidence-gles-work-final-v4/`.
Submit columns are medians of per-run frame medians; CPU columns are medians of
per-run process CPU time divided by frame count. Units are **microseconds/frame**.

| Scene | Submit, capture off | Submit, capture on | Process CPU, off | Process CPU, on |
|---|---:|---:|---:|---:|
| Single triangle | 62.102 | 61.556 | 165.228 | 171.085 |
| Same-bank 64 draws | 223.888 | 222.555 | 384.855 | 385.903 |
| Alternating-bank 64 draws | 393.111 | 404.370 | 530.681 | 547.583 |
| Dirty/resized banks | 96.333 | 97.084 | 208.267 | 208.997 |
| Mixed triangles/textures | 274.158 | 290.278 | 434.991 | 445.320 |
| Fallback geometry | 69.750 | 70.037 | 183.568 | 188.569 |
| Retained contributions | 218.750 | 224.305 | 379.793 | 381.367 |

For mixed triangles/textures, capture-on/off paired submit ratios are
1.0675/1.0479/1.0647/1.0588/1.0680; paired CPU ratios are
1.0331/1.0143/1.0380/1.0154/1.0254. Both increase in every pair. Other cells
include mixed signs or smaller changes: do not turn this into a universal
capture-off speedup. The preceding final collection also found every-pair
submit/CPU increases in the mixed workload.

Existing compiled layout output measures `sizeof(Payload)=128` and a lazy journal
capacity request of **8,388,608 bytes** with the default log capacity. This is
allocation capacity, not measured resident memory or 8 MiB touched at startup.
First-frame medians in this final collection span roughly 30–40 ms across the
specified scenes/modes. No consistent isolated journal-allocation latency is
established by those cold-frame observations.

## Cache experiments and rejection

Both candidates reset state at each execute call and invalidate it after every
non-draw command. They leave draw count/order, dirty uploads and texture behavior
unchanged. The second additionally retains the original setup call order after
uploads and caches only clean draw runs.

The second candidate's capture-off medians illustrate why CPU savings alone are
not sufficient. Submit is a median frame interval; CPU and completion-inclusive
columns are per-block means, then median across runs. Units: microseconds/frame.

| Scene | Submit baseline → candidate | CPU baseline → candidate | Completion-inclusive baseline → candidate |
|---|---:|---:|---:|
| Single triangle | 62.102 → 61.462 | 165.228 → 169.156 | 241.611 → 246.949 |
| Same-bank 64 draws | 223.888 → 163.638 | 384.855 → 328.953 | 1125.130 → 1126.965 |
| Alternating-bank 64 draws | 393.111 → 367.777 | 530.681 → 516.820 | 1149.564 → 1152.545 |
| Dirty/resized banks | 96.333 → 95.278 | 208.267 → 206.517 | 265.297 → 266.644 |
| Mixed triangles/textures | 274.158 → 274.611 | 434.991 → 433.061 | 1134.906 → 1133.373 |
| Fallback geometry | 69.750 → 66.176 | 183.568 → 179.060 | 283.437 → 279.799 |
| Retained contributions | 218.750 → 157.083 | 379.793 → 328.010 | 1118.740 → 1120.059 |

The single-triangle/capture-off completion-inclusive ratios are
1.0451/1.0178/1.0113/1.0326/1.0728. Four of five process-CPU ratios also exceed
one. Dense clean-draw submission and CPU time improve in all pairs, but most
completion-inclusive intervals do not. Even the large submit improvements must
not be advertised as GPU/display acceleration.

The first candidate, retained in `tests/evidence-gles-work-final-v3/`, regressed
dirty/capture-off CPU time in four of five pairs (ratios
1.0381/1.0309/1.0193/1.0227/0.9999). Its replacement does not erase those data.
Neither candidate is accepted by selectively excluding adverse workloads.
No per-edit cache-cost attribution or statistically universal multiplier is
claimed; timing/environment interactions remain unresolved.

Separate GL-call audits confirm genuine eliminated calls, not omitted draws.
For three same-bank frames, setup calls (program/bind/enable/pointer) fall from
192/192/384/384 to 3/3/6/6, while all 192 draws remain. Dirty uploads remain six
over three audit frames; the narrowed cache retains original setup there.
These heterogeneous GL calls are not assigned equal time cost.

## Red-team and verification

* Audit frames include external program/buffer/attribute perturbation, dirty and
  resized buffers, fallback buffers, triangle/texture transitions and a fourth
  **no-CLEAR** frame that cannot accidentally mask cross-frame cache persistence.
* Full RGBA bytes match across baseline/candidate and capture-off/on for each
  scene/audit frame; expected red/green scene pixels are independently checked.
* Three deliberately unsafe caches compiled and linked, then were rejected by
  actual GLES: persistent cache state fails no-clear perturbation, missing
  texture invalidation changes RGBA, and missing buffer transition setup fails
  expected pixels. Compiler failure is not counted as mutation rejection.
* A headless observer of the actual backend source checks program, bound buffer,
  enabled attributes, pointer formats/strides and dirty-upload handling across
  15 draws in three frames. It does not claim to render pixels or measure speed.
* Both final collections passed fourteen separate actual-GLES ASan/UBSan/leak
  qualifications. The initial 11,732-byte leak failure is preserved, not reclassified
  as a pass or hidden by sanitizer suppression.
* Default `make test`, `make sanitize` and `make tsan` now include the headless
  state-transition checks; default regression includes report-integrity controls.
  Performance collection and driver-dependent mutation tests remain explicit.

Production `make check-sources`, `make test`, `make sanitize`, `make tsan`, and
`sudo -n make test-hardware` passed with the repair. The standard hardware target
passed all eight legacy graphics tests, 13 real evdev/uinput checks, participant
lifecycle verification (18 presentations / 36 pixel checks) and equality of
1,536,000 RGBA bytes against direct GLES. Its fresh participant collection is
`tests/evidence-frame-runtime-hardware-1794251603-203035/`.

The final seven-scene no-clear capture contact sheet was visually inspected:
red-only scenes, red/green alternating and retained scenes, the dirty-color
variant, checkerboard/green mixed scene and fallback gradient triangle all match
their baseline/candidate counterparts. Full RGBA comparison covers capture-off
as well. Both final reports were independently re-reduced with identical numeric
results; the reviewed reports add the explicit journal first-touch caveat.

The narrowed candidate's three unsafe cache mutations were also rerun and
rejected. The default headless test observes the unchanged backend doing all
15 program selections / 30 attribute-pointer setups over its 15 test draws;
this confirms that no rejected cache was accidentally installed. Production
`src/hs_backend_gles.c`, `src/hs_gpu.c`, `src/hs_fabric.c` and `src/hs_core.c`
remain byte-identical to HEAD. Only the graphics ownership repair, test wiring,
new diagnostics/regressions and documentation are new working-tree changes.

## Preserved evidence

All failed attempts, pilot, both complete final collections, actual driver
sanitizer output, raw timing/counter/pixel artifacts, unsafe mutations, production
gates and current relevant source files are preserved locally in
`tests/evidence-gles-work-archive.tar.gz`: **45,651,558 bytes**, SHA-256
`f409a5665289cc9e1dd33cdb8f53b1696a0647f22f7b8cfb45fe55c019797b5e`.
The root-capable archive also includes fresh standard hardware evidence and p0
captures. Final results documentation and generated contact sheets are later
supplements, not claimed contents of that archive. No failed measurement was
rewritten or silently substituted. Those experiments preceded the subsequently
approved source commit and direct comparison below.

## Reproduction

```sh
python3 -E tests/test_gles_work_evidence.py --output tests/evidence-gles-repeat --baseline 2bf623a --frames 384 --pairs 5 --sanitizers
python3 -E tests/test_gles_work_report.py tests/evidence-gles-repeat --output tests/evidence-gles-repeat/report.json
python3 -E tests/test_gles_work_mutations.py tests/evidence-gles-repeat --output tests/evidence-gles-mutations-repeat
python3 -E tests/test_gles_work_report_checks.py
make test sanitize tsan
sudo -n make test-hardware
```

The current collector reproduces the **narrowed rejected candidate**, not an
installed production cache. Each historical collection preserves its own driver
and common harness, including the first candidate. Fresh output directories and
the established GLES/device sudo access are required. Machine wall-clock dates
are retained literally; elapsed intervals use monotonic/process CPU clocks.
CI, credentials and the existing workflow-scope push blocker are unchanged.

## Direct committed-source comparison

After user approval to commit and benchmark, the harness also records runtime/
graphics initialization and complete shutdown wall/process-CPU intervals and
whether the current EGL context was released. A direct comparison uses:

```sh
python3 -E tests/test_gles_work_evidence.py --production --baseline HEAD^ --candidate HEAD --output tests/evidence-gles-committed --frames 384 --pairs 5
python3 -E tests/test_gles_work_report.py tests/evidence-gles-committed --output tests/evidence-gles-committed/report.json
```

Unlike the rejected-cache experiments above, `--production` builds exact source
from both revisions: **no cache and no common header repair are injected**.
The known-leaking parent is a performance/lifecycle control, not a newly claimed
leak-clean sanitizer baseline. Candidate sanitizer qualification is recorded
separately. Old experimental evidence remains unchanged; new lifecycle metrics
do not retroactively appear in it. No steady-state speedup is assumed for a
shutdown ownership repair.
