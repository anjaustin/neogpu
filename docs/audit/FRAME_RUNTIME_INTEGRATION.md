# Default participant frame-runtime integration: acceptance contract

Baseline: `7d1889d`. Goal: normal `HSGpu` initialization, consumer service and
frame presentation own the qualified participant fabric, without application
code manually initializing/stepping/composing a second runtime. Legacy message
entry points remain available and keep their ordering/channel/resource contract.

## Invariants and decisions before implementation

* Producer join/submit/leave wrappers preserve bounded admission, canonical
  copies, FIFO, generation validation and drain-before-retirement.
* `hs_gpu_step` services at most one participant event per lane; process retains
  its existing bounded drain-loop behavior. End-frame services one pass even if
  the application did not call process. No implicit legacy-message drain is
  added to end-frame.
* Participants contribute renderer-owned immutable draw banks, not private
  shaders/textures or arbitrary state. Legacy global state applies to their
  draws. Resource reuse remains renderer-owned and follows retirement.
* Execute legacy commands in their original order, followed by retained
  participant draws in slot order, before a trailing FRAME_END/PRESENT suffix.
  Do not inject a clear or erase legacy commands.
* Temporarily extend the existing bounded render list for backend callbacks,
  then restore the legacy builder. Repeated end-frame must not accumulate
  participant draws. Backend frame data is valid only during callbacks; consumer
  calls are externally serialized and end-frame reentry is rejected.
* Preflight the complete command count. On overflow, preserve original command
  data/count, set overflow, skip execution/presentation and return failure from
  a checked end-frame API. Accepted participant state remains retained for a
  later valid frame. Legacy void end-frame delegates to that checked behavior.
* Init failure must be observable; legacy operation remains usable, participant
  admission fails explicitly, and shutdown never destroys uninitialized locks.
* No-participant legacy frames must avoid fabric scans/copies. Measure the
  actual idle overhead and object/init footprint; do not assume zero cost.
  A monotonic highest-joined-slot bound can avoid scanning never-used views;
  it must be published before join returns and read after queue service. It is
  reset only at quiescent init/shutdown, never during concurrent retirement.
* HSSystem channel budgets, fences, capture/replay and IPC counters retain their
  legacy-message scope. Participant frames have separate bounded service and
  are not silently added to HSCAP1 or existing wire results. A unified contract
  belongs to the general-message redesign, not this integration.

## Uncertainty and verification

Measure current/default versus frozen baseline legacy throughput, integrated
versus equivalent manual participant plumbing, and a pinned optimized SPSC
control where work matches. Keep sampler/observer cost and rendering work
separate. No integration speedup or display FPS gain is assumed.

Tests: mixed command order, zero draws, saturation, retention, stale handles,
empty/draining retirement, concurrent producer churn, repeat presentation,
overflow/recovery, init failure, backend reentry and unchanged legacy calls.
Run strict compilation, regression/CLI tests, ASan/UBSan/leak and TSan; verify
actual GLES lifecycle pixels/full-frame equivalence and inspect captures.
Preserve raw commands, source hashes, environment, exact accepted/applied work,
timings and failed attempts. CI, security settings and credentials stay untouched.

## Initial pilot and remediation

The first 100,000-frame-per-producer pilot passed regression, ASan/UBSan/leak,
TSan, all partial-init failures, and actual GLES lifecycle/full-frame equality.
Object sizes were 3,573,184 bytes baseline and 3,656,576 integrated. These are
`sizeof(HSGpu)`, not measured RSS. Idle legacy-scene medians were
1,094,002→1,100,881 scenes/s (one draw) and 129,745→131,386 (64 draws).

The active one-producer/one-draw median fell 2,349,353→2,137,642 applied
contributions/s versus manual plumbing. The implementation scanned all eight
retained views twice regardless of actual slot use and redundantly loaded the
live count before service. Those structural costs were removed using a
monotonic highest-joined-slot bound and a single guarded service call. This is
a hypothesis-driven remediation, not yet a measured causal decomposition;
fresh qualification/comparison must determine its effect. Original pilot data
remains in `tests/evidence-frame-runtime-pilot/`.

The subsequent three-way run compared the original prototype, frozen manual
baseline and refined runtime in one collection. Layout red-teaming then found
two structural false-sharing hazards, so its timings remain exploratory:

* In that runtime, readiness offset 3,656,608, callback guard 3,656,609 and
  applied counter 3,656,584 occupied the same 64-byte line. Producers repeatedly
  read readiness while the consumer updated the other fields. Readiness now has
  a separate immutable line; the fabric itself is explicitly cache-line aligned.
* ELF symbols placed benchmark configuration at `0x30300..0x30338` with its
  callback counter at `0x30318`; producer records also lacked cache-line-sized
  strides. The repaired harness isolates the callback counter and each writer's
  bookkeeping. This avoids measuring unrelated observer sharing as transport
  cost. These are measured placements, not direct ownership-transfer counts.

The inactive end-frame path is isolated from composition's larger register set,
and process retains a direct legacy step call instead of an extra public wrapper
call. A same-work baseline/control/candidate repeat is required after these
changes. Exact earlier harnesses are preserved by their recorded source hashes;
new collections snapshot their complete common harness and toolchain as well.

## Final qualification and measured integration cost

Final evidence: `tests/evidence-frame-runtime-final/`,
`tests/evidence-frame-runtime-final-gates/` and
`tests/evidence-frame-runtime-graphics-final/`. The collector completed 126 timed
runs with 500,000 contributions per producer (207,000,000 accepted/applied units
across legacy and participant cases). Three alternating-order pairs per cell
compare frozen `7d1889d` manual plumbing, the original integration prototype
(`control`), and the current integrated runtime. All use the same repaired
private-counter harness and compiler-optimized command copies. The pinned
Rigtorp SPSC reference is also measured, not a naive queue baseline.

Active scenes now contain CLEAR plus retained draws, matching normal global
frame use; earlier exploratory scenes used explicit frame markers. **Do not
compare absolute active rates across those harness revisions.** The final
collector snapshots runtime headers/sources, common C/C++ harness, driver,
toolchain, command output, CPU affinity, frequency and thermal observations.

Median rates, rounded to whole units:

| Path | Producers | Draws/contribution | Baseline | Prototype | Current |
|---|---:|---:|---:|---:|---:|
| Legacy scenes/s | 1 | 1 | 1,129,231 | 1,075,401 | 1,111,387 |
| Legacy scenes/s | 1 | 64 | 131,193 | 130,037 | 130,020 |
| Native applied contributions/s | 1 | 1 | 2,887,235 | 2,259,027 | 2,754,246 |
| Native applied contributions/s | 1 | 64 | 801,176 | 780,939 | 811,952 |
| Native applied contributions/s | 2 | 1 | 4,072,327 | 3,805,239 | 4,115,142 |
| Native applied contributions/s | 2 | 64 | 820,063 | 813,793 | 800,439 |
| Native applied contributions/s | 8 | 1 | 3,099,585 | 3,094,731 | 3,123,028 |
| Native applied contributions/s | 8 | 64 | 745,172 | 715,651 | 758,880 |
| Static SPSC applied contributions/s | 1 | 1 | 2,476,615 | 2,426,437 | 2,399,189 |
| Static SPSC applied contributions/s | 1 | 64 | 814,348 | 724,398 | 792,612 |
| Static SPSC applied contributions/s | 2 | 1 | 4,166,326 | 4,085,796 | 4,062,430 |
| Static SPSC applied contributions/s | 2 | 64 | 764,623 | 779,500 | 768,841 |
| Static SPSC applied contributions/s | 8 | 1 | 2,929,035 | 2,890,631 | 2,913,763 |
| Static SPSC applied contributions/s | 8 | 64 | 747,265 | 704,545 | 740,403 |

Interpretation and limits:

* This is a lifecycle integration, **not a universal throughput improvement**.
  Current native 1p/1d is below manual plumbing in every pair (ratios
  0.9185/0.9539/0.9673), but above the original prototype in every pair
  (1.1499/1.2192/1.2389). The combined remediation improves this measured case;
  individual edits have not been isolated into additive causal budgets.
* Native 2p/1d and 8p/1d exceed manual plumbing in all three pairs. The 64-draw
  cells vary: 8p/64d ratios span 0.9327–1.0467. A higher median there is not a
  stable speedup claim. Static SPSC and native paths have different lifecycle
  guarantees; the reference is diagnostic, not a drop-in replacement.
* Legacy medians are lower by 1.58% (one draw) and 0.89% (64 draws); paired
  ratios span 0.9715–1.0002 and 0.9737–1.0019 respectively. Do not describe the
  default integration as zero-cost or assert a statistically established idle
  non-regression from these three pairs.
* Final `sizeof(HSGpu)` is 3,656,704 versus baseline 3,573,184 bytes:
  **83,520 bytes added**. Readiness/fabric/consumer-guard offsets are
  3,573,184 / 3,573,248 / 3,656,672. Final baseline/current fabric addresses are
  64-byte aligned; the original prototype remains at offset 8 modulo 64.
  Existing HSGpu storage already requires 64-byte alignment. These are object
  layout measurements, not RSS or total allocator costs.
* Current legacy initialization medians are 6,759,222 ns (one-draw cell) and
  6,499,019 ns (64-draw cell), versus baseline 6,402,815 / 6,354,556 ns. They
  measure per-process runtime/mock setup, not graphics startup. Native baseline
  initialization additionally includes its external fabric initialization.
* `report.json` contains every cell's rate range, paired ratios, retry ratios,
  callback ratios and initialization median. For native 1p/1d, baseline/current
  retry medians are 0.0138/0.0254 per applied contribution and callback medians
  1.0558/1.0780 per contribution. Retention legitimately permits extra callback
  dispatches; every actual dispatch and every applied sequence is checked.
* Every rendered command is validated by a mock backend. These rates include
  that observer and are **not GPU or display FPS**. No new instruction/cycle
  attribution is claimed. More repetitions would be required for stronger
  statistical conclusions.

### Correctness and visible outcome

Strict benchmark compilation, source-preservation checks, default regression,
ASan/UBSan with leak detection, and TSan passed. Baseline, prototype, current
and reference benchmark modes also passed separate sanitizer qualification.
All eight intentional partial-init faults preserve legacy operation and clean
up only initialized locks. Tests cover overflow/no partial callbacks, builder
restoration, terminal suffixes, retained/empty contributions, explicit legacy
draining, idle reentry, sparse high slots, concurrent churn and deterministic
rejoin between retirement and live-count decrement.

Normal HSGpu APIs passed actual GLES lifecycle checks: 18 presentations and 36
pixel checks. Full-frame RGBA matched direct GLES. The final capture contact
sheet was visually inspected: red/green, red-only, restored red/green,
green-only, retained green-only, then blue/green after generation reuse. Direct
and integrated final scenes visibly match. Graphics timings remain diagnostic;
no display refresh improvement is claimed.

The report reducer passed ten synthetic positive/negative controls on host and
Pi, including missing raw fields, raw-summary disagreement, incomplete/failed
runs, invalid counts/timing, duplicate pairs and assertion-disabled execution.
Standalone fabric and general-message source files remain unchanged.

### Reproduction and archive

Run on the same four-core Pi with the preserved prototype and reference:

```sh
make test gpu-participants-sanitize
python3 -E tests/test_gpu_frame_evidence.py --output tests/evidence-frame-runtime-repeat --frames 500000 --control tests/evidence-frame-runtime-pilot/candidate --sanitizers
python3 -E tests/test_gpu_frame_report.py tests/evidence-frame-runtime-repeat --output tests/evidence-frame-runtime-repeat/report.json
python3 -E tests/test_gpu_participants_graphics.py --output tests/evidence-frame-runtime-graphics-repeat
python3 -E tests/test_gpu_frame_report_checks.py
```

Fresh output directories are required. Reference source/license and their
hashes remain in `tests/evidence-comparison-final/reference/`; graphics requires
the established sudo/GLES device access. Wall-clock timestamps are preserved
literally; elapsed timings use the monotonic clock.

All exploratory, failed and final runtime collections were archived locally
using root-capable tar (including graphics artifacts) in
`tests/evidence-frame-runtime-archive.tar.gz`, 25,545,710 bytes, SHA-256
`e34782964a429f29b631baf3d83158eda11448b6cdfff3a368a62ffd4deacc7d`.
Historical harness supplements recovered by matching the original recorded
hashes are retained separately; prior measurements were not rewritten.
The generated visual contact sheet and this results document are later
supplements, not contents claimed for that archive. Changes remain uncommitted;
CI and the previously reported push-credential limitation remain unchanged.

A subsequent [second red-team pass](FRAME_RUNTIME_REDTEAM.md) strengthens the
automated invocation/snapshot validation and adds independent near-capacity
composition and concurrent end-frame tests. Its reducer reproduces the original
numeric results without rewriting this timing collection.

## Accepted default-path checkpoint

The user accepted the measured performance/storage tradeoffs and approved the
integration commit. Runtime initialization, bounded service and composition are
unconditional parts of the normal HSGpu lifecycle; no feature flag or external
fabric plumbing is required. Existing applications retain their message APIs.
The normal `make test`, `make sanitize`, `make tsan` and `make test-hardware`
entry points include the new participant checks (with fresh graphics evidence).
Dedicated participant targets remain available for focused diagnosis. Evidence
collections stay outside the source commit; CI and push permissions are not
changed by this acceptance.

Standard-entry-point acceptance verification: `make`, `make check-sources`,
`make test`, `make sanitize` and `make tsan` passed. The unprivileged hardware
attempt stopped on an existing root-owned `tests/p0-clear.ppm` capture, not a
rendering assertion. Previous p0 captures were preserved before an elevated
`sudo -n make test-hardware` rerun; no permissions were changed. Its result is
recorded in the acceptance evidence rather than silently replacing the failed
attempt. The elevated standard hardware target passed all eight legacy graphics
tests, 13 real evdev/uinput checks, and the participant lifecycle checks (18
presentations / 36 pixel checks) plus equality of 1,536,000 RGBA bytes against
direct GLES. The participant evidence collection created by the standard target
is `tests/evidence-frame-runtime-hardware-1794247621-193418/`; acceptance logs
are in `tests/evidence-frame-runtime-acceptance/`.
