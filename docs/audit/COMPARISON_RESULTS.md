# Same-Pi comparisons: frame transport and retained GLES

## Findings

1. The optimized third-party SPSC reference has higher median completed-frame
   throughput in **all eight** static-participant configurations. The current
   NeoGPU participant implementation is not the faster transport in these tests.
2. For a pixel-identical retained two-triangle scene, direct GLES and NeoGPU's
   participant compositor/executor have close measured total frame times. The
   observed difference of aggregate medians is **2.751 microseconds / 0.599%**.
   Block timing ranges overlap: this is not a confidence bound, universal
   overhead, equivalence proof, or architectural speedup claim.
3. These results measure complete contribution frames and a particular rendering
   workload. They do **not** compare the earlier ~12M/s payload-free legacy
   message sample directly to another system, or establish superiority over bgfx
   or another graphics engine. No runtime optimization was made for this batch.

## Baselines, provenance and controls

Runtime source: commit `44090bb` on the Pi, unchanged by these experiments.
Reference: unmodified MIT-licensed `rigtorp/SPSCQueue` at
`c8e9008dc4639220cc9bcf4c84abd2ff0d344aa5`.

Downloaded dependency SHA-256s, verified before compiling:

- `include/rigtorp/SPSCQueue.h`:
  `62d9f783862666be5c7ca3c8fbcec2e45d9920167c4b9e689bda4957f66dd86e`
- `LICENSE`:
  `21d7e737765bc0c986ee78f16a3e25ee3cb5ad424feece274b64a10654c31912`

The reference and its license remain alongside the raw evidence under
`tests/evidence-comparison-final/reference/`; there is no moving-branch fetch or
modified third-party queue in the comparison. A C++ static adapter adds the
same draw-schema/sequence checks, canonical contents and retained-view copies
as the common workload. It does not implement dynamic participant lifecycles.
The native side calls the **full** current participant admission/service API,
including its mutex and ownership checks. This is a common static workload,
not equivalent implementations of all lifecycle/recovery guarantees.

Toolchain: GCC/G++ Ubuntu 15.2.0-16ubuntu1, aarch64; `-O3`, Cortex-A72 tuning,
`-march=armv8-a+simd`, `-ffast-math`, `-funroll-loops`, no LTO, same shared C
driver and indirect adapter calls. C++23 is used to share the existing frame
types through the toolchain's C atomic compatibility headers. C and C++ layout
assertions require **32-byte commands / 2,064-byte frames**. The reference
template is compiled with the same optimization flags, not a naive mutex queue.

The Pi uses kernel `7.0.0-1023-raspi`, four CPUs and `ondemand` governors.
CPU transport pins the consumer to CPU 0 and producers round-robin to CPUs 1–3;
4/8 producers deliberately oversubscribe those three producer CPUs. Graphics
uses the OS scheduler with CPUs 0–3 allowed, one current context and unchanged
affinity for both paths. No CPU online/offline, governor or frequency controls
were modified; only the stated benchmark-thread affinity was set.
Transport collection boundary frequency snapshots were 1,800,000 kHz; temperature
rose from **36.998 to 51.608 C**. They are not continuous thermal/frequency logs.
Throttle queries failed (`/dev/vcio` unavailable), so no no-throttling claim is
made. Pi UTC differs from the harness date; raw UTC is preserved and duration
timing is monotonic.

## CPU frame transport/application

Both paths have eight lane positions and **four usable queued frames per lane**,
one producer per active lane, no fallback or overwrite, and one visit per lane
per consumer step. Both validate and copy immutable input and apply into retained
consumer views. Allocation/setup is outside timing; first queue-slot touches,
input construction, validation, common per-draw correctness observation and
sampled latency instrumentation remain inside timing. Separate warmup processes
warm each configuration, not the newly allocated data of each measured process.

Each run completes **100,000 frames per producer**, including producer joins in
its completion interval. Each applied sequence and every canonical draw are
verified. Each configuration has **five alternating A/B pairs**. Across the
**80 final measured executions, all 30,000,000 accepted frames were applied**.
There were no lost/overwritten accepted frames or FIFO violations. Busy/full
attempts are counted as retries before acceptance, not hidden as delivery.

Median rates, **millions of complete frames per second**:

| Producers | Draws/frame | NeoGPU | Rigtorp adapter | NeoGPU/reference |
|---|---:|---:|---:|---:|
| 1 | 1 | 0.864 | 4.314 | 20.0% |
| 1 | 64 | 0.357 | 1.092 | 32.7% |
| 2 | 1 | 1.891 | 5.474 | 34.5% |
| 2 | 64 | 0.821 | 1.154 | 71.1% |
| 4 | 1 | 1.239 | 5.175 | 23.9% |
| 4 | 64 | 0.705 | 1.117 | 63.1% |
| 8 | 1 | 1.561 | 3.827 | 40.8% |
| 8 | 64 | 0.976 | 1.015 | 96.2% |

These are not pure queue instruction rates: both include the instrumented common
application workload. Variability is retained, not inferred away. For example,
one-producer/one-draw rates ranged **0.608–0.899M** for NeoGPU and
**3.174–4.522M** for the reference. Eight-producer/64-draw rates ranged
**0.973–0.988M** and **0.985–1.117M** respectively. All configuration ranges and
individual runs are retained in the derived report and raw JSONL.

Sampled first-attempt-to-observed-application latency, median of each run's p99,
in **microseconds** (one sample every 128th frame):

| Producers | Draws/frame | NeoGPU | Rigtorp adapter |
|---|---:|---:|---:|
| 1 | 1 | 10.500 | 1.055 |
| 1 | 64 | 4.371 | 3.907 |
| 2 | 1 | 5.463 | 2.351 |
| 2 | 64 | 14.111 | 10.166 |
| 4 | 1 | 16.611 | 8.352 |
| 4 | 64 | 24.963 | 21.204 |
| 8 | 1 | 25.314 | 14.833 |
| 8 | 64 | 46.741 | 43.870 |

These include rejection/retry delay and the common consumer's observation/check
delay after service. Admission-return timestamps are retained separately;
application is allowed to precede admission return. They are sampled saturated
workload percentiles, not hard deadlines or comparable replacements for the
earlier differently paced, unpinned RT-channel latency experiment.

## Direct GLES versus NeoGPU retained rendering

The two paths render the same red/green triangles, preloaded VBO contents,
vertex layout, shader source text, viewport, clear color, GL state and two draw
calls. Shader-text equality is checked automatically against the existing
backend source. Direct GLES hoists redundant program/attribute-enable calls and
caches EGL display/surface handles outside timing. It is not artificially made
to repeat the executor's dispatch and buffer-state checks.

NeoGPU's scene contributions are admitted/applied before measurement; its timed
path includes begin-frame, retained contribution composition and the stock GLES
executor. This does **not** measure concurrent producer-to-display latency or
uploads. Both paths share one context, EGL surface and real swap operation.
Warmup excludes initial shader/VBO/CRTC setup costs. GPU context reports:
**OpenGL ES 3.1 Mesa 26.0.8-1ubuntu0.3 / V3D 4.2.14.0**.

Every collection verified equality of all **1,536,000 RGBA framebuffer bytes**
at 800×480 outside timing, plus red/green interior pixels to reject a trivially
equal blank frame. Pre-swap captures from both paths and all three final
collections have SHA-256
`963969b77df637e7c0d05ccdaaa3a7e8d44127fda7bc794d6d69ac2dad98bb50`.
The resulting image has been visually inspected as well.

Three collections each contain five alternating pairs of **600 frames per path**:
**9,000 timed frames per path / 18,000 total**. No readbacks or capture writes
occur inside timed blocks. A test-only EGL swap wrapper records timing and calls
the real swap; the production backend is unchanged.

Median of the **15 per-block medians**, in microseconds:

| Boundary | Direct GLES | NeoGPU | Difference |
|---|---:|---:|---:|
| Begin frame → pre-swap submission boundary | 76.935 | 78.370 | +1.435 |
| Begin frame → `glFinish` completion | 416.556 | 418.510 | +1.954 |
| Wait around `glFinish` | 338.852 | 339.232 | +0.380 |
| Real EGL swap call | 41.852 | 41.778 | −0.075 |
| Whole instrumented frame | 459.463 | 462.214 | +2.751 |

Metrics' medians need not add: they are independently aggregated distributions.
Whole-frame block medians ranged **456.213–464.231 us** direct and
**458.241–469.334 us** NeoGPU. Block p99s ranged **494.759–590.722 us** direct
and **499.722–746.834 us** NeoGPU. These overlap and contain tails; a point
difference is not a hard bound or statistical significance assertion.

Completion is a CPU wall interval around explicit `glFinish`, not a hardware GPU
timer. The finish serializes GPU work before swap, so this is not an unrestricted
asynchronous throughput benchmark. EGL swap duration does not prove panel scanout
or physical refresh. No displayed-panel FPS headline is derived from it.

## Verification, artifacts and reproduction

- ASan/UBSan/leak and TSan builds of the shared CPU driver, native adapter, pinned
  reference adapter and native fabric passed self-tests and 1/8-producer,
  64-draw qualification workloads. Both capacity-four/FIFO/copy-ownership
  self-tests passed. All **10 instrumented qualification executions** exited 0.
- Instrumented timings are excluded from reported rates. Full raw qualification
  stdout/stderr, compiler commands and zero exit statuses are retained.
- All-source warning-as-error syntax checks pass for the C sources; the runner
  separately compiles the C++ adapter with warnings as errors.
- `make test` passes: **141 headless assertions, 6,827 P0 assertions**, 100/100
  Unix and TCP CLI batches, and the participant delivery/lifecycle groups.
- `make fabric-sanitize` passes again. Runtime `src/` and `include/` remain
  unchanged relative to `44090bb`; no correctness shortcut/optimization is kept.

Final collections:

- `tests/evidence-comparison-final/`: pinned reference/license, builds,
  qualifications, 80 raw measured JSONL files, summaries and manifest.
- `tests/evidence-graphics-comparison-final-{1..3}/`: each collection's frames,
  captures, shader/byte-equality checks, build commands and manifest.
- `tests/evidence-comparison-report-final.json`: recomputed results, ranges,
  analysis-script hash and collection-manifest SHA-256s. Report SHA-256:
  `e825aca3dd2fb0d2425ccfee6c284a12a8ab0f16dd827833f42c96bd958cac61`.
- Exploratory collections are retained under `tests/evidence-comparison-initial/`
  and `tests/evidence-graphics-comparison-initial/`. The latter used 120-frame
  blocks; final graphics collections increased duration before reporting final
  estimates. Final CPU source adds layout assertions, not a transport change.

Final evidence is copied locally under `neogpu-evidence/tests/` as well as
retained on the Pi. Generated artifacts are ignored, not committed or deleted.
Source/build scripts and this report are reviewable changes; CI remains deferred.

```sh
# Fresh output directory names are required; network fetches are SHA-pinned.
python3 tests/test_comparison_evidence.py --output tests/evidence-transport-NEW --sanitizers
python3 tests/test_comparison_graphics_evidence.py --output tests/evidence-graphics-NEW-1
python3 tests/test_comparison_graphics_evidence.py --output tests/evidence-graphics-NEW-2
python3 tests/test_comparison_graphics_evidence.py --output tests/evidence-graphics-NEW-3
python3 tests/test_comparison_report.py --transport tests/evidence-transport-NEW \
  --graphics tests/evidence-graphics-NEW-1 tests/evidence-graphics-NEW-2 tests/evidence-graphics-NEW-3 \
  --output tests/evidence-report-NEW.json
```

The graphics runner requires passwordless `sudo -n` for DRM access. The CPU
runner requires CPUs 0–3 and GCC/G++ 15-compatible C++23 atomic headers. No
physical input devices or authentication settings are touched. No commit/push
is performed by the comparison tools.

## What this supports next

The evidence supports investigating the participant path's CPU cost while
preserving its lifecycle, ownership and coherent-contribution tests. It does
not justify replacing that contract with an unchecked fast queue, nor claiming
an engine-wide competitive advantage from two triangles. Profile before choosing
an optimization; extend matched graphics/resource workloads before generalizing.
