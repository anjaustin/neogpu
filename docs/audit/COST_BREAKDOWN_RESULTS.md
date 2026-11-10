# Cost Breakdown: Where NeoGPU Spends Time and Compute

Status: diagnostic and uncommitted. This document measures the committed
default path; it does not change production code. Every number here is either
(a) a raw sample from a committed-path run, (b) an independent reduction of
those samples, or (c) explicitly marked as perturbed by tracing/preloading.

## 1. The question and the answer in one paragraph

For a long-running graphics process the dominant cost is **waiting on the
GPU/driver pipeline**, not application code: on the committed E2E path at one
draw the frame is ~346 µs, of which ~294 µs (85%) is wait; at 64 draws it is
~773 µs, of which ~652 µs (84%) is wait. The E2E path finishes every frame for
verification; the production swap-only path is measured separately and is not
more expensive per frame. The waiting is **not free CPU** — the calling
thread busy-polls for roughly one third to one half of it, and kernel/driver
code accounts for 51–77% of sampled cycles. For a short-lived process the
fixed lifecycle is comparable to the whole render loop: startup ~128 ms, first
present ~28–36 ms, shutdown ~32–40 ms, versus ~91 ms for a 256-frame one-draw
loop. Inside startup, `gbm_create_device` alone is ~79.5 ms (62% of the
intercepted startup calls) and is mostly Mesa driver loading.

## 2. Measurement boundaries (read before comparing any two numbers)

These boundaries are why the numbers below are grouped rather than pooled.

| Boundary | What it means |
|---|---|
| Per-frame `glFinish` vs swap-only | The E2E benchmark finishes every frame. Production (`hs_graphics_present`) only submits and swaps. At 64 draws, finishing every frame costs ~1055 µs vs ~808 µs swap-only (+30%). E2E loop numbers are therefore an upper bound on the production path. |
| Traced / preloaded / sampled | `strace`, `LD_PRELOAD`, and `perf` perturb scheduling. They locate cost; they do not rate throughput. The shim's own overhead was measured at +1.8% (1 draw) and +3.6% (64 draws). |
| Submission-only vs completion | `window-noswap-submit-*` never waits. Its 72.3 µs/frame at 64 draws is **CPU submission**; the 217.9 ms `glFinish` after the loop is the deferred GPU work (~851 µs/frame). These must not be compared as throughput. |
| `process_cpu_ns` in the E2E report | This is CPU consumed **during the timed loop across all threads**, not whole-process CPU. At 8 producers it exceeds wall time (3.14 cores average). |
| `whole_process_ns` | External wall clock around `/usr/bin/time` + benchmark process, so it includes exec, dynamic loading, and exit. It is larger than the sum of in-process phases. |
| Additive waits | `swap` and `finish` share the same underlying wait. Summing them as independent costs double-counts. Swap-on already absorbs most of the completion wait. |

## 3. Per-process budget (committed E2E path, capture on, native, medians)

256 frames per producer; `loop` is the timed region; `CPU` is loop-only CPU
across all threads; `wall` is external whole-process wall.

| Case | init | cold (first present) | loop | shutdown | loop CPU | wall |
|---|---:|---:|---:|---:|---:|---:|
| 1p 1d | 128.0 ms | 28.5 ms | 90.6 ms | 32.4 ms | 134.5 ms | 349.9 ms |
| 1p 64d | 125.5 ms | 36.5 ms | 199.7 ms | 35.1 ms | 260.8 ms | 470.5 ms |
| 2p 1d | 128.8 ms | 28.0 ms | 95.2 ms | 45.7 ms | 236.3 ms | 364.5 ms |
| 2p 64d | 127.8 ms | 29.1 ms | 308.9 ms | 35.2 ms | 689.7 ms | 581.9 ms |
| 8p 1d | 129.5 ms | 37.6 ms | 119.0 ms | 36.8 ms | 409.1 ms | 389.2 ms |
| 8p 64d | 128.0 ms | 35.5 ms | 989.4 ms | 39.8 ms | 3102.9 ms | 1299.0 ms |

Shares for 1p 1d: init 36.6%, cold 8.1%, loop 25.9%, shutdown 9.3%, and 20.1%
outside the in-process phases (pre-main loading, exit, harness). For 8p 64d
the loop is 76.2% and init is 9.9%: the crossover is workload length.

Per-frame median submit vs completion (completion = submit + wait + finish):

| Case | submit | completed | wait share |
|---|---:|---:|---:|
| 1p 1d | 52.0 µs | 346.3 µs | 85.0% |
| 1p 64d | 121.4 µs | 773.0 µs | 84.3% |
| 2p 1d | 61.5 µs | 363.6 µs | 83.1% |
| 2p 64d | 194.5 µs | 1195.0 µs | 83.7% |
| 8p 1d | 88.2 µs | 451.4 µs | 80.5% |
| 8p 64d | 633.2 µs | 3819.3 µs | 83.4% |

## 4. Startup: ~128 ms + ~28–36 ms first present

Independent reduction of the raw per-call CSVs (`evidence-cost-calls-2`).
`gbm_create_device` is treated as the outermost GBM call; its internals are
not added again. Connector enumeration sums the init scan and the first-present
scan because the CSV does not timestamp individual calls.

| Item | 1 draw | 64 draws | Share (1 draw) |
|---|---:|---:|---:|
| `gbm_create_device` | 79.49 ms | 79.16 ms | 61.7% |
| `drmModeGetConnector` (6 calls, 2 scans) | 18.59 ms | 22.82 ms | 14.4% |
| `drmModeSetCrtc` (first modeset) | 16.27 ms | 21.17 ms | 12.6% |
| `eglInitialize` | 8.42 ms | 8.51 ms | 6.5% |
| `eglCreateContext` | 5.77 ms | 5.26 ms | 4.5% |
| everything else intercepted | 0.21 ms | 0.24 ms | 0.2% |
| **intercepted startup total** | **128.75 ms** | **137.11 ms** | 100% |
| `eglMakeCurrent` (2 calls: init + teardown) | 2.81 ms | 2.81 ms | separate |

`gbm_create_device` is therefore 62% of the one-time intercepted startup and
62% of the 128.0 ms measured `init_ns`. The traced syscall profile shows
~70.5 ms of loader syscalls across the run (`read` 21.4, `openat` 12.8, `mmap`
9.3, `fstat` 6.4, `lseek` 5.9, `munmap` 5.5, `newfstatat` 4.9, `mprotect` 4.4),
inflated by tracing; most of that sits inside the driver load triggered by
`gbm_create_device`.

The perf init-only profile agrees: kernel 42.3%, `ld-linux` 17.0%, libc 13.5%,
`libgallium` 12.2%, `libexpat` 11.2%, LLVM 1.7%, `libEGL_mesa` 1.1%. Startup
compute is loader + kernel + Mesa, not application code.

## 5. Steady state: GPU throughput, not CPU submission

Frame-cost probe, 5 reps × 256 frames, medians, same real V3D/GBM/EGL stack.
`submit` = clear + draw calls; `swap` = `eglSwapBuffers`; `finish` = `glFinish`.

| Configuration | frame | submit | swap | finish |
|---|---:|---:|---:|---:|
| swap on, finish every frame, 0 draws | 350.0 µs | 27.8 | 104.9 | 211.4 |
| swap on, finish every frame, 1 draw | 385.0 µs | 60.1 | 98.7 | 220.8 |
| swap on, finish every frame, 64 draws | 1055.0 µs | 129.6 | 132.7 | 785.5 |
| swap on, no finish, 64 draws | 808.6 µs | 130.0 | 669.7 | 0.1 |
| swap on, finish every 64, 64 draws | 808.4 µs | 127.9 | 668.0 | 0.2 |
| swap off, finish every frame, 0 draws | 350.5 µs | 24.1 | — | 323.9 |
| swap off, finish every frame, 64 draws | 1048.8 µs | 125.0 | — | 918.8 |
| swap off, never finish, 64 draws | 72.3 µs* | 46.6 | — | — |
| swap off, finish once per 256, 64 draws | 820.4 µs | 46.1 | — | 0.1 µs (one frame) |

\* Submission-only: the deferred `glFinish` after the loop took 217.9 ms for
256 frames, i.e. ~851 µs/frame of GPU work. The 72.3 µs value is CPU
submission, not throughput.

Deep no-swap pipelining does **not** collapse toward submission:

| Finish every | d64 frame | note |
|---|---:|---|
| 8 | 887.8 µs | finish p99 6.2 ms |
| 64 | 851.5 µs | finish p99 47.7 ms |
| 256 | 820.4 µs | one finish at the end |
| 0 draws | 6.1 µs | clear-only throughput |

So the ~820–850 µs at 64 draws is real GPU/driver work, not a sync artifact.
Pipelining recovers only ~19–22% versus finishing every frame (1055 → 820–851
µs). The percentile split confirms the mechanism: at `finish_every=256` the
median frame is 46.2 µs while the mean is 820.4 µs — most frames submit quickly
and the batch waits at the single finish.

Production-shaped behavior: swap-only is ~808 µs at 64 draws, and the E2E path
is ~773 µs with its own (lighter) renderer workload and a per-frame `glFinish`.
The controlled probe comparison at identical workload is the valid one:
finishing every frame costs ~30% more than swap-only (1055 vs 808.6 µs). The
E2E's absolute 773 µs is not directly comparable to the probe because the
renderer's geometry differs.

Main-thread cost is not negligible while waiting:

| Configuration | CPU/frame | wall/frame | CPU share of wall |
|---|---:|---:|---:|
| swap+finish, 1 draw | 211.1 µs | 385.0 µs | 54.8% |
| swap+finish, 64 draws | 334.2 µs | 1055.0 µs | 31.7% |
| no-swap+finish, 0 draws | 145.3 µs | 350.5 µs | 41.5% |
| swap-only, 64 draws | 290.0 µs | 808.6 µs | 35.9% |

The process has only two threads (main + `test_gl:disk$0`, which consumes ~0
CPU), so the wait is polled in the calling thread, not offloaded to a worker.

## 6. Shutdown: final `close()` dominates

| Item | 1 draw | 64 draws |
|---|---:|---:|
| `eglDestroyContext` | 3.30 ms | 3.30 ms |
| `gbm_device_destroy` | 1.76 ms | 1.74 ms |
| `eglTerminate` | 0.094 ms | 0.095 ms |
| `eglDestroySurface` | 0.068 ms | 0.068 ms |
| `drmDropMaster` | 0.009 ms | 0.008 ms |
| `eglReleaseThread`, `gbm_surface_destroy` | 0.004 ms | 0.004 ms |
| **intercepted teardown total** | **5.24 ms** | **5.21 ms** |

Measured `shutdown_ns` is 32.4 ms (1p 1d) to 39.8 ms (8p 64d). The intercepted
teardown calls are only ~16% of that. The remainder is the final DRM
`close(drm_fd)`, measured by `strace -T` at 35.2 ms under tracing (inflated)
and consistent with ~27 ms untraced. The syscall trace put the final
`close(3)` at the top of all individual calls.

## 7. Compute attribution (perf, 4000 Hz, all samples aggregated by shared object)

Kernel symbols are restricted on this host (`kernel.perf_event_paranoid=4`);
kernel time is countable by address, not nameable. No security setting was
changed. Function names in the text come from the flat report; the percentages
come from the full DSO aggregation, which keeps the long tail that the flat
report's 0.4% cutoff drops.

| Case | kernel | libgallium | libc | ld-linux | libexpat | app | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| steady 1 draw | 54.0% | 20.4% | 13.3% | 4.9% | 3.2% | — | 4.2% |
| steady 64 draws | 51.4% | 27.5% | 10.4% | 4.3% | 2.6% | 0.4% | 3.4% |
| no-swap 0 draws | 58.7% | 13.7% | 13.0% | 7.0% | 4.7% | — | 2.9% |
| init only | 42.3% | 12.2% | 13.5% | 17.0% | 11.2% | — | 3.8% |
| E2E 8p 64d | 77.3% | 5.3% | 9.9% | — | — | 5.7% | 1.8% |

In the dense 8-producer E2E run, `__sched_yield` is 5.96% and `hs_fabric_submit`
is 4.19%. Application code is at most ~5.7% of cycles even in the fabric-heavy
case. Kernel + Mesa + libc are the compute.

## 8. Scaling laws

Offscreen resolution sweep, swap off, finish every frame, 256 frames:

| Size | 1 draw | 64 draws | marginal µs/draw |
|---|---:|---:|---:|
| 64² | 99.9 µs | 144.6 µs | 0.71 |
| 128² | 113.8 µs | 185.3 µs | 1.13 |
| 256² | 149.4 µs | 329.2 µs | 2.85 |
| 512² | 274.5 µs | 759.7 µs | 7.70 |
| 800² | 536.9 µs | 1616.4 µs | 17.14 |

d1 fit: ~100.5 µs fixed + **0.68 ns/pixel** (1.47 Gpx/s, ~5.9 GB/s RGBA8).
d64 fit: ~154.2 µs fixed + 2.29 ns/pixel (about 3.4 full-screen passes for the
64-triangle set). The marginal per-draw cost scales with resolution because
each probe triangle covers a fixed fraction of the viewport: at 800×480 the
marginal cost is ~11 µs/draw, which is fill/bandwidth, not fixed draw setup.
At 64² it is 0.71 µs/draw.

Window 800×480 marginal cost of 63 extra draws is ~11.0 µs/draw with swap +
finish, ~10.9 µs/draw no-swap + finish, and ~12.7 µs/draw under deep
no-swap pipelining — consistent across wait policies.

Producer scaling (E2E, 64 draws): frame completion is 773 µs at 1 producer,
1195 µs at 2, and 3819 µs at 8. The GPU is shared; more producers raise
per-frame latency and total draw throughput only modestly.

## 9. What the measurements say (not yet implemented)

These are directions, not results. Nothing below has been changed or verified.

1. **Steady state is GPU-bound.** App-side CPU work has a ≤5.7% ceiling. A
   meaningful frame win requires less GPU work (fewer/cheaper draws, less
   overdraw, smaller target) or a different wait policy, not faster C.
2. **The wait is polled.** 32–55% of the wait burns CPU in the calling thread.
   A genuinely blocking/fence wait could trade latency for CPU, but that is a
   policy change and has not been measured.
3. **Startup is dominated by driver load.** `gbm_create_device` is 79.5 ms and
   ~70 ms of traced loader syscalls surround it. Any change here must preserve
   device initialization; reuse across processes is not available to this path.
4. **The connector scan happens twice** (init and first present) for 18.6–22.8
   ms total. Whether a single scan can serve both without changing hotplug
   behavior is untested.
5. **Shutdown is the final DRM close.** Intercepted teardown is 5.2 ms; the
   rest is implicit kernel cleanup. The four shutdown candidates already
   rejected (close-only, early-RmFB, full saved-CRTC restore, page-flip
   restore) remain rejected; the page-flip path failed its gate with `EINVAL`
   in 6/6 attempts.

## 10. Limitations and evidence status

- Kernel symbols unavailable; kernel attribution is by DSO/address only.
- `strace` durations include tracer overhead; use the call-timer and harness
  totals for rates.
- The preload shim adds 1.8–3.6% and is absent from production.
- The probe's triangles are synthetic; the marginal per-draw cost is
  fill-bound at the tested resolutions and may differ for the real scene mix.
- The E2E per-frame `glFinish` is a harness artifact; production is swap-only.
- `whole_process_ns` includes exec/loader/exit and `/usr/bin/time`.
- `evidence-cost-frames-3` is a **failed partial run**: it stopped at
  `window-noswap-pipe-e256-d64-r1` when the probe rejected `finish_every=256`.
  It is preserved, not used. The corrected rerun is `evidence-cost-frames-4`.

## 11. Evidence index (Pi: `~/Projects/neogpu/tests/`)

| Directory | Contents |
|---|---|
| `evidence-cost-calls-2` | LD_PRELOAD per-call A/B, 3 reps × {1,64} draws, 64 frames; `report.json` is the independent reduction |
| `evidence-cost-frames-2` | 29 frame-cost configurations, 5 reps × 256 frames |
| `evidence-cost-frames-4` | 6 deep no-swap pipelining configurations, 5 reps × 256 frames |
| `evidence-cost-frames-3` | failed partial run (preserved, not used) |
| `evidence-cost-profile-3` | perf data + flat/children/DSO reports + reduced `report.json` |
| `evidence-cost-breakdown-*` | `strace -T` trace + reduced `trace-report.json` |
| `evidence-reference-c9307e1-final` | committed-path E2E phase budget used in section 3 |

Uncommitted diagnostics: `tests/gles_call_timer.c`,
`tests/test_gles_call_timer_run.py`, `tests/test_gles_cost_calls_report.py`,
`tests/test_gles_frame_cost.c`, `tests/test_gles_frame_cost_run.py`,
`tests/test_gles_cpu_profile.py`, `tests/test_gles_cost_profile_report.py`,
`tests/test_gles_cost_trace_report.py`, and this document.