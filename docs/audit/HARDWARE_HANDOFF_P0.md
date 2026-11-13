# P0: hand work to the Pi 4's hardware

Opened 2026-11-16. The P0 items are worked one at a time, in the order below,
and each is measured before anything is built.

**Why these items.** The steady-state frame path is bound by the GPU and the
driver (COST_BREAKDOWN_RESULTS.md):
- Application code is at most 5.7% of cycles.
- Mesa's load is 62% of start-up.
- Every message the GPU handles costs a job, with a floor of ~37 µs
  (VS_MESA.md).

So the gains lie in hardware blocks that NeoGPU does not use yet.

**Hardware confirmed present on darkcrystal:**
- 15 HVS planes (one is used today);
- a writeback connector (`card1-Writeback-1`);
- `EGL_ANDROID_native_fence_sync`;
- the V3D TFU;
- the bcm2835-codec H.264 encoder and decoder, the ISP (`/dev/video10..31`),
  and the HEVC decoder.

| # | Today (CPU or driver) | Hand to | Evidence before the work | Status |
|---|---|---|---|---|
| P0-1 | The present path's waits around `eglSwapBuffers` and the flip | Fence-driven atomic flips (`IN_FENCE_FD`): the kernel and display flip when rendering completes | The thread's CPU is 32–55% of wall while waiting (COST_BREAKDOWN §5). Unknown: where, and whether the thread sleeps or spins | **Rejected as a hand-off after two red teams.** Implicit fences already allow what `IN_FENCE_FD` would. The GPU is saturated on fill-bound frames, an estimated 20–30 of 4,926 flips could land a refresh earlier, and 9–108 µs of CPU per frame is at stake. **Found instead, and made:** the hold cutoff rose from 250‰ to 350‰ of a refresh (`hold_permille`). Frames of 4.2–5.8 ms are shown 4.2–5.9 ms fresher (median) at the same frame rate, with ≤0.34% repeats per run. After any repeated refresh the old quarter-refresh rule applies for 60 flips, which removed the two regressions a third red team found |
| P0-2 | One draw plus one `glBufferData` per glyph (`hs_text.c`) | One instanced draw, with quads built by the GPU's vertex stage | **Measured (P0-1 red team):** about 16 µs of CPU per draw with this pattern. 100 draws give 614 FPS with the GPU 13% busy, so text-heavy frames are CPU-bound | Next |
| P0-3 | The GPU renders every layer at full panel resolution into one plane | HVS planes: static layers composed at scanout, and upscaling from a lower render resolution | 0.68 ns per pixel of fill, measured (COST_BREAKDOWN §8). Unknown: HVS blend and scale cost | Not started |
| P0-4 | Capture: `glReadPixels`, then a CPU swizzle and flip, then a write | The writeback connector (the frame actually scanned out); the H.264 encoder for recording | Unknown: current cost | Not started |
| P0-5 | Each GPU message is a job (~37 µs floor) | A persistent QPU mailbox | 0.76 µs RTT, 110–138× Mesa (VS_MESA.md) | Not started |
| P0-6 | Mesa's start-up (`gbm_create_device` 79.5 ms) | A direct V3D compute → KMS scanout path for scenes compute can render | Direct cold start 0.47 ms, measured. Unknown: sharing a v3d BO with vc4 | Not started |
| P0-7 | Texture uploads | The TFU (raster to tiled, mipmaps) | No streaming-texture user today | Not started |

**Rules for every item:**
- Measure first.
- Every output is checked.
- The visible result is verified by `make present-hardware` and the external
  scanout probe.
- A change is kept only if it survives both measurement and red-team review.

## P0-1: fence-driven flips

**Verdict (2026-11-16, two red-team passes): rejected as a hardware
hand-off. The investigation found a latency tuning instead.**

- **`IN_FENCE_FD` adds nothing on this stack.** The kernel already holds a
  flip of an unfinished frame on the frame's implicit v3d fence (finding 3).
  So any policy a fence-driven flip allows is available today, with no new
  mechanism.
- **For throughput, nothing is left to gain.** On fill-bound frames the GPU
  is saturated. An estimated 20–30 of 4,926 flips could have landed a
  refresh earlier, all of them at load 1.5.
- **For CPU, little is left to gain.** The present path's own CPU is
  9–108 µs per frame. The kernel-thread work is mostly a per-flip commit
  (about 233 µs per flip), and an atomic commit is also one commit per flip.
- **For latency, there is a real gain, and it comes from the existing
  hold policy, not from fences.** Production holds frames from a quarter
  refresh up (4.17 ms frames). Flipping the newest frame instead, and
  letting the kernel hold it, shows frames 4.2–6.3 ms fresher (median) with
  0 repeated refreshes up to 0.30 refresh. Repeats rise to 0.16–0.4% at
  0.33–0.39, and to 3.2% and 16% at 0.42 and 0.45 (finding 7).
- **Moving the cutoff is a production change.** It trades latency against
  smoothness, and it is proposed below, not made.

### Method

`tests/present_cost_probe.c` runs the pacing_sweep shader, calibrated to a
fraction of a refresh, through the production `hs_graphics_present` (vsync
off, the default). Every EGL, GBM and DRM call the present path makes is
wrapped, including the EGL fence calls, `drmModeAddFB*` and
`eglSwapInterval`. Production code is unchanged.

The probe records:
- **Per-phase wall time** from `CLOCK_MONOTONIC`. Each wrapper costs 49 ns,
  which is calibrated and subtracted.
- **Thread CPU** read once at each end of the loop.
- **Process CPU** from `getrusage`.
- **Kernel-thread CPU**, as per-PID deltas from `/proc/<pid>/schedstat`, net
  of their idle rate (the median of 10 windows of 100 ms).
- **GPU engine time** from the v3d driver's `/proc/self/fdinfo`.
- **Which frame each flip shows** (the one just rendered, or an older held one).
- **Per-flip queue and landing times** (`PROBE_FLIPS`).

`PROBE_PHASE_CPU=1` adds thread CPU per phase. Its timer is a syscall
(1.75–1.83 µs per pair), which is also calibrated and subtracted.

Each load runs 600 timed frames after 60 settling frames, 3 repetitions,
with the performance governor and inside the GPU lock. Resets stayed at 205
throughout, with 0 present errors.

Evidence (Pi only), all in `tests/evidence-p0-1-20261116/`:
- `final.jsonl` and `final-phasecpu.jsonl`: the tables below;
- `flips-*.txt` and `v3-flips.jsonl`: flip timing;
- `v3-draws.jsonl`: draw-heavy frames.

The red team's own evidence is in `tests/evidence-p0-1-redteam-20261116/`.

### Results

Per frame. Across the 3 runs, FPS, wall time and GPU time agree within 1%. The CPU figures vary more: up to 13% for kernel threads at 0.25, and the flip ioctl ranges 38–55 µs at 1.5.

| load (refresh) | FPS | wall | GPU render | render busy | thread CPU | kernel threads | flips: newest / held |
|---|---|---|---|---|---|---|---|
| 0.05 | 900.5–901.4 | 1110 µs | 1094 µs | 98.6% ±0.2 | 143–149 µs | 66–68 µs | 40–41 / 0 |
| 0.25 | 238.4 | 4195 µs | 4188 µs | 99.9% ±0.2 | 172–176 µs | 98–111 µs | 0 / 152 |
| 0.5 | 125.2 | 7990 µs | 7995 µs | 100.0% ±0.2 | 200–205 µs | 164–169 µs | 0 / 288–289 |
| 0.95 | 63.9 | 15659 µs | 15669 µs | 100.0% ±0.2 | 271–281 µs | 291–308 µs | 0 / 565 |
| 1.5 | 40.6 | 24628 µs | 24639 µs | 100.0% ±0.2 | 267 µs | 261–271 µs | 0 / 600 |
| 0.5, vsync on | 60.0 | 16658 µs | 7982 µs | 47.9% | 216–227 µs | 272–278 µs | 600 / 0 (after the probe fix; see red team 2) |

Notes on the table:
- **Busy resolution.** The ±0.2 is one job's share of the window. fdinfo
  credits a job when it completes, so readings just above 100% are within
  that resolution. The red team checked busy time independently, timing each
  render job from start to completion IRQ: 99.5% at 0.5 and 99.7% at 0.95.
  The GPU sits idle 16–32 µs between render jobs.
- **The 0.05 row is the shader's floor.** Calibration bottoms out at one
  shader iteration (0.84 ms calibrated). A real frame's render job takes
  1094 µs, because each job adds about 250 µs of per-job cost.
- **Repeats.** At 1.5, 289 of 600 refreshes repeat by construction, since
  frames are longer than a refresh. Vsync on repeated 1 refresh per run.

Where the calling thread's time goes (wall / thread CPU, µs per frame, from
the run with the per-phase CPU timer):

| load | submit | `eglSwapBuffers` | flip ioctl | flip wait | EGL fences | lock + release | rest |
|---|---|---|---|---|---|---|---|
| 0.05 | 43 / 42 | 1035 / 90 | 5 / 4 | 4 / 4 | 0 | 0.5 / 0.4 | 1 |
| 0.25 | 55 / 53 | 4082 / 98 | 15 / 12 | 5 / 5 | 11 / 11 | 1.0 / 0.6 | 1 |
| 0.5 | 57 / 56 | 7845 / 100 | 28 / 21 | 6 / 6 | 22 / 21 | 0.9 / 0.6 | 2 |
| 0.95 | 63 / 62 | 8930 / 107 | 43 / 38 | 6531 / 24 | 44 / 43 | 0.9 / 0.7 | 3 |
| 1.5 | 64 / 63 | 24418 / 106 | 53 / 39 | 9 / 8 | 43 / 41 | 0.7 / 0.6 | 2 |

The "rest" column is wall time from the cheap-timer runs: with the per-phase
CPU timer, its CPU is a residue of overhead subtraction for nested wrappers.
At 0.95 the flip wait is the intended wait for a flip about to land (HSGraphics
`flip_waits`).

### What the measurements establish

1. **Frame rate is GPU render time, for fill-bound frames.**
   - The render engine is busy 98.6–100% of wall time at every vsync-off
     load.
   - The red team's CPU-injection test confirms it directly. Adding
     0–800 µs of spin per frame at 0.05 left 900 FPS unchanged, and
     0–7700 µs at 0.5 left 125.0 FPS unchanged. FPS fell only once thread
     CPU exceeded GPU time.

   **Scope:** this holds for frames where the GPU is the bottleneck. It
   does not hold for frames with many small draws (see 6).

2. **The thread sleeps; it does not spin.** Each frame has one voluntary
   context switch (1.88 at 0.95, where the present path also waits for a
   flip), and `eglSwapBuffers` is 99% sleep (for example
   7845 µs of wall against 100 µs of CPU). COST_BREAKDOWN's "busy-polls a
   third to half of the wait" came from harnesses that call `glFinish` every
   frame. It does not describe this path.

3. **Who waits for rendering depends on the load.**
   - **At 0.05, the kernel waits.** Every flip shows the frame just rendered
     (40–41 per run, 0 held). The red team's bpftrace showed 40 of 40 flips
     waiting 0.5–1 ms on that frame's v3d fence in
     `drm_atomic_helper_wait_for_fences`, then released 8–32 µs after its
     render IRQ.
   - **At 0.25 and above, userspace waits, by design.** Every flip shows an
     older, held frame that has already finished (152–600 per run, 0
     newest). The kernel waits only 4–8 µs, and at 1.5 the buffer had no
     fence at all. Each flip is queued 126–191 µs after the render IRQ
     (median 132, n = 120).
   - **Waiting in userspace is a policy choice, not a missing mechanism.**
     Flipping the newest, unfinished frame and letting the kernel hold it
     is exactly what `IN_FENCE_FD` would do, and implicit fences already
     allow it. Whether that policy is better depends on the load:
     - Before `3d574e7` that was NeoGPU's behavior. At about two frames per
       refresh (`test_08_raycast_orbit`, 110 FPS) it showed a new frame on
       only 65% of refreshes.
     - Measured here, it repeats refreshes from about 0.4 refresh up, but
       not below (finding 7).
     - `3d574e7` is evidence only for the heavier loads.

4. **Of 4,926 flips, at most 29 could land a refresh earlier.** The test
   used the vblank grid, with the previous flip's vblank counted as taken
   (`flips-*.txt`, 1800 frames per load). The table counts how many flips
   landed on the first free vblank after they were queued.

   | load | flips | on the first free vblank | missed it | missed by less than the gap fences could close |
   |---|---|---|---|---|
   | 0.05 | 119 | 119 | 0 | — |
   | 0.25 | 452 | 452 | 0 | — |
   | 0.5 | 864 | 864 | 0 | — |
   | 0.95 | 1693 | 1693 | 0 | — |
   | 1.5 | 1798 | 1313 | 485 | ≤ 18 (gap 105 µs) to ≤ 29 (190 µs), 1.0–1.6% |

   How to read the last column:
   - **What it counts.** Misses whose margin plus the gap would reach the
     deadline.
   - **The gap.** It is the time from the render IRQ to the flip ioctl:
     126–191 µs (median 132, n = 120), traced at 1.5 only. A fence-held flip
     is released 8–32 µs after the IRQ. For the 126–191 µs gap the count is
     20–30 flips. The first version of this table used 105–190 µs, and 105
     had no measured source.
   - **The deadline is sharp.**
     - Every hit had at least 4.0 ms to spare, and only one had less than
       4.44 ms.
     - 476 of the 485 misses had less than 4.44 ms, spread evenly from 0 to
       4.4 ms.
     - So the misses are frames queued too late: a 24.6 ms frame drifts
       against the 16.6 ms refresh. They are not deadline jitter, as the
       first version said.
   - **The counts are estimates, not bounds.** A hit shows the deadline was
     no later than its margin on that refresh only.
   - **The margins are measured against the page-flip event's timestamp.**
     That timestamp comes about 1 ms after the event is delivered (the start
     of the next scanout), so the deadline counted back from the vblank
     interrupt is about 1 ms shorter. At 0.95, 1686 of 1693 flips were
     queued before the previous flip's timestamp. The grid counts are not
     affected.

5. **What P0-1 could save in CPU.**
   - **The present path's own CPU is small.** That is flip ioctl, flip wait,
     EGL fences, lock, release and the rest. It comes to about 9 µs per
     frame at 0.05, 30 at 0.25, 50 at 0.5, 108 at 0.95 and 91 at 1.5.
     Atomic commits would replace, not remove, the ioctl and the fence
     query.
   - **Kernel threads cost more.** They spend 66–308 µs per frame. Across
     the 18 runs this fits 51 µs per frame plus 233 µs per flip (residuals
     −24 to +38 µs), so it is mostly a per-flip commit (red team F3). The
     first baseline missed it. An atomic commit with `IN_FENCE_FD` is also
     one commit per flip; its CPU cost was not measured.
   - **Hard-IRQ time is not accounted separately on this kernel.** It is
     charged to whichever task it interrupts.

6. **Frames with many small draws are bound by the CPU.** The red team's
   `p0_1_redteam_probe draws N` reproduced here:
   - **100 draws** of 16×16 px: 614 FPS, with 1604 µs of CPU per frame and
     render busy at 13%.
   - **1000 draws:** 60 FPS, with 16.6 ms of CPU per frame and render busy
     at 2.4%.

   That is about 16 µs of CPU per draw with the `hs_text` pattern
   (`glBufferData`, `glUniform`, `glDrawArrays`). That is 10× the 1.6 µs
   this plan had derived for P0-2. **P0-2 is therefore a real CPU lever:**
   text-heavy frames are CPU-bound today.

7. **Displayed-frame age: production's hold cutoff costs latency.**
   - **Probe:** `tests/p0_1_redteam2_newest` (`POLICY=prod` or
     `POLICY=newest`). The newest policy makes the present path's "is the GPU
     done?" check answer yes, so it flips the newest frame and the kernel
     holds it on its implicit fence.
   - **Age:** the page-flip event timestamp minus the time the shown frame's
     draw was issued.
   - **Runs:** 1200 frames per cell at 0.20–0.30, 0.5, 0.95 and 1.5, one run
     each, matching the red team's runs. 1800 frames × 2 runs at 0.33–0.45.
   - FPS was identical in every pair, and there were 0 resets.

   | load | frame time | repeats, prod / newest | median age, prod / newest | p90 age, prod / newest |
   |---|---|---|---|---|
   | 0.20 | 3.3 ms | 0 / 0 | 19.3 / 19.3 ms | identical (below the cutoff, production does not hold) |
   | 0.25 | 4.2 ms | 0 / 0 | 23.9 / 19.7 ms | 25.6 / 21.4 |
   | 0.30 | 5.0 ms | 0 / 0 | 24.6 / 20.0 ms | 26.5 / 21.8 |
   | 0.33 | 5.4 ms | 0 / 1 of 589 | 25.8 / 20.4 ms | 28.0 / 22.6 |
   | 0.36 | 5.9 ms | 0 / 1 of 634 | 26.5 / 20.6 ms | 28.7 / 23.0 |
   | 0.39 | 6.3 ms | 0 / 2–3 of 682 | 27.1 / 20.8 ms | 29.5 / 23.3 |
   | 0.42 | 6.7 ms | 0 / 22–25 of 728 | 27.7 / 21.3 ms | 30.4 / 24.0 |
   | 0.45 | 7.1 ms | 0 / 121–123 of 774 | 28.3 / 22.9 ms | 31.2 / 34.8 |
   | 0.5 | 8.0 ms | 0 / 114 of 576 | 29.6 / 24.0 ms | 32.8 / 35.9 |
   | 0.95 | 15.7 ms | 0 / 140 of 1129 | 50.8 / 34.2 ms | 50.9 / 47.5 |
   | 1.5 | 24.6 ms | 576 / 900 of 1775 | 62.1 / 50.9 ms | 68.8 / 64.1 |

   - **The prior audit agrees.** AUDIT_2026-11-13 measured "not holding"
     first repeating at 4.62 ms (1 in 557) and 2.4–3.7% at 5.89 ms.
   - **Made: the hold cutoff is now 350‰ of a refresh** (see "Hold-cutoff
     sweep" below).
   - **Above 0.42 refresh the newest policy trades smoothness for latency.**
     It is a candidate for an opt-in low-latency mode, not the default.

**Conditions.** The `neogpu-kworker-rt` service (KMS commit kworkers at
SCHED_FIFO 1) was active for every run in this section. It moves the display
deadline (AUDIT_2026-11-13).

### Hold-cutoff sweep (the change made)

- **Tools:** `tests/hold_cutoff_sweep.sh` drives `tests/p0_1_redteam2_newest`
  with `HOLD_PERMILLE=250` (held, the old rule) and `1000000` (never held).
- **Runs:** 3 × 1800 frames per cell, kworker-rt service on and off, idle
  (`sweep.jsonl`) and with one spinning process per CPU (`spin.jsonl`).
  Evidence: `tests/evidence-hold-cutoff-20261116/`. 0 errors; resets 205 → 205.
- Loads are quantized by shader iterations, so the frame times measured are
  4.2, 4.6, 5.0, 5.45, 5.87, 6.3 and 6.74 ms. FPS was identical held and not
  held in every idle cell.

| frame time | idle, not held: repeats (per run) | idle, median age held → not held | all CPUs busy, service on: repeats not held | all CPUs busy, service off: repeats held / not held |
|---|---|---|---|---|
| 4.2 ms | 0 | 23.9 → 19.7 ms | — | — |
| 4.6 ms | 0 | 24.6 → 20.0 ms | 0 | 20% / 31% |
| 5.0 ms | 0 | 25.2 → 20.2 ms | 0.06% | 16% / 23% |
| 5.45 ms | 0–0.17% | 25.8 → 20.4 ms | 0.11% | 19% / 33% |
| 5.87 ms | 0–0.16% | 26.5 → 20.6 ms | 0 | 10% / 28% |
| 6.3 ms | 0.3–1.3% | 27.1 → 20.9 ms | not run | not run |
| 6.74 ms | 2.6–3.7% | 27.7 → 21.3 ms | not run | not run |

- **The service made no difference idle.** Service on and off gave the same
  repeats within a run or two. The Nov 11 sweep (before `b52bd35`) had not
  holding repeat 0.2–1.4% from 4.6 ms; it no longer does.
- **Cutoff chosen: 350‰ (5.83 ms at 60 Hz).** Every frame time left unheld is
  below the last measured-safe point (5.87 ms). One run at 6.3 ms reached
  1.3%, over the gate.
- **Regression, measured, then removed:** every CPU busy *without* the
  kworker-rt service. Pacing already fails there (held: 10–20% repeats), and
  not holding made it worse (23–33%), with median age up to 35 ms in some
  runs. With the service, not holding stays ≤0.11%. The fallback below
  (red team 3, H1) brings this case back to the old rule's numbers.
- **Verified with the new default** (`default350.jsonl`): at 3.3–5.44 ms
  every flip was the newest frame (median age 19.3–20.4 ms, ≤2 repeats of
  589); from 5.87 ms frames are held as before (0 repeats).
- **Gate:** `test_present_pacing` gained a phase at ~0.30 refresh that must
  stay unheld with ≤1% repeats, and one at ~0.42 refresh that must be held
  (after ≥99% of presents) with ≤1% repeats.

### Red team 3: the hold-cutoff change

An independent agent attacked the change (evidence:
`tests/evidence-hold-cutoff-redteam-20261116/`, tools:
`tests/redteam-hold-cutoff/`). I reproduced each finding, remediated it, and
re-measured (`tests/evidence-hold-cutoff-20261116/remediation/`, 3 × 1800
frames per cell, resets 205 → 205).

| id | finding | sev | remediation and result |
|---|---|---|---|
| H1 | Every CPU busy, service off, ~5.0 ms frames: 350‰ repeated 4.4–4.7% vs 1.0–1.8% at 250‰. The interval average sat on the cutoff and the policy flipped 79–260 times per run | med | **After any repeated refresh, the quarter-refresh rule applies for 60 flips** (`HS_GRAPHICS_HOLD_AFTER_REPEAT`). Re-measured: 344 of 1,822 refreshes repeated vs 483 of 1,880 at 250‰, same median age (24.5–27.8 vs 27.5–27.8 ms). Pacing there still fails either way; the kworker-rt service is the fix for that |
| H2 | The gate passed cutoffs of 450‰ and 500‰ | med | Added the held phase at ~0.42 refresh. 450, 500 and never-hold now fail (they held 483–525 of 600 presents, 3 of 258 refreshes repeated); 250 fails the unheld phase. **Limit:** a 400‰ mutant still passes. Repeat counts cannot reliably separate 350 from 400 (not holding at 6.3 ms gave 0.3–1.3% per run), so the 350 choice rests on the sweep, not the gate |
| H3 | One gate run failed the unheld phase with 0 repeats | low–med | Two causes: calibration's gap inflated the interval average, and (after H1) earlier phases' deliberate repeats switch the fallback on. The gate resets the average after calibrating and settles until the fallback has expired. 8 of 8 runs passed |
| H4 | Spiky frame costs (7 × 2.9 ms + 1 × 17.4 ms, mean between the cutoffs): 350‰ repeated 26% vs 12–19% | med | The H1 fallback covers it: 286 vs 288 repeats of 1,557 at 250‰. These frames stutter under any cutoff; a mean cannot see the spike |
| H5 | The comment claimed "nothing below 350 repeats more than 1 in 600" | low | Restated from the data: at most 0.34% per 1800-frame run |

**Kept, re-measured after the fallback:** idle at 5.0–5.81 ms, 0–2 repeats
per run and median age 20.1–21.3 ms vs 25.2–26.4 ms at 250‰. CPUs busy with
the service: 0–2 repeats, 20.1–21.1 vs 25.2–25.8 ms. Alternating 2.1/8.8 ms
frames: 0–2 repeats, 21.0–22.9 vs 25.3–26.0 ms.

**Gates after remediation:** build 0 warnings, `check-sources` 0, `make
test` 0, `test_present_pacing` 8 of 8, resets 205 → 205.

**Not measured:** refresh rates other than 60 Hz; real workloads through
`src/hs_backend_gles.c`; the interval average staying high for tens of frames
after a pause (that mechanism predates this change).

### Red team: findings and what was done

An independent agent attacked the first baseline. I then reproduced its
findings and remediated them.

| # | Finding | Severity | Remediation |
|---|---|---|---|
| F1 | "The kernel holds each flip" was wrong at ≥0.25: the probe flips older finished frames there, and userspace waits | high | Measured both regimes (finding 3). Per-flip timing gives an upper bound on what fences could change (finding 4). The prior evidence in `3d574e7` is cited |
| F2 | "No CPU-side change can raise frame rate" is false for frames with many small draws | high | Claim scoped to GPU-bound frames. Draw-heavy result reproduced (finding 6). P0-2's evidence updated |
| F3 | CPU spent in kernel threads (KMS commits, drm_sched) was missing | medium | Per-PID kernel-thread CPU, net of idle (finding 5) |
| F4 | `CLOCK_THREAD_CPUTIME_ID` is a syscall (~1.7 µs): it inflated every phase and the kernel-CPU column | medium | Per-phase timing is now wall-only (49 ns, subtracted). CPU is read once per loop, and per-phase CPU is opt-in with its overhead subtracted |
| F5 | The 0.01 row duplicated 0.05 (calibration floor) | low | Dropped; achieved shader iterations are printed |
| F6 | Fence calls, `drmModeAddFB*` and `eglSwapInterval` were not wrapped | low | Wrapped |

Bugs found in my own remediation, fixed before the final runs:
- **The 500 ms idle measurement entered the present path's frame-interval
  average.** That switched on its wait-for-flip branch: 125 → 116 FPS at
  0.5. The idle measurement now runs before the settling frames.
- **A single idle window gave baselines off by up to 1 ms per frame.** The
  baseline is now the median of 10 windows.
- **A kworker exiting mid-run took its whole run time out of a raw sum**
  (−781 µs per frame). The probe now uses per-PID deltas.

### Red team 2: findings and what was done

A second independent agent attacked the remediated version. I reproduced
each finding before acting on it.

| # | Finding | Severity | Remediation |
|---|---|---|---|
| R2-1 | The verdict never measured displayed-frame age. At 0.25 the newest-frame (fence-held) policy was strictly better, and `3d574e7` covers only heavier loads | high | Reproduced and extended to 0.20–0.45 (finding 7). The verdict now separates the hand-off (rejected) from the hold-cutoff tuning (proposed). Finding 3 is scoped |
| R2-2 | "Most misses are deadline jitter" contradicts the data: the deadline is sharp, and misses come from frames queued late | medium | Finding 4 rewritten |
| R2-3 | The 105 µs gap had no measured source. The measured gap is 126–191 µs, at 1.5 only | low | Uses 126–191 µs (20–30 flips) |
| R2-4 | "≤" bounds rest on an assumption | low | Called estimates |
| R2-5 | Vsync-on flips were misclassified as held: the flip is retired inside present, so `pending_bo` is NULL | medium | Classified by framebuffer id. Re-run: 600 newest / 0 held at vsync 0.5 (3 of 3); 40–41 newest / 0 held at 0.05 (3 of 3) |
| R2-6 | The page-flip timestamp comes about 1 ms after the event | low | Stated in finding 4 |
| R2-7 | Overstatements: one context switch per frame (1.88 at 0.95), "agree within ~3%", the "rest" column | low | Corrected |
| R2-S | Suspicion: the kernel-thread cost "stays either way" was asserted | low | Fitted at 51 µs per frame plus 233 µs per flip; the claim is limited to "one commit per flip either way, not measured" |
| R2-S | Suspicion: the kworker priority service's state was not stated | low | Stated: active for every run |

Checked by red team 2 and confirmed:
- the pairing of queue and landing times;
- every total (4,926; the present-path CPU sums; the kernel-thread range;
  GPU busy, FPS and CPU);
- the overhead subtraction for each phase.

**Not measured:**
- the thermal state and the V3D clock during these runs (`vcgencmd` cannot
  open `/dev/vcio`). FPS and GPU time were consistent within 1% across runs.
- the kernel functions behind the kernel-thread CPU. Kernel symbols are
  restricted, even to `sudo perf`.
- the CPU cost of an atomic commit with `IN_FENCE_FD`.
