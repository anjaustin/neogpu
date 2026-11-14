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
| P0-3 | The GPU renders every layer at full panel resolution into one plane | HVS planes: static layers composed at scanout, and upscaling from a lower render resolution | 0.68 ns per pixel of fill, measured (COST_BREAKDOWN §8). Unknown: HVS blend and scale cost | **Shelved 2026-11-16.** Watched on the panel, the scaled image looked progressively more pixelated, not like a native render: a quality trade, not a speed-up. `hs_graphics_init_scaled` is out of the public API (behind `HS_GRAPHICS_SHELVED_SCALED`, kept for the tests). Kept: the plane-reset fix, the layer measurements and the tests. Reconstruction upscaling (adaptive edge refinement, temporal) was proposed and is shelved too. Before shelving: **Scaling made (opt-in):** `hs_graphics_init_scaled` renders at a reduced size and the HVS scales it at scanout. test_08 runs 1.76× faster at 600×360 and 3.9× at 400×240, on the panel. HVS composition and scaling cost no frame time on this panel (they do use memory bandwidth). Layers and sprites: measured capable, not built (no current user with a GPU cost to save). **Red team found and fixed:** takeover (native too) inherited a previous client's plane alpha, filter and blend mode, and native takeover left stray overlays on. The test now kills 14 of 14 mutants |
| P0-4 | Capture: `glReadPixels`, then a CPU swizzle and flip, then a write | The writeback connector (the frame actually scanned out); the H.264 encoder for recording | Unknown: current cost | Not started |
| P0-5 | Each GPU message is a job (~37 µs floor) | A persistent QPU mailbox | 0.76 µs RTT, 110–138× Mesa (VS_MESA.md) | **Built, red-teamed and hardened (2026-11-17); nothing in NeoGPU calls it yet.** Messages cost 2.1–3.3 µs, against 64–71 µs for a job. **It wins in one regime:** arithmetic-deep work (4,096 steps per element) runs 1.21–1.27× faster than all 4 CPU cores, bit-exact (1.24–1.31× before the chain loop was given thread switches so that it cannot stop a renderer). For gathers and for shallow arithmetic the CPU wins at every size tested. The red team found a defect in shipped code: every 8-batch program took its batch number from the QPU thread index and wrote wrong words beside a renderer (gather kernel: 3,273 of 3,279 jobs); fixed, 0 of 3,206 after. Costs while the mailbox is in use: 2–3% of a Mesa renderer on 1 batch, 34–40% on 8; nothing once the caller has been quiet for 2 ms |
| P0-6 | Mesa's start-up (`gbm_create_device` 79.5 ms) | A direct V3D compute → KMS scanout path for scenes compute can render | Direct cold start 0.47 ms, measured. Unknown: sharing a v3d BO with vc4 | **Started 2026-11-17; the scene runs on the panel.** A buffer vc4 allocates can be imported into v3d and filled by a compute job (the reverse is refused). A test pattern rendered by QPU compute alone is flipped at 60 Hz with no Mesa in the process: 0.37 ms per frame, no late flip, both buffers and the display controller's own capture of the frame checked. The two probes were red-teamed (11 findings fixed, 18 cases, 9 of 9 mutants). Reciprocal and reciprocal square root measured: within 1 ulp, one instruction each. **The scene, stage 1 (primary rays, flat colours):** a hand-assembled kernel renders it in 2.66 ms per frame on 8 batches, identical to a CPU reference except 1–3 edge pixels in 200 frames. **The whole scene** (shading, shadows, one reflection): 725 instructions per 16 pixels, 15.6 ms per frame (64 FPS), about 7 of 384,000 pixels per frame off the CPU reference. Run straight through it is 1.72× slower than Mesa (110 FPS). **With work skipped where its result is known (sky rows, reflections that hit nothing, ground nothing can reach) the same image, bit for bit, takes 3.87 ms of GPU time plus 0.16 ms of CPU per frame.** Red-teamed twice (11 findings fixed, one of them a wrong pixel in 4 of 200 frames that the first check had passed; 43 cases, 12 faults, 7 of 7 mutants). **On the panel through KMS, no Mesa in the process: 247.9 frames rendered per second against Mesa's 110.0 in the same session, 2.25×; 60.2 flips per second, none late; the frames on the panel checked against the reference and the display controller's capture** (25 cases, 12 of 12 mutants). Next: sphere lists for the long path, the path beside another GPU client; plan in `P0_6_PLAN.md` |
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

## P0-3: HVS planes and scaling

> **Status: shelved (2026-11-16).** The user watched the four render sizes on the panel (the rerun matched these numbers to within 1%: 109.9, 193.7, 428.7, 930.7, 110.1 FPS, 0 underruns, resets 205→205). The scaled image looked progressively more pixelated, not truly upscaled. A smaller render plus a stretch is a quality trade, not a speed-up, so `hs_graphics_init_scaled` is now behind `#ifdef HS_GRAPHICS_SHELVED_SCALED`. Still in force: the plane-reset fix on native takeover (F1) and its test. Reconstruction (adaptive edge refinement, or temporal upsampling with jitter and motion vectors) would keep native quality, but is not built: it would need its own P0, gated on error against native output and on the user's eyes.

### What the hardware offers (`tests/p0_3_plane_inventory`, `planes.json`)

- The DSI CRTC can use 1 primary, 16 overlay and 1 cursor plane.
- Every plane has alpha (0–65535), blend mode (pre-multiplied, coverage,
  none), zpos 1–17, rotation (0/180, reflect x/y), a scaling filter
  (Default or Nearest Neighbor) and 39 formats including YUV.
- Atomic modesetting is available. `hvs_underrun` in debugfs counts underruns.

Evidence: `tests/evidence-p0-3-20261116/`.

### What composition and scaling cost (`tests/p0_3_hvs_probe`, `hvs_matrix.jsonl`)

- **Method:** atomic commits of CPU-filled dumb buffers (no GPU, no Mesa),
  one nonblocking commit per frame with its page-flip event, 3 runs × 600
  frames per configuration, governor `performance`.
- **Result** (free for the display, not for memory: 16 full-screen overlays
  cut a concurrent 4-thread memcpy from 3.63 to 3.13 GB/s; one scaled
  400×240 plane reads 4× fewer bytes than native): all 66 runs ran at 60.09 FPS with 0 missed refreshes, 0 HVS
  underruns and 0 failed commits:
  - primary scaled from 800×480, 600×360, 400×240, 266×160 and 200×120 to
    800×480, Default and Nearest filters;
  - 1, 4, 8 and 16 moving 64×64 alpha sprites;
  - 1–16 full-screen 800×480 ARGB overlays;
  - 4 moving 256×256 sprites, 4 sprites scaled 2×, 8 full-screen overlays
    scaled 2×.
- **Commit CPU (calling thread):** ~29 µs, plus ~3.6 µs per moved sprite
  plane (16 sprites: 87 µs). Kernel-worker CPU was not measured here.

### The lever: render smaller, scale at scanout

- **Legacy flips keep the scaling.** After one atomic modeset that scales
  the primary plane, `drmModePageFlip` (what `hs_graphics_present` uses)
  flipped 400×240 buffers onto the 800×480 panel with 0 misses; the kernel's
  plane state mid-run read `src 400x240`, `crtc 800x480`
  (`legacy_flip.jsonl`, `legacy_state_midrun.txt`). So the scaled path
  reuses the tuned pacing code unchanged.
- **GPU throughput of test_08_raycast_orbit** (`tests/p0_3_render_scale.sh`,
  2 runs each, identical offscreen and on the panel):

  | render size | offscreen FPS | on the panel (HVS-scaled) FPS | vs native |
  |---|---|---|---|
  | 800×480 | 110.7 | 110.4–110.5 | 1× |
  | 600×360 | 195.3–195.4 | 195.1 | 1.76× |
  | 400×240 | 433.2–433.4 | 426.9–432.2 | 3.9× |
  | 266×160 | 942.5–942.7 | 930.4–940.0 | 8.5× |

  Throughput tracks pixel count (1.78×, 4×, 9× fewer pixels), so the scene
  is fill-bound and the HVS scaling adds nothing measurable.

### Made: `hs_graphics_init_scaled(gfx, w, h)` (opt-in)

- Renders at w×h and takes the display; takeover is one atomic modeset that
  stretches the primary plane to the mode and turns off other planes on the
  CRTC. Flips after that are the existing legacy path.
- `hs_graphics_init_sized` keeps its meaning (a non-panel size is an
  offscreen target). Each dimension must be at most the display's; the
  panel's own size behaves like `hs_graphics_init`.
- It is opt-in, not the default: a scaled image is softer, and only the
  application knows whether that trade is right.
- **Gate:** `tests/test_present_scaled` in `make present-hardware` (see
  red team 1 for what it checks after hardening).

### What the HVS outputs (`tests/p0_3_writeback_check`, `writeback/`)

The writeback connector captures what the HVS composes, compared with a CPU
reference. It reaches only CRTC 0, so it checks the HVS, not the DSI path
after it.

| case | result |
|---|---|
| unscaled 800×480 | exact |
| 400×240 → 800×480 and 200×120 → 800×480, Nearest | sample mapping is plain replication; each pixel leaks ~1 LSB into its neighbours, so a full-swing pattern is off by ≤2 LSB |
| same, Default filter (what `init_scaled` uses) | a separable 4-tap cubic-like filter with negative lobes, not bilinear. The image sits ~0.38 output px (2×) or ~0.27 px (4×) right and down of a centre-aligned scale. No standard kernel matched; a fitted cubic (B=0.25, C=0.35, offset −0.44 source px) is within 2–3 LSB on line profiles |
| pre-multiplied overlay over primary, alpha 0x80 and a 0–255 ramp | within 1 LSB of src + dst·(1−a) |
| coverage blend and plane alpha | standard DRM formulas within 1 LSB |
| 4 overlapping overlays, zpos 3,1,4,2 | exact painter's order |

### Red team 1 (P0-3)

An independent agent attacked the change (tools `tests/redteam-p0-3/`,
evidence `tests/evidence-p0-3-redteam-20261116/`). It reproduced claims
1–5 (throughput 110.5 / 432.2 FPS on the panel; legacy flips keep the
scaling over thousands of flips; GBM scanout buffers are linear, like the
probe's dumb buffers). Findings and what was done:

| id | finding | sev | remediation and result |
|---|---|---|---|
| F1 | Takeover inherited the primary plane's alpha, filter and blend mode from the previous DRM client; the console's restore resets only rotation. After the author's own probe, `init_scaled` ran in Nearest; the native path inherited alpha and blend the same way | med | **Both takeovers now reset alpha, blend mode, filter and rotation** (scaled: in its one commit; native: only properties that differ, normally none). Native takeover also turns off stray planes on the CRTC, which `SetCrtc` never did. Measured takeover: 10.3 → 11.0 ms median at the native path (10 runs each, ranges 8.6–24.4 vs 8.9–25.7), scaled 11.9 ms (`takeover_cost.jsonl`). The HVS probe restores what it changes |
| F2 | `test_present_scaled` let 6 of 11 mutants through (offset rects, filter, alpha, stray planes, missing ACTIVE or MODE_ID) | med | The test now first sabotages the display from a second DRM client (quarter alpha, Nearest, coverage blend, a stray overlay; or the CRTC off with no mode and the connector detached), then checks every rect and property, the stray overlay, `ACTIVE` and 10 paced flips. `tests/p0_3_mutants.sh`: **14 of 14 mutants fail, the control passes** (`mutants.txt`) |
| F3 | The test never exercised held or paced frames | low | Added a calibrated heavy phase (~0.6 refresh of GPU time at the render size, held by the pacing code): 0 repeats in 182 refreshes at 400×240, 600×360 and 800×480 |
| F4 | The Default filter's edges are uneven at 2×: a 1-px line at the left/top keeps 115% of a mid line's energy, at the right/bottom 79% | low | Documented in `hs_graphics_init_scaled`'s comment |
| F5 | A fractional SRC_X needs SRC_W shrunk by the same amount (else ENOSPC); 0.41 px centres the middle but changes the scale | low | Not changed; a (w+1)×(h+1) buffer with a 0.4 px offset could correct the shift, unmeasured |
| F6 | Odd render widths cost ~0.17 ms per frame (799×479: 1265 vs 1603 FPS on a trivial frame); the cost disappears under load | low | Documented: use a width that is a multiple of 8 |
| F7 | While another process holds the DRM device open, exit leaves the panel black (no last close, so no console restore); same for native | low, pre-existing | Not changed |
| F8 | "Free" needs a qualifier: HVS reads use shared memory bandwidth | low | Reworded above |

Checked and fine by the red team: 1×1 to 800×480 sizes including odd and
non-aspect ones (91 of 91 refreshes each, console restored), a forced
atomic-commit failure (clean error, no crash, no leak), SIGKILL mid-run
(console comes back; native and scaled clients take the panel after),
pacing of scaled vs native frames at equal frame times (no difference).

**Gates after remediation:** `check-sources` 0, build 0 warnings, `make
test` 0, `make present-hardware` 6 of 6, resets 205 → 205, HVS underruns
0 → 0.

### Layers and sprites: capable, not built

The HVS composes 16 full-screen alpha overlays or 16 moving sprites at no
measured cost, and blends exactly. No NeoGPU workload today spends GPU time
that layers would save: a full-screen pass costs ~0.26 ms at 0.68 ns/px, and
the HUD and text cost is CPU per draw (P0-2). The natural first user is a
full-resolution HUD over a scaled scene.

### Not measured

- Displays other than the 800×480 DSI panel (HDMI was not connected).
- Kernel-worker CPU for atomic commits.
- What the DSI panel shows after the HVS (writeback reaches only CRTC 0).
- `hs_backend_gles` has no way to request a scaled render size yet.

## P0-5: persistent QPU mailbox (built and red-teamed; wins on deep arithmetic only)

Probe: `tools/p0_5_mailbox_probe.c` (`make tools/p0_5_mailbox_probe`). Raw
outputs on the Pi: `tests/evidence-p0-5-20261116/`. The probe runs a resident
QPU program as a chain of "leases": each is one CSD job that polls a sequence
word with TMU op 2, sums a payload read from a ring, publishes the result with
op 2, and ends after a poll budget or a message cap, well under the ~1 s job
reset. The next lease is already queued and starts from the state the
previous one stored. Every payload is checked in all 16 lanes.

**Payload coherence: the GPU fetches 256-byte blocks.** A ring slot the GPU
has not read during this lease is seen correctly with a plain load, but only
when slots are 256 bytes:

| slot size | wrong payloads (of 1000) |
|---|---|
| 64 B | 749 |
| 128 B | 499 |
| 256 B | 0 |
| 512 B | 0 |
| 64 B, op 2 before each load | 0 |

Reading one slot pulls its 256-byte block into the GPU's cache, so the next
slots in that block are stale when their messages arrive. Two correct
recipes: fresh 256-byte slots with plain loads (no slot reused within a
lease), or op 2 on each word before loading it.

**Alone on the GPU (10 s runs, 7.3 M messages, 0 lost, 0 wrong, 0 resets):**

| recipe | RTT p50 | p99 | p99.9 | rate |
|---|---|---|---|---|
| 64-word payload, fresh 256 B slots | 1.50 µs | 1.83 µs | 6.24 µs | 332 k/s |
| 16-word payload, op 2 reload | 1.37 µs | 1.67 µs | 6.26 µs | 430 k/s |

- A lease of 300,000 polls lasts ~95 ms.
- **Lease hand-over:** a message that arrives between leases waits for one to
  end and the next to start: 52–182 µs, about once every 100 ms. Nearly
  every message over 20 µs crossed a hand-over (810 of 989, 974 of 1081).
- For comparison, a job round trip is 22–75 µs on NeoGPU's direct path and
  83.9–105.1 µs on Mesa (VS_MESA.md).

**Beside a Mesa renderer** (`test_08` offscreen at 800×480, GPU-bound;
baseline 110.7 FPS in every round):

| poller | test_08 FPS | mailbox p50 | p99 | lease | resets |
|---|---|---|---|---|---|
| none | 110.7 | | | | |
| idle leases, no messages (2 rounds) | 106.7 | | | ~290 ms | 0 |
| polls back to back, messages every 100 µs (2 rounds) | 106.9 | 17.4 µs | 36.3 µs | | 0 |
| ~1 µs ALU wait between polls | 106.8 | 14.0 µs | 36.1 µs | 87 ms | 0 |
| ~1 µs wait with a thread switch per step | 109.9 | 38.7 µs | 179 µs | 283 ms | 0 |
| ~10 µs ALU wait | 100.6 | 22.7 µs | 46.3 µs | 64 ms | 0 |
| ~10 µs wait with thread switches | 110.4 | 46.8 µs | 1162 µs | 1300 ms | **2**, 1 message lost |

What this means:
- A resident poller costs a GPU-bound renderer 3.4–3.6%.
- While Mesa keeps the GPU busy, the mailbox's latency rises ~12×, to
  14–18 µs median. That is still below a job round trip, but far from 1.5 µs:
  the poller only runs when the renderer's shader threads yield.
- Thread switches in the wait give the renderer its time back, but they
  starve the poller. Its lease grew past the ~1 s reset and the GPU was
  reset twice (207 total now, from 205). **A lease bounded by a poll count
  is not bounded in time under contention.** A shipped mailbox must keep
  every lease far under the reset under the worst contention it can meet,
  or have a time bound measured to hold.

### Does a message that triggers work beat the CPU and a GPU job?

`tools/p0_5_mailbox_work.c` runs the same gather three ways, 200 trials per
size, every output word checked (0 wrong, 0 lost, 0 resets in every row;
`work.jsonl`, `work2.jsonl`):
- **cpu:** one Cortex-A72 core, data in ordinary memory;
- **job:** `hs_qpu_kernel_gather` through `hs_v3d_dispatch` and a wait
  (8 batches, k = 8 from 1024 elements; 1 batch, k = 1 below), data already
  in the BO;
- **mailbox:** a message to a resident lease (1 batch, k = 1) that gathers
  and publishes each output line with op 2, data already in the BO.

Median time in µs:

| elements | sequential idx, 64 MiB table: cpu / job / mailbox | random idx, 512 KiB: cpu / job / mailbox | random idx, 64 MiB: cpu / job / mailbox |
|---|---|---|---|
| 16 | 0.20 / 71.0 / 2.11 | 0.56 / 70.8 / 2.39 | 0.56 / 71.2 / 2.57 |
| 64 | 0.32 / 71.2 / 3.20 | 1.30 / 71.7 / 4.43 | 2.04 / 69.9 / 5.57 |
| 256 | 0.67 / 69.7 / 8.26 | 1.91 / 78.8 / 13.2 | 6.63 / 82.9 / 18.5 |
| 1,024 | 2.15 / 63.7 / 28.4 | 5.22 / 87.4 / 48.2 | 28.1 / 102 / 69.4 |
| 4,096 | 5.91 / 67.8 / 110 | 21.7 / 130 / 190 | 116 / 210 / 276 |
| 16,384 | 20.7 / 88.6 / 435 | 100 / 313 / 755 | 521 / 660 / 1102 |
| 65,536 | 89.2 / 195 / 1734 | 445 / 1001 / 3012 | 2250 / 2281 / 4409 |

Sequential indices into a 512 KiB table matched the 64 MiB column within 10%.
A reduction (per-lane sums, a 16-word result, indices written before the
leases, so no reloads and no output to publish) took 1.69 µs at 16 elements
and 2313 µs at 65,536, against 0.63 and 408 µs on the CPU.

What this means:
- **The mailbox beats a job on small messages:** 27–34× at 16 elements, and
  still ahead at 1,024. A job costs 64–71 µs before it does any work.
- **The CPU beats both at every size here.** At these sizes the data sits in
  the CPU's caches. The mailbox's 1.4 µs round trip alone is more than the
  CPU needs for 256 sequential elements.
- **A faster mailbox kernel would not change that.** The job runs the
  shipped kernel at 8 batches and k = 8. Between 16,384 and 65,536 elements
  it costs 2.2 ns per element on sequential indices, against the CPU's
  1.4 ns, and 33 ns against 35 ns on random indices into 64 MiB. A mailbox
  can save the job's fixed cost, not its per-element cost.
- The GPU's measured gather win (612.8 against 245.2 M/s, VS_MESA.md) is at
  64 MiB of sequential traffic, where a job's fixed cost is 1% of the run.

**For gathers the mailbox does not beat the CPU at any size.**

### Arithmetic: `hs_mailbox` on 8 batches against 4 CPU cores

`include/hs_mailbox.h`, `src/hs_mailbox.c` and `tests/test_mailbox.c` are a
library form of the lease chain: 1 or 8 batches, echo, gather, a float
"chain" op (y = x; k times: y = y · c1 + c0) and a stop message. The
leases drain when the caller stops sending, so an unused mailbox costs a
renderer nothing. `tests/test_mailbox` passes 26 checks with 0 wrong words,
the chain bit-exact against the CPU (`test_mailbox_3_verify.jsonl`,
`test_mailbox_4_rerun.jsonl`).

The CPU side is `tools/p0_5_chain_cpu.c`: NEON, 8–24 vectors held in
registers, 1 and 4 cores (a spinning pool), fused and unfused, every output
checked. The fastest was the unfused multiply and add (the same arithmetic
as the GPU): 7.2 GFLOPS on one core and 28.7 on four, flat from 8 to 16
vectors. The fused form reached 5.1 and 20.4: GCC wraps each `fmla` in two
register copies. An earlier baseline in `tests/test_mailbox bench` (fused,
`-O2`) reached only 17.6 on four cores, so its figures are not used here.
Since 2026-11-17 that benchmark uses the unfused form too (28.6 on four
cores, checked).

Median time in µs (mailbox: `bench_2_rerun.jsonl`, 3 rounds; CPU:
`chain_cpu_nofuse.jsonl`, 21 reps; 0 resets):

| elements × steps | mailbox, 8 batches | CPU, 4 cores | CPU, 1 core | mailbox vs 4 cores |
|---|---|---|---|---|
| 256 × 16 | 4.67 | 0.78 | 1.30 | 0.17× |
| 256 × 256 | 7.82 | 5.09 | 18.4 | 0.65× |
| 256 × 4,096 | 59.1 | 73.4 | 291 | **1.24×** |
| 4,096 × 16 | 21.1 | 5.37 | 19.5 | 0.25× |
| 4,096 × 256 | 70.8 | 73.8 | 293 | 1.04× |
| 4,096 × 4,096 | 890 | 1170 | 4683 | **1.31×** |
| 65,536 × 16 | 295 | 78.2 | 311 | 0.27× |
| 65,536 × 256 | 1096 | 1180 | 4720 | 1.08× |

- The GPU peaks at 37.7 GFLOPS (8 batches) against the CPU's 28.7 (4 cores).
- The mailbox needs deep work to get there: at 16 steps per element it does
  1.8–7.1 GFLOPS, because each 32-element block pays its loads, its store
  and its publish.
- The mailbox's caller spins one core while it waits; the CPU figure uses
  all four.
- An echo costs 1.93 µs on 1 batch and 3.07 µs on 8.

The figures above are from the library as first built (commit `be3f863`).
The red team below changed it; the arithmetic table was re-measured
afterwards and is unchanged within 1% (59.4, 71.1, 893 and 1,097 µs for the
four winning rows; `final_bench.txt`).

### Red team (P0-5), 2026-11-17

- **Target:** `hs_mailbox` as committed in `be3f863`.
- **Method:** two independent reads of the code (one of the original, one
  of the rewrite, which found 7 more defects); `tests/redteam-p0-5/rt_mailbox`
  (functional cases, lease timing, soaks, and runs beside
  `tests/test_08_raycast_orbit`); `tools/p0_5_batch_id_probe`; mutation
  tests (`tests/redteam-p0-5/mutants.py`).
- **Evidence:** `tests/evidence-p0-5-rt-20261117/` (`before_*` is the
  library at `be3f863`, `final*` the final code, `after*` the rounds
  between).
- **GPU resets across all of it:** 207 → 207. HVS underruns: 0. (The
  verification further down then caused 4 on purpose.)

**Finding 1, in shipped code: the batch number came from the QPU thread
index.** Every 8-batch program (the gather kernel and the mailbox) computed
its batch as `(tidx >> 2) & 15`. That gives each batch its own number only
while nothing else uses the GPU.

| 8 batches | alone | beside `test_08` |
|---|---|---|
| jobs in which every batch got its own number, thread index | 62,265 of 62,265 | 173 of 78,894 |
| the same, `rf0` (workgroup x) | 65,111 of 65,111 | 79,177 of 79,177 |
| shipped gather kernel, jobs with wrong words, before | 0 of 2,818 | 3,273 of 3,279 |
| shipped gather kernel, jobs with wrong words, after | 0 of 2,290 | 0 of 3,206 |
| mailbox, 8 batches, before | correct | requests time out after 1 s |
| mailbox, 8 batches, after | correct | correct (0 wrong in 44,441 + 28,935 requests) |

- The hardware presets a batch number at program entry. With workgroup
  counts batches × 1 × 1 (what Mesa submits), `rf0` holds 0..7, each once
  (`batch_id_show.txt`).
- **Fix:** `hs_v3d_dispatch` now submits batches × 1 × 1;
  `hs_qpu_kernel_gather` and `hs_mailbox` read `rf0 & 15` before they
  overwrite `rf0`.
- Alone on the GPU nothing changed: the 8-batch gather job takes 1,575 µs
  for 65,536 elements (1,579 before).
- Every earlier 8-batch figure in this tree was measured alone and stands.
  1-batch programs were never affected.

**Finding 2: a lease was bounded by a poll count, and a poll is not a
fixed time.** (`before_lease.txt`, `before_coexist.txt`)

| idle lease | 1 batch | 8 batches |
|---|---|---|
| 100,000 polls, alone | 31.9 ms | 43.7 ms |
| 1,000,000 polls, alone | 320 ms | 437 ms |
| 100,000 polls, beside `test_08` | 244–248 ms | did not finish (finding 1) |
| 20,000 polls, beside `test_08` | 47–51 ms | did not finish (finding 1) |

- Alone, a poll costs 0.32 µs on 1 batch and 0.44 µs on 8, linear in the
  budget. Beside the renderer it costs 2.4 µs.
- The library accepted budgets up to 4,000,000 polls: 1.3–1.7 s alone, past
  the kernel's job reset.
- **Fix:** the default budget is 30,000 polls and the largest is 250,000.
  The poll loop reads three yield words, and a watchdog thread checks every
  1 ms:
  - no lease job has ended for `max_lease_us` (50 ms): every batch ends its
    lease at its next poll, even with requests pending;
  - the caller has not sent or waited for `idle_us` (2 ms): leases end as
    soon as no request is pending.

| after the fix (`final2_*`) | 1 batch | 8 batches |
|---|---|---|
| longest lease, largest budget, beside `test_08`, one echo per ms | 52.7 ms | 53.0 ms |
| the same without the watchdog, alone (reference) | 85.8 ms | 114.9 ms |
| GPU free again after the last call | 2.19 ms (max 2.24) | 2.16 ms (max 2.19) |
| the same without the watchdog, largest budget | 169 ms | 228 ms |

The hardest case for the bound is a caller that never lets the ring empty.
Four of the largest gathers always in flight, budget 100,000, `max_lease_us`
20 ms (`final_pipe.txt`, `final2_pipe.txt`):

| longest lease | 1 batch | 8 batches |
|---|---|---|
| alone, watchdog | 20.6 ms | 22.4 ms |
| alone, no watchdog | 56.0 ms | 241 ms |
| beside `test_08`, watchdog | 24.3 ms | 25.9 ms |

- The watchdog rows are the time between two job ends as the watchdog
  itself measures it (`bound_jobs.txt`, see the verification below). The
  rows without it are the gap between one batch's lease ends as the caller
  sees it.
- The bound is `max_lease_us`, plus one 1 ms tick, plus the requests the
  batches are running when the yield arrives: at most 5.7 ms here.

- Without the thread (a stopped process, or `watchdog_off`) only the poll
  budget ends a lease. At the default budget beside `test_08` the longest
  lease was 76.9 ms on 1 batch and 89.3 ms on 8
  (`after7_coexist_wdoff.txt`); up to 3 can be queued.
- One request cannot be interrupted, so `send` bounds its size: 512 gather
  chunks per batch (8,192 elements on 1 batch, 65,536 on 8) and 65,536
  budget units for a chain. The largest requests take 0.61 ms (gather) and
  2.63 ms (chain) alone on 1 batch, 2.22 and 2.64 ms on 8. The longest
  request in 90 s soaks beside the renderer took 4.7 ms on 1 batch and
  8.0 ms on 8.

**Finding 3: on 8 batches beside a renderer, batches do not all run at
once.** A batch that has served a request and polls for the next holds
back batches that have not started.
- `hs_mailbox_wait` now sets the caller's yield word when some batches have
  acknowledged a request and others have not for 300 µs.
- With it, echo p99 beside the renderer is 0.41 ms; with it removed
  (mutant `m13`), 41.6 ms, with single waits up to 357 ms.

**Findings 4–11, host side** (each confirmed on the unmodified library
where that was safe, `before_cases.txt`; each fixed and covered by a case):

| # | before | after |
|---|---|---|
| 4 | `hs_mailbox_send` checked nothing: a zero step count or a huge element count would loop until the kernel reset the GPU | `send` refuses 17 kinds of bad request (count, alignment, range outside the listed BOs, overlap, unknown op, stop) |
| 5 | after 2³² − 1 requests (2.8 h at 430 k/s) every send failed for good | sequence numbers wrap; 600 checked requests across the wrap |
| 6 | leases the kernel killed were never replaced: every later request timed out | a stalled wait finds the queue empty and resubmits (first request after: 20.3 ms) |
| 7 | a killed lease made the next one start from stale state and replay acknowledged requests from reused slots | a batch's published acknowledgement is its only state: the next lease resumes at the first unacknowledged request |
| 8 | a request could then run twice, wrong for in-place work | output overlapping input is refused |
| 9 | `hs_mailbox_wait` on a sequence never sent spun to its timeout | `-EINVAL` |
| 10 | `stop` dropped unfinished requests' bookkeeping and marked the mailbox stopped even when the stop failed; a stop sent through `send` wedged it | `stop` waits for every request, and only a completed stop changes state; `send` refuses the stop op |
| 11 | `send` published a request before submitting the lease and returned the submit error; `-EBUSY` ignored acknowledgements; `start` used a static buffer | submit first, publish after; `-EBUSY` re-reads the acknowledgements; stack buffer |

**Findings 12–18, from the second read (of the rewrite):**

| # | defect in the rewrite | fix |
|---|---|---|
| 12 | a stop whose wait timed out sent a second stop request when retried; no lease would ever serve it | the stop request is sent once; a retry waits for it; sends are refused meanwhile |
| 13 | the yield was only honoured with no request pending: a caller that kept the ring full held a lease to its budget (241 ms measured on 8 batches) | a third yield word ends the lease with requests pending (table above) |
| 14 | lease counts that differ between batches (a killed job that only some batches finished) stopped all submission | signed accounting against the jobs finished on every batch |
| 15 | gather checked only the table's first word against the output | gather takes the table's length; range and overlap are checked in full |
| 16 | a lease took at most `slots` requests per batch, but a slot one batch had cached could be read by another batch of the same job for a later request | a lease ends where the ring wraps: within a job a slot index means one request for every batch. Reproduced and confirmed fixed in the verification below |
| 17 | the watchdog compared its clock with a time stamped by the caller, and both threads wrote one yield word | the caller bumps a counter; the watchdog times it on its own clock; park and wake use a counter-then-state handshake |
| 18 | the header promised more than the code did about `max_lease_us` | the header states what is measured above |

**What the mailbox costs a renderer** (`test_08` alone: 110.2–110.6 FPS;
`final2_coexist.txt`, `final2_pipe.txt`, `final2_long_coexist.txt`):

| caller | 1 batch | 8 batches |
|---|---|---|
| one echo every 20 ms | 110.2 FPS | 103.8 FPS |
| one echo every 100 µs | 108.2 FPS | 66.8 FPS |
| one 256-step chain every 100 µs | 107.4 FPS | 72.4 FPS |
| four of the largest gathers always in flight | 103.8 FPS | 85.8 FPS |
| 90 s of mixed requests (9 renderer runs) | 105.1–106.1 FPS | 94.0–100.4 FPS |
| echo latency beside the renderer, p50 / p99 | 12.2 / 20.0 µs | 31.0 / 407 µs |

- A resident 8-batch poller occupies every QPU: use 8 batches when the GPU
  is otherwise free, or in bursts.

**Other compute jobs queue behind the leases** (`final2_jobwait.txt`):

| a plain job submitted right after a mailbox request | 1 batch | 8 batches |
|---|---|---|
| no mailbox | 0.065 ms | 0.072 ms |
| behind the leases, watchdog on | 3.11 ms | 3.11 ms |
| behind the leases, watchdog off | 20.4 ms | 27.5 ms |
| `hs_mailbox_stop` first (stop + job) | 0.157 ms | 0.158 ms |

- Two mailboxes used alternately cost each other 2.7–3.0 ms per call.

**Platform facts met on the way:**
- `hs_v3d_bo_wait` with a zero timeout blocks 1.9–2.0 ms while a job holds
  the BO (`after3_bowait.txt`). The library only calls it with the caller
  away or on a 20 ms stall.
- The first request after more than 100–200 ms of GPU idle takes 10.0 ms,
  for a plain job and for the mailbox alike (`after4_idlegap.txt`). This is
  the runtime resume recorded in VS_MESA.md.

**Do the tests have teeth?** `mutants.py` breaks the library one way at a
time (`final_mutants.txt` in the verification evidence): 18 of 18 mutants caught, the unmodified
control passes. They cover the size, overlap and table checks, the wrap
compare, lease replacement and accounting, the watchdog and its age yield,
the ring-wrap lease end and the earlier per-batch slot rule, index reload,
result publish, the single stop request, and the thread-index batch number
in the mailbox and in the gather kernel.

**Final state** (`final*`, and `tests/evidence-p0-5-verify-20261117/`
after the verification below):
- `tests/test_mailbox`: 26 checks, 0 wrong words.
- `rt_mailbox cases`: 47 cases on 1 and 8 batches, all pass.
- Soaks of random checked requests, pauses and stops: 60 s alone and 90 s
  beside back-to-back renderer runs, on 1 and 8 batches: 0 wrong, 0 lost
  leases.
- Cost of the hardening: an echo is 2.09 µs on 1 batch and 3.33 µs on 8
  (1.93 and 3.07 before). The benchmark's CPU chain is now the fastest one
  found (28.6 GFLOPS on four cores, checked).

**Where this leaves P0-5:**
- The mailbox is a real GPU arithmetic path at microsecond latency, worth
  1.2–1.3× all four CPU cores on deep per-element arithmetic and nothing on
  gathers or shallow work.
- It is now safe to leave running: bounded leases, bounded requests, checked
  arguments, and it gives the GPU back 2 ms after the caller goes quiet.
- Nothing in NeoGPU calls it yet.

### Verification of the open points (P0-5), 2026-11-17

The red team left three things unverified. Each was then measured
(`tests/evidence-p0-5-verify-20261117/`).

**1. A real kernel reset of a lease.** `rt_mailbox reset` sends, past the
checks, a chain request that needs about 4 s, with three ordinary gathers
queued behind it. While the lease spins, the test rewrites the request's
step count in its slot to 4, so a lease that runs it again can finish.
`rt_mailbox foreign` has a second client submit a job that never ends while
checked gathers keep flowing (`reset_real.txt`, `reset_foreign.txt`).

| real reset | 1 batch | 8 batches |
|---|---|---|
| lease killed mid-request: all four requests acknowledged after | 1,007 ms | 1,014 ms |
| the killed request's output | the 4-step result, bit-exact | the 4-step result, bit-exact |
| the three queued gathers, wrong words | 0 | 0 |
| 4,000 requests, a stop and a restart afterwards | 0 wrong, all rc 0 | 0 wrong, all rc 0 |
| another client's job hangs: longest request | 1,010 ms | 1,005 ms |
| the same: requests sent, wrong words | 298,276, 0 | 46,192, 0 |

- GPU resets: 207 → 211, one per run, all four deliberate.
- The kernel resets a job after 1.0 s without progress and then runs what
  was queued. The next lease starts from the acknowledgements and runs the
  killed request again: the request ran twice, which is why outputs may not
  overlap inputs.
- The kernel did not drop the queued leases in any of the four runs
  (`lost_leases` stayed 0). The path that replaces lost leases is still only
  exercised by the simulated cases.
- A request that can never finish would be killed and re-run for ever. Only
  the size limits in `hs_mailbox_send` prevent that.

**2. Is a slot cached by one batch served stale to another?** Yes.
`tools/p0_5_batch_cache_probe` runs one 8-batch job per trial: some batches
read a block, the CPU rewrites it, then every batch reads it again
(`batch_cache.txt`, 300 trials, the same alone and beside `test_08`):

| who read the block before the CPU rewrote it | batches 1–7 then read |
|---|---|
| nobody | fresh, 2,100 of 2,100 |
| batch 0 only | **stale, 2,100 of 2,100** |
| every batch | stale, 2,100 of 2,100 |
| batch 0 only, with op 2 before the second load | fresh, 2,100 of 2,100 |

- The read cache belongs to the job, not to the batch or the QPU thread:
  alone, all 2,100 stale reads were on a different thread from batch 0's.
- The library fault was then reproduced. The case "a batch one request
  behind the others, 2 slots" starts batch 0 one request behind, as after a
  lease it ended early. Under the earlier rule (mutant `m19`) the other 7
  batches ran the cached older request: 22,400 wrong words, 448 of 512 in
  each of 50 rounds. With leases ending at the ring wrap: 0
  (`mutant_old_slot_rule.txt`, `cases8.txt`).
- No other case caught the earlier rule: the batches only fall out of step
  when one ends a lease early.

**3. Does the lease bound stretch under the hardest load?** Less than
reported. The 62.5 ms was the gap between one batch's lease ends as the
test's thread saw them, and that gap can span two jobs: a batch that ends
early in one job and late in the next. The watchdog now records the time
between job ends itself (`bound_jobs.txt`; 4 gathers always in flight,
budget 100,000, `max_lease_us` 20 ms):

| beside `test_08`, 8 batches | largest gathers | 64 chunks | 1 chunk |
|---|---|---|---|
| longest job | 25.9 ms | 6.4 ms | 20.8 ms |
| longest time from the yield to the job's end | 5.7 ms | no yield needed | 0.6 ms |
| gap between one batch's lease ends (the earlier measure) | 56.1 ms | 10.2 ms | 21.5 ms |

- The job never ran more than 5.9 ms past the limit. The excess is one
  1 ms tick plus the requests in progress when the yield arrives.
- On 1 batch with the largest gathers the longest job was 24.3 ms beside
  the renderer and 20.6 ms alone.

Not measured at that point (the first two and part of the third are in the
next section):
- A reset while a renderer is running, and what the renderer sees of it.
- The lost-lease path against a kernel that really drops queued jobs (this
  kernel did not).
- Any renderer other than `test_08`; 2 or 4 batches; a job-based chain
  kernel (none exists); CPU code hand-written in assembly.
- Gather indices are not checked by the library: an index outside the table
  reads whatever that GPU address holds.

### Bedrock: resets, lost leases and what compute costs a renderer (P0-5), 2026-11-17

Evidence: `tests/evidence-p0-5-bedrock-20261117/`. Kernel 7.0.0-1023-raspi;
its `v3d_sched.c` and the scheduler's `sched_main.c` were read
(raspberrypi/linux `rpi-7.0.y`) and every step below was then seen in a
trace: `tests/redteam-p0-5/reset_trace.bt` records the kernel's V3D events,
`reset_trace.py` summarises a trace. GPU resets this round: 211 → 240, 23
planned and 6 not (see "Unplanned resets").

**1. How the kernel times a job out** (`reset1_q2.trace`, `reset8_q2.trace`).

- Each queue (bin, render, compute, ...) runs one job at a time on the
  hardware and has its own 500 ms timer, started when the job starts.
- First expiry of a compute job, 501–503 ms after its start: the handler
  reads the hardware's batch counter, sees it differ from the 0 the job
  struct starts with (the counter read 0xFFFFFFFF on 1 and on 8 batches),
  and re-arms. Every compute job gets this pass once.
- Second expiry, 1,004–1,006 ms after the start: the counter is unchanged,
  so the GPU is reset. A compute job is therefore killed about 1.0 s after
  it starts if no batch finished in its second 500 ms, whatever it was doing.
- The reset itself took 35–55 µs in 20 of the 25 resets summarised and
  12.4–17.5 ms in the other 5.
  The whole handler took about 0.55 ms.
- The job the handler ran for is marked guilty, its fence gets `-ECANCELED`,
  and the kernel does not run it again. Jobs in flight on the other queues
  are run again from their start. Jobs still queued are untouched: only one
  job per queue is on the hardware, and this driver gives the scheduler no
  per-client guilty flag.

**2. The lost-lease path, against the real kernel** (`reset1_q1.*`,
`reset8_q1.*`). With `qdepth` 1 there is no second lease queued, so when the
kernel kills the lease nothing is left to run.

| qdepth 1, lease killed mid-request | 1 batch | 8 batches |
|---|---|---|
| `lost_leases` counted by the library | 1 | 1 |
| reset to the replacement lease's submit | 13.0 ms | 13.8 ms |
| all four requests acknowledged after | 1,020.9 ms | 1,021.3 ms |
| wrong words (killed request, queued gathers, 4,000 after) | 0 | 0 |

- The 13 ms is the library's 20 ms lost-lease check period.
- With `qdepth` 2 (the default) the queued lease started 0.10–0.13 ms after
  the reset and `lost_leases` stayed 0.
- After a kill the watchdog's yield word is still set, so the first leases
  end at once until its next 1 ms tick: 1 to 13 leases of a few µs each
  were counted. Harmless, bounded by the tick.

**3. A lease hangs while a renderer runs** (`coexist_reset*`,
`coexist_rep_*`; 13 runs: 7 on 1 batch, 6 on 8; `test_08` at 110 FPS).

| what happened | result |
|---|---|
| render jobs finished while the lease hung | 0, in all 13 runs |
| longest time without a finished render job | 1,017–1,023 ms |
| which job the kernel blamed and cancelled | **the renderer's: 9 runs; the lease: 4 runs** |
| first render job finished after the reset | 12.4 ms later |
| renderer afterwards | 108–112 jobs/s, exit code 0, `PASS` |
| the mailbox's results in all 13 runs | 0 wrong, all acknowledged after 1,004–1,022 ms |
| display underruns | 0 |

- The renderer's bin and render jobs were stuck from the moment the hang
  began. Their timers run too, and whichever queue's second expiry comes
  first is blamed. The renderer's job had usually started first.
- When the renderer is blamed its render job is cancelled (the kernel calls
  the run function, which returns no fence: that frame is never drawn), the
  renderer's reset counter goes up, and the hung lease is **run again from
  its start** by the kernel. In these runs the request had been shortened
  by then, so the re-run lease finished it: the "same job runs twice" case,
  9 of 9 correct.
- A compute job that truly never ends (`hang_coexist_*`, 4 runs,
  `p0_5_spin_probe` variant 0): 1 reset in 2 runs (job gone after 1,006 and
  1,010 ms), 2 resets in 2 runs (renderer blamed first; job gone after
  1,524 ms, because the re-run job keeps its old counter reading and is
  killed at its first expiry).
- Not explained: in those two runs the renderer worked during the re-run
  job's 0.5 s (10, 60, 94 jobs/s in three 0.5 s windows), although the same
  job froze it completely before the reset.

**4. The cause of the freeze: a compute job gives way to a renderer only at
thread switches** (`tools/p0_5_spin_probe`, `spin_coexist*`, `stretch_*`).
One compute job, loop body varied, beside `test_08`:

| loop body | per iteration alone | beside the renderer | render jobs meanwhile |
|---|---|---|---|
| arithmetic only | 32.1 ns | 32.0 ns | **0 in 250 ms** (4 of 4 jobs) |
| arithmetic + TMU store + `tmuwt` | 64.1 ns | 64.1 ns | **0 in 250 ms** (4 of 4 jobs) |
| arithmetic + thread switch | 138.8 ns | 1,933–1,945 ns | 108–111 per s (full speed) |
| arithmetic + TMU load (has a thread switch) | 96.1 ns | over 1,570 ns (killed at 1 s) | not measured |

- The same on 1 batch and on 8: one 16-lane batch without a thread switch
  stopped all of `test_08`'s rendering. (With a lighter renderer it is all
  or nothing per job: see the next section.)
- Without thread switches the compute job runs at full speed and the
  renderer gets nothing. With one in every iteration the renderer runs at
  full speed and the compute job is 14 times slower.
- Between those: `stretch` arithmetic-only iterations (32 ns each) between
  two thread switches, 20 ms jobs back to back for 5 s (`stretch_b1.txt`,
  `stretch_b8.txt`):

| stretch between thread switches | compute slowdown (1 / 8 batches) | render jobs per s, 1 batch | 8 batches | longest gap, 1 / 8 batches |
|---|---|---|---|---|
| 0.03 µs | 12.0× / 12.4× | 110.2 | 106.9 | 9 / 15 ms |
| 0.13 µs | 8.9× / 9.2× | 109.9 | 103.7 | 9 / 15 ms |
| 0.5 µs | 4.7× / 4.8× | 109.4 | 94.5 | 9 / 15 ms |
| 2 µs | 2.4× / 2.3× | 106.3 | 69.2 | 9 / 17 ms |
| 8 µs | 1.43× / 1.61× | 100.5 | 43.9 | 10 / 33 ms |
| 33 µs | 1.10× / 1.09× | 87.1 | 10.0 | 12 / 102 ms |
| 131 µs | 1.02× / 1.02× | 53.4 | 3.2 | 19 / 321 ms |
| 524 µs | 1.003× / 1.004× | 20.0 | 1.4 | 54 / 787 ms |

- A second renderer, `test_05_cube3d` (19.7 render jobs per s, not
  GPU-bound; `cube/stretch_b1.txt`): its rate stayed 19.6–19.8 at every
  stretch and the compute job was not slowed; only its longest gap grew,
  from 51 ms to 60 ms (131 µs) and 74 ms (524 µs).

**5. What this means for `hs_mailbox`** (`chainload_*`). Polls and gathers
switch threads at every load. A chain request does not: each pair of chunks
runs its k steps with no thread switch (one switch per pair).

| chain requests back to back, beside `test_08` | longest stretch | renderer, 1 batch | 8 batches |
|---|---|---|---|
| 1,024 pairs per batch, k = 232 | 3.7 µs | 104 FPS | 60–104 |
| 1 pair, k = 4,096 | 59 µs | 88–90 | not measured |
| 1 pair, k = 65,536 (the largest k) | 0.90 ms | **36–38, frames up to 61 ms apart** | 66–84, up to 29 ms apart |

- 110 FPS without the mailbox. One request in 64 was checked against the
  CPU, all correct; the mailbox was busy 77–86 % of the time on 1 batch and
  44–53 % on 8 (the caller's thread limits it there).
- The request with short stretches took 7.2 ms beside the renderer against
  3.7 ms alone: the renderer takes its turn at every pair.
- The earlier "107–108 FPS on 1 batch" was measured with echoes and gathers.
  It does not hold for chain requests with a large k.
- Changed in the next section: the chain loop now switches threads every
  128 iterations, after the cost was measured in the chain kernel itself.

**Unplanned resets.** Six of the 29 resets were my probe jobs: sized to
250 ms alone, the thread-switching bodies ran 14× slower beside the renderer
and passed 1 s. The script now sizes them for the slow case; the TMU-load
body was killed again at that size, so its speed beside a renderer is only
a lower bound.

**Gates:** build 0 warnings, `check-sources` 0, `rt_mailbox cases` 47 of 47
(`final_cases.txt`). The library's code is unchanged this round; its header
comment was corrected.

Left unmeasured at that point, and measured in the next section: what a
cancelled frame holds and what GL reports; why the re-run compute job did
not freeze the renderer; a hang in a renderer's own job; a chain kernel with
thread switches; 2 and 4 batches.

### Bedrock, second round: the open list closed (P0-5), 2026-11-17

Evidence: `tests/evidence-p0-5-bedrock2-20261117/`. New tool:
`tools/p0_5_gl_probe`, an offscreen GLES renderer (render node, no display)
that fills each frame with its own colour, reads every frame back, compares
every pixel and asks GL for the reset status. GPU resets this round:
240 → 256, 15 planned and 1 not (below).

**1. What a renderer is left with when the kernel cancels its job**
(`gl_hang.txt`; `gl_probe` at 512×512, 89 FPS, beside a compute job that
never ends; 4 runs).

| | renderer blamed (2 runs) | compute job blamed (2 runs) |
|---|---|---|
| the frame in flight | **partly drawn**: 106,496 and 258,048 of 262,144 pixels new, the rest the previous frame | complete and correct |
| that frame took | 1,015 and 1,017 ms | 20 and 24 ms |
| GL reset status | guilty | innocent |
| every GL call afterwards | fails with `GL_CONTEXT_LOST` | fails with `GL_CONTEXT_LOST` |

- The cancelled frame is torn at tile boundaries (both counts are multiples
  of 4,096, a 64×64 tile): the tiles finished before the reset are new.
- `gl_probe` asks for a robust context that is lost on reset. Such a
  renderer loses its context after **any** GPU reset, guilty or not, and has
  to rebuild it. `test_08` does not ask, and kept rendering (previous
  section).

**2. The freeze is all or nothing per job, and depends on where the batch
lands** (`flight.txt`, `tidx.txt`; one 250 ms arithmetic-only job at a time).

| renderer | 1 batch: jobs that froze it | 8 batches |
|---|---|---|
| `test_08` (GPU-bound) | 21 of 21 (previous section) | every job |
| `gl_probe` 1024×1024, 121 ms frames | 12 of 12 | 11 of 11 |
| `gl_probe` 512×512, 10 ms frames, idle between | **5 of 12, then 17 of 30** | 12 of 12 |

- A job that froze the renderer let 0 render jobs start or finish. A job
  that did not let it run at full speed (20–22 render jobs in 250 ms).
  Nothing in between.
- Whether a render job was on the hardware when the compute job started
  does not decide it (true in both groups). The batch's QPU thread index
  does not decide it either: index 28 froze in 1 of 5 jobs, 29 in 2 of 4,
  31 in 4 of 4, 12 in 0 of 4.
- This answers the open point of the previous section: a job the kernel
  runs again after a reset starts on an idle GPU, and it can land clear of
  the renderer.
- On 8 batches beside the light renderer the compute job itself took 250,
  499 or 749 ms for 250 ms of work: batches wait for each other. One such
  job passed 1 s and was reset: the unplanned reset of this round.
- Interpretation, not measured: a QPU runs two threads and hands over only
  at a thread switch, so a batch without switches blocks whatever else has
  a thread on its QPU. The measurements above are consistent with that and
  do not prove it.
- A TMU load per iteration beside `test_08`: 1,674–1,945 ns per iteration
  (96 ns alone), renderer at 110.4 FPS (`tmu_load_beside.txt`).

**3. The renderer's own job hangs while the mailbox is in use**
(`renderer_hangs.txt`; `gl_probe` with one frame whose shader loops 10^8
times per pixel; `rt_mailbox foreign` without its own hung job; 4 runs).

| | 1 batch (2 runs) | 8 batches (2 runs) |
|---|---|---|
| resets until the hung render job was cancelled | 3 and 2 | 2 and 2 |
| jobs the kernel blamed first | **the mailbox's lease**, 4 of 4 runs | |
| longest mailbox request | 2,021 and 1,508 ms | 1,496 and 1,523 ms |
| requests over 100 ms | 1 per run | 1 per run |
| requests sent, wrong words | 227,679 and 250,335; 0 | 40,338 and 40,546; 0 |
| lost leases counted | 1 and 0 | 0 |

- A hung render job stops the lease too, so the lease's timer expires and
  the kernel cancels the lease and runs the hung render job **again**. Only
  when the render job's own timer comes first is it cancelled.
- The mailbox recovered every time with no wrong word. The watchdog cannot
  help here: the lease is not running, so it cannot see the yield.

**4. A chain loop with thread switches: built, wrong, fixed, adopted**
(`chain_stretch_*`). `hs_mailbox_cfg.chain_stretch` sets how many loop
iterations (4 steps each) run between two switches;
`HS_MB_CHAIN_NO_SWITCH` keeps the earlier loop.

- **A thread switch that really runs another thread does not keep the
  accumulators r0–r3.** The first version was bit-exact alone at every
  setting and wrong beside `test_08` at every setting: 1,216 to 2,176 wrong
  words per run, nearly every checked element
  (`chain_stretch_beside_accumulators_lost.txt`). The loop now reloads c1
  and c0 and redoes the pending multiply after each switch: 0 wrong in all
  10 runs. Registers rf0–rf63 are kept (the whole program depends on it).
- Alone, 1 batch, mean request time:

| stretch | 1 pair, k = 65,536 | 1,024 pairs, k = 232 | 16 pairs, k = 4,096 |
|---|---|---|---|
| no switches | 0.900 ms | 3.745 ms | 0.908 ms |
| 64 | 0.941 (+4.6 %) | 3.781 | 0.946 |
| **128 (default)** | **0.921 (+2.3 %)** | **3.784 (+1.0 %)** | **0.927 (+2.1 %)** |
| 256 | 0.910 (+1.1 %) | 3.784 | 0.915 |
| 512 | 0.905 (+0.6 %) | 3.781 | 0.910 |

- Beside `test_08`, k = 65,536 back to back for 5 s:

| stretch | 1 batch: render jobs per s | longest gap | request | 8 batches: render jobs per s | request |
|---|---|---|---|---|---|
| no switches | 37.0 | 61.2 ms | 0.925 ms | 74.9 | 2.51 ms |
| 64 | 105.7 | 9.6 ms | 1.838 ms | 89.0 | 2.58 ms |
| **128** | **102.8** | **10.0 ms** | **1.382 ms** | **88.4** | **2.27 ms** |
| 256 | 100.2 | 10.3 ms | 1.153 ms | 88.0 | 2.29 ms |
| 512 | 95.0 | 11.1 ms | 1.044 ms | 87.3 | 2.29 ms |

- Against 4 CPU cores with the default (`final_bench.txt`, 8 batches,
  k = 4,096): 60.5 µs against 73.3 µs for 256 elements (1.21×, was 1.24×)
  and 922 µs against 1,171 µs for 4,096 (1.27×, was 1.31×). Echo: 2.14 µs
  on 1 batch, 3.24 µs on 8, unchanged.

**5. 2 and 4 batches** (`cases_b2.txt`, `cases_b4.txt`,
`batches_2_4_beside.txt`). `hs_v3d_dispatch` and `hs_mailbox_start` now
accept 1, 2, 4 and 8.

| | 2 batches | 4 batches |
|---|---|---|
| adversarial cases | 25 of 25 | 25 of 25 |
| beside `test_08`: soak, chain load, gathers always in flight, echoes | 0 wrong, all rc 0 | 0 wrong, all rc 0 |
| renderer during those four loads | 107.6 / 104.2 / 99.9 / 106.3 FPS | 105.1 / 102.9 / 96.4 / 93.7 FPS |
| longest job with gathers always in flight (limit 20 ms) | 24.4 ms | 25.8 ms |

- The job-based gather kernel (`hs_qpu_kernel_gather`) still takes 1 or 8
  only: it was not run on 2 or 4.

**Tests.** A new case, "chain step counts around the stretch boundaries"
(12 step counts, bit-exact), and two mutants: `m20` drops the reload after
the switch and is caught only beside the renderer (1,536 wrong words;
alone nothing notices), `m21` stops clamping the last stretch. Mutants: 20
of 20 caught, control passes (`final_mutants.txt`). Cases: 24 on 1 batch,
25 on 2, 4 and 8, 0 failures. `tests/test_mailbox` passes. Build 0
warnings, `check-sources` 0.

**Dropped from the open list, as not worth a measurement:** a job-based
chain kernel (none exists and nothing needs one) and CPU code hand-written
in assembly (the NEON version reaches 28.6 GFLOPS on 4 cores and is the
comparison used).

**Still not measured:**
- Why a batch freezes a renderer in some jobs and not in others: the
  interpretation in 2 is untested.
- Displayed output. Everything here is offscreen or `test_08`'s frame rate;
  no cancelled frame was looked at on the panel.
- The mailbox with a real NeoGPU workload: nothing calls it yet.

### Can the message be the arithmetic? (P0-5), 2026-11-17

Evidence: `tests/evidence-p0-5-inline-20261117/`. `tools/p0_5_inline_probe`
is a resident program like the mailbox's, in which the request block holds
the operands (32 per batch, read once with plain loads) and the
acknowledgement block holds the results (published with op 2 beside the
sequence number). No buffers are pointed at. It is a probe, alone on the
GPU: no yield words, stop flag, watchdog or thread switches in its loop.
The work is the chain (k steps of y = y * c1 + c0), bit-exact in every
checked message (32 of 2,000 per row, 0 wrong).

Round trip as the caller sees it, median of 2,000, in µs. "Mailbox" is
`hs_mailbox_chain` with one pair per batch and the caller copying its
operands in and its results out, as the inline form does by construction
(mean; `mailbox_chain_one_pair.txt`):

| 1 batch, 32 elements | k = 4 | 16 | 64 | 256 | 1,024 | 4,096 |
|---|---|---|---|---|---|---|
| inline message | 2.24 | 2.39 | 3.04 | 5.59 | 15.83 | 56.80 |
| mailbox chain, caller copies | 3.27 | 3.32 | 4.14 | 6.70 | 17.45 | 60.57 |
| one CPU core | 0.09 | 0.20 | 0.63 | 2.35 | 9.17 | 36.48 |

| 8 batches, 256 elements | k = 4 | 16 | 64 | 256 | 1,024 | 4,096 |
|---|---|---|---|---|---|---|
| inline message | 4.93 | 5.11 | 5.80 | 8.35 | 18.61 | 59.56 |
| mailbox chain, caller copies | 6.89 | 7.10 | 7.87 | 10.38 | 21.15 | 65.05 |
| mailbox chain, buffers left in place | 4.97 | 5.15 | 5.88 | 8.39 | 19.27 | 63.20 |
| one CPU core | 0.41 | 1.26 | 4.76 | 18.41 | 73.04 | 291.50 |
| four CPU cores (`final_bench.txt`) | | 0.81 | | 5.04 | | 73.33 |

- **It works, and it saves a fixed amount:** 1.0–1.1 µs per request on 1
  batch and 2.0 µs on 8 (up to 3.8 µs at k = 4,096, where the mailbox's
  chain also pays for its thread switches and the probe has none). That is
  about 30 % of a small request and 6–8 % of a deep one.
- **An inline message with no arithmetic costs 2.26 µs on 1 batch and
  5.0 µs on 8**, against 2.1 and 3.2 µs for an empty mailbox message: on 8
  batches the caller writes 304 words and reads 256 back through uncached
  memory, 1.8 µs.
- **It does not change who wins.** On 1 batch one CPU core is faster at
  every k measured. On 8 batches both forms lose to one core at k = 64 and
  beat it at k = 256; against four cores both lose at k = 256 and win at
  k = 4,096 (1.23× inline, 1.13× mailbox with copies).
- A caller that can leave its data in GPU buffers (the third row) already
  gets the inline form's cost from the existing chain request.
- Not built into `hs_mailbox`: a slot holds 8 lines, and 8 batches need 19
  (the probe uses 2,048-byte blocks). Not measured beside a renderer, and
  the op-carrying TMU writes (atomic add and the like) were not tried.

## P0-6: QPU compute to KMS without Mesa (the scene runs on the panel, 2.25× Mesa)

Evidence: `tests/evidence-p0-6-20261117/`. Nothing here is a renderer yet:
these are the measurements a compute renderer has to stand on.

**Success condition.** A scene rendered by QPU compute alone, no Mesa in the
process, shown on the panel through KMS, checked pixel by pixel against a
CPU reference, with its frame rate measured against the same scene through
Mesa (`test_08`: 110 FPS). Invariants: the display is held only while a test
runs and restored after; the existing tests keep passing.

### Can one buffer be written by the QPUs and scanned out? (`tools/p0_6_scanout_probe`, `scanout_probe.txt`)

| direction | result |
|---|---|
| vc4 dumb buffer → PRIME → v3d handle → GPU address | works; a compute job filled it, 0 of 384,000 pixels wrong (read back through vc4's own mapping); vc4 accepts it as a framebuffer |
| v3d buffer → PRIME → vc4 | refused at import: `EIO` |

Raw ioctls only: no Mesa, no libdrm. The buffer is plain raster XRGB8888
with a pitch of 4 × width, so a TMU store of 16 words writes 16 pixels.

### How fast can compute write pixels? (`fill_rate.txt`)

One job fills 800 × 480; best of 20, job submit and wait included:

| a `tmuwt` after every | 1 batch | 2 | 4 | 8 |
|---|---|---|---|---|
| store | 1.65 ms | 0.88 ms | 0.50 ms | 0.38 ms |
| 2 stores | 1.01 ms | 0.56 ms | 0.37 ms | 0.37 ms |
| 4 stores | 0.65 ms | 0.41 ms | 0.36 ms | 0.36 ms |
| 8 stores | 0.52 ms | 0.37 ms | 0.36 ms | 0.36 ms |
| 20 stores | 0.43 ms | 0.37 ms | 0.36 ms | 0.36 ms |

- The floor is 0.36 ms per frame, 0.94 ns per pixel, reached from 2 batches
  up: writing pixels is not what limits a compute renderer at 60 or 110 FPS.
- Stores do not need a `tmuwt` each: every pixel was correct with one after
  20 stores.

### First frames on the panel (`tools/p0_6_present_probe`, `present.txt`)

Two vc4 dumb buffers, one compute job per frame, a page flip per refresh.
The process contains no Mesa, EGL or GBM (libdrm for the modeset calls).
The image is a test pattern whose every pixel the CPU can predict.

| 10 s on the DSI panel (800×480 at 60 Hz) | 1 batch | 8 batches (2 runs) |
|---|---|---|
| frames | 602 | 602, 602 |
| render time per frame, median / p99 | 0.692 / 0.706 ms | 0.365–0.366 / 0.386–0.396 ms |
| flip interval, median / max | 16.646 / 16.648 ms | 16.646 / 16.650 ms |
| intervals over 1.5 refreshes | 0 | 0 |
| frames checked, wrong pixels | 21, 0 | 21, 0 each |
| process start to first frame on the CRTC | 52.7 ms | 49.4, 45.0 ms |
| display restored, HVS underruns, GPU resets | yes, 0, 0 | yes, 0, 0 |

- The slowest render of each run, 12–16 ms, is the first job after the GPU
  sat idle (its runtime resume, as measured before).
- The start-up figure includes that wake-up and the modeset. It was not
  split further, and Mesa's start on the same path was not re-measured
  here (79.5 ms for `gbm_create_device` alone, earlier).

### What the scene will cost

- Mesa's fragment shader for `test_08` compiles to **959 instructions, no
  loops** (`V3D_DEBUG=shaderdb`, `mesa_shaderdb_test08.txt`), and Mesa
  renders it at 110 FPS: 23.7 ns of all-QPU time per pixel.
- A compute kernel for the same scene must average that or less per pixel
  to tie. It is free to do what a fragment shader here does not: branch per
  16-pixel vector, skip work no lane needs, and take per-region work lists
  from the CPU through the mailbox.

### Red team of the two probes, 2026-11-17 (`tests/evidence-p0-6-redteam-20261117/`)

An independent read of the code and my own found these; all fixed, each
with a case or an injected fault that shows the fix is tested:

| # | finding | fix |
|---|---|---|
| 1 | The display probe checked only even frames at the default interval, so **buffer 1 was never checked**: the "0 wrong in 21 frames" above covered one buffer | frames n and n+1 are checked together; a run that used both buffers and checked one fails |
| 2 | No error path after taking the display restored it, and signals skipped the restore | one exit path; SIGINT and SIGTERM stop the loop and restore |
| 3 | A saved CRTC that was off would have been "restored" with a zeroed mode | it is turned off again instead |
| 4 | The exit status ignored a failed restore | it does not |
| 5 | Takeover did not reset planes a previous client left changed (the P0-3 finding, repeated) | overlays on the CRTC are turned off and the primary plane's alpha, blend mode, filter and rotation set to defaults, as `hs_graphics` does |
| 6 | Neither probe could see a write outside the frame | 16 guard rows above and below, marked and checked; the framebuffer object covers the visible rows only |
| 7 | Reading the buffer's memory cannot show what the display composes from the framebuffer object | the last frame is composed once more by the display controller into a capture buffer (writeback) and compared |
| 8 | The check sat between render and flip, so its 9–11 ms counted against the flip | checks run after the flip |
| 9 | The first-frame time included a full-frame check; the frame rate counted one frame too many | both corrected |
| 10 | `batches` 0 or a zero `wait_every` divided by zero; a skipped fill exited 0 | arguments validated; a skipped fill fails |
| 11 | The 24-bit multiply in the pattern would diverge from the CPU after about 11 hours | the frame term is masked on both sides |

- `tests/redteam-p0-6/present_cases.sh`: **18 of 18 cases pass**
  (`present_cases.txt`). They cover 1, 2, 4 and 8 batches; every frame
  checked; five injected faults that must be reported (wrong frame number,
  a fault in frames 2–11 only, one loop pass too many, a framebuffer that
  starts at the guard rows); SIGINT, SIGTERM and SIGKILL; a second copy
  refused while the first holds the display; bad arguments; and plane
  properties a previous client left dirty. After every case the kernel's
  dump of CRTCs and planes is the same as before it.
- `tests/redteam-p0-6/mutants.sh`: **9 of 9 mutants caught**, control passes
  (`mutants.txt`). Two earlier rounds are kept: 0 of 8 (my harness passed
  the mutant's path in an environment variable that `sudo` drops, so the
  real probe was tested every time) and 7 of 9 (the memory check and the
  capture covered for each other until each got a fault only it can see).
- The capture and the panel agree: with the framebuffer starting at the
  guard rows the user saw the guard marker as a coloured band above the
  pattern, and the capture reported it.
- After the fixes, same panel: 60 Hz, render 0.366 ms, flip interval
  16.646 ms median and 16.649 ms max, both buffers checked, capture of the
  last frame 0 of 384,000 pixels wrong, first frame on the CRTC 28–46 ms
  after `main` starts.
- Not changed: on SIGKILL the probe cannot restore anything; the kernel
  gave the console back (state unchanged in that case too).

### The special functions (`tools/p0_6_sfu_probe`, `sfu.txt`; plan step 1)

On this hardware reciprocal, reciprocal square root and sin are add-ALU
operations with a destination. Their encodings were read from Mesa's
compiled code for `test_08` (`mesa_fs_test08_qpu.txt`) and added to
`hs_qpu` as `HS_QPU_A_RECIP`, `HS_QPU_A_RSQRT` and `HS_QPU_A_SIN`. A first
version of the probe wrote x to the magic registers 48–53 and read r4, as
older V3D versions do: that does nothing here
(`sfu_magic_registers_do_nothing.txt`).

| 320,000 random inputs, 2^-10..2^10 | exact | within 1 ulp | largest relative error |
|---|---|---|---|
| `recip` against a correctly rounded 1/x | 309,981 (96.9 %) | all | 1.19e-07 |
| `rsqrt` against a correctly rounded 1/sqrt(x) | 313,305 (97.9 %) | all | 1.19e-07 |

- Cost: one instruction slot. Used by the very next instruction, a call
  costs 5.33 ns more than a move in its place (one instruction time);
  used one or more instructions later it costs nothing extra.
- Special values: 1/0 = inf, 1/-0 = -inf, 1/denormal = inf;
  rsqrt(negative) = **0**, not NaN; rsqrt(0) = inf.
- `sin` is sin(pi * x) for |x| up to 0.5 and saturates at ±1 beyond (0.25
  gives 0.7071, 1 gives 1). The scene does not need it on the GPU.
- Gate of step 1: met. The rounding rule behind the 1-ulp differences was
  not identified, so the CPU reference will be compared with a tolerance of
  one colour step and the exact-match count reported.

### The scene, stage 1: primary rays (`tools/p0_6_scene_probe`, `scene_stage1*.txt`; plan steps 2 and 3)

A hand-assembled compute kernel casts one ray per pixel against the six
spheres and the ground of `test_08` and writes a flat colour for the
nearest hit (the sphere's albedo, the ground's checker) or the sky. No
shading, shadows or reflection yet. One job per frame; 16 pixels per pass
of the inner loop, **134 instructions per pass**, one thread switch per
pass; scene constants in registers. The reference is the same operations in
the same order in C. Offscreen (a plain buffer).

| 200 frames across the orbit, 800×480 | 1 batch | 2 | 4 | 8 |
|---|---|---|---|---|
| frame time (best) | 20.08 ms | 10.11 ms | 5.14 ms | **2.66 ms** |
| pixels equal to the reference, of 76,800,000 | all but 3 | all but 3 | all but 1 | all but 2 |
| the others | at an object edge | at an edge | at an edge | at an edge |

- **Correct:** bit-identical to the CPU reference except 1–3 pixels in 200
  frames, each on an edge between two objects (where a 1-ulp difference in
  the square root decides which is nearer).
- **Cost:** 2.66 ms per frame on 8 batches, 6.9 ns of all-QPU time per
  pixel, and it scales with batches (7.5× from 1 to 8). Mesa's full shader
  for this scene takes 23.7 ns per pixel. Stage 1 is the smaller part of
  that shader, so this is not yet a comparison.
- Parts of the 8-batch frame (`scene_stage1_cost_parts.txt`): no spheres
  and no ground 0.88 ms; the ground 0.26 ms; six spheres 1.56 ms (0.26 ms
  each, 16 instructions); the thread switch per pass 0.34 ms.
- **A special-function operation whose destination is an accumulator hangs
  the QPU.** The first version wrote `rsqrt` and `recip` results to r0–r2:
  every job ran until the kernel reset it and wrote nothing. With
  register-file destinations (all twelve in Mesa's compiled code are) it
  runs. Five unplanned resets (256 → 261): four from running four frames of
  an untested kernel instead of one, and one from a wrong first guess
  (stores left un-waited across a thread switch, which is not the cause:
  50 un-waited stores are fine, `fill_wait_every_large.txt`).
- Seen once, not explained: in the first job after those resets, lanes 0–3
  of the last batch wrote nothing (12,000 pixels left untouched, the other
  372,000 exact). It did not recur in 800 frames.
- Timings taken outside `wrap.sh` (without the performance governor) were
  5–7 times longer for the same job. Not investigated; every number here is
  under the governor, as the rules require.

### The scene, whole: shading, shadows, reflection (`tools/p0_6_shade_probe`, `shade_all.txt`; plan step 4)

The complete `test_08` picture as one straight-line kernel: per 16 pixels a
ray, a trace of the six spheres and the ground, the shading of the nearest
hit with a shadow test against the six spheres, then the reflected ray
through the same trace and shade, the mix and the pack to 8 bits. Every lane
runs everything; results are selected at the end. Nothing is skipped yet.
The frame was looked at (`scene_s1r1_f0_gpu.ppm`): spheres with highlights,
their shadows on the checkered ground, reflections in each sphere and in
the ground, the sky gradient.

| 8 batches, 800×480, 60 frames checked, then 32 jobs back to back | instructions per 16 pixels | frame time (median) | FPS | pixels differing, of 23,040,000 |
|---|---|---|---|---|
| shading only | 277 | 4.90 ms | 204 | 251, all by one colour step |
| + shadows | 367 | 6.35 ms | 158 | 251, all by one step |
| + reflection, no shadows | 545 | 10.30 ms | 97 | 434 |
| **the whole scene** | **725** | **15.63 ms** | **64.0** | **432** (313 by one step; largest 17) |
| Mesa, same scene, on the panel | 959 | 9.1 ms | 110 | |

- **Correct:** about 7 pixels per frame differ from the CPU reference, at
  edges of objects, shadows or reflections, where a 1-ulp difference in a
  square root or reciprocal decides. Each stage ran correctly the first
  time it ran. The image is the kernel's arithmetic, not Mesa's: the
  difference from Mesa's image is not measured.
- **Speed: the straight port is 1.72 times slower than Mesa** (40.7 against
  23.7 ns of all-QPU time per pixel), with fewer instructions. Mesa's
  shader does not execute all of its 959 for every pixel; this kernel
  executes all 725 for every pixel, the sky included.
- It scales with batches: 131.3 / 65.8 / 33.0 / 15.6 ms on 1 / 2 / 4 / 8.
- 192,000 of the 384,000 pixels are sky, 131,900 ground and 60,100 sphere
  (t = 1). That is the room step 5 has: the sky needs a few dozen
  instructions, not 725.

**Hardware facts met on the way:**

- **Only 8 batches run at once.** A job of 9 to 16 batches takes exactly
  twice as long as one of 8 (`batches_over_8.txt`): compute has 8 QPUs, the
  same ones Mesa's shaders use.
- **A second, rewindable uniform stream** (`tools/p0_6_unifa_probe`,
  `unifa.txt`): write an address to `unifa`, and `ldunifa` (to r5) or
  `ldunifarf` (to a register) reads successive words from it. At least 3
  other instructions must follow the write before the first read (0, 1 or 2
  give garbage for every read); r5 is usable in the next instruction; it
  survives a thread switch (alone on the GPU); a write and four reads per
  pass cost nothing measurable. It cannot ride on an instruction that has a
  small immediate. The whole-scene kernel takes its 204 constants per pass
  this way and keeps the registers for the work.
- **Instruction cache: a loop over 512 instructions (4 KiB) costs more.**
  Padding the pass with nops: 5.3 ns per extra instruction up to 467, then
  about 12 ns per instruction for the next 200, and 7.3 ns from there on
  (`shade_pad.txt`). The whole-scene pass is 725.
- **The special-function unit is shared.** 100 extra `rsqrt` calls per pass
  cost 5.3 ns each on 1 batch and 14.7 ns each on 8. Infinities, NaNs,
  negative inputs and 1/0 cost the same as ordinary values
  (`shade_pad_kinds.txt`).
- **A measurement trap:** timed between reference computations, every job
  was 10 ms slower: the CPU reference of the whole scene takes long enough
  for the GPU to go idle, and the next job pays the wake-up. Frame times
  above are from jobs back to back (`shade_all_timed_with_gpu_wakeups.txt`
  keeps the wrong ones).

### Doing less work (`tools/p0_6_shade_probe`, `tools/p0_6_scene.h`; plan step 5)

The same image, with work left out where its result is known. Each cut is
a switch (`P06_CULL`). Numbers in this section were re-measured after the
red team below (`tests/evidence-p0-6-step5-redteam-20261117/timing.txt`);
the first write-up timed one arrangement of the spheres, not the orbit.

1. **Sky rows.** A ray that goes up from the camera hits nothing (the
   camera is above everything): a row with uy > 0 is filled with its sky
   colour, 16 pixels per store, without casting rays.
2. **No second shade** when all 16 reflected rays hit nothing.
3. **Plain ground.** The CPU sorts each region of the lower half (16
   pixels wide, one row per batch high: 1,500 regions) into three classes
   from conservative bounds: a cone around the region's rays, a sphere
   around the patch of ground they reach, a cone from the camera's mirror
   image for the reflected rays. Class 1 (no sphere seen, shadowing or
   reflected) and class 2 (a sphere may shadow it) take a short path of
   190 instructions that performs the long path's operations for such
   pixels minus those whose result is then fixed; the rest take the long
   path. The class is one word per region in the first uniform stream.

Rows are also dealt to the batches in turn (batch b renders rows b, b + 8,
...). With contiguous blocks, cut 1 changed nothing: the four batches that
held the lower half still did all the work.

200 frames across the orbit, jobs back to back, 8 batches:

| cuts | GPU job per frame: median (min–max) | CPU per frame | frames per second (CPU + job) | against Mesa (9.1 ms) |
|---|---|---|---|---|
| none | 15.73 ms (15.67–15.87) | 0.007 ms | 63.6 | 1.73× slower |
| sky rows | 8.06 ms (7.99–8.19) | 0.006 ms | 123.9 | 1.13× faster |
| sky rows, no second shade | 6.61 ms (6.52–6.74) | 0.005 ms | 151.1 | 1.37× faster |
| **all three** | **3.87 ms (3.63–4.00)** | **0.156 ms** | **249.7** | **2.27× faster** |

- **The cuts change no pixel.** Each frame's image as the GPU wrote it is
  hashed, and a run with all cuts is compared with a run without
  (`tests/redteam-p0-6/scene_ab.sh`): 0 of 200 frames differ at 8 batches,
  0 of 48 at 4, 2 and 1 batches (each has its own region shape), 0 of 48
  without shadows and without reflection. Two runs with the cuts give the
  same 200 hashes.
- Against the CPU reference: 1,647 of 76,800,000 pixels differ over the 200
  frames (at most 16 in a frame), with or without cuts: 1-ulp differences
  of the hardware's reciprocal and square root at edges.
- Regions per frame, mean of 200: 612 long path (558–639), 821 plain
  ground, 67 ground that may be shadowed.
- **The CPU's share** is the scene constants, the region sort and the
  uniforms: 0.156 ms per frame (median; 0.18 max). A first, naive sort with
  trigonometry per region and sphere took 3.5 ms
  (`tests/evidence-p0-6-20261117/cull_slow_classifier.txt`).
- 1 / 2 / 4 / 8 batches, all cuts: 30.4 / 15.0 / 7.6 / 3.9 ms (the CPU's
  share grows as regions get thinner: 1.18 / 0.60 / 0.30 / 0.16 ms).

### Red team of step 5, 2026-11-17 (`tests/evidence-p0-6-step5-redteam-20261117/`)

Two readings: my own, and an independent reviewer given the code and the
hardware rules but not my conclusions (it also modelled the short path and
the region sort on a CPU: 0 misclassified pixels in 1,000 frames at 8
batches and 300 each at 4, 2 and 1). What was wrong, and what was done:

| # | Finding | Measured effect | Fix |
|---|---|---|---|
| 1 | The region sort's reflection bound treated the reflected ray as coming from the camera's mirror image. It starts 0.02 above the ground, so it can reach a sphere that line misses by up to 0.02; the 0.001 margin did not cover it | 4 of 200 frames had a wrong pixel (`ab_oldbound.txt`), and **the check passed them** | The sphere is made 0.02 larger for that test. The old bound is kept as fault 11 |
| 2 | The pixel check allows 38 differing pixels per frame (needed: the 1-ulp edge pixels), so it cannot see a small error like #1 | as #1 | The exact comparison of cut against uncut images (`scene_ab.sh`), which has no allowance. It reports fault 11; the reference check alone does not |
| 3 | The short path ignored the "no shadows" and "no reflection" variants | up to 8,190 and 117,892 wrong pixels per frame, every frame | The short path follows the two switches |
| 4 | Cut 3 without cut 1 (`P06_CULL` 4 or 6): the kernel read a class word in sky rows and the CPU wrote none, so every row read another row's classes (and at 1 batch, past the end of the buffer) | found by reading | Sky rows get class 0 when the sky cut is off; the buffer is sized for the worst case; a region that straddled the horizon is refused |
| 5 | An unwritten pixel could pass: the comparison masked the top byte before looking for the unwritten mark | a pixel whose colour is (238, 238, 238) | Whole word compared; unwritten pixels counted on their own |
| 6 | One guard word after the image | — | 16 guard rows before and after, as on the panel |
| 7 | The timing loop ran the last frame's scene 40 times | 3.85 ms was one arrangement; over the orbit the job takes 3.63–4.00 ms | A second pass over every frame, CPU and job timed separately |
| 8 | A failed build, setup or job gave exit 1, the same as a failed check: a fault case could "pass" because the run died | — | Exit 3 for the run itself; fault cases also require a frame whose job ran and whose check failed |
| 9 | `P06_PAD` could write past the constant table (2,048 entries) | sanitizer report on a build-only run | The table is bounded; the build is refused |
| 10 | Built without `-ffp-contract=off` the reference fuses multiply and add and no longer matches the kernel | about 120 pixels per frame (seen when a mutant build left the flag out) | The probes test for it at start and refuse to run |
| 11 | `P06_MARGIN=nan` classed every region as plain ground; other bad values were read as 0 | found by reading | Arguments and environment are range-checked |

**Verification.** `tests/redteam-p0-6/scene_cases.sh`, 43 cases, 0
failures (`cases.txt`): every combination of cuts, 1/2/4/8 batches, the
variants; 12 built-in faults that must each fail a frame's check (second
shade skipped when any ray misses, constant stream not moved, shadow test
skipped, a row too many, the region sort ignoring reflections, covering
spheres or shadows, rows below the horizon treated as sky, wrong constants
on the short path, batches two rows apart, a pixel left unwritten, sky
rows not stored); the negative margin; bad arguments; and the seven exact
comparisons. `scene_mutants.sh` breaks the checks themselves (no
comparison, no guard check, unwritten pixels ignored, failures not
reported, a 1 % allowance, a hash that ignores the image, no clearing
between frames): 7 of 7 caught, control passes (`mutants.txt`). An eighth
mutant was withdrawn: its edit did not do what its label said. GPU resets
261 before and after; no display underrun.

Clean by the reviewer's reading: the other region bounds (convexity
arguments for the ray cone and the ground patch), every stream rewind (3
instructions before the first read in each block entered by a branch), no
flag reused across a flag-setting instruction, the row arithmetic. The
library's rule checker is off for these programs and models only TMU
loads: the hardware rules above are kept by reading, not by a machine.

### The whole scene on the panel (`tools/p0_6_scene_present_probe`, `tests/evidence-p0-6-panel-20261117/panel.txt`)

The same kernel writing straight into vc4 buffers that are flipped
through KMS; no Mesa, EGL or GBM in the process. Three buffers: the one on
the panel, the one whose flip is pending, the one being rendered. Scene
time is elapsed time, as in `tests/test_08_raycast_orbit`. A finished
frame is flipped if no flip is pending and dropped otherwise, which is
what `hs_graphics_present` does for frames this fast (its vsync-off path).
Two runs of each, alternating with Mesa's test, same session, performance
governor:

| 10 s on the panel, 800×480 at 60 Hz | frames rendered per second | flips per second | flip intervals over 1.5 refreshes | frame to frame, median (p99) |
|---|---|---|---|---|
| Mesa, `test_08` | 109.9, 110.1 | — | — | — |
| **QPU compute, free-running** | **247.9, 247.8** | 60.2, 60.2 | 0, 0 | 4.05 ms (4.29, 4.31) |
| QPU compute, every flip waited for | 60.1, 60.1 | 60.1, 60.1 | 0, 0 | 16.65 ms (16.77, 16.73) |

- **2.25 times Mesa's frame rate, like for like:** both numbers are frames
  rendered per second through each program's whole loop on the panel.
- Per frame: 3.86 ms job (p99 4.04), 0.17 ms CPU, the rest of the 4.05 ms
  is the flip and event calls. At 60 Hz the job takes the same 3.9 ms: the
  GPU does not go idle between frames.
- **What is on the panel is checked.** After the timed phase, 12 frames
  across the orbit are flipped and compared with the CPU reference in all
  three buffers (92 pixels of 4.6 million differ, at most 13 in a frame),
  and their hashes equal those of the same 12 frames rendered offscreen.
  The last framebuffer is then composed once more by the display
  controller into a capture buffer: 0 pixels differ from the buffer's
  memory. Guard rows above and below each buffer: untouched. The timed
  frames themselves are not compared (the reference takes far longer than
  a frame).
- Start of the process to the first frame on the panel: 32–47 ms (four runs).
- `tests/redteam-p0-6/scene_present_cases.sh`, 25 cases, 0 failures
  (`sp_cases.txt`): both pacings, 1/2/4/8 batches, faults that must each
  fail a named check (wrong scene time, framebuffer 16 rows early, four of
  the kernel's faults), signals at start, in the timed phase and in the
  checked phase, SIGKILL, a second copy refused, plane properties left by a
  previous client, and the display state identical after every case.
  `scene_present_mutants.sh`: 12 of 12 caught, control passes
  (`sp_mutants.txt`). "No explicit restore" is not a mutant these cases can
  see: the kernel puts the console back when the process ends; the note is
  in the script.
- The KMS helpers moved from `tools/p0_6_present_probe.c` into
  `tools/p0_6_kms.h`; that probe's 18 cases and 9 mutants were run again:
  0 failures, 9 of 9 (`present_cases.txt`, `present_mutants.txt`).

### No frame dropped: every frame flipped at once (`async.txt`, `tests/evidence-p0-6-coexist-20261117/`)

Direction from the project owner, 2026-11-17: "Don't drop frames. We are
optimizing for the Pi 4, not for the attached display." The free-running
mode above renders 248 frames per second and shows 60 of them. Pacing 2 of
the probe hands every frame to the display controller at once
(`DRM_MODE_PAGE_FLIP_ASYNC`, which vc4 offers): the flip does not wait for
the vertical blank.

| 10 s on the panel, every frame flipped at once | frames rendered per second | flips per second | dropped | wait for the previous flip, median (max) |
|---|---|---|---|---|
| QPU compute alone (two runs) | 246.9, 250.8 | 246.9, 250.8 | 0, 0 | 0.017 ms (0.086, in a 3 s run) |

- Every rendered frame is accepted and its flip completes: 2,469 of 2,469
  and 2,508 of 2,508.
- The 12 checked frames and the capture are as before (they are flipped
  with the wait, so each can be compared while it is shown).
- The flip events of immediate flips carry the time of the last vertical
  blank, so the probe's flip-interval figure says nothing in this mode.
- **Not measured: what the panel shows.** The controller is given four
  buffers per refresh; whether the glass shows bands of several frames
  (tearing) or only the last one given before each scan was not looked at.
  It needs eyes or a camera on the panel.
- With this mode the probe's cases are 26 (0 failures) and its mutants 13
  of 13, one of them "pacing 2 still drops frames" (`sp_cases.txt`,
  `sp_mutants.txt` in the coexistence evidence directory).

### Beside another GPU client (`tests/redteam-p0-6/scene_coexist.sh`, `tests/evidence-p0-6-coexist-20261117/heavy.txt`, `light.txt`)

The second client is `tools/p0_5_gl_probe`: an offscreen GLES renderer
(512×512) that checks every frame it draws. Heavy: 200 loop passes per
pixel, 26 ms a frame. Light: 64 passes, 10 ms a frame. Beside it the
kernel's batches are really switched out at their thread switches (one
per 16 pixels), which is what loses r0–r3 and what the second constant
stream had not been run through.

**Correct on both sides.** 200 frames across the orbit rendered beside
each renderer: every frame passes the reference check, and every frame's
image hash equals that of the same frame rendered alone (0 of 200 differ,
both loads). On the panel the 12 checked frames beside the renderer have
the hashes of the 12 alone. The renderer drew 323–912 frames per run
beside the kernel: none wrong, none in error, none reporting a reset.
GPU resets 261 before and after.

**What each costs the other** (panel, every frame flipped at once, 10 s):

| | compute alone | renderer alone | compute beside | renderer beside | compute job, median (max) beside |
|---|---|---|---|---|---|
| heavy renderer (26 ms frames) | 246.9 FPS | 37.5 FPS | 57.7 FPS (23 %) | 32.2 FPS (86 %) | 17.2 ms (28.9) |
| light renderer (10 ms frames) | 250.8 FPS | 91.4 FPS | 118.9 FPS (47 %) | 65.2 FPS (71 %) | 8.4 ms (12.1) |

- The two shares add up to 109 % and 119 %: the GPU is divided, with a
  little gained from running together.
- The renderer keeps the larger share; a compute job that takes 3.9 ms
  alone takes up to one renderer frame beside it (28.9 ms, 12.1 ms).
- No frame of either was lost, and no job came near the 1 s at which the
  kernel resets the GPU.
- Offscreen, the job's time beside the renderer is the same (median 15.9
  and 8.4 ms). The script's "renderer inside an offscreen scene loop"
  figure (38.2 and 91.1 FPS) is not a cost: that loop spends most of its
  time in the CPU reference, with the GPU free. Only the frames' validity
  is taken from it.

**Not measured:** how the GPU's time is divided and why the renderer gets
the larger share (no trace was taken); three or more clients; the compute
path beside another display client; how the image differs from Mesa's (it matches a
reference of the kernel's own arithmetic); what the panel shows under
immediate flips (above); the long path's
sphere lists (plan step 5, item 3: 612 regions still test six spheres
three times, 745 instructions, over the 512 mark); `exp`, `log` and
`rsqrt2` (their encodings are not in Mesa's output for this shader, and
none was guessed); why the special-function result cannot go to an
accumulator; the lost lanes after a reset.
