# P0-6 plan: a scene rendered by QPU compute, shown through KMS, without Mesa

Written 2026-11-17. Results go in `HARDWARE_HANDOFF_P0.md` (section "P0-6");
this file is the plan and is updated when a step changes it.

## Where we are

Measured (see the P0-6 section of the hand-off document):

- A buffer vc4 allocates can be imported into v3d and written by a compute
  job. The reverse is refused.
- One compute job fills 800×480 in 0.36 ms.
- A test pattern rendered by compute alone was flipped at 60 Hz for 10 s
  with no Mesa in the process: no late flip, no wrong pixel in the checked
  frames.
- Mesa renders the benchmark scene (`tests/test_08_raycast_orbit`) at
  110 FPS with a fragment shader of 959 instructions and no loops:
  23.7 ns of QPU time per pixel.

Not yet: any scene. The probes are tools, not a library.

## Success condition

The `test_08` scene rendered by QPU compute alone, shown on the panel
through KMS with no Mesa in the process, compared pixel by pixel with a CPU
reference of the same arithmetic, and its frame rate measured against Mesa's
on the same panel.

"Faster than Mesa" is the question, not the assumption. The plan is built to
answer it early.

## Invariants

- The display is taken only while a test runs and is given back as it was:
  CRTC, planes, DRM master.
- No boot configuration changes, no reboots.
- GPU and display runs go under the GPU lock with the performance governor;
  resets and display underruns are counted.
- A compute job gives way to other GPU users only at its thread switches:
  no long stretch without one in anything that may run beside a renderer.
- A real thread switch does not keep r0–r3: nothing lives in them across one.
- `recip`, `rsqrt` and `sin` write to a register-file entry, never to an
  accumulator (that hangs the QPU).
- A kernel that has never run is run for one frame first: a hang costs a
  GPU reset per job.
- Every number in the documents is a measurement. Every check is shown to
  fail when the thing it checks is broken.
- Existing tests keep passing (`make test`, `check-sources`,
  `rt_mailbox cases`, `tests/test_mailbox`).

## Steps

Each step ends with a measurement and a decision. A step that fails its gate
stops the line until the plan is changed here.

### 0. Harden what exists — done 2026-11-17

Result: 11 findings fixed, 18 of 18 cases, 9 of 9 mutants; see "Red team of
the two probes" in the hand-off document.

Red-team the two P0-6 probes (`tools/p0_6_scanout_probe`,
`tools/p0_6_present_probe`): display left in a bad state on any exit path,
writes outside the buffer, checks that cannot fail, numbers that are not
what their label says. Fix, re-measure.

### 1. The hardware's special functions — done 2026-11-17

Result: `recip` and `rsqrt` are add-ALU operations here (not the magic
registers of older V3D), within 1 ulp on 320,000 inputs, one instruction
each. The rounding rule was not identified, so step 2's reference is
compared with a tolerance of one colour step. Original plan for this step:

The scene needs a square root and a reciprocal per ray. The QPU provides
reciprocal, reciprocal square root and others through a special-function
unit whose timing and rounding are not measured here.

- Probe: write x to each function's register, read the result after 1, 2, 3
  and 4 instructions, for a few million x. Record when the result is valid,
  the largest error against the CPU, and the cost per call.
- Gate: a square root and a divide good enough that the 8-bit colour of a
  pixel matches a CPU reference built from the same operations. If the unit
  is too coarse, use Newton steps (measured cost) or change the reference
  to model the unit exactly.

### 2. The CPU reference — done for stage 1, 2026-11-17

Result: `reference()` in `tools/p0_6_scene_probe.c`. Because the kernel and
the reference share every operation, they agree bit for bit except where a
1-ulp difference in a square root decides an edge: 1–3 pixels in 200
frames. The difference from Mesa's image is not measured yet (it needs the
full shading). Original plan for this step:

A C implementation of the scene written operation by operation as the
kernel will do it (same order, no fused multiply-add, the special functions
modelled as measured). It is the oracle for every later step. Also record
how it differs from Mesa's image (different arithmetic: expect small
differences; measure them, do not assume).

### 3. Primary rays only — done 2026-11-17

Result: 134 instructions per 16 pixels, 2.66 ms per frame on 8 batches
(6.9 ns per pixel), linear in batches and in instruction count: no
unexplained stall. Two rules learned and added to the invariants: a
special-function result must go to a register-file entry (an accumulator
destination hangs the QPU), and an untested kernel is run for one frame
first. Original plan for this step:

A hand-assembled kernel: ray per pixel, six spheres and the ground, nearest
hit, flat albedo and sky. No shadows, no reflection. One job per frame, 8
batches, thread switches at a measured interval.

- Check: every pixel against the CPU reference, on several frames across
  the orbit.
- Measure: ms per frame, instructions executed per pixel.
- Gate: correct, and its cost per pixel is in line with the instruction
  count (no unexplained stall). This is the first point where the cost of
  hand-written compute against Mesa's compiler can be read off.

### 4. Shading, shadows, reflection — done 2026-11-17

Result: the whole scene in 725 instructions per 16 pixels, correct (about 7
of 384,000 pixels per frame differ, at edges), 15.6 ms per frame on 8
batches: 64 FPS against Mesa's 110, 1.72 times slower. Decision: continue.
The gap is inside the bound set below ("more than about twice"), half of
the frame is sky (192,000 of 384,000 pixels) and the kernel spends the full
725 instructions there. Also learned, and now rules: only 8 batches run at
once; a pass over 512 instructions pays for instruction-cache misses;
constants come from the rewindable stream; time jobs back to back.
Original plan for this step:

Add diffuse, specular and rim, then shadow rays, then the reflection bounce:
three sub-steps, each checked against the reference and timed.

- Gate: the full scene, correct. Compare with Mesa: frame time on the
  panel, same resolution.
- Decision point. If the straight port is slower than Mesa by more than the
  culling in step 5 can plausibly recover, write that down and stop or
  change course. A straight port is not expected to beat Mesa: both run the
  same arithmetic on the same QPUs.

### 5. Do less work per pixel — first three cuts done 2026-11-17

Result: 15.66 ms → 3.85 ms of GPU time per frame (plus 0.16 ms of CPU to
sort 1,500 regions), the image unchanged: 2.3 times faster than Mesa's
9.1 ms, offscreen. Cuts 1, 2 and a class-based form of 3 below (plain
ground takes a 190-instruction path). Not done: sphere lists for the 609
regions that still take the long path (745 instructions, over the
512-instruction mark). Rows are dealt to the batches in turn, or the cuts
do not shorten the frame. Plan as written before the work:

Measured starting point: 15.6 ms per frame, 725 instructions for every 16
pixels; Mesa 9.1 ms. Pixels at t = 1: sky 50 %, ground 34 %, spheres 16 %.
First cuts, in order of expected gain, each to be measured:

1. Rows and passes that are all sky leave after the ray is set up.
2. The second pass (reflection) is skipped when no lane hit anything that
   reflects into a sphere or the ground (all 16 reflected rays miss).
3. Per row band or region, the CPU gives the kernel only the spheres its
   rays can reach, for the primary rays, the shadow rays and the reflected
   rays, so a pass tests two or three spheres, not six.
4. Keep each hot path under 512 instructions.

A branch skips constants in the stream: each block that can be entered by
a branch rewinds the stream to its own start (3 instructions before the
first read).

Where compute can differ from a fragment shader:

- per 16-pixel vector: skip the spheres no lane can hit, skip the
  reflection where no lane hit anything, end early on sky;
- per region: the CPU computes which spheres each screen region's rays can
  reach (bounds of each sphere on screen, and for ground pixels the shadow
  and reflection candidates) and gives each region its own short list.

Measure first how much of the frame is cheap (sky, plain ground) before
building: that fraction bounds the gain.

### 6. The mailbox as the frame loop's dispatch

With per-region work lists there are many small pieces of work per frame.
Compare, on the real scene:

- one job per frame that walks a work list in memory;
- one job per region;
- the resident program (`hs_mailbox`) taking one message per region.

This is where the mailbox is either needed or not. Decide by frame time.

### 7. A library and a test

Only for what survived: a small module for the KMS side (take the display,
buffers shared with v3d, flip, give it back) and the compute renderer for
this scene as a test beside `test_08`, with the same red team as P0-5
(adversarial cases, mutants, coexistence with another GPU client, a real
reset).

## What is uncertain, in order

1. The special-function unit's accuracy and cost (step 1).
2. Whether a hand-written kernel is as tight as Mesa's compiled one (step 3
   gives the first reading).
3. How much of this scene is cheap (step 5): it bounds everything after.
4. Whether many small pieces of work per frame beat one large job once the
   lists exist (step 6).
5. Pacing above 60 FPS: the probe waits for every flip. A renderer that is
   faster than the panel needs the hold and flip logic `hs_graphics` already
   has, or its own.

## What would stop this

- The special functions cannot be matched by a CPU reference within one
  colour step, and Newton refinement costs more than Mesa's whole shader.
- The straight port is more than about twice Mesa's frame time and the
  cheap fraction of the frame is small.
- Compute and display cannot be paced without tearing or late flips.

Each of these is a measurement away, and each would be written up as a
result like any other.
