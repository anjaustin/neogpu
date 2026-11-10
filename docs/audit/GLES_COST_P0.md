# GLES cost P0: speed and efficiency programme

Status: plan. Every item is grounded in measured evidence from
[COST_BREAKDOWN_RESULTS.md](COST_BREAKDOWN_RESULTS.md); every item has an
acceptance criterion and a rejection criterion. No item ships without a
measurement that beats the current committed path.

## 1. Scope and intent

Optimise **speed and efficiency**, agnostic to whichever display is currently
attached. The DSI touchscreen connected during measurement (800×480 @ 60 Hz,
vc4-drm `card1`, rendering on V3D `renderD128`) is one configuration, not the
target. Concretely:

- A faster path must not be tuned to one panel, connector or mode.
- Display-agnosticism is itself a P0 requirement: the current code selects
  connectors by `DRM_MODE_CONNECTOR_DSI` only and falls back to a hard-coded
  800×480, so it is already wrong on any non-DSI display.
- Where the attached panel cannot show frames faster than we produce them,
  that is a presentation limit, not a rendering limit. Optimise production
  throughput, not the panel.

### Invariants (any candidate that breaks one of these is rejected)

- Rendered pixels, pixel order and draw ordering unchanged.
- Capture behaviour and capture defaults unchanged.
- Participant FIFO, admission and applied/accepted counts unchanged.
- Complete lifecycle: DRM fd, EGL display/context/surface, GBM device/surface
  and buffers all released; no fd leak, no deferred cleanup.
- **Per-unit speed must not regress** to buy throughput (see section 3).

### Measurement boundaries

Cost numbers are only comparable inside a boundary. The full set is in
[COST_BREAKDOWN_RESULTS.md](COST_BREAKDOWN_RESULTS.md) section 2. The ones
that bite most often:

- Submission-only measurements are not throughput: `window-noswap-submit-64`
  reads 72.3 µs/frame while the deferred `glFinish` after the same 256 frames
  took 217.9 ms (≈851 µs/frame of GPU work).
- The E2E harness finishes every frame; production does not. At 64 draws,
  finishing every frame costs ~30% more than swap-only.
- Traced, preloaded and sampled runs are for locating cost, not for rates.

## 2. Measured baseline (Pi 4, V3D 4.2.14.0, Mesa 26.0.8, 4 CPUs)

| Region | Measured |
|---|---|
| Startup, total | 128.0 ms (`init_ns`), plus 28.5 ms first present |
| Startup, `gbm_create_device` | 79.5 ms (62% of intercepted startup) |
| Startup, connector sweeps | 18.6–22.8 ms (6 `drmModeGetConnector`, two sweeps) |
| Startup, manual modeset | 16.3 ms (`drmModeSetCrtc`, one call) |
| Frame, 1 draw | 346 µs: 52 µs submit, 294 µs wait (85%) |
| Frame, 64 draws | 773 µs: 121 µs submit, 652 µs wait (84%) |
| Frame, GPU throughput | 820–851 µs at 64 draws even with zero sync |
| Fill cost | 0.68 ns/px (≈5.9 GB/s RGBA8) |
| Teardown, total | 32.4 ms, of which intercepted calls 5.2 ms |
| Teardown, final `close(drm_fd)` | ~27 ms (35.2 ms under tracing) |
| CPU attribution | kernel 51–77%, libgallium 5–27%, libc 10–13%, app ≤5.7% |
| Per-frame kernel work | ~5 syncobj ioctls (create/export/import/wait/destroy) + ~4.3 `V3D_WAIT_BO` + ~2.7 counter ioctls, per frame |

## 3. Throughput-without-slowing rule

When a per-unit cost cannot be reduced without regressing per-unit latency,
prefer changes that increase units per second while leaving the per-unit cost
unchanged:

1. **Batch** work into fewer, larger operations (one present serving N
   producer frames; one sweep serving N lookups).
2. **Overlap** independent work (CPU submission against GPU execution; one
   producer's setup against another's work).
3. **Decouple cadence**: produce frames on a fixed internal rate and present
   on an independent schedule, so a slow panel cannot throttle production.
4. **Reuse** one-time setup across more units of work (amortise the 79.5 ms
   driver load and the 16.3 ms modeset across many frames).

Every such change is gated on a measured **per-unit non-regression**, not on
total throughput alone.

## 4. P0 items

| Item | Title | Status |
|---|---|---|
| P0-1 | Remove the redundant manual modeset | **Rejected (measured)** — it is what puts our frames on the panel |
| P0-11 | Make display ownership optional | Open — isolates 48.6 ms/process of display cost |
| P0-3 | Single connector sweep | Open — duplicate sweep worth ~9–11 ms, validated by ioctl counts |
| P0-2 | Display-agnostic connector and mode selection | Open — the DSI-only check is itself a portability bug |
| P0-10 | Match the surface format to the display's native format | Open — RGB565 vs XRGB8888 halves scanout bandwidth |
| P0-12 | Cost of taking the display from the console | Open — 25.3 ms takeover, ~13.6 ms handback |
| P0-4 | Attribute and reduce the `close()` cost | Open |
| P0-5 | Reduce the per-frame cross-device handshake | Open |
| P0-6 | Reduce GPU work (fill and overdraw) | Open — needs the real scene's coverage |
| P0-7 | Amortise startup across more work | Open — architectural |
| P0-8 | Throughput levers that do not slow the per-unit path | Open |
| P0-9 | Stop burning CPU while waiting | Open |

### P0-1 — Remove the redundant manual modeset — **REJECTED (measured)**

**Hypothesis.** The application calls `drmModeAddFB` + `drmModeSetCrtc`
itself (`include/hs_graphics.h:329-332`) on a GBM surface that Mesa's swap
path already presents. The manual call may be redundant: Mesa issues
`DRM_IOCTL_MODE_SETCRTC` itself when flipping a GBM surface. If so, removing
it saves ~16.3 ms of startup, and it may also remove the DSI/CRTC teardown
that dominates `close()` (~27 ms).

**Method.** Runtime-selectable arms (`manual` = current, `egl` = skip the
manual `AddFB`/`SetCrtc`), same binary, same host, interleaved reps. In each
arm record: startup, first-present, shutdown, the number and timing of
`DRM_IOCTL_MODE_SETCRTC`/`MODE_GETCRTC` ioctls, and the CRTC/FB state after
the first present. Visual confirmation of live panel content is required from
a human observer.

**Result — 5 reps × 64 frames, `tests/evidence-cost-modeset`:**

| Measure | `manual` (committed) | `egl` (skip manual modeset) | Delta |
|---|---:|---:|---:|
| init | 123.17 ms | 120.45 ms | −2.7 ms |
| first present | 33.24 ms | 0.92 ms | **−32.3 ms** |
| shutdown | 42.32 ms | 28.68 ms | **−13.6 ms** |
| `MODE_ADDFB` / `MODE_SETCRTC` ioctls | 1 / 1 (25.30 ms traced) | 0 / 0 | — |
| `MODE_GETCONNECTOR` ioctls | 12 (24.13 ms traced) | 6 (10.46 ms traced) | −6 ioctls |

**Why it is rejected.** EGL's GBM swap path does **not** program the CRTC on
this device: the `egl` arm issues no `MODE_ADDFB` and no `MODE_SETCRTC` at
all. The kernel CRTC state shows why that matters:

- Before NeoGPU presents anything (`after_init`, identical in both arms),
  CRTC 93 already carries **fb_id 724, handle 2, 800×480, pitch 1600, bpp 16**
  — an RGB565 framebuffer that the **DRM console** owns, not NeoGPU.
- `manual` arm after first present: CRTC 93 carries **fb_id 726, handle 4,
  pitch 3200, bpp 32** — NeoGPU's own XRGB8888 GBM buffer. The panel is
  showing our frames.
- `egl` arm after first present and after the loop: CRTC 93 still carries the
  console's **fb_id 724, bpp 16, pitch 1600**. NeoGPU's surface is never
  scanned out.

So the manual modeset is **what puts NeoGPU's output on the panel**, and
removing it would leave the display showing the console while NeoGPU renders
into an unsignalled buffer. The cost is real and belongs to display
ownership, not to waste.

**What the rejection yields.** The measurement isolates **48.6 ms of
per-process lifecycle cost as display ownership** (32.3 ms first present +
13.6 ms shutdown), which is now P0-11 rather than a saving.

### P0-11 — Make display ownership optional (throughput without slowing the display)

**Hypothesis.** Taking the CRTC from the console and handing it back is
~48.6 ms per process. A renderer that never takes the CRTC keeps all of
that time for work instead, and becomes genuinely display-agnostic: it does
not care which panel is attached, or whether one is.

**Method.** Make the CRTC takeover an explicit, supported option rather than
an unconditional side effect of the first present. The `egl` arm already
exists as the experimental form; the product form is a named mode (for
example a no-scanout init flag) so a host can ask for offscreen production
without pretending to display.

**Acceptance.** In no-scanout mode: first present ≤2 ms and shutdown ≤30 ms
(measured: 0.92 ms and 28.68 ms), identical rendered pixels, and complete
resource release. Displayed mode must be unchanged.

**Note.** This is the "agnostic to the connected display" option made
concrete: it is not a faster display path, it is *no* display path, and it
must be selected deliberately rather than by accident.


### P0-2 — Display-agnostic connector and mode selection

**Hypothesis.** Requiring `DRM_MODE_CONNECTOR_DSI` (16) in both the init scan
and the CRTC setup (`include/hs_graphics.h:77`, `:305`), plus the hard-coded
800×480 fallback (`:89-92`), means NeoGPU cannot use an HDMI/DP/DPI display
at all and silently renders at a wrong size when no DSI panel is attached.

**Change.** Select any connected connector that has modes, prefer one that can
drive the current CRTC, and derive the surface size from the chosen mode.
Keep the DSI preference as a tie-break, not a requirement. Never fall back to
a hard-coded size — fail cleanly instead.

**Acceptance.** Correct size and a live CRTC on the attached display, plus
identical pixels for the existing E2E equality gates. **Reject.** Any change
in rendered pixels on the committed path.

### P0-3 — Single connector sweep

**Hypothesis.** The connector scan runs twice — once in `hs_graphics_init`
(`:73-87`) and again inside the one-time CRTC setup in `hs_graphics_present`
(`:296-341`) — costing 18.6–22.8 ms total, 3.6–6.7 ms per `GETCONNECTOR`
ioctl.

**Change.** Cache the connected connector id, encoder id and mode during init
and reuse them. Fall back to a re-scan only when a fresh
`drmModeGetResources` (0.056 ms) no longer lists the cached connector, so
hotplug is still handled.

**Partial validation.** The P0-1 arms bracket the cost directly: the `egl`
arm performs only the init sweep and issues **6 `MODE_GETCONNECTOR` ioctls
(10.46 ms traced)**; the `manual` arm performs both sweeps and issues **12
(24.13 ms traced)**. The duplicate sweep is worth ~13.7 ms traced (~9–11 ms
untraced) and is removable without touching the CRTC takeover, which is the
part P0-1 proved necessary.

**Acceptance.** Startup improves by ≥8 ms with identical CRTC/mode/FB.

### P0-10 — Match the surface format to the display's native format

**Measured.** The console's framebuffer on this panel is **RGB565** (bpp 16,
pitch 1600 at 800×480), while NeoGPU's GBM surface is **XRGB8888** (bpp 32,
pitch 3200). The takeover therefore changes the panel's format, not just its
buffer.

**Hypothesis.** Creating the GBM surface in the panel's native format would
(a) halve scanout bandwidth — the DSI link is the narrowest pipe in the
path — and (b) possibly avoid a full modeset, because the mode and format
would not change, only the buffer.

**Method.** Query the connector's current format, create the surface in that
format, and measure the takeover cost and per-frame present cost against the
current XRGB8888 path.

**Acceptance.** Lower takeover and/or present cost with no regression in
rendered content. **Trade-off to price.** RGB565 costs colour depth; the
product decision is explicit, and the measurement quantifies both sides.

### P0-12 — Cost of taking the display from the console

**Measured.** The CRTC is **already owned by the DRM console** when NeoGPU
starts (fb 724, RGB565). NeoGPU's takeover costs 25.30 ms traced
(`MODE_SETCRTC`), and the handback at `close()` costs ~13.6 ms of shutdown.

**Hypothesis.** Both costs are inherent to the panel's DSI programming, but
they may be smaller when no console holds the CRTC (kiosk/headless
deployment), and the *handback* may be avoidable by handing the console's own
framebuffer back rather than by letting the driver restore it.

**Method.** Measure the takeover and handback with the console not attached,
and compare handing back the console's framebuffer explicitly against
implicit restore.

**Acceptance.** A measured reduction, or a documented floor.

### P0-4 — Attribute and reduce the `close()` cost

**Hypothesis.** ~27 ms of the 32.4 ms shutdown is the final `close(drm_fd)`,
plausibly DSI panel unbind on the same slow path as the connector queries.
P0-1 is the control arm: if the panel is never owned, `close()` should get
much cheaper.

**Method.** Compare `close()` cost with and without a programmed CRTC; also
compare against a device that is never taken over.

**Acceptance.** A measured reduction in shutdown wall time with no lifecycle
regression. **Reject.** If the cost is inherent to releasing the display
device, record it as a floor and stop.

### P0-5 — Reduce the per-frame cross-device handshake

**Hypothesis.** Each frame creates a sync object, exports it to an FD,
imports it on the display device, waits and destroys it (~5 ioctls plus an
FD round trip), plus ~4.3 `V3D_WAIT_BO` and ~2.7 counter ioctls.

**Method.** Measure the cost of those ioctls per frame; test whether the
rendering and scanout devices can share the handshake more cheaply.

**Acceptance.** A measured per-frame reduction with no pixel or ordering
change.

### P0-6 — Reduce GPU work (fill and overdraw)

**Hypothesis.** App CPU is ≤5.7% of cycles, so per-frame time is GPU work:
84–85% of every frame is wait. Fill is 0.68 ns/px (≈5.9 GB/s RGBA8) and the
~11 µs/draw marginal cost at 800×480 is *fill*, not draw setup (0.71 µs/draw
at 64² vs 17.1 µs/draw at 800²).

**Method.** Measure the real scene's coverage and overdraw — the synthetic
benchmark triangles are not the product workload.

**Acceptance.** A measured fill reduction with identical pixels.

### P0-7 — Amortise startup across more work

**Hypothesis.** 79.5 ms of `gbm_create_device` is mapping and faulting a
~183 MB driver image (libLLVM 134 MB + libgallium 50 MB) plus kernel driver
init (V3D `GET_PARAM` alone is 12.9 ms); only 17,368 relocations, so it is
not symbol resolution. The shader cache is already warm.

**Options.** Longer-lived process or a pooled context; or a driver build
without the LLVM dependency. **Note.** Not solvable inside the library; it is
an architectural decision that must be priced against per-process savings.

### P0-8 — Throughput levers that do not slow the per-unit path

Per section 3: batch presents so one `eglSwapBuffers` serves N producer
frames; decouple production cadence from presentation cadence so a slow or
absent panel cannot throttle production; overlap producer submission with GPU
execution.

**Constraint.** Per-frame latency must not regress. Note that deep
no-swap pipelining recovered only 19–22% at 64 draws (1055 → 820–851 µs), so
pipelining alone is not sufficient — batching must also reduce the number of
per-frame kernel round trips.

### P0-9 — Stop burning CPU while waiting

**Hypothesis.** The calling thread busy-polls for 32–55% of the wait
(211 µs CPU in a 385 µs frame at 1 draw); `__sched_yield` is 5.96% of cycles
at 8 producers; the dense E2E case uses 3.14 of 4 cores.

**Method.** Compare the driver's spin wait against a fence-based blocking
wait (`glFenceSync` + `eglWaitSync`) for CPU per frame and for latency.

**Acceptance.** Lower CPU per frame with no latency regression. **Note.**
Production does not call `glFinish` per frame, so this mostly affects
capture, readback and multi-producer scaling paths.

## 5. Out of scope

- Re-running the four rejected shutdown candidates (close-only, early-RmFB,
  full saved-CRTC restore, page-flip restore). The page-flip path failed its
  own gate with `EINVAL` in 6/6 attempts.
- Re-running the rejected GL-state cache candidates.
- App-side CPU micro-optimisation: the measured ceiling is 5.7% of cycles.

## 6. Evidence

- [COST_BREAKDOWN_RESULTS.md](COST_BREAKDOWN_RESULTS.md) — the measurements
  this plan is built on.
- [GLES_SHUTDOWN_PLAN.md](GLES_SHUTDOWN_PLAN.md) — shutdown contract and
  rejected candidates.
- [REFERENCE_REFRESH_RESULTS.md](REFERENCE_REFRESH_RESULTS.md) — committed
  default-path benchmark findings.
- Raw evidence on the Pi host under `tests/evidence-cost-*`;
  `tests/evidence-cost-catalog.tar.gz` carries the hash-verified subset.
