# NeoGPU

**A message-native runtime substrate for ARM/NEON — graphics and tooling
on the Raspberry Pi 4.**

[![Build](https://github.com/anjaustin/neogpu/actions/workflows/ci.yml/badge.svg)](https://github.com/anjaustin/neogpu/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Language: C](https://img.shields.io/badge/language-C%20%7C%20Make-blue.svg)](https://en.wikipedia.org/wiki/C_(programming_language))

NeoGPU combines message-based state/resource/control operations with retained
participant draw contributions. Both feed the normal `HSGpu` frame lifecycle.
Message queues use SPSC/MPSC publication; the overall runtime also uses locks
for lifecycle/state management and is not end-to-end lock-free.

```c
HSGpu gpu;
hs_gpu_init(&gpu);
hs_gpu_set_shader(&gpu, 0);
hs_gpu_load_buffer(&gpu, 0);
hs_gpu_draw(&gpu, 0);
```

---

## Table of Contents

- [What NeoGPU Is](#what-neogpu-is)
- [Status](#status)
- [Performance](#performance)
- [Quick Start](#quick-start)
- [Demos](#demos)
- [Architecture](#architecture)
- [Tooling & IPC](#tooling--ipc)
- [Project Layout](#project-layout)
- [Building](#building)
- [Testing](#testing)
- [Documentation](#documentation)
- [Lineage & Acknowledgments](#lineage--acknowledgments)
- [License](#license)

---

## What NeoGPU Is

A **message-passing GPU substrate**. Rather than calling into a driver API, you
post messages onto a channel; the runtime routes them to nodes, which execute
and record them.

Three consequences fall out of that design:

- **Shared rendering lifecycle.** State/resource messages and retained draw
  contributions meet at the same backend, without pretending they have the
  same admission or ordering contract.
- **Message capture and replay.** `HSCAP1` owns the payloads of recorded legacy
  messages. Participant contributions are not silently added to that format.
- **A focused runtime.** Graphics, input, capture/replay, and tooling share
  the message substrate without model-loading or inference dependencies.

## Status

| Area | State | Notes |
|------|-------|-------|
| Message fabric | **Production-shaped** | Lock-free SPSC/MPSC, channel QoS, overflow accounting |
| Graphics (GBM/EGL/GLES3) | **Working on hardware** | Raspberry Pi 4, DRM/GBM/KMS page flips; verified on an 800x480 DSI panel, any connected display is accepted |
| Tooling / IPC | **Working** | Unix socket + TCP server, query/control surface |
| Capture & replay | **Working** | `HSCAP1`, owned payload journal, strict native-layout file format |
| NEON math library | **Working** | vec4/mat4/quat, SIMD-backed |
| Direct GPU compute (no Mesa) | **Working on hardware** | Own QPU encoder (`hs_qpu`) and kernel v3d runtime (`hs_v3d`), built by default; ships gather (default `HS_QPU_GATHER_BATCHES`/`_K`, the measured best) and store_one, checked word for word on the GPU. Rendering still goes through GLES |

**The runtime and the renderer are the product.** The former BitNet/ML
experiment has moved to a separate repository and is no longer included here.

## Performance

Measured on the Raspberry Pi 4 with GCC 15.2.0. Three-run medians of **applied**
messages per second, with every admitted message verified as applied:

| Producers | 1 | 2 | 4 | 8 | 16 |
|-----------|---:|---:|---:|---:|---:|
| Applied msg/s | 7,161,610 | 6,607,669 | 6,569,234 | 7,382,018 | 6,754,508 |

Runs used the OS scheduler, not pinned CPUs. The single-producer samples ranged
from 6,003,134 to 11,826,346 msg/s; do not treat a peak as a sustained guarantee.
The ledger retains every sample and the exact completion intervals.

Other verified measurements:

- Unix CLI: **100/100** correlated responses; three-run median **39.605 ms** total.
- TCP CLI: **100/100** correlated responses; three-run median **48.781 ms** total.
- Ray tracers (render rate, uncapped): `test_06_raycast` **446 FPS**,
  `test_08_raycast_orbit` **110 FPS**.

Frame rates are uncapped by default: present never waits for the display, so
they measure what the system can render. At each refresh (60 Hz on the DSI
panel) the panel shows the newest frame the GPU has finished, and the rest are
dropped; `tests/test_present_pacing` checks that a GPU-heavy uncapped frame
rate (~0.55 and ~0.95 of a refresh per frame) still puts a new frame on
screen at every refresh, including with every CPU core busy once
`sudo make install-kworker-rt` keeps the kernel workers that commit page
flips at real-time priority; `make pacing-tools` builds a flip tracer that
measures this for any program;
`make present-hardware` verifies with an external CRTC probe that presented
frames really are scanned out. Set `HSGraphics.vsync = true` to pace rendering
to the display instead.

**Reproduce:**

```bash
make benchmark
./neogpu_demo --benchmark-producers 1  # a single producer count, 2-second window
```

Old admission-only, eliminated-work math, and desynchronized IPC benchmark
headlines are not comparable to these measurements. See
[the P0 ledger](docs/audit/P0_REMEDIATION.md) for the failures, fixes, raw samples,
and rejected experiments. Message-only frame rates are not displayed GPU FPS.

> **Size and memory:** measure the actual artifact with `size neogpu_demo`.
> This executable contains numerous static test fixtures; it is not a minimal
> runtime-size benchmark. `bss` is not executable file size or measured resident
> memory. Recording allocates its independent payload journal lazily and releases
> it with `hs_gpu_shutdown()`. Exact layout/memory measurements are in the P0
> ledger; earlier tiny-binary and ~2 MB buffer claims are not current evidence.

### Direct GPU path vs. the reference stack

NeoGPU also drives the Pi 4's V3D GPU directly through the kernel's v3d
driver. That path uses no Mesa, no libdrm and no GLES, and its QPU programs
come from NeoGPU's own encoder. Below, it runs the same jobs on the same GPU
as the reference stack, Mesa 26.0.8's OpenGL ES 3.1 compute shaders, and as
the fastest CPU path (4 × Cortex-A72).

How it was measured (2026-11-16):
- every output word was checked on every trial, and every path was correct;
- NeoGPU and Mesa trials were interleaved;
- each figure is the median of 7 trials;
- Mesa's figure is its best of 12 configurations;
- there were 0 GPU resets.

| Workload | NeoGPU | Mesa | vs. Mesa | Best CPU | vs. CPU |
|---|---:|---:|---:|---:|---:|
| Process start to first checked result, GPU awake | 0.47 ms | 81 ms | **~172×** | — | — |
| Process start to first checked result, GPU runtime-suspended | 10.4 ms | 91 ms | 8.8× | — | — |
| Single-job round trip, p50 (12 runs; two modes per process, see below) | 61.3–62.1 or 73.5–74.0 µs | 83.9–86.0 or 101.8–105.1 µs | 1.35–1.43× | — | — |
| Pipelined small jobs, 1000 queued (12 runs) | 24.2k–27.2k jobs/s | 13.0k–14.1k or 24.2k–27.1k jobs/s | 0.98–1.87× | — | — |
| Gather `out[i] = table[idx[i]]`, sequential, 4M elements | 612.8 M/s | 345.2 M/s | **1.78×** | 245.2 M/s | 2.50× |
| Gather, random, 512 KiB table | 74.6 M/s | 73.3 M/s | 1.02× | 229.6 M/s | 0.32× |
| Gather, random, 64 MiB table | 36.3 M/s | 34.0 M/s | 1.07× | 42.2 M/s | 0.86× |
| CPU ↔ GPU message inside a running job (measured probe, not yet shipped) | 0.76 µs RTT, 1.2 M msg/s | 83.9–105.1 µs per job, 13.0k–27.1k jobs/s | **110–138×** latency, **44–92×** rate | — | — |

What the numbers say:
- **Per-job dispatch cost is largely shared.** NeoGPU's best is the
  kernel's floor of about 37 µs per job.
- **Dispatch timings fall into one of two modes per process.** A fast-or-slow
  split (about 12 µs per round trip) hits both paths, and Mesa's pipelined
  rate also halves in some runs. NeoGPU led the round trip in every run. The
  cause is not yet traced.
- **The in-job mailbox goes around that floor.** It is the route to an order
  of magnitude on CPU ↔ GPU traffic.
- **Random access is bound by memory, not by either GPU path.** Both GPU
  paths sit near DRAM's shared ~43 M/s ceiling. The CPU wins when the table
  fits in its 1 MiB L2 cache.
- **The ~10 ms cold cost is the GPU waking up.** The GPU suspends after
  100 ms idle, and both paths pay the 10.4 ms resume.

**Reproduce:**

```bash
make bench-vs-mesa        # exits 1 if any path's output is wrong
make tools/tmuop_probe
tools/bench_vs_mesa cold neo; tools/bench_vs_mesa cold mesa
flock build/gpu.lock tools/tmuop_probe M 2 3
```

Full method, every configuration and the limits of the comparison are in
[docs/benchmarks/VS_MESA.md](docs/benchmarks/VS_MESA.md). The hardware
measurements behind it are in
[docs/benchmarks/HARDWARE_MAP_2026-11-15.md](docs/benchmarks/HARDWARE_MAP_2026-11-15.md),
and the encoder is described in
[docs/benchmarks/QPU_ENCODER.md](docs/benchmarks/QPU_ENCODER.md).

## Quick Start

Requires an ARM Cortex-A72 (Raspberry Pi 4) with GCC and the GLES/GBM/DRM
development packages.

```bash
git clone https://github.com/anjaustin/neogpu.git
cd neogpu

make                 # build the core and the graphics test suite
./neogpu_demo        # run the headless test suite + benchmarks
```

`neogpu_demo` runs headless — no display or root required. The graphics
tests that drive the DRM/GBM backend do need the display and root:

```bash
sudo ./tests/test_08_raycast_orbit
```

### Retained participant frames

`hs_gpu_init()` prepares the participant runtime by default. Producers can use
the bounded join/submit/leave API; normal `hs_gpu_step()`, `hs_gpu_process()` and
end-frame calls service it. No application-owned `HSFabric` or manual
step/compose plumbing is required.

This is the default runtime path, not an experimental build switch. `make test`
includes participant lifecycle/composition and report-integrity regressions;
`make sanitize` and `make tsan` include their participant checks. The standard
`make test-hardware` (with sudo when device/capture ownership requires it) also
runs normal-HSGpu participant GLES lifecycle and
full-frame equivalence checks, saving a fresh `tests/evidence-frame-runtime-*`
collection. `make gpu-participants-hardware` runs just those graphics checks.

```c
/* Renderer first prepares an immutable vertex bank for the returned slot. */
HSParticipant owner;
if (hs_gpu_participant_join(&gpu, &owner) != HS_FABRIC_OK) return 1;
HSParticipantFrame contribution = {.sequence = 1, .count = 1};
contribution.draws[0] = (HSRenderCmd){.op = HS_RC_DRAW, .a = owner.slot};
if (hs_gpu_participant_submit(&gpu, owner, &contribution) != HS_FABRIC_OK) return 1;

hs_gpu_begin_frame(&gpu);
hs_gpu_clear(&gpu, v4_make(0, 0, 0, 1));
hs_gpu_process(&gpu); /* Legacy messages still require explicit processing. */
if (!hs_gpu_end_frame_checked(&gpu)) return 1;
```

The latest applied contribution persists until replaced by another complete
submission, an empty contribution or retirement. Draws share renderer-global
state and use renderer-owned immutable banks; this is not private shader/texture
state or arbitrary upload ownership. Accepted FIFO application does not promise
presentation of every intermediate contribution. Stop/join producers before
shutdown; handles must not survive shutdown/reinitialization.

Legacy `hs_gpu_*` message APIs remain supported. Their channel budgets, fences,
IPC counters and capture/replay retain their message-only scope. End-frame does
not implicitly drain legacy sends. Retained draws follow legacy commands before
trailing frame-end/present markers. Combined overflow skips backend execution
rather than presenting a partial scene; the checked API reports failure and
begin-frame resets the overflow flag for recovery.

See [integration contract/results](docs/audit/FRAME_RUNTIME_INTEGRATION.md) and
[general-message opportunities](docs/GENERAL-MESSAGE-POTENTIALS.md). Existing
message-only benchmark numbers do not establish an integration or display-FPS
speedup.

The [GLES/capture investigation](docs/audit/GLES_WORK_RESULTS.md) records remaining
CPU costs and rejected GL-state cache experiments. Production rendering retains
its original draw setup and capture defaults; successful graphics teardown now
releases the selected DRM connector and EGL context/display resources. Standard
regression and sanitizer targets include headless backend-state checks, distinct
from actual pixel verification and performance collection.

## Demos

The graphics tests are the clearest proof surface. They run on the Pi's
VideoCore via DRM/GBM.

| Test | What it shows | Verified result |
|------|---------------|-----------------|
| `test_01_clear` | Clear screen | ✅ |
| `test_02_triangle` | Single triangle | ✅ |
| `test_03_instancing` | Hardware instancing | ✅ |
| `test_04_blending` | Alpha blending | ✅ |
| `test_05_cube3d` | 3D rotating cube | ✅ |
| `test_06_raycast` | Shader-driven spheres + live FPS HUD | ✅ |
| `test_07_message_triangle` | Message-driven render path | ✅ |
| `test_08_raycast_orbit` | Full ray tracer: spheres, shadows, reflections, ground plane | ✅ **110 FPS** (uncapped render rate) |

`test_08` is the showpiece — a real ray tracer (sphere intersection, shadow
rays, one reflection bounce, Phong + rim lighting) running on a Cortex-A72.

```bash
make build-tests
sudo ./tests/test_08_raycast_orbit
```

Additional demos live in `tools/`:

| Tool | What it is |
|------|-----------|
| `neogpu_pong` | Pong drawn with direct GLES on the display path; Up/Down or W/S in the launching terminal, P screenshot, Q quit |
| `neogpu_pong_ai` | Self-playing Pong rendered entirely through messages (`OP_DRAW`, `OP_DRAW_TEXT` commentary), then captured, written, re-read and replayed |
| `neogpu_procgame` | Self-playing procedural shooter (no keyboard input; Ctrl-C quits) |
| `neogpu_capture_demo` | Records an `HSCAP1` capture, writes it, reads it back and replays the file |
| `neogpu_tool` | Standalone IPC client for querying a running instance |

The demos need root and a display, and write their files (captures,
screenshots, pong's high scores) to the working directory. They find the font
via `$NEOGPU_FONT`, `src/medodica_font` in the working directory, or next to
the executable.

## Architecture

```
        ┌──────────────────────────────────────────┐
        │              Application                  │
        │   render commands · IPC requests · ops    │
        └───────────────────┬──────────────────────┘
                            │  messages
        ┌───────────────────▼──────────────────────┐
        │        Channel Fabric (QoS)               │
        │  RT (critical) · RENDER (interactive)     │
        │  TELEM (background, auto-drop)            │
        └───────────────────┬──────────────────────┘
                            │
        ┌───────────────────▼──────────────────────┐
        │   Nodes: Shader · Buffer · Texture ·      │
        │          Output · Sound · Input           │
        └───────────────────┬──────────────────────┘
                            │
        ┌───────────────────▼──────────────────────┐
        │   Backend: GBM / EGL / GLES3 → DRM        │
        └──────────────────────────────────────────┘
```

## Tooling & IPC

The runtime exposes its own internals for inspection and control.

**In-process tooling:**

```bash
./neogpu_demo --tool --query-stats --query-fabric
./neogpu_demo --tool --set-record-mask 0x00000006 --query-stats
./neogpu_demo --tool --set-budget 2 8192 --query-stats
```

**IPC server mode:**

```bash
./neogpu_demo --ipc-server "$XDG_RUNTIME_DIR/neogpu.sock" --ipc-ms 5000   # Unix socket (mode 0600)
./neogpu_demo --ipc-tcp 8765 --ipc-ms 5000                                 # TCP/IP, 127.0.0.1
```

**Client:**

```bash
make tool
./neogpu_tool query-stats                       # default: $XDG_RUNTIME_DIR/neogpu.sock, else /tmp/neogpu-<uid>.sock
./neogpu_tool --host localhost --port 8765 query-fabric
```

TCP server mode binds to loopback by default; `--ipc-bind 0.0.0.0` listens on
every interface and should only be used on a trusted network (the protocol is
unauthenticated). The client only talks to a Unix-socket server owned by the
same user or root. Start the matching server before running a client; Unix and
TCP are separate server modes. Services have finite I/O deadlines and bounded
admission. Channel budgets must be at least 1: a zero budget would stop the
channel draining, so `set-budget` rejects it.

**Channel semantics:**

| Channel | QoS | Budget (msgs/tick) | Behavior |
|---------|-----|--------------------|----------|
| RT | Critical | 4096 | Strict priority, no drops |
| RENDER | Interactive | 16384 | Fair share, limited drops |
| TELEM | Background | 1024 | Best-effort, auto-drop |

## Project Layout

```
include/            Public headers
  hs_core.h           Core system, OpCodes, message ABI
  hs_msg.h            Message packing/unpacking (inline)
  hs_gpu.h            High-level GPU API
  hs_nodes.h          Node message handlers
  hs_graphics.h       GBM/EGL/GLES display path (KMS page flips)
  hs_backend_gles.h   GLES backend for the message runtime
  hs_ipc.h            IPC server
  hs_fabric.h         Participant frame fabric
  hs_input_stream.h   evdev input stream
  hs_text.h           Text rendering
  hs_math_neon.h      NEON vec4/mat4/quat math
  hs_neonCompat.h     <arm_neon.h>, or a tested portable fallback

src/                Core runtime (graphics + substrate)
  hs_core.c           Message queue, OpCodes, system core
  hs_nodes.c          Node handlers (Shader, Buffer, Texture, Output, Sound)
  hs_gpu.c            High-level GPU API
  hs_async.c          Async completion handling
  hs_fabric.c         Participant frame fabric
  hs_backend_gles.c   GBM/EGL/GLES3 backend
  hs_ipc.c            IPC server (Unix + TCP), tool-bus
  hs_input_stream.c   evdev readers, hotplug, resync
  hs_text.c           Text rendering
  main.c              Headless test suite + benchmarks
  benchmark.c         Historical port comparison (make bench-legacy)

tests/              Graphics + system tests (test_01 … test_08)
tools/              Standalone demos and utilities
docs/               Documentation (see docs/README.md)
```

## Building

**Native ARM (Cortex-A72):**

```bash
make                  # core + tests
```

**Cross-compile:**

```bash
make CC=aarch64-linux-gnu-gcc STRIP=aarch64-linux-gnu-strip build
```

**Targets:**

| Target | What it does |
|--------|--------------|
| `make` | Build `neogpu_demo` and the graphics test suite |
| `make build` | Build `neogpu_demo` only |
| `make build-tests` | Build the graphics test suite |
| `make test` | Headless regression gate (no display needed) |
| `make check-sources` | Compile every C source with `-Werror` |
| `make check-scalar` | Run the headless suite on the portable intrinsics fallback |
| `sudo make test-hardware` | Display, presentation and evdev checks (needs root + display) |
| `sudo make present-hardware` | Presentation checks only: page flips observed from outside the process |
| `make bench-legacy` | Historical port comparison benchmark |
| `make tool` | Build the standalone IPC client |
| `make run-test N=06` | Build and run one test (needs display + root) |
| `make clean` | Remove build artifacts |
| `make version` | Print the version |

## Testing

```bash
make test                          # headless checks + P0 + Unix/TCP CLI contracts
make sanitize                      # AddressSanitizer + UndefinedBehaviorSanitizer
make tsan                          # ThreadSanitizer, including producer benchmark
make benchmark                     # explicitly separate performance measurement
make build-tests                    # graphics tests (need display + root)
sudo ./tests/test_08_raycast_orbit  # the ray tracer
```

`make test` exercises the message fabric, node handlers, IPC, thread safety,
overflow handling, capture persistence, and synthetic input — without a display.
`sudo make test-hardware` additionally records framebuffer evidence, verifies
presentation with an external scanout probe (every presented frame must be a
page flip the display controller performs, in order, with the console handed
back on exit), checks rendered text against the font atlas, and exercises
virtual evdev devices (hot-add/remove, `SYN_DROPPED` resync, unplug). It
requires DRM and `/dev/uinput` access and does not grab physical input devices.

## Documentation

Full documentation index: **[docs/README.md](docs/README.md)**

| Area | Where to look |
|------|---------------|
| Design & architecture | [docs/architecture/](docs/architecture/) |
| Wire formats & interfaces | [docs/interface/](docs/interface/) |
| Benchmarks & methodology | [docs/benchmarks/](docs/benchmarks/) |
| PicoGPU relationship | [docs/lineage/](docs/lineage/) |
| Self-audit | [docs/audit/DIAMOND.md](docs/audit/DIAMOND.md) |
| Historical material | [docs/archive/](docs/archive/) |

Contributing guidelines: [CONTRIBUTING.md](CONTRIBUTING.md).
Changelog: [CHANGELOG.md](CHANGELOG.md).

## Lineage & Acknowledgments

NeoGPU began as, and remains, a tribute to [**PicoGPU**](https://github.com/ncannasse/picogpu)
by **Nicolas Cannasse** (creator of Haxe). PicoGPU is a ~300 KB software GPU
written in Haxe that compiles to WebGL/HashLink — and it proved the idea at
the heart of this project: that a complete, tiny GPU runtime can be built in
software and still be genuinely useful.

NeoGPU takes that spark somewhere new — a header-oriented **C** implementation
targeting the **Raspberry Pi 4's V3D** on bare-metal DRM/GBM, with a
message-native core. The architectures have
diverged a great deal; that divergence is a measure of how far a good idea
travels, not a reason to forget where it started.

The debt is specific. PicoGPU showed the way on: a compact, self-contained GPU
runtime; the feature surface NeoGPU still mirrors (vertex/fragment buffers,
blending, culling, depth/stencil, color mask, render targets, instancing,
matrix/vector/quaternion math, textured UV mapping); and the 4-channel, 48 kHz
audio model. Where NeoGPU goes its own way — native ARM/NEON, Pi4 V3D, a
message-passing substrate — it does so standing on that
foundation.

Thank you, Nicolas. See [CREDITS.md](CREDITS.md) for the full lineage.

## License

MIT — see [LICENSE](LICENSE).
