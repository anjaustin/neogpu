# NeoGPU

**A message-native runtime substrate for ARM/NEON — graphics, tooling, and
experimental inference on the Raspberry Pi 4.**

[![Build](https://github.com/anjaustin/neogpu/actions/workflows/ci.yml/badge.svg)](https://github.com/anjaustin/neogpu/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Language: C](https://img.shields.io/badge/language-C%20%7C%20Make-blue.svg)](https://en.wikipedia.org/wiki/C_(programming_language))

NeoGPU is a small, complete GPU runtime where *every* operation is a message.
Rendering commands, IPC requests, and (in the experimental path) inference ops
all travel the same lock-free channels — so one fabric carries the whole
system.

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

- **One fabric, many workloads.** The same queue carries render commands and
  IPC requests. Adding a new workload type does not mean adding a new
  transport.
- **Capture and replay come free.** Because work is messages, a frame *is* a
  log. `HSCAP1` captures are portable, inspectable, and replayable.
- **The substrate is separable.** The graphics core builds with no inference
  dependencies at all. The BitNet work is a real experiment, not a
  load-bearing part of the runtime.

## Status

| Area | State | Notes |
|------|-------|-------|
| Message fabric | **Production-shaped** | Lock-free SPSC/MPSC, channel QoS, overflow accounting |
| Graphics (GBM/EGL/GLES3) | **Working on hardware** | Verified on Raspberry Pi 4, DRM/GBM, 800x480 DSI |
| Tooling / IPC | **Working** | Unix socket + TCP server, query/control surface |
| Capture & replay | **Working** | `HSCAP1` format, round-trip verified |
| NEON math library | **Working** | vec4/mat4/quat, SIMD-backed |
| BitNet inference | **Experimental** | Opt-in, off by default — see [below](#architecture) |

The honest summary: **the runtime and the renderer are the product; the
inference engine is a contained experiment.**

## Performance

Measured on a Raspberry Pi 4 (Cortex-A72, 800x480 DSI display).

| Metric | Value | Source |
|--------|-------|--------|
| **Message layer throughput** | **12,000,000 msg/s** | Producer benchmark, `./neogpu_demo` |
| Message layer (measured range) | 8.2M – 16.9M msg/s | 1–16 producer threads |
| GPU frames | ~414K fps | Message-path frame rate |
| NEON math | ~14B ops/sec | Scalar vs NEON mat4/vec4 |
| IPC throughput | ~4,700 req/sec | Round-trip latency test |

**Reproduce the message-layer number:**

```bash
make && ./neogpu_demo
```

The `Comms Producer Benchmark` section prints throughput per producer count:

```
  1 thr,  500 ms: 12000000 msg/s (ok=60000000 fail=0 spsc_ok=60000000 ...)
```

That figure is the headline because it is the one that is *hard*. Rendering
performance is bounded by the display; the fabric's throughput is bounded by
cache-line contention between producers and the consumer, and it took a dozen
red-teamed optimization rounds to get there. The full optimization log,
including the rejected ideas, is in
[docs/benchmarks/ATOMICS_PROFILE_2026-03-13.md](docs/benchmarks/ATOMICS_PROFILE_2026-03-13.md).

> **Binary size:** the core runtime is ~25 KB of `text`+`data` (stripped).
> Verify it yourself with `size neogpu_demo` after `make` — `text` and `data`
> are the loadable code, while `bss` is the ~2 MB of buffers allocated at
> runtime. Older notes in this repo also quote a "73 KB" figure for the whole
> file; that was a different measurement of a different artifact, and the
> `size` output is the number to trust.

## Quick Start

Requires an ARM Cortex-A72 (Raspberry Pi 4) with GCC and the GLES/GBM/DRM
development packages.

```bash
git clone https://github.com/anjaustin/neogpu.git
cd neogpu

make                 # build the core (no ML required)
make build-tests     # build the graphics test suite
./neogpu_demo        # run the headless test suite + benchmarks
```

`neogpu_demo` runs headless — no display or root required. The graphics
tests that drive the DRM/GBM backend do need the display and root:

```bash
sudo ./tests/test_08_raycast_orbit
```

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
| `test_08_raycast_orbit` | Full ray tracer: spheres, shadows, reflections, ground plane | ✅ **107 FPS** |

`test_08` is the showpiece — a real ray tracer (sphere intersection, shadow
rays, one reflection bounce, Phong + rim lighting) running on a Cortex-A72.

```bash
make build-tests
sudo ./tests/test_08_raycast_orbit
```

Additional demos live in `tools/`:

| Tool | What it is |
|------|-----------|
| `neogpu_pong` | Pong, driven by the message fabric |
| `neogpu_pong_ai` | Pong with simulated commentary |
| `neogpu_pong_llm` | Pong alongside real BitNet inference *(experimental)* |
| `neogpu_capture_demo` | Records an `HSCAP1` capture to disk |
| `neogpu_tool` | Standalone IPC client for querying a running instance |

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

### Core vs. experimental

The build is split so the two concerns are structurally separate, not just
nominally:

| | Core (`make`) | ML (`make ml`) |
|---|---|---|
| Sources | `src/hs_core.c`, `hs_nodes.c`, `hs_gpu.c`, `hs_async.c`, `hs_backend_gles.c`, `hs_ipc.c`, `hs_text.c`, `main.c` | Core **plus** `hs_ml.c`, `hs_ml_loader.c`, `hs_ml_ternary_mt.c` |
| `hs_ipc.c` ML ops | Compiled out (`#ifdef HS_ENABLE_ML`) | Compiled in |
| Object tree | `src/*.o` | `build/ml/*.o` (separate) |
| Binary | `neogpu_demo` | `neogpu_demo_ml` |

The object trees are deliberately separate so the two builds can never
contaminate each other — a plain `make` after `make ml` cannot silently
reuse ML-enabled objects.

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
./neogpu_demo --ipc-server /tmp/neogpu.sock --ipc-ms 5000   # Unix socket
./neogpu_demo --ipc-tcp 8765 --ipc-ms 5000                  # TCP/IP
```

**Client:**

```bash
make tool
./neogpu_tool --sock /tmp/neogpu.sock query-stats
./neogpu_tool --host localhost --port 8765 query-fabric
```

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
  hs_graphics.h       GBM/EGL/GLES graphics backend
  hs_text.h           Text rendering
  hs_math_neon.h      NEON vec4/mat4/quat math
  hs_ml*.h            Experimental ML interfaces

src/                Core runtime (graphics + substrate, no ML)
  hs_core.c           Message queue, OpCodes, system core
  hs_nodes.c          Node handlers (Shader, Buffer, Texture, Output, Sound)
  hs_gpu.c            High-level GPU API
  hs_async.c          Async completion handling
  hs_backend_gles.c   GBM/EGL/GLES3 backend
  hs_ipc.c            IPC server (Unix + TCP), tool-bus
  hs_text.c           Text rendering
  main.c              Headless test suite + benchmarks
  hs_ml*.c            Experimental ML (only in the `make ml` build)
  archive_ternary_kernels/  Historical ternary kernel variants

tests/              Graphics + system tests (test_01 … test_08)
tools/              Standalone demos and utilities
docs/               Documentation (see docs/README.md)
experimental/BitNet/  Vendored Microsoft bitnet.cpp reference (MIT)
```

## Building

**Native ARM (Cortex-A72):**

```bash
make                  # core + tests
make ml               # opt-in experimental ML build → neogpu_demo_ml
```

**Cross-compile:**

```bash
make CC=aarch64-linux-gnu-gcc STRIP=aarch64-linux-gnu-strip build
```

**Targets:**

| Target | What it does |
|--------|--------------|
| `make` / `make build` | Build `neogpu_demo` (core only) |
| `make build-tests` | Build the graphics test suite |
| `make ml` | Build `neogpu_demo_ml` (core + experimental ML) |
| `make tool` | Build the standalone IPC client |
| `make run-test N=06` | Build and run one test (needs display + root) |
| `make clean` | Remove build artifacts |
| `make version` | Print the version |

## Testing

```bash
make build && ./neogpu_demo        # headless: system tests + benchmarks
make build-tests                    # graphics tests (need display + root)
sudo ./tests/test_08_raycast_orbit  # the ray tracer
```

`neogpu_demo` exercises the message fabric, node handlers, IPC, thread safety,
overflow handling, and the producer benchmark — all without a display.

## Documentation

Full documentation index: **[docs/README.md](docs/README.md)**

| Area | Where to look |
|------|---------------|
| Design & architecture | [docs/architecture/](docs/architecture/) |
| Wire formats & interfaces | [docs/interface/](docs/interface/) |
| Research & concepts | [docs/research/](docs/research/) |
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
message-native core and experimental BitNet inference. The architectures have
diverged a great deal; that divergence is a measure of how far a good idea
travels, not a reason to forget where it started.

The debt is specific. PicoGPU showed the way on: a compact, self-contained GPU
runtime; the feature surface NeoGPU still mirrors (vertex/fragment buffers,
blending, culling, depth/stencil, color mask, render targets, instancing,
matrix/vector/quaternion math, textured UV mapping); and the 4-channel, 48 kHz
audio model. Where NeoGPU goes its own way — native ARM/NEON, Pi4 V3D, a
message-passing substrate, ML in the same runtime — it does so standing on that
foundation.

Thank you, Nicolas. See [CREDITS.md](CREDITS.md) for the full lineage.

NeoGPU also incorporates the experimental `bitnet.cpp` reference
implementation from Microsoft (MIT, see [experimental/BitNet/](experimental/BitNet/)),
used for comparison against NeoGPU's own ternary kernels.

## License

MIT — see [LICENSE](LICENSE).
