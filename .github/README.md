# NeoGPU

[![Build](https://github.com/anjaustin/neogpu/actions/workflows/ci.yml/badge.svg)](https://github.com/anjaustin/neogpu/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../LICENSE)
[![Language: C](https://img.shields.io/badge/language-C%20%7C%20Make-blue.svg)](https://en.wikipedia.org/wiki/C_(programming_language))

**A message-native runtime substrate for ARM/NEON — graphics, tooling, and
experimental inference on the Raspberry Pi 4.**

Every operation is a message. Render commands, IPC requests, and inference ops
all travel the same lock-free channels, so one fabric carries the whole
system.

```c
HSGpu gpu;
hs_gpu_init(&gpu);
hs_gpu_set_shader(&gpu, 0);
hs_gpu_load_buffer(&gpu, 0);
hs_gpu_draw(&gpu, 0);
```

## Quick start

```bash
git clone https://github.com/anjaustin/neogpu.git
cd neogpu

make                 # core build - no inference dependencies
make build-tests     # graphics test suite
./neogpu_demo        # headless test suite + benchmarks
```

Graphics tests that drive the DRM/GBM backend need the display and root:

```bash
sudo ./tests/test_08_raycast_orbit
```

## Performance

Measured on a Raspberry Pi 4 (Cortex-A72, 800x480 DSI).

| Metric | Value |
|--------|-------|
| **Message layer throughput** | **12,780,000 msg/s** (2 producers) |
| Message layer (1–16 producers) | 6.36M – 12.78M msg/s |
| GPU frames | ~414K fps |
| NEON math | ~14B ops/sec |
| IPC throughput | ~4,700 req/sec |

Reproduce the headline number with `make && ./neogpu_demo` — the
`Comms Producer Benchmark` section prints throughput per producer count.

## What's here

| Area | State |
|------|-------|
| Message fabric (lock-free SPSC/MPSC, channel QoS) | Production-shaped |
| Graphics backend (GBM/EGL/GLES3 on DRM) | Working on hardware |
| Tooling / IPC (Unix socket + TCP) | Working |
| Capture & replay (`HSCAP1`) | Working |
| NEON math library | Working |
| BitNet inference | **Experimental** — opt-in via `make ml`, off by default |

The runtime and the renderer are the product; the inference engine is a
contained experiment, deliberately kept out of the core build.

## Highlights

- **`test_08_raycast_orbit`** — a real ray tracer (sphere intersection, shadow
  rays, one reflection bounce, Phong + rim lighting) running on a Cortex-A72
  at **107 FPS**.
- **Capture and replay come free** — because work is messages, a frame *is* a
  log. Captures are portable, inspectable, and replayable.
- **Clean core/ML separation** — the core build compiles no inference sources
  at all, enforced by `#ifdef HS_ENABLE_ML` and separate object trees.

## Documentation

**[docs/README.md](../docs/README.md)** — full index.

Design specs in [docs/architecture/](../docs/architecture/), wire formats in
[docs/interface/](../docs/interface/), benchmarks and methodology in
[docs/benchmarks/](../docs/benchmarks/), the PicoGPU relationship in
[docs/lineage/](../docs/lineage/), and a full self-audit in
[docs/audit/DIAMOND.md](../docs/audit/DIAMOND.md).

Contributing: [CONTRIBUTING.md](../CONTRIBUTING.md) ·
Security: [SECURITY.md](../SECURITY.md) ·
Changelog: [CHANGELOG.md](../CHANGELOG.md)

## Lineage

NeoGPU began as, and remains, a tribute to
[**PicoGPU**](https://github.com/ncannasse/picogpu) by **Nicolas Cannasse**
(creator of Haxe) — the project that proved a complete, tiny GPU runtime can
be built in software and still be genuinely useful.

NeoGPU takes that spark somewhere new: a header-oriented **C** implementation
targeting the **Raspberry Pi 4's V3D** on bare-metal DRM/GBM, with a
message-native core. The architectures have diverged a great deal; that
divergence is a measure of how far a good idea travels, not a reason to forget
where it started.

Thank you, Nicolas. See [CREDITS.md](../CREDITS.md) for the full lineage.

## License

MIT — see [LICENSE](../LICENSE).
