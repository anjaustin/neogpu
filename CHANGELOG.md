# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Build fails on current Debian/Ubuntu (including the CI runner)**. The
  build did not compile at `src/hs_async.c`: `include/hs_graphics.h` includes
  `<xf86drm.h>`, and libdrm's `xf86drm.h` does `#include <drm.h>`, but
  libdrm installs `drm.h` under `/usr/include/libdrm`. libdrm's own
  `libdrm.pc` prescribes `-I${includedir}/libdrm` for exactly this reason.
  The Makefile now adds `-I/usr/include/libdrm`; gcc ignores a `-I` path that
  does not exist, so this is harmless on systems that lay the headers out
  differently. Verified by building the core, all eight graphics tests, the
  experimental ML target and the standalone IPC client for aarch64.
- **CI was missing a build dependency.** `EGL/eglplatform.h` includes
  `<KHR/khrplatform.h>`, which ships in `libgl-dev` — not in `libegl-dev`
  or `libgles-dev`. Every external header the build needs was mapped to its
  package and the CI dependency list corrected.
- **Buffer overrun in `hs_ml_lmhead_stage1_i8()`** (`src/hs_ml_ternary_neon.c`).
  The int8 activation dequantization loop was bounded by `K * 4` instead of
  `K`, where `K` is the number of hidden dimensions. With the BitNet b1.58
  2B-4T hidden size (`K = 2560`) that read 10240 bytes from a caller's
  2560-byte activation buffer, and wrote 10240 floats into the function's
  own 4096-float static scratch buffer — a 24 KB overrun that landed on
  adjacent globals. Confirmed with AddressSanitizer against the real
  source: `heap-buffer-overflow` on the read, `global-buffer-overflow` on
  the write. The loop bound is now `K`, the scratch buffer is local rather
  than shared static, and a `K` beyond the 4096-element capacity is
  rejected with a diagnostic instead of overrunning memory.
  Regression test: `tests/redteam_lmhead_i8_bounds.c`, also reachable as
  `make test-i8-bounds` and run in CI.

### Changed
- **`hs_ml_lmhead_stage1_i8()` documentation corrected**
  (`include/hs_ml_infer.h`). The header described the function as "int8
  activation path (fastest) ... Uses vmull_s8 integer multiply — 3.4×
  faster than F16 baseline". That described a retired implementation: the
  function contains no NEON integer arithmetic at all and simply
  dequantizes to float32 and delegates to `hs_ml_lmhead_stage1()`. The
  float LUT kernel is the fast path. The formula in the comment was and
  remains correct; the performance claim was not.

## [0.2.0] - 2026-11-11

Everything below accumulated since 0.1.0. The project had no release tags
before this one, so the earlier in-progress work is folded in here rather
than left as a phantom release.

### Changed
- **Repository restructured into professional shape** (three staged commits)
  - *Stage 1 - hygiene*: untracked 17 committed ELF binaries and one `.bak`
    file; hardened `.gitignore` with explicit per-name binary rules (not
    globs, so `tests/*.c` and `tools/*.c` sources are never matched); moved
    `bench_coproc.c` to `tools/`; moved detail docs off the repo root into
    `docs/`. No source deleted, no build impact.
  - *Stage 2 - ML/core separation*: `hs_ipc.c` ML tool-bus path wrapped in
    `#ifdef HS_ENABLE_ML` (include, global ML state, `hs_ipc_handle_ml_op()`,
    and the `OP_ML_*` dispatch). The `OP_ML_*` enum lives in `hs_core.h`, so
    the `case` labels still compile with ML off. `HS_SRC` no longer pulls in
    any ML source; the default `make` builds the core with zero inference
    dependencies, and `make ml` opts in.
  - *Stage 3 - front door*: rewrote `README.md` as a project front door
    (positioning, status table, reproducible performance figures, demo table,
    architecture, build matrix, documentation index). Added `CONTRIBUTING.md`,
    `SECURITY.md`, `.github/README.md` (profile README), CI workflow, issue
    templates, and a pull-request template. Reorganized `docs/` into
    `architecture/`, `interface/`, `research/`, `benchmarks/`, `lineage/`,
    `audit/`, and `archive/`, with a `docs/README.md` index.
- **Performance claim corrected**: the message layer is documented at
  **12,780,000 msg/s** at two producers, within a measured range of
  6.36M-12.78M msg/s across 1-16 producers. The previous "1,000,000+ fps"
  figure predated the lock-free optimization work and understated the result
  by more than an order of magnitude. Reproduce with `make && ./neogpu_demo`.
- **Lineage credit strengthened**: the PicoGPU acknowledgment is now specific
  about what was inherited and where NeoGPU diverges, rather than a generic
  attribution. `docs/audit/DIAMOND.md` carries a dated note reversing its
  original "vestigial framing" verdict.

### Fixed
- **`make tool` did not exist.** The README documented it, but the Makefile
  only had a `.PHONY` declaration and an unused variable. Added a real rule.
- **`make neogpu_pong_llm` could not build from a clean tree.** Eight ML
  objects were used in the link line but never declared as prerequisites, so
  make never built them. All tool targets now declare every object they link.
- **Core and ML builds shared object files.** `make ml` compiled the core
  sources into the same `src/*.o` with an extra `-DHS_ENABLE_ML`, so a plain
  `make` afterwards could silently reuse ML-enabled objects and vice versa.
  The ML build now uses its own object tree (`build/ml/`) and produces a
  distinct binary (`neogpu_demo_ml`).
- **`make run-test` wrote to `/tmp`**, contradicting the project's own
  `AGENTS.md` rule that test programs live in the project tree. Now builds
  into `tests/`, links `hs_backend_gles.o` and `hs_text.o` (several tests
  need them), and lists all eight tests in its usage message.
- **`.gitignore`'s `journal/` rule matched at any depth**, so relocating the
  journal under `docs/` would have silently untracked all twelve files.
  Root-anchored to `/journal/`.
- **Documentation cross-references updated** after the `docs/` reorganization,
  including in `include/hs_msg.h`, `include/hs_ml_routing.h`,
  `src/hs_ml_binary.c`, and `src/hs_ml_mt7.c`.

### Added
- **`test_08_raycast_orbit`**: full ray tracer with five orbiting spheres, a
  central sphere, a checkered ground plane, shadow rays, one reflection
  bounce, and Phong + rim lighting. **107 FPS** on a Raspberry Pi 4.
- **Seven-segment FPS HUD** in `tests/test_06_raycast.c`, replacing the
  broken font-atlas path with self-contained rectangle digits - no atlas, no
  UVs, and no way to smear across the viewport.
- **CI workflow** (`.github/workflows/ci.yml`) building and running the
  headless suite on GitHub's ARM64 runners.
- **Issue and pull-request templates**, `CONTRIBUTING.md`, `SECURITY.md`, and
  a `docs/README.md` documentation index.

### Archived
- `docs/archive/` now holds the LLM conversation corpus (`lmm/`, 48 files),
  the LMM theory explorations (`lmm-theory/`, 4 files), the development
  journal (12 files), and four superseded remediation plans. Nothing was
  deleted; every move is a history-preserving rename.

### Added
- **Input Streaming System** (`hs_input_stream.h`, `src/hs_input_stream.c`)
  - Lock-free SPSC queue with 64-byte cache line alignment
  - Per-device evdev reader threads
  - Double-buffered HSInput for zero-contention game loop access
  - EVIOCGRAB for exclusive device access
  - Hotplug detection via inotify
  - Device capability query via EVIOCGBIT
- **NODE_INPUT Integration** (`hs_nodes.c`)
  - `input_node_init/reset/process()` implementations
  - `OP_INPUT_KEY`, `OP_INPUT_MOUSE`, `OP_INPUT_GAMEPAD` message handlers
- **New Opcodes** (`hs_core.h`)
  - `OP_INPUT_KEY`, `OP_INPUT_MOUSE`, `OP_INPUT_GAMEPAD`
  - `OP_INPUT_DEVICE_ADD`, `OP_INPUT_DEVICE_REMOVE`

### Fixed (13 issues from red-team audit)
- SPSC false sharing (critical performance bug)
- Dropped events untracked
- Double-buffering broken (process thread ignored active state)
- Thread exit fd leak
- Failed start cleanup (orphan threads)
- swap_buffers race condition
- Sticky button state
- No EVIOCGRAB (event duplication)
- Axis mapping conflicts (ABS_Z/ABS_RX)
- Polling inefficiency (blocking instead of polling)
- No HSSystem integration for input
- No capability query
- No hotplug support

### Documentation
- `docs/INPUT_STREAM.md` - Complete input streaming system documentation
- `docs/INPUT_STREAM_REMEDIATION.md` - Red-team analysis and fixes
- `docs/REMEDIATION_PLAN.md` - Updated with input system status

## [0.1.0] - 2024-03-13

### Added
- **TCP/IP IPC Server**: Remote tooling support via `hs_ipc_start_tcp()` on port 8765
- **Explicit QoS Semantics**: RT/RENDER/TELEM channels with guaranteed budgets
  - RT (CHAN_RT): Critical priority, 4096 msgs/tick, no drops
  - RENDER (CHAN_RENDER): Interactive, 16384 msgs/tick, limited drops
  - TELEM (CHAN_TELEM): Background, 1024 msgs/tick, auto-drop when full
- **Per-Channel High-Water Marks**: `submit_hw[CHAN_COUNT]` tracks peak queue depth
- **TELEM Drop Tracking**: `telem_dropped[CHAN_COUNT]` counts dropped messages
- **Standalone neogpu_tool**: Build with `make tool` for remote IPC client
- **IPC Latency Profiling**: Send/wait breakdown in benchmark
- **Red-Team Stress Tests**: Flood, bad-op, disconnect tests

### Performance
- Message layer: ~8.5M msgs/sec
- GPU frames: ~414K fps
- NEON math: ~14B ops/sec
- IPC throughput: ~4,700 req/sec

### Fixed
- SIGPIPE handling in IPC tests
- Buffer overflow in fabric query response (64→96 bytes)

## [0.0.1] - 2024-03-10

### Added
- Initial release
- ARM NEON-optimized message-passing GPU layer
- Channelized fabric (RT/RENDER/TELEM)
- Toolbus for IPC result correlation
- Unix socket IPC server
- In-process tooling via `--tool`
- 135 tests passing

[0.1.0]: https://github.com/anjaustin/neogpu/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/anjaustin/neogpu/tree/v0.0.1
