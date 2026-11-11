# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Commit dates in this repository's history are unreliable: the Raspberry Pi 4
that authored most commits has no real-time clock and its clock drifted by
days. Release headings use the date recorded at the time.

## [Unreleased]

### Fixed (2026-11-13 audit; see docs/audit/AUDIT_2026-11-13.md)
- **Display presentation.** Frames after the first were never page-flipped:
  the CRTC kept scanning out one buffer that was then rendered into, so most
  frames never appeared and the rest tore. Every present is now a KMS page
  flip of its own buffer, verified from outside the process by
  `tests/scanout_probe`. Rendering stays uncapped by default (frames the panel
  cannot show are dropped, never waited for or torn); `vsync = true` paces to
  the display. Render rates are unchanged: 446 FPS (`test_06`), 110 FPS (`test_08`).
- Display takeover now finds a CRTC through `possible_crtcs` when none is
  active, refuses a display whose mode does not match the surface instead of
  failing with `EINVAL` silently, checks `AddFB`/`SetCrtc` results, and a failed
  graphics init releases its descriptor, GBM/EGL objects and DRM master.
- `OP_SET_CHAN_BUDGET` / `hs_set_channel_budget()` accepted 0, which stopped the
  channel draining; on RT one IPC request froze all IPC. Budgets must be >= 1.
- The MPMC submit ring spun forever at the u32 position wrap (sign lost in an
  `intptr_t` difference).
- Malformed messages were acknowledged as success. Nodes now validate every
  message against its opcode schema and NACK with `HS_ACK_INVALID`; input
  opcodes, which had no schema entry, are accepted under validation.
- IPC: a late result of a timed-out request could answer the next request
  (server-assigned correlation ids); transient `accept()` errors no longer end
  the service; no-address TCP binds loopback; the Unix socket is 0600 from
  creation; timed waits use `CLOCK_MONOTONIC`.
- `neogpu_tool` only trusts a server run by the same user or root, defaults to
  a per-user socket, and handles `--help`. CLI values are parsed strictly.
- Payloads allocated off the step thread raced the payload ring;
  `hs_gpu_send_with_payload` / `hs_gpu_set_global` now send inline.
- GLES backend: draws use the bank's vertex count, `OP_DRAW_INSTANCE` honours
  the instance count (with per-instance offsets from the instance bank),
  `OP_DRAW_TEXT` draws text instead of a full-screen checker, a clip rect past
  the bottom edge is trimmed rather than shifted, each backend owns its own
  context, and a failed shader compile no longer leaks.
- Text: glyph texture coordinates were flipped (glyphs rendered as fragments)
  and digits/entities loaded at the wrong codes (scores rendered blank).
- `m4_transpose` swapped rows 1 and 2; the portable NEON fallback did not
  compile and was never included.
- evdev input: `SYN_DROPPED` resync via `EVIOCGKEY`, lost key transitions
  resynced, framing events no longer use queue slots, and an unplugged device
  (with or without hotplug) releases held keys and frees its slot.
- Storage names are never truncated or aliased; storage and font loads reject
  short or oversized files; the font atlas size is bounds-checked.
- Demos: `neogpu_procgame` divided by zero at score 1100; `neogpu_pong_ai` drew
  nothing visible; capture demos replay what they read back; screenshots are
  read before presenting (and in a format GLES supports); outputs no longer use
  predictable `/tmp` paths; `neogpu_pong` uses raw terminal input with real
  arrow keys.
- `make check-sources` failed on `tests/panel_mode_probe.c`, so CI stopped
  before its sanitizer and race stages.
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

### Performance (reference benchmark against d6a59eb)
- The audit fixes cost measurable CPU, now recovered. Node schema checks use
  the inline `hs_message_conforms()` (same verdicts as
  `hs_validate_message()`, checked exhaustively in `test_p0`); plain draws use
  the original two-attribute shader, with a separate program only for
  per-instance offsets; the EGL swap interval is set only when vsync changes.
  Pre-audit vs now: GLES lifecycle submit time is lower in all seven scenes
  (e.g. alternate 399.6 -> 396.9 us, retained 223.4 -> 214.2 us); the
  executor's frame total is 465.5 -> 467.6 us, the difference being the
  page flip that now shows every frame; the legacy message path is +1.3% at
  1 draw per scene and -3% at 64 draws (per-message schema check).

### Added
- `make present-hardware` (part of `make test-hardware`): presentation and
  text-rendering checks against the real display controller.
- `make check-scalar`, `make bench-legacy`; CI runs both.
- `hs_backend_gles_capture_next()` for correct pre-present frame capture.

### Participant runtime and graphics cost work (44090bb..d6a59eb)
- Bounded participant fabric, integrated into the default `HSGpu` lifecycle
  with retained per-participant draw contributions.
- Pinned-transport and equivalent GLES comparisons; measured synchronization gap.
- Graphics resources released on teardown; GLES shutdown profiled and rejected
  shutdown candidates documented.
- Committed-path reference refresh and a real-GLES end-to-end harness.
- Graphics cost breakdown and the cost P0 programme; manual CRTC setup measured.
- Connector selection no longer assumes a DSI panel, and reuses the init scan.

### Remediated audit P0s and removed the ML experiment (4bd8221)

#### Fixed
- Buffer/storage index overflow; record-mask/stencil codec lengths; truncated
  fabric-result decoder overread; buffered capture-write error reporting.
- Capture payload ownership beyond the live transport ring, with lazy aligned
  storage and explicit `hs_destroy()` / `hs_gpu_shutdown()` teardown.
- Transport-to-actor inbox loss, reentrant consumer waits, and shared block-policy
  data races. Benchmarks now verify applied counts, not only admissions.
- IPC accepted-client cancellation, bounded I/O/admission, failed-start cleanup,
  Unix path preservation, and documented Unix/TCP server/client CLI modes.
- Async slot ownership, serialized worker startup, drain/reclamation on shutdown;
  unsupported texture decoding now rejects explicitly with `ENOTSUP`.
- Input queue wrap/wakeup, multi-producer serialization, synchronized held-state
  snapshots, and evdev hot-add/remove lifecycle.
- Missing depth/stencil framebuffer attachments. Graphics tests now capture
  evidence, the clear test terminates, and instancing uses an instanced draw.

#### Verification
- Runnable `make test`, `make sanitize`, `make tsan`, `make check-sources`,
  `make benchmark`, and opt-in `make test-hardware`; ARM CI workflow added.
- Unique in-tree fixtures, complete/correlated IPC reads, malformed peer tests,
  100/100 response assertions, and monotonic observable benchmark work.
- Full evidence and retained limitations: `docs/audit/P0_REMEDIATION.md`.

#### Removed
- BitNet/ML inference code, model sidecars, vendored reference implementation,
  ML-only tests, benchmarks, tools, and build targets. The experiment now lives
  in a separate repository.
- ML IPC handlers and public opcodes. Wire values 43–48 remain retired so
  existing input opcode values and captures do not shift.

#### Changed
- Project documentation and contribution templates now focus on the graphics
  runtime and tooling. Mixed historical notes remain for provenance.
- Headless IPC tests verify rejection of all six retired wire values and pin
  the existing input opcode ABI at compile time.

## [0.2.0] - 2026-11-11 (not tagged)

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
