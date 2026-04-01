# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-04-01

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
