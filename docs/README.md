# NeoGPU Documentation

Reference material for NeoGPU. Start with the [README](../README.md) for the
project overview, quick start, and architecture summary.

## Design & Architecture

Specifications and design documents for the core runtime.

| Doc | What it covers |
|-----|----------------|
| [architecture/PRD.md](architecture/PRD.md) | Product requirements: the message-native runtime vision |
| [architecture/PRD_COMM_FABRIC.md](architecture/PRD_COMM_FABRIC.md) | Channel fabric: acks/results, structured errors, capture I/O, backpressure |
| [architecture/P0_PRD.md](architecture/P0_PRD.md) | Thread-safety and atomics plan for the messaging layer |
| [architecture/CHANNEL_FABRIC.md](architecture/CHANNEL_FABRIC.md) | RT / RENDER / TELEM channelization and QoS semantics |
| [architecture/SEMANTIC_MESSAGE_CLASSES.md](architecture/SEMANTIC_MESSAGE_CLASSES.md) | Semantic classification of message types |
| [architecture/IPC_CONTROL.md](architecture/IPC_CONTROL.md) | IPC server control surface |

## Interfaces & Formats

Wire-level contracts. These describe what actually travels between components.

| Doc | What it covers |
|-----|----------------|
| [interface/MESSAGE_ABI.md](interface/MESSAGE_ABI.md) | Per-opcode schema and payload ranges |
| [interface/CAPTURE_FORMAT.md](interface/CAPTURE_FORMAT.md) | `HSCAP1` capture file format |
| [interface/INPUT_STREAM.md](interface/INPUT_STREAM.md) | Input streaming system: lock-free SPSC queue, evdev, hotplug |
| [interface/ML_API.md](interface/ML_API.md) | Experimental ML inference API |

## Research

Conceptual work behind the implementation.

| Doc | What it covers |
|-----|----------------|
| [research/GEOMETRIC_INFERENCE.md](research/GEOMETRIC_INFERENCE.md) | Neural networks as coordinate transforms |
| [research/TERNARY_KERNEL_NOTES.md](research/TERNARY_KERNEL_NOTES.md) | Nibble-LUT ternary GEMM kernel notes |

## Benchmarks

Measured results. Every number here came from a specific run on specific
hardware; see each document for its methodology.

| Doc | What it covers |
|-----|----------------|
| [benchmarks/ATOMICS_PROFILE_2026-03-13.md](benchmarks/ATOMICS_PROFILE_2026-03-13.md) | Message-layer throughput profiling and red-teamed optimization log |
| [benchmarks/GPU_BENCHMARK_RESULTS.md](benchmarks/GPU_BENCHMARK_RESULTS.md) | GPU-side GEMM benchmarks on the Pi 4 V3D |
| [benchmarks/BITNET_FFN_ATOMICS.md](benchmarks/BITNET_FFN_ATOMICS.md) | BitNet feed-forward atomics analysis |

## Lineage

Relationship to the original PicoGPU project, and the history of the port.

| Doc | What it covers |
|-----|----------------|
| [lineage/PICOGPU_PORT.md](lineage/PICOGPU_PORT.md) | Port notes: what was ported from PicoGPU and how |
| [lineage/FEATURE_COMPARISON.md](lineage/FEATURE_COMPARISON.md) | Feature-by-feature comparison against PicoGPU |
| [lineage/FEATURE_MATRIX.md](lineage/FEATURE_MATRIX.md) | Per-function port status matrix |
| [lineage/HARDWARE.md](lineage/HARDWARE.md) | Target hardware: Raspberry Pi 4, VideoCore, DRM/GBM setup |
| [lineage/PORT_STATUS.md](lineage/PORT_STATUS.md) | Historical port tracking |

See [../CREDITS.md](../CREDITS.md) for the full acknowledgment.

## Audit

| Doc | What it covers |
|-----|----------------|
| [audit/DIAMOND.md](audit/DIAMOND.md) | Full self-audit: what is novel, useful, understated, and what was overstated |

## Archive

Historical material, kept for provenance rather than as current guidance.
Nothing here reflects the current state of the code — see the
[changelog](../CHANGELOG.md) for that.

| Directory | Contents |
|-----------|----------|
| [archive/lmm/](archive/lmm/) | LMM-style design explorations (raw / reflect / synthesize / nodes) |
| [archive/lmm-theory/](archive/lmm-theory/) | LMM explorations of the geometric-inference theory |
| [archive/journal/](archive/journal/) | Working notes and reflections kept during development |
| [archive/REMEDIATION_PLAN.md](archive/REMEDIATION_PLAN.md) | Remediation plan (superseded) |
| [archive/REMEDIATION_PLAN_2026-03-14.md](archive/REMEDIATION_PLAN_2026-03-14.md) | Dated remediation plan (superseded) |
| [archive/REAL_2B4T_REMEDIATION_PLAN.md](archive/REAL_2B4T_REMEDIATION_PLAN.md) | Real-model remediation plan (superseded) |
| [archive/INPUT_STREAM_REMEDIATION.md](archive/INPUT_STREAM_REMEDIATION.md) | Input stream remediation analysis (superseded) |

## Experimental Code

The vendored [../experimental/BitNet/](../experimental/BitNet/) tree is
Microsoft's `bitnet.cpp` reference implementation (MIT, unmodified). It is
present for comparison against NeoGPU's own BitNet kernels and is not part of
any NeoGPU build target.
