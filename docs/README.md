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

## Benchmarks

Measured results. Every number here came from a specific run on specific
hardware; see each document for its methodology.

| Doc | What it covers |
|-----|----------------|
| [benchmarks/ATOMICS_PROFILE_2026-03-13.md](benchmarks/ATOMICS_PROFILE_2026-03-13.md) | Message-layer throughput profiling and red-teamed optimization log |

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
| [audit/TECHNICAL_DEBT_2026-11-11.md](audit/TECHNICAL_DEBT_2026-11-11.md) | Pre-remediation technical debt findings |
| [audit/P0_REMEDIATION.md](audit/P0_REMEDIATION.md) | P0 fixes, acceptance evidence, measurements, and limitations |
| [audit/ARCHITECTURE_EVIDENCE_PLAN.md](audit/ARCHITECTURE_EVIDENCE_PLAN.md) | Falsifiable architectural hypotheses and comparison controls |
| [audit/ARCHITECTURE_EVIDENCE_RESULTS.md](audit/ARCHITECTURE_EVIDENCE_RESULTS.md) | Pi ordering/isolation/frame counterexamples and RT latency measurements |
| [architecture/PARTICIPANT_SLICE.md](architecture/PARTICIPANT_SLICE.md) | Bounded participant FIFO/lifetime/frame-commit contract and explicit limits |
| [audit/PARTICIPANT_SLICE_RESULTS.md](audit/PARTICIPANT_SLICE_RESULTS.md) | Concurrent delivery/lifecycle, sanitizer and real-pixel evidence for the participant slice |
| [audit/COMPARISON_PLAN.md](audit/COMPARISON_PLAN.md) | Pinned optimized-queue and equivalent GLES comparison controls |
| [audit/COMPARISON_RESULTS.md](audit/COMPARISON_RESULTS.md) | Same-Pi throughput/latency and byte-identical direct-GLES measurements |

## Archive

Historical material, kept for provenance rather than as current guidance.
Nothing here reflects the current state of the code — see the
[changelog](../CHANGELOG.md) for that.

| Directory | Contents |
|-----------|----------|
| [archive/lmm/](archive/lmm/) | LMM-style design explorations (raw / reflect / synthesize / nodes) |
| [archive/journal/](archive/journal/) | Working notes and reflections kept during development |
| [archive/REMEDIATION_PLAN.md](archive/REMEDIATION_PLAN.md) | Remediation plan (superseded) |
| [archive/REMEDIATION_PLAN_2026-03-14.md](archive/REMEDIATION_PLAN_2026-03-14.md) | Dated remediation plan (superseded) |
| [archive/INPUT_STREAM_REMEDIATION.md](archive/INPUT_STREAM_REMEDIATION.md) | Input stream remediation analysis (superseded) |
