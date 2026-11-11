# Reflections: NeoGPU Messaging Layer

## Core Insight
If the message stream is going to be the "engine spine" (record/replay, tooling, multiple producers, backend portability), the ABI must be explicit and self-validating; otherwise performance wins today become correctness costs tomorrow.

## Resolved Tensions

### Speed vs Schema
Resolution: keep the fast queue and fixed-size `Message`, but add a typed schema layer around it (pack/unpack helpers + validator). This does not require changing the core loop; it only changes how messages are produced/consumed.

### Fixed payload blocks vs expressive commands
Resolution: preserve 64B payload blocks as the default, and introduce one explicit mechanism for "bigger than 64B" when needed. Prefer indirect handles for large data (meshes, textures, audio) and reserve chained/blob payloads for rare structured control messages.

## Challenged Assumptions

1) "Replay works because we replay immediately": true for the demo, but not true once there are more payload allocations or multiple threads.

2) "Immediate fields are harmless": they are fast, but without a schema they make decoding ambiguous for validators and tools.

## What I Now Understand

- The current system is already a good engine kernel; the biggest missing piece is not speed, it is clarity of contract.
- The smallest high-leverage change is an ABI/spec + helpers + validation; backend execution and feature work become safer afterward.
