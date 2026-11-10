# NeoGPU Capture Format (HSCapture)

This document describes the on-disk binary format used by `hs_capture_write_file()` / `hs_capture_read_file()`.

## File Layout

The file is:

1) `HSCaptureHeader`
2) `count` serialized `Message` structs
3) `count` serialized `Payload` structs

Where `count` is the number of recorded messages in the capture.

## Header

`HSCaptureHeader` fields (see implementation in `src/hs_core.c`):

- `magic[8]`: `"HSCAP1\0"`
- `endian`: `0x01020304`
- `version`: `1` (`HS_CAPTURE_VERSION` in `src/hs_core.c`; files with any other value are rejected)
- `msg_size`: must equal `sizeof(Message)` for this build
- `payload_size`: must equal `sizeof(Payload)` for this build
- `count`: number of messages/payload blocks stored

## Notes

- This format is intentionally strict: it rejects files if struct sizes do not match the current build.
- Captures store native `Payload` structs (96 data bytes, 128-byte aligned struct
  size on aarch64); payload-bearing messages reference capture-local indices.
- The system lazily allocates a recording journal at the first recorded payload:
  one slot per log entry, independent of the live 4,096-slot transport ring.
  Allocation failure or log exhaustion sets overflow; capture then fails
  explicitly, rather than returning a partial success. A caller may instead
  supply `sys->log_payloads` with `log_capacity` slots before any recording;
  leave `owns_log_payloads` false for caller-owned storage.
- `hs_clear()` retains the journal for reuse. `hs_gpu_shutdown()` (or
  `hs_destroy()` for raw systems) releases owned journal memory and synchronization
  resources. Stop/join all producers and services before teardown. Initialize
  before shutdown; do not reinitialize a live instance without tearing it down.
- `hs_capture_from_log()` accepts the system's full log prefix, not arbitrary
  message arrays. Callers must serialize recording/stepping with extraction.
- At most 65,536 entries can be represented by the 16-bit payload index. Capture
  writes validate capacity and report buffered flush/close errors. Writes are
  not atomic file replacements; callers requiring crash durability must use
  a temporary file, `fsync`, and rename at their application boundary.
- The file is native-layout/native-endian, not a cross-architecture interchange
  format. Header checks reject incompatible layouts.

## Capture Policy (Channel Filtering)

Captures are produced from a message log. Which messages enter that log is controlled by the system recording policy:

- `HSSystem.recording`: enables/disables logging.
- `HSSystem.record_mask`: bitmask of channels to record (`1u << HSChannel`).

Default policy (P0): record `CHAN_RENDER` only. This keeps captures deterministic and compact while excluding best-effort telemetry.

You can override the mask at runtime via `hs_set_record_mask()`.
