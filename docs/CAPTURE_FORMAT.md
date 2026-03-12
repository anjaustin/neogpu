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
- `version`: `2`
- `msg_size`: must equal `sizeof(Message)` for this build
- `payload_size`: must equal `sizeof(Payload)` for this build
- `count`: number of messages/payload blocks stored

## Notes

- This format is intentionally strict: it rejects files if struct sizes do not match the current build.
- Captures store payload blocks (64B) as-is; payload-bearing messages should reference capture-local payload indices.

## Capture Policy (Channel Filtering)

Captures are produced from a message log. Which messages enter that log is controlled by the system recording policy:

- `HSSystem.recording`: enables/disables logging.
- `HSSystem.record_mask`: bitmask of channels to record (`1u << HSChannel`).

Default policy (P0): record `CHAN_RENDER` only. This keeps captures deterministic and compact while excluding best-effort telemetry.

You can override the mask at runtime via `hs_set_record_mask()`.
