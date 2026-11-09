# Technical Debt Audit — 2026-11-11

This is the pre-remediation snapshot. For fixes, repeated measurements, sanitizer
results, and explicitly retained limitations, see [P0_REMEDIATION.md](P0_REMEDIATION.md).

## Scope and verification

Audited the post-ML-removal working tree at `~/Projects/neogpu` on the Pi 4.
The removal is still uncommitted. This audit does not change product source.

Success condition: identify actionable correctness and maintenance debt with
specific evidence, rather than infer health from build success. Preserve the
working tree, opcode ABI, graphics state, and existing user files. Uncertainty:
display behavior, live input devices, and adversarial concurrency have not been
tested. Those require separate hardware/stress verification.

Evidence:

- The preceding clean build covered the runtime, eight graphics tests, IPC
  client, and remaining demo targets successfully.
- Its headless suite reported **138 passed, 3 failed**: `set_record_mask_unpack`,
  `async_done_msg`, and `gpu_capture_write`. These also failed before ML removal.
- Isolated audit probes ran on the Pi for payload codecs, buffer indexing,
  queue wraparound, capture integrity/write errors, and idle-client shutdown.
- `src/hs_input_stream.c` and `tests/test_input_stream.c` pass syntax checking;
  this does not validate their runtime behavior.
- No display-driving tests were run. No sanitizer or race-detector campaign
  was run. No performance improvement is claimed.

## Priority findings

### P1 — Buffer/storage bounds arithmetic can wrap

**Locations:** `include/hs_buffer.h:67-147`, `include/hs_storage.h:66-77`.

Buffer accessors compute `u32 offset = index * 4`, then check `offset + size`.
Both operations can wrap. A probe wrote float index `0x40000000` into a 64-byte
buffer: element zero changed from **1 to 99**, rather than rejecting the index.
Other values can bypass bounds checks with genuinely out-of-range addresses.
Storage's `offset + 4` checks have the same addition-overflow pattern (inspection,
not an executed out-of-bounds storage probe).

**Action:** validate indices before multiplication; use subtraction-based bounds
checks and consistent null/disposed-buffer handling. Add boundary and sanitizer
tests for every accessor.

### P1 — Capture silently loses messages and corrupts older payloads

**Locations:** `src/hs_core.c:802-821,1056-1081`, `include/hs_core.h:24-26`.

The message log can hold 65,536 entries, but payload storage is a wrapping ring
of 4,096 slots. Captures copy the ring's current bytes, not the original bytes.
Serializing callers does not restore payloads already overwritten.

**Verified:** after 4,097 individually processed clear messages, capture returned
success with only **4,096** messages; the first clear's value was **4,096**, not
its original **0**. Even payload-free logs are truncated at the payload limit.

**Action:** establish owned payload lifetime for recorded messages; explicitly
report overflow instead of successful truncation. Test beyond the ring boundary
and mix recorded/unrecorded traffic.

### P1 — An idle IPC client blocks service and shutdown

**Locations:** `src/hs_ipc.c:87-115,313-340,395-403`.

The single server thread blocks reading a connected client's header/body without
an I/O deadline. Stop shuts down the listening socket, not the accepted socket,
then joins the blocked thread.

**Verified:** a private test client sent one byte and remained connected;
`hs_ipc_stop()` failed to return within a two-second watchdog. The same design
prevents other clients from being serviced while an idle connection is open.

**Action:** track/cancel accepted connections, bound read/write time, suppress
SIGPIPE locally, and add idle-client, partial-frame, and disconnect tests.
Also repair start/stop ownership: startup failure/timeout can leave an unjoined
thread, and `running` is not a sufficient indication of whether a join is owed.
These additional lifecycle observations are from code inspection.

### P1 — Valid message payloads are rejected by their own unpackers

**Locations:** `include/hs_msg.h:91-94,203-210`,
`src/hs_nodes.c:696-704`, `src/hs_core.c:759-760,797`.

The record-mask packer and validator specify four bytes; its unpacker requires
eight. The four-byte stencil/stencil-function unpacker also requires eight.
Both four-byte probes returned **false**. Thus `set_record_mask_unpack` reflects
a real command-path bug, not just a flaky assertion; valid stencil operations
have the same contradiction.

**Action:** align packer/validator/unpacker lengths, add table-driven round-trip
tests for every opcode, and stop acknowledging failed decoding as handled.

### P1 — Input streaming has broken producer/consumer contracts

**Locations:** `src/hs_input_stream.c:176-258,307-321,466-477`.

Code inspection shows several independent issues:

- Every device reader writes to the **same SPSC queue**; multiple devices make
  it multi-producer without corresponding synchronization.
- The consumer waits on `queue_not_empty`, but readers do not signal it after
  pushing. An empty-queue wait can therefore persist despite available events.
- Buffer swapping clears state without coordinating with the processing thread;
  the worker's pointer can become the active/readable buffer mid-update. Held
  input state is not consistently carried forward between snapshots.
- `hs_input_stream_process_events()` is a no-op; hotplug deletion is empty.

**Verified separately:** after a normal queue wrap with 100 queued events,
`hs_input_spsc_available()` returned **4,294,967,140**, not 100
(`src/hs_input_stream.c:62-65`).

**Action:** choose per-device SPSC lanes or a real MPSC queue; define snapshot
ownership, wakeup, held-state, and device-removal contracts. Add synthetic-event
tests before live-device validation. No live-device correctness claim is made.

### P1 — Async task ownership can be overwritten before completion/reclamation

**Locations:** `src/hs_async.c:42-46,78-87,143-237,240-275`.

Inspection: the worker decrements `pending_count` as soon as it removes a task,
but its slot remains the completion record until `hs_async_process()` reclaims
it. Enqueue admission counts only pending work and advances around the same
array. With enough submissions, in-flight or completed-but-uncollected records
can be replaced, losing completion metadata and allocation ownership.
Worker creation also uses a load/store sequence outside the queue mutex, not
a serialized lifecycle transition; concurrent first submissions can race.
Texture loading is a stub: it never decodes the path or populates `task.result`.

**Action:** separate pending/completed queues or use explicit per-slot ownership;
serialize startup, check thread creation, and define draining/cancellation on
shutdown. Verify with saturation and multi-producer tests. These risks have not
been reproduced with a stress test in this audit.

### P2 — Capture writes can claim success after a failed flush

**Location:** `src/hs_core.c:1105-1119`.

**Verified:** writing a one-message capture to `/dev/full` returned **true**.
Buffered `fwrite()` succeeds, then `fclose()` fails; its result is ignored.
The writer also does not check `count <= capacity` before reading its buffers.

**Action:** validate capture bounds, propagate flush/close errors, and define
atomic replacement behavior for files that must survive interrupted writes.

## Verification and maintenance debt

### P1 — Existing tests can pass on stale files or malformed IPC traffic

**Locations:** `src/main.c:977-1004,1459-1469,1726-1808`.

- Fixed `/tmp/neogpu_async_test.bin` and `/tmp/neogpu_capture.bin` were owned by
  **root**, with timestamps from November 7. The audit runs as `ztflynn`.
  Capture-write failure and async failure therefore have a concrete fixture
  collision explanation; reads can still pass against those older files.
  This is not evidence that the current operation wrote successfully.
- IPC tests read response headers/metadata but do not consume result payloads.
  The next header read consumes leftover payload bytes, desynchronizing framing.
  The logged benchmark counted only **42/100** requests and flood **12/30**, yet
  the flood assertion only requires `good > 0`. The bad-op check only verifies
  that a write succeeded, not the server's rejection response.

**Action:** unique in-tree fixtures, exact cleanup, fail-fast write checks,
complete response decoding/correlation checks, and success-count assertions.
Then re-establish a trustworthy baseline before optimizing the runtime.

### P2 — No automated test gate, incomplete build coverage

**Locations:** `Makefile:29-32,79-110`, `.github/`.

`make test` reports **"Nothing to be done for 'test'"**: the target is phony but
has no recipe. There is no `.github/workflows/` directory despite the build
badge. Input streaming and its test are absent from the normal build; numerous
GPU probes are also outside it. Warning-producing code is accepted, and
`run-test` does not declare the core object dependencies it links.

**Action:** explicit build/check/benchmark targets, an ARM build/headless CI job,
separate opt-in hardware tests, and meaningful compile coverage for supported
modules. Do not turn warnings into errors before triaging existing warnings.

### P2 — Documentation and performance numbers outrun the executable contract

**Locations:** `README.md:210-215`, `SECURITY.md:7-14`,
`docs/interface/MESSAGE_ABI.md:22-24,88-92`,
`src/main.c:1495-1590,1766-1777,1820-1827`.

README/security examples advertise `--ipc-server`, `--ipc-tcp`, and `--ipc-ms`,
but `main()` only dispatches `--tool`; these flags do not start a server.
The ABI document still describes 64-byte payload storage, while the runtime
uses 96 bytes, and documents four-byte instancing where the codec uses eight.
Math benchmark outputs include **0.00 ms** and implausibly huge rates: unused
results can be optimized away. Some IPC timing uses process CPU `clock()`, not
elapsed wall time; the framing defect above further invalidates those numbers.

**Action:** test documented commands; update contracts from executable schema;
use monotonic wall time, observable checksums, repeated samples, and explicit
distinctions between message processing throughput and displayed GPU frame rate.

## Recommended repayment order

1. Repair test isolation, IPC response parsing, and payload codecs; get a
   genuinely green headless baseline and wire it into `make test`/CI.
2. Fix bounds arithmetic and IPC shutdown; add negative-input regression tests.
3. Fix capture payload ownership and durable-write error reporting.
4. Rework async/input ownership with targeted concurrency tests.
5. Reconcile supported APIs, build coverage, documentation, and benchmarks.

Avoid a broad style refactor first. The highest-value debt is correctness and
the ability to detect regressions, not naming or file organization.
