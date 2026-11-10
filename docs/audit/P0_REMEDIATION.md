# P0 remediation ledger

> **Later status (2026-11-13):** the "every C source" gate below regressed after
> this ledger (`tests/panel_mode_probe.c`, added later, did not compile) and the
> published display-path FPS did not measure displayed frames. Both are fixed
> and re-verified in [AUDIT_2026-11-13.md](AUDIT_2026-11-13.md).

## Acceptance contract

- Preserve existing user work and the wire/input opcode ABI.
- Every audit finding needs a regression or explicitly recorded verification
  limitation; no numerical "pristine" score is inferred from build success.
- A clean build, real `make test`, repeated headless runs, negative-input
  regressions, concurrency stress, and available sanitizer checks must pass.
- Measure throughput with monotonic wall time and observable results. Compare
  repeated runs; report measured variability, not assumed improvement.
- Display and real evdev behavior require separate on-device verification.

## Before changes

Three consecutive Pi runs: **138 passed, 3 failed** each. IPC benchmark framing
was broken: **42/100** responses in each run, so those latency numbers are not
a valid baseline. Message throughput (msg/s), raw runs:

| Producers | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| 1 | 12010964 | 12688158 | 11900076 |
| 2 | 6868418 | 8651582 | 6579602 |
| 4 | 6889760 | 7114192 | 6859268 |
| 8 | 7248820 | 7224230 | 7198860 |
| 16 | 6958108 | 6832460 | 6781340 |

Logs: `tests/p0-baseline-{1,2,3}.log` on the Pi. No displayed frame-rate claim
is derived from message-processing benchmarks.

## Acceptance results — final source state

All locally runnable acceptance gates pass. This is a bounded verification
statement, not a claim of 100% code coverage or proof that no other bugs exist.
Assertion counts include repeated boundary/byte/lifecycle checks.

| Gate | Verified result | Pi evidence |
|---|---|---|
| Clean source preservation | All 61 source files unchanged byte-for-byte by `make clean` | `p0-source-manifest.log`, `p0-clean-verified.log` |
| Clean build | Runtime, all graphics tests, client, demos, P0/evdev tests built with `-Wall -Wextra -Werror` | `p0-strict-build.log` |
| Every C source | Syntax checks with warnings as errors pass | `p0-strict-sources.log` |
| Header dependency gate | Forcing `include/hs_storage.h` newer rebuilds P0 executable with explicit source/object inputs | `p0-dependencies.log` |
| Three consecutive `make test` runs | Each: 141 headless assertions, 6,827 P0 assertions, complete Unix/TCP CLI contracts; zero failures | `p0-final-suite-{1,2,3}.log` |
| Additional stress | 20/20 complete P0 executions pass | `p0-final-stress-{1..20}.log` |
| ASan + UBSan + leak detection | 6,827 P0 assertions, 141 headless assertions, malformed/normal standalone-client contracts pass; gate exit 0 | `p0-sanitize-release.log` (71.71 s wall time) |
| TSan | 6,827 P0 assertions, 156 headless/benchmark assertions, standalone-client contracts pass; gate exit 0 | `p0-tsan-release.log` (55.28 s wall time) |
| Privileged hardware gate | All eight graphics tests and 13 virtual-evdev checks pass | `p0-hardware-release.log` |
| Visual evidence | All eight framebuffer captures inspected; final triangle, cube, orbit captures inspected again after lifetime changes | `tests/p0-*.ppm` |

Evidence files above are under `tests/` unless otherwise qualified. Logs and
captures are generated artifacts, not committed source. The combined sanitizer
command initially exceeded the harness's 120-second deadline; each target was
then rerun separately with a 180-second deadline and a verified zero exit status.

## Remediation map

| Audit issue | Remediation and regression |
|---|---|
| Wrapped buffer/storage bounds | Validate ranges before arithmetic; test huge indices, all accessor widths, disposed/null buffers, storage edges; ASan/UBSan |
| Capture corruption/truncation | Independent journal with per-log-entry ownership; verify every payload in a 4,113-message capture and replay; overflow fails explicitly |
| IPC hangs/lifecycle | Accepted socket synchronized with stop; 250ms I/O deadlines; nonblocking queue admission; synchronous startup; repeated shutdown and invalid-start checks; no unlink of caller-owned paths |
| Codec mismatch | Packed-size and every shorter length tested across all decoders; record-mask and four-byte stencil decoding repaired |
| Input concurrency | Mutex-serialized admission and wakeup predicate; thread-local snapshots of protected persistent state; queue-wrap/pause/resume/four-producer tests; real virtual evdev hot-add/remove |
| Async ownership | Separate free/completed slots from pending indices; admission counts occupied slots; serialized worker startup; accepted saves drain; 32 concurrent saves and saturation/reclamation/flush-failure tests |
| Capture flush error | Capacity checked before opening; failed close/flush returns failure; `/dev/full` regression |
| False-passing tests | Unique in-tree fixtures; failed writes cannot pass through stale reads; full IPC frames/correlation/status checked; 100/100 required |
| Missing test gate/coverage | Runnable test/sanitizer/race/source/hardware targets; generated dependencies included; cleanup source hashes verified; ARM workflow added |
| Unsupported docs/benchmarks | Implemented documented server and TCP client modes; corrected ABI/storage contracts; observable math work and monotonic timing; dropped unsupported numeric claims |

### Additional blockers discovered during iteration

1. **Fabric decoder overread:** accepted 64-byte buffers while reading fields
   through byte 79. ASan reproduced a stack-buffer-overflow before the fix
   (`p0-codec-before.log`). Decoder now requires its packed 96-byte result; both
   exact-size and malformed-peer regression checks cover this.
2. **Block-policy data race:** the full TSan benchmark found concurrent writes
   and reads. Shared policy flags are now atomic; the full gate passes.
3. **Actor handoff loss:** admission-only rates hid severe message loss at the
   smaller node inbox. A measured pre-fix current run admitted **19,870,458** but
   applied **531,829** messages; an isolated baseline run admitted **24,901,285**
   and applied **24,532,335**. Full stock inboxes now apply on the sole step
   thread before handoff retries; reentrant consumer sends never wait on
   themselves. A 1,024-message burst and every producer benchmark require exact
   applied/admitted equality.
4. **False graphics passes:** framebuffer inspection exposed incorrect cube
   occlusion. A probe measured **0 depth / 0 stencil bits**. EGL now requests
   **24/8**, and the cube test checks the actual attachments. The instancing
   test previously issued ordinary per-object draws; it now uses one real
   instanced draw per frame.
5. **Recording footprint/lifetime:** the first fix embedded an 8,388,608-byte
   journal in every GPU instance. Replaced by lazy aligned storage with explicit
   teardown; 100 init/record/shutdown cycles are part of each P0 execution.

## Final measurements

### Applied message throughput

The final benchmark gates producer startup, measures the actual completion
interval (including stop/join/drain), and counts shader applications. Every
configuration in all three runs had **applied == admitted** and zero send
failures. These are not admission-only or displayed-frame rates.

| Producers | Run 1 msg/s | Run 2 msg/s | Run 3 msg/s | Median msg/s |
|---|---:|---:|---:|---:|
| 1 | 6003134 | 11826346 | 7161610 | 7161610 |
| 2 | 6646724 | 6607669 | 6576734 | 6607669 |
| 4 | 6597953 | 6569234 | 6522839 | 6569234 |
| 8 | 7485877 | 7382018 | 7276848 | 7382018 |
| 16 | 6754928 | 6754508 | 6743385 | 6754508 |

An additional maximum-configured-producer run applied **12,839,058 of 12,839,058**
admissions with 32 producers over **2003.122110 ms**: **6,409,523 applied msg/s**
(`p0-producers-32.log`). Zero send failures; this is one run, not a median.

Logs: `p0-final-benchmark-{1,2,3}.log`. CPUs were not pinned; single-producer
variability is reported, not attributed to an unverified cause. The original
admission-only baseline is not a valid speedup comparison because it did not
assert delivery. No optimization speedup is claimed for these correctness fixes.

An alternating same-harness A/B experiment and `perf` sampling were used while
investigating the apparent regression (`p0-ab-*`, `p0-perf-*-report.log`). Kernel
symbols were restricted; user-space samples were available. A trial moving
recording fields to struct tails did not improve the measurements and was
**rejected and reverted** (`p0-layout-*`). Application counting subsequently
revealed the handoff-loss defect above. No speculative layout optimization remains.

### CLI request batches

Each entry is the measured total time for **100/100** correlated responses,
including Python-client overhead, on the Pi. These are separate from the
in-process C latency test, not interchangeable measurements.

| Transport | Run 1 ms | Run 2 ms | Run 3 ms | Median ms |
|---|---:|---:|---:|---:|
| Unix | 42.010 | 38.758 | 39.605 | 39.605 |
| TCP/loopback | 49.328 | 48.781 | 47.293 | 48.781 |

Also tested: real standalone C clients over Unix/TCP, malformed headers and
lengths, retired opcodes, invalid payloads/arguments, resets, idle/partial
clients, failed binds, and a fake peer sending malformed/truncated responses.
Negative tests inspect sanitizer diagnostics rather than mistaking a sanitizer
exit for successful rejection.

### Graphics and memory

- Final orbit measurement: **1,092 frames / 10.004662 s = 109.149 FPS** as reported
  by the monotonic display-path counter. This run included one evidence readback.
- Actual default framebuffer: **24 depth bits, 8 stencil bits**.
- Actual `sizeof(HSGpu)`: **3,573,184 bytes**; `sizeof(Payload)`: **128 bytes**.
- Journal at the default maximum capacity: **8,388,608 bytes**, allocated only
  upon the first recorded payload (or supplied by the caller), freed on teardown.
- Strict-build `size neogpu_demo`: text **127,851**, data **2,096**, bss
  **80,143,688** bytes. The demo includes many static test fixtures, not just a
  minimal runtime. BSS size is not measured resident memory.
- `/usr/bin/time -v ./neogpu_demo --check`: **49,980 KiB maximum resident set**
  (`p0-memory-release.log`). This is that specific test run, not an application
  memory guarantee.

## Explicit limits and next-environment work

- No universal "100/100 quality score" or coverage percentage is inferred.
  The local gates above pass; GitHub CI is configured but cannot be claimed as
  remotely executed by these local checks. Consult GitHub Actions for publication
  and remote-run status.
- Input aggregates are last-event-wins; removal resets aggregate state. Generic
  absolute-axis calibration and physical-device-specific behavior are not
  implemented/claimed. Real evdev verification used only test-created virtual
  devices, not a person's active keyboard.
- Async texture decoding/shader compilation are not implemented capabilities;
  the texture API explicitly rejects with `ENOTSUP`. Supported async file saves
  have tested ownership and shutdown behavior.
- Capture files are strict native-layout/native-endian records with at most
  65,536 entries, not a portable cross-architecture interchange format. Write
  failure reporting is tested; atomic replacement/fsync remains the application's
  durability responsibility.
- Init/shutdown and service lifecycle operations require external serialization;
  stop/join users before destroying their storage. Snapshot lifetime is explicitly
  documented, not promised indefinitely.
- IPC remains trusted-local tooling, not an authenticated/TLS network service.
- Verification was completed before publication. Git history and GitHub Actions
  are the authoritative records of subsequent commits, pushes, and remote runs.
