# Participant slice: implementation and evidence

## What changed

`include/hs_fabric.h` and `src/hs_fabric.c` add an opt-in frame-contribution
protocol above the existing GLES executor. They do not silently change the old
message queues, handlers, message ABI, or their historical counterexamples.
The new object is linked only into participant tests/applications that use it.

Its contract is [PARTICIPANT_SLICE.md](../architecture/PARTICIPANT_SLICE.md):
explicit generation-checked handles, per-participant FIFO queues, overload
rejection, rotating service, drain-before-retirement, and consumer-owned retained
complete draw contributions. Only renderer-owned immutable vertex banks and
draw commands are supported. This is an initial working slice, not a general
graphics engine, lock-free queue, process-isolation boundary, or stable ABI.

## Functional evidence on the Pi

The headless test checks:

- Four queued frames fill a participant's queue. The next frame is explicitly
  rejected, then accepted with the **same sequence** after one application.
  All accepted sequences apply in FIFO order, even while closing.
- Mutating the caller's frame immediately after submission cannot mutate the
  admitted frame. Private drafts have no effect until submitted.
- Budget-one steps service all eight uncontended backlogged participants each
  round, rather than repeatedly servicing the first lane.
- Holding one slot mutex busy does not prevent **100** updates to a peer, or
  composition from the busy participant's last complete view. This is a
  controlled critical-section fixture, not a process-crash experiment.
- Maximum-size contributions compose to **516 commands** without overflow.
  Global state/clear commands, cross-bank draws, payload references, invalid
  sizes/sequences and stale generations are rejected.
- **96** sequential join/submit/leave cycles reuse a slot with advancing
  generations; exhausted sequence/generation counters cannot silently wrap.
- Eight actual producer threads admit **40,000** frames. Consumer-side checks
  verify every intermediate per-participant sequence, count, bank, and draw
  opcode, not merely admissions or final totals.
- Eight concurrent lifecycle threads complete **512** join→submit→leave cycles:
  **512 accepted, 512 applied, 512 retired**. Accepted work is drained before
  retirement makes a slot reusable.

`sizeof(HSFabric)` on the Pi is **83,416 bytes**, including bounded queues,
retained views and mutexes. No per-submission allocation is performed.

The final recorded three-run collection used the unpinned OS scheduler and
default optimized Makefile flags, with one consumer and eight producers:

| Run | Accepted / applied | Completion interval | Busy/full admission retries |
|---|---:|---:|---:|
| 1 | 40,000 / 40,000 | 47.690259 ms | 16,112 |
| 2 | 40,000 / 40,000 | 34.955945 ms | 17,287 |
| 3 | 40,000 / 40,000 | 34.897667 ms | 16,921 |

These intervals include correctness observation and joins, and frame sizes cycle
through **0–64 draws**. They are not pure queue benchmarks, stock shader-message
rates, displayed FPS, or evidence of an optimization speedup. Retries are
explicit admission rejections before acceptance, not lost accepted work.
The external optimized-queue and equivalent-graphics comparisons remain pending.

Raw records and source/environment provenance for this final collection:
`tests/evidence-fabric-final/`. The initial collection is also retained under
`tests/evidence-fabric-initial/`, not substituted for the final source. Twenty
additional complete executions passed after the final collection
(`tests/evidence-fabric-final-stress-*.jsonl`).

Final manifest SHA-256:
`ac42e20a024cd04fa11235d43bc9050724c0a676d9ed927dedf1fcf080618f1b`.
The same Pi/compiler/kernel from the initial architectural evidence were used.
CPU allowance was 0–3 with `ondemand` governors. Overall boundary frequency
snapshots were 1,600,000 and 1,500,000 kHz; temperature snapshots were 37.972 and
39.920 C. These are not continuous observations or fixed-frequency controls.
Throttle status remains unavailable (`/dev/vcio` missing). Pi UTC is retained
literally in the manifest despite its disagreement with the harness date;
elapsed intervals use a monotonic clock.

## Actual GLES/presentation evidence

`tests/test_fabric_graphics.c` runs two producer threads and uses the existing
GLES backend. A **test-only linker wrapper** inspects the framebuffer immediately
before the backend's real `eglSwapBuffers`; it neither replaces backend rendering
nor reads a possibly stale post-swap buffer. The initialization swap is explicitly
excluded from the measured test presentations.

| Stage | Producer action | Verified screen |
|---|---|---|
| 1 | Both submit complete contributions | Red A + green B |
| 2 | A edits a private empty draft; B submits empty | Red A retained; B absent |
| 3 | A remains without another commit; B submits visible | Red A + green B |
| 4 | A leaves; its accepted work drains and it retires | Only green B |
| 5 | A rejoins; old handle is rejected; renderer replaces A's retired bank | Only green B; new A has not committed |
| 6 | New generation of A commits | Blue A + green B |

Each stage presents three times. **36 RGB sample checks across 18 presentations**
pass, with **5 complete contribution applications and 1 retirement**. Six
pre-presentation 800×480 captures are retained as
`tests/evidence-fabric-stage-{1..6}.ppm`; all six have been visually inspected
in a row-major contact sheet, `tests/evidence-fabric-contact.png`, locally.
The RGB tests inspect two interior pixels per presentation, not every output
pixel; captures provide the broader visual evidence.

Three final graphics repetitions after a strict rebuild produced identical
stage records and passed all 36 pixel checks each. The final captures are
byte-identical to those already inspected, not merely similarly named files.

The first graphics execution failed its final counter assertion because it
counted the backend's initialization swap. The wrapper was corrected to enable
verification only after backend initialization, and the test was rerun to a
verified zero exit status. The failure was test accounting, not discarded pixels.

## Final gates

| Gate | Verified outcome | In-tree evidence |
|---|---|---|
| New protocol ASan/UBSan/leaks and TSan | Both instrumented executions pass all five test groups, including 40,000-frame delivery and 512 concurrent lifecycles | `tests/evidence-fabric-sanitize-final.log` |
| Legacy ASan/UBSan/leaks | 141 headless + 6,827 P0 assertions and CLI contracts pass | `tests/evidence-fabric-legacy-asan.log` |
| Legacy TSan | 156 headless/benchmark + 6,827 P0 assertions and CLI contracts pass | `tests/evidence-fabric-legacy-tsan.log` |
| Strict build | Runtime, demos/client, old graphics/P0/evdev tests and both participant tests build with `-Werror` | `tests/evidence-fabric-strict-build.log` |
| Source cleanup safety | `make clean` preserves all 69 source files byte-for-byte | `tests/evidence-fabric-source-integrity.json`, `tests/evidence-fabric-clean.log` |
| All-source warning gate | Passes, including new C sources | `tests/evidence-fabric-sources-final.log` |
| Three final full test suites | Each: 141 headless + 6,827 P0 assertions, complete Unix/TCP CLI contracts and five participant groups pass | `tests/evidence-fabric-acceptance-{1..3}.log` |
| Additional participant stress | 20/20 complete executions pass | `tests/evidence-fabric-final-stress-*.jsonl` |
| Existing hardware gate | Eight graphics tests and 13 virtual-evdev checks pass; actual depth/stencil remains 24/8 | `tests/evidence-fabric-legacy-hardware.log` |
| Participant graphics, repeated | 3/3 executions, 36 pixel checks and 18 presentations each | `tests/evidence-fabric-graphics-{1..3}.jsonl` |

Instrumented times are not used in the completion table. Legacy guarantees are
not retrospectively changed: their counterexample probes are retained and still
report their original findings (`tests/evidence-fabric-legacy-probes.jsonl`).

## Reproduce

```sh
make test
python3 tests/test_fabric_evidence.py --output tests/evidence-fabric-NEW
make fabric-sanitize
make tests/test_fabric_graphics
sudo ./tests/test_fabric_graphics
```

Evidence output directories are exclusive; use a fresh name rather than
overwriting prior collections. Generated logs/captures are retained in-tree but
ignored by git. CI is deferred; no commits or pushes are part of this work.

## Interpretation and remaining work

This demonstrates a restricted but real transition from independent message
producers to composable retained graphics contributions on the Pi. Application
is **not** presentation acknowledgement, and independently committed contributions
do not define a globally synchronized simulation epoch. Full arbitrary rendering
state, resource-version ownership/uploads, IPC participant recovery, hard latency
limits and competitive performance are not established.

The next evidence boundary is matched optimized transport and graphics-cost
baselines, followed by extending the resource contract without losing the
invariants already tested. A candidate external SPSC reference was resolved to
`rigtorp/SPSCQueue` revision `c8e9008dc4639220cc9bcf4c84abd2ff0d344aa5`;
it has **not** yet been integrated or benchmarked. No winner claim follows.
