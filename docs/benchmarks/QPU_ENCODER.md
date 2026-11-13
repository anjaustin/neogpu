# QPU encoder: NeoGPU's own path to the Pi 4 GPU, 2026-11-15

`src/hs_qpu.c` and `include/hs_qpu.h` encode V3D 4.2 QPU instructions (the
Pi 4's "VideoCore VI" GPU) directly:
- no Mesa, no libdrm, no allocation;
- table-driven;
- structured instructions, labels and branches.

With `tools/hwprobe_qpu`'s direct v3d submit path, NeoGPU can now write and run
its own GPU programs. See [HARDWARE_MAP_2026-11-15.md](HARDWARE_MAP_2026-11-15.md)
for what the hardware does.

```c
uint64_t code[256];
hs_qpu_prog p;
hs_qpu_prog_init(&p, code, 256);
hs_qpu_emit(&p, &(hs_qpu_alu){...});                     /* or the builders: */
hs_qpu_alu x = hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, HS_QPU_RF(1), HS_QPU_NOREG);
x = hs_qpu_m(x, HS_QPU_M_ADD, HS_QPU_RF(1), HS_QPU_RF(1), HS_QPU_RF(4));  /* dual issue */
x = hs_qpu_s(x, HS_QPU_SIG_THRSW, HS_QPU_NOREG);                          /* signal */
hs_qpu_emit(&p, &x);
int n = hs_qpu_prog_finish(&p);  /* instruction count, or a negative error */
```

## Measured hardware rules, enforced by default

Every `hs_qpu_prog` enforces the two rules whose violation hung the GPU in
HARDWARE_MAP. Turning them off (`hs_qpu_prog_rules(p, 0)`) exists only for the
evidence programs that reproduce the hangs.

| rule | error |
|---|---|
| every TMU load that `ldtmu` reads has a `thrsw` at or after the load request, with at least 2 instructions between that `thrsw` and the `ldtmu` | `HS_QPU_E_RULE_LOAD` |
| a program does not end with TMU writes in flight: `tmuwt`, or the measured `barrierid(syncb)` + `thrsw`, after the last store | `HS_QPU_E_RULE_END` |

Conservative additions. These were not measured to hang, and they are refused
until they are measured:

| refused | error |
|---|---|
| loads never read, or a `tmud` write with no address, at the end | `HS_QPU_E_RULE_END` |
| writes to TMU texture/config registers, which the rules do not model | `HS_QPU_E_RULE_UNCHECKED` |
| `thrsw` or another branch in a branch's 3 delay slots, or a branch without 3 instructions after it | `HS_QPU_E_BRANCH` |

Structural checks, always on:
- One instruction may not write a register twice (add, mul or signal
  destination). It may not issue more than one TMU write either, because the
  order between them would be undefined (`HS_QPU_E_WRITE`).
- An ldtmu and a tmua write in the same instruction are allowed: the
  reference summation kernel does this, and its results were checked on the
  GPU.
- `faddnf` with identical inputs is refused (`HS_QPU_E_ORDER`). Operand order
  is the only thing that tells faddnf from fadd, and identical inputs have
  only one order.
- A label binds once.
- Branch conditions, labels and flags are range-checked, not truncated.
- The register macros give `HS_QPU_BADREG` for out-of-range indexes, and every
  slot refuses it.

Limits of the rule check:
- It follows instructions in program order: a loop body is checked as
  written.
- It assumes one `ldtmu` per load, the default TMU configuration.
- Rule state is tracked even while the rules are off.
  `hs_qpu_prog_finish` reports any violation if the rules are on when it
  runs.

`tests/test_qpu_encode` checks that the rules reject both load forms that hung
the GPU (`ld_delay`, `ld_tmuau`) and a store left in flight, and that they
accept the good form and all 8 gather programs.

## Instruction layout (measured)

The layout was found by assembling single instructions with an independent
assembler and reading the fields back. Nothing was taken from documentation.

ALU instruction:

| bits | field |
|---|---|
| 63..58 | mul-ALU opcode |
| 57..53 | signal combination (index, table below) |
| 52..46 | conditions and flags; for a register-writing signal, its destination (bit 6 = magic/accumulator, bits 5..0 = address) |
| 45 / 44 | mul / add destination is magic or an accumulator |
| 43..38 / 37..32 | mul / add write address |
| 31..24 | add-ALU opcode |
| 23..21, 20..18 | mul input muxes b, a |
| 17..15, 14..12 | add input muxes b, a |
| 11..6 / 5..0 | raddr_a / raddr_b (register file; raddr_b also holds small immediates) |

Operand encodings:
- Input muxes 0–5 select accumulators r0–r5, 6 selects raddr_a, and 7 selects
  raddr_b.
- Write addresses for the magic registers:

  | address | register |
  |---|---|
  | 6 | null |
  | 11 | tmud |
  | 12 | tmua |
  | 13 | tmuau |
  | 16–18 | sync, syncu, syncb |
  | 32–46 | TMU config/texture registers |
  | 55 | r5rep |

  The full list is in `hs_qpu_magic_name`.

Signals: the 5-bit field indexes the combinations the hardware has. Each
"(index) signals" pair below is one combination. Index 23 (rotate) is not
supported. Indexes 26–30 are reserved.

| indexes | combinations |
|---|---|
| 0–7 | (0) none, (1) thrsw, (2) ldunif, (3) thrsw+ldunif, (4) ldtmu, (5) thrsw+ldtmu, (6) ldunif+ldtmu, (7) thrsw+ldunif+ldtmu |
| 8–15 | (8) ldvary, (9) thrsw+ldvary, (10) ldunif+ldvary, (11) thrsw+ldunif+ldvary, (12) ldunifrf, (13) thrsw+ldunifrf, (14) smimm+ldvary, (15) smimm |
| 16–22 | (16) ldtlb, (17) ldtlbu, (18) wrtmuc, (19) thrsw+wrtmuc, (20) ldvary+wrtmuc, (21) thrsw+ldvary+wrtmuc, (22) ucb |
| 24, 25, 31 | (24) ldunifa, (25) ldunifarf, (31) smimm+ldtmu |

Conditions and flags (bits 52..46):
- `ac`/`mc` are the add/mul write conditions: 1 = ifa, 2 = ifb, 3 = ifna,
  4 = ifnb.
- `af`/`mf` are the add/mul flag fields: 1–3 push z/n/c, 4–15 update
  (andz … norc).

| set | value |
|---|---|
| af only / mf only | `af` / `16 | mf` |
| ac (+ mf push) | `32 | (ac-1)<<2 | mf` |
| mc (+ af push) | `48 | (mc-1)<<2 | af` |
| mc + af update | `64 | (mc-1)<<4 | af` |
| ac + mc | `64 | (mc-1)<<4 | (ac-1)` |

Combinations with no encoding are rejected:
- ac with any mul flag update;
- af and mf together;
- a condition or flag on an instruction whose signal writes a register.

Small immediates: integers 0..15 are indexes 0..15, -16..-1 are 16..31, and
the floats 2^-8 .. 2^7 are 32..47. A small immediate uses raddr_b and sets the
smimm signal.

Float operations with two inputs (fadd, faddnf, fsub, fmin, fmax, fcmp, fmul)
carry both input modifiers in the opcode. For the modifier "none" that adds 5.
fadd/faddnf and fmin/fmax share opcodes, and the hardware tells them apart by
operand order. The encoder orders the muxes to match.

Branch: bit 57 set. The byte offset is split into bits 3..23 → 35..55 and
bits 24..31 → 24..31. The condition is in bits 32..34:

| value | condition |
|---|---|
| 0 | always |
| 2 | a0 |
| 3 | na0 |
| 4 | alla |
| 5 | anyna |
| 6 | anya |
| 7 | allna |

Other branch fields:
- bit 14: also set the uniform stream;
- bits 15..17: uniform stream relative (1) or absolute (0);
- bits 12..13: 1 = relative target.

The offset is measured from the branch address + 32 bytes, which covers the
branch's 3 delay slots.

## Verification

`make qpu-encode-check` runs two steps:
1. `tests/test_qpu_encode` writes NeoGPU's encodings.
2. `tools/qpu_oracle_check.py` compares them word for word with Idein's
   py-videocore6 (GPL-2). That assembler is used only as an oracle: it runs
   in its own venv, its code is not copied, and nothing from it enters NeoGPU.

| check | result |
|---|---|
| NeoGPU's programs (end, done, done_nowt, alu64, 8 gathers, 3 load forms) | **15 of 15 identical** |
| branches: 7 conditions × 4 lengths × 3 uniform-stream forms, backward loops and forward jumps | **168 of 168 identical** |
| instruction corpus (every op and operand kind, every magic destination, every small immediate, every legal condition/flag pairing, every signal combination of up to 3, accepted and rejected) | **815 identical**, 502 rejected by both, 0 rejected by NeoGPU only, 5 known divergences (below) |

The check fails on any of these:
- a word difference;
- an instruction NeoGPU accepts and the oracle rejects;
- an instruction NeoGPU alone rejects, unless it is a listed divergence;
- a corpus line the oracle cannot even parse.

The last one keeps a printer bug from passing as agreement.

**Known divergence: faddnf with identical inputs** (3 corpus lines). NeoGPU
refuses it. The oracle silently emits fadd.

**Known divergence: fmov of a small immediate.** For fmov, mul mux b holds
the float input modifier: 0 is abs, 1 is none, 2 is l and 3 is h (measured).
- The oracle encodes abs for every small immediate, even 1.0.
- It encodes none for the same immediates on fadd.
- NeoGPU encodes none.

On the GPU, `fmov` of -16 (a NaN bit pattern) gave 0x00000000 in both forms
(`fmov_imm_none`, `fmov_imm_abs`): fmov flushes that NaN to 0. The positive
immediates are unchanged by abs. So no observable difference was found.

**On hardware:** `tools/hwprobe_qpu build/qpu/neo 0 adg` runs NeoGPU-encoded
programs and reproduces the HARDWARE_MAP figures, with every result checked
and 0 GPU hangs. Output: `tests/evidence-measure-20261115/hwprobe/clean/qpu-neo-encoded.txt`
on the Pi.

| measure | result |
|---|---|
| ALU, 8 QPUs | 44.47 GFLOP/s, 4.000 cycles per instruction |
| empty job, p50 | 74.7 µs |
| random gather, 64 MiB table | 36.9 M/s |
| sequential gather | 580.8 M/s |

The 8-QPU, 4-in-flight sequential gather was slow again: 19.7 ms, 3 of 3
runs. The cause was found and fixed the next day; see the section below.

## NeoGPU's own GPU path: hs_v3d and the shipped kernels

These two modules complete a GPU path with nothing outside the tree: no Mesa,
no libdrm, no GLES, no external assembler.

`src/hs_v3d.c` and `include/hs_v3d.h` talk to the kernel v3d driver directly,
using kernel UAPI headers only:
- find the v3d render node;
- create, map, wait on and free buffer objects;
- load a program;
- dispatch a compute job.

The dispatch accepts only 1 or 8 batches. That is the measured
configuration: workgroup counts 1, workgroup size 16, 16 per supergroup,
CFG4 = batches − 1.

`src/hs_qpu_kernels.c` holds the programs NeoGPU ships, built with hs_qpu:
- `hs_qpu_kernel_gather`: 1 or 8 batches, with 1, 2, 4 or 8 loads in flight.
  An in-kernel guard ends it at once when passes = 0. `hs_qpu_gather_uniforms`
  fills its uniforms, and refuses element counts that are 0 or not a whole
  pass, as well as unaligned addresses.
- `hs_qpu_kernel_store_one`.

The oracle check compares these shipped builders word for word, so what ships
is what was verified.

`tests/test_v3d_compute` (in `make test-hardware`; needs the render group,
not root) runs the whole chain on the GPU: 25 checks, every output word
compared with the CPU, 0 GPU hangs.
- store_one: dispatched, completed, and its write is visible to the CPU.
- 16 gathers: random and sequential indices, 1 and 8 batches, 1–8 loads in
  flight, each over 1 M elements from a 4 MiB table. 0 words wrong in every
  one. Output is pre-filled with a sentinel that equals no table value
  (checked), so an unwritten word cannot pass.
- passes = 0 ends at once (0.14 ms) and writes nothing.
- Refused: element counts that are 0 or not a whole pass; misaligned
  addresses; every batch count except 1 and 8; a dispatch without BOs;
  misaligned or overflowing program loads.

`tests/test_v3d_runtime` (also in `make test-hardware`; 12 checks, clean
under ASan and UBSan) covers the runtime's error paths and resources:
- the display node and missing or non-DRM nodes are refused;
- 200 open/close cycles and 400 BO cycles leak no fd or mapping;
- double free and double close are harmless;
- sizes 0 and above 4 GiB are refused;
- a BO beyond the GPU's 4 GiB address space fails cleanly (-ENOSPC).

The test's timings are not a benchmark. Before the layout fix below, the slow
8-batch 4-in-flight sequential case showed up there too: 3.78 ms, against
2.72 ms for 2 in flight and 1.84 ms for 8. With the fix the figures are
2.02 / 3.01 / 1.90 ms.

## The k = 3..5 gather slowdown: root cause and fix, 2026-11-16

An investigating agent root-caused it with `tools/gather_anom.c`, which runs
kernel variants under a wide perfmon and checks every output word. The key
before/after was reproduced independently.

**Cause.**
- The original kernel interleaved the 8 batches: batch b at 64b, stride
  64 × 8. With sequential indices, each 256-byte memory line was therefore
  shared by 4 QPUs.
- The L2T fetches a shared line once only if those QPUs reach it within
  about 16–32 KiB of each other.
- At k = 3, 4 and 5 the QPUs drift out of step during the run. A CPU-side
  trace of `out` saw leads of up to ~700 × 512 B, growing with time.
- The line is then fetched and written back 2–3 times. The bus moves the
  same ~6.8 GB/s at k=4 and k=8, so the extra bytes are extra time.

**Evidence.**
- The AXI byte counters show the traffic directly. At k=4: 86 MiB read
  against the 32 MiB needed, and 43 MiB written against 16 MiB (2.7×). At
  k = 1, 2, 6, 7 and 8 the traffic is exactly 1×.
- The skew experiment makes the effect appear on demand: starting each QPU
  32 KiB further along gives exactly 4× traffic at k = 2 and k = 8 alike.
  16 KiB gives 1×.
- Address aliasing was ruled out: 9 offsets of out, idx and table, from
  64 B to 8 MiB, changed nothing.
- The slowdown builds up over a run: 1× up to 512 passes, 2.3× by 8192.

**Fix, now shipped** in `hs_qpu_kernel_gather`, with its oracle twin
updated: 8/8 gathers word-identical, 168/168 branches.
- In pass q, batch b's load i reads base + (q·B + b)·64k + 64i + 4·lane. Each
  batch owns a contiguous 64k-byte run per pass.
- At k ≥ 4 a batch owns whole 256-byte lines, so drift no longer matters.
- The uniform interface is unchanged: the kernel computes the per-pass
  skip 64k(B−1) itself.
- With 1 batch the address layout is the same as before.

Measured on the GPU with 8 batches, sequential indices, 64 MiB table, 4 M
elements. Median of 5, every word checked; old = variant 0, shipped =
variant 5, same harness:

| k | old | shipped | traffic, shipped |
|---|---|---|---|
| 1 | 18.05 ms | 17.95 ms | 1.0× |
| 2 | 10.63 ms | 11.57 ms | 1.0× |
| 4 | 19.44 ms (2.7×) | **7.87 ms** | 1.0× |
| 8 | 7.14 ms | 6.89 ms | 1.0× |

- Random indices: unchanged, 114.6–118.3 ms either way.
- **The best sequential rate rose from 581 to 609 M/s.**
- k = 2 is the open question. The harness measured it ~0.8–0.9 ms slower
  in two separate runs, but `hwprobe_qpu` measured 11.28 ms, inside the old
  layout's 11.14–11.96 ms run-to-run range. It is not settled whether that
  is real. Its traffic is 1×. k = 8 remains the fastest choice in both
  layouts.

**Also found.**
- A layout where each batch owns one large contiguous chunk is much worse
  (66 ms at k=1, 4× reads): with chunks 2 MiB apart, 8 QPUs × 3 streams
  conflict in the L2T's sets.
- The first job a process dispatches costs ~10–12 ms extra: the fmov
  experiment measured 12.1 ms, then 0.15 ms for the next job. Later newly
  loaded programs do not pay it; in test_v3d_compute each gather is a fresh
  program and takes 2–5 ms.
- The perfmon counter named "cache hit" counts memory transactions; the AXI
  byte counters (40, 41) are the reliable measure of traffic.
- This kernel's `(tidx >> 2) & 15` batch index is not 0..B−1 for 2 or 4
  batches, which is why hs_v3d accepts only the measured 1 and 8.
- "Whole 256-byte lines at k ≥ 4" assumes idx and out are 256-byte aligned.
  BO addresses are page-aligned; an offset into a BO is the caller's to
  align.

## Red team 2, 2026-11-16

A second independent review covered the runtime, the shipped kernels, the
tests and the probes. It reported 10 findings, several demonstrated. All
are fixed and verified on the GPU with 0 resets.

Correctness:
- **passes = 0** wrapped the loop counter: a ~1 s hang, with stores far past
  `out`. There is now an in-kernel guard, and its oracle twin is updated
  (8/8 identical).
- **Element counts that are not a whole pass** silently skipped the
  remainder. `hs_qpu_gather_uniforms` now refuses them.
- **hs_v3d accepted 2–7 batches,** contrary to its docs. It now accepts only
  1 or 8, and all 15 other counts from 0 to 16 are tested refused.

Tests that could pass while broken:
- out[] was zero-filled while `hash32(0) == 0`, so an unwritten word could
  pass. A sentinel fill is now used, checked against every table value.
- A failed BO creation crashed instead of failing.
- hwprobe_qpu checked only the last of 5 trials; every trial is now
  checked.

Measurement:
- `hwprobe_mailbox` printed 0 resets when dmesg was unreadable. It now says
  "unknown". Section E counts its resets per row, and a missing ack is
  reported as such.
- Section B wrote 1 ms after dispatch, inside the first-job start-up. It now
  writes after 30 ms; the result is unchanged (never seen, 8 runs).

Docs:
- The single "~220 µs" eviction latency is now 216–292 µs over 4 samples.
- The "~10 ms per newly loaded program" claim is corrected (first job per
  process).
- The alignment assumption is stated.

Small fixes:
- the CFG3 supergroup field is masked to 4 bits (same packed value,
  0xF010);
- 64-bit load bounds math;
- `HWPROBE_FLAGKB` is parsed without 32-bit wrap;
- gather_anom: trials ≥ 1, batches must be 1 or 8, table-fill bounds,
  passes = 0 refused.

## Red team, 2026-11-15

An independent review found 14 issues; each one was reproduced with a small
program. All are fixed, and each fix has a regression test in
`tests/test_qpu_encode.c` (34 self-checks, clean under ASan and UBSan).

Rule-check holes:
- Any `barrierid` cleared a pending store.
- A store whose data came through a signal was not seen as a store.
- Unread loads and a dangling `tmud` at the end passed.
- The result depended on add/mul slot order.
- Rules toggled mid-program lost their state.

Missing validation:
- Conflicting writes in one instruction.
- Branch arguments truncated (label to 16 bits, cond and flags to 8 bits).
- faddnf(x, x) encoded as fadd.
- A branch without delay slots, or a thrsw or branch inside them.
- Rebinding a label.
- Register macros wrapping out-of-range indexes.

Checker gaps:
- NeoGPU-only rejections were not counted as failures.
- Rejected signal combinations never reached the oracle.
- No forward branch was tested.
- The rule-1 wording.

The same pass over `tools/hwprobe_qpu.c` found and fixed:
- `load()` accepted empty, truncated or oversized program files.
- Repeated loads could exhaust the code buffer. Programs are now cached by
  name.
- `HWPROBE_FLAGKB` was not bounds-checked.
- One concurrent-reader buffer was never used, and none were freed.
- The GPU-completion thread was pinned to a core a CPU gather thread also
  used.

That last fix was measured. GPU time under concurrency was 142.7–143.6 ms
after the fix and 142.7–143.5 ms before, so the published figures were not
skewed.

## Reproducing

```sh
make test                      # includes the encoder self-checks
make qpu-progs                 # NeoGPU-encoded programs -> build/qpu/neo
make qpu-encode-check          # + oracle comparison (needs the py-videocore6 venv)
tools/hwprobe_qpu build/qpu/neo 0 adgG
```

## Red team 3 (second pass over the round-2 fixes), 2026-11-16

A fresh reviewer verified the round-2 fixes. Findings 1–3 and 5 are
confirmed fixed (the zero-pass guard, uniform validation, the batch
restriction, the BO failure path). It found five remaining defects; all are
now fixed:

| finding | fix |
|---|---|
| `hwprobe_mailbox` still reported 0 resets when dmesg was unreadable: `grep -c` prints 0 on empty input | `resets()` counts in C and requires dmesg's own exit status. Section A now reports resets too. (On this Pi `sudo -n dmesg` works: the counts read 205, so the published zero-reset figures stand.) |
| The sentinel change broke `gather_anom`'s drift trace, which still looked for non-zero words | It now looks for the sentinel (verified: 15 samples traced) |
| `HWPROBE_FLAGKB` could still wrap at 2^54, and non-numbers read as 0 | Bounded before shifting; non-numbers refused (verified) |
| test_v3d_runtime accepted any error for the beyond-address-space BO (the doc said -ENOSPC), and accepted -EACCES for card1, which would skip the name check | It now asserts -ENOSPC, measured on this 8 GB Pi, and -ENODEV for card1 (measured: -19). Mappings are checked over all 400 BO cycles |
| hwprobe_qpu's scopy zero-filled its destination (blind at y[0]), and memset/scopy checked ~1 word in 1000 against an "every result checked" header | Sentinel fill; every word is now checked |

Fixing the last item exposed the GPU idle-clock effect
(HARDWARE_MAP_2026-11-15.md, "The GPU clock drops when the GPU idles"). The
full checks took long enough for the clock to drop. Timed jobs now get a
1 ms warm-up, and the published figures re-measured unchanged.

## Red team 4 (final pass), 2026-11-16

Fixes 1–4 of round 3 were verified. Six further findings were fixed:

| finding | fix (verified on the GPU, 0 resets) |
|---|---|
| `tmuop_probe` still had the `grep -c` reset bug fixed in `hwprobe_mailbox` | The same counting in C with an exit-status check; "unknown" when unreadable |
| The mailbox rate included the ~10 ms first-job start-up, which was the whole of the "unexplained" 0.42–1.2 M/s spread | The rate is now over messages 2..N, a steady 1.20–1.22 M/s. Per-poll cost is a slope between two budgets (321–323 ns), not wall time divided by polls |
| `tmuop_probe` M had no GPU warm-up | A 1 ms warm-up job before each timed job |
| hwprobe_qpu sections c and G had no warm-up, and c never checked its sums (its header also claimed perfmon on every job) | Warm-up before each timed job, untimed; c fills and checks its own data ("sums ok"); the header is corrected |
| M bounds | MSGS ≤ 200000; BUDGET ≥ 2 (0 wrapped the GPU's counter); the uniform stream is sized for 4 + budget + msgs; "msg1" reads n/a when nothing arrived |
| "Every op consumes one uniform" was stated as measured | Now marked as documented (Mesa), not measured |

Minor:
- test_v3d_runtime now proves `card1` opens as a file before asserting the
  driver-name refusal.
- gather_anom refuses negative skew.
- Reset counts read the kernel ring buffer, which can rotate; this is
  noted in the tools.

Figures re-measured with the fixed tools:
- the mailbox table;
- the concurrency section c: 11.63 / 8.26 / 7.67 GB/s, sums ok;
- G: totals 42.2–42.4 M/s, equal to the CPU alone, so the conclusion holds;
  the GPU share under load is 31.0 M/s warm, against 29.3 earlier on a
  cold clock.

## Convergence check, 2026-11-16

A last reviewer, reading only the round-4 changes, verified them and found
one medium and three low issues. All are fixed and verified on the GPU with
0 resets:

| finding | fix |
|---|---|
| (medium) section c's sum check could pass on a correct sum left at the same offset by the memory section, even if a job never ran | The sum slots are cleared before every timed run (expected total 0xFF000000, never 0). Verified with `mc` in one process: "sums ok" |
| (low) a lost mailbox message would include its 50 ms timeout in the rate | The rate now ends at the last ack seen. Unchanged when nothing is lost: 1.20–1.21 M/s |
| (low) a huge skew wrapped gather_anom's 32-bit range check | Bounded in 64 bits ("skip: skew beyond the BO", verified) |
| (low) the Memory table's GPU+CPU row was not updated | It now gives the range over all 5 runs |

Findings per round fell from 14 (round 1) to 10, 5 and 6 (the last mostly
in newly committed code), then 1 medium in this pass. Every finding is
closed with a test or a re-measurement.
