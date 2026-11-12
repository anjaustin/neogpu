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
| every TMU load that `ldtmu` reads has a `thrsw` at or after the load request and at least 2 instructions before the `ldtmu` | `HS_QPU_E_RULE_LOAD` |
| a program does not end with TMU writes in flight (`tmuwt` or a `barrierid` first) | `HS_QPU_E_RULE_END` |

Limits of the rule check:
- It follows instructions in program order: a loop body is checked as
  written.
- It assumes one `ldtmu` per load, the default TMU configuration.
- It tracks only TMUA/TMUAU general memory loads and stores.

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
| branch loops: 7 conditions × 4 lengths × 3 uniform-stream forms | **84 of 84 identical** |
| instruction corpus (every op and operand kind, every magic destination, every small immediate, every legal condition/flag pairing, every signal combination of up to 3) | **818 identical**, 293 rejected by both, 0 rejected by NeoGPU only, 2 known divergences |

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

The 8-QPU, 4-in-flight sequential gather was slow again: 19.7 ms, now 3 of 3
runs.

## Reproducing

```sh
make test                      # includes the encoder self-checks
make qpu-progs                 # NeoGPU-encoded programs -> build/qpu/neo
make qpu-encode-check          # + oracle comparison (needs the py-videocore6 venv)
tools/hwprobe_qpu build/qpu/neo 0 adgG
```
