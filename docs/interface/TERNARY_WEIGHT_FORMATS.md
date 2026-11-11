# Ternary Weight Formats

Byte-level reference for every ternary weight layout NeoGPU reads or
writes, the code mapping those bytes encode, and the two-stage ternary
lm_head format.

Everything here was read out of the source and, where a number is quoted
as measured, measured by executing the code it describes. The file and
function for each claim is given so it can be checked.

Related: [ML_API.md](ML_API.md) (the inference API),
[../research/TERNARY_KERNEL_NOTES.md](../research/TERNARY_KERNEL_NOTES.md)
(kernel internals), and the format comment at the top of
`src/hs_ml_ternary_neon.c`.

---

## 1. The code mapping

A ternary weight is stored as a 2-bit code. NeoGPU's canonical mapping is
**code = weight + 1**:

| code | bits | weight |
|------|------|--------|
| 0    | `00` | −1     |
| 1    | `01` | 0      |
| 2    | `10` | +1     |
| 3    | `11` | unused  |

This is the mapping used by the decode LUT in `src/hs_ml_ternary_neon.c`
(`build_lut`), by the 8-plane encoder in `src/hs_ml_loader_ternary.c`
(`hs_mlt_lmhead_encode`), by the `hs_ml_mt11.c` kernels, and by the
archived `hs_ml_ternary_f32.c`.

Note that the LUT's arithmetic is `(code) - 1`, so a code of 3 decodes to
+2 — outside the ternary range. No NeoGPU-produced data contains code 3,
and upstream's `quantize_i2_s` emits only 0, 1 and 2, so it does not
arise in practice.

### 1.1 One place uses a different mapping — and it is wrong

`f32_to_ternary_packed()` in `src/hs_ml_loader_ternary.c` packs weights
using the opposite convention: `00 = 0`, `01 = +1`, `10 = −1`. Its own
comment calls this "our internal 2-bit code", but it is not the internal
code; it is the only place in the repository that disagrees.

The two are not merely different names for the same thing — they assign
different weights to the same bits:

| code | `f32_to_ternary_packed` writes | the LUT kernel reads back |
|------|-------------------------------|---------------------------|
| 0    | 0                             | −1                        |
| 1    | +1                            | 0                         |
| 2    | −1                            | +1                        |

Measured, using both functions taken verbatim from the repository source:
of a test row `{+0.8, −0.8, +0.2, −0.2, +1.0, −1.0, 0.0, +0.6}`, **8 of 8
weights decode to the wrong value** — signs inverted, and every zero
becomes −1.

This affects only models whose tensors are stored as `GGML_TYPE_F32`,
which the loader packs on the fly. I2_S models bypass this function
entirely (see §2.3), and no test in the repository exercises the F32
path. The fix is to make the packer emit the canonical mapping, but that
is a change to loader behaviour rather than to documentation, so it is
tracked separately from this document.

---

## 2. Bit layouts

Three distinct byte layouts appear in this repository. They are **not**
interchangeable.

### 2.1 Side-by-side

| | NeoGPU internal | upstream BitNet, ARM | upstream BitNet, x86 |
|---|---|---|---|
| Where | `hs_ml_ternary_neon.c`, `hs_mlt_lmhead_encode` | `quantize_i2_s`, ARM branch | `quantize_i2_s`, x86 branch |
| Block size | none — one flat row | 64 weights | 128 weights |
| Bytes per block | — | 16 | 32 |
| Weights sharing a byte | 4 **consecutive** | 4, spaced 16 apart | 4, spaced 32 apart |
| Field order | **low bits first** | **high bits first** | **high bits first** |
| `bits[1:0]` | weight 4b+3 | weight 16b+3 | weight 32b+3 |
| `bits[3:2]` | weight 4b+2 | weight 16b+2 | weight 32b+2 |
| `bits[5:4]` | weight 4b+1 | weight 16b+1 | weight 32b+1 |
| `bits[7:6]` | weight 4b+0 | weight 16b+0 | weight 32b+0 |
| Code mapping | 0=−1, 1=0, 2=+1 | same | same |

Here `b` is the byte index within the block.

The NeoGPU column is the format the LUT kernel and the 8-plane encoder
produce and consume. Both upstream columns come from the same function
in the vendored reference implementation
(`experimental/BitNet/src/ggml-bitnet-mad.cpp`), but the function
compiles to **two different layouts** depending on the target:

```c
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
#define QK_I2_S 128
#elif defined(__ARM_NEON)
#define QK_I2_S 64
#endif
```

with the packing loop then using `group_idx = j / 32` (x86) or
`group_idx = j / 16` (ARM). Both branches write the most significant
field first. **A GGML file containing I2_S tensors carries a different
byte order depending on the machine that produced it.**

The ARM layout is the one this project targets, and it is the layout
already used by the repository's own reference decoders
(`tests/redteam_i2s_neon.c`, `tests/i2s_native_probe.c`).

### 2.2 They are measurably incompatible

Packing a row of random ternary weights with the upstream ARM rule and
then decoding it with the NeoGPU rule recovers **0.3333** of the
original weights — exactly the 1/3 you would expect from three-valued
codes by chance. The upstream x86 layout read with the NeoGPU rule gives
0.3437. Each layout read with its own rule gives 1.0000.

Over a 640-byte row the three byte streams differ in 620 (ARM vs x86),
635 (ARM vs NeoGPU) and 633 (x86 vs NeoGPU) bytes.

So the NeoGPU rule extracts **no information at all** from upstream-layout
bytes. The two must not be mixed.

`tests/redteam_ternary_formats.c` reproduces these figures, and asserts
that each layout is self-consistent while every cross-layout pairing
sits at chance.

### 2.3 What the loader actually does

`load_i2s_raw()` in `src/hs_ml_loader_ternary.c` reads the raw tensor
bytes with a single `fread` and returns them unmodified, with the
comment "No re-packing needed". Those bytes are then handed to
`hs_ml_ternary_f32_proj()`, which decodes them with the **NeoGPU** rule.

That is the open question, and it is described in §4.

---

## 3. The two-stage ternary lm_head

### 3.1 What it is

The lm_head is stored as **8 planes**, each plane holding every weight of
the matrix in 2-bit packed form, plus a per-row F16 scale.

This is *not* "8 bits holding four ternary digits". The two ideas are
different, and conflating them is a common misreading:

- A byte holds **four weights of one plane**. Eight planes of a matrix
  are eight separate byte arrays.
- Each plane is one **digit of a base-3 expansion** of the weight, with
  the digit's place value decreasing by a factor of 3 each plane.

A weight `w` in the normalised range [−1, +1] is written as

```
w ≈ t0 + t1/3 + t2/9 + t3/27 + t4/81 + t5/243 + t6/729 + t7/2187
```

with each `tk ∈ {−1, 0, +1}` — the trits. The plane weights are exactly
`lmh_plane_w[]` in `src/hs_ml_infer.c`:
`1, 1/3, 1/9, 1/27, 1/81, 1/243, 1/729, 1/2187`.

Eight trits give 3⁸ = 6561 distinct levels spanning ±1.499771, with a
finest step of 3⁻⁷ = 0.000457.

### 3.2 The encoder

`hs_mlt_lmhead_encode()` in `src/hs_ml_loader_ternary.c` does this, per
row:

1. Convert the F16 row to F32 and record the row's maximum magnitude as
   the F16 `row_scale`.
2. Normalise the row to [−1, +1] by that scale.
3. For each plane `k` = 0..7, for each weight: round the current
   residual to the nearest of {−1, 0, +1}, store that trit as a 2-bit
   code in the plane's bytes, and multiply the residual by 3 for the
   next plane.

Step 3's "multiply the residual by 3" is what makes the digits a base-3
expansion rather than eight independent quantisations. Because the
rounding threshold is at ±0.5, the residual after each plane is at most
1.5 in magnitude, so:

| after | worst-case error | bound |
|-------|------------------|-------|
| 4 planes (Stage 1) | 1.5 / 3⁴ | 0.018519 |
| 8 planes (Stage 1 + Stage 2) | 1.5 / 3⁸ | 0.000229 |

Measured by running the encoder loop taken verbatim from the repository
source over 20000 random rows of 2560 elements each, normalised to
[−1, +1], then decoding the stored bytes back and comparing:

| planes | max error | mean error | bound |
|--------|-----------|------------|-------|
| 8      | 0.000229  | 0.000114   | 0.000229 |
| 4      | 0.018519  | 0.009255   | 0.018519 |

The measured worst case matches the bound to within float32 rounding — the
excess is 1.4e-8, or 6e-5 relative. The max residual left after 8 planes
measured 1.500000, confirming the bound.

`tests/redteam_ternary_formats.c` reproduces these figures, and also
checks that the source still contains the four encoder lines they depend
on, so the numbers cannot go stale.

After encoding, the F16 lm_head is freed — the planes replace it — and
`use_trit_lmhead` is set.

### 3.3 The two stages

`session_logits_trit()` in `src/hs_ml_infer.c` splits the work by how
much of the answer each plane can still change:

**Stage 1** — planes 0..3, over **every** vocabulary row V. The weights
`1, 1/3, 1/9, 1/27` are applied inside the fused kernel
`hs_ml_lmhead_stage1()`, which does four LUT lookups and four fused
multiply-adds per byte position. The F16 `row_scale` is applied inline
at the end of each row. The result is the coarse logit for every
vocabulary entry; the top `LMH_CANDIDATES` (200) are then selected by
`partial_top_c()`.

**Stage 2** — planes 4..7, over **only those 200 rows**, using the
single-plane kernel `hs_ml_ternary_f32_proj()` with weights
`1/81, 1/243, 1/729, 1/2187`. The residual is added to the coarse
value, with `row_scale` applied again.

**Output contract** — `logits[v]` is the exact value for the 200
candidates and `−1e30` for every other vocabulary entry.

Stage 1's approximation is good to 0.0185 in the normalised row, and
Stage 2 can only change a candidate's logit by at most that much, which
is why the candidate set is chosen on the Stage 1 value and not refined.

The ternary lm_head path is **opt-in**: `hs_mlt_load_gguf()` loads the
lm_head as F16, and `hs_mlt_lmhead_encode()` must be called separately
to produce the planes. `use_trit_lmhead` gates the switch.

### 3.4 Memory traffic

At the BitNet b1.58 2B-4T shape (V = 128256, H = 2560):

| | bytes |
|---|---|
| Stage 1 (4 planes over all V rows) | 4 × V × H/4 = V × H = 328,335,360 (328.3 MB) |
| Stage 2 (4 planes over 200 rows) | 200 × H = 512,000 (0.51 MB) |
| F16 lm_head, for comparison | 2 × V × H = 656,670,720 (656.7 MB) |

So the ternary Stage 1 moves half the weight bytes of the F16 lm_head it
replaces, while doing four planes of arithmetic per weight.

---

## 4. Open question: the layout of `ggml-model-i2_s.gguf`

`hs_mlt_load_gguf()` reads I2_S tensors as raw bytes and feeds them to
the NeoGPU-layout decoder with no reordering. If the model's bytes are in
the upstream layout, every projection weight is wrong — and, per §2.2,
wrong in a way that looks like noise rather than an obvious failure.

The repository contains two reference decoders that disagree:

- `tests/redteam_i2s_neon.c` and `tests/i2s_native_probe.c` decode the
  **upstream ARM** layout (`group_idx = j / 16`, `group_pos = j % 16`,
  field at `6 - 2 * group_idx`).
- `tests/redteam_ternary_v3.c` decodes the **NeoGPU** layout
  (`(byte >> (2*s)) & 3` for weight `4b + s`).

What is not known is which one produced the model file. Two facts make
this harder than it looks:

1. Upstream's own converter (`experimental/BitNet/utils/convert-hf-to-gguf-bitnet.py`)
   emits **only** TL1, TL2, F16 and F32 tensors. It never emits I2_S.
   So the file was not produced by the vendored converter, and its
   producer is unidentified.
2. Even if it was produced by upstream code, which of the two upstream
   layouts it carries depends on whether the producing machine had AVX or
   NEON.

Resolving this needs the model file itself. The plan is to probe it: a
program that measures the agreement rate between the two bit-fields of
the same byte and the same field across adjacent bytes, since in the
upstream ARM layout the four codes of a byte are 16 weights apart while
in the NeoGPU layout they are consecutive. Running the real model under
each decode and keeping whichever produces coherent text settles it
outright.

---

## 5. Source map

| What | Where |
|------|-------|
| Decode LUT and the fused 4-plane kernel | `src/hs_ml_ternary_neon.c` |
| Single-plane projection kernel | `src/hs_ml_ternary_neon.c` (`hs_ml_ternary_f32_proj`) |
| 8-plane encoder, I2_S raw loader | `src/hs_ml_loader_ternary.c` |
| Two-stage lm_head driver, plane weights | `src/hs_ml_infer.c` (`session_logits_trit`, `lmh_plane_w`) |
| Plane buffers, `LMH_CANDIDATES`, `use_trit_lmhead` | `include/hs_ml_infer.h` |
| Upstream I2_S quantiser (both layouts) | `experimental/BitNet/src/ggml-bitnet-mad.cpp` |
| Upstream converter (no I2_S output) | `experimental/BitNet/utils/convert-hf-to-gguf-bitnet.py` |
| Upstream-layout reference decoders | `tests/redteam_i2s_neon.c`, `tests/i2s_native_probe.c` |
| NeoGPU-layout reference decoder | `tests/redteam_ternary_v3.c` |
