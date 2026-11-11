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

There are **two** 2-bit code mappings in this repository. They are not
interchangeable, but each producer's bytes are read by exactly one
decoder, and the two agree within that pairing. Getting this wrong in
either direction silently produces a model made of noise.

### 1.1 The canonical mapping: code = weight + 1

| code | bits | weight |
|------|------|--------|
| 0    | `00` | −1     |
| 1    | `01` | 0      |
| 2    | `10` | +1     |
| 3    | `11` | unused  |

Used by the decode LUT in `src/hs_ml_ternary_neon.c` (`build_lut`), by
the 8-plane encoder in `src/hs_ml_loader_ternary.c`
(`hs_mlt_lmhead_encode`), by the `hs_ml_mt11.c` kernels, by the GPU GEMM
shaders in `src/hs_ml_gpu_gemm.c`, by the coprocessor shaders in
`src/hs_ml_ternary_coproc.c`, by the archived `hs_ml_ternary_f32.c`, by
this repository's own converter `tools/convert_bitnet_to_gguf.py`, and by
upstream BitNet's `quantize_i2_s`.

Note that the LUT's arithmetic is `(code) - 1`, so a code of 3 decodes to
+2 — outside the ternary range. No NeoGPU-produced data contains code 3,
and upstream's `quantize_i2_s` emits only 0, 1 and 2, so it does not
arise in practice.

### 1.2 The routing mapping: code = 0, +1, −1

| code | bits | weight |
|------|------|--------|
| 0    | `00` | 0      |
| 1    | `01` | +1     |
| 2    | `10` | −1     |

Used by the routing stack's GEMM, `hs_ml_gemm_ternary_mt()` in
`src/hs_ml_ternary_mt.c`:

```c
if (bits == 1) sum += a;
else if (bits == 2) sum -= a;
```

and therefore by the two packers that feed it:

- `f32_to_ternary_packed()` in `src/hs_ml_loader_ternary.c`, for models
  whose tensors are stored as `GGML_TYPE_F32`;
- `alloc_ternary()` in `src/hs_ml_infer.c`, for the random synthetic
  models.

Those two packers are **not** wrong. They pack for the routing decoder,
and they are never read by the LUT. The comment on
`f32_to_ternary_packed()` that calls its convention "our internal 2-bit
code" is misleading — it is the routing stack's code, not the internal
one — but the bytes are right.

### 1.3 Which decoder a weight goes to

The ternary inference stack has two projection paths, selected by
`m->use_i2s`:

| `m->use_i2s` | weights come from | decoder | code mapping | layout of the bytes | layout the decoder assumes |
|---|---|---|---|---|---|
| true — the model has any I2_S tensor | `load_i2s_raw()`, raw upstream bytes | `hs_ml_ternary_f32_proj()` (the LUT) | canonical | upstream's (§2) | NeoGPU — **a mismatch unless the file is NeoGPU-layout** |
| false — an F32 model, or a random one | `f32_to_ternary_packed()` or `alloc_ternary()` | `hs_ml_gemm_ternary_mt()` via `hs_ml_route_mt()` | routing | NeoGPU | NeoGPU — matches |

`m->use_i2s` is set only when a projection tensor is actually stored as
`GGML_TYPE_I2_S`, so a weight packed by `f32_to_ternary_packed()` can
only ever reach the routing decoder.

### 1.4 The I2_S path has no upstream-layout decoder

`src/hs_ml_infer.c` contains `i2s_proj_neon()`, which implements the
**upstream ARM layout** (64-weight blocks, `group_idx = j / 16`,
`group_pos = j % 16`, field at `6 - 2 * group_idx`). It is unreachable.
Its only call site is inside `ternary_proj()`'s `if (m->use_i2s)` branch,
and every one of `ternary_proj()`'s seven call sites is itself inside
the `else` branch of the same test.

So for a model with I2_S tensors, the projections go through
`f32_proj()` → `hs_ml_ternary_f32_proj()`, which decodes the **NeoGPU**
layout. If the model's bytes are in one of the upstream layouts, every
projection weight is wrong — see §2.3 for how wrong, and §4 for how to
tell which layout the file actually holds.

`tests/stage_probe_real.c` carries its own private copy of the ARM-layout
decoder, which is how the layout can be probed against the real model
even though the production path cannot decode it.

## 2. Bit layouts

Four distinct byte layouts appear in this repository — one NeoGPU
internal, three from upstream BitNet. They are **not** interchangeable.

### 2.1 The three per-row layouts

| | NeoGPU internal | upstream BitNet, ARM | upstream BitNet, x86 |
|---|---|---|---|
| Where | `hs_ml_ternary_neon.c`, `hs_mlt_lmhead_encode`, `tools/convert_bitnet_to_gguf.py` | `quantize_i2_s`, `__ARM_NEON` branch | `quantize_i2_s`, `__AVX__` branch with `ACT_PARALLEL` |
| Block size | none — one flat row | 64 weights | 128 weights |
| Bytes per block | — | 16 | 32 |
| Weights sharing a byte | 4 **consecutive** | 4, spaced 16 apart | 4, spaced 32 apart |
| Field order | **low bits first** | **high bits first** | **high bits first** |
| `bits[1:0]` | weight 4b+0 | weight 48+b | weight 96+b |
| `bits[3:2]` | weight 4b+1 | weight 32+b | weight 64+b |
| `bits[5:4]` | weight 4b+2 | weight 16+b | weight 32+b |
| `bits[7:6]` | weight 4b+3 | weight b | weight b |
| Code mapping | 0=−1, 1=0, 2=+1 | same | same |

For the NeoGPU column, `b` is the byte index within the row. For the
upstream columns, `b` is the byte index within the block, so the block's
first byte is `b = 0` and holds weights 0, 16 and 32 (ARM) or 0, 32 and
64 (x86) in its three high fields.

The two upstream columns come from the same function in the vendored
reference implementation
(`experimental/BitNet/src/ggml-bitnet-mad.cpp`), but the function
compiles to **different layouts** depending on the target:

```c
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
#define QK_I2_S 128
#elif defined(__ARM_NEON)
#define QK_I2_S 64
#endif
```

with the packing loop then using `group_idx = j / 32` (x86) or
`group_idx = j / 16` (ARM). Both branches write the most significant
field first.

The ARM layout is the one this project targets, and it is the layout
already used by the repository's own reference decoders
(`tests/redteam_i2s_neon.c`, `tests/i2s_native_probe.c`).

### 2.2 The fourth layout: four rows per byte

Upstream's x86 build has a *second* branch that is not a per-row layout
at all. When `ACT_PARALLEL` is not defined, `quantize_i2_s` takes the
`quantize_i2_s_1x4` path, which interleaves **four rows** into every
byte, column-major within a group of four rows:

```c
uint8_t packed = (uint8_t)((q0 << 6) | (q1 << 4) | (q2 << 2) | (q3 << 0));
```

`q0`..`q3` are column `col` of rows `r0`..`r3`, and the byte index is
`rg * n_per_row + col`. So within one byte:

| field | holds |
|-------|-------|
| `bits[7:6]` | row 4*rg+0, column col |
| `bits[5:4]` | row 4*rg+1, column col |
| `bits[3:2]` | row 4*rg+2, column col |
| `bits[1:0]` | row 4*rg+3, column col |

The bytes of one row are therefore strided by `n_per_row`, not
contiguous. A decoder written for a per-row layout cannot read this one
at all.

Which of the two x86 branches is compiled in depends on
`ACT_PARALLEL`. The vendored tree defines it — `include/gemm-config.h`
begins with `#define ACT_PARALLEL` — so the vendored build on x86 uses
the `j / 32` layout, and on ARM the `j / 16` layout.

### 2.3 They are measurably incompatible

Reading bytes produced by one layout with another's decoder recovers no
information:

| bytes produced by | read with | agreement | weights recovered at the right index |
|--------------------|-----------|-----------|--------------------------------------|
| upstream ARM | upstream ARM | 1.0000 | - |
| upstream x86 | upstream x86 | 1.0000 | - |
| NeoGPU | NeoGPU | 1.0000 | - |
| upstream ARM | NeoGPU | 0.3332 | 0 of 2560 |
| upstream 1x4 | NeoGPU | 0.3332 | 0 of 2560 |
| upstream x86 | NeoGPU | 0.3436 | 40 of 2560 |

Chance for three-valued codes is 0.3333.

The x86 row is not quite at chance, and the reason is worth recording.
The upstream x86 layout has a 32-weight stride against the NeoGPU
layout's 4, and the two grids coincide on two weights out of every
128-block. So 2/128 = 1.5625% of the weights are recovered exactly and
the rest only coincidentally, which predicts
0.015625 + 0.984375/3 = 0.34375 — what is measured. The ARM and 1x4
layouts have no such coincidence and land on chance exactly.

The "recovered at the right index" column is measured by perturbation:
set one weight's code to 2 and every other code to 0, then find which
decoded index became 2.

The byte streams differ too. Over 2000 random rows of 640 bytes:
upstream ARM vs upstream x86 differ in 96.3% of bytes, ARM vs NeoGPU in
98.8%, and x86 vs NeoGPU in 98.6%.

So the NeoGPU rule extracts essentially **no information** from
upstream-layout bytes: in the one case where any weight survives, it is
one in 64, surrounded by noise. The two must not be mixed.

`tests/redteam_ternary_formats.c` reproduces these figures, and asserts
that each layout is self-consistent, that every cross-layout pairing
sits at chance apart from the x86 coincidence above, and that the x86
coincidence is exactly two weights per 128-block.

### 2.4 What the loader actually does

`load_i2s_raw()` in `src/hs_ml_loader_ternary.c` reads the raw tensor
bytes with a single `fread` and returns them unmodified, with the
comment "No re-packing needed". Those bytes are then handed to
`hs_ml_ternary_f32_proj()`, which decodes them with the **NeoGPU** rule.

The code mapping is not the problem there — upstream's `quantize_i2_s`
emits 0 = −1, 1 = 0, 2 = +1, the same as the LUT. The problem is the
*layout*, and §1.4 explains why the upstream-layout decoder that exists
in this repository is unreachable.

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

Stage 2 can change a candidate's logit by at most the residual left
after Stage 1, that is 0.0185 in the normalised row, or 0.0185 ×
`row_scale` in absolute terms. A vocabulary entry outside the candidate
set is therefore not refined at all, and its logit stays at the Stage 1
value — which is why the candidate set has to be chosen so that no
excluded entry could overtake an included one. `tests/redteam_lmhead_trit.c`
checks exactly that against a real model: it requires the top 42 of the
exact F16 logits to all appear within the 200 trit candidates, and
requires the top-1 to match exactly.

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

## 4. Which layout does a given model file use?

`hs_mlt_load_gguf()` reads I2_S tensors as raw bytes and feeds them to
the NeoGPU-layout decoder with no reordering — and, per §1.4, that is the
*only* decoder they can reach. If a model's bytes are in one of the
upstream layouts, every projection weight is wrong — and, per §2.3,
wrong in a way that looks like noise rather than an obvious failure.

The repository contains reference decoders for two of the layouts, and
they disagree:

- `tests/redteam_i2s_neon.c` and `tests/i2s_native_probe.c` decode the
  **upstream ARM** layout (`group_idx = j / 16`, `group_pos = j % 16`,
  field at `6 - 2 * group_idx`).
- `tests/redteam_ternary_v3.c` decodes the **NeoGPU** layout
  (`(byte >> (2*s)) & 3` for weight `4b + s`).

### 4.1 Two different producers, two different layouts

**This repository's own converter** — `tools/convert_bitnet_to_gguf.py` —
writes GGML type 36 tensors in the **NeoGPU** layout with the canonical
code mapping. Its header says so, and its packing code
(`codes[:, b::4] << (b * 2)`, mapping −1→0, 0→1, +1→2) matches. Its
output is named `models/bitnet-2b4t-i2s.gguf` by default. A file
produced this way is read correctly by the current loader.

**Upstream BitNet's tooling** produces I2_S GGUFs through the C++
quantizer — `experimental/BitNet/utils/convert-helper-bitnet.py` and
`experimental/BitNet/utils/quantize_embeddings.py` both
invoke the `llama-quantize` binary with the type argument `"I2_S"`. That
quantizer is `quantize_i2_s` in
`experimental/BitNet/src/ggml-bitnet-mad.cpp`, whose layout is one of
the three in §2.1 and §2.2 depending on how it was built.
(`experimental/BitNet/utils/convert-hf-to-gguf-bitnet.py` is not one
of these: its `ftype_map`
offers only f32, f16, tl1 and tl2.)

Upstream's own inference script defaults to a model file named
`models/bitnet_b1_58-3B/ggml-model-i2_s.gguf`, and its embedding
quantizer writes `ggml-model-i2_s-embed-*.gguf`. **`ggml-model-i2_s.gguf`
is upstream's naming convention, not this repository's.**

The model the Pi-side tests use is
`/home/ztflynn/001/neogpu/models/ggml-model-i2_s.gguf` — upstream's
name. So it is unlikely to have come from this repository's converter,
and is likely to be in one of upstream's layouts rather than NeoGPU's.

### 4.2 Settling it

Given the file, the layouts are separable by inspection, because they
place weights into bytes differently:

- In the NeoGPU layout the four codes of a byte are four **consecutive**
  weights, and the byte stream advances by 4 weights per byte.
- In the upstream ARM and x86 layouts the four codes of a byte are
  **strided** (16 or 32 apart), and the codes of one weight are spread
  across 4 or 8 different bytes.
- In the 1x4 layout the codes of a byte belong to four **different
  rows**.

The failure signatures differ, which helps narrow it down. If the file
were in the upstream ARM or 1x4 layout, every weight would decode to
noise. If it were in the upstream x86 layout, one weight in 64 would be
exactly right and the rest would be noise — a subtly wrong model rather
than an obviously broken one.

The decisive test is behavioural rather than statistical: run the real
model under each decode and keep the one that produces coherent text.
`tests/stage_probe_real.c` carries a private copy of the ARM-layout
decoder for exactly this purpose, `tests/i2s_native_probe.c` implements
the same layout, and `tests/redteam_i2s_neon.c` cross-checks the NEON
and scalar decoders against each other.

What is not yet known is which layout that particular file carries.
That needs the file itself.

---

## 5. Source map

| What | Where |
|------|-------|
| Decode LUT and the fused 4-plane kernel | `src/hs_ml_ternary_neon.c` |
| Single-plane projection kernel | `src/hs_ml_ternary_neon.c` (`hs_ml_ternary_f32_proj`) |
| 8-plane encoder, I2_S raw loader, divergent F32 packer | `src/hs_ml_loader_ternary.c` |
| Two-stage lm_head driver, plane weights, divergent random generator | `src/hs_ml_infer.c` (`session_logits_trit`, `lmh_plane_w`, `alloc_ternary`) |
| Plane buffers, `LMH_CANDIDATES`, `use_trit_lmhead` | `include/hs_ml_infer.h` |
| This repository's GGUF converter (NeoGPU layout) | `tools/convert_bitnet_to_gguf.py` |
| Upstream I2_S quantiser (all three layouts) | `experimental/BitNet/src/ggml-bitnet-mad.cpp` |
| Upstream I2_S producers | `experimental/BitNet/utils/convert-helper-bitnet.py`, `experimental/BitNet/utils/quantize_embeddings.py` |
| Upstream-layout reference decoders | `tests/redteam_i2s_neon.c`, `tests/i2s_native_probe.c` |
| NeoGPU-layout reference decoder | `tests/redteam_ternary_v3.c` |
| Real-model lm_head checks (top-42 recall, top-1 match) | `tests/redteam_lmhead_trit.c` |
