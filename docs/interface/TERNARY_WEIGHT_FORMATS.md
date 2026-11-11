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

| `m->use_i2s` | weights come from | decoder | code mapping | layout of the bytes | layout the decoder sees |
|---|---|---|---|---|---|
| true — the model has any I2_S tensor | `load_i2s_raw()`, upstream's bytes, converted at load time (§1.4) | `hs_ml_ternary_f32_proj()` (the LUT) | canonical | upstream's (§2) | NeoGPU — matches after the conversion |
| false — an F32 model, or a random one | `f32_to_ternary_packed()` or `alloc_ternary()` | `hs_ml_gemm_ternary_mt()` via `hs_ml_route_mt()` | routing | NeoGPU | NeoGPU — matches |

`m->use_i2s` is set only when a projection tensor is actually stored as
`GGML_TYPE_I2_S`, so a weight packed by `f32_to_ternary_packed()` can
only ever reach the routing decoder.

### 1.4 The I2_S path converts the layout at load time

An I2_S tensor's bytes are in upstream's layout, but the projection
kernels read NeoGPU's. The loader reconciles the two once, at load time,
in `hs_ml_i2s_to_neogpu()` (`src/hs_ml_i2s_layout.c`): after the
conversion the bytes are byte-identical to what this repository's own
converter produces for the same weights, so the kernels and the GPU
shaders need no special case.

Before that conversion existed, an I2_S model's projections were read
with NeoGPU's layout and every weight was noise — see §2.3 for how
wrong, and §4 for how the layout is decided.

The conversion is checked against upstream BitNet's own code, not
against a copy of it: `tools/extract_upstream_i2s.py` pulls upstream's
`quantize_i2_s()` (the producer) and `ggml_vec_dot_i2_i8_s_1x1()` (the
consumer) out of the vendored source at build time, and
`tests/redteam_i2s_upstream.c` checks that our decode of the converted
bytes equals upstream's own answer once the centring is undone
(upstream accumulates the raw codes 0, 1, 2; ours decodes code − 1).
Measured: upstream's raw dot product 191, ours 143, and
191 − Σ(activations) = 143 exactly.

`src/hs_ml_infer.c` also contains `i2s_proj_neon()`, an int8 ARM-layout
decoder. It is unreachable: its only call site is inside
`ternary_proj()`'s `if (m->use_i2s)` branch, and every one of
`ternary_proj()`'s seven call sites is itself inside the `else` branch
of the same test. It is left in place as a reference implementation of
the upstream layout, but it must not be re-enabled without also
removing the load-time conversion — after the conversion the weights are
in NeoGPU's layout, not upstream's.

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
bytes with a single `fread`. Those bytes are then converted to NeoGPU's
layout by `hs_ml_i2s_to_neogpu()` (§1.4) and handed to
`hs_ml_ternary_f32_proj()`, which decodes them with the **NeoGPU**
rule.

The code mapping is not the problem there — upstream's `quantize_i2_s`
emits 0 = −1, 1 = 0, 2 = +1, the same as the LUT. The problem was the
*layout*, and the conversion fixes it.

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

### 4.1 Two different producers, two different layouts

**This repository's own converter** — `tools/convert_bitnet_to_gguf.py` —
writes GGML type 36 tensors in the **NeoGPU** layout with the canonical
code mapping. Its header says so, and its packing code
(`codes[:, b::4] << (b * 2)`, mapping −1→0, 0→1, +1→2) matches. It
also writes a separate `<name>_scale` F32 tensor holding the per-tensor
scale. Its output is named `models/bitnet-2b4t-i2s.gguf` by default.

**Upstream BitNet's tooling** produces I2_S GGUFs through the C++
quantizer — `experimental/BitNet/utils/convert-helper-bitnet.py` and
`experimental/BitNet/utils/quantize_embeddings.py` both invoke the
`llama-quantize` binary with the type argument `"I2_S"`. That quantizer
is `quantize_i2_s` in
`experimental/BitNet/src/ggml-bitnet-mad.cpp`, whose layout is one of
the three in §2.1 and §2.2 depending on how it was built.
(`experimental/BitNet/utils/convert-hf-to-gguf-bitnet.py` is not one of
these: its `ftype_map` offers only f32, f16, tl1 and tl2.)

Upstream's layout also carries its per-tensor scale differently: the
scale is stored in the 32 bytes that follow the packed data, not in a
separate tensor. That is visible in upstream's ggml fork, where the
tensor's byte size is `nrows * n_per_row / 4 + 32` and `quantize_i2_s`
writes the scale at offset `n / 4`.

Upstream's own inference script defaults to a model file named
`models/bitnet_b1_58-3B/ggml-model-i2_s.gguf`, and its embedding
quantizer writes `ggml-model-i2_s-embed-*.gguf`. **`ggml-model-i2_s.gguf`
is upstream's naming convention, not this repository's.** The model the
Pi-side tests use is named exactly that.

### 4.2 How the loader decides

The two producers differ in a way that is visible in the file itself,
so the loader does not have to guess:

| the model has | the producer was | the layout is |
|---|---|---|
| a separate `<name>_scale` tensor | this repository's converter | NeoGPU — read as-is |
| no separate scale tensor | upstream's quantizer | upstream ARM — converted on load |

`hs_ml_i2s_detect_layout()` in `src/hs_ml_i2s_layout.c` makes that
call, and `hs_ml_i2s_to_neogpu()` does the conversion. After it, the
bytes are byte-identical to what this repository's converter produces for
the same weights, which `tests/redteam_i2s_repack.c` checks.

For an upstream file, the scale is read from the tail of the tensor data
(`hs_ml_i2s_upstream_scale_offset()`) and applied by `f32_proj()` like
any other per-tensor scale.

### 4.3 What is still not covered

- **Upstream's x86 and 1x4 layouts are not handled.** The loader assumes
  the ARM layout for an upstream file. A file produced by upstream's
  x86 build would be misread. Detecting that would need the file, and
  the loader logs which layout it chose so a wrong choice is visible.
- **A model mixing I2_S and F32 tensors.** `m->use_i2s` is set if *any*
  projection tensor in the model is I2_S, and it then selects the path
  for *all* of them. An F32-packed projection in such a model would be
  read by the LUT, which uses a different code mapping from the one it
  was packed with.

### 4.4 Confirming it on the real model

The conversion removes the need to resolve the layout by inspection, but
confirming it against the real file is still worth doing. The
repository carries the reference decoders for the upstream ARM layout:
`tests/stage_probe_real.c` has a private copy of the ARM-layout decoder,
`tests/i2s_native_probe.c` implements the same layout, and
`tests/redteam_i2s_neon.c` cross-checks the NEON and scalar decoders
against each other. Running the real model under each decode and
keeping the one that produces coherent text settles it outright.

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
| I2_S layout conversion, layout decision, scale offset | `src/hs_ml_i2s_layout.c`, `include/hs_ml_i2s_layout.h` |
| Upstream I2_S quantiser (all three layouts) | `experimental/BitNet/src/ggml-bitnet-mad.cpp` |
| Upstream I2_S producers | `experimental/BitNet/utils/convert-helper-bitnet.py`, `experimental/BitNet/utils/quantize_embeddings.py` |
| Upstream-layout reference decoders | `tests/redteam_i2s_neon.c`, `tests/i2s_native_probe.c` |
| NeoGPU-layout reference decoder | `tests/redteam_ternary_v3.c` |
| Real-model lm_head checks (top-42 recall, top-1 match) | `tests/redteam_lmhead_trit.c` |
| I2_S conversion, byte-for-byte against both producers | `tests/redteam_i2s_repack.c` |
| I2_S conversion, against upstream's own producer and consumer | `tests/redteam_i2s_upstream.c`, `tools/extract_upstream_i2s.py` |
