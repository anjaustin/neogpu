/*
 * I2_S (GGML type 36) byte-layout handling.
 *
 * Upstream BitNet packs I2_S weights as a flat stream of 64-weight
 * blocks; NeoGPU packs four consecutive weights per byte. The two use
 * the same 2-bit code values -- 0 = -1, 1 = 0, 2 = +1 -- but they place
 * those codes in different bytes, so the bytes must be converted once
 * rather than decoded differently in the hot path.
 *
 * Converting at load time keeps the inference kernels and the GPU
 * shaders unchanged, and makes the layout a property of the file rather
 * than of the decoder.
 *
 * See docs/interface/TERNARY_WEIGHT_FORMATS.md for the full picture.
 */
#ifndef HS_ML_I2S_LAYOUT_H
#define HS_ML_I2S_LAYOUT_H

#include <stdint.h>

/* Upstream's block geometry: 64 weights in 16 bytes, four groups of 16
 * weights per byte, most significant field first. */
#define HS_I2S_BLOCK_WEIGHTS  64u
#define HS_I2S_BLOCK_BYTES    16u
#define HS_I2S_GROUP_WEIGHTS  16u

/* Which byte layout an I2_S tensor holds. */
typedef enum {
    /* NeoGPU's own: four consecutive weights per byte, low field first.
     * Produced by tools/convert_bitnet_to_gguf.py, which also writes a
     * separate "<name>_scale" tensor. */
    HS_I2S_LAYOUT_NEOGPU = 0,
    /* Upstream BitNet's ARM/NEON build: 64-weight blocks of 16 bytes,
     * four groups of 16 weights per byte, most significant field first.
     * The per-tensor scale is stored in the 32 bytes that follow the
     * packed data, not in a separate tensor. */
    HS_I2S_LAYOUT_UPSTREAM_ARM = 1
} hs_i2s_layout;

/*
 * Decide which layout an I2_S tensor holds.
 *
 * The two producers are distinguishable: this repository's converter
 * writes a separate "<name>_scale" tensor, and upstream's does not --
 * upstream puts the scale in the tail of the tensor data. A model with
 * the separate scale tensor is ours; a model without one is upstream's.
 */
hs_i2s_layout hs_ml_i2s_detect_layout(int has_separate_scale_tensor);

/*
 * Convert upstream's I2_S block layout to NeoGPU's, in place.
 *
 *   buf  : rows * (cols/4) bytes, upstream layout
 *   rows : number of rows
 *   cols : weights per row; must be a multiple of 64
 *
 * Returns 0 on success. Returns -1, leaving buf unchanged, if cols is
 * not a multiple of 64 -- upstream's stream is flat over the whole
 * matrix, so a row boundary only falls on a block boundary when the row
 * length is a multiple of the block size -- or if memory is exhausted.
 */
int hs_ml_i2s_to_neogpu(uint8_t *buf, uint32_t rows, uint32_t cols);

/*
 * Byte offset of upstream's per-tensor scale within the tensor data.
 * Upstream stores it immediately after the packed bytes.
 */
uint64_t hs_ml_i2s_upstream_scale_offset(uint32_t rows, uint32_t cols);

#endif /* HS_ML_I2S_LAYOUT_H */
