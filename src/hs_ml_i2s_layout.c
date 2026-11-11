/*
 * I2_S (GGML type 36) byte-layout handling. See include/hs_ml_i2s_layout.h.
 *
 * Upstream BitNet's ARM/NEON I2_S layout, as produced by
 * quantize_i2_s() in experimental/BitNet/src/ggml-bitnet-mad.cpp:
 *
 *     for (int i = 0; i < n / QK_I2_S; i++)        // 64-weight blocks
 *         for (int j = 0; j < QK_I2_S; j++) {
 *             int group_idx = j / 16;
 *             int group_pos = j % 16;
 *             i2_weight[i * 16 + group_pos] |= code << (6 - 2 * group_idx);
 *         }
 *
 * so within a block, byte p holds:
 *
 *     bits[7:6] -> weight i*64 +  0*16 + p
 *     bits[5:4] -> weight i*64 +  1*16 + p
 *     bits[3:2] -> weight i*64 +  2*16 + p
 *     bits[1:0] -> weight i*64 +  3*16 + p
 *
 * NeoGPU's own layout, as produced by tools/convert_bitnet_to_gguf.py and
 * consumed by the LUT kernel in src/hs_ml_ternary_neon.c, packs four
 * consecutive weights per byte with the low field first:
 *
 *     bits[1:0] -> weight 4b+0, [3:2] -> 4b+1, [5:4] -> 4b+2, [7:6] -> 4b+3
 *
 * The code values are the same in both, so the conversion only has to
 * relocate them.
 */

#include "hs_ml_i2s_layout.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

hs_i2s_layout hs_ml_i2s_detect_layout(int has_separate_scale_tensor) {
    /* This repository's converter writes a separate "<name>_scale"
     * tensor; upstream's quantizer puts the scale in the tail of the
     * tensor data instead. */
    return has_separate_scale_tensor ? HS_I2S_LAYOUT_NEOGPU
                                      : HS_I2S_LAYOUT_UPSTREAM_ARM;
}

uint64_t hs_ml_i2s_upstream_scale_offset(uint32_t rows, uint32_t cols) {
    return (uint64_t)rows * (cols / 4);
}

int hs_ml_i2s_to_neogpu(uint8_t *buf, uint32_t rows, uint32_t cols) {
    if (!buf || cols == 0) return -1;

    if (cols % HS_I2S_BLOCK_WEIGHTS != 0) {
        fprintf(stderr,
                "hs_ml_i2s_to_neogpu: %u columns is not a multiple of %u; "
                "upstream's I2_S stream is flat over the whole matrix, so "
                "rows cannot be converted independently\n",
                cols, HS_I2S_BLOCK_WEIGHTS);
        return -1;
    }

    const uint32_t row_bytes = cols / 4;
    uint8_t *tmp = (uint8_t *)malloc(row_bytes);
    if (!tmp) return -1;

    for (uint32_t r = 0; r < rows; r++) {
        const uint8_t *src = buf + (size_t)r * row_bytes;
        memset(tmp, 0, row_bytes);

        for (uint32_t k = 0; k < cols; k++) {
            const uint32_t blk = k / HS_I2S_BLOCK_WEIGHTS;
            const uint32_t grp = (k % HS_I2S_BLOCK_WEIGHTS) / HS_I2S_GROUP_WEIGHTS;
            const uint32_t pos = k % HS_I2S_GROUP_WEIGHTS;

            /* upstream: most significant field first */
            const uint8_t code = (uint8_t)(
                (src[blk * HS_I2S_BLOCK_BYTES + pos] >> (6 - 2 * grp)) & 3u);

            /* NeoGPU: low field first, four consecutive weights per byte */
            tmp[k / 4] |= (uint8_t)(code << (2 * (k % 4)));
        }

        memcpy(buf + (size_t)r * row_bytes, tmp, row_bytes);
    }

    free(tmp);
    return 0;
}
