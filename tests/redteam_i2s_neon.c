#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_neon.h>

#define I2S_QK 64u

static int32_t dot_scalar(const uint8_t *W, const int8_t *x, uint32_t K) {
    int32_t sum_x = 0;
    for (uint32_t i = 0; i < K; i++) sum_x += (int32_t)x[i];
    int32_t acc = 0;
    uint32_t nblk = K / I2S_QK;
    for (uint32_t bi = 0; bi < nblk; bi++) {
        const uint8_t *block = W + bi * (I2S_QK / 4);
        for (uint32_t j = 0; j < I2S_QK; j++) {
            uint32_t k = bi * I2S_QK + j;
            uint32_t group_idx = j / 16;
            uint32_t group_pos = j % 16;
            uint8_t raw = (block[group_pos] >> (6 - 2 * group_idx)) & 0x3;
            acc += (int32_t)raw * (int32_t)x[k];
        }
    }
    return acc - sum_x;
}

static int32_t dot_neon(const uint8_t *W, const int8_t *x, uint32_t K) {
    const uint8x16_t mask = vdupq_n_u8(3);
    uint32_t nblk = K / I2S_QK;
    uint32_t row_bytes = K / 4;
    int32_t sum_x = 0;
    for (uint32_t i = 0; i < K; i++) sum_x += (int32_t)x[i];

    int32x4_t acc = vdupq_n_s32(0);
    for (uint32_t b = 0; b < nblk; b++) {
        uint8x16_t x3 = vld1q_u8(W + b * 16);
        uint8x16_t x2 = vshrq_n_u8(x3, 2);
        uint8x16_t x1 = vshrq_n_u8(x3, 4);
        uint8x16_t x0 = vshrq_n_u8(x3, 6);
        int8x16_t q0 = vreinterpretq_s8_u8(vandq_u8(x0, mask));
        int8x16_t q1 = vreinterpretq_s8_u8(vandq_u8(x1, mask));
        int8x16_t q2 = vreinterpretq_s8_u8(vandq_u8(x2, mask));
        int8x16_t q3 = vreinterpretq_s8_u8(vandq_u8(x3, mask));
        const int8x16_t y0 = vld1q_s8(x + b * 64 + 0);
        const int8x16_t y1 = vld1q_s8(x + b * 64 + 16);
        const int8x16_t y2 = vld1q_s8(x + b * 64 + 32);
        const int8x16_t y3 = vld1q_s8(x + b * 64 + 48);
        int16x8_t acc16 = vdupq_n_s16(0);
        acc16 = vmlal_s8(acc16, vget_low_s8(q0), vget_low_s8(y0));
        acc16 = vmlal_s8(acc16, vget_high_s8(q0), vget_high_s8(y0));
        acc16 = vmlal_s8(acc16, vget_low_s8(q1), vget_low_s8(y1));
        acc16 = vmlal_s8(acc16, vget_high_s8(q1), vget_high_s8(y1));
        acc16 = vmlal_s8(acc16, vget_low_s8(q2), vget_low_s8(y2));
        acc16 = vmlal_s8(acc16, vget_high_s8(q2), vget_high_s8(y2));
        acc16 = vmlal_s8(acc16, vget_low_s8(q3), vget_low_s8(y3));
        acc16 = vmlal_s8(acc16, vget_high_s8(q3), vget_high_s8(y3));
        acc = vaddq_s32(acc, vmovl_s16(vget_low_s16(acc16)));
        acc = vaddq_s32(acc, vmovl_high_s16(acc16));
    }
    return vaddvq_s32(acc) - sum_x;
}

int main(void) {
    printf("Red-team: I2_S NEON kernel vs scalar reference\n");
    printf("===============================================\n\n");

    int failures = 0;
    uint32_t sizes[] = {64, 128, 256, 512, 1024, 2560, 4096, 6912};
    for (int s = 0; s < 8; s++) {
        uint32_t K = sizes[s];
        uint32_t row_bytes = K / 4;
        uint8_t *W = aligned_alloc(64, row_bytes);
        int8_t *x = aligned_alloc(64, K);

        srand(42 + s);
        for (uint32_t i = 0; i < row_bytes; i++) W[i] = rand() & 0xFF;
        for (uint32_t i = 0; i < K; i++) x[i] = (rand() % 255) - 127;

        int32_t ref = dot_scalar(W, x, K);
        int32_t neon = dot_neon(W, x, K);

        if (ref != neon) {
            printf("  [FAIL] K=%u: scalar=%d neon=%d\n", K, ref, neon);
            failures++;
        } else {
            printf("  [PASS] K=%u: both=%d\n", K, ref);
        }

        free(W);
        free(x);
    }

    /* Multi-row test */
    printf("\nMulti-row (N=64, K=2560):\n");
    {
        uint32_t N = 64, K = 2560;
        uint32_t row_bytes = K / 4;
        uint8_t *W = aligned_alloc(64, (size_t)N * row_bytes);
        int8_t *x = aligned_alloc(64, K);
        int32_t *ref = malloc(N * sizeof(int32_t));
        int32_t *neon_out = malloc(N * sizeof(int32_t));

        srand(99);
        for (size_t i = 0; i < (size_t)N * row_bytes; i++) W[i] = rand() & 0xFF;
        for (uint32_t i = 0; i < K; i++) x[i] = (rand() % 255) - 127;

        for (uint32_t n = 0; n < N; n++)
            ref[n] = dot_scalar(W + n * row_bytes, x, K);

        for (uint32_t n = 0; n < N; n++)
            neon_out[n] = dot_neon(W + n * row_bytes, x, K);

        int mism = 0;
        for (uint32_t n = 0; n < N; n++) {
            if (ref[n] != neon_out[n]) {
                if (mism < 5)
                    printf("  [FAIL] row %u: scalar=%d neon=%d\n", n, ref[n], neon_out[n]);
                mism++;
            }
        }
        if (mism == 0) printf("  [PASS] all %u rows match\n", N);
        else { printf("  [FAIL] %d mismatches\n", mism); failures++; }

        free(W); free(x); free(ref); free(neon_out);
    }

    printf("\n%s\n", failures == 0 ? "All I2_S NEON checks passed." : "FAILURES.");
    return failures;
}
