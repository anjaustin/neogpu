/*
 * Red-team: hs_ml_lmhead_stage1_i8() activation-dequantization bounds.
 *
 * hs_ml_lmhead_stage1_i8() is the int8-activation compatibility entry point for
 * the fused 4-plane lm_head Stage 1 kernel. It dequantizes the int8
 * activation vector to float32 and delegates to hs_ml_lmhead_stage1().
 *
 * This test exercises that dequantization step under AddressSanitizer, which
 * is the only way to observe a write that lands outside the buffer the
 * function owns. Two independent bounds are checked:
 *
 *   Case "read"  the caller's in_i8 buffer is sized to exactly K elements.
 *                A correct implementation reads K elements and stops.
 *
 *   Case "write" the caller's in_i8 buffer is oversized (K*4), so the read
 *                side is in bounds and any violation reported is a write
 *                past the end of the function's own scratch buffer.
 *
 * Build (requires -fsanitize=address for the bounds checks to be visible):
 *
 *   gcc -fsanitize=address -g -Iinclude \
 *       src/hs_ml_ternary_neon.c tests/redteam_lmhead_i8_bounds.c \
 *       -o tests/redteam_lmhead_i8_bounds -lm -lpthread
 *
 * Without the sanitizer the calls still run and the results are still
 * compared against a reference; only the out-of-bounds access goes
 * unobserved. The summary line reports which mode was used.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

extern void hs_ml_lmhead_stage1_i8(float *out, const int8_t *in_i8,
                                    float act_scale,
                                    const uint8_t *P0, const uint8_t *P1,
                                    const uint8_t *P2, const uint8_t *P3,
                                    const uint16_t *row_scale,
                                    uint32_t N, uint32_t K);

#if defined(__SANITIZE_ADDRESS__)
#  define ASAN_ON 1
#elif defined(__has_feature)
#  if __has_feature(address_sanitizer)
#    define ASAN_ON 1
#  endif
#endif

#define K_DIM   2560u          /* BitNet b1.58 2B-4T hidden size */
#define N_ROWS  1u
#define SCRATCH 4096u          /* size of the callee's static scratch buffer */

/* Reference: decode the int8 activation and evaluate the 4-plane dot
 * product directly, with no scratch buffer in between. */
static void reference_stage1(float *out, const int8_t *in_i8, float act_scale,
                             const uint8_t *P0, const uint8_t *P1,
                             const uint8_t *P2, const uint8_t *P3,
                             const uint16_t *row_scale,
                             uint32_t N, uint32_t K) {
    const uint32_t row_bytes = K / 4;
    const float W[4] = { 1.0f, 1.0f / 3.0f, 1.0f / 9.0f, 1.0f / 27.0f };
    const uint8_t *P[4] = { P0, P1, P2, P3 };

    for (uint32_t n = 0; n < N; n++) {
        float acc = 0.0f;
        for (uint32_t bi = 0; bi < row_bytes; bi++) {
            for (int s = 0; s < 4; s++) {
                /* 4 consecutive codes per byte, low bits first, code c -> c-1 */
                float val = 0.0f;
                for (int p = 0; p < 4; p++) {
                    const uint8_t byte = P[p][(size_t)n * row_bytes + bi];
                    const float t = (float)((byte >> (s * 2)) & 3) - 1.0f;
                    val += t * W[p];
                }
                acc += val * ((float)in_i8[bi * 4 + s] / act_scale);
            }
        }
        /* F16 row_scale decode, as the kernel does */
        uint32_t h16 = row_scale[n];
        uint32_t eu  = (h16 >> 10) & 0x1Fu;
        uint32_t mu  = h16 & 0x03FFu;
        uint32_t fu  = ((eu + 112u) << 23) | (mu << 13);
        float rsc; memcpy(&rsc, &fu, 4);
        out[n] = acc * rsc;
    }
}

static uint32_t fail_count = 0;

/* Capacity guard: the callee must reject K beyond its scratch capacity
 * rather than write outside the buffer. It reports the rejection by
 * zeroing out[]. */
static void run_guard_case(void) {
    const uint32_t K = 8192u, N = N_ROWS, row_bytes = K / 4;

    int8_t   *in_i8    = (int8_t *)malloc(K);
    uint8_t  *P[4];
    for (int i = 0; i < 4; i++)
        P[i] = (uint8_t *)calloc((size_t)N * row_bytes, 1);
    uint16_t *row_scale = (uint16_t *)calloc(N, sizeof(uint16_t));
    float    *out = (float *)malloc((size_t)N * sizeof(float));

    for (uint32_t i = 0; i < K; i++) in_i8[i] = (int8_t)(i & 0x7F);
    for (uint32_t n = 0; n < N; n++) out[n] = 12345.0f;

    printf("case guard : K = %u (capacity 4096) -> expect rejection\n", K);

    hs_ml_lmhead_stage1_i8(out, in_i8, 127.0f, P[0], P[1], P[2], P[3],
                          row_scale, N, K);

    int zeroed = (out[0] == 0.0f);
    printf("            out[0] = %.1f  %s\n", out[0],
           zeroed ? "PASS (rejected, output zeroed)" : "FAIL (not rejected)");
    if (!zeroed) fail_count++;

    free(in_i8); free(out); free(row_scale);
    for (int i = 0; i < 4; i++) free(P[i]);
}

static void run_case(const char *name, uint32_t in_bytes) {
    const uint32_t K = K_DIM, N = N_ROWS, row_bytes = K / 4;

    /* in_i8 sized by the caller */
    int8_t  *in_i8   = (int8_t *)malloc(in_bytes ? in_bytes : 1);
    uint8_t *P[4];
    for (int i = 0; i < 4; i++)
        P[i] = (uint8_t *)malloc((size_t)N * row_bytes);
    uint16_t *row_scale = (uint16_t *)malloc((size_t)N * sizeof(uint16_t));
    float    *out  = (float *)malloc((size_t)N * sizeof(float));
    float    *ref  = (float *)malloc((size_t)N * sizeof(float));

    for (uint32_t i = 0; i < in_bytes; i++) in_i8[i] = (int8_t)((i * 37) % 127 - 63);
    for (int p = 0; p < 4; p++)
        for (size_t i = 0; i < (size_t)N * row_bytes; i++)
            P[p][i] = (uint8_t)((i * 91 + p * 17) & 0xFF);
    for (uint32_t n = 0; n < N; n++) row_scale[n] = 0x3C00;  /* 1.0 in F16 */

    printf("case %-6s : in_i8 = %u bytes (K = %u)\n", name, in_bytes, K);

    /* Reference first, so a bounds violation in the callee is not masked. */
    reference_stage1(ref, in_i8, 127.0f, P[0], P[1], P[2], P[3], row_scale, N, K);

    hs_ml_lmhead_stage1_i8(out, in_i8, 127.0f, P[0], P[1], P[2], P[3],
                          row_scale, N, K);

    float max_diff = 0.0f, max_rel = 0.0f;
    for (uint32_t n = 0; n < N; n++) {
        float d = out[n] - ref[n];
        if (d < 0) d = -d;
        if (d > max_diff) max_diff = d;
        float denom = ref[n] < 0 ? -ref[n] : ref[n];
        if (denom < 1.0f) denom = 1.0f;
        float rel = d / denom;
        if (rel > max_rel) max_rel = rel;
    }
    /* Relative tolerance: the ARM build sums the same terms in a different
     * order (four accumulators vs one), so absolute error scales with the
     * magnitude of the result. */
    printf("            max_diff = %.3e  max_rel = %.3e  %s\n",
           max_diff, max_rel, max_rel < 1e-4f ? "PASS" : "FAIL");
    if (!(max_rel < 1e-4f)) fail_count++;

    free(in_i8); free(out); free(ref); free(row_scale);
    for (int i = 0; i < 4; i++) free(P[i]);
}

int main(int argc, char **argv) {
    const char *which = (argc > 1) ? argv[1] : "both";

    printf("Red-team: hs_ml_lmhead_stage1_i8 activation-dequant bounds\n");
    printf("============================================================\n");
#ifdef ASAN_ON
    printf("AddressSanitizer: ACTIVE (bounds violations will be reported)\n\n");
#else
    printf("AddressSanitizer: NOT ACTIVE (bounds violations will not be seen)\n\n");
#endif

    if (!strcmp(which, "read") || !strcmp(which, "both"))
        run_case("read", K_DIM);          /* caller supplies exactly K bytes */

    if (!strcmp(which, "write") || !strcmp(which, "both"))
        run_case("write", K_DIM * 4);     /* oversized input: isolates the write */

    if (!strcmp(which, "guard") || !strcmp(which, "both"))
        run_guard_case();

    printf("\n%s\n", fail_count ? "FAILURES DETECTED" : "DONE");
    return fail_count ? 1 : 0;
}
