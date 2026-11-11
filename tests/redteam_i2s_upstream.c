/*
 * Red-team: our I2_S conversion against upstream BitNet's own code.
 *
 * This is the bedrock check. Everything else about the I2_S layout can
 * be argued about; this one cannot, because it runs against upstream's
 * real producer and upstream's real consumer.
 *
 * Upstream BitNet's I2_S producer is quantize_i2_s() and its consumer is
 * ggml_vec_dot_i2_i8_s_1x1(). Both are extracted verbatim from
 * experimental/BitNet/src/ggml-bitnet-mad.cpp at build time by
 * tools/extract_upstream_i2s.py, so this test always runs against the
 * reference implementation rather than a copy of it.
 *
 * The identity that must hold:
 *
 *      ours == upstream - sum(activations)
 *
 * upstream's kernel accumulates the RAW codes (0, 1, 2); the centred
 * form subtracts one per weight. Ours decodes code-1 directly. So if
 * our conversion puts the codes in the right bytes, the two agree
 * exactly once the centring is undone.
 *
 * The scale is checked too: upstream writes it at offset n/4 of the
 * tensor data, and it must equal max|w|.
 *
 * Build and run (from the repository root):
 *     make test-i2s-upstream
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* upstream's real code, generated at build time */
#include "upstream_i2s_ref.c"

/* our conversion */
#include "hs_ml_i2s_layout.h"

static int checks_failed = 0;

static void check(int ok, const char *what) {
    if (!ok) {
        printf("          FAILED CHECK: %s\n", what);
        checks_failed++;
    }
}

#if defined(__ARM_NEON)

static int part_reference(void) {
    const int64_t NROW = 4;
    const int64_t NCOL = 2560;
    const int64_t N = NROW * NCOL;

    float* w = (float *)malloc((size_t)N * sizeof(float));
    uint8_t* up = (uint8_t *)calloc((size_t)N / 4 + 64, 1);
    uint8_t* conv = (uint8_t *)calloc((size_t)N / 4 + 64, 1);
    int8_t* y = (int8_t *)malloc((size_t)NCOL);
    float* dot = (float *)malloc(sizeof(float));

    /* Ternary weights with a known maximum magnitude. */
    float maxabs = 0.0f;
    uint32_t seed = 12345u;
    for (int64_t i = 0; i < N; i++) {
        seed = seed * 1103515245u + 12345u;
        int r = (int)((seed >> 16) % 5);
        float v = (r == 0) ? 0.0f : ((r == 1) ? 1.0f : ((r == 2) ? -1.0f : 0.25f));
        w[i] = v;
        float a = v < 0 ? -v : v;
        if (a > maxabs) maxabs = a;
    }

    /* upstream's own producer */
    quantize_i2_s(w, up, NROW, NCOL, NULL);

    /* upstream writes the scale at offset n/4 of the tensor data */
    float up_scale = 0.0f;
    memcpy(&up_scale, up + (size_t)(N / 4), sizeof(float));

    /* activations */
    int sum_y = 0;
    for (int64_t k = 0; k < NCOL; k++) {
        seed = seed * 1103515245u + 12345u;
        y[k] = (int8_t)((int)((seed >> 16) % 7) - 3);
        sum_y += y[k];
    }

    /* upstream's own consumer, on row 0 */
    ggml_vec_dot_i2_i8_s_1x1((int)NCOL, dot, sizeof(float),
                              up, (size_t)NCOL, y, 1, 1);

    /* our conversion */
    memcpy(conv, up, (size_t)N / 4 + 64);
    int rc = hs_ml_i2s_to_neogpu(conv, (uint32_t)NROW, (uint32_t)NCOL);

    /* our decode of the converted bytes, row 0, same activations */
    int64_t ours = 0;
    for (int64_t k = 0; k < NCOL; k++) {
        uint8_t code = (uint8_t)((conv[k / 4] >> (2 * (k % 4))) & 3);
        ours += (int64_t)((int)code - 1) * (int64_t)y[k];
    }
    int64_t expect = (int64_t)dot[0] - sum_y;

    /* and the unconverted bytes must NOT satisfy it */
    int64_t raw = 0;
    for (int64_t k = 0; k < NCOL; k++) {
        uint8_t code = (uint8_t)((up[k / 4] >> (2 * (k % 4))) & 3);
        raw += (int64_t)((int)code - 1) * (int64_t)y[k];
    }

    printf("  Reference check against upstream BitNet's own code\n");
    printf("    upstream's scale at offset n/4 : %.6f  (max|w| = %.6f)\n",
           (double)up_scale, (double)maxabs);
    printf("    upstream's raw code dot product : %.1f\n", (double)dot[0]);
    printf("    ours (converted bytes)          : %lld\n", (long long)ours);
    printf("    upstream - sum(activations)      : %lld\n", (long long)expect);
    printf("    identity holds                   : %s\n",
           ours == expect ? "YES" : "NO");
    printf("    without the conversion           : %lld  (identity holds: %s)\n",
           (long long)raw, raw == expect ? "YES" : "NO");

    check(rc == 0, "the conversion failed");
    check(ours == expect, "our decode disagrees with upstream's own kernel");
    check(raw != expect, "the unconverted bytes already agree, so the conversion is unnecessary");
    check(up_scale == maxabs, "upstream's scale is not max|w|");

    free(w); free(up); free(conv); free(y); free(dot);
    return 0;
}

#else  /* !__ARM_NEON */

static int part_reference(void) {
    printf("  Reference check against upstream BitNet's own code\n");
    printf("    SKIPPED -- upstream's I2_S code has an ARM build only, and\n");
    printf("    this test needs the NEON path to be meaningful. CI runs on\n");
    printf("    ARM64 runners, where it does run.\n");
    return 0;
}

#endif

int main(void) {
    printf("Red-team: I2_S conversion vs upstream BitNet's own code\n");
    printf("======================================================\n\n");

    part_reference();

    printf("\n%s\n", checks_failed ? "FAILURES DETECTED" : "DONE");
    return checks_failed ? 1 : 0;
}
