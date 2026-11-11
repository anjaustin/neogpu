/*
 * Red-team: the I2_S byte-layout conversion.
 *
 * Upstream BitNet packs I2_S weights as a flat stream of 64-weight
 * blocks. NeoGPU's kernels pack four consecutive weights per byte. The
 * code values are the same in both (0 = -1, 1 = 0, 2 = +1), but the
 * codes live in different bytes, so reading upstream's bytes with
 * NeoGPU's layout decodes every weight to noise.
 *
 * The loader converts once, at load time, so the kernels and the GPU
 * shaders are unchanged. This test pins that conversion.
 *
 * The load-time decision is which layout the file holds. This
 * repository's converter (tools/convert_bitnet_to_gguf.py) writes
 * NeoGPU's layout and a separate "<name>_scale" tensor; upstream's
 * quantizer writes its own layout and puts the scale in the tail of the
 * tensor data. So the separate scale tensor identifies the producer.
 *
 * Build and run (from the repository root):
 *     make test-i2s-repack
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <unistd.h>

#include "hs_ml_i2s_layout.h"

typedef int8_t  i8;
typedef uint8_t u8;
typedef uint32_t u32;

static int checks_failed = 0;

static void check(int ok, const char *what) {
    if (!ok) {
        printf("          FAILED CHECK: %s\n", what);
        checks_failed++;
    }
}

/* ================================================================
 * The two packers, and the two decoders
 * ================================================================ */

/*
 * Upstream BitNet's ARM/NEON I2_S packing, as in quantize_i2_s() in
 * experimental/BitNet/src/ggml-bitnet-mad.cpp:
 *
 *   for (int i = 0; i < n / QK_I2_S; i++)
 *       for (int j = 0; j < QK_I2_S; j++) {
 *           int group_idx = j / 16;
 *           int group_pos = j % 16;
 *           i2_weight[i * 16 + group_pos] |= code << (6 - 2 * group_idx);
 *       }
 *
 * The code is the weight plus one: 0 = -1, 1 = 0, 2 = +1.
 */
static void pack_upstream_arm(const i8* w, int n, u8* out) {
    memset(out, 0, (size_t)n / 4);
    for (int i = 0; i < n / 64; i++)
        for (int j = 0; j < 64; j++) {
            int gi = j / 16, gp = j % 16;
            out[i * 16 + gp] |= (u8)(((u8)(w[i * 64 + j] + 1) & 3) << (6 - 2 * gi));
        }
}

/*
 * NeoGPU's own packing, as in pack_ternary_simple() in
 * tools/convert_bitnet_to_gguf.py:
 *
 *   for b in range(4):
 *       packed |= codes[:, b::4] << (b * 2)
 *
 * Four consecutive weights per byte, low field first.
 */
static void pack_neogpu(const i8* w, int n, u8* out) {
    memset(out, 0, (size_t)n / 4);
    for (int i = 0; i < n / 4; i++)
        for (int s = 0; s < 4; s++)
            out[i] |= (u8)((((u8)(w[i * 4 + s] + 1) & 3) << (2 * s)));
}

/* The canonical code mapping, as the LUT kernel in
 * src/hs_ml_ternary_neon.c applies it: code - 1. */
static int decode_canonical(u8 code) {
    return (int)code - 1;
}

/* Decode a NeoGPU-layout row back to weights. */
static void decode_neogpu_row(const u8* row, int n, i8* out) {
    for (int k = 0; k < n; k++) {
        u8 code = (u8)((row[k / 4] >> (2 * (k % 4))) & 3);
        out[k] = (i8)decode_canonical(code);
    }
}

/* ================================================================
 * Part 1 -- the conversion makes upstream bytes identical to ours
 * ================================================================ */

static int part1_conversion(void) {
    const int ROWS = 16;
    const int COLS = 2560;
    const int N = ROWS * COLS;

    i8* w   = (i8*)malloc((size_t)N);
    u8* up  = (u8*)malloc((size_t)N / 4);
    u8* neo = (u8*)malloc((size_t)N / 4);
    u8* got = (u8*)malloc((size_t)N / 4);
    i8* back = (i8*)malloc((size_t)N);

    /* Ternary weights, plus the zero and both signs. */
    for (int i = 0; i < N; i++) {
        int r = (i * 37) % 7;
        w[i] = (i8)(r < 3 ? 0 : (r < 5 ? 1 : -1));
    }

    pack_upstream_arm(w, N, up);
    pack_neogpu(w, N, neo);

    memcpy(got, up, (size_t)N / 4);
    int rc = hs_ml_i2s_to_neogpu(got, ROWS, COLS);

    long diff = 0;
    for (int i = 0; i < N / 4; i++) if (got[i] != neo[i]) diff++;

    printf("  Part 1  upstream bytes -> NeoGPU layout\n");
    printf("          conversion returned %d\n", rc);
    printf("          bytes differing from natively-packed NeoGPU bytes: %ld of %d\n",
           diff, N / 4);

    check(rc == 0, "the conversion failed");
    check(diff == 0, "the converted bytes differ from natively-packed NeoGPU bytes");

    /* And the converted bytes decode back to the original weights. */
    int bad = 0;
    for (int r = 0; r < ROWS; r++) {
        decode_neogpu_row(got + (size_t)r * COLS / 4, COLS, back);
        for (int k = 0; k < COLS; k++)
            if (back[k] != w[(size_t)r * COLS + k]) bad++;
    }
    printf("          weights recovered from the converted bytes: %d wrong of %d\n",
           bad, N);
    check(bad == 0, "the converted bytes do not decode to the original weights");

    /* Without the conversion, the same bytes decode to noise. */
    int bad_raw = 0;
    for (int r = 0; r < ROWS; r++) {
        decode_neogpu_row(up + (size_t)r * COLS / 4, COLS, back);
        for (int k = 0; k < COLS; k++)
            if (back[k] != w[(size_t)r * COLS + k]) bad_raw++;
    }
    printf("          weights recovered WITHOUT the conversion: %d wrong of %d\n",
           bad_raw, N);
    check(bad_raw > N / 2, "the unconverted bytes decoded correctly, so the conversion is unnecessary");

    free(w); free(up); free(neo); free(got); free(back);
    return 0;
}

/* ================================================================
 * Part 2 -- the layout decision, and the rejected shapes
 * ================================================================ */

/* A row length that is not a multiple of the upstream block size
 * (64) cannot be converted: upstream's stream is flat over the whole
 * matrix, so a row boundary only falls on a block boundary when the
 * row length is a multiple of 64. The conversion must refuse rather
 * than relocate the wrong codes. */
static int part2_rejects_bad_shape(void) {
    const uint32_t ROWS = 4;
    const uint32_t COLS = 2560 + 32;      /* not a multiple of 64 */
    const size_t  BYTES = (size_t)ROWS * (COLS / 4);

    uint8_t* buf = (uint8_t*)malloc(BYTES);
    uint8_t* before = (uint8_t*)malloc(BYTES);
    for (size_t i = 0; i < BYTES; i++) buf[i] = (uint8_t)(i * 7 + 3);
    memcpy(before, buf, BYTES);

    int rc = hs_ml_i2s_to_neogpu(buf, ROWS, COLS);
    int unchanged = memcmp(buf, before, BYTES) == 0;

    printf("  Part 2  a row length that is not a multiple of 64\n");
    printf("          cols = %u, conversion returned %d\n", COLS, rc);
    printf("          buffer left unchanged: %s\n", unchanged ? "yes" : "NO");

    check(rc != 0, "the conversion accepted a row length that is not a multiple of 64");
    check(unchanged, "the conversion modified the buffer before failing");

    free(buf); free(before);
    return 0;
}

/* ================================================================
 * Part 3 -- the layout decision
 * ================================================================ */

static int part3_detection(void) {
    printf("  Part 3  layout detection\n");
    printf("          separate scale tensor present -> %s\n",
           hs_ml_i2s_detect_layout(1) == HS_I2S_LAYOUT_NEOGPU
               ? "NeoGPU" : "upstream ARM");
    printf("          separate scale tensor absent  -> %s\n",
           hs_ml_i2s_detect_layout(0) == HS_I2S_LAYOUT_UPSTREAM_ARM
               ? "upstream ARM" : "NeoGPU");

    check(hs_ml_i2s_detect_layout(1) == HS_I2S_LAYOUT_NEOGPU,
          "a model with a separate scale tensor must be read as NeoGPU layout");
    check(hs_ml_i2s_detect_layout(0) == HS_I2S_LAYOUT_UPSTREAM_ARM,
          "a model without a separate scale tensor must be read as upstream layout");

    /* Upstream's scale sits immediately after the packed bytes. */
    uint64_t off = hs_ml_i2s_upstream_scale_offset(16, 2560);
    printf("          upstream scale offset for 16 x 2560: %llu (packed bytes = %d)\n",
           (unsigned long long)off, 16 * (2560 / 4));
    check(off == (uint64_t)16 * (2560 / 4), "the upstream scale offset is wrong");

    return 0;
}

/* ================================================================
 * Part 4 -- the source pins
 * ================================================================ */

static char* read_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); return NULL; }
    char* buf = (char*)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, f);
    buf[got] = 0;
    fclose(f);
    return buf;
}

static int find_root(char* out, size_t outlen) {
    char dir[1024];
    if (!getcwd(dir, sizeof(dir))) return -1;
    for (int depth = 0; depth < 6; depth++) {
        char probe[1200];
        snprintf(probe, sizeof(probe), "%s/src/hs_ml_loader_ternary.c", dir);
        FILE* f = fopen(probe, "rb");
        if (f) {
            fclose(f);
            snprintf(out, outlen, "%s", dir);
            return 0;
        }
        char* slash = strrchr(dir, '/');
        if (!slash || slash == dir) break;
        *slash = 0;
    }
    return -1;
}

static int contains(const char* hay, const char* needle) {
    return hay && strstr(hay, needle) != NULL;
}

static void part4_source_pins(const char* root) {
    char path[1200];
    printf("  Part 4  source pins\n");

    snprintf(path, sizeof(path), "%s/experimental/BitNet/src/ggml-bitnet-mad.cpp", root);
    char* up = read_file(path);
    snprintf(path, sizeof(path), "%s/tools/convert_bitnet_to_gguf.py", root);
    char* py = read_file(path);
    snprintf(path, sizeof(path), "%s/src/hs_ml_loader_ternary.c", root);
    char* loader = read_file(path);
    snprintf(path, sizeof(path), "%s/src/hs_ml_i2s_layout.c", root);
    char* layout = read_file(path);

    if (!up || !py || !loader || !layout) {
        printf("          SKIPPED -- could not read the sources\n");
        free(up); free(py); free(loader); free(layout);
        return;
    }

    /* Upstream's ARM packing rule, so the test's copy cannot drift. */
    int up_rule = contains(up, "int group_idx = j / 16;") &&
                  contains(up, "i2_weight[i * 16 + group_pos] |= temp;");
    printf("          upstream ARM packing rule unchanged        %s\n",
           up_rule ? "OK" : "CHANGED");
    check(up_rule, "upstream's ARM I2_S packing rule changed");

    /* This repository's packing, and its separate scale tensor. */
    int our_rule = contains(py, "packed |= codes[:, b::4] << (b * 2)") &&
                   contains(py, "(name + \"_scale\", GGML_TYPE_F32, [1], scale_data.tobytes())");
    printf("          our packing and separate scale tensor      %s\n",
           our_rule ? "OK" : "CHANGED");
    check(our_rule, "our converter's packing or scale tensor changed");

    /* The loader must actually convert, and must decide the layout. */
    int wired = contains(loader, "hs_ml_i2s_detect_layout(si >= 0)") &&
                contains(loader, "hs_ml_i2s_to_neogpu(raw, rows, cols)") &&
                contains(loader, "hs_ml_i2s_upstream_scale_offset(rows, cols)") &&
                contains(loader, "tis[wi].dims[0] != (uint64_t)cols");
    printf("          loader converts and picks the layout       %s\n",
           wired ? "OK" : "CHANGED");
    check(wired, "the loader no longer converts I2_S tensors");

    /* And the conversion itself must still be there. */
    int conv = contains(layout, "int hs_ml_i2s_to_neogpu(uint8_t *buf, uint32_t rows, uint32_t cols)") &&
               contains(layout, "(6 - 2 * grp)") &&
               contains(layout, "code << (2 * (k % 4))");
    printf("          conversion implements both layouts         %s\n",
           conv ? "OK" : "CHANGED");
    check(conv, "the conversion no longer matches the two layouts");

    free(up); free(py); free(loader); free(layout);
}

/* ================================================================ */

int main(void) {
    printf("Red-team: I2_S byte-layout conversion\n");
    printf("========================================\n");
    printf("Docs: docs/interface/TERNARY_WEIGHT_FORMATS.md section 2\n\n");

    part1_conversion();
    part2_rejects_bad_shape();
    part3_detection();

    char root[1024];
    if (find_root(root, sizeof(root)) != 0) {
        printf("\n  Part 4 SKIPPED -- could not locate the repository root\n");
    } else {
        part4_source_pins(root);
    }

    printf("\n%s\n", checks_failed ? "FAILURES DETECTED" : "DONE");
    return checks_failed ? 1 : 0;
}
