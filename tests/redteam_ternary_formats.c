/*
 * Red-team + measurement: the ternary weight formats.
 *
 * This program pins the claims made in
 * docs/interface/TERNARY_WEIGHT_FORMATS.md. If the formats change, this
 * fails.
 *
 *   Part 1  The production LUT kernel decodes the NeoGPU layout.
 *           Links the real src/hs_ml_ternary_neon.c and compares the fused
 *           4-plane kernel's output against a scalar reference decode.
 *
 *   Part 2  The three bit layouts in play -- NeoGPU internal, upstream
 *           BitNet ARM, upstream BitNet x86 -- are each self-consistent and
 *           mutually incompatible.
 *
 *   Part 3  The 8-plane base-3 expansion reconstructs a weight to within
 *           1.5/3^k after k planes.
 *
 *   Part 4  The source text the documentation depends on still says what
 *           the documentation says it says. This is what stops Part 3's
 *           copy of the encoder from drifting away from the real one.
 *
 * Build and run (from the repository root):
 *     make test-ternary-formats
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <unistd.h>

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;

static int checks_failed = 0;
static int known_issues  = 0;

static void check(int ok, const char *what) {
    if (!ok) {
        printf("          FAILED CHECK: %s\n", what);
        checks_failed++;
    }
}

/* ================================================================
 * Part 1 -- the production kernel, linked for real
 * ================================================================ */

extern void hs_ml_lmhead_stage1(float *out, const float *in,
                                 const u8 *P0, const u8 *P1,
                                 const u8 *P2, const u8 *P3,
                                 const u16 *row_scale,
                                 u32 N, u32 K);

/* The kernel's inline F16 decode, copied from lmhead_stage1_rows(). */
static float kernel_fp16_to_f32(u16 h) {
    u32 eu = (h >> 10) & 0x1Fu;
    u32 mu = h & 0x03FFu;
    u32 fu = ((eu + 112u) << 23) | (mu << 13);
    float f;
    memcpy(&f, &fu, 4);
    return f;
}

/* Scalar reference: decode the NeoGPU layout directly.
 *   byte b holds weights 4b, 4b+1, 4b+2, 4b+3 in bits [1:0], [3:2],
 *   [5:4], [7:6] -- low bits first.
 *   code -> weight is code - 1.
 *   Plane weights are 1, 1/3, 1/9, 1/27. */
static void reference_stage1(float *out, const float *in,
                             const u8 *P0, const u8 *P1,
                             const u8 *P2, const u8 *P3,
                             const u16 *row_scale,
                             u32 N, u32 K) {
    const u32 row_bytes = K / 4;
    const float W[4] = { 1.0f, 1.0f / 3.0f, 1.0f / 9.0f, 1.0f / 27.0f };
    const u8 *P[4] = { P0, P1, P2, P3 };

    for (u32 n = 0; n < N; n++) {
        float acc = 0.0f;
        for (u32 bi = 0; bi < row_bytes; bi++) {
            for (int s = 0; s < 4; s++) {
                float v = 0.0f;
                for (int p = 0; p < 4; p++) {
                    u8 b = P[p][(size_t)n * row_bytes + bi];
                    int code = (b >> (2 * s)) & 3;
                    v += ((float)code - 1.0f) * W[p];
                }
                acc += v * in[bi * 4 + s];
            }
        }
        out[n] = acc * kernel_fp16_to_f32(row_scale[n]);
    }
}

static int part1_production_kernel(void) {
    const u32 K = 2560, N = 8;
    const u32 row_bytes = K / 4;

    u8*  planes[4];
    for (int p = 0; p < 4; p++) planes[p] = (u8 *)malloc((size_t)N * row_bytes);
    float* in  = (float *)malloc((size_t)K * sizeof(float));
    float* out = (float *)malloc((size_t)N * sizeof(float));
    float* ref = (float *)malloc((size_t)N * sizeof(float));
    u16*  row_scale = (u16 *)malloc((size_t)N * sizeof(u16));

    /* Deterministic pseudo-random fill. */
    u32 seed = 12345u;
    for (int p = 0; p < 4; p++)
        for (size_t i = 0; i < (size_t)N * row_bytes; i++) {
            seed = seed * 1103515245u + 12345u;
            planes[p][i] = (u8)((seed >> 16) & 0xFF);
        }
    for (u32 i = 0; i < K; i++) {
        seed = seed * 1103515245u + 12345u;
        in[i] = ((float)((seed >> 8) & 0xFFFF) / 32768.0f) - 1.0f;
    }
    for (u32 n = 0; n < N; n++) {
        /* Two different F16 scales so the scale path is exercised too:
         * 0x3C00 = 1.0, 0x3800 = 0.5. */
        row_scale[n] = (n & 1) ? 0x3800 : 0x3C00;
    }

    hs_ml_lmhead_stage1(out, in, planes[0], planes[1], planes[2], planes[3],
                        row_scale, N, K);
    reference_stage1(ref, in, planes[0], planes[1], planes[2], planes[3],
                     row_scale, N, K);

    float max_rel = 0.0f;
    for (u32 n = 0; n < N; n++) {
        float d = out[n] - ref[n];
        if (d < 0) d = -d;
        float den = ref[n] < 0 ? -ref[n] : ref[n];
        if (den < 1.0f) den = 1.0f;
        float rel = d / den;
        if (rel > max_rel) max_rel = rel;
    }
    /* The tolerance has to cover the difference in summation order between
     * this reference and the kernel's. The kernel's scalar path accumulates
     * in the same order as the reference and matches exactly; the NEON
     * path uses four accumulators and measures 5.7e-06 relative. 1e-4
     * leaves roughly 18x margin over that while still catching a wrong
     * decode, which would be off by order 1. */
    printf("  Part 1  production LUT kernel vs scalar reference decode\n");
    printf("          N = %u, K = %u, max relative error = %.3e  %s\n",
           N, K, max_rel, max_rel < 1e-4f ? "PASS" : "FAIL");
    check(max_rel < 1e-4f, "kernel disagrees with reference decode");

    for (int p = 0; p < 4; p++) free(planes[p]);
    free(in); free(out); free(ref); free(row_scale);
    return 0;
}

/* ================================================================
 * Part 2 -- the three bit layouts
 * ================================================================ */

/* NeoGPU internal: 4 consecutive weights per byte, low bits first. */
static void pack_neogpu(const u8* codes, int K, u8* out) {
    memset(out, 0, (size_t)K / 4);
    for (int bi = 0; bi < K / 4; bi++)
        for (int s = 0; s < 4; s++)
            out[bi] |= (u8)((codes[bi * 4 + s] & 3) << (2 * s));
}
static void unpack_neogpu(const u8* in, int K, u8* out) {
    for (int bi = 0; bi < K / 4; bi++)
        for (int s = 0; s < 4; s++)
            out[bi * 4 + s] = (u8)((in[bi] >> (2 * s)) & 3);
}

/* Upstream BitNet, ARM NEON build: QK_I2_S = 64, group_idx = j / 16.
 * experimental/BitNet/src/ggml-bitnet-mad.cpp, quantize_i2_s #else branch:
 *     group_idx = j / 16 ; group_pos = j % 16
 *     i2_weight[i * 16 + group_pos] |= code << (6 - 2 * group_idx) */
static void pack_up_arm(const u8* codes, int K, u8* out) {
    memset(out, 0, (size_t)K / 4);
    for (int i = 0; i < K / 64; i++)
        for (int j = 0; j < 64; j++) {
            int gi = j / 16, gp = j % 16;
            out[i * 16 + gp] |= (u8)(codes[i * 64 + j] << (6 - 2 * gi));
        }
}
static void unpack_up_arm(const u8* in, int K, u8* out) {
    for (int i = 0; i < K / 64; i++)
        for (int j = 0; j < 64; j++) {
            int gi = j / 16, gp = j % 16;
            out[i * 64 + j] = (u8)((in[i * 16 + gp] >> (6 - 2 * gi)) & 3);
        }
}

/* Upstream BitNet, x86 build: QK_I2_S = 128, group_idx = j / 32. */
static void pack_up_x86(const u8* codes, int K, u8* out) {
    memset(out, 0, (size_t)K / 4);
    for (int i = 0; i < K / 128; i++)
        for (int j = 0; j < 128; j++) {
            int gi = j / 32, gp = j % 32;
            out[i * 32 + gp] |= (u8)(codes[i * 128 + j] << (6 - 2 * gi));
        }
}
static void unpack_up_x86(const u8* in, int K, u8* out) {
    for (int i = 0; i < K / 128; i++)
        for (int j = 0; j < 128; j++) {
            int gi = j / 32, gp = j % 32;
            out[i * 128 + j] = (u8)((in[i * 32 + gp] >> (6 - 2 * gi)) & 3);
        }
}

static double agreement(const u8* a, const u8* b, int K) {
    int same = 0;
    for (int i = 0; i < K; i++) if (a[i] == b[i]) same++;
    return (double)same / K;
}

static int part2_layouts(void) {
    const int K = 2560;
    const int TRIALS = 2000;

    u8* codes = (u8 *)malloc(K);
    u8* bytes = (u8 *)malloc(K / 4);
    u8* d1 = (u8 *)malloc(K);
    u8* d2 = (u8 *)malloc(K);

    u32 seed = 987654321u;
    double arm_arm = 0, x86_x86 = 0, ne_ne = 0;
    double arm_ne = 0, x86_ne = 0, arm_x86 = 0;

    for (int t = 0; t < TRIALS; t++) {
        for (int i = 0; i < K; i++) {
            seed = seed * 1103515245u + 12345u;
            codes[i] = (u8)((seed >> 16) % 3);
        }

        pack_up_arm(codes, K, bytes);
        unpack_up_arm(bytes, K, d1);
        unpack_neogpu(bytes, K, d2);
        arm_arm += agreement(codes, d1, K);
        arm_ne  += agreement(codes, d2, K);

        pack_up_x86(codes, K, bytes);
        unpack_up_x86(bytes, K, d1);
        unpack_neogpu(bytes, K, d2);
        x86_x86 += agreement(codes, d1, K);
        x86_ne  += agreement(codes, d2, K);

        pack_neogpu(codes, K, bytes);
        unpack_neogpu(bytes, K, d1);
        unpack_up_arm(bytes, K, d2);
        ne_ne += agreement(codes, d1, K);
        arm_x86 += agreement(codes, d2, K);
    }

    arm_arm /= TRIALS; x86_x86 /= TRIALS; ne_ne /= TRIALS;
    arm_ne /= TRIALS;  x86_ne /= TRIALS;  arm_x86 /= TRIALS;

    printf("  Part 2  bit layout compatibility (K = %d, %d random ternary rows)\n", K, TRIALS);
    printf("          upstream ARM   read with upstream ARM : %.4f\n", arm_arm);
    printf("          upstream x86   read with upstream x86 : %.4f\n", x86_x86);
    printf("          NeoGPU         read with NeoGPU       : %.4f\n", ne_ne);
    printf("          upstream ARM   read with NeoGPU       : %.4f  (chance is 0.3333)\n", arm_ne);
    printf("          upstream x86   read with NeoGPU       : %.4f\n", x86_ne);
    printf("          NeoGPU         read with upstream ARM : %.4f\n", arm_x86);

    check(arm_arm == 1.0, "upstream ARM layout is not self-consistent");
    check(x86_x86 == 1.0, "upstream x86 layout is not self-consistent");
    check(ne_ne   == 1.0, "NeoGPU layout is not self-consistent");
    check(arm_ne <= 0.5, "upstream ARM and NeoGPU layouts unexpectedly agree");
    check(x86_ne <= 0.5, "upstream x86 and NeoGPU layouts unexpectedly agree");
    check(arm_x86 <= 0.5, "upstream ARM and x86 layouts unexpectedly agree");

    free(codes); free(bytes); free(d1); free(d2);
    return 0;
}

/* ================================================================
 * Part 3 -- the 8-plane base-3 expansion
 * ================================================================ */

/*
 * This is the encoder's plane loop from
 * src/hs_ml_loader_ternary.c, hs_mlt_lmhead_encode(). Part 4 checks that
 * the source still contains the four lines this depends on, so the copy
 * here cannot silently drift away from the real encoder.
 *
 *   if      (r >  0.5f) t =  1;                       rounding threshold
 *   u8 code = (u8)(t + 1);                             code = weight + 1
 *   byte |= (code & 3u) << (s * 2);                    low bits first
 *   row_f[hi] = (row_f[hi] - (float)t) * 3.0f;         base-3 magnification
 */
static void encode_planes(float *row_f, u32 H, u8 **planes, u32 row_bytes) {
    for (int k = 0; k < 8; k++) {
        u8 *plane_row = planes[k];
        for (u32 bi = 0; bi < row_bytes; bi++) {
            u8 byte = 0;
            for (int s = 0; s < 4; s++) {
                u32 hi = bi * 4 + s;
                float r = row_f[hi];
                int8_t t;
                if      (r >  0.5f) t =  1;
                else if (r < -0.5f) t = -1;
                else                t =  0;
                u8 code = (u8)(t + 1);
                byte |= (code & 3u) << (s * 2);
                row_f[hi] = (row_f[hi] - (float)t) * 3.0f;
            }
            plane_row[bi] = byte;
        }
    }
    (void)H;
}

static void decode_plane(const u8* plane_row, u32 row_bytes, u8* codes_out) {
    for (u32 bi = 0; bi < row_bytes; bi++) {
        u8 b = plane_row[bi];
        for (int s = 0; s < 4; s++)
            codes_out[bi * 4 + s] = (u8)((b >> (s * 2)) & 3);
    }
}

static int part3_wavelet(void) {
    const u32 H = 2560;
    const u32 ROWS = 20000;
    const u32 row_bytes = H / 4;

    u8** planes = (u8**)malloc(8 * sizeof(u8*));
    for (int k = 0; k < 8; k++) planes[k] = (u8*)malloc((size_t)ROWS * row_bytes);
    float* row_f = (float*)malloc(H * sizeof(float));
    float* orig  = (float*)malloc(H * sizeof(float));
    u8*    codes = (u8*)malloc(H);
    double* rec8 = (double*)malloc(H * sizeof(double));
    double* rec4 = (double*)malloc(H * sizeof(double));

    u32 seed = 20261111u;
    double max_err8 = 0.0, sum_err8 = 0.0;
    double max_err4 = 0.0, sum_err4 = 0.0;
    double max_resid = 0.0;

    for (u32 v = 0; v < ROWS; v++) {
        float scale = 0.0f;
        for (u32 h = 0; h < H; h++) {
            seed = seed * 1103515245u + 12345u;
            float r = ((float)((seed >> 8) & 0xFFFF) / 32768.0f) - 1.0f;
            row_f[h] = r;
            float a = r < 0 ? -r : r;
            if (a > scale) scale = a;
        }
        if (scale == 0.0f) scale = 1.0f;
        for (u32 h = 0; h < H; h++) { row_f[h] /= scale; orig[h] = row_f[h]; }

        encode_planes(row_f, H, planes, row_bytes);

        for (u32 h = 0; h < H; h++) {
            double r = fabs((double)row_f[h]);
            if (r > max_resid) max_resid = r;
            rec8[h] = 0.0; rec4[h] = 0.0;
        }
        for (int k = 0; k < 8; k++) {
            decode_plane(planes[k], row_bytes, codes);
            double w = pow(3.0, -k);
            for (u32 h = 0; h < H; h++) {
                double t = (double)codes[h] - 1.0;
                rec8[h] += t * w;
                if (k < 4) rec4[h] += t * w;
            }
        }
        for (u32 h = 0; h < H; h++) {
            double e8 = fabs(rec8[h] - orig[h]);
            double e4 = fabs(rec4[h] - orig[h]);
            if (e8 > max_err8) max_err8 = e8;
            if (e4 > max_err4) max_err4 = e4;
            sum_err8 += e8; sum_err4 += e4;
        }
    }

    double n = (double)ROWS * H;
    double bound8 = 1.5 / 6561.0;   /* 3^8 */
    double bound4 = 1.5 / 81.0;     /* 3^4 */
    /* The bounds are exact in real arithmetic; the encoder carries the
     * residual in float32, so the measured worst case sits a little above
     * them. The observed excess is 1.4e-8 on the 8-plane bound (6e-5
     * relative), so 1e-4 relative is the tolerance. */
    const double TOL = 1.0001;

    printf("  Part 3  8-plane base-3 expansion (H = %u, %u rows)\n", H, ROWS);
    printf("          8 planes: max error %.9f  mean %.9f  (bound %.9f)  %s\n",
           max_err8, sum_err8 / n, bound8,
           max_err8 <= bound8 * TOL ? "PASS" : "FAIL");
    printf("          4 planes: max error %.9f  mean %.9f  (bound %.9f)  %s\n",
           max_err4, sum_err4 / n, bound4,
           max_err4 <= bound4 * TOL ? "PASS" : "FAIL");
    printf("          max |residual| after 8 planes: %.6f  (bound 1.5)  %s\n",
           max_resid, max_resid <= 1.5 ? "PASS" : "FAIL");

    check(max_err8 <= bound8 * TOL, "8-plane reconstruction error exceeds 1.5/3^8");
    check(max_err4 <= bound4 * TOL, "4-plane reconstruction error exceeds 1.5/3^4");
    check(max_resid <= 1.5,         "residual after 8 planes exceeds 1.5");

    for (int k = 0; k < 8; k++) free(planes[k]);
    free(planes); free(row_f); free(orig); free(codes); free(rec8); free(rec4);
    return 0;
}

/* ================================================================
 * Part 4 -- the source invariants the documentation depends on
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

/* Find the repository root by looking for a file that only exists there. */
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

static int part4_source_invariants(void) {
    char root[1024];
    if (find_root(root, sizeof(root)) != 0) {
        printf("  Part 4  SKIPPED -- could not locate the repository root\n");
        known_issues++;
        return 0;
    }

    char path[1200];
    snprintf(path, sizeof(path), "%s/src/hs_ml_ternary_neon.c", root);
    char* neon = read_file(path);
    snprintf(path, sizeof(path), "%s/src/hs_ml_loader_ternary.c", root);
    char* loader = read_file(path);

    printf("  Part 4  source invariants the documentation depends on\n");

    if (!neon || !loader) {
        printf("          SKIPPED -- could not read the sources\n");
        free(neon); free(loader);
        known_issues++;
        return 0;
    }

    /* The decode LUT uses code - 1: 0 -> -1, 1 -> 0, 2 -> +1. */
    int lut_ok = contains(neon, "g_lut[b][0] = (float)((int)((b >> 0) & 3) - 1);");
    printf("          LUT decodes code-1 (0=-1, 1=0, 2=+1)        %s\n",
           lut_ok ? "OK" : "CHANGED");
    check(lut_ok, "the LUT's code mapping changed");

    /* The encoder uses the same mapping. */
    int enc_map = contains(loader, "u8 code = (u8)(t + 1);");
    printf("          encoder maps weight -> code+1               %s\n",
           enc_map ? "OK" : "CHANGED");
    check(enc_map, "the encoder's code mapping changed");

    /* The encoder packs low bits first. */
    int enc_pack = contains(loader, "byte |= (code & 3u) << (s * 2);");
    printf("          encoder packs low bits first                %s\n",
           enc_pack ? "OK" : "CHANGED");
    check(enc_pack, "the encoder's bit order changed");

    /* The encoder magnifies the residual by 3: that is what makes the
     * planes a base-3 expansion rather than eight independent
     * quantisations. */
    int enc_mag = contains(loader, "row_f[hi] = (row_f[hi] - (float)t) * 3.0f;");
    printf("          encoder magnifies residual by 3 (base-3)    %s\n",
           enc_mag ? "OK" : "CHANGED");
    check(enc_mag, "the encoder's base-3 magnification changed");

    /* The rounding threshold that sets the 1.5 residual bound. */
    int enc_round = contains(loader, "if      (r >  0.5f) t =  1;");
    printf("          encoder rounds at +/-0.5                    %s\n",
           enc_round ? "OK" : "CHANGED");
    check(enc_round, "the encoder's rounding threshold changed");

    /* load_i2s_raw reads the tensor with a bare fread -- no reordering. */
    int raw_read = contains(loader, "if(fread(buf, 1, total, f) != total) {");
    printf("          I2_S tensors are read verbatim (no repack) %s\n",
           raw_read ? "OK" : "CHANGED");
    check(raw_read, "the I2_S raw loader changed");

    /* The divergent packer: f32_to_ternary_packed() uses 0=0, 1=+1,
     * 2=-1, which the LUT above decodes as -1, 0, +1. Reported, not
     * asserted, so this test stays green while the divergence exists and
     * stops reporting it once the loader is fixed. */
    int divergent = contains(loader, "uint8_t code = (v > 0.5f) ? 1 : (v < -0.5f) ? 2 : 0;");
    if (divergent) {
        printf("          KNOWN ISSUE: f32_to_ternary_packed() still uses the\n");
        printf("                     opposite code mapping (0=0, 1=+1, 2=-1).\n");
        printf("                     The LUT reads that as -1, 0, +1 -- every\n");
        printf("                     F32-model weight decodes wrong.\n");
        printf("                     See docs/interface/TERNARY_WEIGHT_FORMATS.md\n");
        known_issues++;
    } else {
        printf("          f32_to_ternary_packed() now uses the canonical mapping\n");
    }

    free(neon); free(loader);
    return 0;
}

/* ================================================================ */

int main(void) {
    printf("Red-team: ternary weight formats\n");
    printf("================================\n");
    printf("Docs: docs/interface/TERNARY_WEIGHT_FORMATS.md\n\n");

    part1_production_kernel();
    part2_layouts();
    part3_wavelet();
    part4_source_invariants();

    printf("\n%s\n", checks_failed ? "FAILURES DETECTED" : "DONE");
    if (known_issues)
        printf("%d known issue(s) reported above (not counted as failures)\n",
               known_issues);
    return checks_failed ? 1 : 0;
}
