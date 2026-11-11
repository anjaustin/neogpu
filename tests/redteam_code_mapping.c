/*
 * Red-team: the ternary code-mapping boundary.
 *
 * This repository has TWO 2-bit code mappings:
 *
 *   canonical  code = weight + 1   -> 0 = -1, 1 = 0, 2 = +1
 *               used by the decode LUT (src/hs_ml_ternary_neon.c), the
 *               8-plane encoder, the GPU shaders, and upstream's
 *               quantize_i2_s.
 *
 *   routing    code = 0, +1, -1    -> 0 = 0, 1 = +1, 2 = -1
 *               used by hs_ml_gemm_ternary_mt() (src/hs_ml_ternary_mt.c),
 *               and therefore by the two packers that feed it:
 *               f32_to_ternary_packed() in the loader, for F32-stored
 *               models, and alloc_ternary() in hs_ml_infer.c, for the
 *               random synthetic models.
 *
 * The invariant this test pins is NOT "one mapping is right". It is:
 * each packer must be read by a decoder with the matching mapping.
 *
 * The failure this guards against is subtle: comparing a packer against
 * the wrong decoder looks exactly like a bug and is not one. Reading
 * f32_to_ternary_packed()'s output with the LUT produces garbage --
 * but the LUT never reads it.
 *
 * Build and run (from the repository root):
 *     make test-code-mapping
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

static void check(int ok, const char *what) {
    if (!ok) {
        printf("          FAILED CHECK: %s\n", what);
        checks_failed++;
    }
}

/* ================================================================
 * The two decode rules
 * ================================================================ */

/* The canonical mapping, as src/hs_ml_ternary_neon.c's build_lut does:
 * lut[b][lane] = (float)((b >> (lane*2)) & 3) - 1. */
static int decode_canonical(u8 code) {
    return (int)code - 1;
}

/* The routing mapping, as src/hs_ml_ternary_mt.c does:
 *   if (bits == 1) sum += a;
 *   else if (bits == 2) sum -= a;
 * so code 1 is +1, code 2 is -1 and code 0 contributes nothing. */
static int decode_routing(u8 code) {
    if (code == 1) return 1;
    if (code == 2) return -1;
    return 0;
}

/* ================================================================
 * The packers, taken verbatim from the repository source
 * ================================================================ */

/*
 * f32_to_ternary_packed(), src/hs_ml_loader_ternary.c.
 *   uint8_t code = (v > 0.5f) ? 1 : (v < -0.5f) ? 2 : 0;
 */
static void pack_f32_routing(const float* in, int N, u8* out) {
    int N4 = N / 4;
    for (int i = 0; i < N4; i++) {
        u8 byte = 0;
        for (int b = 0; b < 4; b++) {
            float v = in[i * 4 + b];
            uint8_t code = (v > 0.5f) ? 1 : (v < -0.5f) ? 2 : 0;
            byte |= (code << (b * 2));
        }
        out[i] = byte;
    }
    if (N % 4 != 0) {
        u8 byte = 0;
        for (int b = 0; b < N % 4; b++) {
            float v = in[N4 * 4 + b];
            uint8_t code = (v > 0.5f) ? 1 : (v < -0.5f) ? 2 : 0;
            byte |= (code << (b * 2));
        }
        out[N4] = byte;
    }
}

/*
 * alloc_ternary()'s inner packing, src/hs_ml_infer.c.
 *   int r = rand_r(rng) % 4;
 *   u8 v = (r == 0) ? 1 : (r == 1) ? 2 : 0;
 *   byte |= (v << (b * 2));
 */
static void pack_random_routing(u32* rng, int N, u8* out) {
    int N4 = N / 4;
    for (int i = 0; i < N4; i++) {
        u8 byte = 0;
        for (int b = 0; b < 4; b++) {
            int r = (int)(rand_r(rng) % 4);
            u8 v = (r == 0) ? 1 : (r == 1) ? 2 : 0;
            byte |= (v << (b * 2));
        }
        out[i] = byte;
    }
}

/* ================================================================
 * The packers under test
 * ================================================================ */

/* 1. The F32 packer feeds the routing decoder. */
static int part1_f32_packer(void) {
    const int N = 4096;
    float* src = (float*)malloc((size_t)N * sizeof(float));
    u8*   packed = (u8*)malloc((size_t)N / 4);

    /* Values well clear of the +/-0.5 threshold, plus exact zeros. */
    for (int i = 0; i < N; i++) {
        switch (i % 5) {
            case 0: src[i] =  1.0f; break;
            case 1: src[i] = -1.0f; break;
            case 2: src[i] =  0.7f; break;
            case 3: src[i] = -0.7f; break;
            default: src[i] = 0.0f; break;
        }
    }

    pack_f32_routing(src, N, packed);

    int routing_ok = 0, routing_bad = 0, canonical_ok = 0, canonical_bad = 0;
    for (int i = 0; i < N; i++) {
        u8 code = (u8)((packed[i / 4] >> (2 * (i % 4))) & 3);
        float want = (src[i] > 0.5f) ? 1.0f : (src[i] < -0.5f) ? -1.0f : 0.0f;

        if ((float)decode_routing(code) == want) routing_ok++; else routing_bad++;
        if ((float)decode_canonical(code) == want) canonical_ok++; else canonical_bad++;
    }

    printf("  Part 1  f32_to_ternary_packed() -> %s decoder\n",
           "routing");
    printf("          routing decoder   : %d correct, %d wrong\n", routing_ok, routing_bad);
    printf("          canonical decoder : %d correct, %d wrong  (never reads it)\n",
           canonical_ok, canonical_bad);

    check(routing_bad == 0, "the F32 packer does not round-trip through the routing decoder");
    check(canonical_bad > 0, "the two code mappings are not actually different");

    free(src); free(packed);
    return 0;
}

/* 2. The random packer feeds the routing decoder, and has the
 *    distribution its comment claims. */
static int part2_random_packer(void) {
    const int N = 400000;
    u8* packed = (u8*)malloc((size_t)N / 4);
    u32 rng = 12345u;

    pack_random_routing(&rng, N, packed);

    long n_zero = 0, n_pos = 0, n_neg = 0;
    for (int i = 0; i < N; i++) {
        u8 code = (u8)((packed[i / 4] >> (2 * (i % 4))) & 3);
        int w = decode_routing(code);
        if (w == 0) n_zero++;
        else if (w > 0) n_pos++;
        else n_neg++;
    }
    double total = (double)N;
    printf("  Part 2  alloc_ternary() -> routing decoder\n");
    printf("          decoded distribution: %.1f%% zero, %.1f%% +1, %.1f%% -1\n",
           100.0 * n_zero / total, 100.0 * n_pos / total, 100.0 * n_neg / total);
    printf("          its comment claims:  50%% zero, 25%% +1, 25%% -1\n");

    /* 0.5 and 0.25 with N = 400000 samples: the tolerance is far wider
     * than the sampling noise and far tighter than a wrong mapping. */
    check(fabs(100.0 * n_zero / total - 50.0) < 1.0, "the random packer is not ~50% zero");
    check(fabs(100.0 * n_pos  / total - 25.0) < 1.0, "the random packer is not ~25% +1");
    check(fabs(100.0 * n_neg  / total - 25.0) < 1.0, "the random packer is not ~25% -1");

    free(packed);
    return 0;
}

/* 3. The 8-plane encoder feeds the canonical decoder. */
static void encode_planes_canonical(float* row_f, u8** planes, u32 row_bytes) {
    for (int k = 0; k < 8; k++) {
        u8* plane_row = planes[k];
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
}

static int part3_encoder(void) {
    const u32 H = 2560;
    const u32 row_bytes = H / 4;
    u8* planes[8];
    for (int k = 0; k < 8; k++) planes[k] = (u8*)malloc(row_bytes);
    float* row_f = (float*)malloc(H * sizeof(float));

    int canonical_ok = 0, canonical_bad = 0, routing_agree = 0;
    u32 seed = 777u;
    for (int trial = 0; trial < 64; trial++) {
        for (u32 h = 0; h < H; h++) {
            seed = seed * 1103515245u + 12345u;
            row_f[h] = ((float)((seed >> 8) & 0xFFFF) / 32768.0f) - 1.0f;
        }
        /* the encoder's first plane is what the packer sees */
        encode_planes_canonical(row_f, planes, row_bytes);

        for (u32 h = 0; h < H; h++) {
            u8 code = (u8)((planes[0][h / 4] >> (2 * (h % 4))) & 3);
            float want = (float)decode_canonical(code);
            /* the canonical decoder must agree with itself */
            if (want == (float)((int)code - 1)) canonical_ok++; else canonical_bad++;
            /* the routing decoder must NOT agree: that is the point */
            if ((float)decode_routing(code) == want) routing_agree++;
        }
    }

    printf("  Part 3  the 8-plane encoder -> canonical decoder\n");
    printf("          canonical decoder agrees with the LUT's rule: %d of %d\n",
           canonical_ok, canonical_ok + canonical_bad);
    printf("          routing decoder would misread it: %d of %d\n",
           canonical_ok + canonical_bad - routing_agree, canonical_ok + canonical_bad);

    check(canonical_bad == 0, "the canonical decode rule is not code-1");
    check(routing_agree == 0, "the two decode rules agree on some weight");

    for (int k = 0; k < 8; k++) free(planes[k]);
    free(row_f);
    return 0;
}

/* ================================================================
 * Part 4 -- the source pins that keep the rules from drifting
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

    snprintf(path, sizeof(path), "%s/src/hs_ml_ternary_neon.c", root);
    char* neon = read_file(path);
    snprintf(path, sizeof(path), "%s/src/hs_ml_ternary_mt.c", root);
    char* mt = read_file(path);
    snprintf(path, sizeof(path), "%s/src/hs_ml_loader_ternary.c", root);
    char* loader = read_file(path);
    snprintf(path, sizeof(path), "%s/src/hs_ml_infer.c", root);
    char* infer = read_file(path);
    snprintf(path, sizeof(path), "%s/src/hs_ml_routing.c", root);
    char* routing = read_file(path);

    if (!neon || !mt || !loader || !infer || !routing) {
        printf("          SKIPPED -- could not read the sources\n");
        free(neon); free(mt); free(loader); free(infer); free(routing);
        return;
    }

    /* The canonical decoder: code - 1. */
    int lut = contains(neon, "g_lut[b][0] = (float)((int)((b >> 0) & 3) - 1);");
    printf("          LUT decodes code-1 (canonical)               %s\n", lut ? "OK" : "CHANGED");
    check(lut, "the canonical decode rule changed");

    /* The routing decoder: code 1 is +1, code 2 is -1. */
    int routing_dec = contains(mt, "if (bits == 1) sum += a;") &&
                      contains(mt, "else if (bits == 2) sum -= a;");
    printf("          routing GEMM decodes code1=+1, code2=-1      %s\n",
           routing_dec ? "OK" : "CHANGED");
    check(routing_dec, "the routing decode rule changed");

    /* The packers that feed it. */
    int pack_f32 = contains(loader, "uint8_t code = (v > 0.5f) ? 1 : (v < -0.5f) ? 2 : 0;");
    int pack_rand = contains(infer, "u8 v = (r == 0) ? 1 : (r == 1) ? 2 : 0;");
    printf("          F32 packer uses the routing mapping          %s\n",
           pack_f32 ? "OK" : "CHANGED");
    printf("          random packer uses the routing mapping        %s\n",
           pack_rand ? "OK" : "CHANGED");
    check(pack_f32,  "the F32 packer's code mapping changed");
    check(pack_rand, "the random packer's code mapping changed");

    /* The path boundary: F32-packed weights reach the routing GEMM. */
    int to_routing = contains(infer, "route.format = HS_ROUTE_TERNARY_2BIT;") &&
                     contains(infer, "hs_ml_route_mt(out, in, &route, 1, threads);") &&
                     contains(routing, "case HS_ROUTE_TERNARY_2BIT:") &&
                     contains(routing, "hs_ml_gemm_ternary_mt(");
    printf("          F32/random weights route to that GEMM        %s\n",
           to_routing ? "OK" : "CHANGED");
    check(to_routing, "the route from the packed weights to the routing GEMM changed");

    free(neon); free(mt); free(loader); free(infer); free(routing);
}

/* ================================================================ */

int main(void) {
    printf("Red-team: ternary code-mapping boundary\n");
    printf("========================================\n");
    printf("Docs: docs/interface/TERNARY_WEIGHT_FORMATS.md section 1\n\n");

    part1_f32_packer();
    part2_random_packer();
    part3_encoder();

    char root[1024];
    if (find_root(root, sizeof(root)) != 0) {
        printf("\n  Part 4 SKIPPED -- could not locate the repository root\n");
    } else {
        part4_source_pins(root);
    }

    printf("\n%s\n", checks_failed ? "FAILURES DETECTED" : "DONE");
    return checks_failed ? 1 : 0;
}
