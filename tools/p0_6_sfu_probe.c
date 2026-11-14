/* p0_6_sfu_probe: the QPU's special-function unit, measured.
 * (P0-6 plan step 1, docs/audit/P0_6_PLAN.md.)
 *
 * A compute renderer needs 1/x and sqrt(x) per ray. On V3D 4.2 reciprocal,
 * reciprocal square root and sin are add-ALU operations with a destination
 * (the encodings were read from Mesa's compiled code for tests/test_08; a
 * first version of this probe wrote x to magic registers 48..53 and read r4,
 * which does nothing on this hardware). This probe measures, for each:
 *   accuracy  16 lanes x `vectors` inputs loaded from memory, the result
 *             used after `d` other instructions (d = 0..4), stored and
 *             compared with the CPU (correctly rounded 1/x and 1/sqrt(x); libm
 *             sin): exact matches, the largest error in units of the last
 *             place, the largest relative error.
 *   cost      an arithmetic-only loop of 1,000,000 passes with and without
 *             the function call, per d: ns per call.
 * Inputs: uniformly random mantissas with exponents 2^-10..2^10 (sin: the
 * range -8..8), then a row of special values reported one by one.
 * Usage: p0_6_sfu_probe [vectors=20000]   (inside the GPU lock, alone)
 * Output: JSON lines.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "hs_qpu.h"
#include "hs_v3d.h"

#define R(n) HS_QPU_R(n)
#define RF(n) HS_QPU_RF(n)
#define X HS_QPU_NOREG
#define I(v) hs_qpu_imm_int(v)

static void emit(hs_qpu_prog* p, hs_qpu_alu x) { hs_qpu_emit(p, &x); }
static void nop(hs_qpu_prog* p) { emit(p, hs_qpu_alu_nop()); }
static void op(hs_qpu_prog* p, int o, hs_qpu_reg d, hs_qpu_reg a, hs_qpu_reg b) { emit(p, hs_qpu_a(o, d, a, b)); }
static void mov(hs_qpu_prog* p, hs_qpu_reg d, hs_qpu_reg s) { op(p, HS_QPU_A_MOV, d, s, X); }
static void ldunifrf(hs_qpu_prog* p, int rf) { emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDUNIFRF, RF(rf))); }
static void loop_end(hs_qpu_prog* p, int counter_rf, int label) {
    hs_qpu_alu dec = hs_qpu_a(HS_QPU_A_SUB, RF(counter_rf), RF(counter_rf), I(1));
    dec.af = HS_QPU_PF_PUSHZ;
    emit(p, dec);
    hs_qpu_branch(p, HS_QPU_B_NA0, label, 0, 0);
    nop(p), nop(p), nop(p);
}

/* accuracy: uniforms in, out, count. out[i] = f(in[i]) */
static int build_acc(hs_qpu_prog* p, unsigned fn, int d) {
    hs_qpu_prog_rules(p, 0);
    for (int i = 0; i < 3; i++) ldunifrf(p, i);
    op(p, HS_QPU_A_EIDX, R(1), X, X);
    op(p, HS_QPU_A_SHL, R(0), R(1), I(2));
    op(p, HS_QPU_A_ADD, RF(0), RF(0), R(0));
    op(p, HS_QPU_A_ADD, RF(1), RF(1), R(0));
    mov(p, R(2), I(1));
    op(p, HS_QPU_A_SHL, RF(4), R(2), I(6));
    int l = hs_qpu_label_new(p);
    hs_qpu_label_bind(p, l);
    emit(p, hs_qpu_s(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(0), X), HS_QPU_SIG_THRSW, X));
    nop(p), nop(p);
    emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDTMU, RF(5)));
    op(p, (int)fn, RF(6), RF(5), X);
    for (int i = 0; i < d; i++) nop(p);
    mov(p, HS_QPU_TMUD, RF(6));
    mov(p, HS_QPU_TMUA, RF(1));
    op(p, HS_QPU_A_TMUWT, X, X, X);
    op(p, HS_QPU_A_ADD, RF(0), RF(0), RF(4));
    op(p, HS_QPU_A_ADD, RF(1), RF(1), RF(4));
    loop_end(p, 2, l);
    hs_qpu_emit_end(p);
    return hs_qpu_prog_finish(p);
}
/* cost: uniforms count, x. A loop with the call (fn) or a plain move in its place (fn 0); the same d nops either way. */
static int build_cost(hs_qpu_prog* p, unsigned fn, int d) {
    hs_qpu_prog_rules(p, 0);
    ldunifrf(p, 2);
    ldunifrf(p, 5);
    mov(p, RF(7), I(0));
    mov(p, RF(8), I(0));
    int l = hs_qpu_label_new(p);
    hs_qpu_label_bind(p, l);
    if (fn) op(p, (int)fn, RF(8), RF(5), X);
    else mov(p, RF(8), RF(5));
    for (int i = 0; i < d; i++) nop(p);
    op(p, HS_QPU_A_FADD, RF(7), RF(7), RF(8));
    loop_end(p, 2, l);
    hs_qpu_emit_end(p);
    return hs_qpu_prog_finish(p);
}

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
static uint32_t rng = 88172645;
static uint32_t rnd(void) {
    rng ^= rng << 13, rng ^= rng >> 17, rng ^= rng << 5;
    return rng;
}
static float ref(unsigned fn, float x) {
    switch (fn) {
    case HS_QPU_A_RECIP: return 1.0f / x;
    case HS_QPU_A_RSQRT: return (float)(1.0 / sqrt((double)x));
    default: return (float)sin((double)x);
    }
}
static const unsigned FNS[3] = {HS_QPU_A_RECIP, HS_QPU_A_RSQRT, HS_QPU_A_SIN};

static hs_v3d dev;
static hs_v3d_bo code, unif, in, out;
static int run(int (*build)(hs_qpu_prog*, unsigned, int), unsigned fn, int d, double* ms) {
    static uint64_t buf[256];
    hs_qpu_prog p;
    hs_qpu_prog_init(&p, buf, 256);
    int len = build(&p, fn, d);
    uint32_t prog = len > 0 ? hs_v3d_load(&code, 0, buf, (uint32_t)len) : 0;
    if (!prog) return len < 0 ? len : -EINVAL;
    uint32_t h[4] = {code.handle, unif.handle, in.handle, out.handle};
    hs_v3d_dispatch_desc dd = {prog, unif.gpu, 1, h, 4, 0};
    uint64_t t = now_ns();
    int e = hs_v3d_dispatch(&dev, &dd);
    if (!e) e = hs_v3d_bo_wait(&dev, &code, 3000000000ull);
    if (ms) *ms = (now_ns() - t) / 1e6;
    return e;
}

int main(int argc, char** argv) {
    uint32_t vectors = argc > 1 ? (uint32_t)atoi(argv[1]) : 20000, n = vectors * 16;
    if (vectors < 1 || vectors > 1000000) return fprintf(stderr, "vectors: 1..1000000\n"), 2;
    if (hs_v3d_open(&dev, NULL) || hs_v3d_bo_new(&dev, &code, 4096) || hs_v3d_bo_new(&dev, &unif, 4096) ||
        hs_v3d_bo_new(&dev, &in, (size_t)n * 4) || hs_v3d_bo_new(&dev, &out, (size_t)n * 4))
        return fprintf(stderr, "setup failed\n"), 1;
    float* x = (float*)in.map;
    uint32_t* y = (uint32_t*)out.map;
    uint32_t* u = (uint32_t*)unif.map;
    int failures = 0;
    static const float special[16] = {0.0f, -0.0f, 1.0f, -1.0f, 2.0f, 0.5f, 4.0f, 1e-30f, 1e30f, 3.0f, 10.0f, 100.0f, -4.0f, 1e-40f, 0.25f, 9.0f};
    for (unsigned fi = 0; fi < 3; fi++) {
        unsigned fn = FNS[fi];
        const char* name = hs_qpu_add_name((int)fn);
        for (int d = 0; d <= 4; d++) {
            for (uint32_t i = 0; i < n; i++) {
                uint32_t bits = (rnd() & 0x007FFFFF) | ((117 + rnd() % 21) << 23);
                memcpy(&x[i], &bits, 4);
                if (fn == HS_QPU_A_SIN) x[i] = ((float)(rnd() % 16000000) / 1e6f) - 8.0f;
            }
            memset(y, 0xEE, (size_t)n * 4);
            u[0] = in.gpu, u[1] = out.gpu, u[2] = vectors;
            int e = run(build_acc, fn, d, NULL);
            uint64_t exact = 0, within1 = 0;
            double max_ulp = 0, max_rel = 0;
            for (uint32_t i = 0; i < n && !e; i++) {
                float r = ref(fn, x[i]), g;
                memcpy(&g, &y[i], 4);
                uint32_t rb;
                memcpy(&rb, &r, 4);
                int64_t du = (int64_t)(int32_t)y[i] - (int64_t)(int32_t)rb;
                if (du < 0) du = -du;
                exact += du == 0, within1 += du <= 1;
                if ((double)du > max_ulp) max_ulp = (double)du;
                double rel = r != 0 ? fabs(((double)g - (double)r) / (double)r) : fabs((double)g);
                if (rel > max_rel) max_rel = rel;
            }
            printf("{\"sfu\":\"%s\",\"result_used_after_instructions\":%d,\"rc\":%d,\"inputs\":%u,\"exact\":%llu,\"within_1_ulp\":%llu,"
                   "\"max_ulp_error\":%.0f,\"max_relative_error\":%.3g}\n",
                   name, d, e, n, (unsigned long long)exact, (unsigned long long)within1, max_ulp, max_rel);
            if (e) failures++;
        }
        /* special values, read after 4 instructions */
        memcpy(x, special, sizeof special);
        u[0] = in.gpu, u[1] = out.gpu, u[2] = 1;
        if (!run(build_acc, fn, 4, NULL)) {
            printf("{\"sfu_special\":\"%s\",\"values\":[", name);
            for (int i = 0; i < 16; i++) {
                float g;
                memcpy(&g, &y[i], 4);
                printf("%s{\"x\":%g,\"gpu\":\"%g\",\"bits\":\"0x%08x\",\"cpu\":\"%g\"}", i ? "," : "", (double)special[i], (double)g, y[i],
                       (double)ref(fn, special[i]));
            }
            printf("]}\n");
        }
        /* cost */
        for (int d = 0; d <= 4; d++) {
            float xv = 1.7f;
            double with = 1e30, without = 1e30, ms;
            u[0] = 1000000;
            memcpy(&u[1], &xv, 4);
            for (int r = 0; r < 5; r++) {
                if (!run(build_cost, fn, d, &ms) && ms < with) with = ms;
                if (!run(build_cost, 0, d, &ms) && ms < without) without = ms;
            }
            printf("{\"sfu_cost\":\"%s\",\"result_used_after_instructions\":%d,\"loop_ns_with_call\":%.2f,\"loop_ns_without\":%.2f,\"ns_per_call\":%.2f}\n",
                   name, d, with, without, with - without);
        }
        fflush(stdout);
    }
    return failures != 0;
}
