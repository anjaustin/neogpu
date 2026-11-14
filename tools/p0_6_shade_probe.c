/* p0_6_shade_probe: the whole test_08 scene as a hand-assembled QPU compute
 * kernel: primary ray, shading, shadow rays and one reflection bounce,
 * against a CPU reference of the same arithmetic.
 * (P0-6 plan step 4, docs/audit/P0_6_PLAN.md.)
 *
 * Per 16 pixels: a ray, then PASS (trace the six spheres and the ground from
 * an origin along a direction; shade the nearest hit, with a shadow test
 * against the six spheres), then the reflected ray and PASS again, then the
 * mix and the 8-bit pack. Straight-line: every lane runs everything and the
 * results are selected at the end. No work is skipped yet (plan step 5).
 *
 * Constants come from a second uniform stream (unifa), rewound once per
 * pass of the inner loop; they ride on other instructions as signals, to r5
 * or straight into a register (measured: tools/p0_6_unifa_probe; at least 3
 * other instructions after the unifa write). Rules kept here: nothing lives
 * in r0..r3 or r5 across the thread switch; special-function results go to
 * a register-file entry.
 *
 * Where it differs from the GLSL: the shadow test avoids the square root
 * (ts > 0.001 is tested as h >= 0, b + eps < 0 and h < (b + eps)^2), the
 * sphere normal is (p - c) / r, and the horizon test on rd.y is left out
 * (no pixel row has |rd.y| < 1e-4 at this size). The reference does the
 * same; how far both are from Mesa's image is a separate measurement.
 *
 * Offscreen. Usage: p0_6_shade_probe [batches=8] [frames=1] [shadows=1] [reflect=1] [ppm_dir]
 *   Start a changed kernel with frames=1: a kernel that hangs costs a GPU
 *   reset per job. (inside the GPU lock, performance governor).
 * Output: JSON lines; exit 1 if a frame has mismatches away from edges or
 * more than 0.5 % in all.
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

enum { W = 800, H = 480, VECS = W / 16, NSPH = 6 };

/* register file */
enum {
    R_OUT = 0, R_ROWS, R_VEC, R_UX0, R_UX, R_DUX16, R_UY, R_DUY, R_64, R_T, R_KBASE, R_UYSQ, R_K1, R_KMERGE, R_SPARE,
    R_SKY = 15,              /* 3: the row's sky colour, float */
    R_EPS = 18, R_NBIG, R_KROW, R_BIGP, R_ZERO, R_REFLK1, R_MISS1,
    R_D = 25,                /* 3: ray direction */
    R_BY = 28, R_NBEST,
    R_H = 30,                /* 6: attributes of the nearest sphere: cx, cz, 1/r, albedo r g b */
    R_TA = 36,               /* 6: the sphere being tested; outside the trace: 3 for the reflected colour, 1 for its miss mark */
    R_P = 42,                /* 3: hit point (during the trace: R_P, R_P+1 hold oy^2 - r^2 for the two radii) */
    R_N = 45,                /* 3: normal */
    R_HK = 48, R_MISS,
    R_COL = 50,              /* 3 */
    R_O = 53,                /* 3: ray origin; in the shade: the shadow ray's origin */
    R_SBY = 56, R_SKS, R_SKC, R_SMAX, R_E, R_S, R_TMP, R_NDR,
};

/* ---- the scene: every constant the kernel and the reference use ---- */
static struct {
    float cx[NSPH], cz[NSPH], invr[NSPH], alb[NSPH][3];
    float r2s, r2c, tiny, roy, roz, m17, gy, g35, g20, k35, k25, missthr, L[3], L01[3], negeps, q25, d85, amb, s7, one, c02, f255, half;
    float gnum, e_lit, e_sh, c75, vecs_bits; /* for the plain-ground path; vecs_bits: the integer 50 */
    float skya[3], skyb[3], c289;
    float ux0, dux, dux16, uy0, duy, duyb, v0, dv, dvb, eps, nbig, bigp, zero;
    uint32_t rows_per_batch;
} S;

static void scene_at(float t, uint32_t batches) {
    const float aspect = (float)W / (float)H, PI = 3.14159265f;
    S.dux = 2.0f * aspect / (float)W, S.ux0 = (1.0f / (float)W - 1.0f) * aspect, S.dux16 = 16.0f * S.dux;
    /* Rows are dealt out in turn: batch b renders rows b, b + batches, ... so
     * that cheap rows (sky) and dear ones are shared between the batches. */
    S.duyb = -2.0f / (float)H, S.uy0 = 1.0f - 1.0f / (float)H; /* duyb: one row, times the batch number at the start */
    S.rows_per_batch = H / batches;
    S.duy = (float)batches * S.duyb;                          /* from one of a batch's rows to its next */
    S.dv = -1.0f / (float)H, S.v0 = 1.0f - 0.5f / (float)H, S.dvb = (float)S.rows_per_batch * S.dv;
    S.eps = 0.001f, S.negeps = -0.001f, S.nbig = -1e9f, S.bigp = 1e9f, S.tiny = 1e-30f, S.zero = 0.0f, S.one = 1.0f, S.half = 0.5f, S.missthr = 1e8f;
    S.roy = 1.1f, S.roz = 4.3f, S.c289 = 1.7f * 1.7f, S.m17 = -1.7f, S.gy = -0.95f, S.f255 = 255.0f;
    S.r2s = 0.5f * 0.5f, S.r2c = 0.85f * 0.85f;
    S.g35 = 0.35f, S.g20 = 0.20f, S.k35 = 0.35f, S.k25 = 0.25f, S.q25 = 0.25f, S.d85 = 0.85f, S.amb = 0.22f, S.s7 = 0.7f, S.c02 = 0.02f;
    float len = sqrtf(0.5f * 0.5f + 1.0f + 0.4f * 0.4f);
    S.L[0] = 0.5f / len, S.L[1] = 1.0f / len, S.L[2] = 0.4f / len;
    for (int c = 0; c < 3; c++) S.L01[c] = S.L[c] * 0.01f;
    for (int i = 0; i < NSPH; i++) {
        if (i < 5) {
            float a = (2.0f * PI * (float)i) / 5.0f + t * 0.6f, fi = (float)i / 5.0f;
            S.cx[i] = cosf(a) * 1.7f, S.cz[i] = sinf(a) * 1.7f, S.invr[i] = 1.0f / 0.5f;
            S.alb[i][0] = 0.5f + 0.5f * cosf(2.0f * PI * (fi + 0.0f));
            S.alb[i][1] = 0.5f + 0.5f * cosf(2.0f * PI * (fi + 0.33f));
            S.alb[i][2] = 0.5f + 0.5f * cosf(2.0f * PI * (fi + 0.67f));
        } else {
            S.cx[i] = 0, S.cz[i] = 0, S.invr[i] = 1.0f / 0.85f;
            S.alb[i][0] = 0.85f, S.alb[i][1] = 0.85f, S.alb[i][2] = 0.9f;
        }
    }
    S.gnum = S.gy - S.roy;
    {   /* the ground's shading terms that do not depend on the pixel: n = (0, 1, 0) */
        float diff = S.L[1];
        S.e_lit = diff * S.d85 + S.amb;
        float dsh = diff * S.q25;
        S.e_sh = dsh * S.d85 + S.amb;
        S.c75 = S.one - S.k25;
        uint32_t n = VECS;
        memcpy(&S.vecs_bits, &n, 4);
    }
    S.skya[0] = 0.55f, S.skya[1] = 0.22f, S.skya[2] = 0.35f;
    S.skyb[0] = 0.08f, S.skyb[1] = 0.10f, S.skyb[2] = 0.28f;
}

/* ---- emitter: an instruction can carry the next constant of the stream ---- */
static const float* ksrc[2048]; /* where each word of the stream comes from */
static int kn, kpend_rf;
static const float* kpend;
static int n_instr_vec, opt_shadows = 1, opt_reflect = 1, opt_pad, opt_pad_kind;
static int opt_cull = 7; /* bit 0: rows that are all sky skip the pass; bit 1: no second shade when all 16 reflected rays
                           * miss; bit 2: regions of ground nothing can reach take a short path (classes from the CPU) */
static double opt_margin = 1e-3; /* slack in the region tests, radians or scene units; negative breaks them (a test of the test) */
static int k_vec, k_merge, k_ground, n_instr_ground; /* stream positions: start of the per-pass block; where the two paths around the second shade meet */

static void emit(hs_qpu_prog* p, hs_qpu_alu x) {
    if (kpend) {
        x = kpend_rf < 0 ? hs_qpu_s(x, HS_QPU_SIG_LDUNIFA, X) : hs_qpu_s(x, HS_QPU_SIG_LDUNIFARF, RF(kpend_rf));
        ksrc[kn++] = kpend;
        kpend = NULL;
    }
    hs_qpu_emit(p, &x);
}
/* the next instruction emitted also loads *src: into r5 (K), or into rf (KRF) */
static void K(const float* src) { kpend = src, kpend_rf = -1; }
static void KRF(const float* src, int rf) { kpend = src, kpend_rf = rf; }
static void nop(hs_qpu_prog* p) { emit(p, hs_qpu_alu_nop()); }
static void A(hs_qpu_prog* p, int o, hs_qpu_reg d, hs_qpu_reg a, hs_qpu_reg b) { emit(p, hs_qpu_a(o, d, a, b)); }
static void MOV(hs_qpu_prog* p, hs_qpu_reg d, hs_qpu_reg s) { A(p, HS_QPU_A_MOV, d, s, X); }
static void M(hs_qpu_prog* p, hs_qpu_reg d, hs_qpu_reg a, hs_qpu_reg b) { emit(p, hs_qpu_m(hs_qpu_alu_nop(), HS_QPU_M_FMUL, d, a, b)); }
/* add-ALU operation and a float multiply in one instruction */
static void AM(hs_qpu_prog* p, int o, hs_qpu_reg d, hs_qpu_reg a, hs_qpu_reg b, hs_qpu_reg md, hs_qpu_reg ma, hs_qpu_reg mb) {
    emit(p, hs_qpu_m(hs_qpu_a(o, d, a, b), HS_QPU_M_FMUL, md, ma, mb));
}
static void AF(hs_qpu_prog* p, int o, hs_qpu_reg d, hs_qpu_reg a, hs_qpu_reg b, int pf) {
    hs_qpu_alu s = hs_qpu_a(o, d, a, b);
    s.af = (uint8_t)pf;
    emit(p, s);
}
static void AC(hs_qpu_prog* p, int cond, hs_qpu_reg d, hs_qpu_reg s) {
    hs_qpu_alu x = hs_qpu_a(HS_QPU_A_MOV, d, s, X);
    x.ac = (uint8_t)cond;
    emit(p, x);
}
/* two conditional moves in one instruction */
static void AC2(hs_qpu_prog* p, int cond, hs_qpu_reg d1, hs_qpu_reg s1, hs_qpu_reg d2, hs_qpu_reg s2) {
    hs_qpu_alu x = hs_qpu_a(HS_QPU_A_MOV, d1, s1, X);
    x.ac = (uint8_t)cond;
    x = hs_qpu_m(x, HS_QPU_M_MOV, d2, s2, X);
    x.mc = (uint8_t)cond;
    emit(p, x);
}
static void ldunif(hs_qpu_prog* p, hs_qpu_reg d) { emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDUNIFRF, d)); }
static void br(hs_qpu_prog* p, int cond, int label) {
    hs_qpu_branch(p, cond, label, 0, 0);
    nop(p), nop(p), nop(p);
}
#define FADD HS_QPU_A_FADD
#define FSUB HS_QPU_A_FSUB
#define FMIN HS_QPU_A_FMIN
#define FMAX HS_QPU_A_FMAX
#define PUSHN HS_QPU_PF_PUSHN
#define IFA HS_QPU_COND_IFA
#define IFNA HS_QPU_COND_IFNA
#define IFB HS_QPU_COND_IFB

/* Trace from R_O along R_D. Leaves: R_P, R_N, R_H+3.. (albedo), R_HK
 * (reflectivity), R_MISS (negative: nothing was hit). */
static void emit_trace(hs_qpu_prog* p) {
    M(p, RF(R_BY), RF(R_O + 1), RF(R_D + 1));
    K(&S.r2s);
    M(p, R(0), RF(R_O + 1), RF(R_O + 1));
    K(&S.r2c);
    A(p, FSUB, RF(R_P), R(0), R(5));       /* oy^2 - r^2, small spheres */
    A(p, FSUB, RF(R_P + 1), R(0), R(5));   /* the core */
    KRF(&S.cx[0], R_TA);
    MOV(p, RF(R_NBEST), RF(R_NBIG));
    for (int i = 0; i < NSPH; i++) {
        KRF(&S.cz[i], R_TA + 1);
        A(p, FSUB, R(0), RF(R_O), RF(R_TA));              /* ocx */
        KRF(&S.invr[i], R_TA + 2);
        A(p, FSUB, R(1), RF(R_O + 2), RF(R_TA + 1));      /* ocz */
        KRF(&S.alb[i][0], R_TA + 3);
        M(p, R(2), R(0), RF(R_D));
        KRF(&S.alb[i][1], R_TA + 4);
        M(p, R(3), R(1), RF(R_D + 2));
        KRF(&S.alb[i][2], R_TA + 5);
        AM(p, FADD, R(2), R(2), R(3), R(0), R(0), R(0));
        AM(p, FADD, R(2), R(2), RF(R_BY), R(1), R(1), R(1)); /* b */
        A(p, FADD, R(0), R(0), R(1));
        AM(p, FADD, R(0), R(0), RF(i < 5 ? R_P : R_P + 1), R(1), R(2), R(2)); /* cc; b*b */
        K(&S.tiny);
        AF(p, FSUB, R(1), R(1), R(0), PUSHN);             /* h; flag: h < 0 */
        A(p, FMAX, R(3), R(1), R(5));
        A(p, HS_QPU_A_RSQRT, RF(R_T), R(3), X);
        nop(p);
        M(p, R(1), R(1), RF(R_T));                        /* sqrt(h) */
        A(p, FADD, R(1), R(2), R(1));                     /* u = b + sqrt(h) = -t */
        AC(p, IFA, R(1), RF(R_NBIG));
        AF(p, FADD, R(3), R(1), RF(R_EPS), PUSHN);        /* flag: t > eps */
        AC(p, IFNA, R(1), RF(R_NBIG));
        AF(p, FSUB, R(3), RF(R_NBEST), R(1), PUSHN);      /* flag: nearer */
        AC2(p, IFA, RF(R_NBEST), R(1), RF(R_H), RF(R_TA));
        AC2(p, IFA, RF(R_H + 1), RF(R_TA + 1), RF(R_H + 2), RF(R_TA + 2));
        AC2(p, IFA, RF(R_H + 3), RF(R_TA + 3), RF(R_H + 4), RF(R_TA + 4));
        AC(p, IFA, RF(R_H + 5), RF(R_TA + 5));
        if (i + 1 < NSPH) {
            KRF(&S.cx[i + 1], R_TA);
            nop(p);
        }
    }
    /* the ground */
    K(&S.gy);
    A(p, HS_QPU_A_RECIP, RF(R_T), RF(R_D + 1), X);
    A(p, FSUB, R(0), R(5), RF(R_O + 1));
    M(p, R(0), R(0), RF(R_T));                            /* tg */
    AF(p, FSUB, R(1), R(0), RF(R_EPS), PUSHN);
    AC(p, IFA, R(0), RF(R_BIGP));
    M(p, R(1), RF(R_D), R(0));
    A(p, FADD, R(1), R(1), RF(R_O));
    M(p, R(2), RF(R_D + 2), R(0));
    A(p, FADD, R(2), R(2), RF(R_O + 2));
    A(p, HS_QPU_A_FFLOOR, R(1), R(1), X);
    A(p, HS_QPU_A_FFLOOR, R(2), R(2), X);
    A(p, FADD, R(1), R(1), R(2));
    A(p, HS_QPU_A_FTOIZ, R(1), R(1), X);
    K(&S.g35);
    AF(p, FADD, R(3), R(0), RF(R_NBEST), PUSHN);          /* flag: the ground is nearer */
    AF(p, HS_QPU_A_BAND, R(1), R(1), I(1), HS_QPU_PF_PUSHZ); /* A: even square; B: ground nearer */
    K(&S.g20);
    MOV(p, R(2), R(5));
    AC(p, IFA, R(2), R(5));                               /* the square's grey */
    A(p, FSUB, R(3), RF(R_ZERO), RF(R_NBEST));            /* t of the nearest sphere */
    AC(p, IFB, R(3), R(0));                               /* t */
    AC2(p, IFB, RF(R_H + 3), R(2), RF(R_H + 4), R(2));
    AC(p, IFB, RF(R_H + 5), R(2));
    M(p, R(0), RF(R_D), R(3));
    A(p, FADD, RF(R_P), R(0), RF(R_O));
    M(p, R(0), RF(R_D + 1), R(3));
    A(p, FADD, RF(R_P + 1), R(0), RF(R_O + 1));
    M(p, R(0), RF(R_D + 2), R(3));
    A(p, FADD, RF(R_P + 2), R(0), RF(R_O + 2));
    A(p, FSUB, R(0), RF(R_P), RF(R_H));
    M(p, RF(R_N), R(0), RF(R_H + 2));
    M(p, RF(R_N + 1), RF(R_P + 1), RF(R_H + 2));
    A(p, FSUB, R(0), RF(R_P + 2), RF(R_H + 1));
    M(p, RF(R_N + 2), R(0), RF(R_H + 2));
    K(&S.one);
    AC2(p, IFB, RF(R_N), RF(R_ZERO), RF(R_N + 2), RF(R_ZERO));
    K(&S.k35);
    AC(p, IFB, RF(R_N + 1), R(5));
    K(&S.k25);
    MOV(p, RF(R_HK), R(5));
    K(&S.missthr);
    AC(p, IFB, RF(R_HK), R(5));
    A(p, FSUB, RF(R_MISS), R(5), R(3));                   /* negative: no hit */
}

/* The shadow test for the point in R_P: R_SMAX > 0 afterwards if any sphere is between it and the light. */
static void emit_shadow(hs_qpu_prog* p, int with_loop) {
    K(&S.L01[0]);
    nop(p);
    K(&S.L01[1]);
    A(p, FADD, RF(R_O), RF(R_P), R(5));
    K(&S.L01[2]);
    A(p, FADD, RF(R_O + 1), RF(R_P + 1), R(5));
    K(&S.L[1]);
    A(p, FADD, RF(R_O + 2), RF(R_P + 2), R(5));
    M(p, RF(R_SBY), RF(R_O + 1), R(5));
    K(&S.r2s);
    M(p, R(0), RF(R_O + 1), RF(R_O + 1));
    K(&S.r2c);
    A(p, FSUB, RF(R_SKS), R(0), R(5));
    A(p, FSUB, RF(R_SKC), R(0), R(5));
    if (with_loop) K(&S.cx[0]);
    MOV(p, RF(R_SMAX), RF(R_NBIG));
    for (int i = 0; with_loop && i < NSPH; i++) {
        K(&S.cz[i]);
        A(p, FSUB, R(0), RF(R_O), R(5));
        K(&S.L[0]);
        A(p, FSUB, R(1), RF(R_O + 2), R(5));
        K(&S.L[2]);
        M(p, R(2), R(0), R(5));
        M(p, R(3), R(1), R(5));
        AM(p, FADD, R(2), R(2), R(3), R(0), R(0), R(0));
        AM(p, FADD, R(2), R(2), RF(R_SBY), R(1), R(1), R(1)); /* b */
        A(p, FADD, R(0), R(0), R(1));
        K(&S.negeps);
        AM(p, FADD, R(0), R(0), RF(i < 5 ? R_SKS : R_SKC), R(1), R(2), R(2));
        A(p, FSUB, R(1), R(1), R(0));                     /* h */
        A(p, FSUB, R(3), R(5), R(2));                     /* -(b + eps) */
        M(p, R(0), R(3), R(3));
        A(p, FSUB, R(0), R(0), R(1));                     /* (b + eps)^2 - h */
        A(p, FMIN, R(0), R(0), R(1));
        A(p, FMIN, R(0), R(0), R(3));
        if (i + 1 < NSPH) K(&S.cx[i + 1]);
        A(p, FMAX, RF(R_SMAX), RF(R_SMAX), R(0));         /* > 0: this sphere shadows the point */
    }
}

/* Shade the hit in R_P, R_N, albedo R_H+3.., seen along R_D. Result in rf out..out+2; n.d left in R_NDR. */
static void emit_shade(hs_qpu_prog* p, int out) {
    emit_shadow(p, opt_shadows);
    /* diffuse */
    K(&S.L[0]);
    nop(p);
    K(&S.L[1]);
    M(p, R(0), RF(R_N), R(5));
    K(&S.L[2]);
    M(p, R(1), RF(R_N + 1), R(5));
    M(p, R(2), RF(R_N + 2), R(5));
    A(p, FADD, R(0), R(0), R(1));
    A(p, FADD, R(0), R(0), R(2));
    A(p, FMAX, R(0), R(0), RF(R_ZERO));
    K(&S.q25);
    AF(p, FSUB, R(3), RF(R_ZERO), RF(R_SMAX), PUSHN);     /* flag A: in shadow; kept until the specular */
    K(&S.d85);
    M(p, R(1), R(0), R(5));
    AC(p, IFA, R(0), R(1));
    K(&S.amb);
    M(p, R(0), R(0), R(5));
    K(&S.L[0]);
    A(p, FADD, RF(R_E), R(0), R(5));
    /* specular: h = normalize(L - d) */
    K(&S.L[1]);
    A(p, FSUB, R(0), R(5), RF(R_D));
    K(&S.L[2]);
    A(p, FSUB, R(1), R(5), RF(R_D + 1));
    A(p, FSUB, R(2), R(5), RF(R_D + 2));
    M(p, R(3), R(0), R(0));
    M(p, RF(R_TMP), R(1), R(1));
    A(p, FADD, R(3), R(3), RF(R_TMP));
    M(p, RF(R_TMP), R(2), R(2));
    A(p, FADD, R(3), R(3), RF(R_TMP));
    A(p, HS_QPU_A_RSQRT, RF(R_T), R(3), X);
    M(p, R(0), R(0), RF(R_N));
    M(p, R(1), R(1), RF(R_N + 1));
    M(p, R(2), R(2), RF(R_N + 2));
    A(p, FADD, R(0), R(0), R(1));
    A(p, FADD, R(0), R(0), R(2));
    M(p, R(0), R(0), RF(R_T));
    A(p, FMAX, R(0), R(0), RF(R_ZERO));
    for (int i = 0; i < 5; i++) M(p, R(0), R(0), R(0));   /* ^32 */
    K(&S.s7);
    AC(p, IFA, R(0), RF(R_ZERO));
    M(p, RF(R_S), R(0), R(5));
    /* rim, and n.d for the reflection */
    M(p, R(0), RF(R_N), RF(R_D));
    M(p, R(1), RF(R_N + 1), RF(R_D + 1));
    M(p, R(2), RF(R_N + 2), RF(R_D + 2));
    A(p, FADD, R(0), R(0), R(1));
    A(p, FADD, RF(R_NDR), R(0), R(2));
    A(p, FSUB, R(1), RF(R_ZERO), RF(R_NDR));
    K(&S.one);
    A(p, FMAX, R(1), R(1), RF(R_ZERO));
    K(&S.q25);
    A(p, FSUB, R(1), R(5), R(1));
    M(p, R(1), R(1), R(1));
    M(p, R(1), R(1), R(5));
    A(p, FADD, RF(R_S), RF(R_S), R(1));
    for (int c = 0; c < 3; c++) {
        M(p, R(0), RF(R_H + 3 + c), RF(R_E));
        A(p, FADD, RF(out + c), R(0), RF(R_S));
    }
}

static int build(hs_qpu_prog* p) {
    hs_qpu_prog_rules(p, 0);
    kn = 0, kpend = NULL;
    A(p, HS_QPU_A_BAND, RF(R_T), RF(0), I(15)); /* the hardware's batch number */
    ldunif(p, RF(R_OUT));
    ldunif(p, RF(R_SKY));     /* bytes per row, for now */
    ldunif(p, RF(R_ROWS));
    ldunif(p, RF(R_UX0));
    ldunif(p, RF(R_UX));      /* dux, for now */
    ldunif(p, RF(R_DUX16));
    ldunif(p, RF(R_UY));
    ldunif(p, RF(R_DUY));
    ldunif(p, RF(R_NBEST));   /* duy per batch, for now */
    ldunif(p, RF(R_KMERGE));
    ldunif(p, RF(R_SPARE));
    ldunif(p, RF(R_BY));      /* unused */
    ldunif(p, RF(R_KROW));
    ldunif(p, RF(R_KBASE));
    ldunif(p, RF(R_K1));
    ldunif(p, RF(R_EPS));
    ldunif(p, RF(R_NBIG));
    ldunif(p, RF(R_BIGP));
    ldunif(p, RF(R_ZERO));
    A(p, HS_QPU_A_EIDX, R(1), X, X);
    A(p, HS_QPU_A_SHL, R(0), R(1), I(2));
    A(p, HS_QPU_A_ADD, RF(R_OUT), RF(R_OUT), R(0));
    emit(p, hs_qpu_m(hs_qpu_alu_nop(), HS_QPU_M_UMUL24, R(2), RF(R_T), RF(R_SKY)));
    A(p, HS_QPU_A_ADD, RF(R_OUT), RF(R_OUT), R(2));
    A(p, HS_QPU_A_ITOF, R(1), R(1), X);
    M(p, R(1), R(1), RF(R_UX));
    A(p, FADD, RF(R_UX0), RF(R_UX0), R(1));
    A(p, HS_QPU_A_ITOF, R(2), RF(R_T), X);
    M(p, R(3), R(2), RF(R_NBEST));
    A(p, FADD, RF(R_UY), RF(R_UY), R(3));
    MOV(p, R(2), I(1));
    A(p, HS_QPU_A_SHL, RF(R_64), R(2), I(6));

    int l_row = hs_qpu_label_new(p), l_vec = hs_qpu_label_new(p);
    /* ---- per row: constants from the start of the stream ---- */
    hs_qpu_label_bind(p, l_row);
    MOV(p, HS_QPU_UNIFA, RF(R_KROW));
    M(p, R(0), RF(R_UY), RF(R_UY));
    MOV(p, RF(R_UX), RF(R_UX0));
    nop(p);
    KRF(&S.vecs_bits, R_VEC);
    nop(p);
    K(&S.c289);
    nop(p);
    K(&S.half);
    A(p, FADD, RF(R_UYSQ), R(0), R(5));
    K(&S.half);
    M(p, R(2), RF(R_UY), R(5));
    K(&S.one);
    A(p, FADD, R(2), R(2), R(5));                         /* v = uy / 2 + 1 / 2 */
    K(&S.skya[0]);
    A(p, FSUB, R(3), R(5), R(2));                         /* 1 - v */
    for (int c = 0; c < 3; c++) {
        K(&S.skyb[c]);
        M(p, R(0), R(5), R(3));
        if (c < 2) K(&S.skya[c + 1]);
        M(p, R(1), R(5), R(2));
        A(p, FADD, RF(R_SKY + c), R(0), R(1));
    }
    int l_skyrow = hs_qpu_label_new(p), l_rowend = hs_qpu_label_new(p);
    if (opt_cull & 1) {
        /* A ray that goes up from the camera (above everything in the scene)
         * hits nothing: a row with uy > 0 is sky. r5 = 255 for that path. */
        K(&S.f255);
        AF(p, FSUB, R(0), RF(R_ZERO), RF(R_UY), PUSHN);
        br(p, HS_QPU_B_A0, l_skyrow);
    }

    /* ---- per 16 pixels ---- */
    k_vec = kn;
    hs_qpu_label_bind(p, l_vec);
    int i0 = (int)p->n;
    emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_THRSW, X)); /* r0..r3 and r5 are dead here */
    nop(p), nop(p);
    int l_ground = hs_qpu_label_new(p);
    if (opt_cull & 4) {
        emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDUNIF, X));   /* this region's class, from the first stream */
        AF(p, HS_QPU_A_MOV, R(0), R(5), X, HS_QPU_PF_PUSHZ);
        br(p, HS_QPU_B_NA0, l_ground);                              /* 1 or 2: plain ground */
    }
    MOV(p, HS_QPU_UNIFA, RF(R_KBASE));
    M(p, R(0), RF(R_UX), RF(R_UX));
    A(p, FADD, R(0), R(0), RF(R_UYSQ));
    A(p, HS_QPU_A_RSQRT, RF(R_T), R(0), X);
    KRF(&S.roy, R_O + 1);
    M(p, RF(R_D), RF(R_UX), RF(R_T));
    KRF(&S.roz, R_O + 2);
    M(p, RF(R_D + 1), RF(R_UY), RF(R_T));
    K(&S.m17);
    MOV(p, RF(R_O), RF(R_ZERO));
    M(p, RF(R_D + 2), R(5), RF(R_T));

    emit_trace(p);
    emit_shade(p, R_COL);
    if (opt_reflect) {
        AC2(p, HS_QPU_COND_NONE, RF(R_REFLK1), RF(R_HK), RF(R_MISS1), RF(R_MISS));
        A(p, FADD, R(3), RF(R_NDR), RF(R_NDR));
        for (int c = 0; c < 3; c++) {
            M(p, R(0), R(3), RF(R_N + c));
            if (c == 2) K(&S.c02);
            A(p, FSUB, RF(R_D + c), RF(R_D + c), R(0));    /* reflected direction */
        }
        for (int c = 0; c < 3; c++) {
            M(p, R(0), RF(R_N + c), R(5));
            A(p, FADD, RF(R_O + c), RF(R_P + c), R(0));    /* its origin */
        }
        emit_trace(p);
        int l_noshade = hs_qpu_label_new(p), l_merge = hs_qpu_label_new(p);
        if (opt_cull & 2) {
            AF(p, HS_QPU_A_MOV, R(0), RF(R_MISS), X, PUSHN);
            br(p, HS_QPU_B_ALLA, l_noshade);               /* all 16 reflected rays hit nothing */
        }
        MOV(p, RF(R_TA + 3), RF(R_MISS));                  /* the trace's temporaries are free now */
        emit_shade(p, R_TA);
        /* the reflected colour is the sky where the second ray hit nothing */
        AF(p, HS_QPU_A_MOV, R(0), RF(R_TA + 3), X, PUSHN);
        AC2(p, IFA, RF(R_TA), RF(R_SKY), RF(R_TA + 1), RF(R_SKY + 1));
        AC(p, IFA, RF(R_TA + 2), RF(R_SKY + 2));
        if (opt_cull & 2) {
            br(p, HS_QPU_B_ALWAYS, l_merge);
            /* the path around the shade read none of its constants: move the stream to where the paths meet */
            hs_qpu_label_bind(p, l_noshade);
            MOV(p, HS_QPU_UNIFA, RF(R_KMERGE));
            AC2(p, HS_QPU_COND_NONE, RF(R_TA), RF(R_SKY), RF(R_TA + 1), RF(R_SKY + 1));
            MOV(p, RF(R_TA + 2), RF(R_SKY + 2));
            nop(p);
            hs_qpu_label_bind(p, l_merge);
        }
        k_merge = kn;
        K(&S.one);
        nop(p);
        A(p, FSUB, R(3), R(5), RF(R_REFLK1));
        for (int c = 0; c < 3; c++) {
            M(p, R(0), RF(R_COL + c), R(3));
            M(p, R(1), RF(R_TA + c), RF(R_REFLK1));
            A(p, FADD, RF(R_COL + c), R(0), R(1));
        }
        AF(p, HS_QPU_A_MOV, R(0), RF(R_MISS1), X, PUSHN);
    } else {
        AF(p, HS_QPU_A_MOV, R(0), RF(R_MISS), X, PUSHN);
    }
    /* timing experiments: a longer pass that changes no result. Kinds: 0 nops;
     * 1 nops that each read a constant; 2..6 special-function calls on an
     * ordinary value, a negative one, infinity, a NaN, and recip of zero. */
    if (opt_pad_kind >= 4) {
        M(p, RF(R_E), RF(R_BIGP), RF(R_BIGP));                         /* inf */
        if (opt_pad_kind == 5) M(p, RF(R_E), RF(R_E), RF(R_ZERO));     /* NaN */
    }
    for (int i = 0; i < opt_pad; i++) {
        if (opt_pad_kind == 1) K(&S.one);
        if (opt_pad_kind == 2) A(p, HS_QPU_A_RSQRT, RF(R_TMP), RF(R_UYSQ), X);
        else if (opt_pad_kind == 3) A(p, HS_QPU_A_RSQRT, RF(R_TMP), RF(R_NBIG), X);
        else if (opt_pad_kind == 4 || opt_pad_kind == 5) A(p, HS_QPU_A_RSQRT, RF(R_TMP), RF(R_E), X);
        else if (opt_pad_kind == 6) A(p, HS_QPU_A_RECIP, RF(R_TMP), RF(R_ZERO), X);
        else nop(p);
    }
    /* the sky where the primary ray hit nothing; then 8 bits per channel */
    AC2(p, IFA, RF(R_COL), RF(R_SKY), RF(R_COL + 1), RF(R_SKY + 1));
    K(&S.f255);
    AC(p, IFA, RF(R_COL + 2), RF(R_SKY + 2));
    for (int c = 0; c < 3; c++) {
        M(p, R(0), RF(R_COL + c), R(5));
        A(p, FMIN, R(0), R(0), R(5));
        A(p, HS_QPU_A_FTOIN, R(0), R(0), X);
        if (c == 0) {
            MOV(p, R(2), R(0));
        } else {
            A(p, HS_QPU_A_SHL, R(2), R(2), I(8));
            A(p, HS_QPU_A_BOR, R(2), R(2), R(0));
        }
    }
    MOV(p, HS_QPU_TMUD, R(2));
    emit(p, hs_qpu_m(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(R_OUT), X), HS_QPU_M_ADD, RF(R_OUT), RF(R_OUT), RF(R_64)));
    A(p, HS_QPU_A_TMUWT, X, X, X);
    A(p, FADD, RF(R_UX), RF(R_UX), RF(R_DUX16));
    AF(p, HS_QPU_A_SUB, RF(R_VEC), RF(R_VEC), I(1), HS_QPU_PF_PUSHZ);
    br(p, HS_QPU_B_NA0, l_vec);
    n_instr_vec = (int)p->n - i0;
    if (opt_cull & 4) {
        br(p, HS_QPU_B_ALWAYS, l_rowend);
        /* ---- 16 pixels of ground that no sphere, shadow (class 1) or reflection can reach.
         * The same operations as the long path performs for such pixels, minus those whose
         * result is then known: n = (0, 1, 0), no sphere hit, the reflected ray hits nothing. */
        enum { G_CLASS = R_TA, G_GREY = R_N, G_SPEC7, G_RIM, G_ELIT = R_HK, G_ESH, G_C75 = R_COL, G_C25, G_C255, G_SLIT = R_E };
        k_ground = kn;
        hs_qpu_label_bind(p, l_ground);
        int g0 = (int)p->n;
        MOV(p, RF(G_CLASS), R(5));
        MOV(p, HS_QPU_UNIFA, RF(R_K1));
        M(p, R(0), RF(R_UX), RF(R_UX));
        A(p, FADD, R(0), R(0), RF(R_UYSQ));
        A(p, HS_QPU_A_RSQRT, RF(R_T), R(0), X);
        KRF(&S.e_lit, G_ELIT);
        M(p, RF(R_D), RF(R_UX), RF(R_T));
        KRF(&S.e_sh, G_ESH);
        M(p, RF(R_D + 1), RF(R_UY), RF(R_T));
        K(&S.m17);
        nop(p);
        K(&S.gnum);
        M(p, RF(R_D + 2), R(5), RF(R_T));
        A(p, HS_QPU_A_RECIP, RF(R_T), RF(R_D + 1), X);
        KRF(&S.c75, G_C75);
        nop(p);
        KRF(&S.k25, G_C25);
        M(p, R(0), R(5), RF(R_T));                             /* t to the ground */
        KRF(&S.f255, G_C255);
        M(p, RF(R_P), RF(R_D), R(0));
        K(&S.roz);
        M(p, R(2), RF(R_D + 2), R(0));
        K(&S.roy);
        A(p, FADD, RF(R_P + 2), R(2), R(5));
        M(p, R(3), RF(R_D + 1), R(0));
        A(p, FADD, RF(R_P + 1), R(3), R(5));
        A(p, HS_QPU_A_FFLOOR, R(1), RF(R_P), X);
        A(p, HS_QPU_A_FFLOOR, R(2), RF(R_P + 2), X);
        A(p, FADD, R(1), R(1), R(2));
        K(&S.g35);
        A(p, HS_QPU_A_FTOIZ, R(1), R(1), X);
        AF(p, HS_QPU_A_BAND, R(1), R(1), I(1), HS_QPU_PF_PUSHZ);
        K(&S.g20);
        MOV(p, RF(G_GREY), R(5));
        K(&S.L[0]);
        AC(p, IFA, RF(G_GREY), R(5));
        /* specular: n.h is h.y / |h| */
        K(&S.L[1]);
        A(p, FSUB, R(0), R(5), RF(R_D));
        K(&S.L[2]);
        A(p, FSUB, R(1), R(5), RF(R_D + 1));
        A(p, FSUB, R(2), R(5), RF(R_D + 2));
        M(p, R(3), R(0), R(0));
        M(p, R(0), R(1), R(1));
        A(p, FADD, R(3), R(3), R(0));
        M(p, R(0), R(2), R(2));
        A(p, FADD, R(3), R(3), R(0));
        A(p, HS_QPU_A_RSQRT, RF(R_T), R(3), X);
        nop(p);
        M(p, R(0), R(1), RF(R_T));
        A(p, FMAX, R(0), R(0), RF(R_ZERO));
        for (int i = 0; i < 4; i++) M(p, R(0), R(0), R(0));
        K(&S.s7);
        M(p, R(0), R(0), R(0));
        M(p, RF(G_SPEC7), R(0), R(5));
        /* rim: n.d is d.y */
        A(p, FSUB, R(0), RF(R_ZERO), RF(R_D + 1));
        K(&S.one);
        A(p, FMAX, R(0), R(0), RF(R_ZERO));
        K(&S.q25);
        A(p, FSUB, R(0), R(5), R(0));
        M(p, R(0), R(0), R(0));
        M(p, RF(G_RIM), R(0), R(5));
        A(p, FADD, RF(G_SLIT), RF(G_SPEC7), RF(G_RIM));
        /* class 2: a sphere may be between these points and the light */
        int l_gdone = hs_qpu_label_new(p);
        MOV(p, RF(R_SMAX), RF(R_NBIG));
        AF(p, HS_QPU_A_SUB, R(0), RF(G_CLASS), I(1), HS_QPU_PF_PUSHZ);
        br(p, HS_QPU_B_A0, l_gdone);
        emit_shadow(p, 1);
        hs_qpu_label_bind(p, l_gdone);
        AF(p, FSUB, R(3), RF(R_ZERO), RF(R_SMAX), PUSHN);    /* flag: in shadow */
        MOV(p, R(0), RF(G_ELIT));
        AC(p, IFA, R(0), RF(G_ESH));
        MOV(p, R(1), RF(G_SLIT));
        AC(p, IFA, R(1), RF(G_RIM));
        M(p, R(2), RF(G_GREY), R(0));
        A(p, FADD, R(2), R(2), R(1));                         /* the ground's colour, the same in every channel */
        M(p, R(3), R(2), RF(G_C75));
        for (int c = 0; c < 3; c++) {                          /* + a quarter of the sky, then 8 bits */
            M(p, R(0), RF(R_SKY + c), RF(G_C25));
            A(p, FADD, R(0), R(3), R(0));
            M(p, R(0), R(0), RF(G_C255));
            A(p, FMIN, R(0), R(0), RF(G_C255));
            A(p, HS_QPU_A_FTOIN, R(0), R(0), X);
            if (c == 0) {
                MOV(p, R(2), R(0));
            } else {
                A(p, HS_QPU_A_SHL, R(2), R(2), I(8));
                A(p, HS_QPU_A_BOR, R(2), R(2), R(0));
            }
        }
        MOV(p, HS_QPU_TMUD, R(2));
        emit(p, hs_qpu_m(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(R_OUT), X), HS_QPU_M_ADD, RF(R_OUT), RF(R_OUT), RF(R_64)));
        A(p, HS_QPU_A_TMUWT, X, X, X);
        A(p, FADD, RF(R_UX), RF(R_UX), RF(R_DUX16));
        AF(p, HS_QPU_A_SUB, RF(R_VEC), RF(R_VEC), I(1), HS_QPU_PF_PUSHZ);
        br(p, HS_QPU_B_NA0, l_vec);
        n_instr_ground = (int)p->n - g0;
    }
    if (opt_cull & 1) {
        br(p, HS_QPU_B_ALWAYS, l_rowend);
        /* ---- a row of sky: one colour, 16 pixels per store ---- */
        hs_qpu_label_bind(p, l_skyrow);
        for (int c = 0; c < 3; c++) {
            M(p, R(0), RF(R_SKY + c), R(5));
            A(p, FMIN, R(0), R(0), R(5));
            A(p, HS_QPU_A_FTOIN, R(0), R(0), X);
            if (c == 0) {
                MOV(p, R(2), R(0));
            } else {
                A(p, HS_QPU_A_SHL, R(2), R(2), I(8));
                A(p, HS_QPU_A_BOR, R(2), R(2), R(0));
            }
        }
        MOV(p, RF(R_COL), R(2));
        int l_sv = hs_qpu_label_new(p);
        hs_qpu_label_bind(p, l_sv);
        MOV(p, HS_QPU_TMUD, RF(R_COL));
        emit(p, hs_qpu_m(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(R_OUT), X), HS_QPU_M_ADD, RF(R_OUT), RF(R_OUT), RF(R_64)));
        A(p, HS_QPU_A_TMUWT, X, X, X);
        AF(p, HS_QPU_A_SUB, RF(R_VEC), RF(R_VEC), I(1), HS_QPU_PF_PUSHZ);
        br(p, HS_QPU_B_NA0, l_sv);
    }
    /* ---- end of row ---- */
    hs_qpu_label_bind(p, l_rowend);
    A(p, FADD, RF(R_UY), RF(R_UY), RF(R_DUY));
    A(p, HS_QPU_A_ADD, RF(R_OUT), RF(R_OUT), RF(R_SPARE)); /* over the other batches' rows */
    AF(p, HS_QPU_A_SUB, RF(R_ROWS), RF(R_ROWS), I(1), HS_QPU_PF_PUSHZ);
    br(p, HS_QPU_B_NA0, l_row);
    hs_qpu_emit_end(p);
    return kpend ? -100 : hs_qpu_prog_finish(p);
}

/* ---- the CPU reference: the kernel's operations, one by one ---- */
static float rsq(float x) { return (float)(1.0 / sqrt((double)x)); }
static float rcp(float x) { return 1.0f / x; }
static float fmx(float a, float b) { return a > b ? a : b; }
static float fmn(float a, float b) { return a < b ? a : b; }
typedef struct {
    float p[3], n[3], alb[3], hk, miss;
    int kind; /* 0 miss, 1..6 sphere, 7 ground */
} hit;
static void ref_trace(const float o[3], const float d[3], hit* h) {
    float by = o[1] * d[1], oy2 = o[1] * o[1], ks = oy2 - S.r2s, kc = oy2 - S.r2c, nbest = S.nbig;
    float hcx = 0, hcz = 0, hir = 0;
    h->alb[0] = h->alb[1] = h->alb[2] = 0, h->kind = 0;
    for (int i = 0; i < NSPH; i++) {
        float ocx = o[0] - S.cx[i], ocz = o[2] - S.cz[i];
        float m2 = ocx * d[0], m3 = ocz * d[2];
        float q0 = ocx * ocx, b = (m2 + m3) + by, q1 = ocz * ocz;
        float cc = (q0 + q1) + (i < 5 ? ks : kc), bb = b * b;
        float hh = bb - cc;
        float u = b + hh * rsq(fmx(hh, S.tiny));
        if (signbit(hh)) u = S.nbig;
        if (!signbit(u + S.eps)) u = S.nbig;
        if (signbit(nbest - u)) {
            nbest = u, hcx = S.cx[i], hcz = S.cz[i], hir = S.invr[i], h->kind = 1 + i;
            for (int c = 0; c < 3; c++) h->alb[c] = S.alb[i][c];
        }
    }
    float tg = (S.gy - o[1]) * rcp(d[1]);
    if (signbit(tg - S.eps)) tg = S.bigp;
    int near = signbit(tg + nbest);
    float t = S.zero - nbest;
    if (near) {
        float gx = d[0] * tg + o[0], gz = d[2] * tg + o[2];
        float f = floorf(gx) + floorf(gz);
        float grey = ((int32_t)f & 1) == 0 ? S.g20 : S.g35;
        t = tg, h->alb[0] = h->alb[1] = h->alb[2] = grey, h->kind = 7;
    }
    for (int c = 0; c < 3; c++) h->p[c] = d[c] * t + o[c];
    h->n[0] = (h->p[0] - hcx) * hir, h->n[1] = h->p[1] * hir, h->n[2] = (h->p[2] - hcz) * hir;
    if (near) h->n[0] = 0, h->n[1] = S.one, h->n[2] = 0;
    h->hk = near ? S.k25 : S.k35;
    h->miss = S.missthr - t;
}
static float ref_shade(const hit* h, const float d[3], float out[3]) {
    float so[3];
    for (int c = 0; c < 3; c++) so[c] = h->p[c] + S.L01[c];
    float sby = so[1] * S.L[1], t0 = so[1] * so[1], sks = t0 - S.r2s, skc = t0 - S.r2c, smax = S.nbig;
    for (int i = 0; opt_shadows && i < NSPH; i++) {
        float ocx = so[0] - S.cx[i], ocz = so[2] - S.cz[i];
        float m2 = ocx * S.L[0], m3 = ocz * S.L[2];
        float q0 = ocx * ocx, b = (m2 + m3) + sby, q1 = ocz * ocz;
        float cc = (q0 + q1) + (i < 5 ? sks : skc), bb = b * b;
        float hh = bb - cc, w2 = S.negeps - b;
        float w3 = w2 * w2 - hh;
        float m = fmn(fmn(w3, hh), w2);
        smax = fmx(smax, m);
    }
    int insh = signbit(S.zero - smax);
    float diff = fmx((h->n[0] * S.L[0] + h->n[1] * S.L[1]) + h->n[2] * S.L[2], S.zero);
    if (insh) diff = diff * S.q25;
    float e = diff * S.d85 + S.amb;
    float hx = S.L[0] - d[0], hy = S.L[1] - d[1], hz = S.L[2] - d[2];
    float inv = rsq((hx * hx + hy * hy) + hz * hz);
    float sp = fmx(((hx * h->n[0] + hy * h->n[1]) + hz * h->n[2]) * inv, S.zero);
    for (int i = 0; i < 5; i++) sp = sp * sp;
    if (insh) sp = S.zero;
    float s = sp * S.s7;
    float ndr = (h->n[0] * d[0] + h->n[1] * d[1]) + h->n[2] * d[2];
    float r1 = S.one - fmx(S.zero - ndr, S.zero);
    r1 = r1 * r1;
    s = s + r1 * S.q25;
    for (int c = 0; c < 3; c++) out[c] = h->alb[c] * e + s;
    return ndr;
}
static void reference(uint32_t batches, uint32_t* img, uint8_t* kind) {
    for (uint32_t b = 0; b < batches; b++) {
        float uy = S.uy0 + (float)b * S.duyb;
        for (uint32_t row = 0; row < S.rows_per_batch; row++) {
            uint32_t y = b + row * batches;
            float uysq = uy * uy + S.c289, v = uy * S.half + S.half, omv = S.one - v, sky[3];
            for (int c = 0; c < 3; c++) sky[c] = S.skya[c] * omv + S.skyb[c] * v;
            float ux[16];
            for (int l = 0; l < 16; l++) ux[l] = S.ux0 + (float)l * S.dux;
            for (int vec = 0; vec < VECS; vec++)
                for (int l = 0; l < 16; l++) {
                    float inv = rsq(ux[l] * ux[l] + uysq);
                    float d[3] = {ux[l] * inv, uy * inv, S.m17 * inv}, o[3] = {S.zero, S.roy, S.roz}, col[3];
                    hit h1, h2;
                    ref_trace(o, d, &h1);
                    int k = h1.kind;
                    if (signbit(h1.miss)) {
                        for (int c = 0; c < 3; c++) col[c] = sky[c];
                        k = 0;
                    } else {
                        float ndr = ref_shade(&h1, d, col);
                        if (opt_reflect) {
                            float k2 = ndr + ndr, rd[3], ro[3], rc[3];
                            for (int c = 0; c < 3; c++) rd[c] = d[c] - k2 * h1.n[c];
                            for (int c = 0; c < 3; c++) ro[c] = h1.p[c] + h1.n[c] * S.c02;
                            ref_trace(ro, rd, &h2);
                            if (signbit(h2.miss)) {
                                for (int c = 0; c < 3; c++) rc[c] = sky[c];
                            } else {
                                ref_shade(&h2, rd, rc);
                                k += 8 * h2.kind;
                            }
                            float omk = S.one - h1.hk;
                            for (int c = 0; c < 3; c++) col[c] = col[c] * omk + rc[c] * h1.hk;
                        }
                    }
                    uint32_t px = 0;
                    for (int c = 0; c < 3; c++) px = px << 8 | (uint32_t)lrintf(fmn(col[c] * S.f255, S.f255));
                    size_t i = (size_t)y * W + (size_t)vec * 16 + (size_t)l;
                    img[i] = px, kind[i] = (uint8_t)k;
                    ux[l] = ux[l] + S.dux16;
                }
            uy = uy + S.duy;
        }
    }
}

/* ---- which regions of ground can nothing reach? (CPU, per frame) ----
 * A region is 16 pixels wide and `batches` rows high (one row per batch).
 * Class 0: the long path. 1: ground that no sphere covers, shadows or is
 * reflected in. 2: as 1, but a sphere may shadow it. The tests are
 * conservative bounds: a cone around the region's rays, a sphere around the
 * patch of ground they reach, a cone from the camera's mirror image for the
 * reflected rays. A wrong class shows as wrong pixels against the reference. */
typedef struct { /* what does not change from frame to frame: the camera is fixed */
    float axis[3], ca, sa;   /* cone of the region's rays from the camera: axis, cos and sin of (half angle + margin) */
    float pc[3], rp;         /* sphere around the patch of ground they reach (radius + margin) */
    float maxis[3], cm, sm;  /* cone of the reflected rays, from the camera's mirror image */
} region;
static region* regions;
static uint32_t n_regions, regions_batches;
static void regions_build(uint32_t batches) {
    const double ro[3] = {0, S.roy, S.roz}, mirror[3] = {0, 2.0 * S.gy - S.roy, S.roz};
    const double duy1 = -2.0 / H, uy0 = 1.0 - 1.0 / H;
    regions = realloc(regions, sizeof(region) * (size_t)S.rows_per_batch * VECS);
    n_regions = 0, regions_batches = batches;
    for (uint32_t r = 0; r < S.rows_per_batch; r++) {
        uint32_t y0 = r * batches, y1 = y0 + batches - 1;
        if (uy0 + y0 * duy1 > 0) continue; /* a sky row: the kernel reads no class for it */
        for (int v = 0; v < VECS; v++) {
            double d[4][3], g[4][3], axis[3] = {0, 0, 0}, pc[3] = {0, 0, 0}, cosh = 1;
            for (int k = 0; k < 4; k++) {
                double ux = S.ux0 + (double)(16 * v + (k & 1 ? 15 : 0)) * S.dux, uy = uy0 + (double)(k & 2 ? y1 : y0) * duy1;
                double len = sqrt(ux * ux + uy * uy + 1.7 * 1.7), t;
                d[k][0] = ux / len, d[k][1] = uy / len, d[k][2] = -1.7 / len;
                t = (S.gy - ro[1]) / d[k][1];
                for (int c = 0; c < 3; c++) g[k][c] = ro[c] + d[k][c] * t, axis[c] += d[k][c], pc[c] += g[k][c] / 4;
            }
            double al = sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
            for (int c = 0; c < 3; c++) axis[c] /= al;
            double max2 = 0, m2axis[3], ml = 0, cosm = 1;
            for (int c = 0; c < 3; c++) m2axis[c] = pc[c] - mirror[c], ml += m2axis[c] * m2axis[c];
            ml = sqrt(ml);
            for (int k = 0; k < 4; k++) {
                double dot = 0, dd = 0, gm = 0, gl = 0;
                for (int c = 0; c < 3; c++) {
                    dot += axis[c] * d[k][c], dd += (g[k][c] - pc[c]) * (g[k][c] - pc[c]);
                    gm += (g[k][c] - mirror[c]) * m2axis[c] / ml, gl += (g[k][c] - mirror[c]) * (g[k][c] - mirror[c]);
                }
                if (dot < cosh) cosh = dot;
                if (dd > max2) max2 = dd;
                if (gm / sqrt(gl) < cosm) cosm = gm / sqrt(gl);
            }
            region* q = &regions[n_regions++];
            double a = acos(cosh) + opt_margin, am = acos(cosm) + opt_margin;
            for (int c = 0; c < 3; c++) q->axis[c] = (float)axis[c], q->pc[c] = (float)pc[c], q->maxis[c] = (float)(m2axis[c] / ml);
            q->ca = (float)cos(a), q->sa = (float)sin(a), q->cm = (float)cos(am), q->sm = (float)sin(am);
            q->rp = (float)(sqrt(max2) + opt_margin);
        }
    }
}
/* Per frame: each region against the six spheres. A sphere of angular radius
 * b around direction w is inside a cone of half angle a around the axis when
 * cos(angle) >= cos(a + b); the comparison is done on the cosines. */
static void classify(uint32_t batches, uint32_t* cls, long count[3]) {
    if (!regions || regions_batches != batches) regions_build(batches);
    const float ro[3] = {0, S.roy, S.roz}, mirror[3] = {0, 2.0f * S.gy - S.roy, S.roz};
    float w[NSPH][3], wm[NSPH][3], cb[NSPH], sb[NSPH], cbm[NSPH], sbm[NSPH], rad[NSPH], c3[NSPH][3];
    for (int i = 0; i < NSPH; i++) {
        float dist = 0, dm = 0;
        c3[i][0] = S.cx[i], c3[i][1] = 0, c3[i][2] = S.cz[i], rad[i] = 1.0f / S.invr[i];
        for (int c = 0; c < 3; c++) {
            w[i][c] = c3[i][c] - ro[c], wm[i][c] = c3[i][c] - mirror[c];
            dist += w[i][c] * w[i][c], dm += wm[i][c] * wm[i][c];
        }
        dist = sqrtf(dist), dm = sqrtf(dm);
        for (int c = 0; c < 3; c++) w[i][c] /= dist, wm[i][c] /= dm;
        sb[i] = rad[i] / dist, cb[i] = sqrtf(1.0f - sb[i] * sb[i]);
        sbm[i] = rad[i] / dm, cbm[i] = sqrtf(1.0f - sbm[i] * sbm[i]);
    }
    count[0] = count[1] = count[2] = 0;
    for (uint32_t n = 0; n < n_regions; n++) {
        const region* q = &regions[n];
        int seen = 0, shadowed = 0, reflected = 0;
        for (int i = 0; i < NSPH; i++) {
            float da = q->axis[0] * w[i][0] + q->axis[1] * w[i][1] + q->axis[2] * w[i][2];
            float dma = q->maxis[0] * wm[i][0] + q->maxis[1] * wm[i][1] + q->maxis[2] * wm[i][2];
            seen |= da >= q->ca * cb[i] - q->sa * sb[i];
            reflected |= dma >= q->cm * cbm[i] - q->sm * sbm[i];
            float px = c3[i][0] - q->pc[0], py = c3[i][1] - q->pc[1], pz = c3[i][2] - q->pc[2];
            float sL = px * S.L[0] + py * S.L[1] + pz * S.L[2], reach = rad[i] + q->rp;
            shadowed |= (px * px + py * py + pz * pz) - sL * sL <= reach * reach && sL > -reach;
        }
        uint32_t c = seen || reflected ? 0 : shadowed ? 2 : 1;
        cls[n] = c, count[c]++;
    }
}

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
static void write_ppm(const char* dir, const char* name, const uint32_t* px) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s.ppm", dir, name);
    FILE* f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    for (int i = 0; i < W * H; i++) {
        unsigned char c[3] = {(unsigned char)(px[i] >> 16), (unsigned char)(px[i] >> 8), (unsigned char)px[i]};
        fwrite(c, 1, 3, f);
    }
    fclose(f);
}
static int chdiff(uint32_t a, uint32_t b) {
    int m = 0;
    for (int s = 0; s < 24; s += 8) {
        int d = (int)(a >> s & 255) - (int)(b >> s & 255);
        if (d < 0) d = -d;
        if (d > m) m = d;
    }
    return m;
}

int main(int argc, char** argv) {
    uint32_t batches = argc > 1 ? (uint32_t)atoi(argv[1]) : 8;
    int frames = argc > 2 ? atoi(argv[2]) : 1;
    opt_shadows = argc > 3 ? atoi(argv[3]) : 1, opt_reflect = argc > 4 ? atoi(argv[4]) : 1;
    const char* ppm_dir = argc > 5 ? argv[5] : NULL;
    if ((batches != 1 && batches != 2 && batches != 4 && batches != 8) || frames < 1 || frames > 100000)
        return fprintf(stderr, "usage: %s [batches 1|2|4|8] [frames] [shadows 0|1] [reflect 0|1] [ppm_dir]\n", argv[0]), 2;
    if (getenv("P06_PAD")) opt_pad = atoi(getenv("P06_PAD"));
    if (getenv("P06_PAD_KIND")) opt_pad_kind = atoi(getenv("P06_PAD_KIND"));
    static uint64_t buf[4096];
    hs_qpu_prog p;
    hs_qpu_prog_init(&p, buf, 4096);
    if (getenv("P06_CULL")) opt_cull = atoi(getenv("P06_CULL"));
    if (getenv("P06_MARGIN")) opt_margin = atof(getenv("P06_MARGIN"));
    int len = build(&p);
    int vec_block = k_vec;
    if (len <= 0) return fprintf(stderr, "build failed: %d at instruction %u\n", len, p.rule_at), 1;
    if (getenv("P06_BUILD_ONLY"))
        return printf("{\"build_only\":true,\"instructions\":%d,\"per_16_pixels\":%d,\"constants\":%d,\"row_constants\":%d}\n", len, n_instr_vec, kn,
                      vec_block), 0;
    hs_v3d dev;
    hs_v3d_bo code, unif, pool, out;
    if (hs_v3d_open(&dev, NULL) || hs_v3d_bo_new(&dev, &code, 65536) || hs_v3d_bo_new(&dev, &unif, 65536) ||
        hs_v3d_bo_new(&dev, &pool, 16384) || hs_v3d_bo_new(&dev, &out, (size_t)W * H * 4 + 4096))
        return fprintf(stderr, "setup failed\n"), 1;
    uint32_t prog = hs_v3d_load(&code, 0, buf, (uint32_t)len);
    if (!prog) return fprintf(stderr, "load failed\n"), 1;
    uint32_t* img = malloc((size_t)W * H * 4);
    uint8_t* kind = malloc((size_t)W * H);
    uint32_t* gpu = (uint32_t*)out.map;
    uint32_t h[4] = {code.handle, unif.handle, pool.handle, out.handle};
    hs_v3d_dispatch_desc d = {prog, unif.gpu, batches, h, 4, 0};
    int failures = 0;
    double best_ms = 1e30;
    printf("{\"scene\":\"full: shading%s%s\",\"cull\":%d,\"width\":%d,\"height\":%d,\"batches\":%u,\"program_instructions\":%d,"
           "\"instructions_per_16_pixels\":%d,\"plain_ground_path_instructions\":%d,\"constants\":%d}\n",
           opt_shadows ? ", shadows" : "", opt_reflect ? ", reflection" : "", opt_cull, W, H, batches, len, n_instr_vec, n_instr_ground, kn);
    for (int f = 0; f < frames; f++) {
        float t = frames > 1 ? 10.0f * (float)f / (float)frames : 1.0f;
        scene_at(t, batches);
        uint32_t* u = (uint32_t*)unif.map;
        float fu[19] = {0, 0, 0, S.ux0, S.dux, S.dux16, S.uy0, S.duy, S.duyb, 0, 0, 0, 0, 0, 0, S.eps, S.nbig, S.bigp, S.zero};
        memcpy(u, fu, sizeof fu);
        u[0] = out.gpu, u[1] = W * 4, u[2] = S.rows_per_batch, u[10] = (batches - 1) * W * 4, u[12] = pool.gpu, u[13] = pool.gpu + 4 * (uint32_t)vec_block,
        u[14] = pool.gpu + 4 * (uint32_t)k_ground, u[9] = pool.gpu + 4 * (uint32_t)k_merge;
        long ncls[3] = {0, 0, 0};
        uint64_t tcl = now_ns();
        if (opt_cull & 4) classify(batches, u + 19, ncls);
        double classify_ms = (now_ns() - tcl) / 1e6;
        for (int i = 0; i < kn; i++) memcpy(pool.map + 4 * i, ksrc[i], 4);
        memset(gpu, 0xEE, (size_t)W * H * 4);
        gpu[W * H] = 0xA5C3F00D;
        uint64_t t0 = now_ns();
        int rc = hs_v3d_dispatch(&dev, &d);
        if (!rc) rc = hs_v3d_bo_wait(&dev, &code, 3000000000ull);
        double ms = (now_ns() - t0) / 1e6;
        if (f > 0 && ms < best_ms) best_ms = ms;
        reference(batches, img, kind);
        long wrong = 0, at_edge = 0, off1 = 0, unwritten = 0, by_hit[3] = {0}, pix[3] = {0};
        int maxd = 0;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                size_t i = (size_t)y * W + (size_t)x;
                int cls = (kind[i] & 7) == 0 ? 0 : (kind[i] & 7) == 7 ? 2 : 1;
                pix[cls]++;
                if ((gpu[i] & 0xFFFFFF) == img[i]) continue;
                wrong++, by_hit[cls]++;
                unwritten += gpu[i] == 0xEEEEEEEE;
                int dd = chdiff(gpu[i] & 0xFFFFFF, img[i]);
                off1 += dd <= 1;
                if (dd > maxd) maxd = dd;
                int edge = 0;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++) {
                        int yy = y + dy, xx = x + dx;
                        if (yy >= 0 && yy < H && xx >= 0 && xx < W && kind[(size_t)yy * W + (size_t)xx] != kind[i]) edge = 1;
                    }
                at_edge += edge;
            }
        int guard_ok = gpu[W * H] == 0xA5C3F00D;
        /* 1/x and 1/sqrt(x) are within 1 ulp of the reference's: a pixel can flip at an
         * edge (object, shadow or reflection). Allowed: 0.01 % of the frame. */
        int ok = !rc && guard_ok && !unwritten && wrong <= W * H / 10000;
        printf("{\"frame\":%d,\"t\":%.2f,\"rc\":%d,\"ok\":%s,\"job_ms\":%.3f,\"pixels\":%d,\"exact\":%ld,\"mismatching\":%ld,\"unwritten\":%ld,"
               "\"at_an_edge_of_the_hit_map\":%ld,\"off_by_one_step\":%ld,\"largest_channel_difference\":%d,\"guard_ok\":%s,"
               "\"regions\":{\"long_path\":%ld,\"plain_ground\":%ld,\"ground_maybe_shadowed\":%ld,\"classify_ms\":%.3f},"
               "\"mismatching_by_primary_hit\":{\"sky\":%ld,\"sphere\":%ld,\"ground\":%ld},\"pixels_by_primary_hit\":{\"sky\":%ld,\"sphere\":%ld,\"ground\":%ld}}\n",
               f, (double)t, rc, ok ? "true" : "false", ms, W * H, (long)W * H - wrong, wrong, unwritten, at_edge, off1, maxd,
               guard_ok ? "true" : "false", ncls[0], ncls[1], ncls[2], classify_ms, by_hit[0], by_hit[1], by_hit[2], pix[0], pix[1], pix[2]);
        if (!ok) failures++;
        if (ppm_dir && (f == 0 || !ok)) {
            char name[64];
            snprintf(name, sizeof name, "scene_s%dr%d_f%d_gpu", opt_shadows, opt_reflect, f);
            write_ppm(ppm_dir, name, gpu);
            snprintf(name, sizeof name, "scene_s%dr%d_f%d_cpu", opt_shadows, opt_reflect, f);
            write_ppm(ppm_dir, name, img);
        }
        if (rc) break;
    }
    /* Timing: jobs back to back. (Timed between reference computations, each job
     * also paid the GPU's ~10 ms wake-up: the reference takes long enough for
     * the GPU to go idle.) */
    double tm[64];
    int nt = 0;
    for (int i = 0; i < 40 && frames > 1; i++) {
        uint64_t t0 = now_ns();
        int rc = hs_v3d_dispatch(&dev, &d);
        if (!rc) rc = hs_v3d_bo_wait(&dev, &code, 3000000000ull);
        if (rc) break;
        if (i >= 8) tm[nt++] = (now_ns() - t0) / 1e6;
    }
    for (int i = 0; i < nt; i++)
        for (int j = i + 1; j < nt; j++)
            if (tm[j] < tm[i]) { double x = tm[i]; tm[i] = tm[j], tm[j] = x; }
    if (nt) best_ms = tm[0];
    if (frames > 1)
        printf("{\"scene_summary\":\"full\",\"cull\":%d,\"shadows\":%d,\"reflection\":%d,\"instructions_per_16_pixels\":%d,\"batches\":%u,\"frames\":%d,"
               "\"failures\":%d,\"job_ms_back_to_back\":{\"best\":%.3f,\"median\":%.3f,\"max\":%.3f,\"jobs\":%d},\"ns_per_pixel_all_qpus\":%.2f,"
               "\"frames_per_second_at_the_median\":%.1f}\n",
               opt_cull, opt_shadows, opt_reflect, n_instr_vec, batches, frames, failures, best_ms, nt ? tm[nt / 2] : 0, nt ? tm[nt - 1] : 0, nt,
               (nt ? tm[nt / 2] : 0) * 1e6 / (W * H), nt ? 1000.0 / tm[nt / 2] : 0);
    return failures != 0;
}
