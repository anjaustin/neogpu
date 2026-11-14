/* p0_6_scene.h: the test_08 scene as a QPU compute kernel, its CPU reference
 * and the per-frame region sort. Shared by tools/p0_6_shade_probe (offscreen,
 * every pixel checked; its header describes the kernel) and
 * tools/p0_6_scene_present_probe (the same frames on the panel).
 * One program per process: the builder keeps its state in file statics.
 */
#ifndef P0_6_SCENE_H
#define P0_6_SCENE_H
#include <errno.h>
#include <math.h>
#include <stdint.h>
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
    float rec_t[8], rec_s[8]; /* stand-ins: what a list loop reads comes from a sphere's record, which the CPU puts in the stream per region */
    float lmark[5], merge_slot; /* stand-ins: where a region's list goes (trace 1, shadow 1, trace 2, shadow 2, ground shadow); the address of the block's merge point */
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
static int kn, kpend_rf, k_overflow;
static const float* kpend;
static int n_instr_vec, opt_shadows = 1, opt_reflect = 1, opt_pad, opt_pad_kind;
static int opt_fault;
static int opt_cull = 15; /* bit 0: rows that are all sky skip the pass; bit 1: no second shade when all 16 reflected rays
                           * miss; bit 2: regions of ground nothing can reach take a short path (classes from the CPU);
                           * bit 3 (needs bit 2): each region tests only the spheres on its lists (from the CPU), in loops */
static double opt_margin = 1e-3; /* slack in the region tests, radians or scene units; negative breaks them (a test of the test) */
static int opt_sites = 31; /* which of the five list loops are used (a bit each; for finding a fault: P06_SITES) */
static int k_vec, k_merge, k_ground, n_instr_ground; /* stream positions: start of the per-pass block; where the two paths around the second shade meet */

static void emit(hs_qpu_prog* p, hs_qpu_alu x) {
    if (kpend) {
        x = kpend_rf < 0 ? hs_qpu_s(x, HS_QPU_SIG_LDUNIFA, X) : hs_qpu_s(x, HS_QPU_SIG_LDUNIFARF, RF(kpend_rf));
        int in_record = (kpend >= S.rec_t && kpend < S.rec_t + 8) || (kpend >= S.rec_s && kpend < S.rec_s + 8);
        if (in_record) ; /* read from a sphere's record: not a word of the stream */
        else if (kn < (int)(sizeof ksrc / sizeof ksrc[0])) ksrc[kn++] = kpend;
        else k_overflow = 1;
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

/* A list loop (cut 8). With lists, every region has its own block of the
 * constant stream (the CPU builds one per distinct set of lists each frame):
 * the constants in the order the kernel reads them and, where a list goes,
 * the records of the region's spheres, each preceded by a non-zero word and
 * the last followed by zero. The loop reads that word, tests it and
 * branches; nothing but sequential reads of the stream.
 * Two forms were tried first and are not used (2026-11-17, GPU resets
 * 262-265, tools/p0_6_unifa_probe branch, sites.txt):
 * - one constant stream with unifa pointed at each record in turn: a branch
 *   that is not taken spoils the stream when the unifa write is one of the 3
 *   instructions before it or in its first delay slot;
 * - with the write moved to the second delay slot, and the record addresses
 *   read from the first uniform stream inside the loop: right at 1, 2 and 4
 *   batches; at 8 batches, wrong in batches 0-3 at regular intervals and
 *   never in 4-7, moving with the instruction count. Not explained. */
static void list_test(hs_qpu_prog* p, int cond, int label, hs_qpu_alu s1, hs_qpu_alu s2) {
    AF(p, HS_QPU_A_MOV, R(1), R(5), X, HS_QPU_PF_PUSHZ);
    hs_qpu_branch(p, cond, label, 0, 0);
    emit(p, s1), emit(p, s2), nop(p);
}
/* The sphere part of the trace as a loop over the region's list. Record: cx, cz, 1/r, albedo r g b, r^2, tiny. */
static void emit_trace_list(hs_qpu_prog* p, int site) {
    enum { OY2 = R_P };
    int l_body = hs_qpu_label_new(p), l_done = hs_qpu_label_new(p);
    M(p, R(0), RF(R_O + 1), RF(R_O + 1));
    K(&S.lmark[site]);
    MOV(p, RF(R_NBEST), RF(R_NBIG));
    list_test(p, HS_QPU_B_A0, l_done, hs_qpu_m(hs_qpu_alu_nop(), HS_QPU_M_FMUL, RF(R_BY), RF(R_O + 1), RF(R_D + 1)),
              hs_qpu_a(HS_QPU_A_MOV, RF(OY2), R(0), X));
    hs_qpu_label_bind(p, l_body);
    KRF(&S.rec_t[0], R_TA);
    nop(p);
    KRF(&S.rec_t[1], R_TA + 1);
    A(p, FSUB, R(0), RF(R_O), RF(R_TA));
    KRF(&S.rec_t[2], R_TA + 2);
    A(p, FSUB, R(1), RF(R_O + 2), RF(R_TA + 1));
    KRF(&S.rec_t[3], R_TA + 3);
    M(p, R(2), R(0), RF(R_D));
    KRF(&S.rec_t[4], R_TA + 4);
    M(p, R(3), R(1), RF(R_D + 2));
    KRF(&S.rec_t[5], R_TA + 5);
    AM(p, FADD, R(2), R(2), R(3), R(0), R(0), R(0));
    AM(p, FADD, R(2), R(2), RF(R_BY), R(1), R(1), R(1));
    K(&S.rec_t[6]);
    A(p, FADD, R(0), R(0), R(1));
    AM(p, FSUB, R(3), RF(OY2), R(5), R(1), R(2), R(2));   /* oy^2 - r^2; b*b */
    K(&S.rec_t[7]);
    A(p, FADD, R(0), R(0), R(3));                         /* cc */
    AF(p, FSUB, R(1), R(1), R(0), PUSHN);
    A(p, FMAX, R(3), R(1), R(5));
    A(p, HS_QPU_A_RSQRT, RF(R_T), R(3), X);
    K(&S.rec_t[0]);                                       /* the word after the record: is there another? Nothing below writes r5 */
    nop(p);
    M(p, R(1), R(1), RF(R_T));
    A(p, FADD, R(1), R(2), R(1));
    AC(p, IFA, R(1), RF(R_NBIG));
    AF(p, FADD, R(3), R(1), RF(R_EPS), PUSHN);
    AC(p, IFNA, R(1), RF(R_NBIG));
    AF(p, FSUB, R(3), RF(R_NBEST), R(1), PUSHN);
    AC2(p, IFA, RF(R_NBEST), R(1), RF(R_H), RF(R_TA));
    AC2(p, IFA, RF(R_H + 1), RF(R_TA + 1), RF(R_H + 2), RF(R_TA + 2));
    AC2(p, IFA, RF(R_H + 3), RF(R_TA + 3), RF(R_H + 4), RF(R_TA + 4));
    AC(p, IFA, RF(R_H + 5), RF(R_TA + 5));
    list_test(p, HS_QPU_B_NA0, l_body, hs_qpu_alu_nop(), hs_qpu_alu_nop());
    hs_qpu_label_bind(p, l_done);
}
/* The shadow test's sphere part as a loop over the region's list. Record: cx, cz, L.x, L.z, r^2, -eps. */
static void emit_shadow_list(hs_qpu_prog* p, int site) {
    enum { OY2 = R_SKS };
    int l_body = hs_qpu_label_new(p), l_done = hs_qpu_label_new(p);
    M(p, R(0), RF(R_O + 1), RF(R_O + 1));
    K(&S.lmark[site]);
    MOV(p, RF(R_SMAX), RF(R_NBIG));
    list_test(p, HS_QPU_B_A0, l_done, hs_qpu_a(HS_QPU_A_MOV, RF(OY2), R(0), X), hs_qpu_alu_nop());
    hs_qpu_label_bind(p, l_body);
    K(&S.rec_s[0]);
    nop(p);
    K(&S.rec_s[1]);
    A(p, FSUB, R(0), RF(R_O), R(5));
    K(&S.rec_s[2]);
    A(p, FSUB, R(1), RF(R_O + 2), R(5));
    K(&S.rec_s[3]);
    M(p, R(2), R(0), R(5));
    M(p, R(3), R(1), R(5));
    AM(p, FADD, R(2), R(2), R(3), R(0), R(0), R(0));
    AM(p, FADD, R(2), R(2), RF(R_SBY), R(1), R(1), R(1));
    K(&S.rec_s[4]);
    A(p, FADD, R(0), R(0), R(1));
    AM(p, FSUB, R(3), RF(OY2), R(5), R(1), R(2), R(2));
    K(&S.rec_s[5]);
    A(p, FADD, R(0), R(0), R(3));
    A(p, FSUB, R(1), R(1), R(0));
    A(p, FSUB, R(3), R(5), R(2));
    M(p, R(0), R(3), R(3));
    A(p, FSUB, R(0), R(0), R(1));
    A(p, FMIN, R(0), R(0), R(1));
    A(p, FMIN, R(0), R(0), R(3));
    K(&S.rec_s[0]);                                       /* the word after the record */
    A(p, FMAX, RF(R_SMAX), RF(R_SMAX), R(0));
    list_test(p, HS_QPU_B_NA0, l_body, hs_qpu_alu_nop(), hs_qpu_alu_nop());
    hs_qpu_label_bind(p, l_done);
}

/* Trace from R_O along R_D. Leaves: R_P, R_N, R_H+3.. (albedo), R_HK
 * (reflectivity), R_MISS (negative: nothing was hit). */
static void emit_trace(hs_qpu_prog* p, int site) {
    if ((opt_cull & 8) && (opt_sites >> site & 1)) {
        emit_trace_list(p, site);
        goto ground;
    }
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
ground:
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
static void emit_shadow(hs_qpu_prog* p, int with_loop, int site) {
    K(&S.L01[0]);
    nop(p);
    K(&S.L01[1]);
    A(p, FADD, RF(R_O), RF(R_P), R(5));
    K(&S.L01[2]);
    A(p, FADD, RF(R_O + 1), RF(R_P + 1), R(5));
    K(&S.L[1]);
    A(p, FADD, RF(R_O + 2), RF(R_P + 2), R(5));
    M(p, RF(R_SBY), RF(R_O + 1), R(5));
    if (with_loop && (opt_cull & 8) && (opt_sites >> site & 1)) {
        emit_shadow_list(p, site);
        return;
    }
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
static void emit_shade(hs_qpu_prog* p, int out, int site) {
    emit_shadow(p, opt_shadows, site);
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
    kn = 0, kpend = NULL, k_overflow = 0;
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
    ldunif(p, RF(R_BY));      /* 0; fault 13: not 0 from the second frame on (read in the sky rows, which come first) */
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
        AF(p, FSUB, R(0), RF(opt_fault == 8 ? R_DUY : R_ZERO), RF(R_UY), PUSHN);
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
        hs_qpu_branch(p, HS_QPU_B_NA0, l_ground, 0, 0);             /* 1 or 2: plain ground */
        if (opt_cull & 8) ldunif(p, RF(R_KBASE));                   /* with lists: the address of the region's block of constants (both paths) */
        else nop(p);
        nop(p), nop(p);
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

    if (opt_cull & 8) KRF(&S.merge_slot, R_KMERGE); /* where this block's two paths around the second shade meet */
    emit_trace(p, 0);
    emit_shade(p, R_COL, 1);
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
        emit_trace(p, 2);
        int l_noshade = hs_qpu_label_new(p), l_merge = hs_qpu_label_new(p);
        if (opt_cull & 2) {
            AF(p, HS_QPU_A_MOV, R(0), RF(R_MISS), X, PUSHN);
            br(p, opt_fault == 1 ? HS_QPU_B_ANYA : HS_QPU_B_ALLA, l_noshade);              /* all 16 reflected rays hit nothing */
        }
        MOV(p, RF(R_TA + 3), RF(R_MISS));                  /* the trace's temporaries are free now */
        emit_shade(p, R_TA, 3);
        /* the reflected colour is the sky where the second ray hit nothing */
        AF(p, HS_QPU_A_MOV, R(0), RF(R_TA + 3), X, PUSHN);
        AC2(p, IFA, RF(R_TA), RF(R_SKY), RF(R_TA + 1), RF(R_SKY + 1));
        AC(p, IFA, RF(R_TA + 2), RF(R_SKY + 2));
        if (opt_cull & 2) {
            br(p, HS_QPU_B_ALWAYS, l_merge);
            /* the path around the shade read none of its constants: move the stream to where the paths meet */
            hs_qpu_label_bind(p, l_noshade);
            if (opt_fault == 2) nop(p);
            else MOV(p, HS_QPU_UNIFA, RF(R_KMERGE));
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
        MOV(p, HS_QPU_UNIFA, RF(opt_cull & 8 ? R_KBASE : R_K1));
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
        br(p, opt_fault == 3 ? HS_QPU_B_ALWAYS : HS_QPU_B_A0, l_gdone);
        emit_shadow(p, opt_shadows, 4);
        hs_qpu_label_bind(p, l_gdone);
        AF(p, FSUB, R(3), RF(R_ZERO), RF(R_SMAX), PUSHN);    /* flag: in shadow */
        MOV(p, R(0), RF(G_ELIT));
        AC(p, IFA, R(0), RF(G_ESH));
        MOV(p, R(1), RF(G_SLIT));
        AC(p, IFA, R(1), RF(G_RIM));
        M(p, R(2), RF(G_GREY), R(0));
        A(p, FADD, R(2), R(2), R(1));                         /* the ground's colour, the same in every channel */
        if (opt_reflect) M(p, R(3), R(2), RF(G_C75));
        else MOV(p, R(3), R(2));
        for (int c = 0; c < 3; c++) {                          /* + a quarter of the sky (its reflection), then 8 bits */
            if (opt_reflect) {
                M(p, R(0), RF(R_SKY + c), RF(G_C25));
                A(p, FADD, R(0), R(3), R(0));
            } else {
                MOV(p, R(0), R(3));
            }
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
        int l_nostore = hs_qpu_label_new(p), l_stored = hs_qpu_label_new(p);
        if (opt_fault == 13) {
            AF(p, HS_QPU_A_MOV, R(0), RF(R_BY), X, HS_QPU_PF_PUSHZ);
            br(p, HS_QPU_B_NA0, l_nostore);
        }
        MOV(p, HS_QPU_TMUD, RF(R_COL));
        emit(p, hs_qpu_m(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(R_OUT), X), HS_QPU_M_ADD, RF(R_OUT), RF(R_OUT), RF(R_64)));
        A(p, HS_QPU_A_TMUWT, X, X, X);
        if (opt_fault == 13) {
            br(p, HS_QPU_B_ALWAYS, l_stored);
            hs_qpu_label_bind(p, l_nostore);
            A(p, HS_QPU_A_ADD, RF(R_OUT), RF(R_OUT), RF(R_64));
        }
        hs_qpu_label_bind(p, l_stored);
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
    return kpend ? -100 : k_overflow ? -101 : hs_qpu_prog_finish(p); /* -101: more constants than ksrc holds */
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
/* Which spheres mattered for a pixel, when ref_mask_img is set: bit i, a valid
 * hit of sphere i by the primary ray; bit 6 + i, sphere i shadows the primary
 * hit; bits 12.. and 18.., the same for the reflected ray and its hit; bit 24,
 * the reflected ray hit the ground. */
static uint32_t* ref_mask_img;
static uint32_t ref_bits;
static int ref_shift;
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
        if (u != S.nbig) ref_bits |= 1u << (ref_shift + i);
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
        if (m > 0) ref_bits |= 1u << (ref_shift + 6 + i);
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
                    ref_bits = 0, ref_shift = 0;
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
                            ref_shift = 12;
                            ref_trace(ro, rd, &h2);
                            if (!signbit(h2.miss) && h2.kind == 7) ref_bits |= 1u << 24; /* the reflected ray hit the ground */
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
                    if (ref_mask_img) ref_mask_img[i] = ref_bits;
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
    float alpha, am, salpha; /* the two cones' half angles (with margin), radians; sin(alpha) */
    int sky;                 /* a row above the horizon that still reads a class (the sky cut is off): always the long path */
} region;
static region* regions;
static uint32_t n_regions, regions_batches;
static int regions_build(uint32_t batches) {
    const double ro[3] = {0, S.roy, S.roz}, mirror[3] = {0, 2.0 * S.gy - S.roy, S.roz};
    const double duy1 = -2.0 / H, uy0 = 1.0 - 1.0 / H;
    region* nr = realloc(regions, sizeof(region) * (size_t)S.rows_per_batch * VECS);
    if (!nr) return -1;
    regions = nr, n_regions = 0, regions_batches = batches;
    for (uint32_t r = 0; r < S.rows_per_batch; r++) {
        uint32_t y0 = r * batches, y1 = y0 + batches - 1;
        int sky0 = uy0 + y0 * duy1 > 0, sky1 = uy0 + y1 * duy1 > 0;
        /* Every batch reads the same class for its row of a region, and reads
         * none in a row it fills as sky: a region must not straddle the horizon. */
        if (sky0 != sky1) return -2;
        if (sky0) {
            if (opt_cull & 1) continue; /* the kernel reads no class for a sky row */
            for (int v = 0; v < VECS; v++) {
                memset(&regions[n_regions], 0, sizeof(region));
                regions[n_regions++].sky = 1;
            }
            continue;
        }
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
            q->sky = 0;
            double a = acos(cosh) + opt_margin, am = acos(cosm) + opt_margin;
            for (int c = 0; c < 3; c++) q->axis[c] = (float)axis[c], q->pc[c] = (float)pc[c], q->maxis[c] = (float)(m2axis[c] / ml);
            q->ca = (float)cos(a), q->sa = (float)sin(a), q->cm = (float)cos(am), q->sm = (float)sin(am);
            q->rp = (float)(sqrt(max2) + opt_margin);
            q->alpha = (float)a, q->am = (float)am, q->salpha = (float)sin(a);
        }
    }
    return 0;
}
/* ---- per-region sphere lists (cut 8; CPU, per frame) ----
 * For a region on the long path: which spheres can its primary rays hit (P),
 * which can shadow the points they hit (S1), which can its reflected rays
 * hit (R), which can shadow those points (S2). Every test is a conservative
 * bound; the lists are checked against the truth for every pixel by
 * tools/p0_6_lists_stats and, on the GPU, by exact comparison with the image
 * rendered without lists.
 *
 * The rays of a region lie in a cone (axis a, half angle alpha) from the
 * camera. Where they enter sphere j is described around the direction w to
 * its centre: a ray at angle psi from w enters at a point whose normal is at
 * theta(psi) = asin(D sin(psi) / r) - psi from -w, at the same azimuth
 * (theta grows with psi when D^2 > 2 r^2). The region's patch on the sphere
 * is cut into opt_cells x opt_cells cells in (psi, azimuth); a cell's
 * normals lie in a cone (n0, gam), its rays in a cone (dm, ac), so its
 * reflected rays start within rho of p0 and lie in a cone (r0, ac + 2 gam):
 * a mirror turns a ray by twice the turn of its normal. */
static int opt_cells = 2;
static double opt_lmargin = 2e-3; /* slack of the list tests, radians or scene units */
typedef struct { uint32_t P, S1, R, S2; } lists;
static struct { double c[NSPH][3], rad[NSPH], w[NSPH][3], D[NSPH], psimax[NSPH], e[NSPH][3], re[NSPH], L[3]; float wf[NSPH][3]; uint32_t tube[NSPH]; } LF; /* per frame */
static double dot3(const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
static double clamp1(double x) { return x > 1 ? 1 : x < -1 ? -1 : x; }
/* can a ray that starts within rho of q and runs along L hit sphere i? */
static int list_tube(const double* q, double rho, int i) {
    double v[3] = {LF.c[i][0] - q[0], LF.c[i][1] - q[1], LF.c[i][2] - q[2]};
    double al = dot3(v, LF.L), reach = LF.rad[i] + rho + opt_lmargin;
    return dot3(v, v) - al * al <= reach * reach && al > -reach;
}
/* can a ray that starts within rho of p0, in the cone (r0, thr), hit the ball (c, rb)? */
static int list_cone(const double* p0, double rho, const double* r0, double thr, const double* c, double rb) {
    double v[3] = {c[0] - p0[0], c[1] - p0[1], c[2] - p0[2]}, dist = sqrt(dot3(v, v));
    if (rb + rho >= dist) return 1;
    return acos(clamp1(dot3(v, r0) / dist)) <= thr + asin((rb + rho) / dist);
}
static void lists_frame(void) {
    const double o[3] = {0, S.roy, S.roz};
    for (int c = 0; c < 3; c++) LF.L[c] = S.L[c];
    for (int i = 0; i < NSPH; i++) {
        LF.c[i][0] = S.cx[i], LF.c[i][1] = 0, LF.c[i][2] = S.cz[i], LF.rad[i] = 1.0 / S.invr[i];
        double d2 = 0;
        for (int c = 0; c < 3; c++) LF.w[i][c] = LF.c[i][c] - o[c], d2 += LF.w[i][c] * LF.w[i][c];
        LF.D[i] = sqrt(d2);
        for (int c = 0; c < 3; c++) LF.w[i][c] /= LF.D[i], LF.wf[i][c] = (float)LF.w[i][c];
        LF.psimax[i] = asin(clamp1(LF.rad[i] / LF.D[i]));
        /* its shadow on the ground: inside a disc around e */
        double t = (LF.c[i][1] - S.gy) / LF.L[1];
        for (int c = 0; c < 3; c++) LF.e[i][c] = LF.c[i][c] - LF.L[c] * t;
        LF.e[i][1] = S.gy, LF.re[i] = LF.rad[i] / LF.L[1] + 0.02;
    }
    for (int i = 0; i < NSPH; i++) { /* tube[i]: the spheres that can shadow a point of sphere i, itself included */
        LF.tube[i] = 0;
        for (int k = 0; k < NSPH; k++)
            if (list_tube(LF.c[i], LF.rad[i], k)) LF.tube[i] |= 1u << k;
    }
}
/* can a ray that starts within rho of p0, in the cone (r0, thr <= 1.5) given by its cosine and sine, hit the ball (c, rb)?
 * angle(c - p0, r0) <= thr + asin(x), x = (rb + rho) / distance, compared on the cosines */
static int cone_hits(const float* p0, float rho, const float* r0, float cthr, float sthr, const double* c, float rb) {
    float v[3] = {(float)c[0] - p0[0], (float)c[1] - p0[1], (float)c[2] - p0[2]}, d2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2], R = rb + rho;
    if (R * R >= d2) return 1;
    float dist = sqrtf(d2), x = R / dist;
    return (v[0] * r0[0] + v[1] * r0[1] + v[2] * r0[2]) / dist >= cthr * sqrtf(1.0f - x * x) - sthr * x;
}
/* One pass over sphere j's patch cut into kc x kc cells: which of the spheres
 * in *cr can a reflected ray hit, into which of the ground shadows in *cg can
 * it land, can a normal face away from the light. The results replace *cr,
 * *cg, *self: every pass is a bound, so a finer pass only needs to look at
 * what a coarser one let through. */
typedef struct { const float* w; float e1[3], e2[3], D, r, plo, phi, beta; int j; } patch;
static void patch_pass(const patch* pt, int kc, uint32_t* cr, uint32_t* cg, int* self) {
    const float m = (float)opt_lmargin, L[3] = {S.L[0], S.L[1], S.L[2]};
    uint32_t outr = 0, outg = 0;
    int outs = 0;
    float side[8][3], hf = pt->beta / (float)kc;
    for (int kf = 0; kf < kc; kf++) {
        float fm = -pt->beta + 2.0f * pt->beta * ((float)kf + 0.5f) / (float)kc, cf = cosf(fm), sf = sinf(fm);
        for (int c = 0; c < 3; c++) side[kf][c] = pt->e1[c] * cf + pt->e2[c] * sf;
    }
    float pb = pt->plo, spb = sinf(pb), q0 = pt->D * spb / pt->r, tb = asinf(q0 > 1 ? 1 : q0) - pb;
    for (int kp = 0; kp < kc; kp++) {
        float pa = pb, ta = tb;
        pb = pt->plo + (pt->phi - pt->plo) * (float)(kp + 1) / (float)kc, spb = sinf(pb), q0 = pt->D * spb / pt->r, tb = asinf(q0 > 1 ? 1 : q0) - pb;
        float pm = 0.5f * (pa + pb), tm = 0.5f * (ta + tb), ac = 0.5f * (pb - pa) + hf * spb + m, gam = 0.5f * (tb - ta) + hf * sinf(tb) + m;
        float cpm = cosf(pm), spm = sinf(pm), ctm = cosf(tm), stm = sinf(tm), sgam = sinf(gam > 1.5f ? 1.5f : gam);
        float thr = ac + 2.0f * gam + m, rho = (pt->r + S.c02) * gam + m;
        int wide = thr > 1.5f; /* the cosine test needs thr + asin(x) < pi: beyond 1.5, everything */
        float cthr = wide ? 0 : cosf(thr), sthr = wide ? 1 : sinf(thr);
        for (int kf = 0; kf < kc; kf++) {
            float dm[3], n0[3], r0[3], p0[3], an = 0, nl = 0;
            for (int c = 0; c < 3; c++) {
                dm[c] = pt->w[c] * cpm + side[kf][c] * spm, n0[c] = -pt->w[c] * ctm + side[kf][c] * stm;
                an += dm[c] * n0[c], nl += n0[c] * L[c];
            }
            if (gam > 1.5f || nl <= sgam) outs = 1;
            if (wide) { outr |= *cr, outg |= *cg; continue; }
            for (int c = 0; c < 3; c++) r0[c] = dm[c] - 2.0f * an * n0[c], p0[c] = (float)LF.c[pt->j][c] + (pt->r + S.c02) * n0[c];
            for (uint32_t t = *cr & ~outr; t; t &= t - 1) {
                int i = __builtin_ctz(t);
                if (cone_hits(p0, rho, r0, cthr, sthr, LF.c[i], (float)LF.rad[i])) outr |= 1u << i;
            }
            if (!(r0[1] - sthr > 0)) /* a reflected ray may go down to the ground: into whose shadow? */
                for (uint32_t t = *cg & ~outg; t; t &= t - 1) {
                    int i = __builtin_ctz(t);
                    if (cone_hits(p0, rho, r0, cthr, sthr, LF.e[i], (float)LF.re[i])) outg |= 1u << i;
                }
        }
    }
    *cr = outr, *cg = outg, *self = outs;
}
static void region_lists(const region* q, lists* out) {
    const double mir[3] = {0, 2.0 * S.gy - S.roy, S.roz};
    const float a[3] = {q->axis[0], q->axis[1], q->axis[2]}, alpha = q->alpha, m = (float)opt_lmargin, PI = 3.14159265f;
    lists l = {0, 0, 0, 0};
    int inside_a_disc = 0;
    for (int j = 0; j < NSPH && !q->sky; j++) {
        patch pt;
        pt.w = LF.wf[j], pt.D = (float)LF.D[j], pt.r = (float)LF.rad[j], pt.j = j;
        float aw = a[0] * pt.w[0] + a[1] * pt.w[1] + a[2] * pt.w[2], psimax = (float)LF.psimax[j];
        float psi0 = acosf(aw > 1 ? 1 : aw < -1 ? -1 : aw);
        if (psi0 - alpha > psimax) continue; /* no ray of the region reaches sphere j */
        l.P |= 1u << j;
        if (psi0 + alpha < psimax - m) inside_a_disc = 1; /* every ray of the region enters sphere j: none reaches the ground first */
        if (!(pt.D * pt.D > 2 * pt.r * pt.r)) { /* the camera is too close for the bounds below: everything */
            l.S1 |= 63, l.R |= 63, l.S2 |= 63;
            continue;
        }
        pt.plo = psi0 - alpha < 0 ? 0 : psi0 - alpha, pt.phi = psi0 + alpha > psimax ? psimax : psi0 + alpha, pt.beta = PI;
        float el = 0;
        for (int c = 0; c < 3; c++) pt.e1[c] = a[c] - aw * pt.w[c], el += pt.e1[c] * pt.e1[c];
        el = sqrtf(el);
        if (psi0 > alpha + 1e-4f && el > 1e-6f) {
            float sb = q->salpha / el; /* el = sin(psi0) */
            pt.beta = asinf(sb > 1 ? 1 : sb) + m;
            for (int c = 0; c < 3; c++) pt.e1[c] /= el;
        } else { /* the axis is (nearly) the direction to the centre: any frame around it, the whole circle */
            float t[3] = {fabsf(pt.w[0]) < 0.9f, fabsf(pt.w[0]) >= 0.9f, 0}, tw = t[0] * pt.w[0] + t[1] * pt.w[1];
            el = 0;
            for (int c = 0; c < 3; c++) pt.e1[c] = t[c] - tw * pt.w[c], el += pt.e1[c] * pt.e1[c];
            el = sqrtf(el);
            for (int c = 0; c < 3; c++) pt.e1[c] /= el;
            pt.plo = 0;
        }
        pt.e2[0] = pt.w[1] * pt.e1[2] - pt.w[2] * pt.e1[1], pt.e2[1] = pt.w[2] * pt.e1[0] - pt.w[0] * pt.e1[2], pt.e2[2] = pt.w[0] * pt.e1[1] - pt.w[1] * pt.e1[0];
        l.S1 |= LF.tube[j] & ~(1u << j); /* the others by their tubes; itself only where a normal can face away from the light */
        uint32_t cr = 63u & ~(1u << j), cg = 63;
        int self = 1;
        patch_pass(&pt, 1, &cr, &cg, &self);
        if (opt_cells > 1 && (cr || cg || self)) patch_pass(&pt, opt_cells > 8 ? 8 : opt_cells, &cr, &cg, &self);
        if (self) l.S1 |= 1u << j;
        l.R |= cr, l.S2 |= cg;
        for (uint32_t t = cr; t; t &= t - 1) l.S2 |= LF.tube[__builtin_ctz(t)];
    }
    if (!inside_a_disc && !q->sky) {
        /* rays of the region may reach the ground: the region sort's tests, per sphere */
        const double ma[3] = {q->maxis[0], q->maxis[1], q->maxis[2]}, pc[3] = {q->pc[0], q->pc[1], q->pc[2]};
        for (int i = 0; i < NSPH; i++) {
            if (list_cone(mir, 0, ma, q->am, LF.c[i], LF.rad[i] + S.c02)) l.R |= 1u << i, l.S2 |= LF.tube[i];
            if (list_tube(pc, q->rp, i)) l.S1 |= 1u << i;
        }
    }
    *out = l;
}
/* ---- a region's block of the constant stream (cut 8) ----
 * The kernel's constants from stream position k0 to k1 with, at each list
 * mark, the records of the spheres on that list. Blocks are shared: one per
 * distinct set of lists per frame, found through a small table. */
enum { BLOCK_SLOTS = 1024, BLOCK_MAX_WORDS = 512 };
static struct {
    uint32_t* map;      /* the pool, as words */
    uint32_t gpu, words, used, first; /* its address, its size, words handed out, where the blocks start */
    uint64_t key[BLOCK_SLOTS];
    uint32_t addr[BLOCK_SLOTS];
    long made;          /* blocks built this frame */
} BK;
static uint32_t block_build(int k0, int k1, const uint32_t mask[5]) {
    if (BK.used + BLOCK_MAX_WORDS > BK.words) return 0;
    uint32_t* start = BK.map + BK.used, *o = start, *fix = NULL, merge_off = 0;
    const float one = 1.0f;
    for (int i = k0; i < k1; i++) {
        const float* src = ksrc[i];
        if (i == k_merge) merge_off = (uint32_t)(o - start);
        if (src >= S.lmark && src < S.lmark + 5) {
            int site = (int)(src - S.lmark), trace = site == 0 || site == 2;
            for (int j = 0; j < NSPH; j++) {
                if (!(mask[site] >> j & 1)) continue;
                float r2 = j < 5 ? S.r2s : S.r2c;
                float tr[8] = {S.cx[j], S.cz[j], S.invr[j], S.alb[j][0], S.alb[j][1], S.alb[j][2], r2, S.tiny};
                float sr[6] = {S.cx[j], S.cz[j], S.L[0], S.L[2], r2, S.negeps};
                memcpy(o++, &one, 4); /* not zero: a record follows */
                if (trace) memcpy(o, tr, sizeof tr), o += 8;
                else memcpy(o, sr, sizeof sr), o += 6;
            }
            *o++ = 0; /* the end of the list */
        } else if (src == &S.merge_slot) {
            fix = o++;
        } else {
            memcpy(o++, src, 4);
        }
    }
    uint32_t addr = BK.gpu + 4u * BK.used;
    if (fix) *fix = addr + 4u * merge_off;
    BK.used += (uint32_t)(o - start), BK.made++;
    return addr;
}
/* the block for these lists: built once per frame. kind 0: the long path; 1: the ground path. 0 when the pool is full. */
static uint32_t block_for(int kind, const uint32_t mask[5]) {
    uint64_t key = 1ull << 40 | (uint64_t)kind << 36 | (uint64_t)mask[0] | (uint64_t)mask[1] << 6 | (uint64_t)mask[2] << 12 | (uint64_t)mask[3] << 18 |
                   (uint64_t)mask[4] << 24;
    uint32_t h = (uint32_t)((key * 0x9E3779B97F4A7C15ull) >> 54) % BLOCK_SLOTS;
    for (int probe = 0; probe < BLOCK_SLOTS; probe++, h = (h + 1) % BLOCK_SLOTS) {
        if (BK.key[h] == key) return BK.addr[h];
        if (!BK.key[h]) {
            uint32_t a = kind ? block_build(k_ground, kn, mask) : block_build(k_vec, k_ground, mask);
            if (a) BK.key[h] = key, BK.addr[h] = a;
            return a;
        }
    }
    return 0;
}
/* Per frame: each region against the six spheres. A sphere of angular radius
 * b around direction w is inside a cone of half angle a around the axis when
 * cos(angle) >= cos(a + b); the comparison is done on the cosines.
 * Writes the regions' part of the uniform stream: a class word per region
 * and, with cut 8, the address of the region's block of constants (built
 * here). Returns the number of words, or < 0. */
static long list_count[4]; /* entries written to the four kinds of list this frame */
static int classify(uint32_t batches, uint32_t* cls, long count[3]) {
    if (!regions || regions_batches != batches) {
        int e = regions_build(batches);
        if (e) return e;
    }
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
        /* The reflected ray starts c02 above the ground: it is the line from the
         * mirror image moved up by c02, so it can reach a sphere that line misses
         * by up to c02. The sphere is made that much larger for this test. */
        sbm[i] = (rad[i] + (opt_fault == 11 ? 0.0f : S.c02)) / dm, cbm[i] = sqrtf(1.0f - sbm[i] * sbm[i]);
        /* the cosine comparisons below need the camera (and its image) outside every sphere */
        if (!(sb[i] < 1.0f) || !(sbm[i] < 1.0f)) return -3;
    }
    int with_lists = (opt_cull & 8) != 0;
    uint32_t* o = cls;
    if (with_lists) {
        const uint32_t all0[5] = {63, 63, 63, 63, 0}, all1[5] = {0, 0, 0, 0, 63};
        lists_frame();
        memset(BK.key, 0, sizeof BK.key);
        BK.used = BK.first, BK.made = 0;
        if (!block_for(0, all0) || !block_for(1, all1)) return -4; /* what a region gets when the pool is full */
    }
    count[0] = count[1] = count[2] = 0;
    list_count[0] = list_count[1] = list_count[2] = list_count[3] = 0;
    for (uint32_t n = 0; n < n_regions; n++) {
        const region* q = &regions[n];
        int seen = q->sky, shadowed = 0, reflected = 0;
        uint32_t sh_mask = 0;
        for (int i = 0; i < NSPH && !q->sky; i++) {
            float da = q->axis[0] * w[i][0] + q->axis[1] * w[i][1] + q->axis[2] * w[i][2];
            float dma = q->maxis[0] * wm[i][0] + q->maxis[1] * wm[i][1] + q->maxis[2] * wm[i][2];
            seen |= da >= q->ca * cb[i] - q->sa * sb[i];
            reflected |= dma >= q->cm * cbm[i] - q->sm * sbm[i];
            float px = c3[i][0] - q->pc[0], py = c3[i][1] - q->pc[1], pz = c3[i][2] - q->pc[2];
            float sL = px * S.L[0] + py * S.L[1] + pz * S.L[2], reach = rad[i] + q->rp;
            int sh = (px * px + py * py + pz * pz) - sL * sL <= reach * reach && sL > -reach;
            shadowed |= sh, sh_mask |= (uint32_t)sh << i;
        }
        if (opt_fault == 5) reflected = 0;
        if (opt_fault == 6) seen = q->sky;
        if (opt_fault == 7) shadowed = 0;
        uint32_t c = seen || reflected ? 0 : shadowed ? 2 : 1;
        *o++ = c, count[c]++;
        if (!with_lists) continue;
        uint32_t mask[5] = {0, 0, 0, 0, 0}, addr;
        if (c == 0) {
            lists l;
            region_lists(q, &l);
            if (opt_fault == 14 && l.P) l.P &= l.P - 1; /* fault: the first sphere of the primary list is left out */
            if (opt_fault == 15) l.R = 0;               /* fault: the reflected ray is tested against nothing */
            if (opt_fault == 16) l.S2 = 0;              /* fault: nothing is tested for shadows of the reflected hit */
            mask[0] = l.P, mask[1] = l.S1, mask[2] = l.R, mask[3] = l.S2;
            list_count[0] += __builtin_popcount(l.P), list_count[1] += __builtin_popcount(l.S1);
            list_count[2] += __builtin_popcount(l.R), list_count[3] += __builtin_popcount(l.S2);
            addr = block_for(0, mask);
            if (!addr) { /* the pool is full: the block with every sphere on every list, built first */
                const uint32_t all[5] = {63, 63, 63, 63, 0};
                addr = block_for(0, all);
            }
        } else {
            mask[4] = c == 2 ? sh_mask : 0;
            addr = block_for(1, mask);
            if (!addr) {
                const uint32_t all[5] = {0, 0, 0, 0, 63};
                addr = block_for(1, all);
            }
        }
        if (!addr) return -4;
        *o++ = addr;
    }
    return (int)(o - cls);
}

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
/* Everything the job reads for a frame at scene time t: the scene's constants,
 * the uniforms and the region classes. The job writes W x H words of
 * 0x00RRGGBB at out_gpu, rows W * 4 bytes apart. stale_sky: for fault 13.
 * Returns 0, or the classifier's error. */
static int frame_setup(float t, int stale_sky, uint32_t batches, hs_v3d_bo* unif, hs_v3d_bo* pool, uint32_t out_gpu, long ncls[3],
                       double* classify_ms) {
    scene_at(t, batches);
    uint32_t* u = (uint32_t*)unif->map;
    float fu[19] = {0, 0, 0, S.ux0, S.dux, S.dux16, S.uy0, S.duy, S.duyb, 0, 0, 0, 0, 0, 0, S.eps, S.nbig, S.bigp, S.zero};
    memcpy(u, fu, sizeof fu);
    u[0] = out_gpu, u[1] = W * 4 * (opt_fault == 10 ? 2 : 1), u[2] = S.rows_per_batch + (opt_fault == 4), u[10] = (batches - 1) * W * 4;
    u[12] = pool->gpu, u[13] = pool->gpu + 4 * (uint32_t)k_vec, u[14] = pool->gpu + 4 * (uint32_t)(opt_fault == 9 ? k_vec : k_ground);
    u[9] = pool->gpu + 4 * (uint32_t)k_merge, u[11] = opt_fault == 13 && stale_sky;
    ncls[0] = ncls[1] = ncls[2] = 0;
    /* cut 8: the regions' blocks go after the stream's own constants */
    BK.map = (uint32_t*)pool->map, BK.gpu = pool->gpu, BK.words = (uint32_t)(pool->size / 4), BK.first = (uint32_t)kn;
    uint64_t tcl = now_ns();
    for (int i = 0; i < kn; i++) memcpy(pool->map + 4 * i, ksrc[i], 4); /* marks copy as zeros: unused where lists are on */
    int e = opt_cull & 4 ? classify(batches, u + 19, ncls) : 0;
    *classify_ms = (now_ns() - tcl) / 1e6;
    return e < 0 ? e : 0;
}
/* bytes of uniforms a frame can need: 19 words, and per region a class word and (cut 8) a block address */
static size_t scene_unif_bytes(uint32_t batches) { return (19 + (size_t)(H / batches + 1) * VECS * 2) * 4; }
/* bytes of the constant pool: the stream's constants and (cut 8) the regions' blocks */
static size_t scene_pool_bytes(void) { return (opt_cull & 8) ? (size_t)BLOCK_SLOTS * BLOCK_MAX_WORDS * 4 : 16384; }
/* The reference must do the kernel's operations one by one: a compiler that
 * fuses a multiply and an add (the default on this CPU without
 * -ffp-contract=off) gives other results (measured: about 120 pixels of a
 * frame). 1 if this build fuses them. */
static int scene_fp_contracted(void) {
    volatile float a = 1.0f + 0x1p-12f, c = -1.0f - 0x1p-11f;
    float x = a, y = a, z = c;
    float r = x * y + z; /* the product rounds to 1 + 2^-11: 0 unless fused */
    return r != 0.0f;
}
/* a whole number in [lo, hi], nothing after it: 0, or -1 */
static int number(const char* s, long lo, long hi, long* out) {
    char* end;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || end == s || *end || v < lo || v > hi) return -1;
    *out = v;
    return 0;
}
#endif
