/* p0_6_scene_probe: the test_08 scene as a hand-assembled QPU compute kernel,
 * against a CPU reference of the same arithmetic.
 * (P0-6 plan steps 2 and 3, docs/audit/P0_6_PLAN.md.)
 *
 * Stage 1 (this file so far): primary rays only. One ray per pixel against
 * the six spheres and the ground of tests/test_08_raycast_orbit; the nearest
 * hit gives a flat colour (the sphere's albedo, the ground's checker), a miss
 * gives the sky. No shading, shadows or reflection yet.
 *
 * The kernel: one compute job per frame, `batches` batches, each rendering
 * a block of rows; 16 pixels per pass of the inner loop; one thread switch
 * per pass (a compute job gives way to a renderer only there, and r0..r3 do
 * not survive one: nothing is kept in them across it). Scene constants live
 * in registers.
 *
 * The reference: the same operations in the same order in C (no fused
 * multiply-add), with 1/x and 1/sqrt(x) correctly rounded. The hardware's
 * are within 1 ulp of that (tools/p0_6_sfu_probe), so a few pixels at edges
 * can differ; the probe reports exact matches and where the rest are.
 *
 * Offscreen: renders into a plain buffer, no display.
 * Usage: p0_6_scene_probe [batches=8] [frames=20] [ppm_dir]
 *        (inside the GPU lock). Output: JSON lines; exit 1 if a frame has
 *        more mismatching pixels than the edge allowance (0.5 %).
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

/* register file use */
enum {
    R_OUT = 0, R_ROWS = 1, R_VEC = 2, R_UX0 = 3, R_UX = 4, R_DUX16 = 5, R_UY = 6, R_DUY = 7, R_64 = 8, R_BATCH = 9, R_T = 9 /* after setup: special-function results */,
    R_RDX = 10, R_RDY = 11, R_RDZ = 12, R_BY = 13, R_NBEST = 14, R_COL = 15,
    R_EPS = 16, R_NBIG = 17, R_TINY = 18, R_OCY = 19, R_C289 = 20, R_M17 = 21, R_GNUM = 22, R_SKY = 23, R_GEVEN = 24, R_GODD = 25,
    R_ROZ = 26, R_VECS = 27, R_V = 28, R_DV = 29,
    R_SPH = 30,  /* 3 per sphere: ocx, ocz, cc */
    R_SCOL = 48, /* 1 per sphere: packed colour */
    R_SKYA = 54, /* 3 */
    R_SKYB = 57, /* 3 */
    R_255 = 60, R_ONE = 61, R_UYSQ = 62, R_BIGP = 63,
};
/* uniforms, in load order */
enum {
    U_OUT, U_BYTES, U_ROWS, U_UX0, U_DUX, U_DUX16, U_UY0, U_DUY, U_DUYB, U_V0, U_DV, U_DVB,
    U_EPS, U_NBIG, U_TINY, U_OCY, U_C289, U_M17, U_GNUM, U_GEVEN, U_GODD, U_ROZ, U_VECS, U_255, U_ONE, U_BIGP,
    U_SPH,                       /* 18 */
    U_SCOL = U_SPH + 3 * NSPH,   /* 6 */
    U_SKYA = U_SCOL + NSPH,      /* 3 */
    U_SKYB = U_SKYA + 3,         /* 3 */
    U_COUNT = U_SKYB + 3
};

static int n_instr_vec; /* instructions in one pass of the inner loop */
static int n_spheres = NSPH, no_ground, no_thrsw, no_tmuwt; /* timing experiments (P06_*): the image is then not the scene */
static void emit(hs_qpu_prog* p, hs_qpu_alu x) { hs_qpu_emit(p, &x); }
static void nop(hs_qpu_prog* p) { emit(p, hs_qpu_alu_nop()); }
static void A(hs_qpu_prog* p, int o, hs_qpu_reg d, hs_qpu_reg a, hs_qpu_reg b) { emit(p, hs_qpu_a(o, d, a, b)); }
static void M(hs_qpu_prog* p, hs_qpu_reg d, hs_qpu_reg a, hs_qpu_reg b) { emit(p, hs_qpu_m(hs_qpu_alu_nop(), HS_QPU_M_FMUL, d, a, b)); }
static void AF(hs_qpu_prog* p, int o, hs_qpu_reg d, hs_qpu_reg a, hs_qpu_reg b, int pf) {
    hs_qpu_alu s = hs_qpu_a(o, d, a, b);
    s.af = (uint8_t)pf;
    emit(p, s);
}
static void AC(hs_qpu_prog* p, int cond, hs_qpu_reg d, hs_qpu_reg s) { /* conditional move */
    hs_qpu_alu x = hs_qpu_a(HS_QPU_A_MOV, d, s, X);
    x.ac = (uint8_t)cond;
    emit(p, x);
}
static void ldunif(hs_qpu_prog* p, hs_qpu_reg d) { emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDUNIFRF, d)); }
static void br(hs_qpu_prog* p, int cond, int label) {
    hs_qpu_branch(p, cond, label, 0, 0);
    nop(p), nop(p), nop(p);
}

static int build(hs_qpu_prog* p) {
    hs_qpu_prog_rules(p, 0);
    A(p, HS_QPU_A_BAND, RF(R_BATCH), RF(0), I(15)); /* the hardware's batch number */
    /* setup: uniforms into registers (some through temporaries) */
    ldunif(p, RF(R_OUT));
    ldunif(p, RF(R_SKY));    /* bytes per batch, for now */
    ldunif(p, RF(R_ROWS));
    ldunif(p, RF(R_UX0));
    ldunif(p, RF(R_UX));     /* dux, for now */
    ldunif(p, RF(R_DUX16));
    ldunif(p, RF(R_UY));
    ldunif(p, RF(R_DUY));
    ldunif(p, RF(R_NBEST));  /* duy per batch, for now */
    ldunif(p, RF(R_V));
    ldunif(p, RF(R_DV));
    ldunif(p, RF(R_COL));    /* dv per batch, for now */
    static const int order[] = {R_EPS, R_NBIG, R_TINY, R_OCY, R_C289, R_M17, R_GNUM, R_GEVEN, R_GODD, R_ROZ, R_VECS, R_255, R_ONE, R_BIGP};
    for (unsigned i = 0; i < sizeof order / sizeof order[0]; i++) ldunif(p, RF(order[i]));
    for (int i = 0; i < 3 * NSPH; i++) ldunif(p, RF(R_SPH + i));
    for (int i = 0; i < NSPH; i++) ldunif(p, RF(R_SCOL + i));
    for (int i = 0; i < 6; i++) ldunif(p, RF(R_SKYA + i));
    /* per lane and per batch starts */
    A(p, HS_QPU_A_EIDX, R(1), X, X);
    A(p, HS_QPU_A_SHL, R(0), R(1), I(2));
    A(p, HS_QPU_A_ADD, RF(R_OUT), RF(R_OUT), R(0));
    emit(p, hs_qpu_m(hs_qpu_alu_nop(), HS_QPU_M_UMUL24, R(2), RF(R_BATCH), RF(R_SKY)));
    A(p, HS_QPU_A_ADD, RF(R_OUT), RF(R_OUT), R(2));
    A(p, HS_QPU_A_ITOF, R(1), R(1), X);
    M(p, R(1), R(1), RF(R_UX));                 /* lane * dux */
    A(p, HS_QPU_A_FADD, RF(R_UX0), RF(R_UX0), R(1));
    A(p, HS_QPU_A_ITOF, R(2), RF(R_BATCH), X);
    M(p, R(3), R(2), RF(R_NBEST));              /* batch * duy per batch */
    A(p, HS_QPU_A_FADD, RF(R_UY), RF(R_UY), R(3));
    M(p, R(3), R(2), RF(R_COL));                /* batch * dv per batch */
    A(p, HS_QPU_A_FADD, RF(R_V), RF(R_V), R(3));
    A(p, HS_QPU_A_MOV, R(2), I(1), X);
    A(p, HS_QPU_A_SHL, RF(R_64), R(2), I(6));

    int l_row = hs_qpu_label_new(p), l_vec = hs_qpu_label_new(p);
    /* ---- per row ---- */
    hs_qpu_label_bind(p, l_row);
    M(p, R(0), RF(R_UY), RF(R_UY));
    A(p, HS_QPU_A_FADD, RF(R_UYSQ), R(0), RF(R_C289));
    A(p, HS_QPU_A_FSUB, R(3), RF(R_ONE), RF(R_V)); /* 1 - v */
    for (int c = 0; c < 3; c++) {                  /* sky = A * (1 - v) + B * v, to 8 bits, packed */
        M(p, R(0), RF(R_SKYA + c), R(3));
        M(p, R(1), RF(R_SKYB + c), RF(R_V));
        A(p, HS_QPU_A_FADD, R(0), R(0), R(1));
        M(p, R(0), R(0), RF(R_255));
        A(p, HS_QPU_A_FTOIN, R(0), R(0), X);
        if (c == 0) {
            A(p, HS_QPU_A_MOV, R(2), R(0), X);
        } else {
            A(p, HS_QPU_A_SHL, R(2), R(2), I(8));
            A(p, HS_QPU_A_BOR, R(2), R(2), R(0));
        }
    }
    A(p, HS_QPU_A_MOV, RF(R_SKY), R(2), X);
    A(p, HS_QPU_A_MOV, RF(R_UX), RF(R_UX0), X);
    A(p, HS_QPU_A_MOV, RF(R_VEC), RF(R_VECS), X);

    /* ---- per 16 pixels ---- */
    hs_qpu_label_bind(p, l_vec);
    int i0 = (int)p->n;
    if (!no_thrsw) {
        emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_THRSW, X)); /* give way to a renderer; r0..r3 are dead here */
        nop(p), nop(p);
    }
    /* the ray: rd = normalize(ux, uy, -1.7) */
    M(p, R(0), RF(R_UX), RF(R_UX));
    A(p, HS_QPU_A_FADD, R(0), R(0), RF(R_UYSQ));
    /* Special-function results go to a register-file entry (rf9: the batch
     * number is not needed after setup). In Mesa's compiled code for this
     * scene all twelve do; whether an accumulator destination works was not
     * established. */
    A(p, HS_QPU_A_RSQRT, RF(R_T), R(0), X);
    A(p, HS_QPU_A_MOV, RF(R_NBEST), RF(R_NBIG), X);
    M(p, RF(R_RDX), RF(R_UX), RF(R_T));
    M(p, RF(R_RDY), RF(R_UY), RF(R_T));
    M(p, RF(R_RDZ), RF(R_M17), RF(R_T));
    M(p, RF(R_BY), RF(R_OCY), RF(R_RDY));
    A(p, HS_QPU_A_MOV, RF(R_COL), RF(R_SKY), X);
    for (int i = 0; i < n_spheres; i++) {
        int s = R_SPH + 3 * i;
        M(p, R(0), RF(s), RF(R_RDX));
        M(p, R(1), RF(s + 1), RF(R_RDZ));
        A(p, HS_QPU_A_FADD, R(0), R(0), R(1));
        A(p, HS_QPU_A_FADD, R(0), R(0), RF(R_BY));              /* b = dot(oc, rd) */
        M(p, R(1), R(0), R(0));
        AF(p, HS_QPU_A_FSUB, R(1), R(1), RF(s + 2), HS_QPU_PF_PUSHN); /* h = b*b - cc; flag: h < 0 */
        A(p, HS_QPU_A_FMAX, R(2), R(1), RF(R_TINY));
        A(p, HS_QPU_A_RSQRT, RF(R_T), R(2), X);
        nop(p);
        M(p, R(1), R(1), RF(R_T));                              /* sqrt(h) */
        A(p, HS_QPU_A_FADD, R(1), R(0), R(1));                  /* u = b + sqrt(h) = -t */
        AC(p, HS_QPU_COND_IFA, R(1), RF(R_NBIG));               /* miss */
        AF(p, HS_QPU_A_FADD, R(2), R(1), RF(R_EPS), HS_QPU_PF_PUSHN); /* flag: t > eps */
        AC(p, HS_QPU_COND_IFNA, R(1), RF(R_NBIG));
        AF(p, HS_QPU_A_FSUB, R(2), RF(R_NBEST), R(1), HS_QPU_PF_PUSHN); /* flag: nearer than the best so far */
        hs_qpu_alu x = hs_qpu_a(HS_QPU_A_MOV, RF(R_NBEST), R(1), X);
        x.ac = HS_QPU_COND_IFA;
        x = hs_qpu_m(x, HS_QPU_M_MOV, RF(R_COL), RF(R_SCOL + i), X);
        x.mc = HS_QPU_COND_IFA;
        emit(p, x);
    }
    if (!no_ground) {
    /* the ground: t = -2.05 / rd.y */
    A(p, HS_QPU_A_RECIP, RF(R_T), RF(R_RDY), X);
    nop(p);
    M(p, R(0), RF(R_GNUM), RF(R_T));
    AF(p, HS_QPU_A_FSUB, R(1), R(0), RF(R_EPS), HS_QPU_PF_PUSHN);
    AC(p, HS_QPU_COND_IFA, R(0), RF(R_BIGP));
    M(p, R(1), RF(R_RDX), R(0));                                /* p.x (ro.x = 0) */
    M(p, R(2), RF(R_RDZ), R(0));
    A(p, HS_QPU_A_FADD, R(2), R(2), RF(R_ROZ));                 /* p.z */
    A(p, HS_QPU_A_FFLOOR, R(1), R(1), X);
    A(p, HS_QPU_A_FFLOOR, R(2), R(2), X);
    A(p, HS_QPU_A_FADD, R(1), R(1), R(2));
    A(p, HS_QPU_A_FTOIZ, R(1), R(1), X);
    AF(p, HS_QPU_A_FADD, R(3), R(0), RF(R_NBEST), HS_QPU_PF_PUSHN); /* flag: the ground is nearer */
    AF(p, HS_QPU_A_BAND, R(1), R(1), I(1), HS_QPU_PF_PUSHZ);     /* flag A: even square; B: ground nearer */
    A(p, HS_QPU_A_MOV, R(2), RF(R_GODD), X);
    AC(p, HS_QPU_COND_IFA, R(2), RF(R_GEVEN));
    AC(p, HS_QPU_COND_IFB, RF(R_COL), R(2));
    }
    /* 16 pixels out */
    A(p, HS_QPU_A_MOV, HS_QPU_TMUD, RF(R_COL), X);
    emit(p, hs_qpu_m(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(R_OUT), X), HS_QPU_M_ADD, RF(R_OUT), RF(R_OUT), RF(R_64)));
    /* The store is waited for before the pass ends, because the next pass
     * starts with a thread switch. (The first version left up to 50 stores
     * un-waited across thread switches; every job hung until the kernel
     * reset it. Whether that combination is the cause is the hypothesis this
     * change tests.) */
    if (!no_tmuwt) A(p, HS_QPU_A_TMUWT, X, X, X);
    A(p, HS_QPU_A_FADD, RF(R_UX), RF(R_UX), RF(R_DUX16));
    AF(p, HS_QPU_A_SUB, RF(R_VEC), RF(R_VEC), I(1), HS_QPU_PF_PUSHZ);
    br(p, HS_QPU_B_NA0, l_vec);
    n_instr_vec = (int)p->n - i0;
    /* ---- end of row ---- */
    A(p, HS_QPU_A_FADD, RF(R_UY), RF(R_UY), RF(R_DUY));
    A(p, HS_QPU_A_FADD, RF(R_V), RF(R_V), RF(R_DV));
    AF(p, HS_QPU_A_SUB, RF(R_ROWS), RF(R_ROWS), I(1), HS_QPU_PF_PUSHZ);
    br(p, HS_QPU_B_NA0, l_row);
    hs_qpu_emit_end(p);
    return hs_qpu_prog_finish(p);
}

/* ---- the scene's constants for time t, shared by the kernel and the reference ---- */
typedef struct {
    float ux0, dux, dux16, uy0, duy, duyb, v0, dv, dvb;
    float eps, nbig, tiny, ocy, c289, m17, gnum, roz, f255, one, bigp;
    float ocx[NSPH], ocz[NSPH], cc[NSPH], skya[3], skyb[3];
    uint32_t scol[NSPH], geven, godd, rows_per_batch;
} scene;

static uint32_t pack(float r, float g, float b) {
    return (uint32_t)lrintf(r * 255.0f) << 16 | (uint32_t)lrintf(g * 255.0f) << 8 | (uint32_t)lrintf(b * 255.0f);
}
static void scene_at(scene* s, float t, uint32_t batches) {
    const float aspect = (float)W / (float)H, PI = 3.14159265f;
    s->dux = 2.0f * aspect / (float)W;
    s->ux0 = (1.0f / (float)W - 1.0f) * aspect;
    s->dux16 = 16.0f * s->dux;
    s->duy = -2.0f / (float)H; /* row 0 is the top of the image: uy = +1 - 1/H */
    s->uy0 = 1.0f - 1.0f / (float)H;
    s->rows_per_batch = H / batches;
    s->duyb = (float)s->rows_per_batch * s->duy;
    s->dv = -1.0f / (float)H;  /* v = (uy + 1) / 2, GL's v_uv.y */
    s->v0 = 1.0f - 0.5f / (float)H;
    s->dvb = (float)s->rows_per_batch * s->dv;
    s->eps = 0.001f, s->nbig = -1e9f, s->bigp = 1e9f, s->tiny = 1e-30f;
    const float roy = 1.1f, roz = 4.3f;
    s->ocy = roy, s->roz = roz, s->c289 = 1.7f * 1.7f, s->m17 = -1.7f, s->gnum = -0.95f - roy, s->f255 = 255.0f, s->one = 1.0f;
    for (int i = 0; i < NSPH; i++) {
        float cx = 0, cz = 0, r = 0.85f, col[3] = {0.85f, 0.85f, 0.9f};
        if (i < 5) {
            float a = (2.0f * PI * (float)i) / 5.0f + t * 0.6f, fi = (float)i / 5.0f;
            cx = cosf(a) * 1.7f, cz = sinf(a) * 1.7f, r = 0.5f;
            col[0] = 0.5f + 0.5f * cosf(2.0f * PI * (fi + 0.0f));
            col[1] = 0.5f + 0.5f * cosf(2.0f * PI * (fi + 0.33f));
            col[2] = 0.5f + 0.5f * cosf(2.0f * PI * (fi + 0.67f));
        }
        float ocx = 0.0f - cx, ocz = roz - cz;
        s->ocx[i] = ocx, s->ocz[i] = ocz;
        s->cc[i] = (ocx * ocx + roy * roy + ocz * ocz) - r * r;
        s->scol[i] = pack(col[0], col[1], col[2]);
    }
    s->geven = pack(0.20f, 0.20f, 0.20f), s->godd = pack(0.35f, 0.35f, 0.35f);
    s->skya[0] = 0.55f, s->skya[1] = 0.22f, s->skya[2] = 0.35f;
    s->skyb[0] = 0.08f, s->skyb[1] = 0.10f, s->skyb[2] = 0.28f;
}
static void uniforms(uint32_t* u, const scene* s, uint32_t out_gpu, uint32_t batches) {
    float f[U_COUNT];
    memset(f, 0, sizeof f);
    f[U_UX0] = s->ux0, f[U_DUX] = s->dux, f[U_DUX16] = s->dux16, f[U_UY0] = s->uy0, f[U_DUY] = s->duy, f[U_DUYB] = s->duyb;
    f[U_V0] = s->v0, f[U_DV] = s->dv, f[U_DVB] = s->dvb, f[U_EPS] = s->eps, f[U_NBIG] = s->nbig, f[U_TINY] = s->tiny, f[U_OCY] = s->ocy;
    f[U_C289] = s->c289, f[U_M17] = s->m17, f[U_GNUM] = s->gnum, f[U_ROZ] = s->roz, f[U_255] = s->f255, f[U_ONE] = s->one, f[U_BIGP] = s->bigp;
    for (int i = 0; i < NSPH; i++) f[U_SPH + 3 * i] = s->ocx[i], f[U_SPH + 3 * i + 1] = s->ocz[i], f[U_SPH + 3 * i + 2] = s->cc[i];
    for (int c = 0; c < 3; c++) f[U_SKYA + c] = s->skya[c], f[U_SKYB + c] = s->skyb[c];
    memcpy(u, f, sizeof f);
    u[U_OUT] = out_gpu, u[U_BYTES] = W * 4 * s->rows_per_batch, u[U_ROWS] = s->rows_per_batch, u[U_VECS] = VECS;
    u[U_GEVEN] = s->geven, u[U_GODD] = s->godd;
    for (int i = 0; i < NSPH; i++) u[U_SCOL + i] = s->scol[i];
    (void)batches;
}

/* ---- the CPU reference: the kernel's operations, one by one ---- */
static float rsq(float x) { return (float)(1.0 / sqrt((double)x)); }
static float rcp(float x) { return 1.0f / x; }
static void reference(const scene* s, uint32_t batches, uint32_t* img, uint8_t* kind) {
    for (uint32_t b = 0; b < batches; b++) {
        float uy = s->uy0 + (float)b * s->duyb, v = s->v0 + (float)b * s->dvb;
        for (uint32_t row = 0; row < s->rows_per_batch; row++) {
            uint32_t y = b * s->rows_per_batch + row;
            float uysq = uy * uy + s->c289, omv = s->one - v;
            uint32_t sky = 0;
            for (int c = 0; c < 3; c++) {
                float a0 = s->skya[c] * omv, a1 = s->skyb[c] * v;
                float m = (a0 + a1) * s->f255;
                sky = sky << 8 | (uint32_t)lrintf(m);
            }
            float ux[16];
            for (int l = 0; l < 16; l++) ux[l] = s->ux0 + (float)l * s->dux;
            for (int vec = 0; vec < VECS; vec++) {
                for (int l = 0; l < 16; l++) {
                    float len2 = ux[l] * ux[l] + uysq, inv = rsq(len2);
                    float rdx = ux[l] * inv, rdy = uy * inv, rdz = s->m17 * inv, by = s->ocy * rdy;
                    float nbest = s->nbig;
                    uint32_t col = sky;
                    uint8_t k = 0;
                    for (int i = 0; i < NSPH; i++) {
                        float m0 = s->ocx[i] * rdx, m1 = s->ocz[i] * rdz;
                        float bb = (m0 + m1) + by;
                        float h = bb * bb - s->cc[i];
                        float hm = h > s->tiny ? h : s->tiny;
                        float u = bb + h * rsq(hm);
                        if (signbit(h)) u = s->nbig;
                        if (!signbit(u + s->eps)) u = s->nbig;
                        if (signbit(nbest - u)) nbest = u, col = s->scol[i], k = (uint8_t)(1 + i);
                    }
                    float tg = s->gnum * rcp(rdy);
                    if (signbit(tg - s->eps)) tg = s->bigp;
                    if (signbit(tg + nbest)) {
                        float px = rdx * tg, pz = rdz * tg + s->roz;
                        float f = floorf(px) + floorf(pz);
                        col = ((int32_t)f & 1) == 0 ? s->geven : s->godd;
                        k = 7;
                    }
                    img[(size_t)y * W + (size_t)vec * 16 + (size_t)l] = col;
                    kind[(size_t)y * W + (size_t)vec * 16 + (size_t)l] = k;
                    ux[l] = ux[l] + s->dux16;
                }
            }
            uy = uy + s->duy;
            v = v + s->dv;
        }
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

int main(int argc, char** argv) {
    uint32_t batches = argc > 1 ? (uint32_t)atoi(argv[1]) : 8;
    int frames = argc > 2 ? atoi(argv[2]) : 20;
    const char* ppm_dir = argc > 3 ? argv[3] : NULL;
    if ((batches != 1 && batches != 2 && batches != 4 && batches != 8) || frames < 1 || frames > 100000)
        return fprintf(stderr, "usage: %s [batches 1|2|4|8] [frames] [ppm_dir]\n", argv[0]), 2;
    if (getenv("P06_SPHERES")) n_spheres = atoi(getenv("P06_SPHERES"));
    no_ground = !!getenv("P06_NO_GROUND"), no_thrsw = !!getenv("P06_NO_THRSW"), no_tmuwt = !!getenv("P06_NO_TMUWT");
    int experiment = n_spheres != NSPH || no_ground || no_thrsw || no_tmuwt;
    if (n_spheres < 0 || n_spheres > NSPH) return 2;
    hs_v3d dev;
    hs_v3d_bo code, unif, out;
    if (hs_v3d_open(&dev, NULL) || hs_v3d_bo_new(&dev, &code, 16384) || hs_v3d_bo_new(&dev, &unif, 4096) ||
        hs_v3d_bo_new(&dev, &out, (size_t)W * H * 4 + 4096))
        return fprintf(stderr, "setup failed\n"), 1;
    static uint64_t buf[1024];
    hs_qpu_prog p;
    hs_qpu_prog_init(&p, buf, 1024);
    int len = build(&p);
    uint32_t prog = len > 0 ? hs_v3d_load(&code, 0, buf, (uint32_t)len) : 0;
    if (!prog) return fprintf(stderr, "build failed: %d\n", len), 1;
    if (getenv("P06_BUILD_ONLY")) /* no GPU job: the encoder's verdict on the program */
        return printf("{\"build_only\":true,\"instructions\":%d,\"per_16_pixels\":%d,\"rule_violation\":%d,\"at_instruction\":%u}\n", len,
                      n_instr_vec, p.viol, p.viol_at), 0;
    uint32_t* img = malloc((size_t)W * H * 4);
    uint8_t* kind = malloc((size_t)W * H);
    uint32_t* gpu = (uint32_t*)out.map;
    uint32_t h[3] = {code.handle, unif.handle, out.handle};
    hs_v3d_dispatch_desc d = {prog, unif.gpu, batches, h, 3, 0};
    int failures = 0;
    double best_ms = 1e30;
    printf("{\"scene\":\"stage 1: primary rays, flat colours\",\"width\":%d,\"height\":%d,\"batches\":%u,\"program_instructions\":%d,"
           "\"instructions_per_16_pixels\":%d}\n",
           W, H, batches, len, n_instr_vec);
    for (int f = 0; f < frames; f++) {
        float t = 10.0f * (float)f / (float)frames;
        scene s;
        scene_at(&s, t, batches);
        uniforms((uint32_t*)unif.map, &s, out.gpu, batches);
        memset(gpu, 0xEE, (size_t)W * H * 4);
        gpu[W * H] = 0xA5C3F00D; /* guard word after the frame */
        uint64_t t0 = now_ns();
        int rc = hs_v3d_dispatch(&dev, &d);
        if (!rc) rc = hs_v3d_bo_wait(&dev, &code, 3000000000ull);
        double ms = (now_ns() - t0) / 1e6;
        if (f > 0 && ms < best_ms) best_ms = ms;
        reference(&s, batches, img, kind);
        long wrong = 0, at_edge = 0, by_kind[8] = {0}, pix_kind[8] = {0};
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                size_t i = (size_t)y * W + (size_t)x;
                pix_kind[kind[i]]++;
                if (gpu[i] == img[i]) continue;
                wrong++;
                by_kind[kind[i]]++;
                /* an edge: a neighbour in the reference is a different object (or checker square) */
                int edge = 0;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++) {
                        int yy = y + dy, xx = x + dx;
                        if (yy >= 0 && yy < H && xx >= 0 && xx < W && img[(size_t)yy * W + (size_t)xx] != img[i]) edge = 1;
                    }
                at_edge += edge;
            }
        int guard_ok = gpu[W * H] == 0xA5C3F00D;
        int ok = !rc && guard_ok && wrong <= W * H / 200 && wrong == at_edge;
        printf("{\"frame\":%d,\"t\":%.2f,\"rc\":%d,\"ok\":%s,\"job_ms\":%.3f,\"pixels\":%d,\"exact\":%ld,\"mismatching\":%ld,"
               "\"mismatches_at_an_object_or_checker_edge\":%ld,\"guard_ok\":%s,\"reference_pixels\":{\"sky\":%ld,\"spheres\":%ld,\"ground\":%ld}}\n",
               f, (double)t, rc, ok ? "true" : "false", ms, W * H, (long)W * H - wrong, wrong, at_edge, guard_ok ? "true" : "false", pix_kind[0],
               pix_kind[1] + pix_kind[2] + pix_kind[3] + pix_kind[4] + pix_kind[5] + pix_kind[6], pix_kind[7]);
        if (!ok && !experiment) failures++;
        if (ppm_dir && (f == 0 || !ok)) {
            char name[64];
            snprintf(name, sizeof name, "scene_stage1_f%d_gpu", f);
            write_ppm(ppm_dir, name, gpu);
            snprintf(name, sizeof name, "scene_stage1_f%d_cpu", f);
            write_ppm(ppm_dir, name, img);
        }
        if (rc) break;
    }
    if (no_tmuwt) hs_v3d_bo_wait(&dev, &code, 1000000000ull);
    printf("{\"scene_summary\":\"stage 1\",\"experiment\":{\"spheres\":%d,\"ground\":%s,\"thrsw\":%s,\"tmuwt\":%s},\"instructions_per_16_pixels\":%d,\"batches\":%u,\"frames\":%d,\"failures\":%d,\"best_job_ms\":%.3f,\"ns_per_pixel_all_qpus\":%.2f,"
           "\"frames_per_second_at_that\":%.1f}\n",
           n_spheres, no_ground ? "false" : "true", no_thrsw ? "false" : "true", no_tmuwt ? "false" : "true", n_instr_vec,
           batches, frames, failures, best_ms, best_ms * 1e6 / (W * H), 1000.0 / best_ms);
    return failures != 0;
}
