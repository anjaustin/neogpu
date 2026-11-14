/* p0_5_inline_probe: can the message be the arithmetic?
 * (P0-5, docs/audit/HARDWARE_HANDOFF_P0.md.)
 *
 * hs_mailbox sends a request that points at buffers: the GPU reloads the
 * inputs with op 2, computes, publishes the outputs, then acknowledges. Here
 * the request block itself holds the operands and the acknowledgement block
 * holds the results:
 *   request block r (2048 bytes, read once with plain loads, so fresh):
 *     line 0  k/4      line 1  c1      line 2  c0
 *     lines 4+2b, 5+2b  batch b's 32 operands (one per lane)
 *   acknowledgement block of batch b: line 0 the sequence number, lines 1
 *     and 2 the 32 results; all three published with op 2.
 * The work is hs_mailbox's chain: y = x, then k times y = y * c1 + c0.
 * One job serves `messages` requests, polling the sequence word with op 2 as
 * hs_mailbox does. No yield words, stop flag, watchdog or thread switches in
 * the loop: this measures the form, alone on the GPU; it is not a library.
 *
 * Timed per round trip, as a caller would see it: write the block, raise the
 * sequence word, spin until every batch acknowledged, copy the results out.
 * Compared in the same run with one CPU core doing the same work (the
 * fastest NEON chain found, as in tests/test_mailbox.c).
 * Usage: p0_5_inline_probe <batches 1|8> [messages=2000]   (inside the GPU lock)
 * Output: one JSON line per k.
 */
#define _GNU_SOURCE
#include <arm_neon.h>
#include <errno.h>
#include <sched.h>
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

enum { OFF_M = 0, OFF_ACK = 4096, OFF_RING = 16384, BLOCK = 2048, POLLS = 3000000 };

static void emit(hs_qpu_prog* p, hs_qpu_alu x) { hs_qpu_emit(p, &x); }
static void nop(hs_qpu_prog* p) { emit(p, hs_qpu_alu_nop()); }
static void op(hs_qpu_prog* p, int o, hs_qpu_reg d, hs_qpu_reg a, hs_qpu_reg b) { emit(p, hs_qpu_a(o, d, a, b)); }
static void mov(hs_qpu_prog* p, hs_qpu_reg d, hs_qpu_reg s) { op(p, HS_QPU_A_MOV, d, s, X); }
static void ldunifrf(hs_qpu_prog* p, int rf) { emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDUNIFRF, RF(rf))); }
static void flags(hs_qpu_prog* p, int o, hs_qpu_reg d, hs_qpu_reg a, hs_qpu_reg b, int pf) {
    hs_qpu_alu s = hs_qpu_a(o, d, a, b);
    s.af = (uint8_t)pf;
    emit(p, s);
}
static void br(hs_qpu_prog* p, int cond, int label) {
    hs_qpu_branch(p, cond, label, 0, 0);
    nop(p), nop(p), nop(p);
}
static void store(hs_qpu_prog* p, hs_qpu_reg addr, hs_qpu_reg v) {
    mov(p, HS_QPU_TMUD, v);
    mov(p, HS_QPU_TMUA, addr);
}
static void op2(hs_qpu_prog* p, hs_qpu_reg addr) { /* publish + reload; takes the next uniform */
    mov(p, HS_QPU_TMUAU, addr);
    op(p, HS_QPU_A_TMUWT, X, X, X);
}
static void chain_i1(hs_qpu_prog* p) {
    emit(p, hs_qpu_m(hs_qpu_a(HS_QPU_A_FADD, RF(36), R(0), R(3)), HS_QPU_M_FMUL, R(1), RF(37), R(2)));
}
static void chain_i2(hs_qpu_prog* p) {
    emit(p, hs_qpu_m(hs_qpu_a(HS_QPU_A_FADD, RF(37), R(1), R(3)), HS_QPU_M_FMUL, R(0), RF(36), R(2)));
}

/* rf0 M+4l  rf1 ACK_b+4l  rf3 polls left  rf4 next sequence  rf5 lane-0 mask
 * rf7 block+4l  rf9 batch  rf10 messages left  rf17 64  rf21 offset of the
 * batch's operand lines  rf41 2048  rf29 k/4  rf30 c1  rf31 c0  rf36/rf37 the
 * two vectors */
static int build(hs_qpu_prog* p) {
    hs_qpu_prog_rules(p, 0); /* TMUAU and multi-load forms are outside the checker */
    op(p, HS_QPU_A_BAND, RF(9), RF(0), I(15)); /* the hardware's batch number */
    ldunifrf(p, 0);
    ldunifrf(p, 1);
    ldunifrf(p, 7);
    ldunifrf(p, 3);
    ldunifrf(p, 10);
    op(p, HS_QPU_A_EIDX, R(1), X, X);
    op(p, HS_QPU_A_SHL, R(0), R(1), I(2));
    op(p, HS_QPU_A_SUB, R(2), R(1), I(1));
    op(p, HS_QPU_A_ASR, RF(5), R(2), I(15));
    op(p, HS_QPU_A_SHL, R(2), RF(9), I(8));
    op(p, HS_QPU_A_ADD, RF(1), RF(1), R(2));
    op(p, HS_QPU_A_ADD, RF(1), RF(1), R(0));
    op(p, HS_QPU_A_ADD, RF(0), RF(0), R(0));
    op(p, HS_QPU_A_ADD, RF(7), RF(7), R(0));
    mov(p, R(2), I(1));
    op(p, HS_QPU_A_SHL, RF(17), R(2), I(6));
    op(p, HS_QPU_A_SHL, RF(41), R(2), I(11));
    op(p, HS_QPU_A_SHL, R(3), R(2), I(8));
    op(p, HS_QPU_A_SHL, R(2), RF(9), I(7));
    op(p, HS_QPU_A_ADD, RF(21), R(2), R(3)); /* (4 + 2b) * 64 */
    mov(p, RF(4), I(1));
    int poll = hs_qpu_label_new(p), done = hs_qpu_label_new(p), out = hs_qpu_label_new(p), l_hl = hs_qpu_label_new(p);

    hs_qpu_label_bind(p, poll);
    op2(p, RF(0));
    emit(p, hs_qpu_s(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(0), X), HS_QPU_SIG_THRSW, X));
    nop(p), nop(p);
    emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDTMU, R(0)));
    flags(p, HS_QPU_A_SUB, RF(3), RF(3), I(1), HS_QPU_PF_PUSHZ);
    br(p, HS_QPU_B_A0, done);
    op(p, HS_QPU_A_SUB, R(1), R(0), RF(4));
    flags(p, HS_QPU_A_BAND, R(1), R(1), RF(5), HS_QPU_PF_PUSHN);
    br(p, HS_QPU_B_ANYA, poll); /* lane 0: M < next sequence */

    /* the request: three header lines and this batch's two operand lines */
    op(p, HS_QPU_A_ADD, R(1), RF(7), RF(17));
    op(p, HS_QPU_A_ADD, R(2), R(1), RF(17));
    op(p, HS_QPU_A_ADD, R(3), RF(7), RF(21));
    mov(p, HS_QPU_TMUA, RF(7));
    mov(p, HS_QPU_TMUA, R(1));
    mov(p, HS_QPU_TMUA, R(2));
    emit(p, hs_qpu_m(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, R(3), X), HS_QPU_M_ADD, R(3), R(3), RF(17)));
    emit(p, hs_qpu_s(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, R(3), X), HS_QPU_SIG_THRSW, X));
    nop(p), nop(p);
    emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDTMU, RF(29)));
    emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDTMU, RF(30)));
    emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDTMU, RF(31)));
    emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDTMU, RF(36)));
    emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDTMU, RF(37)));
    flags(p, HS_QPU_A_MOV, R(1), RF(29), X, HS_QPU_PF_PUSHZ);
    br(p, HS_QPU_B_A0, out); /* k = 0: the operands come back unchanged */
    mov(p, R(2), RF(30));
    mov(p, R(3), RF(31));
    emit(p, hs_qpu_m(hs_qpu_a(HS_QPU_A_MOV, RF(38), RF(29), X), HS_QPU_M_FMUL, R(0), RF(36), R(2)));
    hs_qpu_label_bind(p, l_hl);
    chain_i1(p), chain_i2(p), chain_i1(p), chain_i2(p), chain_i1(p);
    flags(p, HS_QPU_A_SUB, RF(38), RF(38), I(1), HS_QPU_PF_PUSHZ);
    hs_qpu_branch(p, HS_QPU_B_NA0, l_hl, 0, 0);
    chain_i2(p), chain_i1(p), chain_i2(p);

    /* results beside the acknowledgement, then the acknowledgement */
    hs_qpu_label_bind(p, out);
    op(p, HS_QPU_A_ADD, R(1), RF(1), RF(17));
    op(p, HS_QPU_A_ADD, R(2), R(1), RF(17));
    store(p, R(1), RF(36));
    store(p, R(2), RF(37));
    store(p, RF(1), RF(4));
    op(p, HS_QPU_A_TMUWT, X, X, X);
    op2(p, R(1));
    op2(p, R(2));
    op2(p, RF(1));
    op(p, HS_QPU_A_ADD, RF(4), RF(4), I(1));
    op(p, HS_QPU_A_ADD, RF(7), RF(7), RF(41));
    flags(p, HS_QPU_A_SUB, RF(10), RF(10), I(1), HS_QPU_PF_PUSHZ);
    br(p, HS_QPU_B_NA0, poll);
    hs_qpu_label_bind(p, done);
    hs_qpu_emit_end(p);
    return hs_qpu_prog_finish(p);
}

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
static void barrier(void) { __asm__ volatile("dsb sy" ::: "memory"); }
static int cmpd(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}
static float chain_ref(float x, uint32_t k, float c1, float c0) {
    volatile float y = x;
    for (uint32_t i = 0; i < k; i++) {
        float m = y * c1;
        y = m + c0;
    }
    return y;
}
__attribute__((noinline)) static void cpu_chain_neon(const float* x, float* y, uint32_t n, uint32_t k, float c1, float c0) {
    float32x4_t vc1 = vdupq_n_f32(c1), vc0 = vdupq_n_f32(c0);
    for (uint32_t i = 0; i < n; i += 32) {
        float32x4_t v[8];
        for (int j = 0; j < 8; j++) v[j] = vld1q_f32(x + i + 4 * j);
        for (uint32_t t = 0; t < k; t++) {
            _Pragma("GCC unroll 8") for (int j = 0; j < 8; j++) v[j] = vaddq_f32(vmulq_f32(v[j], vc1), vc0);
        }
        for (int j = 0; j < 8; j++) vst1q_f32(y + i + 4 * j, v[j]);
    }
}

int main(int argc, char** argv) {
    uint32_t batches = argc > 1 ? (uint32_t)atoi(argv[1]) : 1, msgs = argc > 2 ? (uint32_t)atoi(argv[2]) : 2000;
    if ((batches != 1 && batches != 8) || msgs < 10 || msgs > 8000) return fprintf(stderr, "usage: %s <1|8> [messages 10..8000]\n", argv[0]), 2;
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(2, &cs);
    sched_setaffinity(0, sizeof cs, &cs);
    static uint64_t buf[512];
    hs_v3d dev;
    hs_v3d_bo code, unif, ctl;
    size_t nunif = 5 + POLLS + 3 * (size_t)msgs + 64;
    if (hs_v3d_open(&dev, NULL) || hs_v3d_bo_new(&dev, &code, 8192) || hs_v3d_bo_new(&dev, &unif, nunif * 4) ||
        hs_v3d_bo_new(&dev, &ctl, OFF_RING + (size_t)msgs * BLOCK))
        return fprintf(stderr, "setup failed\n"), 1;
    hs_qpu_prog p;
    hs_qpu_prog_init(&p, buf, 512);
    int len = build(&p);
    uint32_t prog = len > 0 ? hs_v3d_load(&code, 0, buf, (uint32_t)len) : 0;
    if (!prog) return fprintf(stderr, "build failed: %d\n", len), 1;
    uint32_t* u = (uint32_t*)unif.map;
    u[0] = ctl.gpu + OFF_M, u[1] = ctl.gpu + OFF_ACK, u[2] = ctl.gpu + OFF_RING, u[3] = POLLS, u[4] = msgs;
    for (size_t i = 5; i < nunif; i++) u[i] = 0xFFFFFF80u | 2u << 3 | 7u; /* per-lane op 2, 32-bit */
    uint32_t h[3] = {code.handle, unif.handle, ctl.handle}, n = 32 * batches;
    volatile uint32_t* c = (volatile uint32_t*)ctl.map;
    float* x = aligned_alloc(64, n * 4);
    float* y = aligned_alloc(64, n * 4);
    float* yc = aligned_alloc(64, n * 4);
    double* w = malloc(sizeof(double) * msgs);
    double* wc = malloc(sizeof(double) * msgs);
    float c1 = 0.9995f, c0 = 0.0007f;
    uint32_t c1b, c0b, rng = 2463534242u;
    memcpy(&c1b, &c1, 4);
    memcpy(&c0b, &c0, 4);
    static const uint32_t ks[] = {0, 4, 16, 64, 256, 1024, 4096};
    int failures = 0;
    for (unsigned ki = 0; ki < sizeof ks / sizeof ks[0]; ki++) {
        uint32_t k = ks[ki];
        memset(ctl.map, 0, OFF_RING);
        barrier();
        hs_v3d_dispatch_desc d = {prog, unif.gpu, batches, h, 3, 0};
        int e = hs_v3d_dispatch(&dev, &d);
        if (e) return fprintf(stderr, "dispatch: %s\n", strerror(-e)), 1;
        int bad = 0, timeouts = 0, checked = 0;
        for (uint32_t r = 0; r < msgs && !timeouts; r++) {
            for (uint32_t i = 0; i < n; i++) {
                rng ^= rng << 13, rng ^= rng >> 17, rng ^= rng << 5;
                x[i] = (float)(rng % 1000000) / 1e6f;
            }
            uint64_t t = now_ns();
            volatile uint32_t* blk = c + (OFF_RING + (size_t)r * BLOCK) / 4;
            for (int l = 0; l < 16; l++) blk[l] = k / 4, blk[16 + l] = c1b, blk[32 + l] = c0b;
            memcpy((void*)(blk + 64), x, n * 4);
            barrier();
            c[0] = r + 1;
            barrier();
            for (;;) {
                uint32_t ok = 0;
                for (uint32_t b = 0; b < batches; b++) ok += c[(OFF_ACK + 256 * b) / 4] == r + 1;
                if (ok == batches) break;
                if (now_ns() - t > 2000000000ull) {
                    timeouts++;
                    break;
                }
            }
            for (uint32_t b = 0; b < batches; b++) memcpy(y + 32 * b, (void*)(c + (OFF_ACK + 256 * b + 64) / 4), 128);
            w[r] = (now_ns() - t) / 1e3;
            if (r % 64 == 0) { /* the job keeps polling meanwhile: check sparsely */
                for (uint32_t i = 0; i < n; i++) {
                    float ref = chain_ref(x[i], k, c1, c0);
                    bad += memcmp(&ref, &y[i], 4) != 0;
                }
                checked++;
            }
        }
        if (timeouts) memset(w, 0, sizeof(double) * msgs);
        e = hs_v3d_bo_wait(&dev, &ctl, 3000000000ull);
        /* the same work on one CPU core, after the job: not while it polls */
        for (uint32_t r = 0; r < msgs; r++) {
            uint64_t tc = now_ns();
            cpu_chain_neon(x, yc, n, k, c1, c0);
            wc[r] = (now_ns() - tc) / 1e3;
        }
        for (uint32_t i = 0; i < n; i++) {
            float ref = chain_ref(x[i], k, c1, c0);
            bad += memcmp(&ref, &yc[i], 4) != 0;
        }
        /* the first messages include the job's start: leave out the first 10 */
        qsort(w + 10, msgs - 10, sizeof(double), cmpd);
        qsort(wc + 10, msgs - 10, sizeof(double), cmpd);
        uint32_t m = msgs - 10;
        printf("{\"inline\":true,\"batches\":%u,\"elements\":%u,\"k\":%u,\"messages\":%u,\"checked\":%d,\"rc\":%d,\"timeouts\":%d,\"wrong\":%d,"
               "\"round_trip_us\":{\"p50\":%.2f,\"p99\":%.2f},\"cpu_1_core_us\":{\"p50\":%.2f,\"p99\":%.2f}}\n",
               batches, n, k, msgs, checked, e, timeouts, bad, w[10 + m / 2], w[10 + m * 99 / 100], wc[10 + m / 2], wc[10 + m * 99 / 100]);
        fflush(stdout);
        if (e || timeouts || bad) failures++;
        if (e || timeouts) break;
    }
    hs_v3d_bo_free(&dev, &ctl);
    hs_v3d_bo_free(&dev, &unif);
    hs_v3d_bo_free(&dev, &code);
    hs_v3d_close(&dev);
    return failures != 0;
}
