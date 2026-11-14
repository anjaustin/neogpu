/* p0_5_batch_cache_probe: within one compute job, does a batch's plain load
 * see memory another batch already read, or fresh memory?
 * (P0-5 red team finding 16, docs/audit/HARDWARE_HANDOFF_P0.md.)
 *
 * hs_mailbox reads request slots with plain loads. If the GPU's read cache
 * is shared by the batches of a job, a slot one batch has read can be served
 * stale to another batch after the CPU rewrote it.
 *
 * One 8-batch job per trial. Block D starts as 0x11111111. Some batches
 * "pre-read" D with a plain load and publish what they saw. The CPU then
 * writes 0x22222222 to D and raises a go word (polled with op 2). Every batch
 * then loads D again and publishes the value:
 *   variant 0  nobody pre-reads                      (control: expect fresh)
 *   variant 1  batch 0 pre-reads                     (the question: batches 1..7)
 *   variant 2  every batch pre-reads                 (control: expect stale)
 *   variant 3  batch 0 pre-reads; the second load of
 *              every batch is preceded by op 2       (control: expect fresh)
 * In variants 1 and 3 batch 0's own second load is a same-batch control.
 * Usage: p0_5_batch_cache_probe <trials>      (inside the GPU lock)
 * Output: one JSON line per variant.
 */
#define _GNU_SOURCE
#include <errno.h>
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

enum { OFF_G = 0, OFF_D = 4096, OFF_R = 8192, BUDGET = 400000, OLD = 0x11111111, NEW = 0x22222222 };

static void emit(hs_qpu_prog* p, hs_qpu_alu x) { hs_qpu_emit(p, &x); }
static void nop(hs_qpu_prog* p) { emit(p, hs_qpu_alu_nop()); }
static void op(hs_qpu_prog* p, int o, hs_qpu_reg d, hs_qpu_reg a, hs_qpu_reg b) { emit(p, hs_qpu_a(o, d, a, b)); }
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
static void load(hs_qpu_prog* p, hs_qpu_reg addr, hs_qpu_reg dst) {
    emit(p, hs_qpu_s(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, addr, X), HS_QPU_SIG_THRSW, X));
    nop(p), nop(p);
    emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDTMU, dst));
}
static void store(hs_qpu_prog* p, hs_qpu_reg addr, hs_qpu_reg v) {
    op(p, HS_QPU_A_MOV, HS_QPU_TMUD, v, X);
    op(p, HS_QPU_A_MOV, HS_QPU_TMUA, addr, X);
}
static void op2(hs_qpu_prog* p, hs_qpu_reg addr) { /* publish + reload; takes the next uniform */
    op(p, HS_QPU_A_MOV, HS_QPU_TMUAU, addr, X);
    op(p, HS_QPU_A_TMUWT, X, X, X);
}

/* rf0 G+4l, rf1 D+4l, rf2 R_b+4l (lines: second load, tidx, pre-read), rf3
 * budget, rf5 lane-0 mask, rf6 1, rf7 64, rf9 batch */
static int build(hs_qpu_prog* p, int variant) {
    hs_qpu_prog_rules(p, 0); /* TMUAU is outside the checker */
    op(p, HS_QPU_A_BAND, RF(9), RF(0), I(15)); /* the hardware's batch number */
    for (int i = 0; i < 4; i++) ldunifrf(p, i);
    op(p, HS_QPU_A_EIDX, R(1), X, X);
    op(p, HS_QPU_A_SHL, R(0), R(1), I(2));
    op(p, HS_QPU_A_SUB, R(2), R(1), I(1));
    op(p, HS_QPU_A_ASR, RF(5), R(2), I(15));
    op(p, HS_QPU_A_MOV, RF(6), I(1), X);
    op(p, HS_QPU_A_SHL, RF(7), RF(6), I(6));
    op(p, HS_QPU_A_SHL, R(2), RF(9), I(8));
    op(p, HS_QPU_A_ADD, RF(2), RF(2), R(2));
    for (int i = 0; i < 3; i++) op(p, HS_QPU_A_ADD, RF(i), RF(i), R(0));
    int skip = hs_qpu_label_new(p), poll = hs_qpu_label_new(p), done = hs_qpu_label_new(p);
    if (variant) {
        if (variant != 2) { /* only batch 0 pre-reads */
            flags(p, HS_QPU_A_MOV, R(1), RF(9), X, HS_QPU_PF_PUSHZ);
            br(p, HS_QPU_B_NA0, skip);
        }
        load(p, RF(1), R(2));
        op(p, HS_QPU_A_ADD, R(3), RF(2), RF(7));
        op(p, HS_QPU_A_ADD, R(3), R(3), RF(7));
        store(p, R(3), R(2));
        op(p, HS_QPU_A_TMUWT, X, X, X);
        op2(p, R(3));
    }
    hs_qpu_label_bind(p, skip);
    hs_qpu_label_bind(p, poll);
    op2(p, RF(0));
    load(p, RF(0), R(0));
    flags(p, HS_QPU_A_SUB, RF(3), RF(3), I(1), HS_QPU_PF_PUSHZ);
    hs_qpu_branch(p, HS_QPU_B_A0, done, 0, 0);
    op(p, HS_QPU_A_SUB, R(1), R(0), RF(6));
    flags(p, HS_QPU_A_BAND, R(1), R(1), RF(5), HS_QPU_PF_PUSHN);
    nop(p);
    br(p, HS_QPU_B_ANYA, poll);
    if (variant == 3) op2(p, RF(1));
    load(p, RF(1), R(2));
    store(p, RF(2), R(2));
    op(p, HS_QPU_A_TIDX, R(1), X, X);
    op(p, HS_QPU_A_ADD, R(3), RF(2), RF(7));
    store(p, R(3), R(1));
    op(p, HS_QPU_A_TMUWT, X, X, X);
    op2(p, RF(2));
    op2(p, R(3));
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

int main(int argc, char** argv) {
    int trials = argc > 1 ? atoi(argv[1]) : 200;
    static uint64_t buf[512];
    hs_v3d dev;
    hs_v3d_bo code, unif, data;
    if (hs_v3d_open(&dev, NULL) || hs_v3d_bo_new(&dev, &code, 16384) || hs_v3d_bo_new(&dev, &unif, (BUDGET + 64) * 4) ||
        hs_v3d_bo_new(&dev, &data, 16384)) {
        fprintf(stderr, "setup failed\n");
        return 2;
    }
    uint32_t prog[4];
    for (int v = 0; v < 4; v++) {
        hs_qpu_prog p;
        hs_qpu_prog_init(&p, buf, 512);
        int n = build(&p, v);
        if (n < 0 || !(prog[v] = hs_v3d_load(&code, (size_t)v * 4096, buf, (uint32_t)n))) {
            fprintf(stderr, "build %d failed: %d\n", v, n);
            return 2;
        }
    }
    uint32_t* u = (uint32_t*)unif.map;
    u[0] = data.gpu + OFF_G, u[1] = data.gpu + OFF_D, u[2] = data.gpu + OFF_R, u[3] = BUDGET;
    for (int i = 4; i < BUDGET + 64; i++) u[i] = 0xFFFFFF80u | 2u << 3 | 7u;
    volatile uint32_t* G = (volatile uint32_t*)(data.map + OFF_G);
    volatile uint32_t* D = (volatile uint32_t*)(data.map + OFF_D);
    volatile uint32_t* Rr = (volatile uint32_t*)(data.map + OFF_R);
    uint32_t h[3] = {code.handle, unif.handle, data.handle};
    static const char* what[4] = {"nobody pre-reads", "batch 0 pre-reads", "every batch pre-reads", "batch 0 pre-reads, op 2 before the second load"};
    int rc = 0;
    for (int v = 0; v < 4; v++) {
        long fresh0 = 0, stale0 = 0, fresh = 0, stale = 0, other = 0, skipped = 0, same_thread_stale = 0, other_thread_stale = 0;
        for (int t = 0; t < trials; t++) {
            for (int i = 0; i < 16; i++) D[i] = OLD;
            for (int i = 0; i < 8 * 64; i++) Rr[i] = 0xDEADBEEF;
            G[0] = 0;
            barrier();
            hs_v3d_dispatch_desc d = {prog[v], unif.gpu, 8, h, 3, 0};
            if (hs_v3d_dispatch(&dev, &d)) {
                fprintf(stderr, "dispatch failed\n");
                return 2;
            }
            /* wait for the pre-reads to be published */
            uint64_t t0 = now_ns();
            int ready = 0;
            while (!ready && now_ns() - t0 < 50000000ull) {
                ready = 1;
                if (v == 0) ready = now_ns() - t0 > 200000;
                else
                    for (int b = 0; b < (v == 2 ? 8 : 1); b++) ready &= Rr[b * 64 + 32] == OLD;
            }
            for (int i = 0; i < 16; i++) D[i] = NEW;
            barrier();
            G[0] = 1;
            barrier();
            if (hs_v3d_bo_wait(&dev, &data, 5000000000ull)) {
                fprintf(stderr, "job did not end\n");
                return 2;
            }
            if (!ready) {
                skipped++;
                continue;
            }
            uint32_t t0idx = Rr[16] >> 2; /* batch 0's thread slot */
            for (int b = 0; b < 8; b++) {
                uint32_t val = Rr[b * 64];
                long* f = b ? &fresh : &fresh0;
                long* s = b ? &stale : &stale0;
                if (val == NEW) (*f)++;
                else if (val == OLD) {
                    (*s)++;
                    if (b && v == 1) *(Rr[b * 64 + 16] >> 2 == t0idx ? &same_thread_stale : &other_thread_stale) += 1;
                } else
                    other++;
            }
        }
        printf("{\"variant\":%d,\"what\":\"%s\",\"trials\":%d,\"skipped\":%ld,\"batch0\":{\"fresh\":%ld,\"stale\":%ld},"
               "\"batches1to7\":{\"fresh\":%ld,\"stale\":%ld},\"neither\":%ld",
               v, what[v], trials, skipped, fresh0, stale0, fresh, stale, other);
        if (v == 1) printf(",\"stale_on_batch0s_thread\":%ld,\"stale_on_another_thread\":%ld", same_thread_stale, other_thread_stale);
        printf("}\n");
        if (other || skipped) rc = 1;
    }
    return rc;
}
