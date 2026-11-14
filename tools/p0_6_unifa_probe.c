/* p0_6_unifa_probe: a second, rewindable stream of constants for a kernel.
 * (P0-6 plan step 4, docs/audit/P0_6_PLAN.md.)
 *
 * The scene kernel keeps its constants in registers and has none left. The
 * QPU has a second uniform stream: write an address to `unifa`, and the
 * ldunifa signal (to r5) or ldunifarf (to a register-file entry) then reads
 * successive words from it. A loop can write `unifa` again each pass and
 * read the same constants every time, on signals that ride on other
 * instructions: no registers held, no instructions spent. The encoder knows
 * the signals' bit patterns; what they do was not measured. This probe:
 *   read   per pass: unifa <- table; `d` other instructions; then 8 reads,
 *          each stored to memory (through r5 used `g` instructions later, or
 *          through ldunifarf); `passes` passes. Every stored word of every
 *          lane is compared with the table. For d = 0..3 and g = 0..1.
 *   switch the same with a thread switch between the unifa write and the
 *          reads, and between two reads.
 *   cost   1,000,000 passes of an arithmetic loop with and without
 *          (unifa write + 4 reads).
 *   branch (argument "branch": only these) a branch between the unifa write
 *          and the first read, as a loop over a list of record addresses
 *          needs: not taken, taken; the address written from r5 just loaded
 *          by ldunif; and the whole shape (ldunif, unifa <- r5, compare,
 *          branch, delay slots, read); and where the write may sit relative
 *          to a branch (g: instructions between, or the delay slot). No loop
 *          here depends on what is read: none of these can hang.
 *          Measured 2026-11-17 (unifa_branch.txt): a branch that is NOT
 *          taken spoils every later read when the unifa write is one of the
 *          3 instructions before it or in its first delay slot; a taken
 *          branch does not.
 * Usage: p0_6_unifa_probe [branch]   (inside the GPU lock, alone; each case is one job)
 * Output: JSON lines.
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
enum { NREAD = 8, PASSES = 4 };

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
static void store_rf(hs_qpu_prog* p, int rf) { /* store rf to rf1, advance rf1 */
    mov(p, HS_QPU_TMUD, RF(rf));
    emit(p, hs_qpu_m(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(1), X), HS_QPU_M_ADD, RF(1), RF(1), RF(4)));
    op(p, HS_QPU_A_TMUWT, X, X, X);
}

/* mode 0: ldunifa to r5, used after g instructions; 1: ldunifarf; 2: as 0 with
 * a thread switch after the unifa write; 3: as 0 with one between reads 3 and 4 */
static int build_read(hs_qpu_prog* p, int mode, int d, int g) {
    hs_qpu_prog_rules(p, 0);
    ldunifrf(p, 0); /* table address */
    ldunifrf(p, 1); /* out address */
    ldunifrf(p, 2); /* passes */
    op(p, HS_QPU_A_EIDX, R(1), X, X);
    op(p, HS_QPU_A_SHL, R(0), R(1), I(2));
    op(p, HS_QPU_A_ADD, RF(1), RF(1), R(0));
    mov(p, R(2), I(1));
    op(p, HS_QPU_A_SHL, RF(4), R(2), I(6));
    int l = hs_qpu_label_new(p);
    hs_qpu_label_bind(p, l);
    if (mode >= 14) {
        /* 14: unifa write, g other instructions, then a branch not taken. 15: a branch
         * not taken, the write in its delay slot g (1..3) or just after them (4). */
        hs_qpu_alu f = hs_qpu_a(HS_QPU_A_SUB, R(1), RF(2), I(0));
        f.af = HS_QPU_PF_PUSHZ;
        int lb = hs_qpu_label_new(p);
        if (mode == 14) {
            mov(p, HS_QPU_UNIFA, RF(0));
            for (int i = 0; i < g; i++) nop(p);
            emit(p, f);
            hs_qpu_branch(p, HS_QPU_B_A0, lb, 0, 0);
            nop(p), nop(p), nop(p);
        } else {
            emit(p, f);
            hs_qpu_branch(p, mode == 16 ? HS_QPU_B_NA0 : HS_QPU_B_A0, lb, 0, 0); /* 16: taken */
            for (int i = 1; i <= 3; i++) {
                if (i == g) mov(p, HS_QPU_UNIFA, RF(0));
                else nop(p);
            }
            if (g == 4) mov(p, HS_QPU_UNIFA, RF(0));
        }
        hs_qpu_label_bind(p, lb);
        g = 0;
    } else if (mode >= 10) { /* the unifa write in the branch's first delay slot; 12, 13: the address from r5 */
        if (mode >= 12) emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDUNIF, X));
        hs_qpu_alu f = mode >= 12 ? hs_qpu_a(HS_QPU_A_BXOR, R(1), R(5), RF(4)) : hs_qpu_a(HS_QPU_A_SUB, R(1), RF(2), I(0));
        f.af = HS_QPU_PF_PUSHZ;
        emit(p, f);
        int lb = hs_qpu_label_new(p);
        hs_qpu_branch(p, mode & 1 ? HS_QPU_B_NA0 : HS_QPU_B_A0, lb, 0, 0); /* odd modes: taken */
        mov(p, HS_QPU_UNIFA, mode >= 12 ? R(5) : RF(0));
        nop(p), nop(p);
        hs_qpu_label_bind(p, lb);
    } else if (mode >= 6) { /* the address comes from the first uniform stream, to r5 */
        emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDUNIF, X));
        mov(p, HS_QPU_UNIFA, R(5));
    } else {
        mov(p, HS_QPU_UNIFA, RF(0));
    }
    if (mode == 4 || mode == 5 || (mode >= 7 && mode <= 9)) {
        hs_qpu_alu f = mode == 9 ? hs_qpu_a(HS_QPU_A_BXOR, R(1), R(5), RF(4)) : hs_qpu_a(HS_QPU_A_SUB, R(1), RF(2), I(0));
        f.af = HS_QPU_PF_PUSHZ; /* never zero: the pass counter, or the address xor 64 */
        emit(p, f);
        int lb = hs_qpu_label_new(p);
        hs_qpu_branch(p, mode == 5 || mode == 7 ? HS_QPU_B_NA0 : HS_QPU_B_A0, lb, 0, 0); /* 5, 7: taken (to the next instruction after the slots) */
        nop(p), nop(p), nop(p);
        hs_qpu_label_bind(p, lb);
    }
    for (int i = 0; i < d; i++) nop(p);
    if (mode == 2) {
        emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_THRSW, X));
        nop(p), nop(p);
    }
    for (int k = 0; k < NREAD; k++) {
        if (mode == 3 && k == 4) {
            emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_THRSW, X));
            nop(p), nop(p);
        }
        if (mode == 1) {
            emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDUNIFARF, RF(6)));
        } else {
            emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDUNIFA, X));
            for (int i = 0; i < g; i++) nop(p);
            mov(p, RF(6), R(5));
        }
        store_rf(p, 6);
    }
    loop_end(p, 2, l);
    hs_qpu_emit_end(p);
    return hs_qpu_prog_finish(p);
}
/* cost: mode 0 a plain loop; 1 with a unifa write and 4 ldunifa reads on the same instructions */
static int build_cost(hs_qpu_prog* p, int mode, int d, int g) {
    (void)d, (void)g;
    hs_qpu_prog_rules(p, 0);
    ldunifrf(p, 0);
    ldunifrf(p, 1);
    ldunifrf(p, 2);
    mov(p, RF(7), I(0));
    int l = hs_qpu_label_new(p);
    hs_qpu_label_bind(p, l);
    if (mode) mov(p, HS_QPU_UNIFA, RF(0));
    else mov(p, RF(8), RF(0));
    nop(p), nop(p), nop(p);
    for (int k = 0; k < 4; k++) {
        hs_qpu_alu x = hs_qpu_a(HS_QPU_A_ADD, RF(7), RF(7), RF(2)); /* no small immediate: it has no encoding together with ldunifa */
        if (mode) x = hs_qpu_s(x, HS_QPU_SIG_LDUNIFA, X);
        emit(p, x);
        op(p, HS_QPU_A_ADD, RF(9), RF(7), R(5));
    }
    loop_end(p, 2, l);
    hs_qpu_emit_end(p);
    return hs_qpu_prog_finish(p);
}

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
static hs_v3d dev;
static hs_v3d_bo code, unif, table, out;
static int run(int (*build)(hs_qpu_prog*, int, int, int), int mode, int d, int g, uint32_t passes, double* ms) {
    static uint64_t buf[512];
    hs_qpu_prog p;
    hs_qpu_prog_init(&p, buf, 512);
    int len = build(&p, mode, d, g);
    uint32_t prog = len > 0 ? hs_v3d_load(&code, 0, buf, (uint32_t)len) : 0;
    if (!prog) return len < 0 ? len : -EINVAL;
    uint32_t* u = (uint32_t*)unif.map;
    u[0] = table.gpu, u[1] = out.gpu, u[2] = passes;
    for (uint32_t i = 0; i < 16; i++) u[3 + i] = table.gpu;
    uint32_t h[4] = {code.handle, unif.handle, table.handle, out.handle};
    hs_v3d_dispatch_desc dd = {prog, unif.gpu, 1, h, 4, 0};
    uint64_t t = now_ns();
    int e = hs_v3d_dispatch(&dev, &dd);
    if (!e) e = hs_v3d_bo_wait(&dev, &code, 3000000000ull);
    if (ms) *ms = (now_ns() - t) / 1e6;
    return e;
}

int main(int argc, char** argv) {
    int only_branch = argc > 1 && !strcmp(argv[1], "branch");
    if (hs_v3d_open(&dev, NULL) || hs_v3d_bo_new(&dev, &code, 4096) || hs_v3d_bo_new(&dev, &unif, 4096) ||
        hs_v3d_bo_new(&dev, &table, 4096) || hs_v3d_bo_new(&dev, &out, 65536))
        return fprintf(stderr, "setup failed\n"), 1;
    uint32_t* t = (uint32_t*)table.map;
    uint32_t* y = (uint32_t*)out.map;
    for (int i = 0; i < 64; i++) t[i] = 0xC0DE0000u + (uint32_t)i * 0x0101u;
    static const char* MODE[17] = {"ldunifa to r5", "ldunifarf", "thread switch after the unifa write", "thread switch between two reads",
                                   "a branch not taken after the unifa write", "a branch taken after the unifa write",
                                   "unifa written from r5 (ldunif the instruction before)", "from r5, then a branch taken",
                                   "from r5, then a branch not taken", "the list loop's shape: ldunif, unifa <- r5, xor, branch not taken",
                                   "unifa written in the first delay slot of a branch not taken", "in the first delay slot of a branch taken",
                                   "ldunif, xor, branch not taken, unifa <- r5 in its first delay slot", "ldunif, xor, branch taken, unifa <- r5 in its first delay slot",
                                   "unifa write, G instructions, a branch not taken, its slots", "a branch not taken, the unifa write in its slot G (4: after the slots)",
                                   "a branch taken, the unifa write in its slot G"};
    int failures = 0;
    for (int mode = only_branch ? 4 : 0; mode < 17; mode++)
        for (int d = 0; d <= 3; d++)
            for (int g = mode >= 15 ? 1 : 0; g <= (mode == 1 ? 0 : mode == 14 ? 8 : mode >= 15 ? 4 : 1); g++) {
                if ((mode == 2 || mode == 3) && (d != 3 || g != 0)) continue;
                if (mode >= 4 && mode < 14 && g != 0) continue;
                if (mode == 14 && d != 3) continue;
                if (mode >= 15 && d != 3 && g != 2) continue; /* slot 2 with fewer instructions after it, too */
                if (mode == 16 && g == 4) continue;            /* the write would be jumped over */
                memset(y, 0xEE, 65536);
                int e = run(build_read, mode, d, g, PASSES, NULL);
                int wrong = 0, first_bad = -1;
                uint32_t bad_val = 0;
                for (int ps = 0; ps < PASSES && !e; ps++)
                    for (int k = 0; k < NREAD; k++)
                        for (int l = 0; l < 16; l++) {
                            uint32_t v = y[(ps * NREAD + k) * 16 + l];
                            if (v != t[k]) {
                                if (first_bad < 0) first_bad = ps * NREAD + k, bad_val = v;
                                wrong++;
                            }
                        }
                printf("{\"unifa\":\"%s\",\"instructions_between_unifa_write_and_first_read\":%d,\"g\":%d,\"rc\":%d,"
                       "\"words\":%d,\"wrong\":%d,\"first_wrong_read\":%d,\"its_value\":\"0x%08x\"}\n",
                       MODE[mode], d, g, e, PASSES * NREAD * 16, wrong, first_bad, bad_val);
                fflush(stdout);
                if (e) return 1; /* a job that did not finish: stop, do not run more */
                if (wrong && !(mode <= 1 && d < 3) && mode < 4) failures++; /* the branch cases are measurements */
            }
    double with = 1e30, without = 1e30, ms;
    int ce = 0;
    for (int r = 0; r < 5 && !ce; r++) {
        if (!(ce = run(build_cost, 1, 0, 0, 1000000, &ms)) && ms < with) with = ms;
        if (!ce && !(ce = run(build_cost, 0, 0, 0, 1000000, &ms)) && ms < without) without = ms;
    }
    if (ce) return printf("{\"unifa_cost\":\"failed\",\"rc\":%d}\n", ce), 1;
    printf("{\"unifa_cost\":\"a unifa write and 4 ldunifa signals per pass\",\"pass_ns_with\":%.2f,\"pass_ns_without\":%.2f,\"extra_ns\":%.2f}\n", with,
           without, with - without);
    return failures != 0;
}
