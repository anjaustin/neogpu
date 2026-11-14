/* hs_mailbox: see include/hs_mailbox.h. Register use of the resident program
 * (per batch, 16 lanes; l = lane, b = batch):
 *   rf0 M+4l   rf1 ACK_b+4l   rf3 budget   rf4 next sequence   rf5 lane-0 mask
 *   rf6 STATE_b+4l (stop flag)   rf7 SLOT0+4l   rf8 SLOTEND+4l   rf9 b
 *   rf11 slot+4l   rf12 poll subtrahend (lane 0: next sequence; lanes 1..4:
 *   1)   rf13 ~0 on lanes 5..15   rf14 leases   rf16 STATE_b+128+4l (lease
 *   count)   rf17 64   rf18 1 on lanes 1..4   rf19 slots-1   rf20 ~0 on lane 3
 *   rf22 64*batches (chunk stride)   rf23 64*b+4l   rf24..rf31 request lines
 *   rf33..rf43 op temporaries
 *
 * The sequence line M: word 0 is the last sequence sent. Words 1..3 are
 * yield words (word 4 is never written):
 *   1  set by the caller's thread: end the lease when no request is pending
 *   2  set by the watchdog when the caller is idle: the same
 *   3  set by the watchdog when a lease is too old: end it now, even with
 *      requests pending (the next lease serves them)
 *
 * What survives a lease: only memory the GPU published. A batch's ACK line is
 * its whole durable state: a lease starts at ACK + 1 and takes its slot from
 * that number, so a lease that dies (kernel reset) is resumed by the next one
 * at the first request it had not acknowledged. Requests may therefore run
 * more than once, which is why send refuses outputs that overlap inputs.
 */
#define _GNU_SOURCE
#include "hs_mailbox.h"

#include <errno.h>
#include <sched.h>
#include <string.h>
#include <time.h>

#include "hs_qpu.h"

#define R(n) HS_QPU_R(n)
#define RF(n) HS_QPU_RF(n)
#define X HS_QPU_NOREG
#define I(v) hs_qpu_imm_int(v)

enum {
    OFF_M = 0,        /* word 0: last sequence sent; words 1..3: yield words (see above) */
    OFF_ACK = 4096,   /* + 256 * batch */
    OFF_STATE = 8192, /* + 256 * batch: line 0 stop flag, line 2 leases finished */
    OFF_WARM = 12288,
    OFF_SLOTS = 16384,
    SLOT_BYTES = 512, /* 8 lines; a multiple of the GPU's 256-byte fetch */
    MAX_UNITS = HS_MB_MAX_UNITS,
    HEADER = 9,
    NO_STRETCH = 255,
    STALL_NS = 300000,     /* a request some batches acknowledged and others have not: yield */
    LOST_CHECK_NS = 20000000, /* the check blocks ~2 ms while a job runs: only on a long stall */
    WD_TICK_NS = 1000000,
    WD_RUN = 0,    /* leases may be running; no idle yield */
    WD_IDLE = 1,   /* the idle yield (word 2) is set */
    WD_PARKED = 2, /* nothing queued: the watchdog sleeps until the caller returns */
};

static void emit(hs_qpu_prog* p, hs_qpu_alu x) { hs_qpu_emit(p, &x); }
static void nop(hs_qpu_prog* p) { emit(p, hs_qpu_alu_nop()); }
static void ldunifrf(hs_qpu_prog* p, int rf) { emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDUNIFRF, RF(rf))); }
static void op(hs_qpu_prog* p, int o, hs_qpu_reg d, hs_qpu_reg a, hs_qpu_reg b) { emit(p, hs_qpu_a(o, d, a, b)); }
static void mov(hs_qpu_prog* p, hs_qpu_reg d, hs_qpu_reg s) { op(p, HS_QPU_A_MOV, d, s, X); }
static void tmuwt(hs_qpu_prog* p) { op(p, HS_QPU_A_TMUWT, X, X, X); }
static void flags(hs_qpu_prog* p, int o, hs_qpu_reg d, hs_qpu_reg a, hs_qpu_reg b, int pf) {
    hs_qpu_alu s = hs_qpu_a(o, d, a, b);
    s.af = (uint8_t)pf;
    emit(p, s);
}
static void br(hs_qpu_prog* p, int cond, int label, int slots_filled) {
    hs_qpu_branch(p, cond, label, 0, 0);
    for (int i = slots_filled; i < 3; i++) nop(p);
}
static void load(hs_qpu_prog* p, hs_qpu_reg addr, hs_qpu_reg dst) {
    emit(p, hs_qpu_s(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, addr, X), HS_QPU_SIG_THRSW, X));
    nop(p);
    nop(p);
    emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDTMU, dst));
}
static void store(hs_qpu_prog* p, hs_qpu_reg addr, hs_qpu_reg v) {
    mov(p, HS_QPU_TMUD, v);
    mov(p, HS_QPU_TMUA, addr);
}
/* op 2 (publish, then reload) on each lane's word; takes the next uniform */
static void op2(hs_qpu_prog* p, hs_qpu_reg addr) {
    mov(p, HS_QPU_TMUAU, addr);
    tmuwt(p);
}
static void budget_sub(hs_qpu_prog* p, hs_qpu_reg units) { op(p, HS_QPU_A_SUB, RF(3), RF(3), units); }

/* publish rf40 lines starting at rf39, stride rf22 */
static void publish(hs_qpu_prog* p) {
    int l = hs_qpu_label_new(p);
    hs_qpu_label_bind(p, l);
    mov(p, HS_QPU_TMUAU, RF(39));
    op(p, HS_QPU_A_ADD, RF(39), RF(39), RF(22));
    budget_sub(p, I(1));
    flags(p, HS_QPU_A_SUB, RF(40), RF(40), I(1), HS_QPU_PF_PUSHZ);
    br(p, HS_QPU_B_NA0, l, 0);
    tmuwt(p);
}

static void gather_loop(hs_qpu_prog* p, int table_op) {
    int l = hs_qpu_label_new(p);
    hs_qpu_label_bind(p, l);
    op2(p, RF(33));
    budget_sub(p, I(1));
    load(p, RF(33), R(0));
    op(p, HS_QPU_A_SHL, R(0), R(0), I(2));
    op(p, HS_QPU_A_ADD, R(1), R(0), RF(27));
    if (table_op) {
        op2(p, R(1));
        budget_sub(p, I(1));
    }
    load(p, R(1), R(2));
    store(p, RF(34), R(2));
    op(p, HS_QPU_A_ADD, RF(33), RF(33), RF(22));
    op(p, HS_QPU_A_ADD, RF(34), RF(34), RF(22));
    flags(p, HS_QPU_A_SUB, RF(35), RF(35), I(1), HS_QPU_PF_PUSHZ);
    br(p, HS_QPU_B_NA0, l, 0);
}

/* the two-chain step: add ALU finishes one chain's y = y*c1 + c0 while the
 * mul ALU starts the other's. r2 = c1, r3 = c0. */
static void chain_i1(hs_qpu_prog* p) {
    emit(p, hs_qpu_m(hs_qpu_a(HS_QPU_A_FADD, RF(36), R(0), R(3)), HS_QPU_M_FMUL, R(1), RF(37), R(2)));
}
static void chain_i2(hs_qpu_prog* p) {
    emit(p, hs_qpu_m(hs_qpu_a(HS_QPU_A_FADD, RF(37), R(1), R(3)), HS_QPU_M_FMUL, R(0), RF(36), R(2)));
}

static int build(hs_qpu_prog* p, uint32_t stretch_log2) {
    hs_qpu_prog_rules(p, 0); /* TMUAU and multi-load forms are outside the checker; forms measured in P0-5 */
    mov(p, RF(9), RF(0)); /* the hardware's batch number (workgroup x), before rf0 is overwritten */
    ldunifrf(p, 0);
    ldunifrf(p, 1);
    ldunifrf(p, 3);
    ldunifrf(p, 6);
    ldunifrf(p, 7);
    ldunifrf(p, 8);
    ldunifrf(p, 22);
    ldunifrf(p, 2); /* batches - 1: the batch-number mask */
    ldunifrf(p, 19); /* slots - 1 */
    op(p, HS_QPU_A_EIDX, R(1), X, X);
    op(p, HS_QPU_A_SHL, R(0), R(1), I(2)); /* r0 = 4l */
    op(p, HS_QPU_A_SUB, R(2), R(1), I(1));
    op(p, HS_QPU_A_ASR, RF(5), R(2), I(15)); /* lane 0: ~0; others: 0 */
    op(p, HS_QPU_A_ASR, R(3), R(2), I(2));   /* lane 0: -1; lanes 1..4: 0; others: > 0 */
    mov(p, RF(13), I(0));
    op(p, HS_QPU_A_SUB, RF(13), RF(13), R(3));
    op(p, HS_QPU_A_ASR, RF(13), RF(13), I(15)); /* lanes 5..15: ~0; others: 0 */
    op(p, HS_QPU_A_BOR, R(3), RF(13), RF(5));
    op(p, HS_QPU_A_ADD, RF(18), R(3), I(1)); /* lanes 1..4: 1; others: 0 */
    flags(p, HS_QPU_A_SUB, R(3), R(1), I(3), HS_QPU_PF_PUSHZ);
    mov(p, RF(20), I(0));
    hs_qpu_alu l3 = hs_qpu_a(HS_QPU_A_MOV, RF(20), I(-1), X); /* lane 3: ~0; others: 0 */
    l3.ac = HS_QPU_COND_IFA;
    emit(p, l3);
    op(p, HS_QPU_A_BAND, RF(9), RF(9), RF(2)); /* batch (not the thread index: see hs_v3d.h) */
    op(p, HS_QPU_A_SHL, R(2), RF(9), I(8));
    op(p, HS_QPU_A_ADD, RF(1), RF(1), R(2));
    op(p, HS_QPU_A_ADD, RF(6), RF(6), R(2));
    op(p, HS_QPU_A_SHL, R(2), RF(9), I(6));
    op(p, HS_QPU_A_ADD, RF(23), R(2), R(0));
    static const int lo[] = {0, 1, 6, 7, 8};
    for (int i = 0; i < 5; i++) op(p, HS_QPU_A_ADD, RF(lo[i]), RF(lo[i]), R(0));
    mov(p, R(2), I(1));
    op(p, HS_QPU_A_SHL, RF(17), R(2), I(6)); /* 64 */
    op(p, HS_QPU_A_SHL, RF(41), R(2), I(9)); /* 512: slot size */
    if (stretch_log2 != NO_STRETCH) op(p, HS_QPU_A_SHL, RF(46), R(2), I((int)stretch_log2)); /* chain iterations per stretch */
    op(p, HS_QPU_A_ADD, RF(16), RF(6), RF(17));
    op(p, HS_QPU_A_ADD, RF(16), RF(16), RF(17));
    op(p, HS_QPU_A_ADD, RF(43), RF(22), RF(22));
    load(p, RF(1), R(3)); /* the last request this batch acknowledged */
    op(p, HS_QPU_A_ADD, RF(4), R(3), I(1));
    op(p, HS_QPU_A_BAND, R(3), R(3), RF(19));
    op(p, HS_QPU_A_SHL, R(3), R(3), I(9));
    op(p, HS_QPU_A_ADD, RF(11), RF(7), R(3)); /* request n is in slot (n - 1) mod slots */
    op(p, HS_QPU_A_BAND, R(1), RF(4), RF(5));
    op(p, HS_QPU_A_BOR, RF(12), R(1), RF(18));
    load(p, RF(16), RF(14));
    load(p, RF(6), R(1));

    int poll = hs_qpu_label_new(p), done = hs_qpu_label_new(p);
    int l_gather = hs_qpu_label_new(p), l_gstatic = hs_qpu_label_new(p), l_gpub = hs_qpu_label_new(p);
    int l_chain = hs_qpu_label_new(p), l_stop = hs_qpu_label_new(p), l_ack = hs_qpu_label_new(p);
    int l_cl = hs_qpu_label_new(p), l_hl = hs_qpu_label_new(p);
    flags(p, HS_QPU_A_MOV, R(1), R(1), X, HS_QPU_PF_PUSHZ); /* stop flag set: end at once */
    br(p, HS_QPU_B_NA0, done, 0);

    /* poll while M < next sequence (lane 0) and no yield word (lanes 1..3)
     * is set, until the budget runs out */
    hs_qpu_label_bind(p, poll);
    op2(p, RF(0));
    load(p, RF(0), R(0));
    flags(p, HS_QPU_A_SUB, RF(3), RF(3), I(1), HS_QPU_PF_PUSHZ);
    hs_qpu_branch(p, HS_QPU_B_A0, done, 0, 0);
    op(p, HS_QPU_A_SUB, R(1), R(0), RF(12));
    flags(p, HS_QPU_A_BOR, R(1), R(1), RF(13), HS_QPU_PF_PUSHN);
    nop(p);
    br(p, HS_QPU_B_ALLA, poll, 0);
    /* no request: a yield word ended the poll */
    op(p, HS_QPU_A_SUB, R(1), R(0), RF(4));
    flags(p, HS_QPU_A_BAND, R(1), R(1), RF(5), HS_QPU_PF_PUSHN);
    br(p, HS_QPU_B_ANYA, done, 0);
    /* a request is pending: it is served unless the lease is too old (word 3) */
    flags(p, HS_QPU_A_BAND, R(1), R(0), RF(20), HS_QPU_PF_PUSHZ);
    br(p, HS_QPU_B_ANYNA, done, 0);

    /* the request: 8 lines, one value each (replicated across lanes) */
    mov(p, R(1), RF(11));
    for (int j = 0; j < 8; j++) {
        hs_qpu_alu x = hs_qpu_m(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, R(1), X), HS_QPU_M_ADD, R(1), R(1), RF(17));
        if (j == 7) x = hs_qpu_s(x, HS_QPU_SIG_THRSW, X);
        emit(p, x);
    }
    nop(p);
    nop(p);
    for (int j = 0; j < 8; j++) emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDTMU, RF(24 + j)));
    flags(p, HS_QPU_A_SUB, R(1), RF(24), I(1), HS_QPU_PF_PUSHZ);
    br(p, HS_QPU_B_A0, l_gather, 0);
    flags(p, HS_QPU_A_SUB, R(1), RF(24), I(2), HS_QPU_PF_PUSHZ);
    br(p, HS_QPU_B_A0, l_chain, 0);
    flags(p, HS_QPU_A_SUB, R(1), RF(24), I(3), HS_QPU_PF_PUSHZ);
    br(p, HS_QPU_B_A0, l_stop, 0);
    br(p, HS_QPU_B_ALWAYS, l_ack, 0);

    /* gather: rf25 chunks per batch, rf26 idx, rf27 table, rf28 out, rf29 table op */
    hs_qpu_label_bind(p, l_gather);
    op(p, HS_QPU_A_ADD, RF(33), RF(26), RF(23));
    op(p, HS_QPU_A_ADD, RF(34), RF(28), RF(23));
    mov(p, RF(35), RF(25));
    flags(p, HS_QPU_A_MOV, R(1), RF(25), X, HS_QPU_PF_PUSHZ);
    br(p, HS_QPU_B_A0, l_ack, 0);
    flags(p, HS_QPU_A_MOV, R(1), RF(29), X, HS_QPU_PF_PUSHZ);
    br(p, HS_QPU_B_A0, l_gstatic, 0);
    gather_loop(p, 1);
    br(p, HS_QPU_B_ALWAYS, l_gpub, 0);
    hs_qpu_label_bind(p, l_gstatic);
    gather_loop(p, 0);
    hs_qpu_label_bind(p, l_gpub);
    tmuwt(p);
    op(p, HS_QPU_A_ADD, RF(39), RF(28), RF(23));
    mov(p, RF(40), RF(25));
    publish(p);
    br(p, HS_QPU_B_ALWAYS, l_ack, 0);

    /* chain: rf25 pairs per batch, rf26 in, rf28 out, rf29 k/4, rf30 c1, rf31 c0 */
    hs_qpu_label_bind(p, l_chain);
    op(p, HS_QPU_A_ADD, RF(33), RF(26), RF(23));
    op(p, HS_QPU_A_ADD, RF(34), RF(28), RF(23));
    mov(p, RF(35), RF(25));
    flags(p, HS_QPU_A_MOV, R(1), RF(25), X, HS_QPU_PF_PUSHZ);
    br(p, HS_QPU_B_A0, l_ack, 0);
    hs_qpu_label_bind(p, l_cl);
    op2(p, RF(33));
    op(p, HS_QPU_A_ADD, RF(42), RF(33), RF(22));
    op2(p, RF(42));
    budget_sub(p, I(2));
    mov(p, HS_QPU_TMUA, RF(33));
    emit(p, hs_qpu_s(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(42), X), HS_QPU_SIG_THRSW, X));
    nop(p);
    nop(p);
    emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDTMU, RF(36)));
    emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDTMU, RF(37)));
    mov(p, R(2), RF(30));
    mov(p, R(3), RF(31));
    emit(p, hs_qpu_m(hs_qpu_a(HS_QPU_A_MOV, RF(38), RF(29), X), HS_QPU_M_FMUL, R(0), RF(36), R(2)));
    /* A compute job lets a renderer's jobs run only at its thread switches
     * (measured: tools/p0_5_spin_probe). The k steps therefore run in
     * stretches of rf46 iterations (4 steps each) with a switch after each:
     * rf38 iterations left, rf44 this stretch's. */
    int l_st = hs_qpu_label_new(p);
    if (stretch_log2 != NO_STRETCH) {
        hs_qpu_label_bind(p, l_st);
        op(p, HS_QPU_A_UMIN, RF(44), RF(38), RF(46));
        op(p, HS_QPU_A_SUB, RF(38), RF(38), RF(44));
    }
    hs_qpu_reg count = stretch_log2 != NO_STRETCH ? RF(44) : RF(38);
    hs_qpu_label_bind(p, l_hl);
    chain_i1(p);
    chain_i2(p);
    chain_i1(p);
    chain_i2(p);
    chain_i1(p);
    flags(p, HS_QPU_A_SUB, count, count, I(1), HS_QPU_PF_PUSHZ);
    hs_qpu_branch(p, HS_QPU_B_NA0, l_hl, 0, 0);
    chain_i2(p); /* delay slots: run on both paths */
    chain_i1(p);
    chain_i2(p);
    if (stretch_log2 != NO_STRETCH) { /* more steps left: switch threads, then the next stretch */
        int l_last = hs_qpu_label_new(p);
        flags(p, HS_QPU_A_SUB, RF(45), RF(38), I(0), HS_QPU_PF_PUSHZ);
        br(p, HS_QPU_B_A0, l_last, 0);
        emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_THRSW, X));
        nop(p);
        nop(p);
        /* A switch that really runs another thread does not keep r0..r3
         * (measured beside a renderer: every checked result wrong without
         * these three; alone the switch changes nothing). Reload c1 and c0
         * and redo the multiply that was pending in r0. */
        hs_qpu_branch(p, HS_QPU_B_ALWAYS, l_st, 0, 0);
        mov(p, R(2), RF(30));
        mov(p, R(3), RF(31));
        emit(p, hs_qpu_m(hs_qpu_alu_nop(), HS_QPU_M_FMUL, R(0), RF(36), R(2)));
        hs_qpu_label_bind(p, l_last);
    }
    store(p, RF(34), RF(36));
    op(p, HS_QPU_A_ADD, R(1), RF(34), RF(22));
    store(p, R(1), RF(37));
    budget_sub(p, RF(29));
    op(p, HS_QPU_A_ADD, RF(33), RF(33), RF(43));
    op(p, HS_QPU_A_ADD, RF(34), RF(34), RF(43));
    flags(p, HS_QPU_A_SUB, RF(35), RF(35), I(1), HS_QPU_PF_PUSHZ);
    br(p, HS_QPU_B_NA0, l_cl, 0);
    tmuwt(p);
    op(p, HS_QPU_A_ADD, RF(39), RF(28), RF(23));
    op(p, HS_QPU_A_ADD, RF(40), RF(25), RF(25));
    publish(p);
    br(p, HS_QPU_B_ALWAYS, l_ack, 0);

    /* stop: acknowledge, then set the stop flag so queued leases end at once */
    hs_qpu_label_bind(p, l_stop);
    store(p, RF(1), RF(4));
    tmuwt(p);
    op2(p, RF(1));
    mov(p, R(2), I(1));
    store(p, RF(6), R(2));
    br(p, HS_QPU_B_ALWAYS, done, 0);

    /* acknowledge and move to the next request */
    hs_qpu_label_bind(p, l_ack);
    store(p, RF(1), RF(4));
    tmuwt(p);
    op2(p, RF(1));
    budget_sub(p, I(1));
    op(p, HS_QPU_A_ADD, RF(4), RF(4), I(1));
    op(p, HS_QPU_A_BAND, R(1), RF(4), RF(5));
    op(p, HS_QPU_A_BOR, RF(12), R(1), RF(18));
    op(p, HS_QPU_A_ADD, RF(11), RF(11), RF(41));
    flags(p, HS_QPU_A_SUB, R(1), RF(11), RF(8), HS_QPU_PF_PUSHZ);
    hs_qpu_alu w = hs_qpu_a(HS_QPU_A_MOV, RF(11), RF(7), X);
    w.ac = HS_QPU_COND_IFA;
    emit(p, w);
    /* The lease ends where the ring wraps. Within one job no batch then reads
     * a slot index for two different requests, and the host does not rewrite
     * a slot before every batch acknowledged its request: a slot the GPU
     * cached in this job is never stale, whichever batch cached it. */
    hs_qpu_branch(p, HS_QPU_B_A0, done, 0, 0); /* the ring wrapped: end the lease */
    flags(p, HS_QPU_A_SUB, R(1), RF(3), I(1), HS_QPU_PF_PUSHN); /* delay slot: budget <= 0 */
    nop(p);
    nop(p);
    br(p, HS_QPU_B_A0, done, 0);
    br(p, HS_QPU_B_ALWAYS, poll, 0);

    /* lease end: count it (every lease that is not killed passes here) */
    hs_qpu_label_bind(p, done);
    op(p, HS_QPU_A_ADD, RF(14), RF(14), I(1));
    store(p, RF(16), RF(14));
    tmuwt(p);
    hs_qpu_emit_end(p);
    return hs_qpu_prog_finish(p);
}

static int build_warm(hs_qpu_prog* p) {
    ldunifrf(p, 0);
    int l = hs_qpu_label_new(p);
    hs_qpu_label_bind(p, l);
    flags(p, HS_QPU_A_SUB, RF(0), RF(0), I(1), HS_QPU_PF_PUSHZ);
    br(p, HS_QPU_B_NA0, l, 0);
    hs_qpu_emit_end(p);
    return hs_qpu_prog_finish(p);
}

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
static void barrier(void) { __asm__ volatile("dsb sy" ::: "memory"); }
static volatile uint32_t* ctl32(const hs_mailbox* mb, uint32_t off) { return (volatile uint32_t*)(mb->ctl.map + off); }
static void fill_line(hs_mailbox* mb, uint32_t off, uint32_t v) {
    volatile uint32_t* w = ctl32(mb, off);
    for (int l = 0; l < 16; l++) w[l] = v;
}
static uint32_t ack_of(const hs_mailbox* mb, uint32_t b) { return ctl32(mb, OFF_ACK + 256 * b)[0]; }

uint32_t hs_mailbox_leases_done(const hs_mailbox* mb, uint32_t b) {
    return ctl32(mb, OFF_STATE + 256 * b + 128)[0];
}
/* leases every batch finished: finished jobs, as long as none was killed */
static uint32_t jobs_done(const hs_mailbox* mb) {
    uint32_t m = hs_mailbox_leases_done(mb, 0);
    for (uint32_t b = 1; b < mb->batches; b++) {
        uint32_t c = hs_mailbox_leases_done(mb, b);
        if ((int32_t)(c - m) < 0) m = c;
    }
    return m;
}

/* ---- watchdog: the time bound on a lease, and the idle stop. It reads the
 * GPU-side lease counters and the caller's activity counter, and writes yield
 * words 2 and 3. The caller's thread only ever clears word 2 (present). ---- */
static void* wd_main(void* arg) {
    hs_mailbox* mb = arg;
    cpu_set_t all; /* not the caller's (possibly pinned, spinning) CPU */
    memset(&all, 0xFF, sizeof all);
    sched_setaffinity(0, sizeof all, &all);
    pthread_mutex_lock(&mb->wd_mu);
    for (;;) {
        while (atomic_load(&mb->wd_state) == WD_PARKED && !atomic_load(&mb->wd_quit))
            pthread_cond_wait(&mb->wd_cv, &mb->wd_mu);
        if (atomic_load(&mb->wd_quit)) break;
        pthread_mutex_unlock(&mb->wd_mu);
        uint64_t t_act = now_ns(), t_change = t_act;
        uint32_t act0 = atomic_load(&mb->wd_act), j0 = jobs_done(mb), idle_ticks = 0;
        int aging = 0; /* yield word 3 is set */
        uint64_t t_aged = 0;
        for (;;) {
            struct timespec ts = {0, WD_TICK_NS};
            nanosleep(&ts, NULL);
            if (atomic_load(&mb->wd_quit)) break;
            uint64_t t = now_ns();
            uint32_t a = atomic_load(&mb->wd_act), j = jobs_done(mb);
            int was_idle = atomic_load(&mb->wd_state) == WD_IDLE;
            if (a != act0) act0 = a, t_act = t; /* the caller sent or waited since the last tick */
            if (j != j0 || (was_idle && t_act == t)) { /* a job ended on every batch; or back from idle */
                if (j != j0 && !was_idle && t - t_change > atomic_load(&mb->wd_job_gap_max_ns))
                    atomic_store(&mb->wd_job_gap_max_ns, t - t_change);
                j0 = j, t_change = t;
                if (aging) {
                    ctl32(mb, OFF_M)[3] = 0;
                    barrier();
                    aging = 0;
                    if (t - t_aged > atomic_load(&mb->wd_age_hold_max_ns)) atomic_store(&mb->wd_age_hold_max_ns, t - t_aged);
                }
            }
            int idle = t - t_act > mb->idle_ns;
            if (idle != was_idle) {
                ctl32(mb, OFF_M)[2] = (uint32_t)idle;
                barrier();
                atomic_store(&mb->wd_state, idle ? WD_IDLE : WD_RUN);
                if (idle) {
                    atomic_fetch_add(&mb->wd_idle_yields, 1);
                    if (atomic_load(&mb->wd_act) != act0) { /* it returned while we decided */
                        ctl32(mb, OFF_M)[2] = 0;
                        barrier();
                        atomic_store(&mb->wd_state, WD_RUN);
                        idle = 0;
                    }
                }
            }
            if (!aging && !idle && t - t_change > mb->max_lease_ns) {
                /* No job has ended for max_lease: every batch ends its lease
                 * at its next poll. Held until the job has ended on every
                 * batch (beside a renderer the batches do not all run at
                 * once), watched closely at first so the next job can run. */
                ctl32(mb, OFF_M)[3] = 1;
                barrier();
                aging = 1;
                t_aged = now_ns();
                atomic_fetch_add(&mb->wd_age_yields, 1);
                for (int i = 0; i < 60 && jobs_done(mb) == j0 && !atomic_load(&mb->wd_quit); i++) {
                    struct timespec w = {0, 20000};
                    nanosleep(&w, NULL);
                }
                if (jobs_done(mb) != j0) {
                    ctl32(mb, OFF_M)[3] = 0;
                    barrier();
                    aging = 0;
                    uint64_t t_end = now_ns();
                    if (t_end - t_change > atomic_load(&mb->wd_job_gap_max_ns))
                        atomic_store(&mb->wd_job_gap_max_ns, t_end - t_change);
                    j0 = jobs_done(mb), t_change = t_end;
                    if (t_change - t_aged > atomic_load(&mb->wd_age_hold_max_ns))
                        atomic_store(&mb->wd_age_hold_max_ns, t_change - t_aged);
                }
            }
            idle_ticks = idle ? idle_ticks + 1 : 0;
            /* Drained: sleep until the caller returns. The BO wait is only made
             * here, with the caller away: with a zero timeout it still blocks
             * for about 2 ms while a job runs (measured). */
            if (idle_ticks >= 4 && !hs_v3d_bo_wait(mb->dev, &mb->ctl, 0)) {
                pthread_mutex_lock(&mb->wd_mu);
                atomic_store(&mb->wd_state, WD_PARKED);
                if (atomic_load(&mb->wd_act) != act0) { /* it returned meanwhile: stay awake */
                    atomic_store(&mb->wd_state, WD_IDLE);
                    pthread_mutex_unlock(&mb->wd_mu);
                    idle_ticks = 0;
                    continue;
                }
                ctl32(mb, OFF_M)[2] = ctl32(mb, OFF_M)[3] = 0; /* nothing is queued: nothing to yield */
                barrier();
                goto parked;
            }
        }
        pthread_mutex_lock(&mb->wd_mu);
    parked:;
    }
    pthread_mutex_unlock(&mb->wd_mu);
    return NULL;
}
/* The caller is here (sending or waiting). The activity counter is written
 * before the state is read, and the watchdog writes PARKED before it re-reads
 * the counter: one of the two always sees the other. */
static void present(hs_mailbox* mb) {
    if (!mb->wd_started) return;
    atomic_fetch_add(&mb->wd_act, 1);
    int st = atomic_load(&mb->wd_state);
    if (st == WD_RUN) return;
    ctl32(mb, OFF_M)[2] = 0; /* back from idle: do not wait a watchdog tick for the yield to clear */
    barrier();
    if (st == WD_PARKED) {
        pthread_mutex_lock(&mb->wd_mu);
        if (atomic_load(&mb->wd_state) == WD_PARKED) {
            atomic_store(&mb->wd_state, WD_RUN);
            pthread_cond_signal(&mb->wd_cv);
        }
        pthread_mutex_unlock(&mb->wd_mu);
    }
}

static int dispatch(hs_mailbox* mb, uint32_t code, uint32_t unif_gpu, uint32_t batches) {
    hs_v3d_dispatch_desc d = {code, unif_gpu, batches, mb->handles, mb->nhandles, 0};
    return hs_v3d_dispatch(mb->dev, &d);
}
static int top_up(hs_mailbox* mb) {
    /* the usual case, one read. Signed: after a killed lease batch 0's count
     * can be ahead of the jobs that finished on every batch. */
    if ((int32_t)(mb->submitted - hs_mailbox_leases_done(mb, 0)) >= (int32_t)mb->qdepth) return 0;
    uint32_t ended = jobs_done(mb);
    if ((int32_t)(mb->submitted - ended) < 0) mb->submitted = ended;
    while ((int32_t)(mb->submitted - ended) < (int32_t)mb->qdepth) {
        int e = dispatch(mb, mb->prog, mb->unif.gpu, mb->batches);
        if (e) return e;
        mb->submitted++;
    }
    return 0;
}

int hs_mailbox_start(hs_mailbox* mb, hs_v3d* dev, const hs_mailbox_cfg* cfg) {
    memset(mb, 0, sizeof(*mb));
    if (!dev || !cfg || (cfg->batches != 1 && cfg->batches != 2 && cfg->batches != 4 && cfg->batches != 8) || cfg->nbos > HS_MB_MAX_BOS || (cfg->nbos && !cfg->bos))
        return -EINVAL;
    uint32_t budget = cfg->budget ? cfg->budget : 30000, qdepth = cfg->qdepth ? cfg->qdepth : 2;
    uint32_t slots = cfg->slots ? cfg->slots : 256;
    if (budget < 2 || budget > HS_MB_MAX_BUDGET || qdepth > 8 || slots < 2 || slots > 65536 || (slots & (slots - 1)))
        return -EINVAL;
    for (uint32_t i = 0; i < cfg->nbos; i++)
        if (!cfg->bos[i] || !cfg->bos[i]->map) return -EINVAL;
    uint32_t stretch = cfg->chain_stretch ? cfg->chain_stretch : HS_MB_CHAIN_STRETCH_DEFAULT, stretch_log2 = NO_STRETCH;
    if (stretch != HS_MB_CHAIN_NO_SWITCH) {
        if (stretch > 32768 || (stretch & (stretch - 1))) return -EINVAL;
        stretch_log2 = (uint32_t)__builtin_ctz(stretch);
    }
    mb->dev = dev;
    mb->batches = cfg->batches;
    mb->budget = budget;
    mb->qdepth = qdepth;
    mb->slots = slots;
    mb->idle_ns = (uint64_t)(cfg->idle_us ? cfg->idle_us : 2000) * 1000;
    mb->max_lease_ns = (uint64_t)(cfg->max_lease_us ? cfg->max_lease_us : 50000) * 1000;
    size_t nunif = HEADER + (size_t)mb->budget + MAX_UNITS + 256;
    int e;
    if ((e = hs_v3d_bo_new(dev, &mb->code, 16 << 10)) || (e = hs_v3d_bo_new(dev, &mb->unif, nunif * 4)) ||
        (e = hs_v3d_bo_new(dev, &mb->ctl, OFF_SLOTS + (size_t)mb->slots * SLOT_BYTES))) {
        hs_mailbox_free(mb);
        return e;
    }
    mb->handles[0] = mb->code.handle;
    mb->handles[1] = mb->unif.handle;
    mb->handles[2] = mb->ctl.handle;
    for (uint32_t i = 0; i < cfg->nbos; i++) {
        mb->handles[3 + i] = cfg->bos[i]->handle;
        mb->bo_lo[i] = cfg->bos[i]->gpu;
        mb->bo_hi[i] = (uint64_t)cfg->bos[i]->gpu + cfg->bos[i]->size;
    }
    mb->nbos = cfg->nbos;
    mb->nhandles = 3 + cfg->nbos;
    uint64_t buf[2048];
    hs_qpu_prog p;
    hs_qpu_prog_init(&p, buf, 2048);
    int n = build(&p, stretch_log2);
    if (n < 0 || !(mb->prog = hs_v3d_load(&mb->code, 0, buf, (uint32_t)n))) {
        hs_mailbox_free(mb);
        return -EIO;
    }
    size_t warm_off = ((size_t)n * 8 + 255) & ~(size_t)255;
    hs_qpu_prog_init(&p, buf, 2048);
    int nw = build_warm(&p);
    uint32_t warm = nw > 0 ? hs_v3d_load(&mb->code, warm_off, buf, (uint32_t)nw) : 0;
    if (!warm) {
        hs_mailbox_free(mb);
        return -EIO;
    }
    uint32_t* u = (uint32_t*)mb->unif.map;
    u[0] = mb->ctl.gpu + OFF_M;
    u[1] = mb->ctl.gpu + OFF_ACK;
    u[2] = mb->budget;
    u[3] = mb->ctl.gpu + OFF_STATE;
    u[4] = mb->ctl.gpu + OFF_SLOTS;
    u[5] = mb->ctl.gpu + OFF_SLOTS + mb->slots * SLOT_BYTES;
    u[6] = 64 * mb->batches;
    u[7] = mb->batches - 1;
    u[8] = mb->slots - 1;
    for (size_t i = HEADER; i < nunif; i++) u[i] = 0xFFFFFF80u | 2u << 3 | 7u; /* per-lane op 2, 32-bit */
    memset((void*)mb->ctl.map, 0, mb->ctl.size);
    ctl32(mb, OFF_WARM)[0] = 37500; /* ~1 ms: the GPU clock drops after idle (HARDWARE_MAP) */
    barrier();
    if ((e = dispatch(mb, warm, mb->ctl.gpu + OFF_WARM, 1)) ||
        (e = hs_v3d_bo_wait(dev, &mb->ctl, 2000000000ull))) {
        hs_mailbox_free(mb);
        return e;
    }
    if (!cfg->watchdog_off) {
        pthread_mutex_init(&mb->wd_mu, NULL);
        pthread_cond_init(&mb->wd_cv, NULL);
        atomic_store(&mb->wd_state, WD_PARKED);
        if ((e = pthread_create(&mb->wd_thread, NULL, wd_main, mb))) {
            pthread_mutex_destroy(&mb->wd_mu);
            pthread_cond_destroy(&mb->wd_cv);
            hs_mailbox_free(mb);
            return -e;
        }
        mb->wd_started = 1;
    }
    mb->running = 1;
    return 0;
}

/* [addr, addr + bytes) lies inside one BO the mailbox's jobs hold */
static int covered(const hs_mailbox* mb, uint32_t addr, uint64_t bytes) {
    for (uint32_t i = 0; i < mb->nbos; i++)
        if (addr >= mb->bo_lo[i] && (uint64_t)addr + bytes <= mb->bo_hi[i]) return 1;
    return 0;
}
static int overlap(uint32_t a, uint64_t an, uint32_t b, uint64_t bn) {
    return (uint64_t)a < (uint64_t)b + bn && (uint64_t)b < (uint64_t)a + an;
}
static int valid(const hs_mailbox* mb, uint32_t opc, const uint32_t a[7]) {
    static const uint32_t zero[7];
    if (!a) a = zero;
    if (opc == HS_MB_OP_ECHO) return 1;
    if (opc == HS_MB_OP_GATHER) {
        uint64_t n = a[0], bytes = n * 64 * mb->batches;
        uint64_t tbytes = (uint64_t)a[5] * 4;
        if (a[4] > 1 || a[6] || n > HS_MB_MAX_GATHER_CHUNKS) return 0;
        if (!n) return 1;
        if (((a[1] | a[2] | a[3]) & 3) || !a[5]) return 0;
        if (!covered(mb, a[1], bytes) || !covered(mb, a[3], bytes) || !covered(mb, a[2], tbytes)) return 0;
        return !overlap(a[1], bytes, a[3], bytes) && !overlap(a[2], tbytes, a[3], bytes);
    }
    if (opc == HS_MB_OP_CHAIN) {
        uint64_t n = a[0], bytes = n * 128 * mb->batches;
        if (a[2] || !a[4] || a[4] > 16384 || n * ((uint64_t)a[4] + 4) + 4 > MAX_UNITS) return 0;
        if (!n) return 1;
        if ((a[1] | a[3]) & 3) return 0;
        if (!covered(mb, a[1], bytes) || !covered(mb, a[3], bytes)) return 0;
        return !overlap(a[1], bytes, a[3], bytes);
    }
    return 0; /* HS_MB_OP_STOP goes through hs_mailbox_stop */
}

/* the newest request every batch has acknowledged */
static void refresh_done(hs_mailbox* mb) {
    uint32_t m = ack_of(mb, 0);
    for (uint32_t b = 1; b < mb->batches; b++) {
        uint32_t c = ack_of(mb, b);
        if ((int32_t)(c - m) < 0) m = c;
    }
    if ((int32_t)(m - mb->done_seq) > 0 && (int32_t)(m - mb->seq) <= 0) mb->done_seq = m;
}

static int64_t send_raw(hs_mailbox* mb, uint32_t opc, const uint32_t a[7]) {
    if (!mb->dev) return -EINVAL;
    if (mb->stop_sent) return -EBUSY; /* a stop is under way: finish it with hs_mailbox_stop */
    if (mb->seq - mb->done_seq >= mb->slots) {
        refresh_done(mb);
        if (mb->seq - mb->done_seq >= mb->slots) return -EBUSY;
    }
    uint32_t n = mb->seq + 1;
    if (!mb->running) { /* restart: hs_mailbox_stop returned 0, so no lease is left from before it */
        for (uint32_t b = 0; b < mb->batches; b++) fill_line(mb, OFF_STATE + 256 * b, 0);
        mb->submitted = jobs_done(mb);
        mb->running = 1;
    }
    uint32_t slot = OFF_SLOTS + ((n - 1) & (mb->slots - 1)) * SLOT_BYTES;
    fill_line(mb, slot, opc);
    for (int j = 0; j < 7; j++) fill_line(mb, slot + 64 * (1 + j), a ? a[j] : 0);
    if (mb->yielding) ctl32(mb, OFF_M)[1] = 0, mb->yielding = 0;
    barrier();
    present(mb);
    int e = top_up(mb); /* before the request is published: a failure here sends nothing */
    if (e) return e;
    ctl32(mb, OFF_M)[0] = n;
    barrier();
    mb->seq = n;
    return (int64_t)n;
}

int64_t hs_mailbox_send(hs_mailbox* mb, uint32_t opc, const uint32_t a[7]) {
    if (!mb->dev || !valid(mb, opc, a)) return -EINVAL;
    return send_raw(mb, opc, a);
}

int hs_mailbox_wait(hs_mailbox* mb, uint32_t seq, uint64_t timeout_ns) {
    if (!mb->dev || (int32_t)(seq - mb->seq) > 0) return -EINVAL; /* never sent */
    uint64_t t0 = now_ns(), t = t0, t_partial = 0, t_check = t0;
    uint32_t lease0 = hs_mailbox_leases_done(mb, 0);
    for (uint32_t spin = 0;; spin++) {
        uint32_t acked = 0;
        for (uint32_t b = 0; b < mb->batches; b++) acked += (int32_t)(ack_of(mb, b) - seq) >= 0;
        if (acked == mb->batches) break;
        if ((spin & 63) == 63) {
            t = now_ns();
            present(mb);
            int e = top_up(mb);
            if (e) return e;
            /* Some batches served it and the rest have not started: measured
             * beside a renderer, batches run one after another, and a batch
             * polling for the next request holds the others back. */
            if (!acked) t_partial = 0;
            else if (!t_partial) t_partial = t;
            else if (!mb->yielding && t - t_partial > STALL_NS) {
                ctl32(mb, OFF_M)[1] = 1;
                barrier();
                mb->yielding = 1;
                mb->stall_yields++;
            }
            if (t - t_check > LOST_CHECK_NS) { /* no job left but leases counted as queued: they were killed */
                t_check = t;
                if (!hs_v3d_bo_wait(mb->dev, &mb->ctl, 0)) {
                    uint32_t d = jobs_done(mb);
                    if (mb->submitted != d) mb->submitted = d, mb->lost_leases++;
                    if ((e = top_up(mb))) return e;
                }
            }
            if (t - t0 > timeout_ns) return -ETIME;
        }
    }
    if (mb->yielding) {
        ctl32(mb, OFF_M)[1] = 0;
        barrier();
        mb->yielding = 0;
    }
    if (now_ns() - t0 > 20000) mb->waits_over_20us++;
    if (hs_mailbox_leases_done(mb, 0) != lease0) mb->handovers_seen++;
    if ((int32_t)(seq - mb->done_seq) > 0) mb->done_seq = seq;
    return 0;
}

static int call(hs_mailbox* mb, uint32_t opc, const uint32_t a[7]) {
    int64_t n = hs_mailbox_send(mb, opc, a);
    if (n < 0) return (int)n;
    return hs_mailbox_wait(mb, (uint32_t)n, 1000000000ull);
}

int hs_mailbox_echo(hs_mailbox* mb) { return call(mb, HS_MB_OP_ECHO, NULL); }

int hs_mailbox_gather(hs_mailbox* mb, uint32_t idx_gpu, uint32_t table_gpu, uint32_t table_elements, uint32_t out_gpu,
                      uint32_t elements, int table_changes) {
    uint32_t per = 16 * mb->batches;
    if (!mb->dev || elements % per) return -EINVAL;
    uint32_t a[7] = {elements / per, idx_gpu, table_gpu, out_gpu, table_changes ? 1u : 0u, table_elements, 0};
    return call(mb, HS_MB_OP_GATHER, a);
}

int hs_mailbox_chain(hs_mailbox* mb, uint32_t in_gpu, uint32_t out_gpu, uint32_t elements, uint32_t k, float c1,
                     float c0) {
    uint32_t per = 32 * mb->batches;
    if (!mb->dev || elements % per || k % 4) return -EINVAL;
    uint32_t c1b, c0b;
    memcpy(&c1b, &c1, 4);
    memcpy(&c0b, &c0, 4);
    uint32_t a[7] = {elements / per, in_gpu, 0, out_gpu, k / 4, c1b, c0b};
    return call(mb, HS_MB_OP_CHAIN, a);
}

int hs_mailbox_stop(hs_mailbox* mb) {
    if (!mb->dev || !mb->running) return 0;
    const uint64_t tmo = 1000000000ull;
    int e;
    if (!mb->stop_acked) {
        if (!mb->stop_sent) { /* a stop request is sent once: after it, leases end without polling */
            if (mb->seq != mb->done_seq && (e = hs_mailbox_wait(mb, mb->seq, tmo))) return e;
            int64_t n = send_raw(mb, HS_MB_OP_STOP, NULL);
            if (n < 0) return (int)n;
            mb->stop_sent = 1;
            if (mb->test_stop_fail_once) { /* tests: as if the wait below had timed out */
                mb->test_stop_fail_once = 0;
                return -ETIME;
            }
        }
        if ((e = hs_mailbox_wait(mb, mb->seq, tmo))) return e;
        mb->stop_acked = 1;
    }
    if ((e = hs_v3d_bo_wait(mb->dev, &mb->ctl, 5000000000ull))) return e;
    mb->stop_sent = mb->stop_acked = 0;
    mb->running = 0;
    return 0;
}

void hs_mailbox_free(hs_mailbox* mb) {
    if (mb->dev) {
        if (mb->running) hs_mailbox_stop(mb);
        if (mb->wd_started) {
            pthread_mutex_lock(&mb->wd_mu);
            atomic_store(&mb->wd_quit, 1);
            pthread_cond_signal(&mb->wd_cv);
            pthread_mutex_unlock(&mb->wd_mu);
            pthread_join(mb->wd_thread, NULL);
            pthread_mutex_destroy(&mb->wd_mu);
            pthread_cond_destroy(&mb->wd_cv);
        }
        if (mb->ctl.map) ctl32(mb, OFF_M)[1] = ctl32(mb, OFF_M)[3] = 1; /* leases a failed stop left: end now */
        hs_v3d_bo_free(mb->dev, &mb->code);
        hs_v3d_bo_free(mb->dev, &mb->unif);
        hs_v3d_bo_free(mb->dev, &mb->ctl);
    }
    memset(mb, 0, sizeof(*mb));
}

int hs_mailbox_test_seek(hs_mailbox* mb, uint32_t seq) {
    if (!mb->dev || mb->running || hs_v3d_bo_wait(mb->dev, &mb->ctl, 0)) return -EBUSY;
    for (uint32_t b = 0; b < mb->batches; b++) fill_line(mb, OFF_ACK + 256 * b, seq);
    ctl32(mb, OFF_M)[0] = seq;
    barrier();
    mb->seq = mb->done_seq = seq;
    return 0;
}

void hs_mailbox_test_count_lease(hs_mailbox* mb, uint32_t b) {
    volatile uint32_t* w = ctl32(mb, OFF_STATE + 256 * b + 128);
    uint32_t v = w[0] + 1;
    for (int l = 0; l < 16; l++) w[l] = v;
    barrier();
}

int64_t hs_mailbox_test_send_unchecked(hs_mailbox* mb, uint32_t opc, const uint32_t a[7]) { return send_raw(mb, opc, a); }
void hs_mailbox_test_patch_request(hs_mailbox* mb, uint32_t seq, uint32_t line, uint32_t v) {
    fill_line(mb, OFF_SLOTS + ((seq - 1) & (mb->slots - 1)) * SLOT_BYTES + 64 * line, v);
    barrier();
}
void hs_mailbox_test_set_ack(hs_mailbox* mb, uint32_t b, uint32_t seq) {
    fill_line(mb, OFF_ACK + 256 * b, seq);
    barrier();
}
