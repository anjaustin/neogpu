/* hs_mailbox: see include/hs_mailbox.h. Register use of the resident program
 * (per batch, 16 lanes; l = lane, b = batch):
 *   rf0 M+4l   rf1 ACK_b+4l   rf3 budget   rf4 next sequence   rf5 lane-0 mask
 *   rf6 STATE_b+4l   rf7 SLOT0+4l   rf8 SLOTEND+4l   rf9 b   rf10 requests left
 *   rf11 slot+4l   rf14 leases   rf15/rf16 state lines 1/2   rf17 64
 *   rf22 64*batches (chunk stride)   rf23 64*b+4l   rf24..rf31 request lines
 *   rf33..rf43 op temporaries
 */
#define _GNU_SOURCE
#include "hs_mailbox.h"

#include <errno.h>
#include <string.h>
#include <time.h>

#include "hs_qpu.h"

#define R(n) HS_QPU_R(n)
#define RF(n) HS_QPU_RF(n)
#define X HS_QPU_NOREG
#define I(v) hs_qpu_imm_int(v)

enum {
    OFF_M = 0,
    OFF_ACK = 4096,   /* + 256 * batch */
    OFF_STATE = 8192, /* + 256 * batch: lines next seq, slot offset, leases */
    OFF_WARM = 12288,
    OFF_SLOTS = 16384,
    SLOT_BYTES = 512, /* 8 lines; a multiple of the GPU's 256-byte fetch */
    MAX_UNITS = 65536, /* budget units one request may take per batch */
    HEADER = 9,
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

static int build(hs_qpu_prog* p) {
    hs_qpu_prog_rules(p, 0); /* TMUAU and multi-load forms are outside the checker; forms measured in P0-5 */
    ldunifrf(p, 0);
    ldunifrf(p, 1);
    ldunifrf(p, 3);
    ldunifrf(p, 6);
    ldunifrf(p, 7);
    ldunifrf(p, 8);
    ldunifrf(p, 10);
    ldunifrf(p, 22);
    ldunifrf(p, 2); /* batches - 1: the batch-index mask (1 batch: its tidx field is not 0) */
    op(p, HS_QPU_A_EIDX, R(1), X, X);
    op(p, HS_QPU_A_SHL, R(0), R(1), I(2)); /* r0 = 4l */
    op(p, HS_QPU_A_SUB, R(2), R(1), I(1));
    op(p, HS_QPU_A_ASR, RF(5), R(2), I(15)); /* lane 0: ~0; others: 0 */
    op(p, HS_QPU_A_TIDX, R(2), X, X);
    op(p, HS_QPU_A_SHR, R(2), R(2), I(2));
    op(p, HS_QPU_A_BAND, RF(9), R(2), RF(2)); /* batch */
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
    op(p, HS_QPU_A_ADD, RF(15), RF(6), RF(17));
    op(p, HS_QPU_A_ADD, RF(16), RF(15), RF(17));
    op(p, HS_QPU_A_ADD, RF(43), RF(22), RF(22));
    load(p, RF(6), RF(4));
    load(p, RF(15), R(3));
    op(p, HS_QPU_A_ADD, RF(11), RF(7), R(3));
    load(p, RF(16), RF(14));

    int poll = hs_qpu_label_new(p), done = hs_qpu_label_new(p), quit = hs_qpu_label_new(p);
    int l_gather = hs_qpu_label_new(p), l_gstatic = hs_qpu_label_new(p), l_gpub = hs_qpu_label_new(p);
    int l_chain = hs_qpu_label_new(p), l_stop = hs_qpu_label_new(p), l_ack = hs_qpu_label_new(p);
    int l_cl = hs_qpu_label_new(p), l_hl = hs_qpu_label_new(p);
    flags(p, HS_QPU_A_MOV, R(1), RF(4), X, HS_QPU_PF_PUSHZ); /* next sequence 0: stopped */
    br(p, HS_QPU_B_A0, quit, 0);

    /* poll until M >= next sequence (lane 0 decides) or the budget runs out */
    hs_qpu_label_bind(p, poll);
    op2(p, RF(0));
    load(p, RF(0), R(0));
    flags(p, HS_QPU_A_SUB, RF(3), RF(3), I(1), HS_QPU_PF_PUSHZ);
    hs_qpu_branch(p, HS_QPU_B_A0, done, 0, 0);
    op(p, HS_QPU_A_SUB, R(1), R(0), RF(4));
    flags(p, HS_QPU_A_BAND, R(1), R(1), RF(5), HS_QPU_PF_PUSHN);
    nop(p);
    br(p, HS_QPU_B_ANYA, poll, 0);

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
    hs_qpu_label_bind(p, l_hl);
    chain_i1(p);
    chain_i2(p);
    chain_i1(p);
    chain_i2(p);
    chain_i1(p);
    flags(p, HS_QPU_A_SUB, RF(38), RF(38), I(1), HS_QPU_PF_PUSHZ);
    hs_qpu_branch(p, HS_QPU_B_NA0, l_hl, 0, 0);
    chain_i2(p); /* delay slots: run on both paths */
    chain_i1(p);
    chain_i2(p);
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

    /* stop: acknowledge, then store next sequence 0 so queued leases quit */
    hs_qpu_label_bind(p, l_stop);
    store(p, RF(1), RF(4));
    tmuwt(p);
    op2(p, RF(1));
    mov(p, RF(4), I(0));
    br(p, HS_QPU_B_ALWAYS, done, 0);

    /* acknowledge and move to the next request */
    hs_qpu_label_bind(p, l_ack);
    store(p, RF(1), RF(4));
    tmuwt(p);
    op2(p, RF(1));
    budget_sub(p, I(1));
    op(p, HS_QPU_A_ADD, RF(4), RF(4), I(1));
    op(p, HS_QPU_A_ADD, RF(11), RF(11), RF(41));
    flags(p, HS_QPU_A_SUB, R(1), RF(11), RF(8), HS_QPU_PF_PUSHZ);
    hs_qpu_alu w = hs_qpu_a(HS_QPU_A_MOV, RF(11), RF(7), X);
    w.ac = HS_QPU_COND_IFA;
    emit(p, w);
    flags(p, HS_QPU_A_SUB, RF(10), RF(10), I(1), HS_QPU_PF_PUSHZ);
    hs_qpu_branch(p, HS_QPU_B_A0, done, 0, 0);
    flags(p, HS_QPU_A_SUB, R(1), RF(3), I(1), HS_QPU_PF_PUSHN); /* delay slot: budget <= 0 */
    nop(p);
    nop(p);
    br(p, HS_QPU_B_A0, done, 0);
    br(p, HS_QPU_B_ALWAYS, poll, 0);

    /* lease end: store the state the next lease starts from */
    hs_qpu_label_bind(p, done);
    store(p, RF(6), RF(4));
    op(p, HS_QPU_A_SUB, R(1), RF(11), RF(7));
    store(p, RF(15), R(1));
    op(p, HS_QPU_A_ADD, RF(14), RF(14), I(1));
    store(p, RF(16), RF(14));
    tmuwt(p);
    hs_qpu_label_bind(p, quit);
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

uint32_t hs_mailbox_leases_done(const hs_mailbox* mb, uint32_t b) {
    return ctl32(mb, OFF_STATE + 256 * b + 128)[0];
}

static int dispatch(hs_mailbox* mb, uint32_t code, uint32_t unif_gpu, uint32_t batches) {
    hs_v3d_dispatch_desc d = {code, unif_gpu, batches, mb->handles, mb->nhandles, 0};
    return hs_v3d_dispatch(mb->dev, &d);
}
static int top_up(hs_mailbox* mb) {
    uint32_t ended = hs_mailbox_leases_done(mb, 0);
    while (mb->submitted - ended < mb->qdepth) {
        int e = dispatch(mb, mb->prog, mb->unif.gpu, mb->batches);
        if (e) return e;
        mb->submitted++;
    }
    return 0;
}

int hs_mailbox_start(hs_mailbox* mb, hs_v3d* dev, const hs_mailbox_cfg* cfg) {
    memset(mb, 0, sizeof(*mb));
    if (!cfg || (cfg->batches != 1 && cfg->batches != 8) || cfg->nbos > 13 || (cfg->nbos && !cfg->bos))
        return -EINVAL;
    mb->dev = dev;
    mb->batches = cfg->batches;
    mb->budget = cfg->budget ? cfg->budget : 100000;
    mb->qdepth = cfg->qdepth ? cfg->qdepth : 2;
    mb->slots = cfg->slots ? cfg->slots : 256;
    if (mb->budget < 2 || mb->budget > 4000000 || mb->qdepth > 8 || mb->slots < 2 || mb->slots > 65536) return -EINVAL;
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
    for (uint32_t i = 0; i < cfg->nbos; i++) mb->handles[3 + i] = cfg->bos[i];
    mb->nhandles = 3 + cfg->nbos;
    static uint64_t buf[2048];
    hs_qpu_prog p;
    hs_qpu_prog_init(&p, buf, 2048);
    int n = build(&p);
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
    u[6] = mb->slots;
    u[7] = 64 * mb->batches;
    u[8] = mb->batches - 1;
    for (size_t i = HEADER; i < nunif; i++) u[i] = 0xFFFFFF80u | 2u << 3 | 7u; /* per-lane op 2, 32-bit */
    memset((void*)mb->ctl.map, 0, mb->ctl.size);
    for (uint32_t b = 0; b < mb->batches; b++) fill_line(mb, OFF_STATE + 256 * b, 1);
    ctl32(mb, OFF_WARM)[0] = 37500; /* ~1 ms: the GPU clock drops after idle (HARDWARE_MAP) */
    barrier();
    if ((e = dispatch(mb, warm, mb->ctl.gpu + OFF_WARM, 1)) ||
        (e = hs_v3d_bo_wait(dev, &mb->ctl, 2000000000ull))) {
        hs_mailbox_free(mb);
        return e;
    }
    mb->running = 1;
    return 0;
}

int64_t hs_mailbox_send(hs_mailbox* mb, uint32_t opc, const uint32_t a[7]) {
    if (mb->seq - mb->done_seq >= mb->slots) return -EBUSY;
    uint32_t n = mb->seq + 1;
    if (!n) return -EOVERFLOW;
    if (!mb->running) { /* restart after a stop: queued leases are gone */
        for (uint32_t b = 0; b < mb->batches; b++) {
            fill_line(mb, OFF_STATE + 256 * b, n);
            fill_line(mb, OFF_STATE + 256 * b + 64, ((n - 1) % mb->slots) * SLOT_BYTES);
        }
        mb->submitted = hs_mailbox_leases_done(mb, 0);
        mb->running = 1;
    }
    uint32_t slot = OFF_SLOTS + ((n - 1) % mb->slots) * SLOT_BYTES;
    fill_line(mb, slot, opc);
    for (int j = 0; j < 7; j++) fill_line(mb, slot + 64 * (1 + j), a ? a[j] : 0);
    barrier();
    ctl32(mb, OFF_M)[0] = n;
    barrier();
    mb->seq = n;
    int e = top_up(mb);
    return e ? e : (int64_t)n;
}

int hs_mailbox_wait(hs_mailbox* mb, uint32_t seq, uint64_t timeout_ns) {
    uint64_t t0 = now_ns(), t;
    uint32_t lease0 = hs_mailbox_leases_done(mb, 0);
    for (uint32_t spin = 0;; spin++) {
        uint32_t b = 0;
        while (b < mb->batches && (int32_t)(ctl32(mb, OFF_ACK + 256 * b)[0] - seq) >= 0) b++;
        if (b == mb->batches) break;
        if ((spin & 63) == 63) {
            int e = top_up(mb);
            if (e) return e;
            if ((t = now_ns()) - t0 > timeout_ns) return -ETIME;
        }
    }
    t = now_ns();
    if (t - t0 > 20000) mb->waits_over_20us++;
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

int hs_mailbox_gather(hs_mailbox* mb, uint32_t idx_gpu, uint32_t table_gpu, uint32_t out_gpu, uint32_t elements,
                      int table_changes) {
    uint32_t per = 16 * mb->batches;
    if (elements % per || ((idx_gpu | table_gpu | out_gpu) & 3)) return -EINVAL;
    uint32_t chunks = elements / per;
    if ((uint64_t)chunks * (table_changes ? 3 : 2) + 4 > MAX_UNITS) return -EINVAL;
    uint32_t a[7] = {chunks, idx_gpu, table_gpu, out_gpu, table_changes ? 1u : 0u, 0, 0};
    return call(mb, HS_MB_OP_GATHER, a);
}

int hs_mailbox_chain(hs_mailbox* mb, uint32_t in_gpu, uint32_t out_gpu, uint32_t elements, uint32_t k, float c1,
                     float c0) {
    uint32_t per = 32 * mb->batches;
    if (elements % per || k < 4 || k % 4 || k > 65536 || ((in_gpu | out_gpu) & 3)) return -EINVAL;
    uint32_t pairs = elements / per;
    if ((uint64_t)pairs * (k / 4 + 4) + 4 > MAX_UNITS) return -EINVAL;
    uint32_t c1b, c0b;
    memcpy(&c1b, &c1, 4);
    memcpy(&c0b, &c0, 4);
    uint32_t a[7] = {pairs, in_gpu, 0, out_gpu, k / 4, c1b, c0b};
    return call(mb, HS_MB_OP_CHAIN, a);
}

int hs_mailbox_stop(hs_mailbox* mb) {
    if (!mb->running) return 0;
    int e = call(mb, HS_MB_OP_STOP, NULL);
    int w = hs_v3d_bo_wait(mb->dev, &mb->ctl, 5000000000ull);
    mb->running = 0;
    return e ? e : w;
}

void hs_mailbox_free(hs_mailbox* mb) {
    if (mb->dev) {
        if (mb->running) hs_mailbox_stop(mb);
        hs_v3d_bo_free(mb->dev, &mb->code);
        hs_v3d_bo_free(mb->dev, &mb->unif);
        hs_v3d_bo_free(mb->dev, &mb->ctl);
    }
    memset(mb, 0, sizeof(*mb));
}
