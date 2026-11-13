/* tmuop_probe: decode, by measurement, the per-lookup TMU op field of a
 * general (tmuau) TMU access on V3D 4.2, and what each op does to the
 * caches between the CPU and a running QPU program.
 *
 * Hypothesis under test (read from Mesa's MIT hardware description, not
 * trusted until measured here): the config word that tmuau takes from the
 * uniform stream is 0xFFFFFF00 | perpixel<<7 | op<<3 | type, with
 * type 7 = 32-bit, op 15 = regular; for reads op 0 prefetch, 1 clear,
 * 2/3 flush, 4 full L1 clear, 8 inc, 9 dec, 10 not.
 *
 *   P  per op: does the op return data (zero-hang form: op, then a regular
 *      load of a known value; a second ldtmu only if the first did not
 *      return the known value). Memory after the job shows atomics.
 *   R  per op: CPU write mid-job, then load / op / load (CPU->GPU).
 *   W  per op: store + tmuwt + op, then a long spin (GPU->CPU).
 * Usage: tools/tmuop_probe SECTIONS [ops,comma,separated]
 * Run inside the GPU lock.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "hs_qpu.h"
#include "hs_v3d.h"

#define R(n) HS_QPU_R(n)
#define RF(n) HS_QPU_RF(n)
#define X HS_QPU_NOREG
#define I(v) hs_qpu_imm_int(v)

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
static void die(const char* what, int e) {
    fprintf(stderr, "%s: %s\n", what, strerror(e < 0 ? -e : e));
    exit(1);
}
static void emit(hs_qpu_prog* p, hs_qpu_alu x) { hs_qpu_emit(p, &x); }
static void nop(hs_qpu_prog* p) { emit(p, hs_qpu_alu_nop()); }
static void ldunifrf(hs_qpu_prog* p, int rf) { emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDUNIFRF, RF(rf))); }
static void mov(hs_qpu_prog* p, hs_qpu_reg d, hs_qpu_reg s) { emit(p, hs_qpu_a(HS_QPU_A_MOV, d, s, X)); }
static void add(hs_qpu_prog* p, hs_qpu_reg d, hs_qpu_reg a, hs_qpu_reg b) { emit(p, hs_qpu_a(HS_QPU_A_ADD, d, a, b)); }
static void dec_pushz(hs_qpu_prog* p, int rf) {
    hs_qpu_alu s = hs_qpu_a(HS_QPU_A_SUB, RF(rf), RF(rf), I(1));
    s.af = HS_QPU_PF_PUSHZ;
    emit(p, s);
}
static void branch(hs_qpu_prog* p, int cond, int label) {
    hs_qpu_branch(p, cond, label, 0, 0);
    nop(p);
    nop(p);
    nop(p);
}
/* r0 = 4 * lane; rf(d) = rf(d) + r0 for each listed rf */
static void lane_offsets(hs_qpu_prog* p, const int* rfs, int n) {
    emit(p, hs_qpu_a(HS_QPU_A_EIDX, R(1), X, X));
    emit(p, hs_qpu_a(HS_QPU_A_SHL, R(0), R(1), I(2)));
    for (int i = 0; i < n; i++) add(p, RF(rfs[i]), RF(rfs[i]), R(0));
}
/* rf(d) = rf(s) + 64 * k */
static void plus64k(hs_qpu_prog* p, int d, int s, int k) {
    emit(p, hs_qpu_a(HS_QPU_A_MOV, R(2), I(k), X));
    emit(p, hs_qpu_a(HS_QPU_A_SHL, R(2), R(2), I(6)));
    add(p, RF(d), RF(s), R(2));
}
/* regular load, measured thrsw form, into dst */
static void load(hs_qpu_prog* p, int addr_rf, hs_qpu_reg dst) {
    emit(p, hs_qpu_s(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(addr_rf), X), HS_QPU_SIG_THRSW, X));
    nop(p);
    nop(p);
    emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDTMU, dst));
}
static void store(hs_qpu_prog* p, int addr_rf, hs_qpu_reg v) {
    mov(p, HS_QPU_TMUD, v);
    mov(p, HS_QPU_TMUA, RF(addr_rf));
}
/* the op lookup: tmuau takes its config from the next uniform. ret: the op
 * returns one word per lane (read it, measured form); else tmuwt after it. */
static void op_lookup(hs_qpu_prog* p, int addr_rf, int ret, hs_qpu_reg dst) {
    if (ret) {
        emit(p, hs_qpu_s(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUAU, RF(addr_rf), X), HS_QPU_SIG_THRSW, X));
        nop(p);
        nop(p);
        emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDTMU, dst));
    } else {
        mov(p, HS_QPU_TMUAU, RF(addr_rf));
        emit(p, hs_qpu_a(HS_QPU_A_TMUWT, X, X, X));
    }
}
static void spin(hs_qpu_prog* p, int count_rf) {
    mov(p, RF(20), RF(count_rf));
    int l = hs_qpu_label_new(p);
    hs_qpu_label_bind(p, l);
    dec_pushz(p, 20);
    branch(p, HS_QPU_B_NA0, l);
}

/* P. uniforms: A, B, OUT, KB, cfg(tmuau). out[0..15] = first ldtmu,
 * out[16..31] = second ldtmu (only if the first was not KB+lane) */
static int prog_ret(hs_qpu_prog* p) {
    hs_qpu_prog_rules(p, 0);
    ldunifrf(p, 0);
    ldunifrf(p, 1);
    ldunifrf(p, 2);
    ldunifrf(p, 4);
    static const int rfs[] = {0, 1, 2};
    lane_offsets(p, rfs, 3);
    add(p, RF(4), RF(4), R(1));
    plus64k(p, 5, 2, 1);
    mov(p, R(3), I(0));
    mov(p, HS_QPU_TMUAU, RF(0)); /* the op under test */
    load(p, 1, R(1));            /* the known value */
    hs_qpu_alu x = hs_qpu_a(HS_QPU_A_BXOR, R(2), R(1), RF(4));
    x.af = HS_QPU_PF_PUSHZ;
    emit(p, x);
    int skip = hs_qpu_label_new(p);
    branch(p, HS_QPU_B_ALLA, skip);
    emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDTMU, R(3)));
    hs_qpu_label_bind(p, skip);
    store(p, 2, R(1));
    store(p, 5, R(3));
    emit(p, hs_qpu_a(HS_QPU_A_TMUWT, X, X, X));
    hs_qpu_emit_end(p);
    return hs_qpu_prog_finish(p);
}

/* R. uniforms: M, OUT, SPIN, cfg. out row 0 = load before the CPU write,
 * row 1 = load after it, row 2 = op data (if it returns), row 3 = load
 * after the op */
static int g_ret, g_single;
static uint32_t g_opoff;
static int prog_r(hs_qpu_prog* p) {
    hs_qpu_prog_rules(p, 0);
    ldunifrf(p, 0);
    ldunifrf(p, 2);
    ldunifrf(p, 3);
    ldunifrf(p, 7); /* op address offset */
    static const int rfs[] = {0, 2};
    lane_offsets(p, rfs, 2);
    add(p, RF(7), RF(7), RF(0));
    load(p, 0, RF(10));
    spin(p, 3);
    load(p, 0, RF(11));
    mov(p, RF(12), I(0));
    op_lookup(p, 7, g_ret, RF(12));
    load(p, 0, RF(13));
    for (int k = 0; k < 4; k++) {
        plus64k(p, 6, 2, k);
        store(p, 6, RF(10 + k));
    }
    emit(p, hs_qpu_a(HS_QPU_A_TMUWT, X, X, X));
    hs_qpu_emit_end(p);
    return hs_qpu_prog_finish(p);
}

/* W. uniforms: S, SPIN, cfg. S = 1, tmuwt, op on S, spin, S = 2. */
static int prog_w(hs_qpu_prog* p) {
    hs_qpu_prog_rules(p, 0);
    ldunifrf(p, 0);
    ldunifrf(p, 3);
    ldunifrf(p, 7); /* op address offset */
    static const int rfs[] = {0};
    lane_offsets(p, rfs, 1);
    add(p, RF(7), RF(7), RF(0));
    mov(p, RF(8), I(1));
    store(p, 0, RF(8));
    emit(p, hs_qpu_a(HS_QPU_A_TMUWT, X, X, X));
    op_lookup(p, 7, g_ret, RF(12));
    spin(p, 3);
    if (!g_single) {
        mov(p, RF(8), I(2));
        store(p, 0, RF(8));
        emit(p, hs_qpu_a(HS_QPU_A_TMUWT, X, X, X));
    }
    hs_qpu_emit_end(p);
    return hs_qpu_prog_finish(p);
}


/* M. mailbox ping-pong. uniforms: M, A, MSGS, BUDGET, then one cfg per op
 * lookup. For n = 1..MSGS: poll M (op on M, load M) until it reads n, then
 * A[lane] = n, tmuwt, op on A. BUDGET bounds the total polls (and so the
 * job's run time and the uniforms used). g_wt: tmuwt after each op. */
static int g_wt = 1;
static void op_nr(hs_qpu_prog* p, int addr_rf) {
    mov(p, HS_QPU_TMUAU, RF(addr_rf));
    if (g_wt) emit(p, hs_qpu_a(HS_QPU_A_TMUWT, X, X, X));
}
/* g_payload: a 16-word message. The CPU writes words 1..15 (n*16 + l), then
 * word 0 (n*16) last. Each lane ops and loads its own word; lane 0 alone
 * decides (the rest are masked off); the echo is word + 1 in every lane. */
static int g_payload;
static int prog_m(hs_qpu_prog* p) {
    hs_qpu_prog_rules(p, 0);
    ldunifrf(p, 0);
    ldunifrf(p, 1);
    ldunifrf(p, 2);
    ldunifrf(p, 3);
    if (g_payload) {
        static const int rfs[] = {0, 1};
        lane_offsets(p, rfs, 2); /* r1 = lane */
        emit(p, hs_qpu_a(HS_QPU_A_SUB, R(2), R(1), I(1)));
        emit(p, hs_qpu_a(HS_QPU_A_ASR, RF(5), R(2), I(15))); /* lane 0: -1 -> ~0; lanes 1..15: 0..14 -> 0 */
        emit(p, hs_qpu_a(HS_QPU_A_MOV, R(2), I(1), X));
        emit(p, hs_qpu_a(HS_QPU_A_SHL, RF(6), R(2), I(4))); /* 16 */
        add(p, RF(4), RF(6), R(1));                          /* n = 1: 16 + lane */
    } else {
        static const int rfs[] = {1};
        lane_offsets(p, rfs, 1);
        mov(p, RF(4), I(1));
    }
    int outer = hs_qpu_label_new(p), poll = hs_qpu_label_new(p);
    hs_qpu_label_bind(p, outer);
    hs_qpu_label_bind(p, poll);
    op_nr(p, 0);
    load(p, 0, R(0));
    emit(p, hs_qpu_a(HS_QPU_A_SUB, RF(3), RF(3), I(1)));
    emit(p, hs_qpu_a(HS_QPU_A_BXOR, R(1), R(0), RF(4)));
    if (g_payload) emit(p, hs_qpu_a(HS_QPU_A_BAND, R(1), R(1), RF(5)));
    hs_qpu_alu m = hs_qpu_a(HS_QPU_A_UMIN, R(2), R(1), RF(3));
    m.af = HS_QPU_PF_PUSHZ;
    emit(p, m);
    branch(p, g_payload ? HS_QPU_B_ANYNA : HS_QPU_B_NA0, poll);
    if (g_payload) add(p, R(0), R(0), I(1));
    store(p, 1, R(0));
    emit(p, hs_qpu_a(HS_QPU_A_TMUWT, X, X, X));
    op_nr(p, 1);
    if (g_payload)
        add(p, RF(4), RF(4), RF(6));
    else
        add(p, RF(4), RF(4), I(1));
    emit(p, hs_qpu_a(HS_QPU_A_SUB, RF(2), RF(2), I(1)));
    m = hs_qpu_a(HS_QPU_A_UMIN, R(2), RF(2), RF(3));
    m.af = HS_QPU_PF_PUSHZ;
    emit(p, m);
    branch(p, HS_QPU_B_NA0, outer);
    emit(p, hs_qpu_a(HS_QPU_A_TMUWT, X, X, X));
    hs_qpu_emit_end(p);
    return hs_qpu_prog_finish(p);
}

static hs_v3d dev;
static hs_v3d_bo code, unif, data;
static uint32_t handles[3];
static uint64_t prog[1024];
static size_t code_off;

static uint32_t load_prog(int (*build)(hs_qpu_prog*)) {
    hs_qpu_prog p;
    hs_qpu_prog_init(&p, prog, 1024);
    int n = build(&p);
    if (n < 0) {
        fprintf(stderr, "program build failed: %d at %u\n", n, p.rule_at);
        exit(1);
    }
    uint32_t a = hs_v3d_load(&code, code_off, prog, (uint32_t)n);
    code_off += ((size_t)n * 8 + 255) & ~(size_t)255;
    if (!a) die("load", ENOSPC);
    return a;
}
static int dispatch(uint32_t code_gpu, uint32_t unif_off) {
    hs_v3d_dispatch_desc d = {code_gpu, unif.gpu + unif_off, 1, handles, 3, 0};
    return hs_v3d_dispatch(&dev, &d);
}
static void wbarrier(void) { __asm__ volatile("dsb sy" ::: "memory"); }
/* GPU resets so far, counted in C from `sudo -n dmesg`; -1 unless dmesg
 * itself succeeded (a pipe into grep -c printed 0 on failure: red team 4).
 * Limitation: counts the lines still in the kernel ring buffer. */
static int resets(void) {
    FILE* f = popen("sudo -n dmesg 2>/dev/null", "r");
    if (!f) return -1;
    char line[512];
    int n = 0, lines = 0;
    while (fgets(line, sizeof(line), f)) {
        lines++;
        if (strstr(line, "Resetting GPU for hang")) n++;
    }
    int st = pclose(f);
    return st == 0 && lines > 0 ? n : -1;
}
static const char* reset_delta(int r0, int r1) {
    static char b[4][32];
    static int k;
    char* o = b[k++ & 3];
    if (r0 < 0 || r1 < 0) return "unknown";
    snprintf(o, 32, "%d", r1 - r0);
    return o;
}
static uint32_t cfg_of(int op) { return 0xFFFFFF80u | (uint32_t)op << 3 | 7u; }
static uint32_t* W32(size_t off) { return (uint32_t*)(data.map + off); }
#define SPIN_1MS 37500u
enum { OFF_A = 0, OFF_B = 4096, OFF_OUT = 8192, OFF_M = 12288, OFF_S = 16384 };

static void section_p(int op) {
    int r0 = resets();
    uint32_t pg = load_prog(prog_ret);
    for (int l = 0; l < 16; l++) {
        W32(OFF_A)[l] = 0x0A000000u + 0x100u * (uint32_t)l;
        W32(OFF_B)[l] = 0x0B000000u + (uint32_t)l;
        W32(OFF_OUT)[l] = W32(OFF_OUT)[16 + l] = 0xDEADBEEFu;
    }
    wbarrier();
    uint32_t* u = (uint32_t*)unif.map;
    u[0] = data.gpu + OFF_A;
    u[1] = data.gpu + OFF_B;
    u[2] = data.gpu + OFF_OUT;
    u[3] = 0x0B000000u;
    u[4] = cfg_of(op);
    uint64_t t0 = now_ns();
    int e = dispatch(pg, 0);
    if (e) die("dispatch", e);
    e = hs_v3d_bo_wait(&dev, &data, 5000000000ull);
    double ms = (now_ns() - t0) / 1e6;
    int nk = 0, nA = 0;
    for (int l = 0; l < 16; l++) {
        nk += W32(OFF_OUT)[l] == 0x0B000000u + (uint32_t)l;
        nA += W32(OFF_OUT)[l] == 0x0A000000u + 0x100u * (uint32_t)l;
    }
    printf("P op %2d cfg 0x%08x: %.2f ms%s resets %s | ldtmu1 lanes0-3 %08x %08x %08x %08x (==B in %d/16, ==A in %d/16)"
           " | ldtmu2 lanes0-3 %08x %08x %08x %08x | A after lanes0-3 %08x %08x %08x %08x\n",
           op, cfg_of(op), ms, e ? " WAIT_ERR" : "", reset_delta(r0, resets()), W32(OFF_OUT)[0], W32(OFF_OUT)[1],
           W32(OFF_OUT)[2], W32(OFF_OUT)[3], nk, nA, W32(OFF_OUT)[16], W32(OFF_OUT)[17], W32(OFF_OUT)[18],
           W32(OFF_OUT)[19], W32(OFF_A)[0], W32(OFF_A)[1], W32(OFF_A)[2], W32(OFF_A)[3]);
}

static void section_r(int op, int ret, int runs) {
    g_ret = ret;
    uint32_t pg = load_prog(prog_r);
    for (int run = 0; run < runs; run++) {
        int r0 = resets();
        for (int l = 0; l < 16; l++) W32(OFF_M)[l] = 0x11110000u + (uint32_t)l;
        memset(W32(OFF_OUT), 0xEE, 256);
        wbarrier();
        uint32_t* u = (uint32_t*)unif.map + 64;
        u[0] = data.gpu + OFF_M;
        u[1] = data.gpu + OFF_OUT;
        u[2] = 60 * SPIN_1MS;
        u[3] = g_opoff;
        u[4] = cfg_of(op);
        uint64_t t0 = now_ns();
        int e = dispatch(pg, 256);
        if (e) die("dispatch", e);
        while (now_ns() - t0 < 30000000ull) {
        }
        for (int l = 0; l < 16; l++) W32(OFF_M)[l] = 0x22220000u + (uint32_t)l;
        wbarrier();
        e = hs_v3d_bo_wait(&dev, &data, 5000000000ull);
        double ms = (now_ns() - t0) / 1e6;
        printf("R op %2d ret %d opoff %u run %d: %.1f ms%s resets %s | before %08x | after-write %08x | op-data %08x | after-op "
               "%08x (lane15 %08x)\n",
               op, ret, g_opoff, run, ms, e ? " WAIT_ERR" : "", reset_delta(r0, resets()), W32(OFF_OUT)[0], W32(OFF_OUT)[16],
               W32(OFF_OUT)[32], W32(OFF_OUT)[48], W32(OFF_OUT)[63]);
    }
}

static void section_w(int op, int ret, int runs) {
    g_ret = ret;
    uint32_t pg = load_prog(prog_w);
    volatile uint32_t* s = W32(OFF_S);
    for (int run = 0; run < runs; run++) {
        int r0 = resets();
        for (int l = 0; l < 16; l++) s[l] = 0;
        wbarrier();
        uint32_t* u = (uint32_t*)unif.map + 128;
        u[0] = data.gpu + OFF_S;
        u[1] = 60 * SPIN_1MS;
        u[2] = g_opoff;
        u[3] = cfg_of(op);
        uint64_t t0 = now_ns(), t1 = 0, t2 = 0;
        int e = dispatch(pg, 512);
        if (e) die("dispatch", e);
        while (now_ns() - t0 < 2000000000ull) {
            uint32_t v = s[0];
            if (v == 1 && !t1) t1 = now_ns() - t0;
            if (v == 2) {
                t2 = now_ns() - t0;
                break;
            }
        }
        e = hs_v3d_bo_wait(&dev, &data, 5000000000ull);
        double ms = (now_ns() - t0) / 1e6;
        printf("W%s op %2d ret %d opoff %u run %d: job %.1f ms%s resets %s | CPU saw 1 at %s%.3f ms, 2 at %.3f ms; final lane0 %u "
               "lane15 %u\n",
               g_single ? "1" : "", op, ret, g_opoff, run, ms, e ? " WAIT_ERR" : "", reset_delta(r0, resets()), t1 ? "" : "never ", t1 / 1e6, t2 / 1e6,
               s[0], s[15]);
    }
}


static int cmp_u64(const void* a, const void* b) {
    uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
    return x < y ? -1 : x > y;
}
/* MSGS messages (0: none, just measure the poll cost), BUDGET polls */
/* warm-up: spin ~ms on one QPU, so the timed job does not start on the
 * low clock the GPU drops to after >= 200 ms idle (HARDWARE_MAP) */
static int prog_warm(hs_qpu_prog* p) {
    ldunifrf(p, 0);
    int l = hs_qpu_label_new(p);
    hs_qpu_label_bind(p, l);
    dec_pushz(p, 0);
    branch(p, HS_QPU_B_NA0, l);
    hs_qpu_emit_end(p);
    return hs_qpu_prog_finish(p);
}
static void gpu_warm(uint32_t warm_gpu) {
    uint32_t* u = (uint32_t*)unif.map + 512; /* words 512..: clear of the M stream (1024..) */
    u[0] = SPIN_1MS;                          /* 5 instructions x 4 cycles per pass: ~1 ms */
    int e = dispatch(warm_gpu, 512 * 4);
    if (e) die("dispatch warm", e);
    hs_v3d_bo_wait(&dev, &data, 2000000000ull);
}

static void section_m(int op, int runs) {
    uint32_t msgs = getenv("MSGS") ? (uint32_t)atoi(getenv("MSGS")) : 1000;
    uint32_t budget = getenv("BUDGET") ? (uint32_t)atoi(getenv("BUDGET")) : 100000;
    uint32_t maxu = (uint32_t)(unif.size / 4) - 1024 - 8; /* stream starts 1024 words in */
    if (msgs > 200000) die("MSGS above 200000 (the RTT table)", EINVAL);
    if (budget < 2) die("BUDGET must be at least 2 (0 would wrap the GPU's poll counter)", EINVAL);
    /* the GPU consumes up to 4 + budget + msgs uniforms; all must be filled */
    if (4ull + budget + (msgs ? msgs : 1) > maxu) budget = maxu - 4 - (msgs ? msgs : 1);
    uint32_t warm_gpu = load_prog(prog_warm);
    g_wt = getenv("NOWT") == NULL;
    g_payload = getenv("PAYLOAD") != NULL;
    uint32_t pg = load_prog(prog_m);
    uint32_t torn = 0, torn_fixed = 0;
    volatile uint32_t* mb = W32(OFF_M);
    volatile uint32_t* ack = W32(OFF_S);
    static uint64_t rtt[200000];
    for (int run = 0; run < runs; run++) {
        int r0 = resets();
        for (int l = 0; l < 16; l++) mb[l] = ack[l] = 0;
        uint32_t* u = (uint32_t*)unif.map + 1024;
        u[0] = data.gpu + OFF_M;
        u[1] = data.gpu + OFF_S;
        u[2] = msgs ? msgs : 1;
        u[3] = budget;
        for (uint32_t i = 4; i < maxu; i++) u[i] = cfg_of(op);
        wbarrier();
        if (!msgs) {
            /* per-poll cost without start-up: the slope between a half and
             * a full budget, each job after a warm-up */
            double tj[2];
            for (int h = 0; h < 2; h++) {
                u[3] = h ? budget : budget / 2;
                wbarrier();
                gpu_warm(warm_gpu);
                uint64_t a0 = now_ns();
                int e = dispatch(pg, 4096);
                if (e) die("dispatch", e);
                hs_v3d_bo_wait(&dev, &data, 5000000000ull);
                tj[h] = (double)(now_ns() - a0);
            }
            printf("M op %2d wt %d run %d: no messages; %u vs %u polls: %.2f vs %.2f ms => %.1f ns per poll (slope, "
                   "start-up excluded); resets %s\n",
                   op, g_wt, run, budget / 2, budget, tj[0] / 1e6, tj[1] / 1e6,
                   (tj[1] - tj[0]) / (double)(budget - budget / 2), reset_delta(r0, resets()));
            continue;
        }
        gpu_warm(warm_gpu);
        uint64_t t0 = now_ns(), t_msg1 = 0;
        int e = dispatch(pg, 4096);
        if (e) die("dispatch", e);
        uint32_t done = 0;
        for (uint32_t n = 1; n <= msgs; n++) {
            uint64_t a = now_ns();
            uint32_t want = n;
            if (g_payload) {
                for (int l = 1; l < 16; l++) mb[l] = n * 16 + (uint32_t)l;
                wbarrier();
                mb[0] = n * 16;
                want = n * 16 + 1;
            } else {
                mb[0] = n;
            }
            wbarrier();
            uint64_t b;
            int ok = 0;
            do { /* the time is taken after the read that saw the ack */
                ok = ack[0] == want;
                b = now_ns();
            } while (!ok && b - a < 50000000ull);
            if (!ok) break;
            rtt[n - 1] = b - a;
            done = n;
            if (n == 1) t_msg1 = b; /* the exchange rate is measured from here: start-up excluded */
            if (g_payload) {
                int bad = 0;
                for (int l = 1; l < 16; l++) bad |= ack[l] != n * 16 + (uint32_t)l + 1;
                if (bad) {
                    torn++;
                    uint64_t c = now_ns();
                    while (now_ns() - c < 100000) {
                    }
                    int still = 0;
                    for (int l = 1; l < 16; l++) still |= ack[l] != n * 16 + (uint32_t)l + 1;
                    if (torn <= 3)
                        printf("   torn msg %u: lane1 %08x want %08x; after 100 us %s\n", n, ack[1], n * 16 + 2,
                               still ? "still wrong" : "correct");
                    torn_fixed += !still;
                }
            }
        }
        uint64_t tm = now_ns();
        e = hs_v3d_bo_wait(&dev, &data, 5000000000ull);
        double ms = (now_ns() - t0) / 1e6;

        /* stats over messages 2..done (message 1 includes job start-up) */
        uint32_t k = done > 1 ? done - 1 : 0;
        if (k) qsort(rtt + 1, k, sizeof(rtt[0]), cmp_u64);
        char m1[32];
        if (done) snprintf(m1, sizeof(m1), "%.1f us", rtt[0] / 1e3);
        else snprintf(m1, sizeof(m1), "n/a");
        printf("M%s op %2d wt %d run %d: %u of %u messages round-tripped; msg1 (incl. start-up) %s; RTT (msgs 2..) min %.2f "
               "p50 %.2f p90 %.2f p99 %.2f max %.2f us; %.0f msgs/s over msgs 2.. (start-up excluded); job %.2f ms%s "
               "resets %s; ack lane15 %u\n",
               g_payload ? "16" : "", op, g_wt, run, done, msgs, m1, k ? rtt[1] / 1e3 : 0, k ? rtt[1 + k / 2] / 1e3 : 0,
               k ? rtt[1 + k * 9 / 10] / 1e3 : 0, k ? rtt[1 + k * 99 / 100] / 1e3 : 0, k ? rtt[k] / 1e3 : 0,
               (k && tm > t_msg1) ? k / ((tm - t_msg1) / 1e9) : 0.0, ms, e ? " WAIT_ERR" : "", reset_delta(r0, resets()),
               ack[15]);
        if (g_payload) printf("   payload: %u torn messages (%u correct 100 us later)\n", torn, torn_fixed);
        torn = torn_fixed = 0;
    }
}

int main(int argc, char** argv) {
    const char* which = argc > 1 ? argv[1] : "P";
    int ops[32], nops = 0, rets[32];
    const char* list = argc > 2 ? argv[2] : "15";
    /* ops as N or N:r (r = the op returns data) */
    for (const char* q = list; *q && nops < 32;) {
        ops[nops] = (int)strtol(q, (char**)&q, 10);
        rets[nops] = 0;
        if (*q == ':') {
            rets[nops] = q[1] == 'r';
            q += 2;
        }
        nops++;
        if (*q == ',') q++;
    }
    int runs = argc > 3 ? atoi(argv[3]) : 3;
    g_opoff = getenv("OPOFF") ? (uint32_t)strtoul(getenv("OPOFF"), NULL, 0) : 0;
    g_single = getenv("SINGLE") != NULL;
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(2, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);
    setvbuf(stdout, NULL, _IOLBF, 0);
    int e = hs_v3d_open(&dev, NULL);
    if (e) die("hs_v3d_open", e);
    if ((e = hs_v3d_bo_new(&dev, &code, 256 << 10)) || (e = hs_v3d_bo_new(&dev, &unif, 4 << 20)) ||
        (e = hs_v3d_bo_new(&dev, &data, 2 << 20)))
        die("bo_new", e);
    handles[0] = code.handle;
    handles[1] = unif.handle;
    handles[2] = data.handle;
    printf("# tmuop_probe %s ops %s, resets so far %d\n", which, list, resets());
    for (int i = 0; i < nops; i++) {
        if (strchr(which, 'P')) section_p(ops[i]);
        if (strchr(which, 'R')) section_r(ops[i], rets[i], runs);
        if (strchr(which, 'W')) section_w(ops[i], rets[i], runs);
        if (strchr(which, 'M')) section_m(ops[i], runs);
    }
    return 0;
}
