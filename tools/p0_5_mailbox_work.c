/* p0_5_mailbox_work: P0-5 (docs/audit/HARDWARE_HANDOFF_P0.md). Does a
 * mailbox message that triggers real GPU work beat the CPU and a normal GPU
 * job? The same work is run three ways, every output word checked:
 *   cpu      one Cortex-A72 core, data in ordinary memory
 *   job      hs_qpu_kernel_gather through hs_v3d_dispatch + wait (gather
 *            only; 8 batches k=8 when N is a multiple of 1024, else 1 batch
 *            k=1), data already in the BO
 *   mailbox  a resident lease chain (tools/p0_5_mailbox_probe.c): the
 *            message carries idx, table, out and the pass count; data
 *            already in the BO
 * Work: gather (out[i] = table[idx[i]]; the GPU publishes each output line
 * with op 2) or reduce (per-lane sums of table[idx[i]]: a 16-word result).
 * idx: "fresh" = the CPU writes each message's indices into a ring region
 * the GPU has not read in this lease (plain loads); "op" = one region, op 2
 * before each idx load; "static" = written before the leases start (plain
 * loads, never rewritten). The table (512 KiB) is written before the leases.
 *
 * Usage: tools/p0_5_mailbox_work <gather|reduce> <fresh|op|static> [trials=200]
 *        [sizes...]   Output: one JSON line per size. Run inside the GPU lock.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <math.h>
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
static void tmuwt(hs_qpu_prog* p) { emit(p, hs_qpu_a(HS_QPU_A_TMUWT, X, X, X)); }
static void branch(hs_qpu_prog* p, int cond, int label) {
    hs_qpu_branch(p, cond, label, 0, 0);
    nop(p);
    nop(p);
    nop(p);
}
static void pushz(hs_qpu_prog* p, int op, hs_qpu_reg d, hs_qpu_reg a, hs_qpu_reg b) {
    hs_qpu_alu s = hs_qpu_a(op, d, a, b);
    s.af = HS_QPU_PF_PUSHZ;
    emit(p, s);
}
/* regular load, the measured thrsw form */
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
/* op 2 on each lane's word (config from the next uniform), then tmuwt */
static void op2(hs_qpu_prog* p, hs_qpu_reg addr) {
    mov(p, HS_QPU_TMUAU, addr);
    tmuwt(p);
}


#include "hs_qpu_kernels.h"

enum { MODE_GATHER, MODE_REDUCE };
enum { IDX_FRESH, IDX_OP, IDX_STATIC };
static int g_mode, g_idx;

/* uniforms: M, ASEQ, ASUM, BUDGET, S, P, PEND, CAP, then op-2 configs.
 * The budget counts op-2 lookups (polls and work), so the uniform stream
 * cannot run out: a message that needs more than is left ends the lease
 * unacked and the next lease runs it.
 * rf0 M+4l  rf1 ASEQ+4l  rf2 ASUM+4l  rf3 budget  rf4 expected seq
 * rf5 lane-0 mask  rf6 S+4l  rf7 P+4l  rf8 PEND+4l  rf10 messages left
 * rf11 slot+4l  rf12 sum  rf14 lease count  rf15 S+64+4l  rf16 S+128+4l
 * rf17 64  rf18 line address  rf22 idx+4l  rf23 table  rf24 out+4l
 * rf25 passes  rf26 256 */
static int prog_lease(hs_qpu_prog* p) {
    hs_qpu_prog_rules(p, 0);
    ldunifrf(p, 0);
    ldunifrf(p, 1);
    ldunifrf(p, 2);
    ldunifrf(p, 3);
    ldunifrf(p, 6);
    ldunifrf(p, 7);
    ldunifrf(p, 8);
    ldunifrf(p, 10);
    emit(p, hs_qpu_a(HS_QPU_A_EIDX, R(1), X, X));
    emit(p, hs_qpu_a(HS_QPU_A_SHL, R(0), R(1), I(2))); /* r0 = 4 * lane */
    static const int lo[] = {0, 1, 2, 6, 7, 8};
    for (int i = 0; i < 6; i++) add(p, RF(lo[i]), RF(lo[i]), R(0));
    emit(p, hs_qpu_a(HS_QPU_A_SUB, R(2), R(1), I(1)));
    emit(p, hs_qpu_a(HS_QPU_A_ASR, RF(5), R(2), I(15))); /* lane 0: ~0, others 0 */
    mov(p, R(2), I(1));
    emit(p, hs_qpu_a(HS_QPU_A_SHL, RF(17), R(2), I(6))); /* 64 */
    emit(p, hs_qpu_a(HS_QPU_A_SHL, RF(26), R(2), I(8))); /* 256 */
    add(p, RF(15), RF(6), RF(17));
    add(p, RF(16), RF(15), RF(17));
    load(p, RF(6), R(3));
    mov(p, RF(4), R(3));
    load(p, RF(15), R(3));
    add(p, RF(11), RF(7), R(3));
    load(p, RF(16), R(3));
    mov(p, RF(14), R(3));
    int poll = hs_qpu_label_new(p), done = hs_qpu_label_new(p), wl = hs_qpu_label_new(p);

    hs_qpu_label_bind(p, poll);
    op2(p, RF(0));
    load(p, RF(0), R(0));
    emit(p, hs_qpu_a(HS_QPU_A_SUB, RF(3), RF(3), I(1)));
    emit(p, hs_qpu_a(HS_QPU_A_BXOR, R(1), R(0), RF(4)));
    emit(p, hs_qpu_a(HS_QPU_A_BAND, R(1), R(1), RF(5)));
    pushz(p, HS_QPU_A_UMIN, R(2), R(1), RF(3));
    branch(p, HS_QPU_B_ANYNA, poll);
    emit(p, hs_qpu_a(HS_QPU_A_BXOR, R(1), R(0), RF(4)));
    pushz(p, HS_QPU_A_BAND, R(1), R(1), RF(5));
    branch(p, HS_QPU_B_ANYNA, done); /* the budget ran out */

    /* the message: four payload lines in a fresh 256-byte slot */
    mov(p, RF(18), RF(11));
    load(p, RF(18), R(0));
    mov(p, RF(22), R(0));
    add(p, RF(18), RF(18), RF(17));
    load(p, RF(18), R(0));
    mov(p, RF(23), R(0));
    add(p, RF(18), RF(18), RF(17));
    load(p, RF(18), R(0));
    mov(p, RF(24), R(0));
    add(p, RF(18), RF(18), RF(17));
    load(p, RF(18), R(0));
    mov(p, RF(25), R(0));
    /* op-2 lookups this message needs: 2 per pass (at most) + 2 acks + 2 */
    add(p, R(2), RF(25), RF(25));
    add(p, R(2), R(2), I(4));
    {
        hs_qpu_alu s = hs_qpu_a(HS_QPU_A_SUB, R(3), RF(3), R(2));
        s.af = HS_QPU_PF_PUSHN;
        emit(p, s);
    }
    branch(p, HS_QPU_B_ANYA, done); /* not enough left: the next lease runs it */
    mov(p, RF(3), R(3));
    mov(p, RF(12), I(0));

    hs_qpu_label_bind(p, wl);
    if (g_idx == IDX_OP) op2(p, RF(22));
    load(p, RF(22), R(0));
    emit(p, hs_qpu_a(HS_QPU_A_SHL, R(0), R(0), I(2)));
    add(p, R(0), R(0), RF(23));
    load(p, R(0), R(1));
    if (g_mode == MODE_REDUCE) {
        add(p, RF(12), RF(12), R(1));
    } else {
        store(p, RF(24), R(1));
        tmuwt(p);
        op2(p, RF(24));
        add(p, RF(24), RF(24), RF(17));
    }
    add(p, RF(22), RF(22), RF(17));
    pushz(p, HS_QPU_A_SUB, RF(25), RF(25), I(1));
    branch(p, HS_QPU_B_NA0, wl);

    store(p, RF(2), RF(12));
    store(p, RF(1), RF(4));
    tmuwt(p);
    op2(p, RF(2));
    op2(p, RF(1));
    add(p, RF(4), RF(4), I(1));
    add(p, RF(11), RF(11), RF(26));
    pushz(p, HS_QPU_A_SUB, R(1), RF(11), RF(8)); /* ring end: wrap */
    hs_qpu_alu w = hs_qpu_a(HS_QPU_A_MOV, RF(11), RF(7), X);
    w.ac = HS_QPU_COND_IFA;
    emit(p, w);
    emit(p, hs_qpu_a(HS_QPU_A_SUB, RF(10), RF(10), I(1)));
    pushz(p, HS_QPU_A_UMIN, R(2), RF(10), RF(3));
    branch(p, HS_QPU_B_NA0, poll);

    hs_qpu_label_bind(p, done);
    store(p, RF(6), RF(4));
    emit(p, hs_qpu_a(HS_QPU_A_SUB, R(1), RF(11), RF(7)));
    store(p, RF(15), R(1));
    add(p, RF(14), RF(14), I(1));
    store(p, RF(16), RF(14));
    tmuwt(p);
    hs_qpu_emit_end(p);
    return hs_qpu_prog_finish(p);
}
static int prog_warm(hs_qpu_prog* p) {
    ldunifrf(p, 0);
    int l = hs_qpu_label_new(p);
    hs_qpu_label_bind(p, l);
    pushz(p, HS_QPU_A_SUB, RF(0), RF(0), I(1));
    branch(p, HS_QPU_B_NA0, l);
    hs_qpu_emit_end(p);
    return hs_qpu_prog_finish(p);
}
static int g_kb, g_kk;
static int prog_gather(hs_qpu_prog* p) { return hs_qpu_kernel_gather(p, g_kb, g_kk); }

static hs_v3d dev;
static hs_v3d_bo code, unif, data;
static uint32_t handles[3];
static uint64_t progbuf[2048];
static size_t code_off;
static uint32_t load_prog(int (*build)(hs_qpu_prog*)) {
    hs_qpu_prog p;
    hs_qpu_prog_init(&p, progbuf, 2048);
    int n = build(&p);
    if (n < 0) {
        fprintf(stderr, "program build failed: %d at %u\n", n, p.rule_at);
        exit(1);
    }
    uint32_t a = hs_v3d_load(&code, code_off, progbuf, (uint32_t)n);
    code_off += ((size_t)n * 8 + 255) & ~(size_t)255;
    if (!a) die("load", ENOSPC);
    return a;
}
static void wbarrier(void) { __asm__ volatile("dsb sy" ::: "memory"); }
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
static uint32_t cfg_op2(void) { return 0xFFFFFF80u | 2u << 3 | 7u; }
static uint32_t g_tw = 131072u; /* table words: 512 KiB, or TABLE_MB */
#define TABLE_WORDS g_tw
static int g_seq;
#define MAXN 65536u
#define SLOTS 1024u
#define IDX_RING (16u << 20)
enum {
    OFF_M = 0, OFF_ASEQ = 4096, OFF_ASUM = 8192, OFF_S = 12288, OFF_WARMU = 16384, OFF_JOBU = 20480,
    OFF_P = 65536,                       /* SLOTS x 256 B */
    OFF_OUT = OFF_P + SLOTS * 256,
    OFF_IDX = OFF_OUT + MAXN * 4,        /* IDX_RING */
    OFF_TABLE = OFF_IDX + IDX_RING       /* g_tw words */
};
#define DATA_SIZE ((size_t)OFF_TABLE + (size_t)g_tw * 4)
/* fills dst (and cidx) with one message's indices: random, or with SEQ=1 a
 * sequential run from a random base */
#define FILL_IDX(dst) \
    do { \
        uint32_t base_ = rng() % (g_tw - N); \
        for (uint32_t i = 0; i < N; i++) (dst)[i] = cidx[i] = g_seq ? base_ + i : rng() % g_tw; \
    } while (0)
#define BUDGET 300000u
static volatile uint32_t* W32(size_t off) { return (volatile uint32_t*)(data.map + off); }
static uint32_t rng_s = 0x12345678u;
static uint32_t rng(void) {
    rng_s ^= rng_s << 13;
    rng_s ^= rng_s >> 17;
    rng_s ^= rng_s << 5;
    return rng_s;
}
static int cmp_d(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}
static double med(double* v, int n) {
    qsort(v, (size_t)n, sizeof(double), cmp_d);
    return n ? v[n / 2] : 0;
}
static double pct(double* v, int n, double q) { return n ? v[(int)((n - 1) * q)] : 0; } /* after med() sorted it */

static uint32_t *ctable, cidx[MAXN], cout_[MAXN];
__attribute__((noinline)) static void cpu_gather(const uint32_t* t, const uint32_t* ix, uint32_t* o, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) o[i] = t[ix[i]];
}
__attribute__((noinline)) static void cpu_reduce(const uint32_t* t, const uint32_t* ix, uint32_t* s, uint32_t n) {
    uint32_t a[16] = {0};
    for (uint32_t i = 0; i < n; i += 16)
        for (uint32_t l = 0; l < 16; l++) a[l] += t[ix[i + l]];
    memcpy(s, a, sizeof(a));
}

static uint32_t g_submitted;
static uint32_t g_lease_code;
static void top_up(void) {
    uint32_t ended = W32(OFF_S + 128)[0];
    while (g_submitted - ended < 2) {
        hs_v3d_dispatch_desc d = {g_lease_code, unif.gpu, 1, handles, 3, 0};
        int e = hs_v3d_dispatch(&dev, &d);
        if (e) die("dispatch lease", e);
        g_submitted++;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s gather|reduce fresh|op|static [trials] [sizes...]\n", argv[0]);
        return 2;
    }
    g_mode = !strcmp(argv[1], "reduce") ? MODE_REDUCE : MODE_GATHER;
    g_idx = !strcmp(argv[2], "op") ? IDX_OP : !strcmp(argv[2], "static") ? IDX_STATIC : IDX_FRESH;
    int trials = argc > 3 ? atoi(argv[3]) : 200;
    g_seq = getenv("SEQ") != NULL;
    if (getenv("TABLE_MB")) g_tw = (uint32_t)atoi(getenv("TABLE_MB")) << 18;
    if (g_tw < 131072u || g_tw > (64u << 18)) die("TABLE_MB: 1..64", EINVAL);
    ctable = malloc((size_t)g_tw * 4);
    if (!ctable) die("malloc", ENOMEM);
    if (trials < 3 || trials > 2000) die("trials", EINVAL);
    static const uint32_t defsz[] = {16, 64, 256, 1024, 4096, 16384, 65536};
    uint32_t sizes[16];
    int nsz = 0;
    for (int i = 4; i < argc && nsz < 16; i++) sizes[nsz++] = (uint32_t)atoi(argv[i]);
    if (!nsz)
        for (nsz = 0; nsz < 7; nsz++) sizes[nsz] = defsz[nsz];
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(2, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);
    setvbuf(stdout, NULL, _IOLBF, 0);
    int e = hs_v3d_open(&dev, NULL);
    if (e) die("hs_v3d_open", e);
    size_t nunif = 16 + (size_t)BUDGET + 64;
    if ((e = hs_v3d_bo_new(&dev, &code, 64 << 10)) || (e = hs_v3d_bo_new(&dev, &unif, nunif * 4)) ||
        (e = hs_v3d_bo_new(&dev, &data, DATA_SIZE)))
        die("bo_new", e);
    handles[0] = code.handle;
    handles[1] = unif.handle;
    handles[2] = data.handle;
    uint32_t pw = load_prog(prog_warm);
    g_lease_code = load_prog(prog_lease);
    g_kb = 8, g_kk = 8;
    uint32_t pg88 = load_prog(prog_gather);
    g_kb = 1, g_kk = 1;
    uint32_t pg11 = load_prog(prog_gather);
    memset((void*)data.map, 0, OFF_IDX);
    for (uint32_t i = 0; i < TABLE_WORDS; i++) W32(OFF_TABLE)[i] = ctable[i] = i * 0x9E3779B1u + 12345u;
    W32(OFF_WARMU)[0] = 37500;
    wbarrier();
    int r0 = resets();
    static double t_cpu[2000], t_job[2000], t_mb[2000], t_wr[2000];

    for (int si = 0; si < nsz; si++) {
        uint32_t N = sizes[si];
        if (N < 16 || N > MAXN || (N & 15)) die("size: a multiple of 16, up to 65536", EINVAL);
        uint32_t passes = N / 16, region = (N * 4 + 255) & ~255u;
        uint32_t regions = g_idx == IDX_FRESH ? IDX_RING / region : 1;
        if (2 * passes + 8 > BUDGET / 2) die("size too large for the lease budget", EINVAL);
        uint32_t bad_cpu = 0, bad_job = 0, bad_mb = 0, lost = 0;
        int have_job = g_mode == MODE_GATHER;
        /* ---- cpu ---- */
        for (int t = 0; t < trials; t++) {
            {
                uint32_t base_ = rng() % (g_tw - N);
                for (uint32_t i = 0; i < N; i++) cidx[i] = g_seq ? base_ + i : rng() % g_tw;
            }
            uint32_t s[16];
            uint64_t a = now_ns();
            if (g_mode == MODE_GATHER) cpu_gather(ctable, cidx, cout_, N);
            else cpu_reduce(ctable, cidx, s, N);
            t_cpu[t] = (double)(now_ns() - a);
            if (g_mode == MODE_GATHER) {
                for (uint32_t i = 0; i < N; i++) bad_cpu += cout_[i] != ctable[cidx[i]];
            } else {
                for (uint32_t l = 0; l < 16; l++) {
                    uint32_t x = 0;
                    for (uint32_t i = l; i < N; i += 16) x += ctable[cidx[i]];
                    bad_cpu += s[l] != x;
                }
            }
        }
        /* ---- job (no lease may be running: CSD jobs are serial) ---- */
        const char* jobcfg = "none";
        if (have_job) {
            int b = N % 1024 == 0 ? 8 : 1, k = N % 1024 == 0 ? 8 : 1;
            jobcfg = b == 8 ? "8 batches k=8" : "1 batch k=1";
            uint32_t* ju = (uint32_t*)(data.map + OFF_JOBU);
            if (hs_qpu_gather_uniforms(ju, N, b, k, data.gpu + OFF_IDX, data.gpu + OFF_TABLE, data.gpu + OFF_OUT))
                die("gather uniforms", EINVAL);
            hs_v3d_dispatch_desc warm = {pw, data.gpu + OFF_WARMU, 1, handles, 3, 0};
            if ((e = hs_v3d_dispatch(&dev, &warm))) die("warm", e);
            hs_v3d_bo_wait(&dev, &data, 2000000000ull);
            for (int t = 0; t < trials; t++) {
                FILL_IDX(W32(OFF_IDX));
                for (uint32_t i = 0; i < N; i++) W32(OFF_OUT)[i] = 0xDEADBEEFu;
                wbarrier();
                hs_v3d_dispatch_desc d = {b == 8 ? pg88 : pg11, data.gpu + OFF_JOBU, (uint32_t)b, handles, 3, 0};
                uint64_t a = now_ns();
                if ((e = hs_v3d_dispatch(&dev, &d))) die("dispatch job", e);
                if ((e = hs_v3d_bo_wait(&dev, &data, 2000000000ull))) die("wait job", e);
                t_job[t] = (double)(now_ns() - a);
                for (uint32_t i = 0; i < N; i++) bad_job += W32(OFF_OUT)[i] != ctable[cidx[i]];
            }
        }
        /* ---- mailbox ---- */
        {
            /* fresh idx regions must not repeat within a lease; the others may */
            uint32_t cap = g_idx == IDX_FRESH && regions < SLOTS ? regions : SLOTS;
            uint32_t* u = (uint32_t*)unif.map;
            u[0] = data.gpu + OFF_M;
            u[1] = data.gpu + OFF_ASEQ;
            u[2] = data.gpu + OFF_ASUM;
            u[3] = BUDGET;
            u[4] = data.gpu + OFF_S;
            u[5] = data.gpu + OFF_P;
            u[6] = data.gpu + OFF_P + SLOTS * 256;
            u[7] = cap;
            for (size_t i = 8; i < nunif; i++) u[i] = cfg_op2();
            for (int l = 0; l < 16; l++) {
                W32(OFF_M)[l] = W32(OFF_ASEQ)[l] = 0;
                W32(OFF_S)[l] = 1;
                W32(OFF_S + 64)[l] = 0;
                W32(OFF_S + 128)[l] = 0;
            }
            if (g_idx == IDX_STATIC) FILL_IDX(W32(OFF_IDX));
            wbarrier();
            hs_v3d_dispatch_desc warm = {pw, data.gpu + OFF_WARMU, 1, handles, 3, 0};
            if ((e = hs_v3d_dispatch(&dev, &warm))) die("warm", e);
            hs_v3d_bo_wait(&dev, &data, 2000000000ull);
            g_submitted = 0;
            top_up();
            for (int t = 0; t < trials && !lost; t++) {
                uint32_t n = (uint32_t)t + 1;
                size_t ioff = OFF_IDX + (size_t)((n - 1) % regions) * region;
                uint64_t w0 = now_ns();
                if (g_idx != IDX_STATIC) FILL_IDX(W32(ioff));
                size_t slot = OFF_P + (size_t)((n - 1) % SLOTS) * 256;
                for (uint32_t l = 0; l < 16; l++) {
                    W32(slot)[l] = data.gpu + (uint32_t)ioff + 4 * l;
                    W32(slot + 64)[l] = data.gpu + OFF_TABLE;
                    W32(slot + 128)[l] = data.gpu + OFF_OUT + 4 * l;
                    W32(slot + 192)[l] = passes;
                }
                wbarrier();
                top_up();
                uint64_t a = now_ns();
                t_wr[t] = (double)(a - w0);
                W32(OFF_M)[0] = n;
                wbarrier();
                uint64_t b, last_top = a;
                int ok;
                do {
                    ok = W32(OFF_ASEQ)[0] == n;
                    b = now_ns();
                    if (!ok && b - last_top > 1000000) {
                        top_up();
                        last_top = b;
                    }
                } while (!ok && b - a < 500000000ull);
                if (!ok) {
                    lost++;
                    break;
                }
                t_mb[t] = (double)(b - a);
                if (g_mode == MODE_GATHER) {
                    for (uint32_t i = 0; i < N; i++) bad_mb += W32(OFF_OUT)[i] != ctable[cidx[i]];
                } else {
                    for (uint32_t l = 0; l < 16; l++) {
                        uint32_t x = 0;
                        for (uint32_t i = l; i < N; i += 16) x += ctable[cidx[i]];
                        bad_mb += W32(OFF_ASUM)[l] != x;
                    }
                }
            }
            hs_v3d_bo_wait(&dev, &data, 5000000000ull); /* queued leases run out their budget */
        }
        int nm = lost ? 0 : trials;
        double c50 = med(t_cpu, trials), c99 = pct(t_cpu, trials, 0.99);
        double j50 = have_job ? med(t_job, trials) : 0, j99 = have_job ? pct(t_job, trials, 0.99) : 0;
        double m50 = med(t_mb, nm), m99 = pct(t_mb, nm, 0.99), w50 = med(t_wr, nm);
        printf("{\"work\":\"%s\",\"idx\":\"%s\",\"seq\":%d,\"table_kib\":%u,\"n\":%u,\"trials\":%d,\"cpu_us\":{\"p50\":%.2f,\"p99\":%.2f,\"bad\":%u},"
               "\"job_us\":{\"cfg\":\"%s\",\"p50\":%.2f,\"p99\":%.2f,\"bad\":%u},"
               "\"mailbox_us\":{\"p50\":%.2f,\"p99\":%.2f,\"bad\":%u,\"lost\":%u,\"msg_write_p50\":%.2f},"
               "\"leases\":%u,\"resets\":%d}\n",
               argv[1], argv[2], g_seq, g_tw / 256, N, trials, c50 / 1e3, c99 / 1e3, bad_cpu, jobcfg, j50 / 1e3, j99 / 1e3, bad_job,
               m50 / 1e3, m99 / 1e3, bad_mb, lost, w50 / 1e3, W32(OFF_S + 128)[0], resets() - r0);
    }
    return 0;
}
