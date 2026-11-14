/* p0_5_mailbox_probe: P0-5 (docs/audit/HARDWARE_HANDOFF_P0.md) feasibility
 * measurements for a persistent QPU mailbox, on NeoGPU's own encoder and
 * runtime. A resident program (a "lease": one CSD job that ends after BUDGET
 * polls or CAP messages, well under the ~1 s job reset) polls a sequence word
 * with TMU op 2, reads the message payload from a ring with PLAIN loads (or op
 * 2 + load with OPLOAD=1), sums each lane's W payload words, stores the sums
 * and the sequence, and publishes both with op 2. At lease end it stores its
 * state (next sequence, ring offset, lease count); the next queued lease
 * starts from it. The CPU keeps QDEPTH leases queued.
 *
 * Questions:
 *   fresh lines:  is a ring line the GPU has not read in this lease seen
 *                 correctly with a plain load (no cache op)? CAP <= slots
 *                 guarantees that; CAP > slots (control) forces reuse.
 *   handoff:      what a message costs when it crosses a lease boundary.
 *   coexistence:  run this alongside a Mesa renderer (see the P0-5 scripts).
 *
 * Usage: tools/p0_5_mailbox_probe <seconds> [msgs=0 (idle leases)] [W=1]
 *        [slots=4096] [cap=slots] [budget=300000] [gap_us=0] [qdepth=2]
 * Env: OPLOAD=1 (op 2 before each payload load); POLL_SPIN=n (n wait
 * iterations between polls), POLL_THRSW=1 (a thread switch in each). Output:
 * one JSON line.
 * Run inside the GPU lock: flock build/gpu.lock tools/p0_5_mailbox_probe ...
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

static int g_opload, g_spin, g_thrsw;
/* uniforms: M, ASEQ, ASUM, BUDGET, S, P, PEND, CAP, W, SPIN, then op-2 configs.
 * rf0 M+4l  rf1 ASEQ+4l  rf2 ASUM+4l  rf3 budget  rf4 expected seq
 * rf5 lane-0 mask  rf6 S+4l  rf7 P+4l  rf8 PEND+4l  rf9 W  rf10 messages left
 * rf11 slot+4l  rf12 sum  rf13 line counter  rf14 lease count  rf15 S+64+4l
 * rf16 S+128+4l  rf17 64  rf18 line address */
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
    ldunifrf(p, 9);
    ldunifrf(p, 20); /* spin iterations between polls */
    mov(p, RF(21), I(1));
    emit(p, hs_qpu_a(HS_QPU_A_EIDX, R(1), X, X));
    emit(p, hs_qpu_a(HS_QPU_A_SHL, R(0), R(1), I(2))); /* r0 = 4 * lane */
    static const int lo[] = {0, 1, 2, 6, 7, 8};
    for (int i = 0; i < 6; i++) add(p, RF(lo[i]), RF(lo[i]), R(0));
    emit(p, hs_qpu_a(HS_QPU_A_SUB, R(2), R(1), I(1)));
    emit(p, hs_qpu_a(HS_QPU_A_ASR, RF(5), R(2), I(15))); /* lane 0: ~0, others 0 */
    mov(p, R(2), I(1));
    emit(p, hs_qpu_a(HS_QPU_A_SHL, RF(17), R(2), I(6))); /* 64 */
    add(p, RF(15), RF(6), RF(17));
    add(p, RF(16), RF(15), RF(17));
    /* state, written by the previous lease (or the CPU before the first) */
    load(p, RF(6), R(3));
    mov(p, RF(4), R(3));
    load(p, RF(15), R(3));
    add(p, RF(11), RF(7), R(3));
    load(p, RF(16), R(3));
    mov(p, RF(14), R(3));
    int poll = hs_qpu_label_new(p), done = hs_qpu_label_new(p), quit = hs_qpu_label_new(p);
    int wl = hs_qpu_label_new(p);
    pushz(p, HS_QPU_A_MOV, R(1), RF(4), X); /* expected seq 0: stopped */
    branch(p, HS_QPU_B_A0, quit);

    hs_qpu_label_bind(p, poll);
    op2(p, RF(0));
    load(p, RF(0), R(0));
    emit(p, hs_qpu_a(HS_QPU_A_SUB, RF(3), RF(3), I(1)));
    emit(p, hs_qpu_a(HS_QPU_A_BXOR, R(1), R(0), RF(4)));
    emit(p, hs_qpu_a(HS_QPU_A_BAND, R(1), R(1), RF(5)));
    pushz(p, HS_QPU_A_UMIN, R(2), R(1), RF(3));
    if (g_spin) {
        /* not seen yet: wait g_spin iterations before the next poll (thrsw in
         * each iteration with POLL_THRSW=1, so other threads on the QPU run) */
        int seen = hs_qpu_label_new(p), sp = hs_qpu_label_new(p);
        branch(p, HS_QPU_B_ALLA, seen);
        emit(p, hs_qpu_a(HS_QPU_A_MOV, RF(19), RF(20), X));
        hs_qpu_label_bind(p, sp);
        hs_qpu_alu d = hs_qpu_a(HS_QPU_A_SUB, RF(19), RF(19), RF(21)); /* rf21 = 1: thrsw has no small-immediate form */
        d.af = HS_QPU_PF_PUSHZ;
        if (g_thrsw) d = hs_qpu_s(d, HS_QPU_SIG_THRSW, X);
        emit(p, d);
        if (g_thrsw) {
            nop(p);
            nop(p);
        }
        branch(p, HS_QPU_B_NA0, sp);
        branch(p, HS_QPU_B_ALWAYS, poll);
        hs_qpu_label_bind(p, seen);
    } else {
        branch(p, HS_QPU_B_ANYNA, poll);
    }
    /* out of the poll loop: the message, or the budget ran out */
    emit(p, hs_qpu_a(HS_QPU_A_BXOR, R(1), R(0), RF(4)));
    pushz(p, HS_QPU_A_BAND, R(1), R(1), RF(5));
    branch(p, HS_QPU_B_ANYNA, done);

    mov(p, RF(12), I(0));
    mov(p, RF(13), RF(9));
    mov(p, RF(18), RF(11));
    hs_qpu_label_bind(p, wl);
    if (g_opload) op2(p, RF(18));
    load(p, RF(18), R(0));
    add(p, RF(12), RF(12), R(0));
    add(p, RF(18), RF(18), RF(17));
    pushz(p, HS_QPU_A_SUB, RF(13), RF(13), I(1));
    branch(p, HS_QPU_B_NA0, wl);
    store(p, RF(2), RF(12));
    store(p, RF(1), RF(4));
    tmuwt(p);
    op2(p, RF(2));
    op2(p, RF(1));
    add(p, RF(4), RF(4), I(1));
    mov(p, RF(11), RF(18));
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
    hs_qpu_label_bind(p, quit);
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

static hs_v3d dev;
static hs_v3d_bo code, unif, data;
static uint32_t handles[3];
static uint64_t progbuf[1024];
static size_t code_off;
static uint32_t load_prog(int (*build)(hs_qpu_prog*)) {
    hs_qpu_prog p;
    hs_qpu_prog_init(&p, progbuf, 1024);
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
static int dispatch(uint32_t code_gpu, uint32_t unif_off) {
    hs_v3d_dispatch_desc d = {code_gpu, unif.gpu + unif_off, 1, handles, 3, 0};
    return hs_v3d_dispatch(&dev, &d);
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
enum { OFF_M = 0, OFF_ASEQ = 4096, OFF_ASUM = 8192, OFF_S = 12288, OFF_WARMU = 16384, OFF_P = 65536 };
static volatile uint32_t* W32(size_t off) { return (volatile uint32_t*)(data.map + off); }
static uint32_t val(uint32_t n, uint32_t j, uint32_t l) { return n * 0x9E3779B1u ^ j * 0x85EBCA77u ^ l * 0x27D4EB2Fu; }
static int cmp_u64(const void* a, const void* b) {
    uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
    return x < y ? -1 : x > y;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s seconds [msgs] [W] [slots] [cap] [budget] [gap_us] [qdepth]\n", argv[0]);
        return 2;
    }
    double secs = atof(argv[1]);
    uint32_t msgs = argc > 2 ? (uint32_t)atoi(argv[2]) : 0;
    uint32_t W = argc > 3 ? (uint32_t)atoi(argv[3]) : 1;
    uint32_t slots = argc > 4 ? (uint32_t)atoi(argv[4]) : 4096;
    uint32_t cap = argc > 5 ? (uint32_t)atoi(argv[5]) : slots;
    uint32_t budget = argc > 6 ? (uint32_t)atoi(argv[6]) : 300000;
    uint32_t gap_us = argc > 7 ? (uint32_t)atoi(argv[7]) : 0;
    uint32_t qdepth = argc > 8 ? (uint32_t)atoi(argv[8]) : 2;
    g_opload = getenv("OPLOAD") != NULL;
    g_spin = getenv("POLL_SPIN") ? atoi(getenv("POLL_SPIN")) : 0;
    g_thrsw = getenv("POLL_THRSW") != NULL;
    if (!W || W > 64 || !slots || !cap || budget < 2 || !qdepth || qdepth > 8 || msgs > 4000000) die("arguments", EINVAL);
    size_t ring = (size_t)slots * W * 64;
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(2, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);
    int e = hs_v3d_open(&dev, NULL);
    if (e) die("hs_v3d_open", e);
    size_t nunif = 16 + (size_t)budget + (size_t)cap * (2 + (g_opload ? W : 0)) + 64;
    if ((e = hs_v3d_bo_new(&dev, &code, 64 << 10)) || (e = hs_v3d_bo_new(&dev, &unif, nunif * 4)) ||
        (e = hs_v3d_bo_new(&dev, &data, OFF_P + ring)))
        die("bo_new", e);
    handles[0] = code.handle;
    handles[1] = unif.handle;
    handles[2] = data.handle;
    uint32_t pw = load_prog(prog_warm), pl = load_prog(prog_lease);
    uint32_t* u = (uint32_t*)unif.map;
    u[0] = data.gpu + OFF_M;
    u[1] = data.gpu + OFF_ASEQ;
    u[2] = data.gpu + OFF_ASUM;
    u[3] = budget;
    u[4] = data.gpu + OFF_S;
    u[5] = data.gpu + OFF_P;
    u[6] = data.gpu + OFF_P + (uint32_t)ring;
    u[7] = cap;
    u[8] = W;
    u[9] = g_spin ? g_spin : 1;
    for (size_t i = 10; i < nunif; i++) u[i] = cfg_op2();
    memset((void*)data.map, 0, OFF_P + ring);
    for (int l = 0; l < 16; l++) W32(OFF_S)[l] = 1; /* next seq 1, offset 0, leases 0 */
    W32(OFF_WARMU)[0] = 37500;                      /* ~1 ms */
    wbarrier();
    int r0 = resets();
    /* warm: the uniform stream must be in a tracked BO; the data BO is */
    hs_v3d_dispatch_desc warm = {pw, data.gpu + OFF_WARMU, 1, handles, 3, 0};
    if ((e = hs_v3d_dispatch(&dev, &warm))) die("warm", e);
    hs_v3d_bo_wait(&dev, &data, 2000000000ull);

    static uint64_t rtt[4000000];
    static uint8_t crossed[4000000];
    uint32_t submitted = 0, sent = 0, acked = 0, lost = 0, wrong = 0, wrong_lanes = 0;
    volatile uint32_t* leases = W32(OFF_S + 128);
    uint64_t t0 = now_ns(), tend = t0 + (uint64_t)(secs * 1e9), t_first = 0, t_last = 0;
    uint32_t max_inflight = 0;
    for (;;) {
        uint32_t ended = leases[0];
        while (submitted - ended < qdepth) {
            if ((e = dispatch(pl, 0))) die("dispatch lease", e);
            submitted++;
        }
        if (submitted - ended > max_inflight) max_inflight = submitted - ended;
        uint64_t now = now_ns();
        if (now >= tend) break;
        if (sent >= msgs) {
            usleep(1000);
            continue;
        }
        uint32_t n = sent + 1;
        size_t slot = OFF_P + (size_t)((n - 1) % slots) * W * 64;
        for (uint32_t j = 0; j < W; j++)
            for (uint32_t l = 0; l < 16; l++) W32(slot + j * 64)[l] = val(n, j, l);
        wbarrier();
        uint32_t lease_before = leases[0];
        uint64_t a = now_ns();
        W32(OFF_M)[0] = n;
        wbarrier();
        sent = n;
        uint64_t b;
        int ok;
        do {
            ok = W32(OFF_ASEQ)[0] == n;
            b = now_ns();
            if (!ok && (b - a) % 1000000 < 2000) { /* keep leases queued while waiting */
                uint32_t en = leases[0];
                while (submitted - en < qdepth) {
                    if ((e = dispatch(pl, 0))) die("dispatch lease", e);
                    submitted++;
                }
            }
        } while (!ok && b - a < 200000000ull);
        if (!ok) {
            lost++;
            break; /* the sequence is now out of step: stop */
        }
        rtt[acked] = b - a;
        crossed[acked] = leases[0] != lease_before;
        acked++;
        if (!t_first) t_first = b;
        t_last = b;
        uint32_t exp_bad = 0;
        for (uint32_t l = 0; l < 16; l++) {
            uint32_t s = 0;
            for (uint32_t j = 0; j < W; j++) s += val(n, j, l);
            exp_bad += W32(OFF_ASUM)[l] != s;
        }
        if (exp_bad) {
            wrong++;
            wrong_lanes += exp_bad;
        }
        if (gap_us) {
            uint64_t g = now_ns() + gap_us * 1000ull;
            while (now_ns() < g) {
            }
        }
    }
    /* stop: queued leases run out their budget (no stop message in this probe) */
    uint64_t ts = now_ns();
    hs_v3d_bo_wait(&dev, &data, 5000000000ull);
    double drain_ms = (now_ns() - ts) / 1e6;
    int r1 = resets();
    uint32_t leases_done = leases[0];
    /* statistics */
    uint64_t* s = malloc(sizeof(uint64_t) * (acked ? acked : 1));
    uint32_t n_cross = 0, over20 = 0, over20_cross = 0;
    uint64_t max_cross = 0, max_plain = 0;
    for (uint32_t i = 0; i < acked; i++) {
        s[i] = rtt[i];
        if (crossed[i]) {
            n_cross++;
            if (rtt[i] > max_cross) max_cross = rtt[i];
        } else if (rtt[i] > max_plain)
            max_plain = rtt[i];
        if (rtt[i] > 20000) {
            over20++;
            over20_cross += crossed[i];
        }
    }
    qsort(s, acked ? acked : 1, sizeof(uint64_t), cmp_u64);
    double p50 = acked ? s[acked / 2] / 1e3 : 0, p99 = acked ? s[(uint64_t)acked * 99 / 100] / 1e3 : 0,
           p999 = acked ? s[(uint64_t)acked * 999 / 1000] / 1e3 : 0, mx = acked ? s[acked - 1] / 1e3 : 0;
    double rate = acked > 1 && t_last > t_first ? (acked - 1) / ((t_last - t_first) / 1e9) : 0;
    printf("{\"secs\":%.2f,\"msgs\":%u,\"W\":%u,\"slots\":%u,\"cap\":%u,\"budget\":%u,\"gap_us\":%u,\"qdepth\":%u,"
           "\"opload\":%d,\"spin\":%d,\"thrsw\":%d,\"sent\":%u,\"acked\":%u,\"lost\":%u,\"wrong\":%u,\"wrong_lanes\":%u,"
           "\"rtt_us\":{\"p50\":%.2f,\"p99\":%.2f,\"p999\":%.2f,\"max\":%.2f},\"rate_per_s\":%.0f,"
           "\"leases_submitted\":%u,\"leases_done\":%u,\"max_inflight\":%u,\"crossed\":%u,\"max_cross_us\":%.1f,"
           "\"max_plain_us\":%.1f,\"over20us\":%u,\"over20us_crossed\":%u,\"drain_ms\":%.1f,\"resets\":%d}\n",
           secs, msgs, W, slots, cap, budget, gap_us, qdepth, g_opload, g_spin, g_thrsw, sent, acked, lost, wrong, wrong_lanes, p50,
           p99, p999, mx, rate, submitted, leases_done, max_inflight, n_cross, max_cross / 1e3, max_plain / 1e3,
           over20, over20_cross, drain_ms, r0 < 0 || r1 < 0 ? -1 : r1 - r0);
    return lost || wrong ? 1 : 0;
}
