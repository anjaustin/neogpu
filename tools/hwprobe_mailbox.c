/* hwprobe_mailbox: what a persistent QPU program talking to the CPU through
 * shared memory can rely on, measured. Built only on NeoGPU's own stack
 * (hs_qpu encoder with the hardware rules on, hs_v3d runtime).
 *
 *   A  tick:  a running job stores a counter every ~1 ms. When does the CPU
 *             see each value: as it is written, or only when the job ends?
 *   B  poll:  a job polls one word until the CPU writes it. Does the GPU see
 *             the CPU's write (or a stale cached copy), and how fast?
 *   C  life:  how long can one job run before the kernel resets the GPU?
 *
 * Usage: tools/hwprobe_mailbox [sections=ABE] [cpu=2]
 *   A tick, B poll, E eviction sweeps: no GPU resets.
 *   C job lifetime: ends with one deliberate reset (the over-long job).
 *   O TMU cache ops via tmuc: deliberately hangs the GPU for most ops
 *     (evidence only; 6 resets per op).
 * Run inside the GPU lock: flock build/gpu.lock tools/hwprobe_mailbox
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
static void dec_pushz(hs_qpu_prog* p, int rf) {
    hs_qpu_alu s = hs_qpu_a(HS_QPU_A_SUB, RF(rf), RF(rf), I(1));
    s.af = HS_QPU_PF_PUSHZ;
    emit(p, s);
}
static void loop_back(hs_qpu_prog* p, int label) {
    hs_qpu_branch(p, HS_QPU_B_NA0, label, 0, 0);
    nop(p);
    nop(p);
    nop(p);
}

/* tick: uniforms addr, ticks, spin. Every `spin` iterations (~27 ns each:
 * 5 instructions x 4 cycles at 750 MHz) store the tick number. */
static int prog_tick(hs_qpu_prog* p) {
    ldunifrf(p, 0);
    ldunifrf(p, 1);
    ldunifrf(p, 2);
    emit(p, hs_qpu_a(HS_QPU_A_MOV, RF(4), I(0), X));
    int outer = hs_qpu_label_new(p), inner = hs_qpu_label_new(p);
    hs_qpu_label_bind(p, outer);
    emit(p, hs_qpu_a(HS_QPU_A_MOV, RF(3), RF(2), X));
    hs_qpu_label_bind(p, inner);
    dec_pushz(p, 3);
    loop_back(p, inner);
    emit(p, hs_qpu_a(HS_QPU_A_ADD, RF(4), RF(4), I(1)));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUD, RF(4), X));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(0), X));
    emit(p, hs_qpu_a(HS_QPU_A_TMUWT, X, X, X));
    dec_pushz(p, 1);
    loop_back(p, outer);
    hs_qpu_emit_end(p);
    return hs_qpu_prog_finish(p);
}

/* poll: uniforms mbox, ack, max_polls. Load mbox until it reads 1 or the
 * polls run out; then ack[0] = last value read, ack[1] = polls left. */
static int prog_poll(hs_qpu_prog* p) {
    ldunifrf(p, 0);
    ldunifrf(p, 1);
    ldunifrf(p, 2);
    int l = hs_qpu_label_new(p);
    hs_qpu_label_bind(p, l);
    emit(p, hs_qpu_s(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(0), X), HS_QPU_SIG_THRSW, X));
    nop(p);
    nop(p);
    emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDTMU, R(0)));
    emit(p, hs_qpu_a(HS_QPU_A_BXOR, R(1), R(0), I(1))); /* 0 once the CPU wrote 1 */
    emit(p, hs_qpu_a(HS_QPU_A_SUB, RF(2), RF(2), I(1)));
    hs_qpu_alu m = hs_qpu_a(HS_QPU_A_UMIN, R(2), R(1), RF(2)); /* 0 when seen or out of polls */
    m.af = HS_QPU_PF_PUSHZ;
    emit(p, m);
    loop_back(p, l);
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUD, R(0), X));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(1), X));
    emit(p, hs_qpu_a(HS_QPU_A_ADD, RF(3), RF(1), I(4)));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUD, RF(2), X));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(3), X));
    emit(p, hs_qpu_a(HS_QPU_A_TMUWT, X, X, X));
    hs_qpu_emit_end(p);
    return hs_qpu_prog_finish(p);
}

/* TMU cache-op variants. Hypothesis under test (recalled from Mesa's V3D 4.1
 * headers, not verified): bits 7:4 of a lookup's config byte select a TMU
 * operation (15 = regular; for reads 0 prefetch, 1 clear, 2 and 3 flush,
 * 4 full L1 clear). 0xFF, the default, is op 15 with type 32-bit. The config
 * goes through the tmuc register before each lookup. The rule model does not
 * cover tmuc yet, so these programs run with the rules off; their loads keep
 * the measured thrsw form. */
static int prog_poll_cfg(hs_qpu_prog* p) {
    hs_qpu_prog_rules(p, 0);
    ldunifrf(p, 0);
    ldunifrf(p, 1);
    ldunifrf(p, 2);
    ldunifrf(p, 4); /* tmuc value */
    int l = hs_qpu_label_new(p);
    hs_qpu_label_bind(p, l);
    /* round 2: the cache op returns nothing (round 1: its ldtmu never
     * returned), so issue it without ldtmu, then a regular read */
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUC, RF(4), X));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(0), X));
    emit(p, hs_qpu_a(HS_QPU_A_TMUWT, X, X, X));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUC, I(-1), X));
    emit(p, hs_qpu_s(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(0), X), HS_QPU_SIG_THRSW, X));
    nop(p);
    nop(p);
    emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDTMU, R(0)));
    emit(p, hs_qpu_a(HS_QPU_A_BXOR, R(1), R(0), I(1)));
    emit(p, hs_qpu_a(HS_QPU_A_SUB, RF(2), RF(2), I(1)));
    hs_qpu_alu m = hs_qpu_a(HS_QPU_A_UMIN, R(2), R(1), RF(2));
    m.af = HS_QPU_PF_PUSHZ;
    emit(p, m);
    loop_back(p, l);
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUD, R(0), X));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(1), X));
    emit(p, hs_qpu_a(HS_QPU_A_ADD, RF(3), RF(1), I(4)));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUD, RF(2), X));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(3), X));
    emit(p, hs_qpu_a(HS_QPU_A_TMUWT, X, X, X));
    hs_qpu_emit_end(p);
    return hs_qpu_prog_finish(p);
}
/* tick, then after each store's tmuwt a read of the same address with the
 * tmuc op under test (uniform 3) */
static int prog_tick_cfg(hs_qpu_prog* p) {
    hs_qpu_prog_rules(p, 0);
    ldunifrf(p, 0);
    ldunifrf(p, 1);
    ldunifrf(p, 2);
    ldunifrf(p, 5); /* tmuc value */
    emit(p, hs_qpu_a(HS_QPU_A_MOV, RF(4), I(0), X));
    int outer = hs_qpu_label_new(p), inner = hs_qpu_label_new(p);
    hs_qpu_label_bind(p, outer);
    emit(p, hs_qpu_a(HS_QPU_A_MOV, RF(3), RF(2), X));
    hs_qpu_label_bind(p, inner);
    dec_pushz(p, 3);
    loop_back(p, inner);
    emit(p, hs_qpu_a(HS_QPU_A_ADD, RF(4), RF(4), I(1)));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUD, RF(4), X));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(0), X));
    emit(p, hs_qpu_a(HS_QPU_A_TMUWT, X, X, X));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUC, RF(5), X));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(0), X)); /* cache op, no ldtmu (round 2) */
    emit(p, hs_qpu_a(HS_QPU_A_TMUWT, X, X, X));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUC, I(-1), X));
    dec_pushz(p, 1);
    loop_back(p, outer);
    hs_qpu_emit_end(p);
    return hs_qpu_prog_finish(p);
}

/* eviction: read `lines` x 64 B (one 16-lane lookup each) starting at rf8,
 * one load in flight (the measured thrsw form). Uses rf5..rf7, r3. */
static void evict(hs_qpu_prog* p) {
    emit(p, hs_qpu_a(HS_QPU_A_MOV, RF(5), RF(9), X)); /* line count */
    emit(p, hs_qpu_a(HS_QPU_A_MOV, RF(6), RF(8), X)); /* lane address */
    int l = hs_qpu_label_new(p);
    hs_qpu_label_bind(p, l);
    emit(p, hs_qpu_s(hs_qpu_m(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(6), X), HS_QPU_M_ADD, RF(6), RF(6), RF(7)),
                     HS_QPU_SIG_THRSW, X));
    nop(p);
    nop(p);
    emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDTMU, R(3)));
    dec_pushz(p, 5);
    loop_back(p, l);
}
/* uniforms 3, 4: eviction base, lines; sets rf7 = 64, rf8 = base + 4 * lane, rf9 = lines */
static void evict_setup(hs_qpu_prog* p) {
    ldunifrf(p, 8);
    ldunifrf(p, 9);
    emit(p, hs_qpu_a(HS_QPU_A_EIDX, R(0), X, X));
    emit(p, hs_qpu_a(HS_QPU_A_SHL, R(0), R(0), I(2)));
    emit(p, hs_qpu_a(HS_QPU_A_ADD, RF(8), RF(8), R(0)));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, R(0), I(1), X));
    emit(p, hs_qpu_a(HS_QPU_A_SHL, RF(7), R(0), I(6)));
}
/* poll with an eviction sweep before each poll load */
static int prog_poll_evict(hs_qpu_prog* p) {
    ldunifrf(p, 0);
    ldunifrf(p, 1);
    ldunifrf(p, 2);
    evict_setup(p);
    int l = hs_qpu_label_new(p);
    hs_qpu_label_bind(p, l);
    evict(p);
    emit(p, hs_qpu_s(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(0), X), HS_QPU_SIG_THRSW, X));
    nop(p);
    nop(p);
    emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDTMU, R(0)));
    emit(p, hs_qpu_a(HS_QPU_A_BXOR, R(1), R(0), I(1)));
    emit(p, hs_qpu_a(HS_QPU_A_SUB, RF(2), RF(2), I(1)));
    hs_qpu_alu m = hs_qpu_a(HS_QPU_A_UMIN, R(2), R(1), RF(2));
    m.af = HS_QPU_PF_PUSHZ;
    emit(p, m);
    loop_back(p, l);
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUD, R(0), X));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(1), X));
    emit(p, hs_qpu_a(HS_QPU_A_ADD, RF(3), RF(1), I(4)));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUD, RF(2), X));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(3), X));
    emit(p, hs_qpu_a(HS_QPU_A_TMUWT, X, X, X));
    hs_qpu_emit_end(p);
    return hs_qpu_prog_finish(p);
}
/* tick with an eviction sweep after each store (uniforms: addr, ticks, spin, base, lines) */
static int prog_tick_evict(hs_qpu_prog* p) {
    ldunifrf(p, 0);
    ldunifrf(p, 1);
    ldunifrf(p, 2);
    evict_setup(p);
    emit(p, hs_qpu_a(HS_QPU_A_MOV, RF(4), I(0), X));
    int outer = hs_qpu_label_new(p), inner = hs_qpu_label_new(p);
    hs_qpu_label_bind(p, outer);
    emit(p, hs_qpu_a(HS_QPU_A_MOV, RF(3), RF(2), X));
    hs_qpu_label_bind(p, inner);
    dec_pushz(p, 3);
    loop_back(p, inner);
    emit(p, hs_qpu_a(HS_QPU_A_ADD, RF(4), RF(4), I(1)));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUD, RF(4), X));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(0), X));
    emit(p, hs_qpu_a(HS_QPU_A_TMUWT, X, X, X));
    evict(p);
    dec_pushz(p, 1);
    loop_back(p, outer);
    hs_qpu_emit_end(p);
    return hs_qpu_prog_finish(p);
}

static hs_v3d dev;
static hs_v3d_bo code, unif, data;
static uint32_t handles[3];
static uint64_t prog[512];

static uint32_t load_prog(int (*build)(hs_qpu_prog*), size_t off) {
    hs_qpu_prog p;
    hs_qpu_prog_init(&p, prog, 512);
    int n = build(&p);
    if (n < 0) {
        fprintf(stderr, "program build failed: %d at %u\n", n, p.rule_at);
        exit(1);
    }
    return hs_v3d_load(&code, off, prog, (uint32_t)n);
}
static int dispatch(uint32_t code_gpu, uint32_t unif_off) {
    hs_v3d_dispatch_desc d = {code_gpu, unif.gpu + unif_off, 1, handles, 3, 0};
    return hs_v3d_dispatch(&dev, &d);
}
static void wbarrier(void) { __asm__ volatile("dsb sy" ::: "memory"); }

/* GPU resets so far, from dmesg; -1 if it cannot be read (no sudo) */
static int resets(void) {
    FILE* f = popen("sudo -n dmesg 2>/dev/null | grep -c 'Resetting GPU for hang'", "r");
    int n = -1;
    if (f) {
        if (fscanf(f, "%d", &n) != 1) n = -1;
        pclose(f);
    }
    return n;
}
/* "+n", or "unknown" when either count could not be read: never report
 * 0 resets that were not counted */
static const char* reset_delta(int r0, int r1) {
    static char b[32];
    if (r0 < 0 || r1 < 0) return "unknown (dmesg unreadable)";
    snprintf(b, sizeof(b), "%d", r1 - r0);
    return b;
}

#define SPIN_1MS 37500u

static void section_tick(uint32_t tick_gpu, uint32_t cfg) {
    if (cfg)
        printf("\n## A: GPU stores during a running job, each followed by a read with tmuc 0x%08x (op %u)\n", cfg,
               (cfg >> 4) & 15);
    else
        printf("\n## A: GPU stores during a running job (1 batch, a store + tmuwt every ~1 ms, 100 ticks)\n");
    volatile uint32_t* word = (volatile uint32_t*)data.map;
    for (int run = 0; run < 3; run++) {
        *word = 0;
        wbarrier();
        uint32_t* u = (uint32_t*)unif.map;
        u[0] = data.gpu;
        u[1] = 100;
        u[2] = SPIN_1MS;
        u[3] = cfg;
        static uint64_t seen_at[101];
        memset(seen_at, 0, sizeof(seen_at));
        uint64_t t0 = now_ns();
        int e = dispatch(tick_gpu, 0);
        if (e) die("dispatch", e);
        uint32_t last = 0;
        int distinct = 0;
        while (now_ns() - t0 < 2000000000ull) {
            uint32_t v = *word;
            if (v != last && v <= 100) {
                seen_at[v] = now_ns() - t0;
                distinct++;
                last = v;
                if (v == 100) break;
            }
        }
        e = hs_v3d_bo_wait(&dev, &data, 5000000000ull);
        uint64_t t_done = now_ns() - t0;
        /* gaps between consecutive observed values */
        double gmin = 1e18, gmax = 0, gsum = 0;
        int ng = 0;
        uint64_t prev = 0;
        for (int v = 1; v <= 100; v++) {
            if (!seen_at[v]) continue;
            if (prev) {
                double g = (double)(seen_at[v] - prev);
                gmin = g < gmin ? g : gmin;
                gmax = g > gmax ? g : gmax;
                gsum += g;
                ng++;
            }
            prev = seen_at[v];
        }
        printf("run %d: %d distinct values seen of 100; first (1) at %.3f ms, last (100) at %.3f ms; job done "
               "(WAIT_BO) at %.3f ms; final word %u; gaps between observed values min %.3f / mean %.3f / max %.3f ms%s\n",
               run, distinct, seen_at[1] / 1e6, seen_at[100] / 1e6, t_done / 1e6, *word, gmin / 1e6,
               ng ? gsum / ng / 1e6 : 0, gmax / 1e6, e ? " (WAIT_BO error)" : "");
    }
}

static void section_poll(uint32_t poll_gpu, uint32_t cfg, int runs) {
    if (cfg)
        printf("\n## B: polls with tmuc 0x%08x (op %u) before each load\n", cfg, (cfg >> 4) & 15);
    else
        printf("\n## B: GPU polls one word; the CPU writes it 30 ms after dispatch (past start-up)\n");
    volatile uint32_t* mbox = (volatile uint32_t*)(data.map + 4096);
    volatile uint32_t* ack = (volatile uint32_t*)(data.map + 8192);
    const uint32_t max_polls = 2000000;
    for (int run = 0; run < runs; run++) {
        *mbox = 0;
        ack[0] = 0xFFFFFFFFu;
        ack[1] = 0xFFFFFFFFu;
        wbarrier();
        uint32_t* u = (uint32_t*)unif.map + 64;
        u[0] = data.gpu + 4096;
        u[1] = data.gpu + 8192;
        u[2] = max_polls;
        u[3] = cfg;
        int r0 = resets();
        uint64_t t0 = now_ns();
        int e = dispatch(poll_gpu, 256);
        if (e) die("dispatch", e);
        /* write after 30 ms: past the ~10-12 ms first-job start-up measured
         * in a process, so the GPU is already polling when the write lands */
        while (now_ns() - t0 < 30000000ull) {
        }
        uint64_t tw = now_ns();
        *mbox = 1;
        wbarrier();
        uint64_t ta = 0;
        while (now_ns() - tw < 3000000000ull)
            if (ack[0] != 0xFFFFFFFFu) {
                ta = now_ns();
                break;
            }
        e = hs_v3d_bo_wait(&dev, &data, 5000000000ull);
        uint64_t td = now_ns();
        const char* rd = reset_delta(r0, resets());
        if (ack[0] == 0xFFFFFFFFu) {
            printf("run %d: NO ACK: the job never wrote its result (reset?); GPU resets: %s\n", run, rd);
            continue;
        }
        uint32_t polls = max_polls - ack[1];
        printf("run %d: GPU read %u%s; polls used %u; CPU write -> CPU sees ack %s%.1f us; dispatch -> done %.3f ms; "
               "polls/us averaged from dispatch (start-up included) %.2f; GPU resets: %s%s\n",
               run, ack[0], ack[0] == 1 ? " (saw the CPU's write)" : " (never saw it)", polls, ta ? "" : "never; ",
               ta ? (ta - tw) / 1e3 : 0.0, (td - t0) / 1e6, polls / ((double)(ta ? ta - t0 : td - t0) / 1e3), rd,
               e ? " (WAIT_BO error)" : "");
    }
}

static void section_evict(uint32_t poll_ev, uint32_t tick_ev) {
    static const uint32_t KB[] = {4, 16, 64, 96, 128, 160, 192, 256, 512};
    volatile uint32_t* mbox = (volatile uint32_t*)(data.map + 4096);
    volatile uint32_t* ack = (volatile uint32_t*)(data.map + 8192);
    volatile uint32_t* word = (volatile uint32_t*)data.map;
    printf("\n## E: eviction sweeps of F KiB (other memory) between polls / after each store\n");
    printf("%8s | %-44s | %-34s | %s\n", "F KiB", "poll: saw CPU write? latency, polls", "tick: values seen mid-job (of 20)", "GPU resets");
    for (size_t i = 0; i < sizeof(KB) / sizeof(KB[0]); i++) {
        uint32_t lines = KB[i] * 1024 / 64;
        /* poll */
        *mbox = 0;
        ack[0] = ack[1] = 0xFFFFFFFFu;
        wbarrier();
        uint32_t* u = (uint32_t*)unif.map + 192;
        u[0] = data.gpu + 4096;
        u[1] = data.gpu + 8192;
        /* poll budget: keep the job well under the ~1 s the kernel allows
         * (section C), so "NO" means "finished without seeing it", not a reset.
         * A sweep line costs ~100 ns (measured: 76 polls of 4096 lines in
         * the 30 ms before the write, 256 KiB row of evict-1), a poll ~100 ns. */
        uint32_t budget = (uint32_t)(800e6 / (lines * 100.0 + 100.0));
        u[2] = budget;
        u[3] = data.gpu + (1u << 20);
        u[4] = lines;
        int r0 = resets();
        uint64_t t0 = now_ns();
        int e = dispatch(poll_ev, 768);
        if (e) die("dispatch", e);
        while (now_ns() - t0 < 30000000ull) { /* past first-job start-up, then write */
        }
        uint64_t tw = now_ns(), ta = 0;
        *mbox = 1;
        wbarrier();
        while (now_ns() - tw < 4000000000ull)
            if (ack[0] != 0xFFFFFFFFu) {
                ta = now_ns();
                break;
            }
        hs_v3d_bo_wait(&dev, &data, 10000000000ull);
        char pr[64];
        if (ack[0] == 0xFFFFFFFFu)
            snprintf(pr, sizeof(pr), "NO ACK: job never finished (reset?)");
        else
            snprintf(pr, sizeof(pr), "%s %.1f us, %u of %u polls", ack[0] == 1 ? "yes" : "NO ",
                     ta ? (ta - tw) / 1e3 : 0.0, budget - ack[1], budget);
        /* tick: 20 ticks of ~1 ms */
        *word = 0;
        wbarrier();
        u = (uint32_t*)unif.map + 256;
        u[0] = data.gpu;
        u[1] = 20;
        u[2] = SPIN_1MS;
        u[3] = data.gpu + (1u << 20);
        u[4] = lines;
        t0 = now_ns();
        if ((e = dispatch(tick_ev, 1024))) die("dispatch", e);
        uint32_t last = 0;
        int distinct = 0;
        while (now_ns() - t0 < 4000000000ull) {
            uint32_t v = *word;
            if (v != last) {
                distinct++;
                last = v;
                if (v == 20) break;
            }
        }
        hs_v3d_bo_wait(&dev, &data, 10000000000ull);
        printf("%8u | %-44s | %-34d | resets %s\n", KB[i], pr, distinct, reset_delta(r0, resets()));
    }
}


static void section_life(uint32_t tick_gpu) {
    printf("\n## C: one job running for longer and longer (the tick program, ~1 ms per tick)\n");
    volatile uint32_t* word = (volatile uint32_t*)data.map;
    static const uint32_t TICKS[] = {250, 500, 750, 1000, 1500, 2000, 4000};
    for (size_t i = 0; i < sizeof(TICKS) / sizeof(TICKS[0]); i++) {
        int r0 = resets();
        *word = 0;
        wbarrier();
        uint32_t* u = (uint32_t*)unif.map + 128;
        u[0] = data.gpu;
        u[1] = TICKS[i];
        u[2] = SPIN_1MS;
        uint64_t t0 = now_ns();
        int e = dispatch(tick_gpu, 512);
        if (e) die("dispatch", e);
        e = hs_v3d_bo_wait(&dev, &data, 20000000000ull);
        double ms = (double)(now_ns() - t0) / 1e6;
        printf("%u ticks: job ended after %.1f ms, final tick %u of %u, GPU resets during the job: %s%s\n", TICKS[i], ms,
               *word, TICKS[i], reset_delta(r0, resets()), e ? " (WAIT_BO error)" : "");
        if (*word != TICKS[i]) {
            printf("   the job was cut short; longer runs skipped\n");
            break;
        }
    }
}

int main(int argc, char** argv) {
    const char* which = argc > 1 ? argv[1] : "ABE";
    int cpu = argc > 2 ? atoi(argv[2]) : 2;
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    sched_setaffinity(0, sizeof(s), &s);
    setvbuf(stdout, NULL, _IOLBF, 0);
    int e = hs_v3d_open(&dev, NULL);
    if (e) die("hs_v3d_open", e);
    if ((e = hs_v3d_bo_new(&dev, &code, 64 << 10)) || (e = hs_v3d_bo_new(&dev, &unif, 4 << 10)) ||
        (e = hs_v3d_bo_new(&dev, &data, 2 << 20)))
        die("bo_new", e);
    handles[0] = code.handle;
    handles[1] = unif.handle;
    handles[2] = data.handle;
    uint32_t tick_gpu = load_prog(prog_tick, 0), poll_gpu = load_prog(prog_poll, 4096);
    uint32_t poll_cfg_gpu = load_prog(prog_poll_cfg, 8192), tick_cfg_gpu = load_prog(prog_tick_cfg, 12288);
    uint32_t poll_ev_gpu = load_prog(prog_poll_evict, 16384), tick_ev_gpu = load_prog(prog_tick_evict, 20480);
    printf("# hwprobe_mailbox (NeoGPU encoder + runtime), CPU on core %d\n", cpu);
    if (strchr(which, 'A')) section_tick(tick_gpu, 0);
    if (strchr(which, 'B')) section_poll(poll_gpu, 0, 8);
    /* O: the TMU-op variants, ops 0..4 and 15 (regular, as a control) */
    static const uint32_t OPS[] = {15, 0, 1, 2, 3, 4};
    for (size_t i = 0; strchr(which, 'O') && i < sizeof(OPS) / sizeof(OPS[0]); i++) {
        uint32_t cfg = 0xFFFFFF00u | OPS[i] << 4 | 0xFu;
        int r0 = resets();
        section_poll(poll_cfg_gpu, cfg, 3);
        section_tick(tick_cfg_gpu, cfg);
        printf("   GPU resets during op %u: %s\n", OPS[i], reset_delta(r0, resets()));
    }
    if (strchr(which, 'E')) section_evict(poll_ev_gpu, tick_ev_gpu);
    if (strchr(which, 'C')) section_life(tick_gpu);
    return 0;
}
