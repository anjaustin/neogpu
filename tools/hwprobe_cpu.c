/* hwprobe_cpu: measure the Cortex-A72 side of the Raspberry Pi 4 directly.
 *
 * Every number comes from timing real instruction streams (inline AArch64
 * assembly) on a pinned core, the median of TRIALS runs each long enough to
 * swamp timer resolution. Cycles come from the kernel's cycle counter
 * (perf_event; needs root or perf_event_paranoid <= 2) and are cross-checked
 * against wall time: a chain of dependent 1-cycle adds must give the clock
 * frequency both ways.
 *
 * Sections: timers, clock, instruction latency/throughput (integer, scalar
 * FP, NEON), load-to-use latency across working-set sizes, bandwidth (read,
 * write, non-temporal write, copy; 1 and 4 cores), core-to-core cache-line
 * round trip, atomics and barriers, and kernel crossings (syscall, futex).
 *
 * Usage: sudo tools/hwprobe_cpu [cpu=0]
 * Run with a fixed CPU frequency (scaling_governor=performance), or the
 * cycle/ns cross-check will say so.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <linux/futex.h>
#include <linux/perf_event.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define TRIALS 9
#define TARGET_NS 20000000ull /* each trial runs >= 20 ms */

static int g_cpu;
static int perf_fd = -1;

static inline uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
static inline uint64_t cntvct(void) {
    uint64_t v;
    __asm__ volatile("isb\n mrs %0, cntvct_el0" : "=r"(v));
    return v;
}
static uint64_t cntfrq(void) {
    uint64_t v;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v));
    return v;
}
static void perf_open(void) {
    struct perf_event_attr a;
    memset(&a, 0, sizeof(a));
    a.type = PERF_TYPE_HARDWARE;
    a.size = sizeof(a);
    a.config = PERF_COUNT_HW_CPU_CYCLES;
    a.exclude_hv = 1;
    perf_fd = (int)syscall(__NR_perf_event_open, &a, 0, -1, -1, 0);
}
static inline uint64_t cycles(void) {
    uint64_t v = 0;
    if (perf_fd >= 0 && read(perf_fd, &v, sizeof(v)) != sizeof(v)) v = 0;
    return v;
}
static void pin(int cpu) {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    if (sched_setaffinity(0, sizeof(s), &s) != 0) perror("sched_setaffinity");
}
static int cmpd(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}
static double median(double* v, int n) {
    qsort(v, (size_t)n, sizeof(double), cmpd);
    return v[n / 2];
}

typedef void (*kern_fn)(uint64_t n, void* arg);
typedef struct {
    double ns, cyc, ns_min, ns_max;
} res_t;

/* Median ns and cycles per op over TRIALS runs of f(n, arg), with ops = n *
 * ops_per_n. n is doubled until one run takes TARGET_NS. */
static res_t measure(kern_fn f, void* arg, double ops_per_n) {
    uint64_t n = 16;
    for (;;) {
        uint64_t t0 = now_ns();
        f(n, arg);
        if (now_ns() - t0 >= TARGET_NS / 4) break;
        n *= 2;
    }
    n *= 4;
    double ns[TRIALS], cy[TRIALS];
    for (int t = 0; t < TRIALS; t++) {
        uint64_t c0 = cycles(), t0 = now_ns();
        f(n, arg);
        uint64_t t1 = now_ns(), c1 = cycles();
        ns[t] = (double)(t1 - t0) / ((double)n * ops_per_n);
        cy[t] = (double)(c1 - c0) / ((double)n * ops_per_n);
    }
    res_t r;
    r.cyc = median(cy, TRIALS);
    r.ns = median(ns, TRIALS); /* sorts ns */
    r.ns_min = ns[0];
    r.ns_max = ns[TRIALS - 1];
    return r;
}

/* ---- instruction kernels: .rept 100 of the body per loop iteration ----
 * The prologue runs once and the loop is inside the asm, so a dependent chain
 * stays unbroken for the whole run. (Re-running the prologue every iteration
 * restarts the chain and lets the out-of-order core overlap iterations: that
 * measured dependent adds at 0.93 cycles.) */
#define KERN(name, pro, body, ...)                                                  \
    static void name(uint64_t n, void* arg) {                                       \
        __asm__ volatile(pro "\n1:\n.rept 100\n" body "\n.endr\n"                   \
                         " subs %1, %1, #1\n b.ne 1b\n"                             \
                         : "+r"(arg), "+r"(n)::__VA_ARGS__, "memory", "cc");         \
    }

#define FPINIT "fmov v16.4s, #1.0\n fmov v17.4s, #1.0\n movi v18.16b, #0\n"
#define ACC14 "v18","v19","v20","v21","v22","v23","v24","v25","v26","v27","v28","v29","v30","v31"

/* latency: one dependent chain */
KERN(lat_add, "mov x9, #0", "add x9, x9, #1", "x9")
KERN(lat_mul, "mov x9, #3", "mul x9, x9, x9", "x9")
KERN(lat_ldr, "mov x9, %0", "ldr x9, [x9]", "x9")
KERN(lat_fadd, FPINIT, "fadd d16, d16, d17", "v16", "v17", "v18")
KERN(lat_fmul, FPINIT, "fmul d16, d16, d17", "v16", "v17", "v18")
KERN(lat_fmadd, FPINIT, "fmadd d16, d16, d17, d18", "v16", "v17", "v18")
KERN(lat_vadd, FPINIT, "add v16.4s, v16.4s, v17.4s", "v16", "v17", "v18")
KERN(lat_vfadd, FPINIT, "fadd v16.4s, v16.4s, v17.4s", "v16", "v17", "v18")
KERN(lat_vfmul, FPINIT, "fmul v16.4s, v16.4s, v17.4s", "v16", "v17", "v18")
KERN(lat_vfmla, FPINIT, "fmla v18.4s, v16.4s, v17.4s", "v16", "v17", "v18")
KERN(lat_vmov_gp, "mov x9, #0", "fmov d16, x9\n fmov x9, d16", "x9", "v16")

/* throughput: independent chains */
KERN(tp_add, "",
     "add x9,x9,#1\n add x10,x10,#1\n add x11,x11,#1\n add x12,x12,#1\n add x13,x13,#1\n add x14,x14,#1\n add x15,x15,#1\n add x16,x16,#1",
     "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16")
KERN(tp_mul, "",
     "mul x9,x9,x9\n mul x10,x10,x10\n mul x11,x11,x11\n mul x12,x12,x12\n mul x13,x13,x13\n mul x14,x14,x14\n mul x15,x15,x15\n mul x16,x16,x16",
     "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16")
KERN(tp_ldr, "mov x9, %0",
     "ldr x10,[x9]\n ldr x11,[x9,#8]\n ldr x12,[x9,#16]\n ldr x13,[x9,#24]\n ldr x14,[x9,#32]\n ldr x15,[x9,#40]\n ldr x16,[x9,#48]\n ldr x17,[x9,#56]",
     "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17")
KERN(tp_ldq, "mov x9, %0",
     "ldr q0,[x9]\n ldr q1,[x9,#16]\n ldr q2,[x9,#32]\n ldr q3,[x9,#48]\n ldr q4,[x9,#64]\n ldr q5,[x9,#80]\n ldr q6,[x9,#96]\n ldr q7,[x9,#112]",
     "x9", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7")
KERN(tp_stq, "mov x9, %0",
     "str q0,[x9]\n str q1,[x9,#16]\n str q2,[x9,#32]\n str q3,[x9,#48]\n str q4,[x9,#64]\n str q5,[x9,#80]\n str q6,[x9,#96]\n str q7,[x9,#112]",
     "x9", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7")
#define T14(op) \
    op " v18.4s,v16.4s,v17.4s\n" op " v19.4s,v16.4s,v17.4s\n" op " v20.4s,v16.4s,v17.4s\n" op " v21.4s,v16.4s,v17.4s\n" \
    op " v22.4s,v16.4s,v17.4s\n" op " v23.4s,v16.4s,v17.4s\n" op " v24.4s,v16.4s,v17.4s\n" op " v25.4s,v16.4s,v17.4s\n" \
    op " v26.4s,v16.4s,v17.4s\n" op " v27.4s,v16.4s,v17.4s\n" op " v28.4s,v16.4s,v17.4s\n" op " v29.4s,v16.4s,v17.4s\n" \
    op " v30.4s,v16.4s,v17.4s\n" op " v31.4s,v16.4s,v17.4s"
KERN(tp_vfmla, FPINIT, T14("fmla"), "v16", "v17", ACC14)
KERN(tp_vfadd, FPINIT, T14("fadd"), "v16", "v17", ACC14)
KERN(tp_vfmul, FPINIT, T14("fmul"), "v16", "v17", ACC14)
KERN(tp_vadd, FPINIT, T14("add"), "v16", "v17", ACC14)
KERN(tp_vmul, FPINIT, T14("mul"), "v16", "v17", ACC14)
#define T14D(op) \
    op " d18,d16,d17\n" op " d19,d16,d17\n" op " d20,d16,d17\n" op " d21,d16,d17\n" op " d22,d16,d17\n" op " d23,d16,d17\n" \
    op " d24,d16,d17\n" op " d25,d16,d17\n" op " d26,d16,d17\n" op " d27,d16,d17\n" op " d28,d16,d17\n" op " d29,d16,d17\n" \
    op " d30,d16,d17\n" op " d31,d16,d17"
KERN(tp_fmul, FPINIT, T14D("fmul"), "v16", "v17", ACC14)
/* mixed: one NEON fmla + one 16-byte load + one integer add per step */
KERN(tp_mix, FPINIT "mov x9, %0",
     "fmla v18.4s,v16.4s,v17.4s\n ldr q0,[x9]\n add x10,x10,#1\n fmla v19.4s,v16.4s,v17.4s\n ldr q1,[x9,#16]\n add x11,x11,#1\n"
     "fmla v20.4s,v16.4s,v17.4s\n ldr q2,[x9,#32]\n add x12,x12,#1\n fmla v21.4s,v16.4s,v17.4s\n ldr q3,[x9,#48]\n add x13,x13,#1\n"
     "fmla v22.4s,v16.4s,v17.4s\n ldr q4,[x9,#64]\n add x14,x14,#1\n fmla v23.4s,v16.4s,v17.4s\n ldr q5,[x9,#80]\n add x15,x15,#1\n"
     "fmla v24.4s,v16.4s,v17.4s\n ldr q6,[x9,#96]\n add x16,x16,#1\n fmla v25.4s,v16.4s,v17.4s\n ldr q7,[x9,#112]\n add x17,x17,#1",
     "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
     "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25")
/* barriers and exclusives, each in a dependent sequence */
KERN(lat_dmb, "", "dmb ish", "x9")
KERN(lat_dsb, "", "dsb ish", "x9")
KERN(lat_isb, "", "isb", "x9")
KERN(lat_excl, "mov x9, %0",
     "0: ldxr x10, [x9]\n add x10, x10, #1\n stxr w11, x10, [x9]\n cbnz w11, 0b", "x9", "x10", "x11")
KERN(lat_excl_acqrel, "mov x9, %0",
     "0: ldaxr x10, [x9]\n add x10, x10, #1\n stlxr w11, x10, [x9]\n cbnz w11, 0b", "x9", "x10", "x11")
KERN(lat_ldar, "mov x9, %0", "ldar x10, [x9]", "x9", "x10")
KERN(lat_stlr, "mov x9, %0", "stlr x10, [x9]", "x9", "x10")
KERN(lat_cntvct, "", "mrs x9, cntvct_el0", "x9")

typedef struct {
    const char* name;
    kern_fn fn;
    double ops_per_n; /* instructions timed per n */
    double flops;     /* FP ops per instruction (0: not FP) */
    int is_lat;
} itest_t;

static void section_instructions(double ghz) {
    static uint64_t cell[64] __attribute__((aligned(128)));
    cell[0] = (uint64_t)&cell[0];
    static const itest_t T[] = {
        {"add x (int add)", lat_add, 100, 0, 1},
        {"mul x (int mul)", lat_mul, 100, 0, 1},
        {"ldr x (L1 load-to-use)", lat_ldr, 100, 0, 1},
        {"fadd d (scalar fp64)", lat_fadd, 100, 0, 1},
        {"fmul d", lat_fmul, 100, 0, 1},
        {"fmadd d", lat_fmadd, 100, 0, 1},
        {"add v.4s (NEON int)", lat_vadd, 100, 0, 1},
        {"fadd v.4s (NEON fp32)", lat_vfadd, 100, 0, 1},
        {"fmul v.4s", lat_vfmul, 100, 0, 1},
        {"fmla v.4s", lat_vfmla, 100, 0, 1},
        {"fmov x->d->x round trip", lat_vmov_gp, 100, 0, 1},
        {"ldar (load-acquire)", lat_ldar, 100, 0, 1},
        {"stlr (store-release)", lat_stlr, 100, 0, 1},
        {"ldxr/add/stxr (uncontended)", lat_excl, 100, 0, 1},
        {"ldaxr/add/stlxr", lat_excl_acqrel, 100, 0, 1},
        {"dmb ish", lat_dmb, 100, 0, 1},
        {"dsb ish", lat_dsb, 100, 0, 1},
        {"isb", lat_isb, 100, 0, 1},
        {"mrs cntvct_el0", lat_cntvct, 100, 0, 1},
        {"add x x8 independent", tp_add, 800, 0, 0},
        {"mul x x8 independent", tp_mul, 800, 0, 0},
        {"ldr x x8 (L1)", tp_ldr, 800, 0, 0},
        {"ldr q x8 (L1, 16 B)", tp_ldq, 800, 0, 0},
        {"str q x8 (L1, 16 B)", tp_stq, 800, 0, 0},
        {"fmul d x14", tp_fmul, 1400, 1, 0},
        {"fadd v.4s x14", tp_vfadd, 1400, 4, 0},
        {"fmul v.4s x14", tp_vfmul, 1400, 4, 0},
        {"fmla v.4s x14", tp_vfmla, 1400, 8, 0},
        {"add v.4s x14", tp_vadd, 1400, 0, 0},
        {"mul v.4s x14", tp_vmul, 1400, 0, 0},
        {"fmla+ldr q+add (x8 each)", tp_mix, 2400, 0, 0},
    };
    printf("\n## Instructions (core %d)\n", g_cpu);
    printf("%-30s %10s %10s %12s %14s\n", "instruction", "cycles", "ns", "per cycle", "GFLOP/s/core");
    for (size_t i = 0; i < sizeof(T) / sizeof(T[0]); i++) {
        res_t r = measure(T[i].fn, cell, T[i].ops_per_n);
        double cyc = r.cyc > 0 ? r.cyc : r.ns * ghz;
        char gf[32] = "";
        if (T[i].flops > 0) snprintf(gf, sizeof(gf), "%.2f", T[i].flops / r.ns);
        printf("%-30s %10.3f %10.3f %12.3f %14s  %s\n", T[i].name, cyc, r.ns, 1.0 / cyc, gf,
               T[i].is_lat ? "latency" : "throughput");
    }
}

/* ---- memory latency: random cyclic pointer chase over one cache line per node ---- */
typedef struct {
    void** start;
} chase_arg;
static void* volatile sink;
static void chase(uint64_t n, void* a) {
    void** p = ((chase_arg*)a)->start;
    for (uint64_t i = 0; i < n; i++) {
        p = (void**)*p; p = (void**)*p; p = (void**)*p; p = (void**)*p;
        p = (void**)*p; p = (void**)*p; p = (void**)*p; p = (void**)*p;
        p = (void**)*p; p = (void**)*p; p = (void**)*p; p = (void**)*p;
        p = (void**)*p; p = (void**)*p; p = (void**)*p; p = (void**)*p;
    }
    sink = p;
}
static uint64_t rng = 88172645463325252ull;
static uint64_t xs(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return rng;
}
static void* big_alloc(size_t size, int huge) {
    void* p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    madvise(p, size, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    memset(p, 1, size);
    return p;
}
static double chase_ns(size_t size, int huge, size_t stride) {
    char* buf = big_alloc(size, huge);
    size_t nodes = size / stride;
    size_t* order = malloc(nodes * sizeof(size_t));
    for (size_t i = 0; i < nodes; i++) order[i] = i;
    for (size_t i = nodes - 1; i > 0; i--) { /* Sattolo: one cycle through all nodes */
        size_t j = xs() % i;
        size_t t = order[i]; order[i] = order[j]; order[j] = t;
    }
    for (size_t i = 0; i < nodes; i++)
        *(void**)(buf + order[i] * stride) = buf + order[(i + 1) % nodes] * stride;
    chase_arg a = {(void**)(buf + order[0] * stride)};
    res_t r = measure(chase, &a, 16);
    free(order);
    munmap(buf, size);
    return r.ns;
}
static void section_latency(double ghz) {
    static const size_t sizes[] = {4 << 10, 16 << 10, 24 << 10, 32 << 10, 48 << 10, 64 << 10, 128 << 10, 256 << 10,
                                   512 << 10, 768 << 10, 1 << 20, 2 << 20, 4 << 20, 8 << 20, 32 << 20, 128 << 20, 512 << 20};
    printf("\n## Load-to-use latency, random chase, one 64 B line per node (core %d)\n", g_cpu);
    printf("%10s %12s %10s %14s %10s\n", "set", "4K pages ns", "cycles", "huge pages ns", "cycles");
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        double a = chase_ns(sizes[i], 0, 64), b = chase_ns(sizes[i], 1, 64);
        printf("%8zuK %12.2f %10.1f %14.2f %10.1f\n", sizes[i] >> 10, a, a * ghz, b, b * ghz);
        fflush(stdout);
    }
}

/* ---- bandwidth ---- */
typedef struct {
    char* src;
    char* dst;
    size_t len;
    int kind;
} bw_arg;
static void bw_kernel(uint64_t n, void* va) {
    bw_arg* a = va;
    for (uint64_t r = 0; r < n; r++) {
        char* s = a->src;
        char* d = a->dst;
        size_t len = a->len;
        switch (a->kind) {
        case 0: /* read */
            __asm__ volatile("1: ldp q0,q1,[%0]\n ldp q2,q3,[%0,#32]\n ldp q4,q5,[%0,#64]\n ldp q6,q7,[%0,#96]\n"
                             " add %0,%0,#128\n subs %1,%1,#128\n b.ne 1b"
                             : "+r"(s), "+r"(len)::"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory", "cc");
            break;
        case 1: /* write */
            __asm__ volatile("1: stp q0,q1,[%0]\n stp q2,q3,[%0,#32]\n stp q4,q5,[%0,#64]\n stp q6,q7,[%0,#96]\n"
                             " add %0,%0,#128\n subs %1,%1,#128\n b.ne 1b"
                             : "+r"(d), "+r"(len)::"memory", "cc");
            break;
        case 2: /* non-temporal write */
            __asm__ volatile("1: stnp q0,q1,[%0]\n stnp q2,q3,[%0,#32]\n stnp q4,q5,[%0,#64]\n stnp q6,q7,[%0,#96]\n"
                             " add %0,%0,#128\n subs %1,%1,#128\n b.ne 1b"
                             : "+r"(d), "+r"(len)::"memory", "cc");
            break;
        case 3: /* copy */
            __asm__ volatile("1: ldp q0,q1,[%0]\n ldp q2,q3,[%0,#32]\n ldp q4,q5,[%0,#64]\n ldp q6,q7,[%0,#96]\n"
                             " stp q0,q1,[%1]\n stp q2,q3,[%1,#32]\n stp q4,q5,[%1,#64]\n stp q6,q7,[%1,#96]\n"
                             " add %0,%0,#128\n add %1,%1,#128\n subs %2,%2,#128\n b.ne 1b"
                             : "+r"(s), "+r"(d), "+r"(len)::"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory", "cc");
            break;
        case 4: memcpy(d, s, len); break;
        case 5: memset(d, (int)r, len); break;
        }
    }
}
static const char* bw_names[] = {"read (ldp q)", "write (stp q)", "write (stnp q)", "copy (ldp/stp q)", "memcpy (libc)", "memset (libc)"};

typedef struct {
    int cpu, kind;
    size_t len;
    uint64_t reps;
    pthread_barrier_t* bar;
    double ns;
} bw_thread;
static void* bw_thread_main(void* v) {
    bw_thread* t = v;
    pin(t->cpu);
    bw_arg a = {big_alloc(t->len, 1), big_alloc(t->len, 1), t->len, t->kind};
    bw_kernel(1, &a);
    pthread_barrier_wait(t->bar);
    uint64_t t0 = now_ns();
    bw_kernel(t->reps, &a);
    t->ns = (double)(now_ns() - t0);
    pthread_barrier_wait(t->bar);
    munmap(a.src, t->len);
    munmap(a.dst, t->len);
    return NULL;
}
/* GB/s summed over threads; bytes = bytes read or written (copy counts both). */
static double bw_threads(int nthreads, int kind, size_t len) {
    double best[TRIALS];
    uint64_t reps = (512ull << 20) / len;
    if (reps < 2) reps = 2;
    for (int trial = 0; trial < 5; trial++) {
        pthread_barrier_t bar;
        pthread_barrier_init(&bar, NULL, (unsigned)nthreads);
        bw_thread th[4];
        pthread_t id[4];
        for (int i = 0; i < nthreads; i++) {
            th[i] = (bw_thread){i, kind, len, reps, &bar, 0};
            pthread_create(&id[i], NULL, bw_thread_main, &th[i]);
        }
        double maxns = 0;
        for (int i = 0; i < nthreads; i++) {
            pthread_join(id[i], NULL);
            if (th[i].ns > maxns) maxns = th[i].ns;
        }
        pthread_barrier_destroy(&bar);
        double bytes = (double)len * (double)reps * nthreads * ((kind == 3 || kind == 4) ? 2 : 1);
        best[trial] = bytes / maxns;
    }
    return median(best, 5);
}
static void section_bandwidth(void) {
    static const size_t sizes[] = {16 << 10, 256 << 10, 768 << 10, 64 << 20};
    printf("\n## Bandwidth GB/s (bytes moved; copy counts read + write), huge pages\n");
    printf("%-18s", "kind");
    for (size_t s = 0; s < 4; s++) printf(" %8zuK x1", sizes[s] >> 10);
    printf(" %10s %10s\n", "64M x2", "64M x4");
    for (int k = 0; k < 6; k++) {
        printf("%-18s", bw_names[k]);
        for (size_t s = 0; s < 4; s++) printf(" %11.2f", bw_threads(1, k, sizes[s]));
        printf(" %10.2f %10.2f\n", bw_threads(2, k, 64 << 20), bw_threads(4, k, 64 << 20));
        fflush(stdout);
    }
    pin(g_cpu);
}

/* ---- core to core: one cache line bounced between two pinned threads ---- */
static _Alignas(128) _Atomic uint64_t pp_line;
static _Atomic int pp_go;
typedef struct {
    int cpu;
    uint64_t iters;
} pp_arg;
static void* pp_responder(void* v) {
    pp_arg* a = v;
    pin(a->cpu);
    atomic_store(&pp_go, 1);
    for (uint64_t i = 0; i < a->iters; i++) {
        uint64_t want = 2 * i + 1;
        while (atomic_load_explicit(&pp_line, memory_order_acquire) != want) {}
        atomic_store_explicit(&pp_line, want + 1, memory_order_release);
    }
    return NULL;
}
static double pingpong_ns(int ca, int cb) {
    double r[TRIALS];
    const uint64_t iters = 200000;
    for (int t = 0; t < TRIALS; t++) {
        atomic_store(&pp_line, 0);
        atomic_store(&pp_go, 0);
        pp_arg a = {cb, iters};
        pthread_t id;
        pin(ca);
        pthread_create(&id, NULL, pp_responder, &a);
        while (!atomic_load(&pp_go)) {}
        uint64_t t0 = now_ns();
        for (uint64_t i = 0; i < iters; i++) {
            atomic_store_explicit(&pp_line, 2 * i + 1, memory_order_release);
            while (atomic_load_explicit(&pp_line, memory_order_acquire) != 2 * i + 2) {}
        }
        r[t] = (double)(now_ns() - t0) / (double)iters;
        pthread_join(id, NULL);
    }
    pin(g_cpu);
    return median(r, TRIALS);
}
/* contended exclusive increments from n cores on one line */
static _Alignas(128) uint64_t ct_line;
typedef struct {
    int cpu;
    uint64_t iters;
    pthread_barrier_t* bar;
} ct_arg;
static void* ct_main(void* v) {
    ct_arg* a = v;
    pin(a->cpu);
    pthread_barrier_wait(a->bar);
    uint64_t* p = &ct_line;
    for (uint64_t i = 0; i < a->iters; i++)
        __asm__ volatile("0: ldxr x10, [%0]\n add x10, x10, #1\n stxr w11, x10, [%0]\n cbnz w11, 0b" ::"r"(p)
                         : "x10", "x11", "memory");
    pthread_barrier_wait(a->bar);
    return NULL;
}
static double contended_ns(int n) {
    double r[5];
    const uint64_t iters = 1000000;
    for (int t = 0; t < 5; t++) {
        pthread_barrier_t bar;
        pthread_barrier_init(&bar, NULL, (unsigned)n + 1);
        ct_arg a[4];
        pthread_t id[4];
        for (int i = 0; i < n; i++) {
            a[i] = (ct_arg){i, iters, &bar};
            pthread_create(&id[i], NULL, ct_main, &a[i]);
        }
        pthread_barrier_wait(&bar);
        uint64_t t0 = now_ns();
        pthread_barrier_wait(&bar);
        r[t] = (double)(now_ns() - t0) / (double)(iters * (uint64_t)n);
        for (int i = 0; i < n; i++) pthread_join(id[i], NULL);
        pthread_barrier_destroy(&bar);
    }
    return median(r, 5);
}
static void section_coherence(void) {
    printf("\n## Core to core: one cache line, store then wait for the reply\n");
    printf("%-10s %14s %14s\n", "cores", "round trip ns", "one way ns");
    for (int a = 0; a < 4; a++)
        for (int b = a + 1; b < 4; b++) {
            double ns = pingpong_ns(a, b);
            printf("%d <-> %-4d %14.1f %14.1f\n", a, b, ns, ns / 2);
        }
    printf("\n## Contended ldxr/stxr increments on one line\n");
    printf("%-8s %16s %16s\n", "cores", "ns per increment", "M increments/s");
    for (int n = 1; n <= 4; n++) {
        double ns = contended_ns(n);
        printf("%-8d %16.2f %16.1f\n", n, ns, 1e3 / ns);
    }
}

/* ---- kernel crossings ---- */
static void k_getppid(uint64_t n, void* a) {
    (void)a;
    for (uint64_t i = 0; i < n; i++) syscall(SYS_getppid);
}
static void k_gettime(uint64_t n, void* a) {
    (void)a;
    struct timespec t;
    for (uint64_t i = 0; i < n; i++) clock_gettime(CLOCK_MONOTONIC_RAW, &t);
}
static void k_yield(uint64_t n, void* a) {
    (void)a;
    for (uint64_t i = 0; i < n; i++) sched_yield();
}
static _Atomic uint32_t fx_word;
static void* fx_responder(void* v) {
    pp_arg* a = v;
    pin(a->cpu);
    for (uint64_t i = 0; i < a->iters; i++) {
        uint32_t want = (uint32_t)(2 * i + 1);
        while (atomic_load(&fx_word) != want)
            syscall(SYS_futex, &fx_word, FUTEX_WAIT_PRIVATE, want - 1, NULL, NULL, 0);
        atomic_store(&fx_word, want + 1);
        syscall(SYS_futex, &fx_word, FUTEX_WAKE_PRIVATE, 1, NULL, NULL, 0);
    }
    return NULL;
}
static double futex_rt_ns(int ca, int cb) {
    double r[5];
    const uint64_t iters = 20000;
    for (int t = 0; t < 5; t++) {
        atomic_store(&fx_word, 0);
        pp_arg a = {cb, iters};
        pthread_t id;
        pin(ca);
        pthread_create(&id, NULL, fx_responder, &a);
        uint64_t t0 = now_ns();
        for (uint64_t i = 0; i < iters; i++) {
            uint32_t mine = (uint32_t)(2 * i + 1);
            atomic_store(&fx_word, mine);
            syscall(SYS_futex, &fx_word, FUTEX_WAKE_PRIVATE, 1, NULL, NULL, 0);
            while (atomic_load(&fx_word) != mine + 1)
                syscall(SYS_futex, &fx_word, FUTEX_WAIT_PRIVATE, mine, NULL, NULL, 0);
        }
        r[t] = (double)(now_ns() - t0) / (double)iters;
        pthread_join(id, NULL);
    }
    pin(g_cpu);
    return median(r, 5);
}
static void section_kernel(void) {
    printf("\n## Kernel crossings (core %d)\n", g_cpu);
    res_t a = measure(k_getppid, NULL, 1), b = measure(k_gettime, NULL, 1), c = measure(k_yield, NULL, 1);
    printf("%-34s %10.1f ns\n", "syscall getppid", a.ns);
    printf("%-34s %10.1f ns\n", "clock_gettime (vDSO)", b.ns);
    printf("%-34s %10.1f ns\n", "sched_yield (nothing else runnable)", c.ns);
    printf("%-34s %10.1f ns\n", "futex wait/wake round trip 0<->1", futex_rt_ns(0, 1));
    printf("%-34s %10.1f ns\n", "futex wait/wake round trip 0<->3", futex_rt_ns(0, 3));
}

/* ---- NEON fmla throughput on 1..4 cores at once ---- */
typedef struct {
    int cpu;
    uint64_t n;
    pthread_barrier_t* bar;
    double ns;
} fl_thread;
static void* fl_main(void* v) {
    fl_thread* t = v;
    pin(t->cpu);
    tp_vfmla(t->n / 8, NULL);
    pthread_barrier_wait(t->bar);
    uint64_t t0 = now_ns();
    tp_vfmla(t->n, NULL);
    t->ns = (double)(now_ns() - t0);
    return NULL;
}
static void section_allcores(void) {
    printf("\n## fmla v.4s x14 on several cores at once (GFLOP/s, fmla = 8 flops)\n");
    uint64_t n = 16;
    for (;;) { /* n for ~200 ms on one core */
        uint64_t t0 = now_ns();
        tp_vfmla(n, NULL);
        if (now_ns() - t0 >= 50000000ull) { n *= 4; break; }
        n *= 2;
    }
    for (int k = 1; k <= 4; k++) {
        double r[5], last[4];
        for (int trial = 0; trial < 5; trial++) {
            pthread_barrier_t bar;
            pthread_barrier_init(&bar, NULL, (unsigned)k);
            fl_thread th[4];
            pthread_t id[4];
            for (int i = 0; i < k; i++) {
                th[i] = (fl_thread){i, n, &bar, 0};
                pthread_create(&id[i], NULL, fl_main, &th[i]);
            }
            double maxns = 0;
            for (int i = 0; i < k; i++) {
                pthread_join(id[i], NULL);
                if (th[i].ns > maxns) maxns = th[i].ns;
            }
            pthread_barrier_destroy(&bar);
            r[trial] = (double)n * 1400.0 * 8.0 * k / maxns;
            if (trial == 4) {
                for (int i = 0; i < k; i++) last[i] = (double)n * 1400.0 * 8.0 / th[i].ns;
            }
        }
        printf("%d core%s %8.2f   last trial per core:", k, k > 1 ? "s" : " ", median(r, 5));
        for (int i = 0; i < k; i++) printf(" %.2f", last[i]);
        printf("\n");
    }
    pin(g_cpu);
}

int main(int argc, char** argv) {
    g_cpu = argc > 1 ? atoi(argv[1]) : 0;
    pin(g_cpu);
    setvbuf(stdout, NULL, _IOLBF, 0);
    perf_open();
    char gov[64] = "?";
    FILE* f = fopen("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "r");
    if (f) { if (fscanf(f, "%63s", gov) != 1) strcpy(gov, "?"); fclose(f); }

    printf("# hwprobe_cpu: Cortex-A72, core %d, governor %s, cycle counter %s\n", g_cpu, gov,
           perf_fd >= 0 ? "perf_event" : "UNAVAILABLE (cycles derived from ns)");
    printf("\n## Timers\n");
    uint64_t f0 = cntfrq();
    printf("cntfrq_el0: %llu Hz (%.2f ns per tick)\n", (unsigned long long)f0, 1e9 / (double)f0);
    uint64_t mind = UINT64_MAX;
    for (int i = 0; i < 100000; i++) {
        uint64_t a = cntvct(), b = cntvct();
        if (b > a && b - a < mind) mind = b - a;
    }
    printf("smallest nonzero cntvct step: %llu ticks\n", (unsigned long long)mind);

    /* clock: dependent adds, 1 cycle each, timed by wall clock and by cycle counter */
    res_t r = measure(lat_add, NULL, 100);
    double ghz_wall = 1.0 / r.ns; /* adds per ns, if one add per cycle */
    printf("\n## Clock\n");
    printf("dependent add: %.4f ns each -> %.1f MHz if 1 cycle/add\n", r.ns, ghz_wall * 1000);
    if (r.cyc > 0)
        printf("cycle counter: %.4f cycles per add; counter/wall = %.1f MHz\n", r.cyc, r.cyc / r.ns * 1000);
    double ghz = r.cyc > 0 ? r.cyc / r.ns : ghz_wall;
    printf("ns spread across trials: %.4f-%.4f (%.2f%%)\n", r.ns_min, r.ns_max, (r.ns_max - r.ns_min) / r.ns * 100);

    section_instructions(ghz);
    section_allcores();
    section_latency(ghz);
    section_bandwidth();
    section_coherence();
    section_kernel();
    return 0;
}
