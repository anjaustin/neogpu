/* hwprobe_qpu: run QPU programs on the Raspberry Pi 4 V3D through the kernel's
 * v3d driver directly (compute shader dispatch, no Mesa, no libdrm) and
 * measure: GPU clock, QPU ALU throughput, TMU read/write/copy bandwidth (1 and
 * 8 QPUs), dispatch latency and job rate, and GPU vs CPU memory bandwidth
 * when both run at once. Timed jobs in the ALU, memory and gather sections
 * carry the GPU's own performance counters (perfmon); the dispatch and
 * concurrency sections time bare jobs. Every data result is checked, every
 * word, and timed memory/gather/concurrency jobs follow a 1 ms GPU warm-up
 * (the clock drops after >= 200 ms idle; HWPROBE_WARM_MS).
 *
 * Programs are raw QPU instruction words (little-endian u64 files), produced
 * by tools/hwprobe_qpu_progs.py (make hwprobe-progs writes them to build/qpu).
 *
 * Usage: tools/hwprobe_qpu PROGDIR [cpu=0] [sections=amdcgG]
 *   a ALU, m TMU memory, d dispatch, c GPU+CPU bandwidth, g gather (CPU vs
 *   GPU), G random gather on CPU and GPU at once.
 * Debug/bisection environment: HWPROBE_MEMSEL (s/m/c), HWPROBE_MEMSET_SUFFIX
 * (_ref reproduces the hang), HWPROBE_DONE, HWPROBE_FLAGKB, HWPROBE_GDBG.
 */
#define _GNU_SOURCE
#include <arm_neon.h>
#include <drm/drm.h>
#include <drm/v3d_drm.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define TRIALS 9

static int fd;
static const char* progdir;

static inline uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
static int cmpd(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}
static double median(double* v, int n) {
    qsort(v, (size_t)n, sizeof(double), cmpd);
    return v[n / 2];
}
static int xioctl(unsigned long req, void* arg) {
    int r;
    do r = ioctl(fd, req, arg);
    while (r == -1 && (errno == EINTR || errno == EAGAIN));
    return r;
}
static void die(const char* m) {
    perror(m);
    exit(1);
}
static void pin(int cpu) {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    sched_setaffinity(0, sizeof(s), &s);
}

typedef struct {
    uint32_t handle, gpu;
    size_t size;
    uint8_t* map;
} bo_t;
static void bo_new(bo_t* b, size_t size) {
    struct drm_v3d_create_bo c = {.size = (uint32_t)size};
    if (xioctl(DRM_IOCTL_V3D_CREATE_BO, &c)) die("CREATE_BO");
    struct drm_v3d_mmap_bo m = {.handle = c.handle};
    if (xioctl(DRM_IOCTL_V3D_MMAP_BO, &m)) die("MMAP_BO");
    b->map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)m.offset);
    if (b->map == MAP_FAILED) die("mmap");
    b->handle = c.handle;
    b->gpu = c.offset;
    b->size = size;
}
static void bo_wait(const bo_t* b) {
    struct drm_v3d_wait_bo w = {.handle = b->handle, .timeout_ns = 10000000000ull};
    if (xioctl(DRM_IOCTL_V3D_WAIT_BO, &w)) die("WAIT_BO");
}

/* code: programs at 4 KiB-aligned offsets; unif: uniform streams; data: buffers */
static bo_t code, unif, data;
static uint32_t handles[3];
typedef struct {
    char name[48];
    uint32_t off; /* within code BO */
    uint32_t n;
} prog_t;
static uint32_t code_pos;
static prog_t loaded[128];
static unsigned nloaded;
/* Load a program once (cached by name). The file must be a non-empty whole
 * number of 64-bit instructions that fits in the code BO; anything else is
 * fatal, so the GPU never runs a truncated or empty program. */
static prog_t load(const char* name) {
    for (unsigned i = 0; i < nloaded; i++)
        if (!strcmp(loaded[i].name, name)) return loaded[i];
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", progdir, name);
    FILE* f = fopen(path, "rb");
    if (!f) die(path);
    if (fseek(f, 0, SEEK_END)) die(path);
    long size = ftell(f);
    rewind(f);
    if (size <= 0 || size % 8 || (size_t)size > code.size - code_pos || nloaded == 128 ||
        strlen(name) >= sizeof(loaded[0].name)) {
        fprintf(stderr, "%s: %ld bytes; need a non-empty multiple of 8 within the %zu bytes left\n", path, size,
                code.size - code_pos);
        exit(1);
    }
    prog_t p;
    snprintf(p.name, sizeof(p.name), "%s", name);
    p.off = code_pos;
    p.n = (uint32_t)(size / 8);
    if (fread(code.map + code_pos, 8, p.n, f) != p.n) die(path);
    fclose(f);
    code_pos += (p.n * 8 + 4095) & ~4095u;
    loaded[nloaded++] = p;
    return p;
}

/* perfmon: GPU counters attached to each job */
static const struct {
    uint8_t id;
    const char* name;
} CTR[] = {
    {32, "cycles"},          {86, "compute_active"}, {16, "qpu_valid_instr"}, {13, "qpu_idle"},
    {17, "qpu_wait_tmu"},    {75, "mem_reads"},      {71, "mem_writes"},      {31, "l2t_miss"},
};
#define NCTR (sizeof(CTR) / sizeof(CTR[0]))
static uint32_t perfmon_new(void) {
    struct drm_v3d_perfmon_create c;
    memset(&c, 0, sizeof(c));
    c.ncounters = NCTR;
    for (size_t i = 0; i < NCTR; i++) c.counters[i] = CTR[i].id;
    if (xioctl(DRM_IOCTL_V3D_PERFMON_CREATE, &c)) die("PERFMON_CREATE");
    return c.id;
}
static void perfmon_read(uint32_t id, uint64_t* v) {
    struct drm_v3d_perfmon_get_values g = {.id = id, .values_ptr = (uint64_t)(uintptr_t)v};
    if (xioctl(DRM_IOCTL_V3D_PERFMON_GET_VALUES, &g)) die("PERFMON_GET_VALUES");
}
static void perfmon_free(uint32_t id) {
    struct drm_v3d_perfmon_destroy d = {.id = id};
    xioctl(DRM_IOCTL_V3D_PERFMON_DESTROY, &d);
}

/* One compute dispatch: nqpu batches of one 16-lane workgroup each (the
 * layout the reference programs expect). */
static void submit(const prog_t* p, uint32_t unif_off, int nqpu, uint32_t perfmon) {
    const uint32_t wg_size = 16, wgs_per_sg = 16;
    struct drm_v3d_submit_csd s;
    memset(&s, 0, sizeof(s));
    s.cfg[0] = 1u << 16;
    s.cfg[1] = 1u << 16;
    s.cfg[2] = 1u << 16;
    s.cfg[3] = ((((wgs_per_sg * wg_size + 15) / 16) - 1) << 12) | ((wgs_per_sg & 0xf) << 8) | (wg_size & 0xff);
    s.cfg[4] = (uint32_t)nqpu - 1;
    s.cfg[5] = code.gpu + p->off;
    s.cfg[6] = unif.gpu + unif_off;
    s.bo_handles = (uint64_t)(uintptr_t)handles;
    s.bo_handle_count = 3;
    s.perfmon_id = perfmon;
    if (xioctl(DRM_IOCTL_V3D_SUBMIT_CSD, &s)) die("SUBMIT_CSD");
}
/* run once with counters; returns wall ns from submit to WAIT_BO return */
static double run(const prog_t* p, uint32_t unif_off, int nqpu, uint64_t* ctr) {
    uint32_t pm = ctr ? perfmon_new() : 0;
    uint64_t t0 = now_ns();
    submit(p, unif_off, nqpu, pm);
    bo_wait(&data);
    bo_wait(&code);
    double ns = (double)(now_ns() - t0);
    if (ctr) {
        perfmon_read(pm, ctr);
        perfmon_free(pm);
    }
    return ns;
}

static uint32_t* U(uint32_t off) { return (uint32_t*)(unif.map + off); }

/* ---------------- sections ---------------- */
static void print_ctr(const uint64_t* v) {
    for (size_t i = 0; i < NCTR; i++) printf(" %s=%llu", CTR[i].name, (unsigned long long)v[i]);
    printf("\n");
}

static double g_mhz;
static void section_alu(void) {
    prog_t p = load("alu64");
    printf("## QPU ALU: loop of 64 dual-issue fadd+fmul (16 lanes each), accumulators only\n");
    for (int nq = 1; nq <= 8; nq *= 8) {
        uint32_t loops = 200000;
        *U(0) = loops;
        double ns[TRIALS], mhz[TRIALS], cpi[TRIALS];
        uint64_t v[NCTR];
        for (int t = 0; t < TRIALS; t++) {
            ns[t] = run(&p, 0, nq, v);
            mhz[t] = (double)v[0] / ns[t] * 1e3;
            /* instructions per QPU per loop: 64 body + sub + branch + 3 delay slots = 69 (the qpu_valid_instr counter agrees) */
            double instr = (double)loops * 69.0;
            cpi[t] = (double)v[1] / instr; /* compute-active cycles per loop instruction per QPU */
        }
        double m = median(ns, TRIALS);
        double flops = (double)loops * 64.0 * 32.0 * nq;
        printf("%d QPU: %.2f ms, %.2f GFLOP/s (fadd+fmul, %d lanes), GPU clock by cycle counter %.1f MHz, "
               "compute-active cycles per instruction %.3f\n",
               nq, m / 1e6, flops / m, 16 * nq, median(mhz, TRIALS), median(cpi, TRIALS));
        printf("   counters (last run):");
        print_ctr(v);
        if (nq == 8) g_mhz = median(mhz, TRIALS);
    }
}

/* Keep the GPU busy for ~ms with the ALU loop (8 QPUs) before a timed job.
 * The V3D clock drops when the GPU idles (measured: 749.6 MHz after a short
 * idle, a 493.4 MHz average over a 29 ms job after 1 s idle; same cycle
 * count both times). A loop pass is 69 instructions x 4 cycles. */
/* default 1 ms (measured enough); HWPROBE_WARM_MS overrides, 0 disables */
static double warm_ms(void) { return getenv("HWPROBE_WARM_MS") ? atof(getenv("HWPROBE_WARM_MS")) : 1.0; }
static void gpu_warm(double ms) {
    if (ms <= 0) return;
    prog_t a = load("alu64");
    *U(0) = (uint32_t)(ms * 1e-3 * 749.6e6 / (69.0 * 4.0)) + 1;
    run(&a, 0, 8, NULL);
}

static void section_mem(void) {
    printf("\n## TMU bandwidth (every result checked; memset ends with tmuwt, see HWPROBE_MEMSET_SUFFIX)\n");
    printf("%-26s %10s %10s %12s %14s %14s\n", "program", "MiB", "ms", "GB/s wall", "mem reads M", "mem writes M");
    /* HWPROBE_MEMSEL picks programs: s=summation m=memset c=scopy (default all) */
    const char* sel = getenv("HWPROBE_MEMSEL") ? getenv("HWPROBE_MEMSEL") : "smc";
    const size_t sum_len = 32u << 20; /* u32 elements: 128 MiB */
    for (int nq = 8; nq >= 1 && strchr(sel, 's'); nq -= 7) {
        prog_t p = load(nq == 8 ? "sum8" : "sum1");
        uint32_t* x = (uint32_t*)data.map;
        for (size_t i = 0; i < sum_len; i++) x[i] = (uint32_t)i;
        uint32_t yoff = (uint32_t)(sum_len * 4);
        uint32_t* u = U(4096);
        int nu = 3 + 2 + 1; /* unroll_shift 2 */
        u[0] = (uint32_t)sum_len;
        u[1] = data.gpu;
        u[2] = data.gpu + yoff;
        u[3] = u[4] = 0xFCFCFCFC;
        u[5] = (uint32_t)(4 * (-nu + 3));
        double ns[TRIALS];
        uint64_t v[NCTR];
        int ok = 1;
        for (int t = 0; t < TRIALS; t++) {
            memset(data.map + yoff, 0, 16 * 8 * 4);
            gpu_warm(warm_ms());
            ns[t] = run(&p, 4096, nq, v);
            uint32_t s = 0;
            for (int i = 0; i < 16 * nq; i++) s += ((uint32_t*)(data.map + yoff))[i];
            ok &= s == (uint32_t)((uint64_t)(sum_len - 1) * sum_len / 2);
        }
        double m = median(ns, TRIALS);
        printf("%-26s %10.0f %10.2f %12.2f %14.1f %14.1f %s\n", nq == 8 ? "read: summation 8 QPU" : "read: summation 1 QPU",
               sum_len * 4 / 1048576.0, m / 1e6, sum_len * 4.0 / m, v[5] / 1e6, v[6] / 1e6,
               ok ? "sum ok" : "SUM WRONG");
        printf("   counters (last run):");
        print_ctr(v);
    }
    const size_t ms_len = 16u << 20; /* u32: 64 MiB */
    for (int nq = 8; nq >= 1 && strchr(sel, 'm'); nq -= 7) {
        /* HWPROBE_MEMSET_SUFFIX: run memset<N><suffix>: "" (tmuwt, default), _barrier, _ref (hangs the next tmuwt job) */
        char mname[64];
        snprintf(mname, sizeof(mname), "memset%d%s", nq, getenv("HWPROBE_MEMSET_SUFFIX") ? getenv("HWPROBE_MEMSET_SUFFIX") : "");
        prog_t p = load(mname);
        uint32_t* u = U(8192);
        int nu = 3 + 2 + 1; /* unroll_shift 5: 1 << (5-4) = 2 */
        u[0] = data.gpu;
        u[1] = 0x5A5A5A5A;
        u[2] = (uint32_t)ms_len;
        u[3] = u[4] = 0xFCFCFCFC;
        u[5] = (uint32_t)(4 * (-nu + 3));
        double ns[TRIALS];
        uint64_t v[NCTR];
        int ok = 1;
        for (int t = 0; t < TRIALS; t++) {
            memset(data.map, 0xA5, ms_len * 4);
            gpu_warm(warm_ms());
            ns[t] = run(&p, 8192, nq, v);
            uint32_t* x = (uint32_t*)data.map;
            for (size_t i = 0; i < ms_len; i++) ok &= x[i] == 0x5A5A5A5A; /* every word */
        }
        double m = median(ns, TRIALS);
        printf("%-26s %10.0f %10.2f %12.2f %14.1f %14.1f %s\n", nq == 8 ? "write: memset 8 QPU" : "write: memset 1 QPU",
               ms_len * 4 / 1048576.0, m / 1e6, ms_len * 4.0 / m, v[5] / 1e6, v[6] / 1e6,
               ok ? "fill ok" : "FILL WRONG");
        printf("   counters (last run):");
        print_ctr(v);
    }
    const size_t cp_len = 16u << 20; /* u32: 64 MiB each way */
    for (int nq = 8; nq >= 1 && strchr(sel, 'c'); nq -= 7) {
        prog_t p = load(nq == 8 ? "scopy8" : "scopy1");
        uint32_t* x = (uint32_t*)data.map;
        uint32_t* y = (uint32_t*)(data.map + cp_len * 4);
        for (size_t i = 0; i < cp_len; i++) x[i] = (uint32_t)i;
        uint32_t* u = U(12288);
        int nu = 3 + 1 + 1;
        u[0] = (uint32_t)cp_len;
        u[1] = data.gpu;
        u[2] = data.gpu + (uint32_t)(cp_len * 4);
        u[3] = 0xFC80FCFC;
        u[4] = (uint32_t)(4 * (-nu + 3));
        double ns[TRIALS];
        uint64_t v[NCTR];
        int ok = 1;
        for (int t = 0; t < TRIALS; t++) {
            /* sentinel: 0 would hide y[0], since x[0] == 0. HWPROBE_SCOPY_FILL
             * (a byte) overrides it, to measure whether the destination's
             * prior contents change the copy's speed. */
            memset(y, getenv("HWPROBE_SCOPY_FILL") ? atoi(getenv("HWPROBE_SCOPY_FILL")) : 0xEE, cp_len * 4);
            if (getenv("HWPROBE_SETTLE_MS")) usleep((useconds_t)atoi(getenv("HWPROBE_SETTLE_MS")) * 1000u);
            gpu_warm(warm_ms());
            ns[t] = run(&p, 12288, nq, v);
            /* every word (HWPROBE_CHECK_STRIDE samples instead, for the same experiment) */
            size_t cs = getenv("HWPROBE_CHECK_STRIDE") ? (size_t)atoi(getenv("HWPROBE_CHECK_STRIDE")) : 1;
            for (size_t i = 0; i < cp_len; i += cs ? cs : 1) ok &= y[i] == (uint32_t)i;
        }
        double m = median(ns, TRIALS);
        printf("%-26s %10.0f %10.2f %12.2f %14.1f %14.1f %s  (GB/s counts read + write)\n",
               nq == 8 ? "copy: scopy 8 QPU" : "copy: scopy 1 QPU", cp_len * 4 / 1048576.0, m / 1e6, 2 * cp_len * 4.0 / m,
               v[5] / 1e6, v[6] / 1e6, ok ? "copy ok" : "COPY WRONG");
        printf("   counters (last run):");
        print_ctr(v);
    }
}

static void section_dispatch(void) {
    printf("\n## Dispatch\n");
    prog_t pe = load("end"), pd = load(getenv("HWPROBE_DONE") ? getenv("HWPROBE_DONE") : "done");
    const int N = 2000;
    double* r = malloc(sizeof(double) * N);
    for (int nq = 1; nq <= 8; nq *= 8) {
        for (int i = 0; i < 50; i++) run(&pe, 0, nq, NULL);
        for (int i = 0; i < N; i++) r[i] = run(&pe, 0, nq, NULL);
        qsort(r, N, sizeof(double), cmpd);
        printf("empty job, %d QPU: submit -> WAIT_BO returns: p50 %.1f us  p10 %.1f  p90 %.1f  p99 %.1f  min %.1f\n", nq,
               r[N / 2] / 1e3, r[N / 10] / 1e3, r[N * 9 / 10] / 1e3, r[N * 99 / 100] / 1e3, r[0] / 1e3);
    }
    /* time until a GPU write is visible to a CPU spinning on the mapped word */
    /* HWPROBE_FLAGKB: flag offset in the data BO, KiB (default 65536 = 64 MiB) */
    uint64_t flag_kb = 65536u;
    if (getenv("HWPROBE_FLAGKB")) {
        char* end;
        errno = 0;
        flag_kb = strtoull(getenv("HWPROBE_FLAGKB"), &end, 10);
        if (errno || end == getenv("HWPROBE_FLAGKB") || *end) flag_kb = UINT64_MAX; /* not a number: refused */
    }
    if (flag_kb > (data.size >> 10) || (flag_kb << 10) + 4 > data.size) { /* bounded before shifting: no wrap */
        fprintf(stderr, "HWPROBE_FLAGKB: %llu KiB is outside the %zu-byte data BO\n", (unsigned long long)flag_kb,
                data.size);
        exit(1);
    }
    uint32_t flag_off = (uint32_t)(flag_kb << 10);
    volatile uint32_t* flag = (volatile uint32_t*)(data.map + flag_off);
    *U(16384) = data.gpu + flag_off;
    double vis[N], tot[N];
    int timeouts = 0;
    for (int i = 0; i < N; i++) {
        *flag = 0;
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        uint64_t t0 = now_ns();
        submit(&pd, 16384, 1, 0);
        uint64_t t1;
        for (;;) {
            if (*flag == 1) { t1 = now_ns(); break; }
            if (now_ns() - t0 > 100000000ull) {
                t1 = 0;
                timeouts++;
                bo_wait(&data);
                printf("   job %d: word not seen within 100 ms; after WAIT_BO it reads %u\n", i, *flag);
                break;
            }
        }
        bo_wait(&data);
        uint64_t t2 = now_ns();
        vis[i] = t1 ? (double)(t1 - t0) : 1e12;
        tot[i] = (double)(t2 - t0);
    }
    qsort(vis, N, sizeof(double), cmpd);
    qsort(tot, N, sizeof(double), cmpd);
    printf("1-word write job, 1 QPU: submit -> CPU sees the word: p50 %.1f us  p10 %.1f  p90 %.1f  p99 %.1f  min %.1f  "
           "(not seen within 100 ms: %d)\n",
           vis[N / 2] / 1e3, vis[N / 10] / 1e3, vis[N * 9 / 10] / 1e3, vis[N * 99 / 100] / 1e3, vis[0] / 1e3, timeouts);
    printf("                         submit -> WAIT_BO returns: p50 %.1f us  p99 %.1f\n", tot[N / 2] / 1e3,
           tot[N * 99 / 100] / 1e3);
    /* job rate: many submits, one wait */
    for (int nq = 1; nq <= 8; nq *= 8) {
        double rate[TRIALS];
        for (int t = 0; t < TRIALS; t++) {
            const int J = 2000;
            uint64_t t0 = now_ns();
            for (int i = 0; i < J; i++) submit(&pe, 0, nq, 0);
            bo_wait(&data);
            rate[t] = J / ((double)(now_ns() - t0) / 1e9);
        }
        printf("empty jobs back to back, %d QPU: %.0f jobs/s (%.2f us each)\n", nq, median(rate, TRIALS),
               1e6 / median(rate, TRIALS));
    }
    free(r);
}

/* CPU readers running while the GPU reads */
static atomic_int cpu_stop;
typedef struct {
    int cpu;
    double bytes, ns;
    uint8_t* buf;
} rd_t;
static void* cpu_reader(void* a) {
    rd_t* r = a;
    pin(r->cpu);
    const size_t len = 64u << 20;
    uint64_t t0 = now_ns();
    double bytes = 0;
    while (!atomic_load(&cpu_stop)) {
        char* s = (char*)r->buf;
        size_t l = len;
        __asm__ volatile("1: ldp q0,q1,[%0]\n ldp q2,q3,[%0,#32]\n ldp q4,q5,[%0,#64]\n ldp q6,q7,[%0,#96]\n"
                         " add %0,%0,#128\n subs %1,%1,#128\n b.ne 1b"
                         : "+r"(s), "+r"(l)::"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory", "cc");
        bytes += (double)len;
    }
    r->ns = (double)(now_ns() - t0);
    r->bytes = bytes;
    return NULL;
}
static void section_concurrent(void) {
    printf("\n## GPU and CPU reading DRAM at the same time (GB/s)\n");
    prog_t p = load("sum8");
    const size_t sum_len = 32u << 20;
    uint32_t* u = U(4096);
    int nu = 3 + 2 + 1; /* uniforms as in section_mem (summation, unroll_shift 2) */
    u[0] = (uint32_t)sum_len;
    u[1] = data.gpu;
    u[2] = data.gpu + (uint32_t)(sum_len * 4);
    u[3] = u[4] = 0xFCFCFCFC;
    u[5] = (uint32_t)(4 * (-nu + 3));
    /* own data, so the summation can be checked whatever ran before */
    uint32_t* x = (uint32_t*)data.map;
    for (size_t i = 0; i < sum_len; i++) x[i] = (uint32_t)i;
    uint32_t* ysum = (uint32_t*)(data.map + sum_len * 4);
    int sums_ok = 1;
/* the sum slots are cleared before every timed run (CLEAR_SUM), so a job
 * that never ran cannot pass on a sum left by an earlier section; the
 * expected total, 0xFF000000 for 32 Mi elements, is not 0 */
#define CLEAR_SUM() memset(ysum, 0, 16 * 8 * 4)
#define CHECK_SUM()                                                                                    \
    do {                                                                                               \
        uint32_t s_ = 0;                                                                               \
        for (int i_ = 0; i_ < 16 * 8; i_++) s_ += ysum[i_];                                            \
        sums_ok &= s_ == (uint32_t)((uint64_t)(sum_len - 1) * sum_len / 2);                           \
    } while (0)
    rd_t rd[3];
    for (int i = 0; i < 3; i++) {
        rd[i].buf = mmap(NULL, 64u << 20, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        madvise(rd[i].buf, 64u << 20, MADV_HUGEPAGE);
        memset(rd[i].buf, 1, 64u << 20);
    }
    printf("%-24s %12s %12s %12s\n", "CPU reader threads", "GPU alone", "GPU with CPU", "CPU total");
    for (int nthr = 1; nthr <= 3; nthr += 2) {
        double solo[5], mixed[5], cpu[5];
        for (int t = 0; t < 5; t++) {
            CLEAR_SUM();
            gpu_warm(warm_ms());
            solo[t] = sum_len * 4.0 / run(&p, 4096, 8, NULL);
            CHECK_SUM();
            atomic_store(&cpu_stop, 0);
            pthread_t id[4];
            for (int i = 0; i < nthr; i++) {
                rd[i].cpu = i + 1; /* cores 1..3; the submitting thread stays on core 0 */
                pthread_create(&id[i], NULL, cpu_reader, &rd[i]);
            }
            usleep(20000);
            double g = 0;
            const int reps = 5;
            for (int k = 0; k < reps; k++) {
                CLEAR_SUM();
                gpu_warm(warm_ms()); /* also under CPU load: the warm job is untimed */
                g += run(&p, 4096, 8, NULL);
                CHECK_SUM();
            }
            atomic_store(&cpu_stop, 1);
            double cb = 0;
            for (int i = 0; i < nthr; i++) {
                pthread_join(id[i], NULL);
                cb += rd[i].bytes / rd[i].ns;
            }
            mixed[t] = sum_len * 4.0 * reps / g;
            cpu[t] = cb;
        }
        printf("%-24d %12.2f %12.2f %12.2f %s\n", nthr, median(solo, 5), median(mixed, 5), median(cpu, 5),
               sums_ok ? "sums ok" : "SUM WRONG");
    }
    double alone[5];
    for (int t = 0; t < 5; t++) {
        atomic_store(&cpu_stop, 0);
        pthread_t id[3];
        for (int i = 0; i < 3; i++) {
            rd[i].cpu = i + 1;
            pthread_create(&id[i], NULL, cpu_reader, &rd[i]);
        }
        usleep(300000);
        atomic_store(&cpu_stop, 1);
        double cb = 0;
        for (int i = 0; i < 3; i++) {
            pthread_join(id[i], NULL);
            cb += rd[i].bytes / rd[i].ns;
        }
        alone[t] = cb;
    }
    printf("3 CPU reader threads, GPU idle: %.2f GB/s\n", median(alone, 5));
    for (int i = 0; i < 3; i++) munmap(rd[i].buf, 64u << 20);
}

/* ---------------- gather: out[i] = table[idx[i]] ---------------- */
/* Same table and indices on both sides; the CPU works on ordinary memory
 * (4K or huge pages), the GPU on its BO. Every output word is checked. */
#define G_N (4u << 20) /* elements gathered per run (16 MiB of indices) */
typedef struct {
    const uint32_t *t, *ix;
    uint32_t* o;
    size_t n;
    int cpu, pf;
    double ns;
} gjob_t;
static void gather_cpu(const uint32_t* t, const uint32_t* ix, uint32_t* o, size_t n, int pf) {
    size_t i = 0;
    if (pf)
        for (; i + 32 < n; i++) {
            __builtin_prefetch(&t[ix[i + 32]]);
            o[i] = t[ix[i]];
        }
    for (; i < n; i++) o[i] = t[ix[i]];
}
static void* gjob_main(void* a) {
    gjob_t* j = a;
    pin(j->cpu);
    uint64_t t0 = now_ns();
    gather_cpu(j->t, j->ix, j->o, j->n, j->pf);
    j->ns = (double)(now_ns() - t0);
    return NULL;
}
/* wall ns for nthr threads (cores 0..nthr-1) gathering n elements */
static double gather_cpu_mt(const uint32_t* t, const uint32_t* ix, uint32_t* o, size_t n, int nthr, int pf) {
    pthread_t id[4];
    gjob_t j[4];
    uint64_t t0 = now_ns();
    for (int k = 0; k < nthr; k++) {
        size_t a = n * k / nthr, b = n * (k + 1) / nthr;
        j[k] = (gjob_t){t, ix + a, o + a, b - a, k, pf, 0};
        pthread_create(&id[k], NULL, gjob_main, &j[k]);
    }
    for (int k = 0; k < nthr; k++) pthread_join(id[k], NULL);
    return (double)(now_ns() - t0);
}
static void* galloc(size_t size, int huge) {
    void* p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) die("mmap");
    madvise(p, size, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    memset(p, 0, size);
    return p;
}
/* consume a buffer: NEON sum, returns ns */
static double consume(const uint32_t* b, size_t n, uint32_t* sum) {
    uint32x4_t a0 = vdupq_n_u32(0), a1 = a0;
    uint64_t t0 = now_ns();
    for (size_t i = 0; i < n; i += 8) {
        a0 = vaddq_u32(a0, vld1q_u32(b + i));
        a1 = vaddq_u32(a1, vld1q_u32(b + i + 4));
    }
    double ns = (double)(now_ns() - t0);
    *sum = vaddvq_u32(vaddq_u32(a0, a1));
    return ns;
}
static uint32_t hash32(uint32_t x) { return x * 2654435761u ^ (x >> 7); }

/* ---------------- random gather on CPU and GPU at the same time ---------------- */
typedef struct {
    double done_ns;
    uint64_t t0;
    int cpu; /* core for the waiter, or -1 for any core */
} gwait_t;
/* Times GPU completion. It must not inherit the main thread's pin to core 0,
 * which a CPU gather thread also uses: then its wakeup would wait for the
 * scheduler and the GPU time would include that delay. */
static void* gpu_waiter(void* a) {
    gwait_t* w = a;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c = 0; c < 4; c++)
        if (w->cpu < 0 || w->cpu == c) CPU_SET(c, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    bo_wait(&data);
    w->done_ns = (double)(now_ns() - w->t0);
    return NULL;
}
static void section_gather_concurrent(void) {
    printf("\n## Random gather, 64 MiB tables, CPU and GPU at the same time (each side gathers %u elements\n"
           "## from its own table; the CPU uses huge pages; median of 5; every output word checked)\n", G_N);
    const uint32_t idx_off = 64u << 20, out_off = 80u << 20, tmax = 16u << 20;
    uint32_t* gt = (uint32_t*)data.map;
    uint32_t* gi = (uint32_t*)(data.map + idx_off);
    uint32_t* go = (uint32_t*)(data.map + out_off);
    uint32_t* ct = galloc((size_t)tmax * 4, 1);
    uint32_t* ci = galloc((size_t)G_N * 4, 1);
    uint32_t* co = galloc((size_t)G_N * 4, 1);
    for (uint32_t i = 0; i < tmax; i++) gt[i] = ct[i] = hash32(i);
    uint64_t rs = 0x9E3779B97F4A7C15ull;
    for (uint32_t i = 0; i < G_N; i++) {
        rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
        gi[i] = (uint32_t)(rs % tmax);
        rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
        ci[i] = (uint32_t)(rs % tmax);
    }
    prog_t p = load("gather8_k1");
    uint32_t* u = U(20480);
    u[0] = G_N / (16u * 8u);
    u[1] = data.gpu + idx_off;
    u[2] = data.gpu;
    u[3] = data.gpu + out_off;
    printf("%-30s %10s %10s %12s %12s %14s %s\n", "run", "CPU ms", "GPU ms", "CPU M/s", "GPU M/s", "total M/s", "check");
    for (int mode = 0; mode < 5; mode++) {
        /* 0: GPU alone; 1: CPU 4 cores alone; 2: CPU 3 cores alone; 3: GPU + CPU 4; 4: GPU + CPU 3 */
        int gpu = mode == 0 || mode >= 3, nthr = (mode == 1 || mode == 3) ? 4 : (mode == 2 || mode == 4) ? 3 : 0;
        double cm[5], gm[5], tot[5];
        int ok = 1;
        for (int t = 0; t < 5; t++) {
            memset(go, 0xEE, (size_t)G_N * 4); /* sentinel: hash32(0) == 0 */
            memset(co, 0xEE, (size_t)G_N * 4);
            if (gpu) gpu_warm(warm_ms()); /* before the clock starts: untimed */
            gwait_t w = {0, now_ns(), nthr == 3 ? 3 : -1};
            pthread_t wt;
            if (gpu) {
                submit(&p, 20480, 8, 0);
                pthread_create(&wt, NULL, gpu_waiter, &w);
            }
            double c = nthr ? gather_cpu_mt(ct, ci, co, G_N, nthr, 0) : 0;
            if (gpu) pthread_join(wt, NULL);
            cm[t] = c;
            gm[t] = w.done_ns;
            double wall = c > w.done_ns ? c : w.done_ns;
            tot[t] = ((gpu ? G_N : 0) + (nthr ? G_N : 0)) / wall * 1e3;
            if (gpu) for (uint32_t i = 0; i < G_N; i++) ok &= go[i] == hash32(gi[i]);
            if (nthr) for (uint32_t i = 0; i < G_N; i++) ok &= co[i] == hash32(ci[i]);
        }
        static const char* NM[] = {"GPU 8 QPU alone", "CPU 4 cores alone", "CPU 3 cores alone", "GPU + CPU 4 cores",
                                   "GPU + CPU 3 cores"};
        double c = median(cm, 5), g = median(gm, 5);
        printf("%-30s %10.2f %10.2f %12.1f %12.1f %14.1f %s\n", NM[mode], c / 1e6, g / 1e6, c ? G_N / c * 1e3 : 0,
               g ? G_N / g * 1e3 : 0, median(tot, 5), ok ? "ok" : "WRONG");
    }
    munmap(ct, (size_t)tmax * 4);
    munmap(ci, (size_t)G_N * 4);
    munmap(co, (size_t)G_N * 4);
}

/* HWPROBE_GDBG=prog:passes - one 1-QPU job, prints the first 20 out words */
static void gather_debug(const char* spec) {
    char name[64];
    unsigned passes = 1;
    if (sscanf(spec, "%63[^:]:%u", name, &passes) < 1) die("HWPROBE_GDBG");
    const uint32_t idx_off = 64u << 20, out_off = 80u << 20;
    uint32_t* gt = (uint32_t*)data.map;
    uint32_t* gi = (uint32_t*)(data.map + idx_off);
    uint32_t* go = (uint32_t*)(data.map + out_off);
    for (uint32_t i = 0; i < 4096; i++) gt[i] = hash32(i);
    for (uint32_t i = 0; i < 4096; i++) gi[i] = (i * 7u) & 4095u;
    memset(go, 0xEE, 4096);
    prog_t p = load(name);
    uint32_t* u = U(20480);
    u[0] = passes;
    u[1] = data.gpu + idx_off;
    u[2] = data.gpu;
    u[3] = data.gpu + out_off;
    u[4] = 0xFFFFFFFF; /* TMU config for tmuau: per-lane regular 32-bit */
    double ns = run(&p, 20480, 1, NULL);
    printf("%s passes=%u: %.1f us; out[0..19] (idx, expected gather, got):\n", name, passes, ns / 1e3);
    for (int i = 0; i < 20; i++)
        printf("  %2d idx=%4u want=%08x got=%08x  (out gpu addr %08x)\n", i, gi[i], hash32(gi[i]), go[i],
               data.gpu + out_off + 4u * (uint32_t)i);
}

static void section_gather(void) {
    printf("\n## Gather: out[i] = table[idx[i]], %u elements of 4 B (every output word checked)\n", G_N);
    printf("rates are elements per second; 'GB/s useful' = 4 B out per element\n");
    const uint32_t tbl_off = 0, idx_off = 64u << 20, out_off = 80u << 20;
    uint32_t* gt = (uint32_t*)(data.map + tbl_off);
    uint32_t* gi = (uint32_t*)(data.map + idx_off);
    uint32_t* go = (uint32_t*)(data.map + out_off);
    const uint32_t tmax = 16u << 20; /* 64 MiB table */
    for (uint32_t i = 0; i < tmax; i++) gt[i] = hash32(i);
    uint32_t* ct[2]; uint32_t* ci[2]; uint32_t* co[2];
    for (int h = 0; h < 2; h++) {
        ct[h] = galloc((size_t)tmax * 4, h);
        ci[h] = galloc((size_t)G_N * 4, h);
        co[h] = galloc((size_t)G_N * 4, h);
        for (uint32_t i = 0; i < tmax; i++) ct[h][i] = hash32(i);
    }
    static const struct { const char* name; uint32_t entries; int seq; } CASE[] = {
        {"random, 512 KiB table", 128u << 10, 0},
        {"random, 64 MiB table", 16u << 20, 0},
        {"sequential idx, 64 MiB table", 16u << 20, 1},
    };
    for (size_t c = 0; c < sizeof(CASE) / sizeof(CASE[0]); c++) {
        uint64_t rs = 0x9E3779B97F4A7C15ull;
        for (uint32_t i = 0; i < G_N; i++) {
            rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
            uint32_t v = CASE[c].seq ? i % CASE[c].entries : (uint32_t)(rs % CASE[c].entries);
            gi[i] = v; ci[0][i] = v; ci[1][i] = v;
        }
        printf("\n### %s\n", CASE[c].name);
        printf("%-34s %10s %12s %12s %s\n", "path", "ms", "M elem/s", "GB/s useful", "check");
        /* CPU */
        for (int h = 0; h < 2; h++)
            for (int nthr = 1; nthr <= 4; nthr *= 4)
                for (int pf = 0; pf < 2; pf++) {
                    double ns[5];
                    int ok = 1;
                    for (int t = 0; t < 5; t++) { /* every trial checked; sentinel fill (hash32(0) == 0) */
                        memset(co[h], 0xEE, (size_t)G_N * 4);
                        ns[t] = gather_cpu_mt(ct[h], ci[h], co[h], G_N, nthr, pf);
                        for (uint32_t i = 0; i < G_N; i++) ok &= co[h][i] == hash32(ci[h][i]);
                    }
                    double m = median(ns, 5);
                    char nm[64];
                    snprintf(nm, sizeof(nm), "CPU %d core%s, %s%s", nthr, nthr > 1 ? "s" : "", h ? "huge" : "4K",
                             pf ? ", prefetch" : "");
                    printf("%-34s %10.2f %12.1f %12.2f %s\n", nm, m / 1e6, G_N / m * 1e3, G_N * 4.0 / m,
                           ok ? "ok" : "WRONG");
                }
        /* GPU */
        for (int nq = 1; nq <= 8; nq *= 8)
            for (int k = 1; k <= 8; k *= 2) {
                char pn[32];
                snprintf(pn, sizeof(pn), "gather%d_k%d", nq, k);
                prog_t p = load(pn);
                uint32_t* u = U(20480);
                u[0] = G_N / (16u * (uint32_t)nq * (uint32_t)k);
                u[1] = data.gpu + idx_off;
                u[2] = data.gpu + tbl_off;
                u[3] = data.gpu + out_off;
                double ns[5];
                uint64_t v[NCTR];
                int ok = 1;
                for (int t = 0; t < 5; t++) { /* every trial checked; sentinel fill (hash32(0) == 0) */
                    memset(go, 0xEE, (size_t)G_N * 4);
                    gpu_warm(warm_ms());
                    ns[t] = run(&p, 20480, nq, v);
                    for (uint32_t i = 0; i < G_N; i++) ok &= go[i] == hash32(gi[i]);
                }
                double m = median(ns, 5);
                char nm[64];
                snprintf(nm, sizeof(nm), "GPU %d QPU, %d in flight", nq, k);
                printf("%-34s %10.2f %12.1f %12.2f %s  mem_reads=%llu l2t_miss=%llu wait_tmu=%llu\n", nm, m / 1e6,
                       G_N / m * 1e3, G_N * 4.0 / m, ok ? "ok" : "WRONG", (unsigned long long)v[5],
                       (unsigned long long)v[7], (unsigned long long)v[4]);
            }
        /* what consuming the result costs: GPU output sits in an uncached BO mapping */
        uint32_t s1, s2;
        double cg[5], cc[5];
        for (int t = 0; t < 5; t++) {
            cg[t] = consume(go, G_N, &s1);
            cc[t] = consume(co[1], G_N, &s2);
        }
        printf("CPU reads the 16 MiB result once: from GPU BO %.2f ms (%.2f GB/s), from CPU memory %.2f ms (%.2f GB/s) %s\n",
               median(cg, 5) / 1e6, G_N * 4.0 / median(cg, 5), median(cc, 5) / 1e6, G_N * 4.0 / median(cc, 5),
               s1 == s2 ? "sums match" : "SUMS DIFFER");
    }
    for (int h = 0; h < 2; h++) {
        munmap(ct[h], (size_t)tmax * 4);
        munmap(ci[h], (size_t)G_N * 4);
        munmap(co[h], (size_t)G_N * 4);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s PROGDIR [cpu=0] [sections=amdcgG]\n", argv[0]);
        return 2;
    }
    progdir = argv[1];
    pin(argc > 2 ? atoi(argv[2]) : 0);
    setvbuf(stdout, NULL, _IOLBF, 0);
    fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
    if (fd < 0) die("/dev/dri/renderD128");
    bo_new(&code, 1u << 20);
    bo_new(&unif, 1u << 16);
    bo_new(&data, (128u << 20) + (1u << 20));
    handles[0] = code.handle;
    handles[1] = unif.handle;
    handles[2] = data.handle;
    printf("# hwprobe_qpu: code 0x%08x unif 0x%08x data 0x%08x (GPU addresses)\n\n", code.gpu, unif.gpu, data.gpu);
    if (getenv("HWPROBE_GDBG")) {
        gather_debug(getenv("HWPROBE_GDBG"));
        return 0;
    }
    const char* which = argc > 3 ? argv[3] : "amdcgG";
    if (strchr(which, 'a')) section_alu();
    if (strchr(which, 'm')) section_mem();
    if (strchr(which, 'd')) section_dispatch();
    if (strchr(which, 'c')) section_concurrent();
    if (strchr(which, 'g')) section_gather();
    if (strchr(which, 'G')) section_gather_concurrent();
    return 0;
}
