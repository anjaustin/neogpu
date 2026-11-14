/* test_mailbox: hs_mailbox on the GPU (P0-5, docs/audit/HARDWARE_HANDOFF_P0.md).
 *   tests/test_mailbox            correctness: every output word checked
 *   tests/test_mailbox bench [rounds=3]
 *       latency, median per size: mailbox (warm leases), mailbox cold (a stop
 *       before each request: what one job with this program costs), the
 *       shipped gather as one job (hs_v3d_dispatch), the CPU (1 thread, and 4
 *       threads with a spinning pool; the fastest NEON chain found).
 * Run inside the GPU lock. Output: JSON lines; exit 1 on any wrong word.
 */
#define _GNU_SOURCE
#include <arm_neon.h>
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "hs_mailbox.h"
#include "hs_qpu.h"
#include "hs_qpu_kernels.h"
#include "hs_v3d.h"

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
static void die(const char* w, int e) {
    fprintf(stderr, "%s: %s\n", w, strerror(e < 0 ? -e : e));
    exit(1);
}
static uint32_t rng = 12345;
static uint32_t rnd(void) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}

#define TABLE_N (1u << 20)
#define MAXE (1u << 18)
static hs_v3d dev;
static hs_v3d_bo bt, bi, bo, bx, by; /* table, idx, out, chain in, chain out */
static int failures;

static float chain_ref(float x, uint32_t k, float c1, float c0) {
    volatile float y = x; /* no contraction: the GPU rounds after fmul and after fadd */
    for (uint32_t i = 0; i < k; i++) {
        float m = y * c1;
        y = m + c0;
    }
    return y;
}

static void check(const char* what, uint32_t batches, uint32_t n, int bad, int rc) {
    printf("{\"check\":\"%s\",\"batches\":%u,\"elements\":%u,\"rc\":%d,\"wrong\":%d}\n", what, batches, n, rc, bad);
    if (bad || rc) failures++;
}

static void correctness(void) {
    uint32_t* T = (uint32_t*)bt.map;
    for (uint32_t i = 0; i < TABLE_N; i++) T[i] = i * 2654435761u ^ 0x5A5A5A5Au;
    const hs_v3d_bo* bos[5] = {&bt, &bi, &bo, &bx, &by};
    for (uint32_t batches = 1; batches <= 8; batches += 7) {
        hs_mailbox mb;
        hs_mailbox_cfg cfg = {.batches = batches, .bos = bos, .nbos = 5};
        int e = hs_mailbox_start(&mb, &dev, &cfg);
        if (e) die("hs_mailbox_start", e);
        /* echo */
        int rc = 0;
        for (int i = 0; i < 10000 && !rc; i++) rc = hs_mailbox_echo(&mb);
        check("echo x10000", batches, 0, 0, rc);
        /* gather, new indices every request (op-2 reload), sizes up to 64K */
        static const uint32_t sizes[] = {128, 1024, 8192, 65536};
        for (int s = 0; s < 4; s++) {
            uint32_t n = sizes[s];
            if (n % (16 * batches) || n > 16 * batches * HS_MB_MAX_GATHER_CHUNKS) continue;
            int bad = 0;
            for (int rep = 0; rep < 20; rep++) {
                uint32_t* ix = (uint32_t*)bi.map;
                uint32_t* out = (uint32_t*)bo.map;
                for (uint32_t i = 0; i < n; i++) {
                    ix[i] = rnd() % TABLE_N;
                    out[i] = 0xDEADBEEF;
                }
                rc = hs_mailbox_gather(&mb, bi.gpu, bt.gpu, TABLE_N, bo.gpu, n, 0);
                if (rc) break;
                for (uint32_t i = 0; i < n; i++) bad += out[i] != T[ix[i]];
            }
            check("gather x20 (new idx each)", batches, n, bad, rc);
        }
        /* table changed by the CPU while the leases run: table_changes = 1 */
        {
            uint32_t n = 1024, *ix = (uint32_t*)bi.map, *out = (uint32_t*)bo.map;
            int bad = 0;
            for (int rep = 0; rep < 20 && !rc; rep++) {
                for (uint32_t i = 0; i < n; i++) ix[i] = i * 7 % 4096; /* a small, hot region */
                for (uint32_t i = 0; i < 4096; i++) T[i] = rnd();
                rc = hs_mailbox_gather(&mb, bi.gpu, bt.gpu, TABLE_N, bo.gpu, n, 1);
                for (uint32_t i = 0; i < n; i++) bad += out[i] != T[ix[i]];
            }
            check("gather x20, table rewritten each time, table_changes=1", batches, n, bad, rc);
            for (uint32_t i = 0; i < 4096; i++) T[i] = i * 2654435761u ^ 0x5A5A5A5Au;
        }
        /* chain, bit-exact against the uncontracted CPU reference */
        static const uint32_t ks[] = {4, 64, 1024};
        for (int kk = 0; kk < 3; kk++) {
            uint32_t n = 4096, k = ks[kk];
            float *x = (float*)bx.map, *y = (float*)by.map;
            int bad = 0;
            for (int rep = 0; rep < 5 && !rc; rep++) {
                for (uint32_t i = 0; i < n; i++) {
                    x[i] = (float)(rnd() % 1000000) / 1e6f;
                    y[i] = -1.0f;
                }
                float c1 = 0.999f, c0 = 0.001f * (float)(rep + 1);
                rc = hs_mailbox_chain(&mb, bx.gpu, by.gpu, n, k, c1, c0);
                for (uint32_t i = 0; i < n; i++) {
                    float r = chain_ref(x[i], k, c1, c0);
                    bad += memcmp(&r, &y[i], 4) != 0;
                }
            }
            char w[64];
            snprintf(w, sizeof w, "chain k=%u x5, bit-exact", k);
            check(w, batches, n, bad, rc);
        }
        /* pipelined: 64 requests in flight, then wait for the last */
        {
            int64_t last = 0;
            for (int i = 0; i < 64 && last >= 0; i++) last = hs_mailbox_send(&mb, HS_MB_OP_ECHO, NULL);
            rc = last < 0 ? (int)last : hs_mailbox_wait(&mb, (uint32_t)last, 1000000000ull);
            check("64 echo in flight", batches, 0, 0, rc);
        }
        /* stop, then use again */
        rc = hs_mailbox_stop(&mb);
        int rc2 = hs_mailbox_echo(&mb);
        uint32_t* ix = (uint32_t*)bi.map;
        uint32_t* out = (uint32_t*)bo.map;
        int bad = 0;
        for (uint32_t i = 0; i < 1024; i++) ix[i] = rnd() % TABLE_N;
        int rc3 = hs_mailbox_gather(&mb, bi.gpu, bt.gpu, TABLE_N, bo.gpu, 1024, 0);
        for (uint32_t i = 0; i < 1024; i++) bad += out[i] != T[ix[i]];
        check("stop, then echo and gather", batches, 1024, bad, rc ? rc : rc2 ? rc2 : rc3);
        /* across lease hand-overs: 3 s of spaced requests */
        {
            uint32_t l0 = hs_mailbox_leases_done(&mb, 0);
            uint64_t t_end = now_ns() + 3000000000ull;
            int reqs = 0;
            bad = 0;
            rc = 0;
            while (now_ns() < t_end && !rc) {
                for (uint32_t i = 0; i < 256; i++) ix[i] = rnd() % TABLE_N;
                rc = hs_mailbox_gather(&mb, bi.gpu, bt.gpu, TABLE_N, bo.gpu, 256 - 256 % (16 * batches), 0);
                for (uint32_t i = 0; i < 256 - 256 % (16 * batches); i++) bad += out[i] != T[ix[i]];
                reqs++;
                uint64_t g = now_ns() + 200000;
                while (now_ns() < g) {
                }
            }
            uint32_t leases = hs_mailbox_leases_done(&mb, 0) - l0;
            printf("{\"check\":\"3 s of gathers every 0.2 ms\",\"batches\":%u,\"requests\":%d,\"leases\":%u,"
                   "\"handovers_seen\":%llu,\"rc\":%d,\"wrong\":%d}\n",
                   batches, reqs, leases, (unsigned long long)mb.handovers_seen, rc, bad);
            if (bad || rc || leases < 10) failures++;
        }
        /* argument refusals */
        int r1 = hs_mailbox_gather(&mb, bi.gpu, bt.gpu, TABLE_N, bo.gpu, 8, 0);
        int r2 = hs_mailbox_chain(&mb, bx.gpu, by.gpu, 4096, 6, 1.f, 0.f);
        int r3 = hs_mailbox_gather(&mb, bi.gpu + 2, bt.gpu, TABLE_N, bo.gpu, 1024, 0);
        check("refuses bad sizes/alignment", batches, 0, !(r1 == -EINVAL && r2 == -EINVAL && r3 == -EINVAL), 0);
        hs_mailbox_free(&mb);
    }
}

/* ---- CPU references for timing ---- */
static void cpu_gather(const uint32_t* T, const uint32_t* ix, uint32_t* out, uint32_t a, uint32_t b) {
    for (uint32_t i = a; i < b; i++) out[i] = T[ix[i]];
}
/* The fastest CPU chain found (tools/p0_5_chain_cpu): multiply and add kept
 * separate, 8 vectors. The fused form (vfmaq) is slower on the Cortex-A72
 * (5.1 against 7.2 GFLOPS per core) and rounds differently from the GPU. */
__attribute__((noinline)) static void cpu_chain_neon(const float* x, float* y, uint32_t a, uint32_t b, uint32_t k,
                                                     float c1, float c0) {
    float32x4_t vc1 = vdupq_n_f32(c1), vc0 = vdupq_n_f32(c0);
    for (uint32_t i = a; i < b; i += 32) {
        float32x4_t v[8];
        for (int j = 0; j < 8; j++) v[j] = vld1q_f32(x + i + 4 * j);
        for (uint32_t t = 0; t < k; t++) {
            _Pragma("GCC unroll 8") for (int j = 0; j < 8; j++) v[j] = vaddq_f32(vmulq_f32(v[j], vc1), vc0);
        }
        for (int j = 0; j < 8; j++) vst1q_f32(y + i + 4 * j, v[j]);
    }
}
/* a 4-thread spinning pool: the caller is thread 0 */
static struct {
    _Atomic uint32_t gen, done;
    int kind; /* 0 gather, 1 chain */
    const void* a;
    const void* b;
    void* c;
    uint32_t n, k;
    float c1, c0;
} pool;
static void pool_part(int t) {
    uint32_t per = pool.n / 4, lo = per * (uint32_t)t, hi = t == 3 ? pool.n : lo + per;
    if (pool.kind == 0)
        cpu_gather(pool.a, pool.b, pool.c, lo, hi);
    else
        cpu_chain_neon(pool.a, pool.c, lo, hi, pool.k, pool.c1, pool.c0);
}
static void* pool_worker(void* arg) {
    int t = (int)(intptr_t)arg;
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(t == 1 ? 0 : t == 2 ? 1 : 3, &cs);
    sched_setaffinity(0, sizeof cs, &cs);
    uint32_t seen = 0;
    for (;;) {
        uint32_t g;
        while ((g = atomic_load_explicit(&pool.gen, memory_order_acquire)) == seen) {
        }
        seen = g;
        pool_part(t);
        atomic_fetch_add_explicit(&pool.done, 1, memory_order_release);
    }
    return NULL;
}
static void pool_run(void) {
    atomic_store(&pool.done, 0);
    atomic_fetch_add_explicit(&pool.gen, 1, memory_order_release);
    pool_part(0);
    while (atomic_load_explicit(&pool.done, memory_order_acquire) != 3) {
    }
}

static int cmpd(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}
#define REPS 101
static double med(double* v, int n) {
    qsort(v, n, sizeof(double), cmpd);
    return v[n / 2];
}

static void bench(int rounds) {
    pthread_t th[3];
    for (int t = 1; t <= 3; t++) pthread_create(&th[t - 1], NULL, pool_worker, (void*)(intptr_t)t);
    uint32_t* T = (uint32_t*)bt.map;
    for (uint32_t i = 0; i < TABLE_N; i++) T[i] = i * 2654435761u;
    uint32_t* ix = (uint32_t*)bi.map;
    for (uint32_t i = 0; i < MAXE; i++) ix[i] = rnd() % TABLE_N;
    /* CPU copies (the BO mappings are uncached: the CPU would be measured on
     * uncached memory otherwise) */
    uint32_t* cT = malloc(TABLE_N * 4);
    uint32_t* cI = malloc(MAXE * 4);
    uint32_t* cO = malloc(MAXE * 4);
    float* cx = malloc(MAXE * 4);
    float* cy = malloc(MAXE * 4);
    memcpy(cT, T, TABLE_N * 4);
    memcpy(cI, ix, MAXE * 4);
    for (uint32_t i = 0; i < MAXE; i++) cx[i] = ((float*)bx.map)[i] = (float)(i % 1000) / 1000.f;
    const hs_v3d_bo* bos[5] = {&bt, &bi, &bo, &bx, &by};
    /* the shipped gather as one job */
    hs_v3d_bo code, unif;
    if (hs_v3d_bo_new(&dev, &code, 64 << 10) || hs_v3d_bo_new(&dev, &unif, 4096)) die("bo", EIO);
    static uint64_t buf[4096];
    uint32_t g_code[2];
    for (int v = 0; v < 2; v++) { /* v0: 1 batch k=8 (any multiple of 128); v1: 8 batches k=8 */
        hs_qpu_prog p;
        hs_qpu_prog_init(&p, buf, 4096);
        int n = hs_qpu_kernel_gather(&p, v ? 8 : 1, 8);
        if (n < 0) die("gather build", EIO);
        g_code[v] = hs_v3d_load(&code, (size_t)v * 32768, buf, (uint32_t)n);
    }
    uint32_t jh[5] = {code.handle, unif.handle, bt.handle, bi.handle, bo.handle};
    static const uint32_t gsz[] = {128, 1024, 8192, 65536, 262144};
    static const uint32_t csz[] = {256, 4096, 65536};
    static const uint32_t cks[] = {16, 256, 4096};
    for (int round = 0; round < rounds; round++) {
        for (uint32_t batches = 1; batches <= 8; batches += 7) {
            hs_mailbox mb;
            hs_mailbox_cfg cfg = {.batches = batches, .bos = bos, .nbos = 5};
            if (hs_mailbox_start(&mb, &dev, &cfg)) die("start", EIO);
            double v[REPS];
            for (int i = 0; i < REPS; i++) {
                uint64_t a = now_ns();
                hs_mailbox_echo(&mb);
                v[i] = (now_ns() - a) / 1e3;
            }
            printf("{\"bench\":\"echo\",\"round\":%d,\"batches\":%u,\"mailbox_us\":%.2f}\n", round, batches,
                   med(v, REPS));
            for (int s = 0; s < 5; s++) {
                uint32_t n = gsz[s];
                if (n % (16 * batches) || n > 16 * batches * HS_MB_MAX_GATHER_CHUNKS) continue;
                double w[REPS], c[REPS];
                for (int i = 0; i < REPS; i++) {
                    uint64_t a = now_ns();
                    int rc = hs_mailbox_gather(&mb, bi.gpu, bt.gpu, TABLE_N, bo.gpu, n, 0);
                    w[i] = (now_ns() - a) / 1e3;
                    if (rc) die("gather", rc);
                }
                int bad = 0;
                for (uint32_t i = 0; i < n; i++) bad += ((uint32_t*)bo.map)[i] != T[ix[i]];
                for (int i = 0; i < 21; i++) {
                    hs_mailbox_stop(&mb);
                    uint64_t a = now_ns();
                    hs_mailbox_gather(&mb, bi.gpu, bt.gpu, TABLE_N, bo.gpu, n, 0);
                    c[i] = (now_ns() - a) / 1e3;
                }
                printf("{\"bench\":\"gather\",\"round\":%d,\"batches\":%u,\"elements\":%u,\"mailbox_us\":%.2f,"
                       "\"mailbox_cold_us\":%.2f,\"wrong\":%d}\n",
                       round, batches, n, med(w, REPS), med(c, 21), bad);
                if (bad) failures++;
            }
            for (int s = 0; s < 3; s++)
                for (int kk = 0; kk < 3; kk++) {
                    uint32_t n = csz[s], k = cks[kk];
                    if (n % (32 * batches) || (uint64_t)(n / (32 * batches)) * (k / 4 + 4) + 4 > 65536) continue;
                    double w[REPS];
                    for (int i = 0; i < REPS; i++) {
                        uint64_t a = now_ns();
                        int rc = hs_mailbox_chain(&mb, bx.gpu, by.gpu, n, k, 0.999f, 0.001f);
                        w[i] = (now_ns() - a) / 1e3;
                        if (rc) die("chain", rc);
                    }
                    printf("{\"bench\":\"chain\",\"round\":%d,\"batches\":%u,\"elements\":%u,\"k\":%u,"
                           "\"mailbox_us\":%.2f,\"gflops\":%.2f}\n",
                           round, batches, n, k, med(w, REPS), 2.0 * n * k / (med(w, REPS) * 1e3));
                }
            hs_mailbox_free(&mb);
        }
        /* one job per request (no mailbox running) */
        for (int s = 0; s < 5; s++) {
            uint32_t n = gsz[s];
            int v8 = n % 1024 == 0;
            uint32_t* u = (uint32_t*)unif.map;
            if (hs_qpu_gather_uniforms(u, n, v8 ? 8 : 1, 8, bi.gpu, bt.gpu, bo.gpu)) continue;
            double w[REPS];
            for (int i = 0; i < REPS; i++) {
                ((uint32_t*)bo.map)[0] = 0;
                uint64_t a = now_ns();
                hs_v3d_dispatch_desc d = {g_code[v8], unif.gpu, v8 ? 8 : 1, jh, 5, 0};
                int e = hs_v3d_dispatch(&dev, &d);
                if (!e) e = hs_v3d_bo_wait(&dev, &bo, 2000000000ull);
                w[i] = (now_ns() - a) / 1e3;
                if (e) die("job", e);
            }
            int bad = 0;
            for (uint32_t i = 0; i < n; i++) bad += ((uint32_t*)bo.map)[i] != T[ix[i]];
            printf("{\"bench\":\"gather_job\",\"round\":%d,\"elements\":%u,\"batches\":%d,\"job_us\":%.2f,\"wrong\":%d}\n",
                   round, n, v8 ? 8 : 1, med(w, REPS), bad);
            if (bad) failures++;
        }
        /* CPU */
        for (int s = 0; s < 5; s++) {
            uint32_t n = gsz[s];
            double w1[REPS], w4[REPS];
            for (int i = 0; i < REPS; i++) {
                uint64_t a = now_ns();
                cpu_gather(cT, cI, cO, 0, n);
                w1[i] = (now_ns() - a) / 1e3;
                pool.kind = 0, pool.a = cT, pool.b = cI, pool.c = cO, pool.n = n;
                a = now_ns();
                pool_run();
                w4[i] = (now_ns() - a) / 1e3;
            }
            printf("{\"bench\":\"gather_cpu\",\"round\":%d,\"elements\":%u,\"cpu1_us\":%.2f,\"cpu4_us\":%.2f}\n", round,
                   n, med(w1, REPS), med(w4, REPS));
        }
        for (int s = 0; s < 3; s++)
            for (int kk = 0; kk < 3; kk++) {
                uint32_t n = csz[s], k = cks[kk];
                double w1[REPS], w4[REPS];
                int reps = (uint64_t)n * k > 50000000ull ? 11 : REPS;
                for (int i = 0; i < reps; i++) {
                    uint64_t a = now_ns();
                    cpu_chain_neon(cx, cy, 0, n, k, 0.999f, 0.001f);
                    w1[i] = (now_ns() - a) / 1e3;
                    pool.kind = 1, pool.a = cx, pool.c = cy, pool.n = n, pool.k = k, pool.c1 = 0.999f,
                    pool.c0 = 0.001f;
                    a = now_ns();
                    pool_run();
                    w4[i] = (now_ns() - a) / 1e3;
                }
                double m1 = med(w1, reps), m4 = med(w4, reps);
                int bad = 0;
                for (uint32_t i = 0; i < n; i += n / 64) {
                    float r = chain_ref(cx[i], k, 0.999f, 0.001f);
                    bad += memcmp(&r, &cy[i], 4) != 0;
                }
                printf("{\"bench\":\"chain_cpu\",\"round\":%d,\"elements\":%u,\"k\":%u,\"cpu1_us\":%.2f,\"cpu4_us\":%.2f,"
                       "\"cpu4_gflops\":%.2f,\"wrong\":%d}\n",
                       round, n, k, m1, m4, 2.0 * n * k / (m4 * 1e3), bad);
                if (bad) failures++;
            }
    }
}

int main(int argc, char** argv) {
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(2, &cs);
    sched_setaffinity(0, sizeof cs, &cs);
    setvbuf(stdout, NULL, _IOLBF, 0);
    int e = hs_v3d_open(&dev, NULL);
    if (e) die("hs_v3d_open", e);
    if ((e = hs_v3d_bo_new(&dev, &bt, TABLE_N * 4)) || (e = hs_v3d_bo_new(&dev, &bi, MAXE * 4)) ||
        (e = hs_v3d_bo_new(&dev, &bo, MAXE * 4)) || (e = hs_v3d_bo_new(&dev, &bx, MAXE * 4)) ||
        (e = hs_v3d_bo_new(&dev, &by, MAXE * 4)))
        die("bo", e);
    if (argc > 1 && !strcmp(argv[1], "bench"))
        bench(argc > 2 ? atoi(argv[2]) : 3);
    else
        correctness();
    printf("{\"test\":\"mailbox\",\"failures\":%d,\"passed\":%s}\n", failures, failures ? "false" : "true");
    return failures ? 1 : 0;
}
