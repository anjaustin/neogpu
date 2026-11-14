/* p0_5_chain_cpu: P0-5 (docs/audit/HARDWARE_HANDOFF_P0.md). The fastest CPU
 * implementation found for hs_mailbox's chain op (y = x; k times:
 * y = y * c1 + c0), to compare the mailbox against. Variants: V independent
 * NEON vectors per block held in registers (V = 8, 12, 16, 24), fused
 * (vfmaq) or unfused (vmulq + vaddq, bit-exact with the GPU), on 1 and 4
 * cores. Every output is checked against a scalar reference of the same
 * fusion. Usage: tools/p0_5_chain_cpu [reps=21]   Output: JSON lines. */
#define _GNU_SOURCE
#include <arm_neon.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
#define STEP_F(v) v = vfmaq_f32(vc0, v, vc1)
#define STEP_U(v) v = vaddq_f32(vmulq_f32(v, vc1), vc0)
#define DEF(NAME, V, STEP) \
    __attribute__((noinline)) static void NAME(const float* x, float* y, uint32_t a, uint32_t b, uint32_t k, \
                                                float c1, float c0) { \
        float32x4_t vc1 = vdupq_n_f32(c1), vc0 = vdupq_n_f32(c0); \
        for (uint32_t i = a; i < b; i += 4 * V) { \
            float32x4_t v[V]; \
            for (int j = 0; j < V; j++) v[j] = vld1q_f32(x + i + 4 * j); \
            for (uint32_t t = 0; t < k; t++) { \
                _Pragma("GCC unroll 32") for (int j = 0; j < V; j++) STEP(v[j]); \
            } \
            for (int j = 0; j < V; j++) vst1q_f32(y + i + 4 * j, v[j]); \
        } \
    }
DEF(f8, 8, STEP_F)
DEF(f12, 12, STEP_F)
DEF(f16, 16, STEP_F)
DEF(f24, 24, STEP_F)
DEF(u8, 8, STEP_U)
DEF(u12, 12, STEP_U)
DEF(u16, 16, STEP_U)
DEF(u24, 24, STEP_U)
typedef void (*fn)(const float*, float*, uint32_t, uint32_t, uint32_t, float, float);
static const struct {
    const char* name;
    fn f;
    int v, fused;
} VAR[] = {{"fused, 8 vectors", f8, 8, 1},     {"fused, 12 vectors", f12, 12, 1}, {"fused, 16 vectors", f16, 16, 1},
           {"fused, 24 vectors", f24, 24, 1},   {"unfused, 8 vectors", u8, 8, 0},  {"unfused, 12 vectors", u12, 12, 0},
           {"unfused, 16 vectors", u16, 16, 0}, {"unfused, 24 vectors", u24, 24, 0}};

/* a 4-thread spinning pool: the caller is thread 0 */
static struct {
    _Atomic uint32_t gen, done;
    fn f;
    const float* x;
    float* y;
    uint32_t n, k, block;
} pool;
static void part(int t) {
    uint32_t per = pool.n / 4 / pool.block * pool.block, lo = per * (uint32_t)t, hi = t == 3 ? pool.n : lo + per;
    pool.f(pool.x, pool.y, lo, hi, pool.k, 0.999f, 0.001f);
}
static void* worker(void* arg) {
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
        part(t);
        atomic_fetch_add_explicit(&pool.done, 1, memory_order_release);
    }
    return NULL;
}
static void run4(void) {
    atomic_store(&pool.done, 0);
    atomic_fetch_add_explicit(&pool.gen, 1, memory_order_release);
    part(0);
    while (atomic_load_explicit(&pool.done, memory_order_acquire) != 3) {
    }
}
static int cmpd(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}
static double med(double* v, int n) {
    qsort(v, (size_t)n, sizeof(double), cmpd);
    return v[n / 2];
}
#pragma GCC push_options
#pragma GCC optimize("O1,no-tree-vectorize")
static uint32_t check(const float* x, const float* y, uint32_t n, uint32_t k, int fused) {
    uint32_t bad = 0;
    for (uint32_t i = 0; i < n; i++) {
        float v = x[i];
        for (uint32_t t = 0; t < k; t++) v = fused ? fmaf(v, 0.999f, 0.001f) : (float)((volatile float){v * 0.999f} + 0.001f);
        bad += memcmp(&v, &y[i], 4) != 0;
    }
    return bad;
}
#pragma GCC pop_options

int main(int argc, char** argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 21;
    if (reps < 3 || reps > 201) return 2;
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(2, &cs);
    sched_setaffinity(0, sizeof cs, &cs);
    for (intptr_t t = 1; t < 4; t++) {
        pthread_t id;
        pthread_create(&id, NULL, worker, (void*)t);
    }
    enum { MAXN = 65536 + 96 };
    float* x = aligned_alloc(64, MAXN * 4);
    float* y = aligned_alloc(64, MAXN * 4);
    for (uint32_t i = 0; i < MAXN; i++) x[i] = (float)(i % 1000) * 0.001f;
    static const uint32_t SZ[][2] = {{256, 16}, {256, 256}, {256, 4096}, {4096, 16}, {4096, 256}, {4096, 4096},
                                     {65536, 16}, {65536, 256}};
    static double w1[201], w4[201];
    for (unsigned s = 0; s < sizeof SZ / sizeof SZ[0]; s++)
        for (unsigned v = 0; v < sizeof VAR / sizeof VAR[0]; v++) {
            /* n is rounded up to whole blocks; the rate counts the elements computed */
            uint32_t blk = 4u * (uint32_t)VAR[v].v, n = (SZ[s][0] + blk - 1) / blk * blk, k = SZ[s][1];
            uint32_t bad = 0;
            for (int i = 0; i < reps; i++) {
                uint64_t a = now_ns();
                VAR[v].f(x, y, 0, n, k, 0.999f, 0.001f);
                w1[i] = (double)(now_ns() - a) / 1e3;
                if (i == 0) bad += check(x, y, n, k, VAR[v].fused);
                pool.f = VAR[v].f, pool.x = x, pool.y = y, pool.n = n, pool.k = k, pool.block = blk;
                memset(y, 0, (size_t)n * 4);
                a = now_ns();
                run4();
                w4[i] = (double)(now_ns() - a) / 1e3;
                if (i == 0) bad += check(x, y, n, k, VAR[v].fused);
            }
            double m1 = med(w1, reps), m4 = med(w4, reps);
            printf("{\"elements\":%u,\"computed\":%u,\"k\":%u,\"variant\":\"%s\",\"cpu1_us\":%.2f,\"cpu4_us\":%.2f,"
                   "\"cpu1_gflops\":%.2f,\"cpu4_gflops\":%.2f,\"bad\":%u}\n",
                   SZ[s][0], n, k, VAR[v].name, m1, m4, 2.0 * n * k / (m1 * 1e3), 2.0 * n * k / (m4 * 1e3), bad);
        }
    return 0;
}
