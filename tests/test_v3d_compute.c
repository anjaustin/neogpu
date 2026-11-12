/* test_v3d_compute: NeoGPU's own GPU path end to end on the hardware, with
 * no Mesa, no libdrm and no external assembler:
 *   hs_qpu_kernels (programs) -> hs_qpu (encoding, rules) -> hs_v3d (BOs,
 *   dispatch) -> V3D -> results checked word for word against the CPU.
 * Needs the v3d render node (the "render" group); part of make test-hardware.
 * Exit 0 only if every check passes. */
#include "hs_qpu_kernels.h"
#include "hs_v3d.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static int failures;
static void check(int ok, const char* what) {
    printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}
static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
static uint32_t hash32(uint32_t x) { return x * 2654435761u ^ (x >> 7); }

#define TABLE (1u << 20) /* entries: 4 MiB */
#define N (1u << 20)     /* elements gathered */
/* out[] is filled with this before every run, so an element the GPU never
 * writes is caught: it must not equal any expected value (checked below;
 * 0 would not do, since hash32(0) == 0) */
#define SENTINEL 0xEEEEEEEEu
static void fill_out(uint32_t* out) {
    for (uint32_t i = 0; i < N; i++) out[i] = SENTINEL;
}

int main(void) {
    hs_v3d dev;
    int e = hs_v3d_open(&dev, NULL);
    if (e) {
        printf("FAIL no v3d render node (%s); is the user in the render group?\n", strerror(-e));
        return 1;
    }
    hs_v3d_bo code, unif, data;
    int bo_ok = !hs_v3d_bo_new(&dev, &code, 64 << 10) && !hs_v3d_bo_new(&dev, &unif, 4 << 10) &&
                !hs_v3d_bo_new(&dev, &data, (TABLE + 2 * N) * 4 + 4096);
    check(bo_ok, "buffer objects created and mapped");
    if (!bo_ok) return 1; /* nothing below can run without them */
    const uint32_t handles[3] = {code.handle, unif.handle, data.handle};
    uint32_t* table = (uint32_t*)data.map;
    uint32_t* idx = table + TABLE;
    uint32_t* out = idx + N;
    volatile uint32_t* word = out + N;
    int sentinel_free = 1;
    for (uint32_t i = 0; i < TABLE; i++) {
        table[i] = hash32(i);
        sentinel_free &= table[i] != SENTINEL;
    }
    check(sentinel_free, "the sentinel fill equals no table value, so every unwritten word is caught");

    static uint64_t prog[1024];
    hs_qpu_prog p;
    size_t off = 0;

    /* 1. the smallest visible job: every lane writes 1 to one word */
    hs_qpu_prog_init(&p, prog, 1024);
    int n = hs_qpu_kernel_store_one(&p);
    check(n > 0, "store_one kernel encodes and passes the rules");
    uint32_t store_gpu = hs_v3d_load(&code, off, prog, (uint32_t)n);
    off += 4096;
    *word = 0;
    ((uint32_t*)unif.map)[0] = data.gpu + (uint32_t)((uint8_t*)word - data.map);
    hs_v3d_dispatch_desc d = {store_gpu, unif.gpu, 1, handles, 3, 0};
    e = hs_v3d_dispatch(&dev, &d);
    int w = hs_v3d_bo_wait(&dev, &data, 1000000000ull);
    check(!e && !w && *word == 1, "store_one: dispatched, completed, and the GPU's write is visible to the CPU");

    /* 2. gathers: 1 and 8 batches, 1..8 in flight, random and sequential
     * indices; every output word checked */
    for (int seq = 0; seq < 2; seq++) {
        uint64_t rs = 0x9E3779B97F4A7C15ull;
        for (uint32_t i = 0; i < N; i++) {
            rs ^= rs << 13;
            rs ^= rs >> 7;
            rs ^= rs << 17;
            idx[i] = seq ? i % TABLE : (uint32_t)(rs % TABLE);
        }
        for (int batches = 1; batches <= 8; batches *= 8)
            for (int k = 1; k <= 8; k *= 2) {
                hs_qpu_prog_init(&p, prog, 1024);
                n = hs_qpu_kernel_gather(&p, batches, k);
                uint32_t gpu = n > 0 ? hs_v3d_load(&code, off, prog, (uint32_t)n) : 0;
                uint32_t* u = (uint32_t*)unif.map + 16;
                int ue = hs_qpu_gather_uniforms(u, N, batches, k, data.gpu + TABLE * 4, data.gpu,
                                                data.gpu + (TABLE + N) * 4);
                if (ue) gpu = 0;
                fill_out(out);
                hs_v3d_dispatch_desc g = {gpu, unif.gpu + 64, (uint32_t)batches, handles, 3, 0};
                uint64_t t0 = now_ns();
                e = gpu ? hs_v3d_dispatch(&dev, &g) : -EINVAL;
                w = e ? e : hs_v3d_bo_wait(&dev, &data, 5000000000ull);
                double ms = (double)(now_ns() - t0) / 1e6;
                uint32_t bad = 0;
                for (uint32_t i = 0; i < N; i++) bad += out[i] != hash32(idx[i]);
                char what[160];
                snprintf(what, sizeof(what), "gather %s, %d batch%s, %d in flight: %u of %u words wrong, %.2f ms",
                         seq ? "sequential" : "random", batches, batches > 1 ? "es" : "", k, bad, N, ms);
                check(!e && !w && bad == 0, what);
            }
    }

    /* 3. passes = 0: the kernel's guard ends it at once, writing nothing
     * (without the guard the counter wrapped: a ~1 s hang and writes far
     * past out[]) */
    {
        hs_qpu_prog_init(&p, prog, 1024);
        n = hs_qpu_kernel_gather(&p, 8, 8);
        uint32_t gpu = n > 0 ? hs_v3d_load(&code, off, prog, (uint32_t)n) : 0;
        off += 4096;
        uint32_t* u = (uint32_t*)unif.map + 32;
        u[0] = 0;
        u[1] = data.gpu + TABLE * 4;
        u[2] = data.gpu;
        u[3] = data.gpu + (TABLE + N) * 4;
        fill_out(out);
        hs_v3d_dispatch_desc z = {gpu, unif.gpu + 128, 8, handles, 3, 0};
        uint64_t t0 = now_ns();
        e = gpu ? hs_v3d_dispatch(&dev, &z) : -EINVAL;
        w = e ? e : hs_v3d_bo_wait(&dev, &data, 2000000000ull);
        double ms = (double)(now_ns() - t0) / 1e6;
        uint32_t touched = 0;
        for (uint32_t i = 0; i < N; i++) touched += out[i] != SENTINEL;
        char what[128];
        snprintf(what, sizeof(what), "gather with passes = 0 ends at once (%.2f ms) and writes nothing (%u words touched)",
                 ms, touched);
        check(!e && !w && ms < 100.0 && touched == 0, what);
    }
    uint32_t ub[4];
    check(hs_qpu_gather_uniforms(ub, 1500, 8, 8, data.gpu, data.gpu, data.gpu) == HS_QPU_E_OPERAND &&
              hs_qpu_gather_uniforms(ub, 0, 8, 8, data.gpu, data.gpu, data.gpu) == HS_QPU_E_OPERAND &&
              hs_qpu_gather_uniforms(ub, 1024, 8, 8, data.gpu + 2, data.gpu, data.gpu) == HS_QPU_E_OPERAND &&
              hs_qpu_gather_uniforms(ub, 1024, 4, 8, data.gpu, data.gpu, data.gpu) == HS_QPU_E_OPERAND &&
              hs_qpu_gather_uniforms(ub, 1024, 8, 8, data.gpu, data.gpu, data.gpu) == 0 && ub[0] == 1,
          "gather uniforms: element counts that are 0 or not a whole pass, misaligned addresses and 4 batches are refused");

    /* 4. the runtime refuses what was never measured or cannot be right */
    hs_v3d_dispatch_desc bad = {store_gpu, unif.gpu, 9, handles, 3, 0};
    int refused = 1;
    for (uint32_t b = 0; b <= 16; b++) {
        if (b == 1 || b == 8) continue;
        bad.batches = b;
        refused &= hs_v3d_dispatch(&dev, &bad) == -EINVAL;
    }
    check(refused, "dispatch with any batch count but the measured 1 and 8 is refused (0, 2..7, 9..16)");
    bad.batches = 1;
    bad.nbos = 0;
    check(hs_v3d_dispatch(&dev, &bad) == -EINVAL, "dispatch without BOs is refused");
    check(hs_v3d_load(&code, 3, prog, 1) == 0 && hs_v3d_load(&code, code.size - 8, prog, 2) == 0,
          "misaligned or overflowing program loads are refused");

    hs_v3d_bo_free(&dev, &data);
    hs_v3d_bo_free(&dev, &unif);
    hs_v3d_bo_free(&dev, &code);
    hs_v3d_close(&dev);
    printf("%s\n", failures ? "FAILED" : "all GPU compute checks passed");
    return failures ? 1 : 0;
}
