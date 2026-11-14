/* p0_5_spin_probe: what in a compute job lets a renderer's jobs run beside it?
 * (P0-5 bedrock, docs/audit/HARDWARE_HANDOFF_P0.md.)
 *
 * Measured with tests/redteam-p0-5/rt_mailbox reset beside a renderer: while
 * a lease sat in a request that never ended, no render job finished for the
 * whole second until the kernel reset the GPU. This probe runs one compute
 * job that loops `iterations` times; the loop body is
 *   variant 0  arithmetic only
 *   variant 1  arithmetic + a thread-switch signal (no memory access)
 *   variant 2  arithmetic + one TMU load of the same word (thrsw + ldtmu)
 *   variant 3  arithmetic + one TMU store of the same word + tmuwt
 *   variant 4  `stretch` arithmetic-only iterations, then one thread switch
 *              (iterations counts the outer loop)
 * Run it alone for the job's length, and beside a renderer under
 * tests/redteam-p0-5/reset_trace.bt for what the render jobs do meanwhile.
 * Keep a job far under 1 s beside a renderer: the kernel resets it there, and
 * a job with thread switches ran 14x slower beside a renderer than alone.
 * Usage: p0_5_spin_probe <variant> <iterations> <batches> [repeats | <seconds>s] [stretch]
 *        (inside the GPU lock; "5s" repeats the job for 5 seconds)
 * Output: one JSON line per job (with the QPU thread index each batch ran
 * on), or one summary line with <seconds>s.
 */
#define _GNU_SOURCE
#include <drm/v3d_drm.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "hs_qpu.h"
#include "hs_v3d.h"

#define R(n) HS_QPU_R(n)
#define RF(n) HS_QPU_RF(n)
#define X HS_QPU_NOREG
#define I(v) hs_qpu_imm_int(v)

static void emit(hs_qpu_prog* p, hs_qpu_alu x) { hs_qpu_emit(p, &x); }
static void nop(hs_qpu_prog* p) { emit(p, hs_qpu_alu_nop()); }
static void op(hs_qpu_prog* p, int o, hs_qpu_reg d, hs_qpu_reg a, hs_qpu_reg b) { emit(p, hs_qpu_a(o, d, a, b)); }

static int build(hs_qpu_prog* p, int variant) {
    hs_qpu_prog_rules(p, 0);
    op(p, HS_QPU_A_MOV, RF(5), RF(0), X); /* rf0 at entry: the batch number in its low 4 bits */
    emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDUNIFRF, RF(0))); /* iterations */
    emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDUNIFRF, RF(1))); /* address of a scratch word */
    emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDUNIFRF, RF(4))); /* stretch (variant 4) */
    emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDUNIFRF, RF(6))); /* 256: byte offset of the thread-index words */
    op(p, HS_QPU_A_EIDX, R(1), X, X);
    op(p, HS_QPU_A_SHL, R(0), R(1), I(2));
    op(p, HS_QPU_A_ADD, RF(1), RF(1), R(0)); /* one word per lane */
    op(p, HS_QPU_A_MOV, RF(2), I(0), X);
    /* record the QPU thread index of this batch: word 64 + 16 * batch + lane */
    op(p, HS_QPU_A_BAND, R(2), RF(5), I(15));
    op(p, HS_QPU_A_SHL, R(2), R(2), I(6));
    op(p, HS_QPU_A_ADD, R(2), R(2), RF(1));
    op(p, HS_QPU_A_ADD, R(2), R(2), RF(6));
    op(p, HS_QPU_A_TIDX, R(3), X, X);
    op(p, HS_QPU_A_MOV, HS_QPU_TMUD, R(3), X);
    op(p, HS_QPU_A_MOV, HS_QPU_TMUA, R(2), X);
    op(p, HS_QPU_A_TMUWT, X, X, X);
    int l = hs_qpu_label_new(p);
    hs_qpu_label_bind(p, l);
    if (variant == 4) {
        /* rf3 = stretch; inner arithmetic-only loop, then the thread switch */
        int in = hs_qpu_label_new(p);
        op(p, HS_QPU_A_ADD, RF(3), RF(4), I(0));
        hs_qpu_label_bind(p, in);
        op(p, HS_QPU_A_ADD, RF(2), RF(2), I(1));
        hs_qpu_alu d = hs_qpu_a(HS_QPU_A_SUB, RF(3), RF(3), I(1));
        d.af = HS_QPU_PF_PUSHZ;
        emit(p, d);
        hs_qpu_branch(p, HS_QPU_B_NA0, in, 0, 0);
        nop(p), nop(p), nop(p);
        emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_THRSW, X));
        nop(p), nop(p);
    } else if (variant == 1) {
        emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_THRSW, X));
        nop(p), nop(p);
    } else if (variant == 2) {
        emit(p, hs_qpu_s(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(1), X), HS_QPU_SIG_THRSW, X));
        nop(p), nop(p);
        emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDTMU, R(2)));
    } else if (variant == 3) {
        op(p, HS_QPU_A_MOV, HS_QPU_TMUD, RF(2), X);
        op(p, HS_QPU_A_MOV, HS_QPU_TMUA, RF(1), X);
        op(p, HS_QPU_A_TMUWT, X, X, X);
    }
    op(p, HS_QPU_A_ADD, RF(2), RF(2), I(1));
    hs_qpu_alu dec = hs_qpu_a(HS_QPU_A_SUB, RF(0), RF(0), I(1));
    dec.af = HS_QPU_PF_PUSHZ;
    emit(p, dec);
    hs_qpu_branch(p, HS_QPU_B_NA0, l, 0, 0);
    nop(p), nop(p), nop(p);
    hs_qpu_emit_end(p);
    return hs_qpu_prog_finish(p);
}

/* hs_v3d_dispatch takes 1, 2, 4 or 8 batches. Other counts (up to 16: the
 * batch number is 4 bits) go to the kernel directly, with the same
 * configuration words, to measure how many batches run at once. */
static int dispatch_raw(hs_v3d* dev, uint32_t prog, uint32_t unif_gpu, uint32_t batches, const uint32_t* h, uint32_t nh) {
    struct drm_v3d_submit_csd s;
    memset(&s, 0, sizeof s);
    s.cfg[0] = batches << 16, s.cfg[1] = 1u << 16, s.cfg[2] = 1u << 16, s.cfg[3] = 0xF010, s.cfg[4] = batches - 1;
    s.cfg[5] = prog, s.cfg[6] = unif_gpu;
    s.bo_handles = (uint64_t)(uintptr_t)h, s.bo_handle_count = nh;
    return ioctl(dev->fd, DRM_IOCTL_V3D_SUBMIT_CSD, &s) ? -errno : 0;
}

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

int main(int argc, char** argv) {
    if (argc < 4 || atoi(argv[3]) < 1 || atoi(argv[3]) > 16)
        return fprintf(stderr, "usage: %s <variant 0..4> <iterations> <batches 1..16> [repeats | <seconds>s] [stretch]\n", argv[0]), 2;
    int variant = atoi(argv[1]), repeats = argc > 4 ? atoi(argv[4]) : 1;
    double seconds = argc > 4 && strchr(argv[4], 's') ? atof(argv[4]) : 0;
    uint32_t stretch = argc > 5 ? (uint32_t)strtoul(argv[5], NULL, 0) : 1;
    if (seconds > 0) repeats = 1 << 30;
    uint32_t iters = (uint32_t)strtoul(argv[2], NULL, 0), batches = (uint32_t)atoi(argv[3]);
    static uint64_t buf[128];
    hs_v3d dev;
    hs_v3d_bo code, unif, data;
    if (hs_v3d_open(&dev, NULL) || hs_v3d_bo_new(&dev, &code, 4096) || hs_v3d_bo_new(&dev, &unif, 4096) ||
        hs_v3d_bo_new(&dev, &data, 4096))
        return fprintf(stderr, "setup failed\n"), 1;
    hs_qpu_prog p;
    hs_qpu_prog_init(&p, buf, 128);
    int len = build(&p, variant);
    uint32_t prog = len > 0 ? hs_v3d_load(&code, 0, buf, (uint32_t)len) : 0;
    if (!prog) return fprintf(stderr, "build failed: %d\n", len), 1;
    ((uint32_t*)unif.map)[0] = iters;
    ((uint32_t*)unif.map)[1] = data.gpu;
    ((uint32_t*)unif.map)[2] = stretch;
    ((uint32_t*)unif.map)[3] = 256;
    uint32_t h[3] = {code.handle, unif.handle, data.handle};
    hs_v3d_dispatch_desc d = {prog, unif.gpu, batches, h, 3, 0};
    uint64_t t_all = now_ns(), sum = 0, mx = 0;
    int jobs = 0;
    for (int r = 0; r < repeats; r++) {
        uint64_t t = now_ns();
        int std = batches == 1 || batches == 2 || batches == 4 || batches == 8;
        int e = std ? hs_v3d_dispatch(&dev, &d) : dispatch_raw(&dev, prog, unif.gpu, batches, h, 3);
        if (!e) e = hs_v3d_bo_wait(&dev, &code, 30000000000ull);
        uint64_t dt = now_ns() - t;
        double ms = dt / 1e6;
        sum += dt, jobs++;
        if (dt > mx) mx = dt;
        if (seconds <= 0 || e) {
            char tid[320] = "";
            int o = 0;
            for (uint32_t b = 0; b < batches; b++)
                o += snprintf(tid + o, sizeof tid - (size_t)o, "%s%u", b ? "," : "", ((uint32_t*)data.map)[64 + 16 * b]);
            printf("{\"spin\":%d,\"iterations\":%u,\"stretch\":%u,\"batches\":%u,\"rc\":%d,\"job_ms\":%.3f,\"ns_per_iteration\":%.2f,"
                   "\"thread_index_per_batch\":[%s]}\n",
                   variant, iters, stretch, batches, e, ms, ms * 1e6 / iters, tid);
        }
        fflush(stdout);
        if (e) return 1;
        if (seconds > 0 && now_ns() - t_all > (uint64_t)(seconds * 1e9)) break;
    }
    if (seconds > 0)
        printf("{\"spin\":%d,\"iterations\":%u,\"stretch\":%u,\"batches\":%u,\"jobs\":%d,\"job_ms\":{\"mean\":%.3f,\"max\":%.3f},"
               "\"ns_per_iteration\":%.2f}\n",
               variant, iters, stretch, batches, jobs, sum / 1e6 / jobs, mx / 1e6, (double)sum / jobs / iters);
    hs_v3d_bo_free(&dev, &data);
    hs_v3d_bo_free(&dev, &unif);
    hs_v3d_bo_free(&dev, &code);
    hs_v3d_close(&dev);
    return 0;
}
