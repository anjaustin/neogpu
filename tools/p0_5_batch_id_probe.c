/* p0_5_batch_id_probe: what identifies a compute batch on the QPU?
 * (P0-5 red team, docs/audit/HARDWARE_HANDOFF_P0.md.)
 *
 * Every 8-batch program in this tree took its batch number from the QPU
 * thread index, (tidx >> 2) & 15. Measured 2026-11-17: that is only 0..7
 * once each while nothing else uses the GPU; beside a renderer batches share
 * thread slots and the shipped 8-batch gather writes wrong words in nearly
 * every job. This probe records the registers the hardware presets at program
 * entry (rf0..rf3), and tidx, per batch, for two dispatch configurations:
 *   cfg A: workgroup counts 1 x 1 x 1, CFG4 = batches - 1 (hs_v3d_dispatch)
 *   cfg B: workgroup counts batches x 1 x 1 (what Mesa submits)
 * Usage: p0_5_batch_id_probe show          one job per cfg, slot chosen by tidx
 *        p0_5_batch_id_probe cover <A|B> <tidx|rf0> <seconds>
 *            jobs back to back; counts jobs in which all 8 slots were written
 * Run inside the GPU lock. Output: JSON lines.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>

#include <drm/v3d_drm.h>

#include "hs_qpu.h"
#include "hs_v3d.h"

#define R(n) HS_QPU_R(n)
#define RF(n) HS_QPU_RF(n)
#define X HS_QPU_NOREG
#define I(v) hs_qpu_imm_int(v)

static void emit(hs_qpu_prog* p, hs_qpu_alu x) { hs_qpu_emit(p, &x); }
static void op(hs_qpu_prog* p, int o, hs_qpu_reg d, hs_qpu_reg a, hs_qpu_reg b) { emit(p, hs_qpu_a(o, d, a, b)); }
static void store(hs_qpu_prog* p, hs_qpu_reg addr, hs_qpu_reg v) {
    op(p, HS_QPU_A_MOV, HS_QPU_TMUD, v, X);
    op(p, HS_QPU_A_MOV, HS_QPU_TMUA, addr, X);
}

/* by_rf0: the slot is rf0 & 0xffff (the workgroup x the hardware presets);
 * otherwise (tidx >> 2) & 15. Slot s: 256 bytes at base + 256 s, lines rf0,
 * rf1, rf2, tidx. */
static int build(hs_qpu_prog* p, int by_rf0) {
    hs_qpu_prog_rules(p, 0); /* reads registers the program did not write */
    op(p, HS_QPU_A_MOV, RF(10), RF(0), X);
    op(p, HS_QPU_A_MOV, RF(11), RF(1), X);
    op(p, HS_QPU_A_MOV, RF(12), RF(2), X);
    emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDUNIFRF, RF(20)));
    op(p, HS_QPU_A_TIDX, RF(13), X, X);
    if (by_rf0) {
        op(p, HS_QPU_A_SHL, R(0), RF(10), I(15)); /* keep the low 16 bits */
        op(p, HS_QPU_A_SHL, R(0), R(0), I(1));
        op(p, HS_QPU_A_SHR, R(0), R(0), I(15));
        op(p, HS_QPU_A_SHR, R(0), R(0), I(1));
    } else {
        op(p, HS_QPU_A_SHR, R(0), RF(13), I(2));
        op(p, HS_QPU_A_BAND, R(0), R(0), I(15));
    }
    op(p, HS_QPU_A_SHL, R(0), R(0), I(8));
    op(p, HS_QPU_A_EIDX, R(1), X, X);
    op(p, HS_QPU_A_SHL, R(1), R(1), I(2));
    op(p, HS_QPU_A_ADD, R(0), R(0), R(1));
    op(p, HS_QPU_A_ADD, RF(21), RF(20), R(0));
    op(p, HS_QPU_A_MOV, R(2), I(1), X);
    op(p, HS_QPU_A_SHL, RF(22), R(2), I(6));
    store(p, RF(21), RF(10));
    op(p, HS_QPU_A_ADD, RF(21), RF(21), RF(22));
    store(p, RF(21), RF(11));
    op(p, HS_QPU_A_ADD, RF(21), RF(21), RF(22));
    store(p, RF(21), RF(12));
    op(p, HS_QPU_A_ADD, RF(21), RF(21), RF(22));
    store(p, RF(21), RF(13));
    op(p, HS_QPU_A_TMUWT, X, X, X);
    hs_qpu_emit_end(p);
    return hs_qpu_prog_finish(p);
}

static hs_v3d dev;
static hs_v3d_bo code, unif, data;

static int submit(char cfg, uint32_t prog) {
    struct drm_v3d_submit_csd s;
    uint32_t h[3] = {code.handle, unif.handle, data.handle};
    memset(&s, 0, sizeof s);
    s.cfg[0] = (cfg == 'B' ? 8u : 1u) << 16;
    s.cfg[1] = 1u << 16;
    s.cfg[2] = 1u << 16;
    s.cfg[3] = 0xF010; /* 16 workgroups per supergroup, workgroup size 16 (hs_v3d_dispatch) */
    s.cfg[4] = 7;
    s.cfg[5] = prog;
    s.cfg[6] = unif.gpu;
    s.bo_handles = (uint64_t)(uintptr_t)h;
    s.bo_handle_count = 3;
    int r;
    do r = ioctl(dev.fd, DRM_IOCTL_V3D_SUBMIT_CSD, &s);
    while (r && (errno == EINTR || errno == EAGAIN));
    if (r) return -errno;
    return hs_v3d_bo_wait(&dev, &data, 5000000000ull);
}
static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

int main(int argc, char** argv) {
    static uint64_t buf[256];
    int e = hs_v3d_open(&dev, NULL);
    if (e || hs_v3d_bo_new(&dev, &code, 8192) || hs_v3d_bo_new(&dev, &unif, 4096) || hs_v3d_bo_new(&dev, &data, 16 * 256)) {
        fprintf(stderr, "setup failed\n");
        return 2;
    }
    uint32_t prog[2];
    for (int v = 0; v < 2; v++) {
        hs_qpu_prog p;
        hs_qpu_prog_init(&p, buf, 256);
        int n = build(&p, v);
        if (n < 0 || !(prog[v] = hs_v3d_load(&code, (size_t)v * 4096, buf, (uint32_t)n))) {
            fprintf(stderr, "build failed: %d\n", n);
            return 2;
        }
    }
    ((uint32_t*)unif.map)[0] = data.gpu;
    volatile uint32_t* d = (volatile uint32_t*)data.map;
    if (argc > 1 && !strcmp(argv[1], "show")) {
        for (char cfg = 'A'; cfg <= 'B'; cfg++) {
            for (int i = 0; i < 16 * 64; i++) d[i] = 0xDEADBEEF;
            if ((e = submit(cfg, prog[0]))) {
                fprintf(stderr, "submit: %s\n", strerror(-e));
                return 2;
            }
            for (int s = 0; s < 16; s++) {
                if (d[s * 64] == 0xDEADBEEF) continue;
                printf("{\"cfg\":\"%c\",\"slot_by_tidx\":%d,\"rf0\":\"0x%08x\",\"rf1\":\"0x%08x\",\"rf2_lane0\":\"0x%08x\","
                       "\"rf2_lane15\":\"0x%08x\",\"tidx\":%u}\n",
                       cfg, s, d[s * 64], d[s * 64 + 16], d[s * 64 + 32], d[s * 64 + 47], d[s * 64 + 48]);
            }
        }
        return 0;
    }
    if (argc > 4 && !strcmp(argv[1], "cover")) {
        char cfg = argv[2][0];
        int by_rf0 = !strcmp(argv[3], "rf0");
        uint64_t t_end = now_ns() + (uint64_t)(atof(argv[4]) * 1e9), jobs = 0, full = 0, tidx_seen = 0;
        uint32_t min_slots = 16;
        while (now_ns() < t_end) {
            for (int i = 0; i < 16 * 64; i++) d[i] = 0xDEADBEEF;
            __asm__ volatile("dsb sy" ::: "memory");
            if ((e = submit(cfg, prog[by_rf0]))) {
                fprintf(stderr, "submit: %s\n", strerror(-e));
                return 2;
            }
            uint32_t n = 0;
            for (int s = 0; s < 8; s++) {
                n += d[s * 64] != 0xDEADBEEF;
                if (d[s * 64] != 0xDEADBEEF) tidx_seen |= 1ull << (d[s * 64 + 48] & 63);
            }
            for (int s = 8; s < 16; s++) n += 100 * (d[s * 64] != 0xDEADBEEF); /* a slot outside 0..7 */
            jobs++, full += n == 8;
            if (n < min_slots) min_slots = n;
        }
        printf("{\"cover\":true,\"cfg\":\"%c\",\"slot_by\":\"%s\",\"jobs\":%llu,\"jobs_with_all_8_slots\":%llu,\"fewest_slots_in_a_job\":%u,"
               "\"tidx_values_seen_mask\":\"0x%llx\"}\n",
               cfg, argv[3], (unsigned long long)jobs, (unsigned long long)full, min_slots, (unsigned long long)tidx_seen);
        return full == jobs ? 0 : 1;
    }
    fprintf(stderr, "usage: see the file header\n");
    return 2;
}
