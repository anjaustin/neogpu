/* gather_anom: root-causes the slow 8-batch sequential gather at k = 3..5
 * (2026-11-16; docs/benchmarks/QPU_ENCODER.md). Builds gather kernel
 * variants, runs them through hs_v3d with a perfmon of many counters, and
 * checks every output word.
 *
 * Variants:
 *   0  the interleaved layout shipped until the fix: batch b at 64b, stride
 *      64 * batches. Its instructions were checked identical to the then
 *      shipped hs_qpu_kernel_gather. Kept as the anomaly reference.
 *   2  each batch owns one contiguous chunk (much worse: L2T set conflicts)
 *   3  the fix: each batch owns a contiguous 64k-byte run per pass
 *   4  variant 0 with batch b started b * S passes ahead (the skew test)
 *   5  the shipped hs_qpu_kernel_gather (the fix, computing the skip itself)
 *
 * Usage: tools/gather_anom list            -> kernel perfmon counter names
 *        tools/gather_anom check           -> every variant builds and passes the rules
 *        [TRACE=1] tools/gather_anom [MiB] < cases   -> one line per case:
 *   B k N toff ioff ooff mode variant [trials] [skew]
 *   offsets in hex bytes into one data BO; mode 0 random over 16M entries,
 *   1 sequential idx[i] = i. TRACE=1 (variant 0): the CPU samples how far
 *   each batch has got while the GPU runs (the drift trace).
 * Run inside the GPU lock: flock build/gpu.lock tools/gather_anom ... */
#define _GNU_SOURCE
#include "hs_qpu.h"
#include "hs_qpu_kernels.h"
#include "hs_v3d.h"
#include <drm/v3d_drm.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>

#define R(n) HS_QPU_R(n)
#define RF(n) HS_QPU_RF(n)
#define X HS_QPU_NOREG
#define I(v) hs_qpu_imm_int(v)
static void emit(hs_qpu_prog* p, hs_qpu_alu x) { hs_qpu_emit(p, &x); }
static void nop(hs_qpu_prog* p) { emit(p, hs_qpu_alu_nop()); }
static void ldunifrf(hs_qpu_prog* p, int rf) { emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDUNIFRF, RF(rf))); }

static hs_v3d dev;
static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
static int cmpd(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}
static uint32_t hash32(uint32_t x) { return x * 2654435761u ^ (x >> 7); }

/* variant bits:
 *   0 shipped order: k idx loads | k gathers | k stores
 *   1: stores use rf6 = out pointer advanced separately (same as 0 functionally)
 *   2: per-QPU contiguous layout: batch b owns a contiguous chunk
 *      (lane addr = base + b*chunk + 64*j), stride 64 within a batch */
static int build(hs_qpu_prog* p, int batches, int k, int variant) {
    if (variant == 5) return hs_qpu_kernel_gather(p, batches, k);
    int shift = batches == 8 ? 3 : batches == 4 ? 2 : batches == 2 ? 1 : 0;
    /* rf0 passes, rf1 idx, rf2 table, rf3 out, rf6 chunk bytes (variant 2) */
    for (int i = 0; i < 4; i++) ldunifrf(p, i);
    if (variant == 2 || variant == 3 || variant == 4) ldunifrf(p, 6);
    if (batches == 1) {
        emit(p, hs_qpu_a(HS_QPU_A_MOV, RF(5), I(0), X));
    } else {
        emit(p, hs_qpu_a(HS_QPU_A_TIDX, R(0), X, X));
        emit(p, hs_qpu_a(HS_QPU_A_SHR, R(0), R(0), I(2)));
        emit(p, hs_qpu_a(HS_QPU_A_BAND, RF(5), R(0), I(15)));
    }
    if (variant == 3) {
        /* batch b, pass q, load i at base + (q*B + b)*64k + 64i + 4*lane:
         * each batch reads a contiguous 64k-byte run per pass.
         * rf6 = uniform 64k(B-1): the skip added after each pass */
        emit(p, hs_qpu_a(HS_QPU_A_SHL, R(0), RF(5), I(6)));
        emit(p, hs_qpu_m(hs_qpu_alu_nop(), HS_QPU_M_UMUL24, R(0), R(0), I(k)));
        emit(p, hs_qpu_a(HS_QPU_A_EIDX, R(1), X, X));
        emit(p, hs_qpu_a(HS_QPU_A_SHL, R(1), R(1), I(2)));
        emit(p, hs_qpu_a(HS_QPU_A_ADD, R(0), R(0), R(1)));
        emit(p, hs_qpu_a(HS_QPU_A_ADD, RF(1), RF(1), R(0)));
        emit(p, hs_qpu_a(HS_QPU_A_ADD, RF(3), RF(3), R(0)));
        emit(p, hs_qpu_a(HS_QPU_A_MOV, R(0), I(1), X));
        emit(p, hs_qpu_a(HS_QPU_A_SHL, RF(4), R(0), I(6)));
    } else if (variant == 2) {
        /* byte offset = batch * chunk + 4 * lane; stride 64 */
        emit(p, hs_qpu_m(hs_qpu_alu_nop(), HS_QPU_M_UMUL24, R(0), RF(5), RF(6)));
        emit(p, hs_qpu_a(HS_QPU_A_EIDX, R(1), X, X));
        emit(p, hs_qpu_a(HS_QPU_A_SHL, R(1), R(1), I(2)));
        emit(p, hs_qpu_a(HS_QPU_A_ADD, R(0), R(0), R(1)));
        emit(p, hs_qpu_a(HS_QPU_A_ADD, RF(1), RF(1), R(0)));
        emit(p, hs_qpu_a(HS_QPU_A_ADD, RF(3), RF(3), R(0)));
        emit(p, hs_qpu_a(HS_QPU_A_MOV, R(0), I(1), X));
        emit(p, hs_qpu_a(HS_QPU_A_SHL, RF(4), R(0), I(6)));
    } else {
        emit(p, hs_qpu_a(HS_QPU_A_SHL, R(0), RF(5), I(4)));
        emit(p, hs_qpu_a(HS_QPU_A_EIDX, R(1), X, X));
        emit(p, hs_qpu_a(HS_QPU_A_ADD, R(0), R(0), R(1)));
        emit(p, hs_qpu_a(HS_QPU_A_SHL, R(0), R(0), I(2)));
        emit(p, hs_qpu_a(HS_QPU_A_ADD, RF(1), RF(1), R(0)));
        emit(p, hs_qpu_a(HS_QPU_A_ADD, RF(3), RF(3), R(0)));
        emit(p, hs_qpu_a(HS_QPU_A_MOV, R(0), I(1), X));
        emit(p, hs_qpu_a(HS_QPU_A_SHL, RF(4), R(0), I(6 + shift)));
        if (variant == 4) {
            /* skew: batch b starts b * rf6 bytes further on (rf6 = S passes) */
            emit(p, hs_qpu_m(hs_qpu_alu_nop(), HS_QPU_M_UMUL24, R(0), RF(5), RF(6)));
            emit(p, hs_qpu_a(HS_QPU_A_ADD, RF(1), RF(1), R(0)));
            emit(p, hs_qpu_a(HS_QPU_A_ADD, RF(3), RF(3), R(0)));
        }
    }
    int l = hs_qpu_label_new(p);
    hs_qpu_label_bind(p, l);
    for (int i = 0; i < k; i++) {
        hs_qpu_alu x = hs_qpu_m(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(1), X), HS_QPU_M_ADD, RF(1), RF(1), RF(4));
        emit(p, i == k - 1 ? hs_qpu_s(x, HS_QPU_SIG_THRSW, X) : x);
    }
    nop(p);
    nop(p);
    for (int i = 0; i < k; i++) {
        emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDTMU, R(0)));
        emit(p, hs_qpu_a(HS_QPU_A_SHL, R(0), R(0), I(2)));
        hs_qpu_alu x = hs_qpu_a(HS_QPU_A_ADD, HS_QPU_TMUA, R(0), RF(2));
        emit(p, i == k - 1 ? hs_qpu_s(x, HS_QPU_SIG_THRSW, X) : x);
    }
    nop(p);
    nop(p);
    for (int i = 0; i < k; i++) {
        emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDTMU, R(1)));
        emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUD, R(1), X));
        emit(p, hs_qpu_m(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(3), X), HS_QPU_M_ADD, RF(3), RF(3), RF(4)));
    }
    if (variant == 3) {
        emit(p, hs_qpu_a(HS_QPU_A_ADD, RF(1), RF(1), RF(6)));
        emit(p, hs_qpu_a(HS_QPU_A_ADD, RF(3), RF(3), RF(6)));
    }
    hs_qpu_alu s = hs_qpu_a(HS_QPU_A_SUB, RF(0), RF(0), I(1));
    s.af = HS_QPU_PF_PUSHZ;
    emit(p, s);
    hs_qpu_branch(p, HS_QPU_B_NA0, l, 0, 0);
    nop(p);
    nop(p);
    nop(p);
    emit(p, hs_qpu_a(HS_QPU_A_TMUWT, X, X, X));
    hs_qpu_emit_end(p);
    return hs_qpu_prog_finish(p);
}

static uint8_t CID[32] = {32, 86, 16, 17, 75, 71, 76, 72, 30, 31, 56, 52, 64, 63,
                              24, 25, 85, 53, 54, 50, 51, 13};
static size_t NC = 22;
static char cname[32][64];

static uint32_t pm_new(void) {
    struct drm_v3d_perfmon_create c;
    memset(&c, 0, sizeof(c));
    c.ncounters = NC;
    memcpy(c.counters, CID, NC);
    if (ioctl(dev.fd, DRM_IOCTL_V3D_PERFMON_CREATE, &c)) { perror("perfmon"); exit(1); }
    return c.id;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (hs_v3d_open(&dev, NULL)) { fprintf(stderr, "no v3d\n"); return 1; }
    if (getenv("CTRS")) {
        NC = 0;
        for (char* t = strtok(getenv("CTRS"), ","); t && NC < 32; t = strtok(NULL, ",")) CID[NC++] = (uint8_t)atoi(t);
    }
    for (size_t i = 0; i < NC; i++) {
        struct drm_v3d_perfmon_get_counter c;
        memset(&c, 0, sizeof(c));
        c.counter = CID[i];
        if (!ioctl(dev.fd, DRM_IOCTL_V3D_PERFMON_GET_COUNTER, &c)) snprintf(cname[i], 64, "%s", (char*)c.name);
        else snprintf(cname[i], 64, "c%u", CID[i]);
    }
    if (argc > 1 && !strcmp(argv[1], "list")) {
        for (unsigned i = 0; i < 87; i++) {
            struct drm_v3d_perfmon_get_counter c;
            memset(&c, 0, sizeof(c));
            c.counter = (uint8_t)i;
            if (!ioctl(dev.fd, DRM_IOCTL_V3D_PERFMON_GET_COUNTER, &c))
                printf("%2u %-28s %s\n", i, (char*)c.name, (char*)c.description);
        }
        return 0;
    }
    static uint64_t a[2048], b[2048];
    if (argc > 1 && !strcmp(argv[1], "check")) {
        int all = 1;
        static const int V[] = {0, 2, 3, 4, 5};
        for (size_t v = 0; v < sizeof(V) / sizeof(V[0]); v++)
            for (int B = 1; B <= 8; B *= 8)
                for (int k = 1; k <= 8; k *= 2) {
                    hs_qpu_prog p;
                    hs_qpu_prog_init(&p, a, 2048);
                    int n = build(&p, B, k, V[v]);
                    printf("variant %d B=%d k=%d: %d instructions%s\n", V[v], B, k, n, n > 0 ? "" : " BUILD/RULE ERROR");
                    all &= n > 0;
                }
        (void)b;
        return all ? 0 : 1;
    }
    size_t dsize = (size_t)(argc > 1 ? atoi(argv[1]) : 192) << 20;
    hs_v3d_bo code, unif, data;
    if (hs_v3d_bo_new(&dev, &code, 1 << 20) || hs_v3d_bo_new(&dev, &unif, 64 << 10) || hs_v3d_bo_new(&dev, &data, dsize)) {
        fprintf(stderr, "bo\n");
        return 1;
    }
    const uint32_t handles[3] = {code.handle, unif.handle, data.handle};
    printf("# data BO gpu 0x%08x size %zu MiB\n# B k N toff ioff ooff mode var | ms(median) ok |", data.gpu, dsize >> 20);
    for (size_t i = 0; i < NC; i++) printf(" %s", cname[i]);
    printf("\n");
    char line[256];
    uint32_t last_fill_key[4] = {~0u, ~0u, ~0u, ~0u};
    while (fgets(line, sizeof(line), stdin)) {
        if (line[0] == '#' || line[0] == '\n') { fputs(line, stdout); continue; }
        int B, k, mode, var, trials = 5, skew = 0;
        unsigned N, toff, ioff, ooff;
        if (sscanf(line, "%d %d %u %x %x %x %d %d %d %d", &B, &k, &N, &toff, &ioff, &ooff, &mode, &var, &trials, &skew) < 8) continue;
        if (trials > 15) trials = 15;
        if (trials < 1) trials = 1;
        if (skew < 0) { printf("skip: skew must be >= 0\n"); continue; }
        if (B != 1 && B != 8) { printf("skip: B must be 1 or 8 (the batch index is wrong otherwise)\n"); continue; }
        unsigned per = 16u * (unsigned)B * (unsigned)k;
        unsigned passes = N / per;
        N = passes * per;
        const uint32_t TE = 16u << 20; /* table entries for random */
        if ((size_t)toff + (size_t)TE * 4 > dsize && mode == 0) { printf("skip table range\n"); continue; }
        if ((size_t)ioff + N * 4ull > dsize || (size_t)ooff + N * 4ull > dsize || (size_t)toff + N * 4ull > dsize) { printf("skip range\n"); continue; }
        uint32_t* tb = (uint32_t*)(data.map + toff);
        uint32_t* ix = (uint32_t*)(data.map + ioff);
        uint32_t* ot = (uint32_t*)(data.map + ooff);
        if (var == 4 && (uint64_t)8 * (uint64_t)skew * per * 4 > dsize) { printf("skip: skew beyond the BO\n"); continue; }
        uint32_t NF = var == 4 ? N + 8u * (uint32_t)skew * per : N; /* skewed batches run past N (bounded above) */
        if (var == 4 && ((size_t)ioff + NF * 4ull > dsize || (size_t)ooff + NF * 4ull > dsize || ioff + NF * 4ull > ooff)) { printf("skip skew range\n"); continue; }
        uint32_t tent = mode == 0 ? TE : NF;
        if (passes == 0) { printf("skip: N below one pass (%u elements)\n", per); continue; }
        {
            /* table, idx and out must lie inside the BO and not overlap */
            uint64_t lo[3] = {toff, ioff, ooff}, hi[3] = {toff + (uint64_t)tent * 4, ioff + (uint64_t)NF * 4,
                                                        ooff + (uint64_t)NF * 4};
            int bad = 0;
            for (int x = 0; x < 3; x++) {
                bad |= hi[x] > dsize;
                for (int y = x + 1; y < 3; y++) bad |= lo[x] < hi[y] && lo[y] < hi[x];
            }
            if (bad) {
                printf("skip: table/idx/out out of range or overlapping\n");
                continue;
            }
        }
        /* (re)fill table and idx */
        for (uint32_t i = 0; i < tent; i++) tb[i] = hash32(i);
        uint64_t rs = 0x9E3779B97F4A7C15ull;
        for (uint32_t i = 0; i < NF; i++) {
            rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
            ix[i] = mode == 0 ? (uint32_t)(rs % TE) : i;
        }
        (void)last_fill_key;
        hs_qpu_prog p;
        hs_qpu_prog_init(&p, a, 2048);
        int n = build(&p, B, k, var);
        if (n <= 0) { printf("build error %d at %u\n", n, p.rule_at); continue; }
        uint32_t cg = hs_v3d_load(&code, 0, a, (uint32_t)n);
        uint32_t* u = (uint32_t*)unif.map;
        u[0] = passes;
        u[1] = data.gpu + ioff;
        u[2] = data.gpu + toff;
        u[3] = data.gpu + ooff;
        u[4] = var == 4 ? (uint32_t)skew * 64u * (uint32_t)B * (uint32_t)k : var == 3 ? 64u * (uint32_t)k * (uint32_t)(B - 1) : (uint32_t)passes * (uint32_t)k * 64u;
        double ms[15];
        uint64_t v[32], vs[15][32];
        int ok = 1;
        for (int t = 0; t < trials; t++) {
            memset(ot, 0xEE, (size_t)NF * 4); /* sentinel: hash32(0) == 0, so 0 would hide an unwritten word */
            uint32_t pm = pm_new();
            hs_v3d_dispatch_desc d = {cg, unif.gpu, (uint32_t)B, handles, 3, pm};
            uint64_t t0 = now_ns();
            if (hs_v3d_dispatch(&dev, &d)) { perror("dispatch"); return 1; }
            if (getenv("TRACE") && t == 0 && var == 0) {
                /* CPU watches out[] while the GPU runs: per batch, the highest
                 * load slot L (= pass*k + i) whose first word is visible in
                 * memory (binary search; out was filled with the 0xEE sentinel). */
                uint32_t nl = passes * (uint32_t)k;
                for (int step = 0; step < 4000; step++) {
                    uint64_t tn = now_ns();
                    uint32_t fr[8];
                    int done = 1;
                    for (int bb = 0; bb < B; bb++) {
                        uint32_t lo = 0, hi = nl; /* find last visible slot: first-invisible search */
                        while (lo < hi) {
                            uint32_t mid = (lo + hi) / 2;
                            uint32_t e = ((mid * (uint32_t)B) + (uint32_t)bb) * 16u;
                            if (((volatile uint32_t*)ot)[e] != 0xEEEEEEEEu) lo = mid + 1; else hi = mid;
                        }
                        fr[bb] = lo;
                        done &= lo == nl;
                    }
                    printf("T %.1f", (double)(tn - t0) / 1e3);
                    for (int bb = 0; bb < B; bb++) printf(" %u", fr[bb]);
                    printf("\n");
                    if (done) break;
                    while (now_ns() - tn < 250000) {}
                }
            }
            if (hs_v3d_bo_wait(&dev, &data, 10000000000ull)) { perror("wait"); return 1; }
            ms[t] = (double)(now_ns() - t0) / 1e6;
            struct drm_v3d_perfmon_get_values g = {.id = pm, .values_ptr = (uint64_t)(uintptr_t)vs[t]};
            ioctl(dev.fd, DRM_IOCTL_V3D_PERFMON_GET_VALUES, &g);
            struct drm_v3d_perfmon_destroy de = {.id = pm};
            ioctl(dev.fd, DRM_IOCTL_V3D_PERFMON_DESTROY, &de);
            uint32_t bad = 0;
            if (var != 4) for (uint32_t i = 0; i < N; i++) bad += ot[i] != hash32(ix[i]);
            else for (uint32_t i = 0; i < N; i++) bad += ot[i] != 0xEEEEEEEEu && ot[i] != hash32(ix[i]); /* partial coverage */
            ok &= bad == 0;
        }
        double sorted[15];
        memcpy(sorted, ms, sizeof(double) * (size_t)trials);
        qsort(sorted, (size_t)trials, sizeof(double), cmpd);
        double med = sorted[trials / 2];
        int mi = 0;
        for (int t = 0; t < trials; t++) if (ms[t] == med) mi = t;
        memcpy(v, vs[mi], sizeof(v));
        printf("%d %d %u %x %x %x %d %d s%d | %.2f [%.2f..%.2f] %s |", B, k, N, toff, ioff, ooff, mode, var, skew, med, sorted[0],
               sorted[trials - 1], ok ? "ok" : "WRONG");
        for (size_t i = 0; i < NC; i++) printf(" %llu", (unsigned long long)v[i]);
        printf("\n");
    }
    return 0;
}
