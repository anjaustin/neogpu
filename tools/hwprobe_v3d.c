/* hwprobe_v3d: measure the Raspberry Pi 4 V3D GPU through the kernel's v3d
 * driver directly (no Mesa, no libdrm): hardware identity, ioctl cost, buffer
 * object (BO) lifecycle, CPU access bandwidth to GPU-visible memory, and the
 * GPU performance counters the kernel exposes.
 *
 * Usage: tools/hwprobe_v3d [cpu=2] [/dev/dri/renderD128]
 * Needs read/write on the render node (the "render" group), not root. Run with
 * scaling_governor=performance for stable CPU-side timings.
 */
#define _GNU_SOURCE
#include <drm/drm.h>
#include <drm/v3d_drm.h>
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
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
static int get_param(uint32_t p, uint64_t* v) {
    struct drm_v3d_get_param g = {.param = p};
    int r = xioctl(DRM_IOCTL_V3D_GET_PARAM, &g);
    *v = g.value;
    return r;
}

typedef struct {
    uint32_t handle, gpu_offset;
    size_t size;
    void* map;
} bo_t;
static int bo_create(bo_t* b, size_t size) {
    struct drm_v3d_create_bo c = {.size = (uint32_t)size};
    if (xioctl(DRM_IOCTL_V3D_CREATE_BO, &c)) return -1;
    b->handle = c.handle;
    b->gpu_offset = c.offset;
    b->size = size;
    b->map = NULL;
    return 0;
}
static int bo_map(bo_t* b) {
    struct drm_v3d_mmap_bo m = {.handle = b->handle};
    if (xioctl(DRM_IOCTL_V3D_MMAP_BO, &m)) return -1;
    b->map = mmap(NULL, b->size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)m.offset);
    return b->map == MAP_FAILED ? -1 : 0;
}
static void bo_free(bo_t* b) {
    if (b->map && b->map != MAP_FAILED) munmap(b->map, b->size);
    struct drm_gem_close c = {.handle = b->handle};
    xioctl(DRM_IOCTL_GEM_CLOSE, &c);
}

static void section_ident(void) {
    static const struct { uint32_t p; const char* n; } P[] = {
        {DRM_V3D_PARAM_V3D_UIFCFG, "UIFCFG"},
        {DRM_V3D_PARAM_V3D_HUB_IDENT1, "HUB_IDENT1"},
        {DRM_V3D_PARAM_V3D_HUB_IDENT2, "HUB_IDENT2"},
        {DRM_V3D_PARAM_V3D_HUB_IDENT3, "HUB_IDENT3"},
        {DRM_V3D_PARAM_V3D_CORE0_IDENT0, "CORE0_IDENT0"},
        {DRM_V3D_PARAM_V3D_CORE0_IDENT1, "CORE0_IDENT1"},
        {DRM_V3D_PARAM_V3D_CORE0_IDENT2, "CORE0_IDENT2"},
        {DRM_V3D_PARAM_SUPPORTS_TFU, "SUPPORTS_TFU"},
        {DRM_V3D_PARAM_SUPPORTS_CSD, "SUPPORTS_CSD"},
        {DRM_V3D_PARAM_SUPPORTS_CACHE_FLUSH, "SUPPORTS_CACHE_FLUSH"},
        {DRM_V3D_PARAM_SUPPORTS_PERFMON, "SUPPORTS_PERFMON"},
        {DRM_V3D_PARAM_SUPPORTS_MULTISYNC_EXT, "SUPPORTS_MULTISYNC_EXT"},
        {DRM_V3D_PARAM_SUPPORTS_CPU_QUEUE, "SUPPORTS_CPU_QUEUE"},
        {DRM_V3D_PARAM_MAX_PERF_COUNTERS, "MAX_PERF_COUNTERS"},
        {DRM_V3D_PARAM_SUPPORTS_SUPER_PAGES, "SUPPORTS_SUPER_PAGES"},
    };
    printf("## Identity (raw registers; field decode follows the kernel's v3d_regs.h)\n");
    uint64_t v[32] = {0};
    for (size_t i = 0; i < sizeof(P) / sizeof(P[0]); i++) {
        int r = get_param(P[i].p, &v[P[i].p]);
        printf("%-24s %s0x%08llx (%llu)\n", P[i].n, r ? "ERR " : "", (unsigned long long)v[P[i].p],
               (unsigned long long)v[P[i].p]);
    }
    unsigned long long h1 = v[DRM_V3D_PARAM_V3D_HUB_IDENT1], c0 = v[DRM_V3D_PARAM_V3D_CORE0_IDENT0],
             c1 = v[DRM_V3D_PARAM_V3D_CORE0_IDENT1];
    printf("hub: tech version %llu rev %llu, cores %llu, has L3C %llu, TFU %llu, TSY %llu, MSO %llu\n",
           h1 & 0xf, (h1 >> 4) & 0xf, (h1 >> 8) & 0xf, (h1 >> 16) & 1, (h1 >> 17) & 1, (h1 >> 18) & 1, (h1 >> 19) & 1);
    printf("core0: version %llu; rev %llu, slices %llu, QPUs per slice %llu, TMUs %llu, semaphores %llu, VPM size field %llu\n",
           (c0 >> 24) & 0xff, c1 & 0xf, (c1 >> 4) & 0xf, (c1 >> 8) & 0xf, (c1 >> 12) & 0xf, (c1 >> 16) & 0xff,
           (c1 >> 28) & 0xf);
}

static void section_counters(void) {
    uint64_t n = 0;
    if (get_param(DRM_V3D_PARAM_MAX_PERF_COUNTERS, &n) || n == 0) {
        printf("\n## Performance counters: not reported\n");
        return;
    }
    printf("\n## Performance counters (%llu)\n", (unsigned long long)n);
    for (uint64_t i = 0; i < n && i < 256; i++) {
        struct drm_v3d_perfmon_get_counter c;
        memset(&c, 0, sizeof(c));
        c.counter = (uint8_t)i;
        if (xioctl(DRM_IOCTL_V3D_PERFMON_GET_COUNTER, &c)) {
            printf("%3llu: error %d\n", (unsigned long long)i, errno);
            continue;
        }
        printf("%3llu %-10s %-34s %s\n", (unsigned long long)i, (char*)c.category, (char*)c.name, (char*)c.description);
    }
}

static void section_ioctl(void) {
    printf("\n## Kernel crossings through the v3d driver\n");
    double r[TRIALS];
    uint64_t v;
    for (int t = 0; t < TRIALS; t++) {
        const int n = 20000;
        uint64_t t0 = now_ns();
        for (int i = 0; i < n; i++) get_param(DRM_V3D_PARAM_SUPPORTS_CSD, &v);
        r[t] = (double)(now_ns() - t0) / n;
    }
    printf("%-44s %10.1f ns\n", "GET_PARAM ioctl round trip", median(r, TRIALS));
    bo_t b;
    if (bo_create(&b, 4096)) { perror("create"); return; }
    for (int t = 0; t < TRIALS; t++) {
        const int n = 20000;
        uint64_t t0 = now_ns();
        for (int i = 0; i < n; i++) {
            struct drm_v3d_wait_bo w = {.handle = b.handle, .timeout_ns = 0};
            xioctl(DRM_IOCTL_V3D_WAIT_BO, &w);
        }
        r[t] = (double)(now_ns() - t0) / n;
    }
    printf("%-44s %10.1f ns\n", "WAIT_BO on an idle BO (timeout 0)", median(r, TRIALS));
    bo_free(&b);
}

static void section_bo_lifecycle(void) {
    static const size_t sizes[] = {4096, 65536, 1 << 20, 16 << 20, 64 << 20};
    printf("\n## BO lifecycle, median of %d (us)\n", TRIALS);
    printf("%10s %10s %10s %14s %14s %10s %12s\n", "size", "create", "map", "first touch", "touch MB/s", "free", "gpu offset");
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        double c[TRIALS], m[TRIALS], f[TRIALS], d[TRIALS];
        uint32_t off = 0;
        for (int t = 0; t < TRIALS; t++) {
            bo_t b;
            uint64_t t0 = now_ns();
            if (bo_create(&b, sizes[s])) { perror("create"); return; }
            uint64_t t1 = now_ns();
            if (bo_map(&b)) { perror("map"); return; }
            uint64_t t2 = now_ns();
            for (size_t i = 0; i < sizes[s]; i += 4096) ((volatile char*)b.map)[i] = 1;
            uint64_t t3 = now_ns();
            off = b.gpu_offset;
            bo_free(&b);
            uint64_t t4 = now_ns();
            c[t] = (double)(t1 - t0) / 1e3;
            m[t] = (double)(t2 - t1) / 1e3;
            f[t] = (double)(t3 - t2) / 1e3;
            d[t] = (double)(t4 - t3) / 1e3;
        }
        double fm = median(f, TRIALS);
        printf("%9zuK %10.1f %10.1f %14.1f %14.0f %10.1f   0x%08x\n", sizes[s] >> 10, median(c, TRIALS), median(m, TRIALS),
               fm, (double)sizes[s] / fm, median(d, TRIALS), off);
    }
}

/* CPU bandwidth over a mapping, NEON 128-byte steps (same kernels as hwprobe_cpu). */
static void rd(char* s, size_t len) {
    __asm__ volatile("1: ldp q0,q1,[%0]\n ldp q2,q3,[%0,#32]\n ldp q4,q5,[%0,#64]\n ldp q6,q7,[%0,#96]\n"
                     " add %0,%0,#128\n subs %1,%1,#128\n b.ne 1b"
                     : "+r"(s), "+r"(len)::"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory", "cc");
}
static void wr(char* d, size_t len) {
    __asm__ volatile("1: stp q0,q1,[%0]\n stp q2,q3,[%0,#32]\n stp q4,q5,[%0,#64]\n stp q6,q7,[%0,#96]\n"
                     " add %0,%0,#128\n subs %1,%1,#128\n b.ne 1b"
                     : "+r"(d), "+r"(len)::"memory", "cc");
}
static void rd_scalar(char* s, size_t len) { /* 8-byte loads, one at a time */
    volatile uint64_t* p = (volatile uint64_t*)s;
    uint64_t acc = 0;
    for (size_t i = 0; i < len / 8; i++) acc += p[i];
    __asm__ volatile("" ::"r"(acc));
}
static double gbps(void (*k)(char*, size_t), char* p, size_t len) {
    double r[TRIALS];
    k(p, len);
    for (int t = 0; t < TRIALS; t++) {
        int reps = (int)((64u << 20) / len);
        if (reps < 1) reps = 1;
        uint64_t t0 = now_ns();
        for (int i = 0; i < reps; i++) k(p, len);
        r[t] = (double)len * reps / (double)(now_ns() - t0);
    }
    return median(r, TRIALS);
}
static void section_bandwidth(void) {
    static const size_t sizes[] = {16 << 10, 256 << 10, 16 << 20};
    printf("\n## CPU bandwidth to memory, GB/s (v3d BO mapping vs ordinary anonymous memory)\n");
    printf("%-26s %12s %12s %12s\n", "", "16K", "256K", "16M");
    const char* names[] = {"BO read (ldp q)", "BO read (8 B scalar)", "BO write (stp q)", "anon read (ldp q)", "anon read (8 B scalar)",
                           "anon write (stp q)"};
    void (*ks[])(char*, size_t) = {rd, rd_scalar, wr, rd, rd_scalar, wr};
    for (int k = 0; k < 6; k++) {
        printf("%-26s", names[k]);
        for (int s = 0; s < 3; s++) {
            char* p;
            bo_t b;
            if (k < 3) {
                if (bo_create(&b, sizes[s]) || bo_map(&b)) { perror("bo"); return; }
                p = b.map;
            } else {
                p = mmap(NULL, sizes[s], PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            }
            memset(p, 1, sizes[s]);
            printf(" %12.3f", gbps(ks[k], p, sizes[s]));
            if (k < 3) bo_free(&b);
            else munmap(p, sizes[s]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    int cpu = argc > 1 ? atoi(argv[1]) : 2;
    const char* node = argc > 2 ? argv[2] : "/dev/dri/renderD128";
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    sched_setaffinity(0, sizeof(s), &s);
    setvbuf(stdout, NULL, _IOLBF, 0);
    fd = open(node, O_RDWR | O_CLOEXEC);
    if (fd < 0) { perror(node); return 1; }
    char gov[64] = "?";
    FILE* f = fopen("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "r");
    if (f) { if (fscanf(f, "%63s", gov) != 1) strcpy(gov, "?"); fclose(f); }
    printf("# hwprobe_v3d: %s, CPU core %d, governor %s\n\n", node, cpu, gov);
    section_ident();
    section_ioctl();
    section_bo_lifecycle();
    section_bandwidth();
    section_counters();
    return 0;
}
