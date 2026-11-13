/* bench_vs_mesa: NeoGPU's direct GPU path against the reference stack on the
 * same Raspberry Pi 4 GPU, and against the fastest CPU path.
 *
 *   NeoGPU  hs_qpu_kernels -> hs_qpu -> hs_v3d -> kernel v3d driver
 *   Mesa    GLES 3.1 compute shaders (EGL on GBM, the v3d render node) ->
 *           Mesa's v3d driver -> kernel v3d driver
 *   CPU     4 x Cortex-A72, 1 or 4 threads, with and without prefetch,
 *           huge pages (the fastest CPU gather measured in hwprobe_qpu)
 *
 * Same work on each path; every output word checked on every timed trial;
 * NeoGPU and Mesa trials interleave so drift hits both. Each timed GPU run
 * follows an untimed run of the same job (the GPU clock drops after
 * ~200 ms idle: HARDWARE_MAP_2026-11-15.md). Timing is wall time on the
 * calling thread: NeoGPU from submit to the BO wait returning, Mesa from
 * glDispatchCompute to glFinish returning.
 *
 * Usage (inside the GPU lock, performance governor):
 *   make bench-vs-mesa  (all sections, 7 trials, inside the GPU lock)
 *   tools/bench_vs_mesa [sections=lpg] [trials=7]   exit 1 if any output is wrong
 *     l  single-job round trip, p50/p99 over many jobs
 *     p  pipelined small jobs: jobs per second
 *     g  gather out[i] = table[idx[i]], 4M elements
 *   tools/bench_vs_mesa cold neo|mesa
 *     one process: start to first checked result on that path
 */
#define _GNU_SOURCE
#include "hs_qpu_kernels.h"
#include "hs_v3d.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <errno.h>
#include <fcntl.h>
#include <gbm.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
static void die(const char* m) {
    fprintf(stderr, "FATAL %s\n", m);
    exit(1);
}
static void pin(int cpu) {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    sched_setaffinity(0, sizeof(s), &s);
}
static int cmpd(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}
static double pct(double* v, int n, double p) {
    qsort(v, (size_t)n, sizeof(double), cmpd);
    int i = (int)(p * (n - 1) + 0.5);
    return v[i];
}
static uint32_t hash32(uint32_t x) { return x * 2654435761u ^ (x >> 7); }
/* any wrong output on any path: exit 1, so no speedup is ever reported on a wrong result */
static int g_wrong;

#define SENTINEL 0xEEEEEEEEu
#define N (4u << 20)        /* gather elements (16 MiB of indices, 16 MiB out) */
#define TMAX (16u << 20)    /* largest table: 64 MiB */

/* ---------------------------------------------------------------- NeoGPU */
static hs_v3d dev;
static hs_v3d_bo code, unif, data;
static uint32_t handles[3];
static uint64_t progbuf[1024];
static size_t code_off;
/* data BO layout: table | idx | out | word */
#define D_TBL 0u
#define D_IDX (TMAX * 4u)
#define D_OUT (D_IDX + N * 4u)
#define D_WORD (D_OUT + N * 4u)

static uint32_t neo_load(int (*build)(hs_qpu_prog*, int, int), int a, int b) {
    hs_qpu_prog p;
    hs_qpu_prog_init(&p, progbuf, 1024);
    int n = build(&p, a, b);
    if (n <= 0) die("NeoGPU kernel encode");
    uint32_t gpu = hs_v3d_load(&code, code_off, progbuf, (uint32_t)n);
    if (!gpu) die("NeoGPU kernel load");
    code_off += ((size_t)n * 8 + 255) & ~(size_t)255;
    return gpu;
}
static int store_one_build(hs_qpu_prog* p, int a, int b) {
    (void)a, (void)b;
    return hs_qpu_kernel_store_one(p);
}
static void neo_init(void) {
    if (hs_v3d_open(&dev, NULL)) die("no v3d render node");
    if (hs_v3d_bo_new(&dev, &code, 64 << 10) || hs_v3d_bo_new(&dev, &unif, 4 << 10) ||
        hs_v3d_bo_new(&dev, &data, D_WORD + 4096))
        die("NeoGPU BOs");
    handles[0] = code.handle, handles[1] = unif.handle, handles[2] = data.handle;
}
static int neo_run(uint32_t gpu, uint32_t unif_off, uint32_t batches) {
    hs_v3d_dispatch_desc d = {gpu, unif.gpu + unif_off, batches, handles, 3, 0};
    int e = hs_v3d_dispatch(&dev, &d);
    return e ? e : hs_v3d_bo_wait(&dev, &data, 5000000000ull);
}

/* ------------------------------------------------------------------ Mesa */
static int gfd = -1;
static struct gbm_device* gbm;
static EGLDisplay dpy;
static EGLContext ctx;
static GLint max_ssbo;

static void mesa_init(void) {
    gfd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
    if (gfd < 0) die("open renderD128");
    gbm = gbm_create_device(gfd);
    if (!gbm) die("gbm_create_device");
    PFNEGLGETPLATFORMDISPLAYEXTPROC gpd = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (!gpd) die("eglGetPlatformDisplayEXT");
    dpy = gpd(EGL_PLATFORM_GBM_KHR, gbm, NULL);
    if (dpy == EGL_NO_DISPLAY || !eglInitialize(dpy, NULL, NULL)) die("eglInitialize");
    if (!eglBindAPI(EGL_OPENGL_ES_API)) die("eglBindAPI");
    static const EGLint ca[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR, EGL_NONE};
    EGLConfig cfg;
    EGLint nc = 0;
    if (!eglChooseConfig(dpy, ca, &cfg, 1, &nc) || nc < 1) die("eglChooseConfig");
    static const EGLint xa[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE};
    ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, xa);
    if (ctx == EGL_NO_CONTEXT) die("eglCreateContext ES 3.1");
    if (!eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) die("eglMakeCurrent surfaceless");
    glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &max_ssbo);
}
static void mesa_fini(void) {
    eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(dpy, ctx);
    eglTerminate(dpy);
    gbm_device_destroy(gbm);
    close(gfd);
}
static GLuint mesa_program(const char* src) {
    GLuint s = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(s, sizeof(log), NULL, log);
        fprintf(stderr, "%s\n", log);
        die("compute shader compile");
    }
    GLuint p = glCreateProgram();
    glAttachShader(p, s);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) die("compute program link");
    glDeleteShader(s);
    return p;
}
static const char* STORE_ONE_GLSL =
    "#version 310 es\n"
    "layout(local_size_x = 16) in;\n"
    "layout(std430, binding = 0) buffer W { uint w[]; };\n"
    "void main() { w[0] = 1u; }\n";
static GLuint mesa_gather_program(int ls, int e) {
    char src[1024];
    snprintf(src, sizeof(src),
             "#version 310 es\n"
             "layout(local_size_x = %d) in;\n"
             "layout(std430, binding = 0) readonly buffer T { uint t[]; };\n"
             "layout(std430, binding = 1) readonly buffer I { uint ix[]; };\n"
             "layout(std430, binding = 2) writeonly buffer O { uint o[]; };\n"
             "uniform uint total;\n"
             "void main() {\n"
             "  uint inv = (gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x) * %du + gl_LocalInvocationID.x;\n"
             "  for (uint j = 0u; j < %du; j++) { uint i = inv + j * total; o[i] = t[ix[i]]; }\n"
             "}\n",
             ls, ls, e);
    return mesa_program(src);
}
static GLuint mesa_buf(size_t bytes, const void* init) {
    GLuint b;
    glGenBuffers(1, &b);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, b);
    glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)bytes, init, GL_DYNAMIC_COPY);
    return b;
}
static void* mesa_map(GLuint b, size_t bytes, GLbitfield access) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, b);
    void* p = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)bytes, access);
    if (!p) die("glMapBufferRange");
    return p;
}
static void mesa_unmap(GLuint b) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, b);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
}

/* ------------------------------------------------------------------- CPU */
typedef struct {
    const uint32_t* t;
    const uint32_t* ix;
    uint32_t* o;
    size_t n;
    int cpu, pf;
} gjob_t;
static void* gjob_main(void* a) {
    gjob_t* j = a;
    pin(j->cpu);
    size_t i = 0;
    if (j->pf)
        for (; i + 32 < j->n; i++) {
            __builtin_prefetch(&j->t[j->ix[i + 32]]);
            j->o[i] = j->t[j->ix[i]];
        }
    for (; i < j->n; i++) j->o[i] = j->t[j->ix[i]];
    return NULL;
}
static double cpu_gather(const uint32_t* t, const uint32_t* ix, uint32_t* o, int nthr, int pf) {
    pthread_t id[4];
    gjob_t j[4];
    uint64_t t0 = now_ns();
    for (int k = 0; k < nthr; k++) {
        size_t a = (size_t)N * k / nthr, b = (size_t)N * (k + 1) / nthr;
        j[k] = (gjob_t){t, ix + a, o + a, b - a, k, pf};
        pthread_create(&id[k], NULL, gjob_main, &j[k]);
    }
    for (int k = 0; k < nthr; k++) pthread_join(id[k], NULL);
    return (double)(now_ns() - t0);
}
static void* halloc(size_t size) {
    void* p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) die("mmap");
    madvise(p, size, MADV_HUGEPAGE);
    memset(p, 0, size);
    return p;
}

/* ------------------------------------------------- l: single-job round trip */
#define LAT_N 2000
static void section_latency(void) {
    printf("\n## Single-job round trip: one 16-lane job writes 1 to a word; submit to completion seen by the CPU\n");
    printf("%d jobs per path, interleaved in blocks of 100, after 200 warm-up jobs each; the word is checked after\n",
           LAT_N);
    uint32_t neo_gpu = neo_load(store_one_build, 0, 0);
    volatile uint32_t* nw = (volatile uint32_t*)(data.map + D_WORD);
    ((uint32_t*)unif.map)[0] = data.gpu + D_WORD;
    GLuint mp = mesa_program(STORE_ONE_GLSL);
    GLuint mb = mesa_buf(4096, NULL);
    glUseProgram(mp);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, mb);
    *nw = 0;
    uint32_t* mz = mesa_map(mb, 4, GL_MAP_WRITE_BIT);
    *mz = 0;
    mesa_unmap(mb);
    for (int i = 0; i < 200; i++) {
        if (neo_run(neo_gpu, 0, 1)) die("NeoGPU store_one");
        glDispatchCompute(1, 1, 1);
        glFinish();
    }
    static double tn[LAT_N], tm[LAT_N];
    for (int blk = 0; blk < LAT_N / 100; blk++) {
        for (int i = 0; i < 100; i++) {
            uint64_t t0 = now_ns();
            if (neo_run(neo_gpu, 0, 1)) die("NeoGPU store_one");
            tn[blk * 100 + i] = (double)(now_ns() - t0);
        }
        for (int i = 0; i < 100; i++) {
            uint64_t t0 = now_ns();
            glDispatchCompute(1, 1, 1);
            glFinish();
            tm[blk * 100 + i] = (double)(now_ns() - t0);
        }
    }
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    uint32_t* mr = mesa_map(mb, 4, GL_MAP_READ_BIT);
    int mok = *mr == 1;
    mesa_unmap(mb);
    int nok = *nw == 1;
    printf("%-8s %9s %9s %9s %9s %9s  %s\n", "path", "min us", "p50 us", "p90 us", "p99 us", "max us", "check");
    double* v[2] = {tn, tm};
    const char* nm[2] = {"NeoGPU", "Mesa"};
    int ok[2] = {nok, mok};
    g_wrong |= !nok || !mok;
    double p50[2];
    for (int k = 0; k < 2; k++) {
        double mn = pct(v[k], LAT_N, 0), a = pct(v[k], LAT_N, 0.5), b = pct(v[k], LAT_N, 0.9),
               c = pct(v[k], LAT_N, 0.99), mx = pct(v[k], LAT_N, 1);
        p50[k] = a;
        printf("%-8s %9.1f %9.1f %9.1f %9.1f %9.1f  %s\n", nm[k], mn / 1e3, a / 1e3, b / 1e3, c / 1e3, mx / 1e3,
               ok[k] ? "ok" : "WRONG");
    }
    printf("Mesa / NeoGPU at p50: %.2fx\n", p50[1] / p50[0]);
    glDeleteBuffers(1, &mb);
    glDeleteProgram(mp);
}

/* ------------------------------------------------- p: pipelined small jobs */
static void section_pipelined(void) {
    printf("\n## Pipelined small jobs: submit J jobs back to back, then wait once (jobs per second)\n");
    printf("same 16-lane store_one job; median of 7 interleaved trials\n");
    uint32_t neo_gpu = neo_load(store_one_build, 0, 0);
    ((uint32_t*)unif.map)[0] = data.gpu + D_WORD;
    GLuint mp = mesa_program(STORE_ONE_GLSL);
    GLuint mb = mesa_buf(4096, NULL);
    glUseProgram(mp);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, mb);
    hs_v3d_dispatch_desc d = {neo_gpu, unif.gpu, 1, handles, 3, 0};
    printf("%6s %14s %14s %8s\n", "J", "NeoGPU jobs/s", "Mesa jobs/s", "ratio");
    static const int JS[] = {10, 100, 1000};
    for (size_t q = 0; q < 3; q++) {
        int J = JS[q];
        double tn[7], tm[7];
        for (int t = 0; t < 8; t++) { /* trial 0 is a warm-up */
            uint64_t t0 = now_ns();
            for (int i = 0; i < J; i++)
                if (hs_v3d_dispatch(&dev, &d)) die("NeoGPU submit");
            if (hs_v3d_bo_wait(&dev, &data, 5000000000ull)) die("NeoGPU wait");
            double a = (double)(now_ns() - t0);
            t0 = now_ns();
            for (int i = 0; i < J; i++) glDispatchCompute(1, 1, 1);
            glFinish();
            double b = (double)(now_ns() - t0);
            if (t) tn[t - 1] = a, tm[t - 1] = b;
        }
        double a = pct(tn, 7, 0.5), b = pct(tm, 7, 0.5);
        printf("%6d %14.0f %14.0f %7.2fx\n", J, J / a * 1e9, J / b * 1e9, b / a);
    }
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    uint32_t* mr = mesa_map(mb, 4, GL_MAP_READ_BIT);
    int ok = *mr == 1 && *(volatile uint32_t*)(data.map + D_WORD) == 1;
    mesa_unmap(mb);
    printf("words written by the last jobs: %s\n", ok ? "ok" : "WRONG");
    g_wrong |= !ok;
    glDeleteBuffers(1, &mb);
    glDeleteProgram(mp);
}

/* ----------------------------------------------------------- g: gather */
typedef struct {
    char name[48];
    int is_mesa, a, b; /* NeoGPU: batches, k. Mesa: local size, elements per invocation */
    uint32_t gpu;      /* NeoGPU program */
    uint32_t uoff;     /* NeoGPU uniform offset */
    GLuint prog;       /* Mesa program */
    double ns[16];
    int ok;
} gcfg_t;

static void section_gather(int trials) {
    printf("\n## Gather: out[i] = table[idx[i]], %u elements of 4 B; every output word checked on every trial\n", N);
    printf("median of %d interleaved trials, each after an untimed run of the same job; M elem/s\n", trials);
    printf("Mesa GL_MAX_SHADER_STORAGE_BLOCK_SIZE = %d bytes\n", max_ssbo);
    uint32_t* nt = (uint32_t*)(data.map + D_TBL);
    uint32_t* ni = (uint32_t*)(data.map + D_IDX);
    uint32_t* no = (uint32_t*)(data.map + D_OUT);
    uint32_t* ct = halloc((size_t)TMAX * 4);
    uint32_t* ci = halloc((size_t)N * 4);
    uint32_t* co = halloc((size_t)N * 4);
    for (uint32_t i = 0; i < TMAX; i++) nt[i] = ct[i] = hash32(i);
    GLuint mt = mesa_buf((size_t)TMAX * 4, ct), mi = mesa_buf((size_t)N * 4, NULL), mo = mesa_buf((size_t)N * 4, NULL);

    gcfg_t cf[32];
    int nc = 0;
    for (int b = 1; b <= 8; b *= 8)
        for (int k = 1; k <= 8; k *= 2) {
            gcfg_t* c = &cf[nc];
            memset(c, 0, sizeof(*c));
            snprintf(c->name, sizeof(c->name), "NeoGPU %d batch%s, k=%d", b, b > 1 ? "es" : "", k);
            c->a = b, c->b = k;
            c->gpu = neo_load(hs_qpu_kernel_gather, b, k);
            c->uoff = 256u + 16u * (uint32_t)nc;
            if (hs_qpu_gather_uniforms((uint32_t*)(unif.map + c->uoff), N, b, k, data.gpu + D_IDX, data.gpu + D_TBL,
                                       data.gpu + D_OUT))
                die("gather uniforms");
            nc++;
        }
    static const int LS[] = {16, 64, 128, 256}, EL[] = {1, 4, 16};
    for (int x = 0; x < 4; x++)
        for (int y = 0; y < 3; y++) {
            gcfg_t* c = &cf[nc++];
            memset(c, 0, sizeof(*c));
            snprintf(c->name, sizeof(c->name), "Mesa local %d, %d elem/inv", LS[x], EL[y]);
            c->is_mesa = 1, c->a = LS[x], c->b = EL[y];
            c->prog = mesa_gather_program(LS[x], EL[y]);
        }

    static const struct { const char* name; uint32_t entries; int seq; } CASE[] = {
        {"sequential idx, 64 MiB table", TMAX, 1},
        {"random idx, 512 KiB table", 128u << 10, 0},
        {"random idx, 64 MiB table", TMAX, 0},
    };
    for (size_t cs = 0; cs < 3; cs++) {
        uint64_t rs = 0x9E3779B97F4A7C15ull;
        for (uint32_t i = 0; i < N; i++) {
            rs ^= rs << 13, rs ^= rs >> 7, rs ^= rs << 17;
            ni[i] = ci[i] = CASE[cs].seq ? i % CASE[cs].entries : (uint32_t)(rs % CASE[cs].entries);
        }
        uint32_t* m = mesa_map(mi, (size_t)N * 4, GL_MAP_WRITE_BIT);
        memcpy(m, ci, (size_t)N * 4);
        mesa_unmap(mi);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, mt);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mi);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mo);
        for (int c = 0; c < nc; c++) cf[c].ok = 1;
        double cpu_ns[4][16];
        int cpu_ok[4] = {1, 1, 1, 1};
        for (int t = 0; t < trials; t++) {
            for (int c = 0; c < nc; c++) {
                gcfg_t* g = &cf[c];
                if (!g->is_mesa) {
                    if (neo_run(g->gpu, g->uoff, (uint32_t)g->a)) die("NeoGPU gather warm-up");
                    for (uint32_t i = 0; i < N; i++) no[i] = SENTINEL;
                    uint64_t t0 = now_ns();
                    if (neo_run(g->gpu, g->uoff, (uint32_t)g->a)) die("NeoGPU gather");
                    g->ns[t] = (double)(now_ns() - t0);
                    for (uint32_t i = 0; i < N; i++) g->ok &= no[i] == hash32(ni[i]);
                } else {
                    uint32_t inv = N / (uint32_t)g->b, groups = inv / (uint32_t)g->a;
                    uint32_t gx = groups < 1024 ? groups : 1024, gy = groups / gx;
                    glUseProgram(g->prog);
                    glUniform1ui(glGetUniformLocation(g->prog, "total"), inv);
                    glDispatchCompute(gx, gy, 1);
                    glFinish();
                    uint32_t* o = mesa_map(mo, (size_t)N * 4, GL_MAP_WRITE_BIT);
                    for (uint32_t i = 0; i < N; i++) o[i] = SENTINEL;
                    mesa_unmap(mo);
                    glFinish();
                    uint64_t t0 = now_ns();
                    glDispatchCompute(gx, gy, 1);
                    glFinish();
                    g->ns[t] = (double)(now_ns() - t0);
                    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
                    o = mesa_map(mo, (size_t)N * 4, GL_MAP_READ_BIT);
                    for (uint32_t i = 0; i < N; i++) g->ok &= o[i] == hash32(ci[i]);
                    mesa_unmap(mo);
                }
            }
            for (int v = 0; v < 4; v++) { /* 1 thread, 4 threads; prefetch off/on */
                memset(co, 0xEE, (size_t)N * 4);
                cpu_ns[v][t] = cpu_gather(ct, ci, co, v & 2 ? 4 : 1, v & 1);
                for (uint32_t i = 0; i < N; i++) cpu_ok[v] &= co[i] == hash32(ci[i]);
                pin(0);
            }
        }
        printf("\n### %s\n%-34s %10s %12s  %s\n", CASE[cs].name, "path", "ms", "M elem/s", "check");
        double best[3] = {1e30, 1e30, 1e30};
        const char* bestn[3] = {"", "", ""};
        int all_ok = 1;
        for (int c = 0; c < nc; c++) {
            double med = pct(cf[c].ns, trials, 0.5);
            printf("%-34s %10.2f %12.1f  %s\n", cf[c].name, med / 1e6, N / med * 1e3, cf[c].ok ? "ok" : "WRONG");
            all_ok &= cf[c].ok;
            int w = cf[c].is_mesa;
            if (cf[c].ok && med < best[w]) best[w] = med, bestn[w] = cf[c].name;
        }
        static const char* CN[4] = {"CPU 1 core", "CPU 1 core, prefetch", "CPU 4 cores", "CPU 4 cores, prefetch"};
        for (int v = 0; v < 4; v++) {
            double med = pct(cpu_ns[v], trials, 0.5);
            printf("%-34s %10.2f %12.1f  %s\n", CN[v], med / 1e6, N / med * 1e3, cpu_ok[v] ? "ok" : "WRONG");
            all_ok &= cpu_ok[v];
            if (cpu_ok[v] && med < best[2]) best[2] = med, bestn[2] = CN[v];
        }
        printf("best: NeoGPU %.1f M/s (%s) | Mesa %.1f M/s (%s) | CPU %.1f M/s (%s)\n", N / best[0] * 1e3, bestn[0],
               N / best[1] * 1e3, bestn[1], N / best[2] * 1e3, bestn[2]);
        printf("NeoGPU vs Mesa %.2fx, NeoGPU vs CPU %.2fx%s\n", best[1] / best[0], best[2] / best[0],
               all_ok ? "" : "  (SOME PATH WRONG)");
        g_wrong |= !all_ok;
    }
    for (int c = 0; c < nc; c++)
        if (cf[c].is_mesa) glDeleteProgram(cf[c].prog);
    glDeleteBuffers(1, &mt), glDeleteBuffers(1, &mi), glDeleteBuffers(1, &mo);
    munmap(ct, (size_t)TMAX * 4), munmap(ci, (size_t)N * 4), munmap(co, (size_t)N * 4);
}

/* ------------------------------------------------- cold: first checked result */
static int cold(const char* which) {
    uint64_t t0 = now_ns();
    int ok;
    if (!strcmp(which, "neo")) {
        hs_v3d d;
        hs_v3d_bo c, u, w;
        if (hs_v3d_open(&d, NULL) || hs_v3d_bo_new(&d, &c, 4096) || hs_v3d_bo_new(&d, &u, 4096) ||
            hs_v3d_bo_new(&d, &w, 4096))
            die("NeoGPU init");
        hs_qpu_prog p;
        hs_qpu_prog_init(&p, progbuf, 1024);
        int n = hs_qpu_kernel_store_one(&p);
        uint32_t gpu = n > 0 ? hs_v3d_load(&c, 0, progbuf, (uint32_t)n) : 0;
        ((uint32_t*)u.map)[0] = w.gpu;
        ((volatile uint32_t*)w.map)[0] = 0;
        uint32_t h[3] = {c.handle, u.handle, w.handle};
        hs_v3d_dispatch_desc dd = {gpu, u.gpu, 1, h, 3, 0};
        if (!gpu || hs_v3d_dispatch(&d, &dd) || hs_v3d_bo_wait(&d, &w, 5000000000ull)) die("NeoGPU first job");
        ok = ((volatile uint32_t*)w.map)[0] == 1;
        double ns = (double)(now_ns() - t0);
        printf("cold neo %.3f ms %s\n", ns / 1e6, ok ? "ok" : "WRONG");
        hs_v3d_bo_free(&d, &w), hs_v3d_bo_free(&d, &u), hs_v3d_bo_free(&d, &c);
        hs_v3d_close(&d);
    } else if (!strcmp(which, "mesa")) {
        mesa_init();
        GLuint p = mesa_program(STORE_ONE_GLSL);
        uint32_t zero = 0;
        GLuint b = mesa_buf(4, &zero);
        glUseProgram(p);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, b);
        glDispatchCompute(1, 1, 1);
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        uint32_t* r = mesa_map(b, 4, GL_MAP_READ_BIT);
        ok = *r == 1;
        double ns = (double)(now_ns() - t0);
        mesa_unmap(b);
        printf("cold mesa %.3f ms %s\n", ns / 1e6, ok ? "ok" : "WRONG");
        mesa_fini();
    } else
        die("cold neo|mesa");
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    pin(0);
    if (argc >= 3 && !strcmp(argv[1], "cold")) return cold(argv[2]);
    const char* which = argc > 1 ? argv[1] : "lpg";
    int trials = argc > 2 ? atoi(argv[2]) : 7;
    if (trials < 1 || trials > 16) die("trials must be 1..16");
    neo_init();
    mesa_init();
    printf("# NeoGPU direct (hs_v3d + hs_qpu_kernels) vs Mesa %s / %s, same V3D\n",
           (const char*)glGetString(GL_VERSION), (const char*)glGetString(GL_RENDERER));
    if (strchr(which, 'l')) section_latency();
    if (strchr(which, 'p')) section_pipelined();
    if (strchr(which, 'g')) section_gather(trials);
    mesa_fini();
    hs_v3d_bo_free(&dev, &data), hs_v3d_bo_free(&dev, &unif), hs_v3d_bo_free(&dev, &code);
    hs_v3d_close(&dev);
    if (g_wrong) printf("FAILED: a path produced wrong output\n");
    return g_wrong ? 1 : 0;
}
