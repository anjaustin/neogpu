/* Diagnostic (P0-1, docs/audit/HARDWARE_HANDOFF_P0.md): where the production
 * present path (hs_graphics_present, vsync off by default) spends wall time
 * and CPU time, whether the calling thread sleeps or spins, how busy the GPU
 * is, and which frame each flip shows.
 *
 * The workload is the pacing_sweep shader, calibrated to a fraction of a
 * refresh. Every EGL, GBM and DRM call the present path makes is wrapped:
 * the system headers are included first, then macros redirect the calls in
 * hs_graphics.h (header-only) to timed wrappers, and the EGL fence function
 * pointers are swapped for wrappers after init. Production code is unchanged.
 *
 * Phases (wall time per frame, CLOCK_MONOTONIC, ~45 ns per read):
 *   submit   glUniform + glDrawArrays (the frame's GL calls)
 *   swap     eglSwapBuffers
 *   lock     gbm_surface_lock_front_buffer
 *   addfb    drmModeAddFB / drmModeAddFB2WithModifiers (first use of a buffer)
 *   flip     drmModePageFlip (the ioctl)
 *   wait     poll + drmHandleEvent on the DRM fd (flip completion)
 *   release  gbm_surface_release_buffer
 *   sync     eglCreateSyncKHR + eglClientWaitSyncKHR + eglDestroySyncKHR
 *   interval eglSwapInterval
 *   other    the rest of hs_graphics_present (clock_gettime, arithmetic)
 *
 * CPU:
 *   thread  CLOCK_THREAD_CPUTIME_ID read once at each end of the timed loop
 *   process getrusage (all threads)
 *   kthread run time of every kernel thread, minus their idle rate, (kworkers doing KMS commits and
 *           drm_sched work, IRQ threads), /proc/<pid>/schedstat: CPU spent
 *           in other contexts, which unrelated user processes cannot pollute
 *   system  /proc/schedstat run time summed over every CPU (every task),
 *           minus the idle rate measured over 500 ms before the settling
 *           frames (noisier: any background process lands in it)
 * Hard-IRQ time is not accounted separately on this kernel (/proc/stat irq
 * reads 0); it is charged to whichever task it interrupts.
 * With PROBE_PHASE_CPU=1 each phase also gets thread CPU. That costs a
 * syscall per read (~1.7 us), so the per-call overhead is calibrated at start
 * and subtracted, and is reported.
 *
 * GPU: v3d engine time from /proc/self/fdinfo (drm-engine-bin/render).
 * Flips: each present that queues a flip shows either the frame it just
 * locked ("newest") or an older held frame ("held").
 *
 * Usage: sudo tests/present_cost_probe <fraction_of_refresh> [frames=600] [vsync]
 *        PROBE_FLIPS=<path> writes "queued_ns landed_vblank_ns" per flip.
 * Output: one JSON line. */
#define _GNU_SOURCE
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <dirent.h>
#include <gbm.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

enum { P_SUBMIT, P_SWAP, P_LOCK, P_ADDFB, P_FLIP, P_WAIT, P_RELEASE, P_SYNC, P_INTERVAL, P_PRESENT, P_CAL, NP };
static const char* PN[NP] = {"submit", "swap",  "lock",     "addfb",   "flip", "wait",
                             "release", "sync", "interval", "present", "cal"};
static double wall_ns[NP], cpu_ns[NP];
static long calls[NP];
static int timing, phase_cpu;

static double clk(clockid_t c) {
    struct timespec t;
    clock_gettime(c, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}
typedef struct { double w, c; } stamp_t;
static stamp_t t_start(void) {
    stamp_t s = {clk(CLOCK_MONOTONIC), phase_cpu ? clk(CLOCK_THREAD_CPUTIME_ID) : 0};
    return s;
}
static void t_end(int p, stamp_t s) {
    if (!timing) return;
    if (phase_cpu) cpu_ns[p] += clk(CLOCK_THREAD_CPUTIME_ID) - s.c;
    wall_ns[p] += clk(CLOCK_MONOTONIC) - s.w;
    calls[p]++;
}

static struct gbm_bo* last_locked;
static uint32_t last_flip_fb; /* fb id of the last flip queued */
/* per flip (PROBE_FLIPS=path): when it was queued and the vblank it landed on */
#define MAXF 8192
static double fq[MAXF], fland[MAXF];
static int nfq, nfland;
static const unsigned long long* p_last_flip_ns; /* &gfx.last_flip_ns */
static EGLBoolean w_swap(EGLDisplay d, EGLSurface s) {
    stamp_t t = t_start();
    EGLBoolean r = eglSwapBuffers(d, s);
    t_end(P_SWAP, t);
    return r;
}
static EGLBoolean w_interval(EGLDisplay d, EGLint i) {
    stamp_t t = t_start();
    EGLBoolean r = eglSwapInterval(d, i);
    t_end(P_INTERVAL, t);
    return r;
}
static struct gbm_bo* w_lock(struct gbm_surface* s) {
    stamp_t t = t_start();
    struct gbm_bo* r = gbm_surface_lock_front_buffer(s);
    t_end(P_LOCK, t);
    last_locked = r;
    return r;
}
static int w_addfb(int fd, uint32_t w, uint32_t h, uint8_t depth, uint8_t bpp, uint32_t pitch, uint32_t bo,
                   uint32_t* id) {
    stamp_t t = t_start();
    int r = drmModeAddFB(fd, w, h, depth, bpp, pitch, bo, id);
    t_end(P_ADDFB, t);
    return r;
}
static int w_addfb2(int fd, uint32_t w, uint32_t h, uint32_t fmt, const uint32_t hs[4], const uint32_t ps[4],
                    const uint32_t os[4], const uint64_t ms[4], uint32_t* id, uint32_t flags) {
    stamp_t t = t_start();
    int r = drmModeAddFB2WithModifiers(fd, w, h, fmt, hs, ps, os, ms, id, flags);
    t_end(P_ADDFB, t);
    return r;
}
static int w_flip(int fd, uint32_t crtc, uint32_t fb, uint32_t flags, void* data) {
    stamp_t t = t_start();
    int r = drmModePageFlip(fd, crtc, fb, flags, data);
    t_end(P_FLIP, t);
    last_flip_fb = fb;
    if (timing && r == 0 && nfq < MAXF) fq[nfq++] = t.w;
    return r;
}
static int w_poll(struct pollfd* f, nfds_t n, int ms) {
    stamp_t t = t_start();
    int r = poll(f, n, ms);
    t_end(P_WAIT, t);
    return r;
}
static int w_event(int fd, drmEventContextPtr e) {
    unsigned long long before = p_last_flip_ns ? *p_last_flip_ns : 0;
    stamp_t t = t_start();
    int r = drmHandleEvent(fd, e);
    t_end(P_WAIT, t);
    if (p_last_flip_ns && *p_last_flip_ns != before && nfland < nfq && nfland < MAXF)
        fland[nfland++] = (double)*p_last_flip_ns;
    return r;
}
static void w_release(struct gbm_surface* s, struct gbm_bo* bo) {
    stamp_t t = t_start();
    gbm_surface_release_buffer(s, bo);
    t_end(P_RELEASE, t);
}
#define eglSwapBuffers w_swap
#define eglSwapInterval w_interval
#define gbm_surface_lock_front_buffer w_lock
#define drmModeAddFB w_addfb
#define drmModeAddFB2WithModifiers w_addfb2
#define drmModePageFlip w_flip
#define poll w_poll
#define drmHandleEvent w_event
#define gbm_surface_release_buffer w_release
#include "hs_graphics.h"
#undef eglSwapBuffers
#undef poll

/* the EGL fence calls go through function pointers in HSGraphics */
static PFNEGLCREATESYNCKHRPROC real_create;
static PFNEGLDESTROYSYNCKHRPROC real_destroy;
static PFNEGLCLIENTWAITSYNCKHRPROC real_wait;
static EGLSyncKHR EGLAPIENTRY w_create(EGLDisplay d, EGLenum type, const EGLint* a) {
    stamp_t t = t_start();
    EGLSyncKHR r = real_create(d, type, a);
    t_end(P_SYNC, t);
    return r;
}
static EGLBoolean EGLAPIENTRY w_destroy(EGLDisplay d, EGLSyncKHR s) {
    stamp_t t = t_start();
    EGLBoolean r = real_destroy(d, s);
    t_end(P_SYNC, t);
    return r;
}
static EGLint EGLAPIENTRY w_wait(EGLDisplay d, EGLSyncKHR s, EGLint f, EGLTimeKHR to) {
    stamp_t t = t_start();
    EGLint r = real_wait(d, s, f, to);
    t_end(P_SYNC, t);
    return r;
}

static const char* VS = "attribute vec2 p;\nvoid main() { gl_Position = vec4(p, 0.0, 1.0); }\n";
static const char* FS =
    "precision highp float;\nuniform int n;\nuniform float t;\n"
    "void main() {\n    vec2 q = gl_FragCoord.xy * 0.01 + t;\n    float a = 0.0;\n"
    "    for (int i = 0; i < 4096; i++) {\n        if (i >= n) break;\n"
    "        a += sin(q.x + float(i) * 0.37) * cos(q.y - a);\n    }\n"
    "    gl_FragColor = vec4(fract(a), fract(t), 0.5, 1.0);\n}\n";

static GLuint shader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    return s;
}
static double tv_ns(struct timeval t) { return t.tv_sec * 1e9 + t.tv_usec * 1e3; }
static int cmpd(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}

/* GPU engine time this process used (v3d fdinfo), summed over its DRM fds */
enum { E_BIN, E_RENDER, NE };
static void gpu_ns(double* e, double* jobs) {
    e[E_BIN] = e[E_RENDER] = jobs[E_BIN] = jobs[E_RENDER] = 0;
    DIR* d = opendir("/proc/self/fdinfo");
    if (!d) return;
    struct dirent* de;
    while ((de = readdir(d))) {
        char path[300], line[256];
        snprintf(path, sizeof(path), "/proc/self/fdinfo/%s", de->d_name);
        FILE* f = fopen(path, "r");
        if (!f) continue;
        while (fgets(line, sizeof(line), f)) {
            double v;
            if (sscanf(line, "drm-engine-bin: %lf", &v) == 1) e[E_BIN] += v;
            else if (sscanf(line, "drm-engine-render: %lf", &v) == 1) e[E_RENDER] += v;
            else if (sscanf(line, "v3d-jobs-bin: %lf", &v) == 1) jobs[E_BIN] += v;
            else if (sscanf(line, "v3d-jobs-render: %lf", &v) == 1) jobs[E_RENDER] += v;
        }
        fclose(f);
    }
    closedir(d);
}

/* every task's run time on every CPU (ns), from /proc/schedstat */
static double sched_run_ns(void) {
    FILE* f = fopen("/proc/schedstat", "r");
    if (!f) return -1;
    char line[512];
    double sum = 0;
    while (fgets(line, sizeof(line), f)) {
        unsigned long long v[9];
        if (!strncmp(line, "cpu", 3) &&
            sscanf(line, "cpu%*d %llu %llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2], &v[3], &v[4],
                   &v[5], &v[6], &v[7], &v[8]) == 9)
            sum += (double)v[6];
    }
    fclose(f);
    return sum;
}
/* run time (ns) of every kernel thread: kthreadd (pid 2) and its children.
 * With snap/nsnap given, per-PID values are stored; with base/nbase given,
 * the result is the sum of per-PID deltas against base (a thread new since
 * base counts in full). Summing raw totals is wrong: a kworker that exits
 * during the window takes its whole accumulated run time out of the sum
 * (measured: -781 us per frame in one run). Threads that exit are lost. */
#define MAXK 4096
typedef struct { int pid; double run; } krun_t;
static int kexited;
static double kthread_scan(krun_t* snap, int* nsnap, const krun_t* base, int nbase) {
    DIR* d = opendir("/proc");
    if (!d) return -1;
    struct dirent* de;
    double sum = 0;
    while ((de = readdir(d))) {
        if (de->d_name[0] < '0' || de->d_name[0] > '9') continue;
        char path[300], buf[512];
        snprintf(path, sizeof(path), "/proc/%s/stat", de->d_name);
        FILE* f = fopen(path, "r");
        if (!f) continue;
        size_t k = fread(buf, 1, sizeof(buf) - 1, f);
        fclose(f);
        buf[k] = 0;
        const char* rp = strrchr(buf, ')'); /* comm may contain spaces */
        int ppid = 0;
        if (!rp || sscanf(rp + 2, "%*c %d", &ppid) != 1) continue;
        if (ppid != 2 && strcmp(de->d_name, "2")) continue;
        snprintf(path, sizeof(path), "/proc/%s/schedstat", de->d_name);
        f = fopen(path, "r");
        if (!f) continue;
        unsigned long long run = 0;
        int ok = fscanf(f, "%llu", &run) == 1;
        fclose(f);
        if (!ok) continue;
        int pid = atoi(de->d_name);
        if (snap && *nsnap < MAXK) snap[(*nsnap)++] = (krun_t){pid, (double)run};
        double prev = 0;
        for (int i = 0; base && i < nbase; i++)
            if (base[i].pid == pid) { prev = base[i].run; break; }
        sum += (double)run - prev;
    }
    closedir(d);
    if (base) { /* count base threads that are gone */
        kexited = 0;
        for (int i = 0; i < nbase; i++) {
            char path[64];
            snprintf(path, sizeof(path), "/proc/%d", base[i].pid);
            if (access(path, F_OK)) kexited++;
        }
    }
    return sum;
}
static krun_t kbase[MAXK];
static int nkbase;
static void kthread_mark(void) { nkbase = 0; kthread_scan(kbase, &nkbase, NULL, 0); }
static double kthread_since_mark(void) { return kthread_scan(NULL, NULL, kbase, nkbase); }

int main(int argc, char** argv) {
    double fraction = argc > 1 ? atof(argv[1]) : 0.5;
    int frames = argc > 2 ? atoi(argv[2]) : 600;
    bool vsync = argc > 3 && strcmp(argv[3], "vsync") == 0;
    phase_cpu = getenv("PROBE_PHASE_CPU") && atoi(getenv("PROBE_PHASE_CPU"));
    static HSGraphics gfx;
    if (hs_graphics_init(&gfx) != 0) return 1;
    p_last_flip_ns = (const unsigned long long*)&gfx.last_flip_ns;
    if (gfx.egl_create_sync) {
        real_create = gfx.egl_create_sync, real_destroy = gfx.egl_destroy_sync, real_wait = gfx.egl_client_wait_sync;
        gfx.egl_create_sync = w_create, gfx.egl_destroy_sync = w_destroy, gfx.egl_client_wait_sync = w_wait;
    }
    /* the wrappers' own cost: an empty timed region, 20000 times */
    timing = 1;
    for (int i = 0; i < 20000; i++) t_end(P_CAL, t_start());
    double cal_wall = wall_ns[P_CAL] / calls[P_CAL], cal_cpu = phase_cpu ? cpu_ns[P_CAL] / calls[P_CAL] : 0;
    timing = 0;

    GLuint prog = glCreateProgram();
    glAttachShader(prog, shader(GL_VERTEX_SHADER, VS));
    glAttachShader(prog, shader(GL_FRAGMENT_SHADER, FS));
    glBindAttribLocation(prog, 0, "p");
    glLinkProgram(prog);
    static const float quad[] = {-1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1};
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glUseProgram(prog);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    GLint n_loc = glGetUniformLocation(prog, "n"), t_loc = glGetUniformLocation(prog, "t");
    glUniform1i(n_loc, 1);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    hs_graphics_present(&gfx);
    double refresh_ms = gfx.display_mode.vrefresh ? 1000.0 / gfx.display_mode.vrefresh : 1000.0 / 60;
    int n = 8;
    double frame_ms = 0, want = fraction * refresh_ms;
    for (int round = 0; round < 16; round++) {
        glUniform1i(n_loc, n);
        glFinish();
        double t0 = clk(CLOCK_MONOTONIC);
        for (int i = 0; i < 8; i++) glDrawArrays(GL_TRIANGLES, 0, 6);
        glFinish();
        frame_ms = (clk(CLOCK_MONOTONIC) - t0) / 8e6;
        if (frame_ms > want * 0.95 && frame_ms < want * 1.05) break;
        int next = (int)(n * want / (frame_ms > 0.01 ? frame_ms : 0.01));
        n = next < 1 ? 1 : next > 4096 ? 4096 : next;
    }
    glUniform1i(n_loc, n);
    gfx.vsync = vsync;
    /* idle rate of the whole system. Measured before the settling frames:
     * a gap here would enter the present path's frame-interval average and
     * switch on its wait-for-flip branch (measured: 125 -> 116 FPS at 0.5),
     * and would let the GPU clock drop. */
    /* median of 10 windows of 100 ms: one window catches background bursts
     * (measured: a single 500 ms window gave baselines off by up to 1 ms per
     * frame) */
    double ir[10], kr[10];
    for (int w = 0; w < 10; w++) {
        double s_a = sched_run_ns(), w_a = clk(CLOCK_MONOTONIC);
        kthread_mark();
        usleep(100000);
        double dk = kthread_since_mark(), dw = clk(CLOCK_MONOTONIC) - w_a;
        ir[w] = (sched_run_ns() - s_a) / dw;
        kr[w] = dk / dw;
    }
    qsort(ir, 10, sizeof(double), cmpd);
    qsort(kr, 10, sizeof(double), cmpd);
    double idle_rate = (ir[4] + ir[5]) / 2;  /* CPU-ns per ns, every task */
    double kidle_rate = (kr[4] + kr[5]) / 2; /* kernel threads only */
    for (int i = 0; i < 60; i++) { /* settle into the steady state, untimed */
        glUniform1f(t_loc, (float)i * 0.01f);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        hs_graphics_present(&gfx);
    }

    u64 presented0 = gfx.frames_presented, dropped0 = gfx.frames_dropped, rep0 = gfx.refreshes_repeated;
    long flips_newest = 0, flips_held = 0;
    struct rusage r0, r1;
    double g0[NE], j0[NE], g1[NE], j1[NE];
    gpu_ns(g0, j0);
    double s0 = sched_run_ns();
    kthread_mark();
    getrusage(RUSAGE_SELF, &r0);
    double tc0 = clk(CLOCK_THREAD_CPUTIME_ID);
    timing = 1;
    double start = clk(CLOCK_MONOTONIC);
    for (int i = 0; i < frames; i++) {
        stamp_t s = t_start();
        glUniform1f(t_loc, (float)(i + 60) * 0.01f);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        t_end(P_SUBMIT, s);
        long flips_before = calls[P_FLIP];
        last_locked = NULL;
        s = t_start();
        hs_graphics_present(&gfx);
        t_end(P_PRESENT, s);
        if (calls[P_FLIP] > flips_before) {
            /* by framebuffer id: with vsync on the flip is already retired
             * when present returns, so pending_bo cannot tell (red team 2) */
            const HSScanoutFb* sfb = last_locked ? (const HSScanoutFb*)gbm_bo_get_user_data(last_locked) : NULL;
            if (sfb && sfb->fb_id == last_flip_fb) flips_newest++;
            else flips_held++;
        }
    }
    double wall = clk(CLOCK_MONOTONIC) - start;
    timing = 0;
    double tc1 = clk(CLOCK_THREAD_CPUTIME_ID);
    getrusage(RUSAGE_SELF, &r1);
    double kdelta = kthread_since_mark(), s1 = sched_run_ns();
    gpu_ns(g1, j1);

    double proc_cpu = tv_ns(r1.ru_utime) - tv_ns(r0.ru_utime) + tv_ns(r1.ru_stime) - tv_ns(r0.ru_stime);
    double sys_cpu = (s1 - s0) - idle_rate * wall;
    printf("{\"fraction\":%.3f,\"shader_iterations\":%d,\"calibrated_gpu_ms\":%.3f,\"vsync\":%s,\"frames\":%d,"
           "\"fps\":%.1f,\"wall_us_per_frame\":%.1f,",
           fraction, n, frame_ms, vsync ? "true" : "false", frames, frames * 1e9 / wall, wall / frames / 1e3);
    printf("\"thread_cpu_us_per_frame\":%.1f,\"proc_cpu_us_per_frame\":%.1f,\"proc_user_us\":%.1f,\"proc_sys_us\":%.1f,"
           "\"system_cpu_us_per_frame\":%.1f,\"system_minus_proc_us_per_frame\":%.1f,\"idle_cpu_per_s_ms\":%.2f,"
           "\"kthread_cpu_us_per_frame\":%.1f,\"kthread_idle_per_s_ms\":%.2f,\"kthreads_exited\":%d,",
           (tc1 - tc0) / frames / 1e3, proc_cpu / frames / 1e3, (tv_ns(r1.ru_utime) - tv_ns(r0.ru_utime)) / frames / 1e3,
           (tv_ns(r1.ru_stime) - tv_ns(r0.ru_stime)) / frames / 1e3, sys_cpu / frames / 1e3,
           (sys_cpu - proc_cpu) / frames / 1e3, idle_rate * 1e3, (kdelta - kidle_rate * wall) / frames / 1e3, kidle_rate * 1e3, kexited);
    printf("\"vcsw_per_frame\":%.2f,\"ivcsw_per_frame\":%.2f,", (double)(r1.ru_nvcsw - r0.ru_nvcsw) / frames,
           (double)(r1.ru_nivcsw - r0.ru_nivcsw) / frames);
    printf("\"gpu_render_us_per_frame\":%.1f,\"gpu_bin_us_per_frame\":%.1f,\"render_jobs_per_frame\":%.2f,"
           "\"gpu_render_busy\":%.4f,\"gpu_busy_resolution\":%.4f,",
           (g1[E_RENDER] - g0[E_RENDER]) / frames / 1e3, (g1[E_BIN] - g0[E_BIN]) / frames / 1e3,
           (j1[E_RENDER] - j0[E_RENDER]) / frames, (g1[E_RENDER] - g0[E_RENDER]) / wall,
           /* fdinfo credits a job when it completes: a job in flight at either
            * end of the window moves the figure by up to one job */
           (g1[E_RENDER] - g0[E_RENDER]) / (j1[E_RENDER] - j0[E_RENDER] > 0 ? j1[E_RENDER] - j0[E_RENDER] : 1) / wall);
    printf("\"presented\":%llu,\"dropped\":%llu,\"refreshes_repeated\":%llu,\"flips_newest\":%ld,\"flips_held\":%ld,"
           "\"errors\":%u,",
           (unsigned long long)(gfx.frames_presented - presented0), (unsigned long long)(gfx.frames_dropped - dropped0),
           (unsigned long long)(gfx.refreshes_repeated - rep0), flips_newest, flips_held, gfx.present_errors);
    printf("\"phase_cpu\":%s,\"wrapper_overhead_wall_ns\":%.1f,\"wrapper_overhead_cpu_ns\":%.1f",
           phase_cpu ? "true" : "false", cal_wall, cal_cpu);
    /* per phase, net of the wrappers' own cost; nested wrappers inside
     * present are charged to their own phase, not to "other" */
    double other_w = wall_ns[P_PRESENT] - calls[P_PRESENT] * cal_wall;
    double other_c = cpu_ns[P_PRESENT] - calls[P_PRESENT] * cal_cpu;
    for (int p = 0; p < P_CAL; p++) {
        double w = wall_ns[p] - calls[p] * cal_wall, c = cpu_ns[p] - calls[p] * cal_cpu;
        if (p != P_PRESENT && p != P_SUBMIT) {
            /* the nested wrapper's full cost (net time plus its own overhead) leaves "other" */
            other_w -= wall_ns[p];
            other_c -= cpu_ns[p];
        }
        if (p == P_PRESENT) continue;
        printf(",\"%s\":{\"wall_us\":%.2f,", PN[p], w / frames / 1e3);
        if (phase_cpu) printf("\"cpu_us\":%.2f,", c / frames / 1e3);
        printf("\"calls_per_frame\":%.3f}", (double)calls[p] / frames);
    }
    printf(",\"present_total\":{\"wall_us\":%.2f}", (wall_ns[P_PRESENT] - calls[P_PRESENT] * cal_wall) / frames / 1e3);
    printf(",\"present_other\":{\"wall_us\":%.2f", other_w / frames / 1e3);
    if (phase_cpu) printf(",\"cpu_us\":%.2f", other_c / frames / 1e3);
    printf("}}\n");
    /* flips queued in the window that landed in it, in order (one in flight) */
    const char* fp = getenv("PROBE_FLIPS");
    FILE* ff = fp ? fopen(fp, "w") : NULL;
    if (ff) {
        int m = nfland < nfq ? nfland : nfq;
        for (int i = 0; i < m; i++) fprintf(ff, "%.0f %.0f\n", fq[i], fland[i]);
        fclose(ff);
    }
    hs_graphics_finish(&gfx);
    return 0;
}
