/* P0-1 red team 2 (docs/audit/HARDWARE_HANDOFF_P0.md): what a fence-driven
 * flip of the newest frame does, measured against the production policy.
 * POLICY=newest makes hs_graphics_present's "has the GPU finished this
 * frame?" fence check always answer yes, so every flip queues the frame just
 * swapped, unfinished, and the kernel holds it on its implicit v3d fence
 * (the same wait an atomic commit with IN_FENCE_FD gets). POLICY=prod leaves
 * production unchanged. Per flip it records the queue time, the page-flip
 * event timestamp, and the frame shown; frame age = event timestamp minus
 * the CLOCK_MONOTONIC time that frame's draw was issued.
 * HOLD_PERMILLE=n overrides gfx.hold_permille, the hold cutoff in thousandths
 * of a refresh (1000000: never hold). Used by the hold-cutoff sweep.
 * Usage: sudo tests/p0_1_redteam2_newest <fraction_of_refresh> [frames=600]
 * Output: one JSON line. */
#define _GNU_SOURCE
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <gbm.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

static double clk(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}
#define MAXF 8192
static int timing, cur_frame = -1;
static struct gbm_bo* bo_of[8];
static int frame_of[8], nbo;
static double draw_t[MAXF];
static double fq[MAXF], fland[MAXF];
static int fshown[MAXF], fnewest[MAXF], nfq, nfland;
static const unsigned long long* p_last;
static struct gbm_bo* w_lock(struct gbm_surface* s) {
    struct gbm_bo* r = gbm_surface_lock_front_buffer(s);
    int i;
    for (i = 0; i < nbo && bo_of[i] != r; i++) {}
    if (i == nbo && nbo < 8) bo_of[nbo++] = r;
    if (i < 8) frame_of[i] = cur_frame;
    return r;
}
static int bo_frame(struct gbm_bo* b) {
    for (int i = 0; i < nbo; i++) if (bo_of[i] == b) return frame_of[i];
    return -2;
}
static int w_event(int fd, drmEventContextPtr e) {
    unsigned long long before = *p_last;
    int r = drmHandleEvent(fd, e);
    if (*p_last != before && nfland < nfq) fland[nfland++] = (double)*p_last;
    return r;
}
#define gbm_surface_lock_front_buffer w_lock
#define drmHandleEvent w_event
#include "hs_graphics.h"
#undef gbm_surface_lock_front_buffer
#undef drmHandleEvent

static PFNEGLCLIENTWAITSYNCKHRPROC real_wait;
static EGLint EGLAPIENTRY w_wait_yes(EGLDisplay d, EGLSyncKHR s, EGLint f, EGLTimeKHR to) {
    real_wait(d, s, f, to); /* keep the call's cost */
    return EGL_CONDITION_SATISFIED_KHR;
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
static int cmpd(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}
int main(int argc, char** argv) {
    double fraction = argc > 1 ? atof(argv[1]) : 0.5;
    int frames = argc > 2 ? atoi(argv[2]) : 600;
    if (frames > MAXF) frames = MAXF;
    const char* pol = getenv("POLICY") ? getenv("POLICY") : "prod";
    int newest = !strcmp(pol, "newest");
    static HSGraphics gfx;
    if (hs_graphics_init(&gfx) != 0) return 1;
    p_last = (const unsigned long long*)&gfx.last_flip_ns;
    if (getenv("HOLD_PERMILLE")) gfx.hold_permille = (u32)strtoul(getenv("HOLD_PERMILLE"), NULL, 10);
    if (newest && gfx.egl_client_wait_sync) {
        real_wait = gfx.egl_client_wait_sync;
        gfx.egl_client_wait_sync = w_wait_yes;
    }
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
        double t0 = clk();
        for (int i = 0; i < 8; i++) glDrawArrays(GL_TRIANGLES, 0, 6);
        glFinish();
        frame_ms = (clk() - t0) / 8e6;
        if (frame_ms > want * 0.95 && frame_ms < want * 1.05) break;
        int next = (int)(n * want / (frame_ms > 0.01 ? frame_ms : 0.01));
        n = next < 1 ? 1 : next > 4096 ? 4096 : next;
    }
    glUniform1i(n_loc, n);
    gfx.vsync = false;
    for (int i = 0; i < 60; i++) {
        glUniform1f(t_loc, (float)i * 0.01f);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        hs_graphics_present(&gfx);
    }
    u64 rep0 = gfx.refreshes_repeated, pres0 = gfx.frames_presented, drop0 = gfx.frames_dropped;
    timing = 1;
    double start = clk();
    for (int i = 0; i < frames; i++) {
        cur_frame = i;
        draw_t[i] = clk();
        glUniform1f(t_loc, (float)(i + 60) * 0.01f);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        struct gbm_bo* pend_before = gfx.pending_bo;
        bool fp_before = gfx.flip_pending;
        double tq = clk();
        hs_graphics_present(&gfx);
        if (gfx.flip_pending && (!fp_before || gfx.pending_bo != pend_before) && nfq < MAXF) {
            fq[nfq] = tq; /* present entry: an upper bound on how early the flip was queued */
            fshown[nfq] = bo_frame(gfx.pending_bo);
            fnewest[nfq] = fshown[nfq] == i;
            nfq++;
        }
    }
    double wall = clk() - start;
    timing = 0;
    u64 rep = gfx.refreshes_repeated - rep0;
    int m = nfland < nfq ? nfland : nfq, nnew = 0, nage = 0;
    static double age[MAXF];
    for (int k = 0; k < m; k++) {
        nnew += fnewest[k];
        if (fshown[k] >= 0) age[nage++] = (fland[k] - draw_t[fshown[k]]) / 1e3;
    }
    qsort(age, nage, sizeof(double), cmpd);
    double mean = 0;
    for (int k = 0; k < nage; k++) mean += age[k];
    mean = nage ? mean / nage : 0;
    printf("{\"policy\":\"%s\",\"hold_permille\":%u,\"present_interval_ema_us\":%.0f,\"fraction\":%.3f,\"shader_iterations\":%d,\"calibrated_gpu_ms\":%.3f,\"frames\":%d,"
           "\"fps\":%.1f,\"flips\":%d,\"flips_newest\":%d,\"presented\":%llu,\"dropped\":%llu,"
           "\"refreshes_repeated\":%llu,\"refreshes\":%.0f,\"errors\":%u,"
           "\"age_us\":{\"n\":%d,\"min\":%.0f,\"p10\":%.0f,\"median\":%.0f,\"mean\":%.0f,\"p90\":%.0f,\"max\":%.0f}}\n",
           pol, gfx.hold_permille, gfx.present_interval_ns / 1e3, fraction, n, frame_ms, frames, frames * 1e9 / wall, m, nnew,
           (unsigned long long)(gfx.frames_presented - pres0), (unsigned long long)(gfx.frames_dropped - drop0),
           (unsigned long long)rep, wall / (refresh_ms * 1e6), gfx.present_errors, nage, nage ? age[0] : 0,
           nage ? age[nage / 10] : 0, nage ? age[nage / 2] : 0, mean, nage ? age[nage * 9 / 10] : 0,
           nage ? age[nage - 1] : 0);
    hs_graphics_finish(&gfx);
    return 0;
}
