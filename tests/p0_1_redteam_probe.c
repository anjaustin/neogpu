/* Red-team diagnostic for P0-1 (docs/audit/HARDWARE_HANDOFF_P0.md).
 * Production present path, no wrappers.
 *
 * Modes:
 *   timer                      cost of the clock_gettime calls the probe uses
 *   quad <fraction> <spin_us> [frames] [vsync]
 *                              pacing_sweep shader calibrated as in
 *                              present_cost_probe, plus spin_us of CPU work per
 *                              frame before the draw (app work). Tests whether
 *                              CPU work is hidden behind GPU work.
 *   draws <n> <spin_us> [frames] [vsync]
 *                              n small (16x16 px) draws per frame, each with
 *                              glBufferData + glUniform + glDrawArrays (the
 *                              hs_text pattern), trivial shader.
 * Output: one JSON line. */
#define _GNU_SOURCE
#include "hs_graphics.h"
#include <dirent.h>
#include <stdio.h>
#include <sys/resource.h>

static double mono(clockid_t c) {
    struct timespec t;
    clock_gettime(c, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}
static double tv_ns(struct timeval t) { return t.tv_sec * 1e9 + t.tv_usec * 1e3; }

static void gpu_ns(double* e) { /* bin, render, jobs bin, jobs render */
    e[0] = e[1] = e[2] = e[3] = 0;
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
            if (sscanf(line, "drm-engine-bin: %lf", &v) == 1) e[0] += v;
            else if (sscanf(line, "drm-engine-render: %lf", &v) == 1) e[1] += v;
            else if (sscanf(line, "v3d-jobs-bin: %lf", &v) == 1) e[2] += v;
            else if (sscanf(line, "v3d-jobs-render: %lf", &v) == 1) e[3] += v;
        }
        fclose(f);
    }
    closedir(d);
}

static const char* VS = "attribute vec2 p;\nvoid main() { gl_Position = vec4(p, 0.0, 1.0); }\n";
static const char* FS_HEAVY =
    "precision highp float;\nuniform int n;\nuniform float t;\n"
    "void main() {\n    vec2 q = gl_FragCoord.xy * 0.01 + t;\n    float a = 0.0;\n"
    "    for (int i = 0; i < 4096; i++) {\n        if (i >= n) break;\n"
    "        a += sin(q.x + float(i) * 0.37) * cos(q.y - a);\n    }\n"
    "    gl_FragColor = vec4(fract(a), fract(t), 0.5, 1.0);\n}\n";
static const char* FS_FLAT = "precision mediump float;\nuniform vec4 c;\nvoid main() { gl_FragColor = c; }\n";

static GLuint program(const char* fs) {
    GLuint p = glCreateProgram();
    const char* src[2] = {VS, fs};
    GLenum ty[2] = {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
    for (int i = 0; i < 2; i++) {
        GLuint s = glCreateShader(ty[i]);
        glShaderSource(s, 1, &src[i], NULL);
        glCompileShader(s);
        glAttachShader(p, s);
    }
    glBindAttribLocation(p, 0, "p");
    glLinkProgram(p);
    return p;
}

static void spin(double us) {
    if (us <= 0) return;
    double end = mono(CLOCK_MONOTONIC) + us * 1e3;
    while (mono(CLOCK_MONOTONIC) < end) {}
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "timer";
    if (!strcmp(mode, "timer")) {
        int N = 1000000;
        double t0 = mono(CLOCK_MONOTONIC);
        for (int i = 0; i < N; i++) mono(CLOCK_MONOTONIC);
        double t1 = mono(CLOCK_MONOTONIC);
        for (int i = 0; i < N; i++) mono(CLOCK_THREAD_CPUTIME_ID);
        double t2 = mono(CLOCK_MONOTONIC);
        struct rusage r0, r1;
        getrusage(RUSAGE_SELF, &r0);
        for (int i = 0; i < N; i++) { mono(CLOCK_MONOTONIC); mono(CLOCK_THREAD_CPUTIME_ID); mono(CLOCK_MONOTONIC); mono(CLOCK_THREAD_CPUTIME_ID); }
        getrusage(RUSAGE_SELF, &r1);
        printf("{\"mode\":\"timer\",\"monotonic_ns\":%.1f,\"thread_cputime_ns\":%.1f,\"wrapper_pair_sys_ns\":%.1f,\"wrapper_pair_user_ns\":%.1f}\n",
               (t1 - t0) / N, (t2 - t1) / N, (tv_ns(r1.ru_stime) - tv_ns(r0.ru_stime)) / N,
               (tv_ns(r1.ru_utime) - tv_ns(r0.ru_utime)) / N);
        return 0;
    }
    bool quad = !strcmp(mode, "quad");
    double arg = argc > 2 ? atof(argv[2]) : 0.5;
    double spin_us = argc > 3 ? atof(argv[3]) : 0;
    int frames = argc > 4 ? atoi(argv[4]) : 600;
    bool vsync = argc > 5 && !strcmp(argv[5], "vsync");

    static HSGraphics gfx;
    if (hs_graphics_init(&gfx) != 0) return 1;
    GLuint prog = program(quad ? FS_HEAVY : FS_FLAT);
    glUseProgram(prog);
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    static const float full[] = {-1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1};
    glBufferData(GL_ARRAY_BUFFER, sizeof(full), full, GL_STATIC_DRAW);
    GLint n_loc = glGetUniformLocation(prog, quad ? "n" : "c"), t_loc = glGetUniformLocation(prog, "t");
    double frame_ms = 0;
    int n = 8, draws = quad ? 1 : (int)arg;
    double sw = 2.0 * 16 / 800, sh = 2.0 * 16 / 480;
    if (quad) {
        glUniform1i(n_loc, 1);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        hs_graphics_present(&gfx);
        double refresh_ms = gfx.display_mode.vrefresh ? 1000.0 / gfx.display_mode.vrefresh : 1000.0 / 60;
        double want = arg * refresh_ms;
        for (int round = 0; round < 16; round++) {
            glUniform1i(n_loc, n);
            glFinish();
            double t0 = mono(CLOCK_MONOTONIC);
            for (int i = 0; i < 8; i++) glDrawArrays(GL_TRIANGLES, 0, 6);
            glFinish();
            frame_ms = (mono(CLOCK_MONOTONIC) - t0) / 8e6;
            if (frame_ms > want * 0.95 && frame_ms < want * 1.05) break;
            int next = (int)(n * want / (frame_ms > 0.01 ? frame_ms : 0.01));
            n = next < 1 ? 1 : next > 4096 ? 4096 : next;
        }
        glUniform1i(n_loc, n);
    }
    gfx.vsync = vsync;
    double g0[4], g1[4];
    struct rusage r0, r1;
    double tc0 = 0, start = 0;
    for (int i = 0; i < 60 + frames; i++) {
        if (i == 60) {
            gpu_ns(g0);
            getrusage(RUSAGE_SELF, &r0);
            tc0 = mono(CLOCK_THREAD_CPUTIME_ID);
            start = mono(CLOCK_MONOTONIC);
        }
        spin(spin_us);
        if (quad) {
            glUniform1f(t_loc, (float)i * 0.01f);
            glDrawArrays(GL_TRIANGLES, 0, 6);
        } else {
            glClear(GL_COLOR_BUFFER_BIT);
            for (int k = 0; k < draws; k++) {
                float x = -1 + sw * (k % 50), y = -1 + sh * ((k / 50) % 30);
                float v[12] = {x, y, x + sw, y, x, y + sh, x + sw, y, x + sw, y + sh, x, y + sh};
                glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_STREAM_DRAW);
                glUniform4f(n_loc, (k & 1) * 1.f, (k & 2) * .5f, (float)(i & 255) / 255.f, 1.f);
                glDrawArrays(GL_TRIANGLES, 0, 6);
            }
        }
        hs_graphics_present(&gfx);
    }
    double wall = mono(CLOCK_MONOTONIC) - start;
    double tcpu = mono(CLOCK_THREAD_CPUTIME_ID) - tc0;
    getrusage(RUSAGE_SELF, &r1);
    gpu_ns(g1);
    double proc = tv_ns(r1.ru_utime) - tv_ns(r0.ru_utime) + tv_ns(r1.ru_stime) - tv_ns(r0.ru_stime);
    printf("{\"mode\":\"%s\",\"arg\":%.3f,\"spin_us\":%.0f,\"gpu_frame_ms\":%.3f,\"vsync\":%s,\"frames\":%d,\"fps\":%.1f,"
           "\"wall_us\":%.1f,\"thread_cpu_us\":%.1f,\"proc_cpu_us\":%.1f,\"vcsw\":%.2f,"
           "\"render_us\":%.1f,\"bin_us\":%.1f,\"render_busy\":%.3f,\"bin_busy\":%.3f,\"render_jobs\":%.2f,\"bin_jobs\":%.2f,"
           "\"presented\":%llu,\"errors\":%u}\n",
           mode, arg, spin_us, frame_ms, vsync ? "true" : "false", frames, frames * 1e9 / wall, wall / frames / 1e3,
           tcpu / frames / 1e3, proc / frames / 1e3, (double)(r1.ru_nvcsw - r0.ru_nvcsw) / frames,
           (g1[1] - g0[1]) / frames / 1e3, (g1[0] - g0[0]) / frames / 1e3, (g1[1] - g0[1]) / wall,
           (g1[0] - g0[0]) / wall, (g1[3] - g0[3]) / frames, (g1[2] - g0[2]) / frames,
           (unsigned long long)gfx.frames_presented, gfx.present_errors);
    hs_graphics_finish(&gfx);
    return 0;
}
