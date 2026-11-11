/* Diagnostic: renders a full-screen shader calibrated to a fraction of a
 * refresh per frame, uncapped (or vsync on with "vsync"), for pacing traces
 * with tests/pacing_trace.so.
 *
 * Usage: sudo tests/pacing_sweep <fraction_of_refresh> [frames] [vsync]
 * Output: one JSON line (calibrated GPU ms, FPS). With PACING_CONTENT set,
 * writes the animation timestamp (us, CLOCK_MONOTONIC, taken when the frame
 * starts) of every presented frame in present order, one per line, for
 * tests/pacing_trace_report.py --content. */
#include "hs_graphics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char* VS = "attribute vec2 p;\nvoid main() { gl_Position = vec4(p, 0.0, 1.0); }\n";
static const char* FS =
    "precision highp float;\nuniform int n;\nuniform float t;\n"
    "void main() {\n    vec2 q = gl_FragCoord.xy * 0.01 + t;\n    float a = 0.0;\n"
    "    for (int i = 0; i < 4096; i++) {\n        if (i >= n) break;\n"
    "        a += sin(q.x + float(i) * 0.37) * cos(q.y - a);\n    }\n"
    "    gl_FragColor = vec4(fract(a), fract(t), 0.5, 1.0);\n}\n";

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static GLuint shader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    return s;
}

int main(int argc, char** argv) {
    double fraction = argc > 1 ? atof(argv[1]) : 0.5;
    int frames = argc > 2 ? atoi(argv[2]) : 600;
    bool vsync = argc > 3 && strcmp(argv[3], "vsync") == 0;
    static HSGraphics gfx;
    if (hs_graphics_init(&gfx) != 0) return 1;
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
    static double content[1 << 16];
    int nc = 0;
    glUniform1i(n_loc, 1);
    content[nc++] = now_ms();
    glDrawArrays(GL_TRIANGLES, 0, 6);
    hs_graphics_present(&gfx);
    double refresh_ms = gfx.display_mode.vrefresh ? 1000.0 / gfx.display_mode.vrefresh : 1000.0 / 60;
    int n = 8;
    double frame_ms = 0, want = fraction * refresh_ms;
    for (int round = 0; round < 16; round++) {
        glUniform1i(n_loc, n);
        glFinish();
        double t0 = now_ms();
        for (int i = 0; i < 8; i++) glDrawArrays(GL_TRIANGLES, 0, 6);
        glFinish();
        frame_ms = (now_ms() - t0) / 8;
        if (frame_ms > want * 0.95 && frame_ms < want * 1.05) break;
        int next = (int)(n * want / (frame_ms > 0.01 ? frame_ms : 0.01));
        n = next < 1 ? 1 : next > 4096 ? 4096 : next;
    }
    glUniform1i(n_loc, n);
    gfx.vsync = vsync;
    double start = now_ms();
    for (int i = 0; i < frames; i++) {
        if (nc < (1 << 16)) content[nc++] = now_ms();
        glUniform1f(t_loc, (float)i * 0.01f);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        hs_graphics_present(&gfx);
    }
    double elapsed = now_ms() - start;
    printf("{\"fraction\":%.3f,\"gpu_frame_ms\":%.3f,\"iterations\":%d,\"vsync\":%s,\"fps\":%.1f,\"errors\":%u}\n",
           fraction, frame_ms, n, vsync ? "true" : "false", frames * 1000.0 / elapsed, gfx.present_errors);
    hs_graphics_finish(&gfx);
    const char* path = getenv("PACING_CONTENT");
    FILE* f = path ? fopen(path, "w") : NULL;
    if (f) {
        for (int i = 0; i < nc; i++) fprintf(f, "%.0f\n", content[i] * 1e3);
        fclose(f);
    }
    return 0;
}
