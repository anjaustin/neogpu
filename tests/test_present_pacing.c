/* Hardware check: with vsync off, a GPU-heavy frame rate still puts a new
 * frame on screen at every refresh.
 *
 * A flip lands only once the GPU has finished its frame. When frames take
 * about half a refresh on the GPU (~2 rendered per refresh, as in
 * test_08_raycast_orbit), flipping the frame just submitted, which renders
 * behind frames that will be dropped, misses the next refresh and repeats the
 * old image on about a third of refreshes: visible stutter at full render
 * rate. hs_graphics_present flips the newest finished frame instead.
 *
 * The shader cost is calibrated (with glFinish) to about 0.55 of a refresh
 * per frame, then frames are rendered uncapped. Contract: every frame is
 * shown or dropped, at most 1% of refreshes repeat the previous frame, and
 * hs_graphics_flush_present() shows a held frame and leaves room for the
 * caller's own eglSwapBuffers.
 *
 * Usage: sudo tests/test_present_pacing [frames]
 * Output: one JSON object; exit 0 only if the run met its contract.
 */
#include "hs_graphics.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static const char* VS =
    "attribute vec2 p;\n"
    "void main() { gl_Position = vec4(p, 0.0, 1.0); }\n";
static const char* FS =
    "precision highp float;\n"
    "uniform int n;\n"
    "uniform float t;\n"
    "void main() {\n"
    "    vec2 q = gl_FragCoord.xy * 0.01 + t;\n"
    "    float a = 0.0;\n"
    "    for (int i = 0; i < 4096; i++) {\n"
    "        if (i >= n) break;\n"
    "        a += sin(q.x + float(i) * 0.37) * cos(q.y - a);\n"
    "    }\n"
    "    gl_FragColor = vec4(fract(a), fract(t), 0.5, 1.0);\n"
    "}\n";

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static GLuint shader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    return ok ? s : 0;
}

static void draw(GLint n_loc, GLint t_loc, int n, float t) {
    glUniform1i(n_loc, n);
    glUniform1f(t_loc, t);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

int main(int argc, char** argv) {
    int frames = argc > 1 ? atoi(argv[1]) : 600;
    static HSGraphics gfx;
    if (frames < 60 || hs_graphics_init(&gfx) != 0) {
        printf("{\"init\":false}\n");
        return 1;
    }
    GLuint vs = shader(GL_VERTEX_SHADER, VS), fs = shader(GL_FRAGMENT_SHADER, FS);
    GLuint prog = glCreateProgram();
    if (vs && fs) {
        glAttachShader(prog, vs);
        glAttachShader(prog, fs);
        glBindAttribLocation(prog, 0, "p");
        glLinkProgram(prog);
    }
    GLint linked = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    if (!linked) {
        printf("{\"shader\":false}\n");
        hs_graphics_finish(&gfx);
        return 1;
    }
    static const float quad[] = {-1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1};
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glUseProgram(prog);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    GLint n_loc = glGetUniformLocation(prog, "n"), t_loc = glGetUniformLocation(prog, "t");

    /* Take the display first so calibration runs against the real path. */
    draw(n_loc, t_loc, 1, 0);
    hs_graphics_present(&gfx);
    double refresh_ms = gfx.display_mode.vrefresh ? 1000.0 / gfx.display_mode.vrefresh : 1000.0 / 60;

    /* Calibrate the per-frame GPU time to ~0.55 refresh. */
    int n = 8;
    double frame_ms = 0;
    for (int round = 0; round < 12; round++) {
        glFinish();
        double t0 = now_ms();
        for (int i = 0; i < 4; i++) draw(n_loc, t_loc, n, (float)i);
        glFinish();
        frame_ms = (now_ms() - t0) / 4;
        double want = 0.55 * refresh_ms;
        if (frame_ms > want * 0.9 && frame_ms < want * 1.1) break;
        int next = (int)(n * want / (frame_ms > 0.01 ? frame_ms : 0.01));
        n = next < 1 ? 1 : next > 4096 ? 4096 : next;
    }

    /* One paced flip gives the starting vblank count (takeover has none). */
    gfx.vsync = true;
    draw(n_loc, t_loc, 1, 0);
    hs_graphics_present(&gfx);
    gfx.vsync = false;
    u64 presented0 = gfx.frames_presented, dropped0 = gfx.frames_dropped;
    u64 repeated0 = gfx.refreshes_repeated;
    u32 seq0 = gfx.last_flip_sequence;
    double start = now_ms();
    for (int i = 0; i < frames; i++) {
        draw(n_loc, t_loc, n, (float)i * 0.01f);
        hs_graphics_present(&gfx);
    }
    double elapsed = now_ms() - start;
    /* Keep presenting until a frame is held, flush it (it must be shown, not
     * dropped), then swap the surface directly as a caller mixing raw EGL
     * with hs_graphics may. */
    int extra = 0;
    while (!gfx.held_bo && extra < 64) {
        draw(n_loc, t_loc, n, 0);
        hs_graphics_present(&gfx);
        extra++;
    }
    bool held = gfx.held_bo != NULL;
    u64 owed = (u64)gfx.flip_pending + held;
    u64 presented_before = gfx.frames_presented, dropped_before = gfx.frames_dropped;
    hs_graphics_flush_present(&gfx);
    bool flush_shows = gfx.frames_presented == presented_before + owed && gfx.frames_dropped == dropped_before &&
                       !gfx.held_bo && !gfx.flip_pending;
    u64 presented = gfx.frames_presented - presented0, dropped = gfx.frames_dropped - dropped0;
    u64 repeated = gfx.refreshes_repeated - repeated0;
    u32 refreshes = gfx.last_flip_sequence - seq0;
    /* Each flip covers its own refresh plus the ones that repeated before it. */
    bool counted = refreshes == presented + repeated;
    /* The repeat counter itself: a flip after ~4 idle refreshes adds >= 2. */
    usleep(60000);
    u64 repeated_idle = gfx.refreshes_repeated;
    gfx.vsync = true;
    draw(n_loc, t_loc, 1, 0);
    hs_graphics_present(&gfx);
    gfx.vsync = false;
    counted = counted && gfx.refreshes_repeated >= repeated_idle + 2;
    bool raw_swaps = true;
    for (int i = 0; i < 3; i++) {
        draw(n_loc, t_loc, 1, 0);
        raw_swaps = raw_swaps && eglSwapBuffers(gfx.egl_display, gfx.egl_surface) == EGL_TRUE;
    }
    bool active = gfx.crtc_active;
    hs_graphics_finish(&gfx);
    double fps = frames * 1000.0 / elapsed;
    bool ok = active && gfx.present_errors == 0 && presented + dropped == (u64)(frames + extra) && held && raw_swaps && counted &&
              flush_shows && fps > 1.5 * 1000.0 / refresh_ms && seq0 && refreshes > 0 && repeated * 100 <= refreshes;
    printf("{\"frames\":%d,\"iterations\":%d,\"gpu_frame_ms\":%.2f,\"refresh_ms\":%.2f,\"fps\":%.1f,"
           "\"presented\":%llu,\"dropped\":%llu,\"refreshes\":%u,\"repeated\":%llu,\"errors\":%u,\"held\":%s,\"raw_swaps\":%s,\"counted\":%s,\"flush_owed\":%llu,\"flush_shows\":%s,\"passed\":%s}\n",
           frames, n, frame_ms, refresh_ms, fps, (unsigned long long)presented, (unsigned long long)dropped,
           refreshes, (unsigned long long)repeated, gfx.present_errors, held ? "true" : "false",
           raw_swaps ? "true" : "false", counted ? "true" : "false", (unsigned long long)owed,
           flush_shows ? "true" : "false", ok ? "true" : "false");
    return ok ? 0 : 1;
}
