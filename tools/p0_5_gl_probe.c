/* p0_5_gl_probe: an offscreen GLES renderer that checks every frame it draws.
 * (P0-5 bedrock, docs/audit/HARDWARE_HANDOFF_P0.md.)
 *
 * Questions it answers, run beside tools/p0_5_spin_probe or hs_mailbox:
 * - when the kernel resets the GPU and cancels this renderer's job, what do
 *   the frame's pixels hold, and what does GL report (reset status, errors)?
 * - what happens when this renderer's own job hangs (hang_frame)?
 * - a second GPU-bound renderer for the thread-switch measurements.
 *
 * Renders to a size x size RGBA8 texture through the render node (no display,
 * no DRM master). Frame f fills the target with colour (f & 255, f >> 8 & 255,
 * 77, 255); the fragment shader first runs `iters` loop iterations per pixel
 * so the frame costs GPU time. Every frame is read back and every pixel
 * compared with this frame's colour and the previous frame's.
 *
 * Usage: p0_5_gl_probe <seconds> <size> <iters> [hang_frame hang_iters] [start_delay_ms]
 *   hang_frame  that one frame runs hang_iters iterations per pixel instead
 *   start_delay_ms  wait this long after setup before the first frame
 *   (inside the GPU lock). Output: one JSON line per frame that is slow
 *   (over 100 ms), wrong, or reports a GL error or reset; then a summary.
 */
#define _GNU_SOURCE
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <fcntl.h>
#include <gbm.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifndef GL_NO_ERROR_RESET
#define GL_GUILTY_CONTEXT_RESET_EXT 0x8253
#define GL_INNOCENT_CONTEXT_RESET_EXT 0x8254
#define GL_UNKNOWN_CONTEXT_RESET_EXT 0x8255
#endif

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
static GLuint shader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "shader: %s\n", log);
        exit(1);
    }
    return s;
}
static const char* reset_name(GLenum r) {
    return r == GL_NO_ERROR ? "none" : r == GL_GUILTY_CONTEXT_RESET_EXT ? "guilty" : r == GL_INNOCENT_CONTEXT_RESET_EXT ? "innocent"
           : r == GL_UNKNOWN_CONTEXT_RESET_EXT ? "unknown" : "other";
}

int main(int argc, char** argv) {
    if (argc < 4) return fprintf(stderr, "usage: %s <seconds> <size> <iters> [hang_frame hang_iters] [start_delay_ms]\n", argv[0]), 2;
    double seconds = atof(argv[1]);
    int size = atoi(argv[2]), iters = atoi(argv[3]);
    int hang_frame = argc > 5 ? atoi(argv[4]) : -1, hang_iters = argc > 5 ? atoi(argv[5]) : 0;
    int delay_ms = argc > 6 ? atoi(argv[6]) : 0;
    setvbuf(stdout, NULL, _IOLBF, 0);

    int fd = open("/dev/dri/renderD128", O_RDWR);
    struct gbm_device* gbm = fd >= 0 ? gbm_create_device(fd) : NULL;
    EGLDisplay dpy = gbm ? eglGetDisplay((EGLNativeDisplayType)gbm) : EGL_NO_DISPLAY;
    if (dpy == EGL_NO_DISPLAY || !eglInitialize(dpy, NULL, NULL)) return fprintf(stderr, "no EGL display\n"), 1;
    const char* ext = eglQueryString(dpy, EGL_EXTENSIONS);
    int robust = ext && strstr(ext, "EGL_EXT_create_context_robustness");
    EGLint cattr[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_SURFACE_TYPE, EGL_DONT_CARE, EGL_NONE};
    EGLConfig cfg;
    EGLint n;
    if (!eglChooseConfig(dpy, cattr, &cfg, 1, &n) || !n) return fprintf(stderr, "no EGL config\n"), 1;
    eglBindAPI(EGL_OPENGL_ES_API);
    EGLint xattr[] = {EGL_CONTEXT_CLIENT_VERSION, 3,
                      EGL_CONTEXT_OPENGL_ROBUST_ACCESS_EXT, EGL_TRUE,
                      EGL_CONTEXT_OPENGL_RESET_NOTIFICATION_STRATEGY_EXT, EGL_LOSE_CONTEXT_ON_RESET_EXT,
                      EGL_NONE};
    if (!robust) xattr[2] = EGL_NONE;
    EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, xattr);
    if (ctx == EGL_NO_CONTEXT || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx))
        return fprintf(stderr, "no context: 0x%x\n", eglGetError()), 1;
    GLenum (*reset_status)(void) = (GLenum (*)(void))eglGetProcAddress("glGetGraphicsResetStatusEXT");
    if (!reset_status) reset_status = (GLenum (*)(void))eglGetProcAddress("glGetGraphicsResetStatus");

    const char* vs = "#version 300 es\nin vec2 p;\nvoid main(){gl_Position=vec4(p,0.0,1.0);}\n";
    const char* fs = "#version 300 es\nprecision highp float;\nuniform vec4 u_color;\nuniform int u_iters;\nout vec4 o;\n"
                     "void main(){\n"
                     "  float a = gl_FragCoord.x * 0.001 + gl_FragCoord.y * 0.002;\n"
                     "  for (int i = 0; i < u_iters; i++) a = a * 0.9995 + 0.0007;\n"
                     "  o = u_color + vec4(step(1.0e30, a));\n"
                     "}\n";
    GLuint prog = glCreateProgram();
    glAttachShader(prog, shader(GL_VERTEX_SHADER, vs));
    glAttachShader(prog, shader(GL_FRAGMENT_SHADER, fs));
    glBindAttribLocation(prog, 0, "p");
    glLinkProgram(prog);
    GLint ok;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) return fprintf(stderr, "link failed\n"), 1;
    glUseProgram(prog);
    GLint u_color = glGetUniformLocation(prog, "u_color"), u_iters = glGetUniformLocation(prog, "u_iters");
    static const float quad[] = {-1, -1, 1, -1, -1, 1, 1, 1};
    GLuint vbo, tex, fbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, size, size);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) return fprintf(stderr, "fbo incomplete\n"), 1;
    glViewport(0, 0, size, size);
    uint32_t* px = malloc((size_t)size * size * 4);
    printf("{\"gl_probe\":\"start\",\"renderer\":\"%s\",\"robust_context\":%s,\"reset_status_call\":%s,\"size\":%d,\"iters\":%d}\n",
           (const char*)glGetString(GL_RENDERER), robust ? "true" : "false", reset_status ? "true" : "false", size, iters);
    if (delay_ms) usleep((useconds_t)delay_ms * 1000);

    uint64_t t0 = now_ns(), tmax = 0, tsum = 0;
    int f = 0, bad_frames = 0, slow_frames = 0, reset_frames = 0, err_frames = 0;
    uint32_t prev = 0;
    printf("{\"gl_probe\":\"first_frame\",\"clock_monotonic_ms\":%.3f}\n", t0 / 1e6);
    while (now_ns() - t0 < (uint64_t)(seconds * 1e9)) {
        uint32_t want = 0xFF000000u | (77u << 16) | ((uint32_t)(f >> 8 & 255) << 8) | (uint32_t)(f & 255);
        uint64_t t = now_ns();
        glUniform4f(u_color, (float)(f & 255) / 255.f, (float)(f >> 8 & 255) / 255.f, 77.f / 255.f, 1.f);
        glUniform1i(u_iters, f == hang_frame ? hang_iters : iters);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glReadPixels(0, 0, size, size, GL_RGBA, GL_UNSIGNED_BYTE, px);
        uint64_t dt = now_ns() - t;
        GLenum err = glGetError(), rs = reset_status ? reset_status() : GL_NO_ERROR;
        int n_want = 0, n_prev = 0, n_other = 0;
        uint32_t other = 0;
        for (int i = 0; i < size * size; i++) {
            if (px[i] == want) n_want++;
            else if (px[i] == prev) n_prev++;
            else n_other++, other = px[i];
        }
        int bad = n_want != size * size;
        tsum += dt;
        if (dt > tmax) tmax = dt;
        bad_frames += bad, slow_frames += dt > 100000000ull, reset_frames += rs != GL_NO_ERROR, err_frames += err != GL_NO_ERROR;
        if (bad || dt > 100000000ull || rs != GL_NO_ERROR || err != GL_NO_ERROR)
            printf("{\"frame\":%d,\"at_ms\":%.1f,\"frame_ms\":%.2f,\"pixels\":{\"this_frame\":%d,\"previous_frame\":%d,\"other\":%d,"
                   "\"an_other_value\":\"0x%08x\"},\"gl_error\":\"0x%x\",\"reset_status\":\"%s\"}\n",
                   f, (t - t0) / 1e6, dt / 1e6, n_want, n_prev, n_other, other, err, reset_name(rs));
        if (err_frames > 3) break; /* a lost context fails every call from here on */
        prev = want;
        f++;
    }
    double el = (now_ns() - t0) / 1e9;
    printf("{\"gl_probe\":\"done\",\"frames\":%d,\"seconds\":%.3f,\"fps\":%.2f,\"frame_ms\":{\"mean\":%.3f,\"max\":%.2f},\"wrong_frames\":%d,"
           "\"slow_frames\":%d,\"frames_reporting_reset\":%d,\"frames_with_gl_error\":%d}\n",
           f, el, f / el, tsum / 1e6 / (f ? f : 1), tmax / 1e6, bad_frames, slow_frames, reset_frames, err_frames);
    return 0;
}
