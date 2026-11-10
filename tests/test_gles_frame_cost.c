/* Frame-cost decomposition probe. Diagnostic only, not production.
 *
 * Separates, for the same real V3D/GBM/EGL stack the runtime uses:
 *   - CPU submission (clear + draw calls)
 *   - eglSwapBuffers
 *   - glFinish wait
 *   - per-frame sync policy (finish every N frames, or never)
 *   - swap on/off
 *   - draw count and offscreen resolution
 * and records per-thread CPU so driver-worker cost is visible.
 *
 * Build: gcc -O3 -march=armv8-a+simd -mtune=cortex-a72 -Iinclude \
 *        -o tests/test_gles_frame_cost tests/test_gles_frame_cost.c \
 *        -lm -lGLESv2 -lgbm -ldrm -lEGL
 */
#define _GNU_SOURCE
#include "hs_graphics.h"
#include <dirent.h>
#include <inttypes.h>
#include <stdio.h>
#include <time.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
#define MAX_FRAMES 8192

typedef struct { u64 submit, swap, finish, total; } FrameTime;
static FrameTime frames_times[MAX_FRAMES];
static HSGraphics gfx;
static GLuint program, vbo, fbo, renderbuffer;
static GLint pos_loc;
static GLuint width, height;

static u64 now_ns(void) {
    struct timespec t; REQUIRE(clock_gettime(CLOCK_MONOTONIC, &t) == 0);
    return (u64)t.tv_sec * 1000000000ull + (u64)t.tv_nsec;
}
static u64 cpu_ns(void) {
    struct timespec t; REQUIRE(clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t) == 0);
    return (u64)t.tv_sec * 1000000000ull + (u64)t.tv_nsec;
}
static int compare_u64(const void* a, const void* b) {
    u64 x = *(const u64*)a, y = *(const u64*)b;
    return x < y ? -1 : x > y;
}
static u64 percentile(u64* values, u32 count, u32 p) {
    qsort(values, count, sizeof(*values), compare_u64);
    return values[(u32)(((u64)count * p + 99) / 100) - 1];
}

/* Report CRTC and framebuffer state so a candidate can prove what it left
 * programmed, rather than inferring it from timing. */
static void print_crtcs(const char* phase, int fd) {
    if (fd <= 0) { printf("{\"record\":\"crtcs\",\"phase\":\"%s\",\"crtcs\":[]}\n", phase); return; }
    drmModeRes* res = drmModeGetResources(fd);
    if (!res) { printf("{\"record\":\"crtcs\",\"phase\":\"%s\",\"crtcs\":[]}\n", phase); return; }
    printf("{\"record\":\"crtcs\",\"phase\":\"%s\",\"crtcs\":[", phase);
    bool first = true;
    for (int i = 0; i < res->count_crtcs; i++) {
        drmModeCrtc* crtc = drmModeGetCrtc(fd, res->crtcs[i]);
        if (!crtc) continue;
        uint32_t handle = 0, width = 0, height = 0, pitch = 0, bpp = 0;
        if (crtc->buffer_id) {
            drmModeFB* fb = drmModeGetFB(fd, crtc->buffer_id);
            if (fb) {
                handle = fb->handle; width = fb->width; height = fb->height;
                pitch = fb->pitch; bpp = fb->bpp;
                drmModeFreeFB(fb);
            }
        }
        printf("%s{\"crtc_id\":%u,\"fb_id\":%u,\"handle\":%u,\"width\":%u,\"height\":%u,"
               "\"pitch\":%u,\"bpp\":%u,\"active\":%s}",
               first ? "" : ",", crtc->crtc_id, crtc->buffer_id, handle, width, height,
               pitch, bpp, crtc->buffer_id ? "true" : "false");
        first = false;
        drmModeFreeCrtc(crtc);
    }
    drmModeFreeResources(res);
    printf("]}\n");
}

static void print_threads(const char* phase) {
    DIR* directory = opendir("/proc/self/task"); REQUIRE(directory);
    struct dirent* entry;
    long ticks = sysconf(_SC_CLK_TCK);
    printf("{\"record\":\"threads\",\"phase\":\"%s\",\"threads\":[", phase);
    bool first = true;
    while ((entry = readdir(directory))) {
        if (entry->d_name[0] == '.') continue;
        char path[512], comm[64] = "", stat[2048] = "";
        snprintf(path, sizeof(path), "/proc/self/task/%s/comm", entry->d_name);
        FILE* file = fopen(path, "r");
        if (!file) continue;
        if (fgets(comm, sizeof(comm), file)) comm[strcspn(comm, "\n")] = 0;
        fclose(file);
        snprintf(path, sizeof(path), "/proc/self/task/%s/stat", entry->d_name);
        file = fopen(path, "r");
        if (!file) continue;
        size_t length = fread(stat, 1, sizeof(stat) - 1, file);
        fclose(file);
        stat[length] = 0;
        char* after = strrchr(stat, ')'); REQUIRE(after);
        unsigned long utime = 0, stime = 0;
        int fields = sscanf(after + 2,
            "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %lu %lu", &utime, &stime);
        if (fields != 2) continue;
        u64 thread_cpu = (u64)(utime + stime) * 1000000000ull / (u64)ticks;
        printf("%s{\"tid\":%s,\"comm\":\"%s\",\"cpu_ns\":%" PRIu64 "}", first ? "" : ",",
               entry->d_name, comm, thread_cpu);
        first = false;
    }
    REQUIRE(closedir(directory) == 0);
    printf("]}\n");
}

static GLuint compile(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    GLint ok = 0; glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[1024]; glGetShaderInfoLog(shader, sizeof(log), NULL, log); fprintf(stderr, "%s\n", log); exit(1); }
    return shader;
}

static void setup(void) {
    static const char* vs =
        "attribute vec2 a_position;\n"
        "attribute vec3 a_color;\n"
        "varying vec3 v_color;\n"
        "void main(){ v_color=a_color; gl_Position=vec4(a_position,0.0,1.0); }\n";
    static const char* fs =
        "precision mediump float;\n"
        "varying vec3 v_color;\n"
        "void main(){ gl_FragColor=vec4(v_color,1.0); }\n";
    GLuint v = compile(GL_VERTEX_SHADER, vs), f = compile(GL_FRAGMENT_SHADER, fs);
    program = glCreateProgram();
    glAttachShader(program, v); glAttachShader(program, f); glLinkProgram(program);
    GLint ok = 0; glGetProgramiv(program, GL_LINK_STATUS, &ok); REQUIRE(ok);
    glUseProgram(program);
    pos_loc = glGetAttribLocation(program, "a_position");
    GLint color_loc = glGetAttribLocation(program, "a_color");
    REQUIRE(pos_loc >= 0 && color_loc >= 0);
    const float verts[15] = {-0.6f, 0.6f, 1, 0, 0, -0.8f, -0.6f, 0, 1, 0, -0.4f, -0.6f, 0, 0, 1};
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glEnableVertexAttribArray((GLuint)pos_loc);
    glVertexAttribPointer((GLuint)pos_loc, 2, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)0);
    glEnableVertexAttribArray((GLuint)color_loc);
    glVertexAttribPointer((GLuint)color_loc, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(2*sizeof(float)));
    REQUIRE(glGetError() == GL_NO_ERROR);
}

int main(int argc, char** argv) {
    REQUIRE(argc == 8);
    const char* surface = argv[1];
    bool window = !strcmp(surface, "window"), offscreen = !strcmp(surface, "offscreen"), nogl = !strcmp(surface, "none");
    REQUIRE(window || offscreen || nogl);
    u32 size = (u32)strtoul(argv[2], NULL, 10);
    u32 count = (u32)strtoul(argv[3], NULL, 10);
    u32 draws = (u32)strtoul(argv[4], NULL, 10);
    u32 finish_every = (u32)strtoul(argv[5], NULL, 10);
    bool swap = !strcmp(argv[6], "on");
    const char* prefix = argv[7];
    REQUIRE(offscreen ? (size >= 32 && size <= 1024) : size == 0);
    REQUIRE(count >= 8 && count <= MAX_FRAMES && draws <= 512 && finish_every <= MAX_FRAMES);
    REQUIRE(strncmp(prefix, "tests/evidence-", 15) == 0);

    u64 init_start = now_ns(), init_cpu = cpu_ns();
    if (!nogl) {
        REQUIRE(hs_graphics_init(&gfx) == 0);
        /* Diagnostic arm: "off" skips taking the CRTC from the console, so the
         * probe can measure display ownership without changing the library. */
        {
            const char* display = getenv("HS_GRAPHICS_DISPLAY");
            if (display && !strcmp(display, "off")) gfx.take_display = false;
        }
        width = gfx.screen_width; height = gfx.screen_height;
        if (offscreen) { width = height = size; }
        setup();
        if (offscreen) {
            glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glGenRenderbuffers(1, &renderbuffer); glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, (GLsizei)width, (GLsizei)height);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, renderbuffer);
            REQUIRE(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
        }
        glViewport(0, 0, (GLsizei)width, (GLsizei)height);
        REQUIRE(glGetError() == GL_NO_ERROR);
    }
    u64 init_ns = now_ns() - init_start, init_cpu_ns = cpu_ns() - init_cpu;
    u64 first_present_ns = 0;
    print_crtcs("after_init", nogl ? 0 : gfx.drm_fd);
    if (!nogl && window) {
        u64 start = now_ns(); hs_graphics_present(&gfx); first_present_ns = now_ns() - start;
    }
    print_crtcs("after_first_present", nogl ? 0 : gfx.drm_fd);
    /* Warm up without recording; same draws as the measured loop. */
    for (u32 warm = 0; warm < 8; warm++) {
        if (!nogl) {
            glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
            for (u32 d = 0; d < draws; d++) glDrawArrays(GL_TRIANGLES, 0, 3);
            if (swap) REQUIRE(eglSwapBuffers(gfx.egl_display, gfx.egl_surface) == EGL_TRUE);
            if (finish_every) glFinish();
        }
    }
    if (!nogl) { glFinish(); REQUIRE(glGetError() == GL_NO_ERROR); }
    print_threads("before_timed");

    u64 start = now_ns(), process_cpu = cpu_ns();
    for (u32 i = 0; i < count; i++) {
        u64 t0 = now_ns();
        if (!nogl) {
            glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
            for (u32 d = 0; d < draws; d++) glDrawArrays(GL_TRIANGLES, 0, 3);
        }
        u64 t1 = now_ns();
        if (!nogl && swap) REQUIRE(eglSwapBuffers(gfx.egl_display, gfx.egl_surface) == EGL_TRUE);
        u64 t2 = now_ns();
        if (!nogl && finish_every && (i + 1) % finish_every == 0) glFinish();
        u64 t3 = now_ns();
        frames_times[i] = (FrameTime){t1 - t0, t2 - t1, t3 - t2, t3 - t0};
    }
    u64 elapsed = now_ns() - start, timed_cpu = cpu_ns() - process_cpu;
    u64 submit[MAX_FRAMES], swap_ns[MAX_FRAMES], finish_ns[MAX_FRAMES], total[MAX_FRAMES];
    for (u32 i = 0; i < count; i++) {
        submit[i] = frames_times[i].submit; swap_ns[i] = frames_times[i].swap;
        finish_ns[i] = frames_times[i].finish; total[i] = frames_times[i].total;
    }
    print_threads("after_timed");
    print_crtcs("after_loop", nogl ? 0 : gfx.drm_fd);

    /* One verification frame outside timing: no swap, finish, nonblank pixel. */
    u64 verify_ns = 0;
    if (!nogl) {
        u64 before = now_ns();
        glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
        for (u32 d = 0; d < draws; d++) glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        u8 pixel[4] = {0, 0, 0, 0};
        glReadPixels((GLint)(width/2), (GLint)(height/2), 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
        REQUIRE(glGetError() == GL_NO_ERROR);
        REQUIRE(pixel[3] == 255);
        char path[512]; REQUIRE(snprintf(path, sizeof(path), "%s.verify.rgba", prefix) < (int)sizeof(path));
        size_t bytes = (size_t)width * height * 4;
        u8* pixels = malloc(bytes); REQUIRE(pixels);
        glReadPixels(0, 0, (GLsizei)width, (GLsizei)height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        FILE* file = fopen(path, "wb"); REQUIRE(file);
        REQUIRE(fwrite(pixels, 1, bytes, file) == bytes && fclose(file) == 0);
        free(pixels);
        verify_ns = now_ns() - before;
    }
    u64 shutdown_start = now_ns();
    if (!nogl) hs_graphics_finish(&gfx);
    u64 shutdown_ns = now_ns() - shutdown_start;
    print_threads("after_shutdown");

    printf("{\"record\":\"summary\",\"surface\":\"%s\",\"size\":%u,\"width\":%u,\"height\":%u,"
           "\"frames\":%u,\"draws\":%u,\"finish_every\":%u,\"swap\":\"%s\",\"take_display\":%s,"
           "\"elapsed_ns\":%" PRIu64 ",\"process_cpu_ns\":%" PRIu64 ",\"timed_cpu_ns\":%" PRIu64 ","
           "\"init_ns\":%" PRIu64 ",\"init_cpu_ns\":%" PRIu64 ",\"first_present_ns\":%" PRIu64 ","
           "\"verify_ns\":%" PRIu64 ",\"shutdown_ns\":%" PRIu64 ","
           "\"submit_p50_ns\":%" PRIu64 ",\"submit_p99_ns\":%" PRIu64 ","
           "\"swap_p50_ns\":%" PRIu64 ",\"swap_p99_ns\":%" PRIu64 ","
           "\"finish_p50_ns\":%" PRIu64 ",\"finish_p99_ns\":%" PRIu64 ","
           "\"total_p50_ns\":%" PRIu64 ",\"total_p99_ns\":%" PRIu64 ","
           "\"per_frame_ns\":%" PRIu64 "}\n",
           surface, size, width, height, count, draws, finish_every, swap ? "on" : "off",
           gfx.take_display ? "true" : "false",
           elapsed, timed_cpu, timed_cpu, init_ns, init_cpu_ns, first_present_ns,
           verify_ns, shutdown_ns,
           percentile(submit, count, 50), percentile(submit, count, 99),
           percentile(swap_ns, count, 50), percentile(swap_ns, count, 99),
           percentile(finish_ns, count, 50), percentile(finish_ns, count, 99),
           percentile(total, count, 50), percentile(total, count, 99),
           elapsed / count);
    return 0;
}
