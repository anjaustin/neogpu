#define _POSIX_C_SOURCE 200809L
#include "hs_fabric.h"
#include "hs_gpu.h"
#include "hs_backend_gles.h"
#include "hs_graphics.h"
#include "graphics_capture.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
#define FRAMES 600
static HSGpu gpu;
static HSFabric fabric;
static GLuint program, vbos[2];
static GLint pos_loc, color_loc, width, height;
static EGLDisplay display;
static EGLSurface surface;
static bool active, timed;
static u32 verify_mode;
static u8* reference;
static u64 start;
typedef struct { u64 submit, completion, finish_wait, swap, total; } Timing;
static Timing* current;
static const f32 black[4] = {0, 0, 0, 1};

static u64 ns(void) {
    struct timespec time;
    REQUIRE(clock_gettime(CLOCK_MONOTONIC, &time) == 0);
    return (u64)time.tv_sec * 1000000000ull + time.tv_nsec;
}
extern EGLBoolean __real_eglSwapBuffers(EGLDisplay display, EGLSurface surface);
EGLBoolean __wrap_eglSwapBuffers(EGLDisplay display, EGLSurface surface) {
    if (!active) return __real_eglSwapBuffers(display, surface);
    if (verify_mode) {
        size_t bytes = (size_t)width * height * 4;
        u8* pixels = malloc(bytes);
        REQUIRE(pixels);
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        REQUIRE(glGetError() == GL_NO_ERROR);
        for (u32 side = 0; side < 2; side++) {
            const u8* pixel = &pixels[((size_t)(height / 2) * width + width * (side ? 3 : 1) / 4) * 4];
            for (u32 channel = 0; channel < 3; channel++)
                REQUIRE(pixel[channel] == (channel == side ? 255 : 0));
        }
        if (verify_mode == 1) reference = pixels;
        else {
            REQUIRE(reference && memcmp(reference, pixels, bytes) == 0);
            printf("{\"record\":\"pixel_equivalence\",\"rgba_bytes\":%zu,\"identical\":true}\n", bytes);
            free(pixels);
        }
        REQUIRE(test_capture_frame() == 0);
    }
    if (timed) {
        u64 submitted = ns();
        current->submit = submitted - start;
        glFinish();
        u64 completed = ns();
        current->completion = completed - start;
        current->finish_wait = completed - submitted;
        u64 swap_start = ns();
        EGLBoolean result = __real_eglSwapBuffers(display, surface);
        current->swap = ns() - swap_start;
        REQUIRE(result == EGL_TRUE);
        return result;
    }
    EGLBoolean result = __real_eglSwapBuffers(display, surface);
    REQUIRE(result == EGL_TRUE);
    return result;
}

static GLuint compile(GLenum kind, const char* source) {
    GLuint shader = glCreateShader(kind);
    glShaderSource(shader, 1, &source, NULL); glCompileShader(shader);
    GLint ok;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok); REQUIRE(ok);
    return shader;
}
static void setup_raw(void) {
    /* Identical shader text to hs_backend_gles.c, not a different scene/shader. */
    const char* vertex = "attribute vec2 a_position;\nattribute vec3 a_color;\nvarying vec3 v_color;\nvoid main(){\n  gl_Position = vec4(a_position,0.0,1.0);\n  v_color = a_color;\n}\n";
    const char* fragment = "precision mediump float;\nvarying vec3 v_color;\nvoid main(){\n  gl_FragColor = vec4(v_color,1.0);\n}\n";
    GLuint vs = compile(GL_VERTEX_SHADER, vertex), fs = compile(GL_FRAGMENT_SHADER, fragment);
    program = glCreateProgram(); glAttachShader(program, vs); glAttachShader(program, fs);
    glLinkProgram(program); GLint ok;
    glGetProgramiv(program, GL_LINK_STATUS, &ok); REQUIRE(ok);
    glDeleteShader(vs); glDeleteShader(fs);
    pos_loc = glGetAttribLocation(program, "a_position");
    color_loc = glGetAttribLocation(program, "a_color");
    REQUIRE(pos_loc >= 0 && color_loc >= 0);
    glGenBuffers(2, vbos);
    for (u32 bank = 0; bank < 2; bank++) {
        HSBuffer* buffer = hs_gpu_get_buffer(&gpu, bank);
        glBindBuffer(GL_ARRAY_BUFFER, vbos[bank]);
        glBufferData(GL_ARRAY_BUFFER, buffer->length, buffer->data, GL_STATIC_DRAW);
    }
}
static void draw(bool native) {
    if (native) {
        hs_gpu_begin_frame(&gpu);
        REQUIRE(hs_fabric_compose(&fabric, &gpu.render, black));
        hs_gpu_end_frame(&gpu);
    } else {
        glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(program);
        glEnableVertexAttribArray(pos_loc);
        glEnableVertexAttribArray(color_loc);
        for (u32 bank = 0; bank < 2; bank++) {
            glBindBuffer(GL_ARRAY_BUFFER, vbos[bank]);
            glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
            glVertexAttribPointer(color_loc, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
            glDrawArrays(GL_TRIANGLES, 0, 3);
        }
        REQUIRE(eglSwapBuffers(display, surface) == EGL_TRUE);
    }
}
static void geometry(u32 bank, f32 center) {
    HSBuffer* buffer = hs_gpu_get_buffer(&gpu, (u8)bank);
    REQUIRE(hs_buffer_init(buffer, (u8)bank, 60));
    const f32 data[15] = {
        center, 0.6f, bank == 0, bank == 1, 0,
        center - 0.3f, -0.6f, bank == 0, bank == 1, 0,
        center + 0.3f, -0.6f, bank == 0, bank == 1, 0};
    for (u32 i = 0; i < 15; i++) hs_buffer_set_f32(buffer, i, data[i]);
    buffer->length = sizeof(data);
}
int main(int argc, char** argv) {
    REQUIRE(argc == 2 && strncmp(argv[1], "tests/evidence-", 15) == 0);
    alarm(90);
    hs_gpu_init(&gpu); gpu.system.recording = false;
    REQUIRE(hs_fabric_init(&fabric));
    for (u32 bank = 0; bank < 2; bank++) {
        geometry(bank, bank ? 0.5f : -0.5f);
        HSParticipant owner;
        REQUIRE(hs_fabric_join(&fabric, &owner) == HS_FABRIC_OK && owner.slot == bank);
        HSParticipantFrame frame = {.sequence = 1, .count = 1};
        frame.draws[0] = (HSRenderCmd){.op = HS_RC_DRAW, .a = (u8)bank};
        REQUIRE(hs_fabric_submit(&fabric, owner, &frame) == HS_FABRIC_OK);
    }
    REQUIRE(hs_fabric_step(&fabric, 8) == 2);
    HSBackend backend = hs_backend_gles_create();
    hs_gpu_attach_backend(&gpu, &backend); REQUIRE(gpu.backend);
    display = eglGetCurrentDisplay(); surface = eglGetCurrentSurface(EGL_DRAW);
    REQUIRE(display != EGL_NO_DISPLAY && surface != EGL_NO_SURFACE);
    GLint viewport[4], depth, stencil;
    glGetIntegerv(GL_VIEWPORT, viewport); width = viewport[2]; height = viewport[3];
    glGetIntegerv(GL_DEPTH_BITS, &depth); glGetIntegerv(GL_STENCIL_BITS, &stencil);
    fprintf(stderr, "GL_VERSION: %s\nGL_RENDERER: %s\n", glGetString(GL_VERSION), glGetString(GL_RENDERER));
    printf("{\"record\":\"framebuffer\",\"width\":%d,\"height\":%d,\"depth\":%d,\"stencil\":%d}\n", width, height, depth, stencil);
    REQUIRE(width > 0 && height > 0);
    setup_raw(); active = true;
    for (u32 path = 0; path < 2; path++) for (u32 i = 0; i < 30; i++) draw(path != 0);
    for (u32 path = 0; path < 2; path++) {
        char capture_path[256];
        REQUIRE(snprintf(capture_path, sizeof(capture_path), "%s-%s.ppm", argv[1], path ? "neogpu" : "direct") < (int)sizeof(capture_path));
        REQUIRE(setenv("NEOGPU_CAPTURE", capture_path, 1) == 0);
        verify_mode = path + 1; draw(path != 0);
    }
    verify_mode = 0; free(reference); reference = NULL;
    for (u32 pair = 1; pair <= 5; pair++) {
        for (u32 order = 0; order < 2; order++) {
            bool native = pair % 2 ? order != 0 : order == 0;
            for (u32 i = 0; i < 8; i++) draw(native);
            glFinish(); REQUIRE(glGetError() == GL_NO_ERROR);
            Timing timings[FRAMES];
            timed = true;
            u64 block_start = ns();
            for (u32 i = 0; i < FRAMES; i++) {
                current = &timings[i]; start = ns(); draw(native);
                current->total = ns() - start;
            }
            u64 block_ns = ns() - block_start;
            timed = false;
            REQUIRE(glGetError() == GL_NO_ERROR);
            for (u32 i = 0; i < FRAMES; i++)
                printf("{\"record\":\"frame\",\"pair\":%u,\"path\":\"%s\",\"frame\":%u,"
                       "\"cpu_submit_ns\":%" PRIu64 ",\"completion_ns\":%" PRIu64 ",\"finish_wait_ns\":%" PRIu64
                       ",\"swap_ns\":%" PRIu64 ",\"total_ns\":%" PRIu64 "}\n", pair, native ? "neogpu" : "direct", i,
                       (uint64_t)timings[i].submit, (uint64_t)timings[i].completion, (uint64_t)timings[i].finish_wait,
                       (uint64_t)timings[i].swap, (uint64_t)timings[i].total);
            printf("{\"record\":\"block\",\"pair\":%u,\"path\":\"%s\",\"frames\":%u,\"elapsed_ns\":%" PRIu64 "}\n",
                   pair, native ? "neogpu" : "direct", FRAMES, (uint64_t)block_ns);
        }
    }
    active = false;
    glDeleteBuffers(2, vbos); glDeleteProgram(program);
    hs_gpu_shutdown(&gpu); hs_fabric_destroy(&fabric);
    return 0;
}
