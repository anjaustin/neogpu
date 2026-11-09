#define _POSIX_C_SOURCE 200809L
#include "hs_gpu.h"
#include "hs_backend_gles.h"
#include "hs_graphics.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "%d: %s\n", __LINE__, #x); exit(1); } } while (0)
static HSGpu gpu;
static const char* scene;
enum { SINGLE, SAME, ALTERNATE, DIRTY, MIXED, FALLBACK, RETAINED };
static u32 scene_id;
static const char* prefix;
static bool recording, audit, active, observe, skip_clear;
static u32 frame_index, frames, swaps, commands;
static GLint width, height;
typedef struct { u64 submit, swap, total; } Timing;
static Timing* current;
static u64 frame_start;
static u64 ns(void) {
    struct timespec t; REQUIRE(clock_gettime(CLOCK_MONOTONIC, &t) == 0);
    return (u64)t.tv_sec * 1000000000ull + t.tv_nsec;
}
static u64 process_ns(void) {
    struct timespec t; REQUIRE(clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t) == 0);
    return (u64)t.tv_sec * 1000000000ull + t.tv_nsec;
}

#ifdef HS_GLES_AUDIT_COUNTS
static bool counting;
static struct { u64 program, bind, enable, pointer, draw, upload; } calls;
extern void __real_glUseProgram(GLuint);
void __wrap_glUseProgram(GLuint p) { if (counting) calls.program++; __real_glUseProgram(p); }
extern void __real_glBindBuffer(GLenum, GLuint);
void __wrap_glBindBuffer(GLenum target, GLuint b) { if (counting) calls.bind++; __real_glBindBuffer(target, b); }
extern void __real_glEnableVertexAttribArray(GLuint);
void __wrap_glEnableVertexAttribArray(GLuint a) { if (counting) calls.enable++; __real_glEnableVertexAttribArray(a); }
extern void __real_glVertexAttribPointer(GLuint, GLint, GLenum, GLboolean, GLsizei, const void*);
void __wrap_glVertexAttribPointer(GLuint i, GLint s, GLenum t, GLboolean n, GLsizei stride, const void* p) {
    if (counting) calls.pointer++;
    __real_glVertexAttribPointer(i, s, t, n, stride, p);
}
extern void __real_glDrawArrays(GLenum, GLint, GLsizei);
void __wrap_glDrawArrays(GLenum m, GLint first, GLsizei count) { if (counting) calls.draw++; __real_glDrawArrays(m, first, count); }
extern void __real_glBufferData(GLenum, GLsizeiptr, const void*, GLenum);
void __wrap_glBufferData(GLenum t, GLsizeiptr size, const void* p, GLenum usage) {
    if (counting) calls.upload++;
    __real_glBufferData(t, size, p, usage);
}
#endif

static void pixels(void) {
    size_t bytes = (size_t)width * height * 4;
    u8* rgba = malloc(bytes); REQUIRE(rgba);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
    REQUIRE(glGetError() == GL_NO_ERROR);
    if (scene_id == SINGLE || scene_id == SAME || scene_id == ALTERNATE || scene_id == RETAINED) {
        for (u32 side = 0; side < 2; side++) {
            const u8* p = rgba + ((size_t)(height / 2) * width + width * (side ? 3 : 1) / 4) * 4;
            for (u32 ch = 0; ch < 3; ch++) {
                bool lit = side == 0 ? ch == 0 : (scene_id != SINGLE && scene_id != SAME && ch == 1);
                REQUIRE(p[ch] == (lit ? 255 : 0));
            }
        }
    }
    char path[512];
    REQUIRE(snprintf(path, sizeof(path), "%s-%u.rgba", prefix, frame_index) < (int)sizeof(path));
    FILE* file = fopen(path, "wb"); REQUIRE(file);
    REQUIRE(fwrite(rgba, 1, bytes, file) == bytes && fclose(file) == 0);
    REQUIRE(snprintf(path, sizeof(path), "%s-%u.ppm", prefix, frame_index) < (int)sizeof(path));
    file = fopen(path, "wb"); REQUIRE(file);
    REQUIRE(fprintf(file, "P6\n%d %d\n255\n", width, height) > 0);
    for (GLint y = height - 1; y >= 0; y--) for (GLint x = 0; x < width; x++)
        REQUIRE(fwrite(rgba + ((size_t)y * width + x) * 4, 1, 3, file) == 3);
    REQUIRE(fclose(file) == 0); free(rgba);
}
extern EGLBoolean __real_eglSwapBuffers(EGLDisplay, EGLSurface);
EGLBoolean __wrap_eglSwapBuffers(EGLDisplay d, EGLSurface s) {
    if (!active) return __real_eglSwapBuffers(d, s);
    swaps++;
    if (observe) pixels();
    u64 before = ns();
    if (current) current->submit = before - frame_start;
    EGLBoolean result = __real_eglSwapBuffers(d, s); REQUIRE(result == EGL_TRUE);
    if (current) current->swap = ns() - before;
    return result;
}
static void geometry(u32 bank) {
    HSBuffer* buffer = hs_gpu_get_buffer(&gpu, bank);
    REQUIRE(hs_buffer_init(buffer, bank, 256));
    f32 center = bank ? 0.5f : -0.5f;
    const f32 data[15] = {center, 0.6f, bank == 0, bank == 1, 0,
        center - 0.3f, -0.6f, bank == 0, bank == 1, 0,
        center + 0.3f, -0.6f, bank == 0, bank == 1, 0};
    for (u32 i = 0; i < 15; i++) hs_buffer_set_f32(buffer, i, data[i]);
    buffer->length = sizeof(data);
}
static void render(void) {
    if (scene_id == DIRTY) for (u32 bank = 0; bank < 2; bank++) {
        HSBuffer* buffer = hs_gpu_get_buffer(&gpu, bank);
        hs_buffer_set_f32(buffer, 4, frame_index % 2 ? 0.25f : 0.0f);
        buffer->length = frame_index % 2 ? 80 : 60;
    }
    hs_gpu_begin_frame(&gpu);
    if (!skip_clear) hs_gpu_clear(&gpu, v4_make(0, 0, 0, 1));
    if (scene_id == SINGLE) hs_gpu_draw(&gpu, 0);
    else if (scene_id == SAME || scene_id == ALTERNATE) {
        for (u32 i = 0; i < 64; i++) hs_gpu_draw(&gpu, scene_id == SAME ? 0 : i % 2);
    } else if (scene_id == DIRTY) { hs_gpu_draw(&gpu, 0); hs_gpu_draw(&gpu, 1); }
    else if (scene_id == MIXED) {
        for (u32 i = 0; i < 4; i++) { hs_gpu_draw(&gpu, 0); hs_gpu_show_texture(&gpu, 0); hs_gpu_draw(&gpu, 1); }
    } else if (scene_id == FALLBACK) { hs_gpu_draw(&gpu, 15); hs_gpu_draw(&gpu, 15); }
    else REQUIRE(scene_id == RETAINED);
    u32 processed = hs_gpu_process(&gpu);
    REQUIRE(processed > 0 || (skip_clear && scene_id == RETAINED));
    u32 before = swaps;
    REQUIRE(hs_gpu_end_frame_checked(&gpu) && swaps == before + 1);
}
static void check_log(u32 count) {
    REQUIRE(!hs_gpu_has_overflow(&gpu) && gpu.system.recording == recording);
    REQUIRE(gpu.system.log_head == (recording ? count * commands : 0));
    if (!recording) return;
    const f32 black[4] = {0, 0, 0, 1};
    for (u32 f = 0; f < count; f++) {
        u32 start = f * commands;
        REQUIRE(gpu.log_buffer[start].op == OP_CLEAR && gpu.log_buffer[start].payload_len == sizeof(black));
        REQUIRE(!memcmp(gpu.system.log_payloads[start].data, black, sizeof(black)));
        for (u32 i = 1; i < commands; i++) {
            const Message* msg = &gpu.log_buffer[start + i];
            OpCode expected = !strcmp(scene, "mixed") && (i - 1) % 3 == 1 ? OP_SHOW_TEXTURE : OP_DRAW;
            REQUIRE(msg->op == expected && msg->payload_len == 0);
            u32 bank = 0;
            if (!strcmp(scene, "alternate")) bank = (i - 1) % 2;
            else if (!strcmp(scene, "dirty")) bank = i - 1;
            else if (!strcmp(scene, "fallback")) bank = 15;
            else if (!strcmp(scene, "mixed")) bank = (i - 1) % 3 == 2 ? 1 : 0;
            REQUIRE(msg->payload_idx == bank);
        }
    }
}
#ifdef HS_GLES_AUDIT_COUNTS
static void perturb(void) {
    /* External context access is allowed between backend invocations. */
    GLint program = 0; glGetIntegerv(GL_CURRENT_PROGRAM, &program);
    GLint p = glGetAttribLocation(program, "a_position"), c = glGetAttribLocation(program, "a_color");
    if (p >= 0) glDisableVertexAttribArray(p);
    if (c >= 0) glDisableVertexAttribArray(c);
    glUseProgram(0); glBindBuffer(GL_ARRAY_BUFFER, 0);
    REQUIRE(glGetError() == GL_NO_ERROR);
}
#endif
int main(int argc, char** argv) {
    REQUIRE(argc == 6); alarm(180);
    scene = argv[1]; recording = !strcmp(argv[2], "on");
    const char* names[] = {"single", "same", "alternate", "dirty", "mixed", "fallback", "retained"};
    for (scene_id = 0; scene_id < sizeof(names) / sizeof(names[0]); scene_id++) if (!strcmp(scene, names[scene_id])) break;
    REQUIRE(scene_id <= RETAINED);
    REQUIRE(recording || !strcmp(argv[2], "off"));
    frames = (u32)strtoul(argv[3], NULL, 10); prefix = argv[4]; audit = !strcmp(argv[5], "audit");
    REQUIRE(audit || !strcmp(argv[5], "timed"));
    REQUIRE(frames >= 3 && frames <= 512 && strncmp(prefix, "tests/evidence-", 15) == 0);
    if (!strcmp(scene, "same") || !strcmp(scene, "alternate")) commands = 65;
    else if (!strcmp(scene, "single")) commands = 2;
    else if (!strcmp(scene, "mixed")) commands = 13;
    else if (!strcmp(scene, "retained")) commands = 1;
    else { REQUIRE(!strcmp(scene, "dirty") || !strcmp(scene, "fallback")); commands = 3; }
    REQUIRE(frames * commands < HS_MAX_MSG_LOG);
    u64 init_start = ns();
    hs_gpu_init(&gpu); REQUIRE(hs_gpu_participants_ready(&gpu) && gpu.system.recording);
    if (!recording) hs_gpu_stop_recording(&gpu);
    geometry(0); geometry(1);
    if (!strcmp(scene, "retained")) for (u32 bank = 0; bank < 2; bank++) {
        HSParticipant owner;
        REQUIRE(hs_gpu_participant_join(&gpu, &owner) == HS_FABRIC_OK && owner.slot == bank);
        HSParticipantFrame contribution = {.sequence = 1, .count = 32};
        for (u32 i = 0; i < 32; i++) contribution.draws[i] = (HSRenderCmd){.op = HS_RC_DRAW, .a = bank};
        REQUIRE(hs_gpu_participant_submit(&gpu, owner, &contribution) == HS_FABRIC_OK);
    }
    HSBackend backend = hs_backend_gles_create(); hs_gpu_attach_backend(&gpu, &backend); REQUIRE(gpu.backend);
    u64 init_ns = ns() - init_start;
    GLint viewport[4]; glGetIntegerv(GL_VIEWPORT, viewport); width = viewport[2]; height = viewport[3];
    REQUIRE(width > 0 && height > 0);
    printf("{\"record\":\"context\",\"width\":%d,\"height\":%d,\"renderer\":\"%s\",\"version\":\"%s\"}\n", width, height, glGetString(GL_RENDERER), glGetString(GL_VERSION));
    active = true;
    u64 cold_start = ns(); render(); glFinish(); u64 cold_ns = ns() - cold_start;
    check_log(1);
    printf("{\"record\":\"cold\",\"elapsed_ns\":%" PRIu64 ",\"journal_allocated\":%s}\n", cold_ns, gpu.system.log_payloads ? "true" : "false");
    hs_gpu_clear_log(&gpu);
    for (u32 i = 0; i < 16; i++) { frame_index = i; render(); }
    hs_gpu_clear_log(&gpu); glFinish(); REQUIRE(glGetError() == GL_NO_ERROR);
    if (audit) {
#ifdef HS_GLES_AUDIT_COUNTS
        for (u32 i = 0; i < 3; i++) {
            perturb(); counting = true; observe = true; frame_index = i;
            render(); counting = false; observe = false;
        }
        check_log(3);
        /* No message CLEAR to accidentally invalidate a persistent cache. */
        hs_gpu_clear_log(&gpu); perturb();
        glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
        skip_clear = true; observe = true; frame_index = 3; render(); observe = false;
        REQUIRE(!hs_gpu_has_overflow(&gpu) && gpu.system.recording == recording);
        REQUIRE(gpu.system.log_head == (recording ? commands - 1 : 0));
        for (u32 i = 0; i < gpu.system.log_head; i++) REQUIRE(gpu.log_buffer[i].op != OP_CLEAR && gpu.log_buffer[i].payload_len == 0);
        printf("{\"record\":\"calls\",\"program\":%" PRIu64 ",\"bind\":%" PRIu64 ",\"enable\":%" PRIu64 ",\"pointer\":%" PRIu64 ",\"draw\":%" PRIu64 ",\"upload\":%" PRIu64 "}\n",
               calls.program, calls.bind, calls.enable, calls.pointer, calls.draw, calls.upload);
#else
        REQUIRE(false);
#endif
    } else {
        Timing* timings = calloc(frames, sizeof(*timings)); REQUIRE(timings);
        u64 cpu_start = process_ns(), start = ns();
        for (u32 i = 0; i < frames; i++) {
            current = &timings[i]; frame_index = i; frame_start = ns(); render();
            current->total = ns() - frame_start;
        }
        current = NULL; u64 presented_ns = ns() - start;
        u64 finish_start = ns(); glFinish(); u64 finish_ns = ns() - finish_start;
        u64 process_cpu_ns = process_ns() - cpu_start;
        REQUIRE(glGetError() == GL_NO_ERROR); check_log(frames);
        for (u32 i = 0; i < frames; i++) printf("{\"record\":\"frame\",\"index\":%u,\"submit_ns\":%" PRIu64 ",\"swap_ns\":%" PRIu64 ",\"total_ns\":%" PRIu64 "}\n", i, timings[i].submit, timings[i].swap, timings[i].total);
        printf("{\"record\":\"block\",\"scene\":\"%s\",\"capture\":\"%s\",\"frames\":%u,\"messages\":%u,\"presented_ns\":%" PRIu64 ",\"end_finish_ns\":%" PRIu64 ",\"process_cpu_ns\":%" PRIu64 ",\"journal_count\":%u}\n",
               scene, recording ? "on" : "off", frames, frames * commands, presented_ns, finish_ns, process_cpu_ns, gpu.system.log_head);
        free(timings);
    }
    active = false;
    u64 shutdown_cpu_start = process_ns(), shutdown_start = ns();
    hs_gpu_shutdown(&gpu);
    u64 shutdown_ns = ns() - shutdown_start, shutdown_cpu_ns = process_ns() - shutdown_cpu_start;
    printf("{\"record\":\"lifecycle\",\"init_ns\":%" PRIu64 ",\"shutdown_ns\":%" PRIu64 ",\"shutdown_cpu_ns\":%" PRIu64 ",\"context_released\":%s}\n",
           init_ns, shutdown_ns, shutdown_cpu_ns, eglGetCurrentContext() == EGL_NO_CONTEXT ? "true" : "false");
    return 0;
}
