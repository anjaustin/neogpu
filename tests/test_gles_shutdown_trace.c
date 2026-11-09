#define _POSIX_C_SOURCE 200809L
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <gbm.h>
#include <xf86drm.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

typedef struct { const char* name; uint64_t start, wall, cpu; long result; } Event;
static Event events[64];
static unsigned count;
static bool active;
static uint64_t origin;
static uint64_t now(clockid_t clock) {
    struct timespec t; if (clock_gettime(clock, &t)) abort();
    return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec;
}
void hs_shutdown_trace_begin(void) { count = 0; origin = now(CLOCK_MONOTONIC); active = true; }
void hs_shutdown_trace_end(void) {
    active = false;
    for (unsigned i = 0; i < count; i++) printf("{\"record\":\"shutdown_call\",\"name\":\"%s\",\"start_ns\":%" PRIu64 ",\"wall_ns\":%" PRIu64 ",\"cpu_ns\":%" PRIu64 ",\"result\":%ld}\n",
        events[i].name, events[i].start, events[i].wall, events[i].cpu, events[i].result);
}
static void record(const char* name, uint64_t start, uint64_t cpu, long result) {
    uint64_t end = now(CLOCK_MONOTONIC), cpu_end = now(CLOCK_PROCESS_CPUTIME_ID);
    if (count >= sizeof(events) / sizeof(events[0])) abort();
    events[count++] = (Event){name, start - origin, end - start, cpu_end - cpu, result};
}
#define WRAP_RETURN(type, name, params, args) \
    extern type __real_##name params; \
    type __wrap_##name params { \
        if (!active) return __real_##name args; \
        uint64_t start = now(CLOCK_MONOTONIC), cpu = now(CLOCK_PROCESS_CPUTIME_ID); \
        type result = __real_##name args; record(#name, start, cpu, (long)result); return result; \
    }
#define WRAP_VOID(name, params, args) \
    extern void __real_##name params; \
    void __wrap_##name params { \
        if (!active) { __real_##name args; return; } \
        uint64_t start = now(CLOCK_MONOTONIC), cpu = now(CLOCK_PROCESS_CPUTIME_ID); \
        __real_##name args; record(#name, start, cpu, 0); \
    }
WRAP_RETURN(EGLBoolean, eglMakeCurrent, (EGLDisplay d, EGLSurface draw, EGLSurface read, EGLContext c), (d, draw, read, c))
WRAP_RETURN(EGLBoolean, eglDestroySurface, (EGLDisplay d, EGLSurface s), (d, s))
WRAP_RETURN(EGLBoolean, eglDestroyContext, (EGLDisplay d, EGLContext c), (d, c))
WRAP_RETURN(EGLBoolean, eglTerminate, (EGLDisplay d), (d))
WRAP_RETURN(EGLBoolean, eglReleaseThread, (void), ())
WRAP_VOID(gbm_surface_destroy, (struct gbm_surface* s), (s))
WRAP_VOID(gbm_device_destroy, (struct gbm_device* d), (d))
WRAP_RETURN(int, drmDropMaster, (int fd), (fd))
WRAP_RETURN(int, close, (int fd), (fd))
WRAP_VOID(glDeleteBuffers, (GLsizei n, const GLuint* b), (n, b))
WRAP_VOID(glDeleteProgram, (GLuint p), (p))
WRAP_VOID(glDeleteTextures, (GLsizei n, const GLuint* t), (n, t))
