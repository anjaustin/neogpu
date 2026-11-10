/* LD_PRELOAD diagnostic: per-call wall time for the EGL/GBM/DRM/GL calls that
 * the application itself makes. Kernel-internal driver time is not included
 * (strace covers that). Diagnostic only; it is not part of production.
 *
 * Build: gcc -O2 -shared -fPIC -o tests/gles_call_timer.so \
 *        tests/gles_call_timer.c -ldl -lpthread
 * Use:   sudo env LD_PRELOAD=$PWD/tests/gles_call_timer.so \
 *        GLES_TIMER_OUT=path ./benchmark ...
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <gbm.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#define MAX_NAMES 64
typedef struct { const char* name; uint64_t total_ns, max_ns; uint64_t calls; } Entry;
static Entry entries[MAX_NAMES];
static int entry_count;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
static Entry* entry(const char* name) {
    pthread_mutex_lock(&lock);
    for (int i = 0; i < entry_count; i++) if (!strcmp(entries[i].name, name)) {
        pthread_mutex_unlock(&lock); return &entries[i];
    }
    if (entry_count >= MAX_NAMES) { pthread_mutex_unlock(&lock); return NULL; }
    Entry* e = &entries[entry_count++];
    e->name = name;
    pthread_mutex_unlock(&lock);
    return e;
}
static void record(const char* name, uint64_t start) {
    uint64_t elapsed = now_ns() - start;
    Entry* e = entry(name);
    if (!e) return;
    pthread_mutex_lock(&lock);
    e->calls++; e->total_ns += elapsed;
    if (elapsed > e->max_ns) e->max_ns = elapsed;
    pthread_mutex_unlock(&lock);
}

typedef void (*resolve_fn)(void);
#define REAL(name) \
    static resolve_fn name##_slot; \
    static void name##_resolve(void) { \
        if (!name##_slot) name##_slot = (resolve_fn)dlsym(RTLD_NEXT, #name); \
    }
#define CALL(name, type, args, call_args) do { \
    name##_resolve(); \
    uint64_t start = now_ns(); \
    type result = ((type (*)args)name##_slot) call_args; \
    record(#name, start); \
    return result; \
} while (0)
#define VOID_CALL(name, args, call_args) do { \
    name##_resolve(); \
    uint64_t start = now_ns(); \
    ((void (*)args)name##_slot) call_args; \
    record(#name, start); \
} while (0)

REAL(eglInitialize) REAL(eglChooseConfig) REAL(eglCreateContext) REAL(eglCreateWindowSurface)
REAL(eglMakeCurrent) REAL(eglSwapBuffers) REAL(eglSwapInterval) REAL(eglDestroySurface)
REAL(eglDestroyContext) REAL(eglTerminate) REAL(eglReleaseThread) REAL(eglGetError)
REAL(gbm_create_device) REAL(gbm_surface_create) REAL(gbm_surface_destroy)
REAL(gbm_device_destroy) REAL(gbm_surface_lock_front_buffer) REAL(gbm_surface_release_buffer)
REAL(drmSetMaster) REAL(drmDropMaster) REAL(drmModeGetResources) REAL(drmModeGetConnector)
REAL(drmModeGetEncoder) REAL(drmModeAddFB) REAL(drmModeSetCrtc) REAL(drmModeFreeConnector)
REAL(drmModeFreeResources) REAL(drmModeFreeEncoder) REAL(glFinish) REAL(glClear)
REAL(glDrawArrays) REAL(glReadPixels) REAL(glGetError)

EGLBoolean eglInitialize(EGLDisplay d, EGLint* major, EGLint* minor)
{ CALL(eglInitialize, EGLBoolean, (EGLDisplay, EGLint*, EGLint*), (d, major, minor)); }
EGLBoolean eglChooseConfig(EGLDisplay d, const EGLint* attribs, EGLConfig* configs, EGLint size, EGLint* count)
{ CALL(eglChooseConfig, EGLBoolean, (EGLDisplay, const EGLint*, EGLConfig*, EGLint, EGLint*), (d, attribs, configs, size, count)); }
EGLContext eglCreateContext(EGLDisplay d, EGLConfig c, EGLContext share, const EGLint* attribs)
{ CALL(eglCreateContext, EGLContext, (EGLDisplay, EGLConfig, EGLContext, const EGLint*), (d, c, share, attribs)); }
EGLSurface eglCreateWindowSurface(EGLDisplay d, EGLConfig c, EGLNativeWindowType win, const EGLint* attribs)
{ CALL(eglCreateWindowSurface, EGLSurface, (EGLDisplay, EGLConfig, EGLNativeWindowType, const EGLint*), (d, c, win, attribs)); }
EGLBoolean eglMakeCurrent(EGLDisplay d, EGLSurface draw, EGLSurface read, EGLContext context)
{ CALL(eglMakeCurrent, EGLBoolean, (EGLDisplay, EGLSurface, EGLSurface, EGLContext), (d, draw, read, context)); }
EGLBoolean eglSwapBuffers(EGLDisplay d, EGLSurface s)
{ CALL(eglSwapBuffers, EGLBoolean, (EGLDisplay, EGLSurface), (d, s)); }
EGLBoolean eglSwapInterval(EGLDisplay d, EGLint interval)
{ CALL(eglSwapInterval, EGLBoolean, (EGLDisplay, EGLint), (d, interval)); }
EGLBoolean eglDestroySurface(EGLDisplay d, EGLSurface s)
{ CALL(eglDestroySurface, EGLBoolean, (EGLDisplay, EGLSurface), (d, s)); }
EGLBoolean eglDestroyContext(EGLDisplay d, EGLContext c)
{ CALL(eglDestroyContext, EGLBoolean, (EGLDisplay, EGLContext), (d, c)); }
EGLBoolean eglTerminate(EGLDisplay d) { CALL(eglTerminate, EGLBoolean, (EGLDisplay), (d)); }
EGLBoolean eglReleaseThread(void) { CALL(eglReleaseThread, EGLBoolean, (void), ()); }
EGLint eglGetError(void) { CALL(eglGetError, EGLint, (void), ()); }

struct gbm_device* gbm_create_device(int fd) { CALL(gbm_create_device, struct gbm_device*, (int), (fd)); }
struct gbm_surface* gbm_surface_create(struct gbm_device* dev, uint32_t w, uint32_t h, uint32_t format, uint32_t flags)
{ CALL(gbm_surface_create, struct gbm_surface*, (struct gbm_device*, uint32_t, uint32_t, uint32_t, uint32_t), (dev, w, h, format, flags)); }
void gbm_surface_destroy(struct gbm_surface* s) { VOID_CALL(gbm_surface_destroy, (struct gbm_surface*), (s)); }
void gbm_device_destroy(struct gbm_device* d) { VOID_CALL(gbm_device_destroy, (struct gbm_device*), (d)); }
struct gbm_bo* gbm_surface_lock_front_buffer(struct gbm_surface* s)
{ CALL(gbm_surface_lock_front_buffer, struct gbm_bo*, (struct gbm_surface*), (s)); }
void gbm_surface_release_buffer(struct gbm_surface* s, struct gbm_bo* bo)
{ VOID_CALL(gbm_surface_release_buffer, (struct gbm_surface*, struct gbm_bo*), (s, bo)); }

int drmSetMaster(int fd) { CALL(drmSetMaster, int, (int), (fd)); }
int drmDropMaster(int fd) { CALL(drmDropMaster, int, (int), (fd)); }
drmModeRes* drmModeGetResources(int fd) { CALL(drmModeGetResources, drmModeRes*, (int), (fd)); }
drmModeConnector* drmModeGetConnector(int fd, uint32_t id) { CALL(drmModeGetConnector, drmModeConnector*, (int, uint32_t), (fd, id)); }
drmModeEncoder* drmModeGetEncoder(int fd, uint32_t id) { CALL(drmModeGetEncoder, drmModeEncoder*, (int, uint32_t), (fd, id)); }
int drmModeAddFB(int fd, uint32_t w, uint32_t h, uint8_t depth, uint8_t bpp, uint32_t stride, uint32_t handle, uint32_t* fb)
{ CALL(drmModeAddFB, int, (int, uint32_t, uint32_t, uint8_t, uint8_t, uint32_t, uint32_t, uint32_t*), (fd, w, h, depth, bpp, stride, handle, fb)); }
int drmModeSetCrtc(int fd, uint32_t crtc, uint32_t fb, uint32_t x, uint32_t y, uint32_t* connectors, int count, drmModeModeInfo* mode)
{ CALL(drmModeSetCrtc, int, (int, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t*, int, drmModeModeInfo*), (fd, crtc, fb, x, y, connectors, count, mode)); }
void drmModeFreeConnector(drmModeConnector* c) { VOID_CALL(drmModeFreeConnector, (drmModeConnector*), (c)); }
void drmModeFreeResources(drmModeRes* r) { VOID_CALL(drmModeFreeResources, (drmModeRes*), (r)); }
void drmModeFreeEncoder(drmModeEncoder* e) { VOID_CALL(drmModeFreeEncoder, (drmModeEncoder*), (e)); }

void glFinish(void) { VOID_CALL(glFinish, (void), ()); }
void glClear(GLbitfield mask) { VOID_CALL(glClear, (GLbitfield), (mask)); }
void glDrawArrays(GLenum mode, GLint first, GLsizei count) { VOID_CALL(glDrawArrays, (GLenum, GLint, GLsizei), (mode, first, count)); }
void glReadPixels(GLint x, GLint y, GLsizei w, GLsizei h, GLenum format, GLenum type, void* pixels)
{ VOID_CALL(glReadPixels, (GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void*), (x, y, w, h, format, type, pixels)); }
GLenum glGetError(void) { CALL(glGetError, GLenum, (void), ()); }

__attribute__((destructor)) static void dump(void) {
    const char* path = getenv("GLES_TIMER_OUT");
    if (!path) return;
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return;
    char buffer[256];
    int length = snprintf(buffer, sizeof(buffer), "call,calls,total_ns,max_ns\n");
    if (write(fd, buffer, (size_t)length) < 0) { close(fd); return; }
    for (int i = 0; i < entry_count; i++) {
        Entry* e = &entries[i];
        length = snprintf(buffer, sizeof(buffer), "%s,%llu,%llu,%llu\n", e->name,
                          (unsigned long long)e->calls, (unsigned long long)e->total_ns,
                          (unsigned long long)e->max_ns);
        if (length > 0 && write(fd, buffer, (size_t)length) < 0) break;
    }
    close(fd);
}
