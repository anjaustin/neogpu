/* Diagnostic LD_PRELOAD shim: records every DRM page-flip completion (vblank
 * sequence, kernel timestamp, and which swap's frame it showed) and every
 * eglSwapBuffers call, unmodified binaries included. Written at exit to
 * $PACING_TRACE (default pacing-trace.txt): "F seq t_us swap_index" per flip,
 * "S t_us" per swap, "Q t_us swap_index" per page-flip queued (swap_index
 * counts swaps from 1; -1 if unknown).
 *
 * Build: cc -O2 -shared -fPIC tests/pacing_trace.c -o tests/pacing_trace.so -ldl
 * Use:   PACING_TRACE=out.txt LD_PRELOAD=$PWD/tests/pacing_trace.so sudo -E ./tests/test_08_raycast_orbit
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <xf86drm.h>
#include <EGL/egl.h>
#include <gbm.h>
#include <xf86drmMode.h>

/* fb id -> GEM handle (from AddFB), GEM handle -> swap index (from lock). */
#define MAP 64
static uint32_t map_fb[MAP], map_fb_handle[MAP]; static unsigned map_fb_n;
static uint32_t map_h[MAP]; static long map_h_swap[MAP]; static unsigned map_h_n;
static long queued[64]; static unsigned q_head, q_tail; /* flips in order */
static double qt[1 << 17]; static long qs[1 << 17]; static unsigned qn; /* flip queue calls */

#define CAP (1 << 20)
static unsigned fseq[CAP / 8]; static double ft[CAP / 8]; static long fswap[CAP / 8]; static unsigned fn;
static double st[CAP]; static unsigned sn;
static void (*user_handler)(int, unsigned, unsigned, unsigned, void*);

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static void flip(int fd, unsigned seq, unsigned sec, unsigned usec, void* data) {
    long shown = q_head != q_tail ? queued[q_head++ % 64] : -1;
    if (fn < CAP / 8) { fseq[fn] = seq; ft[fn] = sec * 1e6 + usec; fswap[fn] = shown; fn++; }
    user_handler(fd, seq, sec, usec, data);
}

int drmHandleEvent(int fd, drmEventContextPtr ctx) {
    static int (*real)(int, drmEventContextPtr);
    if (!real) real = dlsym(RTLD_NEXT, "drmHandleEvent");
    drmEventContext copy = *ctx;
    if (ctx->page_flip_handler) { user_handler = ctx->page_flip_handler; copy.page_flip_handler = flip; }
    return real(fd, &copy);
}

static void remember_fb(uint32_t fb, uint32_t handle) {
    for (unsigned i = 0; i < map_fb_n; i++) if (map_fb[i] == fb) { map_fb_handle[i] = handle; return; }
    if (map_fb_n < MAP) { map_fb[map_fb_n] = fb; map_fb_handle[map_fb_n++] = handle; }
}

int drmModeAddFB(int fd, uint32_t w, uint32_t h, uint8_t depth, uint8_t bpp, uint32_t pitch, uint32_t bo, uint32_t* id) {
    static int (*real)(int, uint32_t, uint32_t, uint8_t, uint8_t, uint32_t, uint32_t, uint32_t*);
    if (!real) real = dlsym(RTLD_NEXT, "drmModeAddFB");
    int rc = real(fd, w, h, depth, bpp, pitch, bo, id);
    if (rc == 0) remember_fb(*id, bo);
    return rc;
}

int drmModeAddFB2WithModifiers(int fd, uint32_t w, uint32_t h, uint32_t fmt, const uint32_t handles[4],
                               const uint32_t pitches[4], const uint32_t offsets[4], const uint64_t mods[4],
                               uint32_t* id, uint32_t flags) {
    static int (*real)(int, uint32_t, uint32_t, uint32_t, const uint32_t*, const uint32_t*, const uint32_t*,
                       const uint64_t*, uint32_t*, uint32_t);
    if (!real) real = dlsym(RTLD_NEXT, "drmModeAddFB2WithModifiers");
    int rc = real(fd, w, h, fmt, handles, pitches, offsets, mods, id, flags);
    if (rc == 0) remember_fb(*id, handles[0]);
    return rc;
}

struct gbm_bo* gbm_surface_lock_front_buffer(struct gbm_surface* s) {
    static struct gbm_bo* (*real)(struct gbm_surface*);
    if (!real) real = dlsym(RTLD_NEXT, "gbm_surface_lock_front_buffer");
    struct gbm_bo* bo = real(s);
    if (bo) {
        uint32_t h = gbm_bo_get_handle(bo).u32;
        unsigned i = 0;
        while (i < map_h_n && map_h[i] != h) i++;
        if (i == map_h_n && map_h_n < MAP) map_h[map_h_n++] = h;
        if (i < MAP) map_h_swap[i] = (long)sn;
    }
    return bo;
}

int drmModePageFlip(int fd, uint32_t crtc, uint32_t fb, uint32_t flags, void* data) {
    static int (*real)(int, uint32_t, uint32_t, uint32_t, void*);
    if (!real) real = dlsym(RTLD_NEXT, "drmModePageFlip");
    int rc = real(fd, crtc, fb, flags, data);
    if (rc == 0) {
        long swap = -1;
        for (unsigned i = 0; i < map_fb_n; i++)
            if (map_fb[i] == fb)
                for (unsigned j = 0; j < map_h_n; j++)
                    if (map_h[j] == map_fb_handle[i]) swap = map_h_swap[j];
        queued[q_tail++ % 64] = swap;
        if (qn < (1 << 17)) { qt[qn] = now_us(); qs[qn++] = swap; }
    }
    return rc;
}

EGLBoolean eglSwapBuffers(EGLDisplay d, EGLSurface s) {
    static EGLBoolean (*real)(EGLDisplay, EGLSurface);
    if (!real) real = dlsym(RTLD_NEXT, "eglSwapBuffers");
    if (sn < CAP) st[sn++] = now_us();
    return real(d, s);
}

__attribute__((destructor)) static void dump(void) {
    const char* path = getenv("PACING_TRACE");
    FILE* f = fopen(path ? path : "pacing-trace.txt", "w");
    if (!f) return;
    for (unsigned i = 0; i < fn; i++) fprintf(f, "F %u %.0f %ld\n", fseq[i], ft[i], fswap[i]);
    for (unsigned i = 0; i < sn; i++) fprintf(f, "S %.0f\n", st[i]);
    for (unsigned i = 0; i < qn; i++) fprintf(f, "Q %.0f %ld\n", qt[i], qs[i]);
    fclose(f);
}
