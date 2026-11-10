/* Hardware check: every presented frame reaches the display via a page flip.
 *
 * Renders a known sequence of solid colours (frame i cleared to PALETTE[i % 4])
 * and reports what hs_graphics_present says it did. Run it alongside
 * tests/scanout_probe to confirm from outside the process that the CRTC
 * actually scans out a different buffer each frame with the expected colours.
 *
 * Usage: sudo tests/test_present_flip [frames] [vsync 1|0] [hold_ms_per_frame] [mode]
 *   mode "normal"   - default
 *   mode "idle"     - first disable the active CRTC (as a previous owner that
 *                     exited without a console would leave it); takeover must
 *                     find a CRTC through the encoders' possible_crtcs
 *   mode "mismatch" - corrupt the cached display mode before the first present;
 *                     takeover must refuse (render-only) instead of failing EINVAL
 *   mode "initfail" - hold DRM master elsewhere so init fails; init must leak no
 *                     descriptor and a later init in the same process must work
 * Output: one JSON object; exit 0 only if the run met its own contract.
 */
#include <dirent.h>
#include <string.h>
#include "hs_graphics.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static const float PALETTE[4][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}};

static unsigned open_fds(void) {
    unsigned count = 0;
    DIR* dir = opendir("/proc/self/fd");
    if (!dir) return 0;
    while (readdir(dir)) count++;
    closedir(dir);
    return count;
}

/* Disable the CRTC currently driving a connected connector. Returns the fd
 * used (kept open, without master, so no last-close console restore happens
 * until the caller closes it), or -1. */
static int disable_active_crtc(void) {
    int fd = open("/dev/dri/card1", O_RDWR | O_CLOEXEC);
    if (fd < 0) return -1;
    if (!drmIsMaster(fd) && drmSetMaster(fd) != 0) { close(fd); return -1; }
    drmModeRes* res = drmModeGetResources(fd);
    int disabled = 0;
    for (int i = 0; res && i < res->count_crtcs; i++) {
        drmModeCrtc* crtc = drmModeGetCrtc(fd, res->crtcs[i]);
        if (crtc && crtc->buffer_id &&
            drmModeSetCrtc(fd, crtc->crtc_id, 0, 0, 0, NULL, 0, NULL) == 0) disabled++;
        if (crtc) drmModeFreeCrtc(crtc);
    }
    if (res) drmModeFreeResources(res);
    drmDropMaster(fd);
    if (!disabled) { close(fd); return -1; }
    return fd;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char** argv) {
    int frames = argc > 1 ? atoi(argv[1]) : 240;
    int vsync = argc > 2 ? atoi(argv[2]) : 1;
    int hold_ms = argc > 3 ? atoi(argv[3]) : 0;
    const char* mode = argc > 4 ? argv[4] : "normal";
    if (frames <= 0 || frames > 100000 || hold_ms < 0) return 2;

    static HSGraphics gfx;
    int observer_fd = -1;
    if (!strcmp(mode, "initfail")) {
        unsigned before = open_fds();
        int holder = open("/dev/dri/card1", O_RDWR | O_CLOEXEC);
        bool held = holder >= 0 && (drmIsMaster(holder) || drmSetMaster(holder) == 0);
        bool failed = held && hs_graphics_init(&gfx) != 0;
        unsigned during = open_fds();
        if (holder >= 0) close(holder);
        bool leak_free = during == before + 1; /* only the holder itself */
        bool recovered = hs_graphics_init(&gfx) == 0;
        if (recovered) hs_graphics_finish(&gfx);
        bool ok = held && failed && leak_free && recovered && open_fds() == before;
        printf("{\"mode\":\"initfail\",\"master_held_elsewhere\":%s,\"init_failed\":%s,"
               "\"fds_before\":%u,\"fds_during\":%u,\"later_init_ok\":%s,\"passed\":%s}\n",
               held ? "true" : "false", failed ? "true" : "false", before, during,
               recovered ? "true" : "false", ok ? "true" : "false");
        return ok ? 0 : 1;
    }
    if (!strcmp(mode, "idle")) {
        observer_fd = disable_active_crtc();
        if (observer_fd < 0) { printf("{\"mode\":\"idle\",\"setup\":false}\n"); return 1; }
    }

    double t0 = now_ms();
    if (hs_graphics_init(&gfx) != 0) {
        printf("{\"init\":false}\n");
        if (observer_fd >= 0) close(observer_fd);
        return 1;
    }
    double init_ms = now_ms() - t0;
    gfx.vsync = vsync != 0;
    bool idle_link = !strcmp(mode, "idle") && gfx.display_found && gfx.display_crtc_id;
    if (!strcmp(mode, "mismatch")) gfx.display_mode.hdisplay += 16;

    double start = now_ms();
    for (int i = 0; i < frames; i++) {
        const float* c = PALETTE[i % 4];
        hs_graphics_clear(&gfx, c[0], c[1], c[2], 1.0f);
        hs_graphics_present(&gfx);
        if (hold_ms) usleep((useconds_t)hold_ms * 1000);
    }
    double elapsed = now_ms() - start;
    bool active = gfx.crtc_active;
    /* finish() lands a flip still in flight (vsync off), so count after it. */
    hs_graphics_finish(&gfx);
    unsigned long long presented = gfx.frames_presented, dropped = gfx.frames_dropped;
    unsigned errors = gfx.present_errors;
    int err = gfx.last_present_errno;
    const char* why = gfx.present_error;

    /* Contract: display taken, no errors, and every frame either shown or
     * (vsync off only) explicitly dropped. */
    bool ok = active && errors == 0 && presented + dropped == (unsigned long long)frames &&
              (vsync ? dropped == 0 : true);
    if (!strcmp(mode, "idle")) ok = ok && idle_link;
    if (!strcmp(mode, "mismatch"))  /* must refuse cleanly, once, and keep rendering */
        ok = !active && errors == 1 && presented == 0 && why && strstr(why, "does not match");
    if (observer_fd >= 0) close(observer_fd);
    printf("{\"mode\":\"%s\",\"frames\":%d,\"vsync\":%s,\"init_ms\":%.1f,\"elapsed_ms\":%.1f,\"fps\":%.2f,"
           "\"crtc_active\":%s,\"presented\":%llu,\"dropped\":%llu,\"errors\":%u,\"errno\":%d,"
           "\"present_error\":%s%s%s,\"passed\":%s}\n",
           mode, frames, vsync ? "true" : "false", init_ms, elapsed, frames * 1000.0 / elapsed,
           active ? "true" : "false", presented, dropped, errors, err,
           why ? "\"" : "", why ? why : "null", why ? "\"" : "", ok ? "true" : "false");
    return ok ? 0 : 1;
}
