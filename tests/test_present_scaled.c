/* Hardware check: hs_graphics_init_scaled renders at a reduced size and the
 * display controller (HVS) scales it to the full panel at scanout.
 *
 * Before each takeover another DRM client ("the saboteur") leaves the display
 * in a state a previous program could: the primary plane at quarter alpha,
 * Nearest filter and coverage blending, an overlay plane on the panel, or the
 * CRTC switched off. Contract:
 * - refusals: 0x0 and a size larger than the display fail init;
 * - native size (hs_graphics_init): takeover resets the primary plane's
 *   alpha, blend mode, filter and rotation and turns the stray overlay off;
 * - from a switched-off CRTC, the scaled takeover lights the panel with the
 *   scaled plane;
 * - scaled takeover: the primary plane reads the whole render-size buffer
 *   from (0,0) and covers the whole mode from (0,0), with default properties
 *   and no other plane on the CRTC (kernel atomic state);
 * - cheap frames (thousands per second) and calibrated heavy frames (~0.6
 *   refresh, so frames are held and flipped by the pacing code) each run at
 *   least 3 s with at most 1% of refreshes repeated and no errors; every
 *   frame is presented or dropped;
 * - the render buffer holds what was drawn (glReadPixels);
 * - after hs_graphics_finish the console gets the panel back unscaled, with
 *   default plane properties.
 * What the HVS outputs pixel by pixel is checked through the writeback
 * connector (tests/p0_3_writeback_check.c).
 *
 * Usage: sudo tests/test_present_scaled [width=400] [height=240]
 * Output: one JSON object; exit 0 only if every check passed. */
#define HS_GRAPHICS_SHELVED_SCALED /* P0-3: shelved scaled path */
#include "hs_graphics.h"
#include <stdio.h>
#include <stdlib.h>
#include <drm_fourcc.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static const char* VS = "attribute vec2 p;\nvoid main() { gl_Position = vec4(p, 0.0, 1.0); }\n";
static const char* FS = "precision mediump float;\nuniform vec4 c;\nvoid main() { gl_FragColor = c; }\n";
static const char* FS_HEAVY =
    "precision highp float;\nuniform int n;\nuniform float t;\n"
    "void main() {\n    vec2 q = gl_FragCoord.xy * 0.01 + t;\n    float a = 0.0;\n"
    "    for (int i = 0; i < 4096; i++) {\n        if (i >= n) break;\n"
    "        a += sin(q.x + float(i) * 0.37) * cos(q.y - a);\n    }\n"
    "    gl_FragColor = vec4(fract(a), fract(t), 0.5, 1.0);\n}\n";

static GLuint program(const char* fs) {
    GLuint p = glCreateProgram(), v = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(v, 1, &VS, NULL);
    glCompileShader(v);
    glShaderSource(f, 1, &fs, NULL);
    glCompileShader(f);
    glAttachShader(p, v);
    glAttachShader(p, f);
    glBindAttribLocation(p, 0, "p");
    glLinkProgram(p);
    return p;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int open_card(void) {
    int fd = open("/dev/dri/card1", O_RDWR | O_CLOEXEC);
    return fd >= 0 ? fd : open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
}

/* The primary plane on crtc and what else is there. */
typedef struct {
    bool found;
    uint64_t src_x, src_y, src_w, src_h, dst_x, dst_y, dst_w, dst_h;
    uint64_t alpha, blend, filter, rotation;
    int others; /* other planes on the CRTC */
} PlaneState;

static PlaneState plane_state(int fd, uint32_t crtc) {
    PlaneState s = {0};
    drmSetClientCap(fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1);
    drmSetClientCap(fd, DRM_CLIENT_CAP_ATOMIC, 1);
    drmModePlaneRes* pr = drmModeGetPlaneResources(fd);
    for (uint32_t i = 0; pr && i < pr->count_planes; i++) {
        uint32_t id = pr->planes[i];
        uint64_t type = 0, on = 0;
        hs_graphics_prop(fd, id, DRM_MODE_OBJECT_PLANE, "type", &type);
        hs_graphics_prop(fd, id, DRM_MODE_OBJECT_PLANE, "CRTC_ID", &on);
        if (on != crtc) continue;
        if (type != DRM_PLANE_TYPE_PRIMARY || s.found) { s.others++; continue; }
        s.found = true;
        hs_graphics_prop(fd, id, DRM_MODE_OBJECT_PLANE, "SRC_X", &s.src_x);
        hs_graphics_prop(fd, id, DRM_MODE_OBJECT_PLANE, "SRC_Y", &s.src_y);
        hs_graphics_prop(fd, id, DRM_MODE_OBJECT_PLANE, "SRC_W", &s.src_w);
        hs_graphics_prop(fd, id, DRM_MODE_OBJECT_PLANE, "SRC_H", &s.src_h);
        hs_graphics_prop(fd, id, DRM_MODE_OBJECT_PLANE, "CRTC_X", &s.dst_x);
        hs_graphics_prop(fd, id, DRM_MODE_OBJECT_PLANE, "CRTC_Y", &s.dst_y);
        hs_graphics_prop(fd, id, DRM_MODE_OBJECT_PLANE, "CRTC_W", &s.dst_w);
        hs_graphics_prop(fd, id, DRM_MODE_OBJECT_PLANE, "CRTC_H", &s.dst_h);
        hs_graphics_prop(fd, id, DRM_MODE_OBJECT_PLANE, "alpha", &s.alpha);
        hs_graphics_prop(fd, id, DRM_MODE_OBJECT_PLANE, "pixel blend mode", &s.blend);
        hs_graphics_prop(fd, id, DRM_MODE_OBJECT_PLANE, "SCALING_FILTER", &s.filter);
        hs_graphics_prop(fd, id, DRM_MODE_OBJECT_PLANE, "rotation", &s.rotation);
    }
    drmModeFreePlaneResources(pr);
    return s;
}

/* Full-screen-from-origin rects of the given source size, default properties,
 * nothing else on the CRTC. */
static bool plane_ok(const PlaneState* s, u32 sw, u32 sh, u32 dw, u32 dh) {
    return s->found && s->src_x == 0 && s->src_y == 0 && s->src_w == (uint64_t)sw << 16 &&
           s->src_h == (uint64_t)sh << 16 && s->dst_x == 0 && s->dst_y == 0 && s->dst_w == dw && s->dst_h == dh &&
           s->alpha == 0xffff && s->blend == 0 && s->filter == 0 && s->rotation == 1 && s->others == 0;
}

/* Leave the display as a previous client could: blank=false puts the
 * primary plane at quarter alpha, Nearest, coverage blending and an overlay
 * on the CRTC; blank=true switches the CRTC off with no mode and detaches the
 * connector. Drops master and returns the
 * fd, which must stay open until after the takeover (its framebuffer is the
 * overlay's), or -1. */
static int sabotage(uint32_t crtc, uint32_t connector, bool blank) {
    int fd = open_card();
    if (fd < 0 || drmSetMaster(fd) || drmSetClientCap(fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1) ||
        drmSetClientCap(fd, DRM_CLIENT_CAP_ATOMIC, 1)) {
        if (fd >= 0) close(fd);
        return -1;
    }
    drmModeRes* res = drmModeGetResources(fd);
    int index = -1;
    for (int i = 0; res && i < res->count_crtcs; i++)
        if (res->crtcs[i] == crtc) index = i;
    drmModeFreeResources(res);
    uint32_t primary = 0, overlay = 0;
    drmModePlaneRes* pr = drmModeGetPlaneResources(fd);
    for (uint32_t i = 0; index >= 0 && pr && i < pr->count_planes; i++) {
        drmModePlane* p = drmModeGetPlane(fd, pr->planes[i]);
        uint64_t type = 0;
        if (p && (p->possible_crtcs & (1u << index))) {
            hs_graphics_prop(fd, p->plane_id, DRM_MODE_OBJECT_PLANE, "type", &type);
            if (type == DRM_PLANE_TYPE_PRIMARY && !primary) primary = p->plane_id;
            if (type == DRM_PLANE_TYPE_OVERLAY && !overlay) overlay = p->plane_id;
        }
        drmModeFreePlane(p);
    }
    drmModeFreePlaneResources(pr);
    struct drm_mode_create_dumb cd = {.width = 64, .height = 64, .bpp = 32};
    uint32_t fb = 0;
    bool bad = !primary || !overlay || drmIoctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &cd);
    if (!bad) {
        uint32_t handles[4] = {cd.handle}, pitches[4] = {cd.pitch}, offsets[4] = {0};
        struct drm_mode_map_dumb md = {.handle = cd.handle};
        bad = drmModeAddFB2(fd, 64, 64, DRM_FORMAT_ARGB8888, handles, pitches, offsets, &fb, 0) ||
              drmIoctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &md);
        uint32_t* px = bad ? MAP_FAILED : mmap(0, cd.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, md.offset);
        if (px != MAP_FAILED) {
            for (uint64_t i = 0; i < cd.size / 4; i++) px[i] = 0xff00ff00u; /* opaque green */
            munmap(px, cd.size);
        }
    }
    drmModeAtomicReq* req = drmModeAtomicAlloc();
    struct { uint32_t obj, type; const char* name; uint64_t v; } set[] = {
        {primary, DRM_MODE_OBJECT_PLANE, "alpha", 0x4000},
        {primary, DRM_MODE_OBJECT_PLANE, "SCALING_FILTER", 1},
        {primary, DRM_MODE_OBJECT_PLANE, "pixel blend mode", 1},
        {overlay, DRM_MODE_OBJECT_PLANE, "CRTC_ID", crtc},
        {overlay, DRM_MODE_OBJECT_PLANE, "FB_ID", fb},
        {overlay, DRM_MODE_OBJECT_PLANE, "SRC_W", 64u << 16},
        {overlay, DRM_MODE_OBJECT_PLANE, "SRC_H", 64u << 16},
        {overlay, DRM_MODE_OBJECT_PLANE, "CRTC_W", 64},
        {overlay, DRM_MODE_OBJECT_PLANE, "CRTC_H", 64},
        {primary, DRM_MODE_OBJECT_PLANE, "CRTC_ID", 0},
        {primary, DRM_MODE_OBJECT_PLANE, "FB_ID", 0},
        {crtc, DRM_MODE_OBJECT_CRTC, "ACTIVE", 0},
        {crtc, DRM_MODE_OBJECT_CRTC, "MODE_ID", 0},
        {connector, DRM_MODE_OBJECT_CONNECTOR, "CRTC_ID", 0},
    };
    size_t first = blank ? 9 : 0, last = blank ? 14 : 9;
    for (size_t i = first; !bad && i < last; i++) {
        uint32_t id = hs_graphics_prop(fd, set[i].obj, set[i].type, set[i].name, NULL);
        bad = !id || drmModeAtomicAddProperty(req, set[i].obj, id, set[i].v) < 0;
    }
    bad = bad || drmModeAtomicCommit(fd, req, DRM_MODE_ATOMIC_ALLOW_MODESET, NULL) != 0;
    drmModeAtomicFree(req);
    drmDropMaster(fd);
    if (bad) {
        close(fd);
        return -1;
    }
    return fd;
}

typedef struct {
    double fps;
    u64 presented, dropped, repeated, presents;
    u32 refreshes;
    bool ok;
} Run;

/* Uncapped presents for at least 3 s, counted from a paced flip. */
static Run run(HSGraphics* gfx, void (*draw)(int), int min_frames) {
    Run r = {0};
    gfx->vsync = true;
    draw(0);
    hs_graphics_present(gfx);
    gfx->vsync = false;
    u64 rep0 = gfx->refreshes_repeated, pres0 = gfx->frames_presented, drop0 = gfx->frames_dropped;
    u32 seq0 = gfx->last_flip_sequence;
    double start = now_ms();
    int i;
    for (i = 0; i < min_frames || now_ms() - start < 3000; i++) {
        draw(i);
        hs_graphics_present(gfx);
    }
    r.fps = i * 1000.0 / (now_ms() - start);
    hs_graphics_flush_present(gfx);
    r.presents = (u64)i;
    r.repeated = gfx->refreshes_repeated - rep0;
    r.refreshes = gfx->last_flip_sequence - seq0;
    r.presented = gfx->frames_presented - pres0;
    r.dropped = gfx->frames_dropped - drop0;
    r.ok = r.refreshes > 120 && r.presented + r.dropped == r.presents && r.repeated * 100 <= r.refreshes;
    return r;
}

static GLuint g_flat, g_heavy;
static GLint g_c, g_n, g_t, g_heavy_n;
static u32 g_w, g_h;
static bool g_pixels_ok = true;
static const float k_left[] = {-1, -1, 0, -1, -1, 1, 0, -1, 0, 1, -1, 1};
static const float k_right[] = {0, -1, 1, -1, 0, 1, 1, -1, 1, 1, 0, 1};
static const float k_full[] = {-1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1};

/* Left half red, right half blue (alternating shade). */
static void draw_flat(int i) {
    glUseProgram(g_flat);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glUniform4f(g_c, 1, 0, 0, 1);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, k_left);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glUniform4f(g_c, 0, 0, (float)(i & 1) * 0.5f + 0.5f, 1);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, k_right);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    if (i == 1 || i == 301) {
        unsigned char px[2][4];
        glReadPixels((GLint)g_w / 4, (GLint)g_h / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px[0]);
        glReadPixels((GLint)(g_w * 3 / 4), (GLint)g_h / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px[1]);
        g_pixels_ok = g_pixels_ok && px[0][0] == 255 && px[0][2] == 0 && px[1][0] == 0 && px[1][2] >= 127;
    }
}

static void draw_heavy(int i) {
    glUseProgram(g_heavy);
    glUniform1i(g_n, g_heavy_n);
    glUniform1f(g_t, (float)i * 0.01f);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, k_full);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

static const char* tf(bool b) { return b ? "true" : "false"; }

int main(int argc, char** argv) {
    u32 w = argc > 1 ? (u32)atoi(argv[1]) : 400, h = argc > 2 ? (u32)atoi(argv[2]) : 240;
    static HSGraphics gfx;
    g_w = w;
    g_h = h;

    bool refuses = hs_graphics_init_scaled(&gfx, 8192, 8192) != 0 && hs_graphics_init_scaled(&gfx, 0, 0) != 0;

    /* Native: learn the CRTC and mode, then a sabotaged native takeover. */
    if (hs_graphics_init(&gfx) != 0) {
        printf("{\"init\":false}\n");
        return 1;
    }
    uint32_t crtc = gfx.display_crtc_id, connector = gfx.display_connector_id;
    u32 mode_w = gfx.display_mode.hdisplay, mode_h = gfx.display_mode.vdisplay;
    hs_graphics_finish(&gfx);
    int sab = sabotage(crtc, connector, false);
    PlaneState native = {0};
    bool native_ok = false;
    if (sab >= 0 && hs_graphics_init(&gfx) == 0) {
        hs_graphics_present(&gfx);
        native = plane_state(gfx.drm_fd, crtc);
        native_ok = !gfx.scaled_scanout && gfx.crtc_active && plane_ok(&native, mode_w, mode_h, mode_w, mode_h);
        if (sab >= 0) close(sab);
        hs_graphics_finish(&gfx);
    } else if (sab >= 0) {
        close(sab);
    }
    /* The panel's own size through init_scaled is the plain path. */
    bool native_plain = hs_graphics_init_scaled(&gfx, mode_w, mode_h) == 0 && !gfx.scaled_scanout;
    if (gfx.initialized) hs_graphics_finish(&gfx);

    /* Scaled takeover from a switched-off CRTC. */
    sab = sabotage(crtc, connector, true);
    PlaneState blank = {0};
    bool blank_ok = false;
    if (sab >= 0 && hs_graphics_init_scaled(&gfx, w, h) == 0) {
        hs_graphics_present(&gfx);
        blank = plane_state(gfx.drm_fd, crtc);
        uint64_t active = 0;
        hs_graphics_prop(gfx.drm_fd, crtc, DRM_MODE_OBJECT_CRTC, "ACTIVE", &active);
        /* Paced flips must land: an inactive CRTC has no vblanks. */
        gfx.vsync = true;
        u64 shown0 = gfx.frames_presented;
        for (int i = 0; i < 10; i++) hs_graphics_present(&gfx);
        gfx.vsync = false;
        blank_ok = gfx.crtc_active && active == 1 && gfx.present_errors == 0 && gfx.frames_presented - shown0 == 10 &&
                   plane_ok(&blank, w, h, mode_w, mode_h);
        close(sab);
        hs_graphics_finish(&gfx);
    } else if (sab >= 0) {
        close(sab);
    }

    /* Scaled takeover over a sabotaged plane, then the runs. */
    sab = sabotage(crtc, connector, false);
    if (sab < 0 || hs_graphics_init_scaled(&gfx, w, h) != 0) {
        printf("{\"init_scaled\":false,\"sabotage\":%s}\n", tf(sab >= 0));
        return 1;
    }
    g_flat = program(FS);
    g_heavy = program(FS_HEAVY);
    g_c = glGetUniformLocation(g_flat, "c");
    g_n = glGetUniformLocation(g_heavy, "n");
    g_t = glGetUniformLocation(g_heavy, "t");
    glEnableVertexAttribArray(0);
    glViewport(0, 0, (GLsizei)w, (GLsizei)h);
    draw_flat(0);
    hs_graphics_present(&gfx);
    PlaneState took = plane_state(gfx.drm_fd, crtc);
    bool took_ok = gfx.scaled_scanout == (w != mode_w || h != mode_h) && gfx.crtc_active &&
                   plane_ok(&took, w, h, mode_w, mode_h);
    close(sab);

    Run cheap = run(&gfx, draw_flat, 600);

    /* Heavy frames: ~0.6 refresh of GPU time at the render size. */
    double refresh_ms = 1000.0 / (gfx.display_mode.vrefresh ? gfx.display_mode.vrefresh : 60);
    double want = 0.6 * refresh_ms, heavy_ms = 0;
    g_heavy_n = 8;
    for (int round = 0; round < 12; round++) {
        glFinish();
        double t0 = now_ms();
        for (int i = 0; i < 4; i++) draw_heavy(i);
        glFinish();
        heavy_ms = (now_ms() - t0) / 4;
        if (heavy_ms > want * 0.9 && heavy_ms < want * 1.1) break;
        int next = (int)(g_heavy_n * want / (heavy_ms > 0.01 ? heavy_ms : 0.01));
        g_heavy_n = next < 1 ? 1 : next > 4096 ? 4096 : next;
    }
    gfx.last_present_ns = gfx.present_interval_ns = 0; /* calibration's gap is not a frame */
    for (int i = 0; i < 32; i++) {
        draw_heavy(i);
        hs_graphics_present(&gfx);
    }
    Run heavy = run(&gfx, draw_heavy, 0);
    bool heavy_held = gfx.present_interval_ns * 1000 >= (u64)(refresh_ms * 1e6) * gfx.hold_permille;

    u32 errors = gfx.present_errors;
    const char* why = gfx.present_error;
    PlaneState end = plane_state(gfx.drm_fd, crtc);
    bool end_ok = plane_ok(&end, w, h, mode_w, mode_h);
    hs_graphics_finish(&gfx);

    /* The console's turn. */
    usleep(100000);
    int fd = open_card();
    PlaneState console = fd >= 0 ? plane_state(fd, crtc) : (PlaneState){0};
    if (fd >= 0) close(fd);
    bool console_ok = plane_ok(&console, mode_w, mode_h, mode_w, mode_h);

    bool ok = refuses && native_ok && native_plain && blank_ok && took_ok && cheap.ok && heavy.ok && heavy_held &&
              end_ok && errors == 0 && g_pixels_ok && console_ok;
    printf("{\"render\":[%u,%u],\"mode\":[%u,%u],\"refuses\":%s,\"native_ok\":%s,\"native_alpha\":%llu,"
           "\"native_filter\":%llu,\"native_others\":%d,\"native_plain\":%s,\"blank_ok\":%s,\"took_ok\":%s,"
           "\"took_src\":[%llu,%llu,%llu,%llu],\"took_dst\":[%llu,%llu,%llu,%llu],\"took_alpha\":%llu,"
           "\"took_blend\":%llu,\"took_filter\":%llu,\"took_others\":%d,"
           "\"cheap\":{\"fps\":%.0f,\"refreshes\":%u,\"repeated\":%llu,\"ok\":%s},"
           "\"heavy\":{\"gpu_ms\":%.2f,\"fps\":%.1f,\"refreshes\":%u,\"repeated\":%llu,\"held\":%s,\"ok\":%s},"
           "\"end_ok\":%s,\"errors\":%u,\"error\":\"%s\",\"pixels_ok\":%s,\"console_ok\":%s,"
           "\"console_alpha\":%llu,\"console_filter\":%llu,\"passed\":%s}\n",
           w, h, mode_w, mode_h, tf(refuses), tf(native_ok), (unsigned long long)native.alpha,
           (unsigned long long)native.filter, native.others, tf(native_plain), tf(blank_ok), tf(took_ok),
           (unsigned long long)took.src_x >> 16, (unsigned long long)took.src_y >> 16,
           (unsigned long long)took.src_w >> 16, (unsigned long long)took.src_h >> 16,
           (unsigned long long)took.dst_x, (unsigned long long)took.dst_y, (unsigned long long)took.dst_w,
           (unsigned long long)took.dst_h, (unsigned long long)took.alpha, (unsigned long long)took.blend,
           (unsigned long long)took.filter, took.others, cheap.fps, cheap.refreshes,
           (unsigned long long)cheap.repeated, tf(cheap.ok), heavy_ms, heavy.fps, heavy.refreshes,
           (unsigned long long)heavy.repeated, tf(heavy_held), tf(heavy.ok), tf(end_ok), errors, why ? why : "",
           tf(g_pixels_ok), tf(console_ok), (unsigned long long)console.alpha, (unsigned long long)console.filter,
           tf(ok));
    return ok ? 0 : 1;
}
