/* P0-3 (docs/audit/HARDWARE_HANDOFF_P0.md): what the vc4 HVS costs when it
 * composes and scales at scanout, measured with atomic commits of CPU-filled
 * dumb buffers (no GPU, no Mesa), so every pixel's content is known.
 *
 * The primary plane shows a src_w x src_h buffer scaled to the full panel.
 * `overlays` ARGB planes (premultiplied, half alpha) of ov_w x ov_h sit above
 * it; with move=1 they move every frame (hardware sprites). Each frame flips
 * the primary between two buffers with one nonblocking atomic commit and
 * waits for its page-flip event.
 *
 * Usage: sudo tests/p0_3_hvs_probe <src_w> <src_h> [overlays=0] [ov_w=64]
 *            [ov_h=64] [move=0] [frames=600] [filter=0 default|1 nearest]
 *            [ov_scale_pct=100] [legacy=0]
 * legacy=1: after the atomic modeset, flip with drmModePageFlip (the call
 * hs_graphics_present uses), to see whether legacy flips keep the scaling.
 * Output: one JSON line. Needs DRM master (no other process presenting);
 * the console gets the display back when the probe exits. The plane
 * properties the probe changes (filter, blend mode) are put back to their
 * defaults first: the console's restore does not reset them. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <drm_fourcc.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#define MAXF 4096
#define MAXOV 16

typedef struct { uint32_t handle, pitch, fb, w, h; uint64_t size; uint32_t* map; } Buf;

static int fd;
static uint32_t crtc_id, conn_id, crtc_index;
static drmModeModeInfo mode;

static double now_ns(clockid_t c) {
    struct timespec t;
    clock_gettime(c, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

static uint32_t prop_id(uint32_t obj, uint32_t type, const char* name) {
    drmModeObjectProperties* p = drmModeObjectGetProperties(fd, obj, type);
    uint32_t id = 0;
    for (unsigned i = 0; p && i < p->count_props && !id; i++) {
        drmModePropertyRes* r = drmModeGetProperty(fd, p->props[i]);
        if (r && !strcmp(r->name, name)) id = r->prop_id;
        drmModeFreeProperty(r);
    }
    drmModeFreeObjectProperties(p);
    if (!id) fprintf(stderr, "no property %s on %u\n", name, obj);
    return id;
}

static int add(drmModeAtomicReq* req, uint32_t obj, uint32_t type, const char* name, uint64_t v) {
    uint32_t id = prop_id(obj, type, name);
    return id ? drmModeAtomicAddProperty(req, obj, id, v) : -1;
}

static int make_buf(Buf* b, uint32_t w, uint32_t h, uint32_t format) {
    struct drm_mode_create_dumb c = {.width = w, .height = h, .bpp = 32};
    if (drmIoctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &c)) return -1;
    b->handle = c.handle; b->pitch = c.pitch; b->size = c.size; b->w = w; b->h = h;
    uint32_t handles[4] = {c.handle}, pitches[4] = {c.pitch}, offsets[4] = {0};
    if (drmModeAddFB2(fd, w, h, format, handles, pitches, offsets, &b->fb, 0)) return -1;
    struct drm_mode_map_dumb m = {.handle = c.handle};
    if (drmIoctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &m)) return -1;
    b->map = mmap(0, c.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, m.offset);
    return b->map == MAP_FAILED ? -1 : 0;
}

static unsigned long long underruns(void) {
    char path[128];
    unsigned long long v = 0;
    for (int i = 0; i < 4; i++) {
        snprintf(path, sizeof path, "/sys/kernel/debug/dri/%d/hvs_underrun", i);
        FILE* f = fopen(path, "r");
        if (!f) continue;
        if (fscanf(f, "%llu", &v) != 1) v = 0;
        fclose(f);
        return v;
    }
    return ~0ull;
}

static double ev_ns;
static unsigned ev_seq;
static int ev_got;
static void on_flip(int f, unsigned seq, unsigned s, unsigned us, unsigned crtc, void* data) {
    (void)f; (void)crtc; (void)data;
    ev_ns = s * 1e9 + us * 1e3;
    ev_seq = seq;
    ev_got = 1;
}

static int cmpd(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: see source\n"); return 2; }
    uint32_t sw = atoi(argv[1]), sh = atoi(argv[2]);
    int nov = argc > 3 ? atoi(argv[3]) : 0;
    uint32_t ow = argc > 4 ? atoi(argv[4]) : 64, oh = argc > 5 ? atoi(argv[5]) : 64;
    int move = argc > 6 ? atoi(argv[6]) : 0, frames = argc > 7 ? atoi(argv[7]) : 600;
    int filter = argc > 8 ? atoi(argv[8]) : 0, ov_pct = argc > 9 ? atoi(argv[9]) : 100;
    int legacy = argc > 10 ? atoi(argv[10]) : 0;
    if (nov > MAXOV) nov = MAXOV;
    if (frames > MAXF) frames = MAXF;

    for (int i = 0; i < 8; i++) {
        char p[32];
        snprintf(p, sizeof p, "/dev/dri/card%d", i);
        fd = open(p, O_RDWR | O_CLOEXEC);
        if (fd < 0) continue;
        drmVersion* v = drmGetVersion(fd);
        int ok = v && !strcmp(v->name, "vc4");
        drmFreeVersion(v);
        if (ok) break;
        close(fd);
        fd = -1;
    }
    if (fd < 0 || drmSetClientCap(fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1) ||
        drmSetClientCap(fd, DRM_CLIENT_CAP_ATOMIC, 1) || drmSetMaster(fd)) {
        printf("{\"error\":\"vc4 atomic master unavailable: %s\"}\n", strerror(errno));
        return 1;
    }
    drmModeRes* res = drmModeGetResources(fd);
    for (int i = 0; i < res->count_connectors && !conn_id; i++) {
        drmModeConnector* c = drmModeGetConnector(fd, res->connectors[i]);
        if (c && c->connection == DRM_MODE_CONNECTED && c->count_modes && c->connector_type == DRM_MODE_CONNECTOR_DSI) {
            conn_id = c->connector_id;
            mode = c->modes[0];
            drmModeEncoder* e = drmModeGetEncoder(fd, c->encoder_id ? c->encoder_id : c->encoders[0]);
            for (int k = 0; e && k < res->count_crtcs; k++)
                if (e->crtc_id ? res->crtcs[k] == e->crtc_id : (e->possible_crtcs & (1u << k))) { crtc_id = res->crtcs[k]; crtc_index = k; break; }
            drmModeFreeEncoder(e);
        }
        drmModeFreeConnector(c);
    }
    if (!crtc_id) { printf("{\"error\":\"no DSI crtc\"}\n"); return 1; }

    uint32_t primary = 0, ovl[MAXOV];
    int novl = 0;
    drmModePlaneRes* pr = drmModeGetPlaneResources(fd);
    for (unsigned i = 0; i < pr->count_planes; i++) {
        drmModePlane* p = drmModeGetPlane(fd, pr->planes[i]);
        if (p && (p->possible_crtcs & (1u << crtc_index))) {
            drmModeObjectProperties* op = drmModeObjectGetProperties(fd, p->plane_id, DRM_MODE_OBJECT_PLANE);
            uint64_t type = 99;
            for (unsigned k = 0; k < op->count_props; k++) {
                drmModePropertyRes* r = drmModeGetProperty(fd, op->props[k]);
                if (r && !strcmp(r->name, "type")) type = op->prop_values[k];
                drmModeFreeProperty(r);
            }
            drmModeFreeObjectProperties(op);
            if (type == DRM_PLANE_TYPE_PRIMARY && !primary) primary = p->plane_id;
            else if (type == DRM_PLANE_TYPE_OVERLAY && novl < MAXOV) ovl[novl++] = p->plane_id;
        }
        drmModeFreePlane(p);
    }
    if (nov > novl) nov = novl;

    /* Two primary buffers: a gradient and a frame-specific bar, so a flip
     * changes content. Overlays: premultiplied half-alpha squares. */
    Buf pb[2], ob[MAXOV];
    for (int k = 0; k < 2; k++) {
        if (make_buf(&pb[k], sw, sh, DRM_FORMAT_XRGB8888)) { printf("{\"error\":\"primary buffer\"}\n"); return 1; }
        for (uint32_t y = 0; y < sh; y++)
            for (uint32_t x = 0; x < sw; x++)
                pb[k].map[y * (pb[k].pitch / 4) + x] = (x * 255 / sw) << 16 | (y * 255 / sh) << 8 | (k ? 0xff : 0x40);
    }
    for (int k = 0; k < nov; k++) {
        if (make_buf(&ob[k], ow, oh, DRM_FORMAT_ARGB8888)) { printf("{\"error\":\"overlay buffer\"}\n"); return 1; }
        uint32_t c = 0x80u << 24 | ((k * 37) & 0x7f) << 16 | ((k * 91) & 0x7f) << 8 | 0x40;
        for (uint32_t i = 0; i < oh * (ob[k].pitch / 4); i++) ob[k].map[i] = c;
    }

    uint32_t blob = 0;
    drmModeCreatePropertyBlob(fd, &mode, sizeof mode, &blob);
    uint32_t dw = mode.hdisplay, dh = mode.vdisplay;
    uint32_t odw = ow * ov_pct / 100, odh = oh * ov_pct / 100;
    double refresh_ns = 1e9 / (mode.vrefresh ? mode.vrefresh : 60);

    static double cpu[MAXF], lat[MAXF];
    unsigned seq_prev = 0, missed = 0;
    int fails = 0, first_err = 0, done = 0;
    unsigned long long u0 = underruns();
    double t_start = 0;
    drmEventContext evc = {.version = 3, .page_flip_handler2 = on_flip};

    for (int f = -1; f < frames; f++) {
        drmModeAtomicReq* req = drmModeAtomicAlloc();
        int r = 0;
        if (f < 0) {
            r |= add(req, crtc_id, DRM_MODE_OBJECT_CRTC, "MODE_ID", blob) < 0;
            r |= add(req, crtc_id, DRM_MODE_OBJECT_CRTC, "ACTIVE", 1) < 0;
            r |= add(req, conn_id, DRM_MODE_OBJECT_CONNECTOR, "CRTC_ID", crtc_id) < 0;
            r |= add(req, primary, DRM_MODE_OBJECT_PLANE, "CRTC_ID", crtc_id) < 0;
            r |= add(req, primary, DRM_MODE_OBJECT_PLANE, "SRC_X", 0) < 0;
            r |= add(req, primary, DRM_MODE_OBJECT_PLANE, "SRC_Y", 0) < 0;
            r |= add(req, primary, DRM_MODE_OBJECT_PLANE, "SRC_W", (uint64_t)sw << 16) < 0;
            r |= add(req, primary, DRM_MODE_OBJECT_PLANE, "SRC_H", (uint64_t)sh << 16) < 0;
            r |= add(req, primary, DRM_MODE_OBJECT_PLANE, "CRTC_X", 0) < 0;
            r |= add(req, primary, DRM_MODE_OBJECT_PLANE, "CRTC_Y", 0) < 0;
            r |= add(req, primary, DRM_MODE_OBJECT_PLANE, "CRTC_W", dw) < 0;
            r |= add(req, primary, DRM_MODE_OBJECT_PLANE, "CRTC_H", dh) < 0;
            r |= add(req, primary, DRM_MODE_OBJECT_PLANE, "SCALING_FILTER", filter) < 0;
            for (int k = 0; k < nov; k++) {
                r |= add(req, ovl[k], DRM_MODE_OBJECT_PLANE, "CRTC_ID", crtc_id) < 0;
                r |= add(req, ovl[k], DRM_MODE_OBJECT_PLANE, "FB_ID", ob[k].fb) < 0;
                r |= add(req, ovl[k], DRM_MODE_OBJECT_PLANE, "SRC_X", 0) < 0;
                r |= add(req, ovl[k], DRM_MODE_OBJECT_PLANE, "SRC_Y", 0) < 0;
                r |= add(req, ovl[k], DRM_MODE_OBJECT_PLANE, "SRC_W", (uint64_t)ow << 16) < 0;
                r |= add(req, ovl[k], DRM_MODE_OBJECT_PLANE, "SRC_H", (uint64_t)oh << 16) < 0;
                r |= add(req, ovl[k], DRM_MODE_OBJECT_PLANE, "CRTC_W", odw) < 0;
                r |= add(req, ovl[k], DRM_MODE_OBJECT_PLANE, "CRTC_H", odh) < 0;
                r |= add(req, ovl[k], DRM_MODE_OBJECT_PLANE, "zpos", 1 + k) < 0;
                r |= add(req, ovl[k], DRM_MODE_OBJECT_PLANE, "pixel blend mode", 0) < 0; /* premultiplied */
                r |= add(req, ovl[k], DRM_MODE_OBJECT_PLANE, "SCALING_FILTER", filter) < 0;
            }
        }
        r |= add(req, primary, DRM_MODE_OBJECT_PLANE, "FB_ID", pb[(f + 2) & 1].fb) < 0;
        for (int k = 0; k < nov && (move || f < 0); k++) {
            int span_x = (int)dw - (int)odw, span_y = (int)dh - (int)odh;
            int x = span_x > 0 ? (int)((k * 97 + (f + 1) * 3) % span_x) : 0;
            int y = span_y > 0 ? (int)((k * 53 + (f + 1) * 2) % span_y) : 0;
            r |= add(req, ovl[k], DRM_MODE_OBJECT_PLANE, "CRTC_X", x) < 0;
            r |= add(req, ovl[k], DRM_MODE_OBJECT_PLANE, "CRTC_Y", y) < 0;
        }
        if (r) { printf("{\"error\":\"property lookup\"}\n"); return 1; }
        uint32_t flags = f < 0 ? DRM_MODE_ATOMIC_ALLOW_MODESET : DRM_MODE_ATOMIC_NONBLOCK | DRM_MODE_PAGE_FLIP_EVENT;
        double c0 = now_ns(CLOCK_THREAD_CPUTIME_ID), w0 = now_ns(CLOCK_MONOTONIC);
        int rc = legacy && f >= 0 ? drmModePageFlip(fd, crtc_id, pb[(f + 2) & 1].fb, DRM_MODE_PAGE_FLIP_EVENT, NULL)
                                  : drmModeAtomicCommit(fd, req, flags, NULL);
        double c1 = now_ns(CLOCK_THREAD_CPUTIME_ID);
        drmModeAtomicFree(req);
        if (rc) {
            if (!first_err) first_err = errno;
            fails++;
            if (f < 0) break;
            continue;
        }
        if (f < 0) { t_start = now_ns(CLOCK_MONOTONIC); continue; }
        ev_got = 0;
        while (!ev_got) {
            struct pollfd p = {.fd = fd, .events = POLLIN};
            if (poll(&p, 1, 200) <= 0) break;
            drmHandleEvent(fd, &evc);
        }
        if (!ev_got) { fails++; continue; }
        cpu[done] = (c1 - c0) / 1e3;
        lat[done] = (ev_ns - w0) / 1e6;
        if (seq_prev && ev_seq > seq_prev + 1) missed += ev_seq - seq_prev - 1;
        seq_prev = ev_seq;
        done++;
    }
    double wall = now_ns(CLOCK_MONOTONIC) - t_start;
    unsigned long long u1 = underruns();
    qsort(cpu, done, sizeof(double), cmpd);
    qsort(lat, done, sizeof(double), cmpd);
    printf("{\"src\":[%u,%u],\"dst\":[%u,%u],\"overlays\":%d,\"ov\":[%u,%u],\"ov_dst\":[%u,%u],\"move\":%d,\"legacy\":%d,"
           "\"filter\":%d,\"frames\":%d,\"done\":%d,\"fails\":%d,\"first_errno\":%d,\"fps\":%.2f,"
           "\"refreshes_missed\":%u,\"underruns\":%lld,\"commit_cpu_us\":{\"p50\":%.1f,\"p90\":%.1f,\"max\":%.1f},"
           "\"commit_to_event_ms\":{\"p10\":%.2f,\"p50\":%.2f,\"p90\":%.2f,\"max\":%.2f},\"refresh_ms\":%.3f}\n",
           sw, sh, dw, dh, nov, ow, oh, odw, odh, move, legacy, filter, frames, done, fails, first_err,
           done ? done * 1e9 / wall : 0, missed, (u0 == ~0ull || u1 == ~0ull) ? -1 : (long long)(u1 - u0),
           done ? cpu[done / 2] : 0, done ? cpu[done * 9 / 10] : 0, done ? cpu[done - 1] : 0,
           done ? lat[done / 10] : 0, done ? lat[done / 2] : 0, done ? lat[done * 9 / 10] : 0,
           done ? lat[done - 1] : 0, refresh_ns / 1e6);
    drmModeAtomicReq* req = drmModeAtomicAlloc();
    int r = add(req, primary, DRM_MODE_OBJECT_PLANE, "SCALING_FILTER", 0) < 0;
    for (int k = 0; k < nov; k++) {
        r |= add(req, ovl[k], DRM_MODE_OBJECT_PLANE, "SCALING_FILTER", 0) < 0;
        r |= add(req, ovl[k], DRM_MODE_OBJECT_PLANE, "pixel blend mode", 0) < 0;
    }
    if (r || drmModeAtomicCommit(fd, req, 0, NULL)) fprintf(stderr, "plane property restore failed\n");
    drmModeAtomicFree(req);
    drmDropMaster(fd);
    close(fd);
    return fails ? 1 : 0;
}
