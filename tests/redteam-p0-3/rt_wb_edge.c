/* Writeback (TXP, CRTC 0) capture of a 400x240 -> 800x480 Default-filter scale:
 * impulse columns/rows at the source edges and middle, and the effect of a
 * fractional SRC_X/SRC_Y. Usage: sudo rt_wb_edge [srcw=400] [srch=240]
 * Prints one JSON line per configuration. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <drm_fourcc.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#define W 800
#define H 480
typedef struct { uint32_t handle, pitch, fb, w, h; uint64_t size; uint32_t* map; } Buf;
static int fd; static uint32_t crtc_id, wb_conn, primary; static drmModeModeInfo mode; static int modeset_done; static Buf outb;
static uint32_t prop_id(uint32_t obj, uint32_t type, const char* name) {
    drmModeObjectProperties* p = drmModeObjectGetProperties(fd, obj, type); uint32_t id = 0;
    for (unsigned i = 0; p && i < p->count_props && !id; i++) { drmModePropertyRes* r = drmModeGetProperty(fd, p->props[i]); if (r && !strcmp(r->name, name)) id = r->prop_id; drmModeFreeProperty(r); }
    drmModeFreeObjectProperties(p); return id;
}
static uint64_t prop_val(uint32_t obj, uint32_t type, const char* name) {
    drmModeObjectProperties* p = drmModeObjectGetProperties(fd, obj, type); uint64_t v = ~0ull;
    for (unsigned i = 0; p && i < p->count_props; i++) { drmModePropertyRes* r = drmModeGetProperty(fd, p->props[i]); if (r && !strcmp(r->name, name)) v = p->prop_values[i]; drmModeFreeProperty(r); }
    drmModeFreeObjectProperties(p); return v;
}
static int add(drmModeAtomicReq* q, uint32_t o, uint32_t t, const char* n, uint64_t v) { uint32_t id = prop_id(o, t, n); return id ? drmModeAtomicAddProperty(q, o, id, v) : -1; }
static int make_buf(Buf* b, uint32_t w, uint32_t h) {
    struct drm_mode_create_dumb c = {.width = w, .height = h, .bpp = 32};
    if (drmIoctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &c)) return -1;
    b->handle = c.handle; b->pitch = c.pitch; b->size = c.size; b->w = w; b->h = h;
    uint32_t hs[4] = {c.handle}, ps[4] = {c.pitch}, os[4] = {0};
    if (drmModeAddFB2(fd, w, h, DRM_FORMAT_XRGB8888, hs, ps, os, &b->fb, 0)) return -1;
    struct drm_mode_map_dumb m = {.handle = c.handle};
    if (drmIoctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &m)) return -1;
    b->map = mmap(0, c.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, m.offset);
    return b->map == MAP_FAILED ? -1 : 0;
}
static int capture(Buf* src, uint64_t sx, uint64_t sy, uint64_t sw, uint64_t sh, int filter, uint8_t* out) {
    for (uint32_t y = 0; y < H; y++) for (uint32_t x = 0; x < W; x++) outb.map[y * (outb.pitch / 4) + x] = 0xA5C35A3C;
    drmModeAtomicReq* q = drmModeAtomicAlloc(); int r = 0;
    if (!modeset_done) { uint32_t blob; drmModeCreatePropertyBlob(fd, &mode, sizeof mode, &blob);
        r |= add(q, crtc_id, DRM_MODE_OBJECT_CRTC, "MODE_ID", blob) < 0; r |= add(q, crtc_id, DRM_MODE_OBJECT_CRTC, "ACTIVE", 1) < 0;
        r |= add(q, wb_conn, DRM_MODE_OBJECT_CONNECTOR, "CRTC_ID", crtc_id) < 0; }
    r |= add(q, primary, DRM_MODE_OBJECT_PLANE, "FB_ID", src->fb) < 0; r |= add(q, primary, DRM_MODE_OBJECT_PLANE, "CRTC_ID", crtc_id) < 0;
    r |= add(q, primary, DRM_MODE_OBJECT_PLANE, "SRC_X", sx) < 0; r |= add(q, primary, DRM_MODE_OBJECT_PLANE, "SRC_Y", sy) < 0;
    r |= add(q, primary, DRM_MODE_OBJECT_PLANE, "SRC_W", sw) < 0; r |= add(q, primary, DRM_MODE_OBJECT_PLANE, "SRC_H", sh) < 0;
    r |= add(q, primary, DRM_MODE_OBJECT_PLANE, "CRTC_X", 0) < 0; r |= add(q, primary, DRM_MODE_OBJECT_PLANE, "CRTC_Y", 0) < 0;
    r |= add(q, primary, DRM_MODE_OBJECT_PLANE, "CRTC_W", W) < 0; r |= add(q, primary, DRM_MODE_OBJECT_PLANE, "CRTC_H", H) < 0;
    r |= add(q, primary, DRM_MODE_OBJECT_PLANE, "SCALING_FILTER", filter) < 0;
    r |= add(q, primary, DRM_MODE_OBJECT_PLANE, "alpha", 0xffff) < 0;
    int32_t fence = -1;
    r |= add(q, wb_conn, DRM_MODE_OBJECT_CONNECTOR, "WRITEBACK_FB_ID", outb.fb) < 0;
    r |= add(q, wb_conn, DRM_MODE_OBJECT_CONNECTOR, "WRITEBACK_OUT_FENCE_PTR", (uint64_t)(uintptr_t)&fence) < 0;
    if (r) { drmModeAtomicFree(q); return -1000; }
    int rc = drmModeAtomicCommit(fd, q, DRM_MODE_ATOMIC_ALLOW_MODESET, NULL); int e = errno; drmModeAtomicFree(q);
    if (rc) return -e;
    modeset_done = 1;
    struct pollfd p = {.fd = fence, .events = POLLIN}; int pr = poll(&p, 1, 2000); close(fence);
    if (pr <= 0) return -2000;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) out[y * W + x] = (outb.map[y * (outb.pitch / 4) + x] >> 8) & 0xff; /* G */
    return 0;
}
int main(int argc, char** argv) {
    int SW = argc > 1 ? atoi(argv[1]) : 400, SH = argc > 2 ? atoi(argv[2]) : 240;
    for (int i = 0; i < 8; i++) { char p[32]; snprintf(p, sizeof p, "/dev/dri/card%d", i); fd = open(p, O_RDWR | O_CLOEXEC); if (fd < 0) continue;
        drmVersion* v = drmGetVersion(fd); int ok = v && !strcmp(v->name, "vc4"); drmFreeVersion(v); if (ok) break; close(fd); fd = -1; }
    if (fd < 0 || drmSetClientCap(fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1) || drmSetClientCap(fd, DRM_CLIENT_CAP_ATOMIC, 1) ||
        drmSetClientCap(fd, DRM_CLIENT_CAP_WRITEBACK_CONNECTORS, 1) || drmSetMaster(fd)) { printf("{\"error\":\"setup\"}\n"); return 1; }
    drmModeRes* res = drmModeGetResources(fd); uint32_t mask = 0; int have = 0;
    for (int i = 0; i < res->count_connectors; i++) { drmModeConnector* c = drmModeGetConnector(fd, res->connectors[i]); if (!c) continue;
        if (c->connector_type == DRM_MODE_CONNECTOR_WRITEBACK && !wb_conn) { wb_conn = c->connector_id;
            for (int e = 0; e < c->count_encoders; e++) { drmModeEncoder* en = drmModeGetEncoder(fd, c->encoders[e]); if (en) mask |= en->possible_crtcs; drmModeFreeEncoder(en); } }
        if (c->connector_type == DRM_MODE_CONNECTOR_DSI && !have) for (int m = 0; m < c->count_modes; m++) if (c->modes[m].hdisplay == W && c->modes[m].vdisplay == H) { mode = c->modes[m]; have = 1; break; }
        drmModeFreeConnector(c); }
    int ci = -1; for (int k = 0; k < res->count_crtcs; k++) if (mask & (1u << k)) { crtc_id = res->crtcs[k]; ci = k; break; }
    drmModePlaneRes* pr = drmModeGetPlaneResources(fd);
    for (unsigned i = 0; pr && i < pr->count_planes && !primary; i++) { drmModePlane* p = drmModeGetPlane(fd, pr->planes[i]);
        if (p && ci >= 0 && (p->possible_crtcs & (1u << ci)) && !p->crtc_id && prop_val(p->plane_id, DRM_MODE_OBJECT_PLANE, "type") == DRM_PLANE_TYPE_PRIMARY) primary = p->plane_id;
        drmModeFreePlane(p); }
    if (!wb_conn || !have || !primary || make_buf(&outb, W, H)) { printf("{\"error\":\"resources\"}\n"); return 1; }
    Buf src; if (make_buf(&src, SW, SH)) return 1;
    /* G channel: impulses at columns 0, SW/2, SW-1 and rows 0, SH/2, SH-1 on black (exclusive: columns on rows 1..SH-2 band). */
    for (int y = 0; y < SH; y++) for (int x = 0; x < SW; x++) {
        int on = x == 0 || x == SW / 2 || x == SW - 1 || y == 0 || y == SH / 2 || y == SH - 1;
        src.map[y * (src.pitch / 4) + x] = on ? 0xffffffff : 0xff000000; }
    uint8_t* out = malloc(W * H);
    struct { const char* name; uint64_t sx, sy, sw, sh; int filter; } cfg[] = {
        {"default_src0", 0, 0, (uint64_t)SW << 16, (uint64_t)SH << 16, 0},
        {"nearest_src0", 0, 0, (uint64_t)SW << 16, (uint64_t)SH << 16, 1},
        {"default_srcx0.1875", 0x3000, 0x3000, ((uint64_t)SW << 16) - 0x3000, ((uint64_t)SH << 16) - 0x3000, 0},
        {"default_srcx0.25", 0x4000, 0x4000, ((uint64_t)SW << 16) - 0x4000, ((uint64_t)SH << 16) - 0x4000, 0},
        {"default_srcx0.5", 0x8000, 0x8000, ((uint64_t)SW << 16) - 0x8000, ((uint64_t)SH << 16) - 0x8000, 0},
        {"default_srcx0.1875_fullw", 0x3000, 0x3000, (uint64_t)SW << 16, (uint64_t)SH << 16, 0},
    };
    int ry = SH / 4 * (H / SH) + (H / SH) / 2, rx = SW / 4 * (W / SW) + (W / SW) / 2; /* a row/column away from all impulses */
    for (unsigned k = 0; k < sizeof cfg / sizeof cfg[0]; k++) {
        int rc = capture(&src, cfg[k].sx, cfg[k].sy, cfg[k].sw, cfg[k].sh, cfg[k].filter, out);
        printf("{\"case\":\"%s\",\"src\":[%.4f,%.4f,%.4f,%.4f],\"rc\":%d", cfg[k].name, cfg[k].sx / 65536.0, cfg[k].sy / 65536.0, cfg[k].sw / 65536.0, cfg[k].sh / 65536.0, rc);
        if (rc == 0) {
            /* horizontal profile along row ry; vertical along column rx */
            const int xs[3][2] = {{0, 8}, {W / 2 - 4, W / 2 + 6}, {W - 8, W}};
            const char* nm[3] = {"left", "mid", "right"};
            for (int a = 0; a < 2; a++) {
                int N = a ? H : W;
                const int ys[3][2] = {{0, 8}, {N / 2 - 4, N / 2 + 6}, {N - 8, N}};
                for (int s = 0; s < 3; s++) {
                    const int* r = a ? ys[s] : xs[s];
                    double sum = 0, m1 = 0;
                    printf(",\"%s_%s\":{\"x0\":%d,\"v\":[", a ? "v" : "h", nm[s], r[0]);
                    for (int i = r[0]; i < r[1]; i++) { int v = a ? out[i * W + rx] : out[ry * W + i]; printf("%s%d", i > r[0] ? "," : "", v); sum += v; m1 += v * (double)i; }
                    double ideal = s == 0 ? 0.5 : s == 1 ? (N / 2) + 0.5 : N - 1.5; /* centre of the 2-px replicated impulse (2x) */
                    printf("],\"sum\":%.0f,\"centroid\":%.3f,\"ideal_2x\":%.1f}", sum, sum ? m1 / sum : -1, ideal);
                }
            }
        }
        printf("}\n");
        fflush(stdout);
    }
    /* leave the writeback CRTC off */
    drmModeAtomicReq* q = drmModeAtomicAlloc();
    add(q, primary, DRM_MODE_OBJECT_PLANE, "FB_ID", 0); add(q, primary, DRM_MODE_OBJECT_PLANE, "CRTC_ID", 0);
    add(q, primary, DRM_MODE_OBJECT_PLANE, "SCALING_FILTER", 0);
    add(q, wb_conn, DRM_MODE_OBJECT_CONNECTOR, "CRTC_ID", 0); add(q, crtc_id, DRM_MODE_OBJECT_CRTC, "ACTIVE", 0); add(q, crtc_id, DRM_MODE_OBJECT_CRTC, "MODE_ID", 0);
    int rc = drmModeAtomicCommit(fd, q, DRM_MODE_ATOMIC_ALLOW_MODESET, NULL);
    printf("{\"teardown_rc\":%d}\n", rc);
    return 0;
}
