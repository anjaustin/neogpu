/* P0-3 correctness: what the vc4 HVS actually composes, captured through the
 * writeback connector (TXP) and compared pixel by pixel with CPU references.
 *
 * Every capture is one atomic commit on the writeback CRTC with
 * WRITEBACK_FB_ID = an 800x480 XRGB8888 dumb buffer (pre-filled with a
 * sentinel) and WRITEBACK_OUT_FENCE_PTR; the out-fence is waited with poll()
 * and the buffer is read back. Source buffers are CPU-filled dumb buffers, so
 * every input pixel is known. No GPU, no Mesa.
 *
 * Cases (one JSON line per capture, then a summary line):
 *  1   primary 800x480 unscaled                      exact (0 mismatches)
 *  2   primary 400x240 -> 800x480, Nearest           within 2 LSB of 2x replication
 *      (measured: the HVS Nearest kernel leaks ~1 LSB per neighbour; the
 *      sample mapping itself is plain replication)
 *      (+ a coordinate-coded source that reveals the sample mapping, and
 *      1px line images that show any leakage from neighbouring pixels)
 *  3   primary 400x240 -> 800x480, Default           stats vs nearest, bilinear,
 *      Mitchell-Netravali(1/3,1/3), Catmull-Rom (pixel-centre aligned), and
 *      Mitchell / BC(0.25,0.35) with u = o*sw/dw - 0.44 (the B, C, p0 were
 *      least-squares fitted offline to the impulse profiles this probe prints);
 *      plus line images that measure the horizontal / vertical impulse response
 *  4   200x120 -> 800x480 (4x), Nearest and Default  as 2 and 3
 *  5   primary + premultiplied ARGB overlay, alpha 0x80, unscaled, offset
 *      vs out = src + dst*(255-a)/255 (rounded)     tolerance <= 1 LSB
 *  5b  same with an alpha ramp 0..255 (informational)
 *  6   plane alpha 0x8000: Coverage (pixel alpha 255 and 0x80) and
 *      Pre-multiplied, vs candidate formulas (informational)
 *  7   4 opaque overlays, overlapping, zpos not in plane order -> exact
 * Exit status 1 if case 1, case 2 (hard pattern, nearest) or case 5 fail.
 *
 * Usage: sudo tests/p0_3_writeback_check [ppm_dir]
 *   ppm_dir default tests/evidence-p0-3-20261116/writeback ("-" = none)
 * Needs DRM master; leaves the writeback CRTC disabled on exit. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <drm_fourcc.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#define W 800
#define H 480
#define NOV 4

typedef struct { uint32_t handle, pitch, fb, w, h; uint64_t size; uint32_t* map; } Buf;
typedef struct { int w, h; uint32_t* p; } Img; /* packed 0xAARRGGBB, stride w */

typedef struct {
    uint32_t plane;
    Buf* fb;
    int sw, sh;          /* source rect (origin 0,0) */
    int cx, cy, cw, ch;  /* CRTC rect */
    int filter;          /* SCALING_FILTER */
    int blend;           /* pixel blend mode; -1 = leave */
    int alpha;           /* plane alpha 0..65535 */
    int zpos;            /* -1 = immutable (primary) */
} PlaneCfg;

static int fd;
static uint32_t crtc_id, wb_conn, crtc_index, primary, ovl[NOV];
static drmModeModeInfo mode;
static int modeset_done;
static const char* ppm_dir = "tests/evidence-p0-3-20261116/writeback";
static Buf outb;

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

static uint64_t prop_val(uint32_t obj, uint32_t type, const char* name, int* found) {
    drmModeObjectProperties* p = drmModeObjectGetProperties(fd, obj, type);
    uint64_t v = 0;
    *found = 0;
    for (unsigned i = 0; p && i < p->count_props && !*found; i++) {
        drmModePropertyRes* r = drmModeGetProperty(fd, p->props[i]);
        if (r && !strcmp(r->name, name)) { v = p->prop_values[i]; *found = 1; }
        drmModeFreeProperty(r);
    }
    drmModeFreeObjectProperties(p);
    return v;
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

static void free_buf(Buf* b) {
    munmap(b->map, b->size);
    drmModeRmFB(fd, b->fb);
    struct drm_mode_destroy_dumb d = {.handle = b->handle};
    drmIoctl(fd, DRM_IOCTL_MODE_DESTROY_DUMB, &d);
}

static Img img_new(int w, int h) { Img i = {w, h, calloc((size_t)w * h, 4)}; return i; }

static void upload(Buf* b, const Img* im) {
    for (int y = 0; y < im->h; y++) memcpy(b->map + (size_t)y * (b->pitch / 4), im->p + (size_t)y * im->w, (size_t)im->w * 4);
}

static int src_buf(Buf* b, const Img* im, uint32_t fmt) {
    if (make_buf(b, im->w, im->h, fmt)) return -1;
    upload(b, im);
    return 0;
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

/* ---- patterns ---- */
static uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16;
    return x;
}
static uint32_t rgb(int r, int g, int b) { return 0xff000000u | (uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b; }
static int ch(uint32_t p, int c) { return (p >> (16 - 8 * c)) & 0xff; } /* c: 0=R 1=G 2=B */
static int cha(uint32_t p) { return p >> 24; }

/* hard: R = 1px checkerboard 0/255, G = horizontal ramp, B = hash noise */
static Img pat_hard(int w, int h) {
    Img im = img_new(w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            im.p[y * w + x] = rgb(((x ^ y) & 1) ? 255 : 0, x * 255 / (w - 1), hash32(y * 65536 + x) & 0xff);
    return im;
}
/* coordinate-coded: R = x&255, G = y&255, B = 0x80 | (x>>8)<<2 | (y>>8) */
static Img pat_coord(int w, int h) {
    Img im = img_new(w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            im.p[y * w + x] = rgb(x & 255, y & 255, 0x80 | ((x >> 8) & 3) << 2 | ((y >> 8) & 3));
    return im;
}
/* lines every 16 px at offset 8, hi on lo (192 on 64 shows ringing either way) */
static Img pat_lines(int w, int h, int vertical, int lo, int hi) {
    Img im = img_new(w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            int on = vertical ? (x % 16 == 8) : (y % 16 == 8);
            im.p[y * w + x] = on ? rgb(hi, hi, hi) : rgb(lo, lo, lo);
        }
    return im;
}

/* ---- references ---- (own rounding helpers: the build line has no -lm) */
static double fabsd(double v) { return v < 0 ? -v : v; }
static int ifloor(double v) { int i = (int)v; return (double)i > v ? i - 1 : i; }
static long rnd(double v) { return v >= 0 ? (long)(v + 0.5) : -(long)(-v + 0.5); }
static double k_tent(double t) { t = fabsd(t); return t < 1 ? 1 - t : 0; }
static double k_bc(double t, double B, double C) {
    t = fabsd(t);
    if (t < 1) return ((12 - 9 * B - 6 * C) * t * t * t + (-18 + 12 * B + 6 * C) * t * t + (6 - 2 * B)) / 6;
    if (t < 2) return ((-B - 6 * C) * t * t * t + (6 * B + 30 * C) * t * t + (-12 * B - 48 * C) * t + (8 * B + 24 * C)) / 6;
    return 0;
}
static double k_mitchell(double t) { return k_bc(t, 1.0 / 3, 1.0 / 3); }
static double k_catrom(double t) { return k_bc(t, 0, 0.5); }
/* B,C and p0 least-squares fitted (offline) to the 2x and 4x impulse profiles of the first run */
static double k_fit(double t) { return k_bc(t, 0.25, 0.35); }
#define FIT_P0 (-0.44)

static int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }

/* separable resample, edge clamp, no intermediate rounding. centre=1: pixel-centre
 * aligned u = (o+.5)*sw/dw - .5; centre=0: u = o*sw/dw + p0 (source pixel centres at integers) */
static Img resample(const Img* s, int dw, int dh, double (*k)(double), int centre, double p0) {
    Img o = img_new(dw, dh);
    double* tmp = malloc(sizeof(double) * 3 * dw * s->h);
    for (int y = 0; y < s->h; y++)
        for (int x = 0; x < dw; x++) {
            double u = centre ? (x + 0.5) * s->w / dw - 0.5 : (double)x * s->w / dw + p0, acc[3] = {0, 0, 0}, ws = 0;
            int i0 = ifloor(u);
            for (int i = i0 - 2; i <= i0 + 3; i++) {
                double wt = k(u - i);
                if (wt == 0) continue;
                uint32_t p = s->p[y * s->w + clampi(i, 0, s->w - 1)];
                for (int c = 0; c < 3; c++) acc[c] += wt * ch(p, c);
                ws += wt;
            }
            for (int c = 0; c < 3; c++) tmp[(y * dw + x) * 3 + c] = acc[c] / ws;
        }
    for (int y = 0; y < dh; y++)
        for (int x = 0; x < dw; x++) {
            double v = centre ? (y + 0.5) * s->h / dh - 0.5 : (double)y * s->h / dh + p0, acc[3] = {0, 0, 0}, ws = 0;
            int i0 = ifloor(v);
            for (int i = i0 - 2; i <= i0 + 3; i++) {
                double wt = k(v - i);
                if (wt == 0) continue;
                int yy = clampi(i, 0, s->h - 1);
                for (int c = 0; c < 3; c++) acc[c] += wt * tmp[(yy * dw + x) * 3 + c];
                ws += wt;
            }
            int r[3];
            for (int c = 0; c < 3; c++) r[c] = clampi((int)rnd(acc[c] / ws), 0, 255);
            o.p[y * dw + x] = rgb(r[0], r[1], r[2]);
        }
    free(tmp);
    return o;
}

static Img nearest(const Img* s, int dw, int dh) {
    Img o = img_new(dw, dh);
    for (int y = 0; y < dh; y++)
        for (int x = 0; x < dw; x++) {
            int sx = (2 * x + 1) * s->w / (2 * dw), sy = (2 * y + 1) * s->h / (2 * dh);
            o.p[y * dw + x] = s->p[sy * s->w + sx];
        }
    return o;
}

/* ---- capture ---- */
static void write_ppm(const char* name, const uint32_t* px) {
    if (!strcmp(ppm_dir, "-")) return;
    char path[512];
    snprintf(path, sizeof path, "%s/%s.ppm", ppm_dir, name);
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ppm %s: %s\n", path, strerror(errno)); return; }
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    unsigned char* row = malloc(W * 3);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++)
            for (int c = 0; c < 3; c++) row[x * 3 + c] = ch(px[y * W + x], c);
        fwrite(row, 1, W * 3, f);
    }
    free(row);
    fclose(f);
}

#define SENTINEL 0xA5C35A3Cu
static int last_errno;
static double last_wait_ms;

/* One atomic commit + writeback; on success fills out (W*H) and returns 0. */
static int capture(const PlaneCfg* pl, int n, const char* name, uint32_t* out) {
    for (uint32_t y = 0; y < H; y++)
        for (uint32_t x = 0; x < W; x++) outb.map[y * (outb.pitch / 4) + x] = SENTINEL;
    drmModeAtomicReq* req = drmModeAtomicAlloc();
    int r = 0;
    if (!modeset_done) {
        uint32_t blob = 0;
        drmModeCreatePropertyBlob(fd, &mode, sizeof mode, &blob);
        r |= add(req, crtc_id, DRM_MODE_OBJECT_CRTC, "MODE_ID", blob) < 0;
        r |= add(req, crtc_id, DRM_MODE_OBJECT_CRTC, "ACTIVE", 1) < 0;
        r |= add(req, wb_conn, DRM_MODE_OBJECT_CONNECTOR, "CRTC_ID", crtc_id) < 0;
    }
    uint32_t all[1 + NOV] = {primary, ovl[0], ovl[1], ovl[2], ovl[3]};
    for (int a = 0; a < 1 + NOV; a++) {
        const PlaneCfg* c = NULL;
        for (int k = 0; k < n; k++)
            if (pl[k].plane == all[a]) c = &pl[k];
        uint32_t id = all[a];
        if (!c) {
            r |= add(req, id, DRM_MODE_OBJECT_PLANE, "FB_ID", 0) < 0;
            r |= add(req, id, DRM_MODE_OBJECT_PLANE, "CRTC_ID", 0) < 0;
            continue;
        }
        r |= add(req, id, DRM_MODE_OBJECT_PLANE, "FB_ID", c->fb->fb) < 0;
        r |= add(req, id, DRM_MODE_OBJECT_PLANE, "CRTC_ID", crtc_id) < 0;
        r |= add(req, id, DRM_MODE_OBJECT_PLANE, "SRC_X", 0) < 0;
        r |= add(req, id, DRM_MODE_OBJECT_PLANE, "SRC_Y", 0) < 0;
        r |= add(req, id, DRM_MODE_OBJECT_PLANE, "SRC_W", (uint64_t)c->sw << 16) < 0;
        r |= add(req, id, DRM_MODE_OBJECT_PLANE, "SRC_H", (uint64_t)c->sh << 16) < 0;
        r |= add(req, id, DRM_MODE_OBJECT_PLANE, "CRTC_X", (uint64_t)(int64_t)c->cx) < 0;
        r |= add(req, id, DRM_MODE_OBJECT_PLANE, "CRTC_Y", (uint64_t)(int64_t)c->cy) < 0;
        r |= add(req, id, DRM_MODE_OBJECT_PLANE, "CRTC_W", c->cw) < 0;
        r |= add(req, id, DRM_MODE_OBJECT_PLANE, "CRTC_H", c->ch) < 0;
        r |= add(req, id, DRM_MODE_OBJECT_PLANE, "SCALING_FILTER", c->filter) < 0;
        r |= add(req, id, DRM_MODE_OBJECT_PLANE, "alpha", c->alpha) < 0;
        if (c->blend >= 0) r |= add(req, id, DRM_MODE_OBJECT_PLANE, "pixel blend mode", c->blend) < 0;
        if (c->zpos >= 0) r |= add(req, id, DRM_MODE_OBJECT_PLANE, "zpos", c->zpos) < 0;
    }
    int32_t fence = -1;
    r |= add(req, wb_conn, DRM_MODE_OBJECT_CONNECTOR, "WRITEBACK_FB_ID", outb.fb) < 0;
    r |= add(req, wb_conn, DRM_MODE_OBJECT_CONNECTOR, "WRITEBACK_OUT_FENCE_PTR", (uint64_t)(uintptr_t)&fence) < 0;
    if (r) { drmModeAtomicFree(req); last_errno = -1; return -1; }
    int rc = drmModeAtomicCommit(fd, req, DRM_MODE_ATOMIC_ALLOW_MODESET, NULL);
    last_errno = rc ? errno : 0;
    drmModeAtomicFree(req);
    if (rc) return -1;
    modeset_done = 1;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (fence < 0) { last_errno = -2; return -1; }
    struct pollfd p = {.fd = fence, .events = POLLIN};
    int pr = poll(&p, 1, 2000);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    close(fence);
    last_wait_ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    if (pr <= 0) { last_errno = -3; return -1; }
    for (int y = 0; y < H; y++) memcpy(out + (size_t)y * W, outb.map + (size_t)y * (outb.pitch / 4), W * 4);
    write_ppm(name, out);
    return 0;
}

/* ---- comparison ---- */
typedef struct {
    long mism, over_tol, within1, within2, sentinel;
    int max[3], imax[3];
    double mean[3];
    int nfirst, first[4][2];
    uint32_t fexp[4], fact[4];
} Stats;

/* interior: pixels at least `border` from the edge (for imax) */
static Stats compare(const uint32_t* out, const uint32_t* ref, int tol, int border) {
    Stats s;
    memset(&s, 0, sizeof s);
    double sum[3] = {0, 0, 0};
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            uint32_t a = out[y * W + x], e = ref[y * W + x];
            if (a == SENTINEL) s.sentinel++;
            int m = 0;
            for (int c = 0; c < 3; c++) {
                int d = abs(ch(a, c) - ch(e, c));
                if (d > m) m = d;
                if (d > s.max[c]) s.max[c] = d;
                if (x >= border && y >= border && x < W - border && y < H - border && d > s.imax[c]) s.imax[c] = d;
                sum[c] += d;
            }
            if (m) s.mism++;
            if (m <= 1) s.within1++;
            if (m <= 2) s.within2++;
            if (m > tol) {
                if (s.nfirst < 4) {
                    s.first[s.nfirst][0] = x; s.first[s.nfirst][1] = y;
                    s.fexp[s.nfirst] = e & 0xffffff; s.fact[s.nfirst] = a & 0xffffff;
                    s.nfirst++;
                }
                s.over_tol++;
            }
        }
    for (int c = 0; c < 3; c++) s.mean[c] = sum[c] / (W * H);
    return s;
}

static void print_stats(const char* ref, const Stats* s, int tol) {
    printf("{\"ref\":\"%s\",\"tol\":%d,\"mismatch\":%ld,\"over_tol\":%ld,\"within1\":%.5f,\"within2\":%.5f,"
           "\"max\":[%d,%d,%d],\"max_interior8\":[%d,%d,%d],\"mean\":[%.3f,%.3f,%.3f],\"sentinel_px\":%ld,\"first\":[",
           ref, tol, s->mism, s->over_tol, (double)s->within1 / (W * H), (double)s->within2 / (W * H),
           s->max[0], s->max[1], s->max[2], s->imax[0], s->imax[1], s->imax[2], s->mean[0], s->mean[1], s->mean[2],
           s->sentinel);
    for (int i = 0; i < s->nfirst; i++)
        printf("%s{\"x\":%d,\"y\":%d,\"exp\":\"%06x\",\"act\":\"%06x\"}", i ? "," : "", s->first[i][0], s->first[i][1],
               s->fexp[i], s->fact[i]);
    printf("]}");
}

static void begin(const char* id, const char* desc) { printf("{\"case\":\"%s\",\"desc\":\"%s\"", id, desc); }
static int cap_or_report(const PlaneCfg* pl, int n, const char* name, uint32_t* out) {
    if (capture(pl, n, name, out)) {
        printf(",\"error\":\"capture failed\",\"errno\":%d,\"strerror\":\"%s\"}\n", last_errno,
               last_errno > 0 ? strerror(last_errno) : last_errno == -2 ? "no out fence" : last_errno == -3 ? "fence timeout" : "property");
        fflush(stdout);
        return -1;
    }
    printf(",\"fence_wait_ms\":%.2f", last_wait_ms);
    return 0;
}

static PlaneCfg prim(Buf* b, int filter) {
    PlaneCfg c = {primary, b, (int)b->w, (int)b->h, 0, 0, W, H, filter, -1, 65535, -1};
    return c;
}
static PlaneCfg over(int k, Buf* b, int x, int y, int blend, int alpha, int zpos) {
    PlaneCfg c = {ovl[k], b, (int)b->w, (int)b->h, x, y, (int)b->w, (int)b->h, 0, blend, alpha, zpos};
    return c;
}

/* Nearest-filter case: coord image -> mapping; hard image -> exact vs replication */
static int nearest_case(const char* id, int sw, int sh, uint32_t* out) {
    int fail = 0;
    Img co = pat_coord(sw, sh), hd = pat_hard(sw, sh);
    Buf bc, bh;
    if (src_buf(&bc, &co, DRM_FORMAT_XRGB8888) || src_buf(&bh, &hd, DRM_FORMAT_XRGB8888)) { printf("{\"error\":\"buffer\"}\n"); exit(1); }
    char name[64];

    snprintf(name, sizeof name, "case%s_coord_nearest", id);
    begin(name, "coordinate-coded source, Nearest; decodes which source pixel each output pixel sampled");
    PlaneCfg p = prim(&bc, 1);
    if (!cap_or_report(&p, 1, name, out)) {
        int mapx[W], mapy[H];
        for (int x = 0; x < W; x++) { uint32_t v = out[(H / 2) * W + x]; mapx[x] = ch(v, 0) | ((ch(v, 2) >> 2) & 3) << 8; }
        for (int y = 0; y < H; y++) { uint32_t v = out[y * W + W / 2]; mapy[y] = ch(v, 1) | (ch(v, 2) & 3) << 8; }
        long incons = 0, badtag = 0;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                uint32_t v = out[y * W + x];
                if ((ch(v, 2) & 0xf0) != 0x80) badtag++;
                if ((ch(v, 0) | ((ch(v, 2) >> 2) & 3) << 8) != mapx[x] || (ch(v, 1) | (ch(v, 2) & 3) << 8) != mapy[y]) incons++;
            }
        int dcx = 0, dcy = 0, dfx = 0, dfy = 0, fcx = -1, fcy = -1;
        for (int x = 0; x < W; x++) {
            if (mapx[x] != (2 * x + 1) * sw / (2 * W)) { dcx++; if (fcx < 0) fcx = x; }
            if (mapx[x] != x * sw / W) dfx++;
        }
        for (int y = 0; y < H; y++) {
            if (mapy[y] != (2 * y + 1) * sh / (2 * H)) { dcy++; if (fcy < 0) fcy = y; }
            if (mapy[y] != y * sh / H) dfy++;
        }
        printf(",\"pixels_not_separable_xy\":%ld,\"pixels_bad_tag\":%ld,"
               "\"cols_differ_from_floor((x+.5)*s)\":%d,\"rows_differ_from_floor((y+.5)*s)\":%d,"
               "\"cols_differ_from_floor(x*s)\":%d,\"rows_differ_from_floor(y*s)\":%d,\"first_bad_col\":%d,\"first_bad_row\":%d,"
               "\"mapx_0_15\":[", incons, badtag, dcx, dcy, dfx, dfy, fcx, fcy);
        for (int x = 0; x < 16; x++) printf("%s%d", x ? "," : "", mapx[x]);
        printf("],\"mapx_last4\":[%d,%d,%d,%d],\"mapy_0_15\":[", mapx[W - 4], mapx[W - 3], mapx[W - 2], mapx[W - 1]);
        for (int y = 0; y < 16; y++) printf("%s%d", y ? "," : "", mapy[y]);
        printf("],\"mapy_last4\":[%d,%d,%d,%d]}\n", mapy[H - 4], mapy[H - 3], mapy[H - 2], mapy[H - 1]);
    }

    snprintf(name, sizeof name, "case%s_hard_nearest", id);
    begin(name, "checkerboard/ramp/noise source, Nearest, within 2 LSB of pixel replication");
    p = prim(&bh, 1);
    if (cap_or_report(&p, 1, name, out)) fail = 1;
    else {
        Img ref = nearest(&hd, W, H);
        Stats s = compare(out, ref.p, 2, 8);
        printf(",\"pass\":%s,\"vs\":[", s.over_tol ? "false" : "true");
        print_stats("replication", &s, 2);
        printf("]}\n");
        if (s.over_tol) fail = 1;
        free(ref.p);
    }
    /* leakage of the Nearest kernel: 1px vertical lines 255 on 0 and 0 on 255 */
    for (int inv = 0; inv < 2; inv++) {
        Img li = pat_lines(sw, sh, 1, inv ? 255 : 0, inv ? 0 : 255);
        Buf bl;
        if (src_buf(&bl, &li, DRM_FORMAT_XRGB8888)) { printf("{\"error\":\"buffer\"}\n"); exit(1); }
        int s = W / sw;
        snprintf(name, sizeof name, "case%s_vlines_nearest_%s", id, inv ? "0on255" : "255on0");
        begin(name, "1px vertical lines every 16 src cols, Nearest: horizontal profile across a line (exact replication = s copies of the line value, background elsewhere)");
        p = prim(&bl, 1);
        if (!cap_or_report(&p, 1, name, out)) {
            int l0 = 8 + 16 * (sw / 32), base = s * l0 - 3 * s, n = 8 * s;
            printf(",\"src_line_at\":%d,\"out_first_index\":%d,\"profile_R\":[", l0, base);
            for (int i = 0; i < n; i++) printf("%s%d", i ? "," : "", ch(out[(H / 2) * W + base + i], 0));
            Img ref = nearest(&li, W, H);
            Stats st = compare(out, ref.p, 0, 8);
            printf("],\"vs\":[");
            print_stats("replication", &st, 0);
            printf("]}\n");
            free(ref.p);
        }
        free_buf(&bl); free(li.p);
    }
    free_buf(&bc); free_buf(&bh); free(co.p); free(hd.p);
    fflush(stdout);
    return fail;
}

#define NREF 6
static const char* ref_names[NREF] = {"nearest", "bilinear", "mitchell_1/3_1/3", "catmull_rom",
                                      "mitchell_1/3_1/3_u=o*s-0.44", "bc_0.25_0.35_u=o*s-0.44(fit)"};
static Img make_ref(int i, const Img* src) {
    double (*ks[NREF])(double) = {NULL, k_tent, k_mitchell, k_catrom, k_mitchell, k_fit};
    return i == 0 ? nearest(src, W, H) : resample(src, W, H, ks[i], i < 4, FIT_P0);
}
static void vs_all_refs(const Img* src, const uint32_t* out) {
    printf(",\"vs\":[");
    for (int i = 0; i < NREF; i++) {
        Img ref = make_ref(i, src);
        Stats s = compare(out, ref.p, 2, 8);
        if (i) printf(",");
        print_stats(ref_names[i], &s, 2);
        free(ref.p);
    }
    printf("]");
}

/* Default-filter case: hard pattern vs refs; line images -> impulse response */
static void default_case(const char* id, int sw, int sh, uint32_t* out) {
    int s = W / sw;
    Img hd = pat_hard(sw, sh), lv = pat_lines(sw, sh, 1, 64, 192), lh = pat_lines(sw, sh, 0, 64, 192);
    Buf bh, bv, bl;
    if (src_buf(&bh, &hd, DRM_FORMAT_XRGB8888) || src_buf(&bv, &lv, DRM_FORMAT_XRGB8888) ||
        src_buf(&bl, &lh, DRM_FORMAT_XRGB8888)) { printf("{\"error\":\"buffer\"}\n"); exit(1); }
    char name[64];

    snprintf(name, sizeof name, "case%s_hard_default", id);
    begin(name, "checkerboard/ramp/noise source, Default filter; stats vs candidate references (tol 2 for over_tol)");
    PlaneCfg p = prim(&bh, 0);
    if (!cap_or_report(&p, 1, name, out)) { vs_all_refs(&hd, out); printf("}\n"); }

    for (int vert = 0; vert < 2; vert++) {
        Img* src = vert ? &lh : &lv;
        snprintf(name, sizeof name, "case%s_%s_default", id, vert ? "hlines" : "vlines");
        begin(name, vert ? "1px horizontal lines (192 on 64) every 16 src rows: vertical impulse response"
                         : "1px vertical lines (192 on 64) every 16 src cols: horizontal impulse response");
        p = prim(vert ? &bl : &bv, 0);
        if (cap_or_report(&p, 1, name, out)) continue;
        /* profile across the line at src position 8+16*k, output centre s*l0 */
        int l0 = 8 + 16 * ((vert ? sh : sw) / 32);
        int n = 8 * s, base = s * l0 - 3 * s;
        int prof[64];
        for (int i = 0; i < n; i++) {
            int o = base + i;
            prof[i] = vert ? ch(out[o * W + W / 2], 1) : ch(out[(H / 2) * W + o], 1);
        }
        /* is the response the same for every line and every row/col (all channels)? */
        long diff = 0, checked = 0;
        int extent = vert ? sh : sw;
        for (int l = 8 + 16; l + 16 < extent; l += 16)
            for (int i = 0; i < n; i++)
                for (int a = 0; a < (vert ? W : H); a++) {
                    int o = s * l - 3 * s + i;
                    uint32_t v = vert ? out[o * W + a] : out[a * W + o];
                    for (int c = 0; c < 3; c++) { checked++; if (ch(v, c) != prof[i]) diff++; }
                }
        printf(",\"src_line_at\":%d,\"out_first_index\":%d,\"profile_G\":[", l0, base);
        for (int i = 0; i < n; i++) printf("%s%d", i ? "," : "", prof[i]);
        printf("],\"weights_(v-64)/128\":[");
        for (int i = 0; i < n; i++) printf("%s%.4f", i ? "," : "", (prof[i] - 64) / 128.0);
        printf("],\"samples_differing_from_profile_over_interior_lines\":%ld,\"samples_checked\":%ld", diff, checked);
        printf(",\"ref_profiles\":{");
        for (int k = 0; k < NREF; k++) {
            Img ref = make_ref(k, src);
            printf("%s\"%s\":[", k ? "," : "", ref_names[k]);
            for (int i = 0; i < n; i++) {
                int o = base + i;
                printf("%s%d", i ? "," : "", vert ? ch(ref.p[o * W + W / 2], 1) : ch(ref.p[(H / 2) * W + o], 1));
            }
            printf("]");
            free(ref.p);
        }
        printf("}");
        vs_all_refs(src, out);
        printf("}\n");
    }
    free_buf(&bh); free_buf(&bv); free_buf(&bl); free(hd.p); free(lv.p); free(lh.p);
    fflush(stdout);
}

/* blend references; dst = primary image (unscaled), src overlay at (ox,oy) */
enum { BL_PREMUL_255, BL_PREMUL_256T, BL_PREMUL_255T, BL_COV_255, BL_COV_PA8, BL_PREMUL_PA, BL_PREMUL_PA8 };
static int blend1(int mode_, int s, int d, int a, int pa16) {
    double A = a / 255.0, P = pa16 / 65535.0, P8 = (pa16 >> 8) / 255.0;
    switch (mode_) {
    case BL_PREMUL_255: return clampi((int)rnd(s + d * (255 - a) / 255.0), 0, 255);
    case BL_PREMUL_256T: return clampi(s + ((d * (256 - a)) >> 8), 0, 255);
    case BL_PREMUL_255T: return clampi(s + d * (255 - a) / 255, 0, 255);
    case BL_COV_255: return clampi((int)rnd(s * A * P + d * (1 - A * P)), 0, 255);
    case BL_COV_PA8: return clampi((int)rnd(s * A * P8 + d * (1 - A * P8)), 0, 255);
    case BL_PREMUL_PA: return clampi((int)rnd(s * P + d * (1 - A * P)), 0, 255);
    case BL_PREMUL_PA8: return clampi((int)rnd(s * P8 + d * (1 - A * P8)), 0, 255);
    }
    return 0;
}
static Img blend_ref(const Img* dst, const Img* ov, int ox, int oy, int mode_, int pa16) {
    Img r = img_new(W, H);
    memcpy(r.p, dst->p, (size_t)W * H * 4);
    for (int y = 0; y < ov->h; y++)
        for (int x = 0; x < ov->w; x++) {
            int X = ox + x, Y = oy + y;
            if (X < 0 || Y < 0 || X >= W || Y >= H) continue;
            uint32_t s = ov->p[y * ov->w + x], d = r.p[Y * W + X];
            int o[3];
            for (int c = 0; c < 3; c++) o[c] = blend1(mode_, ch(s, c), ch(d, c), cha(s), pa16);
            r.p[Y * W + X] = rgb(o[0], o[1], o[2]);
        }
    return r;
}

static void blend_vs(const Img* bg, const Img* ov, int ox, int oy, int pa16, const uint32_t* out, const int* modes, int nm, int tol, Stats* first) {
    static const char* names[] = {"premul_over_/255_round", "premul_over_(256-a)>>8_trunc", "premul_over_/255_trunc",
                                  "coverage_a*pa16", "coverage_a*(pa16>>8)/255", "premul_with_plane_alpha_pa16",
                                  "premul_with_plane_alpha_(pa16>>8)/255"};
    printf(",\"vs\":[");
    for (int i = 0; i < nm; i++) {
        Img ref = blend_ref(bg, ov, ox, oy, modes[i], pa16);
        Stats s = compare(out, ref.p, tol, 0);
        if (i == 0 && first) *first = s;
        if (i) printf(",");
        print_stats(names[modes[i]], &s, tol);
        free(ref.p);
    }
    printf("]");
}

int main(int argc, char** argv) {
    if (argc > 1) ppm_dir = argv[1];
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
    if (fd < 0) { printf("{\"error\":\"no vc4 card\"}\n"); return 1; }
    if (drmSetClientCap(fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1) || drmSetClientCap(fd, DRM_CLIENT_CAP_ATOMIC, 1) ||
        drmSetClientCap(fd, DRM_CLIENT_CAP_WRITEBACK_CONNECTORS, 1)) {
        printf("{\"error\":\"client caps (universal/atomic/writeback): %s\"}\n", strerror(errno));
        return 1;
    }
    if (drmSetMaster(fd)) { printf("{\"error\":\"drmSetMaster: %s (another master holds vc4?)\"}\n", strerror(errno)); return 1; }

    drmModeRes* res = drmModeGetResources(fd);
    uint32_t enc_mask = 0;
    int have_mode = 0;
    for (int i = 0; i < res->count_connectors; i++) {
        drmModeConnector* c = drmModeGetConnector(fd, res->connectors[i]);
        if (!c) continue;
        if (c->connector_type == DRM_MODE_CONNECTOR_WRITEBACK && !wb_conn) {
            wb_conn = c->connector_id;
            for (int e = 0; e < c->count_encoders; e++) {
                drmModeEncoder* en = drmModeGetEncoder(fd, c->encoders[e]);
                if (en) enc_mask |= en->possible_crtcs;
                drmModeFreeEncoder(en);
            }
        }
        if (c->connector_type == DRM_MODE_CONNECTOR_DSI && c->count_modes && !have_mode) {
            for (int m = 0; m < c->count_modes; m++)
                if (c->modes[m].hdisplay == W && c->modes[m].vdisplay == H) { mode = c->modes[m]; have_mode = 1; break; }
        }
        drmModeFreeConnector(c);
    }
    if (!wb_conn) { printf("{\"error\":\"no writeback connector\"}\n"); return 1; }
    if (!have_mode) {
        printf("{\"error\":\"no 800x480 DSI mode to copy\"}\n");
        return 1;
    }
    for (int k = 0; k < res->count_crtcs; k++)
        if (enc_mask & (1u << k)) { crtc_id = res->crtcs[k]; crtc_index = k; break; }
    if (!crtc_id) { printf("{\"error\":\"writeback encoder has no possible crtc\"}\n"); return 1; }

    /* writeback pixel formats */
    int found;
    uint64_t fmt_blob = prop_val(wb_conn, DRM_MODE_OBJECT_CONNECTOR, "WRITEBACK_PIXEL_FORMATS", &found);
    char fmts[256] = "";
    int has_xr24 = 0;
    if (found && fmt_blob) {
        drmModePropertyBlobRes* b = drmModeGetPropertyBlob(fd, (uint32_t)fmt_blob);
        for (uint32_t i = 0; b && i < b->length / 4; i++) {
            uint32_t f = ((uint32_t*)b->data)[i];
            if (f == DRM_FORMAT_XRGB8888) has_xr24 = 1;
            size_t l = strlen(fmts);
            if (l + 8 < sizeof fmts) snprintf(fmts + l, sizeof fmts - l, "%s%.4s", l ? "," : "", (char*)&f);
        }
        drmModeFreePropertyBlob(b);
    }
    if (!has_xr24) { printf("{\"error\":\"writeback lacks XRGB8888\",\"formats\":\"%s\"}\n", fmts); return 1; }

    int novl = 0;
    drmModePlaneRes* pr = drmModeGetPlaneResources(fd);
    for (unsigned i = 0; i < pr->count_planes; i++) {
        drmModePlane* p = drmModeGetPlane(fd, pr->planes[i]);
        if (p && (p->possible_crtcs & (1u << crtc_index)) && !p->crtc_id) {
            uint64_t type = prop_val(p->plane_id, DRM_MODE_OBJECT_PLANE, "type", &found);
            if (type == DRM_PLANE_TYPE_PRIMARY && !primary) primary = p->plane_id;
            else if (type == DRM_PLANE_TYPE_OVERLAY && novl < NOV) ovl[novl++] = p->plane_id;
        }
        drmModeFreePlane(p);
    }
    if (!primary || novl < NOV) { printf("{\"error\":\"planes: primary %u, free overlays %d\"}\n", primary, novl); return 1; }

    if (strcmp(ppm_dir, "-")) mkdir(ppm_dir, 0755);
    if (make_buf(&outb, W, H, DRM_FORMAT_XRGB8888)) { printf("{\"error\":\"writeback buffer\"}\n"); return 1; }
    unsigned long long u0 = underruns();
    printf("{\"setup\":{\"writeback_connector\":%u,\"encoder_possible_crtcs\":\"0x%x\",\"crtc\":%u,\"crtc_index\":%u,"
           "\"mode\":\"%s\",\"mode_clock\":%u,\"primary\":%u,\"overlays\":[%u,%u,%u,%u],\"wb_formats\":\"%s\",\"out_pitch\":%u}}\n",
           wb_conn, enc_mask, crtc_id, crtc_index, mode.name, mode.clock, primary, ovl[0], ovl[1], ovl[2], ovl[3], fmts, outb.pitch);
    fflush(stdout);

    uint32_t* out = malloc((size_t)W * H * 4);
    int fail1 = 0, fail2 = 0, fail4 = 0, fail5 = 0;

    /* 1: unscaled */
    Img hard800 = pat_hard(W, H);
    Buf b800;
    if (src_buf(&b800, &hard800, DRM_FORMAT_XRGB8888)) { printf("{\"error\":\"buffer\"}\n"); return 1; }
    begin("1", "primary 800x480 XRGB8888 unscaled, exact");
    PlaneCfg p = prim(&b800, 0);
    if (cap_or_report(&p, 1, "case1_unscaled", out)) { fail1 = 1; if (last_errno) goto done; }
    else {
        Stats s = compare(out, hard800.p, 0, 0);
        fail1 = s.over_tol != 0;
        printf(",\"pass\":%s,\"vs\":[", fail1 ? "false" : "true");
        print_stats("source", &s, 0);
        printf("]}\n");
    }
    fflush(stdout);

    /* 2, 3: 2x */
    fail2 = nearest_case("2", 400, 240, out);
    default_case("3", 400, 240, out);
    /* 4: 4x */
    fail4 = nearest_case("4", 200, 120, out);
    default_case("4", 200, 120, out);

    /* 5: premultiplied half alpha overlay */
    {
        const int ox = 100, oy = 60;
        Img ov = img_new(256, 128);
        for (int y = 0; y < 128; y++)
            for (int x = 0; x < 256; x++) ov.p[y * 256 + x] = 0x80u << 24 | (uint32_t)(x / 2) << 16 | (uint32_t)y << 8 | 0x40;
        Buf bo;
        if (src_buf(&bo, &ov, DRM_FORMAT_ARGB8888)) { printf("{\"error\":\"buffer\"}\n"); return 1; }
        begin("5", "primary + ARGB8888 premultiplied overlay a=0x80 at (100,60), blend Pre-multiplied, plane alpha 0xffff; tol 1 LSB vs first ref");
        PlaneCfg pl[2] = {prim(&b800, 0), over(0, &bo, ox, oy, 0, 65535, 1)};
        if (cap_or_report(pl, 2, "case5_premul_half", out)) fail5 = 1;
        else {
            int modes[3] = {BL_PREMUL_255, BL_PREMUL_256T, BL_PREMUL_255T};
            Stats s;
            blend_vs(&hard800, &ov, ox, oy, 65535, out, modes, 3, 1, &s);
            fail5 = s.over_tol != 0;
            printf(",\"pass\":%s}\n", fail5 ? "false" : "true");
        }
        fflush(stdout);
        /* 5b: alpha ramp, colour = premultiplied of (2y, 255-2y, x) */
        Img ov2 = img_new(256, 128);
        for (int y = 0; y < 128; y++)
            for (int x = 0; x < 256; x++) {
                int a = x, r = (2 * y) * a / 255, g = (255 - 2 * y) * a / 255, b = x * a / 255;
                ov2.p[y * 256 + x] = (uint32_t)a << 24 | (uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b;
            }
        Buf bo2;
        if (src_buf(&bo2, &ov2, DRM_FORMAT_ARGB8888)) { printf("{\"error\":\"buffer\"}\n"); return 1; }
        begin("5b", "premultiplied overlay with alpha ramp a=x (0..255) at (100,60); informational");
        pl[1] = over(0, &bo2, ox, oy, 0, 65535, 1);
        if (!cap_or_report(pl, 2, "case5b_premul_ramp", out)) {
            int modes[3] = {BL_PREMUL_255, BL_PREMUL_256T, BL_PREMUL_255T};
            blend_vs(&hard800, &ov2, ox, oy, 65535, out, modes, 3, 1, NULL);
            printf("}\n");
        }
        fflush(stdout);

        /* 6: plane alpha 50% */
        Img ov3 = img_new(256, 128), ov4 = img_new(256, 128);
        for (int y = 0; y < 128; y++)
            for (int x = 0; x < 256; x++) {
                uint32_t c = (uint32_t)x << 16 | (uint32_t)(2 * y) << 8 | 0xc0;
                ov3.p[y * 256 + x] = 0xffu << 24 | c;
                ov4.p[y * 256 + x] = 0x80u << 24 | c;
            }
        Buf bo3, bo4;
        if (src_buf(&bo3, &ov3, DRM_FORMAT_ARGB8888) || src_buf(&bo4, &ov4, DRM_FORMAT_ARGB8888)) { printf("{\"error\":\"buffer\"}\n"); return 1; }
        int all_modes[7] = {BL_COV_255, BL_COV_PA8, BL_PREMUL_PA, BL_PREMUL_PA8, BL_PREMUL_255, BL_PREMUL_256T, BL_PREMUL_255T};
        struct { const char* id; const char* desc; Buf* b; Img* im; int blend; } c6[3] = {
            {"6a", "Coverage, pixel alpha 0xff (straight colour), plane alpha 0x8000", &bo3, &ov3, 1},
            {"6b", "Coverage, pixel alpha 0x80 (straight colour), plane alpha 0x8000", &bo4, &ov4, 1},
            {"6c", "Pre-multiplied overlay of case 5 (a=0x80), plane alpha 0x8000", &bo, &ov, 0},
        };
        for (int i = 0; i < 3; i++) {
            begin(c6[i].id, c6[i].desc);
            pl[1] = over(0, c6[i].b, ox, oy, c6[i].blend, 0x8000, 1);
            char name[64];
            snprintf(name, sizeof name, "case%s_plane_alpha", c6[i].id);
            if (!cap_or_report(pl, 2, name, out)) {
                blend_vs(&hard800, c6[i].im, ox, oy, 0x8000, out, all_modes, 7, 1, NULL);
                printf("}\n");
            }
            fflush(stdout);
        }
        free_buf(&bo); free_buf(&bo2); free_buf(&bo3); free_buf(&bo4);
        free(ov.p); free(ov2.p); free(ov3.p); free(ov4.p);
    }

    /* 7: z-order */
    {
        uint32_t col[NOV] = {rgb(255, 0, 0), rgb(0, 255, 0), rgb(0, 0, 255), rgb(255, 255, 255)};
        int zp[NOV] = {3, 1, 4, 2};
        Img ob[NOV];
        Buf bb[NOV];
        PlaneCfg pl[1 + NOV];
        pl[0] = prim(&b800, 0);
        for (int k = 0; k < NOV; k++) {
            ob[k] = img_new(200, 150);
            for (int i = 0; i < 200 * 150; i++) ob[k].p[i] = col[k];
            if (src_buf(&bb[k], &ob[k], DRM_FORMAT_ARGB8888)) { printf("{\"error\":\"buffer\"}\n"); return 1; }
            pl[1 + k] = over(k, &bb[k], 40 + 110 * k, 60 + 60 * k, 0, 65535, zp[k]);
        }
        begin("7", "4 opaque overlays (R,G,B,W) at (40+110k,60+60k) 200x150, zpos 3,1,4,2; exact vs painter by zpos");
        if (!cap_or_report(pl, 1 + NOV, "case7_zorder", out)) {
            Img ref = img_new(W, H);
            memcpy(ref.p, hard800.p, (size_t)W * H * 4);
            for (int z = 1; z <= NOV; z++)
                for (int k = 0; k < NOV; k++)
                    if (zp[k] == z)
                        for (int y = 0; y < 150; y++)
                            for (int x = 0; x < 200; x++) ref.p[(60 + 60 * k + y) * W + 40 + 110 * k + x] = col[k];
            Stats s = compare(out, ref.p, 0, 0);
            printf(",\"pass\":%s,\"vs\":[", s.over_tol ? "false" : "true");
            print_stats("painter_by_zpos", &s, 0);
            /* which colour wins inside each pairwise overlap (only planes k, k+1 cover it) */
            printf("],\"overlap_winners\":[");
            for (int k = 0; k + 1 < NOV; k++) {
                int x = 40 + 110 * (k + 1) + 45, y = 60 + 60 * (k + 1) + 45;
                printf("%s{\"planes\":[%d,%d],\"zpos\":[%d,%d],\"at\":[%d,%d],\"act\":\"%06x\",\"expect\":\"%06x\"}", k ? "," : "", k, k + 1,
                       zp[k], zp[k + 1], x, y, out[y * W + x] & 0xffffff, (zp[k] > zp[k + 1] ? col[k] : col[k + 1]) & 0xffffff);
            }
            printf("]}\n");
            free(ref.p);
        }
        for (int k = 0; k < NOV; k++) { free_buf(&bb[k]); free(ob[k].p); }
        fflush(stdout);
    }

done:;
    unsigned long long u1 = underruns();
    /* leave the writeback CRTC off */
    drmModeAtomicReq* req = drmModeAtomicAlloc();
    uint32_t all[1 + NOV] = {primary, ovl[0], ovl[1], ovl[2], ovl[3]};
    for (int a = 0; a < 1 + NOV; a++) {
        add(req, all[a], DRM_MODE_OBJECT_PLANE, "FB_ID", 0);
        add(req, all[a], DRM_MODE_OBJECT_PLANE, "CRTC_ID", 0);
    }
    add(req, wb_conn, DRM_MODE_OBJECT_CONNECTOR, "CRTC_ID", 0);
    add(req, crtc_id, DRM_MODE_OBJECT_CRTC, "ACTIVE", 0);
    add(req, crtc_id, DRM_MODE_OBJECT_CRTC, "MODE_ID", 0);
    int rc = drmModeAtomicCommit(fd, req, DRM_MODE_ATOMIC_ALLOW_MODESET, NULL);
    int cleanup_errno = rc ? errno : 0;
    drmModeAtomicFree(req);
    int fail = fail1 || fail2 || fail5;
    printf("{\"summary\":{\"case1_pass\":%s,\"case2_pass\":%s,\"case4_nearest_pass\":%s,\"case5_pass\":%s,"
           "\"case5_tolerance_lsb\":1,\"underruns\":%lld,\"cleanup_errno\":%d,\"exit\":%d}}\n",
           fail1 ? "false" : "true", fail2 ? "false" : "true", fail4 ? "false" : "true", fail5 ? "false" : "true",
           (u0 == ~0ull || u1 == ~0ull) ? -1 : (long long)(u1 - u0), cleanup_errno, fail);
    free_buf(&outb);
    drmDropMaster(fd);
    close(fd);
    return fail;
}
