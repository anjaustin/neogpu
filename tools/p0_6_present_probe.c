/* p0_6_present_probe: frames rendered by QPU compute alone, shown on the
 * panel through KMS. No Mesa, EGL or GBM in the process (libdrm for the
 * modeset calls only). (P0-6, docs/audit/HARDWARE_HANDOFF_P0.md.)
 *
 * Two vc4 dumb buffers are imported into v3d (tools/p0_6_scanout_probe showed
 * that direction works). Each frame one compute job fills the back buffer and
 * the buffer is page-flipped at the next refresh. The image is a test
 * pattern, not a scene: pixel i of frame f is (i + 7 * f) * 0x00010203 (low
 * 32 bits), so every pixel of a checked frame can be compared on the CPU.
 *
 * The display is taken only while this runs: it becomes DRM master, saves the
 * CRTC, turns off planes another client left on it and puts the primary
 * plane's properties back to their defaults (as hs_graphics does), and on
 * every exit path, SIGINT and SIGTERM included, restores the CRTC and drops
 * master. If something else is DRM master it reports that and exits.
 * Each buffer has 16 guard rows above and below the visible frame, marked
 * once and checked with the frame: a write outside the frame is counted.
 * The framebuffer object covers the visible rows only. Checked frames are
 * compared after their flip, in pairs, so both buffers are checked. At the
 * end the last framebuffer shown is composed once more by the display
 * controller into a capture buffer (the writeback connector) and compared:
 * that checks the framebuffer object, not only the memory behind it.
 * Usage: p0_6_present_probe [seconds=5] [batches=8] [check_every=30] [fault=0]
 *   fault 1: the job is given the wrong frame number (every checked pixel
 *            must then be reported wrong: shows the check can fail)
 *   fault 2: raises SIGINT in itself after 1 s (the restore path)
 *   fault 3: one loop pass too many per batch (guard words must be reported
 *            overwritten)
 *   fault 4: wrong frame number for frames 2..11 only (the memory check sees
 *            it, the capture of the last frame cannot)
 *   fault 5: the framebuffer object starts at the guard rows (only the
 *            scanout capture sees it: the memory is correct)
 *        (inside the GPU lock). Output: one JSON line. Exit 1 on any failure.
 */
#define _GNU_SOURCE
#include <drm/v3d_drm.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm_fourcc.h>

#include "hs_qpu.h"
#include "hs_v3d.h"

#define R(n) HS_QPU_R(n)
#define RF(n) HS_QPU_RF(n)
#define X HS_QPU_NOREG
#define I(v) hs_qpu_imm_int(v)
enum { MULT = 0x00010203, WAIT_EVERY = 4, GUARD_ROWS = 16, GUARD = 0xA5C3F00D };

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
static void emit(hs_qpu_prog* p, hs_qpu_alu x) { hs_qpu_emit(p, &x); }
static void nop(hs_qpu_prog* p) { emit(p, hs_qpu_alu_nop()); }
static void op(hs_qpu_prog* p, int o, hs_qpu_reg d, hs_qpu_reg a, hs_qpu_reg b) { emit(p, hs_qpu_a(o, d, a, b)); }

/* uniforms: base address, loop count, pixels per batch, 7 * frame, MULT, 16 * MULT.
 * Batch b fills its share: rf5 = (b * pixels + lane + 7f) * MULT, then
 * += 16 * MULT per store. */
static int build(hs_qpu_prog* p) {
    hs_qpu_prog_rules(p, 0);
    op(p, HS_QPU_A_BAND, RF(9), RF(0), I(15)); /* the hardware's batch number */
    for (int i = 0; i < 4; i++) emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDUNIFRF, RF(i)));
    emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDUNIFRF, RF(6)));
    emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDUNIFRF, RF(7)));
    op(p, HS_QPU_A_EIDX, R(1), X, X);
    op(p, HS_QPU_A_SHL, R(0), R(1), I(2));
    emit(p, hs_qpu_m(hs_qpu_alu_nop(), HS_QPU_M_UMUL24, R(2), RF(9), RF(2))); /* r2 = b * pixels */
    op(p, HS_QPU_A_ADD, RF(0), RF(0), R(0));
    op(p, HS_QPU_A_SHL, R(3), R(2), I(2));
    op(p, HS_QPU_A_ADD, RF(0), RF(0), R(3)); /* address of this lane's first pixel */
    op(p, HS_QPU_A_ADD, R(2), R(2), R(1));
    op(p, HS_QPU_A_ADD, R(2), R(2), RF(3));
    emit(p, hs_qpu_m(hs_qpu_a(HS_QPU_A_MOV, R(3), I(1), X), HS_QPU_M_UMUL24, RF(5), R(2), RF(6)));
    op(p, HS_QPU_A_SHL, RF(4), R(3), I(6)); /* 64 */
    int l = hs_qpu_label_new(p);
    hs_qpu_label_bind(p, l);
    for (int i = 0; i < (int)WAIT_EVERY; i++) {
        emit(p, hs_qpu_m(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUD, RF(5), X), HS_QPU_M_ADD, RF(5), RF(5), RF(7)));
        emit(p, hs_qpu_m(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(0), X), HS_QPU_M_ADD, RF(0), RF(0), RF(4)));
    }
    op(p, HS_QPU_A_TMUWT, X, X, X);
    hs_qpu_alu dec = hs_qpu_a(HS_QPU_A_SUB, RF(1), RF(1), I(1));
    dec.af = HS_QPU_PF_PUSHZ;
    emit(p, dec);
    hs_qpu_branch(p, HS_QPU_B_NA0, l, 0, 0);
    nop(p), nop(p), nop(p);
    hs_qpu_emit_end(p);
    return hs_qpu_prog_finish(p);
}

typedef struct {
    uint32_t handle, fb, v3d_handle, gpu, pitch; /* gpu: address of the first visible pixel */
    uint64_t size;
    uint32_t* all; /* the whole buffer: guard rows, the visible frame, guard rows */
    uint32_t* map; /* the visible frame */
} buf;

static int flip_pending;
static uint64_t flip_ns;
static volatile sig_atomic_t stop_requested;
static void on_signal(int sig) {
    (void)sig;
    stop_requested = 1;
}
static void on_flip(int fd, unsigned seq, unsigned sec, unsigned usec, void* data) {
    (void)fd, (void)seq, (void)data;
    flip_pending = 0;
    flip_ns = (uint64_t)sec * 1000000000ull + (uint64_t)usec * 1000ull;
}
static int cmpd(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}
static uint32_t prop(int fd, uint32_t obj, const char* name, uint64_t* value) {
    drmModeObjectProperties* props = drmModeObjectGetProperties(fd, obj, DRM_MODE_OBJECT_PLANE);
    uint32_t id = 0;
    for (uint32_t i = 0; props && i < props->count_props && !id; i++) {
        drmModePropertyRes* pr = drmModeGetProperty(fd, props->props[i]);
        if (pr && !strcmp(pr->name, name)) id = pr->prop_id, *value = props->prop_values[i];
        drmModeFreeProperty(pr);
    }
    drmModeFreeObjectProperties(props);
    return id;
}
/* As hs_graphics_reset_plane: a previous client's overlays stay on the CRTC
 * across SetCrtc, and the primary plane keeps its alpha, blend mode, filter
 * and rotation (measured in P0-3). Returns the number of changes made. */
static int reset_planes(int fd, uint32_t crtc_id, int crtc_index) {
    static const struct { const char* name; uint64_t value; } defaults[] = {
        {"alpha", 0xffff}, {"pixel blend mode", 0}, {"SCALING_FILTER", 0}, {"rotation", 1}};
    int changes = 0;
    if (drmSetClientCap(fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1)) return 0;
    drmModePlaneRes* planes = drmModeGetPlaneResources(fd);
    uint32_t primary = 0;
    for (uint32_t i = 0; planes && i < planes->count_planes && !primary; i++) {
        drmModePlane* pl = drmModeGetPlane(fd, planes->planes[i]);
        uint64_t type = 0;
        if (pl && (pl->possible_crtcs & (1u << crtc_index)) && prop(fd, pl->plane_id, "type", &type) && type == DRM_PLANE_TYPE_PRIMARY)
            primary = pl->plane_id;
        drmModeFreePlane(pl);
    }
    for (uint32_t i = 0; primary && planes && i < planes->count_planes; i++) {
        drmModePlane* pl = drmModeGetPlane(fd, planes->planes[i]);
        if (pl && pl->plane_id != primary && pl->crtc_id == crtc_id)
            changes += !drmModeSetPlane(fd, pl->plane_id, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        drmModeFreePlane(pl);
    }
    drmModeFreePlaneResources(planes);
    for (size_t i = 0; primary && i < sizeof defaults / sizeof defaults[0]; i++) {
        uint64_t v = 0;
        uint32_t id = prop(fd, primary, defaults[i].name, &v);
        if (id && v != defaults[i].value) changes += !drmModeObjectSetProperty(fd, primary, DRM_MODE_OBJECT_PLANE, id, defaults[i].value);
    }
    return changes;
}

static uint32_t obj_prop(int fd, uint32_t obj, uint32_t type, const char* name) {
    drmModeObjectProperties* props = drmModeObjectGetProperties(fd, obj, type);
    uint32_t id = 0;
    for (uint32_t i = 0; props && i < props->count_props && !id; i++) {
        drmModePropertyRes* pr = drmModeGetProperty(fd, props->props[i]);
        if (pr && !strcmp(pr->name, name)) id = pr->prop_id;
        drmModeFreeProperty(pr);
    }
    drmModeFreeObjectProperties(props);
    return id;
}
static int padd(drmModeAtomicReq* q, int fd, uint32_t obj, uint32_t type, const char* name, uint64_t v) {
    uint32_t id = obj_prop(fd, obj, type, name);
    return id ? drmModeAtomicAddProperty(q, obj, id, v) < 0 : 1;
}
/* What vc4 composes from framebuffer `fb`: the display controller's
 * writeback connector renders it into `out` (w * h words), as
 * tests/p0_3_writeback_check does. This is the check of the framebuffer
 * object itself (its offset, pitch and format), which reading the buffer's
 * memory cannot give. Uses the writeback CRTC and a free plane, and leaves
 * both off. 0, or a negative step number. */
static int capture_scanout(int fd, drmModeRes* res, uint32_t fb, uint32_t w, uint32_t h, const drmModeModeInfo* mode, uint32_t* out) {
    if (drmSetClientCap(fd, DRM_CLIENT_CAP_ATOMIC, 1) || drmSetClientCap(fd, DRM_CLIENT_CAP_WRITEBACK_CONNECTORS, 1)) return -1;
    drmModeRes* r2 = drmModeGetResources(fd); /* again: writeback connectors are listed only with the cap */
    uint32_t wb = 0, mask = 0, crtc = 0, plane = 0;
    int ci = -1;
    for (int i = 0; r2 && i < r2->count_connectors && !wb; i++) {
        drmModeConnector* c = drmModeGetConnectorCurrent(fd, r2->connectors[i]);
        if (c && c->connector_type == DRM_MODE_CONNECTOR_WRITEBACK) {
            wb = c->connector_id;
            for (int e = 0; e < c->count_encoders; e++) {
                drmModeEncoder* en = drmModeGetEncoder(fd, c->encoders[e]);
                if (en) mask |= en->possible_crtcs;
                drmModeFreeEncoder(en);
            }
        }
        drmModeFreeConnector(c);
    }
    for (int k = 0; r2 && k < r2->count_crtcs && !crtc; k++)
        if (mask & (1u << k)) crtc = r2->crtcs[k], ci = k;
    drmModeFreeResources(r2);
    (void)res;
    if (!wb || !crtc) return -2;
    drmModePlaneRes* pr = drmModeGetPlaneResources(fd);
    for (uint32_t i = 0; pr && i < pr->count_planes && !plane; i++) {
        drmModePlane* pl = drmModeGetPlane(fd, pr->planes[i]);
        if (pl && (pl->possible_crtcs & (1u << ci)) && !pl->crtc_id) plane = pl->plane_id;
        drmModeFreePlane(pl);
    }
    drmModeFreePlaneResources(pr);
    if (!plane) return -3;
    struct drm_mode_create_dumb cd = {.width = w, .height = h, .bpp = 32};
    if (drmIoctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &cd)) return -4;
    struct drm_mode_map_dumb md = {.handle = cd.handle};
    uint32_t ofb = 0, blob = 0, oh[4] = {cd.handle}, op_[4] = {cd.pitch}, oo[4] = {0};
    uint32_t* om = MAP_FAILED;
    int rc = -5, fence = -1;
    if (!drmModeAddFB2(fd, w, h, DRM_FORMAT_XRGB8888, oh, op_, oo, &ofb, 0) && !drmIoctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &md))
        om = mmap(NULL, cd.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)md.offset);
    if (om != MAP_FAILED) {
        for (uint64_t i = 0; i < cd.size / 4; i++) om[i] = 0xDEADBEEF;
        drmModeCreatePropertyBlob(fd, mode, sizeof *mode, &blob);
        drmModeAtomicReq* q = drmModeAtomicAlloc();
        int bad = padd(q, fd, crtc, DRM_MODE_OBJECT_CRTC, "MODE_ID", blob) | padd(q, fd, crtc, DRM_MODE_OBJECT_CRTC, "ACTIVE", 1) |
                  padd(q, fd, wb, DRM_MODE_OBJECT_CONNECTOR, "CRTC_ID", crtc) | padd(q, fd, plane, DRM_MODE_OBJECT_PLANE, "FB_ID", fb) |
                  padd(q, fd, plane, DRM_MODE_OBJECT_PLANE, "CRTC_ID", crtc) | padd(q, fd, plane, DRM_MODE_OBJECT_PLANE, "SRC_X", 0) |
                  padd(q, fd, plane, DRM_MODE_OBJECT_PLANE, "SRC_Y", 0) | padd(q, fd, plane, DRM_MODE_OBJECT_PLANE, "SRC_W", (uint64_t)w << 16) |
                  padd(q, fd, plane, DRM_MODE_OBJECT_PLANE, "SRC_H", (uint64_t)h << 16) | padd(q, fd, plane, DRM_MODE_OBJECT_PLANE, "CRTC_X", 0) |
                  padd(q, fd, plane, DRM_MODE_OBJECT_PLANE, "CRTC_Y", 0) | padd(q, fd, plane, DRM_MODE_OBJECT_PLANE, "CRTC_W", w) |
                  padd(q, fd, plane, DRM_MODE_OBJECT_PLANE, "CRTC_H", h) | padd(q, fd, plane, DRM_MODE_OBJECT_PLANE, "alpha", 0xffff) |
                  padd(q, fd, wb, DRM_MODE_OBJECT_CONNECTOR, "WRITEBACK_FB_ID", ofb) |
                  padd(q, fd, wb, DRM_MODE_OBJECT_CONNECTOR, "WRITEBACK_OUT_FENCE_PTR", (uint64_t)(uintptr_t)&fence);
        rc = bad ? -6 : drmModeAtomicCommit(fd, q, DRM_MODE_ATOMIC_ALLOW_MODESET, NULL) ? -7 : 0;
        drmModeAtomicFree(q);
        if (!rc) {
            struct pollfd pf = {.fd = fence, .events = POLLIN};
            if (fence < 0 || poll(&pf, 1, 2000) <= 0) rc = -8;
            if (fence >= 0) close(fence);
        }
        if (!rc)
            for (uint32_t y = 0; y < h; y++) memcpy(out + (size_t)y * w, om + (size_t)y * (cd.pitch / 4), (size_t)w * 4);
        /* leave the writeback CRTC and the plane off */
        q = drmModeAtomicAlloc();
        padd(q, fd, plane, DRM_MODE_OBJECT_PLANE, "FB_ID", 0), padd(q, fd, plane, DRM_MODE_OBJECT_PLANE, "CRTC_ID", 0);
        padd(q, fd, wb, DRM_MODE_OBJECT_CONNECTOR, "CRTC_ID", 0), padd(q, fd, crtc, DRM_MODE_OBJECT_CRTC, "ACTIVE", 0);
        padd(q, fd, crtc, DRM_MODE_OBJECT_CRTC, "MODE_ID", 0);
        if (drmModeAtomicCommit(fd, q, DRM_MODE_ATOMIC_ALLOW_MODESET, NULL) && !rc) rc = -9;
        drmModeAtomicFree(q);
        if (blob) drmModeDestroyPropertyBlob(fd, blob);
        munmap(om, cd.size);
    }
    if (ofb) drmModeRmFB(fd, ofb);
    struct drm_mode_destroy_dumb dd = {.handle = cd.handle};
    drmIoctl(fd, DRM_IOCTL_MODE_DESTROY_DUMB, &dd);
    return rc;
}

int main(int argc, char** argv) {
    uint64_t t_start = now_ns();
    double seconds = argc > 1 ? atof(argv[1]) : 5;
    uint32_t batches = argc > 2 ? (uint32_t)atoi(argv[2]) : 8;
    int check_every = argc > 3 ? atoi(argv[3]) : 30, fault = argc > 4 ? atoi(argv[4]) : 0;
    const char* fail = NULL;
    int card = -1, rc = 0, master = 0, crtc_set = 0, restore = 0, crtc_index = -1, plane_changes = 0;
    hs_v3d dev = {.fd = -1};
    hs_v3d_bo code = {0}, unif = {0};
    drmModeRes* res = NULL;
    drmModeCrtc* saved = NULL;
    buf b[2];
    memset(b, 0, sizeof b);
    uint32_t conn_id = 0, crtc_id = 0, w = 0, h = 0;
    drmModeModeInfo mode;
    memset(&mode, 0, sizeof mode);
    int frames = 0, checked_buf[2] = {0, 0}, cap = 0, nr = 0, ni = 0, late = 0;
    uint64_t wrong = 0, guard_hit = 0, last_flip = 0, t0 = 0, scan_wrong = 0;
    int scan_rc = 1; /* 1: not run */
    double first_frame_ms = 0, el = 0, check_ms_max = 0, *render = NULL, *interval = NULL;
    if (seconds <= 0 || seconds > 3600 || (batches != 1 && batches != 2 && batches != 4 && batches != 8) || check_every < 0)
        return fprintf(stderr, "usage: seconds 0 < s <= 3600, batches 1|2|4|8, check_every >= 0\n"), 2;
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    for (int i = 0; i < 8 && card < 0; i++) {
        char path[32];
        snprintf(path, sizeof path, "/dev/dri/card%d", i);
        int fd = open(path, O_RDWR | O_CLOEXEC);
        if (fd < 0) continue;
        drmVersion* v = drmGetVersion(fd);
        if (v && !strcmp(v->name, "vc4")) card = fd;
        else close(fd);
        if (v) drmFreeVersion(v);
    }
    if (card < 0) { fail = "no vc4 card"; goto out; }
    if ((rc = hs_v3d_open(&dev, NULL))) { fail = "no v3d render node"; goto out; }
    if (drmSetMaster(card)) { rc = -errno; fail = rc == -EBUSY || rc == -EACCES ? "not DRM master: something else holds the display, or no permission" : "drmSetMaster failed"; goto out; }
    master = 1;

    /* the connected output: prefer one that already has a CRTC, then the DSI
     * panel. The kernel's cached connector state is used first: probing a
     * DSI panel is slow (hs_graphics). */
    res = drmModeGetResources(card);
    int best = -1;
    for (int i = 0; res && i < res->count_connectors; i++) {
        drmModeConnector* c = drmModeGetConnectorCurrent(card, res->connectors[i]);
        if (c && c->connection == DRM_MODE_CONNECTED && !c->count_modes) {
            drmModeFreeConnector(c);
            c = drmModeGetConnector(card, res->connectors[i]);
        }
        if (!c) continue;
        if (c->connection == DRM_MODE_CONNECTED && c->count_modes > 0) {
            uint32_t crtc = 0;
            int score = c->connector_type == DRM_MODE_CONNECTOR_DSI;
            drmModeEncoder* e = c->encoder_id ? drmModeGetEncoder(card, c->encoder_id) : NULL;
            if (e && e->crtc_id) crtc = e->crtc_id, score += 2;
            if (e) drmModeFreeEncoder(e);
            for (int k = 0; !crtc && k < c->count_encoders; k++) {
                e = drmModeGetEncoder(card, c->encoders[k]);
                for (int x = 0; e && x < res->count_crtcs && x < 32 && !crtc; x++)
                    if (e->possible_crtcs & (1u << x)) crtc = res->crtcs[x];
                if (e) drmModeFreeEncoder(e);
            }
            if (crtc && score > best) best = score, conn_id = c->connector_id, crtc_id = crtc, mode = c->modes[0];
        }
        drmModeFreeConnector(c);
    }
    if (best < 0) { rc = -ENODEV; fail = "no connected output"; goto out; }
    for (int i = 0; i < res->count_crtcs; i++)
        if (res->crtcs[i] == crtc_id) crtc_index = i;
    saved = drmModeGetCrtc(card, crtc_id);
    w = mode.hdisplay, h = mode.vdisplay;
    uint64_t npix = (uint64_t)w * h;
    /* 24-bit multiply operands: pixels per batch, and batch * pixels + lane + frame term */
    if (!npix || npix % (16ull * WAIT_EVERY * batches) || npix + 16 + 0xFFFF >= (1u << 24)) {
        rc = -EINVAL;
        fail = "mode size does not fit the kernel (batches, or over 2^24 pixels)";
        goto out;
    }

    uint32_t guard_words = GUARD_ROWS * w;
    for (int i = 0; i < 2; i++) {
        struct drm_mode_create_dumb cd = {.width = w, .height = h + 2 * GUARD_ROWS, .bpp = 32};
        if (drmIoctl(card, DRM_IOCTL_MODE_CREATE_DUMB, &cd)) { rc = -errno; fail = "create dumb buffer"; goto out; }
        b[i].handle = cd.handle, b[i].pitch = cd.pitch, b[i].size = cd.size;
        struct drm_mode_map_dumb md = {.handle = cd.handle};
        if (cd.pitch != w * 4 || drmIoctl(card, DRM_IOCTL_MODE_MAP_DUMB, &md)) { rc = -EINVAL; fail = "map dumb buffer (or pitch != 4 * width)"; goto out; }
        void* m = mmap(NULL, cd.size, PROT_READ | PROT_WRITE, MAP_SHARED, card, (off_t)md.offset);
        if (m == MAP_FAILED) { rc = -errno; fail = "mmap dumb buffer"; goto out; }
        b[i].all = m, b[i].map = b[i].all + guard_words;
        for (uint32_t k = 0; k < guard_words; k++) b[i].all[k] = GUARD, b[i].map[npix + k] = GUARD;
        int pfd = -1;
        if (drmPrimeHandleToFD(card, cd.handle, DRM_CLOEXEC | DRM_RDWR, &pfd)) { rc = -errno; fail = "export"; goto out; }
        struct drm_prime_handle ih = {.fd = pfd};
        int e = drmIoctl(dev.fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &ih) ? -errno : 0;
        close(pfd);
        if (e) { rc = e; fail = "import into v3d"; goto out; }
        b[i].v3d_handle = ih.handle;
        struct drm_v3d_get_bo_offset go = {.handle = ih.handle};
        if (drmIoctl(dev.fd, DRM_IOCTL_V3D_GET_BO_OFFSET, &go)) { rc = -errno; fail = "GPU address"; goto out; }
        b[i].gpu = go.offset + guard_words * 4;
        /* the framebuffer is the visible frame only: it starts after the guard rows */
        uint32_t fh[4] = {cd.handle}, fp[4] = {cd.pitch}, fo[4] = {fault == 5 ? 0 : guard_words * 4};
        if (drmModeAddFB2(card, w, h, DRM_FORMAT_XRGB8888, fh, fp, fo, &b[i].fb, 0)) { rc = -errno; fail = "add framebuffer"; goto out; }
    }
    static uint64_t code_buf[256];
    if ((rc = hs_v3d_bo_new(&dev, &code, 4096)) || (rc = hs_v3d_bo_new(&dev, &unif, 4096))) { fail = "code buffers"; goto out; }
    hs_qpu_prog p;
    hs_qpu_prog_init(&p, code_buf, 256);
    int len = build(&p);
    uint32_t prog = len > 0 ? hs_v3d_load(&code, 0, code_buf, (uint32_t)len) : 0;
    if (!prog) { rc = len < 0 ? len : -EINVAL; fail = "program build"; goto out; }
    uint32_t pixels = (uint32_t)(npix / batches);
    uint32_t* u = (uint32_t*)unif.map;
    cap = (int)(seconds * 400) + 16;
    render = malloc(sizeof(double) * (size_t)cap), interval = malloc(sizeof(double) * (size_t)cap);
    if (!render || !interval) { rc = -ENOMEM; fail = "malloc"; goto out; }

    drmEventContext ev = {.version = 2, .page_flip_handler = on_flip};
    while (!rc && !stop_requested) {
        buf* back = &b[frames & 1];
        uint32_t fterm = (7u * (uint32_t)frames) & 0xFFFF; /* keeps the multiply's operand under 2^24 for any run length */
        u[0] = back->gpu, u[1] = pixels / (16 * WAIT_EVERY) + (fault == 3), u[2] = pixels, u[3] = fault == 1 || (fault == 4 && frames >= 2 && frames < 12) ? fterm + 1 : fterm, u[4] = MULT,
        u[5] = 16 * MULT;
        uint32_t hh[3] = {code.handle, unif.handle, back->v3d_handle};
        hs_v3d_dispatch_desc d = {prog, unif.gpu, batches, hh, 3, 0};
        uint64_t t = now_ns();
        rc = hs_v3d_dispatch(&dev, &d);
        if (!rc) rc = hs_v3d_bo_wait(&dev, &code, 2000000000ull);
        if (rc) { fail = "compute job"; break; }
        if (nr < cap) render[nr++] = (now_ns() - t) / 1e6;
        if (frames == 0) {
            plane_changes = reset_planes(card, crtc_id, crtc_index);
            if (drmModeSetCrtc(card, crtc_id, back->fb, 0, 0, &conn_id, 1, &mode)) { rc = -errno; fail = "set CRTC"; break; }
            crtc_set = 1;
            first_frame_ms = (now_ns() - t_start) / 1e6;
            t0 = now_ns();
            last_flip = 0;
        } else {
            if (drmModePageFlip(card, crtc_id, back->fb, DRM_MODE_PAGE_FLIP_EVENT, NULL)) { rc = -errno; fail = "page flip"; break; }
            flip_pending = 1;
            while (flip_pending && !rc) {
                struct pollfd pf = {.fd = card, .events = POLLIN};
                int pr = poll(&pf, 1, 1000);
                if (pr < 0 && errno == EINTR) continue; /* a signal: the flip is still owed, wait for it */
                if (pr <= 0) { rc = -ETIME; fail = "no flip event within 1 s"; break; }
                drmHandleEvent(card, &ev);
            }
            if (rc) break;
            if (last_flip && ni < cap) interval[ni++] = (flip_ns - last_flip) / 1e6;
            last_flip = flip_ns;
        }
        /* Check after the frame is on its way to the panel, so the check
         * costs the next frame's slack and not this frame's flip. Frames n
         * and n + 1 are checked together: one in each buffer. */
        if (check_every > 0 && (frames % check_every == 0 || frames % check_every == 1 || check_every == 1)) {
            uint64_t tc = now_ns();
            for (uint32_t i = 0; i < w * h; i++) wrong += back->map[i] != (i + fterm) * (uint32_t)MULT;
            for (uint32_t k = 0; k < guard_words; k++) guard_hit += (back->all[k] != GUARD) + (back->map[npix + k] != GUARD);
            checked_buf[frames & 1]++;
            double cm = (now_ns() - tc) / 1e6;
            if (cm > check_ms_max) check_ms_max = cm;
        }
        frames++;
        if (now_ns() - t0 > (uint64_t)(seconds * 1e9)) break;
        if (fault == 2 && now_ns() - t0 > 1000000000ull) raise(SIGINT);
    }
    if (t0) el = (now_ns() - t0) / 1e9;
    /* what the display controller composes from the framebuffer last shown */
    if (!rc && frames > 0 && check_every > 0) {
        uint32_t* cap_px = malloc((size_t)w * h * 4);
        uint32_t ft = (7u * (uint32_t)(frames - 1)) & 0xFFFF;
        scan_rc = cap_px ? capture_scanout(card, res, b[(frames - 1) & 1].fb, w, h, &mode, cap_px) : -ENOMEM;
        for (uint32_t i = 0; !scan_rc && i < w * h; i++) scan_wrong += (cap_px[i] & 0x00FFFFFF) != (((i + ft) * (uint32_t)MULT) & 0x00FFFFFF);
        free(cap_px);
    }

out:
    /* give the display back as it was: the saved CRTC (or off, if it was off), then master */
    if (crtc_set) {
        if (saved && saved->mode_valid && saved->buffer_id)
            restore = drmModeSetCrtc(card, saved->crtc_id, saved->buffer_id, saved->x, saved->y, &conn_id, 1, &saved->mode) ? -errno : 0;
        else
            restore = drmModeSetCrtc(card, crtc_id, 0, 0, 0, NULL, 0, NULL) ? -errno : 0;
    }
    for (int i = 0; i < 2; i++) {
        if (b[i].fb) drmModeRmFB(card, b[i].fb);
        if (b[i].all) munmap(b[i].all, b[i].size);
        if (b[i].v3d_handle) {
            struct drm_gem_close gc = {.handle = b[i].v3d_handle};
            drmIoctl(dev.fd, DRM_IOCTL_GEM_CLOSE, &gc);
        }
        if (b[i].handle) {
            struct drm_mode_destroy_dumb dd = {.handle = b[i].handle};
            drmIoctl(card, DRM_IOCTL_MODE_DESTROY_DUMB, &dd);
        }
    }
    if (master) drmDropMaster(card);
    if (saved) drmModeFreeCrtc(saved);
    if (res) drmModeFreeResources(res);
    if (unif.handle) hs_v3d_bo_free(&dev, &unif);
    if (code.handle) hs_v3d_bo_free(&dev, &code);
    if (dev.fd >= 0) hs_v3d_close(&dev);
    if (card >= 0) close(card);
    if (render) qsort(render, (size_t)nr, sizeof(double), cmpd);
    if (interval) qsort(interval, (size_t)ni, sizeof(double), cmpd);
    double refresh_ms = 1000.0 / (mode.vrefresh ? mode.vrefresh : 60);
    for (int i = 0; i < ni; i++) late += interval[i] > 1.5 * refresh_ms;
    printf("{\"present\":\"QPU compute to KMS, no Mesa\",\"mode\":\"%ux%u@%u\",\"batches\":%u,\"fault\":%d,\"rc\":%d,\"failed_at\":\"%s\","
           "\"stopped_by_signal\":%s,\"frames\":%d,\"seconds\":%.3f,\"fps\":%.2f,\"process_start_to_first_frame_on_crtc_ms\":%.2f,"
           "\"render_ms\":{\"p50\":%.3f,\"p99\":%.3f,\"max\":%.3f},\"flip_interval_ms\":{\"p50\":%.3f,\"p99\":%.3f,\"max\":%.3f},"
           "\"intervals_over_1.5_refreshes\":%d,\"frames_checked_in_buffer\":[%d,%d],\"wrong_pixels\":%llu,\"guard_words_overwritten\":%llu,"
           "\"check_ms_max\":%.2f,\"scanout_capture\":{\"rc\":%d,\"wrong_pixels\":%llu},\"plane_changes_at_takeover\":%d,"
           "\"display_restored_rc\":%d}\n",
           w, h, mode.vrefresh, batches, fault, rc, fail ? fail : "", stop_requested ? "true" : "false", frames, el, el > 0 && frames > 1 ? (frames - 1) / el : 0,
           first_frame_ms, nr ? render[nr / 2] : 0, nr ? render[nr * 99 / 100] : 0, nr ? render[nr - 1] : 0, ni ? interval[ni / 2] : 0,
           ni ? interval[ni * 99 / 100] : 0, ni ? interval[ni - 1] : 0, late, checked_buf[0], checked_buf[1], (unsigned long long)wrong, (unsigned long long)guard_hit, check_ms_max,
           scan_rc, (unsigned long long)scan_wrong, plane_changes, restore);
    free(render), free(interval);
    /* a run long enough to use both buffers must have checked both */
    int unchecked = check_every > 0 && !rc && !stop_requested && frames > check_every + 1 && (!checked_buf[0] || !checked_buf[1]);
    int scan_bad = scan_rc < 0 || scan_wrong || (scan_rc == 1 && check_every > 0 && !rc && frames > 0);
    return rc || wrong || guard_hit || restore || unchecked || scan_bad;
}
