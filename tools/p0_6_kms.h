/* p0_6_kms.h: the KMS side of the P0-6 tools, with libdrm only (no Mesa,
 * EGL or GBM): plane reset at takeover, the scanout capture through the
 * writeback connector, and (kms_*) taking the display, buffers shared with
 * v3d, and giving the display back. Used by tools/p0_6_present_probe (the
 * first two; its own main does the rest, as red-teamed) and by
 * tools/p0_6_scene_present_probe (all of it).
 */
#ifndef P0_6_KMS_H
#define P0_6_KMS_H
#include <drm/v3d_drm.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm_fourcc.h>

#include "hs_v3d.h"

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

/* ---- taking the display, buffers, giving it back ---- */
typedef struct {
    uint32_t handle, fb, v3d_handle, gpu, pitch; /* gpu: address of the first visible pixel */
    uint64_t size;
    uint32_t guard_words;
    uint32_t* all; /* the whole buffer: guard rows, the visible frame, guard rows */
    uint32_t* map; /* the visible frame */
} kms_buf;
typedef struct {
    int card, master, crtc_set, crtc_index, plane_changes;
    drmModeRes* res;
    drmModeCrtc* saved;
    uint32_t conn_id, crtc_id, w, h;
    drmModeModeInfo mode;
} kms;
#define KMS_GUARD 0xA5C3F00Du

/* Opens the vc4 card, becomes DRM master, picks the connected output (one
 * that already has a CRTC first, then the DSI panel) and saves its CRTC.
 * Nothing on the display changes yet. NULL, or what failed (*rc: -errno). */
static inline const char* kms_open(kms* k, int* rc) {
    memset(k, 0, sizeof *k);
    k->card = -1, k->crtc_index = -1;
    for (int i = 0; i < 8 && k->card < 0; i++) {
        char path[32];
        snprintf(path, sizeof path, "/dev/dri/card%d", i);
        int fd = open(path, O_RDWR | O_CLOEXEC);
        if (fd < 0) continue;
        drmVersion* v = drmGetVersion(fd);
        if (v && !strcmp(v->name, "vc4")) k->card = fd;
        else close(fd);
        if (v) drmFreeVersion(v);
    }
    if (k->card < 0) return *rc = -ENODEV, "no vc4 card";
    if (drmSetMaster(k->card)) {
        *rc = -errno;
        return *rc == -EBUSY || *rc == -EACCES ? "not DRM master: something else holds the display, or no permission" : "drmSetMaster failed";
    }
    k->master = 1;
    k->res = drmModeGetResources(k->card);
    int best = -1;
    for (int i = 0; k->res && i < k->res->count_connectors; i++) {
        /* the kernel's cached connector state first: probing a DSI panel is slow */
        drmModeConnector* c = drmModeGetConnectorCurrent(k->card, k->res->connectors[i]);
        if (c && c->connection == DRM_MODE_CONNECTED && !c->count_modes) {
            drmModeFreeConnector(c);
            c = drmModeGetConnector(k->card, k->res->connectors[i]);
        }
        if (!c) continue;
        if (c->connection == DRM_MODE_CONNECTED && c->count_modes > 0) {
            uint32_t crtc = 0;
            int score = c->connector_type == DRM_MODE_CONNECTOR_DSI;
            drmModeEncoder* e = c->encoder_id ? drmModeGetEncoder(k->card, c->encoder_id) : NULL;
            if (e && e->crtc_id) crtc = e->crtc_id, score += 2;
            if (e) drmModeFreeEncoder(e);
            for (int n = 0; !crtc && n < c->count_encoders; n++) {
                e = drmModeGetEncoder(k->card, c->encoders[n]);
                for (int x = 0; e && x < k->res->count_crtcs && x < 32 && !crtc; x++)
                    if (e->possible_crtcs & (1u << x)) crtc = k->res->crtcs[x];
                if (e) drmModeFreeEncoder(e);
            }
            if (crtc && score > best) best = score, k->conn_id = c->connector_id, k->crtc_id = crtc, k->mode = c->modes[0];
        }
        drmModeFreeConnector(c);
    }
    if (best < 0) return *rc = -ENODEV, "no connected output";
    for (int i = 0; i < k->res->count_crtcs; i++)
        if (k->res->crtcs[i] == k->crtc_id) k->crtc_index = i;
    k->saved = drmModeGetCrtc(k->card, k->crtc_id);
    k->w = k->mode.hdisplay, k->h = k->mode.vdisplay;
    return NULL;
}
/* A vc4 dumb buffer of the mode's size with guard_rows marked rows above and
 * below, imported into v3d, and a framebuffer object over the visible rows
 * (fb_offset_fault: over the first rows of the buffer, the guard rows: a
 * fault for the scanout capture to find). */
static inline const char* kms_buf_new(kms* k, hs_v3d* dev, kms_buf* b, uint32_t guard_rows, int fb_offset_fault, int* rc) {
    uint64_t npix = (uint64_t)k->w * k->h;
    struct drm_mode_create_dumb cd = {.width = k->w, .height = k->h + 2 * guard_rows, .bpp = 32};
    if (drmIoctl(k->card, DRM_IOCTL_MODE_CREATE_DUMB, &cd)) return *rc = -errno, "create dumb buffer";
    b->handle = cd.handle, b->pitch = cd.pitch, b->size = cd.size, b->guard_words = guard_rows * k->w;
    struct drm_mode_map_dumb md = {.handle = cd.handle};
    if (cd.pitch != k->w * 4 || drmIoctl(k->card, DRM_IOCTL_MODE_MAP_DUMB, &md)) return *rc = -EINVAL, "map dumb buffer (or pitch != 4 * width)";
    void* m = mmap(NULL, cd.size, PROT_READ | PROT_WRITE, MAP_SHARED, k->card, (off_t)md.offset);
    if (m == MAP_FAILED) return *rc = -errno, "mmap dumb buffer";
    b->all = m, b->map = b->all + b->guard_words;
    for (uint32_t i = 0; i < b->guard_words; i++) b->all[i] = KMS_GUARD, b->map[npix + i] = KMS_GUARD;
    int pfd = -1;
    if (drmPrimeHandleToFD(k->card, cd.handle, DRM_CLOEXEC | DRM_RDWR, &pfd)) return *rc = -errno, "export";
    struct drm_prime_handle ih = {.fd = pfd};
    int e = drmIoctl(dev->fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &ih) ? -errno : 0;
    close(pfd);
    if (e) return *rc = e, "import into v3d";
    b->v3d_handle = ih.handle;
    struct drm_v3d_get_bo_offset go = {.handle = ih.handle};
    if (drmIoctl(dev->fd, DRM_IOCTL_V3D_GET_BO_OFFSET, &go)) return *rc = -errno, "GPU address";
    b->gpu = go.offset + b->guard_words * 4;
    uint32_t fh[4] = {cd.handle}, fp[4] = {cd.pitch}, fo[4] = {fb_offset_fault ? 0 : b->guard_words * 4};
    if (drmModeAddFB2(k->card, k->w, k->h, DRM_FORMAT_XRGB8888, fh, fp, fo, &b->fb, 0)) return *rc = -errno, "add framebuffer";
    return NULL;
}
static inline uint64_t kms_buf_guard_hits(const kms* k, const kms_buf* b) {
    uint64_t hits = 0, npix = (uint64_t)k->w * k->h;
    for (uint32_t i = 0; b->all && i < b->guard_words; i++) hits += (b->all[i] != KMS_GUARD) + (b->map[npix + i] != KMS_GUARD);
    return hits;
}
/* The first frame: planes another client left are turned off, the primary
 * plane's properties go back to their defaults, and the buffer is put on
 * the CRTC. 0 or -errno. */
static inline int kms_show_first(kms* k, const kms_buf* b) {
    k->plane_changes = reset_planes(k->card, k->crtc_id, k->crtc_index);
    if (drmModeSetCrtc(k->card, k->crtc_id, b->fb, 0, 0, &k->conn_id, 1, &k->mode)) return -errno;
    k->crtc_set = 1;
    return 0;
}
static inline void kms_buf_free(kms* k, hs_v3d* dev, kms_buf* b) {
    if (b->fb) drmModeRmFB(k->card, b->fb);
    if (b->all) munmap(b->all, b->size);
    if (b->v3d_handle) {
        struct drm_gem_close gc = {.handle = b->v3d_handle};
        drmIoctl(dev->fd, DRM_IOCTL_GEM_CLOSE, &gc);
    }
    if (b->handle) {
        struct drm_mode_destroy_dumb dd = {.handle = b->handle};
        drmIoctl(k->card, DRM_IOCTL_MODE_DESTROY_DUMB, &dd);
    }
    memset(b, 0, sizeof *b);
}
/* Gives the display back as it was: the saved CRTC (or off, if it was off).
 * Call before the buffers are freed. 0 or -errno. */
static inline int kms_restore(kms* k) {
    int r = 0;
    if (k->crtc_set) {
        if (k->saved && k->saved->mode_valid && k->saved->buffer_id)
            r = drmModeSetCrtc(k->card, k->saved->crtc_id, k->saved->buffer_id, k->saved->x, k->saved->y, &k->conn_id, 1, &k->saved->mode) ? -errno : 0;
        else
            r = drmModeSetCrtc(k->card, k->crtc_id, 0, 0, 0, NULL, 0, NULL) ? -errno : 0;
        k->crtc_set = 0;
    }
    return r;
}
static inline void kms_close(kms* k) {
    if (k->master) drmDropMaster(k->card);
    if (k->saved) drmModeFreeCrtc(k->saved);
    if (k->res) drmModeFreeResources(k->res);
    if (k->card >= 0) close(k->card);
    k->master = 0, k->saved = NULL, k->res = NULL, k->card = -1;
}
#endif
