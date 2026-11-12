/* hs_v3d: direct v3d kernel-driver access. See include/hs_v3d.h. */
#define _GNU_SOURCE
#include "hs_v3d.h"

#include <drm/drm.h>
#include <drm/v3d_drm.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

static int xioctl(int fd, unsigned long req, void* arg) {
    int r;
    do r = ioctl(fd, req, arg);
    while (r == -1 && (errno == EINTR || errno == EAGAIN));
    return r == -1 ? -errno : 0;
}

/* 1 if the node is driven by v3d */
static int is_v3d(int fd) {
    char name[16] = {0};
    struct drm_version v;
    memset(&v, 0, sizeof(v));
    v.name = name;
    v.name_len = sizeof(name) - 1;
    if (xioctl(fd, DRM_IOCTL_VERSION, &v)) return 0;
    return v.name_len == 3 && memcmp(name, "v3d", 3) == 0;
}

int hs_v3d_open(hs_v3d* dev, const char* path) {
    dev->fd = -1;
    if (path) {
        int fd = open(path, O_RDWR | O_CLOEXEC);
        if (fd < 0) return -errno;
        if (!is_v3d(fd)) {
            close(fd);
            return -ENODEV;
        }
        dev->fd = fd;
        return 0;
    }
    int last = -ENODEV;
    for (int i = 128; i < 192; i++) {
        char p[32];
        snprintf(p, sizeof(p), "/dev/dri/renderD%d", i);
        int fd = open(p, O_RDWR | O_CLOEXEC);
        if (fd < 0) {
            if (errno != ENOENT) last = -errno; /* e.g. EACCES: not in the render group */
            continue;
        }
        if (is_v3d(fd)) {
            dev->fd = fd;
            return 0;
        }
        close(fd);
    }
    return last;
}

void hs_v3d_close(hs_v3d* dev) {
    if (dev->fd >= 0) close(dev->fd);
    dev->fd = -1;
}

static void gem_close(int fd, uint32_t handle) {
    struct drm_gem_close c = {.handle = handle};
    xioctl(fd, DRM_IOCTL_GEM_CLOSE, &c);
}

int hs_v3d_bo_new(hs_v3d* dev, hs_v3d_bo* bo, size_t size) {
    memset(bo, 0, sizeof(*bo));
    if (size == 0 || size > 0xFFFFFFFFu) return -EINVAL;
    struct drm_v3d_create_bo c = {.size = (uint32_t)size};
    int e = xioctl(dev->fd, DRM_IOCTL_V3D_CREATE_BO, &c);
    if (e) return e;
    struct drm_v3d_mmap_bo m = {.handle = c.handle};
    if ((e = xioctl(dev->fd, DRM_IOCTL_V3D_MMAP_BO, &m))) {
        gem_close(dev->fd, c.handle);
        return e;
    }
    void* map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, dev->fd, (off_t)m.offset);
    if (map == MAP_FAILED) {
        e = -errno;
        gem_close(dev->fd, c.handle);
        return e;
    }
    bo->handle = c.handle;
    bo->gpu = c.offset;
    bo->size = size;
    bo->map = map;
    return 0;
}

void hs_v3d_bo_free(hs_v3d* dev, hs_v3d_bo* bo) {
    if (bo->map) munmap(bo->map, bo->size);
    if (bo->handle) gem_close(dev->fd, bo->handle);
    memset(bo, 0, sizeof(*bo));
}

int hs_v3d_bo_wait(hs_v3d* dev, const hs_v3d_bo* bo, uint64_t timeout_ns) {
    struct drm_v3d_wait_bo w = {.handle = bo->handle, .timeout_ns = timeout_ns};
    return xioctl(dev->fd, DRM_IOCTL_V3D_WAIT_BO, &w);
}

uint32_t hs_v3d_load(hs_v3d_bo* code, size_t off, const uint64_t* prog, uint32_t n) {
    if ((off & 7) || n == 0 || off > code->size || (uint64_t)n * 8 > (uint64_t)(code->size - off)) return 0;
    memcpy(code->map + off, prog, (size_t)n * 8);
    return code->gpu + (uint32_t)off;
}

int hs_v3d_dispatch(hs_v3d* dev, const hs_v3d_dispatch_desc* d) {
    if ((d->batches != 1 && d->batches != 8) || !d->code_gpu || !d->bos || !d->nbos) return -EINVAL;
    const uint32_t wg_size = 16, wgs_per_sg = 16;
    struct drm_v3d_submit_csd s;
    memset(&s, 0, sizeof(s));
    s.cfg[0] = 1u << 16; /* workgroup counts x, y, z: 1 */
    s.cfg[1] = 1u << 16;
    s.cfg[2] = 1u << 16;
    /* the supergroup field is 4 bits, 16 encoded as 0 (masked as Mesa does;
     * the packed value is the measured 0xF010 either way) */
    s.cfg[3] = ((((wgs_per_sg * wg_size + 15) / 16) - 1) << 12) | ((wgs_per_sg & 0xf) << 8) | (wg_size & 0xff);
    s.cfg[4] = d->batches - 1;
    s.cfg[5] = d->code_gpu;
    s.cfg[6] = d->unif_gpu;
    s.bo_handles = (uint64_t)(uintptr_t)d->bos;
    s.bo_handle_count = d->nbos;
    s.perfmon_id = d->perfmon;
    return xioctl(dev->fd, DRM_IOCTL_V3D_SUBMIT_CSD, &s);
}
