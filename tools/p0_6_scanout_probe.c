/* p0_6_scanout_probe: can one buffer be written by the QPUs and scanned out
 * by the display driver? (P0-6, docs/audit/HARDWARE_HANDOFF_P0.md.)
 *
 * The Pi 4 has two DRM devices: v3d (the QPUs, a render node) and vc4 (the
 * display). Mesa renders into buffers that vc4 allocated. This probe tries
 * both directions with raw ioctls, no Mesa, no libdrm:
 *   A  vc4 dumb buffer -> PRIME -> v3d handle -> GPU address -> a QPU job
 *      fills it -> the CPU checks every pixel through vc4's own mapping
 *   B  v3d buffer -> PRIME -> vc4 handle -> framebuffer object
 * and reports whether vc4 accepts each as a framebuffer (ADDFB2 only: no
 * modeset, no DRM master, nothing changes on the panel).
 * The job writes each 32-bit pixel as its own GPU address XOR 0x5A5A5A5A, 16
 * pixels per store. `wait_every`: a tmuwt after every n-th store (1: after
 * each; the form measured so far). The buffer has 16 guard rows above and
 * below the filled area, set to a marker before every job and checked after:
 * a write outside the area is counted.
 * Usage: p0_6_scanout_probe [width=800 height=480] [batches=1] [wait_every=1] [reps=20] [fault=0]
 *   fault 1: the job is given the wrong XOR constant (every pixel must be
 *            reported wrong); fault 2: one store too many per batch (guard
 *            words must be reported overwritten). Both show the checks can
 *            fail and must make the probe exit 1.
 *        (inside the GPU lock). Output: JSON lines. Exit 1 on any failure
 *        of direction A; direction B's refusal is reported, not a failure.
 */
#define _GNU_SOURCE
#include <drm/drm.h>
#include <drm/drm_fourcc.h>
#include <drm/drm_mode.h>
#include <drm/v3d_drm.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "hs_qpu.h"
#include "hs_v3d.h"

#define R(n) HS_QPU_R(n)
#define RF(n) HS_QPU_RF(n)
#define X HS_QPU_NOREG
#define I(v) hs_qpu_imm_int(v)

static int xioctl(int fd, unsigned long req, void* arg) {
    int r;
    do r = ioctl(fd, req, arg);
    while (r == -1 && (errno == EINTR || errno == EAGAIN));
    return r == -1 ? -errno : 0;
}
static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
static int open_card(const char* driver) {
    for (int i = 0; i < 8; i++) {
        char p[32], name[16] = {0};
        snprintf(p, sizeof p, "/dev/dri/card%d", i);
        int fd = open(p, O_RDWR | O_CLOEXEC);
        if (fd < 0) continue;
        struct drm_version v = {.name = name, .name_len = sizeof name - 1};
        if (!xioctl(fd, DRM_IOCTL_VERSION, &v) && !strcmp(name, driver)) return fd;
        close(fd);
    }
    return -1;
}
static void emit(hs_qpu_prog* p, hs_qpu_alu x) { hs_qpu_emit(p, &x); }
static void nop(hs_qpu_prog* p) { emit(p, hs_qpu_alu_nop()); }
static void op(hs_qpu_prog* p, int o, hs_qpu_reg d, hs_qpu_reg a, hs_qpu_reg b) { emit(p, hs_qpu_a(o, d, a, b)); }

/* uniforms: base address, stores per batch, bytes per batch, XOR constant.
 * Batch b fills [base + b * bytes, + bytes): 16 pixels per store. */
static int build(hs_qpu_prog* p, uint32_t wait_every) {
    hs_qpu_prog_rules(p, 0);
    op(p, HS_QPU_A_BAND, RF(9), RF(0), I(15)); /* the hardware's batch number */
    for (int i = 0; i < 4; i++) emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDUNIFRF, RF(i)));
    op(p, HS_QPU_A_EIDX, R(1), X, X);
    op(p, HS_QPU_A_SHL, R(0), R(1), I(2));
    op(p, HS_QPU_A_ADD, RF(0), RF(0), R(0));
    emit(p, hs_qpu_m(hs_qpu_alu_nop(), HS_QPU_M_UMUL24, R(2), RF(9), RF(2)));
    op(p, HS_QPU_A_ADD, RF(0), RF(0), R(2));
    op(p, HS_QPU_A_MOV, R(2), I(1), X);
    op(p, HS_QPU_A_SHL, RF(4), R(2), I(6)); /* 64 */
    int l = hs_qpu_label_new(p);
    hs_qpu_label_bind(p, l);
    for (uint32_t i = 0; i < wait_every; i++) {
        op(p, HS_QPU_A_BXOR, HS_QPU_TMUD, RF(0), RF(3));
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

enum { GUARD_ROWS = 16, GUARD = 0xA5C3F00D };

int main(int argc, char** argv) {
    uint32_t w = argc > 2 ? (uint32_t)atoi(argv[1]) : 800, h = argc > 2 ? (uint32_t)atoi(argv[2]) : 480;
    uint32_t batches = argc > 3 ? (uint32_t)atoi(argv[3]) : 1, wait_every = argc > 4 ? (uint32_t)atoi(argv[4]) : 1;
    int reps = argc > 5 ? atoi(argv[5]) : 20, fault = argc > 6 ? atoi(argv[6]) : 0;
    uint64_t npix = (uint64_t)w * h;
    /* the kernel multiplies batch by bytes per batch with a 24-bit multiply */
    if (!w || !h || w > 4096 || h > 4096 || reps < 1 || reps > 10000 || !wait_every || wait_every > 64 ||
        (batches != 1 && batches != 2 && batches != 4 && batches != 8) || npix % (16ull * batches) || npix * 4 / batches >= (1u << 24) ||
        (npix / 16 / batches) % wait_every)
        return fprintf(stderr, "bad arguments: width*height must divide into 16 * batches * wait_every pixels, "
                               "and bytes per batch must be under 2^24\n"), 2;
    hs_v3d dev;
    int e = hs_v3d_open(&dev, NULL), card = open_card("vc4");
    if (e || card < 0) return fprintf(stderr, "open: v3d %d, vc4 card %d\n", e, card), 1;

    /* A: vc4 dumb buffer, imported into v3d */
    uint32_t rows = h + 2 * GUARD_ROWS;
    struct drm_mode_create_dumb cd = {.width = w, .height = rows, .bpp = 32};
    if ((e = xioctl(card, DRM_IOCTL_MODE_CREATE_DUMB, &cd))) return fprintf(stderr, "create dumb: %s\n", strerror(-e)), 1;
    struct drm_mode_map_dumb md = {.handle = cd.handle};
    e = xioctl(card, DRM_IOCTL_MODE_MAP_DUMB, &md);
    uint32_t* all = e ? MAP_FAILED : mmap(NULL, cd.size, PROT_READ | PROT_WRITE, MAP_SHARED, card, (off_t)md.offset);
    if (all == MAP_FAILED) return fprintf(stderr, "map dumb failed\n"), 1;
    struct drm_prime_handle ph = {.handle = cd.handle, .flags = DRM_CLOEXEC | DRM_RDWR, .fd = -1};
    int e_export = xioctl(card, DRM_IOCTL_PRIME_HANDLE_TO_FD, &ph);
    struct drm_prime_handle pi = {.fd = ph.fd};
    int e_import = e_export ? e_export : xioctl(dev.fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &pi);
    if (!e_export) close(ph.fd);
    struct drm_v3d_get_bo_offset go = {.handle = pi.handle};
    int e_offset = e_import ? e_import : xioctl(dev.fd, DRM_IOCTL_V3D_GET_BO_OFFSET, &go);
    struct drm_mode_fb_cmd2 fb = {.width = w, .height = rows, .pixel_format = DRM_FORMAT_XRGB8888};
    fb.handles[0] = cd.handle, fb.pitches[0] = cd.pitch;
    int e_fb = xioctl(card, DRM_IOCTL_MODE_ADDFB2, &fb);
    printf("{\"A\":\"vc4 dumb buffer into v3d\",\"width\":%u,\"height\":%u,\"guard_rows\":%d,\"pitch\":%u,\"bytes\":%llu,\"export\":%d,"
           "\"import\":%d,\"gpu_offset_rc\":%d,\"gpu_address\":\"0x%08x\",\"addfb2\":%d,\"fb_id\":%u}\n",
           w, h, GUARD_ROWS, cd.pitch, (unsigned long long)cd.size, e_export, e_import, e_offset, go.offset, e_fb, fb.fb_id);
    int failures = e_export || e_import || e_offset || e_fb;

    if (!e_offset && cd.pitch == w * 4) {
        static uint64_t buf[256];
        hs_v3d_bo code, unif;
        if (hs_v3d_bo_new(&dev, &code, 4096) || hs_v3d_bo_new(&dev, &unif, 4096)) return fprintf(stderr, "bo\n"), 1;
        hs_qpu_prog p;
        hs_qpu_prog_init(&p, buf, 256);
        int len = build(&p, wait_every);
        uint32_t prog = len > 0 ? hs_v3d_load(&code, 0, buf, (uint32_t)len) : 0;
        if (!prog) return fprintf(stderr, "build failed: %d\n", len), 1;
        uint32_t per = (uint32_t)(npix * 4 / batches), stores = per / 64, guard_words = GUARD_ROWS * w;
        uint32_t base = go.offset + guard_words * 4;
        uint32_t* pix = all + guard_words;
        uint32_t* u = (uint32_t*)unif.map;
        u[0] = base, u[1] = stores / wait_every + (fault == 2), u[2] = per, u[3] = fault == 1 ? 0x5A5A5A5B : 0x5A5A5A5A;
        uint32_t hh[3] = {code.handle, unif.handle, pi.handle};
        hs_v3d_dispatch_desc d = {prog, unif.gpu, batches, hh, 3, 0};
        double best = 1e30, sum = 0;
        uint64_t wrong = 0, guard_hit = 0;
        int rc = 0;
        for (int r = 0; r < reps && !rc; r++) {
            for (uint32_t i = 0; i < guard_words; i++) all[i] = GUARD, pix[npix + i] = GUARD;
            memset(pix, 0, npix * 4);
            uint64_t t = now_ns();
            rc = hs_v3d_dispatch(&dev, &d);
            struct drm_v3d_wait_bo wb = {.handle = code.handle, .timeout_ns = 3000000000ull};
            if (!rc) rc = xioctl(dev.fd, DRM_IOCTL_V3D_WAIT_BO, &wb);
            double ms = (now_ns() - t) / 1e6;
            if (ms < best) best = ms;
            sum += ms;
            for (uint32_t i = 0; i < npix; i++) wrong += pix[i] != ((base + 4 * i) ^ 0x5A5A5A5A);
            for (uint32_t i = 0; i < guard_words; i++) guard_hit += (all[i] != GUARD) + (pix[npix + i] != GUARD);
        }
        printf("{\"A_fill\":\"a QPU job fills the vc4 buffer\",\"batches\":%u,\"wait_every\":%u,\"fault\":%d,\"reps\":%d,\"rc\":%d,"
               "\"wrong_pixels\":%llu,\"guard_words_overwritten\":%llu,\"job_ms_submit_to_done\":{\"best\":%.3f,\"mean\":%.3f},"
               "\"ns_per_pixel_best\":%.2f}\n",
               batches, wait_every, fault, reps, rc, (unsigned long long)wrong, (unsigned long long)guard_hit, best, sum / reps,
               best * 1e6 / (double)npix);
        failures += rc || wrong || guard_hit;
        hs_v3d_bo_free(&dev, &unif);
        hs_v3d_bo_free(&dev, &code);
    } else {
        printf("{\"A_fill\":\"skipped\",\"pitch\":%u}\n", cd.pitch);
        failures++;
    }
    if (!e_fb) xioctl(card, DRM_IOCTL_MODE_RMFB, &fb.fb_id);
    munmap(all, cd.size);
    if (!e_import) {
        struct drm_gem_close gc = {.handle = pi.handle};
        xioctl(dev.fd, DRM_IOCTL_GEM_CLOSE, &gc);
    }
    struct drm_mode_destroy_dumb dd = {.handle = cd.handle};
    xioctl(card, DRM_IOCTL_MODE_DESTROY_DUMB, &dd);

    /* B: a v3d buffer, imported into vc4 */
    hs_v3d_bo vb;
    if (!hs_v3d_bo_new(&dev, &vb, (size_t)npix * 4)) {
        struct drm_prime_handle eh = {.handle = vb.handle, .flags = DRM_CLOEXEC | DRM_RDWR, .fd = -1};
        int b_export = xioctl(dev.fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &eh);
        struct drm_prime_handle ih = {.fd = eh.fd};
        int b_import = b_export ? b_export : xioctl(card, DRM_IOCTL_PRIME_FD_TO_HANDLE, &ih);
        struct drm_mode_fb_cmd2 fb2 = {.width = w, .height = h, .pixel_format = DRM_FORMAT_XRGB8888};
        fb2.handles[0] = ih.handle, fb2.pitches[0] = w * 4;
        int b_fb = b_import ? b_import : xioctl(card, DRM_IOCTL_MODE_ADDFB2, &fb2);
        printf("{\"B\":\"v3d buffer into vc4\",\"export\":%d,\"import\":%d,\"addfb2\":%d,\"errno_text\":\"%s\"}\n", b_export, b_import,
               b_fb, b_fb ? strerror(-b_fb) : "ok");
        if (!b_fb) xioctl(card, DRM_IOCTL_MODE_RMFB, &fb2.fb_id);
        if (!b_import) {
            struct drm_gem_close gc = {.handle = ih.handle};
            xioctl(card, DRM_IOCTL_GEM_CLOSE, &gc);
        }
        if (!b_export) close(eh.fd);
        hs_v3d_bo_free(&dev, &vb);
    }
    close(card);
    hs_v3d_close(&dev);
    return failures != 0;
}
