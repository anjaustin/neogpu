/* p0_6_scene_present_probe: the test_08 scene rendered by QPU compute alone
 * and shown on the panel through KMS. No Mesa, EGL or GBM in the process
 * (libdrm for the modeset calls only). The kernel, its CPU reference and the
 * region sort are tools/p0_6_scene.h (described in tools/p0_6_shade_probe.c);
 * the KMS side is tools/p0_6_kms.h. (P0-6, docs/audit/P0_6_PLAN.md.)
 *
 * Three vc4 dumb buffers are imported into v3d; one compute job per frame
 * writes the image straight into the one that is neither on the panel nor
 * waiting for its flip. The timed phase is a pipeline: while a job renders,
 * the CPU prepares the next frame (region sort, sphere lists, constants) in
 * a second set of buffers.
 *
 * Two phases:
 *   timed    for `seconds`, the scene at t = elapsed time, as
 *            tests/test_08_raycast_orbit runs it. pacing 0: frames are
 *            rendered back to back; a finished frame is flipped if no flip
 *            is pending and dropped otherwise (what hs_graphics_present does
 *            for frames this fast). pacing 1: every frame is flipped and its
 *            flip waited for. pacing 2: no frame is dropped and none waits
 *            for the panel: every frame is flipped at once
 *            (DRM_MODE_PAGE_FLIP_ASYNC: the display controller is given the
 *            new buffer without waiting for the vertical blank, so the
 *            panel can show parts of several frames in one refresh). The
 *            loop waits only for the previous flip's event. No pixel is
 *            checked in this phase: the reference takes far longer than a
 *            frame.
 *   checked  `checked` frames at t = 10 * n / checked, each flipped and then
 *            compared pixel by pixel with the CPU reference (same allowance
 *            as the offscreen probe: 0.01 % of a frame), in all three
 *            buffers in turn. Each frame's hash is printed: the offscreen
 *            probe run with frames = `checked` renders the same t and must
 *            give the same hashes.
 * Then the last framebuffer shown is composed once more by the display
 * controller into a capture buffer (the writeback connector) and compared
 * with the buffer's memory: that checks the framebuffer object, not only
 * the memory behind it. Guard rows above and below every buffer are checked
 * after each phase.
 *
 * The display is taken only while this runs and given back on every exit
 * path, SIGINT and SIGTERM included (as tools/p0_6_present_probe).
 *
 * Usage: p0_6_scene_present_probe [seconds=10] [batches=8] [pacing=0] [checked=6] [fault=0]
 *   fault 1: the checked frames are rendered 0.05 s of scene time late
 *   fault 2: raises SIGINT in itself after 1 s
 *   fault 3: the framebuffer objects start at the guard rows (only the
 *            scanout capture sees it)
 *   and the kernel's own faults through P06_FAULT (tools/p0_6_shade_probe.c);
 *   P06_CULL and P06_MARGIN as there.
 *        (inside the GPU lock). Output: one JSON line. Exit 1 when a check
 *        failed, 2 on bad arguments, 3 when the run itself failed (display
 *        held by something else, setup, a job, a flip).
 */
#define _GNU_SOURCE
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hs_qpu.h"
#include "hs_v3d.h"
#include "p0_6_kms.h"
#include "p0_6_scene.h"

enum { NBUF = 3, GUARD_ROWS = 16 };

static int flip_pending = -1; /* the buffer whose flip is queued, or -1 */
static int front = -1;        /* the buffer on the panel */
static uint64_t flip_ns;
static int flips_done;
static volatile sig_atomic_t stop_requested;
static void on_signal(int sig) {
    (void)sig;
    stop_requested = 1;
}
static void on_flip(int fd, unsigned seq, unsigned sec, unsigned usec, void* data) {
    (void)fd, (void)seq, (void)data;
    front = flip_pending, flip_pending = -1, flips_done++;
    flip_ns = (uint64_t)sec * 1000000000ull + (uint64_t)usec * 1000ull;
}
static int cmpd(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}
/* Handles flip events that have arrived; with wait_ms > 0, waits that long
 * for a pending one. 0, or -ETIME when a flip was waited for and did not come. */
static int flip_events(int card, int wait_ms) {
    drmEventContext ev = {.version = 2, .page_flip_handler = on_flip};
    for (;;) {
        struct pollfd pf = {.fd = card, .events = POLLIN};
        int pr = poll(&pf, 1, flip_pending >= 0 ? wait_ms : 0);
        if (pr < 0 && errno == EINTR) continue; /* a signal: a pending flip is still owed */
        if (pr > 0) {
            drmHandleEvent(card, &ev);
            continue;
        }
        return flip_pending >= 0 && wait_ms > 0 ? -ETIME : 0;
    }
}
static int back_buffer(void) {
    for (int i = 1; i < NBUF; i++) {
        int b = ((front < 0 ? 0 : front) + i) % NBUF;
        if (b != flip_pending) return b;
    }
    return -1;
}
typedef struct {
    double* v;
    int n, cap;
} series;
static void put(series* s, double x) {
    if (s->v && s->n < s->cap) s->v[s->n++] = x;
}
static double at(const series* s, int num, int den) { return s->n ? s->v[(long)(s->n - 1) * num / den] : 0; }

int main(int argc, char** argv) {
    double seconds = argc > 1 ? atof(argv[1]) : 10;
    long v[4] = {8, 0, 6, 0};
    int bad = 0;
    for (int i = 0; i < 4 && i + 2 < argc; i++) bad |= number(argv[i + 2], 0, 1000, &v[i]);
    uint32_t batches = (uint32_t)v[0];
    int pacing = (int)v[1], checked = (int)v[2], fault = (int)v[3];
    long ev = 0;
    const char* e;
    if ((e = getenv("P06_CULL"))) bad |= number(e, 0, 15, &ev), opt_cull = (int)ev, bad |= (opt_cull & 12) == 8;
    if ((e = getenv("P06_CELLS"))) bad |= number(e, 1, 8, &ev), opt_cells = (int)ev;
    if ((e = getenv("P06_FAULT"))) bad |= number(e, 0, 16, &ev), opt_fault = (int)ev;
    if ((e = getenv("P06_MARGIN"))) {
        char* end;
        opt_margin = strtod(e, &end);
        bad |= end == e || *end || !(opt_margin > -1.0 && opt_margin < 1.0);
    }
    if (bad || !(seconds >= 0 && seconds <= 3600) || (batches != 1 && batches != 2 && batches != 4 && batches != 8) || pacing > 2 || fault > 3 ||
        (seconds == 0 && checked == 0))
        return fprintf(stderr, "usage: %s [seconds 0..3600] [batches 1|2|4|8] [pacing 0 free, drop when a flip is pending|1 every flip waited for|2 every frame flipped at once, none dropped] [checked frames 0..1000] [fault 0..3]\n",
                       argv[0]), 2;
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    const char* fail = NULL;
    int rc = 0, restore = 0, len = 0;
    kms k;
    memset(&k, 0, sizeof k);
    k.card = -1;
    hs_v3d dev = {.fd = -1};
    hs_v3d_bo code = {0}, unif[2] = {{0}, {0}}, pool[2] = {{0}, {0}}; /* two sets: the next frame's is filled while a job reads the other */
    kms_buf b[NBUF];
    memset(b, 0, sizeof b);
    static uint64_t code_buf[4096];
    uint32_t* img = malloc((size_t)W * H * 4), *cap_px = malloc((size_t)W * H * 4);
    uint8_t* kind = malloc((size_t)W * H);
    series job = {0}, cpu = {0}, loop = {0}, interval = {0}, fwait = {0};
    uint64_t async_cap = 0;
    job.cap = cpu.cap = loop.cap = interval.cap = fwait.cap = (int)(seconds * 1000) + 64;
    job.v = malloc(sizeof(double) * (size_t)job.cap), cpu.v = malloc(sizeof(double) * (size_t)job.cap);
    loop.v = malloc(sizeof(double) * (size_t)job.cap), interval.v = malloc(sizeof(double) * (size_t)job.cap);
    fwait.v = malloc(sizeof(double) * (size_t)job.cap);
    long rendered = 0, flips_queued = 0, dropped = 0, late = 0, long_min = 0, long_max = 0;
    int checked_buf[NBUF] = {0}, frames_checked = 0, frames_failed = 0, scan_rc = 1;
    long wrong_total = 0, wrong_max = 0, unwritten = 0;
    uint64_t guard_hit = 0, scan_wrong = 0, t0 = 0, last_flip = 0, t_start = now_ns();
    double el = 0, first_frame_ms = 0;
    char hashes[1024] = "";

    if (scene_fp_contracted()) { rc = -EINVAL; fail = "built without -ffp-contract=off: the reference would not match the kernel"; goto out; }
    if (!img || !cap_px || !kind || !job.v || !cpu.v || !loop.v || !interval.v || !fwait.v) { rc = -ENOMEM; fail = "malloc"; goto out; }
    if ((fail = kms_open(&k, &rc))) goto out;
    drmGetCap(k.card, DRM_CAP_ASYNC_PAGE_FLIP, &async_cap);
    if (pacing == 2 && !async_cap) { rc = -ENOTSUP; fail = "the driver does not offer immediate (async) page flips"; goto out; }
    if (k.w != W || k.h != H) { rc = -EINVAL; fail = "the mode is not 800x480: the kernel is built for that size"; goto out; }
    if ((rc = hs_v3d_open(&dev, NULL))) { fail = "no v3d render node"; goto out; }
    for (int i = 0; i < NBUF; i++)
        if ((fail = kms_buf_new(&k, &dev, &b[i], GUARD_ROWS, fault == 3, &rc))) goto out;
    hs_qpu_prog p;
    hs_qpu_prog_init(&p, code_buf, 4096);
    len = build(&p);
    if (len <= 0) { rc = len ? len : -EINVAL; fail = "program build"; goto out; }
    size_t unif_bytes = scene_unif_bytes(batches);
    if ((rc = hs_v3d_bo_new(&dev, &code, 65536)) || (rc = hs_v3d_bo_new(&dev, &unif[0], unif_bytes + 4096)) || (rc = hs_v3d_bo_new(&dev, &pool[0], scene_pool_bytes())) ||
        (rc = hs_v3d_bo_new(&dev, &unif[1], unif_bytes + 4096)) || (rc = hs_v3d_bo_new(&dev, &pool[1], scene_pool_bytes()))) {
        fail = "code buffers";
        goto out;
    }
    uint32_t prog = hs_v3d_load(&code, 0, code_buf, (uint32_t)len);
    if (!prog) { rc = -EINVAL; fail = "program load"; goto out; }

    /* ---- timed phase, then the checked frames ---- */
    int n_checked_done = 0, in_checked = seconds == 0, cur = 0, have_next = 0;
    long ncls_next[3] = {0, 0, 0};
    uint64_t t_prev = 0;
    while (!rc && !stop_requested) {
        uint64_t ta = now_ns();
        double tsec = t0 ? (ta - t0) / 1e9 : 0;
        if (!in_checked && t0 && tsec >= seconds) {
            /* the end of the timed phase: the last flip lands, the guard rows are read */
            el = tsec;
            if ((rc = flip_events(k.card, 1000))) { fail = "no flip event within 1 s"; break; }
            for (int i = 0; i < NBUF; i++) guard_hit += kms_buf_guard_hits(&k, &b[i]);
            in_checked = 1, have_next = 0; /* the frame prepared ahead is not used */
            continue;
        }
        if (in_checked && n_checked_done >= checked) break;
        float t = in_checked ? 10.0f * (float)n_checked_done / (float)checked + (fault == 1 ? 0.05f : 0.0f) : (float)tsec;
        int bi = back_buffer();
        if (bi < 0) { rc = -EBUSY; fail = "no free buffer"; break; }
        kms_buf* back = &b[bi];
        long ncls[3];
        double classify_ms;
        if (in_checked) memset(back->map, 0xEE, (size_t)W * H * 4); /* not on the panel: the back buffer */
        /* The timed phase is a pipeline: while a job renders, the CPU prepares the next
         * frame (scene time: now) in the other set of buffers. The first frame and the
         * checked frames are prepared here, before their job. */
        double setup_ms = 0;
        if (have_next) {
            memcpy(ncls, ncls_next, sizeof ncls);
        } else {
            uint64_t ts = now_ns();
            if ((rc = frame_setup(t, in_checked && n_checked_done > 0, batches, &unif[cur], &pool[cur], back->gpu, ncls, &classify_ms))) { fail = "region sort"; break; }
            setup_ms = (now_ns() - ts) / 1e6;
        }
        ((uint32_t*)unif[cur].map)[0] = back->gpu; /* the buffer is chosen when the job starts */
        uint32_t hh[4] = {code.handle, unif[cur].handle, pool[cur].handle, back->v3d_handle};
        hs_v3d_dispatch_desc d = {prog, unif[cur].gpu, batches, hh, 4, 0};
        uint64_t tb = now_ns();
        rc = hs_v3d_dispatch(&dev, &d);
        have_next = 0;
        if (!rc && !in_checked && t0) {
            uint64_t ts = now_ns();
            if ((rc = frame_setup((float)((ts - t0) / 1e9), 0, batches, &unif[cur ^ 1], &pool[cur ^ 1], 0, ncls_next, &classify_ms))) { fail = "region sort"; break; }
            setup_ms = (now_ns() - ts) / 1e6, have_next = 1, cur ^= 1;
        }
        if (!rc) rc = hs_v3d_bo_wait(&dev, &code, 2000000000ull);
        if (rc) { fail = "compute job"; break; }
        uint64_t tc = now_ns();
        if (front < 0) {
            if ((rc = kms_show_first(&k, back))) { fail = "set CRTC"; break; }
            front = bi;
            first_frame_ms = (now_ns() - t_start) / 1e6;
            t0 = now_ns(), t_prev = 0;
            continue; /* the clock starts with the first frame on the panel; that frame is not counted */
        }
        if (!in_checked) {
            put(&cpu, setup_ms), put(&job, (tc - tb) / 1e6);
            if (t_prev) put(&loop, (ta - t_prev) / 1e6);
            t_prev = ta;
            if (!rendered || ncls[0] < long_min) long_min = ncls[0];
            if (!rendered || ncls[0] > long_max) long_max = ncls[0];
            rendered++;
        }
        int wait = in_checked || pacing == 1, now_flip = !in_checked && pacing == 2;
        /* pacing 2: the previous flip's event is owed before the next flip is accepted; that wait is timed */
        uint64_t tw = now_ns();
        if ((rc = flip_events(k.card, wait || now_flip ? 1000 : 0))) { fail = "no flip event within 1 s"; break; }
        if (now_flip) put(&fwait, (now_ns() - tw) / 1e6);
        if (flip_pending < 0) {
            if (drmModePageFlip(k.card, k.crtc_id, back->fb, DRM_MODE_PAGE_FLIP_EVENT | (now_flip ? DRM_MODE_PAGE_FLIP_ASYNC : 0), NULL)) {
                rc = -errno;
                fail = now_flip ? "immediate (async) page flip" : "page flip";
                break;
            }
            flip_pending = bi;
            if (!in_checked) flips_queued++;
        } else {
            dropped++; /* a flip is still pending: this frame is not shown */
        }
        if (wait && (rc = flip_events(k.card, 1000))) { fail = "no flip event within 1 s"; break; }
        if (!in_checked && flip_ns != last_flip) {
            if (last_flip) put(&interval, (flip_ns - last_flip) / 1e6);
            last_flip = flip_ns;
        }
        if (in_checked) {
            /* on the panel now; compare the memory with the reference */
            scene_at(t - (fault == 1 ? 0.05f : 0.0f), batches);
            reference(batches, img, kind);
            long wrong = 0;
            uint64_t hash = 1469598103934665603ull;
            for (size_t i = 0; i < (size_t)W * H; i++) {
                hash = (hash ^ back->map[i]) * 1099511628211ull;
                unwritten += back->map[i] == 0xEEEEEEEE;
                wrong += back->map[i] != img[i];
            }
            uint64_t gh = kms_buf_guard_hits(&k, back);
            guard_hit += gh;
            wrong_total += wrong;
            if (wrong > wrong_max) wrong_max = wrong;
            if (wrong > W * H / 10000 || gh) frames_failed++;
            size_t hl = strlen(hashes);
            if (hl + 20 < sizeof hashes) snprintf(hashes + hl, sizeof hashes - hl, "%s\"%016llx\"", hl ? "," : "", (unsigned long long)hash);
            checked_buf[bi]++, frames_checked++, n_checked_done++;
        }
        if (fault == 2 && t0 && now_ns() - t0 > 1000000000ull) raise(SIGINT);
    }
    if (!el && t0) el = (now_ns() - t0) / 1e9;
    if (!rc) flip_events(k.card, 200);
    /* what the display controller composes from the framebuffer on the panel, against that buffer's memory */
    if (!rc && !stop_requested && front >= 0 && checked > 0) {
        scan_rc = capture_scanout(k.card, k.res, b[front].fb, W, H, &k.mode, cap_px);
        for (size_t i = 0; !scan_rc && i < (size_t)W * H; i++) scan_wrong += (cap_px[i] & 0x00FFFFFF) != (b[front].map[i] & 0x00FFFFFF);
    }

out:
    restore = kms_restore(&k);
    for (int i = 0; i < NBUF; i++) kms_buf_free(&k, &dev, &b[i]);
    kms_close(&k);
    for (int i = 0; i < 2; i++) {
        if (pool[i].handle) hs_v3d_bo_free(&dev, &pool[i]);
        if (unif[i].handle) hs_v3d_bo_free(&dev, &unif[i]);
    }
    if (code.handle) hs_v3d_bo_free(&dev, &code);
    if (dev.fd >= 0) hs_v3d_close(&dev);
    series* all[5] = {&job, &cpu, &loop, &interval, &fwait};
    double refresh_ms = 1000.0 / (k.mode.vrefresh ? k.mode.vrefresh : 60);
    for (int i = 0; i < interval.n; i++) late += interval.v[i] > 1.5 * refresh_ms;
    for (int i = 0; i < 5; i++)
        if (all[i]->v) qsort(all[i]->v, (size_t)all[i]->n, sizeof(double), cmpd);
    printf("{\"present\":\"test_08 scene by QPU compute to KMS, no Mesa\",\"mode\":\"%ux%u@%u\",\"batches\":%u,\"pacing\":\"%s\",\"cull\":%d,"
           "\"fault\":%d,\"kernel_fault\":%d,\"rc\":%d,\"failed_at\":\"%s\",\"stopped_by_signal\":%s,"
           "\"timed\":{\"seconds\":%.3f,\"frames_rendered\":%ld,\"frames_rendered_per_second\":%.2f,\"flips\":%ld,\"flips_per_second\":%.2f,"
           "\"frames_dropped\":%ld,\"job_submit_to_done_ms\":{\"p50\":%.3f,\"p99\":%.3f,\"max\":%.3f},\"cpu_setup_ms_during_the_previous_job\":{\"p50\":%.3f,\"max\":%.3f},"
           "\"frame_to_frame_ms\":{\"p50\":%.3f,\"p99\":%.3f,\"max\":%.3f},\"flip_interval_ms\":{\"p50\":%.3f,\"p99\":%.3f,\"max\":%.3f},"
           "\"wait_for_previous_flip_ms\":{\"p50\":%.3f,\"p99\":%.3f,\"max\":%.3f},\"driver_offers_async_flips\":%s,\"flip_intervals_over_1.5_refreshes\":%ld,\"long_path_regions\":{\"min\":%ld,\"max\":%ld}},"
           "\"checked\":{\"frames\":%d,\"frames_failed\":%d,\"in_buffer\":[%d,%d,%d],\"pixels_different_from_reference\":%ld,\"most_in_a_frame\":%ld,"
           "\"unwritten\":%ld,\"image_hashes\":[%s]},\"guard_words_overwritten\":%llu,\"scanout_capture\":{\"rc\":%d,\"pixels_different_from_memory\":%llu},"
           "\"process_start_to_first_frame_on_crtc_ms\":%.2f,\"plane_changes_at_takeover\":%d,\"display_restored_rc\":%d}\n",
           k.w, k.h, k.mode.vrefresh, batches, pacing == 2 ? "every frame flipped at once (async), none dropped" : pacing ? "every flip waited for" : "free: flip when none is pending, else drop", opt_cull, fault, opt_fault,
           rc, fail ? fail : "", stop_requested ? "true" : "false", el, rendered, el > 0 ? rendered / el : 0, flips_queued, el > 0 ? flips_queued / el : 0,
           dropped, at(&job, 1, 2), at(&job, 99, 100), at(&job, 1, 1), at(&cpu, 1, 2), at(&cpu, 1, 1), at(&loop, 1, 2), at(&loop, 99, 100), at(&loop, 1, 1),
           at(&interval, 1, 2), at(&interval, 99, 100), at(&interval, 1, 1), at(&fwait, 1, 2), at(&fwait, 99, 100), at(&fwait, 1, 1),
           async_cap ? "true" : "false", late, long_min, long_max, frames_checked, frames_failed, checked_buf[0],
           checked_buf[1], checked_buf[2], wrong_total, wrong_max, unwritten, hashes, (unsigned long long)guard_hit, scan_rc,
           (unsigned long long)scan_wrong, first_frame_ms, k.plane_changes, restore);
    free(img), free(cap_px), free(kind), free(job.v), free(cpu.v), free(loop.v), free(interval.v), free(fwait.v), free(regions);
    /* a complete run checked every frame it was asked to, in every buffer when there were at least three */
    int complete = !rc && !stop_requested;
    int unchecked = complete && (frames_checked != checked || (checked >= NBUF && (!checked_buf[0] || !checked_buf[1] || !checked_buf[2])));
    int scan_bad = scan_rc < 0 || scan_wrong || (scan_rc == 1 && complete && checked > 0);
    if (rc) return 3;
    return frames_failed || unwritten || guard_hit || restore || unchecked || scan_bad;
}
