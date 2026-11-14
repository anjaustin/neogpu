/* p0_6_shade_probe: the whole test_08 scene as a hand-assembled QPU compute
 * kernel: primary ray, shading, shadow rays and one reflection bounce,
 * against a CPU reference of the same arithmetic.
 * (P0-6 plan steps 4 and 5, docs/audit/P0_6_PLAN.md.) The kernel, the
 * reference and the region sort are in tools/p0_6_scene.h.
 *
 * Per 16 pixels: a ray, then PASS (trace the six spheres and the ground from
 * an origin along a direction; shade the nearest hit, with a shadow test
 * against the six spheres), then the reflected ray and PASS again, then the
 * mix and the 8-bit pack. Within a path every lane runs everything and the
 * results are selected at the end.
 *
 * Work that is skipped (plan step 5; P06_CULL, a bit mask, default 7):
 *   1  a row above the horizon is filled with its sky colour, no rays;
 *   2  the second shade is skipped when all 16 reflected rays hit nothing;
 *   4  the CPU sorts the 16-pixel regions below the horizon each frame
 *      (classify()); ground that no sphere covers or is reflected in takes
 *      a short path.
 * Rows are dealt to the batches in turn (batch b renders rows b,
 * b + batches, ...), or the cuts would not shorten the frame.
 *
 * Constants come from a second uniform stream (unifa), rewound once per
 * pass of the inner loop; they ride on other instructions as signals, to r5
 * or straight into a register (measured: tools/p0_6_unifa_probe; at least 3
 * other instructions after the unifa write). Rules kept here: nothing lives
 * in r0..r3 or r5 across the thread switch; special-function results go to
 * a register-file entry.
 *
 * Where it differs from the GLSL: the shadow test avoids the square root
 * (ts > 0.001 is tested as h >= 0, b + eps < 0 and h < (b + eps)^2), the
 * sphere normal is (p - c) / r, and the horizon test on rd.y is left out
 * (no pixel row has |rd.y| < 1e-4 at this size). The reference does the
 * same; how far both are from Mesa's image is a separate measurement.
 *
 * Offscreen. Usage: p0_6_shade_probe [batches=8] [frames=1] [shadows=1] [reflect=1] [ppm_dir]
 *   Start a changed kernel with frames=1: a kernel that hangs costs a GPU
 *   reset per job. (inside the GPU lock, performance governor).
 *   Environment: P06_CULL (above), P06_MARGIN (slack of the region tests),
 *   P06_FAULT (below), P06_BUILD_ONLY, P06_PAD, P06_PAD_KIND.
 * Checks, per frame: every pixel against the reference (a frame fails with
 * more than 0.01 % of its pixels different, any pixel left unwritten, or a
 * guard word changed: 16 rows of them before and after the image). Each
 * frame's line carries a hash of the image, so two runs (with and without
 * the cuts) can be compared exactly: tests/redteam-p0-6/scene_ab.sh.
 * Timing, when frames > 1: a second pass over the same frames with no
 * checking, jobs back to back, the CPU's per-frame work timed separately.
 * P06_FAULT breaks one thing; every one must make the run fail
 * (tests/redteam-p0-6/scene_cases.sh):
 *   1 the second shade is skipped when ANY reflected ray misses
 *   2 the path around the second shade does not move the constant stream
 *   3 ground that may be shadowed skips the shadow test
 *   4 one row too many per batch (the guard rows must show it)
 *   5 6 7 the region sort ignores reflections / covering spheres / shadows
 *   8 rows just below the horizon are treated as sky
 *   9 the short path reads the long path's constants
 *   10 the batches start two rows apart, not one
 *   11 the reflection bound of the region sort ignores the 0.02 by which
 *      the reflected ray starts above the ground (a few pixels in a few
 *      frames: under the reference check's allowance, seen only by the
 *      exact comparison of scene_ab.sh)
 *   12 one pixel is put back to the unwritten mark after the job
 *   13 from the second frame on, sky rows are not stored (the sky does not
 *      change: without the clearing of the image between frames the stale
 *      rows would pass)
 * Output: JSON lines. Exit 1 if a frame fails its check, 2 on bad
 * arguments, 3 when the run itself failed (build, setup, a job).
 */
#define _GNU_SOURCE
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "hs_qpu.h"
#include "hs_v3d.h"
#include "p0_6_scene.h"
static void write_ppm(const char* dir, const char* name, const uint32_t* px) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s.ppm", dir, name);
    FILE* f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    for (int i = 0; i < W * H; i++) {
        unsigned char c[3] = {(unsigned char)(px[i] >> 16), (unsigned char)(px[i] >> 8), (unsigned char)px[i]};
        fwrite(c, 1, 3, f);
    }
    fclose(f);
}
static int chdiff(uint32_t a, uint32_t b) {
    int m = 0;
    for (int s = 0; s < 24; s += 8) {
        int d = (int)(a >> s & 255) - (int)(b >> s & 255);
        if (d < 0) d = -d;
        if (d > m) m = d;
    }
    return m;
}

enum { GUARD_ROWS = 16, GUARD_WORDS = GUARD_ROWS * W, GUARD = 0xA5C3F00D };
static int cmpd(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}
int main(int argc, char** argv) {
    long v[4] = {8, 1, 1, 1};
    for (int i = 0; i < 4 && i + 1 < argc; i++)
        if (number(argv[i + 1], 0, 100000, &v[i])) v[0] = -1;
    uint32_t batches = (uint32_t)v[0];
    int frames = (int)v[1];
    opt_shadows = (int)v[2], opt_reflect = (int)v[3];
    const char* ppm_dir = argc > 5 ? argv[5] : NULL;
    long ev = 0;
    const char* e;
    int bad = v[0] < 0 || (batches != 1 && batches != 2 && batches != 4 && batches != 8) || frames < 1 || opt_shadows > 1 || opt_reflect > 1;
    if ((e = getenv("P06_PAD"))) bad |= number(e, 0, 2000, &ev), opt_pad = (int)ev;
    if ((e = getenv("P06_PAD_KIND"))) bad |= number(e, 0, 6, &ev), opt_pad_kind = (int)ev;
    if ((e = getenv("P06_CULL"))) bad |= number(e, 0, 7, &ev), opt_cull = (int)ev;
    if ((e = getenv("P06_FAULT"))) bad |= number(e, 0, 13, &ev), opt_fault = (int)ev;
    if ((e = getenv("P06_MARGIN"))) {
        char* end;
        opt_margin = strtod(e, &end);
        bad |= end == e || *end || !(opt_margin > -1.0 && opt_margin < 1.0);
    }
    if (bad)
        return fprintf(stderr, "usage: %s [batches 1|2|4|8] [frames 1..100000] [shadows 0|1] [reflect 0|1] [ppm_dir]\n"
                               "  environment: P06_CULL 0..7, P06_MARGIN (-1, 1), P06_FAULT 0..13, P06_PAD 0..2000, P06_PAD_KIND 0..6\n", argv[0]), 2;
    static uint64_t buf[4096];
    hs_qpu_prog p;
    hs_qpu_prog_init(&p, buf, 4096);
    int len = build(&p);
    if (len <= 0) return fprintf(stderr, "build failed: %d at instruction %u\n", len, p.rule_at), 3;
    if (scene_fp_contracted()) return fprintf(stderr, "built without -ffp-contract=off: the reference would not match the kernel\n"), 3;
    if (getenv("P06_BUILD_ONLY"))
        return printf("{\"build_only\":true,\"instructions\":%d,\"per_16_pixels\":%d,\"plain_ground_path_instructions\":%d,\"constants\":%d,"
                      "\"row_constants\":%d}\n", len, n_instr_vec, n_instr_ground, kn, k_vec), 0;
    /* class words: one per region, at most one row of regions per row of a batch */
    size_t unif_bytes = (19 + (size_t)(H / batches + 1) * VECS) * 4;
    hs_v3d dev;
    hs_v3d_bo code, unif, pool, out;
    if (hs_v3d_open(&dev, NULL) || hs_v3d_bo_new(&dev, &code, 65536) || hs_v3d_bo_new(&dev, &unif, unif_bytes + 4096) ||
        hs_v3d_bo_new(&dev, &pool, 16384) || hs_v3d_bo_new(&dev, &out, ((size_t)W * H + 2 * GUARD_WORDS) * 4))
        return fprintf(stderr, "setup failed\n"), 3;
    uint32_t prog = hs_v3d_load(&code, 0, buf, (uint32_t)len);
    if (!prog) return fprintf(stderr, "load failed\n"), 3;
    uint32_t* img = malloc((size_t)W * H * 4);
    uint8_t* kind = malloc((size_t)W * H);
    double* gpu_ms = malloc(sizeof(double) * (size_t)frames), *cpu_ms = malloc(sizeof(double) * (size_t)frames);
    if (!img || !kind || !gpu_ms || !cpu_ms) return fprintf(stderr, "malloc failed\n"), 3;
    uint32_t* all = (uint32_t*)out.map, *gpu = all + GUARD_WORDS;
    uint32_t out_gpu = out.gpu + GUARD_WORDS * 4;
    uint32_t h[4] = {code.handle, unif.handle, pool.handle, out.handle};
    hs_v3d_dispatch_desc d = {prog, unif.gpu, batches, h, 4, 0};
    int failures = 0, rc = 0;
    printf("{\"scene\":\"full: shading%s%s\",\"cull\":%d,\"fault\":%d,\"margin\":%g,\"width\":%d,\"height\":%d,\"batches\":%u,\"program_instructions\":%d,"
           "\"instructions_per_16_pixels\":%d,\"plain_ground_path_instructions\":%d,\"constants\":%d}\n",
           opt_shadows ? ", shadows" : "", opt_reflect ? ", reflection" : "", opt_cull, opt_fault, opt_margin, W, H, batches, len, n_instr_vec,
           n_instr_ground, kn);
    for (int f = 0; f < frames && !rc; f++) {
        float t = frames > 1 ? 10.0f * (float)f / (float)frames : 1.0f;
        long ncls[3];
        double classify_ms;
        if ((rc = frame_setup(t, f > 0, batches, &unif, &pool, out_gpu, ncls, &classify_ms))) {
            fprintf(stderr, "region sort failed: %d\n", rc);
            break;
        }
        memset(gpu, 0xEE, (size_t)W * H * 4);
        for (size_t k = 0; k < GUARD_WORDS; k++) all[k] = GUARD, gpu[(size_t)W * H + k] = GUARD;
        uint64_t t0 = now_ns();
        rc = hs_v3d_dispatch(&dev, &d);
        if (!rc) rc = hs_v3d_bo_wait(&dev, &code, 3000000000ull);
        double ms = (now_ns() - t0) / 1e6;
        if (opt_fault == 12) gpu[(size_t)W * 300 + 123] = 0xEEEEEEEE;
        reference(batches, img, kind);
        long wrong = 0, at_edge = 0, off1 = 0, unwritten = 0, guard_hit = 0, by_hit[3] = {0}, pix[3] = {0};
        int maxd = 0;
        uint64_t hash = 1469598103934665603ull;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                size_t i = (size_t)y * W + (size_t)x;
                int cls = (kind[i] & 7) == 0 ? 0 : (kind[i] & 7) == 7 ? 2 : 1;
                pix[cls]++;
                hash = (hash ^ gpu[i]) * 1099511628211ull;
                unwritten += gpu[i] == 0xEEEEEEEE;
                if (gpu[i] == img[i]) continue; /* the kernel writes 0x00RRGGBB: the top byte is compared too */
                wrong++, by_hit[cls]++;
                int dd = chdiff(gpu[i] & 0xFFFFFF, img[i]);
                off1 += dd <= 1;
                if (dd > maxd) maxd = dd;
                int edge = 0;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++) {
                        int yy = y + dy, xx = x + dx;
                        if (yy >= 0 && yy < H && xx >= 0 && xx < W && kind[(size_t)yy * W + (size_t)xx] != kind[i]) edge = 1;
                    }
                at_edge += edge;
            }
        for (size_t k = 0; k < GUARD_WORDS; k++) guard_hit += (all[k] != GUARD) + (gpu[(size_t)W * H + k] != GUARD);
        /* 1/x and 1/sqrt(x) are within 1 ulp of the reference's: a pixel can flip at an
         * edge (object, shadow or reflection). Allowed: 0.01 % of the frame. */
        int ok = !rc && !guard_hit && !unwritten && wrong <= W * H / 10000;
        printf("{\"frame\":%d,\"t\":%.2f,\"rc\":%d,\"ok\":%s,\"job_ms_after_idle\":%.3f,\"pixels\":%d,\"exact\":%ld,\"mismatching\":%ld,\"unwritten\":%ld,"
               "\"at_an_edge_of_the_hit_map\":%ld,\"off_by_one_step\":%ld,\"largest_channel_difference\":%d,\"guard_words_overwritten\":%ld,"
               "\"gpu_image_hash\":\"%016llx\","
               "\"regions\":{\"long_path\":%ld,\"plain_ground\":%ld,\"ground_maybe_shadowed\":%ld,\"classify_ms\":%.3f},"
               "\"mismatching_by_primary_hit\":{\"sky\":%ld,\"sphere\":%ld,\"ground\":%ld},\"pixels_by_primary_hit\":{\"sky\":%ld,\"sphere\":%ld,\"ground\":%ld}}\n",
               f, (double)t, rc, ok ? "true" : "false", ms, W * H, (long)W * H - wrong, wrong, unwritten, at_edge, off1, maxd, guard_hit,
               (unsigned long long)hash, ncls[0], ncls[1], ncls[2], classify_ms, by_hit[0], by_hit[1], by_hit[2], pix[0], pix[1], pix[2]);
        if (!ok) failures++;
        if (ppm_dir && (f == 0 || !ok)) {
            char name[64];
            snprintf(name, sizeof name, "scene_s%dr%d_f%d_gpu", opt_shadows, opt_reflect, f);
            write_ppm(ppm_dir, name, gpu);
            snprintf(name, sizeof name, "scene_s%dr%d_f%d_cpu", opt_shadows, opt_reflect, f);
            write_ppm(ppm_dir, name, img);
        }
    }
    /* Timing: the same frames again with no checking, jobs back to back. (Timed
     * between reference computations, each job also pays the GPU's ~10 ms
     * wake-up: the reference takes long enough for the GPU to go idle.) Per
     * frame: the CPU's work (scene constants, region sort, uniforms) and the
     * job. The first 8 passes are a warm-up on frame 0 and are not counted. */
    int nt = 0, worst = 0;
    long long_min = 0, long_max = 0;
    for (int i = -8; i < frames && frames > 1 && !rc && !failures; i++) {
        long ncls[3];
        double classify_ms;
        uint64_t t0 = now_ns();
        if ((rc = frame_setup(10.0f * (float)(i < 0 ? 0 : i) / (float)frames, 0, batches, &unif, &pool, out_gpu, ncls, &classify_ms))) break;
        uint64_t t1 = now_ns();
        rc = hs_v3d_dispatch(&dev, &d);
        if (!rc) rc = hs_v3d_bo_wait(&dev, &code, 3000000000ull);
        uint64_t t2 = now_ns();
        if (rc || i < 0) continue;
        cpu_ms[nt] = (t1 - t0) / 1e6, gpu_ms[nt] = (t2 - t1) / 1e6;
        if (gpu_ms[nt] > gpu_ms[worst]) worst = nt;
        if (!nt || ncls[0] < long_min) long_min = ncls[0];
        if (!nt || ncls[0] > long_max) long_max = ncls[0];
        nt++;
    }
    if (frames > 1 && nt == frames) {
        double sum = 0;
        for (int i = 0; i < nt; i++) sum += gpu_ms[i] + cpu_ms[i];
        qsort(gpu_ms, (size_t)nt, sizeof(double), cmpd), qsort(cpu_ms, (size_t)nt, sizeof(double), cmpd);
        printf("{\"scene_summary\":\"full\",\"cull\":%d,\"shadows\":%d,\"reflection\":%d,\"instructions_per_16_pixels\":%d,\"batches\":%u,\"frames\":%d,"
               "\"failures\":%d,\"timed_frames_back_to_back\":%d,\"job_ms\":{\"min\":%.3f,\"median\":%.3f,\"p99\":%.3f,\"max\":%.3f,\"slowest_frame\":%d},"
               "\"cpu_ms_per_frame\":{\"median\":%.3f,\"max\":%.3f},\"mean_ms_per_frame_cpu_and_job\":%.3f,\"frames_per_second_at_the_mean\":%.1f,"
               "\"long_path_regions\":{\"min\":%ld,\"max\":%ld}}\n",
               opt_cull, opt_shadows, opt_reflect, n_instr_vec, batches, frames, failures, nt, gpu_ms[0], gpu_ms[nt / 2], gpu_ms[nt * 99 / 100],
               gpu_ms[nt - 1], worst, cpu_ms[nt / 2], cpu_ms[nt - 1], sum / nt, 1000.0 * nt / sum, long_min, long_max);
    } else if (frames > 1) {
        printf("{\"scene_summary\":\"full\",\"cull\":%d,\"frames\":%d,\"failures\":%d,\"rc\":%d,\"timed_frames_back_to_back\":0}\n", opt_cull, frames, failures, rc);
    }
    free(img), free(kind), free(gpu_ms), free(cpu_ms), free(regions);
    hs_v3d_bo_free(&dev, &out), hs_v3d_bo_free(&dev, &pool), hs_v3d_bo_free(&dev, &unif), hs_v3d_bo_free(&dev, &code);
    hs_v3d_close(&dev);
    return rc ? 3 : failures != 0;
}
