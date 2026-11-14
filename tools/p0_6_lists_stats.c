/* p0_6_lists_stats: how short could per-region sphere lists be? CPU only.
 * (P0-6 plan step 5, item 3.) For every region the region sort sends down
 * the long path, the spheres that any of its pixels really needs: valid hits
 * of the primary ray (P), spheres that shadow the primary hit (S1), valid
 * hits of the reflected ray (R), spheres that shadow that hit (S2). That is
 * the floor for any conservative list. Beside it, the lists the region sort
 * builds (region_lists in tools/p0_6_scene.h): their sizes, and whether any
 * leaves out a sphere a pixel needs (a violation: exit 1).
 * Usage: p0_6_lists_stats [batches=8] [frames=50] [cells=2] [margin=2e-3]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hs_qpu.h"
#include "hs_v3d.h"
#include "p0_6_scene.h"

int main(int argc, char** argv) {
    uint32_t batches = argc > 1 ? (uint32_t)atoi(argv[1]) : 8;
    int frames = argc > 2 ? atoi(argv[2]) : 50;
    if (argc > 3) opt_cells = atoi(argv[3]);
    if (argc > 4) opt_lmargin = atof(argv[4]);
    if (opt_cells < 1 || opt_cells > 64 || frames < 1) return 2;
    long viol[4] = {0}, lsum[4] = {0};
    double lmodel = 0, lists_ms = 0;
    static uint64_t buf[4096];
    hs_qpu_prog p;
    hs_qpu_prog_init(&p, buf, 4096);
    opt_cull = 7; /* one class word per region, no lists in the stream */
    long unused;
    (void)number("0", 0, 0, &unused), (void)frame_setup, (void)scene_unif_bytes, (void)scene_pool_bytes;
    if (build(&p) <= 0 || scene_fp_contracted()) return 3; /* build: sets nothing the statistics need, but keeps every function used */
    uint32_t* img = malloc((size_t)W * H * 4), *mask = malloc((size_t)W * H * 4), *cls = malloc(sizeof(uint32_t) * (H / batches + 1) * VECS);
    uint8_t* kind = malloc((size_t)W * H);
    ref_mask_img = mask;
    long hist[4][7] = {{0}}, nlong = 0, sum[4] = {0}, noshade2 = 0, shape[8] = {0}, missed_class = 0;
    double model = 0;
    for (int f = 0; f < frames; f++) {
        long ncls[3];
        scene_at(10.0f * (float)f / (float)frames, batches);
        if (classify(batches, cls, ncls) < 0) return 3;
        reference(batches, img, kind);
        lists_frame();
        uint32_t rows = S.rows_per_batch, first = H / 2 / batches; /* first region row below the horizon */
        for (uint32_t r = first, n = 0; r < rows; r++)
            for (int v = 0; v < VECS; v++, n++) {
                uint32_t u = 0;
                int sphere_px = 0, ground_px = 0;
                for (uint32_t y = r * batches; y < (r + 1) * batches; y++)
                    for (int x = 16 * v; x < 16 * v + 16; x++) {
                        u |= mask[(size_t)y * W + x];
                        if ((kind[(size_t)y * W + x] & 7) == 7) ground_px++; else sphere_px++;
                    }
                if (cls[n] != 0) { /* the short path: nothing may be needed but, for class 2, shadows of the primary hit */
                    if (u & ~(cls[n] == 2 ? 0xFC0u : 0u)) missed_class++;
                    continue;
                }
                nlong++;
                int c[4];
                for (int k = 0; k < 4; k++) c[k] = __builtin_popcount(u >> (6 * k) & 63), hist[k][c[k]]++, sum[k] += c[k];
                if (!c[2]) noshade2++;
                shape[(sphere_px ? 1 : 0) | (ground_px ? 2 : 0) | (c[2] ? 4 : 0)]++;
                model += 24.0 * (c[0] + c[2]) + 17.0 * (c[1] + c[3]);
                lists l;
                uint64_t t0 = now_ns();
                region_lists(&regions[n], &l);
                lists_ms += (now_ns() - t0) / 1e6;
                uint32_t g[4] = {l.P, l.S1, l.R, l.S2};
                for (int k = 0; k < 4; k++) viol[k] += ((u >> (6 * k) & 63) & ~g[k]) != 0, lsum[k] += __builtin_popcount(g[k]);
                lmodel += 24.0 * (__builtin_popcount(l.P) + __builtin_popcount(l.R)) + 17.0 * (__builtin_popcount(l.S1) + __builtin_popcount(l.S2));
            }
    }
    printf("{\"batches\":%u,\"frames\":%d,\"long_path_regions_per_frame\":%.1f,\"short_path_regions_needing_a_sphere_they_do_not_test\":%ld,\n", batches, frames,
           (double)nlong / frames, missed_class);
    const char* nm[4] = {"primary", "shadow_of_primary_hit", "reflected", "shadow_of_reflected_hit"};
    for (int k = 0; k < 4; k++) {
        printf(" \"%s\":{\"mean\":%.2f,\"regions_by_count_0_to_6\":[", nm[k], (double)sum[k] / (double)nlong);
        for (int i = 0; i < 7; i++) printf("%s%.1f", i ? "," : "", 100.0 * (double)hist[k][i] / (double)nlong);
        printf("]},\n");
    }
    printf(" \"percent_of_regions\":{\"spheres_only\":%.1f,\"ground_only\":%.1f,\"both\":%.1f,\"no_reflected_hit_at_all\":%.1f},\n",
           100.0 * (double)(shape[1] + shape[5]) / (double)nlong, 100.0 * (double)(shape[2] + shape[6]) / (double)nlong,
           100.0 * (double)(shape[3] + shape[7]) / (double)nlong, 100.0 * (double)noshade2 / (double)nlong);
    printf(" \"sphere_test_instructions_per_16_pixels\":{\"now\":%d,\"with_minimal_lists\":%.1f,\"with_the_region_sort_lists\":%.1f},\n", 24 * 12 + 17 * 12,
           model / (double)nlong, lmodel / (double)nlong);
    printf(" \"region_sort_lists\":{\"cells\":%d,\"margin\":%g,\"mean_entries\":[%.2f,%.2f,%.2f,%.2f],\"regions_with_a_needed_sphere_missing\":[%ld,%ld,%ld,%ld],"
           "\"cpu_ms_per_frame\":%.3f}}\n", opt_cells, opt_lmargin, (double)lsum[0] / (double)nlong, (double)lsum[1] / (double)nlong, (double)lsum[2] / (double)nlong,
           (double)lsum[3] / (double)nlong, viol[0], viol[1], viol[2], viol[3], lists_ms / frames);
    return missed_class || viol[0] || viol[1] || viol[2] || viol[3];
}
