/* Diagnostic: how long before a vblank must a page flip of an already
 * finished buffer be queued to land on that vblank?
 *
 * Each trial: clear (cheap) + glFinish so the buffer is complete, wait for
 * the previous flip to land, sleep until <offset> ms after it, then present
 * with vsync on (queues the flip and waits for it). A flip that lands one
 * refresh later counts as made; two or more as missed.
 *
 * Usage: sudo tests/flip_deadline_probe [trials_per_offset]
 * Output: one line per offset (0.0 .. 16.0 ms in 0.5 ms steps). */
#include "hs_graphics.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void sleep_until(double t) {
    double left;
    while ((left = t - now_ms()) > 0.3) {
        struct timespec ts = {0, (long)((left - 0.25) * 1e6)};
        nanosleep(&ts, NULL);
    }
    while (now_ms() < t) {}
}

int main(int argc, char** argv) {
    int trials = argc > 1 ? atoi(argv[1]) : 40;
    static HSGraphics gfx;
    if (hs_graphics_init(&gfx) != 0) return 1;
    gfx.vsync = true;
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    hs_graphics_present(&gfx); /* takeover */
    glClear(GL_COLOR_BUFFER_BIT);
    hs_graphics_present(&gfx); /* first flip: sequence starts */
    for (int step = 0; step <= 32; step++) {
        double offset = step * 0.5;
        int made = 0, missed = 0;
        double cpu = 0;
        for (int t = 0; t < trials; t++) {
            glClearColor((t & 1) ? 1.0f : 0.0f, offset / 16.0f, 0.3f, 1);
            glClear(GL_COLOR_BUFFER_BIT);
            glFinish();
            /* Re-sync: one paced flip, then measure from when it returns. */
            hs_graphics_present(&gfx);
            u32 seq0 = gfx.last_flip_sequence;
            double v = now_ms();
            glClear(GL_COLOR_BUFFER_BIT);
            glFinish();
            sleep_until(v + offset);
            double q0 = now_ms();
            hs_graphics_present(&gfx);
            cpu += now_ms() - q0;
            u32 gap = gfx.last_flip_sequence - seq0;
            if (gap == 1) made++; else missed++;
        }
        printf("offset_ms=%.1f made=%d missed=%d\n", offset, made, missed);
        fflush(stdout);
    }
    hs_graphics_finish(&gfx);
    return 0;
}
