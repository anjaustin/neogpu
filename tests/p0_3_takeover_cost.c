/* P0-3: time hs_graphics_init and the first present (the display takeover)
 * of whichever hs_graphics.h it is built against, so a change to takeover
 * can be compared with the committed one. scaled=0 uses hs_graphics_init.
 * Usage: sudo tests/p0_3_takeover_cost [w h]   Output: one JSON line. */
#define HS_GRAPHICS_SHELVED_SCALED /* P0-3: shelved scaled path */
#include "hs_graphics.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}
int main(int argc, char** argv) {
    static HSGraphics gfx;
    double t0 = now_ms();
    int rc;
#ifdef SCALED
    rc = hs_graphics_init_scaled(&gfx, argc > 2 ? atoi(argv[1]) : 400, argc > 2 ? atoi(argv[2]) : 240);
#else
    (void)argc; (void)argv;
    rc = hs_graphics_init(&gfx);
#endif
    double t1 = now_ms();
    if (rc) { printf("{\"init\":false}\n"); return 1; }
    glClear(GL_COLOR_BUFFER_BIT);
    hs_graphics_present(&gfx);
    double t2 = now_ms();
    printf("{\"init_ms\":%.2f,\"takeover_ms\":%.2f,\"active\":%d,\"errors\":%u}\n", t1 - t0, t2 - t1, gfx.crtc_active, gfx.present_errors);
    hs_graphics_finish(&gfx);
    return 0;
}
