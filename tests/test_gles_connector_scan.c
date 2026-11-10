/* Connector selection test for the graphics path.
 *
 * Covers the selection helper and the memory handling on every path:
 *  - the attached display is selected, and it is one we can drive;
 *  - a type filter that matches nothing selects nothing, and mishandles no
 *    connector on the way out;
 *  - the legacy selection loop, kept only as a negative control, is shown to
 *    double free when no connector matches. The defect was real: the loop
 *    released each non-matching connector inside the loop and then released
 *    the last one again after it.
 *
 * Build (plain):
 *   gcc -O1 -g -o tests/test_gles_connector_scan tests/test_gles_connector_scan.c \
 *       -Iinclude -I/usr/include/libdrm -ldrm -lEGL -lGLESv2 -lgbm
 * Build (sanitised):
 *   gcc -O1 -g -fsanitize=address,undefined -o tests/test_gles_connector_scan_asan \
 *       tests/test_gles_connector_scan.c -Iinclude -I/usr/include/libdrm \
 *       -ldrm -lEGL -lGLESv2 -lgbm
 *
 * Run:
 *   ./tests/test_gles_connector_scan            expects a clean exit
 *   ./tests/test_gles_connector_scan --legacy   expects a sanitiser report
 */
#define _GNU_SOURCE
#include "hs_graphics.h"
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

/* The selection loop as it was written before the fix. A connector type that
 * nothing reports makes the loop finish without a match, which is exactly the
 * case the old code got wrong. Kept only to prove the defect was real. */
static void legacy_select(int fd) {
    drmModeRes* resources = drmModeGetResources(fd);
    if (!resources) return;
    drmModeConnector* conn = NULL;
    drmModeEncoder* enc = NULL;
    int crtc_idx = -1;

    for (int i = 0; i < resources->count_connectors; i++) {
        conn = drmModeGetConnector(fd, resources->connectors[i]);
        if (conn && conn->connection == DRM_MODE_CONNECTED && conn->count_modes > 0 &&
            conn->connector_type == (uint32_t)0xDEAD) {
            if (conn->encoder_id) {
                enc = drmModeGetEncoder(fd, conn->encoder_id);
                if (enc && enc->crtc_id) {
                    crtc_idx = -1;
                    for (int j = 0; j < resources->count_crtcs; j++) {
                        if (resources->crtcs[j] == enc->crtc_id) { crtc_idx = j; break; }
                    }
                }
            }
            break;
        }
        drmModeFreeConnector(conn);
    }

    if (conn) drmModeFreeConnector(conn);
    if (enc) drmModeFreeEncoder(enc);
    drmModeFreeResources(resources);
    (void)crtc_idx;   /* the original used it to pick the CRTC; unreachable here */
}

int main(int argc, char** argv) {
    bool legacy = (argc > 1 && !strcmp(argv[1], "--legacy"));

    int fd = open("/dev/dri/card1", O_RDWR | O_CLOEXEC);
    if (fd < 0) fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    REQUIRE(fd >= 0);
    if (drmSetMaster(fd) != 0) fprintf(stderr, "warning: drmSetMaster failed\n");

    if (legacy) {
        printf("{\"record\":\"negative_control\",\"case\":\"legacy_no_match\"}\n");
        fflush(stdout);
        legacy_select(fd);
        printf("{\"record\":\"summary\",\"ok\":true,\"note\":\"legacy control returned without a report\"}\n");
        drmDropMaster(fd);
        close(fd);
        return 0;
    }

    /* The attached display must be selected, whatever its type. */
    HSGraphics gfx;
    memset(&gfx, 0, sizeof(gfx));
    int found = hs_graphics_select_display(fd, 0, &gfx);
    printf("{\"record\":\"selection\",\"case\":\"any_type\",\"found\":%s,\"connector_id\":%u,"
           "\"crtc_id\":%u,\"width\":%u,\"height\":%u}\n",
           found ? "true" : "false", gfx.display_connector_id, gfx.display_crtc_id,
           gfx.display_mode.hdisplay, gfx.display_mode.vdisplay);
    REQUIRE(found);
    REQUIRE(gfx.display_crtc_id != 0);
    REQUIRE(gfx.display_mode.hdisplay > 0 && gfx.display_mode.vdisplay > 0);

    /* A filter that nothing reports must select nothing, and must not
     * mishandle the connectors it looked at. */
    HSGraphics none;
    memset(&none, 0, sizeof(none));
    int not_found = hs_graphics_select_display(fd, (uint32_t)0xDEAD, &none);
    printf("{\"record\":\"selection\",\"case\":\"impossible_type\",\"found\":%s}\n",
           not_found ? "true" : "false");
    REQUIRE(not_found == 0);
    REQUIRE(none.display_found == false);
    REQUIRE(none.display_crtc_id == 0);

    /* A display with no resources must be handled without a crash. */
    int other = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    if (other >= 0) {
        HSGraphics empty;
        memset(&empty, 0, sizeof(empty));
        int empty_found = hs_graphics_select_display(other, 0, &empty);
        printf("{\"record\":\"selection\",\"case\":\"no_resources\",\"found\":%s}\n",
               empty_found ? "true" : "false");
        REQUIRE(empty_found == 0);
        close(other);
    }

    printf("{\"record\":\"summary\",\"ok\":true}\n");
    drmDropMaster(fd);
    close(fd);
    return 0;
}
