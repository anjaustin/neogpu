/* P0-3 (docs/audit/HARDWARE_HANDOFF_P0.md): what the vc4 HVS exposes as KMS
 * planes on this kernel. Read-only: queries only, no commits, so it works
 * while another process owns the display.
 * Usage: tests/p0_3_plane_inventory [/dev/dri/cardN]   (default: the vc4 card)
 * Output: one JSON object. */
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

static int open_vc4(const char* path) {
    if (path) return open(path, O_RDWR | O_CLOEXEC);
    char p[32];
    for (int i = 0; i < 8; i++) {
        snprintf(p, sizeof p, "/dev/dri/card%d", i);
        int fd = open(p, O_RDWR | O_CLOEXEC);
        if (fd < 0) continue;
        drmVersion* v = drmGetVersion(fd);
        int ok = v && !strcmp(v->name, "vc4");
        drmFreeVersion(v);
        if (ok) return fd;
        close(fd);
    }
    return -1;
}

int main(int argc, char** argv) {
    int fd = open_vc4(argc > 1 ? argv[1] : NULL);
    if (fd < 0) { printf("{\"error\":\"no vc4 card\"}\n"); return 1; }
    int universal = drmSetClientCap(fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1) == 0;
    int atomic = drmSetClientCap(fd, DRM_CLIENT_CAP_ATOMIC, 1) == 0;
    drmModeRes* res = drmModeGetResources(fd);
    printf("{\"universal\":%d,\"atomic\":%d,\"crtcs\":[", universal, atomic);
    for (int i = 0; res && i < res->count_crtcs; i++) {
        drmModeCrtc* c = drmModeGetCrtc(fd, res->crtcs[i]);
        printf("%s{\"id\":%u,\"active\":%d,\"w\":%u,\"h\":%u,\"fb\":%u}", i ? "," : "", res->crtcs[i],
               c ? c->mode_valid : 0, c ? c->mode.hdisplay : 0, c ? c->mode.vdisplay : 0, c ? c->buffer_id : 0);
        drmModeFreeCrtc(c);
    }
    printf("],\"planes\":[");
    drmModePlaneRes* pr = drmModeGetPlaneResources(fd);
    for (unsigned i = 0; pr && i < pr->count_planes; i++) {
        drmModePlane* pl = drmModeGetPlane(fd, pr->planes[i]);
        if (!pl) continue;
        printf("%s\n{\"id\":%u,\"possible_crtcs\":%u,\"crtc\":%u,\"fb\":%u,\"formats\":[", i ? "," : "", pl->plane_id,
               pl->possible_crtcs, pl->crtc_id, pl->fb_id);
        for (unsigned f = 0; f < pl->count_formats; f++) {
            unsigned x = pl->formats[f];
            printf("%s\"%c%c%c%c\"", f ? "," : "", x & 0xff, (x >> 8) & 0xff, (x >> 16) & 0xff, (x >> 24) & 0xff);
        }
        printf("],\"props\":{");
        drmModeObjectProperties* op = drmModeObjectGetProperties(fd, pl->plane_id, DRM_MODE_OBJECT_PLANE);
        for (unsigned p = 0; op && p < op->count_props; p++) {
            drmModePropertyRes* pp = drmModeGetProperty(fd, op->props[p]);
            if (!pp) continue;
            printf("%s\"%s\":{\"value\":%llu", p ? "," : "", pp->name, (unsigned long long)op->prop_values[p]);
            if (drm_property_type_is(pp, DRM_MODE_PROP_ENUM) || drm_property_type_is(pp, DRM_MODE_PROP_BITMASK)) {
                printf(",\"enums\":[");
                for (int e = 0; e < pp->count_enums; e++)
                    printf("%s\"%s=%llu\"", e ? "," : "", pp->enums[e].name, (unsigned long long)pp->enums[e].value);
                printf("]");
            } else if (drm_property_type_is(pp, DRM_MODE_PROP_RANGE) && pp->count_values == 2) {
                printf(",\"range\":[%llu,%llu]", (unsigned long long)pp->values[0], (unsigned long long)pp->values[1]);
            }
            printf(",\"immutable\":%d}", (pp->flags & DRM_MODE_PROP_IMMUTABLE) ? 1 : 0);
            drmModeFreeProperty(pp);
        }
        drmModeFreeObjectProperties(op);
        printf("}}");
        drmModeFreePlane(pl);
    }
    printf("]}\n");
    drmModeFreePlaneResources(pr);
    drmModeFreeResources(res);
    close(fd);
    return 0;
}
