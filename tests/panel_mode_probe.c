/* Diagnostic: enumerate DRM connectors with their names and types.
 * Diagnostic only; not part of production.
 *
 * Build: gcc -O2 -o tests/panel_mode_probe tests/panel_mode_probe.c -ldrm
 */
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

static const char* connector_name(int fd, uint32_t id) {
    static char buffer[64];
    drmModeObjectProperties* props = drmModeObjectGetProperties(fd, id, DRM_MODE_OBJECT_CONNECTOR);
    if (!props) return "?";
    buffer[0] = 0;
    for (uint32_t i = 0; i < props->count_props; i++) {
        drmModePropertyRes* prop = drmModeGetProperty(fd, props->props[i]);
        if (!prop) continue;
        if (!strcmp(prop->name, "name") && prop->prop_type == DRM_MODE_PROP_STRING &&
            prop->count_values && props->prop_values[i]) {
            snprintf(buffer, sizeof(buffer), "%s", (const char*)props->prop_values[i]);
        }
        drmModeFreeProperty(prop);
    }
    drmModeFreeObjectProperties(props);
    return buffer;
}

int main(void) {
    printf("header constants: DSI=%d DPI=%d WRITEBACK=%d HDMI_A=%d\n",
           DRM_MODE_CONNECTOR_DSI, DRM_MODE_CONNECTOR_DPI,
           DRM_MODE_CONNECTOR_WRITEBACK, DRM_MODE_CONNECTOR_HDMI_A);
    const char* nodes[] = {"/dev/dri/card1", "/dev/dri/card0"};
    for (size_t n = 0; n < sizeof(nodes)/sizeof(*nodes); n++) {
        int fd = open(nodes[n], O_RDWR | O_CLOEXEC);
        if (fd < 0) { printf("%s: open failed\n", nodes[n]); continue; }
        printf("%s:\n", nodes[n]);
        drmModeRes* res = drmModeGetResources(fd);
        if (!res) { printf("  no resources\n"); close(fd); continue; }
        printf("  crtcs=%d connectors=%d encoders=%d\n", res->count_crtcs,
               res->count_connectors, res->count_encoders);
        for (int i = 0; i < res->count_connectors; i++) {
            drmModeConnector* c = drmModeGetConnector(fd, res->connectors[i]);
            if (!c) continue;
            printf("  id=%u name=%-16s type=%u status=%d modes=%u encoders=%u\n",
                   c->connector_id, connector_name(fd, c->connector_id),
                   c->connector_type, c->connection, c->count_modes, c->count_encoders);
            for (int m = 0; m < c->count_modes; m++)
                printf("     mode %dx%d vrefresh=%u clock=%u\n", c->modes[m].hdisplay,
                       c->modes[m].vdisplay, c->modes[m].vrefresh, c->modes[m].clock);
            drmModeFreeConnector(c);
        }
        drmModeFreeResources(res);
        close(fd);
    }
    return 0;
}
