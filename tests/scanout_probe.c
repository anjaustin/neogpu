/* Diagnostic: observe what the display controller is actually scanning out.
 *
 * Samples the CRTC's current framebuffer id for a fixed duration and, at a
 * lower rate, reads the centre pixel of that framebuffer through a dma-buf
 * mapping. This is evidence from outside the rendering process: a frame only
 * counts as presented if the hardware is scanning out its buffer.
 *
 * Requires root (CAP_SYS_ADMIN) to obtain framebuffer handles while another
 * process is DRM master. Drops the master status the first opener of a
 * primary node receives, and never modifies display state.
 *
 * Usage: sudo tests/scanout_probe [duration_ms] [pixel_every_ms]
 * Output: one JSON object on stdout.
 */
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#define MAX_DISTINCT 64
#define MAX_PIXELS   512

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Find the CRTC currently driving a connected connector (DSI preferred). */
static uint32_t active_crtc(int fd) {
    drmModeRes* res = drmModeGetResources(fd);
    if (!res) return 0;
    uint32_t best = 0;
    int best_score = -1;
    for (int i = 0; i < res->count_connectors; i++) {
        drmModeConnector* c = drmModeGetConnectorCurrent(fd, res->connectors[i]);
        if (!c) continue;
        if (c->connection == DRM_MODE_CONNECTED && c->encoder_id) {
            drmModeEncoder* e = drmModeGetEncoder(fd, c->encoder_id);
            if (e && e->crtc_id) {
                int score = c->connector_type == DRM_MODE_CONNECTOR_DSI ? 1 : 0;
                if (score > best_score) { best_score = score; best = e->crtc_id; }
            }
            if (e) drmModeFreeEncoder(e);
        }
        drmModeFreeConnector(c);
    }
    drmModeFreeResources(res);
    return best;
}

/* Centre pixel of a framebuffer as 0xRRGGBB, or -1 if it cannot be mapped. */
static long centre_pixel(int fd, uint32_t fb_id, uint64_t* modifier_out) {
    drmModeFB2* fb = drmModeGetFB2(fd, fb_id);
    if (!fb) return -1;
    long pixel = -1;
    *modifier_out = fb->modifier;
    if (fb->handles[0]) {
        int prime = -1;
        if (drmPrimeHandleToFD(fd, fb->handles[0], DRM_CLOEXEC, &prime) == 0) {
            size_t size = (size_t)fb->pitches[0] * fb->height + fb->offsets[0];
            uint8_t* map = mmap(NULL, size, PROT_READ, MAP_SHARED, prime, 0);
            if (map != MAP_FAILED) {
                size_t at = fb->offsets[0] + (size_t)(fb->height / 2) * fb->pitches[0] + (size_t)(fb->width / 2) * 4;
                uint32_t value;
                memcpy(&value, map + at, sizeof(value));
                pixel = (long)(value & 0xFFFFFFu); /* XRGB8888 little-endian */
                munmap(map, size);
            }
            close(prime);
        }
        drmCloseBufferHandle(fd, fb->handles[0]);
    }
    drmModeFreeFB2(fb);
    return pixel;
}

int main(int argc, char** argv) {
    int duration = argc > 1 ? atoi(argv[1]) : 3000;
    int pixel_every = argc > 2 ? atoi(argv[2]) : 50;
    if (duration <= 0 || duration > 600000 || pixel_every <= 0) {
        fprintf(stderr, "usage: %s [duration_ms 1..600000] [pixel_every_ms >0]\n", argv[0]);
        return 2;
    }
    int fd = open("/dev/dri/card1", O_RDWR | O_CLOEXEC);
    if (fd < 0) fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    if (fd < 0) { fprintf(stderr, "open: %s\n", strerror(errno)); return 1; }
    /* The first opener of a primary node becomes master automatically; an
     * observer must never hold it, or the program under test cannot take it. */
    if (drmIsMaster(fd)) drmDropMaster(fd);

    uint32_t distinct[MAX_DISTINCT];
    unsigned distinct_count = 0, samples = 0, changes = 0, no_crtc = 0;
    uint32_t last_fb = 0, crtc = 0;
    double t_pixels[MAX_PIXELS];
    long pixels[MAX_PIXELS];
    uint32_t pixel_fbs[MAX_PIXELS];
    unsigned pixel_count = 0;
    uint64_t modifier = 0;

    double start = now_ms(), next_pixel = start;
    while (now_ms() - start < duration) {
        if (!crtc) crtc = active_crtc(fd);
        drmModeCrtc* c = crtc ? drmModeGetCrtc(fd, crtc) : NULL;
        uint32_t fb = c ? c->buffer_id : 0;
        if (c) drmModeFreeCrtc(c);
        if (!c) { no_crtc++; crtc = 0; }
        samples++;
        if (fb && fb != last_fb) {
            if (last_fb) changes++;
            bool seen = false;
            for (unsigned i = 0; i < distinct_count; i++) seen |= distinct[i] == fb;
            if (!seen && distinct_count < MAX_DISTINCT) distinct[distinct_count++] = fb;
            last_fb = fb;
        }
        if (fb && now_ms() >= next_pixel && pixel_count < MAX_PIXELS) {
            t_pixels[pixel_count] = now_ms() - start;
            pixel_fbs[pixel_count] = fb;
            pixels[pixel_count++] = centre_pixel(fd, fb, &modifier);
            next_pixel += pixel_every;
        }
        usleep(500);
    }
    double elapsed = now_ms() - start;

    printf("{\"crtc\":%u,\"duration_ms\":%.1f,\"samples\":%u,\"no_crtc_samples\":%u,"
           "\"fb_changes\":%u,\"changes_per_s\":%.2f,\"distinct_fbs\":%u,\"modifier\":\"0x%llx\",\"pixels\":[",
           crtc, elapsed, samples, no_crtc, changes, changes * 1000.0 / elapsed, distinct_count,
           (unsigned long long)modifier);
    for (unsigned i = 0; i < pixel_count; i++) {
        if (pixels[i] < 0) printf("%s[%.1f,%u,null]", i ? "," : "", t_pixels[i], pixel_fbs[i]);
        else printf("%s[%.1f,%u,\"%06lx\"]", i ? "," : "", t_pixels[i], pixel_fbs[i], pixels[i]);
    }
    printf("]}\n");
    close(fd);
    return 0;
}
