#ifndef HS_GRAPHICS_H
#define HS_GRAPHICS_H

#include "hs_core.h"
#include "hs_buffer.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __linux__

#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <poll.h>
#include <time.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <gbm.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>

#define HS_MAX_TEXTURES 16
#define HS_MAX_BUFFERS  16

typedef struct {
    GLuint gl_texture;
    u32    width;
    u32    height;
    GLenum format;
    bool   disposed;
    bool   dirty;
} HSGLTexture;

typedef struct {
    int            drm_fd;
    struct gbm_device* gbm_device;
    struct gbm_surface* gbm_surface;
    EGLDisplay     egl_display;
    EGLContext     egl_context;
    EGLSurface     egl_surface;
    EGLConfig      egl_config;
    
    u32           screen_width;
    u32           screen_height;
    
    HSGLTexture    textures[HS_MAX_TEXTURES];
    u8             textures_active;
    
    bool           initialized;
    bool           vsync;
    int            swap_interval; /* EGL swap interval last applied; -1 = not yet */
    bool           take_display;
    bool           reuse_connector_scan;
    bool           display_found;
    bool           size_defaulted;
    /* hs_graphics_init_scaled: the surface is smaller than the display's
     * mode and the display controller (HVS) scales it to the full panel at
     * scanout. Takeover programs the primary plane with one atomic commit;
     * flips after that are the same legacy page flips as unscaled. */
    bool           scaled_scanout;
    bool           crtc_setup_done;   /* takeover attempted (success or not) */
    bool           crtc_active;       /* our framebuffer is being scanned out */
    bool           flip_pending;      /* a page flip was queued and has not completed */
    bool           closing;           /* finish(): let close() release framebuffers */
    uint32_t       display_connector_id;
    uint32_t       display_encoder_id;
    uint32_t       display_crtc_id;
    drmModeModeInfo display_mode;

    /* KMS presentation. scanout_bo is on screen; pending_bo is queued to
     * replace it at the next vblank. Each locked GBM buffer carries its own
     * DRM framebuffer (created once per buffer, see hs_graphics_fb_for_bo). */
    struct gbm_bo* scanout_bo;
    struct gbm_bo* pending_bo;
    /* vsync off: the newest frame presented while a flip was in flight, kept
     * for the next present that finds the slot free. With it, three buffers
     * are locked; call hs_graphics_flush_present() before swapping the
     * surface yourself. */
    struct gbm_bo* held_bo;
    /* Smoothed time between vsync-off presents. */
    u64            last_present_ns;
    u64            present_interval_ns;
    /* Frames are held when present_interval_ns is at least this many
     * thousandths of a refresh (init sets 350), or at least a quarter refresh
     * within HS_GRAPHICS_HOLD_AFTER_REPEAT flips of a repeated refresh.
     * Holding adds latency: at 4.2-5.9 ms frames (60 Hz) the shown frame's
     * median age was 23.9-26.5 ms held and 19.7-20.6 ms not, at the same
     * frame rate. On an idle CPU, or a busy one with the kworker-rt service,
     * not holding repeated at most 0.34% of refreshes per run up to 5.87 ms,
     * 0.3-1.3% at 6.3 ms and 3% at 6.7 ms. With every CPU busy and no
     * service, or with spiky frame costs, it repeated more than holding;
     * there repeats keep the quarter-refresh rule in force
     * (tests/evidence-hold-cutoff-20261116, docs/audit/HARDWARE_HANDOFF_P0.md). */
    u32            hold_permille;
    u64            repeats_seen;      /* refreshes_repeated at the last present */
    u64            hold_until_flip;   /* frames_presented below which the quarter rule applies */
    u64            last_flip_ns;      /* vblank timestamp of the last completed flip (CLOCK_MONOTONIC) */
    u64            flip_waits;        /* vsync off: presents that waited for a flip about to land */
    /* EGL_KHR_fence_sync; NULL when unsupported (frames then count as unfinished). */
    PFNEGLCREATESYNCKHRPROC     egl_create_sync;
    PFNEGLDESTROYSYNCKHRPROC    egl_destroy_sync;
    PFNEGLCLIENTWAITSYNCKHRPROC egl_client_wait_sync;
    u64            frames_presented;  /* flips (and the takeover) that completed */
    u64            frames_dropped;    /* vsync off: replaced by a newer frame before it could be shown */
    u32            last_flip_sequence; /* vblank counter of the last completed flip */
    u64            refreshes_repeated; /* refreshes between consecutive flips that kept the old frame */
    u32            present_errors;
    int            last_present_errno;
    const char*    present_error;     /* why presentation is off, or NULL */
} HSGraphics;

/* Choose the connector to drive.
 *
 * A candidate is a connected connector that reports at least one mode. Among
 * the candidates, prefer one whose encoder is attached to a CRTC, then one
 * that is a DSI panel, then any. The type filter is not a requirement: any
 * connected connector with a mode can be driven. A require_type of 0 accepts
 * every type; a non-zero value restricts the candidates to that type.
 *
 * Fills the display fields of gfx and returns 1 when a candidate was found.
 * Every connector is released exactly once, including when nothing matches.
 */
static inline int hs_graphics_select_display(int fd, uint32_t require_type, HSGraphics* gfx) {
    drmModeRes* resources = drmModeGetResources(fd);
    if (!resources) return 0;

    int found = 0;
    int best_score = -1;
    uint32_t best_connector = 0, best_encoder = 0, best_crtc = 0;
    drmModeModeInfo best_mode;
    memset(&best_mode, 0, sizeof(best_mode));

    for (int i = 0; i < resources->count_connectors; i++) {
        drmModeConnector* conn = drmModeGetConnector(fd, resources->connectors[i]);
        if (!conn) continue;
        bool usable = conn->connection == DRM_MODE_CONNECTED && conn->count_modes > 0 &&
                      (require_type == 0 || conn->connector_type == require_type);
        if (!usable) { drmModeFreeConnector(conn); continue; }

        uint32_t encoder_id = conn->encoder_id;
        uint32_t crtc_id = 0;
        int score = 0;
        if (encoder_id) {
            drmModeEncoder* enc = drmModeGetEncoder(fd, encoder_id);
            if (enc) {
                if (enc->crtc_id) { crtc_id = enc->crtc_id; score += 2; }
                drmModeFreeEncoder(enc);
            }
        }
        /* An idle connector (no console on it, or a CRTC left disabled by a
         * previous owner) has no active encoder->CRTC link. Any CRTC one of
         * its encoders can drive will do. */
        for (int e = 0; !crtc_id && e < conn->count_encoders; e++) {
            drmModeEncoder* enc = drmModeGetEncoder(fd, conn->encoders[e]);
            if (!enc) continue;
            for (int c = 0; c < resources->count_crtcs && c < 32; c++) {
                if (enc->possible_crtcs & (1u << c)) {
                    crtc_id = resources->crtcs[c];
                    encoder_id = enc->encoder_id;
                    break;
                }
            }
            drmModeFreeEncoder(enc);
        }
        if (conn->connector_type == DRM_MODE_CONNECTOR_DSI) score += 1;

        if (score > best_score) {
            best_score = score;
            best_connector = conn->connector_id;
            best_encoder = encoder_id;
            best_crtc = crtc_id;
            best_mode = conn->modes[0];
            found = 1;
        }
        drmModeFreeConnector(conn);
    }
    drmModeFreeResources(resources);

    if (found) {
        gfx->display_found = true;
        gfx->display_connector_id = best_connector;
        gfx->display_encoder_id = best_encoder;
        gfx->display_crtc_id = best_crtc;
        gfx->display_mode = best_mode;
    }
    return found;
}

/* Initialize the graphics path.
 *
 * A width and height of 0 means "use the attached display's mode". A caller
 * that knows its target size passes it, which also means the display is not
 * taken unless that size is the panel's own mode.
 */
static inline int hs_graphics_init_common(HSGraphics* gfx, u32 width, u32 height, bool scaled);
static inline int hs_graphics_init_sized(HSGraphics* gfx, u32 width, u32 height) {
    return hs_graphics_init_common(gfx, width, height, false);
}

#ifdef HS_GRAPHICS_SHELVED_SCALED
/* SHELVED (P0-3, docs/audit/HARDWARE_HANDOFF_P0.md): not part of the public
 * API. On the panel the stretched image looked pixelated, not like a native
 * render, so this is a quality trade, not a speed-up. Kept, behind this define,
 * only for the P0-3 tests and probes. */
/* Render at width x height and show it on the attached display, scaled to the
 * display's full mode by the display controller (the vc4 HVS) at scanout, at
 * no GPU cost. The GPU's fill work shrinks with the pixel count: the ray-cast
 * scene of test_08 renders 3.9x faster at 400x240 than at 800x480. The image
 * is stretched to the panel (keep the panel's aspect ratio to avoid
 * distortion) and is softer than a native-resolution render. Each dimension
 * must be at most the display's; the size of the display's own mode behaves
 * exactly like hs_graphics_init. Fails (returns -1) without a display or if
 * the size is larger than the display's mode.
 * - Use a width that is a multiple of 8: odd widths measured ~0.17 ms of
 *   extra time per frame (799x479: 1265 vs 1603 FPS on a trivial frame).
 * - The HVS's default filter is a 4-tap cubic-like kernel; it places the image
 *   ~0.38 output px right and down at 2x, and a 1-px border looks heavier at
 *   the left/top than at the right/bottom (tests/evidence-p0-3-20261116).
 * - Scanout is free for the display but reads memory: a smaller render size
 *   reads less than native. */
static inline int hs_graphics_init_scaled(HSGraphics* gfx, u32 width, u32 height) {
    if (!width || !height) return -1;
    return hs_graphics_init_common(gfx, width, height, true);
}
#endif /* HS_GRAPHICS_SHELVED_SCALED */

static inline int hs_graphics_init_common(HSGraphics* gfx, u32 width, u32 height, bool scaled) {
    memset(gfx, 0, sizeof(HSGraphics));
    gfx->drm_fd = -1;
    gfx->egl_display = EGL_NO_DISPLAY;
    gfx->egl_context = EGL_NO_CONTEXT;
    gfx->egl_surface = EGL_NO_SURFACE;
    /* Uncapped by default: present never waits for the display, so frame
     * rates measure what the system can render. The panel shows the newest
     * completed flip at each refresh; frames it cannot show are dropped, not
     * waited for. Set vsync = true to pace to the display instead. */
    gfx->vsync = false;
    gfx->swap_interval = -1;
    gfx->hold_permille = 350;
    /* Taking the CRTC away from whatever currently owns it is what puts our
     * frames on the panel; it costs a full modeset on the first present. At
     * close() the kernel hands the CRTC back to the console (fbcon); an
     * explicit restore was measured and rejected (docs/audit/GLES_SHUTDOWN_PLAN.md).
     * Hosts that render without presenting can clear this to skip display
     * ownership entirely; the default keeps it. */
    gfx->take_display = true;
    /* The connector scan is slow on this class of panel, and the display setup
     * needs the same information, so keep it from being done twice. */
    gfx->reuse_connector_scan = true;

    gfx->drm_fd = open("/dev/dri/card1", O_RDWR | O_CLOEXEC);
    if (gfx->drm_fd < 0) {
        gfx->drm_fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    }
    if (gfx->drm_fd < 0) {
        return -1;
    }

    if (drmSetMaster(gfx->drm_fd) != 0) {
        goto fail;
    }

    /* Pick the connector to drive: any connected connector that reports a
     * mode, not only a DSI panel, so the path works on whatever display is
     * attached. The DSI panel is preferred among equals, not required. */
    hs_graphics_select_display(gfx->drm_fd, 0, gfx);
    u32 detected_width = 0, detected_height = 0;
    if (gfx->display_found) {
        detected_width = gfx->display_mode.hdisplay;
        detected_height = gfx->display_mode.vdisplay;
    }

    if (scaled) {
        if (!gfx->display_found || width > detected_width || height > detected_height) goto fail;
        gfx->screen_width = width;
        gfx->screen_height = height;
        gfx->scaled_scanout = width != detected_width || height != detected_height;
    } else if (width && height) {
        /* The caller knows the target size. Present on the attached panel
         * only when the requested size is that panel's own mode; otherwise
         * this is an offscreen target and the display is not ours to take. */
        gfx->screen_width = width;
        gfx->screen_height = height;
        if (!gfx->display_found || width != detected_width || height != detected_height) {
            gfx->take_display = false;
        }
    } else if (!gfx->display_found) {
        /* No display attached. Keep the legacy default so a host still gets a
         * surface, and record that the size was defaulted rather than
         * detected so the fallback is visible instead of silent. */
        gfx->screen_width = 800;
        gfx->screen_height = 480;
        gfx->size_defaulted = true;
    } else {
        gfx->screen_width = detected_width;
        gfx->screen_height = detected_height;
    }
    
    if (gfx->screen_width == 0) {
        gfx->screen_width = 800;
        gfx->screen_height = 480;
        gfx->size_defaulted = true;
    }

    gfx->gbm_device = gbm_create_device(gfx->drm_fd);
    if (!gfx->gbm_device) {
        goto fail;
    }
    
    gfx->gbm_surface = gbm_surface_create(
        gfx->gbm_device,
        gfx->screen_width,
        gfx->screen_height,
        GBM_FORMAT_XRGB8888,
        GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING
    );
    if (!gfx->gbm_surface) {
        goto fail;
    }
    
    EGLint egl_major, egl_minor;
    const char* client_extensions = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
    
    typedef EGLDisplay (EGLAPIENTRY *PFNEGLGETPLATFORMDISPLAYEXTPROC)(EGLenum, void*, const EGLint*);
    PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    
    if (eglGetPlatformDisplayEXT && client_extensions && strstr(client_extensions, "EGL_KHR_platform_gbm")) {
        gfx->egl_display = eglGetPlatformDisplayEXT(EGL_PLATFORM_GBM_KHR, gfx->gbm_device, NULL);
    } else {
        gfx->egl_display = eglGetDisplay((EGLNativeDisplayType)gfx->gbm_device);
    }
    
    if (gfx->egl_display == EGL_NO_DISPLAY) {
        goto fail;
    }
    
    if (!eglInitialize(gfx->egl_display, &egl_major, &egl_minor)) {
        gfx->egl_display = EGL_NO_DISPLAY; /* nothing to terminate */
        goto fail;
    }
    
    EGLint config_attribs[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 24,
        EGL_STENCIL_SIZE, 8,
        EGL_SAMPLE_BUFFERS, 0,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES_BIT,
        EGL_NONE
    };
    
    EGLint num_configs;
    if (!eglChooseConfig(gfx->egl_display, config_attribs, &gfx->egl_config, 1, &num_configs) || num_configs == 0) {
        goto fail;
    }
    
    EGLint context_attribs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 3,
        EGL_NONE
    };
    
    gfx->egl_context = eglCreateContext(gfx->egl_display, gfx->egl_config, EGL_NO_CONTEXT, context_attribs);
    if (gfx->egl_context == EGL_NO_CONTEXT) {
        context_attribs[1] = 2;
        gfx->egl_context = eglCreateContext(gfx->egl_display, gfx->egl_config, EGL_NO_CONTEXT, context_attribs);
        if (gfx->egl_context == EGL_NO_CONTEXT) {
            goto fail;
        }
    }
    
    gfx->egl_surface = eglCreateWindowSurface(gfx->egl_display, gfx->egl_config, 
                                              (EGLNativeWindowType)gfx->gbm_surface, NULL);
    if (gfx->egl_surface == EGL_NO_SURFACE) {
        goto fail;
    }
    
    if (!eglMakeCurrent(gfx->egl_display, gfx->egl_surface, gfx->egl_surface, gfx->egl_context)) {
        goto fail;
    }

    const char* display_extensions = eglQueryString(gfx->egl_display, EGL_EXTENSIONS);
    if (display_extensions && strstr(display_extensions, "EGL_KHR_fence_sync")) {
        gfx->egl_create_sync = (PFNEGLCREATESYNCKHRPROC)eglGetProcAddress("eglCreateSyncKHR");
        gfx->egl_destroy_sync = (PFNEGLDESTROYSYNCKHRPROC)eglGetProcAddress("eglDestroySyncKHR");
        gfx->egl_client_wait_sync = (PFNEGLCLIENTWAITSYNCKHRPROC)eglGetProcAddress("eglClientWaitSyncKHR");
        if (!gfx->egl_create_sync || !gfx->egl_destroy_sync || !gfx->egl_client_wait_sync)
            gfx->egl_create_sync = NULL;
    }
    
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    eglSwapBuffers(gfx->egl_display, gfx->egl_surface);
    
    gfx->initialized = 1;
    return 0;

fail:
    /* Release exactly what was acquired, in reverse order, so a failed init
     * leaks nothing and does not keep DRM master (which would make every
     * later init in this process fail at drmSetMaster). */
    if (gfx->egl_display != EGL_NO_DISPLAY) {
        eglMakeCurrent(gfx->egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (gfx->egl_surface != EGL_NO_SURFACE) eglDestroySurface(gfx->egl_display, gfx->egl_surface);
        if (gfx->egl_context != EGL_NO_CONTEXT) eglDestroyContext(gfx->egl_display, gfx->egl_context);
        eglTerminate(gfx->egl_display);
        eglReleaseThread();
    }
    if (gfx->gbm_surface) gbm_surface_destroy(gfx->gbm_surface);
    if (gfx->gbm_device) gbm_device_destroy(gfx->gbm_device);
    close(gfx->drm_fd); /* also releases master */
    gfx->drm_fd = -1;
    gfx->gbm_surface = NULL;
    gfx->gbm_device = NULL;
    gfx->egl_display = EGL_NO_DISPLAY;
    gfx->egl_context = EGL_NO_CONTEXT;
    gfx->egl_surface = EGL_NO_SURFACE;
    return -1;
}

/* Initialize using the attached display's mode. */
static inline int hs_graphics_init(HSGraphics* gfx) {
    return hs_graphics_init_sized(gfx, 0, 0);
}

/* ---- KMS presentation ---------------------------------------------------- */

typedef struct {
    int         fd;
    uint32_t    fb_id;
    const bool* closing;
} HSScanoutFb;

static inline void hs_graphics_fb_destroy(struct gbm_bo* bo, void* data) {
    (void)bo;
    HSScanoutFb* fb = (HSScanoutFb*)data;
    if (!fb) return;
    /* During finish() the fd is about to be closed, which releases every
     * framebuffer it owns; removing the scanned-out one first would force an
     * extra CRTC disable (measured slower: GLES_SHUTDOWN_PLAN "early-RmFB"). */
    if (fb->fb_id && !*fb->closing) drmModeRmFB(fb->fd, fb->fb_id);
    free(fb);
}

/* The DRM framebuffer for a locked GBM buffer, created once and cached on it. */
static inline uint32_t hs_graphics_fb_for_bo(HSGraphics* gfx, struct gbm_bo* bo) {
    HSScanoutFb* fb = (HSScanoutFb*)gbm_bo_get_user_data(bo);
    if (fb) return fb->fb_id;
    fb = (HSScanoutFb*)calloc(1, sizeof(*fb));
    if (!fb) return 0;
    fb->fd = gfx->drm_fd;
    fb->closing = &gfx->closing;
    uint32_t width = gbm_bo_get_width(bo), height = gbm_bo_get_height(bo);
    int rc = drmModeAddFB(gfx->drm_fd, width, height, 24, 32, gbm_bo_get_stride(bo),
                          gbm_bo_get_handle(bo).u32, &fb->fb_id);
    if (rc != 0) {
        /* Buffers with a layout modifier need the explicit-modifier form. */
        uint32_t handles[4] = {0}, strides[4] = {0}, offsets[4] = {0};
        uint64_t modifiers[4] = {0};
        int planes = gbm_bo_get_plane_count(bo);
        uint64_t modifier = gbm_bo_get_modifier(bo);
        for (int i = 0; i < planes && i < 4; i++) {
            handles[i] = gbm_bo_get_handle_for_plane(bo, i).u32;
            strides[i] = gbm_bo_get_stride_for_plane(bo, i);
            offsets[i] = gbm_bo_get_offset(bo, i);
            modifiers[i] = modifier;
        }
        rc = drmModeAddFB2WithModifiers(gfx->drm_fd, width, height, gbm_bo_get_format(bo), handles,
                                        strides, offsets, modifiers, &fb->fb_id, DRM_MODE_FB_MODIFIERS);
    }
    if (rc != 0) {
        gfx->last_present_errno = errno;
        free(fb);
        return 0;
    }
    gbm_bo_set_user_data(bo, fb, hs_graphics_fb_destroy);
    return fb->fb_id;
}

static inline void hs_graphics_flip_done(int fd, unsigned int sequence, unsigned int sec,
                                         unsigned int usec, void* data) {
    (void)fd;
    HSGraphics* gfx = (HSGraphics*)data;
    gfx->last_flip_ns = (u64)sec * 1000000000ull + (u64)usec * 1000ull;
    if (gfx->last_flip_sequence && sequence - gfx->last_flip_sequence > 1)
        gfx->refreshes_repeated += sequence - gfx->last_flip_sequence - 1;
    gfx->last_flip_sequence = sequence;
    gfx->flip_pending = false;
}

/* Service DRM events until the queued flip completes or timeout_ms passes
 * (0 = only consume events already delivered). Returns true when no flip is
 * pending afterwards. */
static inline bool hs_graphics_wait_flip(HSGraphics* gfx, int timeout_ms) {
    drmEventContext events;
    memset(&events, 0, sizeof(events));
    events.version = 2;
    events.page_flip_handler = hs_graphics_flip_done;
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
    while (gfx->flip_pending) {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        long elapsed = (long)(now.tv_sec - start.tv_sec) * 1000L + (now.tv_nsec - start.tv_nsec) / 1000000L;
        int remaining = timeout_ms - (int)elapsed;
        if (remaining < 0) remaining = 0;
        struct pollfd pfd = {.fd = gfx->drm_fd, .events = POLLIN};
        int ready = poll(&pfd, 1, remaining);
        if (ready < 0 && errno == EINTR) continue;
        if (ready <= 0) break;
        if (drmHandleEvent(gfx->drm_fd, &events) != 0) break;
    }
    return !gfx->flip_pending;
}

/* The queued buffer is now on screen: the previous one can be rendered into. */
static inline void hs_graphics_retire_flip(HSGraphics* gfx) {
    if (gfx->scanout_bo) gbm_surface_release_buffer(gfx->gbm_surface, gfx->scanout_bo);
    gfx->scanout_bo = gfx->pending_bo;
    gfx->pending_bo = NULL;
    gfx->frames_presented++;
}

/* GPU completion fence for the frame just swapped; EGL_NO_SYNC_KHR without
 * EGL_KHR_fence_sync. */
static inline EGLSyncKHR hs_graphics_fence(HSGraphics* gfx) {
    if (!gfx->egl_create_sync) return EGL_NO_SYNC_KHR;
    return gfx->egl_create_sync(gfx->egl_display, EGL_SYNC_FENCE_KHR, NULL);
}

static inline bool hs_graphics_fence_done(HSGraphics* gfx, EGLSyncKHR sync) {
    if (sync == EGL_NO_SYNC_KHR) return false;
    return gfx->egl_client_wait_sync(gfx->egl_display, sync, EGL_SYNC_FLUSH_COMMANDS_BIT_KHR, 0) ==
           EGL_CONDITION_SATISFIED_KHR;
}

static inline void hs_graphics_fence_destroy(HSGraphics* gfx, EGLSyncKHR sync) {
    if (sync != EGL_NO_SYNC_KHR) gfx->egl_destroy_sync(gfx->egl_display, sync);
}

/* The held frame is replaced without being shown. */
static inline void hs_graphics_drop_held(HSGraphics* gfx) {
    if (!gfx->held_bo) return;
    gbm_surface_release_buffer(gfx->gbm_surface, gfx->held_bo);
    gfx->held_bo = NULL;
    gfx->frames_dropped++;
}

static inline void hs_graphics_present_failed(HSGraphics* gfx, const char* why);

/* Queue a flip to bo (the flip slot must be free). Releases bo on failure. */
static inline void hs_graphics_queue_flip(HSGraphics* gfx, struct gbm_bo* bo) {
    uint32_t fb = hs_graphics_fb_for_bo(gfx, bo);
    if (!fb || drmModePageFlip(gfx->drm_fd, gfx->display_crtc_id, fb, DRM_MODE_PAGE_FLIP_EVENT, gfx) != 0) {
        hs_graphics_present_failed(gfx, fb ? "drmModePageFlip failed" : "drmModeAddFB failed");
        gbm_surface_release_buffer(gfx->gbm_surface, bo);
        return;
    }
    gfx->pending_bo = bo;
    gfx->flip_pending = true;
}

static inline void hs_graphics_present_failed(HSGraphics* gfx, const char* why) {
    gfx->present_errors++;
    if (!gfx->last_present_errno) gfx->last_present_errno = errno;
    gfx->present_error = why;
}

/* Property id of name on a KMS object, or 0. */
static inline uint32_t hs_graphics_prop(int fd, uint32_t obj, uint32_t type, const char* name, uint64_t* value) {
    drmModeObjectProperties* props = drmModeObjectGetProperties(fd, obj, type);
    uint32_t id = 0;
    for (uint32_t i = 0; props && i < props->count_props && !id; i++) {
        drmModePropertyRes* p = drmModeGetProperty(fd, props->props[i]);
        if (p && !strcmp(p->name, name)) {
            id = p->prop_id;
            if (value) *value = props->prop_values[i];
        }
        drmModeFreeProperty(p);
    }
    drmModeFreeObjectProperties(props);
    return id;
}

/* Plane properties a previous DRM client may have left changed, and their
 * defaults; the console's restore resets only rotation (measured,
 * tests/evidence-p0-3-redteam-20261116/inherited_props.jsonl). An enum or
 * range property the kernel lacks is skipped. */
static const struct { const char* name; uint64_t value; } hs_graphics_plane_defaults[] = {
    {"alpha", 0xffff},
    {"pixel blend mode", 0},  /* Pre-multiplied (the kernel default) */
    {"SCALING_FILTER", 0},    /* Default */
    {"rotation", 1},          /* rotate-0 */
};

/* The primary plane that can drive gfx's CRTC (universal planes on), or 0. */
static inline uint32_t hs_graphics_primary_plane(HSGraphics* gfx) {
    int fd = gfx->drm_fd;
    if (drmSetClientCap(fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1)) return 0;
    drmModeRes* res = drmModeGetResources(fd);
    int crtc_index = -1;
    for (int i = 0; res && i < res->count_crtcs; i++)
        if (res->crtcs[i] == gfx->display_crtc_id) crtc_index = i;
    drmModeFreeResources(res);
    drmModePlaneRes* planes = drmModeGetPlaneResources(fd);
    uint32_t primary = 0;
    for (uint32_t i = 0; crtc_index >= 0 && planes && i < planes->count_planes && !primary; i++) {
        drmModePlane* p = drmModeGetPlane(fd, planes->planes[i]);
        uint64_t type = 0;
        if (p && (p->possible_crtcs & (1u << crtc_index)) &&
            hs_graphics_prop(fd, p->plane_id, DRM_MODE_OBJECT_PLANE, "type", &type) && type == DRM_PLANE_TYPE_PRIMARY)
            primary = p->plane_id;
        drmModeFreePlane(p);
    }
    drmModeFreePlaneResources(planes);
    return primary;
}

/* Native takeover: turn off any other plane a previous client left on the
 * CRTC (SetCrtc does not), and put the primary plane's properties back to
 * their defaults where they differ. Each change is a commit; normally none
 * is needed. */
static inline void hs_graphics_reset_plane(HSGraphics* gfx) {
    uint32_t primary = hs_graphics_primary_plane(gfx);
    drmModePlaneRes* planes = primary ? drmModeGetPlaneResources(gfx->drm_fd) : NULL;
    for (uint32_t i = 0; planes && i < planes->count_planes; i++) {
        drmModePlane* p = drmModeGetPlane(gfx->drm_fd, planes->planes[i]);
        if (p && p->plane_id != primary && p->crtc_id == gfx->display_crtc_id)
            drmModeSetPlane(gfx->drm_fd, p->plane_id, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        drmModeFreePlane(p);
    }
    drmModeFreePlaneResources(planes);
    for (size_t i = 0; primary && i < sizeof(hs_graphics_plane_defaults) / sizeof(hs_graphics_plane_defaults[0]); i++) {
        uint64_t value = 0;
        uint32_t id = hs_graphics_prop(gfx->drm_fd, primary, DRM_MODE_OBJECT_PLANE, hs_graphics_plane_defaults[i].name, &value);
        if (id && value != hs_graphics_plane_defaults[i].value)
            drmModeObjectSetProperty(gfx->drm_fd, primary, DRM_MODE_OBJECT_PLANE, id, hs_graphics_plane_defaults[i].value);
    }
}

/* Scaled takeover: one atomic modeset that shows fb (the render size) on the
 * CRTC's primary plane stretched to the full mode, puts the plane's
 * alpha, blend mode, filter and rotation back to their defaults, and turns
 * off any other plane left on this CRTC. Later legacy page flips keep the plane's source
 * and destination rectangles (measured: tests/evidence-p0-3-20261116). */
static inline int hs_graphics_set_scaled(HSGraphics* gfx, uint32_t fb) {
    int fd = gfx->drm_fd;
    if (drmSetClientCap(fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1) || drmSetClientCap(fd, DRM_CLIENT_CAP_ATOMIC, 1))
        return -1;
    drmModeRes* res = drmModeGetResources(fd);
    int crtc_index = -1;
    for (int i = 0; res && i < res->count_crtcs; i++)
        if (res->crtcs[i] == gfx->display_crtc_id) crtc_index = i;
    drmModeFreeResources(res);
    drmModePlaneRes* planes = drmModeGetPlaneResources(fd);
    if (crtc_index < 0 || !planes) {
        drmModeFreePlaneResources(planes);
        return -1;
    }
    drmModeAtomicReq* req = drmModeAtomicAlloc();
    uint32_t primary = 0, blob = 0;
    int bad = !req;
    for (uint32_t i = 0; !bad && i < planes->count_planes; i++) {
        drmModePlane* p = drmModeGetPlane(fd, planes->planes[i]);
        if (!p) continue;
        uint64_t type = 0;
        bool ours = p->possible_crtcs & (1u << crtc_index);
        if (ours && !primary && hs_graphics_prop(fd, p->plane_id, DRM_MODE_OBJECT_PLANE, "type", &type) &&
            type == DRM_PLANE_TYPE_PRIMARY) {
            primary = p->plane_id;
        } else if (p->crtc_id == gfx->display_crtc_id) {
            uint32_t crtc_prop = hs_graphics_prop(fd, p->plane_id, DRM_MODE_OBJECT_PLANE, "CRTC_ID", NULL);
            uint32_t fb_prop = hs_graphics_prop(fd, p->plane_id, DRM_MODE_OBJECT_PLANE, "FB_ID", NULL);
            bad |= !crtc_prop || drmModeAtomicAddProperty(req, p->plane_id, crtc_prop, 0) < 0;
            bad |= !fb_prop || drmModeAtomicAddProperty(req, p->plane_id, fb_prop, 0) < 0;
        }
        drmModeFreePlane(p);
    }
    drmModeFreePlaneResources(planes);
    if (!bad && primary && !drmModeCreatePropertyBlob(fd, &gfx->display_mode, sizeof(gfx->display_mode), &blob)) {
        struct { uint32_t obj, type; const char* name; uint64_t value; } set[] = {
            {gfx->display_crtc_id, DRM_MODE_OBJECT_CRTC, "MODE_ID", blob},
            {gfx->display_crtc_id, DRM_MODE_OBJECT_CRTC, "ACTIVE", 1},
            {gfx->display_connector_id, DRM_MODE_OBJECT_CONNECTOR, "CRTC_ID", gfx->display_crtc_id},
            {primary, DRM_MODE_OBJECT_PLANE, "FB_ID", fb},
            {primary, DRM_MODE_OBJECT_PLANE, "CRTC_ID", gfx->display_crtc_id},
            {primary, DRM_MODE_OBJECT_PLANE, "SRC_X", 0},
            {primary, DRM_MODE_OBJECT_PLANE, "SRC_Y", 0},
            {primary, DRM_MODE_OBJECT_PLANE, "SRC_W", (uint64_t)gfx->screen_width << 16},
            {primary, DRM_MODE_OBJECT_PLANE, "SRC_H", (uint64_t)gfx->screen_height << 16},
            {primary, DRM_MODE_OBJECT_PLANE, "CRTC_X", 0},
            {primary, DRM_MODE_OBJECT_PLANE, "CRTC_Y", 0},
            {primary, DRM_MODE_OBJECT_PLANE, "CRTC_W", gfx->display_mode.hdisplay},
            {primary, DRM_MODE_OBJECT_PLANE, "CRTC_H", gfx->display_mode.vdisplay},
        };
        for (size_t i = 0; !bad && i < sizeof(set) / sizeof(set[0]); i++) {
            uint32_t id = hs_graphics_prop(fd, set[i].obj, set[i].type, set[i].name, NULL);
            bad |= !id || drmModeAtomicAddProperty(req, set[i].obj, id, set[i].value) < 0;
        }
        for (size_t i = 0; !bad && i < sizeof(hs_graphics_plane_defaults) / sizeof(hs_graphics_plane_defaults[0]); i++) {
            uint32_t id = hs_graphics_prop(fd, primary, DRM_MODE_OBJECT_PLANE, hs_graphics_plane_defaults[i].name, NULL);
            bad |= id && drmModeAtomicAddProperty(req, primary, id, hs_graphics_plane_defaults[i].value) < 0;
        }
    } else {
        bad = 1;
    }
    int rc = bad ? -1 : drmModeAtomicCommit(fd, req, DRM_MODE_ATOMIC_ALLOW_MODESET, NULL);
    drmModeAtomicFree(req);
    if (blob) drmModeDestroyPropertyBlob(fd, blob);
    return rc;
}

/* First present: resolve the display and program the CRTC with this frame. */
static inline void hs_graphics_takeover(HSGraphics* gfx) {
    gfx->crtc_setup_done = 1;
    bool resolved = false;

    /* Reuse the scan done at init while the connector still reports the same
     * state. drmModeGetConnectorCurrent returns the kernel's cached state
     * without re-probing the panel (the probe is what is slow on DSI). */
    if (gfx->reuse_connector_scan && gfx->display_found && gfx->display_crtc_id) {
        drmModeConnector* current = drmModeGetConnectorCurrent(gfx->drm_fd, gfx->display_connector_id);
        if (current) {
            resolved = current->connection == DRM_MODE_CONNECTED;
            drmModeFreeConnector(current);
        }
    }
    if (!resolved) {
        /* The connector may have appeared, changed, or gone since init. */
        resolved = hs_graphics_select_display(gfx->drm_fd, 0, gfx) && gfx->display_crtc_id;
    }
    if (!resolved) {
        hs_graphics_present_failed(gfx, "no connected display with a usable CRTC");
        return;
    }
    if (!gfx->scaled_scanout &&
        (gfx->display_mode.hdisplay != gfx->screen_width || gfx->display_mode.vdisplay != gfx->screen_height)) {
        /* The surface was sized for a different (or no) display; scanning it
         * out with this mode would fail with EINVAL. Stay render-only. */
        errno = EINVAL;
        hs_graphics_present_failed(gfx, "display mode does not match surface size");
        return;
    }

    struct gbm_bo* bo = gbm_surface_lock_front_buffer(gfx->gbm_surface);
    if (!bo) {
        hs_graphics_present_failed(gfx, "gbm_surface_lock_front_buffer failed");
        return;
    }
    uint32_t fb = hs_graphics_fb_for_bo(gfx, bo);
    uint32_t connector = gfx->display_connector_id;
    int set = !fb ? -1
            : gfx->scaled_scanout ? hs_graphics_set_scaled(gfx, fb)
            : drmModeSetCrtc(gfx->drm_fd, gfx->display_crtc_id, fb, 0, 0, &connector, 1, &gfx->display_mode);
    if (set != 0) {
        hs_graphics_present_failed(gfx, !fb ? "drmModeAddFB failed"
                                        : gfx->scaled_scanout ? "scaled atomic modeset failed"
                                                              : "drmModeSetCrtc failed");
        gbm_surface_release_buffer(gfx->gbm_surface, bo);
        return;
    }
    if (!gfx->scaled_scanout) hs_graphics_reset_plane(gfx);
    gfx->scanout_bo = bo;
    gfx->crtc_active = true;
    gfx->present_error = NULL;
    gfx->frames_presented++;
}

/* Wait for a flip in flight and show a held frame (vsync off), so that only
 * the scanned-out buffer stays locked. Blocks up to about two refreshes.
 * Call it before swapping this surface with eglSwapBuffers yourself: Mesa's
 * GBM surfaces have four buffers and a raw swap needs two free ones. */
static inline void hs_graphics_flush_present(HSGraphics* gfx) {
    if (!gfx->initialized) return;
    if (gfx->flip_pending && hs_graphics_wait_flip(gfx, 100)) hs_graphics_retire_flip(gfx);
    if (!gfx->held_bo) return;
    struct gbm_bo* bo = gfx->held_bo;
    gfx->held_bo = NULL;
    if (gfx->crtc_active && !gfx->flip_pending) {
        hs_graphics_queue_flip(gfx, bo);
        if (gfx->flip_pending && hs_graphics_wait_flip(gfx, 100)) hs_graphics_retire_flip(gfx);
    } else {
        gbm_surface_release_buffer(gfx->gbm_surface, bo);
        gfx->frames_dropped++;
    }
}

static inline void hs_graphics_finish(HSGraphics* gfx) {
    if (!gfx->initialized) return;

    /* Let a queued flip land before its buffer is destroyed, and show a
     * frame still held for the flip slot. */
    hs_graphics_flush_present(gfx);
    gfx->closing = true; /* framebuffers are released by close() below */
    if (gfx->pending_bo) gbm_surface_release_buffer(gfx->gbm_surface, gfx->pending_bo);
    if (gfx->scanout_bo) gbm_surface_release_buffer(gfx->gbm_surface, gfx->scanout_bo);
    gfx->pending_bo = gfx->scanout_bo = NULL;
    gfx->flip_pending = false;
    gfx->crtc_active = false;

    for (int i = 0; i < HS_MAX_TEXTURES; i++) {
        if (gfx->textures[i].gl_texture != 0 && !gfx->textures[i].disposed) {
            glDeleteTextures(1, &gfx->textures[i].gl_texture);
        }
    }
    
    eglMakeCurrent(gfx->egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (gfx->egl_surface != EGL_NO_SURFACE) {
        eglDestroySurface(gfx->egl_display, gfx->egl_surface);
    }
    if (gfx->egl_context != EGL_NO_CONTEXT) {
        eglDestroyContext(gfx->egl_display, gfx->egl_context);
    }
    eglTerminate(gfx->egl_display);
    eglReleaseThread();
    if (gfx->gbm_surface) {
        gbm_surface_destroy(gfx->gbm_surface);
    }
    if (gfx->gbm_device) {
        gbm_device_destroy(gfx->gbm_device);
    }
    if (gfx->drm_fd >= 0) {
        drmDropMaster(gfx->drm_fd);
        close(gfx->drm_fd);
    }
    
    gfx->initialized = 0;
}

static inline HSGLTexture* hs_graphics_create_texture(HSGraphics* gfx, u8 slot, u32 width, u32 height, const u8* data) {
    if (!gfx->initialized || slot >= HS_MAX_TEXTURES) return NULL;
    
    HSGLTexture* tex = &gfx->textures[slot];
    
    if (tex->gl_texture != 0 && !tex->disposed) {
        glDeleteTextures(1, &tex->gl_texture);
    }
    
    glGenTextures(1, &tex->gl_texture);
    glBindTexture(GL_TEXTURE_2D, tex->gl_texture);
    
    GLenum format = GL_RGBA;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, format, GL_UNSIGNED_BYTE, data);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    tex->width = width;
    tex->height = height;
    tex->format = format;
    tex->disposed = false;
    tex->dirty = false;
    
    gfx->textures_active |= (1 << slot);
    
    return tex;
}

static inline void hs_graphics_update_texture(HSGraphics* gfx, u8 slot, const u8* data) {
    if (!gfx->initialized || slot >= HS_MAX_TEXTURES) return;
    
    HSGLTexture* tex = &gfx->textures[slot];
    if (tex->gl_texture == 0 || tex->disposed) return;
    
    glBindTexture(GL_TEXTURE_2D, tex->gl_texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tex->width, tex->height, tex->format, GL_UNSIGNED_BYTE, data);
    tex->dirty = false;
}

static inline bool hs_graphics_texture_disposed(HSGraphics* gfx, u8 slot) {
    if (slot >= HS_MAX_TEXTURES) return true;
    return gfx->textures[slot].disposed;
}

static inline void hs_graphics_dispose_texture(HSGraphics* gfx, u8 slot) {
    if (!gfx->initialized || slot >= HS_MAX_TEXTURES) return;
    
    HSGLTexture* tex = &gfx->textures[slot];
    if (tex->gl_texture != 0 && !tex->disposed) {
        glDeleteTextures(1, &tex->gl_texture);
        tex->gl_texture = 0;
        tex->disposed = true;
    }
}

static inline GLuint hs_graphics_get_gl_texture(HSGraphics* gfx, u8 slot) {
    if (slot >= HS_MAX_TEXTURES) return 0;
    return gfx->textures[slot].gl_texture;
}

static inline void hs_graphics_clear(HSGraphics* gfx, float r, float g, float b, float a) {
    if (!gfx->initialized) return;
    glClearColor(r, g, b, a);
    glClear(GL_COLOR_BUFFER_BIT);
}

/* Present the frame just rendered.
 *
 * Every frame is put on screen with a page flip of its own buffer:
 *   vsync off (default) - the call never waits, so rendering runs at full
 *               speed. A flip lands only once the GPU has finished its frame,
 *               so the frame flipped is the newest one the GPU has finished
 *               (or, if none has, the older candidate): flipping a frame still
 *               rendering behind frames that will be dropped would miss the
 *               next refresh and repeat the old image. While a flip is in
 *               flight the newest frame is held (when frames take at least
 *               hold_permille of a refresh; faster ones finish in time and
 *               are dropped)
 *               and flipped at the next present that finds the slot free
 *               (frames of 7/8 refresh or more wait for a flip about to
 *               land, so the next one is queued before the display's
 *               deadline); a
 *               frame replaced by a newer one is not shown (frames_dropped). A held frame keeps a
 *               third buffer locked: call hs_graphics_flush_present() before
 *               calling eglSwapBuffers on this surface yourself.
 *   vsync on  - the call returns after the flip completes (one frame per
 *               vblank; the scanned-out buffer is never rendered into).
 * Without take_display (offscreen use) the frame is only swapped. */
#define HS_GRAPHICS_HOLD_AFTER_REPEAT 60
static inline void hs_graphics_present(HSGraphics* gfx) {
    if (!gfx->initialized) return;

    /* Applied only when vsync changes: an EGL call per frame is measurable. */
    int interval = gfx->vsync ? 1 : 0;
    if (gfx->swap_interval != interval && eglSwapInterval(gfx->egl_display, interval))
        gfx->swap_interval = interval;
    eglSwapBuffers(gfx->egl_display, gfx->egl_surface);

    if (!gfx->take_display) return;
    if (!gfx->crtc_setup_done) {
        hs_graphics_takeover(gfx);
        return;
    }
    if (!gfx->crtc_active) return; /* takeover failed: render-only, reason in present_error */

    if (gfx->flip_pending && hs_graphics_wait_flip(gfx, gfx->vsync ? 100 : 0)) {
        hs_graphics_retire_flip(gfx);
    }

    struct gbm_bo* bo = gbm_surface_lock_front_buffer(gfx->gbm_surface);
    if (!bo) {
        hs_graphics_present_failed(gfx, "gbm_surface_lock_front_buffer failed");
        return;
    }
    if (gfx->vsync) {
        hs_graphics_drop_held(gfx); /* left from vsync off */
        if (gfx->flip_pending) { /* the 100 ms wait above timed out */
            gbm_surface_release_buffer(gfx->gbm_surface, bo);
            gfx->frames_dropped++;
            return;
        }
        hs_graphics_queue_flip(gfx, bo);
        if (gfx->flip_pending && hs_graphics_wait_flip(gfx, 100)) hs_graphics_retire_flip(gfx);
        return;
    }

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    u64 now = (u64)ts.tv_sec * 1000000000ull + (u64)ts.tv_nsec;
    if (gfx->last_present_ns) {
        u64 dt = now - gfx->last_present_ns;
        gfx->present_interval_ns = gfx->present_interval_ns ? (gfx->present_interval_ns * 7 + dt) / 8 : dt;
    }
    gfx->last_present_ns = now;
    u64 refresh_ns = 1000000000ull / (gfx->display_mode.vrefresh ? gfx->display_mode.vrefresh : 60);
    if (gfx->flip_pending && gfx->present_interval_ns * 8 >= refresh_ns * 7 && gfx->last_flip_ns) {
        /* Frames take about a refresh or more. A flip must be queued some
         * 4-7 ms before a vblank to land on it; the next present would often
         * come too late and repeat a refresh. When the pending flip lands
         * before the next present is due, wait for it and queue the next
         * frame at once. Measured smoother below 68 FPS, not above. */
        u64 next_vblank = gfx->last_flip_ns + refresh_ns;
        while (next_vblank < now) next_vblank += refresh_ns;
        u64 until = next_vblank - now;
        if (until < gfx->present_interval_ns) {
            gfx->flip_waits++;
            if (hs_graphics_wait_flip(gfx, (int)(until / 1000000ull) + 3)) hs_graphics_retire_flip(gfx);
        }
    }
    if (gfx->refreshes_repeated != gfx->repeats_seen) {
        /* A refresh repeated: frames are not reaching the display in time
         * (busy CPUs, uneven frame costs). Hold from a quarter refresh. */
        gfx->repeats_seen = gfx->refreshes_repeated;
        gfx->hold_until_flip = gfx->frames_presented + HS_GRAPHICS_HOLD_AFTER_REPEAT;
    }
    if (gfx->flip_pending) {
        hs_graphics_drop_held(gfx);
        bool repeating = gfx->frames_presented < gfx->hold_until_flip;
        if (gfx->present_interval_ns * 1000 >= (u64)refresh_ns * gfx->hold_permille ||
            (repeating && gfx->present_interval_ns * 4 >= refresh_ns)) {
            /* Hold the newest frame for the next free flip slot. */
            gfx->held_bo = bo;
        } else {
            gbm_surface_release_buffer(gfx->gbm_surface, bo);
            gfx->frames_dropped++;
        }
        return;
    }
    if (gfx->held_bo) {
        /* At most once per refresh: flip this frame if the GPU has already
         * finished it, else the older held one (it finishes first) and hold
         * this one. Without a fence, the held one. */
        EGLSyncKHR sync = hs_graphics_fence(gfx);
        bool done = hs_graphics_fence_done(gfx, sync);
        hs_graphics_fence_destroy(gfx, sync);
        if (!done) {
            struct gbm_bo* show = gfx->held_bo;
            gfx->held_bo = bo;
            hs_graphics_queue_flip(gfx, show);
            return;
        }
        hs_graphics_drop_held(gfx);
    }
    hs_graphics_queue_flip(gfx, bo);
}

/* Compatibility name for hs_graphics_present(). A bare eglSwapBuffers would
 * skip the page flip, leaving the display on a stale buffer. */
static inline void hs_graphics_swap_buffers(HSGraphics* gfx) {
    hs_graphics_present(gfx);
}

#else

typedef struct {
    bool initialized;
    u32 screen_width;
    u32 screen_height;
    bool vsync;
} HSGraphics;

static inline int hs_graphics_init(HSGraphics* gfx) {
    memset(gfx, 0, sizeof(HSGraphics));
    gfx->screen_width = 800;
    gfx->screen_height = 480;
    gfx->initialized = 1;
    return 0;
}

static inline void hs_graphics_finish(HSGraphics* gfx) {
    gfx->initialized = 0;
}

static inline void hs_graphics_swap_buffers(HSGraphics* gfx) {}
static inline void hs_graphics_clear(HSGraphics* gfx, float r, float g, float b, float a) {}
static inline void hs_graphics_present(HSGraphics* gfx) {}
static inline void hs_graphics_flush_present(HSGraphics* gfx) {}
static inline bool hs_graphics_texture_disposed(HSGraphics* gfx, u8 slot) { return false; }
static inline void hs_graphics_dispose_texture(HSGraphics* gfx, u8 slot) {}
static inline GLuint hs_graphics_get_gl_texture(HSGraphics* gfx, u8 slot) { return 0; }
static inline HSGLTexture* hs_graphics_create_texture(HSGraphics* gfx, u8 slot, u32 w, u32 h, const u8* d) { return NULL; }
static inline void hs_graphics_update_texture(HSGraphics* gfx, u8 slot, const u8* data) {}

#endif

#endif
