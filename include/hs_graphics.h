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
    u64            frames_presented;  /* flips (and the takeover) that completed */
    u64            frames_dropped;    /* vsync off: frame skipped, flip still in flight */
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
static inline int hs_graphics_init_sized(HSGraphics* gfx, u32 width, u32 height) {
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

    if (width && height) {
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
    (void)fd; (void)sequence; (void)sec; (void)usec;
    ((HSGraphics*)data)->flip_pending = false;
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

static inline void hs_graphics_present_failed(HSGraphics* gfx, const char* why) {
    gfx->present_errors++;
    if (!gfx->last_present_errno) gfx->last_present_errno = errno;
    gfx->present_error = why;
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
    if (gfx->display_mode.hdisplay != gfx->screen_width || gfx->display_mode.vdisplay != gfx->screen_height) {
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
    if (!fb || drmModeSetCrtc(gfx->drm_fd, gfx->display_crtc_id, fb, 0, 0, &connector, 1,
                              &gfx->display_mode) != 0) {
        hs_graphics_present_failed(gfx, fb ? "drmModeSetCrtc failed" : "drmModeAddFB failed");
        gbm_surface_release_buffer(gfx->gbm_surface, bo);
        return;
    }
    gfx->scanout_bo = bo;
    gfx->crtc_active = true;
    gfx->present_error = NULL;
    gfx->frames_presented++;
}

static inline void hs_graphics_finish(HSGraphics* gfx) {
    if (!gfx->initialized) return;

    /* Let a queued flip land before its buffer is destroyed. */
    if (gfx->flip_pending && hs_graphics_wait_flip(gfx, 100)) hs_graphics_retire_flip(gfx);
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
 *   vsync off (default) - the flip is queued and the call returns at once, so
 *               rendering runs at full speed. If the previous flip has not
 *               completed yet, this frame is not shown (frames_dropped) rather
 *               than shown torn or waited for.
 *   vsync on  - the call returns after the flip completes (one frame per
 *               vblank; the scanned-out buffer is never rendered into).
 * Without take_display (offscreen use) the frame is only swapped. */
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
    if (gfx->flip_pending) {
        /* vsync off and the previous flip is still in flight. */
        gbm_surface_release_buffer(gfx->gbm_surface, bo);
        gfx->frames_dropped++;
        return;
    }
    uint32_t fb = hs_graphics_fb_for_bo(gfx, bo);
    if (!fb || drmModePageFlip(gfx->drm_fd, gfx->display_crtc_id, fb, DRM_MODE_PAGE_FLIP_EVENT, gfx) != 0) {
        hs_graphics_present_failed(gfx, fb ? "drmModePageFlip failed" : "drmModeAddFB failed");
        gbm_surface_release_buffer(gfx->gbm_surface, bo);
        return;
    }
    gfx->pending_bo = bo;
    gfx->flip_pending = true;
    if (gfx->vsync && hs_graphics_wait_flip(gfx, 100)) hs_graphics_retire_flip(gfx);
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
static inline bool hs_graphics_texture_disposed(HSGraphics* gfx, u8 slot) { return false; }
static inline void hs_graphics_dispose_texture(HSGraphics* gfx, u8 slot) {}
static inline GLuint hs_graphics_get_gl_texture(HSGraphics* gfx, u8 slot) { return 0; }
static inline HSGLTexture* hs_graphics_create_texture(HSGraphics* gfx, u8 slot, u32 w, u32 h, const u8* d) { return NULL; }
static inline void hs_graphics_update_texture(HSGraphics* gfx, u8 slot, const u8* data) {}

#endif

#endif
