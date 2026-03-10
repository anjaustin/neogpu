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
} HSTexture;

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
    
    HSTexture      textures[HS_MAX_TEXTURES];
    u8             textures_active;
    
    bool           initialized;
    bool           vsync;
} HSGraphics;

static inline int hs_graphics_init(HSGraphics* gfx) {
    memset(gfx, 0, sizeof(HSGraphics));
    gfx->vsync = true;
    
    gfx->drm_fd = open("/dev/dri/card1", O_RDWR | O_CLOEXEC);
    if (gfx->drm_fd < 0) {
        gfx->drm_fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    }
    if (gfx->drm_fd < 0) {
        return -1;
    }
    
    if (drmSetMaster(gfx->drm_fd) != 0) {
        return -1;
    }
    
    drmModeConnector* conn = NULL;
    drmModeRes* resources = drmModeGetResources(gfx->drm_fd);
    if (resources) {
        for (int i = 0; i < resources->count_connectors; i++) {
            conn = drmModeGetConnector(gfx->drm_fd, resources->connectors[i]);
            if (conn && conn->connection == DRM_MODE_CONNECTED && conn->count_modes > 0) {
                gfx->screen_width = conn->modes[0].hdisplay;
                gfx->screen_height = conn->modes[0].vdisplay;
                break;
            }
            drmModeFreeConnector(conn);
        }
        drmModeFreeResources(resources);
    }
    
    if (gfx->screen_width == 0) {
        gfx->screen_width = 800;
        gfx->screen_height = 480;
    }
    
    gfx->gbm_device = gbm_create_device(gfx->drm_fd);
    if (!gfx->gbm_device) {
        return -1;
    }
    
    gfx->gbm_surface = gbm_surface_create(
        gfx->gbm_device,
        gfx->screen_width,
        gbm->screen_height,
        GBM_FORMAT_XRGB8888,
        GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING
    );
    if (!gfx->gbm_surface) {
        return -1;
    }
    
    EGLint egl_major, egl_minor;
    const char* client_extensions = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
    
    gfx->egl_display = eglGetDisplay((EGLNativeDisplayType)gfx->gbm_device);
    if (gfx->egl_display == EGL_NO_DISPLAY) {
        return -1;
    }
    
    if (!eglInitialize(gfx->egl_display, &egl_major, &egl_minor)) {
        return -1;
    }
    
    EGLint config_attribs[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 0,
        EGL_DEPTH_SIZE, 0,
        EGL_STENCIL_SIZE, 0,
        EGL_SAMPLE_BUFFERS, 0,
        EGL_NONE
    };
    
    EGLint num_configs;
    if (!eglChooseConfig(gfx->egl_display, config_attribs, &gfx->egl_config, 1, &num_configs) || num_configs == 0) {
        return -1;
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
            return -1;
        }
    }
    
    gfx->egl_surface = eglCreateWindowSurface(gfx->egl_display, gfx->egl_config, 
                                              (EGLNativeWindowType)gfx->gbm_surface, NULL);
    if (gfx->egl_surface == EGL_NO_SURFACE) {
        return -1;
    }
    
    if (!eglMakeCurrent(gfx->egl_display, gfx->egl_surface, gfx->egl_surface, gfx->egl_context)) {
        return -1;
    }
    
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    eglSwapBuffers(gfx->egl_display, gfx->egl_surface);
    
    gfx->initialized = 1;
    return 0;
}

static inline void hs_graphics_finish(HSGraphics* gfx) {
    if (!gfx->initialized) return;
    
    for (int i = 0; i < HS_MAX_TEXTURES; i++) {
        if (gfx->textures[i].gl_texture != 0 && !gfx->textures[i].disposed) {
            glDeleteTextures(1, &gfx->textures[i].gl_texture);
        }
    }
    
    if (gfx->egl_surface != EGL_NO_SURFACE) {
        eglDestroySurface(gfx->egl_display, gfx->egl_surface);
    }
    if (gfx->egl_context != EGL_NO_CONTEXT) {
        eglDestroyContext(gfx->egl_display, gfx->egl_context);
    }
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

static inline HSTexture* hs_graphics_create_texture(HSGraphics* gfx, u8 slot, u32 width, u32 height, const u8* data) {
    if (!gfx->initialized || slot >= HS_MAX_TEXTURES) return NULL;
    
    HSTexture* tex = &gfx->textures[slot];
    
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
    
    HSTexture* tex = &gfx->textures[slot];
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
    
    HSTexture* tex = &gfx->textures[slot];
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

static inline void hs_graphics_swap_buffers(HSGraphics* gfx) {
    if (!gfx->initialized) return;
    eglSwapBuffers(gfx->egl_display, gfx->egl_surface);
    
    struct gbm_bo* bo = gbm_surface_lock_front_buffer(gfx->gbm_surface);
    if (bo) {
        uint32_t handle = gbm_bo_get_handle(bo).u32;
        uint32_t stride = gbm_bo_get_stride(bo);
        
        drmModeFBCreate(gfx->drm_fd, &handle, 0, 
                       gfx->screen_width, gfx->screen_height, 24, 32,
                       stride, handle);
        
        drmModeSetCrtc(gfx->drm_fd, handle, 0, 0, 0, 
                       0, NULL, 0, NULL, 0);
        
        gbm_surface_release_buffer(gfx->gbm_surface, bo);
    }
}

static inline void hs_graphics_clear(HSGraphics* gfx, float r, float g, float b, float a) {
    if (!gfx->initialized) return;
    glClearColor(r, g, b, a);
    glClear(GL_COLOR_BUFFER_BIT);
}

static inline void hs_graphics_present(HSGraphics* gfx) {
    if (!gfx->initialized) return;
    if (gfx->vsync) {
        eglSwapBuffers(gfx->egl_display, gfx->egl_surface);
    } else {
        eglSwapInterval(gfx->egl_display, 0);
        eglSwapBuffers(gfx->egl_display, gfx->egl_surface);
    }
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
static inline HSTexture* hs_graphics_create_texture(HSGraphics* gfx, u8 slot, u32 w, u32 h, const u8* d) { return NULL; }
static inline void hs_graphics_update_texture(HSGraphics* gfx, u8 slot, const u8* data) {}

#endif

#endif
