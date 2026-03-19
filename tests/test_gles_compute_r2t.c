/*
 * NeoGPU GLES Compute Test - Render-to-Texture Approach
 *
 * Uses GLES fragment shaders to perform compute via rendering to texture.
 * Works on Pi4 with GLES2/GLES3 (no compute shaders needed).
 *
 * Build on Pi:
 * cd ~/001/neogpu
 * gcc -O3 -march=armv8-a+simd+fp16 -mtune=cortex-a72 -ffast-math \
 *     -funroll-loops -DNDEBUG \
 *     tests/test_gles_compute_r2t.c -o /tmp/test_gles_compute_r2t \
 *     -lGLESv2 -lgbm -ldrm -lm -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/mman.h>
#include <time.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <gbm.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>

#define BUF_SIZE 256

static uint64_t ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static GLuint create_shader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);
    
    GLint compiled;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        char log[512];
        glGetShaderInfoLog(shader, sizeof(log), NULL, log);
        fprintf(stderr, "Shader error: %s\n", log);
        return 0;
    }
    return shader;
}

static GLuint create_program(const char* vs_src, const char* fs_src) {
    GLuint vs = create_shader(GL_VERTEX_SHADER, vs_src);
    GLuint fs = create_shader(GL_FRAGMENT_SHADER, fs_src);
    if (!vs || !fs) return 0;
    
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    
    GLint linked;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[512];
        glGetProgramInfoLog(program, sizeof(log), NULL, log);
        fprintf(stderr, "Link error: %s\n", log);
        return 0;
    }
    
    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

int main(int argc, char* argv[]) {
    printf("=== NeoGPU GLES Compute (Render-to-Texture) ===\n\n");
    
    int drm_fd = open("/dev/dri/card1", O_RDWR | O_CLOEXEC);
    if (drm_fd < 0) {
        drm_fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    }
    if (drm_fd < 0) {
        fprintf(stderr, "No DRM device\n");
        return 1;
    }
    printf("Opened DRM\n");
    
    if (drmSetMaster(drm_fd) != 0) {
        printf("Note: DRM master failed\n");
    }
    
    struct gbm_device* gbm_dev = gbm_create_device(drm_fd);
    if (!gbm_dev) {
        fprintf(stderr, "No GBM device\n");
        return 1;
    }
    printf("Created GBM\n");
    
    int width = 256, height = 256;
    
    drmModeConnector* conn = NULL;
    drmModeRes* resources = drmModeGetResources(drm_fd);
    if (resources) {
        for (int i = 0; i < resources->count_connectors; i++) {
            conn = drmModeGetConnector(drm_fd, resources->connectors[i]);
            if (conn && conn->connection == DRM_MODE_CONNECTED && conn->count_modes > 0) {
                width = conn->modes[0].hdisplay;
                height = conn->modes[0].vdisplay;
                break;
            }
            drmModeFreeConnector(conn);
        }
        drmModeFreeResources(resources);
    }
    
    printf("Display: %dx%d\n", width, height);
    
    struct gbm_surface* gbm_surf = gbm_surface_create(
        gbm_dev, width, height,
        GBM_FORMAT_XRGB8888,
        GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING
    );
    if (!gbm_surf) {
        fprintf(stderr, "No GBM surface\n");
        return 1;
    }
    
    EGLDisplay egl_display = eglGetDisplay((EGLNativeDisplayType)gbm_dev);
    if (egl_display == EGL_NO_DISPLAY) {
        fprintf(stderr, "No EGL display\n");
        return 1;
    }
    
    EGLint egl_major, egl_minor;
    if (!eglInitialize(egl_display, &egl_major, &egl_minor)) {
        fprintf(stderr, "EGL init failed\n");
        return 1;
    }
    printf("EGL: %d.%d\n", egl_major, egl_minor);
    
    EGLint config_attribs[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 0, EGL_STENCIL_SIZE, 0,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_NONE
    };
    
    EGLConfig egl_config;
    EGLint num_configs;
    if (!eglChooseConfig(egl_display, config_attribs, &egl_config, 1, &num_configs) || num_configs == 0) {
        fprintf(stderr, "No EGL config\n");
        return 1;
    }
    
    EGLint context_attribs[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
    EGLContext egl_context = eglCreateContext(egl_display, egl_config, EGL_NO_CONTEXT, context_attribs);
    if (egl_context == EGL_NO_CONTEXT) {
        fprintf(stderr, "No EGL context\n");
        return 1;
    }
    
    EGLSurface egl_surface = eglCreateWindowSurface(egl_display, egl_config, 
                                                     (EGLNativeWindowType)gbm_surf, NULL);
    if (egl_surface == EGL_NO_SURFACE) {
        fprintf(stderr, "No EGL surface: 0x%x\n", eglGetError());
        return 1;
    }
    
    if (!eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context)) {
        fprintf(stderr, "MakeCurrent failed\n");
        return 1;
    }
    
    printf("GL: %s\n", glGetString(GL_VERSION));
    
    // Vertex shader - fullscreen quad
    const char* vs =
        "attribute vec2 a_pos;\n"
        "varying vec2 v_uv;\n"
        "void main() {\n"
        "    v_uv = a_pos * 0.5 + 0.5;\n"
        "    gl_Position = vec4(a_pos, 0.0, 1.0);\n"
        "}\n";
    
    // Fragment shader - compute: multiply by 2
    // Maps 16x16 texture where each pixel = one float
    const char* fs =
        "precision mediump float;\n"
        "varying vec2 v_uv;\n"
        "uniform sampler2D u_input;\n"
        "void main() {\n"
        "    vec4 c = texture2D(u_input, v_uv);\n"
        "    gl_FragColor = vec4(c.r * 2.0, 0.0, 0.0, 1.0);\n"
        "}\n";
    
    GLuint program = create_program(vs, fs);
    if (!program) {
        fprintf(stderr, "No program\n");
        return 1;
    }
    printf("Created render program\n");
    
    // Setup geometry
    float quad[] = { -1,-1, 1,-1, -1,1, 1,1 };
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    
    // Create input texture (16x16 = 256 floats)
    float input_data[BUF_SIZE];
    for (int i = 0; i < BUF_SIZE; i++) input_data[i] = (float)i;
    
    GLuint tex_in;
    glGenTextures(1, &tex_in);
    glBindTexture(GL_TEXTURE_2D, tex_in);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, 16, 16, 0, GL_LUMINANCE, GL_FLOAT, input_data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
    // Output texture
    GLuint tex_out;
    glGenTextures(1, &tex_out);
    glBindTexture(GL_TEXTURE_2D, tex_out);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, 16, 16, 0, GL_LUMINANCE, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
    // Framebuffer
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_out, 0);
    
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "FBO incomplete\n");
        return 1;
    }
    printf("FBO ready\n");
    
    // Render-to-texture compute
    glUseProgram(program);
    
    GLint pos_loc = glGetAttribLocation(program, "a_pos");
    glEnableVertexAttribArray(pos_loc);
    glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_in);
    glUniform1i(glGetUniformLocation(program, "u_input"), 0);
    
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, 16, 16);
    
    uint64_t start = ns_now();
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    uint64_t end = ns_now();
    
    printf("GPU time: %.3f ms\n", (end - start) / 1000000.0);
    
    // Read back
    glReadPixels(0, 0, 16, 16, GL_LUMINANCE, GL_FLOAT, input_data);
    
    // Verify
    int errors = 0;
    for (int i = 0; i < 16; i++) {
        float expected = (float)i * 2.0f;
        if (input_data[i] != expected) {
            errors++;
            if (errors <= 5) {
                printf("ERROR at %d: got %f, expected %f\n", i, input_data[i], expected);
            }
        }
    }
    
    if (errors == 0) {
        printf("\n=== SUCCESS: All values correct! ===\n");
    } else {
        printf("\n=== FAILED: %d errors ===\n", errors);
    }
    
    // Cleanup
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex_in);
    glDeleteTextures(1, &tex_out);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(program);
    
    eglDestroySurface(egl_display, egl_surface);
    eglDestroyContext(egl_display, egl_context);
    gbm_surface_destroy(gbm_surf);
    gbm_device_destroy(gbm_dev);
    close(drm_fd);
    
    return errors;
}
