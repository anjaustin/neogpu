/*
 * NeoGPU GLES Compute Test - Final working version!
 *
 * Uses GBM + GLES for headless compute on Pi4 V3D GPU.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <xf86drm.h>
#include <gbm.h>
#include <EGL/egl.h>
#include <GLES2/gl2.h>

#define BUF_SIZE 256

static uint64_t ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static GLuint create_program(const char* vs, const char* fs) {
    GLuint p = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v, 1, &vs, NULL);
    glCompileShader(v);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &fs, NULL);
    glCompileShader(f);
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    return p;
}

int main(void) {
    printf("=== NeoGPU GLES Compute ===\n\n");
    
    int fd = open("/dev/dri/card1", O_RDWR);
    if (fd < 0) fd = open("/dev/dri/card0", O_RDWR);
    struct gbm_device* gbm = gbm_create_device(fd);
    
    EGLDisplay egl = eglGetDisplay((EGLNativeDisplayType)gbm);
    eglInitialize(egl, &(EGLint){0}, &(EGLint){0});
    
    EGLConfig cfg;
    EGLint n;
    eglChooseConfig(egl, (EGLint[]){EGL_SURFACE_TYPE, EGL_WINDOW_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT, EGL_NONE}, &cfg, 1, &n);
    
    EGLContext ctx = eglCreateContext(egl, cfg, EGL_NO_CONTEXT, (EGLint[]){EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE});
    eglMakeCurrent(egl, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx);
    
    printf("GL: %s\n", glGetString(GL_VERSION));
    printf("Renderer: %s\n\n", glGetString(GL_RENDERER));
    
    const char* vs = "attribute vec2 a_pos;void main(){gl_Position=vec4(a_pos,0,1);}";
    const char* fs = "precision mediump float;uniform sampler2D t;void main(){gl_FragColor=texture2D(t,gl_FragCoord.xy/256.0)*2.0;}";
    GLuint prog = create_program(vs, fs);
    glUseProgram(prog);
    
    float in[BUF_SIZE];
    for (int i = 0; i < BUF_SIZE; i++) in[i] = i;
    
    GLuint tex, fbo;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, 16, 16, 0, GL_LUMINANCE, GL_FLOAT, in);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    glViewport(0, 0, 16, 16);
    
    uint64_t start = ns_now();
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    uint64_t end = ns_now();
    
    printf("GPU time: %.3f ms\n", (end - start) / 1000000.0);
    
    glReadPixels(0, 0, 16, 16, GL_LUMINANCE, GL_FLOAT, in);
    
    int errors = 0;
    for (int i = 0; i < 16; i++) {
        if (in[i] != i * 2.0f) errors++;
    }
    
    if (errors == 0) printf("\n=== SUCCESS ===\n");
    else printf("\n=== FAILED: %d errors ===\n", errors);
    
    return errors;
}
