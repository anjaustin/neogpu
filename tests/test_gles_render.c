/*
 * GLES 2.0 Compute via Rendering Test
 *
 * Uses fragment shader rendering to texture to perform computation.
 * This is a fallback when compute shaders aren't available.
 *
 * Build on Pi:
 * gcc -O3 -march=armv8-a+simd+fp16 -mtune=cortex-a72 -ffast-math \
 *     -funroll-loops -DNDEBUG \
 *     ~/001/neogpu/tests/test_gles_render.c -o /tmp/test_gles_render \
 *     -lGLESv2 -lEGL -lm -lpthread
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
        fprintf(stderr, "Shader compile error: %s\n", log);
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
        fprintf(stderr, "Program link error: %s\n", log);
        return 0;
    }
    
    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

int main(void) {
    printf("=== GLES 2.0 Render-to-Texture Compute Test ===\n\n");
    
    // EGL setup
    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (display == EGL_NO_DISPLAY) {
        fprintf(stderr, "Failed to get EGL display\n");
        return 1;
    }
    
    EGLint major, minor;
    if (!eglInitialize(display, &major, &minor)) {
        fprintf(stderr, "Failed to init EGL\n");
        return 1;
    }
    printf("EGL: %d.%d\n", major, minor);
    
    // Config for GLES2
    EGLint config_attribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_NONE
    };
    
    EGLConfig config;
    EGLint num_configs;
    if (!eglChooseConfig(display, config_attribs, &config, 1, &num_configs) || num_configs == 0) {
        fprintf(stderr, "No config\n");
        return 1;
    }
    
    eglBindAPI(EGL_OPENGL_ES_API);
    
    // Pbuffer
    EGLint pbuffer_attribs[] = {
        EGL_WIDTH, 256,
        EGL_HEIGHT, 256,
        EGL_NONE
    };
    
    EGLSurface surface = eglCreatePbufferSurface(display, config, pbuffer_attribs);
    if (surface == EGL_NO_SURFACE) {
        fprintf(stderr, "No pbuffer: 0x%x\n", eglGetError());
        return 1;
    }
    
    // GLES2 context
    EGLint context_attribs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 2,
        EGL_NONE
    };
    
    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attribs);
    if (context == EGL_NO_CONTEXT) {
        fprintf(stderr, "No context: 0x%x\n", eglGetError());
        return 1;
    }
    
    if (!eglMakeCurrent(display, surface, surface, context)) {
        fprintf(stderr, "MakeCurrent failed\n");
        return 1;
    }
    
    printf("GL: %s\n", glGetString(GL_VERSION));
    
    // Vertex shader - simple fullscreen quad
    const char* vs =
        "attribute vec2 a_pos;\n"
        "void main() {\n"
        "    gl_Position = vec4(a_pos, 0.0, 1.0);\n"
        "}\n";
    
    // Fragment shader - multiply input by 2
    const char* fs =
        "precision mediump float;\n"
        "uniform sampler2D u_tex;\n"
        "void main() {\n"
        "    vec4 c = texture2D(u_tex, gl_FragCoord.xy / 256.0);\n"
        "    gl_FragColor = vec4(c.r * 2.0, 0.0, 0.0, 1.0);\n"
        "}\n";
    
    GLuint program = create_program(vs, fs);
    if (!program) {
        printf("FAILED to create program\n");
        return 1;
    }
    printf("Program created\n");
    
    // Setup geometry (fullscreen quad)
    float quad[] = {
        -1, -1,
         1, -1,
        -1,  1,
         1,  1,
    };
    
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    
    GLint pos_loc = glGetAttribLocation(program, "a_pos");
    glEnableVertexAttribArray(pos_loc);
    glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
    
    // Create input texture
    float input_data[BUF_SIZE];
    for (int i = 0; i < BUF_SIZE; i++) {
        input_data[i] = (float)i;
    }
    
    GLuint tex_in;
    glGenTextures(1, &tex_in);
    glBindTexture(GL_TEXTURE_2D, tex_in);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, 16, 16, 0, GL_LUMINANCE, GL_FLOAT, input_data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
    // Create output texture (framebuffer)
    GLuint tex_out;
    glGenTextures(1, &tex_out);
    glBindTexture(GL_TEXTURE_2D, tex_out);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, 16, 16, 0, GL_LUMINANCE, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_out, 0);
    
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        printf("FBO incomplete\n");
        return 1;
    }
    printf("FBO ready\n");
    
    // Render
    glUseProgram(program);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, 16, 16);
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_in);
    glUniform1i(glGetUniformLocation(program, "u_tex"), 0);
    
    uint64_t start = ns_now();
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    uint64_t end = ns_now();
    
    printf("GPU time: %.3f ms\n", (end - start) / 1000000.0);
    
    // Read back
    float output_data[BUF_SIZE];
    glReadPixels(0, 0, 16, 16, GL_LUMINANCE, GL_FLOAT, output_data);
    
    // Verify
    int errors = 0;
    for (int i = 0; i < 16; i++) {
        float expected = input_data[i] * 2.0f;
        if (output_data[i] != expected) {
            errors++;
            if (errors <= 5) {
                printf("ERROR at %d: got %f, expected %f\n", i, output_data[i], expected);
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
    
    eglDestroySurface(display, surface);
    eglDestroyContext(display, context);
    eglTerminate(display);
    
    return errors;
}
