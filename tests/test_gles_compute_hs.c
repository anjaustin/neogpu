/*
 * NeoGPU GLES Compute Test - Using hs_graphics infrastructure
 *
 * Uses existing NeoGPU graphics infrastructure for GLES compute.
 *
 * Build on Pi:
 * cd ~/001/neogpu
 * gcc -O3 -march=armv8-a+simd+fp16 -mtune=cortex-a72 -ffast-math \
 *     -funroll-loops -DNDEBUG -Iinclude \
 *     tests/test_gles_compute_hs.c -o /tmp/test_gles_compute_hs \
 *     src/hs_core.o src/hs_nodes.o src/hs_gpu.o src/hs_async.o src/hs_backend_gles.o \
 *     -lGLESv2 -lgbm -ldrm -lEGL -lm -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "hs_graphics.h"

#define BUF_SIZE 256

static uint64_t ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static GLuint create_program(const char* vs_src, const char* fs_src) {
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vs_src, NULL);
    glCompileShader(vs);
    
    GLint compiled;
    glGetShaderiv(vs, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        char log[512];
        glGetShaderInfoLog(vs, sizeof(log), NULL, log);
        fprintf(stderr, "VS error: %s\n", log);
        return 0;
    }
    
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fs_src, NULL);
    glCompileShader(fs);
    
    glGetShaderiv(fs, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        char log[512];
        glGetShaderInfoLog(fs, sizeof(log), NULL, log);
        fprintf(stderr, "FS error: %s\n", log);
        return 0;
    }
    
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
    
    return program;
}

int main(void) {
    printf("=== NeoGPU GLES Compute (via hs_graphics) ===\n\n");
    
    HSGraphics gfx;
    if (hs_graphics_init(&gfx) != 0) {
        fprintf(stderr, "Graphics init failed\n");
        return 1;
    }
    printf("Graphics initialized: %dx%d\n", gfx.screen_width, gfx.screen_height);
    printf("GL: %s\n", glGetString(GL_VERSION));
    
    // Shaders for compute
    const char* vs =
        "attribute vec2 a_pos;\n"
        "varying vec2 v_uv;\n"
        "void main() {\n"
        "    v_uv = a_pos * 0.5 + 0.5;\n"
        "    gl_Position = vec4(a_pos, 0.0, 1.0);\n"
        "}\n";
    
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
        fprintf(stderr, "Program failed\n");
        return 1;
    }
    printf("Program created\n");
    
    // Geometry
    float quad[] = { -1,-1, 1,-1, -1,1, 1,1 };
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    
    // Input texture (16x16 = 256 floats)
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
    
    // FBO
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_out, 0);
    
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "FBO incomplete\n");
        return 1;
    }
    printf("FBO ready\n");
    
    // Render compute
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
    
    hs_graphics_finish(&gfx);
    
    return errors;
}
