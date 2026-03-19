/*
 * GLES 3.1 Compute Shader Test
 *
 * Tests if GLES 3.1 compute shaders work on Pi4
 *
 * Build:
 * gcc -O3 -march=armv8-a+simd+fp16 -mtune=cortex-a72 -ffast-math \
 *     -funroll-loops -DNDEBUG \
 *     /tmp/test_gles_compute.c -o /tmp/test_gles_compute \
 *     -lGLESv3 -lEGL -lm -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>

static uint64_t ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(void) {
    printf("=== GLES 3.1 Compute Shader Test ===\n\n");
    
    // EGL setup
    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (display == EGL_NO_DISPLAY) {
        fprintf(stderr, "Failed to get EGL display\n");
        return 1;
    }
    printf("Got EGL display\n");
    
    EGLint major, minor;
    if (!eglInitialize(display, &major, &minor)) {
        fprintf(stderr, "Failed to initialize EGL\n");
        return 1;
    }
    printf("EGL version: %d.%d\n", major, minor);
    
    // Choose config
    EGLint config_attribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
        EGL_NONE
    };
    
    EGLConfig config;
    EGLint num_configs;
    if (!eglChooseConfig(display, config_attribs, &config, 1, &num_configs) || num_configs == 0) {
        fprintf(stderr, "Failed to choose EGL config\n");
        return 1;
    }
    printf("Found %d configs\n", num_configs);
    
    // Bind GLES API
    if (!eglBindAPI(EGL_OPENGL_ES_API)) {
        fprintf(stderr, "Failed to bind GLES API\n");
        return 1;
    }
    printf("Bound GLES API\n");
    
    // Create pbuffer surface
    EGLint pbuffer_attribs[] = {
        EGL_WIDTH, 256,
        EGL_HEIGHT, 256,
        EGL_NONE
    };
    
    EGLSurface surface = eglCreatePbufferSurface(display, config, pbuffer_attribs);
    if (surface == EGL_NO_SURFACE) {
        fprintf(stderr, "Failed to create pbuffer: 0x%x\n", eglGetError());
        return 1;
    }
    printf("Created pbuffer surface\n");
    
    // Create context with GLES 3.1
    EGLint context_attribs[] = {
        EGL_CONTEXT_MAJOR_VERSION, 3,
        EGL_CONTEXT_MINOR_VERSION, 1,
        EGL_NONE
    };
    
    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attribs);
    if (context == EGL_NO_CONTEXT) {
        fprintf(stderr, "Failed to create context: 0x%x\n", eglGetError());
        return 1;
    }
    printf("Created GLES 3.1 context\n");
    
    if (!eglMakeCurrent(display, surface, surface, context)) {
        fprintf(stderr, "Failed to make context current\n");
        return 1;
    }
    printf("Context is current\n");
    
    // Check GLES version
    const char* gl_version = (const char*)glGetString(GL_VERSION);
    printf("GL version: %s\n", gl_version);
    
    // Check compute shader support
    GLint compute_supported = 0;
    glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &compute_supported);
    printf("Max compute work group invocations: %d\n", compute_supported);
    
    if (compute_supported == 0) {
        printf("ERROR: Compute shaders not supported!\n");
        eglTerminate(display);
        return 1;
    }
    
    // Simple compute shader that multiplies by 2
    const char* compute_src = 
        "#version 310 es\n"
        "precision highp float;\n"
        "layout(local_size_x = 8) in;\n"
        "layout(std430, binding = 0) readonly buffer InputBuffer { float inputs[]; };\n"
        "layout(std430, binding = 1) writeonly buffer OutputBuffer { float outputs[]; };\n"
        "void main() {\n"
        "    uint idx = gl_GlobalInvocationID.x;\n"
        "    outputs[idx] = inputs[idx] * 2.0;\n"
        "}\n";
    
    GLuint program = glCreateProgram();
    GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(cs, 1, &compute_src, NULL);
    glCompileShader(cs);
    
    GLint compiled;
    glGetShaderiv(cs, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        char log[512];
        glGetShaderInfoLog(cs, sizeof(log), NULL, log);
        fprintf(stderr, "Shader compile failed: %s\n", log);
        return 1;
    }
    printf("Shader compiled!\n");
    
    glAttachShader(program, cs);
    glLinkProgram(program);
    
    GLint linked;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[512];
        glGetProgramInfoLog(program, sizeof(log), NULL, log);
        fprintf(stderr, "Program link failed: %s\n", log);
        return 1;
    }
    printf("Program linked!\n");
    
    // Create buffers
    #define BUF_SIZE 256
    float input_data[BUF_SIZE];
    float output_data[BUF_SIZE];
    
    for (int i = 0; i < BUF_SIZE; i++) {
        input_data[i] = (float)i;
        output_data[i] = 0;
    }
    
    GLuint ssbo[2];
    glGenBuffers(2, ssbo);
    
    // Input buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(input_data), input_data, GL_STATIC_DRAW);
    
    // Output buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[1]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(output_data), NULL, GL_STATIC_DRAW);
    
    // Run compute
    glUseProgram(program);
    
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[0]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo[1]);
    
    uint64_t start = ns_now();
    glDispatchCompute(BUF_SIZE / 8, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    uint64_t end = ns_now();
    
    printf("GPU time: %.3f ms\n", (end - start) / 1000000.0);
    
    // Read back
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[1]);
    void* map = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(output_data), GL_MAP_READ_BIT);
    memcpy(output_data, map, sizeof(output_data));
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    
    // Verify
    int errors = 0;
    for (int i = 0; i < BUF_SIZE; i++) {
        float expected = input_data[i] * 2.0f;
        if (output_data[i] != expected) {
            errors++;
            if (errors <= 3) {
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
    glDeleteBuffers(2, ssbo);
    glDeleteProgram(program);
    glDeleteShader(cs);
    eglDestroySurface(display, surface);
    eglDestroyContext(display, context);
    eglTerminate(display);
    
    return errors;
}
