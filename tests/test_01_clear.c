/*
 * Test 01: Clear Screen
 * 
 * The most basic GPU test - just clear the screen to a solid color.
 * This verifies:
 * - GBM/EGL initialization works
 * - GLES context is valid
 * - Display flip works
 * 
 * Expected: Screen fills with solid color
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "hs_graphics.h"

static const char* vertex_shader_src = R"(
    attribute vec2 a_position;
    void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
    }
)";

static const char* fragment_shader_src = R"(
    precision mediump float;
    uniform vec4 u_color;
    void main() {
        gl_FragColor = u_color;
    }
)";

static GLuint compile_shader(GLenum type, const char* src) {
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
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_src);
    
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
    return program;
}

int main(int argc, char** argv) {
    printf("=== Test 01: Clear Screen ===\n");
    
    HSGraphics gfx;
    if (hs_graphics_init(&gfx) != 0) {
        fprintf(stderr, "Failed to init graphics\n");
        return 1;
    }
    
    printf("Screen: %dx%d\n", gfx.screen_width, gfx.screen_height);
    
    GLuint program = create_program(vertex_shader_src, fragment_shader_src);
    if (!program) {
        fprintf(stderr, "Failed to create program\n");
        return 1;
    }
    
    GLint pos_loc = glGetAttribLocation(program, "a_position");
    GLint color_loc = glGetUniformLocation(program, "u_color");
    
    // Full-screen quad vertices
    float vertices[] = {
        -1.0f, -1.0f,
         1.0f, -1.0f,
        -1.0f,  1.0f,
         1.0f, -1.0f,
         1.0f,  1.0f,
        -1.0f,  1.0f,
    };
    
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    glUseProgram(program);
    glEnableVertexAttribArray(pos_loc);
    glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
    
    // Cycle through colors
    float colors[][4] = {
        {1.0f, 0.0f, 0.0f, 1.0f},  // Red
        {0.0f, 1.0f, 0.0f, 1.0f},  // Green
        {0.0f, 0.0f, 1.0f, 1.0f},  // Blue
        {1.0f, 1.0f, 0.0f, 1.0f},  // Yellow
        {0.0f, 1.0f, 1.0f, 1.0f},  // Cyan
        {1.0f, 0.0f, 1.0f, 1.0f},  // Magenta
    };
    
    printf("Rendering color cycle (Ctrl+C to exit)...\n");
    
    int frame = 0;
    while (1) {
        float* color = colors[frame % 6];
        
        glClearColor(color[0], color[1], color[2], color[3]);
        glClear(GL_COLOR_BUFFER_BIT);
        
        hs_graphics_present(&gfx);
        
        usleep(500000);  // 0.5 seconds
        frame++;
    }
    
    // Cleanup (never reached in demo mode)
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(program);
    hs_graphics_finish(&gfx);
    
    printf("PASS: Clear screen test\n");
    return 0;
}
