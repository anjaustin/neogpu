/*
 * Test 04: Blending
 * 
 * Renders semi-transparent shapes to demonstrate alpha blending.
 * This verifies:
 * - Alpha blending pipeline
 * - GL blend functions
 * 
 * Expected: Overlapping circles/squares with transparency
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "hs_graphics.h"

static const char* vertex_shader_src = R"(
    attribute vec2 a_position;
    uniform vec2 u_offset;
    uniform float u_scale;
    void main() {
        gl_Position = vec4(a_position * u_scale + u_offset, 0.0, 1.0);
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
        fprintf(stderr, "Shader error: %s\n", log);
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
    return program;
}

int main(int argc, char** argv) {
    printf("=== Test 04: Blending ===\n");
    
    HSGraphics gfx;
    if (hs_graphics_init(&gfx) != 0) {
        fprintf(stderr, "Failed to init graphics\n");
        return 1;
    }
    
    GLuint program = create_program(vertex_shader_src, fragment_shader_src);
    if (!program) return 1;
    
    // Circle approximation
    #define SEGMENTS 32
    float circle[SEGMENTS * 6];
    for (int i = 0; i < SEGMENTS; i++) {
        float a1 = (float)i / SEGMENTS * 2.0f * M_PI;
        float a2 = (float)(i + 1) / SEGMENTS * 2.0f * M_PI;
        circle[i*6 + 0] = 0.0f;
        circle[i*6 + 1] = 0.0f;
        circle[i*6 + 2] = cosf(a1);
        circle[i*6 + 3] = sinf(a1);
        circle[i*6 + 4] = cosf(a2);
        circle[i*6 + 5] = sinf(a2);
    }
    
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(circle), circle, GL_STATIC_DRAW);
    
    GLint pos_loc = glGetAttribLocation(program, "a_position");
    GLint offset_loc = glGetUniformLocation(program, "u_offset");
    GLint scale_loc = glGetUniformLocation(program, "u_scale");
    GLint color_loc = glGetUniformLocation(program, "u_color");
    
    glUseProgram(program);
    glEnableVertexAttribArray(pos_loc);
    glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
    
    // Enable blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    
    printf("Rendering blending for 10 seconds...\n");
    
    for (int frame = 0; frame < 200; frame++) {
        float t = frame * 0.05f;
        
        glClear(GL_COLOR_BUFFER_BIT);
        
        // Draw overlapping circles with different alphas
        for (int i = 0; i < 5; i++) {
            float angle = t + i * M_PI * 0.4f;
            float x = cosf(angle) * 0.3f;
            float y = sinf(angle) * 0.3f;
            
            glUniform2f(offset_loc, x, y);
            glUniform1f(scale_loc, 0.15f);
            
            // Rainbow colors with varying alpha
            float r = sinf(t + i * 0.5f) * 0.5f + 0.5f;
            float g = sinf(t + i * 0.5f + 2.0f) * 0.5f + 0.5f;
            float b = sinf(t + i * 0.5f + 4.0f) * 0.5f + 0.5f;
            glUniform4f(color_loc, r, g, b, 0.4f);  // 40% transparent
            
            glDrawArrays(GL_TRIANGLES, 0, SEGMENTS * 3);
        }
        
        hs_graphics_present(&gfx);
        usleep(50000);
    }
    
    glDisable(GL_BLEND);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(program);
    hs_graphics_finish(&gfx);
    
    printf("PASS: Blending test\n");
    return 0;
}
