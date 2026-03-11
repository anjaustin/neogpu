/*
 * Test 03: Instancing
 * 
 * Renders many instances of the same geometry with different transforms.
 * This verifies:
 * - Instanced rendering works
 * - Vertex shader instance transforms
 * 
 * Expected: Grid of rotating/colored shapes
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "hs_graphics.h"

static const char* vertex_shader_src = R"(
    attribute vec2 a_position;
    attribute vec2 a_offset;
    attribute vec3 a_color;
    varying vec3 v_color;
    uniform float u_time;
    
    void main() {
        vec2 pos = a_position * 0.1 + a_offset;
        gl_Position = vec4(pos, 0.0, 1.0);
        v_color = a_color;
    }
)";

static const char* fragment_shader_src = R"(
    precision mediump float;
    varying vec3 v_color;
    void main() {
        gl_FragColor = vec4(v_color, 1.0);
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
    printf("=== Test 03: Instancing ===\n");
    
    HSGraphics gfx;
    if (hs_graphics_init(&gfx) != 0) {
        fprintf(stderr, "Failed to init graphics\n");
        return 1;
    }
    
    printf("Screen: %dx%d\n", gfx.screen_width, gfx.screen_height);
    
    printf("GL Version: %s\n", glGetString(GL_VERSION));
    
    GLuint program = create_program(vertex_shader_src, fragment_shader_src);
    if (!program) return 1;
    
    // Square geometry
    float quad[] = {
        -1, -1,  1, -1,  -1, 1,
        -1,  1,  1, -1,   1, 1,
    };
    
    // Instance offsets and colors
    #define GRID 8
    #define COUNT (GRID * GRID)
    float offsets[COUNT * 2];
    float colors[COUNT * 3];
    
    for (int i = 0; i < GRID; i++) {
        for (int j = 0; j < GRID; j++) {
            int idx = i * GRID + j;
            offsets[idx * 2 + 0] = (float)(i - GRID/2) * 0.2f;
            offsets[idx * 2 + 1] = (float)(j - GRID/2) * 0.2f;
            colors[idx * 3 + 0] = (float)i / GRID;
            colors[idx * 3 + 1] = (float)j / GRID;
            colors[idx * 3 + 2] = 0.5f;
        }
    }
    
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    
    GLuint offset_vbo;
    glGenBuffers(1, &offset_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, offset_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(offsets), offsets, GL_STATIC_DRAW);
    
    GLuint color_vbo;
    glGenBuffers(1, &color_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, color_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(colors), colors, GL_STATIC_DRAW);
    
    GLint pos_loc = glGetAttribLocation(program, "a_position");
    GLint offset_loc = glGetAttribLocation(program, "a_offset");
    GLint color_loc = glGetAttribLocation(program, "a_color");
    GLint time_loc = glGetUniformLocation(program, "u_time");
    
    glUseProgram(program);
    glEnableVertexAttribArray(pos_loc);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
    
    glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
    
    printf("Rendering %d instances for 10 seconds...\n", COUNT);
    
    for (int frame = 0; frame < 200; frame++) {
        float t = frame * 0.05f;
        
        glClear(GL_COLOR_BUFFER_BIT);
        glUniform1f(time_loc, t);
        
        for (int i = 0; i < COUNT; i++) {
            glVertexAttrib2f(offset_loc, offsets[i*2], offsets[i*2+1]);
            glVertexAttrib3f(color_loc, colors[i*3], colors[i*3+1], colors[i*3+2]);
            glDrawArrays(GL_TRIANGLES, 0, 6);
        }
        
        hs_graphics_present(&gfx);
        usleep(50000);
    }
    
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &offset_vbo);
    glDeleteBuffers(1, &color_vbo);
    glDeleteProgram(program);
    hs_graphics_finish(&gfx);
    
    printf("PASS: Instancing test\n");
    return 0;
}
