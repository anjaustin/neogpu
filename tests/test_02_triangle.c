/*
 * Test 02: Colored Triangle
 * 
 * Renders a single triangle with vertex colors.
 * This verifies:
 * - Vertex shader working
 * - Fragment shader working  
 * - Basic geometry rendering
 * 
 * Expected: Colorful triangle in center of screen
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "hs_graphics.h"

static const char* vertex_shader_src = R"(
    attribute vec2 a_position;
    attribute vec3 a_color;
    varying vec3 v_color;
    void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
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
    (void)argc;
    (void)argv;
    printf("=== Test 02: Colored Triangle ===\n");
    
    HSGraphics gfx;
    if (hs_graphics_init(&gfx) != 0) {
        fprintf(stderr, "Failed to init graphics\n");
        return 1;
    }
    
    printf("Screen: %dx%d\n", gfx.screen_width, gfx.screen_height);
    
    GLuint program = create_program(vertex_shader_src, fragment_shader_src);
    if (!program) return 1;
    
    GLint pos_loc = glGetAttribLocation(program, "a_position");
    GLint color_loc = glGetAttribLocation(program, "a_color");
    
    // Triangle: position (x,y) + color (r,g,b)
    float vertices[] = {
        // x, y,    r, g, b
         0.0f,  0.5f,  1.0f, 0.0f, 0.0f,  // top - red
        -0.5f, -0.5f,  0.0f, 1.0f, 0.0f,  // bottom left - green
         0.5f, -0.5f,  0.0f, 0.0f, 1.0f,  // bottom right - blue
    };
    
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    glUseProgram(program);
    
    glEnableVertexAttribArray(pos_loc);
    glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 20, 0);
    
    glEnableVertexAttribArray(color_loc);
    glVertexAttribPointer(color_loc, 3, GL_FLOAT, GL_FALSE, 20, (void*)(8));
    
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    
    hs_graphics_present(&gfx);
    
    printf("Displaying triangle for 5 seconds...\n");
    sleep(5);
    
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(program);
    hs_graphics_finish(&gfx);
    
    printf("PASS: Colored triangle test\n");
    return 0;
}
