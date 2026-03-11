/*
 * Test 05: 3D Cube
 * 
 * Renders a rotating 3D cube with perspective projection.
 * This verifies:
 * - 3D perspective projection matrix
 * - 3D vertex transformations
 * - Depth testing
 * 
 * Expected: Rotating colored cube
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "hs_graphics.h"

static const char* vertex_shader_src = R"(
    attribute vec3 a_position;
    attribute vec3 a_color;
    varying vec3 v_color;
    uniform mat4 u_matrix;
    
    void main() {
        gl_Position = u_matrix * vec4(a_position, 1.0);
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
    (void)argc;
    (void)argv;
    printf("=== Test 05: 3D Cube ===\n");
    
    HSGraphics gfx;
    if (hs_graphics_init(&gfx) != 0) {
        fprintf(stderr, "Failed to init graphics\n");
        return 1;
    }
    
    printf("Screen: %dx%d\n", gfx.screen_width, gfx.screen_height);
    
    GLuint program = create_program(vertex_shader_src, fragment_shader_src);
    if (!program) return 1;
    
    float cube_vertices[] = {
        -0.5, -0.5, -0.5,  1, 0, 0,
         0.5, -0.5, -0.5,  1, 0, 0,
         0.5,  0.5, -0.5,  1, 0, 0,
        -0.5,  0.5, -0.5,  1, 0, 0,
        -0.5, -0.5,  0.5,  0, 1, 0,
         0.5, -0.5,  0.5,  0, 1, 0,
         0.5,  0.5,  0.5,  0, 1, 0,
        -0.5,  0.5,  0.5,  0, 1, 0,
    };
    
    unsigned short cube_indices[] = {
        0, 1, 2,  0, 2, 3,
        4, 5, 6,  4, 6, 7,
        0, 1, 5,  0, 5, 4,
        2, 3, 7,  2, 7, 6,
        0, 3, 7,  0, 7, 4,
        1, 2, 6,  1, 6, 5,
    };
    
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertices), cube_vertices, GL_STATIC_DRAW);
    
    GLuint ibo;
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cube_indices), cube_indices, GL_STATIC_DRAW);
    
    GLint pos_loc = glGetAttribLocation(program, "a_position");
    GLint color_loc = glGetAttribLocation(program, "a_color");
    GLint matrix_loc = glGetUniformLocation(program, "u_matrix");
    
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    
    printf("Rendering rotating cube for 15 seconds...\n");
    
    for (int frame = 0; frame < 300; frame++) {
        float t = frame * 0.05f;
        
        float aspect = (float)gfx.screen_width / gfx.screen_height;
        mat4 proj = m4_perspective(1.0f, aspect, 0.1f, 10.0f);
        
        mat4 view = m4_translation(0, 0, -2.5f);
        
        mat4 rot_y = m4_rotation_y(t);
        mat4 rot_x = m4_rotation_x(t * 0.7f);
        
        mat4 model = m4_multiply(rot_y, rot_x);
        mat4 mvp = m4_multiply(proj, m4_multiply(view, model));
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        glUseProgram(program);
        
        float mat_arr[16];
        m4_to_array(mvp, mat_arr);
        glUniformMatrix4fv(matrix_loc, 1, GL_FALSE, mat_arr);
        
        glEnableVertexAttribArray(pos_loc);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glVertexAttribPointer(pos_loc, 3, GL_FLOAT, GL_FALSE, 24, 0);
        
        glEnableVertexAttribArray(color_loc);
        glVertexAttribPointer(color_loc, 3, GL_FLOAT, GL_FALSE, 24, (void*)12);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, 0);
        
        hs_graphics_present(&gfx);
        usleep(50000);
    }
    
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ibo);
    glDeleteProgram(program);
    hs_graphics_finish(&gfx);
    
    printf("PASS: 3D Cube test\n");
    return 0;
}
