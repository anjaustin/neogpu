/*
 * Test 06: Ray Cast Spheres
 * 
 * Renders spheres using ray casting in fragment shader.
 * This verifies:
 * - Ray-sphere intersection
 * - Lighting and shading
 * - FPS counter overlay
 * 
 * Expected: Multiple colored spheres with lighting
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include "hs_graphics.h"
#include "hs_text.h"

static HSFont g_font;
static bool g_font_loaded = false;

static const char* vertex_shader_src = R"(
    attribute vec2 a_position;
    varying vec2 v_uv;
    
    void main() {
        v_uv = a_position * 0.5 + 0.5;
        gl_Position = vec4(a_position, 0.0, 1.0);
    }
)";

static const char* fragment_shader_src = R"(
    precision mediump float;
    varying vec2 v_uv;
    uniform float u_time;
    uniform vec2 u_resolution;
    
    #define MAX_SPHERES 3
    
    float intersect_sphere(vec3 ro, vec3 rd, vec3 center, float radius) {
        vec3 oc = ro - center;
        float b = dot(oc, rd);
        float c = dot(oc, oc) - radius * radius;
        float h = b * b - c;
        if (h < 0.0) return -1.0;
        return -b - sqrt(h);
    }
    
    void main() {
        vec2 uv = (v_uv * 2.0 - 1.0) * vec2(u_resolution.x / u_resolution.y, 1.0);
        
        vec3 ro = vec3(0.0, 0.0, 3.5);
        vec3 rd = normalize(vec3(uv, -1.5));
        
        float t = 1000.0;
        vec3 color = vec3(0.1, 0.1, 0.15);
        
        vec3 spheres[3];
        float radii[3];
        vec3 colors[3];
        
        spheres[0] = vec3(-1.0, 0.0, 0.0);
        radii[0] = 0.8;
        colors[0] = vec3(1.0, 0.3, 0.3);
        
        spheres[1] = vec3(1.0, 0.0, 0.0);
        radii[1] = 0.8;
        colors[1] = vec3(0.3, 0.3, 1.0);
        
        spheres[2] = vec3(0.0, -0.3, 0.5);
        radii[2] = 0.5;
        colors[2] = vec3(0.3, 1.0, 0.3);
        
        spheres[0].x += sin(u_time * 0.7) * 0.5;
        spheres[1].x += cos(u_time * 0.7) * 0.5;
        
        vec3 light = normalize(vec3(0.5, 0.8, 1.0));
        
        for (int i = 0; i < MAX_SPHERES; i++) {
            float ts = intersect_sphere(ro, rd, spheres[i], radii[i]);
            if (ts > 0.0 && ts < t) {
                t = ts;
                vec3 p = ro + rd * ts;
                vec3 n = normalize(p - spheres[i]);
                float diff = max(dot(n, light), 0.0);
                float amb = 0.3;
                color = colors[i] * (amb + diff * 0.7);
            }
        }
        
        gl_FragColor = vec4(color, 1.0);
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
    printf("=== Test 06: Ray Cast Spheres ===\n");
    
    HSGraphics gfx;
    if (hs_graphics_init(&gfx) != 0) {
        fprintf(stderr, "Failed to init graphics\n");
        return 1;
    }
    
    printf("Screen: %dx%d\n", gfx.screen_width, gfx.screen_height);
    
    g_font_loaded = hs_font_load(&g_font, "src/medodica_font");
    if (g_font_loaded) {
        hs_font_upload_texture(&g_font);
        printf("Font loaded successfully\n");
    }
    
    GLuint program = create_program(vertex_shader_src, fragment_shader_src);
    if (!program) return 1;
    
    float quad[] = {
        -1, -1,
         1, -1,
        -1,  1,
         1, -1,
         1,  1,
        -1,  1,
    };
    
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    
    GLint pos_loc = glGetAttribLocation(program, "a_position");
    GLint time_loc = glGetUniformLocation(program, "u_time");
    GLint res_loc = glGetUniformLocation(program, "u_resolution");
    
    glUseProgram(program);
    glEnableVertexAttribArray(pos_loc);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
    
    glUniform2f(res_loc, (float)gfx.screen_width, (float)gfx.screen_height);
    
    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);
    int frame = 0;
    float fps = 0.0f;
    
    printf("Rendering ray cast spheres for 15 seconds...\n");
    printf("FPS will be shown on screen\n");
    
    while (frame < 300) {
        float t = frame * 0.05f;
        
        glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        glUniform1f(time_loc, t);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        
        hs_graphics_present(&gfx);
        
        if (g_font_loaded && frame > 0) {
            char fps_buf[32];
            snprintf(fps_buf, sizeof(fps_buf), "FPS: %.1f", fps);
            glDisable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            hs_font_render_text(&g_font, fps_buf, -0.95f, 0.85f, 0.04f, 1.0f, 1.0f, 0.0f, 1.0f);
            glDisable(GL_BLEND);
        }
        
        clock_gettime(CLOCK_MONOTONIC, &now);
        double elapsed = (now.tv_sec - start.tv_sec) + (now.tv_nsec - start.tv_nsec) / 1e9;
        if (elapsed > 0.5 && frame % 30 == 0) {
            fps = frame / elapsed;
            printf("FPS: %.1f\n", fps);
        }
        
        frame++;
    }
    
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(program);
    hs_graphics_finish(&gfx);
    
    printf("PASS: Ray cast spheres test\n");
    return 0;
}
