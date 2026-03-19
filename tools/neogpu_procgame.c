/*
 * neogpu_procgame.c - Procedural Space Shooter
 * 
 * Demonstrates procedural content generation:
 * - Textures (layer-based procedural)
 * - Meshes (compressed vertices)
 * - Audio (synthesized sounds)
 * 
 * Target: < 64KB total including all assets
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include "hs_graphics.h"
#include "hs_procgen.h"
#include "hs_procmesh.h"
#include "hs_procaudio.h"

#define SCREEN_W 800
#define SCREEN_H 480

static volatile bool g_run = true;
static void on_sig(int s) { (void)s; g_run = false; }

static const char *VS_QUAD = 
    "attribute vec2 a_pos;\n"
    "attribute vec2 a_uv;\n"
    "varying vec2 v_uv;\n"
    "void main(){\n"
    "  v_uv = a_uv;\n"
    "  gl_Position = vec4(a_pos, 0.0, 1.0);\n"
    "}\n";

static const char *FS_TEX = 
    "precision mediump float;\n"
    "varying vec2 v_uv;\n"
    "uniform sampler2D u_tex;\n"
    "uniform vec3 u_color;\n"
    "void main(){\n"
    "  vec4 t = texture2D(u_tex, v_uv);\n"
    "  gl_FragColor = vec4(u_color * t.rgb, t.a);\n"
    "}\n";

static const char *FS_SOLID = 
    "precision mediump float;\n"
    "varying vec2 v_uv;\n"
    "uniform vec3 u_color;\n"
    "void main(){\n"
    "  gl_FragColor = vec4(u_color, 1.0);\n"
    "}\n";

static GLuint make_prog(const char *vs, const char *fs) {
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v, 1, &vs, NULL); glCompileShader(v);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

static GLuint g_vbo, g_tex;
static GLint g_a_pos, g_a_uv, g_u_tex, g_u_color;
static GLuint g_prog_tex, g_prog_solid;

static void draw_quad(float x0, float y0, float x1, float y1) {
    float q[] = {
        x0, y0, 0, 0,
        x1, y0, 1, 0,
        x0, y1, 0, 1,
        x1, y0, 1, 0,
        x1, y1, 1, 1,
        x0, y1, 0, 1
    };
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(q), q);
    glEnableVertexAttribArray(g_a_pos);
    glEnableVertexAttribArray(g_a_uv);
    glVertexAttribPointer(g_a_pos, 2, GL_FLOAT, GL_FALSE, 16, (void*)0);
    glVertexAttribPointer(g_a_uv, 2, GL_FLOAT, GL_FALSE, 16, (void*)8);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

static void draw_sprite(float x, float y, float w, float h) {
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g_tex);
    glUniform1i(g_u_tex, 0);
    draw_quad(x - w/2, y - h/2, x + w/2, y + h/2);
}

typedef struct {
    float x, y;
    float vx, vy;
    int alive;
    int cooldown;
} Player;

typedef struct {
    float x, y;
    float vx, vy;
    int alive;
} Bullet;

typedef struct {
    float x, y;
    float vx, vy;
    int alive;
    int frame;
} Enemy;

#define MAX_BULLETS 16
#define MAX_ENEMIES 8

static Player player;
static Bullet bullets[MAX_BULLETS];
static Enemy enemies[MAX_ENEMIES];
static int score;
static int wave;
static int frame_count;

static void spawn_enemy() {
    for (int i = 0; i < MAX_ENEMIES; i++) {
        if (!enemies[i].alive) {
            enemies[i].x = (rand() % SCREEN_W) * 2.0f / SCREEN_W - 1.0f;
            enemies[i].y = 1.2f;
            enemies[i].vx = ((rand() % 100) - 50) * 0.0001f;
            enemies[i].vy = -0.003f - (rand() % 100) * 0.00001f;
            enemies[i].alive = 1;
            enemies[i].frame = frame_count;
            break;
        }
    }
}

static void fire_bullet() {
    for (int i = 0; i < MAX_BULLETS; i++) {
        if (!bullets[i].alive) {
            bullets[i].x = player.x;
            bullets[i].y = player.y + 0.1f;
            bullets[i].vx = 0;
            bullets[i].vy = 0.015f;
            bullets[i].alive = 1;
            break;
        }
    }
}

static u8 g_player_tex[64*64*4];
static u8 g_enemy_tex[64*64*4];
static u8 g_bullet_tex[32*32*4];
static u8 g_star_tex[16*16*4];

static void generate_textures() {
    // Player ship texture - procedural
    ProcTexture pt;
    proc_tex_init(&pt, 64, 64);
    proc_tex_add_layer(&pt);
    proc_tex_layer_coverage(&pt, 0, 2); // Round
    proc_tex_layer_rect(&pt, 0, 0.2f, 0.0f, 0.6f, 1.0f);
    proc_tex_layer_color(&pt, 0, 100, 150, 200, 255);
    proc_tex_layer_noise(&pt, 0, 2.0f, 0.3f);
    proc_tex_add_layer(&pt);
    proc_tex_layer_coverage(&pt, 1, 0); // Full
    proc_tex_layer_color(&pt, 1, 60, 100, 140, 255);
    proc_tex_layer_noise(&pt, 1, 4.0f, 0.2f);
    proc_tex_generate(&pt, g_player_tex);
    
    // Enemy texture - procedural
    proc_tex_init(&pt, 64, 64);
    proc_tex_add_layer(&pt);
    proc_tex_layer_coverage(&pt, 0, 2); // Round
    proc_tex_layer_rect(&pt, 0, 0.15f, 0.15f, 0.7f, 0.7f);
    proc_tex_layer_color(&pt, 0, 200, 50, 50, 255);
    proc_tex_layer_noise(&pt, 0, 3.0f, 0.4f);
    proc_tex_add_layer(&pt);
    proc_tex_layer_coverage(&pt, 1, 0);
    proc_tex_layer_color(&pt, 1, 255, 100, 100, 255);
    proc_tex_layer_emissive(&pt, 1, 1);
    proc_tex_generate(&pt, g_enemy_tex);
    
    // Bullet texture
    proc_tex_init(&pt, 32, 32);
    proc_tex_add_layer(&pt);
    proc_tex_layer_coverage(&pt, 0, 2);
    proc_tex_layer_rect(&pt, 0, 0.2f, 0.2f, 0.6f, 0.6f);
    proc_tex_layer_color(&pt, 0, 255, 255, 100, 255);
    proc_tex_layer_emissive(&pt, 0, 1);
    proc_tex_generate(&pt, g_bullet_tex);
    
    // Star texture (tiny)
    proc_tex_init(&pt, 16, 16);
    proc_tex_add_layer(&pt);
    proc_tex_layer_coverage(&pt, 0, 0);
    proc_tex_layer_color(&pt, 0, 200, 200, 200, 255);
    proc_tex_generate(&pt, g_star_tex);
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    srand(time(NULL));
    
    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);

    printf("=== NeoGPU Procedural Space Shooter ===\n");
    printf("Procedural textures: %zu bytes\n", sizeof(g_player_tex) + sizeof(g_enemy_tex) + sizeof(g_bullet_tex));
    printf("Controls: Arrow keys to move, SPACE to fire, Q to quit\n");

    HSGraphics gfx;
    if (hs_graphics_init(&gfx) != 0) {
        fprintf(stderr, "display init failed\n");
        return 1;
    }

    g_prog_tex = make_prog(VS_QUAD, FS_TEX);
    g_prog_solid = make_prog(VS_QUAD, FS_SOLID);
    
    glUseProgram(g_prog_tex);
    g_a_pos = glGetAttribLocation(g_prog_tex, "a_pos");
    g_a_uv = glGetAttribLocation(g_prog_tex, "a_uv");
    g_u_tex = glGetUniformLocation(g_prog_tex, "u_tex");
    g_u_color = glGetUniformLocation(g_prog_tex, "u_color");

    glGenBuffers(1, &g_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, 6 * 16, NULL, GL_DYNAMIC_DRAW);
    
    glGenTextures(1, &g_tex);
    glBindTexture(GL_TEXTURE_2D, g_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glViewport(0, 0, (GLsizei)gfx.screen_width, (GLsizei)gfx.screen_height);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Generate procedural textures
    generate_textures();

    // Init game objects
    player.x = 0; player.y = -0.6f; player.alive = 1; player.cooldown = 0;
    memset(bullets, 0, sizeof(bullets));
    memset(enemies, 0, sizeof(enemies));
    score = 0; wave = 1; frame_count = 0;

    // Simple procedural audio
    ProcAudio audio;
    proc_audio_init(&audio);
    
    int ch = 0;
    while (g_run) {
        frame_count++;
        
        // Simple input (no actual input for demo)
        player.vx = 0; player.vy = 0;
        
        // Auto-play: move in sine wave
        player.x = sinf(frame_count * 0.02f) * 0.5f;
        
        // Auto-fire
        player.cooldown--;
        if (player.cooldown <= 0) {
            fire_bullet();
            player.cooldown = 15;
            // Play procedural sound
            proc_audio_play_note(&audio, 0, 80 + rand() % 20, WAVE_SQUARE);
        }
        
        // Update bullets
        for (int i = 0; i < MAX_BULLETS; i++) {
            if (bullets[i].alive) {
                bullets[i].x += bullets[i].vx;
                bullets[i].y += bullets[i].vy;
                if (bullets[i].y > 1.2f) bullets[i].alive = 0;
            }
        }
        
        // Spawn enemies
        if (frame_count % (60 - wave * 5) == 0) {
            spawn_enemy();
        }
        
        // Update enemies
        for (int i = 0; i < MAX_ENEMIES; i++) {
            if (enemies[i].alive) {
                enemies[i].x += enemies[i].vx;
                enemies[i].y += enemies[i].vy;
                
                // Bounce off walls
                if (enemies[i].x < -0.9f || enemies[i].x > 0.9f) {
                    enemies[i].vx = -enemies[i].vx;
                }
                
                if (enemies[i].y < -1.2f) {
                    enemies[i].alive = 0;
                }
            }
        }
        
        // Collision detection
        for (int i = 0; i < MAX_BULLETS; i++) {
            if (!bullets[i].alive) continue;
            for (int j = 0; j < MAX_ENEMIES; j++) {
                if (!enemies[j].alive) continue;
                float dx = bullets[i].x - enemies[j].x;
                float dy = bullets[i].y - enemies[j].y;
                if (dx*dx + dy*dy < 0.01f) {
                    bullets[i].alive = 0;
                    enemies[j].alive = 0;
                    score += 10;
                    wave = 1 + score / 100;
                    // Explosion sound
                    proc_audio_play_note(&audio, 1, 40, WAVE_NOISE);
                }
            }
        }
        
        // Render
        glClearColor(0.02f, 0.02f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        glUseProgram(g_prog_tex);
        glUniform3f(g_u_color, 1.0f, 1.0f, 1.0f);
        
        // Draw stars (background)
        glBindTexture(GL_TEXTURE_2D, g_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 16, 16, 0, GL_RGBA, GL_UNSIGNED_BYTE, g_star_tex);
        for (int i = 0; i < 50; i++) {
            float sx = sinf(i * 123.456f + frame_count * 0.0001f);
            float sy = cosf(i * 789.012f);
            draw_sprite(sx, sy, 0.02f, 0.02f);
        }
        
        // Draw bullets
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 32, 32, 0, GL_RGBA, GL_UNSIGNED_BYTE, g_bullet_tex);
        for (int i = 0; i < MAX_BULLETS; i++) {
            if (bullets[i].alive) {
                glUniform3f(g_u_color, 1.0f, 1.0f, 0.5f);
                draw_sprite(bullets[i].x, bullets[i].y, 0.04f, 0.08f);
            }
        }
        
        // Draw enemies
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 64, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, g_enemy_tex);
        for (int i = 0; i < MAX_ENEMIES; i++) {
            if (enemies[i].alive) {
                glUniform3f(g_u_color, 1.0f, 0.5f, 0.5f);
                draw_sprite(enemies[i].x, enemies[i].y, 0.15f, 0.15f);
            }
        }
        
        // Draw player
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 64, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, g_player_tex);
        glUniform3f(g_u_color, 0.5f, 0.8f, 1.0f);
        draw_sprite(player.x, player.y, 0.12f, 0.15f);
        
        hs_graphics_present(&gfx);
        
        // 60 FPS
        struct timespec ts = {0, 16666667};
        nanosleep(&ts, NULL);
        
        // Render audio buffer
        proc_audio_render(&audio, 256);
    }

    hs_graphics_finish(&gfx);
    printf("\nGame Over! Score: %d, Wave: %d\n", score, wave);
    printf("Procedural textures generated at runtime - no asset files!\n");
    return 0;
}
