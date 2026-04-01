/*
 * neogpu_pong.c -- Pong on NeoGPU
 * 
 * Simple pong game demonstrating the game engine.
 * Controls: Arrow keys or WASD to move paddle
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <signal.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/select.h>
#include <sys/stat.h>
#include "hs_graphics.h"
#include "hs_input.h"
#include "hs_text.h"

static HSFont g_font;
static bool g_font_loaded = false;

#define PADDLE_W 0.05f
#define PADDLE_H 0.2f
#define BALL_SIZE 0.04f
#define PADDLE_SPEED 0.03f
#define AI_SPEED 0.015f

#define SAVE_PATH "/tmp/neogpu_pong_save.bin"
#define HS_STORAGE_MAGIC 0x504F4E47  // "PONG"

typedef struct {
    u32 magic;
    int player_high_score;
    int ai_high_score;
} PongSave;

typedef struct {
    float x, y;
    float vx, vy;
} Ball;

typedef struct {
    float x, y;
    int score;
} Paddle;

static const char *VS_QUAD = 
    "attribute vec2 a_pos;\n"
    "attribute vec2 a_uv;\n"
    "varying vec2 v_uv;\n"
    "void main(){\n"
    "  v_uv = a_uv;\n"
    "  gl_Position = vec4(a_pos, 0.0, 1.0);\n"
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

static GLuint g_vbo;
static GLint g_a_pos, g_a_uv;
static GLuint g_prog;
static GLint g_u_color;

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

static volatile bool g_run = true;
static void on_sig(int s) { (void)s; g_run = false; }

static void save_screenshot(HSGraphics* gfx, const char* path) {
    int w = gfx->screen_width;
    int h = gfx->screen_height;
    u8* pixels = malloc(w * h * 3);
    if (!pixels) return;
    
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, pixels);
    
    FILE* f = fopen(path, "wb");
    if (f) {
        fprintf(f, "P6\n%d %d\n255\n", w, h);
        for (int y = h - 1; y >= 0; y--) {
            fwrite(pixels + y * w * 3, 1, w * 3, f);
        }
        fclose(f);
        fprintf(stderr, "screenshot saved: %s\n", path);
    }
    free(pixels);
}

static void reset_ball(Ball *b) {
    b->x = 0.0f;
    b->y = 0.0f;
    b->vx = (rand() % 2 == 0 ? 1.0f : -1.0f) * (0.005f + 0.01f * (rand() / (float)RAND_MAX));
    b->vy = (rand() % 2 == 0 ? 1.0f : -1.0f) * (0.005f + 0.01f * (rand() / (float)RAND_MAX));
}

static bool kbhit(void) {
    fd_set rfds;
    struct timeval tv;
    FD_ZERO(&rfds);
    FD_SET(STDIN_FILENO, &rfds);
    tv.tv_sec = 0; tv.tv_usec = 0;
    return select(1, &rfds, NULL, NULL, &tv) > 0;
}

static void load_scores(int *player_high, int *ai_high) {
    *player_high = 0;
    *ai_high = 0;
    FILE *f = fopen(SAVE_PATH, "rb");
    if (!f) return;
    PongSave save;
    if (fread(&save, sizeof(save), 1, f) == 1 && save.magic == HS_STORAGE_MAGIC) {
        *player_high = save.player_high_score;
        *ai_high = save.ai_high_score;
    }
    fclose(f);
}

static void save_scores(int player_high, int ai_high) {
    FILE *f = fopen(SAVE_PATH, "wb");
    if (!f) return;
    PongSave save = {
        .magic = HS_STORAGE_MAGIC,
        .player_high_score = player_high,
        .ai_high_score = ai_high
    };
    fwrite(&save, sizeof(save), 1, f);
    fclose(f);
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    srand(time(NULL));
    
    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);

    HSGraphics gfx;
    if (hs_graphics_init(&gfx) != 0) {
        fprintf(stderr, "pong: display init failed\n");
        return 1;
    }
    fprintf(stderr, "pong: %ux%u display ready\n", gfx.screen_width, gfx.screen_height);

    g_prog = make_prog(VS_QUAD, FS_SOLID);
    glUseProgram(g_prog);
    g_a_pos = glGetAttribLocation(g_prog, "a_pos");
    g_a_uv = glGetAttribLocation(g_prog, "a_uv");
    g_u_color = glGetUniformLocation(g_prog, "u_color");

    glGenBuffers(1, &g_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, 6 * 16, NULL, GL_DYNAMIC_DRAW);

    glViewport(0, 0, (GLsizei)gfx.screen_width, (GLsizei)gfx.screen_height);
    glDisable(GL_DEPTH_TEST);

    g_font_loaded = hs_font_load(&g_font, "src/medodica_font");
    if (g_font_loaded) {
        fprintf(stderr, "Font loaded: size=%d line_height=%d\n", g_font.font_size, g_font.line_height);
        fprintf(stderr, "Font glyph 'A': tex=(%d,%d) size=%dx%d offset=(%d,%d) advance=%d\n",
                g_font.glyphs['A'].tex_x, g_font.glyphs['A'].tex_y,
                g_font.glyphs['A'].width, g_font.glyphs['A'].height,
                g_font.glyphs['A'].offset_x, g_font.glyphs['A'].offset_y,
                g_font.glyphs['A'].advance);
        hs_font_upload_texture(&g_font);
        fprintf(stderr, "Font texture uploaded: tex=%u\n", g_font.atlas_tex);
    } else {
        fprintf(stderr, "Font FAILED to load\n");
    }

    Paddle player = { -0.9f, 0.0f, 0 };
    Paddle ai = { 0.9f, 0.0f, 0 };
    Ball ball;
    reset_ball(&ball);

    int player_high = 0, ai_high = 0;
    load_scores(&player_high, &ai_high);
    fprintf(stderr, "High Scores - Player: %d  AI: %d\n", player_high, ai_high);

    fprintf(stderr, "PONG - Use UP/DOWN arrows or W/S to move\n");
    fprintf(stderr, "Press Q to quit\n");

    int ch = 0;
    static int screenshot_num = 0;
    int frame = 0;
    while (g_run) {
        frame++;
        if (kbhit()) {
            ch = getchar();
            if (ch == 'q' || ch == 'Q') break;
            if (ch == 'p' || ch == 'P') {
                char path[64];
                snprintf(path, sizeof(path), "/tmp/pong_%03d.ppm", screenshot_num++);
                save_screenshot(&gfx, path);
                ch = 0;
            }
        }

        if (ch == -1 || ch == 27) {
            ch = 0;
        }

        float move = 0.0f;
        if (ch == 'w' || ch == 'W' || ch == 126) {
            move = PADDLE_SPEED;
            ch = 0;
        }
        if (ch == 's' || ch == 'S' || ch == 125) {
            move = -PADDLE_SPEED;
            ch = 0;
        }

        player.y += move;
        if (player.y > 1.0f - PADDLE_H) player.y = 1.0f - PADDLE_H;
        if (player.y < -1.0f) player.y = -1.0f;

        float ai_target = ball.y;
        if (ai.y < ai_target - 0.05f) ai.y += AI_SPEED;
        else if (ai.y > ai_target + 0.05f) ai.y -= AI_SPEED;
        if (ai.y > 1.0f - PADDLE_H) ai.y = 1.0f - PADDLE_H;
        if (ai.y < -1.0f) ai.y = -1.0f;

        ball.x += ball.vx;
        ball.y += ball.vy;

        if (ball.y > 1.0f - BALL_SIZE || ball.y < -1.0f) {
            ball.vy = -ball.vy;
            ball.y = ball.y > 0 ? 1.0f - BALL_SIZE : -1.0f;
        }

        if (ball.x < player.x + PADDLE_W && 
            ball.x > player.x - PADDLE_W &&
            ball.y > player.y - PADDLE_H/2 && 
            ball.y < player.y + PADDLE_H/2) {
            ball.vx = fabsf(ball.vx) * 1.05f;
            if (ball.vx > 0.03f) ball.vx = 0.03f;
            float offset = (ball.y - player.y) / (PADDLE_H/2);
            ball.vy += offset * 0.01f;
            ball.x = player.x + PADDLE_W;
        }

        if (ball.x > ai.x - PADDLE_W && 
            ball.x < ai.x + PADDLE_W &&
            ball.y > ai.y - PADDLE_H/2 && 
            ball.y < ai.y + PADDLE_H/2) {
            ball.vx = -fabsf(ball.vx) * 1.05f;
            if (ball.vx < -0.03f) ball.vx = -0.03f;
            float offset = (ball.y - ai.y) / (PADDLE_H/2);
            ball.vy += offset * 0.01f;
            ball.x = ai.x - PADDLE_W;
        }

        if (ball.x < -1.5f) {
            ai.score++;
            if (ai.score > ai_high) ai_high = ai.score;
            fprintf(stderr, "\rPlayer: %d  AI: %d  (Best: %d)", player.score, ai.score, ai_high);
            reset_ball(&ball);
        }
        if (ball.x > 1.5f) {
            player.score++;
            if (player.score > player_high) player_high = player.score;
            fprintf(stderr, "\rPlayer: %d  AI: %d  (Best: %d)", player.score, ai.score, player_high);
            reset_ball(&ball);
        }

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUniform3f(g_u_color, 0.2f, 0.6f, 1.0f);
        draw_quad(player.x - PADDLE_W, player.y - PADDLE_H/2, 
                  player.x + PADDLE_W, player.y + PADDLE_H/2);

        glUniform3f(g_u_color, 1.0f, 0.2f, 0.2f);
        draw_quad(ai.x - PADDLE_W, ai.y - PADDLE_H/2, 
                  ai.x + PADDLE_W, ai.y + PADDLE_H/2);

        glUniform3f(g_u_color, 1.0f, 1.0f, 1.0f);
        draw_quad(ball.x - BALL_SIZE/2, ball.y - BALL_SIZE/2,
                  ball.x + BALL_SIZE/2, ball.y + BALL_SIZE/2);

        if (g_font_loaded && frame <= 2) {
            fprintf(stderr, "Frame %d: rendering text\n", frame);
            hs_font_render_text(&g_font, "HELLO", -0.3f, 0.8f, 0.05f, 1.0f, 1.0f, 1.0f, 1.0f);
            glDisable(GL_BLEND);
            glUseProgram(g_prog);
        }

        hs_graphics_present(&gfx);

        if (frame % 60 == 0) {
            char path[64];
            snprintf(path, sizeof(path), "/tmp/pong_%03d.ppm", frame/60);
            save_screenshot(&gfx, path);
        }

        struct timespec ts = {0, 16666667};
        nanosleep(&ts, NULL);
    }

    hs_graphics_finish(&gfx);
    save_scores(player_high, ai_high);
    fprintf(stderr, "\npong: final score - Player: %d  AI: %d\n", player.score, ai.score);
    return 0;
}
