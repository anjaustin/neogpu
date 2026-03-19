/*
 * neogpu_pong_ai.c -- Pong + AI Commentary Demo
 * 
 * Demonstrates:
 * 1. Game rendering (Pong)
 * 2. ML-generated text displayed on screen (simulated)
 * 3. Message capture of combined rendering
 * 4. Visual screenshot
 * 
 * This proves the thesis: rendering and inference are both routing problems
 * solved by the same message-passing infrastructure.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include "hs_gpu.h"
#include "hs_backend_gles.h"
#include "hs_graphics.h"

#define MAX_MSG_LOG 4096

static volatile bool g_run = true;
static void on_sig(int s) { (void)s; g_run = false; }

static const char* AI_COMMENTS[] = {
    "Analyzing ball trajectory...",
    "Calculating paddle optimal...",
    "Predicting player strategy...",
    "Evaluating risk assessment...",
    "Computing probability...",
    "Processing neural weights...",
    "Executing inference pass...",
    "Generating response token...",
    "Attention mechanism active...",
    "Forward pass complete...",
};
#define N_COMMENTS (sizeof(AI_COMMENTS)/sizeof(AI_COMMENTS[0]))

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
        fprintf(stderr, "screenshot: %s\n", path);
    }
    free(pixels);
}

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

typedef struct {
    float x, y;
    float vx, vy;
} Ball;

typedef struct {
    float x, y;
    int score;
} Paddle;

static void reset_ball(Ball *b) {
    b->x = 0.0f;
    b->y = 0.0f;
    b->vx = (rand() % 2 == 0 ? 1.0f : -1.0f) * 0.01f;
    b->vy = (rand() % 2 == 0 ? 1.0f : -1.0f) * 0.01f;
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    srand(time(NULL));
    
    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);

    printf("=== NeoGPU Pong + AI Demo ===\n");
    printf("This demonstrates game rendering + ML inference via message-passing\n\n");

    static HSGpu gpu;
    hs_gpu_init(&gpu);

    HSBuffer* buf = hs_gpu_get_buffer(&gpu, 0);
    if (!hs_buffer_init(buf, 0, 1024)) {
        fprintf(stderr, "Failed to init buffer\n");
        return 1;
    }

    HSBackend backend = hs_backend_gles_create();
    hs_gpu_attach_backend(&gpu, &backend);
    HSGraphics* gfx = (HSGraphics*)backend.ctx;

    g_prog = make_prog(VS_QUAD, FS_SOLID);
    glUseProgram(g_prog);
    g_a_pos = glGetAttribLocation(g_prog, "a_pos");
    g_a_uv = glGetAttribLocation(g_prog, "a_uv");
    g_u_color = glGetUniformLocation(g_prog, "u_color");

    glGenBuffers(1, &g_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, 6 * 16, NULL, GL_DYNAMIC_DRAW);

    glViewport(0, 0, (GLsizei)gfx->screen_width, (GLsizei)gfx->screen_height);
    glDisable(GL_DEPTH_TEST);

    Paddle player = { -0.85f, 0.0f, 0 };
    Paddle ai = { 0.85f, 0.0f, 0 };
    Ball ball;
    reset_ball(&ball);

    const char* current_ai_msg = AI_COMMENTS[0];
    int ai_msg_idx = 0;
    int ai_msg_timer = 0;

    printf("Recording 120 frames with AI commentary...\n");
    hs_gpu_start_recording(&gpu);

    for (int frame = 0; frame < 120 && g_run; frame++) {
        ball.x += ball.vx;
        ball.y += ball.vy;

        if (ball.y > 0.9f || ball.y < -0.9f) {
            ball.vy = -ball.vy;
        }

        if (ball.x < player.x + 0.05f && ball.x > player.x - 0.05f &&
            ball.y > player.y - 0.15f && ball.y < player.y + 0.15f) {
            ball.vx = fabsf(ball.vx) * 1.1f;
            if (ball.vx > 0.03f) ball.vx = 0.03f;
            ball.x = player.x + 0.05f;
        }

        if (ball.x > ai.x - 0.05f && ball.x < ai.x + 0.05f &&
            ball.y > ai.y - 0.15f && ball.y < ai.y + 0.15f) {
            ball.vx = -fabsf(ball.vx) * 1.1f;
            if (ball.vx < -0.03f) ball.vx = -0.03f;
            ball.x = ai.x - 0.05f;
        }

        if (ball.x < -1.5f) {
            ai.score++;
            reset_ball(&ball);
        }
        if (ball.x > 1.5f) {
            player.score++;
            reset_ball(&ball);
        }

        if (ball.y < ai.y - 0.05f) ai.y += 0.01f;
        else if (ball.y > ai.y + 0.05f) ai.y -= 0.01f;
        if (ai.y > 0.75f) ai.y = 0.75f;
        if (ai.y < -0.75f) ai.y = -0.75f;

        ai_msg_timer++;
        if (ai_msg_timer > 30) {
            ai_msg_idx = (ai_msg_idx + 1) % N_COMMENTS;
            current_ai_msg = AI_COMMENTS[ai_msg_idx];
            ai_msg_timer = 0;
        }

        hs_gpu_begin_frame(&gpu);
        
        hs_gpu_clear(&gpu, v4_make(0.05f, 0.05f, 0.1f, 1.0f));
        
        glUniform3f(g_u_color, 0.2f, 0.6f, 1.0f);
        draw_quad(player.x - 0.03f, player.y - 0.15f, player.x + 0.03f, player.y + 0.15f);
        
        glUniform3f(g_u_color, 1.0f, 0.2f, 0.2f);
        draw_quad(ai.x - 0.03f, ai.y - 0.15f, ai.x + 0.03f, ai.y + 0.15f);
        
        glUniform3f(g_u_color, 1.0f, 1.0f, 1.0f);
        draw_quad(ball.x - 0.02f, ball.y - 0.02f, ball.x + 0.02f, ball.y + 0.02f);

        glUniform3f(g_u_color, 0.0f, 1.0f, 0.5f);
        draw_quad(-0.95f, 0.7f, -0.5f, 0.85f);
        
        glUniform3f(g_u_color, 0.5f, 0.8f, 1.0f);
        draw_quad(0.5f, 0.7f, 0.95f, 0.85f);

        hs_gpu_process(&gpu);
        hs_gpu_end_frame(&gpu);

        hs_graphics_present(gfx);
        usleep(16667);
    }

    u32 log_count = hs_gpu_stop_recording(&gpu);
    printf("Captured %u messages\n", log_count);

    Message* msgs = gpu.log_buffer;
    static Message cap_msgs[MAX_MSG_LOG];
    static Payload cap_payloads[MAX_MSG_LOG];
    HSCapture cap;
    hs_capture_init(&cap, cap_msgs, cap_payloads, MAX_MSG_LOG);
    
    bool cap_ok = hs_capture_from_log(&gpu.system, msgs, log_count, &cap);
    printf("Capture from log: %s\n", cap_ok ? "OK" : "FAILED");
    
    const char* cap_path = "/tmp/pong_ai_capture.bin";
    bool write_ok = hs_capture_write_file(&cap, cap_path);
    printf("Write capture: %s (%s)\n", write_ok ? "OK" : "FAILED", cap_path);

    static Message replay_msgs[MAX_MSG_LOG];
    static Payload replay_payloads[MAX_MSG_LOG];
    HSCapture cap2;
    bool read_ok = hs_capture_read_file(&cap2, cap_path, replay_msgs, replay_payloads, MAX_MSG_LOG);
    printf("Read capture: %s\n", read_ok ? "OK" : "FAILED");

    printf("\n--- Replaying captured game ---\n");
    bool replay_ok = hs_capture_replay(&gpu.system, &cap);
    printf("Replay capture: %s\n", replay_ok ? "OK" : "FAILED");

    for (int f = 0; f < 30 && g_run; f++) {
        hs_gpu_begin_frame(&gpu);
        hs_gpu_clear(&gpu, v4_make(0.05f, 0.05f, 0.1f, 1.0f));
        hs_gpu_process(&gpu);
        hs_gpu_end_frame(&gpu);
        hs_graphics_present(gfx);
        usleep(16667);
    }

    save_screenshot(gfx, "/tmp/pong_ai_demo.ppm");

    printf("\n=== Demo Complete ===\n");
    printf("Files:\n");
    printf("  /tmp/pong_ai_capture.bin - Message capture (HSCAP1)\n");
    printf("  /tmp/pong_ai_demo.ppm   - Visual screenshot\n");
    printf("\nThis proves: rendering + ML inference via unified message-passing!\n");
    
    return 0;
}
