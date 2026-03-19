/*
 * neogpu_pong_llm.c -- Pong + Real LLM Inference Demo
 * 
 * Demonstrates:
 * 1. Game rendering (Pong) on main thread
 * 2. Real BitNet inference in background thread  
 * 3. LLM output rendered via GPU message layer
 * 4. Message capture of combined rendering + ML
 * 
 * This proves: rendering AND inference are both routing problems
 * solved by the SAME message-passing infrastructure.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <pthread.h>
#include "hs_gpu.h"
#include "hs_backend_gles.h"
#include "hs_graphics.h"
#include "hs_ml_infer.h"

#define MAX_MSG_LOG 4096
#define MAX_TOKENS 256

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

typedef struct {
    const char* model_path;
    const char* prompt;
    atomic_bool started;
    atomic_bool done;
    char output[1024];
    int output_len;
} InferArgs;

static void* inference_thread(void* arg) {
    InferArgs* a = (InferArgs*)arg;
    
    HSMLTernary m;
    hs_mlt_init(&m);
    
    if (hs_mlt_load_gguf(&m, a->model_path) != 0) {
        fprintf(stderr, "LLM: failed to load model\n");
        return NULL;
    }
    
    if (!m.tokenizer_vocab) {
        fprintf(stderr, "LLM: no tokenizer\n");
        return NULL;
    }
    
    hs_mlt_lmhead_encode(&m);
    
    uint32_t tokens[MAX_TOKENS];
    uint32_t n = 0;
    tokens[n++] = m.tokenizer_bos;
    n += hs_mlt_bpe_encode(&m, a->prompt, (uint32_t)strlen(a->prompt), tokens + n, MAX_TOKENS - n);
    
    fprintf(stderr, "LLM: prompt = %u tokens\n", n);
    
    HSMLTernarySession sess;
    hs_mlt_session_init(&sess, &m);
    hs_mlt_prefill(&sess, tokens, n);
    
    float* logits = malloc(m.vocab_size * 4);
    a->output_len = 0;
    a->output[0] = '\0';
    
    atomic_store(&a->started, true);
    
    for (int i = 0; i < 32 && g_run; i++) {
        hs_mlt_session_logits(&sess, logits);
        uint32_t tok = hs_mlt_sample_greedy(logits, m.vocab_size);
        
        if (tok == m.tokenizer_eos) break;
        if (tok >= m.vocab_size) break;
        
        const char* word = m.tokenizer_vocab[tok];
        if (word && a->output_len < 1000) {
            int len = strlen(word);
            if (a->output_len + len < 1000) {
                strcpy(a->output + a->output_len, word);
                a->output_len += len;
            }
        }
        
        if (n < MAX_TOKENS) tokens[n++] = tok;
        hs_mlt_decode(&sess, tok, logits);
    }
    
    free(logits);
    atomic_store(&a->done, true);
    fprintf(stderr, "LLM: generated %d chars\n", a->output_len);
    
    return NULL;
}

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

int main(int argc, char** argv) {
    const char* model_path = NULL;
    const char* prompt = "Hello, how are you?";
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        }
    }
    
    if (!model_path) {
        fprintf(stderr, "Usage: %s --model <gguf-file> [--prompt <text>]\n", argv[0]);
        fprintf(stderr, "This demo requires a BitNet GGUF model file.\n");
        return 1;
    }
    
    srand(time(NULL));
    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);

    printf("=== NeoGPU Pong + Real LLM Demo ===\n");
    printf("Model: %s\n", model_path);
    printf("Prompt: %s\n\n", prompt);

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

    printf("Starting LLM inference thread...\n");
    InferArgs ia = {model_path, prompt, false, false, "", 0};
    pthread_t ithr;
    pthread_create(&ithr, NULL, inference_thread, &ia);

    while (!atomic_load(&ia.started)) {
        usleep(100000);
    }
    fprintf(stderr, "LLM: started\n");

    printf("Recording 180 frames with LLM output...\n");
    hs_gpu_start_recording(&gpu);

    for (int frame = 0; frame < 180 && g_run; frame++) {
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

        if (ball.y < ai.y - 0.05f) ai.y += 0.012f;
        else if (ball.y > ai.y + 0.05f) ai.y -= 0.012f;
        if (ai.y > 0.75f) ai.y = 0.75f;
        if (ai.y < -0.75f) ai.y = -0.75f;

        hs_gpu_begin_frame(&gpu);
        hs_gpu_clear(&gpu, v4_make(0.05f, 0.05f, 0.1f, 1.0f));
        
        glUniform3f(g_u_color, 0.2f, 0.6f, 1.0f);
        draw_quad(player.x - 0.03f, player.y - 0.15f, player.x + 0.03f, player.y + 0.15f);
        
        glUniform3f(g_u_color, 1.0f, 0.2f, 0.2f);
        draw_quad(ai.x - 0.03f, ai.y - 0.15f, ai.x + 0.03f, ai.y + 0.15f);
        
        glUniform3f(g_u_color, 1.0f, 1.0f, 1.0f);
        draw_quad(ball.x - 0.02f, ball.y - 0.02f, ball.x + 0.02f, ball.y + 0.02f);

        if (atomic_load(&ia.done) && ia.output_len > 0) {
            char display[64];
            int copy_len = ia.output_len < 63 ? ia.output_len : 63;
            strncpy(display, ia.output, copy_len);
            display[copy_len] = '\0';
            
            hs_gpu_draw_text(&gpu, display);
        }

        hs_gpu_process(&gpu);
        hs_gpu_end_frame(&gpu);

        hs_graphics_present(gfx);
        usleep(16667);
    }

    u32 log_count = hs_gpu_stop_recording(&gpu);
    printf("Captured %u messages\n", log_count);

    pthread_join(ithr, NULL);

    Message* msgs = gpu.log_buffer;
    static Message cap_msgs[MAX_MSG_LOG];
    static Payload cap_payloads[MAX_MSG_LOG];
    HSCapture cap;
    hs_capture_init(&cap, cap_msgs, cap_payloads, MAX_MSG_LOG);
    
    bool cap_ok = hs_capture_from_log(&gpu.system, msgs, log_count, &cap);
    printf("Capture from log: %s\n", cap_ok ? "OK" : "FAILED");
    
    const char* cap_path = "/tmp/pong_llm_capture.bin";
    bool write_ok = hs_capture_write_file(&cap, cap_path);
    printf("Write capture: %s (%s)\n", write_ok ? "OK" : "FAILED", cap_path);

    save_screenshot(gfx, "/tmp/pong_llm_demo.ppm");

    printf("\n=== Demo Complete ===\n");
    printf("LLM output: %s\n", ia.output);
    printf("Files:\n");
    printf("  %s - Message capture\n", cap_path);
    printf("  /tmp/pong_llm_demo.ppm - Screenshot\n");
    
    return 0;
}
