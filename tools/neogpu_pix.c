/*
 * neogpu_pix.c -- Semantic Image Description Visualizer
 * 
 * Two-window visualization:
 * - Left: Target image drawn with GL primitives (circle, heart, square)
 * - Right: Model's geometric reconstruction, updated as concept tokens appear
 * 
 * Concept tokens trigger geometric primitives:
 *   circle=26942, heart=18207, square=38576, curve=51151
 *   line=1074, point=2837, edge=7334, center=3133
 *   red=1171, blue=12481, green=11787, yellow=30516
 *   round=1067, left=2414, right=1315, up=455, down=2996
 *   symmetric=45621, two=1899, four=2568, three=2820
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <signal.h>
#include <fcntl.h>
#include <unistd.h>
#include "hs_graphics.h"
#include "hs_ml_infer.h"

#define MAX_TOKENS  512
#define MAX_CONCEPTS 64

typedef struct {
    uint32_t token_id;
    const char *concept;
    const char *primitive;
    float r, g, b;
} ConceptEntry;

static const ConceptEntry g_concepts[] = {
    {18207, "heart",     "heart",     0.9f, 0.2f, 0.3f},
    {26942, "circle",    "circle",    0.2f, 0.6f, 0.9f},
    {38576, "square",    "rect",      0.8f, 0.5f, 0.2f},
    {12134, "star",      "star",      1.0f, 0.9f, 0.2f},
    {51151, "curve",     "curve",     0.6f, 0.3f, 0.8f},
    {1074,  "line",      "line",      0.7f, 0.7f, 0.7f},
    {2837,  "point",     "point",     1.0f, 1.0f, 1.0f},
    {7334,  "edge",      "line",      0.5f, 0.5f, 0.5f},
    {3133,  "center",    "point",     1.0f, 0.5f, 0.0f},
    {1171,  "red",       "color_r",   0.9f, 0.2f, 0.2f},
    {12481, "blue",      "color_b",   0.2f, 0.4f, 0.9f},
    {11787, "green",     "color_g",   0.2f, 0.8f, 0.3f},
    {30516, "yellow",    "color_y",   1.0f, 0.9f, 0.2f},
    {1067,  "round",     "circle",    0.3f, 0.7f, 0.6f},
    {2414,  "left",      "dir_left",  0.8f, 0.3f, 0.3f},
    {1315,  "right",     "dir_right", 0.3f, 0.8f, 0.3f},
    {455,   "up",        "dir_up",    0.3f, 0.3f, 0.9f},
    {2996,  "down",      "dir_down",  0.9f, 0.6f, 0.3f},
    {46220, "symmetric", "symmetric", 0.6f, 0.6f, 0.8f},
    {1899,  "two",       "count_2",   0.5f, 0.8f, 0.9f},
    {2568,  "four",      "count_4",   0.9f, 0.6f, 0.5f},
    {2820,  "three",     "count_3",   0.7f, 0.9f, 0.5f},
};
static const int g_n_concepts = sizeof(g_concepts) / sizeof(g_concepts[0]);

typedef struct {
    uint32_t tokens[MAX_TOKENS];
    uint32_t n_tokens;
    uint32_t concept_indices[MAX_CONCEPTS];
    uint32_t n_concepts;
    float color_r, color_g, color_b;
    bool symmetric;
    int direction;
    int count;
    double step_ms;
    uint32_t step_num;
    bool valid;
} PixSnapshot;

static PixSnapshot g_snaps[2];
static atomic_int g_write_idx = 0;

typedef struct {
    const char *model_path, *norms_path, *target_shape;
    const char *prompt;
    float temp, top_p, rep_penalty;
    uint32_t top_k;
    int n_predict;
    atomic_bool *stop;
} InferArgs;

static int match_concept(uint32_t token_id) {
    for (int i = 0; i < g_n_concepts; i++) {
        if (g_concepts[i].token_id == token_id) return i;
    }
    return -1;
}

static uint32_t sample_topk(const float *logits, uint32_t V,
                              float temp, uint32_t topk,
                              const uint32_t *prev, uint32_t nprev,
                              float rep_pen) {
    static float buf[131072];
    uint32_t Vc = V < 131072 ? V : 131072;
    memcpy(buf, logits, Vc * 4);
    for (uint32_t i = 0; i < nprev && i < 64; i++) {
        uint32_t t = prev[nprev - 1 - i];
        if (t < Vc) buf[t] /= rep_pen;
    }
    float mx = -1e30f;
    for (uint32_t v = 0; v < Vc; v++)
        if (buf[v] > -1e29f && buf[v] > mx) mx = buf[v];
    static uint32_t idx[131072];
    static float probs[131072];
    uint32_t cnt = 0;
    for (uint32_t v = 0; v < Vc; v++)
        if (buf[v] > -1e29f) {
            probs[cnt] = expf((buf[v] - mx) / temp);
            idx[cnt] = v;
            cnt++;
        }
    uint32_t k = topk < cnt ? topk : cnt;
    for (uint32_t i = 0; i < k; i++)
        for (uint32_t j = i + 1; j < cnt; j++)
            if (probs[j] > probs[i]) {
                float tp = probs[i];
                probs[i] = probs[j];
                probs[j] = tp;
                uint32_t ti = idx[i];
                idx[i] = idx[j];
                idx[j] = ti;
            }
    float s = 0;
    for (uint32_t i = 0; i < k; i++) s += probs[i];
    float r = ((float)rand() / RAND_MAX) * s, cum = 0;
    for (uint32_t i = 0; i < k; i++) {
        cum += probs[i];
        if (cum >= r) return idx[i];
    }
    return idx[0];
}

static void make_snap(PixSnapshot *s, const uint32_t *toks, uint32_t ntoks,
                      const float *logits, uint32_t V,
                      double ms, uint32_t step) {
    memset(s, 0, sizeof(*s));
    s->step_ms = ms;
    s->step_num = step;
    s->n_tokens = ntoks < MAX_TOKENS ? ntoks : MAX_TOKENS;
    memcpy(s->tokens, toks, s->n_tokens * 4);
    
    s->color_r = 0.3f;
    s->color_g = 0.6f;
    s->color_b = 0.9f;
    s->symmetric = false;
    s->direction = 0;
    s->count = 1;
    
    if (logits && V > 0) {
        float max_logit = -1e30f;
        for (uint32_t v = 0; v < V; v++) if (logits[v] > max_logit) max_logit = logits[v];
        
        for (int i = 0; i < g_n_concepts && s->n_concepts < MAX_CONCEPTS; i++) {
            uint32_t tid = g_concepts[i].token_id;
            if (tid < V) {
                float prob = expf(logits[tid] - max_logit);
                if (prob > 0.005f) {
                    fprintf(stderr, "[step %u] concept '%s' prob=%.3f\n", step, g_concepts[i].concept, prob);
                    s->concept_indices[s->n_concepts++] = (uint32_t)i;
                    const char *prim = g_concepts[i].primitive;
                    if (strcmp(prim, "color_r") == 0) s->color_r = 0.9f;
                    else if (strcmp(prim, "color_b") == 0) s->color_b = 0.9f;
                    else if (strcmp(prim, "color_g") == 0) s->color_g = 0.9f;
                    else if (strcmp(prim, "color_y") == 0) { s->color_r = 1.0f; s->color_g = 0.9f; s->color_b = 0.2f; }
                    else if (strcmp(prim, "symmetric") == 0) s->symmetric = true;
                    else if (strcmp(prim, "dir_left") == 0) s->direction = -1;
                    else if (strcmp(prim, "dir_right") == 0) s->direction = 1;
                    else if (strcmp(prim, "dir_up") == 0) s->direction = 2;
                    else if (strcmp(prim, "dir_down") == 0) s->direction = 3;
                    else if (strcmp(prim, "count_2") == 0) s->count = 2;
                    else if (strcmp(prim, "count_3") == 0) s->count = 3;
                    else if (strcmp(prim, "count_4") == 0) s->count = 4;
                }
            }
        }
    }
    s->valid = true;
}

static void *infer_thread(void *arg) {
    InferArgs *a = (InferArgs *)arg;
    HSMLTernary m;
    hs_mlt_init(&m);
    if (hs_mlt_load_gguf(&m, a->model_path) != 0) {
        fprintf(stderr, "pix: load failed\n");
        return NULL;
    }
    if (a->norms_path) hs_mlt_load_norms_sidecar(&m, a->norms_path);
    hs_mlt_lmhead_encode(&m);
    
    uint32_t V = m.vocab_size;
    uint32_t tokens[MAX_TOKENS];
    uint32_t n = 0;
    float *logits = malloc(V * 4);
    if (!logits) { fprintf(stderr, "pix: malloc failed\n"); return NULL; }
    
    char prompt[512];
    snprintf(prompt, sizeof(prompt),
        "The shape is a %s. It has", a->target_shape);
    
    tokens[n++] = m.tokenizer_bos;
    n += hs_mlt_bpe_encode(&m, prompt, (uint32_t)strlen(prompt), tokens + n, MAX_TOKENS - n);
    
    fprintf(stderr, "pix: prompt tokens=%u, shape=%s\n", n, a->target_shape);
    
    HSMLTernarySession sess;
    hs_mlt_session_init(&sess, &m);
    hs_mlt_prefill(&sess, tokens, n);
    hs_mlt_session_logits(&sess, logits);
    
    int wi = 0;
    make_snap(&g_snaps[wi], tokens, n, logits, V, 0, 0);
    atomic_store(&g_write_idx, wi);
    
    struct timespec t0, t1;
    for (int step = 0; step < a->n_predict && !atomic_load(a->stop); step++) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        uint32_t tok = sample_topk(logits, V, a->temp, a->top_k, tokens, n, a->rep_penalty);
        if (tok == m.tokenizer_eos) break;
        if (n < MAX_TOKENS) tokens[n++] = tok;
        
        int ci = match_concept(tok);
        if (ci >= 0) {
            fprintf(stderr, "[%d] concept: %s\n", step, g_concepts[ci].concept);
        }
        
        if (tok < m.vocab_size && m.tokenizer_vocab[tok]) {
            const char *tv = m.tokenizer_vocab[tok];
            while (*tv) {
                if ((unsigned char)tv[0] == 0xC4 && (unsigned char)tv[1] == 0xA0) {
                    putchar(' ');
                    tv += 2;
                } else {
                    putchar(*tv++);
                }
            }
            fflush(stdout);
        }
        
        hs_mlt_decode(&sess, tok, logits);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = ((double)(t1.tv_sec - t0.tv_sec) * 1e3
                    + (double)(t1.tv_nsec - t0.tv_nsec) / 1e6);
        
        int nwi = 1 - atomic_load(&g_write_idx);
        make_snap(&g_snaps[nwi], tokens, n, logits, V, ms, (uint32_t)step + 1);
        atomic_store(&g_write_idx, nwi);
    }
    
    free(logits);
    hs_mlt_session_free(&sess);
    hs_mlt_free(&m);
    fprintf(stderr, "\npix: done\n");
    return NULL;
}

static const char *VS_QUAD =
    "attribute vec2 a_pos;\n"
    "varying vec2 v_uv;\n"
    "void main(){v_uv=a_pos*0.5+0.5;gl_Position=vec4(a_pos,0.0,1.0);}\n";

static const char *FS_CIRCLE =
    "precision mediump float;\n"
    "varying vec2 v_uv;\n"
    "uniform vec3 u_color;\n"
    "uniform float u_r;\n"
    "void main(){\n"
    "  vec2 c=v_uv-0.5;\n"
    "  float d=length(c);\n"
    "  float ring=smoothstep(u_r-0.02,u_r,d)*smoothstep(u_r+0.02,u_r,d);\n"
    "  float fill=smoothstep(u_r,0.0,d)*0.3;\n"
    "  gl_FragColor=vec4(u_color,fill+ring);\n"
    "}\n";

static const char *FS_HEART =
    "precision mediump float;\n"
    "varying vec2 v_uv;\n"
    "uniform vec3 u_color;\n"
    "void main(){\n"
    "  vec2 p=v_uv*2.0-1.0;\n"
    "  float h=p.x+sign(p.y)*sqrt(1.0-abs(p.x));\n"
    "  h=1.0-h*h;\n"
    "  float d=smoothstep(0.0,0.02,h-abs(p.y-0.5));\n"
    "  gl_FragColor=vec4(u_color,d*0.8);\n"
    "}\n";

static const char *FS_RECT =
    "precision mediump float;\n"
    "varying vec2 v_uv;\n"
    "uniform vec3 u_color;\n"
    "uniform float u_aspect;\n"
    "void main(){\n"
    "  vec2 p=v_uv*2.0-1.0;\n"
    "  p.x/=u_aspect;\n"
    "  float bx=step(-0.8,abs(p.x))-step(0.8,abs(p.x));\n"
    "  float by=step(-0.6,abs(p.y))-step(0.6,abs(p.y));\n"
    "  float edge=bx*by;\n"
    "  float fill=(1.0-edge)*0.2;\n"
    "  gl_FragColor=vec4(u_color,fill+edge*0.8);\n"
    "}\n";

static const char *FS_STAR =
    "precision mediump float;\n"
    "varying vec2 v_uv;\n"
    "uniform vec3 u_color;\n"
    "void main(){\n"
    "  vec2 p=(v_uv-0.5)*2.0;\n"
    "  float a=atan(p.y,p.x);\n"
    "  float r=length(p);\n"
    "  float star=sin(a*5.0)*0.3+0.7;\n"
    "  float d=smoothstep(star+0.05,star,r)*(1.0-smoothstep(star,star-0.05,r));\n"
    "  gl_FragColor=vec4(u_color,d);\n"
    "}\n";

static const char *FS_CURVE =
    "precision mediump float;\n"
    "varying vec2 v_uv;\n"
    "uniform vec3 u_color;\n"
    "uniform float u_dir;\n"
    "void main(){\n"
    "  vec2 p=v_uv*2.0-1.0;\n"
    "  float c=p.y-p.x*p.x*u_dir;\n"
    "  float line=smoothstep(0.05,0.0,abs(c));\n"
    "  gl_FragColor=vec4(u_color,line);\n"
    "}\n";

typedef struct { float x, y, z; } Vec3;

static void set_vec3(Vec3 *v, float x, float y, float z) { v->x = x; v->y = y; v->z = z; }

static GLuint make_prog(const char *vs, const char *fs) {
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v, 1, &vs, NULL);
    glCompileShader(v);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &fs, NULL);
    glCompileShader(f);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}

static GLuint g_prog_circle, g_prog_heart, g_prog_rect;
static GLuint g_prog_star, g_prog_curve;
static GLuint g_vbo;

static void draw_shape(GLuint prog, float x0, float y0, float x1, float y1,
                       Vec3 color, float param) {
    float q[] = {
        x0, y0, x1, y0, x0, y1,
        x1, y0, x1, y1, x0, y1
    };
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(q), q);
    GLint pos = glGetAttribLocation(prog, "a_pos");
    glEnableVertexAttribArray(pos);
    glVertexAttribPointer(pos, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glUseProgram(prog);
    glUniform3fv(glGetUniformLocation(prog, "u_color"), 1, (float *)&color);
    if (param != 0.0f) {
        glUniform1f(glGetUniformLocation(prog, "u_r"), param);
        glUniform1f(glGetUniformLocation(prog, "u_aspect"), param);
    }
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

static volatile bool g_run = true;
static void on_sig(int s) {
    (void)s;
    g_run = false;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s --model <path> [--norms <n>] [--shape <circle|heart|square|star>]\n"
                "  [--prompt <p>] [--temp <f>] [--top-k <n>] [--n-predict <n>]\n", argv[0]);
        return 1;
    }
    
    const char *mpath = NULL, *npath = NULL;
    const char *shape = "heart";
    const char *prompt = NULL;
    float temp = 0.432f, top_p = 0.9531f, rep = 1.1229f;
    uint32_t topk = 42;
    int npred = 128;
    
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) mpath = argv[++i];
        else if (!strcmp(argv[i], "--norms") && i + 1 < argc) npath = argv[++i];
        else if (!strcmp(argv[i], "--shape") && i + 1 < argc) shape = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) temp = atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-k") && i + 1 < argc) topk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n-predict") && i + 1 < argc) npred = atoi(argv[++i]);
    }
    
    if (!mpath) {
        fprintf(stderr, "--model required\n");
        return 1;
    }
    
    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);
    
    HSGraphics gfx;
    if (hs_graphics_init(&gfx) != 0) {
        fprintf(stderr, "pix: display init failed\n");
        return 1;
    }
    fprintf(stderr, "pix: %ux%u display, shape=%s\n", gfx.screen_width, gfx.screen_height, shape);
    
    g_prog_circle = make_prog(VS_QUAD, FS_CIRCLE);
    g_prog_heart = make_prog(VS_QUAD, FS_HEART);
    g_prog_rect = make_prog(VS_QUAD, FS_RECT);
    g_prog_star = make_prog(VS_QUAD, FS_STAR);
    g_prog_curve = make_prog(VS_QUAD, FS_CURVE);
    
    glGenBuffers(1, &g_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, 6 * 8, NULL, GL_DYNAMIC_DRAW);
    
    glViewport(0, 0, (GLsizei)gfx.screen_width, (GLsizei)gfx.screen_height);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    atomic_bool stop;
    atomic_store(&stop, false);
    InferArgs ia = {mpath, npath, shape, prompt, temp, top_p, rep, topk, npred, &stop};
    pthread_t tid;
    pthread_create(&tid, NULL, infer_thread, &ia);
    
    float win_left = -1.0f, win_right = 1.0f;
    float win_top = 1.0f, win_bottom = -1.0f;
    float mid = 0.0f;
    
    uint32_t frame = 0;
    while (g_run) {
        const PixSnapshot *s = &g_snaps[atomic_load(&g_write_idx)];
        
        glClearColor(0.02f, 0.02f, 0.04f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        if (s->valid) {
            Vec3 target_color;
            if (strcmp(shape, "heart") == 0) set_vec3(&target_color, 0.9f, 0.2f, 0.3f);
            else if (strcmp(shape, "circle") == 0) set_vec3(&target_color, 0.2f, 0.6f, 0.9f);
            else if (strcmp(shape, "square") == 0) set_vec3(&target_color, 0.8f, 0.5f, 0.2f);
            else if (strcmp(shape, "star") == 0) set_vec3(&target_color, 1.0f, 0.9f, 0.2f);
            else set_vec3(&target_color, 0.9f, 0.2f, 0.3f);
            
            if (strcmp(shape, "heart") == 0) {
                draw_shape(g_prog_heart, win_left, win_bottom, mid, win_top, target_color, 0.0f);
            } else if (strcmp(shape, "circle") == 0) {
                draw_shape(g_prog_circle, win_left, win_bottom, mid, win_top, target_color, 0.35f);
            } else if (strcmp(shape, "square") == 0) {
                draw_shape(g_prog_rect, win_left, win_bottom, mid, win_top, target_color, 1.2f);
            } else if (strcmp(shape, "star") == 0) {
                draw_shape(g_prog_star, win_left, win_bottom, mid, win_top, target_color, 0.0f);
            }
            
            Vec3 recon_color;
            set_vec3(&recon_color, s->color_r, s->color_g, s->color_b);
            if (s->n_concepts == 0) {
                Vec3 gray; set_vec3(&gray, 0.1f, 0.1f, 0.1f);
                draw_shape(g_prog_circle, mid, win_bottom, win_right, win_top, gray, 0.3f);
            } else {
                for (uint32_t i = 0; i < s->n_concepts; i++) {
                    const ConceptEntry *c = &g_concepts[s->concept_indices[i]];
                    if (strcmp(c->primitive, "circle") == 0 || strcmp(c->primitive, "round") == 0) {
                        draw_shape(g_prog_circle, mid, win_bottom, win_right, win_top, recon_color, 0.35f);
                    } else if (strcmp(c->primitive, "heart") == 0) {
                        draw_shape(g_prog_heart, mid, win_bottom, win_right, win_top, recon_color, 0.0f);
                    } else if (strcmp(c->primitive, "rect") == 0) {
                        draw_shape(g_prog_rect, mid, win_bottom, win_right, win_top, recon_color, 1.2f);
                    } else if (strcmp(c->primitive, "star") == 0) {
                        draw_shape(g_prog_star, mid, win_bottom, win_right, win_top, recon_color, 0.0f);
                    } else if (strcmp(c->primitive, "curve") == 0) {
                        draw_shape(g_prog_curve, mid, win_bottom, win_right, win_top, recon_color, 0.5f);
                    }
                }
            }
            
            if ((frame % 30) == 0 && s->step_num > 0) {
                fprintf(stderr, "\rstep=%u concepts=%u",
                    s->step_num, s->n_concepts);
            }
        }
        
        hs_graphics_present(&gfx);
        frame++;
        usleep(10000);
    }
    
    atomic_store(&stop, true);
    pthread_join(tid, NULL);
    hs_graphics_finish(&gfx);
    fprintf(stderr, "\npix: %u frames\n", frame);
    return 0;
}
