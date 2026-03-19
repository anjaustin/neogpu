#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "hs_ml_infer.h"
#include "hs_ml_gpu_gemm.h"

typedef struct {
    int byte_to_cp[256];
    int cp_to_byte[512];
} GPT2ByteMap;

static void gpt2_bytemap_init(GPT2ByteMap *m) {
    int bs[256], cs[256];
    int n = 0;
    for (int i = 33; i <= 126; i++) bs[n++] = i;
    for (int i = 161; i <= 172; i++) bs[n++] = i;
    for (int i = 174; i <= 255; i++) bs[n++] = i;
    for (int i = 0; i < n; i++) cs[i] = bs[i];
    int extra = 0;
    for (int b = 0; b < 256; b++) {
        int seen = 0;
        for (int i = 0; i < n; i++) {
            if (bs[i] == b) { seen = 1; break; }
        }
        if (!seen) {
            bs[n] = b;
            cs[n] = 256 + extra;
            n++;
            extra++;
        }
    }
    for (int i = 0; i < 256; i++) m->byte_to_cp[i] = -1;
    for (int i = 0; i < 512; i++) m->cp_to_byte[i] = -1;
    for (int i = 0; i < n; i++) {
        m->byte_to_cp[bs[i]] = cs[i];
        if (cs[i] < 512) m->cp_to_byte[cs[i]] = bs[i];
    }
}

static int utf8_encode_cp(int cp, char *out) {
    if (cp <= 0x7F) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp <= 0x7FF) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp <= 0xFFFF) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

static int utf8_decode_next(const char *s, u32 len, u32 *pos, int *cp) {
    if (*pos >= len) return 0;
    unsigned char c = (unsigned char)s[*pos];
    if (c < 0x80) {
        *cp = c;
        (*pos)++;
        return 1;
    }
    if ((c & 0xE0) == 0xC0 && *pos + 1 < len) {
        *cp = ((c & 0x1F) << 6) | ((unsigned char)s[*pos + 1] & 0x3F);
        *pos += 2;
        return 1;
    }
    if ((c & 0xF0) == 0xE0 && *pos + 2 < len) {
        *cp = ((c & 0x0F) << 12) |
              (((unsigned char)s[*pos + 1] & 0x3F) << 6) |
              ((unsigned char)s[*pos + 2] & 0x3F);
        *pos += 3;
        return 1;
    }
    if ((c & 0xF8) == 0xF0 && *pos + 3 < len) {
        *cp = ((c & 0x07) << 18) |
              (((unsigned char)s[*pos + 1] & 0x3F) << 12) |
              (((unsigned char)s[*pos + 2] & 0x3F) << 6) |
              ((unsigned char)s[*pos + 3] & 0x3F);
        *pos += 4;
        return 1;
    }
    *cp = c;
    (*pos)++;
    return 1;
}

static char *gpt2_transform_input(const char *text, const GPT2ByteMap *m) {
    size_t n = strlen(text);
    char *out = malloc(n * 4 + 1);
    if (!out) return NULL;
    u32 pos = 0;
    for (size_t i = 0; i < n; i++) {
        int cp = m->byte_to_cp[(unsigned char)text[i]];
        pos += utf8_encode_cp(cp, out + pos);
    }
    out[pos] = '\0';
    return out;
}

static char *gpt2_inverse_output(const char *text, const GPT2ByteMap *m) {
    u32 len = (u32)strlen(text), pos = 0, outp = 0;
    char *out = malloc(len + 1);
    if (!out) return NULL;
    while (utf8_decode_next(text, len, &pos, &(int){0})) {
        u32 old = pos;
        int cp;
        pos = old - 1;
        utf8_decode_next(text, len, &pos, &cp);
        if (cp >= 0 && cp < 512 && m->cp_to_byte[cp] >= 0) {
            out[outp++] = (char)m->cp_to_byte[cp];
        }
    }
    out[outp] = '\0';
    return out;
}

static u32 sample_with_controls(float *logits, u32 vocab_size,
                                const u32 *history, u32 hist_len,
                                float temp, u32 top_k, float top_p,
                                float rep_penalty) {
    for (u32 i = 0; i < hist_len; i++) {
        u32 t = history[i];
        if (t >= vocab_size) continue;
        if (logits[t] > 0) logits[t] /= rep_penalty;
        else logits[t] *= rep_penalty;
    }
    if (temp > 0.0f && temp != 1.0f) {
        for (u32 i = 0; i < vocab_size; i++) logits[i] /= temp;
    }
    if (top_k == 0 || top_k > vocab_size) top_k = vocab_size;

    typedef struct { u32 id; float logit; float prob; } Cand;
    Cand *c = malloc(top_k * sizeof(Cand));
    if (!c) return hs_mlt_sample_greedy(logits, vocab_size);

    for (u32 i = 0; i < top_k; i++) { c[i].id = i; c[i].logit = logits[i]; }
    for (u32 i = top_k; i < vocab_size; i++) {
        u32 min_idx = 0;
        float min_val = c[0].logit;
        for (u32 j = 1; j < top_k; j++) if (c[j].logit < min_val) { min_val = c[j].logit; min_idx = j; }
        if (logits[i] > min_val) { c[min_idx].id = i; c[min_idx].logit = logits[i]; }
    }
    for (u32 i = 0; i < top_k; i++) {
        for (u32 j = i + 1; j < top_k; j++) {
            if (c[j].logit > c[i].logit) { Cand t = c[i]; c[i] = c[j]; c[j] = t; }
        }
    }
    float max = c[0].logit, sum = 0.0f;
    for (u32 i = 0; i < top_k; i++) { c[i].prob = expf(c[i].logit - max); sum += c[i].prob; }
    for (u32 i = 0; i < top_k; i++) c[i].prob /= sum;
    u32 keep = top_k;
    if (top_p > 0.0f && top_p < 1.0f) {
        float cum = 0.0f;
        keep = 0;
        while (keep < top_k) {
            cum += c[keep].prob;
            keep++;
            if (cum >= top_p) break;
        }
    }
    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0.0f;
    u32 pick = c[0].id;
    for (u32 i = 0; i < keep; i++) {
        cum += c[i].prob;
        if (r <= cum) { pick = c[i].id; break; }
    }
    free(c);
    return pick;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s --model <path> [options]\n"
        "  --model <path>        GGUF model file (required)\n"
        "  --prompt <text>       Input prompt (default: control prompt)\n"
        "  --norms <path>        Norms sidecar .bin (optional)\n"
        "  --temp <float>        Sampling temperature (default: 0.432)\n"
        "  --top-k <int>         Top-K (default: 42)\n"
        "  --top-p <float>       Top-P (default: 0.9531)\n"
        "  --rep-penalty <float> Repetition penalty (default: 1.1229)\n"
        "  --n-predict <int>     Max tokens to generate (default: 64)\n"
        "  --gpu                 Use GPU acceleration (default: off)\n",
        prog);
}

int main(int argc, char **argv) {
    const char *model_path  = NULL;
    const char *norms_path  = NULL;
    const char *prompt = "Hypothetically, might reflective recursion be a function of cognition?";
    float temp = 0.432f;
    u32 top_k = 42;
    float top_p = 0.9531f;
    float rep_penalty = 1.1229f;
    int n_predict = 64;
    int use_gpu = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--norms") == 0 && i + 1 < argc) {
            norms_path = argv[++i];
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            temp = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            top_k = (u32)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            top_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--rep-penalty") == 0 && i + 1 < argc) {
            rep_penalty = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--n-predict") == 0 && i + 1 < argc) {
            n_predict = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gpu") == 0) {
            use_gpu = 1;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!model_path) {
        fprintf(stderr, "Error: --model is required\n");
        usage(argv[0]);
        return 1;
    }

    HSMLTernary m;
    hs_mlt_init(&m);
    if (hs_mlt_load_gguf(&m, model_path) != 0) return 1;

    /* Load BF16 norms from sidecar — GGUF norm tensors are corrupt in the
     * upstream converter. The sidecar was extracted directly from the HF
     * safetensors checkpoint via HTTP range request. */
    if (hs_mlt_load_norms_sidecar(&m, norms_path) != 0) {
        fprintf(stderr, "warning: norms sidecar '%s' not loaded, "
                "using GGUF norms (may be corrupt)\n", norms_path);
    }

    if (!m.tokenizer_vocab) {
        fprintf(stderr, "tokenizer vocab not loaded from GGUF\n");
        return 2;
    }
    if (!m.tokenizer_merges || m.num_merges == 0) {
        fprintf(stderr, "warning: no BPE merges loaded, tokenization may be wrong\n");
    }

    /* Encode lm_head to ternary for faster CPU computation */
    if (hs_mlt_lmhead_encode(&m) != 0) {
        fprintf(stderr, "warning: lm_head encoding failed\n");
    }

    /* Initialize GPU for lm_head if requested */
    if (use_gpu) {
        if (gpu_gemm_init() == 0) {
            uint32_t V = m.vocab_size;
            uint32_t H = m.hidden_size;
            
            /* Set dimensions for output buffer large enough for vocab */
            gpu_gemm_set_dims(H, H, V > H * 3 ? V : H * 3);
            
            uint32_t row_bytes = H / 4;
            size_t plane_size = (size_t)V * row_bytes;
            /* Only use first 4 planes for GPU (Stage 1) */
            size_t weight_size = plane_size * 4;
            
            void* gpu_weight_buf = gpu_gemm_alloc_lmhead(weight_size);
            if (gpu_weight_buf) {
                /* Copy only planes 0-3 to GPU */
                for (int k = 0; k < 4; k++) {
                    memcpy(gpu_weight_buf + k * plane_size, m.lm_head_planes[k], plane_size);
                }
                m.gpu_enabled = 1;
                m.gpu_lmhead_ready = 1;
                fprintf(stderr, "GPU: enabled for lm_head (4 planes)\n");
            } else {
                fprintf(stderr, "GPU: failed to allocate lm_head weights\n");
            }
        } else {
            fprintf(stderr, "GPU: not available\n");
        }
    }

    u32 *tokens = malloc((4096 + n_predict) * sizeof(u32));
    float *logits = malloc(m.vocab_size * sizeof(float));
    if (!tokens || !logits) return 4;

    /* BPE encode prompt */
    u32 n = 0;
    tokens[n++] = m.tokenizer_bos;  /* BOS = 128000 */
    n += hs_mlt_bpe_encode(&m, prompt, (u32)strlen(prompt), tokens + n, 4096 - n);

    printf("prompt: %s\n", prompt);
    printf("encoded_tokens=%u\n", n);
    printf("token_ids:");
    for (u32 i = 0; i < n; i++) printf(" %u", tokens[i]);
    printf("\n");
    fflush(stdout);

    HSMLTernarySession s;
    if (hs_mlt_session_init(&s, &m) != 0) return 5;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (hs_mlt_prefill(&s, tokens, n) != 0) return 6;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double prefill_ms = ((double)(t1.tv_sec - t0.tv_sec)*1e3
                       + (double)(t1.tv_nsec - t0.tv_nsec)/1e6);

    /* Get logits after prefill */
    if (hs_mlt_session_logits(&s, logits) != 0) return 7;

    /* Decode loop */
    u32 total = n;
    u32 n_generated = 0;
    double decode_total_ms = 0.0;
    double decode_min_ms = 1e9, decode_max_ms = 0.0;
    srand(42);
    printf("\nresponse:\n");
    for (int i = 0; i < n_predict; i++) {
        u32 tok_id = sample_with_controls(logits, m.vocab_size,
                                          tokens, total, temp, top_k, top_p, rep_penalty);
        tokens[total++] = tok_id;
        if (tok_id == m.tokenizer_eos) break;

        /* Print token (decode Ġ back to space) */
        if (tok_id < m.vocab_size && m.tokenizer_vocab[tok_id]) {
            const char *s_tok = m.tokenizer_vocab[tok_id];
            while (*s_tok) {
                if ((u8)s_tok[0] == 0xC4 && (u8)s_tok[1] == 0xA0) {
                    putchar(' ');
                    s_tok += 2;
                } else {
                    putchar(*s_tok++);
                }
            }
        }
        fflush(stdout);

        /* Run decode step for next token, timed */
        clock_gettime(CLOCK_MONOTONIC, &t0);
        if (hs_mlt_decode(&s, tok_id, logits) != 0) break;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double step_ms = ((double)(t1.tv_sec - t0.tv_sec)*1e3
                        + (double)(t1.tv_nsec - t0.tv_nsec)/1e6);
        decode_total_ms += step_ms;
        if (step_ms < decode_min_ms) decode_min_ms = step_ms;
        if (step_ms > decode_max_ms) decode_max_ms = step_ms;
        n_generated++;
    }

    double avg_ms = n_generated > 0 ? decode_total_ms / n_generated : 0.0;
    double tok_per_sec = n_generated > 0 ? 1000.0 / avg_ms : 0.0;

    printf("\n\nmeta: temp=%.4f top_k=%u top_p=%.4f rep_penalty=%.4f generated=%u\n",
           temp, top_k, top_p, rep_penalty, total - n);
    printf("perf: prefill=%.0fms  decode_avg=%.0fms  decode_min=%.0fms  decode_max=%.0fms\n",
           prefill_ms, avg_ms, decode_min_ms, decode_max_ms);
    printf("      %.3f tokens/sec  (%.0f ms/token)\n", tok_per_sec, avg_ms);

    free(logits);
    free(tokens);
    hs_mlt_session_free(&s);
    hs_mlt_free(&m);
    return 0;
}
