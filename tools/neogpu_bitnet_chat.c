#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "hs_ml_infer.h"

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

int main(int argc, char **argv) {
    const char *model_path = "/home/ztflynn/001/neogpu/models/ggml-model-i2_s.gguf";
    const char *prompt = "Hypothetically, might reflective recursion be a function of cognition?";
    float temp = 0.432f;
    u32 top_k = 42;
    float top_p = 0.9531f;
    float rep_penalty = 1.1229f;
    int n_predict = 64;
    if (argc > 1) prompt = argv[1];

    HSMLTernary m;
    hs_mlt_init(&m);
    if (hs_mlt_load_gguf(&m, model_path) != 0) return 1;
    if (!m.tokenizer_vocab) {
        fprintf(stderr, "tokenizer vocab not loaded from GGUF\n");
        return 2;
    }

    GPT2ByteMap map;
    gpt2_bytemap_init(&map);
    char *xform = gpt2_transform_input(prompt, &map);
    if (!xform) return 3;

    HSTokenizer tok;
    hs_tokenizer_init(&tok, m.tokenizer_vocab, m.vocab_size, 0, m.tokenizer_bos, m.tokenizer_eos, 0);

    u32 *tokens = malloc((4096 + n_predict) * sizeof(u32));
    float *logits = malloc(m.vocab_size * sizeof(float));
    char *decoded = malloc(1 << 20);
    if (!tokens || !logits || !decoded) return 4;

    u32 n = 0;
    if (m.tokenizer_bos) tokens[n++] = m.tokenizer_bos;
    n += hs_tokenizer_encode(&tok, xform, (u32)strlen(xform), tokens + n, 4096 - n);

    printf("prompt: %s\n", prompt);
    printf("encoded_tokens=%u\n", n);

    HSMLTernarySession s;
    if (hs_mlt_session_init(&s, &m) != 0) return 5;
    if (hs_mlt_prefill(&s, tokens, n) != 0) return 6;

    u32 total = n;
    srand(42);

    if (hs_mlt_session_logits(&s, logits) != 0) return 7;
    u32 tok_id = sample_with_controls(logits, m.vocab_size, tokens, total, temp, top_k, top_p, rep_penalty);
    tokens[total++] = tok_id;

    for (int i = 1; i < n_predict; i++) {
        if (tok_id == m.tokenizer_eos) break;
        if (hs_mlt_decode(&s, tok_id, logits) != 0) break;
        tok_id = sample_with_controls(logits, m.vocab_size, tokens, total, temp, top_k, top_p, rep_penalty);
        tokens[total++] = tok_id;
    }

    u32 out_len = hs_tokenizer_decode(&tok, tokens + n, total - n, decoded, 1 << 20);
    decoded[out_len] = '\0';
    char *final = gpt2_inverse_output(decoded, &map);

    printf("\nresponse:\n%s\n", final ? final : decoded);
    printf("\nmeta: temp=%.4f top_k=%u top_p=%.4f rep_penalty=%.4f generated=%u\n",
           temp, top_k, top_p, rep_penalty, total - n);

    free(final);
    free(decoded);
    free(logits);
    free(tokens);
    free(xform);
    hs_tokenizer_free(&tok);
    hs_mlt_session_free(&s);
    hs_mlt_free(&m);
    return 0;
}
