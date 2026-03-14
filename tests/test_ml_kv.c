/*
 * NeoGPU ML - KV Cache & Attention Test
 * Tests attention mechanism correctness
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "hs_ml.h"

#define EPSILON 0.001f

int test_kv_cache_init(void) {
    printf("\n=== Test: KV Cache Init ===\n");
    
    HSKVCache* cache = calloc(1, sizeof(HSKVCache));
    hs_kv_cache_init(cache, 128, 4, 2, 64);
    
    int passed = 1;
    if (cache->max_seq != 128) { printf("  FAIL: max_seq\n"); passed = 0; }
    if (cache->num_heads != 4) { printf("  FAIL: num_heads\n"); passed = 0; }
    if (cache->num_kv_heads != 2) { printf("  FAIL: num_kv_heads\n"); passed = 0; }
    if (cache->head_dim != 64) { printf("  FAIL: head_dim\n"); passed = 0; }
    if (cache->k_cache == NULL) { printf("  FAIL: k_cache alloc\n"); passed = 0; }
    if (cache->v_cache == NULL) { printf("  FAIL: v_cache alloc\n"); passed = 0; }
    if (cache->cache_len != 0) { printf("  FAIL: cache_len init\n"); passed = 0; }
    
    if (passed) printf("  PASSED: KV cache initialized correctly\n");
    
    hs_kv_cache_free(cache);
    free(cache);
    return passed;
}

int test_kv_cache_clear(void) {
    printf("\n=== Test: KV Cache Clear ===\n");
    
    HSKVCache* cache = calloc(1, sizeof(HSKVCache));
    hs_kv_cache_init(cache, 32, 2, 2, 32);
    
    /* Fill with some data */
    for (u32 i = 0; i < 10 * cache->num_kv_heads * cache->head_dim; i++) {
        cache->k_cache[i] = 1.0f;
        cache->v_cache[i] = 2.0f;
    }
    cache->cache_len = 10;
    
    hs_kv_cache_clear(cache);
    
    int passed = 1;
    if (cache->cache_len != 0) {
        printf("  FAIL: cache_len = %u after clear\n", cache->cache_len);
        passed = 0;
    }
    
    /* Check data is zeroed */
    for (u32 i = 0; i < cache->max_seq * cache->num_kv_heads * cache->head_dim; i++) {
        if (cache->k_cache[i] != 0.0f || cache->v_cache[i] != 0.0f) {
            printf("  FAIL: cache not cleared at index %u\n", i);
            passed = 0;
            break;
        }
    }
    
    if (passed) printf("  PASSED: KV cache cleared correctly\n");
    
    hs_kv_cache_free(cache);
    free(cache);
    return passed;
}

int test_rope_basic(void) {
    printf("\n=== Test: RoPE Basic ===\n");
    
    const u32 num_heads = 2;
    const u32 head_dim = 32;
    const float theta = 10000.0f;
    
    float* q = malloc(num_heads * head_dim * sizeof(float));
    float* k = malloc(num_heads * head_dim * sizeof(float));
    
    /* Identity input */
    for (u32 i = 0; i < num_heads * head_dim; i++) {
        q[i] = 1.0f;
        k[i] = 1.0f;
    }
    
    hs_rope_apply(q, k, num_heads, head_dim, 0, theta);
    
    /* Position 0: angles are all 0, cos(0)=1, sin(0)=0
     * So q[0] = 1*1 - 1*0 = 1
     *    q[half] = 1*0 + 1*1 = 1
     */
    int passed = 1;
    for (u32 h = 0; h < num_heads; h++) {
        for (u32 d = 0; d < head_dim / 2; d++) {
            float expected = 1.0f;
            if (fabsf(q[h * head_dim + d] - expected) > EPSILON ||
                fabsf(q[h * head_dim + d + head_dim/2] - expected) > EPSILON) {
                printf("  FAIL: q[%u][%u] = %f, q[%u][%u] = %f\n",
                       h, d, q[h*head_dim + d], h, d+head_dim/2, q[h*head_dim + d+head_dim/2]);
                passed = 0;
            }
        }
    }
    
    /* Test position 1 - should change */
    float q_copy[2][32];
    memcpy(q_copy, q, sizeof(q_copy));
    
    hs_rope_apply(q, k, num_heads, head_dim, 1, theta);
    
    /* Position 1 should produce different values than position 0 */
    int changed = 0;
    for (u32 h = 0; h < num_heads; h++) {
        for (u32 d = 0; d < head_dim; d++) {
            if (fabsf(q[h*head_dim + d] - q_copy[h][d]) > EPSILON) {
                changed = 1;
                break;
            }
        }
        if (changed) break;
    }
    
    if (!changed) {
        printf("  FAIL: RoPE did not change values at position 1\n");
        passed = 0;
    }
    
    if (passed) printf("  PASSED: RoPE applied correctly\n");
    
    free(q);
    free(k);
    return passed;
}

int test_attention_correctness(void) {
    printf("\n=== Test: Attention Correctness ===\n");
    
    const u32 num_heads = 2;
    const u32 head_dim = 4;
    const u32 seq_len = 2;
    
    float* q = malloc(num_heads * head_dim * sizeof(float));
    float* k = malloc(seq_len * head_dim * sizeof(float));
    float* v = malloc(seq_len * head_dim * sizeof(float));
    float* out = malloc(num_heads * head_dim * sizeof(float));
    
    /* Q matches K[0], should favor V[0] */
    q[0] = 1.0f; q[1] = 0.0f; q[2] = 0.0f; q[3] = 0.0f;
    q[4] = 1.0f; q[5] = 0.0f; q[6] = 0.0f; q[7] = 0.0f;
    
    k[0] = 1.0f; k[1] = 0.0f; k[2] = 0.0f; k[3] = 0.0f;
    k[4] = 0.0f; k[5] = 1.0f; k[6] = 0.0f; k[7] = 0.0f;
    
    v[0] = 1.0f; v[1] = 2.0f; v[2] = 3.0f; v[3] = 4.0f;
    v[4] = 5.0f; v[5] = 6.0f; v[6] = 7.0f; v[7] = 8.0f;
    
    hs_attention_score(q, k, v, out, num_heads, head_dim, seq_len);
    
    /* Compute expected: scale = 0.5, softmax([0.5, 0]) = [0.622, 0.378] */
    float expected[4];
    float exp0 = expf(0.5f);
    float exp1 = expf(0.0f);
    float sum = exp0 + exp1;
    float w0 = exp0 / sum;
    float w1 = exp1 / sum;
    for (u32 d = 0; d < head_dim; d++) {
        expected[d] = w0 * v[d] + w1 * v[head_dim + d];
    }
    
    int passed = 1;
    for (u32 d = 0; d < head_dim; d++) {
        if (fabsf(out[d] - expected[d]) > 0.01f) {
            printf("  FAIL: head0 out[%u] = %f, expected %f\n", d, out[d], expected[d]);
            passed = 0;
        }
    }
    
    if (passed) printf("  PASSED: Attention computes weighted sum correctly\n");
    
    free(q);
    free(k);
    free(v);
    free(out);
    return passed;
}

int test_attention_uniform(void) {
    printf("\n=== Test: Attention Uniform ===\n");
    
    const u32 num_heads = 2;
    const u32 head_dim = 4;
    const u32 seq_len = 2;
    
    float* q = malloc(num_heads * head_dim * sizeof(float));
    float* k = malloc(seq_len * head_dim * sizeof(float));
    float* v = malloc(seq_len * head_dim * sizeof(float));
    float* out = malloc(num_heads * head_dim * sizeof(float));
    
    /* Q = uniform, K = uniform */
    for (u32 i = 0; i < num_heads * head_dim; i++) {
        q[i] = 1.0f;
    }
    for (u32 i = 0; i < seq_len * head_dim; i++) {
        k[i] = 1.0f;
        v[i] = (float)(i + 1);
    }
    
    hs_attention_score(q, k, v, out, num_heads, head_dim, seq_len);
    
    /* With uniform Q and uniform K, output should be average of V rows */
    int passed = 1;
    for (u32 h = 0; h < num_heads; h++) {
        for (u32 d = 0; d < head_dim; d++) {
            float expected = (v[d] + v[head_dim + d]) / 2.0f;
            if (fabsf(out[h*head_dim + d] - expected) > 0.1f) {
                printf("  FAIL: head%u out[%u] = %f, expected %f\n", h, d, out[h*head_dim + d], expected);
                passed = 0;
            }
        }
    }
    
    if (passed) printf("  PASSED: Attention computes uniform correctly\n");
    
    free(q);
    free(k);
    free(v);
    free(out);
    return passed;
}

int main(void) {
    printf("NeoGPU ML - KV Cache & Attention Tests\n");
    printf("========================================\n");
    
    int total = 0;
    int passed = 0;
    
    printf("Running test 1...\n");
    total++; passed += test_kv_cache_init();
    printf("Test 1 done\n");
    
    printf("Running test 2...\n");
    total++; passed += test_kv_cache_clear();
    printf("Test 2 done\n");
    
    printf("Running test 3...\n");
    total++; passed += test_rope_basic();
    printf("Test 3 done\n");
    
    printf("Running test 4...\n");
    total++; passed += test_attention_correctness();
    printf("Test 4 done\n");
    
    printf("Running test 5...\n");
    total++; passed += test_attention_uniform();
    printf("Test 5 done\n");
    
    printf("\n========================================\n");
    printf("Results: %d/%d tests passed\n", passed, total);
    
    return (passed == total) ? 0 : 1;
}
