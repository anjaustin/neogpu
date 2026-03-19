/*
 * Simple benchmark test
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

extern void hs_ml_ternary_f32_proj(float *out, const float *in,
                                    const uint8_t *W, uint32_t N, uint32_t K);

static uint64_t ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(void) {
    printf("=== Simple Projection Test ===\n\n");
    
    uint32_t N = 2560, K = 2560;
    
    float* out = malloc(N * sizeof(float));
    float* in = malloc(K * sizeof(float));
    uint8_t* w = malloc((N/4) * K);
    
    printf("Allocated buffers\n");
    
    srand(42);
    for (uint32_t i = 0; i < K; i++) in[i] = (float)(rand() % 100) / 10.0f;
    for (uint32_t i = 0; i < (N/4)*K; i++) w[i] = rand() % 256;
    
    printf("Initialized data\n");
    
    printf("Calling hs_ml_ternary_f32_proj...\n");
    fflush(stdout);
    
    hs_ml_ternary_f32_proj(out, in, w, N, K);
    
    printf("Done! out[0] = %f\n", out[0]);
    
    free(out);
    free(in);
    free(w);
    
    return 0;
}
