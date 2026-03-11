/*
 * Benchmark: Compare Haxe-style vs C/NEON implementation
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "hs_gpu.h"
#include "hs_math_neon.h"

#define BENCH_FRAMES 10000

/* Simulated Haxe-style message processing */
typedef struct {
    int to, from, op;
    float payload[16];
} HaxeMessage;

static void benchmark_haxe_style(int frames) {
    printf("\n--- Simulated Haxe Message Benchmark ---\n");
    
    HaxeMessage* msgs = (HaxeMessage*)malloc(sizeof(HaxeMessage) * 20);
    
    clock_t start = clock();
    for (int f = 0; f < frames; f++) {
        int count = 0;
        msgs[count].op = 1; count++;  /* CLEAR */
        msgs[count].op = 2; count++;  /* SET_SHADER */
        msgs[count].op = 3; count++;  /* SET_PARAM */
        msgs[count].op = 4; count++;  /* SET_CAMERA */
        msgs[count].op = 5; count++;  /* LOAD_BUFFER */
        msgs[count].op = 6; count++;  /* DRAW */
        
        for (int i = 0; i < count; i++) {
            switch(msgs[i].op) {
                case 1: case 2: case 3: case 4: case 5: case 6: break;
            }
        }
    }
    clock_t end = clock();
    double ms = (end - start) * 1000.0 / CLOCKS_PER_SEC;
    printf("  Haxe-style (simulated): %.2f ms for %d frames (%.0f fps)\n", 
           ms, frames, frames * 1000.0 / ms);
    
    free(msgs);
}

static void benchmark_c_neon(int frames) {
    printf("\n--- C/NEON Message Benchmark ---\n");
    
    static HSGpu gpu;
    hs_gpu_init(&gpu);
    
    clock_t start = clock();
    for (int f = 0; f < frames; f++) {
        hs_gpu_start_recording(&gpu);
        
        vec4 color = v4_make(0.1f, 0.2f, 0.3f, 1.0f);
        hs_gpu_clear(&gpu, color);
        
        mat4 cam = m4_identity();
        hs_gpu_set_camera(&gpu, cam);
        
        hs_gpu_set_shader(&gpu, 0);
        
        vec4 param = v4_make(1, 0, 0, 0);
        hs_gpu_set_param(&gpu, 0, param);
        
        hs_gpu_load_buffer(&gpu, 0);
        hs_gpu_draw(&gpu, 0);
        
        hs_gpu_process(&gpu);
        hs_gpu_stop_recording(&gpu);
    }
    clock_t end = clock();
    double ms = (end - start) * 1000.0 / CLOCKS_PER_SEC;
    printf("  C/NEON: %.2f ms for %d frames (%.0f fps)\n", 
           ms, frames, frames * 1000.0 / ms);
}

static void benchmark_math_only(int iterations) {
    printf("\n--- Math-Only Benchmark ---\n");
    
    mat4 m1 = m4_identity();
    mat4 m2 = m4_translation(1, 2, 3);
    mat4 m3 = m4_rotation_x(0.5f);
    
    /* C/NEON */
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        mat4 r = m4_multiply(m1, m2);
        r = m4_multiply(r, m3);
    }
    clock_t end = clock();
    double neon_ms = (end - start) * 1000.0 / CLOCKS_PER_SEC;
    
    /* Simulated scalar (what Haxe does) */
    float m1f[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float m2f[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 1,2,3,1};
    
    start = clock();
    for (int i = 0; i < iterations; i++) {
        float result[16] = {0};
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                for (int k = 0; k < 4; k++) {
                    result[r*4+c] += m1f[r*4+k] * m2f[k*4+c];
                }
            }
        }
    }
    end = clock();
    double scalar_ms = (end - start) * 1000.0 / CLOCKS_PER_SEC;
    
    printf("  Scalar (Haxe-style): %.2f ms for %d mat4 mul\n", scalar_ms, iterations);
    printf("  NEON SIMD:           %.2f ms for %d mat4 mul\n", neon_ms, iterations);
    printf("  Speedup:             %.1fx\n", scalar_ms / neon_ms);
}

int main(void) {
    printf("========================================\n");
    printf("  PicoGPU Benchmark: Haxe vs C/NEON\n");
    printf("========================================\n");
    
    benchmark_c_neon(10);
    benchmark_haxe_style(BENCH_FRAMES);
    benchmark_c_neon(BENCH_FRAMES);
    benchmark_math_only(100000);
    
    printf("\n========================================\n");
    return 0;
}
