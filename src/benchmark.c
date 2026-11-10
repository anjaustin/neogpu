/*
 * Benchmark: Compare Haxe-style vs C/NEON implementation
 *
 * Historical comparison from the port (see docs/lineage). Built by
 * `make bench-legacy`; not part of the runtime or the test gate.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
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
    if (!msgs) {
        fprintf(stderr, "out of memory\n");
        return;
    }
    
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

static volatile float g_sink; /* keeps results observable so loops are not elided */

static void benchmark_math_only(int iterations) {
    printf("\n--- Math-Only Benchmark ---\n");

    /* Same work on both sides: one dependent 4x4 multiply per iteration by
     * a rotation (orthonormal, so values stay bounded). */
    mat4 rot = m4_rotation_x(0.5f);
    float rotf[16];
    m4_to_array(rot, rotf);

    /* C/NEON */
    mat4 acc = m4_identity();
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) acc = m4_multiply(acc, rot);
    clock_t end = clock();
    double neon_ms = (end - start) * 1000.0 / CLOCKS_PER_SEC;
    float out[16];
    m4_to_array(acc, out);
    g_sink = out[5];

    /* Scalar (what the Haxe version does), same column-major layout. */
    float accf[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    start = clock();
    for (int i = 0; i < iterations; i++) {
        float result[16];
        for (int c = 0; c < 4; c++)
            for (int r = 0; r < 4; r++) {
                float sum = 0;
                for (int k = 0; k < 4; k++) sum += accf[k * 4 + r] * rotf[c * 4 + k];
                result[c * 4 + r] = sum;
            }
        memcpy(accf, result, sizeof(accf));
    }
    end = clock();
    double scalar_ms = (end - start) * 1000.0 / CLOCKS_PER_SEC;
    g_sink = accf[5];

    printf("  Scalar (Haxe-style): %.2f ms for %d mat4 mul\n", scalar_ms, iterations);
    printf("  NEON SIMD:           %.2f ms for %d mat4 mul\n", neon_ms, iterations);
    printf("  Results agree:       %s\n", fabsf(out[5] - accf[5]) < 1e-3f ? "yes" : "NO");
    if (neon_ms > 0) printf("  Speedup:             %.1fx\n", scalar_ms / neon_ms);
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
