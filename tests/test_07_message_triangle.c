/*
 * Test 07: Message-driven Triangle (Backend)
 *
 * Uses hs_gpu_* messages to build a render list and a GLES backend
 * to execute it.
 */

#include <stdio.h>
#include <unistd.h>

#include "hs_gpu.h"
#include "hs_backend_gles.h"

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    printf("=== Test 07: Message-driven Triangle ===\n");

    static HSGpu gpu;
    hs_gpu_init(&gpu);

    HSBackend backend = hs_backend_gles_create();
    hs_gpu_attach_backend(&gpu, &backend);

    for (int f = 0; f < 300; f++) {
        hs_gpu_begin_frame(&gpu);

        hs_gpu_clear(&gpu, v4_make(0.08f, 0.08f, 0.10f, 1.0f));
        hs_gpu_draw(&gpu, 0);
        hs_gpu_process(&gpu);

        hs_gpu_end_frame(&gpu);
        usleep(16000);
    }

    printf("PASS: Message-driven triangle test\n");
    return 0;
}
