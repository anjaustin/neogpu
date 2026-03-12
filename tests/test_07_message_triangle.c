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

    /* Provide a CPU-side vertex buffer in bank 0: x,y,r,g,b per vertex */
    HSBuffer* buf = hs_gpu_get_buffer(&gpu, 0);
    if (!hs_buffer_init(buf, 0, 256)) {
        fprintf(stderr, "Failed to init buffer bank 0\n");
        return 1;
    }
    /* v0 */
    hs_buffer_set_f32(buf, 0,  0.0f);
    hs_buffer_set_f32(buf, 1,  0.6f);
    hs_buffer_set_f32(buf, 2,  1.0f);
    hs_buffer_set_f32(buf, 3,  0.0f);
    hs_buffer_set_f32(buf, 4,  0.0f);
    /* v1 */
    hs_buffer_set_f32(buf, 5, -0.6f);
    hs_buffer_set_f32(buf, 6, -0.6f);
    hs_buffer_set_f32(buf, 7,  0.0f);
    hs_buffer_set_f32(buf, 8,  1.0f);
    hs_buffer_set_f32(buf, 9,  0.0f);
    /* v2 */
    hs_buffer_set_f32(buf, 10, 0.6f);
    hs_buffer_set_f32(buf, 11, -0.6f);
    hs_buffer_set_f32(buf, 12, 0.0f);
    hs_buffer_set_f32(buf, 13, 0.0f);
    hs_buffer_set_f32(buf, 14, 1.0f);
    buf->length = 15 * 4;

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
