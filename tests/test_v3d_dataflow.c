/*
 * V3D CSD Data Flow Test - Copy data to GPU, run shader, copy back
 *
 * Builds on Pi:
 *   gcc -O3 -march=armv8-a+simd+fp16 -mtune=cortex-a72 -ffast-math \
 *       -funroll-loops -DNDEBUG \
 *       tests/test_v3d_dataflow.c -o /tmp/test_v3d_dataflow -ldrm -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <drm.h>
#include <drm/v3d_drm.h>
#include <sys/ioctl.h>
#include <time.h>

#define V3D_DEV "/dev/dri/renderD128"
#define SHADER_FILE "/home/ztflynn/001/neogpu/tests/ternary_v3d.spv"

#define TEST_SIZE 256

static uint64_t ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static int v3d_create_bo(int fd, size_t size, uint32_t* handle) {
    struct drm_v3d_create_bo bo = { .size = size };
    if (ioctl(fd, DRM_IOCTL_V3D_CREATE_BO, &bo) != 0) return -1;
    *handle = bo.handle;
    return 0;
}

static void* v3d_mmap_bo(int fd, uint32_t handle, size_t size) {
    struct drm_v3d_mmap_bo mmap_bo = { .handle = handle, .flags = 0 };
    if (ioctl(fd, DRM_IOCTL_V3D_MMAP_BO, &mmap_bo) != 0) return NULL;
    void* ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mmap_bo.offset);
    return ptr == MAP_FAILED ? NULL : ptr;
}

static int v3d_submit_csd(int fd, uint32_t shader_bo, size_t shader_size,
                          uint32_t* bo_handles, int bo_count,
                          uint32_t wg_x, uint32_t wg_y, uint32_t wg_z) {
    struct drm_v3d_submit_csd csd = {
        .cfg[0] = wg_x,
        .cfg[1] = wg_y,
        .cfg[2] = wg_z,
        .cfg[3] = 8,
        .cfg[4] = 1,
        .cfg[5] = 1,
        .cfg[6] = 0,
        .coef[0] = 0, .coef[1] = 0, .coef[2] = 0, .coef[3] = 0,
        .bo_handles = (uint64_t)(uintptr_t)bo_handles,
        .bo_handle_count = bo_count,
        .in_sync = 0,
        .out_sync = 0,
        .perfmon_id = 0,
        .extensions = 0,
        .flags = 0,
        .pad = 0,
    };
    
    if (ioctl(fd, DRM_IOCTL_V3D_SUBMIT_CSD, &csd) != 0) {
        fprintf(stderr, "CSD failed: %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

int main(void) {
    printf("=== V3D Data Flow Test ===\n\n");
    
    // Test data
    float input_data[TEST_SIZE];
    float output_data[TEST_SIZE];
    float expected[TEST_SIZE];
    
    for (int i = 0; i < TEST_SIZE; i++) {
        input_data[i] = (float)i;
        expected[i] = input_data[i] * 2.0f;  // Shader multiplies by 2
        output_data[i] = 0;
    }
    
    printf("Test data: input[0]=%f, expected[0]=%f\n", input_data[0], expected[0]);
    
    int fd = open(V3D_DEV, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "Failed to open %s: %s\n", V3D_DEV, strerror(errno));
        return 1;
    }
    printf("Opened %s\n", V3D_DEV);
    
    // Load shader
    int sfd = open(SHADER_FILE, O_RDONLY);
    struct stat st;
    fstat(sfd, &st);
    size_t shader_size = st.st_size;
    void* shader_data = mmap(NULL, shader_size, PROT_READ, MAP_PRIVATE, sfd, 0);
    close(sfd);
    
    // Create shader BO
    uint32_t bo_shader, bo_input, bo_output;
    v3d_create_bo(fd, shader_size, &bo_shader);
    v3d_create_bo(fd, sizeof(input_data), &bo_input);
    v3d_create_bo(fd, sizeof(output_data), &bo_output);
    
    void* map_shader = v3d_mmap_bo(fd, bo_shader, shader_size);
    void* map_input = v3d_mmap_bo(fd, bo_input, sizeof(input_data));
    void* map_output = v3d_mmap_bo(fd, bo_output, sizeof(output_data));
    
    memcpy(map_shader, shader_data, shader_size);
    munmap(shader_data, shader_size);
    memcpy(map_input, input_data, sizeof(input_data));
    
    printf("Copied data to GPU\n");
    
    uint32_t bo_handles[3] = { bo_shader, bo_input, bo_output };
    
    uint64_t start = ns_now();
    v3d_submit_csd(fd, bo_shader, shader_size, bo_handles, 3, TEST_SIZE / 8, 1, 1);
    uint64_t end = ns_now();
    
    printf("GPU execution time: %.3f ms\n", (end - start) / 1000000.0);
    
    // Copy output back
    memcpy(output_data, map_output, sizeof(output_data));
    
    printf("Output[0] = %f (expected %f)\n", output_data[0], expected[0]);
    
    // Verify
    int errors = 0;
    for (int i = 0; i < TEST_SIZE; i++) {
        if (output_data[i] != expected[i]) {
            errors++;
            if (errors <= 3) {
                printf("ERROR at %d: got %f, expected %f\n", i, output_data[i], expected[i]);
            }
        }
    }
    
    if (errors == 0) {
        printf("\n=== SUCCESS: All %d values correct! ===\n", TEST_SIZE);
    } else {
        printf("\n=== FAILED: %d errors ===\n", errors);
    }
    
    close(fd);
    return errors;
}
