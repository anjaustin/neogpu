/*
 * Test QPU Shader Submission
 * 
 * Builds on Pi:
 *   gcc -O3 -march=armv8-a+simd+fp16 -mtune=cortex-a72 -ffast-math \
 *       -funroll-loops -DNDEBUG -Iinclude \
 *       src/hs_ml_v3d_gpu.c tests/test_qpu_submit.c -o /tmp/test_qpu_submit \
 *       -lv3d -lm -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <xf86drm.h>
#include <drm/v3d_drm.h>

#define V3D_DEV_PATH "/dev/dri/card1"

extern uint32_t ternary_gemm_bin[];
extern uint32_t ternary_gemm_bin_size;

static int v3d_get_param(int fd, uint32_t param, uint64_t* value) {
    struct drm_v3d_get_param p = { .param = param };
    if (ioctl(fd, DRM_IOCTL_V3D_GET_PARAM, &p) != 0) {
        return -1;
    }
    *value = p.value;
    return 0;
}

static int v3d_create_bo(int fd, size_t size, uint32_t* handle_out) {
    struct drm_v3d_create_bo create = { .size = size };
    if (ioctl(fd, DRM_IOCTL_V3D_CREATE_BO, &create) != 0) {
        return -1;
    }
    *handle_out = create.handle;
    return 0;
}

static int v3d_mmap_bo(int fd, uint32_t handle, size_t size, void** map_out) {
    struct drm_v3d_mmap_bo mmap_bo = { .handle = handle, .flags = 0 };
    if (ioctl(fd, DRM_IOCTL_V3D_MMAP_BO, &mmap_bo) != 0) {
        return -1;
    }
    void* map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mmap_bo.offset);
    if (map == MAP_FAILED) {
        return -1;
    }
    *map_out = map;
    return 0;
}

int main(void) {
    printf("=== QPU Shader Submission Test ===\n\n");
    
    int fd = open(V3D_DEV_PATH, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "Failed to open %s: %s\n", V3D_DEV_PATH, strerror(errno));
        return 1;
    }
    printf("Opened V3D device\n");
    
    uint64_t ver = 0;
    v3d_get_param(fd, DRM_V3D_PARAM_V3D_UCODE_ID, &ver);
    printf("V3D UCODE ID: %lu\n", ver);
    
    uint64_t qpus = 0;
    v3d_get_param(fd, DRM_V3D_PARAM_V3D_QPU_COUNT, &qpus);
    printf("QPU count: %lu\n", qpus);
    
    uint64_t sop = 0;
    v3d_get_param(fd, DRM_V3D_PARAM_SUPPORTS_OQ, &sop);
    printf("Supports QPU: %lu\n", sop);
    
    /* Create BO for shader */
    size_t shader_size = 256;
    uint32_t shader_handle = 0;
    if (v3d_create_bo(fd, shader_size, &shader_handle) != 0) {
        fprintf(stderr, "Failed to create shader BO\n");
        close(fd);
        return 1;
    }
    printf("Created shader BO (handle=%u)\n", shader_handle);
    
    /* Map and copy shader */
    void* shader_map = NULL;
    if (v3d_mmap_bo(fd, shader_handle, shader_size, &shader_map) != 0) {
        fprintf(stderr, "Failed to mmap shader BO\n");
        close(fd);
        return 1;
    }
    printf("Mapped shader BO\n");
    
    /* Copy shader to BO (use placeholder for now) */
    memset(shader_map, 0, shader_size);
    printf("Shader memory initialized\n");
    
    /* Create uniform BO */
    size_t uniform_size = 256;
    uint32_t uniform_handle = 0;
    if (v3d_create_bo(fd, uniform_size, &uniform_handle) != 0) {
        fprintf(stderr, "Failed to create uniform BO\n");
        close(fd);
        return 1;
    }
    printf("Created uniform BO (handle=%u)\n", uniform_handle);
    
    /* Submit QPU job */
    uint32_t bo_handles[2] = { shader_handle, uniform_handle };
    
    struct drm_v3d_submit_cs submit = {
        .qpu = 1,
        .qpu_offsets = 0,
        .qpu_size = shader_size,
        .qpu_bo_handles = (uint64_t)(uintptr_t)bo_handles,
        .uniforms = 0,
        .qpu_count = 1,
    };
    
    printf("Submitting QPU job...\n");
    int ret = ioctl(fd, DRM_IOCTL_V3D_SUBMIT_CS, &submit);
    if (ret != 0) {
        fprintf(stderr, "QPU submit failed: %s\n", strerror(errno));
    } else {
        printf("QPU job submitted successfully!\n");
    }
    
    /* Cleanup */
    munmap(shader_map, shader_size);
    close(fd);
    
    printf("\n=== Test Complete ===\n");
    return ret != 0 ? 1 : 0;
}
