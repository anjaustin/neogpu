/*
 * V3D CSD Test - Direct DRM Compute Shader Dispatch
 *
 * Builds on Pi:
 *   gcc -O3 -march=armv8-a+simd+fp16 -mtune=cortex-a72 -ffast-math \
 *       -funroll-loops -DNDEBUG \
 *       tests/test_v3d_csd.c -o /tmp/test_v3d_csd -lv3d -ldrm -lm
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

#define V3D_DEV "/dev/dri/renderD128"
#define SHADER_FILE "ternary_v3d.spv"

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
                          uint32_t uniform_bo, uint32_t wg_x, uint32_t wg_y, uint32_t wg_z) {
    // cfg[0-2]: number of workgroups
    // cfg[3-5]: local workgroup size
    // cfg[6]: config flags
    uint32_t cfg[7] = {
        wg_x, wg_y, wg_z,  // Workgroup counts
        8, 1, 1,           // Local size (must match shader)
        0                  // Flags
    };
    
    // BO handles: shader, uniform
    uint32_t bo_handles[2] = { shader_bo, uniform_bo };
    
    struct drm_v3d_submit_csd csd = {
        .cfg[0] = wg_x,
        .cfg[1] = wg_y,
        .cfg[2] = wg_z,
        .cfg[3] = 8,  // local_size_x from shader
        .cfg[4] = 1,  // local_size_y from shader
        .cfg[5] = 1,  // local_size_z from shader
        .cfg[6] = 0,
        .coef[0] = 0,
        .coef[1] = 0,
        .coef[2] = 0,
        .coef[3] = 0,
        .bo_handles = (uint64_t)(uintptr_t)bo_handles,
        .bo_handle_count = 2,
        .in_sync = 0,
        .out_sync = 0,
        .perfmon_id = 0,
        .extensions = 0,
        .flags = 0,
        .pad = 0,
    };
    
    int ret = ioctl(fd, DRM_IOCTL_V3D_SUBMIT_CSD, &csd);
    if (ret != 0) {
        fprintf(stderr, "CSD ioctl failed: %s (ret=%d)\n", strerror(errno), ret);
        return -1;
    }
    printf("CSD job submitted successfully!\n");
    return 0;
}

int main(void) {
    printf("=== V3D CSD Test ===\n\n");
    
    int fd = open(V3D_DEV, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "Failed to open %s: %s\n", V3D_DEV, strerror(errno));
        return 1;
    }
    printf("Opened %s\n", V3D_DEV);
    
    // Load shader from file
    int sfd = open(SHADER_FILE, O_RDONLY);
    if (sfd < 0) {
        fprintf(stderr, "Failed to open %s: %s\n", SHADER_FILE, strerror(errno));
        close(fd);
        return 1;
    }
    
    struct stat st;
    fstat(sfd, &st);
    size_t shader_size = st.st_size;
    printf("Shader size: %zu bytes\n", shader_size);
    
    void* shader_data = mmap(NULL, shader_size, PROT_READ, MAP_PRIVATE, sfd, 0);
    close(sfd);
    if (!shader_data) {
        fprintf(stderr, "Failed to mmap shader\n");
        close(fd);
        return 1;
    }
    
    uint32_t shader_bo;
    if (v3d_create_bo(fd, shader_size, &shader_bo) != 0) {
        fprintf(stderr, "Failed to create shader BO\n");
        close(fd);
        return 1;
    }
    printf("Created shader BO: %u\n", shader_bo);
    
    void* shader_map = v3d_mmap_bo(fd, shader_bo, shader_size);
    if (!shader_map) {
        fprintf(stderr, "Failed to mmap shader BO\n");
        close(fd);
        return 1;
    }
    memcpy(shader_map, shader_data, shader_size);
    munmap(shader_data, shader_size);
    printf("Copied shader to BO\n");
    
    uint32_t uniform_bo;
    if (v3d_create_bo(fd, 256, &uniform_bo) != 0) {
        fprintf(stderr, "Failed to create uniform BO\n");
        close(fd);
        return 1;
    }
    printf("Created uniform BO: %u\n", uniform_bo);
    
    printf("Submitting CSD job (8x1x1 workgroups)...\n");
    if (v3d_submit_csd(fd, shader_bo, shader_size, uniform_bo, 8, 1, 1) != 0) {
        fprintf(stderr, "CSD submit failed\n");
        close(fd);
        return 1;
    }
    printf("CSD job completed!\n");
    
    close(fd);
    printf("\n=== SUCCESS ===\n");
    return 0;
}
