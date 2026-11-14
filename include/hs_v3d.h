/* hs_v3d: NeoGPU's direct path to the Raspberry Pi 4 GPU (V3D 4.2) through
 * the kernel's v3d driver. Dependency-free: kernel UAPI headers only, no
 * Mesa, no libdrm, no GLES. Programs come from hs_qpu (include/hs_qpu.h).
 *
 * Measured costs (docs/benchmarks/HARDWARE_MAP_2026-11-15.md):
 * - a buffer object (BO) costs 20 us (4K) to 20 ms (64M) to create, so
 *   allocate up front;
 * - a dispatch costs 22-75 us from submit to completion, against a few us of
 *   GPU work;
 * - CPU reads of a BO mapping run at 1.48 GB/s (uncached); writes run at
 *   4.16 GB/s.
 */
#ifndef HS_V3D_H
#define HS_V3D_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int fd;
} hs_v3d;

typedef struct {
    uint32_t handle;
    uint32_t gpu; /* GPU virtual address */
    size_t size;
    uint8_t* map; /* CPU mapping (uncached) */
} hs_v3d_bo;

/* Opens the first render node driven by "v3d" (or `path` if given).
 * Returns 0, or a negative errno. */
int hs_v3d_open(hs_v3d* dev, const char* path);
void hs_v3d_close(hs_v3d* dev);

int hs_v3d_bo_new(hs_v3d* dev, hs_v3d_bo* bo, size_t size);
void hs_v3d_bo_free(hs_v3d* dev, hs_v3d_bo* bo);
/* waits until no submitted job references the BO; 0, -ETIME or -errno */
int hs_v3d_bo_wait(hs_v3d* dev, const hs_v3d_bo* bo, uint64_t timeout_ns);

/* Copies a program into `code` at byte offset `off` (8-byte aligned).
 * Returns the program's GPU address, or 0 if it does not fit. */
uint32_t hs_v3d_load(hs_v3d_bo* code, size_t off, const uint64_t* prog, uint32_t n);

/* One compute dispatch: workgroup counts batches x 1 x 1, workgroup size 16,
 * 16 workgroups per supergroup, and CFG4 = batches - 1. That runs `batches`
 * 16-lane batches. A program reads its batch number from the register the
 * hardware presets at entry: rf0 & 15 (the workgroup x; 0..batches-1, once
 * each, in no fixed order), before it overwrites rf0.
 * Do not use the QPU thread index for this. Until 2026-11-17 programs here
 * read (tidx >> 2) & 15, which is only distinct per batch while nothing else
 * uses the GPU: beside a renderer 173 of 78,894 eight-batch jobs gave every
 * batch its own number, and the gather kernel wrote wrong words in 3,273 of
 * 3,279 jobs; with rf0, 79,177 of 79,177 (tools/p0_5_batch_id_probe,
 * docs/audit/HARDWARE_HANDOFF_P0.md P0-5).
 * Only batches = 1 or 8 is accepted: the other counts were never measured. `bos` lists every BO the job touches (code, uniforms, data), so
 * the kernel can track them. Returns 0 or -errno (-EINVAL for a refused
 * descriptor). */
typedef struct {
    uint32_t code_gpu;  /* from hs_v3d_load */
    uint32_t unif_gpu;  /* uniform stream */
    uint32_t batches;   /* 1 or 8 */
    const uint32_t* bos; /* BO handles */
    uint32_t nbos;
    uint32_t perfmon;   /* 0, or a perfmon id */
} hs_v3d_dispatch_desc;
int hs_v3d_dispatch(hs_v3d* dev, const hs_v3d_dispatch_desc* d);

#endif
