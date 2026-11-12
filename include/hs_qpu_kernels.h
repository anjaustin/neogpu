/* hs_qpu_kernels: QPU programs NeoGPU ships, built with hs_qpu. Each one is
 * word-identical to its reference-assembler twin (make qpu-encode-check) and
 * runs on the GPU in tests/test_v3d_compute.c. */
#ifndef HS_QPU_KERNELS_H
#define HS_QPU_KERNELS_H

#include "hs_qpu.h"

/* out[e] = table[idx[e]] for 4-byte elements; dispatch with `batches`
 * (1 or 8) and k (1, 2, 4 or 8) loads in flight per QPU.
 * Uniforms: passes, idx GPU address, table GPU address, out GPU address;
 * fill them with hs_qpu_gather_uniforms. passes = 0 ends the program at
 * once (an in-kernel guard; without it the loop counter would wrap).
 * Each batch owns a contiguous 64k-byte run per pass, so with 256-byte
 * aligned idx and out and k >= 4 it owns whole memory lines (the
 * performance claim needs that alignment; correctness needs 4 bytes).
 * Measured: HARDWARE_MAP_2026-11-15.md and QPU_ENCODER.md (8 batches, k=8:
 * 609 M/s on sequential indices; random DRAM access is capped near 43 M/s
 * system-wide). Returns hs_qpu_prog_finish's result. */
int hs_qpu_kernel_gather(hs_qpu_prog* p, int batches, int k);

/* Fills u[0..3] for hs_qpu_kernel_gather. Returns 0, or HS_QPU_E_OPERAND
 * when elements is 0 or not a multiple of 16 * batches * k (the kernel
 * would skip the remainder), when batches or k is unsupported, or when an
 * address is not 4-byte aligned. */
int hs_qpu_gather_uniforms(uint32_t u[4], uint32_t elements, int batches, int k, uint32_t idx_gpu,
                           uint32_t table_gpu, uint32_t out_gpu);

/* Writes the 32-bit value 1 to the address in uniform 0, from every lane,
 * then waits for the write: the smallest job with a visible effect. */
int hs_qpu_kernel_store_one(hs_qpu_prog* p);

#endif
