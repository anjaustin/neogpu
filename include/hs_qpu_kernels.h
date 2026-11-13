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

/* The default gather configuration: the measured best against Mesa's GLES
 * 3.1 compute and the CPU (docs/benchmarks/VS_MESA.md, 2026-11-16, median of
 * 7, every word checked). 8 batches with k = 8 was fastest on sequential
 * indices (612.8 M/s, 1.78x Mesa's best) and on random indices into a
 * 512 KiB table (74.6 M/s). On random indices into 64 MiB it was within
 * 0.8% of the best (36.0 vs 36.3 M/s at k = 1; DRAM-bound). Use these unless
 * a measurement of your workload says otherwise. */
#define HS_QPU_GATHER_BATCHES 8
#define HS_QPU_GATHER_K 8

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
