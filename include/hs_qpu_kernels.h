/* hs_qpu_kernels: QPU programs NeoGPU ships, built with hs_qpu. Each one is
 * word-identical to its reference-assembler twin (make qpu-encode-check) and
 * runs on the GPU in tests/test_v3d_compute.c. */
#ifndef HS_QPU_KERNELS_H
#define HS_QPU_KERNELS_H

#include "hs_qpu.h"

/* out[e] = table[idx[e]] for 4-byte elements; dispatch with `batches`
 * (1 or 8) and k (1, 2, 4 or 8) loads in flight per QPU.
 * Uniforms: passes, idx GPU address, table GPU address, out GPU address.
 * passes = elements / (16 * batches * k), so elements must be a multiple
 * of 16 * batches * k. Measured: HARDWARE_MAP_2026-11-15.md (8 batches, k=8:
 * 581 M/s on sequential indices; random DRAM access is capped near
 * 43 M/s system-wide). Returns hs_qpu_prog_finish's result. */
int hs_qpu_kernel_gather(hs_qpu_prog* p, int batches, int k);

/* Writes the 32-bit value 1 to the address in uniform 0, from every lane,
 * then waits for the write: the smallest job with a visible effect. */
int hs_qpu_kernel_store_one(hs_qpu_prog* p);

#endif
