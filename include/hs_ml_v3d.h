/*
 * NeoGPU ML - V3D GPU Kernel Header
 */

#ifndef HS_ML_V3D_H
#define HS_ML_V3D_H

#include <stdint.h>
#include <stdbool.h>

/*============================================================================
 * GPU Node Lifecycle
 *============================================================================*/

/* Initialize V3D GPU context */
int v3d_init(void);

/* Shutdown V3D GPU */
void v3d_shutdown(void);

/* Check if GPU is available */
int v3d_available(void);

/*============================================================================
 * Unified ML Operations
 * 
 * These route to GPU when available, otherwise use NEON ASM.
 * Seamless 0-friction API.
 *============================================================================*/

/* Softmax: out[i] = exp(x[i] - max) / sum */
void ml_softmax(float* out, const float* in, uint32_t N);

/* Add: out[i] = a[i] + b[i] */
void ml_add(float* out, const float* a, const float* b, uint32_t N);

/* Activate: out[i] = relu2(gate[i]) * up[i] */
void ml_activate(float* out, const float* gate, const float* up, uint32_t N);

/* RoPE: rotary position embedding */
void ml_rope(float* x, uint32_t num_heads, uint32_t head_dim, 
             uint32_t position, float theta);

/*============================================================================
 * Direct NEON paths (bypass GPU routing)
 *============================================================================*/

void neon_softmax(float* out, const float* in, uint32_t N);
void neon_add(float* out, const float* a, const float* b, uint32_t N);
void neon_activate(float* out, const float* gate, const float* up, uint32_t N);
void neon_rope(float* x, uint32_t num_heads, uint32_t head_dim, 
               uint32_t position, float theta);

#endif /* HS_ML_V3D_H */
