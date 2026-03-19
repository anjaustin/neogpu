/*
 * Hybrid CPU/GPU Pipeline Header
 */

#ifndef HS_ML_HYBRID_H
#define HS_ML_HYBRID_H

#include <stdint.h>

/*
 * Start hybrid pipeline - launch GPU thread for lm_head
 * while CPU processes transformer layers
 */
void hybrid_start(float* hidden, float* logits);

/*
 * Wait for GPU to complete
 */
void hybrid_wait(void);

/*
 * Check if GPU is ready
 */
int hybrid_gpu_ready(void);

#endif /* HS_ML_HYBRID_H */
