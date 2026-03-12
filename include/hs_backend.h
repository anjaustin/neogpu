/*
 * NeoGPU - Backend Interface
 *
 * Backends consume decoded node state + a render command list and execute
 * platform-specific work (e.g. GLES).
 */

#ifndef HS_BACKEND_H
#define HS_BACKEND_H

#include "hs_core.h"

typedef struct HSGpu HSGpu;
typedef struct HSFrameContext HSFrameContext;

typedef struct {
    bool (*init)(void* ctx, HSGpu* gpu);
    void (*shutdown)(void* ctx, HSGpu* gpu);
    void (*begin_frame)(void* ctx, const HSFrameContext* frame);
    void (*execute)(void* ctx, const HSFrameContext* frame);
    void (*end_frame)(void* ctx, const HSFrameContext* frame);
} HSBackendOps;

typedef struct {
    void* ctx;
    const HSBackendOps* ops;
} HSBackend;

#endif
