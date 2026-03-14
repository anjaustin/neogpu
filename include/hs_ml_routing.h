/*
 * NeoGPU ML - Routing Abstraction Layer
 * 
 * Neural network inference reframed as coordinate transformation through routing.
 * 
 * Core insight:
 *   - Weights are not multipliers, they are routing decisions
 *   - +1: route activation to positive accumulator
 *   - -1: route activation to negative accumulator
 *   - 0: no route (orthogonal dimensions)
 *   
 * output[n] = sum(input[k] where route[k->n] = +1)
 *           - sum(input[k] where route[k->n] = -1)
 * 
 * This is gather-accumulate, not multiply-accumulate.
 * 
 * See: docs/theory/GEOMETRIC_INFERENCE.md
 */

#ifndef HS_ML_ROUTING_H
#define HS_ML_ROUTING_H

#include "hs_core.h"

/*
 * Routing formats - different representations of the same routing decisions
 * 
 * All formats encode the same information: for each (input, output) pair,
 * what is the routing decision {+1, -1, 0} or {+1, -1} for binary.
 */
typedef enum {
    /* Binary: 1 bit per weight, {-1, +1} encoded as {0, 1}
     * Storage: K/8 bytes per output dimension
     * Operations: XNOR + popcount
     * Throughput: ~30 GOPS on Pi4 (memory-bound)
     */
    HS_ROUTE_BINARY = 0,
    
    /* Ternary 2-bit: 2 bits per weight, {-1, 0, +1} as {10, 00, 01}
     * Storage: K/4 bytes per output dimension
     * Operations: LUT decode + signed accumulate
     * Throughput: ~10 GOPS single-thread, ~25 GOPS with 3 threads
     */
    HS_ROUTE_TERNARY_2BIT = 1,
    
    /* Ternary INT8: 8 bits per weight, {-1, 0, +1} as int8_t values
     * Storage: K bytes per output dimension (4x larger than 2-bit)
     * Operations: direct signed operations
     * Use case: small models or when decode latency matters
     */
    HS_ROUTE_TERNARY_INT8 = 2,
    
    /* Sparse: only store non-zero routes
     * Storage: variable, depends on sparsity
     * Format: (index, sign) pairs
     * Use case: highly sparse routing (attention patterns)
     */
    HS_ROUTE_SPARSE = 3,
    
} HSRouteFormat;

/*
 * Route descriptor - describes the routing table for a layer
 */
typedef struct {
    HSRouteFormat format;   /* How routes are encoded */
    u32 K;                  /* Input dimension (number of input coordinates) */
    u32 N;                  /* Output dimension (number of output coordinates) */
    const void* routes;     /* Routing table data */
    
    /* Format-specific metadata */
    union {
        /* For HS_ROUTE_SPARSE */
        struct {
            const u32* row_offsets;   /* CSR-style row offsets [N+1] */
            const u32* col_indices;   /* Column indices */
            const i8* signs;          /* Signs: +1 or -1 */
            u32 nnz;                  /* Number of non-zero routes */
        } sparse;
        
        /* For ternary 2-bit with scales */
        struct {
            const float* scales;      /* Per-block scales (if needed) */
            u32 block_size;           /* Scale block size */
        } ternary;
    } meta;
    
} HSRouteDesc;

/*
 * Apply routing: transform input coordinates to output coordinates
 * 
 * output: [M, N] int32 accumulators (caller must zero-initialize)
 * input: [M, K] activations (format depends on route format)
 * route: routing table descriptor
 * M: batch size (number of vectors to route)
 * 
 * For binary routes: input is uint8_t packed bits [M, K/8]
 * For ternary routes: input is int8_t values [M, K]
 * For sparse routes: input is int8_t values [M, K]
 */
void hs_ml_route(i32* output,
                 const void* input,
                 const HSRouteDesc* route,
                 u32 M);

/*
 * Apply routing with multi-threading
 * 
 * Same as hs_ml_route but uses multiple threads.
 * num_threads: 0 or 1 = single-threaded, 2-4 = multi-threaded
 */
void hs_ml_route_mt(i32* output,
                    const void* input,
                    const HSRouteDesc* route,
                    u32 M,
                    int num_threads);

/*
 * Binary routing (specialized, maximum throughput)
 * 
 * Both activations and weights are binary {-1, +1} encoded as {0, 1} bits.
 * Uses XNOR + popcount for dot product.
 * 
 * result = 2 * popcount(A XNOR W) - K
 *        = count(matches) - count(mismatches)
 * 
 * output: [M, N] int32 results
 * input: [M, K/8] packed activation bits
 * weights: [N, K/8] packed weight bits (note: transposed)
 * 
 * K must be multiple of 128 for NEON alignment.
 */
void hs_ml_route_binary(i32* output,
                        const u8* input,
                        const u8* weights,
                        u32 M, u32 N, u32 K);

/*
 * Binary routing optimized (4 columns at once)
 */
void hs_ml_route_binary_opt(i32* output,
                            const u8* input,
                            const u8* weights,
                            u32 M, u32 N, u32 K);

/*
 * Ternary × Ternary routing (both activations and weights are ternary)
 * 
 * Uses bitplane representation:
 *   active_plane: 1 if value is non-zero
 *   sign_plane: 1 if value is negative (only meaningful where active)
 * 
 * Result computation:
 *   result_active = a_active AND w_active
 *   result_sign = a_sign XOR w_sign  
 *   output = count(result_active AND NOT result_sign)
 *          - count(result_active AND result_sign)
 * 
 * A_active, A_sign: [M, K/8] bitplanes for activations
 * B_active, B_sign: [N, K/8] bitplanes for weights
 */
void hs_ml_route_ternary_x_ternary(i32* output,
                                   const u8* A_active,
                                   const u8* A_sign,
                                   const u8* B_active,
                                   const u8* B_sign,
                                   u32 M, u32 N, u32 K);

/*
 * Get recommended thread count for routing operation
 */
int hs_ml_route_optimal_threads(const HSRouteDesc* route, u32 M);

/*
 * Memory requirements for different routing formats
 */
static inline size_t hs_ml_route_weight_size(HSRouteFormat fmt, u32 N, u32 K) {
    switch (fmt) {
        case HS_ROUTE_BINARY:       return (size_t)N * (K / 8);
        case HS_ROUTE_TERNARY_2BIT: return (size_t)N * (K / 4);
        case HS_ROUTE_TERNARY_INT8: return (size_t)N * K;
        case HS_ROUTE_SPARSE:       return 0; /* variable, depends on sparsity */
    }
    return 0;
}

/*
 * Information content per weight (bits)
 */
static inline float hs_ml_route_bits_per_weight(HSRouteFormat fmt) {
    switch (fmt) {
        case HS_ROUTE_BINARY:       return 1.0f;
        case HS_ROUTE_TERNARY_2BIT: return 1.58f; /* log2(3) */
        case HS_ROUTE_TERNARY_INT8: return 1.58f; /* same info, more storage */
        case HS_ROUTE_SPARSE:       return 0.0f;  /* depends on sparsity */
    }
    return 0.0f;
}

#endif /* HS_ML_ROUTING_H */
