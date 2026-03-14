/*
 * NeoGPU ML - Routing Implementation
 *
 * Implements the unified routing dispatch promised by hs_ml_routing.h.
 * This is a thin dispatcher: it validates the descriptor and calls the
 * correct kernel. All kernel logic lives in the kernel files.
 *
 * Single source of truth for:
 *   - format -> kernel mapping
 *   - thread-count heuristic
 *   - descriptor validation
 */

#include "hs_ml_routing.h"
#include <string.h>

/*============================================================================
 * External kernel declarations
 * (defined in hs_ml_binary.c and hs_ml_ternary_mt.c)
 *============================================================================*/

extern void hs_ml_route_binary_opt(s32* C, const u8* A, const u8* B,
                                   u32 M, u32 N, u32 K);

extern void hs_ml_gemm_ternary_mt(s32* C, const s8* A, const u8* B_ternary,
                                  u32 M, u32 N, u32 K, int num_threads);

extern void hs_ml_route_ternary_x_ternary(s32* C,
                                          const u8* A_active, const u8* A_sign,
                                          const u8* B_active, const u8* B_sign,
                                          u32 M, u32 N, u32 K);

/*============================================================================
 * Thread count heuristic
 *============================================================================*/

int hs_ml_route_optimal_threads(const HSRouteDesc* route, u32 M) {
    if (!route) return 1;

    /* Binary is fast enough single-threaded; threading adds overhead */
    if (route->format == HS_ROUTE_BINARY) return 1;

    /* Ternary: memory-bound, 3 threads saturates Pi4 LPDDR4 bandwidth.
     *
     * Measured crossover points (Pi4 Cortex-A72, best-of-5x20):
     *   1T beats 2T/3T below N*K ~ 4M (e.g. N=2048 K=1024 still single-thread)
     *   2T beats 1T  from  N*K ~ 4M
     *   3T beats 2T  above N*K ~ 8M
     *
     * Using M*N*K as the op count so batched (M>1) calls also scale correctly.
     */
    u64 ops = (u64)M * route->N * route->K;
    /* Two-threshold: 1T below 3M ops, 3T above.
     * 2T never cleanly beats both 1T and 3T on Pi4 — the crossover
     * jumps directly from 1T-optimal to 3T-optimal around N*K=3M.
     * Measured on Cortex-A72 @ 1800 MHz, LPDDR4. */
    if (ops < 3000000ULL) return 1;
    return 3;
}

/*============================================================================
 * hs_ml_route — single-threaded dispatch
 *============================================================================*/

void hs_ml_route(s32* output, const void* input,
                 const HSRouteDesc* route, u32 M) {
    hs_ml_route_mt(output, input, route, M, 1);
}

/*============================================================================
 * hs_ml_route_mt — multi-threaded dispatch
 *============================================================================*/

void hs_ml_route_mt(s32* output, const void* input,
                    const HSRouteDesc* route, u32 M, int num_threads) {
    if (!output || !input || !route || M == 0) return;
    if (route->N == 0 || route->K == 0)       return;
    if (!route->routes)                         return;

    /* Clamp threads */
    if (num_threads < 1) num_threads = 1;
    if (num_threads > 4) num_threads = 4;

    switch (route->format) {

        case HS_ROUTE_BINARY:
            /* Binary: XNOR + popcount. Threading not beneficial; ignore num_threads. */
            hs_ml_route_binary_opt(
                output,
                (const u8*)input,
                (const u8*)route->routes,
                M, route->N, route->K
            );
            break;

        case HS_ROUTE_TERNARY_2BIT:
            hs_ml_gemm_ternary_mt(
                output,
                (const s8*)input,
                (const u8*)route->routes,
                M, route->N, route->K,
                num_threads
            );
            break;

        case HS_ROUTE_TERNARY_INT8:
            /* Not yet implemented. Caller gets zeroed output.
             * This is an explicit contract: unimplemented formats produce
             * zero rather than garbage or a silent wrong result. */
            memset(output, 0, (size_t)M * route->N * sizeof(s32));
            break;

        case HS_ROUTE_SPARSE:
            /* Not yet implemented. Zero output, same contract. */
            memset(output, 0, (size_t)M * route->N * sizeof(s32));
            break;

        default:
            /* Unknown format: zero output. */
            memset(output, 0, (size_t)M * route->N * sizeof(s32));
            break;
    }
}
