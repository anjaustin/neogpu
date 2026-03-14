/*
 * NeoGPU ML - Multi-threaded Ternary GEMM v2
 * Uses persistent thread pool to avoid creation overhead
 */

#include "hs_ml.h"
#include <arm_neon.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>

#define MAX_THREADS 4

/* Thread pool state */
static pthread_t g_threads[MAX_THREADS];
static volatile int g_thread_run = 0;
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cond_work = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_cond_done = PTHREAD_COND_INITIALIZER;
static atomic_int g_work_count;
static atomic_int g_done_count;
static int g_pool_initialized = 0;

typedef struct {
    int32_t* C;
    const int8_t* A;
    const u8* B_ternary;
    u32 M, N, K;
    u32 n_start, n_end;
} work_item_t;

static work_item_t g_work[MAX_THREADS];
static int g_num_workers = 0;

/* Core kernel - single threaded portion */
static void gemm_kernel(
    int32_t* C, const int8_t* A, const u8* B_ternary,
    u32 M, u32 N, u32 K, u32 n_start, u32 n_end)
{
    static const int8_t nibble_w0[16] __attribute__((aligned(16))) = {
        0, 1, -1, 0,  0, 1, -1, 0,  0, 1, -1, 0,  0, 1, -1, 0
    };
    static const int8_t nibble_w1[16] __attribute__((aligned(16))) = {
        0, 0, 0, 0,  1, 1, 1, 1,  -1, -1, -1, -1,  0, 0, 0, 0
    };
    int8x16_t lut_w0 = vld1q_s8(nibble_w0);
    int8x16_t lut_w1 = vld1q_s8(nibble_w1);
    
    const u32 K64 = K & ~63u;
    const u32 Kstride = K / 4;
    
    for (u32 m = 0; m < M; m++) {
        const int8_t* A_row = A + m * K;
        int32_t* C_row = C + m * N;
        
        u32 n = n_start;
        for (; n + 4 <= n_end; n += 4) {
            const u8* B0 = B_ternary + (n + 0) * Kstride;
            const u8* B1 = B_ternary + (n + 1) * Kstride;
            const u8* B2 = B_ternary + (n + 2) * Kstride;
            const u8* B3 = B_ternary + (n + 3) * Kstride;
            
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);
            
            for (u32 k = 0; k < K64; k += 64) {
                __builtin_prefetch(A_row + k + 128, 0, 3);
                __builtin_prefetch(B0 + (k + 128) / 4, 0, 3);
                int8x16_t a0 = vld1q_s8(A_row + k + 0);
                int8x16_t a1 = vld1q_s8(A_row + k + 16);
                int8x16_t a2 = vld1q_s8(A_row + k + 32);
                int8x16_t a3 = vld1q_s8(A_row + k + 48);
                
                int8x16x2_t d0 = vuzpq_s8(a0, a1);
                int8x16x2_t d1 = vuzpq_s8(a2, a3);
                int8x16x2_t g01 = vuzpq_s8(d0.val[0], d1.val[0]);
                int8x16x2_t g23 = vuzpq_s8(d0.val[1], d1.val[1]);
                
                int8x16_t ag0 = g01.val[0], ag1 = g23.val[0];
                int8x16_t ag2 = g01.val[1], ag3 = g23.val[1];
                
                u32 ko = k / 4;
                
                #define PROC_COL(Bptr, acc) do {                     uint8x16_t wb = vld1q_u8(Bptr + ko);                     uint8x16_t lo = vandq_u8(wb, vdupq_n_u8(0x0F));                     uint8x16_t hi = vshrq_n_u8(wb, 4);                     int8x16_t w0 = vqtbl1q_s8(lut_w0, lo);                     int8x16_t w1 = vqtbl1q_s8(lut_w1, lo);                     int8x16_t w2 = vqtbl1q_s8(lut_w0, hi);                     int8x16_t w3 = vqtbl1q_s8(lut_w1, hi);                     int16x8_t prod = vmull_s8(vget_low_s8(w0), vget_low_s8(ag0));                     prod = vmlal_s8(prod, vget_high_s8(w0), vget_high_s8(ag0));                     prod = vmlal_s8(prod, vget_low_s8(w1), vget_low_s8(ag1));                     prod = vmlal_s8(prod, vget_high_s8(w1), vget_high_s8(ag1));                     prod = vmlal_s8(prod, vget_low_s8(w2), vget_low_s8(ag2));                     prod = vmlal_s8(prod, vget_high_s8(w2), vget_high_s8(ag2));                     prod = vmlal_s8(prod, vget_low_s8(w3), vget_low_s8(ag3));                     prod = vmlal_s8(prod, vget_high_s8(w3), vget_high_s8(ag3));                     acc = vpadalq_s16(acc, prod);                 } while(0)
                
                PROC_COL(B0, acc0);
                PROC_COL(B1, acc1);
                PROC_COL(B2, acc2);
                PROC_COL(B3, acc3);
                #undef PROC_COL
            }
            
            C_row[n+0] = vaddvq_s32(acc0);
            C_row[n+1] = vaddvq_s32(acc1);
            C_row[n+2] = vaddvq_s32(acc2);
            C_row[n+3] = vaddvq_s32(acc3);
            
            for (u32 col = 0; col < 4; col++) {
                const u8* B_col = B_ternary + (n + col) * Kstride;
                for (u32 k = K64; k < K; k++) {
                    u8 byte = B_col[k / 4];
                    u8 bits = (byte >> ((k % 4) * 2)) & 0x03;
                    int8_t a = A_row[k];
                    if (bits == 1) C_row[n + col] += a;
                    else if (bits == 2) C_row[n + col] -= a;
                }
            }
        }
        
        for (; n < n_end; n++) {
            const u8* B_col = B_ternary + n * Kstride;
            int32_t sum = 0;
            for (u32 k = 0; k < K; k++) {
                u8 byte = B_col[k / 4];
                u8 bits = (byte >> ((k % 4) * 2)) & 0x03;
                int8_t a = A_row[k];
                if (bits == 1) sum += a;
                else if (bits == 2) sum -= a;
            }
            C_row[n] = sum;
        }
    }
}

/* Simple parallel - spawn threads each call (lower overhead than pool for our case) */
typedef struct {
    int32_t* C;
    const int8_t* A;
    const u8* B_ternary;
    u32 M, N, K;
    u32 n_start, n_end;
} targ_t;

static void* thread_fn(void* arg) {
    targ_t* t = (targ_t*)arg;
    gemm_kernel(t->C, t->A, t->B_ternary, t->M, t->N, t->K, t->n_start, t->n_end);
    return NULL;
}

void hs_ml_gemm_ternary_mt(int32_t* C,
                           const int8_t* A,
                           const u8* B_ternary,
                           u32 M, u32 N, u32 K,
                           int num_threads) {
    
    memset(C, 0, M * N * sizeof(int32_t));
    
    if (num_threads < 1) num_threads = 1;
    if (num_threads > MAX_THREADS) num_threads = MAX_THREADS;
    
    /* For small problems, single thread is faster */
    if (N < 256 || M * N * K < 1000000) {
        num_threads = 1;
    }
    
    if (num_threads == 1) {
        gemm_kernel(C, A, B_ternary, M, N, K, 0, N);
        return;
    }
    
    pthread_t threads[MAX_THREADS];
    targ_t args[MAX_THREADS];
    
    /* Divide N among threads, aligned to 4 */
    u32 base = (N / num_threads) & ~3u;
    if (base < 4) base = 4;
    
    u32 n_pos = 0;
    for (int t = 0; t < num_threads; t++) {
        args[t].C = C;
        args[t].A = A;
        args[t].B_ternary = B_ternary;
        args[t].M = M;
        args[t].N = N;
        args[t].K = K;
        args[t].n_start = n_pos;
        args[t].n_end = (t == num_threads - 1) ? N : n_pos + base;
        n_pos = args[t].n_end;
        
        if (t < num_threads - 1) {
            pthread_create(&threads[t], NULL, thread_fn, &args[t]);
        }
    }
    
    /* Main thread does last chunk */
    gemm_kernel(C, A, B_ternary, M, N, K, args[num_threads-1].n_start, N);
    
    /* Wait for others */
    for (int t = 0; t < num_threads - 1; t++) {
        pthread_join(threads[t], NULL);
    }
}

/*
 * OpenMP version (simpler, if available)
 */
#ifdef _OPENMP
#include <omp.h>

void hs_ml_gemm_ternary_omp(int32_t* C,
                            const int8_t* A,
                            const u8* B_ternary,
                            u32 M, u32 N, u32 K,
                            int num_threads) {
    
    memset(C, 0, M * N * sizeof(int32_t));
    
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        
        u32 chunk = (N + nth - 1) / nth;
        chunk = (chunk + 3) & ~3u;
        
        u32 n_start = tid * chunk;
        u32 n_end = n_start + chunk;
        if (n_end > N) n_end = N;
        if (n_start < N) {
            gemm_kernel(C, A, B_ternary, M, N, K, n_start, n_end);
        }
    }
}
#endif


/*
 * Auto-select optimal thread count based on matrix dimensions
 */
int hs_ml_gemm_ternary_optimal_threads(u32 M, u32 N, u32 K) {
    size_t ops = (size_t)M * N * K;
    
    /* Very small matrices - threading overhead hurts */
    if (N < 256 || ops < 1000000) {
        return 1;
    }
    
    /* Medium matrices - 2 threads */
    if (ops < 10000000) {
        return 2;
    }
    
    /* Large matrices - 3 threads optimal on Pi4 */
    /* 4 threads hits memory bandwidth wall */
    return 3;
}
