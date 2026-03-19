/*
 * NeoGPU ML - Message-Passing Inference Implementation
 * 
 * Implements the message routing layer for ML inference.
 * Single consumer thread processes messages in priority order.
 */

#include "hs_ml_msg.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/*============================================================================
 * Kernel dispatch via routing abstraction (hs_ml_routing.h / hs_ml_routing.c)
 *============================================================================*/

static inline u64 get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (u64)ts.tv_sec * 1000000000ULL + (u64)ts.tv_nsec;
}

/*============================================================================
 * System Lifecycle
 *============================================================================*/

void ml_sys_init(MLSystem* sys) {
    memset(sys, 0, sizeof(*sys));
    
    /* Initialize queues */
    for (int ch = 0; ch < ML_CHAN_COUNT; ch++) {
        for (int p = 0; p < ML_MAX_PRODUCERS; p++) {
            ml_spsc_init(&sys->producers[ch][p]);
        }
    }
    
    /* Default channel policies */
    sys->channels[ML_CHAN_PREFILL].budget = 4096;   /* High throughput for prefill */
    sys->channels[ML_CHAN_PREFILL].blocking = true;
    
    sys->channels[ML_CHAN_DECODE].budget = 256;     /* Lower budget, low latency */
    sys->channels[ML_CHAN_DECODE].blocking = true;
    
    sys->channels[ML_CHAN_TELEM].budget = 64;       /* Best effort */
    sys->channels[ML_CHAN_TELEM].blocking = false;
    
    /* Initialize backpressure primitives */
    pthread_mutex_init(&sys->bp_lock, NULL);
    pthread_cond_init(&sys->bp_cv, NULL);
    
    atomic_store(&sys->running, true);
    atomic_store(&sys->tick, 0);
    atomic_store(&sys->producer_count, 0);
}

void ml_sys_free(MLSystem* sys) {
    atomic_store(&sys->running, false);
    
    /* Wake any blocked producers */
    pthread_mutex_lock(&sys->bp_lock);
    pthread_cond_broadcast(&sys->bp_cv);
    pthread_mutex_unlock(&sys->bp_lock);
    
    pthread_mutex_destroy(&sys->bp_lock);
    pthread_cond_destroy(&sys->bp_cv);
}

int ml_sys_register_producer(MLSystem* sys) {
    u32 id = atomic_fetch_add(&sys->producer_count, 1);
    if (id >= ML_MAX_PRODUCERS) {
        atomic_fetch_sub(&sys->producer_count, 1);
        return -1;
    }
    return (int)id;
}

/*============================================================================
 * Message Submission
 *============================================================================*/

bool ml_sys_submit(MLSystem* sys, int producer_id, const MLMsg* msg) {
    if (!sys || producer_id < 0 || producer_id >= ML_MAX_PRODUCERS) {
        return false;
    }
    
    MLChannel ch = (MLChannel)msg->channel;
    if (ch >= ML_CHAN_COUNT) ch = ML_CHAN_PREFILL;
    
    MLSpscQueue* q = &sys->producers[ch][producer_id];
    MLChannelPolicy* policy = &sys->channels[ch];
    
    /* Try to push */
    if (ml_spsc_push(q, msg)) {
        return true;
    }
    
    /* Queue full */
    if (!policy->blocking) {
        /* Non-blocking channel: drop the message */
        policy->dropped++;
        return false;
    }
    
    /* Blocking channel: wait for space */
    atomic_fetch_add(&sys->bp_waiters, 1);
    pthread_mutex_lock(&sys->bp_lock);
    
    while (atomic_load(&sys->running) && ml_spsc_full(q)) {
        pthread_cond_wait(&sys->bp_cv, &sys->bp_lock);
    }
    
    pthread_mutex_unlock(&sys->bp_lock);
    atomic_fetch_sub(&sys->bp_waiters, 1);
    
    /* Try again after wakeup */
    if (!atomic_load(&sys->running)) {
        return false;
    }
    
    return ml_spsc_push(q, msg);
}

bool ml_sys_submit_gemm(MLSystem* sys, int producer_id,
                        MLLayerType layer, u8 layer_idx,
                        MLChannel channel, HSRouteFormat format,
                        void* input, void* weights, void* output,
                        u32 M, u32 N, u32 K) {
    MLMsg msg = ml_msg_gemm(layer, layer_idx, format, channel,
                            input, weights, output, M, N, K);
    return ml_sys_submit(sys, producer_id, &msg);
}

/*============================================================================
 * Kernel Dispatch
 *============================================================================*/

static void dispatch_gemm(MLSystem* sys, MLMsg* msg) {
    u64 t0 = get_time_ns();

    HSRouteDesc route;
    route.format = (HSRouteFormat)msg->format;
    route.K      = msg->K;
    route.N      = msg->N;
    route.routes = msg->weights;

    int threads = hs_ml_route_optimal_threads(&route, msg->M);
    hs_ml_route_mt((s32*)msg->output, msg->input, &route, msg->M, threads);
    
    u64 elapsed = get_time_ns() - t0;
    
    /* Update stats */
    MLLayerStats* stats = &sys->layer_stats[msg->layer_type];
    stats->messages_processed++;
    stats->total_ops += (u64)msg->M * msg->N * msg->K;
    stats->total_ns += elapsed;
}

static void dispatch_norm(MLSystem* sys, MLMsg* msg) {
    (void)sys;
    /* RMSNorm implementation */
    float* input = (float*)msg->input;
    float* output = (float*)msg->output;
    u32 N = msg->N;
    
    /* Compute sum of squares */
    float ss = 0.0f;
    for (u32 i = 0; i < N; i++) {
        ss += input[i] * input[i];
    }
    
    /* Normalize */
    float scale = 1.0f / sqrtf(ss / N + 1e-5f);
    for (u32 i = 0; i < N; i++) {
        output[i] = input[i] * scale;
    }
}

static void dispatch_fence(MLSystem* sys, MLMsg* msg) {
    (void)msg;
    /* Fence is a no-op - it just ensures ordering */
    /* The message being processed means all prior messages are done */
    atomic_fetch_add(&sys->tick, 1);
}

void ml_sys_dispatch(MLSystem* sys, MLMsg* msg) {
    switch ((MLOpCode)msg->op) {
        case ML_OP_GEMM:
            dispatch_gemm(sys, msg);
            break;
        case ML_OP_NORM:
            dispatch_norm(sys, msg);
            break;
        case ML_OP_FENCE:
            dispatch_fence(sys, msg);
            break;
        case ML_OP_ROPE:
        case ML_OP_SOFTMAX:
        case ML_OP_ACTIVATE:
        case ML_OP_ADD:
        case ML_OP_COPY:
            /* TODO: Implement these */
            break;
        default:
            break;
    }
    
    /* Record for capture if enabled */
    if (sys->capturing && sys->capture_count < sys->capture_capacity) {
        sys->capture_buf[sys->capture_count++] = *msg;
    }
}

/*============================================================================
 * Processing (Consumer Side)
 *============================================================================*/

u32 ml_sys_step(MLSystem* sys) {
    u32 processed = 0;
    u32 prod_count = atomic_load(&sys->producer_count);
    if (prod_count > ML_MAX_PRODUCERS) prod_count = ML_MAX_PRODUCERS;
    
    /* Process channels in priority order: PREFILL, DECODE, TELEM */
    for (int ch = 0; ch < ML_CHAN_COUNT; ch++) {
        u32 budget = sys->channels[ch].budget;
        u32 used = 0;
        
        /* Drain all producer queues for this channel */
        for (u32 p = 0; p < prod_count && used < budget; p++) {
            MLSpscQueue* q = &sys->producers[ch][p];
            
            /* Drain up to remaining budget from this producer */
            while (used < budget) {
                MLMsg msg;
                if (!ml_spsc_pop(q, &msg)) {
                    break;  /* Queue empty */
                }
                
                /* Assign tick */
                msg.tick = atomic_load(&sys->tick);
                
                /* Dispatch to kernel */
                ml_sys_dispatch(sys, &msg);
                
                used++;
                processed++;
            }
        }
    }
    
    /* Increment tick */
    atomic_fetch_add(&sys->tick, 1);
    
    /* Wake blocked producers if any */
    if (atomic_load(&sys->bp_waiters) > 0) {
        pthread_mutex_lock(&sys->bp_lock);
        pthread_cond_broadcast(&sys->bp_cv);
        pthread_mutex_unlock(&sys->bp_lock);
    }
    
    return processed;
}

/*============================================================================
 * Capture/Replay
 *============================================================================*/

void ml_sys_capture_start(MLSystem* sys, MLMsg* buf, u32 capacity) {
    sys->capture_buf = buf;
    sys->capture_capacity = capacity;
    sys->capture_count = 0;
    sys->capturing = true;
}

u32 ml_sys_capture_stop(MLSystem* sys) {
    sys->capturing = false;
    return sys->capture_count;
}

bool ml_sys_replay(MLSystem* sys, const MLMsg* msgs, u32 count) {
    /* Reset stats */
    ml_sys_reset_stats(sys);
    atomic_store(&sys->tick, 0);
    
    /* Process each message directly (single-threaded replay) */
    for (u32 i = 0; i < count; i++) {
        MLMsg msg = msgs[i];
        msg.tick = atomic_load(&sys->tick);
        ml_sys_dispatch(sys, &msg);
        atomic_fetch_add(&sys->tick, 1);
    }
    
    return true;
}

/*============================================================================
 * Statistics
 *============================================================================*/

const MLLayerStats* ml_sys_layer_stats(MLSystem* sys, MLLayerType layer) {
    if (layer >= ML_LAYER_COUNT) return NULL;
    return &sys->layer_stats[layer];
}

void ml_sys_reset_stats(MLSystem* sys) {
    memset(sys->layer_stats, 0, sizeof(sys->layer_stats));
    for (int ch = 0; ch < ML_CHAN_COUNT; ch++) {
        sys->channels[ch].dropped = 0;
    }
}

void ml_sys_set_budget(MLSystem* sys, MLChannel ch, u32 budget) {
    if (ch < ML_CHAN_COUNT) {
        sys->channels[ch].budget = budget;
    }
}

void ml_sys_set_blocking(MLSystem* sys, MLChannel ch, bool blocking) {
    if (ch < ML_CHAN_COUNT) {
        sys->channels[ch].blocking = blocking;
    }
}
