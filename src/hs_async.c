/*
 * NeoGPU - Async Operations Implementation
 * 
 * Background thread for non-blocking I/O operations.
 */

#include "hs_async.h"
#include "hs_nodes.h"
#include "hs_msg.h"
#include "hs_storage.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

#ifdef HS_DEBUG
#define DBG_PRINT(...) printf(__VA_ARGS__)
#else
#define DBG_PRINT(...) ((void)0)
#endif

static void* async_worker(void* arg) {
    HSAsync* async = (HSAsync*)arg;
    
    while (atomic_load_explicit(&async->running, memory_order_acquire)) {
        sem_wait(&async->sem);
        
        if (!atomic_load_explicit(&async->running, memory_order_acquire)) break;
        
        pthread_mutex_lock(&async->mutex);
        
        if (async->pending_count == 0) {
            pthread_mutex_unlock(&async->mutex);
            continue;
        }
        
        AsyncTask task = async->tasks[async->tail];
        async->tail = (async->tail + 1) % HS_ASYNC_MAX_PENDING;
        async->pending_count--;
        
        pthread_mutex_unlock(&async->mutex);
        
        task.done = 1;
        
        switch (task.type) {
            case ASYNC_LOAD_TEXTURE: {
                if (task.result && async->gfx) {
                    hs_graphics_create_texture(async->gfx, task.slot, 
                        ((HSGLTexture*)task.result)->width,
                        ((HSGLTexture*)task.result)->height,
                        NULL);
                }
                task.success = (task.result != NULL);
                break;
            }
            
            case ASYNC_SAVE_FILE: {
                if (task.result && task.user_data && task.size > 0) {
                    FILE* f = fopen((const char*)task.user_data, "wb");
                    if (f) {
                        size_t wrote = fwrite(task.result, 1, task.size, f);
                        fclose(f);
                        task.success = (wrote == task.size) ? 1 : 0;
                    }
                }
                break;
            }
            
            default:
                break;
        }
        
        pthread_mutex_lock(&async->mutex);
        
        for (int i = 0; i < HS_ASYNC_MAX_PENDING; i++) {
            if (async->tasks[i].task_id == task.task_id && !async->tasks[i].done) {
                async->tasks[i].done = task.done;
                async->tasks[i].success = task.success;
                async->tasks[i].result = task.result;
                async->tasks[i].size = task.size;
                break;
            }
        }
        
        pthread_mutex_unlock(&async->mutex);
    }
    
    return NULL;
}

void hs_async_init(HSAsync* async, HSGraphics* gfx) {
    memset(async, 0, sizeof(HSAsync));
    async->gfx = gfx;
    atomic_init(&async->running, false);
    async->notify_to = NODE_SYSTEM;
    
    sem_init(&async->sem, 0, 0);
    pthread_mutex_init(&async->mutex, NULL);
}

void hs_async_attach_system(HSAsync* async, HSSystem* sys, u8 notify_to) {
    if (!async) return;
    async->sys = sys;
    async->notify_to = notify_to;
}

void hs_async_shutdown(HSAsync* async) {
    if (!atomic_load_explicit(&async->running, memory_order_acquire)) return;
    
    atomic_store_explicit(&async->running, false, memory_order_release);
    sem_post(&async->sem);
    
    pthread_join(async->thread, NULL);
    
    sem_destroy(&async->sem);
    pthread_mutex_destroy(&async->mutex);
}

bool hs_async_load_texture(HSAsync* async, u8 slot, const char* path) {
    pthread_mutex_lock(&async->mutex);
    
    if (async->pending_count >= HS_ASYNC_MAX_PENDING) {
        pthread_mutex_unlock(&async->mutex);
        return false;
    }
    
    AsyncTask task = {
        .type = ASYNC_LOAD_TEXTURE,
        .task_id = ++async->next_task_id,
        .slot = slot,
        .done = 0,
        .success = 0,
        .size = 0,
        .result = NULL,
        .user_data = strdup(path)
    };
    
    async->tasks[async->head] = task;
    async->head = (async->head + 1) % HS_ASYNC_MAX_PENDING;
    async->pending_count++;
    
    pthread_mutex_unlock(&async->mutex);
    
    if (!atomic_load_explicit(&async->running, memory_order_acquire)) {
        atomic_store_explicit(&async->running, true, memory_order_release);
        pthread_create(&async->thread, NULL, async_worker, async);
    }
    
    sem_post(&async->sem);
    
    return true;
}

bool hs_async_save_file(HSAsync* async, const char* path, const void* data, u32 size) {
    pthread_mutex_lock(&async->mutex);
    
    if (async->pending_count >= HS_ASYNC_MAX_PENDING) {
        pthread_mutex_unlock(&async->mutex);
        return false;
    }
    
    void* copy = malloc(size);
    memcpy(copy, data, size);
    
    AsyncTask task = {
        .type = ASYNC_SAVE_FILE,
        .task_id = ++async->next_task_id,
        .slot = 0,
        .done = 0,
        .success = 0,
        .size = size,
        .result = copy,
        .user_data = strdup(path)
    };
    
    async->tasks[async->head] = task;
    async->head = (async->head + 1) % HS_ASYNC_MAX_PENDING;
    async->pending_count++;
    
    pthread_mutex_unlock(&async->mutex);
    
    if (!atomic_load_explicit(&async->running, memory_order_acquire)) {
        atomic_store_explicit(&async->running, true, memory_order_release);
        pthread_create(&async->thread, NULL, async_worker, async);
    }
    
    sem_post(&async->sem);
    
    return true;
}

bool hs_async_process(HSAsync* async) {
    bool has_done = false;
    
    pthread_mutex_lock(&async->mutex);
    
    for (int i = 0; i < HS_ASYNC_MAX_PENDING; i++) {
        if (async->tasks[i].done) {
            has_done = true;

            if (async->sys) {
                u8 payload[8];
                hs_pack_async_done(payload, async->tasks[i].task_id, (u8)async->tasks[i].type, async->tasks[i].slot, async->tasks[i].success);
                Message m = {
                    .to = async->notify_to,
                    .from = NODE_CPU,
                    .op = OP_ASYNC_DONE,
                    .flags = 0,
                    .cid = 0,
                    .tick = 0,
                    .payload_idx = 0,
                    .payload_len = 0,
                };
                (void)hs_send_with_payload(async->sys, &m, payload, sizeof(payload));
            }
            
            if (async->tasks[i].type == ASYNC_LOAD_TEXTURE) {
                DBG_PRINT("[Async] Texture %d loaded: %s\n", 
                    async->tasks[i].slot,
                    async->tasks[i].success ? "OK" : "FAILED");
            }
            
            if (async->tasks[i].user_data) free(async->tasks[i].user_data);
            if (async->tasks[i].result) free(async->tasks[i].result);
            
            memset(&async->tasks[i], 0, sizeof(AsyncTask));
        }
    }
    
    pthread_mutex_unlock(&async->mutex);
    
    return has_done;
}
