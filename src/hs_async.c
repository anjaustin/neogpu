#include "hs_async.h"
#include "hs_nodes.h"
#include "hs_msg.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HS_ASYNC_MAX_PATH 1024

static void release_task(AsyncTask* task) {
    free(task->user_data);
    free(task->result);
    memset(task, 0, sizeof(*task));
}

static void* async_worker(void* arg) {
    HSAsync* async = arg;
    for (;;) {
        int rc;
        do { rc = sem_wait(&async->sem); } while (rc != 0 && errno == EINTR);
        if (rc != 0) return NULL;
        pthread_mutex_lock(&async->mutex);
        if (!async->pending_count) {
            bool closing = async->closing;
            pthread_mutex_unlock(&async->mutex);
            if (closing) break;
            continue;
        }
        u8 index = async->queued[async->tail];
        async->tail = (async->tail + 1) % HS_ASYNC_MAX_PENDING;
        async->pending_count--;
        AsyncTask task = async->tasks[index];
        pthread_mutex_unlock(&async->mutex);

        FILE* file = fopen(task.user_data, "wb");
        bool success = false;
        if (file) {
            success = fwrite(task.result, 1, task.size, file) == task.size;
            if (fclose(file) != 0) success = false;
        }
        pthread_mutex_lock(&async->mutex);
        /* Slot remains owned until the consumer reclaims the completion. */
        async->tasks[index].success = success;
        async->tasks[index].done = 1;
        pthread_mutex_unlock(&async->mutex);
    }
    atomic_store_explicit(&async->running, false, memory_order_release);
    return NULL;
}

void hs_async_init(HSAsync* async, HSGraphics* gfx) {
    if (!async) return;
    memset(async, 0, sizeof(*async));
    atomic_init(&async->running, false);
    async->gfx = gfx;
    async->notify_to = NODE_SYSTEM;
    if (pthread_mutex_init(&async->mutex, NULL) != 0) return;
    if (sem_init(&async->sem, 0, 0) != 0) {
        pthread_mutex_destroy(&async->mutex);
        return;
    }
    async->initialized = true;
}

void hs_async_attach_system(HSAsync* async, HSSystem* sys, u8 notify_to) {
    if (!async || !async->initialized) return;
    pthread_mutex_lock(&async->mutex);
    async->sys = sys;
    async->notify_to = notify_to;
    pthread_mutex_unlock(&async->mutex);
}

void hs_async_shutdown(HSAsync* async) {
    if (!async || !async->initialized) return;
    pthread_mutex_lock(&async->mutex);
    async->closing = true;
    bool started = async->worker_started;
    pthread_mutex_unlock(&async->mutex);
    if (started) {
        sem_post(&async->sem);
        pthread_join(async->thread, NULL);
    }
    for (unsigned i = 0; i < HS_ASYNC_MAX_PENDING; i++) release_task(&async->tasks[i]);
    sem_destroy(&async->sem);
    pthread_mutex_destroy(&async->mutex);
    async->initialized = false;
    async->worker_started = false;
    atomic_store_explicit(&async->running, false, memory_order_release);
}

bool hs_async_load_texture(HSAsync* async, u8 slot, const char* path) {
    (void)async; (void)slot; (void)path;
    errno = ENOTSUP;
    return false;
}

bool hs_async_save_file(HSAsync* async, const char* path, const void* data, u32 size) {
    if (!async || !async->initialized || !path || !data || !size) return false;
    size_t length = strnlen(path, HS_ASYNC_MAX_PATH);
    if (!length || length >= HS_ASYNC_MAX_PATH) return false;
    pthread_mutex_lock(&async->mutex);
    int slot = -1;
    if (!async->closing) {
        for (int i = 0; i < HS_ASYNC_MAX_PENDING; i++) {
            if (async->tasks[i].type == ASYNC_NONE) { slot = i; break; }
        }
    }
    if (slot < 0) { pthread_mutex_unlock(&async->mutex); return false; }
    char* path_copy = malloc(length + 1);
    void* copy = malloc(size);
    if (!path_copy || !copy) {
        free(path_copy); free(copy);
        pthread_mutex_unlock(&async->mutex);
        return false;
    }
    memcpy(path_copy, path, length + 1);
    memcpy(copy, data, size);
    if (!async->worker_started) {
        atomic_store_explicit(&async->running, true, memory_order_release);
        int rc = pthread_create(&async->thread, NULL, async_worker, async);
        if (rc != 0) {
            atomic_store_explicit(&async->running, false, memory_order_release);
            free(path_copy); free(copy);
            pthread_mutex_unlock(&async->mutex);
            errno = rc;
            return false;
        }
        async->worker_started = true;
    }
    async->tasks[slot] = (AsyncTask){.type = ASYNC_SAVE_FILE,
        .task_id = ++async->next_task_id, .size = size, .result = copy, .user_data = path_copy};
    async->queued[async->head] = (u8)slot;
    async->head = (async->head + 1) % HS_ASYNC_MAX_PENDING;
    async->pending_count++;
    sem_post(&async->sem);
    pthread_mutex_unlock(&async->mutex);
    return true;
}

bool hs_async_process(HSAsync* async) {
    if (!async || !async->initialized) return false;
    bool completed = false;
    pthread_mutex_lock(&async->mutex);
    for (int i = 0; i < HS_ASYNC_MAX_PENDING; i++) {
        AsyncTask* task = &async->tasks[i];
        if (!task->done) continue;
        if (async->sys) {
            u8 payload[8];
            hs_pack_async_done(payload, task->task_id, (u8)task->type, task->slot, task->success);
            Message msg = {.to = async->notify_to, .from = NODE_CPU, .op = OP_ASYNC_DONE};
            /* Retain ownership and retry if notification admission is full. */
            if (!hs_try_send(async->sys, &msg, payload, sizeof(payload))) continue;
        }
        completed = true;
        release_task(task);
    }
    pthread_mutex_unlock(&async->mutex);
    return completed;
}
