/*
 * NeoGPU - Async Operations
 * 
 * Non-blocking operations via message queue.
 * File I/O, texture decoding, shader compilation run async.
 */

#ifndef HS_ASYNC_H
#define HS_ASYNC_H

#include "hs_core.h"
#include "hs_graphics.h"
#include <pthread.h>
#include <semaphore.h>
#include <stdatomic.h>

#define HS_ASYNC_MAX_PENDING 32
#define HS_ASYNC_STACK_SIZE 8192

typedef enum {
    ASYNC_NONE = 0,
    ASYNC_LOAD_TEXTURE,
    ASYNC_LOAD_FILE,
    ASYNC_SAVE_FILE,
    ASYNC_SHADER_COMPILE
} AsyncType;

typedef struct {
    AsyncType type;
    u32      task_id;
    u8       slot;
    u8       done;
    u8       success;
    u8       pad0;
    u32      size;
    void*    result;
    void*    user_data;
} AsyncTask;

typedef struct {
    AsyncTask    tasks[HS_ASYNC_MAX_PENDING];
    u8           pending_count;
    u8           head;
    u8           tail;
    u32          next_task_id;
    sem_t        sem;
    pthread_t    thread;
    pthread_mutex_t mutex;
    atomic_bool   running;
    HSGraphics*  gfx;
    HSSystem*    sys;
    u8           notify_to;
} HSAsync;

void hs_async_init(HSAsync* async, HSGraphics* gfx);
void hs_async_shutdown(HSAsync* async);

/* If attached, hs_async_process emits OP_ASYNC_DONE messages to notify_to. */
void hs_async_attach_system(HSAsync* async, HSSystem* sys, u8 notify_to);

bool hs_async_load_texture(HSAsync* async, u8 slot, const char* path);
bool hs_async_save_file(HSAsync* async, const char* path, const void* data, u32 size);
bool hs_async_process(HSAsync* async);

typedef void (*AsyncCallback)(void* result, void* user_data);

#endif // HS_ASYNC_H
