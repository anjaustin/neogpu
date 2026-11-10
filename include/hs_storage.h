#ifndef HS_STORAGE_H
#define HS_STORAGE_H

#include "hs_core.h"
#include <stdio.h>
#include <string.h>

#define HS_STORAGE_NAME_MAX  32
#define HS_STORAGE_SIZE      256
#define HS_STORAGE_SLOTS     16

typedef struct {
    char     name[HS_STORAGE_NAME_MAX];
    u8      data[HS_STORAGE_SIZE];
    u8      dirty;
    u8      loaded;
} HSStorageSlot;

typedef struct {
    HSStorageSlot slots[HS_STORAGE_SLOTS];
    u8            initialized;
} HSStorage;

static inline void hs_storage_init(HSStorage* store) {
    memset(store, 0, sizeof(HSStorage));
    store->initialized = 1;
}

static inline HSStorageSlot* hs_storage_load(HSStorage* store, const char* name) {
    if (!store || !name || !store->initialized) return NULL;
    /* Names must fit with their terminator. Truncating would make a long name
     * never match its stored (shortened) copy, or alias another name. */
    if (!name[0] || strnlen(name, HS_STORAGE_NAME_MAX) >= HS_STORAGE_NAME_MAX) return NULL;

    for (int i = 0; i < HS_STORAGE_SLOTS; i++) {
        if (store->slots[i].loaded &&
            strcmp(store->slots[i].name, name) == 0) {
            return &store->slots[i];
        }
    }
    
    for (int i = 0; i < HS_STORAGE_SLOTS; i++) {
        if (!store->slots[i].loaded) {
            strncpy(store->slots[i].name, name, HS_STORAGE_NAME_MAX - 1);
            store->slots[i].name[HS_STORAGE_NAME_MAX - 1] = 0;
            store->slots[i].loaded = 1;
            store->slots[i].dirty = 0;
            memset(store->slots[i].data, 0, HS_STORAGE_SIZE);
            return &store->slots[i];
        }
    }
    
    return NULL;
}

static inline u8 hs_storage_get_u8(HSStorage* store, const char* name, u32 offset) {
    HSStorageSlot* slot = hs_storage_load(store, name);
    if (!slot || offset >= HS_STORAGE_SIZE) return 0;
    return slot->data[offset];
}

static inline void hs_storage_set_u8(HSStorage* store, const char* name, u32 offset, u8 value) {
    HSStorageSlot* slot = hs_storage_load(store, name);
    if (!slot || offset >= HS_STORAGE_SIZE) return;
    slot->data[offset] = value;
    slot->dirty = 1;
}

static inline f32 hs_storage_get_f32(HSStorage* store, const char* name, u32 offset) {
    HSStorageSlot* slot = hs_storage_load(store, name);
    if (!slot || offset > HS_STORAGE_SIZE - sizeof(f32)) return 0.0f;
    f32 val;
    memcpy(&val, slot->data + offset, 4);
    return val;
}

static inline void hs_storage_set_f32(HSStorage* store, const char* name, u32 offset, f32 value) {
    HSStorageSlot* slot = hs_storage_load(store, name);
    if (!slot || offset > HS_STORAGE_SIZE - sizeof(f32)) return;
    memcpy(slot->data + offset, &value, 4);
    slot->dirty = 1;
}

/* Native-layout snapshot of all slots. Returns false on any write/close error. */
static inline bool hs_storage_save(HSStorage* store, const char* filename) {
    if (!store || !filename) return false;
    FILE* f = fopen(filename, "wb");
    if (!f) return false;
    bool ok = fwrite(store->slots, sizeof(HSStorageSlot), HS_STORAGE_SLOTS, f) == HS_STORAGE_SLOTS;
    if (fclose(f) != 0) ok = false;
    return ok;
}

/* Loads a snapshot written by hs_storage_save. A short/oversized file or a
 * slot without a terminated name is rejected and leaves the store unchanged. */
static inline bool hs_storage_load_file(HSStorage* store, const char* filename) {
    if (!store || !filename) return false;
    FILE* f = fopen(filename, "rb");
    if (!f) return false;
    HSStorageSlot loaded[HS_STORAGE_SLOTS];
    bool ok = fread(loaded, sizeof(HSStorageSlot), HS_STORAGE_SLOTS, f) == HS_STORAGE_SLOTS &&
              fgetc(f) == EOF && !ferror(f);
    fclose(f);
    for (int i = 0; ok && i < HS_STORAGE_SLOTS; i++) {
        if (memchr(loaded[i].name, 0, HS_STORAGE_NAME_MAX) == NULL) ok = false;
    }
    if (!ok) return false;
    memcpy(store->slots, loaded, sizeof(loaded));
    store->initialized = 1;
    return true;
}

static inline void hs_storage_sync(HSStorage* store) {
    for (int i = 0; i < HS_STORAGE_SLOTS; i++) {
        store->slots[i].dirty = 0;
    }
}

#endif
