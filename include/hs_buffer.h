#ifndef HS_BUFFER_H
#define HS_BUFFER_H

#include "hs_core.h"
#include "hs_math_neon.h"
#include <stdbool.h>
#include <stdlib.h>

/*
 * Buffer - mirrors PicoApi Buffer class.
 *
 * A contiguous block of memory that can hold vertex data, index
 * data, or arbitrary typed values.  All accessors use a float-
 * index (i.e. index into 32-bit slots, same as the Haxe version).
 *
 * Memory layout: raw bytes accessed as either f32[] or s32[]
 * via a union.  The capacity is in bytes; the "length" is also
 * in bytes for consistency with the Haxe side.
 */

#define HS_BUFFER_MAX_SIZE  (640 * 480 * 4)  /* matches PicoApi MAX_SIZE * 4 bytes */

typedef enum {
    HS_BUF_TYPE_I32,
    HS_BUF_TYPE_F32,
    HS_BUF_TYPE_TEXTURE,
    HS_BUF_TYPE_UNKNOWN
} HSBufferType;

typedef struct {
    u8*          data;
    u32          capacity;    /* bytes allocated */
    u32          length;      /* bytes used */
    HSBufferType type;
    u8           index;       /* memory bank index (0-15) */
    bool         dirty;       /* needs re-upload */
    bool         disposed;    /* buffer has been disposed */
    u32          tex_width;   /* texture width if type == TEXTURE */
    u32          tex_height;  /* texture height if type == TEXTURE */
} HSBuffer;

static inline bool hs_buffer_init(HSBuffer* buf, u8 index, u32 size_bytes) {
    if (size_bytes > HS_BUFFER_MAX_SIZE) size_bytes = HS_BUFFER_MAX_SIZE;
    buf->data = (u8*)calloc(1, size_bytes);
    if (!buf->data) return false;
    buf->capacity = size_bytes;
    buf->length = size_bytes;
    buf->type = HS_BUF_TYPE_UNKNOWN;
    buf->index = index;
    buf->dirty = false;
    buf->disposed = false;
    buf->tex_width = 0;
    buf->tex_height = 0;
    return true;
}

static inline void hs_buffer_free(HSBuffer* buf) {
    if (buf->data) {
        free(buf->data);
        buf->data = NULL;
    }
    buf->capacity = 0;
    buf->length = 0;
}

/* Read 32-bit int at float-index (index into 4-byte slots) */
static inline s32 hs_buffer_get_i32(const HSBuffer* buf, u32 index) {
    u32 offset = index * 4;
    if (offset + 4 > buf->length) return 0;
    s32 val;
    memcpy(&val, buf->data + offset, 4);
    return val;
}

/* Write 32-bit int at float-index */
static inline void hs_buffer_set_i32(HSBuffer* buf, u32 index, s32 val) {
    u32 offset = index * 4;
    if (offset + 4 > buf->length) return;
    memcpy(buf->data + offset, &val, 4);
    buf->dirty = true;
}

/* Read 32-bit float at float-index */
static inline f32 hs_buffer_get_f32(const HSBuffer* buf, u32 index) {
    if (!buf || !buf->data) return 0.0f;
    u32 offset = index * 4;
    if (offset + 4 > buf->length) return 0.0f;
    f32 val;
    memcpy(&val, buf->data + offset, 4);
    return val;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

/* Write 32-bit float at float-index */
static inline void hs_buffer_set_f32(HSBuffer* buf, u32 index, f32 val) {
    if (!buf || !buf->data) return;
    u32 offset = index * 4;
    if (offset + 4 > buf->length) return;
    memcpy(buf->data + offset, &val, 4);
    buf->dirty = true;
}

/* Write vec4 (4 floats) starting at float-index */
static inline void hs_buffer_set_vec4(HSBuffer* buf, u32 index, vec4 v) {
    if (!buf || !buf->data) return;
    u32 offset = index * 4;
    if (offset + 16 > buf->length) return;
    f32 arr[4];
    arr[0] = vgetq_lane_f32(v, 0);
    arr[1] = vgetq_lane_f32(v, 1);
    arr[2] = vgetq_lane_f32(v, 2);
    arr[3] = vgetq_lane_f32(v, 3);
    memcpy(buf->data + offset, arr, 16);
    buf->dirty = true;
}

/* Read vec4 (4 floats) starting at float-index */
static inline vec4 hs_buffer_get_vec4(const HSBuffer* buf, u32 index) {
    u32 offset = index * 4;
    if (offset + 16 > buf->length) return v4_zero();
    f32 arr[4];
    memcpy(arr, buf->data + offset, 16);
    return v4_make(arr[0], arr[1], arr[2], arr[3]);
}

/* Write mat4 (16 floats) starting at float-index -- column-major */
static inline void hs_buffer_set_mat4(HSBuffer* buf, u32 index, mat4 m) {
    u32 offset = index * 4;
    if (offset + 64 > buf->length) return;
    f32 arr[16];
    m4_to_array(m, arr);
    memcpy(buf->data + offset, arr, 64);
    buf->dirty = true;
}

/* Write mat3x4 (12 floats) starting at float-index -- first 3 columns */
static inline void hs_buffer_set_mat3x4(HSBuffer* buf, u32 index, mat4 m) {
    u32 offset = index * 4;
    if (offset + 48 > buf->length) return;
    f32 arr[16];
    m4_to_array(m, arr);
    /* Write columns 0-2 (12 floats), skip column 3 */
    memcpy(buf->data + offset,      &arr[0],  16);  /* col 0 */
    memcpy(buf->data + offset + 16, &arr[4],  16);  /* col 1 */
    memcpy(buf->data + offset + 32, &arr[8],  16);  /* col 2 */
    buf->dirty = true;
}

/*
 * Texture descriptor - mirrors PicoApi Texture class.
 *
 * This is metadata only; actual pixel data lives in a Buffer
 * or is managed by the GPU backend.
 */
typedef enum {
    HS_TEX_RGBA8,
    HS_TEX_R32F,
    HS_TEX_DEPTH24_STENCIL8
} HSTexFormat;

typedef struct {
    u32         width;
    u32         height;
    HSTexFormat format;
    bool        filter_linear;   /* false = nearest, true = linear */
    bool        wrap_repeat;     /* false = clamp, true = repeat */
    u32         handle;          /* GPU-side handle (opaque) */
    HSBuffer*   pixels;          /* optional CPU-side pixel data */
} HSTexture;

static inline void hs_texture_init(HSTexture* tex, u32 w, u32 h, HSTexFormat fmt) {
    tex->width = w;
    tex->height = h;
    tex->format = fmt;
    tex->filter_linear = false;  /* default: nearest (matches PicoApi) */
    tex->wrap_repeat = true;     /* default: repeat (matches PicoApi) */
    tex->handle = 0;
    tex->pixels = NULL;
}

static inline void hs_texture_filter(HSTexture* tex, bool linear) {
    tex->filter_linear = linear;
}

static inline void hs_texture_wrap(HSTexture* tex, bool repeat) {
    tex->wrap_repeat = repeat;
}

static inline bool hs_texture_is_disposed(const HSTexture* tex) {
    return tex->handle == 0 && tex->pixels == NULL;
}

static inline void hs_texture_dispose(HSTexture* tex) {
    tex->handle = 0;
    if (tex->pixels) {
        hs_buffer_free(tex->pixels);
        tex->pixels = NULL;
    }
}

static inline bool hs_buffer_is_disposed(const HSBuffer* buf) {
    return buf->disposed || buf->data == NULL;
}

static inline void hs_buffer_dispose(HSBuffer* buf) {
    if (buf->data) {
        free(buf->data);
        buf->data = NULL;
    }
    buf->disposed = true;
    buf->capacity = 0;
    buf->length = 0;
}

static inline HSTexture* hs_buffer_get_texture(HSBuffer* buf, u8 slot) {
    (void)slot;
    if (hs_buffer_is_disposed(buf)) return NULL;
    
    u32 size = buf->length >> 2;
    u32 w = 1, h = 1;
    while (w * h < size) {
        if (w == h) h <<= 1;
        else w <<= 1;
    }
    
    buf->type = HS_BUF_TYPE_TEXTURE;
    buf->tex_width = w;
    buf->tex_height = h;
    buf->dirty = true;
    
    return NULL;
}

/*
 * Memory bank manager - holds up to 16 buffers, matching PicoApi.
 */
#define HS_MAX_BANKS 16

typedef struct {
    HSBuffer  banks[HS_MAX_BANKS];
    HSTexture textures[HS_MAX_BANKS];
    u8        bank_count;
    u8        texture_count;
} HSMemory;

static inline void hs_memory_init(HSMemory* mem) {
    memset(mem, 0, sizeof(HSMemory));
}

static inline HSBuffer* hs_memory_get_buffer(HSMemory* mem, u8 index) {
    if (index >= HS_MAX_BANKS) return NULL;
    return &mem->banks[index];
}

static inline HSTexture* hs_memory_get_texture(HSMemory* mem, u8 index) {
    if (index >= HS_MAX_BANKS) return NULL;
    return &mem->textures[index];
}

static inline void hs_memory_free(HSMemory* mem) {
    for (int i = 0; i < HS_MAX_BANKS; i++) {
        hs_buffer_free(&mem->banks[i]);
    }
    memset(mem, 0, sizeof(HSMemory));
}

#pragma GCC diagnostic pop

#endif
