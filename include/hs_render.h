/*
 * NeoGPU - Render Command List
 */

#ifndef HS_RENDER_H
#define HS_RENDER_H

#include "hs_core.h"

#define HS_RENDER_MAX_CMDS 2048

typedef enum {
    HS_RC_FRAME_BEGIN = 1,
    HS_RC_FRAME_END,
    HS_RC_PRESENT,

    HS_RC_CLEAR,
    HS_RC_CLEAR_DS,
    HS_RC_SET_CULL,
    HS_RC_SET_BLEND,
    HS_RC_SET_ALPHA,
    HS_RC_SET_DEPTH,
    HS_RC_SET_DEPTH_COMPARE,
    HS_RC_SET_COLOR_MASK,
    HS_RC_SET_CLIP,
    HS_RC_SET_TEX_FILTER,
    HS_RC_SET_TEX_WRAP,
    HS_RC_DRAW,
    HS_RC_DRAW_INSTANCE,
    HS_RC_DRAW_TEXT,
    HS_RC_SHOW_TEXTURE,
} HSRenderOp;

typedef struct {
    u8 op;
    u8 a;
    u8 b;
    u8 c;
    u16 payload_idx;
    u16 payload_len;
    u32 x;
    u32 y;
    f32 f0;
    f32 f1;
    f32 f2;
    f32 f3;
} HSRenderCmd;

typedef struct HSRenderList {
    HSRenderCmd cmds[HS_RENDER_MAX_CMDS];
    u32 count;
    bool overflow;
} HSRenderList;

static inline void hs_render_reset(HSRenderList* rl) {
    if (!rl) return;
    rl->count = 0;
    rl->overflow = false;
}

static inline bool hs_render_push(HSRenderList* rl, const HSRenderCmd* cmd) {
    if (!rl || !cmd) return false;
    if (rl->count >= HS_RENDER_MAX_CMDS) {
        rl->overflow = true;
        return false;
    }
    rl->cmds[rl->count++] = *cmd;
    return true;
}

#endif
