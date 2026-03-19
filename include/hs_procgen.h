/*
 * hs_procgen.h - Procedural Content Generation
 * 
 * Inspired by David (64KB game) - procedural generation for:
 * - Textures (layer-based)
 * - Meshes (compression + mirroring)
 * - Audio (synthesis from primitives)
 */

#ifndef HS_PROCGEN_H
#define HS_PROCGEN_H

#include "hs_core.h"
#include <math.h>
#include <string.h>

#define HS_PROC_TEX_SIZE 64
#define HS_PROC_MAX_LAYERS 8

typedef enum {
    LAYER_COVERAGE_FULL = 0,
    LAYER_COVERAGE_RECT,
    LAYER_COVERAGE_ROUND,
    LAYER_COVERAGE_MASK,
} LayerCoverage;

typedef enum {
    LAYER_BLEND_NORMAL = 0,
    LAYER_BLEND_ADD,
    LAYER_BLEND_MULT,
} LayerBlend;

typedef struct {
    u8 coverage;       // 0=full, 1=rect, 2=round, 3=mask
    u8 blend;          // 0=normal, 1=add, 2=mult
    u8 has_color;
    u8 has_noise;
    u8 has_bevel;
    u8 is_emissive;
    
    // Rectangle (if coverage = 1)
    f32 rect_x, rect_y, rect_w, rect_h;
    
    // Color (16-bit: 4444 RGBA)
    u16 color;
    
    // Noise
    f32 noise_freq;
    f32 noise_amp;
    
    // Bevel
    f32 bevel_depth;
    f32 bevel_softness;
} ProcLayer;

typedef struct {
    ProcLayer layers[HS_PROC_MAX_LAYERS];
    u8 num_layers;
    u8 width;
    u8 height;
} ProcTexture;

static u32 g_prng_state = 12345;

#ifndef FMIN
#define FMIN(a,b) ((a)<(b)?(a):(b))
#endif

static u32 prng_next(void) {
    g_prng_state = g_prng_state * 1103515245 + 12345;
    return (g_prng_state >> 16) & 0x7FFF;
}

static void prng_seed(u32 seed) {
    g_prng_state = seed;
}

static f32 noise2D(f32 x, f32 y, u32 seed) {
    f32 n = sinf(x * 12.9898f + y * 78.233f + seed * 43758.5453f);
    return n - floorf(n);
}

static f32 fbm(f32 x, f32 y, u32 seed, int octaves) {
    f32 val = 0.0f;
    f32 amp = 1.0f;
    f32 freq = 1.0f;
    for (int i = 0; i < octaves; i++) {
        val += amp * noise2D(x * freq, y * freq, seed + i * 100);
        amp *= 0.5f;
        freq *= 2.0f;
    }
    return val;
}

static void proc_tex_init(ProcTexture* tex, u8 w, u8 h) {
    memset(tex, 0, sizeof(*tex));
    tex->width = w;
    tex->height = h;
}

static void proc_tex_add_layer(ProcTexture* tex) {
    if (tex->num_layers < HS_PROC_MAX_LAYERS) {
        tex->num_layers++;
    }
}

static void proc_tex_layer_coverage(ProcTexture* tex, u8 layer, u8 coverage) {
    if (layer < tex->num_layers) {
        tex->layers[layer].coverage = coverage;
    }
}

static void proc_tex_layer_color(ProcTexture* tex, u8 layer, u8 r, u8 g, u8 b, u8 a) {
    if (layer < tex->num_layers) {
        tex->layers[layer].has_color = 1;
        tex->layers[layer].color = ((r >> 4) << 12) | ((g >> 4) << 8) | ((b >> 4) << 4) | (a >> 4);
    }
}

static void proc_tex_layer_noise(ProcTexture* tex, u8 layer, f32 freq, f32 amp) {
    if (layer < tex->num_layers) {
        tex->layers[layer].has_noise = 1;
        tex->layers[layer].noise_freq = freq;
        tex->layers[layer].noise_amp = amp;
    }
}

static void proc_tex_layer_bevel(ProcTexture* tex, u8 layer, f32 depth, f32 softness) {
    if (layer < tex->num_layers) {
        tex->layers[layer].has_bevel = 1;
        tex->layers[layer].bevel_depth = depth;
        tex->layers[layer].bevel_softness = softness;
    }
}

static void proc_tex_layer_rect(ProcTexture* tex, u8 layer, f32 x, f32 y, f32 w, f32 h) {
    if (layer < tex->num_layers) {
        tex->layers[layer].rect_x = x;
        tex->layers[layer].rect_y = y;
        tex->layers[layer].rect_w = w;
        tex->layers[layer].rect_h = h;
    }
}

static void proc_tex_layer_emissive(ProcTexture* tex, u8 layer, u8 emissive) {
    if (layer < tex->num_layers) {
        tex->layers[layer].is_emissive = emissive;
    }
}

static void proc_tex_generate(const ProcTexture* tex, u8* output) {
    u8 bg_r = 30, bg_g = 30, bg_b = 35;
    
    for (int y = 0; y < tex->height; y++) {
        for (int x = 0; x < tex->width; x++) {
            f32 fx = (f32)x / tex->width;
            f32 fy = (f32)y / tex->height;
            
            u8 r = bg_r, g = bg_g, b = bg_b, a = 255;
            
            for (int l = 0; l < tex->num_layers; l++) {
                const ProcLayer* layer = &tex->layers[l];
                
                int in_coverage = 0;
                if (layer->coverage == 0) {
                    in_coverage = 1;
                } else if (layer->coverage == 1) {
                    if (fx >= layer->rect_x && fx <= layer->rect_x + layer->rect_w &&
                        fy >= layer->rect_y && fy <= layer->rect_y + layer->rect_h) {
                        in_coverage = 1;
                    }
                } else if (layer->coverage == 2) {
                    f32 cx = layer->rect_x + layer->rect_w * 0.5f;
                    f32 cy = layer->rect_y + layer->rect_h * 0.5f;
                    f32 dx = (fx - cx) / (layer->rect_w * 0.5f);
                    f32 dy = (fy - cy) / (layer->rect_h * 0.5f);
                    if (dx*dx + dy*dy <= 1.0f) in_coverage = 1;
                }
                
                if (in_coverage && layer->has_color) {
                    u8 lr = ((layer->color >> 12) & 0xF) * 17;
                    u8 lg = ((layer->color >> 8) & 0xF) * 17;
                    u8 lb = ((layer->color >> 4) & 0xF) * 17;
                    
                    if (layer->has_noise) {
                        f32 n = fbm(fx * layer->noise_freq * 10.0f, fy * layer->noise_freq * 10.0f, l, 3);
                        lr = (u8)(lr * (0.7f + n * layer->noise_amp * 0.6f));
                        lg = (u8)(lg * (0.7f + n * layer->noise_amp * 0.6f));
                        lb = (u8)(lb * (0.7f + n * layer->noise_amp * 0.6f));
                    }
                    
                    if (layer->bevel_depth > 0) {
                        f32 bevel_edge = layer->bevel_depth;
                        f32 dist = 1.0f;
                        if (layer->coverage == 1) {
                            f32 dx1 = fabsf(fx - layer->rect_x);
                            f32 dx2 = fabsf(fx - (layer->rect_x + layer->rect_w));
                            f32 dy1 = fabsf(fy - layer->rect_y);
                            f32 dy2 = fabsf(fy - (layer->rect_y + layer->rect_h));
                            dist = fminf(fminf(dx1, dx2), fminf(dy1, dy2)) / bevel_edge;
                        }
                        if (dist < 1.0f) {
                            f32 shadow = 1.0f - dist * 0.4f;
                            lr = (u8)(lr * shadow);
                            lg = (u8)(lg * shadow);
                            lb = (u8)(lb * shadow);
                        }
                    }
                    
                    if (layer->blend == 0) {
                        r = lr; g = lg; b = lb;
                    } else if (layer->blend == 1) {
                        r = (u8)FMIN(255, r + lr);
                        g = (u8)FMIN(255, g + lg);
                        b = (u8)FMIN(255, b + lb);
                    } else if (layer->blend == 2) {
                        r = (u8)(r * lr / 255);
                        g = (u8)(g * lg / 255);
                        b = (u8)(b * lb / 255);
                    }
                }
            }
            
            output[(y * tex->width + x) * 4 + 0] = r;
            output[(y * tex->width + x) * 4 + 1] = g;
            output[(y * tex->width + x) * 4 + 2] = b;
            output[(y * tex->width + x) * 4 + 3] = a;
        }
    }
}

static void proc_tex_to_rgba8888(const ProcTexture* tex, u8* output) {
    proc_tex_generate(tex, output);
}

#endif
