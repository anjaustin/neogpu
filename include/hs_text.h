#ifndef HS_TEXT_H
#define HS_TEXT_H

#include "hs_core.h"
#include "hs_math_neon.h"
#include <stdint.h>

#ifdef __linux__
#include <GLES3/gl3.h>
#endif

#define HS_FONT_MAX_GLYPHS 128
#define HS_FONT_ATLAS_W 128
#define HS_FONT_ATLAS_H 64

typedef struct {
    u16 width;
    u16 height;
    int16_t offset_x;
    int16_t offset_y;
    u8 advance;
} HSGlyph;

typedef struct {
    u8 atlas_data[HS_FONT_ATLAS_W * HS_FONT_ATLAS_H * 4];
    u16 atlas_w;
    u16 atlas_h;
    u16 font_size;
    u16 line_height;
    HSGlyph glyphs[HS_FONT_MAX_GLYPHS];
    bool loaded;
    GLuint atlas_tex;
} HSFont;

bool hs_font_load_fnt(HSFont* font, const char* fnt_path, const u8* atlas_data, int atlas_w, int atlas_h);
bool hs_font_load(HSFont* font, const char* base_path);
void hs_font_unload(HSFont* font);
void hs_font_upload_texture(HSFont* font);

void hs_font_render_text(const HSFont* font, const char* text, float x, float y, float scale, float r, float g, float b, float a);

float hs_font_text_width(const HSFont* font, const char* text, float scale);
float hs_font_text_height(const HSFont* font, float scale);

#endif
