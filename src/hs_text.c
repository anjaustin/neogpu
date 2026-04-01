#include "hs_text.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* g_vs_text =
    "attribute vec2 a_pos;\n"
    "attribute vec2 a_uv;\n"
    "varying vec2 v_uv;\n"
    "void main(){\n"
    "  v_uv = a_uv;\n"
    "  gl_Position = vec4(a_pos, 0.0, 1.0);\n"
    "}\n";

static const char* g_fs_text =
    "precision mediump float;\n"
    "varying vec2 v_uv;\n"
    "uniform sampler2D u_tex;\n"
    "uniform vec4 u_color;\n"
    "void main(){\n"
    "  float a = texture2D(u_tex, v_uv).a;\n"
    "  gl_FragColor = vec4(u_color.rgb, u_color.a * a);\n"
    "}\n";

static GLuint g_text_prog = 0;
static GLuint g_text_vbo = 0;
static GLint g_a_pos = -1, g_a_uv = -1;
static GLint g_u_tex = -1, g_u_color = -1;
static bool g_text_init = false;

static bool hs_text_init(void) {
    if (g_text_init) return true;
    
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &g_vs_text, NULL);
    glCompileShader(vs);
    
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &g_fs_text, NULL);
    glCompileShader(fs);
    
    g_text_prog = glCreateProgram();
    glAttachShader(g_text_prog, vs);
    glAttachShader(g_text_prog, fs);
    glLinkProgram(g_text_prog);
    
    glDeleteShader(vs);
    glDeleteShader(fs);
    
    g_a_pos = glGetAttribLocation(g_text_prog, "a_pos");
    g_a_uv = glGetAttribLocation(g_text_prog, "a_uv");
    g_u_tex = glGetUniformLocation(g_text_prog, "u_tex");
    g_u_color = glGetUniformLocation(g_text_prog, "u_color");
    
    glGenBuffers(1, &g_text_vbo);
    
    g_text_init = true;
    return true;
}

static int hs_font_parse_glyph(HSGlyph* g, const char* line) {
    int width = 0, ox = 0, oy = 0, rect_x = 0, rect_y = 0, rect_w = 0, rect_h = 0;
    int code = -1;
    
    if (sscanf(line, " <Char width=\"%d\" offset=\"%d %d\" rect=\"%d %d %d %d\" code=\"%d\"",
               &width, &ox, &oy, &rect_x, &rect_y, &rect_w, &rect_h, &code) < 8) {
        // Try parsing code as a single character
        const char* p = strstr(line, "code=\"");
        if (p) {
            p += 6;
            if (*p >= '0' && *p <= '9') {
                code = atoi(p);
            } else {
                code = (unsigned char)*p;
            }
        }
    }
    
    if (code >= 0 && code < HS_FONT_MAX_GLYPHS) {
        g[code].width = (u16)rect_w;
        g[code].height = (u16)rect_h;
        g[code].offset_x = (int16_t)ox;   // render offset X
        g[code].offset_y = (int16_t)oy;   // render offset Y
        g[code].tex_x = (int16_t)rect_x;  // texture position X
        g[code].tex_y = (int16_t)rect_y;  // texture position Y
        g[code].advance = (u8)width;
        return code;
    }
    return -1;
}

bool hs_font_load_fnt(HSFont* font, const char* fnt_path, const u8* atlas_data, int atlas_w, int atlas_h) {
    if (!font) return false;
    
    memset(font, 0, sizeof(*font));
    
    if (fnt_path) {
        FILE* f = fopen(fnt_path, "r");
        if (f) {
            char line[256];
            while (fgets(line, sizeof(line), f)) {
                if (strstr(line, "<Char ")) {
                    hs_font_parse_glyph(font->glyphs, line);
                } else if (strstr(line, "size=\"")) {
                    int size;
                    if (sscanf(line, " <Font size=\"%d\"", &size) == 1) {
                        font->font_size = (u16)size;
                        font->line_height = (u16)(size + 4);
                    }
                }
            }
            fclose(f);
        } else {
            fprintf(stderr, "DEBUG: could not open %s\n", fnt_path);
        }
    }
    
    if (atlas_data && atlas_w > 0 && atlas_h > 0) {
        font->atlas_w = (u16)atlas_w;
        font->atlas_h = (u16)atlas_h;
        memcpy(font->atlas_data, atlas_data, (size_t)atlas_w * atlas_h * 4);
    }
    
    font->atlas_tex = 0;
    font->loaded = true;
    return true;
}

bool hs_font_load(HSFont* font, const char* base_path) {
    if (!font || !base_path) return false;
    
    char fnt_path[512];
    char atlas_path[512];
    
    snprintf(fnt_path, sizeof(fnt_path), "%s.fnt", base_path);
    snprintf(atlas_path, sizeof(atlas_path), "%s.atlas", base_path);
    
    FILE* f = fopen(atlas_path, "rb");
    if (!f) return false;
    
    u8 atlas_data[HS_FONT_ATLAS_W * HS_FONT_ATLAS_H * 4];
    size_t read_size = fread(atlas_data, 1, sizeof(atlas_data), f);
    (void)read_size;
    fclose(f);
    
    return hs_font_load_fnt(font, fnt_path, atlas_data, HS_FONT_ATLAS_W, HS_FONT_ATLAS_H);
}

void hs_font_upload_texture(HSFont* font) {
    if (!font || !font->loaded || font->atlas_tex) return;
    
    glGenTextures(1, &font->atlas_tex);
    glBindTexture(GL_TEXTURE_2D, font->atlas_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, font->atlas_w, font->atlas_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, font->atlas_data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void hs_font_unload(HSFont* font) {
    if (!font) return;
    if (font->atlas_tex) {
        glDeleteTextures(1, &font->atlas_tex);
        font->atlas_tex = 0;
    }
    memset(font, 0, sizeof(*font));
}

void hs_font_render_text(const HSFont* font, const char* text, float x, float y, float scale, float r, float g, float b, float a) {
    if (!font || !text || !font->loaded || !font->atlas_tex) return;
    
    hs_text_init();
    
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    glUseProgram(g_text_prog);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, font->atlas_tex);
    glUniform1i(g_u_tex, 0);
    glUniform4f(g_u_color, r, g, b, a);
    
    glBindBuffer(GL_ARRAY_BUFFER, g_text_vbo);
    glEnableVertexAttribArray(g_a_pos);
    glEnableVertexAttribArray(g_a_uv);
    glVertexAttribPointer(g_a_pos, 2, GL_FLOAT, GL_FALSE, 16, (void*)0);
    glVertexAttribPointer(g_a_uv, 2, GL_FLOAT, GL_FALSE, 16, (void*)8);
    
    float cursor_x = x;
    float cursor_y = y;
    float atlas_w = (float)font->atlas_w;
    float atlas_h = (float)font->atlas_h;
    
    while (*text) {
        unsigned char c = (unsigned char)*text;
        
        if (c == '\n') {
            cursor_x = x;
            cursor_y -= font->line_height * scale;
            text++;
            continue;
        }
        
        if (c < HS_FONT_MAX_GLYPHS && font->glyphs[c].width > 0) {
            const HSGlyph* g = &font->glyphs[c];
            
            float gw = g->width * scale;
            float gh = g->height * scale;
            float ox = g->offset_x * scale;
            float oy = g->offset_y * scale;
            
            float u0 = (float)g->tex_x / atlas_w;
            float v0 = (float)(atlas_h - g->tex_y - g->height) / atlas_h;
            float u1 = (float)(g->tex_x + g->width) / atlas_w;
            float v1 = (float)(atlas_h - g->tex_y) / atlas_h;
            
            float q[] = {
                cursor_x + ox, cursor_y + oy + gh, u0, v1,
                cursor_x + ox + gw, cursor_y + oy + gh, u1, v1,
                cursor_x + ox, cursor_y + oy, u0, v0,
                cursor_x + ox + gw, cursor_y + oy + gh, u1, v1,
                cursor_x + ox + gw, cursor_y + oy, u1, v0,
                cursor_x + ox, cursor_y + oy, u0, v0,
            };
            
            glBufferData(GL_ARRAY_BUFFER, sizeof(q), q, GL_DYNAMIC_DRAW);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            
            cursor_x += g->advance * scale;
        } else {
            cursor_x += font->font_size * 0.5f * scale;
        }
        
        text++;
    }
}

float hs_font_text_width(const HSFont* font, const char* text, float scale) {
    if (!font || !text || !font->loaded) return 0.0f;
    float width = 0.0f;
    while (*text) {
        unsigned char c = (unsigned char)*text;
        if (c < HS_FONT_MAX_GLYPHS && font->glyphs[c].advance > 0) {
            width += font->glyphs[c].advance * scale;
        } else {
            width += font->font_size * 0.5f * scale;
        }
        text++;
    }
    return width;
}

float hs_font_text_height(const HSFont* font, float scale) {
    if (!font || !font->loaded) return 0.0f;
    return font->line_height * scale;
}
