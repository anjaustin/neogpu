#include "hs_text.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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
    "  vec4 tex = texture2D(u_tex, v_uv);\n"
    "  gl_FragColor = vec4(u_color.rgb * tex.a, tex.a);\n"
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

/* The code attribute is the glyph's character as XML text: one literal byte
 * ("0" is the digit zero, not code point 0), a named entity, or &#N;/&#xN;. */
static int hs_font_decode_code_attr(const char* p) {
    if (!p || !*p) return -1;
    if (*p != '&') return p[1] == '"' ? (unsigned char)*p : -1;
    static const struct { const char* name; int code; } entities[] = {
        {"&quot;", '"'}, {"&amp;", '&'}, {"&lt;", '<'}, {"&gt;", '>'}, {"&apos;", '\''},
    };
    for (size_t i = 0; i < sizeof(entities) / sizeof(entities[0]); i++) {
        size_t n = strlen(entities[i].name);
        if (strncmp(p, entities[i].name, n) == 0 && p[n] == '"') return entities[i].code;
    }
    if (p[1] == '#') {
        char* end = NULL;
        long value = (p[2] == 'x' || p[2] == 'X') ? strtol(p + 3, &end, 16) : strtol(p + 2, &end, 10);
        if (end && end[0] == ';' && end[1] == '"' && value >= 0 && value < 0x110000) return (int)value;
    }
    return -1;
}

static int hs_font_parse_glyph(HSGlyph* g, const char* line) {
    int width = 0, ox = 0, oy = 0, rect_x = 0, rect_y = 0, rect_w = 0, rect_h = 0;
    if (sscanf(line, " <Char width=\"%d\" offset=\"%d %d\" rect=\"%d %d %d %d\"",
               &width, &ox, &oy, &rect_x, &rect_y, &rect_w, &rect_h) != 7) return -1;
    const char* p = strstr(line, "code=\"");
    int code = p ? hs_font_decode_code_attr(p + 6) : -1;

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
        /* atlas_data is a fixed HS_FONT_ATLAS_W x HS_FONT_ATLAS_H RGBA array. */
        if (atlas_w > HS_FONT_ATLAS_W || atlas_h > HS_FONT_ATLAS_H) {
            font->loaded = false;
            return false;
        }
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
    bool complete = read_size == sizeof(atlas_data) && fgetc(f) == EOF && !ferror(f);
    fclose(f);
    if (!complete) return false; /* truncated or oversized atlas */
    
    return hs_font_load_fnt(font, fnt_path, atlas_data, HS_FONT_ATLAS_W, HS_FONT_ATLAS_H);
}

bool hs_font_load_default(HSFont* font) {
    const char* env = getenv("NEOGPU_FONT");
    if (env && env[0]) return hs_font_load(font, env);
    if (hs_font_load(font, "src/medodica_font")) return true;
    char exe[512];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n <= 0) return false;
    exe[n] = 0;
    char* slash = strrchr(exe, '/');
    if (!slash) return false;
    *slash = 0;
    char path[600];
    snprintf(path, sizeof(path), "%s/src/medodica_font", exe);
    return hs_font_load(font, path);
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
    hs_font_render_text_xy(font, text, x, y, scale, scale, r, g, b, a);
}

void hs_font_render_text_xy(const HSFont* font, const char* text, float x, float y, float scale_x, float scale_y,
                            float r, float g, float b, float a) {
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
            cursor_y -= font->line_height * scale_y;
            text++;
            continue;
        }
        
        if (c < HS_FONT_MAX_GLYPHS && font->glyphs[c].width > 0) {
            const HSGlyph* g = &font->glyphs[c];
            
            float gw = g->width * scale_x;
            float gh = g->height * scale_y;
            float ox = g->offset_x * scale_x;
            float oy = g->offset_y * scale_y;
            
            // Glyph quad in NDC: a small rectangle placed at the pen position,
            // offset by the glyph's bearing. x,y and scale arrive already in NDC,
            // so the quad spans [left,right] x [bottom,top] around the cursor.
            // (Previously the quad was hardcoded to the full -1..1 viewport, which
            // stretched every glyph across the whole screen.)
            float left   = cursor_x + ox;
            float right  = left + gw;
            float top    = cursor_y - oy;
            float bottom = top - gh;
            
            /* glTexImage2D stores atlas row 0 at t = 0, so a glyph whose
             * rect starts at row tex_y spans t = tex_y (top of the glyph)
             * to t = tex_y + height (bottom). v0 is the quad's bottom edge. */
            float u0 = (float)g->tex_x / atlas_w;
            float v0 = (float)(g->tex_y + g->height) / atlas_h;
            float u1 = (float)(g->tex_x + g->width) / atlas_w;
            float v1 = (float)g->tex_y / atlas_h;
            
            float q[] = {
                left,  bottom, u0, v0,
                right, bottom, u1, v0,
                left,  top,    u0, v1,
                right, bottom, u1, v0,
                right, top,    u1, v1,
                left,  top,    u0, v1,
            };
            
            glBufferData(GL_ARRAY_BUFFER, sizeof(q), q, GL_DYNAMIC_DRAW);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            
            cursor_x += g->advance * scale_x;
        } else {
            cursor_x += font->font_size * 0.5f * scale_x;
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
