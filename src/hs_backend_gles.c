#include "hs_backend_gles.h"
#include "hs_gpu.h"
#include "hs_graphics.h"
#include "hs_nodes.h"
#include "hs_text.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    HSGraphics gfx;
    bool gfx_ok;
    GLuint program;
    GLuint fallback_vbo;
    GLuint vbos[16];
    u32    vbo_sizes[16];
    GLint pos_loc;
    GLint color_loc;
    /* Instanced draws with per-instance offsets use their own program, so
     * plain draws keep a two-attribute shader (an extra constant attribute
     * costs every draw that rebinds its vertex buffer). */
    GLuint inst_program;
    GLint inst_pos_loc;
    GLint inst_color_loc;
    GLint offset_loc;  /* a_offset in inst_program */

    HSFont font;       /* loaded at init; OP_DRAW_TEXT is a no-op without it */
    bool   font_ok;
    u32    text_line;  /* OP_DRAW_TEXT lines drawn so far in this frame */
    char   capture_path[256]; /* non-empty: read back the next frame before presenting */
    bool   capture_ok;

    GLuint tex_program;
    GLuint quad_vbo;
    GLint quad_pos_loc;
    GLint quad_uv_loc;
    GLint quad_tex_loc;
} HSGLESBackend;

static inline GLenum hs_gles_blend_factor(u8 v) {
    switch (v) {
        case 0: return GL_ZERO;
        case 1: return GL_ONE;
        case 5: return GL_SRC_ALPHA;
        case 6: return GL_ONE_MINUS_SRC_ALPHA;
        default: return GL_ONE;
    }
}

static inline GLenum hs_gles_depth_func(u8 v) {
    switch (v) {
        case 0: return GL_ALWAYS;
        case 1: return GL_LESS;
        case 2: return GL_LEQUAL;
        case 3: return GL_EQUAL;
        case 4: return GL_GEQUAL;
        case 5: return GL_GREATER;
        case 6: return GL_NOTEQUAL;
        case 7: return GL_NEVER;
        default: return GL_LEQUAL;
    }
}

static GLuint hs_compile_shader(GLenum type, const char* src) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    GLint compiled = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        glDeleteShader(sh);
        return 0;
    }
    return sh;
}

static GLuint hs_create_program(const char* vs_src, const char* fs_src) {
    GLuint vs = hs_compile_shader(GL_VERTEX_SHADER, vs_src);
    GLuint fs = hs_compile_shader(GL_FRAGMENT_SHADER, fs_src);
    if (!vs || !fs) {
        if (vs) glDeleteShader(vs);
        if (fs) glDeleteShader(fs);
        return 0;
    }
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint linked = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &linked);
    if (!linked) {
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

static bool gles_init(void* ctx, HSGpu* gpu) {
    (void)gpu;
    HSGLESBackend* b = (HSGLESBackend*)ctx;
    if (!b) return false;
    memset(b, 0, sizeof(*b));

    b->gfx_ok = (hs_graphics_init(&b->gfx) == 0);
    if (!b->gfx_ok) return false;

    static const char* vs =
        "attribute vec2 a_position;\n"
        "attribute vec3 a_color;\n"
        "varying vec3 v_color;\n"
        "void main(){\n"
        "  gl_Position = vec4(a_position,0.0,1.0);\n"
        "  v_color = a_color;\n"
        "}\n";

    static const char* vs_instanced =
        "attribute vec2 a_position;\n"
        "attribute vec3 a_color;\n"
        "attribute vec2 a_offset;\n"
        "varying vec3 v_color;\n"
        "void main(){\n"
        "  gl_Position = vec4(a_position + a_offset,0.0,1.0);\n"
        "  v_color = a_color;\n"
        "}\n";

    static const char* fs =
        "precision mediump float;\n"
        "varying vec3 v_color;\n"
        "void main(){\n"
        "  gl_FragColor = vec4(v_color,1.0);\n"
        "}\n";

    b->program = hs_create_program(vs, fs);
    if (!b->program) return false;

    b->pos_loc = glGetAttribLocation(b->program, "a_position");
    b->color_loc = glGetAttribLocation(b->program, "a_color");

    b->inst_program = hs_create_program(vs_instanced, fs);
    if (!b->inst_program) return false;
    b->inst_pos_loc = glGetAttribLocation(b->inst_program, "a_position");
    b->inst_color_loc = glGetAttribLocation(b->inst_program, "a_color");
    b->offset_loc = glGetAttribLocation(b->inst_program, "a_offset");

    float verts[] = {
        /* x, y, r, g, b */
         0.0f,  0.6f,  1.0f, 0.0f, 0.0f,
        -0.6f, -0.6f,  0.0f, 1.0f, 0.0f,
         0.6f, -0.6f,  0.0f, 0.0f, 1.0f,
    };
    glGenBuffers(1, &b->fallback_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, b->fallback_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

    glViewport(0, 0, (GLsizei)b->gfx.screen_width, (GLsizei)b->gfx.screen_height);

    /* Simple texture program for SHOW_TEXTURE */
    static const char* qvs =
        "attribute vec2 a_position;\n"
        "attribute vec2 a_uv;\n"
        "varying vec2 v_uv;\n"
        "void main(){ v_uv=a_uv; gl_Position=vec4(a_position,0.0,1.0); }\n";

    static const char* qfs =
        "precision mediump float;\n"
        "varying vec2 v_uv;\n"
        "uniform sampler2D u_tex;\n"
        "void main(){ gl_FragColor = texture2D(u_tex, v_uv); }\n";

    b->tex_program = hs_create_program(qvs, qfs);
    if (!b->tex_program) return false;
    b->quad_pos_loc = glGetAttribLocation(b->tex_program, "a_position");
    b->quad_uv_loc = glGetAttribLocation(b->tex_program, "a_uv");
    b->quad_tex_loc = glGetUniformLocation(b->tex_program, "u_tex");

    float quad[] = {
        /* x, y, u, v */
        -1, -1, 0, 0,
         1, -1, 1, 0,
        -1,  1, 0, 1,
         1, -1, 1, 0,
         1,  1, 1, 1,
        -1,  1, 0, 1,
    };
    glGenBuffers(1, &b->quad_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, b->quad_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);

    /* Create a simple checker texture in slot 0 (for debugging). */
    {
        u8 pixels[16 * 16 * 4];
        for (int y = 0; y < 16; y++) {
            for (int x = 0; x < 16; x++) {
                int on = ((x >> 2) ^ (y >> 2)) & 1;
                int i = (y * 16 + x) * 4;
                pixels[i + 0] = on ? 255 : 40;
                pixels[i + 1] = on ? 255 : 40;
                pixels[i + 2] = on ? 255 : 40;
                pixels[i + 3] = 255;
            }
        }
        hs_graphics_create_texture(&b->gfx, 0, 16, 16, pixels);
    }

    /* Font for OP_DRAW_TEXT (see hs_font_load_default for the search order). */
    b->font_ok = hs_font_load_default(&b->font);
    if (b->font_ok) hs_font_upload_texture(&b->font);
    b->font_ok = b->font_ok && b->font.atlas_tex != 0;

    return true;
}

static void gles_shutdown(void* ctx, HSGpu* gpu) {
    (void)gpu;
    HSGLESBackend* b = (HSGLESBackend*)ctx;
    if (!b) return;
    if (b->fallback_vbo) glDeleteBuffers(1, &b->fallback_vbo);
    if (b->quad_vbo) glDeleteBuffers(1, &b->quad_vbo);
    for (int i = 0; i < 16; i++) {
        if (b->vbos[i]) glDeleteBuffers(1, &b->vbos[i]);
    }
    if (b->program) glDeleteProgram(b->program);
    if (b->inst_program) glDeleteProgram(b->inst_program);
    if (b->tex_program) glDeleteProgram(b->tex_program);
    if (b->font_ok) hs_font_unload(&b->font);
    if (b->gfx_ok) hs_graphics_finish(&b->gfx);
    memset(b, 0, sizeof(*b));
    free(b); /* the context belongs to this backend instance; see hs_backend_gles_create */
}

#define HS_GLES_VERTEX_BYTES   (5u * 4u)  /* x,y,r,g,b */
#define HS_GLES_INSTANCE_BYTES (2u * 4u)  /* per-instance x,y offset */

/* Upload (when dirty) a memory bank's bytes to its VBO. 0 if the bank is empty. */
static GLuint gles_upload_bank(HSGLESBackend* b, HSBuffer* buf, u8 idx) {
    if (!b || !buf || !buf->data || buf->length == 0) return 0;
    u32 byte_len = buf->length;

    if (!b->vbos[idx]) {
        glGenBuffers(1, &b->vbos[idx]);
        b->vbo_sizes[idx] = 0;
    }

    if (buf->dirty || b->vbo_sizes[idx] != byte_len) {
        glBindBuffer(GL_ARRAY_BUFFER, b->vbos[idx]);
        glBufferData(GL_ARRAY_BUFFER, byte_len, buf->data, GL_STATIC_DRAW);
        b->vbo_sizes[idx] = byte_len;
        buf->dirty = false;
    }

    return b->vbos[idx];
}

/* Vertex source for a draw: the bank when it holds at least one triangle,
 * otherwise the built-in triangle. *vertices is the number of whole vertices. */
static GLuint gles_get_vbo_for_buffer(HSGLESBackend* b, HSBuffer* buf, u8 idx, GLsizei* vertices) {
    *vertices = 3;
    if (!b) return 0;
    if (!buf || !buf->data || buf->length < 3 * HS_GLES_VERTEX_BYTES) return b->fallback_vbo;
    GLuint vbo = gles_upload_bank(b, buf, idx);
    if (!vbo) return b->fallback_vbo;
    *vertices = (GLsizei)(buf->length / HS_GLES_VERTEX_BYTES);
    return vbo;
}

/* Binary PPM of the current back buffer, top row first. Must run before the
 * frame is presented: after a page flip the back buffer content is undefined. */
static bool gles_capture_ppm(HSGLESBackend* b, const char* path) {
    u32 w = b->gfx.screen_width, h = b->gfx.screen_height;
    if (!w || !h) return false;
    /* GL_RGBA/GL_UNSIGNED_BYTE is the only read format GLES guarantees;
     * GL_RGB fails with GL_INVALID_OPERATION and leaves the buffer untouched. */
    u8* pixels = malloc((size_t)w * h * 4);
    u8* row_rgb = malloc((size_t)w * 3);
    if (!pixels || !row_rgb) { free(pixels); free(row_rgb); return false; }
    while (glGetError() != GL_NO_ERROR) {}
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, (GLsizei)w, (GLsizei)h, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    bool ok = glGetError() == GL_NO_ERROR;
    FILE* f = ok ? fopen(path, "wb") : NULL;
    ok = ok && f && fprintf(f, "P6\n%u %u\n255\n", w, h) > 0;
    for (u32 row = 0; ok && row < h; row++) {
        const u8* src = pixels + (size_t)(h - 1 - row) * w * 4;
        for (u32 x = 0; x < w; x++) memcpy(row_rgb + (size_t)x * 3, src + (size_t)x * 4, 3);
        ok = fwrite(row_rgb, 1, (size_t)w * 3, f) == (size_t)w * 3;
    }
    if (f && fclose(f) != 0) ok = false;
    free(pixels);
    free(row_rgb);
    return ok;
}

static void gles_execute(void* ctx, const HSFrameContext* frame) {
    HSGLESBackend* b = (HSGLESBackend*)ctx;
    if (!b->gfx_ok || !frame || !frame->render) return;
    b->text_line = 0;

    for (u32 i = 0; i < frame->render->count; i++) {
        const HSRenderCmd* c = &frame->render->cmds[i];
        switch ((HSRenderOp)c->op) {
            case HS_RC_FRAME_BEGIN:
            case HS_RC_FRAME_END:
            case HS_RC_PRESENT:
                /* Markers only; backend may ignore. */
                break;

            case HS_RC_SET_CULL:
                if (c->a == 0) {
                    glDisable(GL_CULL_FACE);
                } else {
                    glEnable(GL_CULL_FACE);
                    if (c->a == 255) glCullFace(GL_FRONT);
                    else glCullFace(GL_BACK);
                }
                break;

            case HS_RC_SET_BLEND:
                glEnable(GL_BLEND);
                glBlendFunc(hs_gles_blend_factor(c->a), hs_gles_blend_factor(c->b));
                break;

            case HS_RC_SET_ALPHA:
                if (c->a) {
                    glEnable(GL_BLEND);
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                } else {
                    glDisable(GL_BLEND);
                }
                break;

            case HS_RC_SET_DEPTH:
                if (c->a) glEnable(GL_DEPTH_TEST);
                else glDisable(GL_DEPTH_TEST);
                break;

            case HS_RC_SET_DEPTH_COMPARE:
                glEnable(GL_DEPTH_TEST);
                glDepthFunc(hs_gles_depth_func(c->a));
                glDepthMask(c->b ? GL_TRUE : GL_FALSE);
                break;

            case HS_RC_SET_COLOR_MASK:
                glColorMask((c->a & 1) != 0, (c->a & 2) != 0, (c->a & 4) != 0, (c->a & 8) != 0);
                break;

            case HS_RC_SET_CLIP: {
                u32 x = c->x;
                u32 y = c->y;
                u32 w = c->payload_idx;
                u32 h = c->payload_len;
                if (w == 0 || h == 0) {
                    glDisable(GL_SCISSOR_TEST);
                } else {
                    glEnable(GL_SCISSOR_TEST);
                    /* Messages are treated as top-left origin pixels. */
                    /* GL's origin is bottom-left: trim the part of the rect
                     * below the bottom edge instead of shifting it up. */
                    long sy = (long)b->gfx.screen_height - (long)(y + h);
                    long sh = (long)h;
                    if (sy < 0) { sh += sy; sy = 0; }
                    if (sh < 0) sh = 0;
                    glScissor((int)x, (int)sy, (int)w, (int)sh);
                }
                break;
            }

            case HS_RC_SET_TEX_FILTER: {
                u8 slot = c->a & 0x0F;
                u8 linear = c->b & 1;
                GLuint tex = hs_graphics_get_gl_texture(&b->gfx, slot);
                if (tex) {
                    glBindTexture(GL_TEXTURE_2D, tex);
                    GLint f = linear ? GL_LINEAR : GL_NEAREST;
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, f);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, f);
                }
                break;
            }

            case HS_RC_SET_TEX_WRAP: {
                u8 slot = c->a & 0x0F;
                u8 repeat = c->b & 1;
                GLuint tex = hs_graphics_get_gl_texture(&b->gfx, slot);
                if (tex) {
                    glBindTexture(GL_TEXTURE_2D, tex);
                    GLint w = repeat ? GL_REPEAT : GL_CLAMP_TO_EDGE;
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, w);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, w);
                }
                break;
            }

            case HS_RC_CLEAR:
                glClearColor(c->f0, c->f1, c->f2, c->f3);
                glClear(GL_COLOR_BUFFER_BIT);
                break;

            case HS_RC_DRAW:
            case HS_RC_DRAW_INSTANCE:
            {
                bool instanced = (HSRenderOp)c->op == HS_RC_DRAW_INSTANCE;
                u32 instances = instanced ? c->x : 1;
                if (instances == 0) break;
                u8 buf_idx = c->a & 0xF;
                HSBuffer* cpu_buf = NULL;
                if (frame->gpu) {
                    cpu_buf = hs_gpu_get_buffer((HSGpu*)frame->gpu, buf_idx);
                }
                /* Per-instance offsets come from the instance bank (2 floats
                 * per instance) when it holds one for every instance; then
                 * inst_program draws them. Otherwise every instance is drawn
                 * at offset (0,0) by the plain program. */
                GLuint instance_vbo = 0;
                if (instanced && frame->gpu && b->inst_program && b->offset_loc >= 0) {
                    u8 inst_idx = c->b & 0xF;
                    HSBuffer* inst = hs_gpu_get_buffer((HSGpu*)frame->gpu, inst_idx);
                    if (inst && inst->data && (u64)inst->length >= (u64)instances * HS_GLES_INSTANCE_BYTES)
                        instance_vbo = gles_upload_bank(b, inst, inst_idx);
                }
                GLint pos_loc = instance_vbo ? b->inst_pos_loc : b->pos_loc;
                GLint color_loc = instance_vbo ? b->inst_color_loc : b->color_loc;
                GLsizei vertices = 3;
                GLuint vbo = gles_get_vbo_for_buffer(b, cpu_buf, buf_idx, &vertices);
                glUseProgram(instance_vbo ? b->inst_program : b->program);
                glBindBuffer(GL_ARRAY_BUFFER, vbo);
                glEnableVertexAttribArray(pos_loc);
                glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
                glEnableVertexAttribArray(color_loc);
                glVertexAttribPointer(color_loc, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
                if (instance_vbo) {
                    glBindBuffer(GL_ARRAY_BUFFER, instance_vbo);
                    glEnableVertexAttribArray((GLuint)b->offset_loc);
                    glVertexAttribPointer((GLuint)b->offset_loc, 2, GL_FLOAT, GL_FALSE, HS_GLES_INSTANCE_BYTES, (void*)0);
                    glVertexAttribDivisor((GLuint)b->offset_loc, 1);
                }
                if (instanced) glDrawArraysInstanced(GL_TRIANGLES, 0, vertices, (GLsizei)instances);
                else glDrawArrays(GL_TRIANGLES, 0, vertices);
                if (instance_vbo) {
                    /* Leave no per-instance array for other programs. */
                    glVertexAttribDivisor((GLuint)b->offset_loc, 0);
                    glDisableVertexAttribArray((GLuint)b->offset_loc);
                }
                break;
            }

            case HS_RC_CLEAR_DS:
                glClearDepthf(c->f0);
                glClearStencil((GLint)c->a);
                glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
                break;
            case HS_RC_DRAW_TEXT: {
                /* One line per command, stacked from the top-left corner. The
                 * string is the (NUL-terminated) payload of the message. */
                const HSSystem* sys = frame->sys;
                if (!b->font_ok || !sys || !sys->payloads || c->payload_idx >= sys->payload_capacity) break;
                char text[HS_PAYLOAD_SIZE + 1];
                u32 len = c->payload_len > HS_PAYLOAD_SIZE ? HS_PAYLOAD_SIZE : c->payload_len;
                memcpy(text, sys->payloads[c->payload_idx].data, len);
                text[len] = 0;
                /* NDC per screen pixel on each axis; glyphs drawn at 2x. */
                float px = 2.0f / (float)(b->gfx.screen_width ? b->gfx.screen_width : 1);
                float py = 2.0f / (float)(b->gfx.screen_height ? b->gfx.screen_height : 1);
                float sx = 2.0f * px, sy = 2.0f * py;
                float line = (float)(b->font.line_height ? b->font.line_height : 16) * sy;
                float x = -1.0f + 8.0f * px;
                float y = 1.0f - 8.0f * py - (float)b->text_line * line;
                b->text_line++;
                /* The text renderer enables alpha blending; keep the frame's own blend state. */
                GLboolean blend_on = glIsEnabled(GL_BLEND);
                GLint src = GL_ONE, dst = GL_ZERO;
                glGetIntegerv(GL_BLEND_SRC_RGB, &src);
                glGetIntegerv(GL_BLEND_DST_RGB, &dst);
                hs_font_render_text_xy(&b->font, text, x, y, sx, sy, 1.0f, 1.0f, 1.0f, 1.0f);
                glBlendFunc((GLenum)src, (GLenum)dst);
                if (!blend_on) glDisable(GL_BLEND);
                break;
            }
            case HS_RC_SHOW_TEXTURE: {
                u8 slot = c->a & 0x0F;
                GLuint tex = hs_graphics_get_gl_texture(&b->gfx, slot);
                if (!tex) tex = hs_graphics_get_gl_texture(&b->gfx, 0);

                glUseProgram(b->tex_program);
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, tex);
                glUniform1i(b->quad_tex_loc, 0);
                glBindBuffer(GL_ARRAY_BUFFER, b->quad_vbo);
                glEnableVertexAttribArray(b->quad_pos_loc);
                glVertexAttribPointer(b->quad_pos_loc, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
                glEnableVertexAttribArray(b->quad_uv_loc);
                glVertexAttribPointer(b->quad_uv_loc, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
                glDrawArrays(GL_TRIANGLES, 0, 6);
                break;
            }
            default:
                break;
        }
    }

    if (b->capture_path[0]) {
        b->capture_ok = gles_capture_ppm(b, b->capture_path);
        b->capture_path[0] = 0;
    }
    hs_graphics_present(&b->gfx);
}

static const HSBackendOps g_ops = {
    .init = gles_init,
    .shutdown = gles_shutdown,
    .begin_frame = NULL,
    .execute = gles_execute,
    .end_frame = NULL,
};

HSBackend hs_backend_gles_create(void) {
    /* Each backend owns its own context (previously one static instance was
     * shared, so a second init wiped the first backend's live state). The
     * context is released by the backend's shutdown, which hs_gpu_shutdown
     * and a failed hs_gpu_attach_backend both call. ctx is NULL if out of memory. */
    HSBackend b;
    b.ctx = calloc(1, sizeof(HSGLESBackend));
    b.ops = &g_ops;
    return b;
}

bool hs_backend_gles_capture_next(HSBackend* backend, const char* path) {
    if (!backend || backend->ops != &g_ops || !backend->ctx || !path) return false;
    HSGLESBackend* b = (HSGLESBackend*)backend->ctx;
    if (!b->gfx_ok || strlen(path) >= sizeof(b->capture_path)) return false;
    memcpy(b->capture_path, path, strlen(path) + 1);
    b->capture_ok = false;
    return true;
}

bool hs_backend_gles_capture_ok(const HSBackend* backend) {
    if (!backend || backend->ops != &g_ops || !backend->ctx) return false;
    const HSGLESBackend* b = (const HSGLESBackend*)backend->ctx;
    return b->capture_ok && !b->capture_path[0];
}

void hs_backend_gles_flush_present(HSBackend* backend) {
    if (!backend || backend->ops != &g_ops || !backend->ctx) return;
    HSGLESBackend* b = (HSGLESBackend*)backend->ctx;
    if (b->gfx_ok) hs_graphics_flush_present(&b->gfx);
}
