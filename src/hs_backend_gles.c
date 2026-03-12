#include "hs_backend_gles.h"
#include "hs_gpu.h"
#include "hs_graphics.h"
#include "hs_nodes.h"

typedef struct {
    HSGraphics gfx;
    bool gfx_ok;
    GLuint program;
    GLuint fallback_vbo;
    GLuint vbos[16];
    u32    vbo_sizes[16];
    GLint pos_loc;
    GLint color_loc;

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
    if (!vs || !fs) return 0;
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

    return true;
}

static void gles_shutdown(void* ctx, HSGpu* gpu) {
    (void)gpu;
    HSGLESBackend* b = (HSGLESBackend*)ctx;
    if (b->fallback_vbo) glDeleteBuffers(1, &b->fallback_vbo);
    if (b->quad_vbo) glDeleteBuffers(1, &b->quad_vbo);
    for (int i = 0; i < 16; i++) {
        if (b->vbos[i]) glDeleteBuffers(1, &b->vbos[i]);
    }
    if (b->program) glDeleteProgram(b->program);
    if (b->tex_program) glDeleteProgram(b->tex_program);
    if (b->gfx_ok) hs_graphics_finish(&b->gfx);
    memset(b, 0, sizeof(*b));
}

static GLuint gles_get_vbo_for_buffer(HSGLESBackend* b, HSBuffer* buf, u8 idx) {
    if (!b) return 0;
    if (!buf || !buf->data || buf->length == 0) return b->fallback_vbo;

    /* Expect 5 floats per vertex: x,y,r,g,b */
    u32 stride = 5 * 4;
    u32 byte_len = buf->length;
    if (byte_len < 3 * stride) return b->fallback_vbo;

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

static void gles_execute(void* ctx, const HSFrameContext* frame) {
    HSGLESBackend* b = (HSGLESBackend*)ctx;
    if (!b->gfx_ok || !frame || !frame->render) return;

    for (u32 i = 0; i < frame->render->count; i++) {
        const HSRenderCmd* c = &frame->render->cmds[i];
        switch ((HSRenderOp)c->op) {
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
                    int sy = (int)b->gfx.screen_height - (int)(y + h);
                    if (sy < 0) sy = 0;
                    glScissor((int)x, sy, (int)w, (int)h);
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
                u8 buf_idx = c->a & 0xF;
                HSBuffer* cpu_buf = NULL;
                if (frame->gpu) {
                    cpu_buf = hs_gpu_get_buffer((HSGpu*)frame->gpu, buf_idx);
                }
                GLuint vbo = gles_get_vbo_for_buffer(b, cpu_buf, buf_idx);
                glUseProgram(b->program);
                glBindBuffer(GL_ARRAY_BUFFER, vbo);
                glEnableVertexAttribArray(b->pos_loc);
                glVertexAttribPointer(b->pos_loc, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
                glEnableVertexAttribArray(b->color_loc);
                glVertexAttribPointer(b->color_loc, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
                glDrawArrays(GL_TRIANGLES, 0, 3);
                break;
            }

            case HS_RC_CLEAR_DS:
                glClearDepthf(c->f0);
                glClearStencil((GLint)c->a);
                glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
                break;
            case HS_RC_DRAW_TEXT:
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
    static HSGLESBackend backend_state;
    HSBackend b;
    b.ctx = &backend_state;
    b.ops = &g_ops;
    return b;
}
