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
} HSGLESBackend;

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

    return true;
}

static void gles_shutdown(void* ctx, HSGpu* gpu) {
    (void)gpu;
    HSGLESBackend* b = (HSGLESBackend*)ctx;
    if (b->fallback_vbo) glDeleteBuffers(1, &b->fallback_vbo);
    for (int i = 0; i < 16; i++) {
        if (b->vbos[i]) glDeleteBuffers(1, &b->vbos[i]);
    }
    if (b->program) glDeleteProgram(b->program);
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
            case HS_RC_DRAW_TEXT:
            case HS_RC_SHOW_TEXTURE:
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
