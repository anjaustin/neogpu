/* Headless semantic observer for the real backend implementation. GL setup
 * calls are intercepted; no context/device or fake rendering-speed claim. */
#include "hs_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#ifndef HS_GLES_SOURCE
#define HS_GLES_SOURCE "../src/hs_backend_gles.c"
#endif
#include HS_GLES_SOURCE

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "%d: %s\n", __LINE__, #x); exit(1); } } while (0)
static HSGpu gpu;
static HSGLESBackend backend;
static GLuint current_program, current_buffer;
static struct { bool enabled; GLint size; GLsizei stride; const void* pointer; GLuint buffer; } attributes[2];
static struct { GLuint program, buffer; GLsizei vertices; } expected[32];
static u32 expected_count, draws, uploads, program_calls, pointer_calls;

void __wrap_glUseProgram(GLuint program) { current_program = program; program_calls++; }
void __wrap_glBindBuffer(GLenum target, GLuint buffer) { REQUIRE(target == GL_ARRAY_BUFFER); current_buffer = buffer; }
void __wrap_glEnableVertexAttribArray(GLuint index) { REQUIRE(index < 2); attributes[index].enabled = true; }
void __wrap_glVertexAttribPointer(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void* pointer) {
    REQUIRE(index < 2 && type == GL_FLOAT && normalized == GL_FALSE && current_buffer);
    attributes[index].size = size; attributes[index].stride = stride; attributes[index].pointer = pointer;
    attributes[index].buffer = current_buffer; pointer_calls++;
}
void __wrap_glDrawArrays(GLenum mode, GLint first, GLsizei count) {
    REQUIRE(mode == GL_TRIANGLES && first == 0 && draws < expected_count);
    REQUIRE(current_program == expected[draws].program && count == expected[draws].vertices);
    bool textured = current_program == backend.tex_program;
    for (u32 i = 0; i < 2; i++) {
        REQUIRE(attributes[i].enabled && attributes[i].buffer == expected[draws].buffer);
        REQUIRE(attributes[i].size == (i == 0 || textured ? 2 : 3));
        REQUIRE(attributes[i].stride == (GLsizei)((textured ? 4 : 5) * sizeof(float)));
        REQUIRE(attributes[i].pointer == (void*)(uintptr_t)(i ? 2 * sizeof(float) : 0));
    }
    draws++;
}
void __wrap_glBufferData(GLenum target, GLsizeiptr size, const void* data, GLenum usage) {
    REQUIRE(target == GL_ARRAY_BUFFER && current_buffer && size >= 60 && data && usage == GL_STATIC_DRAW); uploads++;
}
void __wrap_glActiveTexture(GLenum unit) { REQUIRE(unit == GL_TEXTURE0); }
void __wrap_glBindTexture(GLenum target, GLuint texture) { REQUIRE(target == GL_TEXTURE_2D && texture == 123); }
void __wrap_glUniform1i(GLint location, GLint value) { REQUIRE(location == 2 && value == 0); }

static void reset_external_state(void) {
    memset(attributes, 0, sizeof(attributes)); current_program = 999; current_buffer = 0;
    draws = expected_count = 0;
}
static void push(u8 op, u8 bank) {
    HSRenderCmd command = {.op = op, .a = bank}; REQUIRE(hs_render_push(&gpu.render, &command));
    if (op != HS_RC_DRAW && op != HS_RC_DRAW_INSTANCE && op != HS_RC_SHOW_TEXTURE) return;
    REQUIRE(expected_count < sizeof(expected) / sizeof(expected[0]));
    bool textured = op == HS_RC_SHOW_TEXTURE;
    expected[expected_count].program = textured ? 2 : 1;
    expected[expected_count].buffer = textured ? 77 : bank < 2 ? backend.vbos[bank] : backend.fallback_vbo;
    expected[expected_count].vertices = textured ? 6 : 3; expected_count++;
}
static void execute(void) {
    HSFrameContext frame = {.gpu = &gpu, .render = &gpu.render};
    gles_execute(&backend, &frame); REQUIRE(draws == expected_count);
}
int main(void) {
    backend.gfx_ok = true;
    /* initialized remains false: hs_graphics_present must not touch EGL. */
    backend.program = 1; backend.tex_program = 2; backend.fallback_vbo = 99; backend.quad_vbo = 77;
    backend.vbos[0] = 10; backend.vbos[1] = 11;
    backend.pos_loc = backend.quad_pos_loc = 0; backend.color_loc = backend.quad_uv_loc = 1; backend.quad_tex_loc = 2;
    backend.gfx.textures[0].gl_texture = 123;
    hs_memory_init(&gpu.memory);
    for (u32 i = 0; i < 2; i++) {
        HSBuffer* buffer = hs_gpu_get_buffer(&gpu, i);
        REQUIRE(hs_buffer_init(buffer, i, 128)); buffer->length = 60;
    }
    reset_external_state();
    push(HS_RC_DRAW, 0); push(HS_RC_DRAW, 0); push(HS_RC_DRAW, 1); push(HS_RC_DRAW_INSTANCE, 1);
    push(HS_RC_SHOW_TEXTURE, 0); push(HS_RC_DRAW, 1); push(HS_RC_DRAW, 0);
    push(HS_RC_FRAME_END, 0); push(HS_RC_DRAW, 0); push(HS_RC_DRAW, 15); push(HS_RC_DRAW, 15);
    execute(); REQUIRE(uploads == 2);
    /* A new no-clear frame must recover arbitrary external program/array state. */
    hs_render_reset(&gpu.render); reset_external_state(); push(HS_RC_DRAW, 0); push(HS_RC_DRAW, 0); execute();
    REQUIRE(uploads == 2);
    /* Upload helper binds a different bank; pointers must follow that binding. */
    hs_render_reset(&gpu.render); reset_external_state();
    gpu.memory.banks[0].dirty = true; gpu.memory.banks[1].dirty = true; gpu.memory.banks[1].length = 80;
    push(HS_RC_DRAW, 0); push(HS_RC_DRAW, 1); push(HS_RC_DRAW, 1); execute(); REQUIRE(uploads == 4);
    hs_memory_free(&gpu.memory);
    printf("{\"test\":\"gles_real_backend_headless_state_transitions\",\"frames\":3,\"draws\":15,\"uploads\":%u,\"program_calls\":%u,\"pointer_calls\":%u,\"passed\":true}\n", uploads, program_calls, pointer_calls);
    return 0;
}
