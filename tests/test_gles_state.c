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
/* Attribute 2 is the per-instance offset (a_offset). */
static struct { bool enabled; GLint size; GLsizei stride; const void* pointer; GLuint buffer; GLuint divisor; } attributes[3];
static struct { GLuint program, buffer; GLsizei vertices; GLsizei instances; GLuint offsets; } expected[32];
static u32 expected_count, draws, uploads, program_calls, pointer_calls, constant_offsets;
static GLint scissor[4];
static bool scissor_enabled;

void __wrap_glUseProgram(GLuint program) { current_program = program; program_calls++; }
void __wrap_glBindBuffer(GLenum target, GLuint buffer) { REQUIRE(target == GL_ARRAY_BUFFER); current_buffer = buffer; }
void __wrap_glEnableVertexAttribArray(GLuint index) { REQUIRE(index < 3); attributes[index].enabled = true; }
void __wrap_glDisableVertexAttribArray(GLuint index) { REQUIRE(index == 2); attributes[index].enabled = false; }
void __wrap_glVertexAttrib2f(GLuint index, GLfloat x, GLfloat y) { REQUIRE(index == 2 && x == 0 && y == 0); constant_offsets++; }
void __wrap_glVertexAttribDivisor(GLuint index, GLuint divisor) { REQUIRE(index == 2 && divisor <= 1); attributes[index].divisor = divisor; }
void __wrap_glScissor(GLint x, GLint y, GLsizei w, GLsizei h) { scissor[0] = x; scissor[1] = y; scissor[2] = w; scissor[3] = h; }
void __wrap_glEnable(GLenum cap) { if (cap == GL_SCISSOR_TEST) scissor_enabled = true; }
void __wrap_glDisable(GLenum cap) { if (cap == GL_SCISSOR_TEST) scissor_enabled = false; }
void __wrap_glVertexAttribPointer(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void* pointer) {
    REQUIRE(index < 3 && type == GL_FLOAT && normalized == GL_FALSE && current_buffer);
    attributes[index].size = size; attributes[index].stride = stride; attributes[index].pointer = pointer;
    attributes[index].buffer = current_buffer; pointer_calls++;
}
static void check_draw(GLenum mode, GLint first, GLsizei count, GLsizei instances) {
    REQUIRE(mode == GL_TRIANGLES && first == 0 && draws < expected_count);
    REQUIRE(current_program == expected[draws].program && count == expected[draws].vertices);
    REQUIRE(instances == expected[draws].instances);
    if (expected[draws].offsets) {
        REQUIRE(attributes[2].enabled && attributes[2].divisor == 1 && attributes[2].size == 2);
        REQUIRE(attributes[2].buffer == expected[draws].offsets && attributes[2].stride == 8);
    } else {
        REQUIRE(!attributes[2].enabled);
    }
    bool textured = current_program == backend.tex_program;
    for (u32 i = 0; i < 2; i++) {
        REQUIRE(attributes[i].enabled && attributes[i].buffer == expected[draws].buffer);
        REQUIRE(attributes[i].size == (i == 0 || textured ? 2 : 3));
        REQUIRE(attributes[i].stride == (GLsizei)((textured ? 4 : 5) * sizeof(float)));
        REQUIRE(attributes[i].pointer == (void*)(uintptr_t)(i ? 2 * sizeof(float) : 0));
    }
    draws++;
}
void __wrap_glDrawArrays(GLenum mode, GLint first, GLsizei count) { check_draw(mode, first, count, 0); }
void __wrap_glDrawArraysInstanced(GLenum mode, GLint first, GLsizei count, GLsizei instances) {
    REQUIRE(instances > 0);
    check_draw(mode, first, count, instances);
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
/* Banks 0 and 1 hold geometry; the vertex count follows the bank length. */
static GLsizei bank_vertices(u8 bank) {
    return bank < 2 ? (GLsizei)(hs_gpu_get_buffer(&gpu, bank)->length / 20) : 3;
}
static void push_full(u8 op, u8 bank, u8 instance_bank, u32 instances) {
    HSRenderCmd command = {.op = op, .a = bank, .b = instance_bank, .x = instances};
    REQUIRE(hs_render_push(&gpu.render, &command));
    if (op != HS_RC_DRAW && op != HS_RC_DRAW_INSTANCE && op != HS_RC_SHOW_TEXTURE) return;
    if (op == HS_RC_DRAW_INSTANCE && instances == 0) return; /* nothing is drawn */
    REQUIRE(expected_count < sizeof(expected) / sizeof(expected[0]));
    bool textured = op == HS_RC_SHOW_TEXTURE;
    expected[expected_count].program = textured ? 2 : 1;
    expected[expected_count].buffer = textured ? 77 : bank < 2 ? backend.vbos[bank] : backend.fallback_vbo;
    expected[expected_count].vertices = textured ? 6 : bank_vertices(bank);
    expected[expected_count].instances = op == HS_RC_DRAW_INSTANCE ? (GLsizei)instances : 0;
    /* Offsets are used only when the instance bank holds 8 bytes per instance. */
    bool offsets = op == HS_RC_DRAW_INSTANCE && instance_bank < 2 &&
                   hs_gpu_get_buffer(&gpu, instance_bank)->length >= instances * 8;
    expected[expected_count].offsets = offsets ? backend.vbos[instance_bank] : 0;
    expected_count++;
}
static void push(u8 op, u8 bank) { push_full(op, bank, 0, 0); }
static u32 draws_total;
static void execute(void) {
    HSFrameContext frame = {.gpu = &gpu, .render = &gpu.render};
    gles_execute(&backend, &frame); REQUIRE(draws == expected_count); draws_total += draws;
}
int main(void) {
    backend.gfx_ok = true;
    /* initialized remains false: hs_graphics_present must not touch EGL. */
    backend.program = 1; backend.tex_program = 2; backend.fallback_vbo = 99; backend.quad_vbo = 77;
    backend.vbos[0] = 10; backend.vbos[1] = 11;
    backend.pos_loc = backend.quad_pos_loc = 0; backend.color_loc = backend.quad_uv_loc = 1; backend.quad_tex_loc = 2;
    backend.offset_loc = 2; backend.gfx.screen_height = 480;
    backend.gfx.textures[0].gl_texture = 123;
    hs_memory_init(&gpu.memory);
    for (u32 i = 0; i < 2; i++) {
        HSBuffer* buffer = hs_gpu_get_buffer(&gpu, i);
        REQUIRE(hs_buffer_init(buffer, i, 128)); buffer->length = 60;
    }
    reset_external_state();
    push(HS_RC_DRAW, 0); push(HS_RC_DRAW, 0); push(HS_RC_DRAW, 1);
    push_full(HS_RC_DRAW_INSTANCE, 1, 0, 7);   /* bank 0 holds 60 B >= 7*8: per-instance offsets */
    push_full(HS_RC_DRAW_INSTANCE, 1, 0, 8);   /* 64 B needed, 60 B held: constant offset */
    push_full(HS_RC_DRAW_INSTANCE, 1, 0, 0);   /* zero instances: no draw call */
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
    REQUIRE(expected[1].vertices == 4); /* an 80-byte bank is 4 vertices, not a fixed 3 */
    REQUIRE(constant_offsets > 0);
    /* A clip rect that runs off the bottom edge is trimmed, not shifted up:
     * top-left rows 400..479 of a 480-row target are GL rows 0..79. */
    hs_render_reset(&gpu.render); reset_external_state();
    HSRenderCmd clip = {.op = HS_RC_SET_CLIP, .x = 0, .y = 400, .payload_idx = 100, .payload_len = 200};
    REQUIRE(hs_render_push(&gpu.render, &clip)); execute();
    REQUIRE(scissor_enabled && scissor[0] == 0 && scissor[1] == 0 && scissor[2] == 100 && scissor[3] == 80);
    clip.y = 10; clip.payload_len = 20; hs_render_reset(&gpu.render);
    REQUIRE(hs_render_push(&gpu.render, &clip)); execute();
    REQUIRE(scissor[1] == 450 && scissor[3] == 20);
    clip.y = 600; hs_render_reset(&gpu.render);   /* entirely below the target */
    REQUIRE(hs_render_push(&gpu.render, &clip)); execute();
    REQUIRE(scissor[3] == 0);
    hs_memory_free(&gpu.memory);
    printf("{\"test\":\"gles_real_backend_headless_state_transitions\",\"frames\":6,\"draws\":%u,\"uploads\":%u,\"program_calls\":%u,\"pointer_calls\":%u,\"passed\":true}\n", draws_total, uploads, program_calls, pointer_calls);
    return 0;
}
