/*
 * NeoGPU - V3D Ternary GEMM Coprocessor
 *
 * GPU-accelerated ternary GEMM using GLES 3.1 compute shaders.
 * Falls back to CPU (NEON) when GPU is unavailable.
 *
 * Build with GPU support:
 *   -DHAS_GLES_COMPUTE -lGLESv2 -lEGL
 *
 * Build CPU-only:
 *   (no extra flags needed)
 */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#include "hs_ml_ternary_coproc.h"

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;
typedef float f32;

/* Forward declare CPU fallback */
extern void hs_ml_ternary_f32_proj(float *out, const float *in,
                                    const uint8_t *W, uint32_t N, uint32_t K);

/*============================================================================
 * Timing
 *============================================================================*/

static u64 ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (u64)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/*============================================================================
 * GPU Context (GLES 3.1 Compute)
 *============================================================================*/

#ifdef HAS_GLES_COMPUTE

#include <fcntl.h>
#include <unistd.h>
#include <gbm.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>

typedef struct {
    int drm_fd;
    struct gbm_device* gbm;
    EGLDisplay display;
    EGLContext context;
    EGLSurface surface;
    GLuint program;
    GLuint ssbo_weights;
    GLuint ssbo_input;
    GLuint ssbo_output;
    GLuint ssbo_uniforms;
    size_t weights_size;
    size_t input_size;
    size_t output_size;
    bool active;
    u64 total_time_ns;
    u32 num_projections;
} GLESContext;

static GLESContext g_ctx = { .active = false };

/* Ternary GEMM compute shader
 *
 * Each thread computes one output element.
 * 
 * BitNet I2_S encoding (GGUF type 36):
 *   Sequential 2-bit packing, 4 weights per byte, low bits first.
 *   Code mapping: 0=-1, 1=0, 2=+1, 3=0
 */
static const char* TERNARY_GEMM_SHADER =
    "#version 310 es\n"
    "precision highp float;\n"
    "precision highp int;\n"
    "\n"
    "layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;\n"
    "\n"
    "layout(std430, binding = 0) readonly buffer WeightBuffer {\n"
    "    uint weights[];\n"
    "};\n"
    "\n"
    "layout(std430, binding = 1) readonly buffer InputBuffer {\n"
    "    float inputs[];\n"
    "};\n"
    "\n"
    "layout(std430, binding = 2) writeonly buffer OutputBuffer {\n"
    "    float outputs[];\n"
    "};\n"
    "\n"
    "layout(std430, binding = 3) readonly buffer UniformBuffer {\n"
    "    uint N;\n"
    "    uint K;\n"
    "};\n"
    "\n"
    "void main() {\n"
    "    uint n = gl_GlobalInvocationID.x;\n"
    "    if (n >= N) return;\n"
    "\n"
    "    uint bytes_per_row = K / 4u;\n"
    "    uint uints_per_row = (bytes_per_row + 3u) / 4u;\n"
    "    uint row_offset = n * uints_per_row;\n"
    "\n"
    "    float acc = 0.0;\n"
    "\n"
    "    for (uint k = 0u; k < K; k += 4u) {\n"
    "        uint byte_idx = k / 4u;\n"
    "        uint uint_idx = byte_idx / 4u;\n"
    "        uint byte_off = byte_idx & 3u;\n"
    "\n"
    "        uint wdata = weights[row_offset + uint_idx];\n"
    "        uint b = (wdata >> (byte_off * 8u)) & 0xFFu;\n"
    "\n"
    "        float a0 = inputs[k + 0u];\n"
    "        float a1 = inputs[k + 1u];\n"
    "        float a2 = inputs[k + 2u];\n"
    "        float a3 = inputs[k + 3u];\n"
    "\n"
    "        uint c0 = b & 3u;\n"
    "        uint c1 = (b >> 2u) & 3u;\n"
    "        uint c2 = (b >> 4u) & 3u;\n"
    "        uint c3 = (b >> 6u) & 3u;\n"
    "\n"
        "        // Code: 0=-1, 1=0, 2=+1, 3=0\n"
    "        if (c0 == 2u) acc += a0; else if (c0 == 0u) acc -= a0;\n"
    "        if (c1 == 2u) acc += a1; else if (c1 == 0u) acc -= a1;\n"
    "        if (c2 == 2u) acc += a2; else if (c2 == 0u) acc -= a2;\n"
    "        if (c3 == 2u) acc += a3; else if (c3 == 0u) acc -= a3;\n"
    "    }\n"
    "\n"
    "    outputs[n] = acc;\n"
    "}\n";

static int gles_init_context(void) {
    /* Open DRM device for headless GPU access */
    g_ctx.drm_fd = open("/dev/dri/renderD128", O_RDWR);
    if (g_ctx.drm_fd < 0) {
        /* Try card1 as fallback */
        g_ctx.drm_fd = open("/dev/dri/card1", O_RDWR);
        if (g_ctx.drm_fd < 0) {
            g_ctx.drm_fd = open("/dev/dri/card0", O_RDWR);
        }
    }
    if (g_ctx.drm_fd < 0) {
        fprintf(stderr, "GLES: failed to open DRM device\n");
        return -1;
    }

    /* Create GBM device */
    g_ctx.gbm = gbm_create_device(g_ctx.drm_fd);
    if (!g_ctx.gbm) {
        fprintf(stderr, "GLES: failed to create GBM device\n");
        close(g_ctx.drm_fd);
        return -1;
    }

    /* Get EGL display from GBM */
    g_ctx.display = eglGetDisplay((EGLNativeDisplayType)g_ctx.gbm);
    if (g_ctx.display == EGL_NO_DISPLAY) {
        fprintf(stderr, "GLES: failed to get EGL display from GBM\n");
        gbm_device_destroy(g_ctx.gbm);
        close(g_ctx.drm_fd);
        return -1;
    }

    EGLint major, minor;
    if (!eglInitialize(g_ctx.display, &major, &minor)) {
        fprintf(stderr, "GLES: failed to initialize EGL: 0x%x\n", eglGetError());
        gbm_device_destroy(g_ctx.gbm);
        close(g_ctx.drm_fd);
        return -1;
    }
    fprintf(stderr, "GLES: EGL %d.%d initialized\n", major, minor);

    if (!eglBindAPI(EGL_OPENGL_ES_API)) {
        fprintf(stderr, "GLES: failed to bind GLES API\n");
        eglTerminate(g_ctx.display);
        gbm_device_destroy(g_ctx.gbm);
        close(g_ctx.drm_fd);
        return -1;
    }

    /* For surfaceless compute, use minimal config */
    EGLint config_attribs[] = {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
        EGL_NONE
    };

    EGLConfig config;
    EGLint num_configs;
    if (!eglChooseConfig(g_ctx.display, config_attribs, &config, 1, &num_configs) || num_configs == 0) {
        fprintf(stderr, "GLES: failed to choose EGL config (tried %d)\n", num_configs);
        eglTerminate(g_ctx.display);
        gbm_device_destroy(g_ctx.gbm);
        close(g_ctx.drm_fd);
        return -1;
    }
    fprintf(stderr, "GLES: found %d configs\n", num_configs);

    /* For compute-only, we can use surfaceless context */
    EGLint context_attribs[] = {
        EGL_CONTEXT_MAJOR_VERSION, 3,
        EGL_CONTEXT_MINOR_VERSION, 1,
        EGL_NONE
    };

    g_ctx.context = eglCreateContext(g_ctx.display, config, EGL_NO_CONTEXT, context_attribs);
    if (g_ctx.context == EGL_NO_CONTEXT) {
        fprintf(stderr, "GLES: failed to create context (GLES 3.1 required): 0x%x\n", eglGetError());
        eglTerminate(g_ctx.display);
        gbm_device_destroy(g_ctx.gbm);
        close(g_ctx.drm_fd);
        return -1;
    }

    /* Make context current without a surface (surfaceless) */
    if (!eglMakeCurrent(g_ctx.display, EGL_NO_SURFACE, EGL_NO_SURFACE, g_ctx.context)) {
        fprintf(stderr, "GLES: failed to make context current (surfaceless): 0x%x\n", eglGetError());
        eglDestroyContext(g_ctx.display, g_ctx.context);
        eglTerminate(g_ctx.display);
        gbm_device_destroy(g_ctx.gbm);
        close(g_ctx.drm_fd);
        return -1;
    }

    /* Check compute shader support */
    GLint max_invocations = 0;
    glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &max_invocations);
    if (max_invocations == 0) {
        fprintf(stderr, "GLES: compute shaders not supported\n");
        eglMakeCurrent(g_ctx.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglDestroyContext(g_ctx.display, g_ctx.context);
        eglDestroySurface(g_ctx.display, g_ctx.surface);
        eglTerminate(g_ctx.display);
        return -1;
    }

    fprintf(stderr, "GLES: initialized (max compute invocations: %d)\n", max_invocations);
    return 0;
}

static int gles_compile_shader(void) {
    g_ctx.program = glCreateProgram();
    GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
    
    glShaderSource(cs, 1, &TERNARY_GEMM_SHADER, NULL);
    glCompileShader(cs);

    GLint compiled;
    glGetShaderiv(cs, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        char log[1024];
        glGetShaderInfoLog(cs, sizeof(log), NULL, log);
        fprintf(stderr, "GLES: shader compile failed: %s\n", log);
        glDeleteShader(cs);
        return -1;
    }

    glAttachShader(g_ctx.program, cs);
    glLinkProgram(g_ctx.program);

    GLint linked;
    glGetProgramiv(g_ctx.program, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[1024];
        glGetProgramInfoLog(g_ctx.program, sizeof(log), NULL, log);
        fprintf(stderr, "GLES: program link failed: %s\n", log);
        glDeleteShader(cs);
        glDeleteProgram(g_ctx.program);
        return -1;
    }

    glDeleteShader(cs);
    fprintf(stderr, "GLES: ternary GEMM shader compiled\n");
    return 0;
}

static void gles_ensure_buffers(size_t weights_bytes, size_t input_floats, size_t output_floats) {
    size_t weights_size = ((weights_bytes + 3) / 4) * 4;  /* Round up to uint alignment */
    size_t input_size = input_floats * sizeof(float);
    size_t output_size = output_floats * sizeof(float);

    if (g_ctx.ssbo_weights == 0 || g_ctx.weights_size < weights_size) {
        if (g_ctx.ssbo_weights) glDeleteBuffers(1, &g_ctx.ssbo_weights);
        glGenBuffers(1, &g_ctx.ssbo_weights);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, g_ctx.ssbo_weights);
        glBufferData(GL_SHADER_STORAGE_BUFFER, weights_size, NULL, GL_DYNAMIC_DRAW);
        g_ctx.weights_size = weights_size;
    }

    if (g_ctx.ssbo_input == 0 || g_ctx.input_size < input_size) {
        if (g_ctx.ssbo_input) glDeleteBuffers(1, &g_ctx.ssbo_input);
        glGenBuffers(1, &g_ctx.ssbo_input);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, g_ctx.ssbo_input);
        glBufferData(GL_SHADER_STORAGE_BUFFER, input_size, NULL, GL_DYNAMIC_DRAW);
        g_ctx.input_size = input_size;
    }

    if (g_ctx.ssbo_output == 0 || g_ctx.output_size < output_size) {
        if (g_ctx.ssbo_output) glDeleteBuffers(1, &g_ctx.ssbo_output);
        glGenBuffers(1, &g_ctx.ssbo_output);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, g_ctx.ssbo_output);
        glBufferData(GL_SHADER_STORAGE_BUFFER, output_size, NULL, GL_DYNAMIC_DRAW);
        g_ctx.output_size = output_size;
    }

    if (g_ctx.ssbo_uniforms == 0) {
        glGenBuffers(1, &g_ctx.ssbo_uniforms);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, g_ctx.ssbo_uniforms);
        glBufferData(GL_SHADER_STORAGE_BUFFER, 2 * sizeof(u32), NULL, GL_DYNAMIC_DRAW);
    }
}

static int gles_run_projection(const TernaryProj* proj) {
    u32 N = proj->N;
    u32 K = proj->K;
    size_t weights_bytes = (size_t)N * K / 4;

    gles_ensure_buffers(weights_bytes, K, N);

    /* Upload data */
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, g_ctx.ssbo_weights);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, weights_bytes, proj->weights);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, g_ctx.ssbo_input);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, K * sizeof(float), proj->input);

    u32 uniforms[2] = { N, K };
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, g_ctx.ssbo_uniforms);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uniforms), uniforms);

    /* Bind buffers */
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, g_ctx.ssbo_weights);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, g_ctx.ssbo_input);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, g_ctx.ssbo_output);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, g_ctx.ssbo_uniforms);

    /* Dispatch */
    glUseProgram(g_ctx.program);
    u32 workgroups = (N + 63) / 64;
    glDispatchCompute(workgroups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    /* Read back */
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, g_ctx.ssbo_output);
    void* map = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, N * sizeof(float), GL_MAP_READ_BIT);
    if (map) {
        memcpy(proj->output, map, N * sizeof(float));
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }

    return 0;
}

#endif /* HAS_GLES_COMPUTE */

/*============================================================================
 * CPU Fallback Context
 *============================================================================*/

typedef struct {
    bool active;
    u64 total_time_ns;
    u32 num_projections;
} CPUContext;

static CPUContext g_cpu = { .active = true };

/*============================================================================
 * Public API
 *============================================================================*/

int ternary_coproc_init(void) {
#ifdef HAS_GLES_COMPUTE
    if (gles_init_context() == 0 && gles_compile_shader() == 0) {
        g_ctx.active = true;
        fprintf(stderr, "V3D: ternary coprocessor initialized (GLES 3.1 compute)\n");
        return 0;
    }
    fprintf(stderr, "V3D: falling back to CPU\n");
#else
    fprintf(stderr, "V3D: GPU support not compiled (use -DHAS_GLES_COMPUTE -lGLESv2 -lEGL)\n");
#endif
    g_cpu.active = true;
    return 0;  /* CPU fallback always available */
}

void ternary_coproc_shutdown(void) {
#ifdef HAS_GLES_COMPUTE
    if (g_ctx.active) {
        if (g_ctx.ssbo_weights) glDeleteBuffers(1, &g_ctx.ssbo_weights);
        if (g_ctx.ssbo_input) glDeleteBuffers(1, &g_ctx.ssbo_input);
        if (g_ctx.ssbo_output) glDeleteBuffers(1, &g_ctx.ssbo_output);
        if (g_ctx.ssbo_uniforms) glDeleteBuffers(1, &g_ctx.ssbo_uniforms);
        if (g_ctx.program) glDeleteProgram(g_ctx.program);
        eglMakeCurrent(g_ctx.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglDestroyContext(g_ctx.display, g_ctx.context);
        if (g_ctx.surface != EGL_NO_SURFACE) {
            eglDestroySurface(g_ctx.display, g_ctx.surface);
        }
        eglTerminate(g_ctx.display);
        if (g_ctx.gbm) gbm_device_destroy(g_ctx.gbm);
        if (g_ctx.drm_fd >= 0) close(g_ctx.drm_fd);
        g_ctx.active = false;
    }
#endif
    g_cpu.active = false;
}

int ternary_coproc_available(void) {
#ifdef HAS_GLES_COMPUTE
    return g_ctx.active ? 1 : 0;
#else
    return 0;
#endif
}

int ternary_coproc_load_layer(uint32_t layer_idx,
                               const uint8_t* q_proj,
                               const uint8_t* k_proj,
                               const uint8_t* v_proj,
                               const uint8_t* o_proj,
                               const uint8_t* gate_proj,
                               const uint8_t* up_proj,
                               const uint8_t* down_proj) {
    /* TODO: Pre-upload weights to GPU for persistent storage */
    (void)layer_idx; (void)q_proj; (void)k_proj; (void)v_proj;
    (void)o_proj; (void)gate_proj; (void)up_proj; (void)down_proj;
    return 0;
}

void ternary_coproc_unload(void) {
    /* TODO: Free pre-uploaded weights */
}

int ternary_coproc_run_batch(const TernaryProj* projs, uint32_t count) {
    if (!projs || count == 0) return -1;

#ifdef HAS_GLES_COMPUTE
    if (g_ctx.active) {
        u64 start = ns_now();
        for (u32 i = 0; i < count; i++) {
            if (gles_run_projection(&projs[i]) != 0) {
                return -2;  /* GPU failed, caller should use CPU */
            }
        }
        u64 end = ns_now();
        g_ctx.total_time_ns += (end - start);
        g_ctx.num_projections += count;
        return 0;
    }
#endif

    return -1;  /* No GPU available */
}

int ternary_coproc_batch(const TernaryProj* projs, uint32_t count) {
    if (!projs || count == 0) return -1;

    /* Try GPU first */
    if (ternary_coproc_run_batch(projs, count) == 0) {
        return 0;
    }

    /* Fall back to CPU */
    u64 start = ns_now();
    for (u32 i = 0; i < count; i++) {
        hs_ml_ternary_f32_proj(projs[i].output,
                                projs[i].input,
                                projs[i].weights,
                                projs[i].N,
                                projs[i].K);
    }
    u64 end = ns_now();
    g_cpu.total_time_ns += (end - start);
    g_cpu.num_projections += count;

    return 0;
}

void ternary_coproc_get_stats(TernaryCoprocStats* stats) {
    if (!stats) return;

#ifdef HAS_GLES_COMPUTE
    if (g_ctx.active) {
        stats->total_time_ns = g_ctx.total_time_ns;
        stats->num_projections = g_ctx.num_projections;
        stats->num_qpus_used = 12;  /* V3D has 12 QPUs */
        return;
    }
#endif

    stats->total_time_ns = g_cpu.total_time_ns;
    stats->num_projections = g_cpu.num_projections;
    stats->num_qpus_used = 0;  /* CPU mode */
}

void ternary_coproc_reset_stats(void) {
#ifdef HAS_GLES_COMPUTE
    g_ctx.total_time_ns = 0;
    g_ctx.num_projections = 0;
#endif
    g_cpu.total_time_ns = 0;
    g_cpu.num_projections = 0;
}
