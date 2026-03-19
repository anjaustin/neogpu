/*
 * NeoGPU - Zero-Copy GPU Ternary GEMM
 *
 * Uses GL_EXT_buffer_storage for persistent mapped buffers.
 * CPU and GPU share the same physical memory - no copies.
 *
 * Architecture:
 *   1. Model weights loaded directly into persistent-mapped SSBOs
 *   2. Activation buffers (small) also persistent-mapped
 *   3. GPU dispatches read weights in-place, write outputs in-place
 *   4. Only synchronization needed, no data transfer
 *
 * Build: -DHAS_GLES_COMPUTE -lGLESv2 -lEGL -lgbm
 */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

typedef uint8_t  u8;
typedef uint32_t u32;
typedef uint64_t u64;

#ifdef HAS_GLES_COMPUTE

#include <fcntl.h>
#include <unistd.h>
#include <gbm.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>

/* GL_EXT_buffer_storage */
#ifndef GL_MAP_PERSISTENT_BIT_EXT
#define GL_MAP_PERSISTENT_BIT_EXT 0x0040
#endif
#ifndef GL_MAP_COHERENT_BIT_EXT
#define GL_MAP_COHERENT_BIT_EXT 0x0080
#endif
#ifndef GL_DYNAMIC_STORAGE_BIT_EXT
#define GL_DYNAMIC_STORAGE_BIT_EXT 0x0100
#endif
#ifndef GL_CLIENT_STORAGE_BIT_EXT
#define GL_CLIENT_STORAGE_BIT_EXT 0x0200
#endif

typedef void (*PFNGLBUFFERSTORAGEEXTPROC)(GLenum target, GLsizeiptr size, 
                                           const void* data, GLbitfield flags);
static PFNGLBUFFERSTORAGEEXTPROC glBufferStorageEXT = NULL;

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;

/*============================================================================
 * GPU Context
 *============================================================================*/

typedef struct {
    int drm_fd;
    struct gbm_device* gbm;
    EGLDisplay display;
    EGLContext context;
    GLuint program;        /* Single-layer shader */
    GLuint program_batch; /* Batched shader (multiple layers per dispatch) */
    bool initialized;
} GPUContext;

static GPUContext g_gpu = {0};

/* Persistent buffer - CPU and GPU share this memory */
typedef struct {
    GLuint ssbo;
    void* map;           /* CPU-accessible pointer */
    size_t size;
    bool valid;
} PersistentBuffer;

/* Weight buffer pool - one per layer to avoid re-upload */
#define MAX_LAYERS 64
typedef struct {
    PersistentBuffer weights[MAX_LAYERS];  /* Layer weights, ~16MB each */
    PersistentBuffer unified_weights;      /* All layers in one buffer (for batched) */
    PersistentBuffer input;                 /* Activation input, ~10KB */
    PersistentBuffer output;                /* Activation output, ~28KB */
    PersistentBuffer batch_output;         /* Batched output (num_layers * N) */
    PersistentBuffer uniforms;              /* Shader uniforms */
    PersistentBuffer lm_head;               /* LM head weights for vocab projection */
    u32 num_layers;
    u32 H, kv, F;                           /* Model dimensions */
    bool unified_available;
} BufferPool;

static BufferPool g_pool = {0};

/*============================================================================
 * Compute Shader - Single Layer
 *============================================================================*/

/*
 * Tiled GEMM shader for single layer projection.
 */
static const char* GEMM_SHADER =
    "#version 310 es\n"
    "precision highp float;\n"
    "precision highp int;\n"
    "\n"
    "#define TILE_N 4\n"
    "#define WG_SIZE 64\n"
    "\n"
    "layout(local_size_x = WG_SIZE) in;\n"
    "\n"
    "layout(std430, binding = 0) readonly buffer Weights { uint weights[]; };\n"
    "layout(std430, binding = 1) readonly buffer Input { float inputs[]; };\n"
    "layout(std430, binding = 2) writeonly buffer Output { float outputs[]; };\n"
    "layout(std430, binding = 3) readonly buffer Uniforms {\n"
    "    uint N; uint K; uint weight_offset; uint input_offset; uint output_offset;\n"
    "};\n"
    "\n"
    "shared float shared_input[2560];\n"
    "\n"
    "void main() {\n"
    "    uint lid = gl_LocalInvocationID.x;\n"
    "    uint gid = gl_WorkGroupID.x;\n"
    "    uint wg_size = uint(WG_SIZE);\n"
    "\n"
    "    for (uint i = lid; i < K; i += wg_size) {\n"
    "        shared_input[i] = inputs[input_offset + i];\n"
    "    }\n"
    "    memoryBarrierShared();\n"
    "    barrier();\n"
    "\n"
    "    uint base_n = gid * wg_size * uint(TILE_N) + lid * uint(TILE_N);\n"
    "    uint bytes_per_row = K / 4u;\n"
    "\n"
    "    float acc[TILE_N];\n"
    "    for (int t = 0; t < TILE_N; t++) acc[t] = 0.0;\n"
    "\n"
    "    for (uint k = 0u; k < K; k += 4u) {\n"
    "        float a0 = shared_input[k + 0u];\n"
    "        float a1 = shared_input[k + 1u];\n"
    "        float a2 = shared_input[k + 2u];\n"
    "        float a3 = shared_input[k + 3u];\n"
    "\n"
    "        for (int t = 0; t < TILE_N; t++) {\n"
    "            uint n = base_n + uint(t);\n"
    "            if (n >= N) continue;\n"
    "\n"
    "            uint abs_byte = weight_offset + n * bytes_per_row + k / 4u;\n"
    "            uint uint_idx = abs_byte / 4u;\n"
    "            uint byte_off = abs_byte & 3u;\n"
    "            uint wdata = weights[uint_idx];\n"
    "            uint b = (wdata >> (byte_off * 8u)) & 0xFFu;\n"
    "\n"
    "            uint c0 = b & 3u; uint c1 = (b >> 2u) & 3u;\n"
    "            uint c2 = (b >> 4u) & 3u; uint c3 = (b >> 6u) & 3u;\n"
    "            if (c0 == 2u) acc[t] += a0; else if (c0 == 0u) acc[t] -= a0;\n"
    "            if (c1 == 2u) acc[t] += a1; else if (c1 == 0u) acc[t] -= a1;\n"
    "            if (c2 == 2u) acc[t] += a2; else if (c2 == 0u) acc[t] -= a2;\n"
    "            if (c3 == 2u) acc[t] += a3; else if (c3 == 0u) acc[t] -= a3;\n"
    "        }\n"
    "    }\n"
    "\n"
    "    for (int t = 0; t < TILE_N; t++) {\n"
    "        uint n = base_n + uint(t);\n"
    "        if (n < N) outputs[output_offset + n] = acc[t];\n"
    "    }\n"
    "}\n";

/*============================================================================
 * Compute Shader - Batched (Multiple Layers Per Dispatch)
 *============================================================================*/

/*
 * Batched GEMM shader - one workgroup per layer.
 * Each workgroup processes one layer's projection.
 * 
 * Uses uniform variables instead of buffer block for simplicity.
 * Each workgroup gid = layer index
 */
static const char* GEMM_BATCH_SHADER =
    "#version 310 es\n"
    "precision highp float;\n"
    "precision highp int;\n"
    "\n"
    "#define TILE_N 4\n"
    "#define WG_SIZE 64\n"
    "#define MAX_LAYERS 32\n"
    "\n"
    "layout(local_size_x = WG_SIZE) in;\n"
    "\n"
    "layout(std430, binding = 0) readonly buffer Weights { uint weights[]; };\n"
    "layout(std430, binding = 1) readonly buffer Input { float inputs[]; };\n"
    "layout(std430, binding = 2) writeonly buffer Output { float outputs[]; };\n"
    "\n"
    "uniform uint uN;\n"
    "uniform uint uK;\n"
    "uniform uint uNumLayers;\n"
    "uniform uint uLayerOffsets[MAX_LAYERS];\n"
    "\n"
    "shared float shared_input[2560];\n"
    "\n"
    "void main() {\n"
    "    uint lid = gl_LocalInvocationID.x;\n"
    "    uint gid = gl_WorkGroupID.x;\n"
    "    uint wg_size = uint(WG_SIZE);\n"
    "\n"
    "    // Bounds check\n"
    "    if (gid >= uNumLayers) return;\n"
    "\n"
    "    // Get weight offset for this layer\n"
    "    uint weight_offset = uLayerOffsets[gid];\n"
    "\n"
    "    // Load input into shared memory\n"
    "    for (uint i = lid; i < uK; i += wg_size) {\n"
    "        shared_input[i] = inputs[i];\n"
    "    }\n"
    "    memoryBarrierShared();\n"
    "    barrier();\n"
    "\n"
    "    // Each thread computes TILE_N rows\n"
    "    uint base_n = gid * uint(TILE_N) * wg_size + lid * uint(TILE_N);\n"
    "    uint bytes_per_row = uK / 4u;\n"
    "\n"
    "    float acc[TILE_N];\n"
    "    for (int t = 0; t < TILE_N; t++) acc[t] = 0.0;\n"
    "\n"
    "    for (uint k = 0u; k < uK; k += 4u) {\n"
    "        float a0 = shared_input[k + 0u];\n"
    "        float a1 = shared_input[k + 1u];\n"
    "        float a2 = shared_input[k + 2u];\n"
    "        float a3 = shared_input[k + 3u];\n"
    "\n"
    "        for (int t = 0; t < TILE_N; t++) {\n"
    "            uint n = base_n + uint(t);\n"
    "            if (n >= uN) continue;\n"
    "\n"
    "            uint abs_byte = weight_offset + n * bytes_per_row + k / 4u;\n"
    "            uint uint_idx = abs_byte / 4u;\n"
    "            uint byte_off = abs_byte & 3u;\n"
    "            uint wdata = weights[uint_idx];\n"
    "            uint b = (wdata >> (byte_off * 8u)) & 0xFFu;\n"
    "\n"
    "            uint c0 = b & 3u; uint c1 = (b >> 2u) & 3u;\n"
    "            uint c2 = (b >> 4u) & 3u; uint c3 = (b >> 6u) & 3u;\n"
    "            if (c0 == 2u) acc[t] += a0; else if (c0 == 0u) acc[t] -= a0;\n"
    "            if (c1 == 2u) acc[t] += a1; else if (c1 == 0u) acc[t] -= a1;\n"
    "            if (c2 == 2u) acc[t] += a2; else if (c2 == 0u) acc[t] -= a2;\n"
    "            if (c3 == 2u) acc[t] += a3; else if (c3 == 0u) acc[t] -= a3;\n"
    "        }\n"
    "    }\n"
    "\n"
    "    // Write output at layer-specific offset\n"
    "    uint output_base = gid * uN;\n"
    "    for (int t = 0; t < TILE_N; t++) {\n"
    "        uint n = base_n + uint(t);\n"
    "        if (n < uN) outputs[output_base + n] = acc[t];\n"
    "    }\n"
    "}\n";

/*============================================================================
 * Initialization
 *============================================================================*/

static int init_egl(void) {
    g_gpu.drm_fd = open("/dev/dri/renderD128", O_RDWR);
    if (g_gpu.drm_fd < 0) {
        g_gpu.drm_fd = open("/dev/dri/card1", O_RDWR);
        if (g_gpu.drm_fd < 0) { fprintf(stderr, "GPU: failed to open DRM\n"); return -1; }
    }
    fprintf(stderr, "GPU: DRM opened, fd=%d\n", g_gpu.drm_fd);

    g_gpu.gbm = gbm_create_device(g_gpu.drm_fd);
    if (!g_gpu.gbm) { fprintf(stderr, "GPU: gbm_create_device failed\n"); return -1; }
    fprintf(stderr, "GPU: GBM created\n");

    g_gpu.display = eglGetPlatformDisplay(EGL_PLATFORM_GBM_MESA, g_gpu.gbm, NULL);
    if (g_gpu.display == EGL_NO_DISPLAY) { fprintf(stderr, "GPU: eglGetPlatformDisplay failed\n"); return -1; }
    fprintf(stderr, "GPU: EGL display obtained\n");

    if (!eglInitialize(g_gpu.display, NULL, NULL)) { fprintf(stderr, "GPU: eglInitialize failed\n"); return -1; }
    fprintf(stderr, "GPU: EGL initialized\n");

    EGLint cfg_attr[] = { EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_NONE };
    EGLConfig cfg; EGLint n;
    if (!eglChooseConfig(g_gpu.display, cfg_attr, &cfg, 1, &n) || n == 0) { fprintf(stderr, "GPU: eglChooseConfig failed\n"); return -1; }
    fprintf(stderr, "GPU: EGL config chosen\n");

    eglBindAPI(EGL_OPENGL_ES_API);
    EGLint ctx_attr[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
    g_gpu.context = eglCreateContext(g_gpu.display, cfg, EGL_NO_CONTEXT, ctx_attr);
    if (g_gpu.context == EGL_NO_CONTEXT) { fprintf(stderr, "GPU: eglCreateContext failed\n"); return -1; }
    fprintf(stderr, "GPU: EGL context created\n");

    if (!eglMakeCurrent(g_gpu.display, EGL_NO_SURFACE, EGL_NO_SURFACE, g_gpu.context)) { fprintf(stderr, "GPU: eglMakeCurrent failed\n"); return -1; }
    fprintf(stderr, "GPU: EGL context made current\n");

    /* Get buffer storage extension */
    glBufferStorageEXT = (PFNGLBUFFERSTORAGEEXTPROC)
        eglGetProcAddress("glBufferStorageEXT");
    if (!glBufferStorageEXT) {
        fprintf(stderr, "GPU: GL_EXT_buffer_storage not available\n");
        return -1;
    }
    fprintf(stderr, "GPU: GL_EXT_buffer_storage available\n");

    return 0;
}

static int compile_shader(void) {
    /* Compile single-layer shader */
    GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
    const char* src = GEMM_SHADER;
    glShaderSource(cs, 1, &src, NULL);
    glCompileShader(cs);

    GLint ok;
    glGetShaderiv(cs, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(cs, 512, NULL, log);
        fprintf(stderr, "GPU: shader compile error: %s\n", log);
        return -1;
    }

    g_gpu.program = glCreateProgram();
    glAttachShader(g_gpu.program, cs);
    glLinkProgram(g_gpu.program);
    glGetProgramiv(g_gpu.program, GL_LINK_STATUS, &ok);
    glDeleteShader(cs);

    if (!ok) {
        char log[512];
        glGetShaderInfoLog(g_gpu.program, 512, NULL, log);
        fprintf(stderr, "GPU: shader link error: %s\n", log);
        return -1;
    }

    /* Compile batched shader (multiple layers per dispatch) */
    cs = glCreateShader(GL_COMPUTE_SHADER);
    src = GEMM_BATCH_SHADER;
    glShaderSource(cs, 1, &src, NULL);
    glCompileShader(cs);

    glGetShaderiv(cs, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(cs, 512, NULL, log);
        fprintf(stderr, "GPU: batched shader compile error: %s\n", log);
        return -1;
    }

    g_gpu.program_batch = glCreateProgram();
    glAttachShader(g_gpu.program_batch, cs);
    glLinkProgram(g_gpu.program_batch);
    glGetProgramiv(g_gpu.program_batch, GL_LINK_STATUS, &ok);
    glDeleteShader(cs);

    if (!ok) {
        char log[512];
        glGetShaderInfoLog(g_gpu.program_batch, 512, NULL, log);
        fprintf(stderr, "GPU: batched shader link error: %s\n", log);
        return -1;
    }

    return 0;
}

/*============================================================================
 * Persistent Buffer Management
 *============================================================================*/

static int create_persistent_buffer(PersistentBuffer* buf, size_t size, GLenum usage) {
    glGenBuffers(1, &buf->ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf->ssbo);

    /* First try with persistent + coherent mapping (fastest) */
    GLbitfield flags = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT |
                       GL_MAP_PERSISTENT_BIT_EXT | GL_MAP_COHERENT_BIT_EXT;
    glBufferStorageEXT(GL_SHADER_STORAGE_BUFFER, size, NULL, flags);

    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        fprintf(stderr, "GPU: persistent buffer failed (0x%x), trying regular\n", err);
        
        /* Fall back to regular buffer (still DMA-accessible) */
        glBufferData(GL_SHADER_STORAGE_BUFFER, size, NULL, usage);
        err = glGetError();
        if (err != GL_NO_ERROR) {
            fprintf(stderr, "GPU: regular buffer also failed (0x%x)\n", err);
            return -1;
        }
        
        /* Map it once for CPU access */
        buf->map = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, size, 
                                    GL_MAP_READ_BIT | GL_MAP_WRITE_BIT);
        if (!buf->map) {
            fprintf(stderr, "GPU: buffer map failed\n");
            return -1;
        }
        fprintf(stderr, "GPU: regular buffer OK, mapped at %p\n", buf->map);
    } else {
        /* Persistent mapping succeeded */
        buf->map = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, size, flags);
        if (!buf->map) {
            fprintf(stderr, "GPU: persistent map failed\n");
            return -1;
        }
        fprintf(stderr, "GPU: persistent buffer OK, mapped at %p\n", buf->map);
    }

    buf->size = size;
    buf->valid = true;
    return 0;
}

static void destroy_persistent_buffer(PersistentBuffer* buf) {
    if (buf->valid) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf->ssbo);
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        glDeleteBuffers(1, &buf->ssbo);
        buf->valid = false;
    }
}

/*============================================================================
 * Public API
 *============================================================================*/

int gpu_gemm_available(void) {
#ifdef HAS_GLES_COMPUTE
    return g_gpu.initialized ? 1 : 0;
#else
    return 0;
#endif
}

int gpu_gemm_init(void) {
    if (g_gpu.initialized) return 0;

    if (init_egl() != 0) {
        fprintf(stderr, "GPU: EGL init failed\n");
        return -1;
    }

    if (compile_shader() != 0) {
        fprintf(stderr, "GPU: shader compile failed\n");
        return -1;
    }

    /* Create uniform buffer - need 36 u32s for batched shader (4 header + 32 layer offsets) */
    if (create_persistent_buffer(&g_pool.uniforms, 36 * sizeof(u32), GL_DYNAMIC_DRAW) != 0)
        return -1;

    g_gpu.initialized = true;
    fprintf(stderr, "GPU: zero-copy GEMM initialized\n");
    return 0;
}

void gpu_gemm_shutdown(void) {
    if (!g_gpu.initialized) return;

    for (u32 i = 0; i < g_pool.num_layers; i++)
        destroy_persistent_buffer(&g_pool.weights[i]);
    destroy_persistent_buffer(&g_pool.input);
    destroy_persistent_buffer(&g_pool.output);
    destroy_persistent_buffer(&g_pool.batch_output);
    destroy_persistent_buffer(&g_pool.uniforms);
    destroy_persistent_buffer(&g_pool.unified_weights);

    if (g_gpu.program) glDeleteProgram(g_gpu.program);
    if (g_gpu.program_batch) glDeleteProgram(g_gpu.program_batch);
    eglMakeCurrent(g_gpu.display, EGL_NO_SURFACE, EGL_NO_SURFACE, g_gpu.context);
    eglDestroyContext(g_gpu.display, g_gpu.context);
    eglTerminate(g_gpu.display);
    if (g_gpu.gbm) gbm_device_destroy(g_gpu.gbm);
    if (g_gpu.drm_fd >= 0) close(g_gpu.drm_fd);

    memset(&g_gpu, 0, sizeof(g_gpu));
    memset(&g_pool, 0, sizeof(g_pool));
}

/*
 * Allocate model weight buffer in shared memory.
 * Returns CPU pointer where caller should load weights.
 * GPU will read from same memory with zero copy.
 */
void* gpu_gemm_alloc_weights(u32 layer_idx, size_t bytes) {
    if (!g_gpu.initialized || layer_idx >= MAX_LAYERS) return NULL;

    PersistentBuffer* buf = &g_pool.weights[layer_idx];
    if (buf->valid && buf->size >= bytes) {
        return buf->map;  /* Reuse existing buffer */
    }

    destroy_persistent_buffer(buf);
    if (create_persistent_buffer(buf, bytes, GL_STATIC_DRAW) != 0)
        return NULL;

    if (layer_idx >= g_pool.num_layers)
        g_pool.num_layers = layer_idx + 1;

    return buf->map;
}

/*
 * Set model dimensions (needed for computing offsets).
 */
void gpu_gemm_set_dims(u32 H, u32 kv, u32 F) {
    g_pool.H = H;
    g_pool.kv = kv;
    g_pool.F = F;

    /* Allocate activation buffers sized for largest projections */
    size_t max_input = (H > F ? H : F) * sizeof(float);
    size_t max_output = (H > F ? H : F) * sizeof(float);

    fprintf(stderr, "GPU: set_dims: max_input=%zu max_output=%zu\n", max_input, max_output);
    fflush(stderr);

    if (!g_pool.input.valid || g_pool.input.size < max_input) {
        fprintf(stderr, "GPU: creating input buffer\n");
        fflush(stderr);
        destroy_persistent_buffer(&g_pool.input);
        create_persistent_buffer(&g_pool.input, max_input, GL_DYNAMIC_DRAW);
    }
    fprintf(stderr, "GPU: input buffer done\n");
    fflush(stderr);
    if (!g_pool.output.valid || g_pool.output.size < max_output) {
        fprintf(stderr, "GPU: creating output buffer\n");
        fflush(stderr);
        destroy_persistent_buffer(&g_pool.output);
        create_persistent_buffer(&g_pool.output, max_output, GL_DYNAMIC_DRAW);
    }
    fprintf(stderr, "GPU: output buffer done\n");
    fflush(stderr);
}

/*
 * Run a single projection using unified weight buffer.
 * Input/output are copied to/from persistent buffers.
 */
int gpu_gemm_run(u32 layer_idx, u32 weight_offset,
                 const float* input, float* output,
                 u32 N, u32 K) {
    if (!g_gpu.initialized) return -1;
    
    /* Use unified buffer if available, otherwise fall back to per-layer */
    bool use_unified = g_pool.unified_available && g_pool.unified_weights.valid;
    if (!use_unified) {
        fprintf(stderr, "GPU: unified buffer not available!\n");
        return -1;
    }
    GLuint weight_ssbo = g_pool.unified_weights.ssbo;
    fprintf(stderr, "GPU: running layer %u weight_offset=%u N=%u K=%u\n", 
            layer_idx, weight_offset, N, K);

    /* GPU path - execute compute shader */
    /* Copy input to persistent buffer (small: K floats = 2.5-10KB) */
    memcpy(g_pool.input.map, input, K * sizeof(float));

    /* Set uniforms */
    u32* u = (u32*)g_pool.uniforms.map;
    u[0] = N;
    u[1] = K;
    u[2] = weight_offset;  /* Byte offset into layer weight buffer */
    u[3] = 0;              /* Input offset (always 0, we copy exact input) */
    u[4] = 0;              /* Output offset */

    /* Bind buffers */
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weight_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, g_pool.input.ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, g_pool.output.ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, g_pool.uniforms.ssbo);

    /* Dispatch - each thread handles TILE_N rows, workgroup is 64 threads */
    glUseProgram(g_gpu.program);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    u32 rows_per_wg = 64 * 4;  /* WG_SIZE * TILE_N */
    glDispatchCompute((N + rows_per_wg - 1) / rows_per_wg, 1, 1);
    
    /* Don't sync here - let pipeline buffer. Copy will wait if needed. */

    /* Copy output from persistent buffer */
    memcpy(output, g_pool.output.map, N * sizeof(float));

    return 0;
}

/*
 * Sync after batched dispatches - call after gpu_gemm_run_nosync.
 */
void gpu_gemm_sync(void) {
    if (!g_gpu.initialized) return;
    glFinish();
}

/*
 * Allocate unified weight buffer for batched processing.
 * Uses regular GL buffer (fallback from CMA).
 */
int gpu_gemm_alloc_unified_weights(size_t total_bytes) {
    if (!g_gpu.initialized) return -1;
    
    destroy_persistent_buffer(&g_pool.unified_weights);
    
    if (create_persistent_buffer(&g_pool.unified_weights, total_bytes, GL_STATIC_DRAW) != 0) {
        fprintf(stderr, "GPU: failed to allocate unified weights buffer (%zu MB)\n", 
                total_bytes / 1024 / 1024);
        g_pool.unified_available = false;
        return -1;
    }
    
    g_pool.unified_available = true;
    fprintf(stderr, "GPU: allocated unified weights buffer (%zu MB)\n",
            total_bytes / 1024 / 1024);
    fflush(stderr);
    return 0;
}

/*
 * Get pointer to unified weight buffer for loading weights.
 * Call this after gpu_gemm_alloc_unified_weights succeeds.
 */
void* gpu_gemm_get_unified_weight_ptr(void) {
    if (!g_pool.unified_available || !g_pool.unified_weights.valid) {
        return NULL;
    }
    return g_pool.unified_weights.map;
}

/*
 * Get direct pointer to input/output buffers for benchmarking.
 */
void gpu_gemm_get_buffer_ptrs(void** input, void** output) {
    if (input) *input = g_pool.input.valid ? g_pool.input.map : NULL;
    if (output) *output = g_pool.output.valid ? g_pool.output.map : NULL;
}

/*
 * Run Q projection for ALL layers in ONE dispatch.
 * Each workgroup handles one layer.
 * 
 * input: [K] input activations (same for all layers)
 * outputs: [num_layers][N] output for each layer
 * layer_offsets: byte offset for each layer's Q weights in unified buffer
 * N: output dimension per layer
 * K: input dimension
 * num_layers: number of layers
 */
int gpu_gemm_run_all_layers_q(const float* input, 
                             float** outputs,
                             const u32* layer_offsets,
                             u32 N, u32 K, 
                             u32 num_layers) {
    if (!g_gpu.initialized) return -1;
    if (!g_pool.unified_available || !g_pool.unified_weights.valid) return -1;
    if (num_layers > 32) return -1;  /* MAX_LAYERS limit in shader */
    
    /* Ensure batch output buffer is large enough */
    size_t batch_output_size = (size_t)num_layers * N * sizeof(float);
    if (!g_pool.batch_output.valid || g_pool.batch_output.size < batch_output_size) {
        destroy_persistent_buffer(&g_pool.batch_output);
        if (create_persistent_buffer(&g_pool.batch_output, batch_output_size, GL_DYNAMIC_DRAW) != 0) {
            return -1;
        }
    }
    
    /* Copy input to persistent buffer */
    memcpy(g_pool.input.map, input, K * sizeof(float));
    
    /* Set up uniforms using uniform variables */
    glUseProgram(g_gpu.program_batch);
    glUniform1ui(glGetUniformLocation(g_gpu.program_batch, "uN"), N);
    glUniform1ui(glGetUniformLocation(g_gpu.program_batch, "uK"), K);
    glUniform1ui(glGetUniformLocation(g_gpu.program_batch, "uNumLayers"), num_layers);
    glUniform1uiv(glGetUniformLocation(g_gpu.program_batch, "uLayerOffsets"), num_layers, layer_offsets);
    
    /* Bind buffers */
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, g_pool.unified_weights.ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, g_pool.input.ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, g_pool.batch_output.ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, g_pool.uniforms.ssbo);
    
    /* Dispatch - one workgroup per layer */
    glUseProgram(g_gpu.program_batch);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glDispatchCompute(num_layers, 1, 1);
    
    /* Wait for completion */
    glFinish();
    
    /* Copy outputs */
    float* out_base = (float*)g_pool.batch_output.map;
    for (u32 l = 0; l < num_layers; l++) {
        memcpy(outputs[l], out_base + (size_t)l * N, N * sizeof(float));
    }
    
    return 0;
}

/*
 * Run multiple projections using unified weight buffer.
 * Projections are defined by (weight_offset, N, K) tuples.
 * All share the same input, outputs are written to separate regions.
 *
 * Key optimization: uses unified buffer (one SSBO bind) instead of per-layer.
 */
int gpu_gemm_run_batch(u32 layer_idx,
                       const float* input, u32 input_K,
                       float* const* outputs,
                       const u32* weight_offsets,
                       const u32* Ns, const u32* Ks,
                       u32 count) {
    if (!g_gpu.initialized) return -1;
    if (count == 0) return 0;
    
    /* Use unified buffer if available, otherwise fall back to per-layer */
    bool use_unified = g_pool.unified_available && g_pool.unified_weights.valid;
    GLuint weight_ssbo = 0;
    if (use_unified) {
        weight_ssbo = g_pool.unified_weights.ssbo;
    } else {
        if (layer_idx >= g_pool.num_layers) return -1;
        if (!g_pool.weights[layer_idx].valid) return -1;
        weight_ssbo = g_pool.weights[layer_idx].ssbo;
    }

    /* Calculate total output size and offsets */
    u32 output_offsets[16];  /* Max batch size */
    u32 total_output = 0;
    for (u32 i = 0; i < count && i < 16; i++) {
        output_offsets[i] = total_output;
        total_output += Ns[i];
    }

    /* Ensure output buffer is large enough */
    size_t needed = total_output * sizeof(float);
    if (!g_pool.output.valid || g_pool.output.size < needed) {
        destroy_persistent_buffer(&g_pool.output);
        if (create_persistent_buffer(&g_pool.output, needed, GL_DYNAMIC_DRAW) != 0)
            return -1;
    }

    /* Copy input once */
    memcpy(g_pool.input.map, input, input_K * sizeof(float));

    glUseProgram(g_gpu.program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weight_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, g_pool.input.ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, g_pool.output.ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, g_pool.uniforms.ssbo);

    /* Dispatch all shaders without waiting - let them run in parallel */
    for (u32 i = 0; i < count; i++) {
        u32* u = (u32*)g_pool.uniforms.map;
        u[0] = Ns[i];
        u[1] = Ks[i];
        u[2] = weight_offsets[i];
        u[3] = 0;                    /* Input offset always 0 */
        u[4] = output_offsets[i];    /* Output offset for this projection */

        u32 rows_per_wg = 64 * 4;  /* WG_SIZE * TILE_N */
        glDispatchCompute((Ns[i] + rows_per_wg - 1) / rows_per_wg, 1, 1);
    }

    /* Single sync point after all dispatches */
    glFinish();

    /* Copy all outputs */
    float* out_base = (float*)g_pool.output.map;
    for (u32 i = 0; i < count; i++) {
        memcpy(outputs[i], out_base + output_offsets[i], Ns[i] * sizeof(float));
    }

    return 0;
}

/*
 * Layer-level attention: Q, K, V, O projections in one batch.
 * Input: layer input [H]
 * Outputs: q [H], k [kv], v [kv], o_out [H] (uses attn_out as input)
 */
int gpu_gemm_run_attn(uint32_t layer_idx,
                      const float* input,
                      float* q_out, float* k_out, float* v_out,
                      const float* attn_out, float* o_out) {
    if (!g_gpu.initialized) return -1;
    if (layer_idx >= g_pool.num_layers) return -1;
    if (!g_pool.weights[layer_idx].valid) return -1;

    const uint32_t H = g_pool.H;
    const uint32_t kv = g_pool.kv;

    /* QKV share same input, batch them together */
    float* outputs_qkv[3] = { q_out, k_out, v_out };
    uint32_t offsets_qkv[3] = {
        g_pool.weights[layer_idx].valid ? 0 : 0,
        g_pool.weights[layer_idx].valid ? 
            (uint32_t)((H * H / 4)) : 0,
        g_pool.weights[layer_idx].valid ? 
            (uint32_t)((H * H / 4) + (kv * H / 4)) : 0
    };
    uint32_t Ns_qkv[3] = { H, kv, kv };
    uint32_t Ks_qkv[3] = { H, H, H };

    /* Copy input to GPU */
    memcpy(g_pool.input.map, input, H * sizeof(float));

    /* Bind buffers once */
    glUseProgram(g_gpu.program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, g_pool.weights[layer_idx].ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, g_pool.input.ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, g_pool.output.ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, g_pool.uniforms.ssbo);

    /* Allocate output space */
    size_t output_size = (H + kv + kv) * sizeof(float);
    if (g_pool.output.size < output_size) {
        destroy_persistent_buffer(&g_pool.output);
        create_persistent_buffer(&g_pool.output, output_size, GL_DYNAMIC_DRAW);
    }

    /* Dispatch QKV */
    uint32_t output_offsets[3] = { 0, H, H + kv };
    for (int i = 0; i < 3; i++) {
        uint32_t* u = (uint32_t*)g_pool.uniforms.map;
        u[0] = Ns_qkv[i];
        u[1] = Ks_qkv[i];
        u[2] = offsets_qkv[i];
        u[3] = 0;
        u[4] = output_offsets[i];
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        uint32_t rows_per_wg = 64 * 4;
        glDispatchCompute((Ns_qkv[i] + rows_per_wg - 1) / rows_per_wg, 1, 1);
    }

    /* Copy QKV outputs back */
    float* out_base = (float*)g_pool.output.map;
    memcpy(q_out, out_base, H * sizeof(float));
    memcpy(k_out, out_base + H, kv * sizeof(float));
    memcpy(v_out, out_base + H + kv, kv * sizeof(float));

    /* O projection uses attn_out as input */
    memcpy(g_pool.input.map, attn_out, H * sizeof(float));

    uint32_t o_offset = (uint32_t)((H * H / 4) + 2 * (kv * H / 4));
    uint32_t* u = (uint32_t*)g_pool.uniforms.map;
    u[0] = H;
    u[1] = H;
    u[2] = o_offset;
    u[3] = 0;
    u[4] = 0;
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    uint32_t rows_per_wg = 64 * 4;
    glDispatchCompute((H + rows_per_wg - 1) / rows_per_wg, 1, 1);
    glFinish();

    memcpy(o_out, g_pool.output.map, H * sizeof(float));

    return 0;
}

/*
 * Layer-level FFN: gate, up, down projections in one batch.
 */
int gpu_gemm_run_ffn(uint32_t layer_idx,
                     const float* input,
                     float* gate_out, float* up_out,
                     const float* ffn_in, float* down_out) {
    if (!g_gpu.initialized) return -1;
    if (layer_idx >= g_pool.num_layers) return -1;
    if (!g_pool.weights[layer_idx].valid) return -1;

    const uint32_t H = g_pool.H;
    const uint32_t F = g_pool.F;

    /* Gate and Up share input */
    memcpy(g_pool.input.map, input, H * sizeof(float));

    glUseProgram(g_gpu.program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, g_pool.weights[layer_idx].ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, g_pool.input.ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, g_pool.output.ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, g_pool.uniforms.ssbo);

    size_t output_size = (F + F) * sizeof(float);
    if (g_pool.output.size < output_size) {
        destroy_persistent_buffer(&g_pool.output);
        create_persistent_buffer(&g_pool.output, output_size, GL_DYNAMIC_DRAW);
    }

    /* Gate offset: after q,k,v,o = H*H/4 + 2*kv*H/4 + H*H/4 = H*H/4 * 2 + kv*H/4 * 2 */
    uint32_t gate_offset = (uint32_t)(2 * (H * H / 4) + 2 * (g_pool.kv * H / 4));
    uint32_t up_offset = gate_offset + (uint32_t)(F * H / 4);

    uint32_t output_offsets[2] = { 0, F };
    float* outputs[2] = { gate_out, up_out };
    uint32_t Ns[2] = { F, F };
    uint32_t Ks[2] = { H, H };
    uint32_t offsets[2] = { gate_offset, up_offset };

    for (int i = 0; i < 2; i++) {
        uint32_t* u = (uint32_t*)g_pool.uniforms.map;
        u[0] = Ns[i];
        u[1] = Ks[i];
        u[2] = offsets[i];
        u[3] = 0;
        u[4] = output_offsets[i];
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        uint32_t rows_per_wg = 64 * 4;
        glDispatchCompute((Ns[i] + rows_per_wg - 1) / rows_per_wg, 1, 1);
    }

    glFinish();

    float* out_base = (float*)g_pool.output.map;
    memcpy(gate_out, out_base, F * sizeof(float));
    memcpy(up_out, out_base + F, F * sizeof(float));

    /* Down projection uses ffn_in */
    memcpy(g_pool.input.map, ffn_in, F * sizeof(float));

    uint32_t down_offset = up_offset + (uint32_t)(F * H / 4);
    uint32_t* u = (uint32_t*)g_pool.uniforms.map;
    u[0] = H;
    u[1] = F;
    u[2] = down_offset;
    u[3] = 0;
    u[4] = 0;
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    uint32_t rows_per_wg = 64 * 4;
    glDispatchCompute((H + rows_per_wg - 1) / rows_per_wg, 1, 1);
    glFinish();

    memcpy(down_out, g_pool.output.map, H * sizeof(float));

    return 0;
}

/*
 * Allocate lm_head weights buffer in GPU memory.
 */
void* gpu_gemm_alloc_lmhead(size_t bytes) {
    if (!g_gpu.initialized) return NULL;
    
    destroy_persistent_buffer(&g_pool.lm_head);
    
    if (create_persistent_buffer(&g_pool.lm_head, bytes, GL_STATIC_DRAW) != 0) {
        fprintf(stderr, "GPU: failed to allocate lm_head buffer (%zu MB)\n", 
                bytes / 1024 / 1024);
        return NULL;
    }
    
    return g_pool.lm_head.map;
}

/*
 * Run lm_head projection on GPU.
 * This is a large GEMV: [V, H] x [H] -> [V]
 */
int gpu_gemm_run_lmhead(const float* input, float* output, u32 V, u32 K) {
    if (!g_gpu.initialized) return -1;
    if (!g_pool.lm_head.valid) return -1;
    
    fprintf(stderr, "GPU lm_head: V=%u K=%u\n", V, K);
    /* Copy input to GPU */
    memcpy(g_pool.input.map, input, K * sizeof(float));
    fprintf(stderr, "GPU lm_head: input copied\n");
    
    /* Set uniforms */
    u32* u = (u32*)g_pool.uniforms.map;
    u[0] = V;      /* N = vocab size */
    u[1] = K;       /* K = hidden size */
    u[2] = 0;       /* weight offset = 0 */
    u[3] = 0;       /* input offset */
    u[4] = 0;       /* output offset */
    fprintf(stderr, "GPU lm_head: uniforms set\n");
    
    /* Bind buffers */
    glUseProgram(g_gpu.program);
    fprintf(stderr, "GPU lm_head: program bound\n");
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, g_pool.lm_head.ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, g_pool.input.ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, g_pool.output.ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, g_pool.uniforms.ssbo);
    fprintf(stderr, "GPU lm_head: buffers bound\n");
    
    /* Dispatch - lm_head is large, many workgroups */
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    u32 rows_per_wg = 64 * 4;
    u32 dispatch_size = (V + rows_per_wg - 1) / rows_per_wg;
    fprintf(stderr, "GPU lm_head: dispatching %u workgroups\n", dispatch_size);
    glDispatchCompute(dispatch_size, 1, 1);
    fprintf(stderr, "GPU lm_head: compute dispatched\n");
    
    /* Wait and copy output */
    glFinish();
    fprintf(stderr, "GPU lm_head: finished\n");
    memcpy(output, g_pool.output.map, V * sizeof(float));
    fprintf(stderr, "GPU lm_head: output copied\n");
    
    return 0;
}

#else /* !HAS_GLES_COMPUTE */

int gpu_gemm_available(void) { return 0; }
int gpu_gemm_init(void) { return -1; }
void gpu_gemm_shutdown(void) {}
void* gpu_gemm_alloc_weights(u32 layer_idx, size_t bytes) { 
    (void)layer_idx; (void)bytes; return NULL; 
}
void gpu_gemm_set_dims(u32 H, u32 kv, u32 F) { (void)H; (void)kv; (void)F; }
int gpu_gemm_run(u32 layer_idx, u32 weight_offset,
                 const float* input, float* output,
                 u32 N, u32 K) {
    (void)layer_idx; (void)weight_offset; (void)input; (void)output;
    (void)N; (void)K;
    return -1;
}
int gpu_gemm_run_batch(u32 layer_idx,
                       const float* input, u32 input_K,
                       float* const* outputs,
                       const u32* weight_offsets,
                       const u32* Ns, const u32* Ks,
                       u32 count) {
    (void)layer_idx; (void)input; (void)input_K; (void)outputs;
    (void)weight_offsets; (void)Ns; (void)Ks; (void)count;
    return -1;
}

int gpu_gemm_run_attn(u32 layer_idx,
                      const float* input,
                      float* q_out, float* k_out, float* v_out,
                      const float* attn_out, float* o_out) {
    (void)layer_idx; (void)input; (void)q_out; (void)k_out; (void)v_out;
    (void)attn_out; (void)o_out;
    return -1;
}

int gpu_gemm_run_ffn(u32 layer_idx,
                     const float* input,
                     float* gate_out, float* up_out,
                     const float* ffn_in, float* down_out) {
    (void)layer_idx; (void)input; (void)gate_out; (void)up_out;
    (void)ffn_in; (void)down_out;
    return -1;
}

int gpu_gemm_alloc_unified_weights(size_t total_bytes) {
    (void)total_bytes;
    return -1;
}

void* gpu_gemm_get_unified_weight_ptr(void) {
    return NULL;
}

void* gpu_gemm_alloc_lmhead(size_t bytes) {
    (void)bytes;
    return NULL;
}

int gpu_gemm_run_lmhead(const float* input, float* output, u32 V, u32 K) {
    (void)input; (void)output; (void)V; (void)K;
    return -1;
}

int gpu_gemm_run_all_layers_q(const float* input, 
                               float** outputs,
                               const u32* layer_offsets,
                               u32 N, u32 K, 
                               u32 num_layers) {
    (void)input; (void)outputs; (void)layer_offsets;
    (void)N; (void)K; (void)num_layers;
    return -1;
}

#endif /* HAS_GLES_COMPUTE */
