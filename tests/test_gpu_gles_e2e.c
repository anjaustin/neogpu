/* Reuse the existing schema/FIFO/mock observer, replacing only its backend.
 * The uncalled original main stays available as frame_mock_main; the real
 * benchmark below uses the same native/reference admission and list checks. */
#define main frame_mock_main
#include "test_gpu_frame_benchmark.c"
#undef main
#include "hs_backend_gles.h"
#include "hs_graphics.h"
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>

typedef struct { u64 attempt, admitted, completed; } ContributionTime;
typedef struct { u64 submit, total, cpu; } RealFrameTime;
static ContributionTime* stamps[HS_FABRIC_PARTICIPANTS];
static RealFrameTime timings[65536];
static HSBackend real_backend;
static bool validating, snapshot;
static const char* prefix;
static GLint width, height;
static u32 timed_frames;
static u64 submitted_at;

static u64 cpu_ns(void) {
    struct timespec t; REQUIRE(clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t) == 0);
    return (u64)t.tv_sec * 1000000000ull + t.tv_nsec;
}
static int descriptors(void) {
    DIR* directory = opendir("/proc/self/fd"); REQUIRE(directory);
    struct dirent* entry; int count = 0, own = dirfd(directory);
    while ((entry = readdir(directory)))
        if (entry->d_name[0] != '.' && atoi(entry->d_name) != own) count++;
    REQUIRE(closedir(directory) == 0); return count;
}
static void report_driver_affinity(void) {
    DIR* directory = opendir("/proc/self/task"); REQUIRE(directory);
    struct dirent* entry;
    while ((entry = readdir(directory))) {
        if (entry->d_name[0] == '.') continue;
        int tid = atoi(entry->d_name); cpu_set_t set;
        if (sched_getaffinity(tid, sizeof(set), &set)) { REQUIRE(errno == ESRCH); continue; }
        printf("{\"record\":\"driver_thread_affinity\",\"tid\":%d,\"cpus\":[", tid);
        bool first = true;
        for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) if (CPU_ISSET(cpu, &set)) {
            printf("%s%d", first ? "" : ",", cpu); first = false;
        }
        printf("]}\n");
    }
    REQUIRE(closedir(directory) == 0);
}
extern EGLBoolean __real_eglSwapBuffers(EGLDisplay, EGLSurface);
EGLBoolean __wrap_eglSwapBuffers(EGLDisplay display, EGLSurface surface) {
    submitted_at = ns();
    if (snapshot) {
        size_t bytes = (size_t)width * height * 4;
        u8* pixels = malloc(bytes); REQUIRE(pixels);
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        REQUIRE(glGetError() == GL_NO_ERROR);
        for (u32 lane = 0; lane < producers; lane++) {
            u32 x = (u32)width * (2 * lane + 1) / 16;
            const u8* pixel = pixels + ((size_t)(height / 2) * width + x) * 4;
            REQUIRE(pixel[0] == (lane % 2 ? 0 : 255));
            REQUIRE(pixel[1] == (lane % 2 ? 255 : 0));
            REQUIRE(pixel[2] == 0 && pixel[3] == 255);
        }
        char filename[512]; REQUIRE(snprintf(filename, sizeof(filename), "%s.rgba", prefix) < (int)sizeof(filename));
        FILE* file = fopen(filename, "wb"); REQUIRE(file);
        REQUIRE(fwrite(pixels, 1, bytes, file) == bytes && fclose(file) == 0);
        free(pixels);
    }
    EGLBoolean result = __real_eglSwapBuffers(display, surface);
    REQUIRE(result == EGL_TRUE); return result;
}
static void execute_real(void* context, const HSFrameContext* frame) {
    if (validating) execute(context, frame);
    else backend_calls++;
    real_backend.ops->execute(context, frame);
}
static void geometry_real(u32 lane) {
    HSBuffer* buffer = hs_gpu_get_buffer(&gpu, (u8)lane);
    REQUIRE(buffer && hs_buffer_init(buffer, (u8)lane, 60));
    f32 x = -0.875f + 0.25f * lane, red = lane % 2 == 0, green = lane % 2 != 0;
    const f32 data[15] = {x, 0.7f, red, green, 0, x - 0.10f, -0.7f, red, green, 0,
                         x + 0.10f, -0.7f, red, green, 0};
    for (u32 i = 0; i < 15; i++) hs_buffer_set_f32(buffer, i, data[i]);
    buffer->length = sizeof(data);
}
static void begin_real(void) {
    hs_gpu_begin_frame(&gpu);
    hs_gpu_clear(&gpu, v4_make(0, 0, 0, 1));
    /* hs_gpu_process deliberately drains participant frames as well. Pump
     * only the legacy CLEAR here; checked end-frame owns bounded service. */
    while (hs_step(&gpu.system)) {}
    REQUIRE(gpu.render.count == 1);
}
static void frame_real(void) {
    begin_real();
    u32 added = 0;
    if (is_reference) { comparison_rigtorp.step(reference); added = append_manual(); }
    end_frame();
    if (is_reference) { REQUIRE(gpu.render.count == added + 1); gpu.render.count = 1; }
    REQUIRE(gpu.render.count == 1);
}
static void* produce_real(void* pointer) {
    Producer* producer = pointer;
    pin(1 + producer->id % 3);
    atomic_fetch_add_explicit(&ready, 1, memory_order_release);
    while (!atomic_load_explicit(&go, memory_order_acquire)) sched_yield();
    for (u32 sequence = 1; sequence <= frames; sequence++) {
        HSParticipantFrame input = {.sequence = sequence, .count = draws};
        for (u32 i = 0; i < draws; i++) input.draws[i] = (HSRenderCmd){.op = HS_RC_DRAW, .a = (u8)producer->id};
        stamps[producer->id][sequence].attempt = ns();
        while (!submit(producer, &input)) { producer->retries++; sched_yield(); }
        stamps[producer->id][sequence].admitted = ns(); producer->accepted++;
    }
    return NULL;
}
int main(int argc, char** argv) {
    REQUIRE(argc == 7); alarm(180);
    mode = argv[1]; is_reference = !strcmp(mode, "reference");
    REQUIRE(is_reference || !strcmp(mode, "native"));
    producers = (u32)strtoul(argv[2], NULL, 10); frames = (u32)strtoul(argv[3], NULL, 10);
    draws = (u32)strtoul(argv[4], NULL, 10);
    REQUIRE(producers && producers <= 8 && frames >= 3 && frames <= 512 && (draws == 1 || draws == 64));
    bool recording = !strcmp(argv[5], "on"); REQUIRE(recording || !strcmp(argv[5], "off"));
    prefix = argv[6]; REQUIRE(strncmp(prefix, "tests/evidence-", 15) == 0);
    int initial_fds = descriptors(); pin(0);
    u64 init_start = ns(), init_cpu_start = cpu_ns();
    hs_gpu_init(&gpu); gpu.system.recording = recording; REQUIRE(hs_gpu_participants_ready(&gpu));
    if (is_reference) { reference = comparison_rigtorp.create(producers); REQUIRE(reference); }
    for (u32 lane = 0; lane < producers; lane++) {
        workers[lane].id = lane; geometry_real(lane);
        stamps[lane] = calloc(frames + 1, sizeof(**stamps)); REQUIRE(stamps[lane]);
        if (!is_reference) REQUIRE(join(&workers[lane].owner) == HS_FABRIC_OK && workers[lane].owner.slot == lane);
    }
    real_backend = hs_backend_gles_create();
    HSBackendOps observed_ops = *real_backend.ops; observed_ops.execute = execute_real;
    HSBackend observed_backend = real_backend; observed_backend.ops = &observed_ops;
    hs_gpu_attach_backend(&gpu, &observed_backend);
    REQUIRE(eglGetCurrentContext() != EGL_NO_CONTEXT && gpu.backend);
    int graphics_fd = ((HSGraphics*)observed_backend.ctx)->drm_fd;
    GLint viewport[4]; glGetIntegerv(GL_VIEWPORT, viewport); width = viewport[2]; height = viewport[3];
    REQUIRE(width > 0 && height > 0);
    printf("{\"record\":\"context\",\"width\":%d,\"height\":%d,\"renderer\":\"%s\",\"version\":\"%s\"}\n",
           width, height, glGetString(GL_RENDERER), glGetString(GL_VERSION));
    u64 init_ns = ns() - init_start, init_cpu_ns = cpu_ns() - init_cpu_start, cold_ns = 0;
    /* Both paths pre-upload the same geometry, without admitting contributions. */
    for (u32 warm = 0; warm < 9; warm++) {
        u64 start = ns(); begin_real();
        for (u32 lane = 0; lane < producers; lane++)
            for (u32 d = 0; d < draws; d++) {
                HSRenderCmd cmd = {.op = HS_RC_DRAW, .a = (u8)lane}; REQUIRE(hs_render_push(&gpu.render, &cmd));
            }
        end_frame(); glFinish(); if (!warm) cold_ns = ns() - start;
        REQUIRE(glGetError() == GL_NO_ERROR);
    }
    report_driver_affinity();
    hs_gpu_clear_log(&gpu); backend_calls = 0; validating = true;
    pthread_t threads[HS_FABRIC_PARTICIPANTS]; u64 observed[HS_FABRIC_PARTICIPANTS] = {0};
    atomic_init(&go, false); atomic_init(&ready, 0);
    for (u32 lane = 0; lane < producers; lane++) REQUIRE(pthread_create(&threads[lane], NULL, produce_real, &workers[lane]) == 0);
    while (atomic_load_explicit(&ready, memory_order_acquire) < producers) sched_yield();
    u64 applied = 0, accepted = 0, retries = 0, start = ns(), cpu_start = cpu_ns();
    atomic_store_explicit(&go, true, memory_order_release);
    while (applied < (u64)frames * producers) {
        REQUIRE(timed_frames < 65536 && gpu.system.log_head < HS_MAX_MSG_LOG - 1);
        u64 frame_start = ns(), frame_cpu = cpu_ns(); frame_real();
        u64 returned = ns(); glFinish(); u64 completed = ns();
        timings[timed_frames++] = (RealFrameTime){submitted_at - frame_start, completed - frame_start, cpu_ns() - frame_cpu};
        REQUIRE(returned >= submitted_at && glGetError() == GL_NO_ERROR);
        for (u32 lane = 0; lane < producers; lane++) {
            const HSParticipantFrame* current = view(lane);
            if (!current || current->sequence == observed[lane]) continue;
            REQUIRE(current->sequence == observed[lane] + 1 && current->count == draws && current->sequence <= frames);
            observed[lane]++; applied++; stamps[lane][current->sequence].completed = completed;
        }
    }
    for (u32 lane = 0; lane < producers; lane++) {
        REQUIRE(pthread_join(threads[lane], NULL) == 0 && observed[lane] == frames);
        accepted += workers[lane].accepted; retries += workers[lane].retries;
    }
    u64 elapsed_ns = ns() - start, process_cpu_ns = cpu_ns() - cpu_start;
    REQUIRE(accepted == applied && backend_calls == timed_frames);
    REQUIRE(gpu.system.log_head == (recording ? timed_frames : 0));
    for (u32 i = 0; i < gpu.system.log_head; i++) REQUIRE(gpu.log_buffer[i].op == OP_CLEAR);
    snapshot = true; frame_real(); glFinish(); snapshot = false;
    for (u32 lane = 0; lane < producers; lane++) {
        for (u32 seq = 1; seq <= frames; seq++) {
            ContributionTime t = stamps[lane][seq];
            REQUIRE(t.attempt >= start && t.admitted >= t.attempt && t.completed >= t.attempt);
            printf("{\"record\":\"contribution\",\"lane\":%u,\"sequence\":%u,\"attempt_ns\":%" PRIu64 ",\"admitted_ns\":%" PRIu64 ",\"completed_ns\":%" PRIu64 "}\n",
                   lane, seq, t.attempt-start, t.admitted-start, t.completed-start);
        }
        free(stamps[lane]);
    }
    for (u32 i = 0; i < timed_frames; i++)
        printf("{\"record\":\"frame\",\"index\":%u,\"submit_ns\":%" PRIu64 ",\"completed_ns\":%" PRIu64 ",\"cpu_ns\":%" PRIu64 "}\n",
               i, timings[i].submit, timings[i].total, timings[i].cpu);
    u64 shutdown_start = ns(), shutdown_cpu = cpu_ns();
    if (reference) { comparison_rigtorp.destroy(reference); reference = NULL; }
    hs_gpu_shutdown(&gpu);
    u64 shutdown_ns = ns()-shutdown_start, shutdown_cpu_ns = cpu_ns()-shutdown_cpu;
    errno = 0; REQUIRE(fcntl(graphics_fd, F_GETFD) == -1 && errno == EBADF);
    REQUIRE(descriptors() == initial_fds && eglGetCurrentContext() == EGL_NO_CONTEXT);
    printf("{\"record\":\"summary\",\"mode\":\"%s\",\"producers\":%u,\"draws\":%u,\"frames_per_producer\":%u,\"capture\":\"%s\",\"accepted\":%" PRIu64 ",\"applied\":%" PRIu64 ",\"retries\":%" PRIu64 ",\"rendered_frames\":%u,\"elapsed_ns\":%" PRIu64 ",\"process_cpu_ns\":%" PRIu64 ",\"init_ns\":%" PRIu64 ",\"init_cpu_ns\":%" PRIu64 ",\"cold_ns\":%" PRIu64 ",\"shutdown_ns\":%" PRIu64 ",\"shutdown_cpu_ns\":%" PRIu64 ",\"resources_released\":true}\n",
           mode, producers, draws, frames, recording ? "on" : "off", accepted, applied, retries, timed_frames,
           elapsed_ns, process_cpu_ns, init_ns, init_cpu_ns, cold_ns, shutdown_ns, shutdown_cpu_ns);
    return 0;
}
