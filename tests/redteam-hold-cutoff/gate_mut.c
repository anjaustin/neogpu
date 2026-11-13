/* Mutation copy of tests/test_present_pacing.c: env HOLD_PERMILLE overrides gfx.hold_permille after init (hold-cutoff red team). */
/* Hardware check: with vsync off, a GPU-heavy frame rate still puts a new
 * frame on screen at every refresh.
 *
 * A flip lands only once the GPU has finished its frame. When frames take
 * about half a refresh on the GPU (~2 rendered per refresh, as in
 * test_08_raycast_orbit), flipping the frame just submitted, which renders
 * behind frames that will be dropped, misses the next refresh and repeats the
 * old image on about a third of refreshes: visible stutter at full render
 * rate. hs_graphics_present flips the newest finished frame instead.
 *
 * The shader cost is calibrated (with glFinish) to about 0.55 of a refresh
 * per frame, then frames are rendered uncapped. Contract: every frame is
 * shown or dropped, at most 1% of refreshes repeat the previous frame, and
 * hs_graphics_flush_present() shows a held frame and leaves room for the
 * caller's own eglSwapBuffers.
 *
 * Then frames of ~0.30 refresh, below hold_permille: they are never held
 * (the newest frame is flipped and the kernel waits for its render), and at
 * most 1% of refreshes repeat.
 *
 * Then frames of ~0.95 refresh (~64 FPS): a flip must be queued 4-7 ms
 * before a vblank, and without waiting for a flip about to land ~12% of
 * refreshes repeated. Contract: at most 1% repeat, at no frame-rate cost
 * (>= 99% of the same frames presented without the wait, measured first),
 * and the wait is used there but not
 * at ~0.55 refresh, where it measured only added latency.
 *
 * Then the ~0.55-refresh frames again with every CPU busy (one spinning
 * process per core). The kernel's commit work for a flip runs in an unbound
 * kworker and includes firmware clock calls; at normal priority their
 * wakeups waited for a CPU and 6-60 of ~330 refreshes repeated. With those
 * workers at SCHED_FIFO (tools/neogpu_kworker_rt, `sudo make
 * install-kworker-rt`) the contract is at most 1% repeats under that load.
 * Without it the load phase is still measured and reported, and
 * "load_checked" is false.
 *
 * Usage: sudo tests/test_present_pacing [frames]
 * Output: one JSON object; exit 0 only if the run met its contract.
 */
#include "hs_graphics.h"
#include <ctype.h>
#include <dirent.h>
#include <sched.h>
#include <signal.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>


static const char* VS =
    "attribute vec2 p;\n"
    "void main() { gl_Position = vec4(p, 0.0, 1.0); }\n";
static const char* FS =
    "precision highp float;\n"
    "uniform int n;\n"
    "uniform float t;\n"
    "void main() {\n"
    "    vec2 q = gl_FragCoord.xy * 0.01 + t;\n"
    "    float a = 0.0;\n"
    "    for (int i = 0; i < 4096; i++) {\n"
    "        if (i >= n) break;\n"
    "        a += sin(q.x + float(i) * 0.37) * cos(q.y - a);\n"
    "    }\n"
    "    gl_FragColor = vec4(fract(a), fract(t), 0.5, 1.0);\n"
    "}\n";

/* True when every default-pool unbound kworker ("kworker/u...", nice 0) runs
 * SCHED_FIFO, as tools/neogpu_kworker_rt keeps them. */
static bool unbound_workers_rt(void) {
    DIR* d = opendir("/proc");
    if (!d) return false;
    int workers = 0, rt = 0;
    struct dirent* e;
    while ((e = readdir(d))) {
        if (!isdigit((unsigned char)e->d_name[0])) continue;
        char path[300], buf[1024];
        snprintf(path, sizeof(path), "/proc/%s/stat", e->d_name);
        FILE* f = fopen(path, "r");
        if (!f) continue;
        size_t n = fread(buf, 1, sizeof(buf) - 1, f);
        fclose(f);
        buf[n] = '\0';
        if (!strstr(buf, "(kworker/u")) continue;
        char* p = strrchr(buf, ')');
        long field = 2, nice = -1, policy = -1;
        for (char* tok = p ? strtok(p + 1, " ") : NULL; tok; tok = strtok(NULL, " ")) {
            field++;
            if (field == 19) nice = strtol(tok, NULL, 10);
            if (field == 41) { policy = strtol(tok, NULL, 10); break; }
        }
        if (nice != 0) continue;
        workers++;
        rt += policy == SCHED_FIFO;
    }
    closedir(d);
    return workers > 0 && rt == workers;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static GLuint shader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    return ok ? s : 0;
}

static void draw(GLint n_loc, GLint t_loc, int n, float t) {
    glUniform1i(n_loc, n);
    glUniform1f(t_loc, t);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

/* Shader iterations for want_ms (within +-tolerance) of GPU time per frame. */
static int calibrate(GLint n_loc, GLint t_loc, double want, double tolerance, double* frame_ms) {
    int n = 8;
    for (int round = 0; round < 12; round++) {
        glFinish();
        double t0 = now_ms();
        for (int i = 0; i < 4; i++) draw(n_loc, t_loc, n, (float)i);
        glFinish();
        *frame_ms = (now_ms() - t0) / 4;
        if (*frame_ms > want * (1 - tolerance) && *frame_ms < want * (1 + tolerance)) break;
        int next = (int)(n * want / (*frame_ms > 0.01 ? *frame_ms : 0.01));
        n = next < 1 ? 1 : next > 4096 ? 4096 : next;
    }
    return n;
}

int main(int argc, char** argv) {
    int frames = argc > 1 ? atoi(argv[1]) : 600;
    static HSGraphics gfx;
    if (frames < 60 || hs_graphics_init(&gfx) != 0 || (getenv("HOLD_PERMILLE") && (gfx.hold_permille = (u32)strtoul(getenv("HOLD_PERMILLE"), NULL, 10), 0))) {
        printf("{\"init\":false}\n");
        return 1;
    }
    GLuint vs = shader(GL_VERTEX_SHADER, VS), fs = shader(GL_FRAGMENT_SHADER, FS);
    GLuint prog = glCreateProgram();
    if (vs && fs) {
        glAttachShader(prog, vs);
        glAttachShader(prog, fs);
        glBindAttribLocation(prog, 0, "p");
        glLinkProgram(prog);
    }
    GLint linked = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    if (!linked) {
        printf("{\"shader\":false}\n");
        hs_graphics_finish(&gfx);
        return 1;
    }
    static const float quad[] = {-1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1};
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glUseProgram(prog);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    GLint n_loc = glGetUniformLocation(prog, "n"), t_loc = glGetUniformLocation(prog, "t");

    /* Take the display first so calibration runs against the real path. */
    draw(n_loc, t_loc, 1, 0);
    hs_graphics_present(&gfx);
    double refresh_ms = gfx.display_mode.vrefresh ? 1000.0 / gfx.display_mode.vrefresh : 1000.0 / 60;

    /* Calibrate the per-frame GPU time to ~0.55 refresh. */
    double frame_ms = 0;
    int n = calibrate(n_loc, t_loc, 0.55 * refresh_ms, 0.1, &frame_ms);

    /* One paced flip gives the starting vblank count (takeover has none). */
    gfx.vsync = true;
    draw(n_loc, t_loc, 1, 0);
    hs_graphics_present(&gfx);
    gfx.vsync = false;
    u64 presented0 = gfx.frames_presented, dropped0 = gfx.frames_dropped;
    u64 repeated0 = gfx.refreshes_repeated;
    u32 seq0 = gfx.last_flip_sequence;
    double start = now_ms();
    for (int i = 0; i < frames; i++) {
        draw(n_loc, t_loc, n, (float)i * 0.01f);
        hs_graphics_present(&gfx);
    }
    double elapsed = now_ms() - start;
    u64 fast_waits = gfx.flip_waits; /* ~0.55 refresh: below the wait gate */
    /* Keep presenting until a frame is held, flush it (it must be shown, not
     * dropped), then swap the surface directly as a caller mixing raw EGL
     * with hs_graphics may. */
    int extra = 0;
    while (!gfx.held_bo && extra < 64) {
        draw(n_loc, t_loc, n, 0);
        hs_graphics_present(&gfx);
        extra++;
    }
    bool held = gfx.held_bo != NULL;
    u64 owed = (u64)gfx.flip_pending + held;
    u64 presented_before = gfx.frames_presented, dropped_before = gfx.frames_dropped;
    hs_graphics_flush_present(&gfx);
    bool flush_shows = gfx.frames_presented == presented_before + owed && gfx.frames_dropped == dropped_before &&
                       !gfx.held_bo && !gfx.flip_pending;
    u64 presented = gfx.frames_presented - presented0, dropped = gfx.frames_dropped - dropped0;
    u64 repeated = gfx.refreshes_repeated - repeated0;
    u32 refreshes = gfx.last_flip_sequence - seq0;
    /* Each flip covers its own refresh plus the ones that repeated before it. */
    bool counted = refreshes == presented + repeated;
    /* The repeat counter itself: a flip after ~4 idle refreshes adds >= 2. */
    usleep(60000);
    u64 repeated_idle = gfx.refreshes_repeated;
    gfx.vsync = true;
    draw(n_loc, t_loc, 1, 0);
    hs_graphics_present(&gfx);
    gfx.vsync = false;
    counted = counted && gfx.refreshes_repeated >= repeated_idle + 2;
    bool raw_swaps = true;
    for (int i = 0; i < 3; i++) {
        draw(n_loc, t_loc, 1, 0);
        raw_swaps = raw_swaps && eglSwapBuffers(gfx.egl_display, gfx.egl_surface) == EGL_TRUE;
    }
    /* Cheap frames (far below hold_permille) finish in time: they are
     * dropped while a flip is in flight, never held. */
    bool cheap_unheld = true;
    for (int i = 0; i < 200; i++) {
        draw(n_loc, t_loc, 1, 0);
        hs_graphics_present(&gfx);
        if (i >= 32) cheap_unheld = cheap_unheld && !gfx.held_bo;
    }
    /* Frames between the cheap ones and hold_permille: not held. */
    double mid_ms = 0;
    int mid_n = calibrate(n_loc, t_loc, 0.30 * refresh_ms, 0.08, &mid_ms);
    for (int i = 0; i < 32; i++) { /* settle the smoothed present interval */
        draw(n_loc, t_loc, mid_n, 0);
        hs_graphics_present(&gfx);
    }
    bool mid_unheld = mid_ms * 1000 < refresh_ms * gfx.hold_permille;
    u64 mid_rep0 = gfx.refreshes_repeated;
    u32 mid_seq0 = gfx.last_flip_sequence;
    for (int i = 0; i < 600; i++) {
        draw(n_loc, t_loc, mid_n, (float)i * 0.01f);
        hs_graphics_present(&gfx);
        mid_unheld = mid_unheld && !gfx.held_bo;
    }
    u64 mid_repeated = gfx.refreshes_repeated - mid_rep0;
    u32 mid_refreshes = gfx.last_flip_sequence - mid_seq0;
    bool mid_ok = mid_unheld && mid_refreshes > 120 && mid_repeated * 100 <= mid_refreshes;
    /* Slow frames: ~0.95 refresh each, uncapped. */
    double slow_ms = 0;
    int slow_n = calibrate(n_loc, t_loc, 0.95 * refresh_ms, 0.04, &slow_ms);
    for (int i = 0; i < 16; i++) { /* settle the smoothed present interval */
        draw(n_loc, t_loc, slow_n, 0);
        hs_graphics_present(&gfx);
    }
    /* Baseline without the wait (it needs the last flip's timestamp). */
    u64 base_rep0 = gfx.refreshes_repeated;
    u32 base_seq0 = gfx.last_flip_sequence;
    double base_start = now_ms();
    for (int i = 0; i < 240; i++) {
        draw(n_loc, t_loc, slow_n, (float)i * 0.01f);
        gfx.last_flip_ns = 0;
        hs_graphics_present(&gfx);
    }
    double base_fps = 240 * 1000.0 / (now_ms() - base_start);
    u64 base_repeated = gfx.refreshes_repeated - base_rep0;
    u32 base_refreshes = gfx.last_flip_sequence - base_seq0;
    for (int i = 0; i < 16; i++) { /* a flip timestamp again */
        draw(n_loc, t_loc, slow_n, 0);
        hs_graphics_present(&gfx);
    }
    u64 slow_rep0 = gfx.refreshes_repeated, slow_waits0 = gfx.flip_waits;
    u32 slow_seq0 = gfx.last_flip_sequence;
    double slow_start = now_ms();
    for (int i = 0; i < 240; i++) {
        draw(n_loc, t_loc, slow_n, (float)i * 0.01f);
        hs_graphics_present(&gfx);
    }
    double slow_fps = 240 * 1000.0 / (now_ms() - slow_start);
    u64 slow_waits = gfx.flip_waits - slow_waits0;
    u64 slow_repeated = gfx.refreshes_repeated - slow_rep0;
    u32 slow_refreshes = gfx.last_flip_sequence - slow_seq0;
    /* Waiting for the flip must not cost frame rate: the GPU stays busy. */
    bool slow_ok = slow_refreshes > 120 && slow_repeated * 100 <= slow_refreshes && slow_waits > 0 &&
                   slow_fps >= 0.99 * base_fps;
    /* Every CPU busy, from separate processes: spinning threads in this
     * process measured weaker (1-78 vs 7-38 repeats without the commit
     * thread). */
    long cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpus < 1) cpus = 1;
    if (cpus > 64) cpus = 64;
    pid_t spin_pids[64];
    int spun = 0;
    bool workers_rt = unbound_workers_rt();
    for (long c = 0; c < cpus; c++) {
        pid_t pid = fork();
        if (pid == 0) {
            prctl(PR_SET_PDEATHSIG, SIGKILL); /* never outlive the test */
            if (getppid() == 1) _exit(0);
            for (;;) {} /* killed below */
        }
        if (pid > 0) spin_pids[spun++] = pid;
    }
    usleep(200000);
    for (int i = 0; i < 16; i++) {
        draw(n_loc, t_loc, n, 0);
        hs_graphics_present(&gfx);
    }
    u64 load_rep0 = gfx.refreshes_repeated;
    u32 load_seq0 = gfx.last_flip_sequence;
    for (int i = 0; i < 400; i++) {
        draw(n_loc, t_loc, n, (float)i * 0.01f);
        hs_graphics_present(&gfx);
    }
    u64 load_repeated = gfx.refreshes_repeated - load_rep0;
    u32 load_refreshes = gfx.last_flip_sequence - load_seq0;
    for (int i = 0; i < spun; i++) {
        kill(spin_pids[i], SIGKILL);
        waitpid(spin_pids[i], NULL, 0);
    }
    bool load_checked = workers_rt && unbound_workers_rt();
    bool load_ok = !load_checked || (spun == cpus && load_refreshes > 100 && load_repeated * 100 <= load_refreshes);
    bool active = gfx.crtc_active;
    hs_graphics_finish(&gfx);
    double fps = frames * 1000.0 / elapsed;
    bool ok = active && gfx.present_errors == 0 && presented + dropped == (u64)(frames + extra) && held && raw_swaps && counted && cheap_unheld && mid_ok && slow_ok && load_ok && fast_waits == 0 &&
              flush_shows && fps > 1.5 * 1000.0 / refresh_ms && seq0 && refreshes > 0 && repeated * 100 <= refreshes;
    printf("{\"frames\":%d,\"iterations\":%d,\"gpu_frame_ms\":%.2f,\"refresh_ms\":%.2f,\"fps\":%.1f,"
           "\"presented\":%llu,\"dropped\":%llu,\"refreshes\":%u,\"repeated\":%llu,\"errors\":%u,\"held\":%s,\"raw_swaps\":%s,\"counted\":%s,\"cheap_unheld\":%s,\"flush_owed\":%llu,\"flush_shows\":%s,"
           "\"mid_gpu_frame_ms\":%.2f,\"mid_unheld\":%s,\"mid_refreshes\":%u,\"mid_repeated\":%llu,"
           "\"fast_waits\":%llu,\"slow_gpu_frame_ms\":%.2f,\"slow_fps\":%.1f,\"slow_refreshes\":%u,"
           "\"slow_repeated\":%llu,\"slow_waits\":%llu,\"no_wait_fps\":%.1f,\"no_wait_refreshes\":%u,"
           "\"no_wait_repeated\":%llu,\"load_checked\":%s,\"load_procs\":%d,\"load_refreshes\":%u,"
           "\"load_repeated\":%llu,\"passed\":%s}\n",
           frames, n, frame_ms, refresh_ms, fps, (unsigned long long)presented, (unsigned long long)dropped,
           refreshes, (unsigned long long)repeated, gfx.present_errors, held ? "true" : "false",
           raw_swaps ? "true" : "false", counted ? "true" : "false",
           cheap_unheld ? "true" : "false", (unsigned long long)owed,
           flush_shows ? "true" : "false", mid_ms, mid_unheld ? "true" : "false", mid_refreshes,
           (unsigned long long)mid_repeated, (unsigned long long)fast_waits, slow_ms, slow_fps, slow_refreshes,
           (unsigned long long)slow_repeated, (unsigned long long)slow_waits, base_fps, base_refreshes,
           (unsigned long long)base_repeated, load_checked ? "true" : "false", spun, load_refreshes,
           (unsigned long long)load_repeated, ok ? "true" : "false");
    return ok ? 0 : 1;
}
