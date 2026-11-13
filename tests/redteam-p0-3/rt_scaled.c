/* Red-team probe for hs_graphics_init_scaled (P0-3).
 * Usage:
 *   rt_scaled run W H secs [work=0] [native=0]   init_scaled (native=1: hs_graphics_init) + uncapped presents
 *   rt_scaled props                                print primary-plane state on the DSI CRTC (fresh fd)
 *   rt_scaled setprops alpha filter rotation blend set those on the plane fbcon shows, then exit
 * One JSON line per run. */
#define HS_GRAPHICS_SHELVED_SCALED /* P0-3: shelved scaled path */
#include "hs_graphics.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

static unsigned long long underruns(void) {
    unsigned long long v = ~0ull;
    FILE* f = fopen("/sys/kernel/debug/dri/1/hvs_underrun", "r");
    if (f) { if (fscanf(f, "%llu", &v) != 1) v = ~0ull; fclose(f); }
    return v;
}

static uint64_t pv(int fd, uint32_t obj, const char* n) { uint64_t v = ~0ull; hs_graphics_prop(fd, obj, DRM_MODE_OBJECT_PLANE, n, &v); return v; }

/* JSON fragment for the planes on crtc (0 = the CRTC fbcon/DSI uses: first active). */
static void plane_json(int fd, uint32_t crtc, char* out, size_t n) {
    drmSetClientCap(fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1);
    drmSetClientCap(fd, DRM_CLIENT_CAP_ATOMIC, 1);
    drmModePlaneRes* pr = drmModeGetPlaneResources(fd);
    size_t l = 0;
    int on = 0;
    l += snprintf(out + l, n - l, "\"planes\":[");
    for (uint32_t i = 0; pr && i < pr->count_planes; i++) {
        uint32_t id = pr->planes[i];
        uint64_t c = pv(fd, id, "CRTC_ID");
        if (!c || (crtc && c != crtc)) continue;
        uint64_t fbid = pv(fd, id, "FB_ID");
        uint64_t mod = ~0ull;
        drmModeFB2* fb2 = drmModeGetFB2(fd, (uint32_t)fbid);
        if (fb2) { mod = fb2->modifier; drmModeFreeFB2(fb2); }
        l += snprintf(out + l, n - l,
            "%s{\"id\":%u,\"crtc\":%llu,\"type\":%llu,\"src\":[%.4f,%.4f,%.4f,%.4f],\"dst\":[%lld,%lld,%llu,%llu],"
            "\"alpha\":%llu,\"filter\":%llu,\"rotation\":%llu,\"blend\":%llu,\"fb\":%llu,\"mod\":\"0x%llx\"}",
            on++ ? "," : "", id, (unsigned long long)c, (unsigned long long)pv(fd, id, "type"),
            pv(fd, id, "SRC_X") / 65536.0, pv(fd, id, "SRC_Y") / 65536.0, pv(fd, id, "SRC_W") / 65536.0, pv(fd, id, "SRC_H") / 65536.0,
            (long long)(int32_t)pv(fd, id, "CRTC_X"), (long long)(int32_t)pv(fd, id, "CRTC_Y"),
            (unsigned long long)pv(fd, id, "CRTC_W"), (unsigned long long)pv(fd, id, "CRTC_H"),
            (unsigned long long)pv(fd, id, "alpha"), (unsigned long long)pv(fd, id, "SCALING_FILTER"),
            (unsigned long long)pv(fd, id, "rotation"), (unsigned long long)pv(fd, id, "pixel blend mode"),
            (unsigned long long)fbid, (unsigned long long)mod);
    }
    l += snprintf(out + l, n - l, "]");
    drmModeFreePlaneResources(pr);
}

static int fresh_fd(void) {
    int fd = open("/dev/dri/card1", O_RDWR | O_CLOEXEC);
    if (fd < 0) fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    return fd;
}

static GLuint shader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[512]; glGetShaderInfoLog(s, 512, NULL, log); fprintf(stderr, "shader: %s\n", log); }
    return s;
}

static int do_run(int argc, char** argv) {
    u32 w = (u32)atoi(argv[2]), h = (u32)atoi(argv[3]);
    double secs = atof(argv[4]);
    int work = argc > 5 ? atoi(argv[5]) : 0, native = argc > 6 ? atoi(argv[6]) : 0;
    static HSGraphics gfx;
    unsigned long long u0 = underruns();
    errno = 0;
    int rc = native ? hs_graphics_init(&gfx) : hs_graphics_init_scaled(&gfx, w, h);
    if (rc != 0) {
        int fd = fresh_fd(); static char pj[8192]; plane_json(fd, 0, pj, sizeof pj); close(fd);
        printf("{\"render\":[%u,%u],\"init\":false,\"errno\":%d,%s}\n", w, h, errno, pj);
        return 2;
    }
    if (native) { w = gfx.screen_width; h = gfx.screen_height; }
    char fs[1024];
    snprintf(fs, sizeof fs,
        "precision highp float;\nuniform vec4 c;\nvoid main() {\n"
        " float v = gl_FragCoord.x * 0.001 + gl_FragCoord.y * 0.002;\n"
        " for (int i = 0; i < %d; i++) v = fract(sin(v * 1.37 + 0.11) * 43758.5);\n"
        " gl_FragColor = c + vec4(v * 0.001);\n}\n", work);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, shader(GL_VERTEX_SHADER, "attribute vec2 p;\nvoid main() { gl_Position = vec4(p, 0.0, 1.0); }\n"));
    glAttachShader(prog, shader(GL_FRAGMENT_SHADER, fs));
    glBindAttribLocation(prog, 0, "p");
    glLinkProgram(prog);
    glUseProgram(prog);
    GLint cl = glGetUniformLocation(prog, "c");
    static const float quad[] = {-1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1};
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, quad);
    glViewport(0, 0, (GLsizei)w, (GLsizei)h);

    /* first present = takeover */
    glUniform4f(cl, 0.1f, 0.1f, 0.1f, 1);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    double t0 = now_ms();
    hs_graphics_present(&gfx);
    double takeover_ms = now_ms() - t0;
    const char* takeover_err = gfx.present_error;
    int takeover_errno = gfx.last_present_errno;
    static char pj_take[8192];
    plane_json(gfx.drm_fd, gfx.display_crtc_id, pj_take, sizeof pj_take);
    gfx.vsync = true; hs_graphics_present(&gfx); gfx.vsync = false;
    u64 rep0 = gfx.refreshes_repeated, pres0 = gfx.frames_presented, drop0 = gfx.frames_dropped, fw0 = gfx.flip_waits;
    u32 seq0 = gfx.last_flip_sequence;
    double start = now_ms();
    long n = 0;
    while (now_ms() - start < secs * 1000) {
        glUniform4f(cl, (n & 1) ? 0.8f : 0.2f, 0.3f, 0.5f, 1);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        hs_graphics_present(&gfx);
        n++;
    }
    double el = now_ms() - start;
    hs_graphics_flush_present(&gfx);
    u32 refreshes = gfx.last_flip_sequence - seq0;
    u64 presented = gfx.frames_presented - pres0;
    static char pj_end[8192];
    plane_json(gfx.drm_fd, gfx.display_crtc_id, pj_end, sizeof pj_end);
    uint32_t crtc = gfx.display_crtc_id;
    int active = gfx.crtc_active, scaled = gfx.scaled_scanout;
    u32 errs = gfx.present_errors;
    const char* why = gfx.present_error;
    u64 rep = gfx.refreshes_repeated - rep0, drop = gfx.frames_dropped - drop0, fw = gfx.flip_waits - fw0;
    hs_graphics_finish(&gfx);
    usleep(100000);
    int fd = fresh_fd(); static char pj_con[8192]; plane_json(fd, crtc, pj_con, sizeof pj_con); close(fd);
    unsigned long long u1 = underruns();
    printf("{\"render\":[%u,%u],\"native\":%d,\"work\":%d,\"init\":true,\"scaled\":%d,\"active\":%d,\"takeover_ms\":%.1f,"
           "\"takeover_error\":\"%s\",\"takeover_errno\":%d,\"errors\":%u,\"error\":\"%s\",\"fps\":%.1f,\"presents\":%ld,"
           "\"presented\":%llu,\"dropped\":%llu,\"refreshes\":%u,\"repeated\":%llu,\"repeat_pct\":%.2f,\"flip_waits\":%llu,"
           "\"underruns\":%lld,\"at_takeover\":{%s},\"at_end\":{%s},\"console\":{%s}}\n",
           w, h, native, work, scaled, active, takeover_ms, takeover_err ? takeover_err : "", takeover_errno, errs,
           why ? why : "", n * 1000.0 / el, n, (unsigned long long)presented, (unsigned long long)drop, refreshes,
           (unsigned long long)rep, refreshes ? 100.0 * rep / refreshes : 0, (unsigned long long)fw,
           (long long)(u1 - u0), pj_take, pj_end, pj_con);
    return 0;
}

static int do_setprops(int argc, char** argv) {
    int fd = fresh_fd();
    if (drmSetMaster(fd)) { printf("{\"error\":\"master\"}\n"); return 1; }
    drmSetClientCap(fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1);
    drmSetClientCap(fd, DRM_CLIENT_CAP_ATOMIC, 1);
    drmModePlaneRes* pr = drmModeGetPlaneResources(fd);
    uint32_t plane = 0;
    for (uint32_t i = 0; pr && i < pr->count_planes && !plane; i++)
        if (pv(fd, pr->planes[i], "type") == DRM_PLANE_TYPE_PRIMARY && pv(fd, pr->planes[i], "CRTC_ID") != 0) plane = pr->planes[i];
    drmModeFreePlaneResources(pr);
    if (!plane) { printf("{\"error\":\"no active primary\"}\n"); return 1; }
    const char* names[4] = {"alpha", "SCALING_FILTER", "rotation", "pixel blend mode"};
    drmModeAtomicReq* req = drmModeAtomicAlloc();
    for (int k = 0; k < 4 && k + 2 < argc; k++) {
        uint32_t id = hs_graphics_prop(fd, plane, DRM_MODE_OBJECT_PLANE, names[k], NULL);
        drmModeAtomicAddProperty(req, plane, id, strtoull(argv[k + 2], NULL, 0));
    }
    int rc = drmModeAtomicCommit(fd, req, 0, NULL);
    int e = errno;
    drmModeAtomicFree(req);
    static char pj[8192]; plane_json(fd, 0, pj, sizeof pj);
    printf("{\"setprops_rc\":%d,\"errno\":%d,%s}\n", rc, rc ? e : 0, pj);
    close(fd);
    return rc ? 1 : 0;
}

int main(int argc, char** argv) {
    if (argc >= 5 && !strcmp(argv[1], "run")) return do_run(argc, argv);
    if (argc >= 2 && !strcmp(argv[1], "setprops")) return do_setprops(argc, argv);
    if (argc >= 2 && !strcmp(argv[1], "props")) {
        int fd = fresh_fd(); static char pj[8192]; plane_json(fd, 0, pj, sizeof pj); close(fd);
        printf("{%s}\n", pj); return 0;
    }
    fprintf(stderr, "usage: see source\n");
    return 1;
}
