/*
 * Test 08: Ray Cast Orbit
 *
 * Five ray-traced spheres orbiting a common central (vertical) axis, against a
 * vertical gradient background, with a seven-segment FPS HUD.
 *
 * Verifies:
 * - Ray-sphere intersection with multiple animated spheres
 * - Per-frame orbital animation driven by a time uniform
 * - Gradient background (screen-space vertical lerp)
 * - Self-contained seven-segment HUD (no font atlas)
 *
 * Expected: 5 colored, lit spheres revolving around the centre of the screen
 * about the vertical axis, over a top-to-bottom gradient, FPS in the corner.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include "hs_graphics.h"

static const char* vertex_shader_src = R"(
    attribute vec2 a_position;
    varying vec2 v_uv;

    void main() {
        v_uv = a_position * 0.5 + 0.5;
        gl_Position = vec4(a_position, 0.0, 1.0);
    }
)";

static const char* fragment_shader_src = R"(
    precision mediump float;
    varying vec2 v_uv;
    uniform float u_time;
    uniform vec2 u_resolution;

    #define NUM_ORBIT 5
    #define PI 3.14159265

    // Feature toggles (bisect compile/perf by flipping these).
    #define ENABLE_SHADOWS    1
    #define ENABLE_REFLECT    1

    #define ORBIT_R 1.7
    #define SPH_R   0.5
    #define CORE_R  0.85
    #define GROUND_Y -0.95
    #define SPEED   0.6

    // Centre of orbit sphere i (i = 0..NUM_ORBIT-1); i = NUM_ORBIT -> core.
    vec3 sphere_center(int i, float t) {
        if (i >= NUM_ORBIT) return vec3(0.0, 0.0, 0.0);       // central anchor
        float a = (2.0 * PI * float(i)) / float(NUM_ORBIT) + t * SPEED;
        return vec3(cos(a) * ORBIT_R, 0.0, sin(a) * ORBIT_R);
    }
    float sphere_radius(int i) { return (i >= NUM_ORBIT) ? CORE_R : SPH_R; }

    vec3 sphere_albedo(int i) {
        if (i >= NUM_ORBIT) return vec3(0.85, 0.85, 0.9);     // pale core
        float fi = float(i) / float(NUM_ORBIT);
        return 0.5 + 0.5 * cos(2.0 * PI * (fi + vec3(0.0, 0.33, 0.67)));
    }

    float hit_sphere(vec3 ro, vec3 rd, vec3 c, float r) {
        vec3 oc = ro - c;
        float b = dot(oc, rd);
        float cc = dot(oc, oc) - r * r;
        float h = b * b - cc;
        if (h < 0.0) return -1.0;
        float t = -b - sqrt(h);
        return t;
    }

    float hit_ground(vec3 ro, vec3 rd) {
        if (abs(rd.y) < 1e-4) return -1.0;
        float t = (GROUND_Y - ro.y) / rd.y;
        return (t > 0.0) ? t : -1.0;
    }

    // Trace scene. Returns nearest t (or -1). Outputs hit id: 0..5 sphere,
    // 100 = ground, -1 = miss.
    float trace(vec3 ro, vec3 rd, float tm, out int id, float t) {
        float best = tm;
        int bid = -1;
        for (int i = 0; i <= NUM_ORBIT; i++) {
            float ts = hit_sphere(ro, rd, sphere_center(i, t), sphere_radius(i));
            if (ts > 0.001 && ts < best) { best = ts; bid = i; }
        }
        float tg = hit_ground(ro, rd);
        if (tg > 0.001 && tg < best) { best = tg; bid = 100; }
        id = bid;
        return (bid < 0) ? -1.0 : best;
    }

    float in_shadow(vec3 p, vec3 ldir, float t) {
        for (int i = 0; i <= NUM_ORBIT; i++) {
            float ts = hit_sphere(p + ldir * 0.01, ldir, sphere_center(i, t), sphere_radius(i));
            if (ts > 0.001) return 1.0;
        }
        return 0.0;
    }

    vec3 sky(vec3 rd, vec2 uv_y) {
        return mix(vec3(0.55, 0.22, 0.35), vec3(0.08, 0.10, 0.28), uv_y.y);
    }

    vec3 shade(vec3 p, vec3 n, vec3 rd, vec3 albedo, vec3 light, float t) {
        float amb = 0.22;
        float diff = max(dot(n, light), 0.0);
        float sh = 0.0;
        #if ENABLE_SHADOWS
        sh = in_shadow(p, light, t);
        #endif
        diff *= (1.0 - 0.75 * sh);
        vec3 h = normalize(light - rd);
        float spec = pow(max(dot(n, h), 0.0), 32.0) * (1.0 - sh);
        float rim = pow(1.0 - max(dot(n, -rd), 0.0), 2.0) * 0.25;
        return albedo * (amb + diff * 0.85) + vec3(spec) * 0.7 + vec3(rim);
    }

    void main() {
        float t = u_time;
        float aspect = u_resolution.x / u_resolution.y;
        vec2 uv = (v_uv * 2.0 - 1.0) * vec2(aspect, 1.0);

        vec3 ro = vec3(0.0, 1.1, 4.3);
        vec3 rd = normalize(vec3(uv, -1.7));
        vec3 light = normalize(vec3(0.5, 1.0, 0.4));

        vec3 color;
        int id;
        float tt = trace(ro, rd, 1e9, id, t);

        if (id < 0) {
            color = sky(rd, v_uv);
        } else {
            vec3 p = ro + rd * tt;
            vec3 n; vec3 albedo; float reflk;
            if (id == 100) {
                n = vec3(0.0, 1.0, 0.0);
                float chk = mod(floor(p.x) + floor(p.z), 2.0);
                albedo = mix(vec3(0.20), vec3(0.35), chk);
                reflk = 0.25;
            } else {
                n = normalize(p - sphere_center(id, t));
                albedo = sphere_albedo(id);
                reflk = 0.35;
            }
            color = shade(p, n, rd, albedo, light, t);

            #if ENABLE_REFLECT
            // one reflection bounce
            vec3 rrd = reflect(rd, n);
            vec3 rro = p + n * 0.02;
            int rid;
            float rt = trace(rro, rrd, 1e9, rid, t);
            vec3 rcol;
            if (rid < 0) {
                rcol = sky(rrd, v_uv);
            } else {
                vec3 rp = rro + rrd * rt;
                vec3 rn; vec3 ralb;
                if (rid == 100) {
                    rn = vec3(0.0, 1.0, 0.0);
                    float c2 = mod(floor(rp.x) + floor(rp.z), 2.0);
                    ralb = mix(vec3(0.20), vec3(0.35), c2);
                } else {
                    rn = normalize(rp - sphere_center(rid, t));
                    ralb = sphere_albedo(rid);
                }
                rcol = shade(rp, rn, rrd, ralb, light, t);
            }
            color = mix(color, rcol, reflk);
            #endif
        }

        gl_FragColor = vec4(color, 1.0);
    }
)";

static GLuint compile_shader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);
    GLint compiled;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        char log[512];
        glGetShaderInfoLog(shader, sizeof(log), NULL, log);
        fprintf(stderr, "Shader error: %s\n", log);
        return 0;
    }
    return shader;
}

static GLuint create_program(const char* vs_src, const char* fs_src) {
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_src);
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    return program;
}

/* ------------------------------------------------------------------------
 * Seven-segment FPS HUD (solid-color rectangles; no font atlas, no UVs).
 * Mirrors the renderer proven in test_06.
 * ---------------------------------------------------------------------- */

static const char* hud_vs_src = R"(
    attribute vec2 a_position;
    void main() { gl_Position = vec4(a_position, 0.0, 1.0); }
)";
static const char* hud_fs_src = R"(
    precision mediump float;
    uniform vec4 u_color;
    void main() { gl_FragColor = u_color; }
)";

typedef struct {
    GLuint program;
    GLuint vbo;
    GLint  a_pos;
    GLint  u_color;
    float  verts[4096];
    int    nverts;
} HudCtx;

static void hud_init(HudCtx* c) {
    memset(c, 0, sizeof(*c));
    c->program = create_program(hud_vs_src, hud_fs_src);
    c->a_pos   = glGetAttribLocation(c->program, "a_position");
    c->u_color = glGetUniformLocation(c->program, "u_color");
    glGenBuffers(1, &c->vbo);
}

static void hud_rect(HudCtx* c, float x0, float y0, float x1, float y1) {
    if (c->nverts + 12 > (int)(sizeof(c->verts) / sizeof(float))) return;
    float* v = &c->verts[c->nverts];
    v[0]=x0; v[1]=y0;  v[2]=x1; v[3]=y0;  v[4]=x0; v[5]=y1;
    v[6]=x1; v[7]=y0;  v[8]=x1; v[9]=y1;  v[10]=x0; v[11]=y1;
    c->nverts += 12;
}

static const unsigned char HUD_SEG[10] = {
    0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F,
};

static void hud_draw_number(HudCtx* c, const char* str, float x, float y,
                            float dh, int screen_w, int screen_h,
                            float r, float g, float b, float a) {
    c->nverts = 0;
    float aspect = (float)screen_h / (float)screen_w;
    float dw = dh * 0.6f * aspect;
    float t  = dh * 0.14f;
    float tx = t * aspect;
    float gap = dw * 0.35f;
    float cx = x;
    for (const char* p = str; *p; ++p) {
        if (*p >= '0' && *p <= '9') {
            unsigned char s = HUD_SEG[*p - '0'];
            float xl = cx, xr = cx + dw;
            float yt = y, ym = y - dh*0.5f, yb = y - dh;
            if (s & 0x01) hud_rect(c, xl+tx, yt,        xr-tx, yt-t);
            if (s & 0x02) hud_rect(c, xr-tx, yt-t,      xr,    ym+t*0.5f);
            if (s & 0x04) hud_rect(c, xr-tx, ym-t*0.5f, xr,    yb+t);
            if (s & 0x08) hud_rect(c, xl+tx, yb,        xr-tx, yb+t);
            if (s & 0x10) hud_rect(c, xl,    ym-t*0.5f, xl+tx, yb+t);
            if (s & 0x20) hud_rect(c, xl,    yt-t,      xl+tx, ym+t*0.5f);
            if (s & 0x40) hud_rect(c, xl+tx, ym-t*0.5f, xr-tx, ym+t*0.5f);
            cx += dw + gap;
        } else if (*p == '.') {
            hud_rect(c, cx, y - dh, cx + tx, y - dh + t);
            cx += tx + gap;
        } else {
            cx += dw * 0.5f + gap;
        }
    }
    if (c->nverts == 0) return;
    glUseProgram(c->program);
    glBindBuffer(GL_ARRAY_BUFFER, c->vbo);
    glBufferData(GL_ARRAY_BUFFER, c->nverts * sizeof(float), c->verts, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(c->a_pos);
    glVertexAttribPointer(c->a_pos, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glUniform4f(c->u_color, r, g, b, a);
    glDrawArrays(GL_TRIANGLES, 0, c->nverts / 2);
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    printf("=== Test 08: Ray Cast Orbit (5 spheres) ===\n");

    HSGraphics gfx;
    if (hs_graphics_init(&gfx) != 0) {
        fprintf(stderr, "Failed to init graphics\n");
        return 1;
    }
    printf("Screen: %dx%d\n", gfx.screen_width, gfx.screen_height);

    HudCtx hud;
    hud_init(&hud);

    GLuint program = create_program(vertex_shader_src, fragment_shader_src);
    if (!program) return 1;

    float quad[] = {
        -1, -1,  1, -1,  -1,  1,
         1, -1,  1,  1,  -1,  1,
    };

    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);

    GLint pos_loc  = glGetAttribLocation(program, "a_position");
    GLint time_loc = glGetUniformLocation(program, "u_time");
    GLint res_loc  = glGetUniformLocation(program, "u_resolution");

    glUseProgram(program);
    glEnableVertexAttribArray(pos_loc);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glUniform2f(res_loc, (float)gfx.screen_width, (float)gfx.screen_height);

    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);
    int frame = 0;
    float fps = 0.0f;

    printf("Rendering 5 orbiting spheres for 10 seconds...\n");

    while (1) {
        clock_gettime(CLOCK_MONOTONIC, &now);
        double elapsed = (now.tv_sec - start.tv_sec) + (now.tv_nsec - start.tv_nsec) / 1e9;
        if (elapsed >= 10.0) break;

        float t = (float)elapsed;

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        /* Re-bind the raycast program + vertex state each frame; the HUD below
         * leaves its own program/VBO/attrib bound. */
        glUseProgram(program);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(pos_loc);
        glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 0, 0);

        glUniform1f(time_loc, t);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        if (frame > 0) {
            char fps_buf[32];
            snprintf(fps_buf, sizeof(fps_buf), "%.1f", fps);
            hud_draw_number(&hud, fps_buf, -0.96f, 0.92f, 0.12f,
                            gfx.screen_width, gfx.screen_height,
                            1.0f, 1.0f, 0.0f, 1.0f);
        }

        hs_graphics_present(&gfx);

        if (frame % 30 == 0) {
            fps = frame / elapsed;
            printf("FPS: %.1f\n", fps);
        }
        frame++;
    }

    glDeleteBuffers(1, &vbo);
    glDeleteProgram(program);
    hs_graphics_finish(&gfx);

    printf("PASS: Ray cast orbit test\n");
    return 0;
}
