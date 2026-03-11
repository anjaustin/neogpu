#ifndef HS_INPUT_H
#define HS_INPUT_H

#include "hs_core.h"
#include <time.h>

#define HS_FPS    60
#define HS_WIDTH  640
#define HS_HEIGHT 480

/*
 * Input state - mirrors PicoApi controls.
 *
 * On a real platform (SDL, Linux evdev, etc.) these would be
 * populated by an event-polling loop.  The struct is platform-
 * agnostic; only the polling function is platform-specific.
 */
typedef struct {
    /* Directional input  [-1.0 .. 1.0] */
    f32  dir_x;
    f32  dir_y;

    /* Buttons (true = pressed this frame, one-shot) */
    bool button1;
    bool button2;

    /* Raw key state for dir computation */
    bool key_left;
    bool key_right;
    bool key_up;
    bool key_down;

    /* Mouse */
    f32  mouse_x;       /* 0 .. HS_WIDTH  */
    f32  mouse_y;       /* 0 .. HS_HEIGHT */
    bool mouse_left;
    bool mouse_right;

    /* Gamepad axes (raw) */
    f32  pad_x;
    f32  pad_y;
    bool pad_a;
    bool pad_b;

    /* Timing */
    u32  frame;         /* frame counter since reset */
    f64  elapsed;       /* seconds since reset (frame / FPS) */
    f64  epoch;         /* unix timestamp (seconds) */
} HSInput;

static inline void hs_input_init(HSInput* in) {
    memset(in, 0, sizeof(HSInput));
}

/* Call once per frame to recompute derived values */
static inline void hs_input_tick(HSInput* in) {
    /* Directional input from keys + gamepad */
    f32 dx = 0.0f, dy = 0.0f;
    if (in->key_left)  dx -= 1.0f;
    if (in->key_right) dx += 1.0f;
    if (in->key_up)    dy -= 1.0f;
    if (in->key_down)  dy += 1.0f;
    dx += in->pad_x;
    dy += in->pad_y;
    /* Clamp to [-1, 1] */
    if (dx < -1.0f) dx = -1.0f;
    if (dx >  1.0f) dx =  1.0f;
    if (dy < -1.0f) dy = -1.0f;
    if (dy >  1.0f) dy =  1.0f;
    in->dir_x = dx;
    in->dir_y = dy;

    /* Timing */
    in->frame++;
    in->elapsed = (f64)in->frame / (f64)HS_FPS;
    in->epoch = (f64)time(NULL);
}

/* Convenience accessors matching PicoApi names */
static inline f32  hs_dir_x(const HSInput* in)   { return in->dir_x; }
static inline f32  hs_dir_y(const HSInput* in)   { return in->dir_y; }
static inline bool hs_button(const HSInput* in)  { return in->button1 || in->mouse_left || in->pad_a; }
static inline bool hs_button2(const HSInput* in) { return in->button2 || in->mouse_right || in->pad_b; }
static inline f32  hs_mouse_x(const HSInput* in) { return in->mouse_x; }
static inline f32  hs_mouse_y(const HSInput* in) { return in->mouse_y; }
static inline f32  hs_time(const HSInput* in)    { return (f32)in->elapsed; }
static inline f64  hs_date(const HSInput* in)    { return in->epoch; }

#endif
