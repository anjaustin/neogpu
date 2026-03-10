#include "hs_math_neon.h"
#include <stdio.h>

void m4_to_array(mat4 m, float* out) {
    for (int i = 0; i < 4; i++) {
        float32x2_t pair = vget_low_f32(m.val[i]);
        out[i*4 + 0] = vget_lane_f32(pair, 0);
        out[i*4 + 1] = vget_lane_f32(pair, 1);
        float32x2_t pair2 = vget_high_f32(m.val[i]);
        out[i*4 + 2] = vget_lane_f32(pair2, 0);
        out[i*4 + 3] = vget_lane_f32(pair2, 1);
    }
}

mat4 m4_from_array(float* m) {
    mat4 result;
    for (int i = 0; i < 4; i++) {
        result.val[i] = v4_make(m[i*4 + 0], m[i*4 + 1], m[i*4 + 2], m[i*4 + 3]);
    }
    return result;
}

void m4_print(mat4 m) {
    float arr[16];
    m4_to_array(m, arr);
    printf("[%.3f %.3f %.3f %.3f]\n", arr[0], arr[4], arr[8], arr[12]);
    printf("[%.3f %.3f %.3f %.3f]\n", arr[1], arr[5], arr[9], arr[13]);
    printf("[%.3f %.3f %.3f %.3f]\n", arr[2], arr[6], arr[10], arr[14]);
    printf("[%.3f %.3f %.3f %.3f]\n", arr[3], arr[7], arr[11], arr[15]);
}
