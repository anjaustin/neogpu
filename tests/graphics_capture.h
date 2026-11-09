#ifndef NEOGPU_TEST_GRAPHICS_CAPTURE_H
#define NEOGPU_TEST_GRAPHICS_CAPTURE_H
#include <GLES2/gl2.h>
#include <stdio.h>
#include <stdlib.h>

/* Test-only framebuffer evidence, enabled explicitly by NEOGPU_CAPTURE. */
static inline int test_capture_frame(void) {
    const char* path = getenv("NEOGPU_CAPTURE");
    if (!path) return 0;
    GLint view[4];
    glGetIntegerv(GL_VIEWPORT, view);
    if (view[2] <= 0 || view[3] <= 0 || view[2] > 8192 || view[3] > 8192) return -1;
    GLenum previous = glGetError();
    if (previous != GL_NO_ERROR) {
        fprintf(stderr, "GL error before capture: 0x%x\n", previous);
        return -1;
    }
    size_t row = (size_t)view[2] * 4;
    unsigned char* pixels = malloc(row * view[3]);
    if (!pixels) return -1;
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(view[0], view[1], view[2], view[3], GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        fprintf(stderr, "GL capture read error: 0x%x\n", error);
        free(pixels); return -1;
    }
    FILE* file = fopen(path, "wb");
    if (!file) { free(pixels); return -1; }
    int result = fprintf(file, "P6\n%d %d\n255\n", view[2], view[3]) < 0 ? -1 : 0;
    for (int y = view[3] - 1; y >= 0; y--)
        for (int x = 0; x < view[2]; x++)
            if (fwrite(pixels + y * row + x * 4, 1, 3, file) != 3) result = -1;
    if (fclose(file)) result = -1;
    free(pixels);
    return result;
}
#endif
