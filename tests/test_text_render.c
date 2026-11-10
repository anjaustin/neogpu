/* Hardware check: rendered glyphs match the font atlas.
 *
 * Renders a string at exactly one screen pixel per font pixel into an
 * offscreen surface (the display is not taken) and compares every glyph cell
 * with that glyph's coverage in the atlas. A wrong texture-coordinate mapping
 * (flipped or offset rows), a glyph table filled at the wrong codes, or a
 * broken atlas load all show up as mismatched cells.
 *
 * Usage: sudo tests/test_text_render   (needs a DRM device; display untouched)
 */
#include "hs_graphics.h"
#include "hs_text.h"
#include <stdio.h>
#include <stdlib.h>

#define W 320
#define H 64

int main(void) {
    static HSGraphics gfx;
    if (hs_graphics_init_sized(&gfx, W, H) != 0) { printf("{\"init\":false}\n"); return 1; }
    gfx.take_display = false; /* offscreen: never touch the panel */
    static HSFont font;
    if (!hs_font_load_default(&font)) { printf("{\"font\":false}\n"); hs_graphics_finish(&gfx); return 1; }
    hs_font_upload_texture(&font);

    const char* text = "Ag0&9\"Wy";
    const int pen_x = 8, pen_y = 8; /* top-left pen position in pixels */
    glViewport(0, 0, W, H);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    float sx = 2.0f / W, sy = 2.0f / H;
    hs_font_render_text_xy(&font, text, -1.0f + pen_x * sx, 1.0f - pen_y * sy, sx, sy, 1, 1, 1, 1);
    static unsigned char pixels[W * H * 4];
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    bool read_ok = glGetError() == GL_NO_ERROR;

    unsigned cells = 0, mismatches = 0, ink = 0, glyphs = 0;
    int cursor = pen_x;
    for (const char* p = text; *p; p++) {
        const HSGlyph* g = &font.glyphs[(unsigned char)*p];
        if (!g->advance) { mismatches += 1000; continue; } /* glyph missing */
        glyphs++;
        for (int gy = 0; gy < g->height; gy++) {
            for (int gx = 0; gx < g->width; gx++) {
                /* Atlas coverage (alpha) for this glyph pixel; atlas row 0 is the top. */
                int ax = g->tex_x + gx, ay = g->tex_y + gy;
                unsigned char coverage = font.atlas_data[((size_t)ay * font.atlas_w + ax) * 4 + 3];
                /* Screen pixel (top-left origin) -> GL row (bottom-left origin). */
                int px = cursor + g->offset_x + gx, py = pen_y + g->offset_y + gy;
                if (px < 0 || px >= W || py < 0 || py >= H) { mismatches++; continue; }
                unsigned char lit = pixels[((size_t)(H - 1 - py) * W + px) * 4];
                bool expect = coverage > 127, got = lit > 127;
                cells++;
                ink += expect;
                mismatches += expect != got;
            }
        }
        cursor += g->advance;
    }
    hs_font_unload(&font);
    hs_graphics_finish(&gfx);
    /* Linear filtering at exact 1:1 alignment reproduces texels; allow a few
     * edge cells for rounding, never a systematic mismatch. */
    bool ok = read_ok && glyphs == 8 && ink > 40 && mismatches * 50 <= cells;
    printf("{\"test\":\"text_render_matches_atlas\",\"glyphs\":%u,\"cells\":%u,\"ink\":%u,"
           "\"mismatches\":%u,\"passed\":%s}\n", glyphs, cells, ink, mismatches, ok ? "true" : "false");
    return ok ? 0 : 1;
}
