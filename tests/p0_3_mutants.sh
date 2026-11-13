#!/bin/bash
# Mutation check for tests/test_present_scaled (P0-3, docs/audit/HARDWARE_HANDOFF_P0.md):
# builds copies of include/hs_graphics.h with one fault each (in build/p0_3_mutants/,
# artifacts) and runs the test against each. Every mutant but the control must fail.
# Usage (repo root, sudo for the display): tests/p0_3_mutants.sh <out.txt>
set -u
out=$1; M=build/p0_3_mutants; rm -rf $M; mkdir -p $M
H=include/hs_graphics.h
mk() { # name sed-expr description
  mkdir -p $M/$1; sed "$2" $H > $M/$1/hs_graphics.h
  if [ "$1" != m0_control ] && cmp -s $H $M/$1/hs_graphics.h; then echo "$1: SED DID NOT APPLY" | tee -a "$out"; rm -rf $M/$1; return; fi
  echo "$1: $3" >> $M/list.txt
  gcc -O2 -I$M/$1 -Iinclude -I/usr/include/libdrm tests/test_present_scaled.c -lm -lGLESv2 -lgbm -ldrm -lEGL -lpthread \
    -o $M/$1/test_present_scaled 2>$M/$1/build.err || echo "$1: build failed" | tee -a "$out"
}
: > $M/list.txt
mk m0_control 's/x/x/' "unchanged (must pass)"
mk m1_dst_half_w 's/"CRTC_W", gfx->display_mode.hdisplay}/"CRTC_W", gfx->display_mode.hdisplay \/ 2}/' "CRTC_W = mode width / 2"
mk m2_no_disable_others '/p->plane_id, crtc_prop, 0) < 0;/d;/p->plane_id, fb_prop, 0) < 0;/d' "scaled: other planes not turned off"
mk m3_no_active '/"ACTIVE", 1}/d' "ACTIVE not set"
mk m4_no_mode_id '/"MODE_ID", blob}/d' "MODE_ID not set"
mk m5_src_w_minus1 's/"SRC_W", (uint64_t)gfx->screen_width << 16}/"SRC_W", (uint64_t)(gfx->screen_width - 1) << 16}/' "SRC_W one column short"
mk m6_crtc_y_8 's/"CRTC_Y", 0}/"CRTC_Y", 8}/' "CRTC_Y = 8"
mk m7_src_x_half 's/"SRC_X", 0}/"SRC_X", 0x8000}/;s/"SRC_W", (uint64_t)gfx->screen_width << 16}/"SRC_W", ((uint64_t)gfx->screen_width << 16) - 0x8000}/' "SRC_X = 0.5 with SRC_W - 0.5"
mk m8_filter_nearest 's/{"SCALING_FILTER", 0},/{"SCALING_FILTER", 1},/' "default filter changed to Nearest"
mk m9_alpha_half 's/{"alpha", 0xffff},/{"alpha", 0x8000},/' "default alpha changed to half"
mk m10_flip_moves_plane 's/drmModePageFlip(gfx->drm_fd, gfx->display_crtc_id, fb, DRM_MODE_PAGE_FLIP_EVENT, gfx) != 0/drmModeSetPlane(gfx->drm_fd, 82, gfx->display_crtc_id, fb, 0, 0, 0, gfx->display_mode.hdisplay \/ 2, gfx->display_mode.vdisplay \/ 2, 0, 0, gfx->screen_width << 16, gfx->screen_height << 16) != 0 || drmModePageFlip(gfx->drm_fd, gfx->display_crtc_id, fb, DRM_MODE_PAGE_FLIP_EVENT, gfx) != 0/' "every flip moves the plane to a quarter (plane 82: the DSI primary on this Pi)"
mk m11_no_native_reset 's/    if (!gfx->scaled_scanout) hs_graphics_reset_plane(gfx);/    (void)0;/' "native takeover does not reset the plane"
mk m12_native_keeps_overlay 's/drmModeSetPlane(gfx->drm_fd, p->plane_id, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);/(void)0;/' "native takeover leaves other planes on"
mk m13_no_scaled_defaults 's/bad |= id \&\& drmModeAtomicAddProperty(req, primary, id, hs_graphics_plane_defaults\[i\].value) < 0;/(void)id;/' "scaled takeover does not reset plane properties"
mk m14_always_unscaled 's/gfx->scaled_scanout = width != detected_width || height != detected_height;/gfx->scaled_scanout = false;/' "scaled path never taken"
cat $M/list.txt >> "$out"
for d in $M/m*/; do n=$(basename $d); [ -x $d/test_present_scaled ] || continue
  res=$(flock build/gpu.lock $d/test_present_scaled 400 240 2>&1 | tail -1)
  echo "$n: $(echo "$res" | python3 -c 'import sys,json
try:
    d=json.loads(sys.stdin.read())
    print("passed=%s failed_checks=%s" % (d.get("passed"), [k for k,v in d.items() if v is False] + [k for k in ("cheap","heavy") if isinstance(d.get(k),dict) and not d[k]["ok"]]))
except Exception as e: print("unparsed:", e)')" | tee -a "$out"
done
