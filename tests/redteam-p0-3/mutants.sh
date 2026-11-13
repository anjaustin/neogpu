#!/bin/bash
# Build mutants of include/hs_graphics.h and run tests/test_present_scaled against each.
# Usage (repo root): tests/redteam-p0-3/mutants.sh <evidence.txt>
out=$1; M=tests/redteam-p0-3/mutants; mkdir -p $M
H=include/hs_graphics.h
mk() { # name sed-expr description
  mkdir -p $M/$1; sed "$2" $H > $M/$1/hs_graphics.h
  if cmp -s $H $M/$1/hs_graphics.h; then echo "$1: SED DID NOT APPLY" | tee -a $out; return; fi
  echo "$1: $3 ($(diff $H $M/$1/hs_graphics.h | grep -c '^[<>]') diff lines)" >> $M/list.txt
  gcc -O2 -I$M/$1 -Iinclude -I/usr/include/libdrm tests/test_present_scaled.c -lm -lGLESv2 -lgbm -ldrm -lEGL -lpthread -o $M/$1/test_present_scaled 2>$M/$1/build.err || echo "$1 build failed" | tee -a $out
}
: > $M/list.txt
mk m0_identity 's/^\(#define HS_GRAPHICS_HOLD_AFTER_REPEAT 60\)$/\1 \/\/ m0/' "no behavioural change (control)"
mk m1_dst_half_w 's/"CRTC_W", gfx->display_mode.hdisplay}/"CRTC_W", gfx->display_mode.hdisplay \/ 2}/' "CRTC_W = mode width / 2"
mk m2_no_disable_others '/p->plane_id, crtc_prop, 0) < 0;/d;/p->plane_id, fb_prop, 0) < 0;/d' "other planes on the CRTC are not turned off"
mk m3_no_active '/"ACTIVE", 1}/d' "ACTIVE not set"
mk m4_no_mode_id '/"MODE_ID", blob}/d' "MODE_ID not set"
mk m5_src_w_minus1 's/"SRC_W", (uint64_t)gfx->screen_width << 16}/"SRC_W", (uint64_t)(gfx->screen_width - 1) << 16}/' "SRC_W one column short"
mk m6_crtc_y_8 's/"CRTC_Y", 0}/"CRTC_Y", 8}/' "CRTC_Y = 8: image shifted down 8 px, bottom rows clipped"
mk m7_src_x_half 's/"SRC_X", 0}/"SRC_X", 0x8000}/;s/"SRC_W", (uint64_t)gfx->screen_width << 16}/"SRC_W", ((uint64_t)gfx->screen_width << 16) - 0x8000}/' "SRC_X = 0.5 px with SRC_W reduced by 0.5 (W check rounds?)"
mk m8_filter_nearest 's/{primary, DRM_MODE_OBJECT_PLANE, "CRTC_ID", gfx->display_crtc_id},/&\n            {primary, DRM_MODE_OBJECT_PLANE, "SCALING_FILTER", 1},/' "SCALING_FILTER forced to Nearest"
mk m9_alpha_half 's/{primary, DRM_MODE_OBJECT_PLANE, "CRTC_ID", gfx->display_crtc_id},/&\n            {primary, DRM_MODE_OBJECT_PLANE, "alpha", 0x8000},/' "plane alpha 0.5 (image at half brightness)"
mk m10_flip_resets_rects 's/drmModePageFlip(gfx->drm_fd, gfx->display_crtc_id, fb, DRM_MODE_PAGE_FLIP_EVENT, gfx) != 0/drmModeSetPlane(gfx->drm_fd, 82, gfx->display_crtc_id, fb, 0, 0, 0, gfx->display_mode.hdisplay \/ 2, gfx->display_mode.vdisplay \/ 2, 0, 0, gfx->screen_width << 16, gfx->screen_height << 16) != 0 || drmModePageFlip(gfx->drm_fd, gfx->display_crtc_id, fb, DRM_MODE_PAGE_FLIP_EVENT, gfx) != 0/' "every flip moves the plane to the top-left quarter"
cat $M/list.txt | tee -a $out
for d in $M/m*/; do n=$(basename $d); [ -x $d/test_present_scaled ] || continue
  for i in 1; do
    res=$(sudo flock build/gpu.lock $d/test_present_scaled 400 240 60 2>&1 | tail -1); rc=$?
    echo "$n: $(echo "$res" | python3 -c 'import sys,json; d=json.loads(sys.stdin.read()); print("passed=%s scaled_ok=%s active=%s err=%r plane_src=%s plane_dst=%s repeated=%s/%s console_ok=%s"%(d["passed"],d.get("scaled_ok"),d.get("active"),d.get("error"),d.get("plane_src"),d.get("plane_dst"),d.get("repeated"),d.get("refreshes"),d.get("console_ok")))' 2>&1)" | tee -a $out
  done
done
