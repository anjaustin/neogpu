#!/bin/bash
# Usage: tests/redteam-p0-6/scene_present_mutants.sh <evidence-file> [one mutant name]   (from repo root)
# Breaks tools/p0_6_scene_present_probe (and tools/p0_6_kms.h) one way at a
# time and runs scene_present_cases.sh against the broken build: a mutant is
# "caught" when the cases report at least one failure. The control passes.
out=$1; only=$2; d=build/p06_sp_mutants; mkdir -p $d
caught=0; missed=0
mk() { # name sed-expression description
  if [ -n "$only" ] && [ "$only" != "$1" ]; then return; fi
  mkdir -p $d/$1.src
  cp tools/p0_6_scene.h $d/$1.src/
  sed -E "$2" tools/p0_6_scene_present_probe.c > $d/$1.src/p0_6_scene_present_probe.c
  sed -E "$2" tools/p0_6_kms.h > $d/$1.src/p0_6_kms.h
  if cmp -s $d/$1.src/p0_6_scene_present_probe.c tools/p0_6_scene_present_probe.c && cmp -s $d/$1.src/p0_6_kms.h tools/p0_6_kms.h && [ "$1" != m0_control ]; then echo "$1: EDIT DID NOT APPLY" | tee -a $out; missed=$((missed+1)); return; fi
  gcc -O2 -ffp-contract=off -Wall -Wextra -Iinclude -I/usr/include/libdrm $d/$1.src/p0_6_scene_present_probe.c src/hs_qpu.o src/hs_v3d.o -ldrm -lm -o $d/$1 || { echo "$1: BUILD FAILED" | tee -a $out; missed=$((missed+1)); return; }
  tests/redteam-p0-5/wrap.sh $d/$1.txt tests/redteam-p0-6/scene_present_cases.sh $d/$1 > $d/$1.out 2>&1
  n=$(grep -c "CASE FAIL" $d/$1.out)
  if [ "$1" = m0_control ]; then
    if [ $n = 0 ]; then echo "$1: control passes [$3]" | tee -a $out; else echo "$1: CONTROL FAILED" | tee -a $out; grep "CASE FAIL" $d/$1.out | cut -c1-160 | tee -a $out; missed=$((missed+1)); fi
  elif [ $n -gt 0 ]; then echo "$1: caught ($n cases) [$3]" | tee -a $out; grep "CASE FAIL" $d/$1.out | sed 's/^/    /' | cut -c1-160 | head -4 | tee -a $out; caught=$((caught+1))
  else echo "$1: NOT CAUGHT [$3]" | tee -a $out; missed=$((missed+1)); fi
}
mk m0_control 's/XXXX_NO_MATCH//' "no change"
mk m1_no_plane_reset 's/k->plane_changes = reset_planes\(k->card, k->crtc_id, k->crtc_index\);/k->plane_changes = 0;/' "takeover keeps a previous client's plane properties"
mk m2_no_compare 's/wrong \+= back->map\[i\] != img\[i\];/(void)img;/' "checked frames are not compared with the reference"
mk m3_no_guard_check 's/hits \+= \(b->all\[i\] != KMS_GUARD\) \+ \(b->map\[npix \+ i\] != KMS_GUARD\);/(void)npix;/' "guard rows are not compared"
mk m4_exit_ignores_failed_frames 's/return frames_failed \|\| unwritten \|\| guard_hit \|\| restore \|\| unchecked \|\| scan_bad;/return restore;/' "failed frames do not fail the run"
mk m5_unwritten_ignored 's/return frames_failed \|\| unwritten \|\|/return frames_failed ||/; s/unwritten \+= back->map\[i\] == 0xEEEEEEEE;//; s/wrong \+= back->map\[i\] != img\[i\];/wrong += back->map[i] != img[i] \&\& back->map[i] != 0xEEEEEEEE;/' "an unwritten pixel is neither counted nor different"
mk m6_no_clear_before_checked_frame 's/if \(in_checked\) memset\(back->map, 0xEE, \(size_t\)W \* H \* 4\);//' "the back buffer keeps an older frame under a checked one"
mk m7_no_signal_handler 's/sigaction\(SIG(INT|TERM), &sa, NULL\);//g' "signals kill the probe before it restores"
mk m8_fb_without_guard_offset 's/fo\[4\] = \{fb_offset_fault \? 0 : b->guard_words \* 4\}/fo[4] = {0}/' "the framebuffers start at the guard rows: the panel shows the image 16 rows low"
mk m9_no_scanout_compare 's/scan_wrong \+= \(cap_px\[i\] & 0x00FFFFFF\) != /scan_wrong += 0 \&\& (cap_px[i] \& 0x00FFFFFF) != /' "the captured scanout is not compared"
mk m10_two_buffers_only 's/int b = \(\(front < 0 \? 0 : front\) \+ i\) % NBUF;/int b = ((front < 0 ? 0 : front) + i) % 2;/' "the third buffer is never used: it cannot be checked"
mk m11_hash_ignores_pixels 's/hash = \(hash \^ back->map\[i\]\) \* 1099511628211ull;/hash = hash * 1099511628211ull;/' "the image hash does not depend on the image"
mk m12_checked_frames_all_at_t0 's/float t = in_checked \? 10\.0f \* \(float\)n_checked_done \/ \(float\)checked/float t = in_checked ? 0.0f * (float)n_checked_done \/ (float)checked/' "every checked frame is the same frame (the reference follows): the orbit is not covered"
# Not a mutant here: leaving out kms_restore() changes nothing these cases can see. When the
# process ends the kernel puts the console back by itself (the SIGKILL case shows it), so the
# explicit restore only matters when the CRTC was another running client's, which cannot be
# set up while this probe is DRM master. Measured 2026-11-17: 'no restore' passed all 25 cases.
mk m13_async_mode_still_drops 's/int wait = in_checked \|\| pacing == 1, now_flip = !in_checked \&\& pacing == 2;/int wait = in_checked || pacing == 1, now_flip = 0;/' "pacing 2 behaves as pacing 0: frames are dropped while a flip is pending"
echo "p0-6 scene-present mutants caught: $caught; problems (not caught / build / control): $missed" | tee -a $out
