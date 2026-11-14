#!/bin/bash
# Usage: tests/redteam-p0-6/mutants.sh <evidence-file> [one mutant name]   (from repo root)
# Breaks tools/p0_6_present_probe one way at a time and runs
# present_cases.sh against the broken build: a mutant is "caught" when the
# cases report at least one failure. The control (no change) must pass.
out=$1; only=$2; d=build/p06_mutants; mkdir -p $d
caught=0; missed=0
mk() { # name sed-expression description
  if [ -n "$only" ] && [ "$only" != "$1" ]; then return; fi
  sed -E "$2" tools/p0_6_present_probe.c > $d/$1.c
  if cmp -s $d/$1.c tools/p0_6_present_probe.c && [ "$1" != m0_control ]; then echo "$1: EDIT DID NOT APPLY" | tee -a $out; missed=$((missed+1)); return; fi
  gcc -O2 -Wall -Wextra -Iinclude -I/usr/include/libdrm $d/$1.c src/hs_qpu.o src/hs_v3d.o -ldrm -o $d/$1 || { echo "$1: BUILD FAILED" | tee -a $out; missed=$((missed+1)); return; }
  tests/redteam-p0-5/wrap.sh $d/$1.txt tests/redteam-p0-6/present_cases.sh $d/$1 > $d/$1.out 2>&1
  n=$(grep -c "CASE FAIL" $d/$1.out)
  if [ "$1" = m0_control ]; then
    if [ $n = 0 ]; then echo "$1: control passes [$3]" | tee -a $out; else echo "$1: CONTROL FAILED" | tee -a $out; missed=$((missed+1)); fi
  elif [ $n -gt 0 ]; then echo "$1: caught ($n cases) [$3]" | tee -a $out; grep "CASE FAIL" $d/$1.out | sed 's/^/    /' | cut -c1-160 | tee -a $out; caught=$((caught+1))
  else echo "$1: NOT CAUGHT [$3]" | tee -a $out; missed=$((missed+1)); fi
}
mk m0_control 's/XXXX_NO_MATCH//' "no change"
mk m1_no_plane_reset 's/plane_changes = reset_planes\(card, crtc_id, crtc_index\);/plane_changes = 0;/' "takeover keeps a previous client's plane properties"
mk m2_check_one_buffer 's/frames % check_every == 0 \|\| frames % check_every == 1 \|\| check_every == 1/frames % check_every == 0/' "only every check_every-th frame is checked: one buffer only when it is even"
mk m3_no_frame_check 's/wrong \+= back->map\[i\] != \(i \+ fterm\) \* \(uint32_t\)MULT;/(void)i;/' "pixels are not compared"
mk m4_no_guard_check 's/guard_hit \+= \(back->all\[k\] != GUARD\) \+ \(back->map\[npix \+ k\] != GUARD\);/(void)k;/' "guard rows are not compared"
mk m5_exit_ignores_wrong 's/return rc \|\| wrong \|\| guard_hit \|\| restore \|\| unchecked \|\| scan_bad;/return rc || restore;/' "wrong pixels do not fail the run"
mk m6_second_buffer_wrong_address 's/b\[i\]\.gpu = go\.offset \+ guard_words \* 4;/b[i].gpu = go.offset + guard_words * 4 + (i ? 64u : 0u);/' "buffer 1 is rendered 16 pixels off"
mk m7_no_signal_handler 's/sigaction\(SIG(INT|TERM), &sa, NULL\);//g' "signals kill the probe before it restores (the kernel still hands the console back)"
mk m8_fb_without_guard_offset 's/fo\[4\] = \{fault == 5 \? 0 : guard_words \* 4\}/fo[4] = {0}/' "the framebuffer starts at the guard rows: the panel would show the image 16 rows low"
mk m9_no_scanout_compare 's/scan_wrong \+= \(cap_px\[i\] & 0x00FFFFFF\) != /scan_wrong += 0 \&\& (cap_px[i] \& 0x00FFFFFF) != /' "the captured scanout is not compared (m8 would then pass)"
echo "p0-6 mutants caught: $caught; problems (not caught / build / control): $missed" | tee -a $out
