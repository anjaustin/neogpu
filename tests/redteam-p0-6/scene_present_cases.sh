#!/bin/bash
# Usage: tests/redteam-p0-6/scene_present_cases.sh [probe-binary]   (from repo root, as root, inside tests/redteam-p0-5/wrap.sh)
# Adversarial runs of tools/p0_6_scene_present_probe (the test_08 scene by
# QPU compute on the panel). For each: the probe's JSON line, its exit code,
# and whether the display state (the kernel's own dump of CRTCs and planes)
# is the same after as before. Prints "CASE ok|FAIL".
P=${1:-tools/p0_6_scene_present_probe}   # argument: a mutant build, to show a case fails
state() { grep -E "^plane|^crtc|fb=|crtc=|enable=|active=|allocated by|crtc-pos|src-pos|rotation|alpha|blend" /sys/kernel/debug/dri/1/state; }
fails=0
run() { # name expected_exit cmd...
  name=$1; want=$2; shift 2
  state > build/p06_before.txt
  "$@" > build/p06_out.txt 2>&1; got=$?
  sleep 0.3
  state > build/p06_after.txt
  same=yes; cmp -s build/p06_before.txt build/p06_after.txt || same=NO
  sed 's/^/    /' build/p06_out.txt | cut -c1-1500
  shown=yes; if [ -n "$MUST" ] && ! grep -Eq "$MUST" build/p06_out.txt; then shown=NO; fi
  if [ $got = $want ] && [ $same = yes ] && [ $shown = yes ]; then echo "CASE ok: $name (exit $got, display state unchanged${MUST:+, the named check failed})"
  elif [ $shown = NO ]; then echo "CASE FAIL: $name (exit $got; the output does not show: $MUST)"; fails=$((fails+1))
  else echo "CASE FAIL: $name (exit $got, wanted $want; display state unchanged: $same)"; diff build/p06_before.txt build/p06_after.txt | head -10; fails=$((fails+1)); fi
}
run "3 s free-running, 8 batches, 6 frames checked" 0 $P 3 8 0 6
run "3 s with every flip waited for" 0 $P 3 8 1 6
run "1, 2 and 4 batches" 0 sh -c "$P 1 1 0 3 && $P 1 2 0 3 && $P 1 4 0 3"
run "checked frames only, 9 of them" 0 $P 0 8 0 9
run "no cuts" 0 env P06_CULL=0 $P 1 8 0 3
run "timed only: nothing checked, no capture" 0 $P 1 8 0 0
MUST='"rc":0,.*"frames_failed":[1-9]' run "checked frames rendered at the wrong scene time: the check must fail" 1 $P 1 8 0 6 1
MUST='"frames_failed":0,.*"pixels_different_from_memory":[1-9]' run "framebuffer objects start 16 rows early: the scanout capture must catch it" 1 $P 1 8 0 6 3
MUST='"rc":0,.*"guard_words_overwritten":[1-9]' run "kernel fault 4, one row too many per batch: the guard rows must catch it" 1 env P06_FAULT=4 $P 1 8 0 6
MUST='"rc":0,.*"frames_failed":[1-9]' run "kernel fault 6, region sort ignores covering spheres" 1 env P06_FAULT=6 $P 1 8 0 6
MUST='"rc":0,.*"frames_failed":[1-9]' run "kernel fault 3, no shadow test on ground that may be shadowed" 1 env P06_FAULT=3 $P 1 8 0 6
MUST='"rc":0,.*"unwritten":[1-9]' run "kernel fault 13, sky rows not stored after the first checked frame" 1 env P06_FAULT=13 $P 1 8 0 6
run "stopped by its own SIGINT after 1 s" 0 $P 5 8 0 6 2
run "SIGTERM from outside after 1.5 s" 0 sh -c "$P 5 8 0 6 & p=\$!; sleep 1.5; kill -TERM \$p; wait \$p"
run "SIGINT from outside during start-up" 0 sh -c "$P 5 8 0 6 & p=\$!; sleep 0.02; kill -INT \$p; wait \$p"
run "SIGTERM during the checked frames" 0 sh -c "$P 0 8 0 200 & p=\$!; sleep 2; kill -TERM \$p; wait \$p"
run "a second copy while the first holds the display: refused (exit 3: the run, not a check)" 3 sh -c "$P 3 8 0 3 > build/p06_first.txt & p=\$!; sleep 1; $P 1 8 0 3; r=\$?; wait \$p; cat build/p06_first.txt; exit \$r"
run "bad arguments: seconds" 2 $P -1 8 0 6
run "bad arguments: batches 3" 2 $P 1 3 0 6
run "bad arguments: pacing 2" 2 $P 1 8 2 6
run "bad arguments: nothing to do" 2 $P 0 8 0 0
run "bad arguments: cuts 9" 2 env P06_CULL=9 $P 1 8 0 6
run "SIGKILL after 1.5 s: the kernel must give the console back" 137 sh -c "$P 5 8 0 6 & p=\$!; sleep 1.5; kill -KILL \$p; wait \$p"
# the panel's frames are the offscreen probe's frames: same scene times, same image hashes
S=tools/p0_6_shade_probe
$P 0 8 0 12 > build/p06_out.txt 2>&1; got=$?
a=$(grep -o '"image_hashes":\[[^]]*\]' build/p06_out.txt | grep -o '[0-9a-f]\{16\}' | paste -sd,)
b=$($S 8 12 | grep -o '"gpu_image_hash":"[0-9a-f]*"' | grep -o '[0-9a-f]\{16\}' | paste -sd,)
if [ $got = 0 ] && [ -n "$a" ] && [ "$a" = "$b" ] && [ $(echo "$a" | tr ',' '\n' | sort -u | wc -l) = 12 ]
then echo "CASE ok: 12 frames on the panel have the image hashes of the same 12 frames offscreen"
else echo "CASE FAIL: panel and offscreen frames differ (exit $got)"; echo "    panel:     $a"; echo "    offscreen: $b"; fails=$((fails+1)); fi
# a previous client left the plane the console shows at half alpha, nearest
# filter and coverage blending: the takeover must put the defaults back
R=tests/redteam-p0-3/rt_scaled
if [ -x $R ]; then
  $R props > build/p06_props_clean.txt
  $R setprops 0x8000 1 1 1 > build/p06_props_dirty.txt
  $P 1 8 0 3 > build/p06_out.txt 2>&1; got=$?
  $R props > build/p06_props_after.txt
  sed 's/^/    /' build/p06_out.txt | cut -c1-640
  ch=$(sed -E 's/.*"plane_changes_at_takeover":([0-9]+).*/\1/' build/p06_out.txt)
  if [ $got = 0 ] && [ "$ch" -ge 2 ] && cmp -s build/p06_props_clean.txt build/p06_props_after.txt && ! cmp -s build/p06_props_clean.txt build/p06_props_dirty.txt
  then echo "CASE ok: inherited plane properties reset at takeover ($ch changes; properties as before they were dirtied)"
  else echo "CASE FAIL: inherited plane properties (exit $got, changes $ch)"; cat build/p06_props_dirty.txt build/p06_props_after.txt | cut -c1-300; fails=$((fails+1)); $R setprops 0xffff 0 1 0 > /dev/null; fi
else echo "CASE FAIL: $R not built"; fails=$((fails+1)); fi
echo "scene_present_cases failures: $fails"
exit $fails
