#!/bin/bash
# Usage: tests/redteam-p0-6/scene_cases.sh [probe-binary] [frames=24]   (from repo root, inside tests/redteam-p0-5/wrap.sh)
# Adversarial runs of tools/p0_6_shade_probe (the whole scene by QPU compute,
# checked against the CPU reference). Every cut alone and together must give
# a passing run; every built-in fault must give a failing one. Prints
# "CASE ok|FAIL" per case and the number of failures. NOAB=1 leaves out the
# two exact comparisons (scene_ab.sh, 6 runs of 200 frames).
P=${1:-tools/p0_6_shade_probe}; N=${2:-24}
fails=0
run() { # name expected_exit env... -- args...
  name=$1; want=$2; shift 2
  envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  env "${envs[@]}" $P "$@" > build/p06_scene_out.txt 2>&1; got=$?
  bad=$(grep -c '"rc":0,"ok":false' build/p06_scene_out.txt); tot=$(grep -c '"frame":' build/p06_scene_out.txt)
  worst=$(grep -o '"mismatching":[0-9]*' build/p06_scene_out.txt | cut -d: -f2 | sort -n | tail -1)
  note="exit $got, $bad of $tot frames failed their check, most pixels different in a frame: ${worst:-none}"
  # exit 1 must mean a frame's job ran and its check failed (the probe exits 3 when the run itself fails)
  if [ $got = $want ] && { [ $want != 1 ] || [ $bad -gt 0 ]; } && { [ $want != 0 ] || [ $tot -gt 0 ]; }; then echo "CASE ok: $name ($note)"
  else echo "CASE FAIL: $name (wanted exit $want; $note)"; tail -2 build/p06_scene_out.txt | cut -c1-400; fails=$((fails+1)); fi
}
ab() { # name expected(same|differ) frames batches shadows reflect [env for run B]
  name=$1; want=$2; shift 2
  PROBE=$P tests/redteam-p0-6/scene_ab.sh "$@" > build/p06_scene_out.txt 2>&1; got=$?
  line=$(head -1 build/p06_scene_out.txt | cut -c1-330)
  if [ $want = same ] && [ $got = 0 ]; then echo "CASE ok: $name ($line)"
  elif [ $want = differ ] && [ $got != 0 ] && grep -q '"exit_all_cuts":0' build/p06_scene_out.txt && ! grep -q '"frames_whose_image_differs_between_no_cuts_and_all_cuts":0' build/p06_scene_out.txt; then echo "CASE ok: $name ($line)"
  elif [ $want = differ2 ] && [ $got != 0 ] && ! grep -q '"frames_whose_image_differs_between_no_cuts_and_all_cuts":0' build/p06_scene_out.txt; then echo "CASE ok: $name ($line)"
  else echo "CASE FAIL: $name (wanted $want)"; head -1 build/p06_scene_out.txt; fails=$((fails+1)); fi
}
for c in 0 1 2 3 4 5 6 7 12 13 14 15; do run "cuts $c, 8 batches" 0 P06_CULL=$c -- 8 $N; done
run "lists with one cell per patch" 0 P06_CELLS=1 -- 8 $N
run "lists with four cells per side" 0 P06_CELLS=4 -- 8 $N
for b in 1 2 4; do run "all cuts, $b batches" 0 -- $b $N; done
run "no cuts, 1 batch" 0 P06_CULL=0 -- 1 4
run "no shadows" 0 -- 8 $N 0 1
run "no reflection" 0 -- 8 $N 1 0
run "neither" 0 -- 8 $N 0 0
run "one frame (t = 1)" 0 -- 8 1
run "fault 1: second shade skipped when any reflected ray misses" 1 P06_FAULT=1 -- 8 $N
run "fault 2: constant stream not moved on the path around the second shade" 1 P06_FAULT=2 -- 8 $N
run "fault 3: ground that may be shadowed skips the shadow test" 1 P06_FAULT=3 -- 8 $N
run "fault 4: one row too many per batch (guard rows)" 1 P06_FAULT=4 -- 8 $N
run "fault 5: region sort ignores reflections" 1 P06_FAULT=5 -- 8 $N
run "fault 6: region sort ignores covering spheres" 1 P06_FAULT=6 -- 8 $N
run "fault 7: region sort ignores shadows" 1 P06_FAULT=7 -- 8 $N
run "fault 8: rows just below the horizon treated as sky" 1 P06_FAULT=8 -- 8 $N
run "fault 9: short path reads the long path's constants (no lists)" 1 P06_FAULT=9 P06_CULL=7 -- 8 $N
run "fault 14: the primary list lacks its first sphere" 1 P06_FAULT=14 -- 8 $N
run "fault 15: the reflected ray's list is empty" 1 P06_FAULT=15 -- 8 $N
run "fault 10: batches start two rows apart" 1 P06_FAULT=10 -- 8 $N
run "fault 12: one pixel left unwritten" 1 P06_FAULT=12 -- 8 2
run "fault 13: sky rows not stored after the first frame" 1 P06_FAULT=13 -- 8 3
run "region tests with a negative margin" 1 P06_MARGIN=-0.01 -- 8 $N
run "bad arguments: batches 3" 2 -- 3 1
run "bad arguments: batches 0" 2 -- 0 1
run "bad arguments: frames 0" 2 -- 8 0
run "bad arguments: frames not a number" 2 -- 8 x
run "bad arguments: cuts 16" 2 P06_CULL=16 -- 8 1
run "bad arguments: lists without the region sort (cuts 9)" 2 P06_CULL=9 -- 8 1
run "bad arguments: margin not a number" 2 P06_MARGIN=nan -- 8 1
run "a constant table that would overflow: refused at build" 3 P06_PAD=1900 P06_PAD_KIND=1 P06_BUILD_ONLY=1 -- 8 1
# the exact comparison: the cuts change no pixel, for every region shape and variant;
# and it sees a bound that is a little too tight, which the reference check lets through
if [ -z "$NOAB" ]; then
  ab "8 batches: 200 frames with all cuts are identical to 200 without" same 200 8 1 1
  ab "4 batches, 48 frames" same 48 4 1 1
  ab "2 batches, 48 frames" same 48 2 1 1
  ab "1 batch, 48 frames" same 48 1 1 1
  ab "no shadows, 48 frames" same 48 8 0 1
  ab "no reflection, 48 frames" same 48 8 1 0
  ab "without lists (cuts 7), 48 frames" same 48 8 1 1 P06_CULL=7
  ab "fault 11 (reflection bound of the region sort too tight, no lists) passes the reference check and is seen by the exact comparison" differ 200 8 1 1 P06_FAULT=11 P06_CULL=7
  ab "fault 16 (no list for shadows of the reflected hit): seen by the exact comparison" differ2 200 8 1 1 P06_FAULT=16
fi
echo "scene_cases failures: $fails"
exit $fails
