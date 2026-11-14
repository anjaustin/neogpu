#!/bin/bash
# Usage: tests/redteam-p0-6/present_cases.sh [probe-binary]   (from repo root, as root, inside tests/redteam-p0-5/wrap.sh)
# Adversarial runs of tools/p0_6_present_probe. For each: the probe's JSON
# line, its exit code, and whether the display state (the kernel's own dump
# of CRTCs and planes) is the same after as before. Prints "CASE ok|FAIL".
P=${1:-tools/p0_6_present_probe}   # argument: a mutant build, to show a case fails
state() { grep -E "^plane|^crtc|fb=|crtc=|enable=|active=|allocated by|crtc-pos|src-pos|rotation|alpha|blend" /sys/kernel/debug/dri/1/state; }
fails=0
run() { # name expected_exit cmd...
  name=$1; want=$2; shift 2
  state > build/p06_before.txt
  "$@" > build/p06_out.txt 2>&1; got=$?
  sleep 0.3
  state > build/p06_after.txt
  same=yes; cmp -s build/p06_before.txt build/p06_after.txt || same=NO
  sed 's/^/    /' build/p06_out.txt | cut -c1-640
  if [ $got = $want ] && [ $same = yes ]; then echo "CASE ok: $name (exit $got, display state unchanged)"
  else echo "CASE FAIL: $name (exit $got, wanted $want; display state unchanged: $same)"; diff build/p06_before.txt build/p06_after.txt | head -10; fails=$((fails+1)); fi
}
run "3 s, 8 batches, every 10th frame checked" 0 $P 3 8 10
run "3 s, 1 batch" 0 $P 3 1 10
run "2 and 4 batches" 0 sh -c "$P 1 2 10 && $P 1 4 10"
run "wrong frame number given to the job: the check must fail" 1 $P 2 8 10 1
run "one loop pass too many: the guard rows must catch it" 1 $P 2 8 10 3
run "wrong frame number in frames 2..11 only: the memory check must catch it" 1 $P 2 8 5 4
run "framebuffer object starts 16 rows early: the scanout capture must catch it" 1 $P 2 8 10 5
run "stopped by its own SIGINT after 1 s" 0 $P 5 8 10 2
run "SIGTERM from outside after 1.5 s" 0 sh -c "$P 5 8 10 & p=\$!; sleep 1.5; kill -TERM \$p; wait \$p"
run "SIGINT from outside during start-up" 0 sh -c "$P 5 8 10 & p=\$!; sleep 0.02; kill -INT \$p; wait \$p"
run "a second copy while the first holds the display: refused" 1 sh -c "$P 3 8 10 > build/p06_first.txt & p=\$!; sleep 1; $P 1 8 10; r=\$?; wait \$p; cat build/p06_first.txt; exit \$r"
run "bad arguments: seconds" 2 $P 0 8 10
run "bad arguments: batches 3" 2 $P 1 3 10
run "bad arguments: batches 0" 2 $P 1 0 10
run "every frame checked" 0 $P 2 8 1
run "odd check interval" 0 $P 2 8 7
run "SIGKILL after 1.5 s: the kernel must give the console back" 137 sh -c "$P 5 8 10 & p=\$!; sleep 1.5; kill -KILL \$p; wait \$p"
# a previous client left the plane the console shows at half alpha, nearest
# filter and coverage blending: the takeover must put the defaults back
R=tests/redteam-p0-3/rt_scaled
if [ -x $R ]; then
  $R props > build/p06_props_clean.txt
  $R setprops 0x8000 1 1 1 > build/p06_props_dirty.txt
  $P 1 8 10 > build/p06_out.txt 2>&1; got=$?
  $R props > build/p06_props_after.txt
  sed 's/^/    /' build/p06_out.txt | cut -c1-640
  ch=$(sed -E 's/.*"plane_changes_at_takeover":([0-9]+).*/\1/' build/p06_out.txt)
  if [ $got = 0 ] && [ "$ch" -ge 2 ] && cmp -s build/p06_props_clean.txt build/p06_props_after.txt && ! cmp -s build/p06_props_clean.txt build/p06_props_dirty.txt
  then echo "CASE ok: inherited plane properties reset at takeover ($ch changes; properties as before they were dirtied)"
  else echo "CASE FAIL: inherited plane properties (exit $got, changes $ch)"; cat build/p06_props_dirty.txt build/p06_props_after.txt | cut -c1-300; fails=$((fails+1)); $R setprops 0xffff 0 1 0 > /dev/null; fi
else echo "CASE FAIL: $R not built"; fails=$((fails+1)); fi
echo "present_cases failures: $fails"
exit $fails
