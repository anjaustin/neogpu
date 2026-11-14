#!/bin/bash
# Usage: tests/redteam-p0-6/scene_mutants.sh <evidence-file> [one mutant name]   (from repo root)
# Breaks the checks of tools/p0_6_shade_probe (and tools/p0_6_scene.h) one at a time and runs
# scene_cases.sh against the broken build: a mutant is "caught" when the
# cases report at least one failure. The control (no change) must pass.
out=$1; only=$2; d=build/p06_scene_mutants; mkdir -p $d
caught=0; missed=0
mk() { # name noab sed-expression description
  if [ -n "$only" ] && [ "$only" != "$1" ]; then return; fi
  # the probe is two files: the edit is applied to both, and must change one of them
  mkdir -p $d/$1.src
  sed -E "$3" tools/p0_6_shade_probe.c > $d/$1.src/p0_6_shade_probe.c
  sed -E "$3" tools/p0_6_scene.h > $d/$1.src/p0_6_scene.h
  if cmp -s $d/$1.src/p0_6_shade_probe.c tools/p0_6_shade_probe.c && cmp -s $d/$1.src/p0_6_scene.h tools/p0_6_scene.h && [ "$1" != m0_control ]; then echo "$1: EDIT DID NOT APPLY" | tee -a $out; missed=$((missed+1)); return; fi
  gcc -O2 -ffp-contract=off -Wall -Wextra -Iinclude $d/$1.src/p0_6_shade_probe.c src/hs_qpu.o src/hs_v3d.o -lm -o $d/$1 || { echo "$1: BUILD FAILED" | tee -a $out; missed=$((missed+1)); return; }
  tests/redteam-p0-5/wrap.sh $d/$1.txt env NOAB=$2 tests/redteam-p0-6/scene_cases.sh $d/$1 8 > $d/$1.out 2>&1
  n=$(grep -c "CASE FAIL" $d/$1.out)
  if [ "$1" = m0_control ]; then
    if [ $n = 0 ]; then echo "$1: control passes [$4]" | tee -a $out; else echo "$1: CONTROL FAILED" | tee -a $out; missed=$((missed+1)); fi
  elif [ $n -gt 0 ]; then echo "$1: caught ($n cases) [$4]" | tee -a $out; grep "CASE FAIL" $d/$1.out | sed 's/^/    /' | cut -c1-150 | head -4 | tee -a $out; caught=$((caught+1))
  else echo "$1: NOT CAUGHT [$4]" | tee -a $out; missed=$((missed+1)); fi
}
mk m0_control "" 's/XXXX_NO_MATCH//' "no change"
mk m1_no_compare 1 's/if \(gpu\[i\] == img\[i\]\) continue;/continue;/' "pixels are not compared with the reference"
mk m2_no_guard_check 1 's/guard_hit \+= \(all\[k\] != GUARD\) \+ \(gpu\[\(size_t\)W \* H \+ k\] != GUARD\);/(void)k;/' "guard rows are not compared"
mk m3_unwritten_ignored 1 's/int ok = !rc \&\& !guard_hit \&\& !unwritten \&\&/int ok = !rc \&\& !guard_hit \&\&/' "an unwritten pixel does not fail the frame"
mk m4_exit_ignores_failures 1 's/return rc \? 3 : failures != 0;/return rc ? 3 : 0;/' "failed frames do not fail the run"
mk m5_allowance_one_percent 1 's/wrong <= W \* H \/ 10000;/wrong <= W * H \/ 100;/' "the allowance is 1 % of the frame, not 0.01 %"
mk m6_hash_ignores_pixels "" 's/hash = \(hash \^ gpu\[i\]\) \* 1099511628211ull;/hash = hash * 1099511628211ull;/' "the image hash does not depend on the image"
mk m7_no_poison 1 's/memset\(gpu, 0xEE, \(size_t\)W \* H \* 4\);/if (f == 0) memset(gpu, 0xEE, (size_t)W * H * 4);/' "the image is not cleared between frames: a stale frame can pass"
echo "p0-6 scene mutants caught: $caught; problems (not caught / build / control): $missed" | tee -a $out
