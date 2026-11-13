#!/bin/bash
# Gate mutation for the hold cutoff (docs/audit/HARDWARE_HANDOFF_P0.md):
# builds a copy of the current tests/test_present_pacing.c whose
# gfx.hold_permille comes from env HOLD_PERMILLE, then runs it per value.
# A correct gate fails for cutoffs it should reject.
# Usage (repo root): tests/redteam-hold-cutoff/gate_mutants.sh <out.jsonl> <permille...>
set -u
out=$1; shift
R=tests/redteam-hold-cutoff
CF="-O3 -march=armv8-a+simd -mtune=cortex-a72 -ffast-math -funroll-loops -Wall -Wextra -DNDEBUG -Iinclude -I/usr/include/libdrm"
sed 's|if (frames < 60 \|\| hs_graphics_init(&gfx) != 0) {|if (frames < 60 \|\| hs_graphics_init(\&gfx) != 0 \|\| (getenv("HOLD_PERMILLE") \&\& (gfx.hold_permille = (u32)strtoul(getenv("HOLD_PERMILLE"), NULL, 10), 0))) {|' \
  tests/test_present_pacing.c > $R/gate_mut2.c
grep -q 'getenv("HOLD_PERMILLE")' $R/gate_mut2.c || { echo "mutation did not apply" >&2; exit 1; }
gcc $CF $R/gate_mut2.c -lm -lGLESv2 -lgbm -ldrm -lEGL -lpthread -o $R/gate_mut2 || exit 1
gov() { for g in /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_governor; do echo "$1" | sudo tee $g >/dev/null; done; }
trap 'gov ondemand' EXIT; gov performance
for h in "$@"; do
  echo -n "{\"mutant\":$h," >> "$out"
  sudo env HOLD_PERMILLE=$h flock build/gpu.lock $R/gate_mut2 600 | sed 's/^{//' >> "$out"
  echo "mutant $h rc=${PIPESTATUS[0]}"
done
