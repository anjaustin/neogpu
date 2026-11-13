#!/bin/bash
# P0-3 (docs/audit/HARDWARE_HANDOFF_P0.md): what test_08_raycast_orbit's
# GPU throughput is at reduced render sizes. Builds a copy of the test (in
# build/, an artifact) that renders offscreen at RWxRH, uncapped, and prints
# its FPS. ON_DISPLAY=1 instead presents on the panel through
# hs_graphics_init_scaled (the HVS scales to the full mode; needs sudo).
# Run from the repo root after `make build`.
# Usage: tests/p0_3_render_scale.sh <out.txt> [runs=2] [sizes...]
set -u
out=$1; runs=${2:-2}; shift 2 2>/dev/null || shift $#
sizes=${*:-800x480 600x360 400x240 266x160}
mkdir -p build/p0_3
if [ "${ON_DISPLAY:-0}" = 1 ]; then init='hs_graphics_init_scaled(\&gfx, RW, RH) != 0'; tag=disp
else init='(hs_graphics_init_sized(\&gfx, RW, RH) != 0 || (gfx.take_display = false))'; tag=off; fi
sed "s/hs_graphics_init(&gfx) != 0/$init/" \
  tests/test_08_raycast_orbit.c > build/p0_3/orbit_sized.c
grep -q "(&gfx, RW, RH)" build/p0_3/orbit_sized.c || { echo "substitution failed" >&2; exit 1; }
for s in $sizes; do
  w=${s%x*}; h=${s#*x}
  gcc -O3 -march=armv8-a+simd -mtune=cortex-a72 -Iinclude -Itests -I/usr/include/libdrm -DHS_GRAPHICS_SHELVED_SCALED -DRW=$w -DRH=$h \
    -c build/p0_3/orbit_sized.c -o build/p0_3/orbit_${tag}_$s.o &&
  gcc src/hs_core.o src/hs_nodes.o src/hs_gpu.o src/hs_fabric.o src/hs_async.o src/hs_backend_gles.o src/hs_text.o \
    build/p0_3/orbit_${tag}_$s.o -lm -lGLESv2 -lgbm -ldrm -lEGL -lpthread -o build/p0_3/orbit_${tag}_$s || exit 1
done
for r in $(seq 1 "$runs"); do
  for s in $sizes; do
    fps=$(flock build/gpu.lock build/p0_3/orbit_${tag}_$s | grep "^FPS" | tail -1)
    echo "mode=$tag size=$s run=$r last_$fps" | tee -a "$out"
  done
done
