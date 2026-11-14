#!/bin/bash
# Usage: tests/redteam-p0-5/coexist.sh <label> <cmd...>   (inside wrap.sh, from repo root)
# Runs cmd while tests/test_08_raycast_orbit renders (10 s); prints the renderer's FPS.
label=$1; shift
./tests/test_08_raycast_orbit > build/rt_t08.log 2>&1 &
t=$!
sleep 2
if [ $# -gt 0 ]; then timeout 30 "$@"; echo "# $label rc=$?"; fi
wait $t
echo "# $label renderer: $(grep 'FPS:' build/rt_t08.log | tail -1) $(grep -c 'FPS:' build/rt_t08.log) samples; $(grep 'Measured' build/rt_t08.log)"
