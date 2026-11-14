#!/bin/bash
# Usage: tests/redteam-p0-5/reset_coexist.sh <label> <renderer> <delay_s> <cmd...>   (inside wrap.sh + traced.sh)
# Starts <renderer>, and <delay_s> later runs cmd (one that gets the GPU reset) beside it.
label=$1; r=$2; d=$3; shift 3
$r > build/rt_reset_renderer.log 2>&1 &
t=$!
sleep $d
timeout 30 "$@"; echo "# $label cmd rc=$?"
wait $t; echo "# $label renderer rc=$?"
echo "# $label renderer: $(grep 'FPS:' build/rt_reset_renderer.log | tr '\n' ' ') $(grep 'Measured\|PASS\|FAIL\|rror' build/rt_reset_renderer.log | tr '\n' ' ')"
