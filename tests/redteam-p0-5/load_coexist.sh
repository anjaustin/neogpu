#!/bin/bash
# Usage: tests/redteam-p0-5/load_coexist.sh <label> <renderer> <cmd...>   (inside wrap.sh + traced.sh)
# Runs cmd (give it about 5 s) from 2.5 s into the renderer's run; the trace
# shows the renderer before, during and after.
label=$1; r=$2; shift 2
$r > build/rt_load_renderer.log 2>&1 &
t=$!
sleep 2.5
timeout 30 "$@"; echo "# $label cmd rc=$?"
wait $t; echo "# $label renderer rc=$? $(grep 'Measured\|PASS\|FAIL' build/rt_load_renderer.log | tr '\n' ' ')"
