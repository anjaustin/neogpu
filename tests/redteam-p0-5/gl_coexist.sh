#!/bin/bash
# Usage: tests/redteam-p0-5/gl_coexist.sh <label> <delay_s> "<gl_probe args>" <cmd...>   (inside wrap.sh + traced.sh)
# Runs tools/p0_5_gl_probe (the renderer that checks its own frames) and,
# <delay_s> later, cmd beside it. Prints the probe's lines.
label=$1; d=$2; g=$3; shift 3
tools/p0_5_gl_probe $g > build/rt_gl_probe.log 2>&1 &
t=$!
sleep $d
if [ $# -gt 0 ]; then timeout 40 "$@"; echo "# $label cmd rc=$?"; fi
wait $t; echo "# $label gl_probe rc=$?"
sed "s/^/# $label gl: /" build/rt_gl_probe.log
