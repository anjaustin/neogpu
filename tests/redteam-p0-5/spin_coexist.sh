#!/bin/bash
# Usage: tests/redteam-p0-5/spin_coexist.sh <batches> [renderer]   (inside wrap.sh + traced.sh)
# Beside the renderer, one tools/p0_5_spin_probe job of each loop body. Bodies
# 0 and 3 run at their speed alone (32.1 / 64.1 ns per iteration): sized to
# 250 ms. Bodies 1 and 2 (thread switches) ran 4.1x slower beside the renderer
# than alone (138.8 / 96.1 ns): sized to about 250 ms there. At 250 ms alone
# they passed 1 s and the kernel reset them (2026-11-17, four resets).
b=$1; r=${2:-./tests/test_08_raycast_orbit}
$r > build/rt_spin_renderer.log 2>&1 &
t=$!
sleep 2.5
for vi in "0 7800000" "1 440000" "2 640000" "3 3900000"; do tools/p0_5_spin_probe $vi $b; sleep 1; done
wait $t; echo "# renderer rc=$? $(grep 'Measured\|PASS\|FAIL' build/rt_spin_renderer.log | tr '\n' ' ')"
