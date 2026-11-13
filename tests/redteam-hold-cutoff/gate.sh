#!/bin/bash
# Gate mutation: run the copy of test_present_pacing with HOLD_PERMILLE overrides.
# Usage: sudo tests/redteam-hold-cutoff/gate.sh <out.jsonl> <permille...>
set -u
out=$1; shift
gov() { for g in /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_governor; do echo "$1" > "$g"; done; }
trap 'systemctl start neogpu-kworker-rt; gov ondemand' EXIT
gov performance; systemctl start neogpu-kworker-rt; sleep 0.5
for h in "$@"; do
  line=$(HOLD_PERMILLE=$h flock build/gpu.lock tests/redteam-hold-cutoff/gate_mut 2>/dev/null | tail -1)
  echo "{\"mutant_hold_permille\":$h,${line#\{}" >> "$out"
done
