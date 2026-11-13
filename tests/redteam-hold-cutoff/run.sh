#!/bin/bash
# Hold-cutoff red team driver. Run as root from the repo root.
# Usage: sudo tests/redteam-hold-cutoff/run.sh <out.jsonl> <plan>
# Plan lines: <svc on|off> <spin 0|1> <env,comma,separated|-> <fraction> <frames>
# Each probe run is under build/gpu.lock; governor=performance during the run,
# restored to ondemand on exit; kworker-rt restored to active on exit.
set -u
out=$1; plan=$2
gov() { for g in /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_governor; do echo "$1" > "$g"; done; }
spin=()
stop_spin() { for p in "${spin[@]}"; do kill "$p" 2>/dev/null; done; spin=(); }
cleanup() { stop_spin; systemctl start neogpu-kworker-rt; gov ondemand; }
trap cleanup EXIT
gov performance
echo "{\"resets_before\":$(dmesg | grep -ci 'v3d.*reset')}" >> "$out"
while read -r svc sp envs frac frames; do
  [ -z "$svc" ] && continue; case $svc in \#*) continue;; esac
  if [ "$svc" = on ]; then systemctl start neogpu-kworker-rt; else systemctl stop neogpu-kworker-rt; fi
  stop_spin
  if [ "$sp" = 1 ]; then for c in 1 2 3 4; do (while :; do :; done) & spin+=($!); done; fi
  sleep 0.5
  e=(); [ "$envs" != - ] && IFS=, read -ra e <<< "$envs"
  line=$(env "${e[@]}" flock build/gpu.lock tests/redteam-hold-cutoff/probe "$frac" "$frames") || { echo "{\"failed\":\"$svc $sp $envs $frac\"}" >> "$out"; continue; }
  echo "{\"kworker_rt\":\"$svc\",\"spin\":$sp,${line#\{}" >> "$out"
done < "$plan"
stop_spin
echo "{\"resets_after\":$(dmesg | grep -ci 'v3d.*reset')}" >> "$out"
