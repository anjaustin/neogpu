#!/bin/bash
# Usage: tests/redteam-p0-3/wrap.sh <evidence-file> <cmd...>   (from repo root)
# Runs cmd under the GPU lock with the performance governor; appends output to the
# evidence file plus GPU reset / HVS underrun counts before and after.
out=$1; shift
gov() { for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo $1 | sudo tee $c >/dev/null; done; }
trap 'gov ondemand' EXIT
r0=$(sudo dmesg | grep -ci "v3d.*reset"); u0=$(sudo cat /sys/kernel/debug/dri/1/hvs_underrun)
gov performance
sudo flock build/gpu.lock "$@" 2>&1 | tee -a "$out"
r1=$(sudo dmesg | grep -ci "v3d.*reset"); u1=$(sudo cat /sys/kernel/debug/dri/1/hvs_underrun)
echo "# cmd=[$*] gpu_resets $r0->$r1 hvs_underrun $u0->$u1" | tee -a "$out"
