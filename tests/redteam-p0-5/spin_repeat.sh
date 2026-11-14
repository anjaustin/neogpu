#!/bin/bash
# Usage: tests/redteam-p0-5/spin_repeat.sh <n> <gap_s> <spin_probe args...>
# n spin jobs, gap_s apart.
n=$1; g=$2; shift 2
for i in $(seq $n); do tools/p0_5_spin_probe "$@"; sleep $g; done
