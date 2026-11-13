#!/bin/bash
# Hold-cutoff sweep (docs/audit/HARDWARE_HANDOFF_P0.md, P0-1 finding 7):
# at each load, the present path with frames held (HOLD_PERMILLE=250, the
# old cutoff) and never held (1000000), with the neogpu-kworker-rt service on
# and off. Run from the repo root with sudo; appends JSON lines to $1.
# SPIN=1 keeps every CPU busy (one spinning process per core, as
# test_present_pacing's load phase does).
# Usage: sudo [SPIN=1] [SERVICES="on off"] tests/hold_cutoff_sweep.sh <out.jsonl> [runs=3] [frames=1800] [loads...]
set -u
out=$1; runs=${2:-3}; frames=${3:-1800}; shift 3 2>/dev/null || shift $#
loads=${*:-0.25 0.28 0.30 0.32 0.34 0.36 0.38 0.40 0.42}
was_active=$(systemctl is-active neogpu-kworker-rt)
restore() { [ "$was_active" = active ] && systemctl start neogpu-kworker-rt; }
spin=()
stop_spin() { for p in "${spin[@]}"; do kill "$p" 2>/dev/null; done; wait 2>/dev/null; }
trap 'stop_spin; restore' EXIT
if [ "${SPIN:-0}" = 1 ]; then
  for c in $(seq 1 "$(nproc)"); do (while :; do :; done) & spin+=($!); done
fi
for svc in ${SERVICES:-on off}; do
  if [ $svc = on ]; then systemctl start neogpu-kworker-rt; else systemctl stop neogpu-kworker-rt; fi
  sleep 1
  for r in $(seq 1 "$runs"); do
    for f in $loads; do
      for h in 250 1000000; do
        line=$(HOLD_PERMILLE=$h tests/p0_1_redteam2_newest "$f" "$frames") || continue
        echo "{\"kworker_rt\":\"$svc\",\"spin\":${SPIN:-0},\"run\":$r,${line#\{}" >> "$out"
      done
    done
  done
done
