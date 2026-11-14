#!/bin/bash
# Usage: tests/redteam-p0-5/long_coexist.sh <batches> <seconds>   (inside wrap.sh, from repo root)
# A soak of random checked mailbox requests while the renderer runs back to back.
b=$1; s=$2
tests/redteam-p0-5/rt_mailbox soak $b $s > build/rt_soak.log 2>&1 &
m=$!
n=0
while kill -0 $m 2>/dev/null; do
  ./tests/test_08_raycast_orbit > build/rt_t08.log 2>&1
  n=$((n+1)); echo "# renderer run $n: $(grep 'Measured' build/rt_t08.log)"
done
wait $m; echo "# soak rc=$?"; cat build/rt_soak.log
