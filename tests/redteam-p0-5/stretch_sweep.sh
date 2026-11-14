#!/bin/bash
# Usage: tests/redteam-p0-5/stretch_sweep.sh <evidence-dir> <batches> [renderer]   (from repo root)
# For each stretch length (arithmetic-only iterations between two thread
# switches, 32 ns each): tools/p0_5_spin_probe variant 4 as 20 ms jobs back to
# back, 2 s alone and then 5 s beside the renderer under reset_trace.bt.
E=$1; b=$2; r=${3:-./tests/test_08_raycast_orbit}
for L in 1 4 16 64 256 1024 4096 16384; do
  outer=$(( 20000000 / (L * 32 + 107) ))
  tests/redteam-p0-5/wrap.sh $E/stretch_b$b.txt tools/p0_5_spin_probe 4 $outer $b 2s $L
  tests/redteam-p0-5/wrap.sh $E/stretch_b$b.txt tests/redteam-p0-5/traced.sh $E/stretch_b${b}_L$L.trace \
    tests/redteam-p0-5/load_coexist.sh L$L $r tools/p0_5_spin_probe 4 $outer $b 5s $L
  python3 tests/redteam-p0-5/reset_trace.py $E/stretch_b${b}_L$L.trace 500 > $E/stretch_b${b}_L$L.summary.json
  python3 -c "
import json;d=json.load(open('$E/stretch_b${b}_L$L.summary.json'))
print('# L=$L', json.dumps(d['render_jobs_per_s_while_submitting'].get('p0_5_spin_probe')), 'resets', d['resets'])" | tee -a $E/stretch_b$b.txt
done
