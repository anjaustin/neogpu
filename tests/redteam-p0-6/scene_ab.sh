#!/bin/bash
# Usage: tests/redteam-p0-6/scene_ab.sh [frames=200] [batches=8] [shadows=1] [reflect=1] [env for run B ...]
#        (from repo root, inside tests/redteam-p0-5/wrap.sh; PROBE names another binary)
# Do the cuts change any pixel? Runs tools/p0_6_shade_probe with no cuts (A)
# and with all of them (B: the probe's default, plus any environment given) and compares the hash
# of every frame's image as the GPU wrote it. Also B twice: the same hashes
# show the kernel is deterministic. Exit 0 when every frame is identical.
N=${1:-200}; B=${2:-8}; SH=${3:-1}; RF=${4:-1}
for i in 1 2 3 4; do [ $# -gt 0 ] && shift; done
P=${PROBE:-tools/p0_6_shade_probe}
hashes() { grep -o '"frame":[0-9]*\|"gpu_image_hash":"[0-9a-f]*"' "$1" | paste - -; }
P06_CULL=0 $P $B $N $SH $RF > build/p06_ab_a.txt 2>&1; ra=$?
env "$@" $P $B $N $SH $RF > build/p06_ab_b.txt 2>&1; rb=$?
env "$@" $P $B $N $SH $RF > build/p06_ab_b2.txt 2>&1; rb2=$?
hashes build/p06_ab_a.txt > build/p06_ab_a.h; hashes build/p06_ab_b.txt > build/p06_ab_b.h; hashes build/p06_ab_b2.txt > build/p06_ab_b2.h
na=$(wc -l < build/p06_ab_a.h); differ=$(diff build/p06_ab_a.h build/p06_ab_b.h | grep -c '^<'); rep=$(diff build/p06_ab_b.h build/p06_ab_b2.h | grep -c '^<')
echo "{\"frames\":$na,\"batches\":$B,\"shadows\":$SH,\"reflection\":$RF,\"run_b_env\":\"$*\",\"exit_no_cuts\":$ra,\"exit_all_cuts\":$rb,\"exit_all_cuts_again\":$rb2,\"frames_whose_image_differs_between_no_cuts_and_all_cuts\":$differ,\"frames_that_differ_between_two_runs_with_all_cuts\":$rep}"
grep scene_summary build/p06_ab_a.txt build/p06_ab_b.txt | cut -c1-700
[ "$na" = "$N" ] && [ $differ = 0 ] && [ $rep = 0 ] && [ $ra = 0 ] && [ $rb = 0 ] && [ $rb2 = 0 ]
