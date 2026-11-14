#!/bin/bash
# Usage: tests/redteam-p0-6/scene_coexist.sh [iters=200] [frames=200]   (from repo root, as root, inside tests/redteam-p0-5/wrap.sh)
# The scene kernel beside a second GPU client: tools/p0_5_gl_probe, an
# offscreen GLES renderer (512x512, `iters` loop passes per pixel) that
# checks every frame it draws. Beside it the kernel's batches really are
# switched out at their thread switches. Measures, alone and together:
#   - is every frame of the scene still exactly the image it is alone
#     (image hashes of tools/p0_6_shade_probe), and correct by the reference;
#   - what each costs the other (job time, frame rates), offscreen and on
#     the panel with every frame flipped at once;
#   - are the renderer's frames still right.
# Prints JSON-ish lines and "COEXIST ok|FAIL".
I=${1:-200}; N=${2:-200}
S=tools/p0_6_shade_probe; P=tools/p0_6_scene_present_probe; G=tools/p0_5_gl_probe
fails=0
hashes() { grep -o '"gpu_image_hash":"[0-9a-f]*"' "$1"; }
gl() { grep '"gl_probe":"done"' "$1" | cut -c1-260; }
echo "# renderer alone, 10 s"; $G 10 512 $I > build/p06_co_gl_alone.txt 2>&1; gl build/p06_co_gl_alone.txt
echo "# scene alone, $N frames"; $S 8 $N > build/p06_co_s_alone.txt 2>&1; ra=$?; grep scene_summary build/p06_co_s_alone.txt | cut -c1-520
# offscreen, together: the renderer runs for the whole of the scene probe's run
stdbuf -oL $G 3600 512 $I > build/p06_co_gl_with.txt 2>&1 & g=$!
sleep 2
echo "# scene beside the renderer, $N frames"; $S 8 $N > build/p06_co_s_with.txt 2>&1; rb=$?; grep scene_summary build/p06_co_s_with.txt | cut -c1-520
kill -TERM $g; wait $g 2>/dev/null
grep -c '"frame":' build/p06_co_gl_with.txt | sed 's/^/# renderer frames reported slow, wrong or in error while beside the scene probe: /'
hashes build/p06_co_s_alone.txt > build/p06_co_a.h; hashes build/p06_co_s_with.txt > build/p06_co_b.h
na=$(wc -l < build/p06_co_a.h); differ=$(diff build/p06_co_a.h build/p06_co_b.h | grep -c '^[<>]')
bad=$(grep -c '"ok":false' build/p06_co_s_with.txt)
echo "{\"frames\":$na,\"exit_alone\":$ra,\"exit_beside\":$rb,\"frames_failing_the_reference_check_beside\":$bad,\"frames_whose_image_differs_from_the_run_alone\":$differ}"
if [ $ra = 0 ] && [ $rb = 0 ] && [ "$na" = "$N" ] && [ $differ = 0 ]; then echo "COEXIST ok: offscreen, every frame beside the renderer is the image rendered alone"
else echo "COEXIST FAIL: offscreen"; fails=$((fails+1)); fi
# the renderer inside the scene probe's run: its frame rate beside compute, and its own frame check
echo "# renderer (10 s) inside a 16 s offscreen scene loop"
( for i in 1 2 3 4 5 6; do $S 8 $N > /dev/null 2>&1; done ) & s=$!
sleep 3; $G 10 512 $I > build/p06_co_gl_in.txt 2>&1; rg=$?; gl build/p06_co_gl_in.txt
kill $s 2>/dev/null; pkill -f "p0_6_shade_probe 8 $N" 2>/dev/null; wait $s 2>/dev/null
if [ $rg = 0 ] && grep -q '"wrong_frames":0' build/p06_co_gl_in.txt; then echo "COEXIST ok: the renderer's frames are right beside the scene probe"
else echo "COEXIST FAIL: renderer beside the scene probe (exit $rg)"; fails=$((fails+1)); fi
# on the panel, every frame flipped at once
echo "# panel alone, 10 s"; $P 10 8 2 12 > build/p06_co_p_alone.txt 2>&1; pa=$?; cut -c1-900 build/p06_co_p_alone.txt
stdbuf -oL $G 3600 512 $I > build/p06_co_gl_with2.txt 2>&1 & g=$!
sleep 2
echo "# panel beside the renderer, 10 s"; $P 10 8 2 12 > build/p06_co_p_with.txt 2>&1; pb=$?; cut -c1-900 build/p06_co_p_with.txt
kill -TERM $g; wait $g 2>/dev/null
ha=$(grep -o '"image_hashes":\[[^]]*\]' build/p06_co_p_alone.txt); hb=$(grep -o '"image_hashes":\[[^]]*\]' build/p06_co_p_with.txt)
if [ $pa = 0 ] && [ $pb = 0 ] && [ -n "$ha" ] && [ "$ha" = "$hb" ]; then echo "COEXIST ok: on the panel, the 12 checked frames beside the renderer are the frames rendered alone"
else echo "COEXIST FAIL: panel (exit alone $pa, beside $pb)"; fails=$((fails+1)); fi
echo "# renderer (10 s) inside a 20 s panel run"
$P 20 8 2 0 > build/p06_co_p_long.txt 2>&1 & s=$!
sleep 4; $G 10 512 $I > build/p06_co_gl_in2.txt 2>&1; rg=$?; gl build/p06_co_gl_in2.txt
wait $s; pl=$?
if [ $rg = 0 ] && [ $pl = 0 ] && grep -q '"wrong_frames":0' build/p06_co_gl_in2.txt; then echo "COEXIST ok: the renderer's frames are right beside the panel run"
else echo "COEXIST FAIL: renderer beside the panel run (exit $rg, panel $pl)"; fails=$((fails+1)); fi
echo "scene_coexist failures: $fails"
exit $fails
