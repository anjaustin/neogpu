"""ROI-gated PMU, exact diagnostic counts and bounded same-driver ablations."""
import argparse
import csv
import glob
import hashlib
import json
import os
import re
from pathlib import Path
import statistics
import subprocess
import time
from bedrock_trace import decode, selftest as trace_selftest
from bedrock_pmu import counter_rows
from bedrock_driver import roi_driver

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--reference", default="tests/evidence-comparison-final/reference", type=Path)
parser.add_argument("--phase", choices=("build", "qualify", "counts", "count-pmu", "pmu", "ablations", "trace", "all"), default="all")
parser.add_argument("--groups", help="comma-separated PMU groups for capability checks")
parser.add_argument("--pairs", type=int, default=3)
parser.add_argument("--variants", help="comma-separated build/ablation variants")
parser.add_argument("--pmu-variants", default="control", help="comma-separated uninstrumented PMU variants")
args = parser.parse_args()
if not args.output.name.startswith("evidence-") or not args.output.resolve().is_relative_to((ROOT / "tests").resolve()):
    parser.error("output must be under tests/evidence-*")
args.output.mkdir(exist_ok=args.phase != "all")
names = ["src/hs_fabric.c", "tests/comparison_native.c", "tests/comparison_rigtorp.cc", "tests/test_comparison_transport.c"]
original = {name: Path(name).read_text() for name in names}
assert hashlib.sha256((args.reference / "include/rigtorp/SPSCQueue.h").read_bytes()).hexdigest() == "62d9f783862666be5c7ca3c8fbcec2e45d9920167c4b9e689bda4957f66dd86e"
manifest_path = args.output / "manifest.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
    "source_sha256": {name: hashlib.sha256(data.encode()).hexdigest() for name, data in original.items()},
    "commands": [], "phases": {}}
if manifest['phases'].get(args.phase):
    parser.error('phase already completed; use a fresh evidence directory rather than overwrite evidence')
manifest.setdefault('tool_versions', []).append({
    'phase': args.phase,
    'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in [Path(__file__), Path('tests/bedrock_support.h'), Path('tests/bedrock_support.c'),
                                   Path('tests/comparison_transport.h'), Path('tests/bedrock_trace.py'), Path('tests/bedrock_pmu.py'), Path('tests/bedrock_driver.py'), *sorted(Path('include').glob('*.h'))]}})
if 'runtime_revision' not in manifest:
    manifest['runtime_revision'] = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()


def environment():
    result = {'wall_time_ns':time.time_ns(), 'affinity':sorted(os.sched_getaffinity(0)), 'sysfs':{}}
    for pattern in ('/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq',
                    '/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor', '/sys/class/thermal/thermal_zone*/temp'):
        for name in sorted(glob.glob(pattern)):
            try: result['sysfs'][name] = Path(name).read_text().strip()
            except OSError as error: result['sysfs'][name] = str(error)
    return result


manifest.setdefault('phase_environments', []).append({'phase':args.phase,'before':environment()})


def checked(argv, label, env=None, required=True):
    if (args.output / (label + '.json')).exists():
        raise FileExistsError('refusing to overwrite command evidence: ' + label)
    start = time.monotonic_ns()
    try:
        result = subprocess.run([str(a) for a in argv], text=True, capture_output=True, timeout=120, env=env)
        record = {"argv": [str(a) for a in argv], "exit_status": result.returncode,
                  "elapsed_ns": time.monotonic_ns() - start, "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired as error:
        record = {"argv": [str(a) for a in argv], "exit_status": None, "timeout": True,
                  "stdout": str(error.stdout), "stderr": str(error.stderr)}
    (args.output / (label + ".json")).write_text(json.dumps(record, indent=2))
    manifest["commands"].append({k: v for k, v in record.items() if k != "stdout"})
    if required:
        assert record["exit_status"] == 0, record
    return record


def edit(source, before, after):
    assert source.count(before) == 1, before
    return source.replace(before, after)


def driver(variant):
    source = roi_driver(original["tests/test_comparison_transport.c"])
    if variant == "cpu-yield":
        source = edit(source, 'producer->retries++; sched_yield();', 'producer->retries++; __asm__ volatile("yield" ::: "memory");')
        source = edit(source, 'alarm(90);', 'alarm(15); /* bounded diagnostic, not a production retry policy */')
    if variant == "prebuilt-input":
        before = '''for (u32 i = 1; i <= frames; i++) {
        HSParticipantFrame input = {.sequence = i, .count = draws};
        for (u32 j = 0; j < draws; j++)
            input.draws[j] = (HSRenderCmd){.op = HS_RC_DRAW, .a = (u8)producer->id};'''
        after = '''HSParticipantFrame input = {.count = draws};
    for (u32 j = 0; j < draws; j++)
        input.draws[j] = (HSRenderCmd){.op = HS_RC_DRAW, .a = (u8)producer->id};
    for (u32 i = 1; i <= frames; i++) {
        input.sequence = i;'''
        source = edit(source, before, after)
    if variant == "block-observer":
        source = edit(source, 'u64 observed[HS_FABRIC_PARTICIPANTS] = {0};', '''u64 observed[HS_FABRIC_PARTICIPANTS] = {0};
    HSRenderCmd expected[HS_FABRIC_PARTICIPANTS][HS_FABRIC_DRAWS] = {0};
    for (u32 i = 0; i < participants; i++) for (u32 j = 0; j < draws; j++)
        expected[i][j] = (HSRenderCmd){.op = HS_RC_DRAW, .a = (u8)i};''')
        source = edit(source, '''for (u32 j = 0; j < draws; j++) {
                HSRenderCmd expected = {.op = HS_RC_DRAW, .a = (u8)i};
                REQUIRE(memcmp(&view->draws[j], &expected, sizeof(expected)) == 0);
            }''', 'REQUIRE(memcmp(view->draws, expected[i], draws * sizeof(HSRenderCmd)) == 0);')
    return source


def instrument_native(source):
    split = 'slot->read_index' in source
    source = source.replace('#include "hs_fabric.h"', '#include "hs_fabric.h"\n#include "bedrock_support.h"', 1)
    begin = source.index('HSFabricResult hs_fabric_submit(')
    end = source.index('u32 hs_fabric_step(', begin)
    submit = source[begin:end]
    submit = edit(submit, 'HSParticipantSlot* slot = &fabric->slots[owner.slot];', 'BR_INC(attempts); BR_ADD(validated_draws, frame->count);\n    HSParticipantSlot* slot = &fabric->slots[owner.slot];')
    submit = edit(submit, 'if (pthread_mutex_trylock(&slot->mutex) != 0) return HS_FABRIC_BUSY;', 'if (pthread_mutex_trylock(&slot->mutex) != 0) { BR_INC(busy); return HS_FABRIC_BUSY; }')
    full = 'else if ((u32)(write - read) >= HS_FABRIC_QUEUE)' if split else 'else if (slot->count == HS_FABRIC_QUEUE)'
    submit = edit(submit, full + ' result = HS_FABRIC_FULL;', full + ' { BR_INC(full); result = HS_FABRIC_FULL; }')
    submit = edit(submit, 'slot->admitted = frame->sequence;', 'slot->admitted = frame->sequence; BR_INC(admitted); BR_ADD(canonical_bytes, frame->count * sizeof(HSRenderCmd));')
    source = source[:begin] + submit + source[end:]
    begin = source.index('u32 hs_fabric_step('); end = source.index('bool hs_fabric_compose(', begin)
    step = source[begin:end]
    step = edit(step, 'HSParticipantSlot* slot = &fabric->slots[i];', 'BR_INC(visits); if (i < br_participants) { BR_INC(active_visits); } else { BR_INC(inactive_visits); }\n        HSParticipantSlot* slot = &fabric->slots[i];')
    step = edit(step, 'if (pthread_mutex_trylock(&slot->mutex) != 0) continue;', 'if (pthread_mutex_trylock(&slot->mutex) != 0) { BR_INC(service_busy); continue; }')
    if split:
        # Account before releasing ring storage for immediate producer reuse.
        step = edit(step, '__atomic_store_n(&slot->read_index, read + 1, __ATOMIC_RELEASE);',
                    'BR_ADD(copy_bytes, source->count * sizeof(HSRenderCmd));\n            __atomic_store_n(&slot->read_index, read + 1, __ATOMIC_RELEASE);')
        step = edit(step, 'fabric->applied++;', 'fabric->applied++; BR_INC(applied);')
        step = edit(step, 'if (read != write) {', 'if (read == write) BR_INC(empty);\n        if (read != write) {')
    else:
        step = edit(step, 'fabric->applied++;', 'fabric->applied++; BR_INC(applied); BR_ADD(copy_bytes, source->count * sizeof(HSRenderCmd));')
    # Count emptiness before the state machine (retirement is absent in this workload).
    if not split:
        step = edit(step, 'if (slot->active && slot->count) {', 'if (!slot->count) BR_INC(empty);\n        if (slot->active && slot->count) {')
    return source[:begin] + step + source[end:]


def instrument_reference(source):
    source = source.replace('#include "comparison_transport.h"', '#include "comparison_transport.h"\n#include "bedrock_support.h"', 1)
    source = edit(source, 'auto& lane = context->lanes[index];', 'BR_INC(attempts); BR_ADD(validated_draws, input->count);\n    auto& lane = context->lanes[index];')
    source = edit(source, 'if (!lane.queue.try_emplace(*input)) return false;', 'if (!lane.queue.try_emplace(*input)) { BR_INC(full); return false; }')
    source = edit(source, 'lane.admitted = input->sequence;', 'lane.admitted = input->sequence; BR_INC(admitted); BR_ADD(canonical_bytes, input->count * sizeof(HSRenderCmd));')
    source = edit(source, 'auto& lane = context->lanes[context->cursor];', 'BR_INC(visits); if (context->cursor < br_participants) { BR_INC(active_visits); } else { BR_INC(inactive_visits); }\n        auto& lane = context->lanes[context->cursor];')
    source = edit(source, 'if (!entry) continue;', 'if (!entry) { BR_INC(empty); continue; }')
    source = edit(source, 'applied++;', 'applied++; BR_INC(applied); BR_ADD(copy_bytes, entry->frame.count * sizeof(HSRenderCmd));')
    # Read count before pop: queue storage can be reused immediately after pop.
    source = source.replace('lane.queue.pop();\n        applied++; BR_INC(applied); BR_ADD(copy_bytes, entry->frame.count * sizeof(HSRenderCmd));',
                            'BR_ADD(copy_bytes, entry->frame.count * sizeof(HSRenderCmd));\n        lane.queue.pop();\n        applied++; BR_INC(applied);')
    return source


def build(variant):
    directory = args.output / variant
    directory.mkdir()
    flags = ['-O3', '-g', '-march=armv8-a+simd', '-mtune=cortex-a72', '-ffast-math', '-funroll-loops',
             '-Wall', '-Wextra', '-Werror', '-DNDEBUG', '-Iinclude', '-Itests']
    counted = variant.startswith('counts')
    if counted: flags += ['-DBEDROCK_COUNTS']
    if variant.endswith('-asan'): flags += ['-O1', '-fno-omit-frame-pointer', '-fsanitize=address,undefined']
    if variant.endswith('-tsan'): flags += ['-O1', '-fno-omit-frame-pointer', '-fsanitize=thread']
    objects = []
    for name in names + ['tests/bedrock_support.c']:
        source = driver(variant) if name.endswith('test_comparison_transport.c') else Path(name).read_text()
        if counted and name.endswith('hs_fabric.c'): source = instrument_native(source)
        if counted and name.endswith('comparison_rigtorp.cc'): source = instrument_reference(source)
        destination = directory / Path(name).name
        destination.write_text(source)
        obj = destination.with_suffix('.o'); cpp = name.endswith('.cc')
        checked(['g++' if cpp else 'gcc', *flags, *(['-std=c++23', '-isystem', args.reference / 'include'] if cpp else []),
                 '-c', destination, '-o', obj], variant + '-compile-' + Path(name).stem)
        objects.append(obj)
    binary = directory / 'comparison'
    checked(['g++', *flags, *objects, '-lpthread', '-lm', *(['-Wl,--wrap=sched_yield'] if counted else []), '-o', binary], variant + '-link')
    checked([binary, '--selftest'], variant + '-selftest')


def run(variant, implementation, label, frames=1000000, draws=1, perf_events=None, trace=False, allow_failure=False):
    binary = args.output / variant / 'comparison'
    command = [binary, implementation, 1, frames, draws]
    env = None
    before = environment()
    if perf_events or trace:
        ctl = args.output / (label + '-ctl'); ack = args.output / (label + '-ack')
        os.mkfifo(ctl); os.mkfifo(ack)
        env = {**os.environ, 'BEDROCK_CTL': str(ctl), 'BEDROCK_ACK': str(ack)}
        prefix = ['sudo', '-n', 'env', 'BEDROCK_CTL=' + str(ctl), 'BEDROCK_ACK=' + str(ack), 'perf']
        control = ['--delay=-1', '--control=fifo:' + str(ctl) + ',' + str(ack)]
        if trace:
            command = [*prefix, 'record', '-a', '-C', '0,1', *control, '-m', '1024', '-o', args.output / (label + '.perf.data'),
                       '-e', 'syscalls:sys_enter_sched_yield,sched:sched_switch,sched:sched_migrate_task', '--', *command]
        else:
            command = [*prefix, 'stat', '-x;', '--no-big-num', *control, '-o', args.output / (label + '-stat.csv'),
                       '-e', perf_events, '--', *command]
    result = checked(command, label, env=env, required=not allow_failure)
    if result['exit_status'] != 0:
        return {'variant':variant,'implementation':implementation,'label':label,'failed':True,
                'exit_status':result['exit_status'],'timeout':result.get('timeout',False)}
    records = [json.loads(line) for line in result['stdout'].splitlines()]
    summary = next(r for r in records if r['record'] == 'summary')
    assert summary['accepted'] == summary['applied'] == frames
    summary.update(variant=variant, implementation=implementation, label=label,
                   fps=frames * 1e9 / summary['elapsed_ns'], environment_before=before, environment_after=environment())
    counts = [r for r in records if r['record'] == 'bedrock_counts']
    if variant.startswith('counts'):
        producer = counts[1]; consumer = counts[0]
        assert producer['attempts'] == producer['busy'] + producer['full'] + producer['admitted']
        assert producer['admitted'] == consumer['applied'] == frames
        assert producer['busy'] + producer['full'] == summary['retries'] == producer['yields']
        assert consumer['visits'] == consumer['service_busy'] + consumer['empty'] + consumer['applied']
        summary['counts'] = counts
    if perf_events:
        summary['pmu'] = list(csv.reader((args.output / (label + '-stat.csv')).read_text().splitlines(), delimiter=';'))
        parsed = counter_rows(summary['pmu'])
        assert parsed and all(r['available'] and r['unmultiplexed'] for r in parsed.values()), parsed
    return summary


try:
    if args.phase in ('all', 'build'):
        variants = args.variants.split(',') if args.variants else ('control', 'counts', 'prebuilt-input', 'block-observer')
        for variant in variants:
            assert variant in ('control','counts','prebuilt-input','block-observer','cpu-yield')
            build(variant)
        manifest['phases']['build'] = True
    if args.phase in ('all', 'qualify'):
        for variant in ('counts-asan', 'counts-tsan'):
            build(variant)
            # Runtime tools default to fail-fast; subprocess status is checked.
            os.environ.update(ASAN_OPTIONS='detect_leaks=1:halt_on_error=1', UBSAN_OPTIONS='halt_on_error=1', TSAN_OPTIONS='halt_on_error=1')
            for impl in ('neogpu','rigtorp'):
                run(variant, impl, variant + '-' + impl, frames=4096, draws=64)
        manifest['phases']['qualify'] = True
    if args.phase in ('all', 'counts'):
        results = [run('counts', impl, f'counts-{pair}-{impl}', frames=100000) for pair in range(1, 4) for impl in ('neogpu', 'rigtorp')]
        (args.output / 'counts.json').write_text(json.dumps(results, indent=2))
        print(json.dumps({'phase': 'counts', 'results': results}), flush=True)
        manifest['phases']['counts'] = True
    if args.phase in ('all', 'count-pmu'):
        events = '{armv8_cortex_a72/event=0x6c,name=ldrex_user/u,armv8_cortex_a72/event=0x6d,name=strex_pass_user/u,armv8_cortex_a72/event=0x6e,name=strex_fail_user/u},syscalls:sys_enter_sched_yield'
        results = []
        for pair in range(1,4):
            for impl in (('neogpu','rigtorp') if pair % 2 else ('rigtorp','neogpu')):
                summary = run('counts',impl,f'count-pmu-{pair}-{impl}',frames=100000,perf_events=events)
                consumer, producer = summary['counts']
                consumer_locks = 0 if 'slot->read_index' in original['src/hs_fabric.c'] else consumer['visits'] - consumer['service_busy']
                # No leave/retirement occurs in this static diagnostic workload.
                summary['successful_mutex_pairs'] = consumer_locks + producer['attempts'] - producer['busy'] if impl == 'neogpu' else 0
                summary['mutex_store_exclusive_budget'] = 2 * summary['successful_mutex_pairs']
                rows = {r[2]:r for r in summary['pmu'] if len(r)>=5 and r[0] and not r[0].startswith('#')}
                summary['measured_strex_pass'] = float(rows['strex_pass_user'][0])
                summary['pass_minus_mutex_budget'] = summary['measured_strex_pass'] - summary['mutex_store_exclusive_budget']
                summary['yield_pmu_matches'] = float(rows['syscalls:sys_enter_sched_yield'][0]) == producer['yields']
                assert summary['yield_pmu_matches']
                results.append(summary)
        (args.output/'count-pmu.json').write_text(json.dumps(results,indent=2))
        manifest['phases']['count-pmu'] = True
        print(json.dumps({'phase':'count-pmu','results':results}),flush=True)
    if args.phase in ('all', 'pmu'):
        groups = {
            'core': '{cycles,instructions,branch-misses},task-clock,context-switches,cpu-migrations,page-faults,syscalls:sys_enter_sched_yield',
            'core-untraced': '{cycles,instructions,branch-misses},task-clock,context-switches,cpu-migrations,page-faults',
            'exclusive': '{armv8_cortex_a72/event=0x6c,name=ldrex_user/u,armv8_cortex_a72/event=0x6d,name=strex_pass_user/u,armv8_cortex_a72/event=0x6e,name=strex_fail_user/u}',
            'l1': '{armv8_cortex_a72/l1d_cache/u,armv8_cortex_a72/l1d_cache_refill/u,armv8_cortex_a72/event=0x48,name=l1d_inval_user/u}',
            'l2-bus': '{armv8_cortex_a72/l2d_cache/u,armv8_cortex_a72/l2d_cache_refill/u,armv8_cortex_a72/bus_access/u}',
            'tlb': '{armv8_cortex_a72/l1d_tlb_refill/u,armv8_cortex_a72/l1i_cache_refill/u,armv8_cortex_a72/l1i_tlb_refill/u}',
            'user-core': '{cycles:u,instructions:u,branch-misses:u}',
        }
        results = []
        if args.groups:
            selected = args.groups.split(',')
            assert all(name in groups for name in selected)
            groups = {name: groups[name] for name in selected}
        for group, events in groups.items():
            for pair in range(1, args.pairs + 1):
                variants = args.pmu_variants.split(',')
                assert all(v in ('control','prebuilt-input','block-observer','cpu-yield') for v in variants)
                order = [(v, i) for v in variants for i in ('neogpu','rigtorp')]
                if pair % 2 == 0: order.reverse()
                for variant, impl in order:
                    results.append(run(variant, impl, f'pmu-{group}-{pair}-{variant}-{impl}', perf_events=events))
        (args.output / 'pmu.json').write_text(json.dumps(results, indent=2))
        manifest['phases']['pmu'] = True
        print(json.dumps({'phase': 'pmu', 'executions': len(results)}), flush=True)
    if args.phase in ('all', 'ablations'):
        results = []
        variants = args.variants.split(',') if args.variants else ('control','prebuilt-input','block-observer')
        for draws in (1, 64):
            for pair in range(1, 4):
                order = [(v, i) for v in variants for i in ('neogpu','rigtorp')]
                if pair % 2 == 0: order.reverse()
                for variant, impl in order:
                    results.append(run(variant, impl, f'ablation-{draws}-{pair}-{variant}-{impl}', frames=200000, draws=draws,
                                       allow_failure=variant == 'cpu-yield'))
        (args.output / 'ablations.json').write_text(json.dumps(results, indent=2))
        manifest['phases']['ablations'] = True
        print(json.dumps({'phase': 'ablations', 'executions': len(results)}), flush=True)
    if args.phase in ('all', 'trace'):
        trace_selftest()
        results = []
        for impl in ('neogpu','rigtorp'):
            label = 'trace-systemwide-' + impl
            summary = run('counts', impl, label, frames=20000, trace=True)
            script = checked(['sudo','-n','perf','script','--ns','--show-lost-events','-i',args.output / (label + '.perf.data'),
                              '-F','comm,tid,cpu,time,event,trace'], label + '-script')['stdout']
            (args.output / (label + '-script.txt')).write_text(script)
            summary.update(decode(script,summary['counts']),trace_scope='CPU 0/1 systemwide, post-filtered to benchmark TIDs')
            assert not summary['lost_events'] and summary['yield_count_matches'], summary
            results.append(summary)
        (args.output / 'trace.json').write_text(json.dumps(results, indent=2))
        manifest['phases']['trace'] = True
        print(json.dumps({'phase':'trace','results':results}), flush=True)
    assert all(Path(name).read_text() == source for name, source in original.items())
    manifest['runtime_unchanged'] = True
finally:
    manifest['phase_environments'][-1]['after'] = environment()
    manifest_path.write_text(json.dumps(manifest, indent=2))
