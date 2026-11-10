"""Time the application's own EGL/GBM/DRM/GL calls via an LD_PRELOAD shim.

Diagnostic only. Compares preload-on against preload-off so shim overhead is
measured rather than assumed, and reports kernel-side hotspots separately
(strace). Output stays in a fresh tests/evidence-* directory.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

if not __debug__:
    raise RuntimeError('Cost evidence requires assertions enabled')
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--binary', type=Path, default=Path('tests/evidence-e2e-c9307e1-bounded/optimized/benchmark'))
parser.add_argument('--reps', type=int, default=3)
parser.add_argument('--frames', type=int, default=64)
args = parser.parse_args()
out = args.output.resolve()
assert out.parent == ROOT/'tests' and out.name.startswith('evidence-')
assert 1 <= args.reps <= 9 and 8 <= args.frames <= 1024
assert args.binary.resolve().is_file()
out.mkdir(exist_ok=False)
out = out.relative_to(ROOT)

SHIM = Path('tests/gles_call_timer.so')
subprocess.run(['gcc', '-O2', '-Wall', '-Wextra', '-shared', '-fPIC', '-o', str(SHIM),
                'tests/gles_call_timer.c', '-ldl', '-lpthread'], check=True)

manifest = {'completed': False, 'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'binary': str(args.binary), 'binary_sha256': hashlib.sha256(args.binary.resolve().read_bytes()).hexdigest(),
            'shim_sha256': hashlib.sha256(SHIM.read_bytes()).hexdigest(), 'runs': [],
            'contract': 'Application-visible call durations under LD_PRELOAD on the committed real-GLES '
                        'end-to-end benchmark. Kernel driver internals are not included; strace covers those. '
                        'Traced/preloaded runs are diagnostics, not throughput samples.'}
(out/'driver.py').write_bytes(Path(__file__).read_bytes())

def environment():
    data = {'wall_ns': time.time_ns(), 'affinity': sorted(os.sched_getaffinity(0)), 'sysfs': {}}
    for base, pattern in [('/sys/devices/system/cpu', 'cpu*/cpufreq/scaling_*'), ('/sys/class/thermal', 'thermal_zone*/temp')]:
        for path in sorted(Path(base).glob(pattern)):
            if path.is_file():
                try: data['sysfs'][str(path)] = path.read_text().strip()
                except OSError as error: data['sysfs'][str(path)] = str(error)
    return data

def run(label, draws, preload):
    prefix = out/label
    argv = ['/usr/bin/time', '-v', '-o', str(out/(label+'.time')), 'sudo', '-n', 'env']
    if preload:
        argv += ['LD_PRELOAD='+str(SHIM.resolve()), 'GLES_TIMER_OUT='+str((out/(label+'.calls.csv')).resolve())]
    argv += [str(args.binary.resolve()), 'native', '1', str(args.frames), str(draws), 'off', str(prefix)]
    before = environment(); start = time.monotonic_ns()
    result = subprocess.run(argv, capture_output=True, text=True, timeout=600, env={**os.environ, 'LC_ALL': 'C'})
    record = {'label': label, 'draws': draws, 'preload': preload, 'argv': argv, 'status': result.returncode,
              'wall_ns': time.monotonic_ns()-start, 'stdout': result.stdout, 'stderr': result.stderr,
              'before': before, 'after': environment()}
    (out/(label+'.command.json')).write_text(json.dumps(record, indent=2))
    assert result.returncode == 0, record
    summaries = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    summary = [r for r in summaries if r['record'] == 'summary']
    assert len(summary) == 1
    summary = summary[0]
    assert summary['mode'] == 'native' and summary['producers'] == 1 and summary['draws'] == draws
    assert summary['capture'] == 'off' and summary['resources_released'] is True
    assert summary['accepted'] == summary['applied'] == args.frames
    calls = {}
    if preload:
        with (out/(label+'.calls.csv')).open() as handle:
            for row in csv.DictReader(handle):
                calls[row['call']] = {'calls': int(row['calls']), 'total_ns': int(row['total_ns']), 'max_ns': int(row['max_ns'])}
    return {**summary, 'calls': calls, 'whole_process_ns': record['wall_ns']}

frames = args.frames
INIT_ONCE = ('eglInitialize', 'eglChooseConfig', 'eglCreateContext', 'eglCreateWindowSurface',
             'gbm_create_device', 'gbm_surface_create', 'drmSetMaster', 'drmModeGetResources')
SHUTDOWN_ONCE = ('eglDestroySurface', 'eglDestroyContext', 'eglTerminate', 'eglReleaseThread',
                 'gbm_surface_destroy', 'gbm_device_destroy', 'drmDropMaster')

try:
    manifest['environment'] = environment()
    for draws in (1, 64):
        baseline, preloaded = [], []
        for rep in range(1, args.reps+1):
            baseline.append(run(f'draws{draws}-off-{rep}', draws, False))
            preloaded.append(run(f'draws{draws}-on-{rep}', draws, True))
        totals = {}
        for record in preloaded:
            for name, value in record['calls'].items():
                entry = totals.setdefault(name, {'calls': 0, 'total_ns': 0, 'max_ns': 0, 'runs': 0})
                entry['calls'] += value['calls']
                entry['total_ns'] += value['total_ns']
                entry['max_ns'] = max(entry['max_ns'], value['max_ns'])
                entry['runs'] += 1
        assert totals, 'preload shim produced no call records'
        expected_swaps = frames + 10 + 1  # warmup 9 + cold 1, timed frames, one snapshot frame
        assert abs(totals['eglSwapBuffers']['calls']/len(preloaded) - expected_swaps) <= 1
        for name, entry in totals.items():
            entry['per_run_mean_ns'] = entry['total_ns']/entry['runs']
            entry['per_call_mean_ns'] = entry['total_ns']/entry['calls']
        init_ns = sum(totals.get(name, {}).get('per_run_mean_ns', 0) for name in INIT_ONCE)
        shutdown_ns = sum(totals.get(name, {}).get('per_run_mean_ns', 0) for name in SHUTDOWN_ONCE)
        manifest['runs'].append({
            'draws': draws,
            'baseline': {'elapsed_ns': statistics.median(r['elapsed_ns'] for r in baseline),
                         'process_cpu_ns': statistics.median(r['process_cpu_ns'] for r in baseline),
                         'whole_process_ns': statistics.median(r['whole_process_ns'] for r in baseline),
                         'timed_frames_ns': [r['elapsed_ns'] for r in baseline]},
            'preloaded': {'elapsed_ns': statistics.median(r['elapsed_ns'] for r in preloaded),
                          'process_cpu_ns': statistics.median(r['process_cpu_ns'] for r in preloaded),
                          'whole_process_ns': statistics.median(r['whole_process_ns'] for r in preloaded),
                          'timed_frames_ns': [r['elapsed_ns'] for r in preloaded]},
            'shim_overhead': {'elapsed_ratio': statistics.median(r['elapsed_ns'] for r in preloaded)/
                                          statistics.median(r['elapsed_ns'] for r in baseline)},
            'calls': totals,
            'once_call_buckets_ns': {'init_once': init_ns, 'shutdown_once': shutdown_ns},
            'note': 'Buckets sum application-visible call wall time only; nested library work is '
                    'counted once at each intercepted call and must not be double-counted.'})
        print(draws, 'draws: preload/baseline elapsed ratio',
              round(manifest['runs'][-1]['shim_overhead']['elapsed_ratio'], 4), flush=True)
    manifest['completed'] = True
finally:
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
