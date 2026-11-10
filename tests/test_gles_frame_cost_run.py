"""Run the frame-cost decomposition probe and reduce it.

Diagnostic only. Establishes whether the fixed per-frame cost is GL submission,
swap, GPU completion wait, or resolution-scaled fill, by varying one factor at a
time on the same real V3D/GBM/EGL stack. Also records per-thread CPU.

Output stays in a fresh tests/evidence-* directory. No production code changes.
"""
import argparse
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
parser.add_argument('--reps', type=int, default=5)
parser.add_argument('--suite', choices=('full', 'extras'), default='full')
parser.add_argument('--frames', type=int, default=256)
args = parser.parse_args()
out = args.output.resolve()
assert out.parent == ROOT/'tests' and out.name.startswith('evidence-')
assert 3 <= args.reps <= 9 and 64 <= args.frames <= 1024
out.mkdir(exist_ok=False)
out = out.relative_to(ROOT)

BINARY = Path('tests/test_gles_frame_cost')
subprocess.run(['gcc', '-O3', '-march=armv8-a+simd', '-mtune=cortex-a72', '-ffast-math',
                '-Wall', '-Wextra', '-Werror', '-DNDEBUG', '-Iinclude', '-I/usr/include/libdrm',
                '-o', str(BINARY), 'tests/test_gles_frame_cost.c',
                '-lm', '-lGLESv2', '-lgbm', '-ldrm', '-lEGL'], check=True)
binary_hash = hashlib.sha256(BINARY.read_bytes()).hexdigest()

# surface, size, frames, draws, finish_every, swap
CONFIGS = []
def add(name, surface, size, draws, finish_every, swap):
    CONFIGS.append({'name': name, 'surface': surface, 'size': size, 'draws': draws,
                    'finish_every': finish_every, 'swap': swap})
for draws in (0, 1, 16, 64, 256):
    add(f'window-sync-d{draws}', 'window', 0, draws, 1, 'on')
for every in (2, 4, 8, 16, 32, 64, 0):
    add(f'window-pipeline-e{every}-d64', 'window', 0, 64, every, 'on')
add('window-swaponly-d64', 'window', 0, 64, 0, 'on')
add('window-noswap-sync-d0', 'window', 0, 0, 1, 'off')
add('window-noswap-sync-d1', 'window', 0, 1, 1, 'off')
add('window-noswap-sync-d64', 'window', 0, 64, 1, 'off')
add('window-noswap-submit-d64', 'window', 0, 64, 0, 'off')
add('window-noswap-submit-d0', 'window', 0, 0, 0, 'off')
for size in (64, 128, 256, 512, 800):
    add(f'offscreen-sync-s{size}-d1', 'offscreen', size, 1, 1, 'off')
    add(f'offscreen-sync-s{size}-d64', 'offscreen', size, 64, 1, 'off')
add('none-loop', 'none', 0, 0, 0, 'off')
EXTRAS = []
def extra(name, surface, size, draws, finish_every, swap):
    EXTRAS.append({'name': name, 'surface': surface, 'size': size, 'draws': draws,
                   'finish_every': finish_every, 'swap': swap})
extra('window-noswap-pipe-e8-d64', 'window', 0, 64, 8, 'off')
extra('window-noswap-pipe-e64-d64', 'window', 0, 64, 64, 'off')
extra('window-noswap-pipe-e64-d0', 'window', 0, 0, 64, 'off')
extra('window-noswap-pipe-e256-d64', 'window', 0, 64, 256, 'off')
extra('offscreen-noswap-pipe-e64-s800-d64', 'offscreen', 800, 64, 64, 'off')
extra('offscreen-noswap-pipe-e64-s800-d1', 'offscreen', 800, 1, 64, 'off')
if args.suite == 'extras':
    CONFIGS = EXTRAS

manifest = {'completed': False, 'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'binary_sha256': binary_hash, 'reps': args.reps, 'frames': args.frames,
            'configs': CONFIGS, 'runs': [],
            'contract': 'One factor varied per configuration on the same real GBM/EGL/V3D stack. '
                        'Sync variants wait per frame with glFinish; pipeline variants finish every N frames. '
                        'Swap-off variants never call eglSwapBuffers, so they measure submit plus GPU finish, '
                        'not presentation. Offscreen variants render to an FBO of the given size. These are '
                        'diagnostic costs, not application frame rates; the verify readback is outside timing.'}
(out/'driver.py').write_bytes(Path(__file__).read_bytes())

def environment():
    data = {'wall_ns': time.time_ns(), 'affinity': sorted(os.sched_getaffinity(0)), 'sysfs': {}}
    for base, pattern in [('/sys/devices/system/cpu', 'cpu*/cpufreq/scaling_*'), ('/sys/class/thermal', 'thermal_zone*/temp')]:
        for path in sorted(Path(base).glob(pattern)):
            if path.is_file():
                try: data['sysfs'][str(path)] = path.read_text().strip()
                except OSError as error: data['sysfs'][str(path)] = str(error)
    return data

def run(config, rep):
    label = f"{config['name']}-r{rep}"
    prefix = out/label
    argv = ['sudo', '-n', str(BINARY.resolve()), config['surface'], str(config['size']),
            str(args.frames), str(config['draws']), str(config['finish_every']), config['swap'], str(prefix)]
    before = environment(); start = time.monotonic_ns()
    result = subprocess.run(argv, capture_output=True, text=True, timeout=300, env={**os.environ, 'LC_ALL': 'C'})
    record = {'label': label, 'config': config['name'], 'argv': argv, 'status': result.returncode,
              'wall_ns': time.monotonic_ns()-start, 'stdout': result.stdout, 'stderr': result.stderr,
              'before': before, 'after': environment()}
    (out/(label+'.json')).write_text(json.dumps(record, indent=2))
    assert result.returncode == 0, record
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    summaries = [r for r in rows if r['record'] == 'summary']
    assert len(summaries) == 1, rows
    summary = summaries[0]
    assert summary['surface'] == config['surface'] and summary['draws'] == config['draws']
    assert summary['frames'] == args.frames and summary['finish_every'] == config['finish_every']
    assert summary['swap'] == config['swap']
    assert summary['elapsed_ns'] > 0 and summary['process_cpu_ns'] > 0
    assert summary['verify_ns'] == 0 or summary['verify_ns'] > 0
    if config['surface'] != 'none':
        assert summary['width'] > 0 and summary['height'] > 0
        pixel_file = out/(label+'.verify.rgba')
        pixels = pixel_file.read_bytes()
        assert len(pixels) == summary['width']*summary['height']*4
        assert max(pixels) > 0, 'verification frame is blank'
        assert any(pixels[i] for i in range(3, len(pixels), 4)), 'no opaque pixel'
    threads = {r['phase']: r['threads'] for r in rows if r['record'] == 'threads'}
    assert set(threads) == {'before_timed', 'after_timed', 'after_shutdown'}
    return summary, threads

def stats(values):
    ordered = sorted(values)
    return {'median': statistics.median(ordered), 'min': ordered[0], 'max': ordered[-1]}

try:
    manifest['environment'] = environment()
    for config in CONFIGS:
        collected, threads = [], []
        for rep in range(1, args.reps+1):
            summary, thread_data = run(config, rep)
            collected.append(summary)
            threads.append(thread_data)
        entry = {'name': config['name'], 'config': config,
                 'frames': args.frames,
                 'per_frame_ns': stats([r['per_frame_ns'] for r in collected]),
                 'elapsed_ns': stats([r['elapsed_ns'] for r in collected]),
                 'timed_cpu_ns': stats([r['timed_cpu_ns'] for r in collected]),
                 'init_ns': stats([r['init_ns'] for r in collected]),
                 'first_present_ns': stats([r['first_present_ns'] for r in collected]),
                 'shutdown_ns': stats([r['shutdown_ns'] for r in collected]),
                 'submit_ns': {'p50': stats([r['submit_p50_ns'] for r in collected]),
                               'p99': stats([r['submit_p99_ns'] for r in collected])},
                 'swap_ns': {'p50': stats([r['swap_p50_ns'] for r in collected]),
                             'p99': stats([r['swap_p99_ns'] for r in collected])},
                 'finish_ns': {'p50': stats([r['finish_p50_ns'] for r in collected]),
                               'p99': stats([r['finish_p99_ns'] for r in collected])},
                 'thread_cpu': {}}
        for phase in ('before_timed', 'after_timed', 'after_shutdown'):
            totals, per_comm = [], {}
            for record in threads:
                total = 0
                comms = {}
                for thread in record[phase]:
                    total += thread['cpu_ns']
                    comms[thread['comm']] = comms.get(thread['comm'], 0) + thread['cpu_ns']
                totals.append(total)
                for comm, cpu in comms.items():
                    per_comm.setdefault(comm, []).append((1, cpu))
            entry['thread_cpu'][phase] = {
                'total_ns_median': statistics.median(totals),
                'threads_median': statistics.median(len(record[phase]) for record in threads),
                'by_comm': {comm: {'count_median': statistics.median([c for c, _ in values]),
                                   'cpu_ns_sum_median': statistics.median([cpu for _, cpu in values])}
                            for comm, values in sorted(per_comm.items())}}
        manifest['runs'].append(entry)
        print(config['name'], 'per-frame us', round(entry['per_frame_ns']['median']/1000, 1), flush=True)
    manifest['completed'] = True
finally:
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
