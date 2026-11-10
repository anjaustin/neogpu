"""Concurrent producer to completed real GLES: default HSGpu vs pinned SPSC plumbing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

if not __debug__:
    raise RuntimeError('Evidence requires assertions enabled')
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--reference', type=Path, default=Path('tests/evidence-comparison-final/reference'))
parser.add_argument('--frames', type=int, default=256)
parser.add_argument('--pairs', type=int, default=5)
parser.add_argument('--sanitizers', action='store_true')
args = parser.parse_args()
assert args.output.resolve().parent == ROOT/'tests' and args.output.name.startswith('evidence-')
assert 3 <= args.frames <= 512 and 3 <= args.pairs <= 9
args.output = args.output.resolve().relative_to(ROOT)
args.output.mkdir(exist_ok=False)
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
manifest = {'completed': False, 'revision': revision, 'source_sha256': {}, 'binary_sha256': {}, 'commands': [], 'runs': [],
    'pairs': args.pairs, 'frames_per_producer': args.frames, 'qualification': [],
    'reference_revision': 'c8e9008dc4639220cc9bcf4c84abd2ff0d344aa5',
    'contract': 'One producer per lane, four usable queued contributions; identical real GLES backend, stable geometry and ordered draws. Each render frame ends with glFinish; first-attempt-to-completed-frame latency includes retries/queueing/service/rendering. All contributions validated FIFO and all final RGBA bytes equal, with nonblank lane pixels checked. Capture off/on records one real CLEAR message per frame; participant draws remain retained contributions. No panel scanout timing or engine-wide speed claim.'}
runtime = ('hs_core', 'hs_nodes', 'hs_gpu', 'hs_fabric', 'hs_async', 'hs_backend_gles', 'hs_ipc', 'hs_text', 'hs_input_stream')

def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    manifest['source_sha256'][str(path)] = hashlib.sha256(data).hexdigest()

def environment():
    data = {'wall_ns': time.time_ns(), 'affinity': sorted(os.sched_getaffinity(0)), 'sysfs': {}}
    for base, pattern in [('/sys/devices/system/cpu', 'cpu*/cpufreq/scaling_*'), ('/sys/class/thermal', 'thermal_zone*/temp')]:
        for p in sorted(Path(base).glob(pattern)):
            if p.is_file():
                try: data['sysfs'][str(p)] = p.read_text().strip()
                except OSError as error: data['sysfs'][str(p)] = str(error)
    return data

def command(argv, label):
    before = environment(); start = time.monotonic_ns()
    result = subprocess.run(['/usr/bin/time', '-v', '-o', str(args.output/(label+'.time')), *map(str, argv)], capture_output=True,
                            text=True, timeout=240, env={**os.environ, 'LC_ALL': 'C'})
    record = {'argv': list(map(str, argv)), 'status': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
        'whole_process_ns': time.monotonic_ns()-start, 'before': before, 'after': environment()}
    (args.output/(label+'.json')).write_text(json.dumps(record, indent=2))
    manifest['commands'].append({'label': label, 'status': result.returncode})
    assert result.returncode == 0, record
    return [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]

def build(sanitizer=False):
    directory = args.output/('asan' if sanitizer else 'optimized'); directory.mkdir()
    flags = ['-O3', '-g', '-march=armv8-a+simd', '-mtune=cortex-a72', '-ffast-math', '-funroll-loops', '-Wall', '-Wextra', '-Werror', '-DNDEBUG',
        '-I'+str(args.output/'runtime/include'), '-I'+str(args.output/'common'), '-I/usr/include/libdrm']
    if sanitizer: flags += ['-O1', '-fno-omit-frame-pointer', '-fsanitize=address,undefined']
    objects = []
    sources = [args.output/'runtime/src'/(n+'.c') for n in runtime]
    sources += [args.output/'common/test_gpu_gles_e2e.c', args.output/'common/comparison_rigtorp.cc']
    for source in sources:
        cpp = source.suffix == '.cc'; obj = directory/(source.stem+'.o'); objects.append(obj)
        command(['g++' if cpp else 'gcc', *flags, *(['-std=c++23', '-isystem', args.output/'reference/include'] if cpp else []),
            '-c', source, '-o', obj], directory.name+'-compile-'+source.stem)
    binary = directory/'benchmark'
    command(['g++', *flags, *objects, '-Wl,--wrap=eglSwapBuffers', '-lm', '-lGLESv2', '-lgbm', '-ldrm', '-lEGL', '-lpthread', '-o', binary], directory.name+'-link')
    manifest['binary_sha256'][str(binary)] = hashlib.sha256(binary.read_bytes()).hexdigest()
    return binary

def validate(records, mode, producers, draws, frames, capture, prefix):
    summaries = [r for r in records if r['record'] == 'summary']; assert len(summaries) == 1
    summary = summaries[0]
    assert (summary['mode'], summary['producers'], summary['draws'], summary['frames_per_producer'], summary['capture']) == (mode, producers, draws, frames, capture)
    assert summary['resources_released'] is True and summary['accepted'] == summary['applied'] == frames*producers
    contributions = [r for r in records if r['record'] == 'contribution']
    assert [(r['lane'], r['sequence']) for r in contributions] == [(lane, seq) for lane in range(producers) for seq in range(1, frames+1)]
    for r in contributions:
        assert r['completed_ns'] >= r['attempt_ns'] >= 0 and r['admitted_ns'] >= r['attempt_ns']
    rendered = [r for r in records if r['record'] == 'frame']
    assert [r['index'] for r in rendered] == list(range(summary['rendered_frames']))
    assert len(rendered) >= frames and all(r['completed_ns'] >= r['submit_ns'] > 0 and r['cpu_ns'] > 0 for r in rendered)
    contexts = [r for r in records if r['record'] == 'context']; assert len(contexts) == 1
    rgba = Path(str(prefix)+'.rgba').read_bytes()
    assert len(rgba) == contexts[0]['width']*contexts[0]['height']*4
    def stats(values):
        values = sorted(values)
        return {'p50': statistics.median(values), 'p95': values[(len(values)*95+99)//100-1], 'p99': values[(len(values)*99+99)//100-1], 'max': values[-1]}
    return {**summary, 'context': contexts[0], 'rgba_sha256': hashlib.sha256(rgba).hexdigest(),
        'contribution_latency_ns': stats([r['completed_ns']-r['attempt_ns'] for r in contributions]),
        'admission_latency_ns': stats([r['admitted_ns']-r['attempt_ns'] for r in contributions]),
        'frame_completed_ns': stats([r['completed_ns'] for r in rendered]),
        'frame_submit_ns': stats([r['submit_ns'] for r in rendered]),
        'completed_contributions_per_second': summary['applied']*1e9/summary['elapsed_ns']}

def run(binary, mode, producers, draws, frames, capture, label, sanitizer=False):
    prefix = args.output/label
    argv = ['sudo', '-n']
    if sanitizer: argv += ['env', 'ASAN_OPTIONS=detect_leaks=1:halt_on_error=1', 'UBSAN_OPTIONS=halt_on_error=1']
    argv += [binary, mode, producers, frames, draws, capture, prefix]
    records = command(argv, label)
    return validate(records, mode, producers, draws, frames, capture, prefix)

try:
    assert set(range(4)).issubset(os.sched_getaffinity(0))
    save(args.output/'driver.py', Path(__file__).read_bytes())
    manifest['toolchain'] = {name: subprocess.check_output(argv, text=True) for name, argv in
        [('gcc', ['gcc', '--version']), ('g++', ['g++', '--version']), ('kernel', ['uname', '-a']), ('working_tree', ['git', 'status', '--short'])]}
    headers = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', revision, 'include'], text=True).splitlines()
    for name in [n for n in headers if n.endswith('.h')]+['src/'+n+'.c' for n in runtime]:
        save(args.output/'runtime'/name, subprocess.check_output(['git', 'show', revision+':'+name]))
    for name in ('test_gpu_gles_e2e.c', 'test_gpu_frame_benchmark.c', 'comparison_rigtorp.cc', 'comparison_transport.h'):
        data = (ROOT/'tests'/name).read_bytes()
        if name != 'test_gpu_gles_e2e.c': assert data == subprocess.check_output(['git', 'show', revision+':tests/'+name])
        save(args.output/'common'/name, data)
    for name, digest in [('include/rigtorp/SPSCQueue.h', '62d9f783862666be5c7ca3c8fbcec2e45d9920167c4b9e689bda4957f66dd86e'),
                         ('LICENSE', '21d7e737765bc0c986ee78f16a3e25ee3cb5ad424feece274b64a10654c31912')]:
        data = (args.reference/name).read_bytes(); assert hashlib.sha256(data).hexdigest() == digest
        save(args.output/'reference'/name, data)
    if args.sanitizers:
        sanitized = build(True)
        for mode in ('native', 'reference'):
            for producers, draws, capture in [(1, 1, 'off'), (8, 64, 'on')]:
                label = f'asan-{mode}-{producers}-{draws}-{capture}'
                manifest['qualification'].append(run(sanitized, mode, producers, draws, 16, capture, label, True))
    binary = build()
    images = {}
    for producers in (1, 2, 8):
        for draws in (1, 64):
            for capture in ('off', 'on'):
                for pair in range(1, args.pairs+1):
                    for mode in (('native', 'reference') if pair % 2 else ('reference', 'native')):
                        label = f'timed-{producers}-{draws}-{capture}-{pair}-{mode}'
                        result = run(binary, mode, producers, draws, args.frames, capture, label)
                        key = producers, draws
                        if key in images: assert images[key] == result['rgba_sha256']
                        images[key] = result['rgba_sha256']
                        manifest['runs'].append({**result, 'pair': pair, 'label': label})
                print(producers, draws, capture, 'passed', flush=True)
    manifest['completed'] = True
finally:
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
