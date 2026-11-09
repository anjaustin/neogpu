"""Measure default HSGpu integration versus frozen manual/legacy and SPSC paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

if not __debug__:
    raise RuntimeError('Frame evidence requires Python assertions enabled')
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--baseline', default='7d1889d')
parser.add_argument('--frames', type=int, default=200000)
parser.add_argument('--sanitizers', action='store_true')
parser.add_argument('--control', type=Path, help='preserved include/src snapshot of an earlier integration')
parser.add_argument('--reference', type=Path, default=Path('tests/evidence-comparison-final/reference'))
args = parser.parse_args()
if not args.output.name.startswith('evidence-') or not args.output.resolve().is_relative_to(ROOT/'tests'):
    parser.error('output must be a fresh tests/evidence-* directory')
if not 128 <= args.frames <= 1000000:
    parser.error('frames must be 128..1000000 per producer')
args.output.mkdir(exist_ok=False)
(args.output/'driver.py').write_bytes(Path(__file__).read_bytes())
common = args.output/'common'; common.mkdir()
for name in ('test_gpu_frame_benchmark.c', 'comparison_rigtorp.cc', 'comparison_transport.h'):
    (common/name).write_bytes((Path('tests')/name).read_bytes())
manifest = {'completed': False, 'runs': [], 'commands': [], 'source_sha256': {},
            'baseline': subprocess.check_output(['git', 'rev-parse', args.baseline], text=True).strip(),
            'frames_per_producer': args.frames,
            'contract': 'Legacy scenes and participant contributions are separate units. Active scenes are renderer-global CLEAR plus retained draws (no explicit frame markers); legacy scenes use actual message APIs. Mock backend validates every command, callback and applied sequence. No GPU/display FPS claim.'}
runtime = ['hs_core', 'hs_nodes', 'hs_gpu', 'hs_fabric', 'hs_async', 'hs_backend_gles', 'hs_ipc', 'hs_text', 'hs_input_stream']
variants = ('baseline', 'control', 'candidate') if args.control else ('baseline', 'candidate')


def environment():
    result = {'wall_ns': time.time_ns(), 'affinity': sorted(os.sched_getaffinity(0)), 'sysfs': {}}
    for base, pattern in [('/sys/devices/system/cpu', 'cpu*/cpufreq/scaling_*'), ('/sys/class/thermal', 'thermal_zone*/temp')]:
        for path in sorted(Path(base).glob(pattern)):
            if path.is_file():
                try: result['sysfs'][str(path)] = path.read_text().strip()
                except OSError as error: result['sysfs'][str(path)] = str(error)
    return result


def command(argv, label, env=None):
    before = environment()
    result = subprocess.run(list(map(str, argv)), text=True, capture_output=True, timeout=180, env=env)
    record = {'argv': list(map(str, argv)), 'status': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
              'environment_before': before, 'environment_after': environment()}
    with (args.output/(label+'.json')).open('x') as out: json.dump(record, out, indent=2)
    manifest['commands'].append({'label': label, 'status': result.returncode})
    assert result.returncode == 0, record
    return result.stdout


def build(variant, sanitizer=None):
    label = variant + ('-'+sanitizer.replace(',', '-') if sanitizer else '')
    directory = args.output/label
    include, src = directory/'include', directory/'src'
    include.mkdir(parents=True); src.mkdir()
    if variant == 'baseline':
        headers = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', manifest['baseline'], 'include'], text=True).splitlines()
        headers = [name for name in headers if name.endswith('.h')]
    elif variant == 'control': headers = [str(path.relative_to(args.control)) for path in (args.control/'include').rglob('*.h')]
    else: headers = [str(path) for path in Path('include').rglob('*.h')]
    sources = ['src/'+name+'.c' for name in runtime]
    for name in headers + sources:
        if variant == 'baseline': data = subprocess.check_output(['git', 'show', manifest['baseline']+':'+name])
        elif variant == 'control': data = (args.control/name).read_bytes()
        else: data = Path(name).read_bytes()
        destination = directory/name
        destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(data)
        manifest['source_sha256'][str(destination)] = hashlib.sha256(data).hexdigest()
    flags = ['-O3', '-g', '-march=armv8-a+simd', '-mtune=cortex-a72', '-ffast-math', '-funroll-loops',
             '-Wall', '-Wextra', '-Werror', '-DNDEBUG', '-I'+str(include), '-Itests', '-I/usr/include/libdrm']
    if sanitizer: flags += ['-O1', '-fno-omit-frame-pointer', '-fsanitize='+sanitizer]
    objects = []
    for name in [*sources, 'tests/test_gpu_frame_benchmark.c', 'tests/comparison_rigtorp.cc']:
        source = directory/name if name in sources else common/Path(name).name
        cpp = source.suffix == '.cc'; obj = directory/(source.stem+'.o')
        if name not in sources: manifest['source_sha256'][name] = hashlib.sha256(source.read_bytes()).hexdigest()
        command(['g++' if cpp else 'gcc', *flags, '-I'+str(common), *(['-std=c++23', '-isystem', args.reference/'include'] if cpp else []),
                 '-c', source, '-o', obj], label+'-build-'+source.stem)
        objects.append(obj)
    command(['g++', *flags, *objects, '-lm', '-lGLESv2', '-lgbm', '-ldrm', '-lEGL', '-lpthread', '-o', directory/'benchmark'], label+'-link')
    return directory/'benchmark'


try:
    for path in [args.output/'driver.py', *common.iterdir()]:
        manifest['source_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['toolchain'] = {name: subprocess.check_output(argv, text=True) for name, argv in
        [('gcc', ['gcc', '--version']), ('g++', ['g++', '--version']), ('kernel', ['uname', '-a'])]}
    manifest['working_tree'] = subprocess.check_output(['git', 'status', '--short'], text=True)
    assert set(range(4)).issubset(os.sched_getaffinity(0))
    for name, expected in [('include/rigtorp/SPSCQueue.h', '62d9f783862666be5c7ca3c8fbcec2e45d9920167c4b9e689bda4957f66dd86e'),
                           ('LICENSE', '21d7e737765bc0c986ee78f16a3e25ee3cb5ad424feece274b64a10654c31912')]:
        actual = hashlib.sha256((args.reference/name).read_bytes()).hexdigest(); assert actual == expected
        manifest['source_sha256'][str(args.reference/name)] = actual
    manifest['source_sha256'][str(Path(__file__).relative_to(ROOT)) if Path(__file__).is_absolute() else str(Path(__file__))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['variants'] = list(variants)
    if args.control: manifest['control'] = str(args.control)
    binaries = {variant: build(variant) for variant in variants}
    if args.sanitizers:
        env = {**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1', 'UBSAN_OPTIONS': 'halt_on_error=1', 'TSAN_OPTIONS': 'halt_on_error=1'}
        for variant in variants:
            for sanitizer in ('address,undefined', 'thread'):
                binary = build(variant, sanitizer)
                for mode, producers in [('legacy', 1), ('native', 1), ('native', 8), ('reference', 1), ('reference', 8)]:
                    command([binary, mode, producers, 1024, 64], variant+'-'+sanitizer.replace(',', '-')+'-'+mode+'-'+str(producers), env)
    for draws in (1, 64):
        for variant in variants:
            command([binaries[variant], 'legacy', 1, 4096, draws], f'legacy-warm-{draws}-{variant}')
        for pair in (1, 2, 3):
            order = list(variants) if pair % 2 else list(reversed(variants))
            for variant in order:
                output = command([binaries[variant], 'legacy', 1, args.frames, draws], f'legacy-{draws}-{pair}-{variant}')
                summary = json.loads(output); assert summary['accepted'] == summary['applied'] == args.frames
                summary.update(variant=variant, pair=pair); manifest['runs'].append(summary)
    for producers in (1, 2, 8):
        for draws in (1, 64):
            for variant in variants:
                for mode in ('native', 'reference'):
                    command([binaries[variant], mode, producers, 4096, draws], f'active-warm-{producers}-{draws}-{variant}-{mode}')
            for pair in (1, 2, 3):
                order = [(v, m) for m in ('native', 'reference') for v in variants]
                if pair % 2 == 0: order.reverse()
                for variant, mode in order:
                    output = command([binaries[variant], mode, producers, args.frames, draws], f'active-{producers}-{draws}-{pair}-{variant}-{mode}')
                    summary = json.loads(output); assert summary['accepted'] == summary['applied'] == producers*args.frames
                    summary.update(variant=variant, pair=pair); manifest['runs'].append(summary)
    manifest['completed'] = True
finally:
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
