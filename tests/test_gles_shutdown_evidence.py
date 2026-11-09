"""Gated real-call shutdown profiling against frozen committed GLES source."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

if not __debug__:
    raise RuntimeError('Shutdown evidence requires assertions enabled')
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--baseline', default='3aaab60')
args = parser.parse_args()
if not args.output.name.startswith('evidence-') or not args.output.resolve().is_relative_to(ROOT/'tests'):
    parser.error('fresh tests/evidence-* output required')
args.output = args.output.resolve().relative_to(ROOT); args.output.mkdir(exist_ok=False)
manifest = {'completed': False, 'baseline': subprocess.check_output(['git', 'rev-parse', args.baseline], text=True).strip(),
            'source_sha256': {}, 'commands': [], 'runs': []}
runtime = ('hs_core', 'hs_nodes', 'hs_gpu', 'hs_fabric', 'hs_async', 'hs_backend_gles', 'hs_ipc', 'hs_text', 'hs_input_stream')
wraps = ('eglMakeCurrent', 'eglDestroySurface', 'eglDestroyContext', 'eglTerminate', 'eglReleaseThread',
         'gbm_surface_destroy', 'gbm_device_destroy', 'drmDropMaster', 'close', 'glDeleteBuffers', 'glDeleteProgram', 'glDeleteTextures')


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    manifest['source_sha256'][str(path)] = hashlib.sha256(data).hexdigest()


def command(argv, label):
    result = subprocess.run(list(map(str, argv)), text=True, capture_output=True, timeout=240)
    record = {'argv': list(map(str, argv)), 'status': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    with (args.output/(label+'.json')).open('x') as out: json.dump(record, out, indent=2)
    manifest['commands'].append({'label': label, 'status': result.returncode})
    assert result.returncode == 0, record
    return [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]


try:
    save(args.output/'driver.py', Path(__file__).read_bytes())
    save(args.output/'trace.c', Path('tests/test_gles_shutdown_trace.c').read_bytes())
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', manifest['baseline'], 'include'], text=True).splitlines()
    names = [n for n in names if n.endswith('.h')] + ['src/'+n+'.c' for n in runtime]
    for name in names: save(args.output/'runtime'/name, subprocess.check_output(['git', 'show', manifest['baseline']+':'+name]))
    harness = subprocess.check_output(['git', 'show', manifest['baseline']+':tests/test_gles_work.c']).decode()
    save(args.output/'control.c', harness.encode())
    profiled = 'void hs_shutdown_trace_begin(void);\nvoid hs_shutdown_trace_end(void);\n'+harness
    assert profiled.count('    u64 shutdown_cpu_start =') == 1
    profiled = profiled.replace('    u64 shutdown_cpu_start =', '    hs_shutdown_trace_begin();\n    u64 shutdown_cpu_start =')
    marker = '    printf("{\\\"record\\\":\\\"lifecycle\\\"'
    assert profiled.count(marker) == 1, marker
    profiled = profiled.replace(marker, '    hs_shutdown_trace_end();\n'+marker)
    save(args.output/'profile.c', profiled.encode())
    flags = ['-O3', '-g', '-march=armv8-a+simd', '-mtune=cortex-a72', '-ffast-math', '-funroll-loops',
             '-Wall', '-Wextra', '-Werror', '-DNDEBUG', '-I'+str(args.output/'runtime/include'), '-I/usr/include/libdrm']
    objects = []
    for name in runtime:
        obj = args.output/(name+'.o'); objects.append(obj)
        command(['gcc', *flags, '-c', args.output/'runtime/src'/(name+'.c'), '-o', obj], 'build-'+name)
    for name in ('control', 'profile', 'trace'):
        command(['gcc', *flags, '-c', args.output/(name+'.c'), '-o', args.output/(name+'.o')], 'build-'+name)
    for variant in ('control', 'profile'):
        command(['gcc', *flags, args.output/(variant+'.o'), *objects,
                 *([args.output/'trace.o'] if variant == 'profile' else []),
                 '-Wl,--wrap=eglSwapBuffers', *(['-Wl,--wrap='+n for n in wraps] if variant == 'profile' else []),
                 '-lm', '-lGLESv2', '-lgbm', '-ldrm', '-lEGL', '-lpthread', '-o', args.output/variant], 'link-'+variant)
    for scene in ('single', 'retained'):
        for pair in range(1, 6):
            for variant in (('control', 'profile') if pair % 2 else ('profile', 'control')):
                label = f'{scene}-{pair}-{variant}'
                records = command(['sudo', '-n', args.output/variant, scene, 'on', '8', args.output/label, 'timed'], label)
                lifecycle = [r for r in records if r['record'] == 'lifecycle']; assert len(lifecycle) == 1
                assert lifecycle[0]['context_released'] is True
                calls = [r for r in records if r['record'] == 'shutdown_call']
                assert bool(calls) == (variant == 'profile')
                manifest['runs'].append({'scene': scene, 'pair': pair, 'variant': variant, 'lifecycle': lifecycle[0], 'calls': calls})
    records = command(['sudo', '-n', 'strace', '-f', '-ttt', '-T', '-o', args.output/'syscalls.txt',
                       args.output/'profile', 'single', 'on', '3', args.output/'strace-single', 'timed'], 'strace-single')
    manifest['trace_records'] = [r for r in records if r['record'] in ('lifecycle', 'shutdown_call')]
    manifest['completed'] = True
finally:
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
