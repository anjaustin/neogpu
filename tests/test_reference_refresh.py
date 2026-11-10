"""Serial, auditable refresh of committed-path/reference benchmarks and gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

if not __debug__:
    raise RuntimeError('Evidence requires assertions enabled')
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--resume', action='store_true', help='preserve failed attempt and reuse verified successful stage invocations')
args = parser.parse_args()
out = args.output.resolve()
assert out.parent == ROOT/'tests' and out.name.startswith('evidence-')
if args.resume: assert out.is_dir()
else: out.mkdir(exist_ok=False)
out = out.relative_to(ROOT)
manifest = {'completed': False, 'commands': [], 'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'contract': 'Serial runs; unchanged production. Optimized pinned static queue, validating mock frame execution, retained direct GLES, and exact committed GLES lifecycle are distinct boundaries. No panel-FPS or engine-wide superiority claim.'}
if args.resume:
    old = (out/'manifest.json').read_bytes()
    prior = json.loads(old); assert not prior['completed'] and prior['revision'] == manifest['revision']
    assert hashlib.sha256((out/'committed-sources.tar').read_bytes()).hexdigest() == prior['source_archive_sha256']
    attempt = 1
    while (out/f'manifest-before-resume-{attempt}.json').exists(): attempt += 1
    (out/f'manifest-before-resume-{attempt}.json').write_bytes(old)
    (out/f'driver-resume-{attempt}.py').write_bytes(Path(__file__).read_bytes())
    manifest = prior
else:
    (out/'driver.py').write_bytes(Path(__file__).read_bytes())
    with (out/'committed-sources.tar').open('xb') as f:
        subprocess.run(['git', 'archive', manifest['revision']], stdout=f, check=True)
    manifest['source_archive_sha256'] = hashlib.sha256((out/'committed-sources.tar').read_bytes()).hexdigest()

def env():
    data = {'wall_ns': time.time_ns(), 'affinity': sorted(os.sched_getaffinity(0)), 'sysfs': {}}
    for base, pattern in [('/sys/devices/system/cpu', 'cpu*/cpufreq/scaling_*'), ('/sys/class/thermal', 'thermal_zone*/temp')]:
        for p in sorted(Path(base).glob(pattern)):
            if p.is_file():
                try: data['sysfs'][str(p)] = p.read_text().strip()
                except OSError as error: data['sysfs'][str(p)] = str(error)
    return data

def run(label, argv):
    stage = label
    previous = [c for c in manifest['commands'] if c.get('stage', c['label']) == stage]
    if args.resume and previous and previous[-1]['status'] == 0:
        record = json.loads((out/previous[-1].get('record_file', previous[-1]['label']+'.json')).read_text())
        assert record['status'] == 0 and record['argv'] == list(map(str, argv))
        print(stage, 'reusing successful recorded invocation', flush=True); return
    if previous:
        retry = 1
        while any((out/(label+f'-retry{retry}'+suffix)).exists() for suffix in ('.json', '.command.json')): retry += 1
        label += f'-retry{retry}'
    before = env(); start = time.monotonic_ns()
    result = subprocess.run(['/usr/bin/time', '-v', '-o', str(out/(label+'.time')), *map(str, argv)], capture_output=True, text=True,
                            timeout=7200, env={**os.environ, 'LC_ALL': 'C'})
    record = {'label': label, 'argv': list(map(str, argv)), 'status': result.returncode,
        'whole_process_tree_wall_ns': time.monotonic_ns()-start, 'stdout': result.stdout, 'stderr': result.stderr,
        'before': before, 'after': env()}
    record_file = label+'.command.json'
    with (out/record_file).open('x') as f: json.dump(record, f, indent=2)
    manifest['commands'].append({'stage': stage, 'label': label, 'record_file': record_file, 'status': result.returncode})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    assert result.returncode == 0, record
    print(label, 'passed', flush=True)

def path(name): return 'tests/'+out.name+'-'+name

try:
    run('provenance', ['git', 'status', '--short', '--branch'])
    run('tracked-clean', ['git', 'diff', '--exit-code', manifest['revision']])
    run('sources', ['make', 'check-sources'])
    run('regression', ['make', 'test'])
    run('sanitizers', ['make', 'sanitize'])
    run('transport', ['python3', '-E', 'tests/test_comparison_evidence.py', '--output', path('transport'), '--runs', '7', '--frames', '200000', '--sanitizers'])
    run('frames', ['python3', '-E', 'tests/test_gpu_frame_evidence.py', '--output', path('frames'), '--baseline', '7d1889d', '--frames', '200000', '--sanitizers', '--reference', path('transport')+'/reference'])
    run('frame-report', ['python3', '-E', 'tests/test_gpu_frame_report.py', path('frames'), '--output', out/'evidence-frame-report.json'])
    for repeat in range(1, 4):
        run('graphics-'+str(repeat), ['python3', '-E', 'tests/test_comparison_graphics_evidence.py', '--output', path('graphics-'+str(repeat))])
    run('comparison-report', ['python3', '-E', 'tests/test_comparison_report.py', '--transport', path('transport'), '--graphics',
        *[path('graphics-'+str(i)) for i in range(1, 4)], '--output', out/'evidence-comparison-report.json'])
    run('lifecycle', ['python3', '-E', 'tests/test_gles_work_evidence.py', '--output', path('lifecycle'), '--production', '--baseline', '2bf623a',
        '--candidate', manifest['revision'], '--pairs', '7', '--frames', '512'])
    run('lifecycle-report', ['python3', '-E', 'tests/test_gles_work_report.py', path('lifecycle'), '--output', out/'evidence-lifecycle-report.json'])
    # Both revisions here have complete EGL release: actual-GLES leak qualification, not a speed comparison.
    run('gles-sanitizers', ['python3', '-E', 'tests/test_gles_work_evidence.py', '--output', path('gles-sanitizers'), '--production', '--baseline', '3aaab60',
        '--candidate', manifest['revision'], '--pairs', '1', '--frames', '32', '--sanitizers'])
    run('hardware', ['sudo', '-n', 'make', 'test-hardware', 'GPU_PARTICIPANTS_GRAPHICS_OUTPUT='+path('hardware')])
    run('microbenchmark', ['make', 'benchmark'])
    run('tracked-clean-after', ['git', 'diff', '--exit-code', manifest['revision']])
    manifest['completed'] = True
finally:
    manifest['after'] = env()
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
