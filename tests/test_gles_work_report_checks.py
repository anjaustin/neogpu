"""Synthetic integrity controls for the GLES report, not performance data."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

if not __debug__:
    raise RuntimeError('GLES controls require assertions enabled')
ROOT = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='evidence-gles-report-controls-', dir=ROOT/'tests') as temporary:
    directory = Path(temporary); relative = directory.relative_to(ROOT)
    manifest = {'completed': True, 'frames': 3, 'pairs': 1, 'audit_frames': 4, 'commands': [], 'source_sha256': {}, 'binary_sha256': {}, 'qualification': [], 'runs': []}
    manifest['comparison'] = 'committed-default-path'
    paths = ['driver.py', 'common/test_gles_work.c']
    paths += [v+'/'+p for v in ('baseline', 'candidate') for p in ('include/hs_gpu.h', 'src/hs_backend_gles.c')]
    for name in paths + [v+'/'+p for v in ('baseline', 'candidate') for p in ('audit', 'timed')]:
        path = directory/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'synthetic fixture\n')
        group = 'source_sha256' if name in paths else 'binary_sha256'
        manifest[group][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for scene, draws, commands in [('single', 1, 2), ('same', 64, 65), ('alternate', 64, 65), ('dirty', 2, 3), ('mixed', 12, 13), ('fallback', 2, 3), ('retained', 64, 1)]:
        for capture in ('off', 'on'):
            qualification = {'scene': scene, 'capture': capture, 'rgba_identical': True, 'calls': {}}
            for variant in ('baseline', 'candidate'):
                context = {'record': 'context', 'width': 1, 'height': 1}
                cold = {'record': 'cold', 'elapsed_ns': 1000, 'journal_allocated': capture == 'on'}
                lifecycle = dict(record='lifecycle', init_ns=1000, shutdown_ns=100, shutdown_cpu_ns=100, context_released=variant == 'candidate')
                for audit in (True, False):
                    label = f'audit-{scene}-{capture}-{variant}' if audit else f'timed-{scene}-1-{capture}-{variant}'
                    argv = ['sudo', '-n', str(relative/variant/('audit' if audit else 'timed')), scene, capture, '3', str(relative/label), 'audit' if audit else 'timed']
                    if audit:
                        calls = dict(record='calls', draw=draws*3, upload=6 if scene == 'dirty' else 0, program=3, bind=3, enable=6, pointer=6)
                        qualification['calls'][variant] = calls
                        records = [context, cold, calls, lifecycle]
                        for frame in range(4): (directory/f'{label}-{frame}.rgba').write_bytes(b'\0\0\0\xff')
                    else:
                        frames = [dict(record='frame', index=i, submit_ns=10, swap_ns=20, total_ns=40) for i in range(3)]
                        block = dict(record='block', scene=scene, capture=capture, frames=3, messages=commands*3, presented_ns=150,
                                     end_finish_ns=15, process_cpu_ns=120, journal_count=commands*3 if capture == 'on' else 0)
                        records = [context, cold, *frames, block, lifecycle]
                        manifest['runs'].append(dict(block, median_submit_ns=10, median_swap_ns=20, median_total_ns=40,
                                                     cold=cold, lifecycle=lifecycle, variant=variant, pair=1, label=label))
                    (directory/(label+'.json')).write_text(json.dumps({'argv': argv, 'status': 0, 'stdout': '\n'.join(json.dumps(r) for r in records)}))
                    manifest['commands'].append({'label': label, 'status': 0})
            manifest['qualification'].append(qualification)

    def run(data, expected, optimized=False):
        (directory/'manifest.json').write_text(json.dumps(data))
        output = directory/f'report-{run.count}.json'; run.count += 1
        result = subprocess.run([sys.executable, *(['-O'] if optimized else []), str(ROOT/'tests/test_gles_work_report.py'),
                                 str(directory), '--output', str(output)], capture_output=True, text=True)
        assert (result.returncode == 0) == expected, result.stderr
        assert output.exists() == expected
    run.count = 0
    run(manifest, True)
    for mutate in (lambda m: m.update(completed=False), lambda m: m['runs'].pop(),
                   lambda m: m['runs'][0].update(median_submit_ns=11),
                   lambda m: m['commands'][0].update(status=1),
                   lambda m: m['qualification'][0]['calls']['baseline'].update(draw=0),
                   lambda m: m['source_sha256'].update({str(relative/'driver.py'): '0'*64}),
                   lambda m: m['runs'].__setitem__(0, copy.deepcopy(m['runs'][1]))):
        changed = copy.deepcopy(manifest); mutate(changed); run(changed, False)
    path = directory/'timed-single-1-off-baseline.json'; saved = path.read_bytes()
    record = json.loads(saved); record['argv'][3] = 'wrong-scene'; path.write_text(json.dumps(record))
    run(manifest, False); path.write_bytes(saved)
    path = directory/'audit-single-on-candidate-0.rgba'; path.write_bytes(b'\xff\0\0\xff')
    run(manifest, False); path.write_bytes(b'\0\0\0\xff')
    path = directory/'candidate/timed'; path.write_bytes(b'changed binary\n')
    run(manifest, False); path.write_bytes(b'synthetic fixture\n')
    path = directory/'timed-single-1-off-candidate.json'; saved = path.read_bytes()
    for field, value in [('context_released', False), ('shutdown_ns', -1)]:
        changed = copy.deepcopy(manifest)
        entry = next(r for r in changed['runs'] if r['label'] == path.stem)
        entry['lifecycle'][field] = value
        raw = json.loads(saved); records = [json.loads(line) for line in raw['stdout'].splitlines()]
        next(r for r in records if r['record'] == 'lifecycle')[field] = value
        raw['stdout'] = '\n'.join(json.dumps(r) for r in records); path.write_text(json.dumps(raw))
        run(changed, False)
    path.write_bytes(saved)
    run(manifest, False, optimized=True)
    print(json.dumps({'gles_report_controls': run.count, 'passed': True}))
