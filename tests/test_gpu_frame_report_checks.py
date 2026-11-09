"""Synthetic negative controls for the frame evidence reducer (not timing data)."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

if not __debug__:
    raise RuntimeError('Report controls require Python assertions enabled')

ROOT = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='evidence-report-checks-', dir=ROOT/'tests') as temporary:
    directory = Path(temporary)
    manifest = {'completed': True, 'commands': [], 'source_sha256': {},
                'variants': ['baseline', 'candidate'], 'frames_per_producer': 128, 'runs': []}
    paths = ['driver.py', 'common/test_gpu_frame_benchmark.c', 'common/comparison_rigtorp.cc', 'common/comparison_transport.h']
    paths += [variant+'/'+name for variant in manifest['variants'] for name in ('include/hs_gpu.h', 'src/hs_gpu.c')]
    for name in paths:
        path = directory/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'synthetic snapshot\n')
        manifest['source_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for mode in ('legacy', 'native', 'reference'):
        for producers in ((1,) if mode == 'legacy' else (1, 2, 8)):
            for draws in (1, 64):
                for pair in (1, 2, 3):
                    for variant in manifest['variants']:
                        record = dict(mode=mode, producers=producers, draws=draws, accepted=128*producers,
                                      applied=128*producers, elapsed_ns=1000000, retries=0, backend_calls=128,
                                      init_ns=1000, gpu_bytes=1024, fabric_bytes=128, fabric_address_mod64=0)
                        label = (f'legacy-{draws}-{pair}-{variant}' if mode == 'legacy' else
                                 f'active-{producers}-{draws}-{pair}-{variant}-{mode}')
                        argv = [str(directory/variant/'benchmark'), mode, str(producers), '128', str(draws)]
                        (directory/(label+'.json')).write_text(json.dumps({'status': 0, 'argv': argv, 'stdout': json.dumps(record)}))
                        manifest['commands'].append({'label': label, 'status': 0})
                        manifest['runs'].append(dict(record, variant=variant, pair=pair))
    def run(data, expected, optimized=False):
        (directory/'manifest.json').write_text(json.dumps(data))
        output = directory/(f'output-{run.count}.json'); run.count += 1
        result = subprocess.run([sys.executable, *(['-O'] if optimized else []),
                                 str(ROOT/'tests/test_gpu_frame_report.py'), str(directory), '--output', str(output)],
                                capture_output=True, text=True)
        assert (result.returncode == 0) == expected, result.stderr
        assert output.exists() == expected
    run.count = 0
    run(manifest, True)
    for mutate in (lambda m: m.update(completed=False),
                   lambda m: m['commands'][0].update(status=1),
                   lambda m: m['runs'][0].update(applied=127),
                   lambda m: m['runs'][0].update(elapsed_ns=0),
                   lambda m: m['runs'][0].update(retries=1),
                   lambda m: m['runs'][0].update(pair=2),
                   lambda m: m['runs'].pop(),
                   lambda m: m['commands'].pop(0),
                   lambda m: m['source_sha256'].update({str(directory/'driver.py'): '0'*64}),
                   lambda m: m['source_sha256'].pop(str(directory/'driver.py'))):
        changed = copy.deepcopy(manifest); mutate(changed); run(changed, False)
    raw = directory/'legacy-1-1-baseline.json'
    saved = raw.read_text()
    for argv in (['wrong-binary', 'native', '8', '1', '64'],
                 [str(directory/'candidate/benchmark'), 'legacy', '1', '128', '1']):
        changed = json.loads(saved); changed['argv'] = argv; raw.write_text(json.dumps(changed)); run(manifest, False)
    raw.write_text(saved)
    path = directory/'baseline/src/hs_gpu.c'; path.write_bytes(b'changed snapshot\n')
    run(manifest, False); path.write_bytes(b'synthetic snapshot\n')
    raw.write_text(json.dumps({'status': 0, 'argv': json.loads(saved)['argv'], 'stdout': '{}'}))
    run(manifest, False)
    run(manifest, False, optimized=True)
    print(json.dumps({'synthetic_report_controls': run.count, 'passed': True}))
