"""Positive and corrupted-input reducer controls on an isolated actual-evidence copy."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

if not __debug__:
    raise RuntimeError('Reducer controls require assertions enabled')
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('collection', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
source = args.collection.resolve(); out = args.output.resolve()
assert out.parent == source.parent and out.name.startswith('evidence-')
out.mkdir(exist_ok=False)
fixture = out/'tests'/source.name
shutil.copytree(source, fixture)
original = json.loads((fixture/'manifest.json').read_text()); assert original['completed']
index = next(i for i, r in enumerate(original['runs']) if r['mode'] == 'native')
label = original['runs'][index]['label']; raw_path = fixture/(label+'.json')
original_raw = json.loads(raw_path.read_text())
results = []
for case in ('positive', 'raw_discordance', 'live_resources', 'wrong_argv', 'sequence_duplicate', 'frame_missing',
             'wrong_pixel_hash', 'source_hash', 'binary_hash', 'run_missing', 'wrong_pair', 'zero_elapsed', 'forged_latency'):
    manifest, command = copy.deepcopy(original), copy.deepcopy(original_raw)
    record = manifest['runs'][index]
    raw = [json.loads(line) for line in command['stdout'].splitlines() if line.startswith('{')]
    summary = next(r for r in raw if r['record'] == 'summary')
    if case == 'raw_discordance': record['elapsed_ns'] += 1
    elif case == 'live_resources': record['resources_released'] = summary['resources_released'] = False
    elif case == 'wrong_argv': command['argv'][-2] = 'wrong-capture'
    elif case == 'sequence_duplicate': next(r for r in raw if r['record'] == 'contribution')['sequence'] = 2
    elif case == 'frame_missing': raw.remove(next(r for r in raw if r['record'] == 'frame'))
    elif case == 'wrong_pixel_hash': record['rgba_sha256'] = '0'*64
    elif case == 'source_hash': manifest['source_sha256'][next(iter(manifest['source_sha256']))] = '0'*64
    elif case == 'binary_hash': manifest['binary_sha256'][next(iter(manifest['binary_sha256']))] = '0'*64
    elif case == 'run_missing': manifest['runs'].pop()
    elif case == 'wrong_pair': record['pair'] = 0
    elif case == 'zero_elapsed': record['elapsed_ns'] = summary['elapsed_ns'] = 0
    elif case == 'forged_latency': record['contribution_latency_ns']['p99'] += 1
    command['stdout'] = '\n'.join(json.dumps(r) for r in raw)+'\n'
    (fixture/'manifest.json').write_text(json.dumps(manifest)); raw_path.write_text(json.dumps(command))
    result = subprocess.run([sys.executable, '-E', str(Path(__file__).with_name('test_gpu_gles_e2e_report.py')), str(fixture),
        '--output', str(out/(case+'-report.json'))], capture_output=True, text=True)
    assert (result.returncode == 0) == (case == 'positive'), (case, result.stdout, result.stderr)
    results.append({'case': case, 'status': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
(fixture/'manifest.json').write_text(json.dumps(original)); raw_path.write_text(json.dumps(original_raw))
(out/'controls.json').write_text(json.dumps(results, indent=2))
print('Actual evidence positive control accepted;', len(results)-1, 'corrupt controls rejected.')
