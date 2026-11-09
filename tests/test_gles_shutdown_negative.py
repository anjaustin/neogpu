"""Exercise shutdown reducer rejection paths on an isolated evidence copy."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('collection', type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
source = args.collection.resolve()
output = args.output.resolve()
assert output.name.startswith('evidence-') and output.parent == source.parent
output.mkdir(exist_ok=False)
fixture = output/'tests'/source.name
shutil.copytree(source, fixture)
reducer = Path(__file__).with_name('test_gles_shutdown_report.py')
original = json.loads((fixture/'manifest.json').read_text())
assert original['completed'] and original['variants'][-1] in ('flip_restore', 'restore_crtc')
index = next(i for i, r in enumerate(original['runs']) if r['variant'] == original['variants'][-1])
run = original['runs'][index]
label = f"{run['scene']}-{run['pair']}-{run['variant']}"
raw_path = fixture/(label+'.json')
original_raw = json.loads(raw_path.read_text())
results = []

cases = ['positive', 'raw_discordance', 'context_live', 'fd_leak', 'wrong_argv', 'missing_mode', 'source_hash']
if run['variant'] == 'flip_restore': cases += ['unverified_flip', 'wrong_target']
for case in cases:
    manifest, command = copy.deepcopy(original), copy.deepcopy(original_raw)
    records = [json.loads(line) for line in command['stdout'].splitlines() if line.startswith('{')]
    record = manifest['runs'][index]
    def change(group, kind, key, value):
        record[group][key] = value
        next(r for r in records if r['record'] == kind)[key] = value
    if case == 'raw_discordance': record['lifecycle']['shutdown_ns'] += 1
    elif case == 'context_live': change('lifecycle', 'lifecycle', 'context_released', False)
    elif case == 'fd_leak': change('resources', 'shutdown_resources', 'remaining_fds', record['resources']['initial_fds']+1)
    elif case == 'wrong_argv': command['argv'][-1] = 'audit'
    elif case == 'unverified_flip': change('flip_audit', 'flip_audit', 'completed', 1 - record['flip_audit']['completed'])
    elif case == 'wrong_target': change('flip_audit', 'flip_audit', 'target', record['mode_audit']['saved_fb']+1)
    elif case == 'missing_mode':
        del record['mode_audit']
        records = [r for r in records if r['record'] != 'mode_audit']
    elif case == 'source_hash': manifest['source_sha256'][next(iter(manifest['source_sha256']))] = '0'*64
    command['stdout'] = '\n'.join(json.dumps(r) for r in records)+'\n'
    (fixture/'manifest.json').write_text(json.dumps(manifest))
    raw_path.write_text(json.dumps(command))
    report = output/(case+'-report.json')
    result = subprocess.run([sys.executable, '-E', str(reducer), str(fixture), '--output', str(report)], capture_output=True, text=True)
    accepted = result.returncode == 0
    assert accepted == (case == 'positive'), (case, result.stdout, result.stderr)
    results.append({'case': case, 'status': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
# Restore copied inputs; originals were never modified. Keep all diagnostic outputs.
(fixture/'manifest.json').write_text(json.dumps(original))
raw_path.write_text(json.dumps(original_raw))
(output/'negative-controls.json').write_text(json.dumps(results, indent=2))
print(f"Positive control accepted; {len(results)-1} corrupted evidence controls rejected.")
