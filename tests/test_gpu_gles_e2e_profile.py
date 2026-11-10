"""Separate, whole-process perf/strace diagnostics; never mix these timings into E2E rates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

if not __debug__:
    raise RuntimeError('Profile provenance requires assertions enabled')
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('collection', type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
collection = args.collection.resolve()
out = args.output.resolve()
assert out.parent == ROOT/'tests' and out.name.startswith('evidence-')
out.mkdir(exist_ok=False); out = out.relative_to(ROOT)
manifest = json.loads((collection/'manifest.json').read_text())
assert manifest['completed']
binary = collection/'optimized/benchmark'
assert hashlib.sha256(binary.read_bytes()).hexdigest() == manifest['binary_sha256'][str(binary.relative_to(ROOT))]
records = []
(out/'driver.py').write_bytes(Path(__file__).read_bytes())
for mode in ('native', 'reference'):
    for producers, draws in ((1, 1), (8, 64)):
        label = f'{mode}-{producers}-{draws}'
        for observer in ('perf', 'strace'):
            prefix = out/(observer+'-'+label)
            if observer == 'perf':
                argv = ['sudo', '-n', 'perf', 'stat', '-x,', '-o', str(prefix)+'.counters', '-e',
                    'task-clock,cycles,instructions,branches,branch-misses,cache-references,cache-misses,context-switches,cpu-migrations,page-faults', '--']
            else: argv = ['sudo', '-n', 'strace', '-f', '-c', '-o', str(prefix)+'.syscalls']
            argv += [str(binary.relative_to(ROOT)), mode, str(producers), '256', str(draws), 'on', str(prefix)]
            start = time.monotonic_ns()
            result = subprocess.run(argv, capture_output=True, text=True, timeout=240)
            record = {'observer': observer, 'mode': mode, 'producers': producers, 'draws': draws,
                'argv': argv, 'status': result.returncode, 'whole_process_ns': time.monotonic_ns()-start,
                'stdout': result.stdout, 'stderr': result.stderr}
            (out/(observer+'-'+label+'.json')).write_text(json.dumps(record, indent=2))
            records.append({k: v for k, v in record.items() if k not in ('stdout', 'stderr')})
            # Unsupported/forbidden counters are recorded, not bypassed via policy changes.
            print(observer, label, 'status', result.returncode, flush=True)
(out/'manifest.json').write_text(json.dumps({'completed': True, 'collection_manifest_sha256': hashlib.sha256((collection/'manifest.json').read_bytes()).hexdigest(),
    'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(), 'commands': records,
    'limitations': ['Observers cover whole process including startup, output, verification and cleanup, not gated steady-state GPU work.',
        'Instrumented elapsed times are diagnostic only and excluded from benchmark rates.',
        'Failed counter collection is retained without changing perf permissions or security settings.']}, indent=2))
