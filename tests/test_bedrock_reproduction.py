"""Verify reference pinning and evidence overwrite rejection without modifying inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from bedrock_driver import roi_driver

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--qualified', required=True, type=Path)
parser.add_argument('--completed', required=True, type=Path)
parser.add_argument('--reference', type=Path, default=Path('tests/evidence-comparison-final/reference'))
args = parser.parse_args()
if not args.output.name.startswith('evidence-') or not args.output.resolve().is_relative_to(ROOT/'tests'):
    parser.error('output must be a fresh tests/evidence-* directory')
args.output.mkdir(exist_ok=False)
reference = args.output/'synthetic-bad-reference'
(reference/'include/rigtorp').mkdir(parents=True)
(reference/'include/rigtorp/SPSCQueue.h').write_bytes((args.reference/'include/rigtorp/SPSCQueue.h').read_bytes())
(reference/'LICENSE').write_bytes((args.reference/'LICENSE').read_bytes()+b'\nSYNTHETIC HASH MISMATCH CONTROL\n')


def reject(argv, label, message, status):
    result = subprocess.run([sys.executable, *map(str, argv)], text=True, capture_output=True, timeout=30)
    (args.output/(label+'.json')).write_text(json.dumps({'argv': list(map(str, argv)), 'status': result.returncode,
        'stdout': result.stdout, 'stderr': result.stderr}, indent=2))
    assert result.returncode == status and message in result.stderr, result.stderr


for script in ('test_bedrock_p0.py', 'test_bedrock_p0_pmu.py'):
    label = script.removesuffix('.py')
    output = args.output/('evidence-'+label)
    command = ['tests/'+script, '--output', output, '--reference', reference]
    command += ['--source', args.qualified] if script.endswith('_pmu.py') else []
    reject(command, label, 'LICENSE', 1)
    failed = json.loads((output/'manifest.json').read_text())
    assert not failed['completed']
    if script == 'test_bedrock_p0.py': assert failed['candidate'] == 'split'
manifest = args.completed/'manifest.json'
before = hashlib.sha256(manifest.read_bytes()).hexdigest()
reject(['tests/test_bedrock.py', '--output', args.completed, '--phase', 'count-pmu'],
       'completed-phase', 'phase already completed', 2)
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == before
generated = roi_driver(Path('tests/test_comparison_transport.c').read_text())
(args.output/'roi-driver.c').write_text(generated)
historical = Path('tests/evidence-bedrock-untraced/control/test_comparison_transport.c')
if historical.exists():
    assert generated == historical.read_text(), 'ROI generation changed measured driver work'
print('Reference tampering and completed-phase overwrite rejected; ROI source generation verified')
