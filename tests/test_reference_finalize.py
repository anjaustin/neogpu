"""Post-E2E serial diagnostics, forced graphics rebuild, and final evidence gates."""
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
parser.add_argument('--e2e', required=True, type=Path)
parser.add_argument('--transport', required=True, type=Path)
args = parser.parse_args()
out = args.output.resolve(); assert out.parent == ROOT/'tests' and out.name.startswith('evidence-')
out.mkdir(exist_ok=False); out = out.relative_to(ROOT)
manifest = {'completed': False, 'commands': [], 'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
(out/'driver.py').write_bytes(Path(__file__).read_bytes())

def run(label, argv):
    start = time.monotonic_ns()
    result = subprocess.run(['/usr/bin/time', '-v', '-o', str(out/(label+'.time')), *map(str, argv)], capture_output=True, text=True, timeout=1200)
    record = {'argv': list(map(str, argv)), 'status': result.returncode, 'whole_process_ns': time.monotonic_ns()-start,
        'stdout': result.stdout, 'stderr': result.stderr}
    with (out/(label+'.command.json')).open('x') as f: json.dump(record, f, indent=2)
    manifest['commands'].append({'label': label, 'status': result.returncode})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    assert result.returncode == 0, record
    print(label, 'passed', flush=True)

try:
    run('e2e-report', ['python3', '-E', 'tests/test_gpu_gles_e2e_report.py', args.e2e, '--output', out/'evidence-e2e-report.json'])
    run('e2e-report-controls', ['python3', '-E', 'tests/test_gpu_gles_e2e_report_checks.py', args.e2e, '--output', str(out)+'-e2e-controls'])
    run('profiles', ['python3', '-E', 'tests/test_gpu_gles_e2e_profile.py', args.e2e, '--output', str(out)+'-profiles'])
    run('force-graphics-build', ['make', '-B', 'tests/test_comparison_graphics'])
    manifest['graphics_binary_sha256'] = hashlib.sha256(Path('tests/test_comparison_graphics').read_bytes()).hexdigest()
    graphics = []
    for i in range(1, 4):
        directory = str(out)+'-graphics-'+str(i); graphics.append(directory)
        run('graphics-'+str(i), ['python3', '-E', 'tests/test_comparison_graphics_evidence.py', '--output', directory])
        assert hashlib.sha256(Path('tests/test_comparison_graphics').read_bytes()).hexdigest() == manifest['graphics_binary_sha256']
    run('comparison-report', ['python3', '-E', 'tests/test_comparison_report.py', '--transport', args.transport, '--graphics', *graphics,
        '--output', out/'evidence-comparison-report.json'])
    run('sources', ['make', 'check-sources'])
    run('validator-controls', ['python3', '-E', 'tests/test_gpu_gles_e2e_checks.py'])
    run('regression', ['make', 'test'])
    run('runtime-clean', ['git', 'diff', '--exit-code', manifest['revision'], '--', 'src', 'include', 'Makefile'])
    run('working-tree', ['git', 'status', '--short', '--branch'])
    manifest['completed'] = True
finally:
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
