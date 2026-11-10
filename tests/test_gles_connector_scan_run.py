"""Run the connector selection test, including its negative control.

Diagnostic and regression tooling. The plain and the sanitised build must both
be clean; the negative control must trip the sanitiser, which is the proof
that the double free it reproduces was a real defect and not a theory.

Output stays in a fresh tests/evidence-* directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

if not __debug__:
    raise RuntimeError('Connector scan evidence requires assertions enabled')
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
out = args.output.resolve()
assert out.parent == ROOT/'tests' and out.name.startswith('evidence-')
out.mkdir(exist_ok=False)
out = out.relative_to(ROOT)

PLAIN = Path('tests/test_gles_connector_scan')
ASAN = Path('tests/test_gles_connector_scan_asan')
COMMON = ['-O1', '-g', '-Wall', '-Wextra', '-Iinclude', '-I/usr/include/libdrm',
          'tests/test_gles_connector_scan.c', '-ldrm', '-lEGL', '-lGLESv2', '-lgbm']
subprocess.run(['gcc', *COMMON, '-o', str(PLAIN)], check=True)
subprocess.run(['gcc', *COMMON, '-fsanitize=address,undefined',
                '-fno-omit-frame-pointer', '-o', str(ASAN)], check=True)

manifest = {'completed': False,
            'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'plain_sha256': hashlib.sha256(PLAIN.read_bytes()).hexdigest(),
            'asan_sha256': hashlib.sha256(ASAN.read_bytes()).hexdigest(),
            'cases': [],
            'contract': 'The selection helper must find the attached display whatever its type, '
                        'must select nothing when no connector matches, and must release every '
                        'connector exactly once on both paths. The negative control replays the '
                        'pre-fix selection loop with a filter nothing matches and must trip the '
                        'sanitiser, which is the evidence that the defect was real.'}
(out/'driver.py').write_bytes(Path(__file__).read_bytes())


def run(binary, args, expect):
    argv = ['sudo', '-n', str(binary.resolve()), *args]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=120,
                            env={**os.environ, 'LC_ALL': 'C', 'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=0'})
    output = result.stdout + result.stderr
    (out/(binary.stem + ('-' + '-'.join(args) if args else '') + '.log')).write_text(output)
    case = {'binary': binary.name, 'args': args, 'expect': expect,
            'status': result.returncode, 'output': output[-4000:]}
    if expect == 'clean':
        assert result.returncode == 0, case
        assert 'ERROR: AddressSanitizer' not in output, case
        assert 'runtime error' not in output, case
    else:
        assert result.returncode != 0, case
        assert 'AddressSanitizer' in output, case
        double_free = re.search(r'attempting double-free|double-free', output)
        case['double_free_reported'] = bool(double_free)
        assert double_free, case
    manifest['cases'].append(case)
    print(f"{binary.name} {' '.join(args) or '(no args)'} -> rc {result.returncode} "
          f"(expected {expect})", flush=True)


try:
    run(PLAIN, [], 'clean')
    run(ASAN, [], 'clean')
    run(ASAN, ['--legacy'], 'report')
    manifest['completed'] = True
finally:
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
