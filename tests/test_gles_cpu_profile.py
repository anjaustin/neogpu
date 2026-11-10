"""Flat CPU attribution with perf for the frame-cost probe and the real-GLES harness.

Saves raw perf reports. Kernel symbols are restricted on this host; userspace
symbols and driver-library exports are what this attributes. Diagnostic only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

if not __debug__:
    raise RuntimeError('Profile evidence requires assertions enabled')
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--probe-frames', type=int, default=2048)
args = parser.parse_args()
assert 8 <= args.probe_frames <= 4096
out = args.output.resolve()
assert out.parent == ROOT/'tests' and out.name.startswith('evidence-')
out.mkdir(exist_ok=False)
out = out.relative_to(ROOT)

PROBE = Path('tests/test_gles_frame_cost_profile')
subprocess.run(['gcc', '-O3', '-g', '-march=armv8-a+simd', '-mtune=cortex-a72', '-ffast-math',
                '-Wall', '-Wextra', '-Werror', '-DNDEBUG', '-Iinclude', '-I/usr/include/libdrm',
                '-o', str(PROBE), 'tests/test_gles_frame_cost.c',
                '-lm', '-lGLESv2', '-lgbm', '-ldrm', '-lEGL'], check=True)
E2E = Path('tests/evidence-e2e-c9307e1-bounded/optimized/benchmark')
assert E2E.is_file()

# label, argv after perf
frames = str(args.probe_frames)
CASES = [
    ('probe-window-sync-d1', [str(PROBE.resolve()), 'window', '0', frames, '1', '1', 'on', str(out/'probe-window-sync-d1')]),
    ('probe-window-sync-d64', [str(PROBE.resolve()), 'window', '0', frames, '64', '1', 'on', str(out/'probe-window-sync-d64')]),
    ('probe-window-noswap-sync-d0', [str(PROBE.resolve()), 'window', '0', frames, '0', '1', 'off', str(out/'probe-window-noswap-sync-d0')]),
    ('probe-init-only', [str(PROBE.resolve()), 'window', '0', '8', '1', '1', 'on', str(out/'probe-init-only')]),
    ('e2e-native-8p64d', [str(E2E.resolve()), 'native', '8', '256', '64', 'on', str(out/'e2e-native-8p64d')]),
]

manifest = {'completed': False, 'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'probe_sha256': hashlib.sha256(PROBE.read_bytes()).hexdigest(),
            'e2e_sha256': hashlib.sha256(E2E.read_bytes()).hexdigest(), 'cases': [],
            'contract': 'Flat self-time samples from perf record at 4000 Hz. Kernel symbols are '
                        'restricted, so kernel driver functions appear as unknown addresses; no '
                        'security setting was changed. Sampling perturbs the workload; these are '
                        'attribution diagnostics, not rates.'}
(out/'driver.py').write_bytes(Path(__file__).read_bytes())
flat_pattern = re.compile(r'^\s+([0-9.]+)%\s+(\S+)\s+(\S+)\s+(\S.*)$')

def environment():
    data = {'wall_ns': time.time_ns(), 'affinity': sorted(os.sched_getaffinity(0)), 'sysfs': {}}
    for base, pattern in [('/sys/devices/system/cpu', 'cpu*/cpufreq/scaling_*'), ('/sys/class/thermal', 'thermal_zone*/temp')]:
        for path in sorted(Path(base).glob(pattern)):
            if path.is_file():
                try: data['sysfs'][str(path)] = path.read_text().strip()
                except OSError as error: data['sysfs'][str(path)] = str(error)
    return data

def command(argv, label, data_file=None):
    before = environment(); start = time.monotonic_ns()
    result = subprocess.run(['sudo', '-n', *argv], capture_output=True, text=True, timeout=600,
                            env={**os.environ, 'LC_ALL': 'C'})
    record = {'label': label, 'argv': argv, 'status': result.returncode,
              'wall_ns': time.monotonic_ns()-start, 'stdout': result.stdout, 'stderr': result.stderr,
              'before': before, 'after': environment()}
    with (out/(label+'.command.json')).open('x') as handle:
        json.dump(record, handle, indent=2)
    assert result.returncode == 0, record
    return record

manifest['environment'] = environment()
try:
    for label, argv in CASES:
        data_file = out/(label+'.data')
        recorded = command(['perf', 'record', '-F', '4000', '-g', '--call-graph', 'dwarf',
                            '-o', str(data_file.resolve()), '--', *argv], label+'-record')
        captured = re.search(r'\((\d+) samples\)', recorded['stderr'])
        assert captured and int(captured.group(1)) > 100, recorded['stderr']
        flat = command(['perf', 'report', '-i', str(data_file.resolve()), '--stdio', '-g', 'none',
                        '--no-children', '--sort', 'dso,symbol', '--percent-limit', '0.4'], label+'-flat')
        (out/(label+'.flat.txt')).write_text(flat['stdout'])
        children = command(['perf', 'report', '-i', str(data_file.resolve()), '--stdio', '--children',
                            '--sort', 'dso,symbol', '--percent-limit', '0.4'], label+'-children')
        (out/(label+'.children.txt')).write_text(children['stdout'])
        samples = int(captured.group(1))
        entries = []
        for line in flat['stdout'].splitlines():
            match = flat_pattern.match(line)
            if match:
                entries.append({'percent': float(match.group(1)), 'comm': match.group(2),
                                'dso': match.group(3), 'symbol': match.group(4).strip()})
        assert entries, (label, flat['stdout'][:800])
        manifest['cases'].append({'label': label, 'argv': argv, 'samples': samples,
                                  'flat_top': entries[:25]})
        print(label, 'samples', samples, 'top', entries[0]['symbol'], flush=True)
    manifest['completed'] = True
finally:
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
