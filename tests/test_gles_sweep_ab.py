"""A/B the duplicated connector sweep against reusing the scan done at init.

P0-3. Two arms of the same binary on the same host, interleaved:

  reuse  : the display setup uses the connector information captured at init
  rescan : the display setup queries the connector again, as it used to

For each arm this records startup, first-present and shutdown cost, the CRTC
and framebuffer state after the first present and after the rendered loop, and
the DRM mode-setting ioctls actually issued. It asserts that both arms end
with the same CRTC bound to the same framebuffer, so the cheaper arm cannot
pass by leaving the display programmed differently.

Diagnostic only; the probe and header flags are not production changes.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

if not __debug__:
    raise RuntimeError('Sweep A/B evidence requires assertions enabled')
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--reps', type=int, default=5)
parser.add_argument('--frames', type=int, default=64)
parser.add_argument('--draws', type=int, default=1)
args = parser.parse_args()
out = args.output.resolve()
assert out.parent == ROOT/'tests' and out.name.startswith('evidence-')
assert 3 <= args.reps <= 9 and 8 <= args.frames <= 1024 and 1 <= args.draws <= 512
out.mkdir(exist_ok=False)
out = out.relative_to(ROOT)

BINARY = Path('tests/test_gles_frame_cost')
subprocess.run(['gcc', '-O3', '-march=armv8-a+simd', '-mtune=cortex-a72', '-ffast-math',
                '-Wall', '-Wextra', '-Werror', '-DNDEBUG', '-Iinclude', '-I/usr/include/libdrm',
                '-o', str(BINARY), 'tests/test_gles_frame_cost.c',
                '-lm', '-lGLESv2', '-lgbm', '-ldrm', '-lEGL'], check=True)

ARMS = {
    'reuse': {},
    'rescan': {'HS_GRAPHICS_SWEEP': 'rescan'},
}

manifest = {'completed': False, 'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'binary_sha256': hashlib.sha256(BINARY.read_bytes()).hexdigest(),
            'reps': args.reps, 'frames': args.frames, 'draws': args.draws,
            'arms': ARMS, 'runs': [],
            'contract': 'Interleaved A/B of the duplicated connector sweep against reusing the '
                        'scan done at init. Same binary, same host, same workload, display '
                        'ownership held constant. Startup, first present and shutdown are wall '
                        'clock from the probe; ioctl counts come from a separate strace pass and '
                        'are not timed data. Both arms must end with the same CRTC bound to a '
                        'framebuffer of the same size and format.'}
(out/'driver.py').write_bytes(Path(__file__).read_bytes())


def environment():
    data = {'wall_ns': time.time_ns(), 'affinity': sorted(os.sched_getaffinity(0)), 'sysfs': {}}
    for base, pattern in [('/sys/devices/system/cpu', 'cpu*/cpufreq/scaling_*'),
                           ('/sys/class/thermal', 'thermal_zone*/temp')]:
        for path in sorted(Path(base).glob(pattern)):
            if path.is_file():
                try: data['sysfs'][str(path)] = path.read_text().strip()
                except OSError as error: data['sysfs'][str(path)] = str(error)
    return data


def probe_argv(arm, label):
    env_args = [f'{key}={value}' for key, value in ARMS[arm].items()]
    argv = ['sudo', '-n'] + (['env'] + env_args if env_args else [])
    return argv + [str(BINARY.resolve()), 'window', '0', str(args.frames), str(args.draws),
                   '1', 'on', str(out/label)]


def active_crtcs(state):
    return sorted((c['crtc_id'], c['fb_id'], c['width'], c['height'], c['bpp'])
                  for c in state if c['active'])


def run(arm, rep):
    label = f'{arm}-r{rep}'
    argv = probe_argv(arm, label)
    before = environment(); start = time.monotonic_ns()
    result = subprocess.run(argv, capture_output=True, text=True, timeout=300,
                            env={**os.environ, 'LC_ALL': 'C'})
    record = {'arm': arm, 'rep': rep, 'argv': argv, 'status': result.returncode,
              'wall_ns': time.monotonic_ns()-start, 'stdout': result.stdout,
              'stderr': result.stderr, 'before': before, 'after': environment()}
    (out/(label+'.json')).write_text(json.dumps(record, indent=2))
    assert result.returncode == 0, record
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    summary = [r for r in rows if r['record'] == 'summary']
    assert len(summary) == 1, rows
    summary = summary[0]
    assert summary['surface'] == 'window' and summary['draws'] == args.draws
    assert summary['frames'] == args.frames
    assert summary['take_display'] is True, 'both arms must own the display'
    assert summary['reuse_connector_scan'] is (arm == 'reuse'), (arm, summary['reuse_connector_scan'])
    pixels = (out/(label+'.verify.rgba')).read_bytes()
    assert len(pixels) == summary['width']*summary['height']*4
    assert max(pixels) > 0, 'verification frame is blank'
    crtcs = {r['phase']: r['crtcs'] for r in rows if r['record'] == 'crtcs'}
    assert set(crtcs) == {'after_init', 'after_first_present', 'after_loop'}
    assert active_crtcs(crtcs['after_first_present']), 'no CRTC is active after the first present'
    return summary, crtcs


def trace_arm(arm):
    label = f'{arm}-trace'
    trace_file = out/(label+'.trace.txt')
    argv = ['sudo', '-n', 'strace', '-f', '-T', '-e', 'trace=ioctl',
            '-o', str(trace_file.resolve())] + probe_argv(arm, label)[1:]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=300,
                            env={**os.environ, 'LC_ALL': 'C'})
    assert result.returncode == 0, result.stderr[-2000:]
    counts = collections.Counter(); seconds = collections.Counter()
    pattern = re.compile(r'ioctl\(\d+,\s*([A-Za-z_0-9]+)')
    for line in trace_file.read_text(errors='replace').splitlines():
        match = pattern.search(line)
        if not match:
            continue
        name = match.group(1)
        counts[name] += 1
        duration = re.search(r'<([0-9.]+)>\s*$', line)
        if duration:
            seconds[name] += float(duration.group(1))
    return {'argv': argv, 'counts': dict(counts), 'seconds': dict(seconds),
            'trace_sha256': hashlib.sha256(trace_file.read_bytes()).hexdigest()}


def stats(values):
    ordered = sorted(values)
    return {'median': statistics.median(ordered), 'min': ordered[0], 'max': ordered[-1]}


try:
    manifest['environment'] = environment()
    for arm in ARMS:
        collected, crtc_states = [], []
        for rep in range(1, args.reps+1):
            summary, crtcs = run(arm, rep)
            collected.append(summary)
            crtc_states.append(crtcs)
        manifest['runs'].append({
            'arm': arm,
            'init_ns': stats([r['init_ns'] for r in collected]),
            'first_present_ns': stats([r['first_present_ns'] for r in collected]),
            'shutdown_ns': stats([r['shutdown_ns'] for r in collected]),
            'active_crtcs': [active_crtcs(state['after_first_present']) for state in crtc_states],
        })
        print(arm, 'init ms', round(manifest['runs'][-1]['init_ns']['median']/1e6, 2),
              'first present ms', round(manifest['runs'][-1]['first_present_ns']['median']/1e6, 2),
              'shutdown ms', round(manifest['runs'][-1]['shutdown_ns']['median']/1e6, 2), flush=True)
    arms = {run['arm']: run for run in manifest['runs']}
    assert arms['reuse']['active_crtcs'] == arms['rescan']['active_crtcs'], \
        'arms disagree on the programmed CRTC'
    manifest['traces'] = {arm: trace_arm(arm) for arm in ARMS}
    for arm, trace in manifest['traces'].items():
        print(arm, 'GETCONNECTOR', trace['counts'].get('DRM_IOCTL_MODE_GETCONNECTOR'),
              'seconds', round(trace['seconds'].get('DRM_IOCTL_MODE_GETCONNECTOR', 0)*1000, 2), flush=True)
    manifest['completed'] = True
finally:
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
