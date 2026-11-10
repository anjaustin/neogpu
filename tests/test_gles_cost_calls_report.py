"""Recompute application-visible GL/EGL/GBM/DRM call costs from raw collector output.

Verifies frozen hashes, exact invocations, call counts, and derives the
startup/steady/teardown buckets independently of the collector's summary.

The bucket is deliberately non-overlapping: gbm_create_device is the outermost
GBM initialization call and no call nested inside it is added again. Connector
enumeration is summed across the init scan and the first-present scan because
both are one-time startup work and the raw CSV does not timestamp each call.

Diagnostic only; preload overhead is measured by the collector's A/B runs.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

if not __debug__:
    raise RuntimeError('Cost report requires assertions enabled')
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('collection', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
collection = args.collection.resolve()
root = collection.parent.parent
manifest = json.loads((collection/'manifest.json').read_text())
assert manifest['completed'], 'collection is not complete'
assert manifest['revision'] == os.popen('git rev-parse HEAD').read().strip()

for name, digest in [(manifest['binary'], manifest['binary_sha256'])]:
    path = Path(name); path = path if path.is_absolute() else root/path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
shim = root/'tests/gles_call_timer.so'
assert hashlib.sha256(shim.read_bytes()).hexdigest() == manifest['shim_sha256']

# One-time startup work. gbm_create_device is outermost; do not add its internals.
STARTUP_CALLS = ('drmSetMaster', 'drmModeGetResources', 'drmModeGetConnector',
                 'drmModeGetEncoder', 'drmModeAddFB', 'drmModeSetCrtc',
                 'gbm_create_device', 'gbm_surface_create', 'gbm_surface_lock_front_buffer',
                 'gbm_surface_release_buffer', 'eglInitialize', 'eglChooseConfig',
                 'eglCreateContext', 'eglCreateWindowSurface')
# Exactly one per run.
SHUTDOWN_CALLS = ('eglDestroySurface', 'eglDestroyContext', 'eglTerminate',
                  'eglReleaseThread', 'gbm_surface_destroy', 'gbm_device_destroy',
                  'drmDropMaster')
STEADY_NAMES = ('eglSwapBuffers', 'glFinish', 'glClear', 'glDrawArrays')

suffix = '.calls.csv'
frame_counts = {}
for run in manifest['runs']:
    draws = run['draws']
    calls_by_label = {}
    for label in sorted(p.name[:-len(suffix)] for p in collection.glob(f'draws{draws}-on-*'+suffix)):
        with (collection/(label+suffix)).open() as handle:
            calls_by_label[label] = {row['call']: {'calls': int(row['calls']),
                                                   'total_ns': int(row['total_ns']),
                                                   'max_ns': int(row['max_ns'])}
                                     for row in csv.DictReader(handle)}
    assert len(calls_by_label) == len(run['baseline']['timed_frames_ns'])
    structures = {tuple(sorted(c)) for c in calls_by_label.values()}
    assert len(structures) == 1, 'preloaded runs disagree on intercepted call set'
    swap_counts = {c['eglSwapBuffers']['calls'] for c in calls_by_label.values()}
    assert len(swap_counts) == 1, swap_counts
    frames = swap_counts.pop() - 11  # initial present + 9 warmup + timed + one snapshot swap
    assert frames >= 8
    frame_counts[draws] = frames

    totals = {}
    for calls in calls_by_label.values():
        for name, value in calls.items():
            entry = totals.setdefault(name, {'calls': 0, 'total_ns': 0, 'max_ns': 0, 'runs': 0})
            entry['calls'] += value['calls']
            entry['total_ns'] += value['total_ns']
            entry['max_ns'] = max(entry['max_ns'], value['max_ns'])
            entry['runs'] += 1
    runs = len(calls_by_label)
    for entry in totals.values():
        entry['calls_per_run'] = entry['calls']/runs
        entry['total_ns_per_run'] = entry['total_ns']/runs
        entry['mean_ns_per_call'] = entry['total_ns']/entry['calls']

    assert totals['eglSwapBuffers']['calls_per_run'] == frames + 11
    assert totals['glFinish']['calls_per_run'] == frames + 10
    assert totals['glDrawArrays']['calls_per_run'] == (frames + 10) * draws
    for name in STARTUP_CALLS:
        assert name in totals and totals[name]['calls_per_run'] >= 1, name
    for name in SHUTDOWN_CALLS:
        assert totals.get(name, {}).get('calls_per_run') == 1, (name, totals.get(name))
    assert totals['eglMakeCurrent']['calls_per_run'] == 2  # one init, one teardown
    assert totals['drmModeGetResources']['calls_per_run'] == 2
    assert totals['drmModeGetConnector']['calls_per_run'] >= 2

    run['recomputed'] = {
        'frames': frames,
        'startup_calls_ns': sum(totals[n]['total_ns_per_run'] for n in STARTUP_CALLS),
        'shutdown_calls_ns': sum(totals[n]['total_ns_per_run'] for n in SHUTDOWN_CALLS),
        'make_current_ns_per_run': totals['eglMakeCurrent']['total_ns_per_run'],
        'steady_mean_ns_per_call': {n: totals[n]['mean_ns_per_call']
                                    for n in STEADY_NAMES if n in totals},
        'startup_items_ns': {n: totals[n]['total_ns_per_run'] for n in STARTUP_CALLS
                             if totals[n]['total_ns_per_run'] >= 1000},
        'calls': totals,
    }
    run['shim_overhead']['elapsed_ratio_recomputed'] = (
        run['preloaded']['elapsed_ns']/run['baseline']['elapsed_ns'])
    assert abs(run['shim_overhead']['elapsed_ratio_recomputed'] -
               run['shim_overhead']['elapsed_ratio']) < 1e-9
    del run['calls']

report = {'manifest_sha256': hashlib.sha256((collection/'manifest.json').read_bytes()).hexdigest(),
          'reducer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'revision': manifest['revision'], 'binary_sha256': manifest['binary_sha256'],
          'shim_sha256': manifest['shim_sha256'], 'frame_counts': frame_counts,
          'runs': manifest['runs'],
          'limitations': ['Application-visible call time only; kernel driver internals need strace.',
                          'Preload adds measured 1.8-3.6% to the timed loop; do not use as throughput.',
                          'glReadPixels is the harness verification readback, outside timed frames.',
                          'gbm_create_device includes Mesa driver/screen initialization and driver loading; '
                          'its internal DRM calls are not added again.',
                          'drmModeGetConnector sums the init scan and the first-present scan; the raw CSV '
                          'does not timestamp individual calls, so the split cannot be recovered here.',
                          'eglMakeCurrent is 2 calls (init + teardown) and is reported separately, not split.',
                          'Buckets are sums of separate call durations, not a critical path.']}
with args.output.open('x') as handle:
    json.dump(report, handle, indent=2)

for run in manifest['runs']:
    rec = run['recomputed']
    print(run['draws'], 'draws: startup calls ms', round(rec['startup_calls_ns']/1e6, 2),
          '| teardown calls ms', round(rec['shutdown_calls_ns']/1e6, 2),
          '| make-current ms', round(rec['make_current_ns_per_run']/1e6, 2),
          '| swap/call us', round(rec['steady_mean_ns_per_call']['eglSwapBuffers']/1e3, 1),
          '| finish/call us', round(rec['steady_mean_ns_per_call']['glFinish']/1e3, 1))
    for name, ns in sorted(rec['startup_items_ns'].items(), key=lambda kv: -kv[1]):
        print(f"    {name:32s} {ns/1e6:8.3f} ms")