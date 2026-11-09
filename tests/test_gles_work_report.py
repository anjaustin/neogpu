"""Verify raw GLES work evidence and reduce paired CPU/presentation costs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

if not __debug__:
    raise RuntimeError('GLES report requires assertions enabled')
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('collection', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
directory = args.collection.resolve(); root = directory.parent.parent
manifest = json.loads((directory/'manifest.json').read_text())
assert manifest['completed'] and all(r['status'] == 0 for r in manifest['commands'])
assert manifest.get('comparison', 'diagnostic-cache') in ('diagnostic-cache', 'committed-default-path')
scenes = ('single', 'same', 'alternate', 'dirty', 'mixed', 'fallback', 'retained')
variants = ('baseline', 'candidate')
audit_frames = manifest.get('audit_frames', 3)
assert type(audit_frames) is int and audit_frames in (3, 4)
assert type(manifest['frames']) is int and 3 <= manifest['frames'] <= 512
assert type(manifest['pairs']) is int and 1 <= manifest['pairs'] <= 9
assert len(manifest['runs']) == len(scenes)*4*manifest['pairs']
assert len(manifest['qualification']) == len(scenes)*2
for group in ('source_sha256', 'binary_sha256'):
    assert manifest[group]
    for name, expected in manifest[group].items():
        path = (root/name).resolve()
        assert path.is_relative_to(directory) and path.is_file(), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
required = [directory/'driver.py', directory/'common/test_gles_work.c']
required += [directory/v/name for v in variants for name in ('include/hs_gpu.h', 'src/hs_backend_gles.c')]
assert all(str(path.relative_to(root)) in manifest['source_sha256'] for path in required)
labels = [r['label'] for r in manifest['commands']]
assert len(labels) == len(set(labels))


def raw(label, scene, capture, variant, audit=False):
    assert label in labels
    record = json.loads((directory/(label+'.json')).read_text())
    expected = ['sudo', '-n', str(directory.relative_to(root)/variant/('audit' if audit else 'timed')),
                scene, capture, str(manifest['frames']), str(directory.relative_to(root)/label), 'audit' if audit else 'timed']
    assert record['status'] == 0 and record['argv'] == expected, label
    records = [json.loads(line) for line in record['stdout'].splitlines() if line.startswith('{')]
    assert len([r for r in records if r['record'] == 'context']) == 1
    cold = [r for r in records if r['record'] == 'cold']; assert len(cold) == 1
    assert type(cold[0]['elapsed_ns']) is int and cold[0]['elapsed_ns'] > 0
    assert cold[0]['journal_allocated'] is (capture == 'on')
    if manifest.get('comparison') == 'committed-default-path':
        lifecycle = [r for r in records if r['record'] == 'lifecycle']; assert len(lifecycle) == 1
        for name in ('init_ns', 'shutdown_ns', 'shutdown_cpu_ns'):
            assert type(lifecycle[0][name]) is int and lifecycle[0][name] > 0
        assert type(lifecycle[0]['context_released']) is bool
        if variant == 'candidate': assert lifecycle[0]['context_released'] is True
    return records


qualified = set()
for entry in manifest['qualification']:
    scene, capture = entry['scene'], entry['capture']
    assert scene in scenes and capture in ('off', 'on') and (scene, capture) not in qualified
    qualified.add((scene, capture)); assert entry['rgba_identical'] is True
    for variant in variants:
        records = raw(f'audit-{scene}-{capture}-{variant}', scene, capture, variant, True)
        calls = [r for r in records if r['record'] == 'calls']; assert len(calls) == 1 and calls[0] == entry['calls'][variant]
        for name in ('program', 'bind', 'enable', 'pointer', 'draw', 'upload'):
            assert type(calls[0][name]) is int and calls[0][name] >= 0
        assert calls[0]['draw'] == 3*{'single': 1, 'same': 64, 'alternate': 64, 'dirty': 2, 'mixed': 12, 'fallback': 2, 'retained': 64}[scene]
        assert calls[0]['upload'] == (6 if scene == 'dirty' else 0)
        context = next(r for r in records if r['record'] == 'context')
        for frame in range(audit_frames):
            image = directory/f'audit-{scene}-{capture}-{variant}-{frame}.rgba'
            data = image.read_bytes(); assert len(data) == context['width']*context['height']*4
            reference = directory/f'audit-{scene}-off-baseline-{frame}.rgba'
            assert data == reference.read_bytes(), image

runs = []
keys = set()
for entry in manifest['runs']:
    scene, capture, variant, pair = (entry[k] for k in ('scene', 'capture', 'variant', 'pair'))
    key = (scene, capture, variant, pair)
    assert scene in scenes and capture in ('off', 'on') and variant in variants
    assert type(pair) is int and 1 <= pair <= manifest['pairs'] and key not in keys
    keys.add(key)
    label = f'timed-{scene}-{pair}-{capture}-{variant}'; assert entry['label'] == label
    records = raw(label, scene, capture, variant)
    blocks = [r for r in records if r['record'] == 'block']; assert len(blocks) == 1
    block = blocks[0]
    for name in ('frames', 'messages', 'presented_ns', 'end_finish_ns', 'process_cpu_ns', 'journal_count'):
        assert type(block[name]) is int and block[name] >= 0, (label, name)
    assert block['frames'] == manifest['frames'] and block['presented_ns'] > 0 and block['process_cpu_ns'] > 0
    assert block['messages'] == manifest['frames']*{'single': 2, 'same': 65, 'alternate': 65, 'dirty': 3, 'mixed': 13, 'fallback': 3, 'retained': 1}[scene]
    assert block['journal_count'] == (block['messages'] if capture == 'on' else 0)
    frames = [r for r in records if r['record'] == 'frame']
    assert [r['index'] for r in frames] == list(range(manifest['frames']))
    for frame in frames:
        for name in ('submit_ns', 'swap_ns', 'total_ns'): assert type(frame[name]) is int and frame[name] > 0
        assert frame['total_ns'] >= frame['submit_ns'] + frame['swap_ns']
    derived = {**block, 'median_submit_ns': statistics.median(r['submit_ns'] for r in frames),
               'median_swap_ns': statistics.median(r['swap_ns'] for r in frames),
               'median_total_ns': statistics.median(r['total_ns'] for r in frames),
               'cold': next(r for r in records if r['record'] == 'cold'), 'variant': variant, 'pair': pair, 'label': label}
    lifecycle = [r for r in records if r['record'] == 'lifecycle']
    if 'lifecycle' in entry or manifest.get('comparison') == 'committed-default-path':
        assert len(lifecycle) == 1
        for name in ('init_ns', 'shutdown_ns', 'shutdown_cpu_ns'):
            assert type(lifecycle[0][name]) is int and lifecycle[0][name] > 0
        assert type(lifecycle[0]['context_released']) is bool
        if variant == 'candidate' and manifest.get('comparison') == 'committed-default-path':
            assert lifecycle[0]['context_released'] is True
        derived['lifecycle'] = lifecycle[0]
    assert derived == entry, label
    runs.append({**entry, 'mean_cpu_ns': block['process_cpu_ns']/block['frames'],
                 'mean_completed_ns': (block['presented_ns']+block['end_finish_ns'])/block['frames']})
    if 'lifecycle' in derived:
        runs[-1].update({name: derived['lifecycle'][name] for name in ('init_ns', 'shutdown_ns', 'shutdown_cpu_ns')})

metrics = ('median_submit_ns', 'median_swap_ns', 'median_total_ns', 'mean_cpu_ns', 'mean_completed_ns')
if all('lifecycle' in r for r in runs): metrics += ('init_ns', 'shutdown_ns', 'shutdown_cpu_ns')


def stats(rows):
    assert len(rows) == manifest['pairs']
    return {metric: {'median': statistics.median(r[metric] for r in rows),
                     'range': [min(r[metric] for r in rows), max(r[metric] for r in rows)]} for metric in metrics}


def ratios(numerator, denominator):
    return {metric: [next(r[metric] for r in numerator if r['pair'] == pair) /
                     next(r[metric] for r in denominator if r['pair'] == pair)
                     for pair in range(1, manifest['pairs']+1)] for metric in metrics}


report = {'manifest_sha256': hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest(),
          'reducer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'scenes': [],
          'comparison': manifest.get('comparison', 'diagnostic-cache'), 'revisions': manifest.get('revisions', {}),
          'limitations': ['Specified workload probes, not every product application.',
              'Normal EGL presentation with one end-block completion wait; no display FPS claim.',
              'Process CPU time includes all process threads, not a GPU execution counter.',
              'Cold frame includes geometry upload and first-use driver/journal work; not isolated journal allocation cost.',
              'Call counts are from separate audit binaries and do not assign equal cost to different GL calls.',
              'Capture logging remains enabled throughout each bounded on-block; reset is outside measured work.',
              'Rendering has 16 warmup frames, but longer journal blocks reach new slots: capture-on includes journal page first-touch, not only fully resident steady-state copying.',
              'Paired timing differences are not additive causal cost partitions.']}
for scene in scenes:
    item = {'scene': scene, 'variants': {}, 'candidate_over_baseline': {}, 'capture_on_over_off': {}}
    for variant in variants:
        item['variants'][variant] = {capture: stats([r for r in runs if r['scene'] == scene and r['variant'] == variant and r['capture'] == capture]) for capture in ('off', 'on')}
        item['capture_on_over_off'][variant] = ratios(
            [r for r in runs if r['scene'] == scene and r['variant'] == variant and r['capture'] == 'on'],
            [r for r in runs if r['scene'] == scene and r['variant'] == variant and r['capture'] == 'off'])
    for capture in ('off', 'on'):
        item['candidate_over_baseline'][capture] = ratios(
            [r for r in runs if r['scene'] == scene and r['variant'] == 'candidate' and r['capture'] == capture],
            [r for r in runs if r['scene'] == scene and r['variant'] == 'baseline' and r['capture'] == capture])
    item['calls_per_three_frames'] = next(q['calls'] for q in manifest['qualification'] if q['scene'] == scene and q['capture'] == 'off')
    report['scenes'].append(item)
with args.output.open('x') as file: json.dump(report, file, indent=2)
for item in report['scenes']:
    print(item['scene'], {v: {c: round(s['median_submit_ns']['median']/1000, 3) for c, s in captures.items()} for v, captures in item['variants'].items()}, 'submit microseconds/frame')
