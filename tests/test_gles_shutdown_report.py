"""Recompute shutdown experiments from raw commands, not cached summaries."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

if not __debug__:
    raise RuntimeError('Shutdown reports require assertions enabled')
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('collection', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
directory = args.collection.resolve(); root = directory.parent.parent
manifest = json.loads((directory/'manifest.json').read_text())
assert manifest['completed'] and all(c['status'] == 0 for c in manifest['commands'])
variants = manifest.get('variants', ['baseline', 'close_only'])
assert variants in (['baseline', 'close_only'], ['baseline', 'owned_fb', 'rmfb_early'], ['baseline', 'saved_state', 'restore_crtc'], ['baseline', 'saved_state', 'flip_restore'])
assert type(manifest['pairs']) is int and 3 <= manifest['pairs'] <= 31
assert type(manifest['frames']) is int and 3 <= manifest['frames'] <= 512
for group in ('source_sha256', 'binary_sha256'):
    for name, digest in manifest.get(group, {}).items():
        path = (root/name).resolve(); assert path.is_relative_to(directory) and path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
required = ['driver.py', 'harness.c']+[v+'/'+n for v in variants for n in ('include/hs_graphics.h', 'src/hs_backend_gles.c')]
assert all(str((directory/name).relative_to(root)) in manifest['source_sha256'] for name in required)
assert len(manifest['runs']) == 2 * len(variants) * manifest['pairs']
labels = [c['label'] for c in manifest['commands']]; assert len(labels) == len(set(labels))
runs = []; keys = set()
for record in manifest['runs']:
    scene, pair, variant = (record[k] for k in ('scene', 'pair', 'variant'))
    key = scene, pair, variant
    assert scene in ('single', 'retained') and variant in variants and key not in keys
    assert type(pair) is int and 1 <= pair <= manifest['pairs']; keys.add(key)
    label = f'{scene}-{pair}-{variant}'; assert label in labels
    command = json.loads((directory/(label+'.json')).read_text())
    expected = ['sudo', '-n', str(directory.relative_to(root)/variant/'benchmark'), scene, 'on', str(manifest['frames']), str(directory.relative_to(root)/label), 'timed']
    assert command['status'] == 0 and command['argv'] == expected, label
    raw = [json.loads(line) for line in command['stdout'].splitlines() if line.startswith('{')]
    for name, kind in [('lifecycle', 'lifecycle'), ('resources', 'shutdown_resources'), ('block', 'block')]:
        matching = [r for r in raw if r['record'] == kind]
        assert len(matching) == 1 and matching[0] == record[name], (label, kind)
    if 'mode_audit' in record:
        modes = [r for r in raw if r['record'] == 'mode_audit']
        assert len(modes) == 1 and modes[0] == record['mode_audit'] and modes[0]['mode_failures'] == 0
        if variant == 'restore_crtc': assert modes[0]['mode_sets'] == 2 and modes[0]['matched_restores'] == 1
    elif variant in ('restore_crtc', 'flip_restore'): raise AssertionError('missing mode audit')
    if 'flip_audit' in record:
        flips = [r for r in raw if r['record'] == 'flip_audit']
        assert len(flips) == 1 and flips[0] == record['flip_audit']
        flip = flips[0]
        if variant == 'flip_restore':
            assert all(type(flip[k]) is int for k in ('attempts', 'submitted', 'completed', 'target', 'error'))
            assert 0 <= flip['completed'] <= flip['submitted'] <= flip['attempts'] <= 1
            if flip['attempts']: assert flip['target'] == record['mode_audit']['saved_fb'] > 0
            if flip['submitted']: assert flip['error'] == 0
            elif flip['attempts']: assert flip['error'] > 0
            if flip['completed']:
                assert record['mode_audit']['mode_sets'] == 1 and record['mode_audit']['matched_restores'] == 0
            else:
                assert record['mode_audit']['mode_sets'] == 2 and record['mode_audit']['matched_restores'] == 1
        else: assert flip['attempts'] == flip['submitted'] == flip['completed'] == flip['target'] == 0
    elif variant == 'flip_restore': raise AssertionError('missing flip audit')
    life, resources, block = record['lifecycle'], record['resources'], record['block']
    assert life['context_released'] is True and resources['fd_closed'] is True
    for name in ('init_ns', 'shutdown_ns', 'shutdown_cpu_ns'): assert type(life[name]) is int and life[name] > 0
    for name in ('initial_fds', 'remaining_fds', 'reacquire_ns', 'ownership_probe_ns'):
        assert type(resources[name]) is int and resources[name] > 0
    assert resources['initial_fds'] == resources['remaining_fds']
    assert resources['ownership_probe_ns'] >= resources['reacquire_ns']
    assert block['scene'] == scene and block['capture'] == 'on' and block['frames'] == manifest['frames']
    assert block['messages'] == block['journal_count'] == manifest['frames'] * (2 if scene == 'single' else 1)
    frames = [r for r in raw if r['record'] == 'frame']
    assert [r['index'] for r in frames] == list(range(manifest['frames']))
    assert type(command['whole_process_ns']) is int and command['whole_process_ns'] > 0
    cold = [r for r in raw if r['record'] == 'cold']; assert len(cold) == 1
    assert type(cold[0]['elapsed_ns']) is int and cold[0]['elapsed_ns'] > 0
    runs.append({**record, 'shutdown_ns': life['shutdown_ns'], 'shutdown_cpu_ns': life['shutdown_cpu_ns'],
                 'probe_ns': resources['ownership_probe_ns'], 'reacquire_ns': resources['reacquire_ns'],
                 'combined_ns': life['shutdown_ns']+resources['ownership_probe_ns'],
                 'whole_process_ns': command['whole_process_ns'],
                 'init_ns': life['init_ns'], 'cold_ns': cold[0]['elapsed_ns'],
                 'mean_frame_ns': (block['presented_ns']+block['end_finish_ns'])/block['frames']})
metrics = ('shutdown_ns', 'shutdown_cpu_ns', 'probe_ns', 'reacquire_ns', 'combined_ns', 'whole_process_ns', 'mean_frame_ns', 'init_ns', 'cold_ns')
report = {'manifest_sha256': hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest(),
          'reducer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'scenes': [],
          'limitations': ['Probe open/set-master/drop/close is a separate adversarial device-use check, not an ordinary render frame.',
              'Combined intervals detect shifted waits; they are not additive causal attribution to kernel internals.',
              'No display FPS, energy saving or universal platform multiplier is claimed.']}
if 'flip_restore' in variants:
    flips = [r['flip_audit'] for r in runs if r['variant'] == 'flip_restore']
    report['flip_path'] = {'runs': len(flips), 'attempts': sum(f['attempts'] for f in flips),
        'submitted': sum(f['submitted'] for f in flips), 'completed': sum(f['completed'] for f in flips),
        'errors': sorted(set(f['error'] for f in flips)),
        'intended_path_exercised_every_run': all(f['completed'] == 1 for f in flips)}
    report['limitations'].append('Fallback runs qualify cleanup only, not page-flip performance. Inspect flip_path before interpreting candidate timings.')
for scene in ('single', 'retained'):
    entry = {'scene': scene, 'variants': {}, 'paired_over_baseline': {}}
    for variant in variants:
        selected = [r for r in runs if r['scene'] == scene and r['variant'] == variant]
        assert len(selected) == manifest['pairs']
        entry['variants'][variant] = {k: {'median': statistics.median(r[k] for r in selected),
            'range': [min(r[k] for r in selected), max(r[k] for r in selected)]} for k in metrics}
        if variant != 'baseline':
            entry['paired_over_baseline'][variant] = {k: [
                next(r[k] for r in selected if r['pair'] == pair) /
                next(r[k] for r in runs if r['scene'] == scene and r['variant'] == 'baseline' and r['pair'] == pair)
                for pair in range(1, manifest['pairs']+1)] for k in metrics}
    report['scenes'].append(entry)
with args.output.open('x') as out: json.dump(report, out, indent=2)
for entry in report['scenes']:
    print(entry['scene'], {v: {k: round(stats[k]['median']/1e6, 3) for k in ('shutdown_ns', 'shutdown_cpu_ns', 'combined_ns')} for v, stats in entry['variants'].items()})
