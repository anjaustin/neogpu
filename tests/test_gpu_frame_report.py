"""Recompute default-frame integration results from raw command summaries."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

if not __debug__:
    raise RuntimeError('Frame evidence requires Python assertions enabled')
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('collection', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
collection = args.collection.resolve()
root = collection.parent.parent
manifest = json.loads((args.collection/'manifest.json').read_text())
assert manifest['completed'] and all(command['status'] == 0 for command in manifest['commands'])
variants = manifest.get('variants', ['baseline', 'candidate'])
assert variants in (['baseline', 'candidate'], ['baseline', 'control', 'candidate'])
sources = manifest['source_sha256']
snapshots = {(Path(name) if Path(name).is_absolute() else root/name).resolve(): digest
             for name, digest in sources.items()}
# Verify immutable collection snapshots, not mutable aliases in the live tree.
required = [collection/'driver.py', *(collection/'common'/name for name in
            ('test_gpu_frame_benchmark.c', 'comparison_rigtorp.cc', 'comparison_transport.h'))]
required += [collection/variant/name for variant in variants for name in ('include/hs_gpu.h', 'src/hs_gpu.c')]
assert all(path in snapshots for path in required), 'Missing reproduction snapshots'
verified = 0
for path, digest in snapshots.items():
    if path.is_relative_to(collection):
        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
        verified += 1
assert len(manifest['runs']) == 42 * len(variants)
frames = manifest['frames_per_producer']
assert type(frames) is int and frames > 0
report = {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'manifest_sha256': hashlib.sha256((args.collection/'manifest.json').read_bytes()).hexdigest(),
          'legacy': [], 'active': [], 'layout': {}, 'runs': len(manifest['runs']),
          'verified_snapshot_files': verified}
runs = []
for record in manifest['runs']:
    assert record['variant'] in variants and record['pair'] in (1, 2, 3)
    for name in ('producers', 'draws', 'accepted', 'applied', 'elapsed_ns', 'retries', 'backend_calls', 'init_ns', 'gpu_bytes', 'fabric_bytes'):
        assert type(record[name]) is int and record[name] >= 0, (name, record[name])
    assert record['mode'] in ('legacy', 'native', 'reference') and record['producers'] in (1, 2, 8) and record['draws'] in (1, 64)
    assert record['accepted'] == record['applied'] == frames * record['producers']
    assert record['elapsed_ns'] > 0 and record['backend_calls'] >= frames
    if record['mode'] == 'legacy':
        assert record['producers'] == 1 and record['backend_calls'] == frames
        label = f"legacy-{record['draws']}-{record['pair']}-{record['variant']}"
    else:
        label = f"active-{record['producers']}-{record['draws']}-{record['pair']}-{record['variant']}-{record['mode']}"
    command = json.loads((args.collection/(label+'.json')).read_text())
    assert command['status'] == 0
    argv = command['argv']
    assert len(argv) == 5 and argv[1:] == [record['mode'], str(record['producers']), str(frames), str(record['draws'])], label
    binary = Path(argv[0])
    binary = (binary if binary.is_absolute() else root/binary).resolve()
    assert binary == collection/record['variant']/'benchmark', label
    assert any(item.get('label') == label and item['status'] == 0 for item in manifest['commands']), label
    raw = json.loads(command['stdout'])
    assert raw == {key: value for key, value in record.items() if key not in ('variant', 'pair')}, label
    runs.append({**record, 'rate': record['applied']*1e9/record['elapsed_ns']})


def summarize(selected):
    result = {}
    for variant in variants:
        values = [r for r in selected if r['variant'] == variant]
        assert len(values) == 3 and {r['pair'] for r in values} == {1, 2, 3}
        result[variant] = {'median_rate': statistics.median(r['rate'] for r in values),
                           'range_rate': [min(r['rate'] for r in values), max(r['rate'] for r in values)],
                           'median_retries_per_applied': statistics.median(r['retries']/r['applied'] for r in values),
                           'median_backend_calls_per_applied': statistics.median(r['backend_calls']/r['applied'] for r in values),
                           'median_init_ns': statistics.median(r['init_ns'] for r in values)}
    ratios = {}
    for control in [v for v in variants if v != 'candidate']:
        ratios[control] = [next(r['rate'] for r in selected if r['variant'] == 'candidate' and r['pair'] == pair) /
                           next(r['rate'] for r in selected if r['variant'] == control and r['pair'] == pair) for pair in (1, 2, 3)]
    return {'variants': result, 'paired_candidate_over_control': ratios}


for draws in (1, 64):
    selected = [r for r in runs if r['mode'] == 'legacy' and r['draws'] == draws]
    report['legacy'].append({'draws': draws, 'unit': 'validated legacy scenes/s', **summarize(selected)})
for mode in ('native', 'reference'):
    for producers in (1, 2, 8):
        for draws in (1, 64):
            selected = [r for r in runs if r['mode'] == mode and r['producers'] == producers and r['draws'] == draws]
            report['active'].append({'mode': mode, 'producers': producers, 'draws': draws,
                                     'unit': 'validated applied contributions/s', **summarize(selected)})
for variant in variants:
    selected = [r for r in runs if r['variant'] == variant]
    sizes = {r['gpu_bytes'] for r in selected}; assert len(sizes) == 1
    report['layout'][variant] = {'gpu_bytes': sizes.pop(),
        'fabric_bytes': sorted({r['fabric_bytes'] for r in selected}),
        'fabric_address_mod64': sorted({r['fabric_address_mod64'] for r in selected})}
report['limitations'] = ['Legacy scenes and participant contributions are different work units.',
    'Includes a validating mock backend, not GPU execution or display FPS.',
    'Initial control/manual plumbing and the static SPSC reference have explicitly different integration/lifecycle contracts.',
    'Three pairs per cell; timing differences are not additive causal partitions.',
    'Initialization intervals are per-process HSGpu/mock setup, including manual native fabric init on baseline; not full graphics startup.']
with args.output.open('x') as out: json.dump(report, out, indent=2)
for cell in report['legacy']:
    print('legacy', cell['draws'], {v: round(r['median_rate']) for v, r in cell['variants'].items()})
for cell in report['active']:
    print(cell['mode'], cell['producers'], cell['draws'], {v: round(r['median_rate']) for v, r in cell['variants'].items()})
