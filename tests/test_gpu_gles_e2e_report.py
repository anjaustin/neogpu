"""Recompute actual-GLES E2E results from frozen validator and raw process outputs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import statistics

if not __debug__:
    raise RuntimeError('Evidence requires assertions enabled')
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('collection', type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
collection = args.collection.resolve(); root = collection.parent.parent
manifest = json.loads((collection/'manifest.json').read_text())
assert manifest['completed'] and all(c['status'] == 0 for c in manifest['commands'])
for group in ('source_sha256', 'binary_sha256'):
    assert manifest[group]
    for name, digest in manifest[group].items():
        path = (root/name).resolve(); assert path.is_relative_to(collection)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
assert all(str((collection/name).relative_to(root)) in manifest['source_sha256'] for name in
    ('driver.py', 'common/test_gpu_gles_e2e.c', 'common/test_gpu_frame_benchmark.c', 'common/comparison_rigtorp.cc',
     'reference/include/rigtorp/SPSCQueue.h', 'reference/LICENSE', 'runtime/src/hs_gpu.c', 'runtime/src/hs_backend_gles.c'))
assert str((collection/'optimized/benchmark').relative_to(root)) in manifest['binary_sha256']
tree = ast.parse((collection/'driver.py').read_text())
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate']
assert len(functions) == 1
namespace = {'Path': Path, 'hashlib': hashlib, 'statistics': statistics}
exec(compile(ast.Module(body=functions, type_ignores=[]), 'frozen-validator', 'exec'), namespace)
pairs = manifest['pairs']; frames = manifest['frames_per_producer']
assert type(pairs) is int and 3 <= pairs <= 9 and type(frames) is int and 3 <= frames <= 512
assert len(manifest['runs']) == 24*pairs
labels = [c['label'] for c in manifest['commands']]; assert len(labels) == len(set(labels))
seen = set(); runs = []
for record in manifest['runs']:
    mode, producers, draws, capture, pair = (record[k] for k in ('mode', 'producers', 'draws', 'capture', 'pair'))
    assert mode in ('native', 'reference') and producers in (1, 2, 8) and draws in (1, 64) and capture in ('off', 'on')
    assert type(pair) is int and 1 <= pair <= pairs
    key = mode, producers, draws, capture, pair; assert key not in seen; seen.add(key)
    label = f'timed-{producers}-{draws}-{capture}-{pair}-{mode}'; assert record['label'] == label and label in labels
    command = json.loads((collection/(label+'.json')).read_text())
    expected = ['sudo', '-n', str(collection.relative_to(root)/'optimized/benchmark'), mode, str(producers), str(frames), str(draws), capture, str(collection.relative_to(root)/label)]
    assert command['status'] == 0 and command['argv'] == expected
    raw = [json.loads(line) for line in command['stdout'].splitlines() if line.startswith('{')]
    recomputed = namespace['validate'](raw, mode, producers, draws, frames, capture, collection/label)
    assert recomputed == {k: v for k, v in record.items() if k not in ('pair', 'label')}, label
    for name in ('elapsed_ns', 'process_cpu_ns', 'init_ns', 'init_cpu_ns', 'cold_ns', 'shutdown_ns', 'shutdown_cpu_ns'):
        assert type(record[name]) is int and record[name] > 0
    time_fields = {}
    for line in (collection/(label+'.time')).read_text().splitlines():
        if ': ' in line:
            name, value = line.strip().rsplit(': ', 1); time_fields[name] = value
    assert time_fields.get('Exit status') == '0'
    runs.append({**record, 'whole_process_ns': command['whole_process_ns'],
        'peak_rss_kib': int(time_fields['Maximum resident set size (kbytes)']),
        'voluntary_context_switches': int(time_fields['Voluntary context switches']),
        'involuntary_context_switches': int(time_fields['Involuntary context switches']),
        'minor_faults': int(time_fields['Minor (reclaiming a frame) page faults']),
        'major_faults': int(time_fields['Major (requiring I/O) page faults']),
        'process_tree_cpu_seconds': float(time_fields['User time (seconds)'])+float(time_fields['System time (seconds)'])})
cells = []
metrics = ('completed_contributions_per_second', 'elapsed_ns', 'process_cpu_ns', 'init_ns', 'cold_ns', 'shutdown_ns',
    'shutdown_cpu_ns', 'retries', 'rendered_frames', 'whole_process_ns', 'peak_rss_kib', 'voluntary_context_switches',
    'involuntary_context_switches', 'minor_faults', 'major_faults', 'process_tree_cpu_seconds')
for producers in (1, 2, 8):
    for draws in (1, 64):
        for capture in ('off', 'on'):
            selected = [r for r in runs if (r['producers'], r['draws'], r['capture']) == (producers, draws, capture)]
            assert len({r['rgba_sha256'] for r in selected}) == 1
            cell = {'producers': producers, 'draws': draws, 'capture': capture, 'variants': {}}
            for mode in ('native', 'reference'):
                rows = [r for r in selected if r['mode'] == mode]; assert len(rows) == pairs
                def stats(values): return {'median': statistics.median(values), 'range': [min(values), max(values)]}
                cell['variants'][mode] = {k: stats([r[k] for r in rows]) for k in metrics}
                for group in ('contribution_latency_ns', 'admission_latency_ns', 'frame_completed_ns', 'frame_submit_ns'):
                    cell['variants'][mode][group] = {percentile: stats([r[group][percentile] for r in rows]) for percentile in ('p50', 'p95', 'p99', 'max')}
            cell['paired_native_over_reference_rate'] = [
                next(r['completed_contributions_per_second'] for r in selected if r['mode'] == 'native' and r['pair'] == pair)/
                next(r['completed_contributions_per_second'] for r in selected if r['mode'] == 'reference' and r['pair'] == pair) for pair in range(1, pairs+1)]
            cells.append(cell)
report = {'manifest_sha256': hashlib.sha256((collection/'manifest.json').read_bytes()).hexdigest(),
    'reducer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'revision': manifest['revision'], 'cells': cells,
    'runs': len(runs), 'applied': sum(r['applied'] for r in runs), 'rendered_frames': sum(r['rendered_frames'] for r in runs),
    'limitations': ['Every frame glFinish serializes completion; this is not unrestricted asynchronous throughput or panel FPS.',
        'Reference shares actual NeoGPU GLES executor; it substitutes static SPSC admission/composition only, not a complete third-party graphics engine.',
        'Per-contribution clocks and frame/FIFO validators are included for both paths; observer cache traffic is not production-free.',
        'Stable geometry validates draw order/schema and contribution completion, not sequence-dependent visible animation.',
        'Admission return can follow consumption/completion. Latency is first attempt to glFinish after the consuming frame.',
        'Whole-process metrics include setup, verification, output, cleanup and sudo/time overhead; peak RSS is not allocated GPU memory.']}
with args.output.open('x') as f: json.dump(report, f, indent=2)
for cell in cells:
    print(cell['producers'], cell['draws'], cell['capture'], {m: round(s['completed_contributions_per_second']['median'], 2) for m, s in cell['variants'].items()})
