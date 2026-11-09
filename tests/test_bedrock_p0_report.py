"""Describe paired P0 measurements; never turn one median into acceptance."""
import argparse
import json
from pathlib import Path
import statistics
import math

if not __debug__:
    raise RuntimeError('Bedrock evidence requires Python assertions enabled')

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('collection', type=Path)
args = parser.parse_args()
manifest = json.loads((args.collection/'manifest.json').read_text())
assert manifest['completed']
assert len(manifest['runs']) == 54, 'incomplete matrix'
for run in manifest['runs']:
    assert all(type(run[field]) is int for field in ('participants', 'draws', 'accepted', 'applied', 'elapsed_ns', 'retries', 'pair')), 'noninteger timing metadata'
    assert run['participants'] in (1, 2, 8) and run['draws'] in (1, 64) and run['retries'] >= 0
    assert run['accepted'] == run['applied'] > 0 and run['elapsed_ns'] > 0
    assert run['applied'] % run['participants'] == 0
    measured_fps = run['applied'] * 1e9 / run['elapsed_ns']
    assert math.isclose(run['fps'], measured_fps, rel_tol=1e-12), 'cached rate contradicts elapsed time'
    run['fps'] = measured_fps
assert len({run['applied'] // run['participants'] for run in manifest['runs']}) == 1, 'unequal frame work'
report = []
for producers in (1, 2, 8):
    for draws in (1, 64):
        cell = {'producers': producers, 'draws': draws, 'variants': {}}
        selected = [r for r in manifest['runs'] if r['participants'] == producers and r['draws'] == draws]
        for variant, impl in [('baseline', 'neogpu'), ('candidate', 'neogpu'), ('baseline', 'rigtorp')]:
            values = [r for r in selected if r['variant'] == variant and r['implementation'] == impl]
            assert len(values) == 3 and {r['pair'] for r in values} == {1, 2, 3}, 'missing/duplicate pair'
            cell['variants'][variant+'-'+impl] = {
                'runs': len(values), 'median_fps': statistics.median(r['fps'] for r in values),
                'range_fps': [min(r['fps'] for r in values), max(r['fps'] for r in values)],
                'median_retries_per_frame': statistics.median(r['retries']/r['applied'] for r in values)}
            samples = []
            for value in values:
                label = f"run-{producers}-{draws}-{value['pair']}-{variant}-{impl}"
                raw = json.loads((args.collection/(label+'.json')).read_text())
                assert raw['status'] == 0
                raw_summary = next(r for line in raw['stdout'].splitlines() if (r := json.loads(line))['record'] == 'summary')
                for field in ('participants', 'draws', 'accepted', 'applied', 'elapsed_ns', 'retries', 'implementation'):
                    assert raw_summary[field] == value[field], (label, field)
                samples += [r for line in raw['stdout'].splitlines() if (r := json.loads(line))['record'] == 'sample']
            latency = {'pooled_sample_count': len(samples)}
            for name, end in [('admission', 'admission_return_ns'), ('application', 'applied_ns')]:
                ordered = sorted(r[end] - r['attempt_ns'] for r in samples)
                assert all(value >= 0 for value in ordered)
                latency[name+'_sample_p50_ns'] = ordered[math.ceil(len(ordered)*0.50)-1]
                latency[name+'_sample_p99_ns'] = ordered[math.ceil(len(ordered)*0.99)-1]
            cell['variants'][variant+'-'+impl]['sampled_latency'] = latency
        ratios = []
        for pair in sorted({r['pair'] for r in selected}):
            base = next(r for r in selected if r['pair'] == pair and r['variant'] == 'baseline' and r['implementation'] == 'neogpu')
            candidate = next(r for r in selected if r['pair'] == pair and r['variant'] == 'candidate')
            ratios.append(candidate['fps']/base['fps'])
        cell['paired_candidate_over_baseline_fps'] = ratios
        report.append(cell)
print(json.dumps(report, indent=2))
