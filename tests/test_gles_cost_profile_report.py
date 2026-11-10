"""Recompute perf flat-profile attribution from saved reports.

Kernel symbols are restricted on this host, so kernel time is countable by
address but not nameable. This reducer reports the measured split, not an
interpretation of the unnamed kernel addresses.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

if not __debug__:
    raise RuntimeError('Profile report requires assertions enabled')
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('collection', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
collection = args.collection.resolve()
root = collection.parent.parent
manifest = json.loads((collection/'manifest.json').read_text())
assert manifest['completed'], 'profile collection is not complete'
for key in ('probe_sha256', 'e2e_sha256'):
    assert re.fullmatch(r'[0-9a-f]{64}', manifest[key])
assert len(manifest['cases']) == 5

line = re.compile(r'^\s+([0-9.]+)%\s+(\S+)\s+(\[[k.]\])\s+(.+?)\s+-+\s+-+\s*$')
cases = []
for case in manifest['cases']:
    label = case['label']
    text = (collection/(label+'.flat.txt')).read_text()
    entries = []
    for raw in text.splitlines():
        match = line.match(raw)
        if match:
            entries.append({'percent': float(match.group(1)), 'dso': match.group(2),
                            'space': 'kernel' if match.group(3) == '[k]' else 'user',
                            'symbol': match.group(4).strip()})
    assert entries, label
    assert all(0 < e['percent'] <= 100 for e in entries)
    assert sum(e['percent'] for e in entries) <= 100.001
    record_command = json.loads((collection/(label+'-record.command.json')).read_text())
    assert record_command['status'] == 0
    samples = int(re.search(r'\((\d+) samples\)', record_command['stderr']).group(1))
    assert samples == case['samples']
    kernel = sum(e['percent'] for e in entries if e['space'] == 'kernel')
    user = sum(e['percent'] for e in entries if e['space'] == 'user')
    users = sorted((e for e in entries if e['space'] == 'user'), key=lambda e: -e['percent'])
    kernels = sorted((e for e in entries if e['space'] == 'kernel'), key=lambda e: -e['percent'])
    cases.append({'label': label, 'argv': case['argv'], 'samples': samples,
                  'listed_percent': kernel+user, 'listed_kernel_percent': kernel,
                  'listed_user_percent': user,
                  'top_user': users[:10], 'top_kernel_addresses': kernels[:5],
                  'flat_report_sha256': hashlib.sha256((collection/(label+'.flat.txt')).read_bytes()).hexdigest()})
report = {'manifest_sha256': hashlib.sha256((collection/'manifest.json').read_bytes()).hexdigest(),
          'reducer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'revision': manifest['revision'], 'cases': cases,
          'limitations': ['Kernel symbols are restricted; kernel time is counted by address only.',
                          'Perf sampling at 4000 Hz perturbs the workload; use for attribution, not rates.',
                          'Entries below 0.4% are not listed, so listed totals are lower bounds.',
                          'No security or kernel setting was changed to obtain symbols.']}
with args.output.open('x') as handle:
    json.dump(report, handle, indent=2)
for case in cases:
    print(f"{case['label']:32s} samples={case['samples']:6d} listed kernel={case['listed_kernel_percent']:5.1f}% "
          f"user={case['listed_user_percent']:5.1f}% top_user={case['top_user'][0]['symbol'] if case['top_user'] else '-'}")
