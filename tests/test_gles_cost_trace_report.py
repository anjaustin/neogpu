"""Summarize an strace -T trace: totals by syscall, longest individual calls.

Diagnostic only. Traced durations perturb scheduling; use for locating
candidates, not as production timing.
"""
import argparse
import collections
import json
import re
from pathlib import Path

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('trace', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()

pattern = re.compile(r'^(\d+)\s+(\S+)\((.*)\)\s+=\s+(.*?)\s+<([0-9.]+)>$')
rows = []
for line in args.trace.read_text(errors='replace').splitlines():
    match = pattern.match(line)
    if match:
        rows.append({'tid': int(match.group(1)), 'syscall': match.group(2),
                     'seconds': float(match.group(5)), 'args': match.group(3),
                     'result': match.group(4)})

total = collections.Counter()
count = collections.Counter()
for row in rows:
    total[row['syscall']] += row['seconds']
    count[row['syscall']] += 1

def describe(row):
    return {'seconds': row['seconds'], 'tid': row['tid'], 'syscall': row['syscall'],
            'args': row['args'][:160], 'result': row['result'][:80]}

ordered = sorted(rows, key=lambda r: -r['seconds'])
report = {
    'trace': str(args.trace),
    'parsed_calls': len(rows),
    'unparsed_lines': len(args.trace.read_text(errors='replace').splitlines()) - len(rows),
    'traced_seconds_total': sum(r['seconds'] for r in rows),
    'by_syscall': [{'syscall': name, 'calls': count[name], 'seconds': value}
                   for name, value in total.most_common()],
    'longest': [describe(r) for r in ordered[:40]],
    'top10_seconds': sum(r['seconds'] for r in ordered[:10]),
    'threads': len({r['tid'] for r in rows}),
    'note': 'Traced durations include tracer overhead and interleaving; summed '
            'thread time may exceed wall time. Not production timing.',
}
args.output.write_text(json.dumps(report, indent=2))
print(f"parsed {len(rows)} calls, traced {report['traced_seconds_total']*1000:.1f} ms total")
for row in report['by_syscall'][:10]:
    print(f"  {row['syscall']:14s} {row['seconds']*1000:9.3f} ms {row['calls']:6d} calls")
print('longest:')
for row in report['longest'][:10]:
    print(f"  {row['seconds']*1000:9.3f} ms tid {row['tid']} {row['syscall']}({row['args'][:90]})")
