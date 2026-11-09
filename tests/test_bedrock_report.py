"""Recompute bedrock summaries without treating counters as causal cost partitions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
from bedrock_trace import decode, selftest as trace_selftest
from bedrock_pmu import counter_rows

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--primary', required=True, type=Path)
parser.add_argument('--deep', required=True, type=Path)
parser.add_argument('--backoff', required=True, type=Path)
parser.add_argument('--syscalls', required=True, type=Path)
parser.add_argument('--inventory', required=True, type=Path)
parser.add_argument('--extra', type=Path, help='untraced core/count-PMU validation collection')
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
if not args.output.name.startswith('evidence-') or not args.output.resolve().is_relative_to((ROOT/'tests').resolve()):
    parser.error('output must be a fresh tests/evidence-* file')
report = {'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'artifacts_sha256':{}, 'pmu':[], 'counts':[], 'ablations':[], 'syscalls':[]}
report['helper_sha256'] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                           for name in ('tests/bedrock_trace.py', 'tests/bedrock_pmu.py')}


def load(path):
    data = path.read_bytes()
    report['artifacts_sha256'][str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def trace_text(path):
    data = path.read_bytes()
    report['artifacts_sha256'][str(path)] = hashlib.sha256(data).hexdigest()
    return data.decode()


primary = load(args.primary/'manifest.json')
assert all(primary['phases'].get(p) for p in ('build','qualify','counts','pmu','ablations','trace'))
counts = load(args.primary/'counts.json')
for record in counts:
    consumer, producer = record['counts']
    assert producer['attempts'] == producer['admitted'] + producer['busy'] + producer['full']
    assert producer['admitted'] == consumer['applied'] == record['accepted'] == record['applied']
    assert producer['yields'] == producer['busy'] + producer['full'] == record['retries']
    assert consumer['visits'] == consumer['service_busy'] + consumer['empty'] + consumer['applied']
    assert consumer['inactive_visits'] == consumer['active_visits'] * 7
    report['counts'].append({'implementation':record['implementation'], 'consumer':consumer,'producer':producer})
    raw = load(args.primary/(record['label']+'.json'))
    layouts = [json.loads(line) for line in raw['stdout'].splitlines() if '"record":"layout"' in line]
    if layouts: report.setdefault('layouts', []).append(layouts)

for directory in (args.primary,args.deep,*([args.extra] if args.extra else [])):
    data = load(directory/'pmu.json')
    selected = {}
    for record in data:
        assert record['accepted'] == record['applied'] == 1000000
        group = re.match(r'pmu-(.*?)-[0-9]+-',record['label'])[1]
        for event, values in counter_rows(record['pmu']).items():
            key = (record['variant'],record['implementation'],group,event)
            selected.setdefault(key,[]).append((record,values))
    for (variant,impl,group,event), values in selected.items():
        valid = [(r,v) for r,v in values if v['available'] and v['unmultiplexed']]
        item = {'collection':str(directory),'variant':variant,'implementation':impl,'group':group,'event':event,
                'runs':len(values),'accepted_unmultiplexed_runs':len(valid)}
        if valid:
            normalized = [v['value']/r['applied'] for r,v in valid]
            item.update(median_per_frame=statistics.median(normalized),range_per_frame=[min(normalized),max(normalized)],
                        units=valid[0][1]['unit'],percent_running_range=[min(v['percent_running'] for _,v in valid),max(v['percent_running'] for _,v in valid)])
        report['pmu'].append(item)

for directory in (args.primary,args.backoff):
    results = load(directory/'ablations.json')
    for draws in (1,64):
        for variant in sorted({r['variant'] for r in results}):
            for impl in ('neogpu','rigtorp'):
                values = [r for r in results if r.get('draws')==draws and r['variant']==variant and r['implementation']==impl and not r.get('failed')]
                if not values: continue
                assert all(v['accepted']==v['applied']==200000 for v in values)
                report['ablations'].append({'collection':str(directory),'draws':draws,'variant':variant,'implementation':impl,'runs':len(values),
                    'median_fps':statistics.median(r['fps'] for r in values), 'range_fps':[min(r['fps'] for r in values),max(r['fps'] for r in values)],
                    'median_retries':statistics.median(r['retries'] for r in values)})

trace_selftest()
report['trace_task_attached'] = load(args.primary/'trace.json')
report['trace_cpu_wide'] = load(args.deep/'trace.json')
for directory,key in ((args.primary,'trace_task_attached'),(args.deep,'trace_cpu_wide')):
    for record in report[key]:
        text = trace_text(directory/(record['label']+'-script.txt'))
        record.update(decode(text,record['counts']))
for record in report['trace_cpu_wide']:
    assert record['yield_count_matches'] and not record['lost_events']

syscalls = load(args.syscalls/'manifest.json')
assert syscalls['completed']
report['security'] = syscalls['security']
report['relevant_kernel_config'] = syscalls.get('relevant_kernel_config',[])
report['untraced_wall_controls'] = syscalls.get('untraced_wall_controls',[])
for group in sorted({r['group'] for r in syscalls['results']}):
    for mode in ('loop','cpu-yield','getpid','sched-yield'):
        results = [r for r in syscalls['results'] if r['group']==group and r['mode']==mode]
        events = {}
        for record in results:
            for event, values in counter_rows(record['pmu']).items():
                if values['available'] and values['unmultiplexed']:
                    events.setdefault(event, []).append(values['value']/record['iterations'])
        report['syscalls'].append({'group':group,'mode':mode,'runs':len(results),
            'median_elapsed_ns_per_iteration':statistics.median(r['elapsed_ns']/r['iterations'] for r in results),
            'counters_per_iteration':{event:statistics.median(values) for event,values in events.items()}})
inventory = load(args.inventory/'inventory.json')
if args.extra:
    report['count_pmu'] = load(args.extra/'count-pmu.json')
    assert all(r['yield_pmu_matches'] for r in report['count_pmu'])
    report['trace_validation'] = load(args.extra/'trace.json')
    for record in report['trace_validation']:
        text = trace_text(args.extra/(record['label']+'-script.txt'))
        record.update(decode(text,record['counts']))
        assert record['yield_count_matches'] and not record['lost_events']
report['hwcap'] = inventory['hwcap']
report['hwcap_atomics'] = bool(inventory['hwcap'] & (1<<8))
report['cache_cpu0'] = {name:value for name,value in inventory['files'].items() if '/cpu0/cache/' in name}
with args.output.open('x') as output:
    output.write(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k in ('hwcap','hwcap_atomics','pmu','ablations','syscalls')},indent=2))
