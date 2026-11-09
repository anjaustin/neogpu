"""ROI-gated counters for an already-qualified P0 diagnostic collection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import csv
from bedrock_pmu import counter_rows
from bedrock_driver import roi_driver

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--source', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--reference', type=Path, default=Path('tests/evidence-comparison-final/reference'))
args = parser.parse_args()
assert args.output.name.startswith('evidence-') and args.output.resolve().is_relative_to(ROOT/'tests')
assert json.loads((args.source/'manifest.json').read_text())['completed']
args.output.mkdir(exist_ok=False)
manifest = {'completed': False, 'runs': [], 'sources': {}}
for name in ('tests/bedrock_driver.py', 'tests/bedrock_pmu.py', str(Path(__file__).resolve().relative_to(ROOT))):
    manifest['sources'][name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()


def run(argv, label):
    result = subprocess.run(list(map(str, argv)), text=True, capture_output=True, timeout=180)
    record = {'argv': list(map(str, argv)), 'status': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    (args.output/(label+'.json')).write_text(json.dumps(record, indent=2))
    assert result.returncode == 0, record
    return result.stdout


try:
    for name, expected in [('include/rigtorp/SPSCQueue.h', '62d9f783862666be5c7ca3c8fbcec2e45d9920167c4b9e689bda4957f66dd86e'),
                           ('LICENSE', '21d7e737765bc0c986ee78f16a3e25ee3cb5ad424feece274b64a10654c31912')]:
        actual = hashlib.sha256((args.reference/name).read_bytes()).hexdigest()
        assert actual == expected, 'pinned reference changed: '+name
        manifest['sources'][str(args.reference/name)] = actual
    flags = ['-O3', '-g', '-march=armv8-a+simd', '-mtune=cortex-a72', '-ffast-math', '-funroll-loops', '-Wall', '-Wextra', '-Werror', '-DNDEBUG']
    for variant in ('baseline', 'candidate'):
        directory = args.output/variant
        directory.mkdir()
        for name in ('hs_fabric.c', 'hs_fabric.h'):
            (directory/name).write_bytes((args.source/variant/name).read_bytes())
        driver = directory/'test_comparison_transport.c'
        driver.write_text(roi_driver(Path('tests/test_comparison_transport.c').read_text()))
        support = directory/'bedrock_support.c'
        support_text = Path('tests/bedrock_support.c').read_text()
        if 'u32 gate;' in (directory/'hs_fabric.h').read_text():
            support_text = support_text.replace('offsetof(HSParticipantSlot, mutex)', 'offsetof(HSParticipantSlot, gate)').replace('&s->mutex', '&s->gate')
        if 'u32 read_index, write_index;' in (directory/'hs_fabric.h').read_text():
            support_text = '#define HS_FABRIC_SPLIT_PUBLICATION 1\n' + support_text
            support_text = support_text.replace('offsetof(HSParticipantSlot, head)', 'offsetof(HSParticipantSlot, read_index)').replace('offsetof(HSParticipantSlot, count)', 'offsetof(HSParticipantSlot, write_index)').replace('&s->count', '&s->write_index')
        support.write_text(support_text)
        objects = []
        for name in [directory/'hs_fabric.c', 'tests/comparison_native.c', 'tests/comparison_rigtorp.cc', driver, support]:
            name = Path(name)
            manifest['sources'][str(name)] = hashlib.sha256(name.read_bytes()).hexdigest()
            cpp = name.suffix == '.cc'
            obj = directory/(name.stem+'.o')
            run(['g++' if cpp else 'gcc', *flags, '-I'+str(directory), '-Iinclude', '-Itests',
                 *(['-std=c++23', '-isystem', args.reference/'include'] if cpp else []),
                 '-c', name, '-o', obj], variant+'-compile-'+name.stem)
            objects.append(obj)
        run(['g++', *objects, '-lpthread', '-o', directory/'comparison'], variant+'-link')
        run([directory/'comparison', '--selftest'], variant+'-selftest')
    groups = {'core': '{cycles,instructions,branch-misses},task-clock,context-switches,cpu-migrations,page-faults',
              'exclusive': '{armv8_cortex_a72/event=0x6c,name=ldrex_user/u,armv8_cortex_a72/event=0x6d,name=strex_pass_user/u,armv8_cortex_a72/event=0x6e,name=strex_fail_user/u}'}
    for draws in (1, 64):
        for group, events in groups.items():
            for pair in range(1, 4):
                order = [('baseline', 'neogpu'), ('candidate', 'neogpu'), ('baseline', 'rigtorp')]
                if pair % 2 == 0: order.reverse()
                for variant, impl in order:
                    label = f'{draws}-{group}-{pair}-{variant}-{impl}'
                    ctl, ack, stat = [args.output/(label+suffix) for suffix in ('-ctl', '-ack', '-stat.csv')]
                    os.mkfifo(ctl); os.mkfifo(ack)
                    output = run(['sudo', '-n', 'env', 'BEDROCK_CTL='+str(ctl), 'BEDROCK_ACK='+str(ack),
                        'perf', 'stat', '-x;', '--no-big-num', '--delay=-1', '--control=fifo:'+str(ctl)+','+str(ack),
                        '-o', stat, '-e', events, '--', args.output/variant/'comparison', impl, 1, 1000000, draws], label)
                    summary = next(json.loads(line) for line in output.splitlines() if '"record":"summary"' in line)
                    assert summary['accepted'] == summary['applied'] == 1000000
                    summary.update(variant=variant, group=group, pair=pair, stat=stat.read_text())
                    parsed = counter_rows(list(csv.reader(summary['stat'].splitlines(), delimiter=';')))
                    expected = {'cycles', 'instructions', 'branch-misses', 'task-clock', 'context-switches', 'cpu-migrations', 'page-faults'} if group == 'core' else {'ldrex_user', 'strex_pass_user', 'strex_fail_user'}
                    assert set(parsed) == expected, ('missing/unexpected counters', parsed)
                    assert parsed and all(r['available'] and r['unmultiplexed'] for r in parsed.values()), parsed
                    manifest['runs'].append(summary)
    manifest['completed'] = True
finally:
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
