"""Decode both keyed and perf-pretty scheduler records; fail on unknown formats."""
import re

if not __debug__:
    raise RuntimeError('Bedrock evidence requires Python assertions enabled')


def decode(script, counters):
    observed = {str(r['tid']): {'yields':0,'switch_out':0,'switch_in_observed':0,'migrations_observed':0} for r in counters}
    if not counters or len(observed) != len(counters) or any(r['tid'] <= 0 or r['yields'] < 0 for r in counters):
        raise ValueError('missing, duplicate or invalid thread counters')
    lost = [line for line in script.splitlines() if 'LOST' in line or 'lost' in line]
    switches = 0
    for line in script.splitlines():
        # perf can emit TID -1 for an unknown/exiting event context; scheduler
        # payload PIDs still identify the actual tasks and must be decoded.
        match = re.match(r'\s*(.+?)\s+(-?\d+)\s+\[(\d+)\]\s+([0-9.]+):\s+(\S+):\s*(.*)',line)
        if not match:
            if any(event in line for event in ('sched:sched_switch:', 'sched:sched_migrate_task:', 'syscalls:sys_enter_sched_yield:')):
                raise ValueError(f'unrecognized perf event header: {line}')
            continue
        _, tid, _, _, event, detail = match.groups()
        if event == 'syscalls:sys_enter_sched_yield' and tid in observed:
            observed[tid]['yields'] += 1
        if event == 'sched:sched_switch':
            switches += 1
            if 'prev_pid=' in detail:
                previous = re.search(r'(?:^|\s)prev_pid=(\d+)(?:\s|$)',detail)
                following = re.search(r'(?:^|\s)next_pid=(\d+)(?:\s|$)',detail)
            else:
                parts = detail.split(' ==> ')
                assert len(parts) == 2, line
                previous = re.search(r':(\d+)\s+\[',parts[0])
                following = re.search(r':(\d+)\s+\[',parts[1])
            assert previous and following, line
            if previous[1] in observed: observed[previous[1]]['switch_out'] += 1
            if following[1] in observed: observed[following[1]]['switch_in_observed'] += 1
        if event == 'sched:sched_migrate_task':
            pid = re.search(r'pid=(\d+)',detail)
            assert pid, line
            if pid[1] in observed: observed[pid[1]]['migrations_observed'] += 1
    return {'trace':observed,'lost_events':lost,'sched_switch_records':switches,
            'yield_count_matches':all(observed[str(r['tid'])]['yields'] == r['yields'] for r in counters)}


def selftest():
    counters = [{'tid':100,'yields':1},{'tid':200,'yields':0}]
    text = '''  p 100 [001] 1.000000000: syscalls:sys_enter_sched_yield: ()
  p 100 [001] 1.000000001: sched:sched_switch: br-prod:100 [120] R ==> kworker/0:0:200 [120]
  p 200 [001] 1.000000002: sched:sched_switch: prev_comm=k prev_pid=200 prev_prio=120 prev_state=S ==> next_comm=b next_pid=100 next_prio=120
  p 100 [001] 1.000000003: sched:sched_migrate_task: comm=b pid=100 prio=120 orig_cpu=1 dest_cpu=0
'''
    result = decode(text,counters)
    assert result['yield_count_matches'] and result['sched_switch_records'] == 2
    assert result['trace']['100'] == {'yields':1,'switch_out':1,'switch_in_observed':1,'migrations_observed':1}
    assert result['trace']['200']['switch_out'] == result['trace']['200']['switch_in_observed'] == 1
    unknown = decode(':-1 -1 [001] 1.0: sched:sched_switch: p:100 [120] X ==> swapper/1:0 [120]', counters)
    assert unknown['trace']['100']['switch_out'] == 1 and unknown['sched_switch_records'] == 1
    for line in ('p 100 [001] 1.0: sched:sched_switch: unsupported format',
                 'bad header sched:sched_switch: prev_pid=100 next_pid=200',
                 'bad header syscalls:sys_enter_sched_yield: ()'):
        try:
            decode(line,counters)
        except (AssertionError, ValueError):
            continue
        raise AssertionError('unknown scheduler formats must fail closed')
    for invalid in ([], [counters[0], counters[0]], [{'tid': 0, 'yields': 0}]):
        try:
            decode('', invalid)
        except ValueError:
            continue
        raise AssertionError('invalid counter metadata was accepted')


if __name__ == '__main__':
    selftest()
    print('Scheduler decoder: keyed, pretty, colon-containing names and rejection controls passed')
