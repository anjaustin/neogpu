"""Isolated kernel-entry/cache controls. No mitigations or OS settings are changed."""
import argparse
import csv
import glob
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

if not __debug__:
    raise RuntimeError('Bedrock evidence requires Python assertions enabled')

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
if not args.output.name.startswith('evidence-') or not args.output.resolve().is_relative_to((ROOT/'tests').resolve()):
    parser.error('output must be a fresh tests/evidence-* directory')
args.output.mkdir(exist_ok=False)
manifest = {'completed':False, 'security':{}, 'commands':[], 'results':[]}


def environment():
    result = {'wall_time_ns':time.time_ns(), 'affinity':sorted(os.sched_getaffinity(0)), 'sysfs':{}}
    for pattern in ('/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq',
                    '/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor','/sys/class/thermal/thermal_zone*/temp'):
        for name in sorted(glob.glob(pattern)):
            try: result['sysfs'][name] = Path(name).read_text().strip()
            except OSError as error: result['sysfs'][name] = str(error)
    return result


def checked(argv, label, required=True):
    result = subprocess.run([str(a) for a in argv], text=True, capture_output=True, timeout=60)
    record = {'argv':[str(a) for a in argv], 'exit_status':result.returncode, 'stdout':result.stdout, 'stderr':result.stderr}
    (args.output/(label+'.json')).write_text(json.dumps(record,indent=2))
    manifest['commands'].append({k:v for k,v in record.items() if k != 'stdout'})
    if required: assert result.returncode == 0, record
    return record


try:
    manifest['environment_before'] = environment()
    for name in sorted(glob.glob('/sys/devices/system/cpu/vulnerabilities/*')):
        try: manifest['security'][name] = Path(name).read_text().strip()
        except OSError as error: manifest['security'][name] = str(error)
    manifest['cmdline'] = Path('/proc/cmdline').read_text()
    config = Path('/boot/config-' + os.uname().release)
    record = checked(['sudo','-n','cat',config], 'kernel-config', required=False)
    if record['exit_status'] == 0:
        manifest['kernel_config_sha256'] = hashlib.sha256(record['stdout'].encode()).hexdigest()
        manifest['relevant_kernel_config'] = [line for line in record['stdout'].splitlines() if any(word in line for word in
            ('UNMAP_KERNEL_AT_EL0','SW_TTBR0_PAN','SPECTRE','HARDEN_BRANCH','HARDEN_EL2','MITIGATE','ARM64_BTI_KERNEL','PREEMPT'))]
    manifest['source_sha256'] = {name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
        ('tests/test_bedrock_syscalls.c','tests/bedrock_support.c','tests/bedrock_support.h',str(Path(__file__).resolve().relative_to(ROOT)))}
    binary = args.output/'syscalls'
    checked(['gcc','-O3','-g','-march=armv8-a+simd','-mtune=cortex-a72','-Wall','-Wextra','-Werror','-Iinclude','-Itests',
             'tests/test_bedrock_syscalls.c','tests/bedrock_support.c','-lpthread','-o',binary], 'build')
    groups = {
      'core':'{cycles,instructions,branch-misses},task-clock,context-switches,cpu-migrations,syscalls:sys_enter_sched_yield,syscalls:sys_enter_getpid',
      'refills':'{armv8_cortex_a72/l1d_tlb_refill/u,armv8_cortex_a72/l1i_cache_refill/u,armv8_cortex_a72/l1i_tlb_refill/u}',
      'user-core':'{cycles:u,instructions:u,branch-misses:u}'}
    groups['core-untraced'] = '{cycles,instructions,branch-misses},task-clock,context-switches,cpu-migrations'
    for group, events in groups.items():
        for pair in range(1,4):
            order = ('loop','cpu-yield','getpid','sched-yield') if pair % 2 else ('sched-yield','getpid','cpu-yield','loop')
            for mode in order:
                label = f'{group}-{pair}-{mode}'
                ctl, ack = args.output/(label+'-ctl'), args.output/(label+'-ack')
                os.mkfifo(ctl);os.mkfifo(ack)
                stat = args.output/(label+'-stat.csv')
                before = environment()
                result = checked(['sudo','-n','env','BEDROCK_CTL='+str(ctl),'BEDROCK_ACK='+str(ack),
                    'perf','stat','-x;','--no-big-num','--delay=-1','--control=fifo:'+str(ctl)+','+str(ack),
                    '-o',stat,'-e',events,'--',binary,mode],label)
                summary = json.loads(result['stdout'])
                summary.update(group=group,pair=pair,pmu=list(csv.reader(stat.read_text().splitlines(),delimiter=';')),
                               environment_before=before,environment_after=environment())
                manifest['results'].append(summary)
    manifest['untraced_wall_controls'] = []
    for pair in range(1,4):
        for mode in (('loop','cpu-yield','getpid','sched-yield') if pair % 2 else ('sched-yield','getpid','cpu-yield','loop')):
            result = checked(['sudo','-n',binary,mode], f'wall-{pair}-{mode}')
            manifest['untraced_wall_controls'].append(dict(json.loads(result['stdout']),pair=pair))
    manifest['completed'] = True
finally:
    manifest['environment_after'] = environment()
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
