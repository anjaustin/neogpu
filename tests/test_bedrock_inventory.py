"""Read-only hardware/OS counter capability inventory; failures remain evidence."""
import argparse
import ctypes
import glob
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
if not args.output.name.startswith("evidence-") or not args.output.resolve().is_relative_to((ROOT / "tests").resolve()):
    parser.error("output must be a new tests/evidence-* directory")
args.output.mkdir(exist_ok=False)


def command(argv):
    result = subprocess.run(argv, text=True, capture_output=True, timeout=30)
    return {"argv": argv, "exit_status": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


libc = ctypes.CDLL(None)
libc.getauxval.argtypes = [ctypes.c_ulong]
libc.getauxval.restype = ctypes.c_ulong
records = {"cpuinfo": Path("/proc/cpuinfo").read_text(), "affinity": sorted(os.sched_getaffinity(0)),
           "hwcap": int(libc.getauxval(16)), "files": {}, "commands": [],
           "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
records['monotonic_clock'] = vars(time.get_clock_info('monotonic'))
patterns = ("/sys/devices/system/cpu/cpu*/cache/index*/*", "/sys/bus/event_source/devices/armv8_cortex_a72/*",
            "/sys/bus/event_source/devices/armv8_cortex_a72/events/*", "/sys/bus/event_source/devices/armv8_cortex_a72/format/*",
            "/sys/devices/system/cpu/cpu*/cpufreq/scaling_*", "/sys/class/thermal/thermal_zone*/temp")
for pattern in patterns:
    for name in sorted(glob.glob(pattern)):
        path = Path(name)
        if not path.is_file():
            continue
        try:
            records["files"][name] = path.read_text()
        except OSError as error:
            records["files"][name] = {"error": str(error)}
for argv in (["uname", "-a"], ["sudo", "-n", "perf", "list", "--details"],
             ["sudo", "-n", "cat", "/sys/kernel/tracing/events/syscalls/sys_enter_sched_yield/id"],
             ["sudo", "-n", "cat", "/sys/kernel/tracing/events/sched/sched_switch/id"],
             ["sudo", "-n", "cat", "/sys/kernel/tracing/events/sched/sched_migrate_task/id"],
             ["sudo", "-n", "ls", "/sys/bus/event_source/devices/armv8_cortex_a72/caps"],
             ["sudo", "-n", "perf", "stat", "-x,", "-e", "cycles,instructions", "--", "true"]):
    records["commands"].append(command(argv))
header = Path("/usr/include/aarch64-linux-gnu/asm/hwcap.h")
if header.exists():
    records["hwcap_header"] = header.read_text()
    records["hwcap_header_sha256"] = hashlib.sha256(header.read_bytes()).hexdigest()
records['capacity_probes'] = []
events = ['instructions','branch-misses','armv8_cortex_a72/l1d_cache/','armv8_cortex_a72/l1d_cache_refill/',
          'armv8_cortex_a72/l2d_cache/','armv8_cortex_a72/l2d_cache_refill/','armv8_cortex_a72/l1d_tlb_refill/','armv8_cortex_a72/l1i_cache_refill/']
for count in range(1,len(events)+1):
    group = '{cycles,' + ','.join(events[:count]) + '}'
    records['capacity_probes'].append(command(['sudo','-n','perf','stat','-x;','--no-big-num','-e',group,'--','true']))
records['energy_sensors'] = {}
for pattern in ('/sys/class/hwmon/hwmon*/power*_input','/sys/class/hwmon/hwmon*/energy*_input',
                '/sys/class/powercap/*/energy_uj','/sys/bus/event_source/devices/arm_spe*','/sys/bus/event_source/devices/cs_etm*'):
    for name in sorted(glob.glob(pattern)):
        path = Path(name)
        try: records['energy_sensors'][name] = path.read_text() if path.is_file() else 'present directory'
        except OSError as error: records['energy_sensors'][name] = str(error)
records['libc'] = {'commands':[command(['ldd','--version']),command(['readelf','-n','/lib/aarch64-linux-gnu/libc.so.6'])],
                  'sha256':hashlib.sha256(Path('/lib/aarch64-linux-gnu/libc.so.6').read_bytes()).hexdigest()}
(args.output / "inventory.json").write_text(json.dumps(records, indent=2))
for record in records["commands"]:
    print(json.dumps(record))
