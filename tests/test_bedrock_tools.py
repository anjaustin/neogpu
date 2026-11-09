"""Negative controls for bedrock evidence parsers and the perf FIFO client."""
import os
from pathlib import Path
import select
import subprocess
import threading
import time
import json
import sys
from bedrock_pmu import selftest as pmu_selftest
from bedrock_trace import selftest as trace_selftest
from bedrock_driver import roi_driver

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
pmu_selftest()
trace_selftest()
driver_source = Path('tests/test_comparison_transport.c').read_text()
generated_driver = roi_driver(driver_source)
assert generated_driver.count('bedrock_control("enable")') == generated_driver.count('bedrock_control("disable")') == 1
try:
    roi_driver('invalid driver')
except ValueError:
    pass
else:
    raise AssertionError('missing ROI anchors were accepted')
for script in ('test_bedrock_tools.py', 'test_bedrock_p0.py', 'test_bedrock_p0_report.py',
               'test_bedrock_adversarial.py', 'test_bedrock_syscall_evidence.py'):
    result = subprocess.run([sys.executable, '-O', 'tests/'+script, '--help'], capture_output=True, timeout=10)
    assert result.returncode != 0 and b'assertions enabled' in result.stderr
output = Path(f'tests/evidence-bedrock-tools-{time.time_ns()}-{os.getpid()}')
output.mkdir()
source = output/'client.c'
source.write_text('''#include "bedrock_support.h"
#include <sys/resource.h>
int main(void) {
    struct rlimit limit = {0, 0};
    if (setrlimit(RLIMIT_CORE, &limit)) return 1;
    bedrock_control("enable"); bedrock_control("disable"); return 0;
}
''')
subprocess.run(['gcc', '-Wall', '-Wextra', '-Werror', '-Iinclude', '-Itests', str(source),
                'tests/bedrock_support.c', '-lpthread', '-o', str(output/'client')], check=True)
clean_env = {key: value for key, value in os.environ.items() if key not in ('BEDROCK_CTL', 'BEDROCK_ACK')}
subprocess.run([str(output/'client')], env=clean_env, check=True, timeout=5)
for key in ('BEDROCK_CTL', 'BEDROCK_ACK'):
    result = subprocess.run([str(output/'client')], env={**clean_env, key: 'unused'}, capture_output=True, timeout=5)
    assert result.returncode != 0 and b'requires both FIFO paths' in result.stderr


def check(index, replies, valid):
    ctl, ack = output/f'ctl-{index}', output/f'ack-{index}'
    os.mkfifo(ctl); os.mkfifo(ack)
    writer = os.open(ctl, os.O_RDWR)
    reader = os.open(ack, os.O_RDWR)
    errors = []

    def server():
        try:
            for expected, reply in zip((b'enable\n', b'disable\n'), replies):
                command = b''
                while not command.endswith(b'\n'):
                    if not select.select([writer], [], [], 5)[0]:
                        raise AssertionError('client command timed out')
                    command += os.read(writer, 1)
                assert command == expected
                # Exercise the client with individually written stream bytes.
                for byte in reply:
                    os.write(reader, bytes([byte]))
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=server, daemon=True)
    worker.start()
    try:
        result = subprocess.run([str(output/'client')], env={**os.environ, 'BEDROCK_CTL': str(ctl),
            'BEDROCK_ACK': str(ack)}, capture_output=True, timeout=10)
        worker.join(6)
        assert not worker.is_alive() and not errors, errors
        assert (result.returncode == 0) == valid, result.stderr
        (output/f'case-{index}.txt').write_text(f'status={result.returncode}\n{result.stderr.decode()}')
    finally:
        os.close(writer); os.close(reader)


for index, replies in enumerate(((b'ack\n', b'ack\n'), (b'ack\n\0', b'ack\n\0'),
                                 (b'ack\n\0', b'ack\n'), (b'ack\n', b'ack\n\0'))):
    check(index, replies, True)
for index, reply in enumerate((b'\0ack\n', b'bad\n', b'ac\0\n'), start=4):
    check(index, (reply,), False)
# Explicit synthetic fixtures: these are report unit tests, never measurements.
fixture = output/'synthetic-report-fixture'
fixture.mkdir()
runs = []
first_raw = None
for producers in (1, 2, 8):
    for draws in (1, 64):
        for pair in (1, 2, 3):
            for variant, implementation in (('baseline', 'neogpu'), ('candidate', 'neogpu'), ('baseline', 'rigtorp')):
                summary = {'record': 'summary', 'participants': producers, 'draws': draws,
                           'implementation': implementation, 'accepted': producers*128, 'applied': producers*128,
                           'elapsed_ns': producers*12800, 'retries': 0}
                run = {**summary, 'pair': pair, 'variant': variant, 'fps': 10000000.0}
                runs.append(run)
                sample = {'record': 'sample', 'attempt_ns': 1, 'admission_return_ns': 2, 'applied_ns': 3}
                raw = fixture/f'run-{producers}-{draws}-{pair}-{variant}-{implementation}.json'
                raw.write_text(json.dumps({'status': 0, 'stdout': json.dumps(sample)+'\n'+json.dumps(summary)+'\n'}))
                if first_raw is None: first_raw = raw
manifest_path = fixture/'manifest.json'


def report_case(valid, reason=''):
    manifest_path.write_text(json.dumps({'completed': True, 'synthetic': True, 'runs': runs}))
    result = subprocess.run([sys.executable, 'tests/test_bedrock_p0_report.py', str(fixture)],
                            capture_output=True, text=True, timeout=10)
    assert (result.returncode == 0) == valid and (valid or reason in result.stderr), result.stderr


report_case(True)
runs[0]['fps'] *= 2
report_case(False, 'cached rate contradicts elapsed time')
runs[0]['fps'] /= 2
runs[0]['elapsed_ns'] = float('inf')
report_case(False, 'noninteger timing metadata')
runs[0]['elapsed_ns'] = 12800
raw = json.loads(first_raw.read_text())
records = [json.loads(line) for line in raw['stdout'].splitlines()]
records[-1]['elapsed_ns'] += 1
raw['stdout'] = '\n'.join(map(json.dumps, records))
first_raw.write_text(json.dumps(raw))
report_case(False, 'elapsed_ns')
runs.pop()
report_case(False, 'incomplete matrix')
print('Bedrock tooling: parser/ACK controls and synthetic rate/raw-summary/completeness controls passed')
