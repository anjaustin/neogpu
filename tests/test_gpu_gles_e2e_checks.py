"""Synthetic fail-closed validator controls; these are NOT hardware benchmark results."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import statistics

if not __debug__:
    raise RuntimeError('Validator controls require assertions enabled')

root = Path(__file__).resolve().parents[1]
output = root/('tests/evidence-e2e-validator-controls-'+str(__import__('os').getpid()))
output.mkdir(exist_ok=False)
tree = ast.parse(Path(__file__).with_name('test_gpu_gles_e2e.py').read_text())
namespace = {'Path': Path, 'hashlib': hashlib, 'statistics': statistics}
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate']
assert len(functions) == 1
exec(compile(ast.Module(body=functions, type_ignores=[]), 'validator-controls', 'exec'), namespace)
prefix = output/'SYNTHETIC-NOT-HARDWARE'
Path(str(prefix)+'.rgba').write_bytes(bytes(16*8*4))
summary = {'record': 'summary', 'mode': 'native', 'producers': 1, 'draws': 1, 'frames_per_producer': 3,
    'capture': 'off', 'resources_released': True, 'accepted': 3, 'applied': 3, 'rendered_frames': 3, 'elapsed_ns': 1000}
original = [summary, {'record': 'context', 'width': 16, 'height': 8}]
original += [{'record': 'contribution', 'lane': 0, 'sequence': seq, 'attempt_ns': seq*10, 'admitted_ns': seq*10+1, 'completed_ns': seq*10+2} for seq in range(1, 4)]
original += [{'record': 'frame', 'index': index, 'submit_ns': 1, 'completed_ns': 2, 'cpu_ns': 1} for index in range(3)]
results = []
for case in ('positive', 'live_resources', 'lost_contribution', 'wrong_sequence', 'negative_latency', 'negative_admission',
             'missing_frame', 'bad_frame_index', 'completion_before_submit', 'duplicate_summary', 'duplicate_context', 'truncated_rgba', 'wrong_capture'):
    records = copy.deepcopy(original)
    if case == 'live_resources': records[0]['resources_released'] = False
    elif case == 'lost_contribution': records[0]['applied'] -= 1
    elif case == 'wrong_sequence': records[2]['sequence'] = 2
    elif case == 'negative_latency': records[2]['completed_ns'] = 0
    elif case == 'negative_admission': records[2]['admitted_ns'] = 0
    elif case == 'missing_frame': records.pop()
    elif case == 'bad_frame_index': records[-1]['index'] = 0
    elif case == 'completion_before_submit': records[-1]['completed_ns'] = 0
    elif case == 'duplicate_summary': records.append(copy.deepcopy(records[0]))
    elif case == 'duplicate_context': records.append(copy.deepcopy(records[1]))
    elif case == 'truncated_rgba': Path(str(prefix)+'.rgba').write_bytes(b'')
    elif case == 'wrong_capture': records[0]['capture'] = 'on'
    try:
        namespace['validate'](records, 'native', 1, 1, 3, 'off', prefix)
        accepted = True
    except (AssertionError, ValueError, ZeroDivisionError): accepted = False
    assert accepted == (case == 'positive'), case
    results.append({'case': case, 'accepted': accepted})
    Path(str(prefix)+'.rgba').write_bytes(bytes(16*8*4))
(output/'controls.json').write_text(json.dumps({'synthetic_only': True, 'cases': results}, indent=2))
print(json.dumps({'synthetic_only': True, 'positive': 1, 'rejected': len(results)-1, 'passed': True}))
