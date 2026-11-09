"""Deterministic publication/retirement interleavings with bug-mutation controls."""
import argparse
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
parser.add_argument('--output', type=Path)
parser.add_argument('--sanitizers', action='store_true')
args = parser.parse_args()
output = args.output or Path(f'tests/evidence-bedrock-adversarial-{time.time_ns()}-{os.getpid()}')
if not output.name.startswith('evidence-') or not output.resolve().is_relative_to(ROOT/'tests'):
    parser.error('output must be a fresh tests/evidence-* directory')
output.mkdir(exist_ok=False)
original = Path('src/hs_fabric.c').read_text()
manifest = {'completed': False, 'production_sha256': hashlib.sha256(original.encode()).hexdigest(), 'commands': []}


def edit(source, before, after):
    assert source.count(before) == 1, before
    return source.replace(before, after)


def run(argv, label, expected_failure=None, env=None):
    result = subprocess.run(list(map(str, argv)), text=True, capture_output=True, timeout=30, env=env)
    record = {'argv': list(map(str, argv)), 'status': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    (output/(label+'.json')).write_text(json.dumps(record, indent=2))
    manifest['commands'].append({'label': label, 'status': result.returncode})
    if expected_failure:
        assert result.returncode == 1 and expected_failure in result.stderr, record
    else:
        assert result.returncode == 0, record


try:
    source = edit(original, '#include <string.h>', '#include <string.h>\nextern void bedrock_adversarial_hook(unsigned stage);')
    publish = '__atomic_store_n(&slot->write_index, write + 1, __ATOMIC_RELEASE);'
    release = '__atomic_store_n(&slot->read_index, read + 1, __ATOMIC_RELEASE);'
    source = edit(source, publish, 'bedrock_adversarial_hook(1);\n            '+publish)
    source = edit(source, '        if (read != write) {', '        bedrock_adversarial_hook(2);\n        if (read != write) {')
    source = edit(source, release, 'bedrock_adversarial_hook(3);\n            '+release)
    flags = ['-O3', '-g', '-Wall', '-Wextra', '-Werror', '-Iinclude', '-Itests', '-lpthread']
    variants = [('correct', source, None, [])]
    if args.sanitizers:
        variants += [('asan', source, None, ['-O1', '-fno-omit-frame-pointer', '-fsanitize=address,undefined']),
                     ('tsan', source, None, ['-O1', '-fno-omit-frame-pointer', '-fsanitize=thread'])]
    early = edit(source, 'bedrock_adversarial_hook(3);\n            '+release,
                 release+'\n            bedrock_adversarial_hook(3);')
    stale = edit(source, '__atomic_load_n(&slot->read_index, __ATOMIC_RELAXED) ==\n                __atomic_load_n(&slot->write_index, __ATOMIC_ACQUIRE)', 'true')
    variants += [('bad-early-reuse', early, 'hs_fabric_submit(&fabric, owner, &excess) == HS_FABRIC_FULL', []),
                 ('bad-unchecked-retirement', stale, 'step_result == 0 && fabric.retired == 0 && fabric.applied == 0', [])]
    env = {**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1', 'UBSAN_OPTIONS': 'halt_on_error=1', 'TSAN_OPTIONS': 'halt_on_error=1'}
    for label, text, failure, extra in variants:
        path = output/(label+'.c')
        path.write_text(text)
        run(['gcc', *flags, *extra, path, 'tests/test_bedrock_adversarial.c', '-o', output/label], label+'-build')
        run([output/label], label, failure, env)
    assert Path('src/hs_fabric.c').read_text() == original
    manifest['completed'] = True
    print('Adversarial interleavings passed; early-reuse and unchecked-retirement mutations rejected')
finally:
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
