"""Reject deliberately unsafe GL-state caches using actual frozen GLES probes."""
import argparse
import json
import os
from pathlib import Path
import subprocess

if not __debug__:
    raise RuntimeError('GLES mutations require assertions enabled')
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('collection', type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
if not args.output.name.startswith('evidence-') or not args.output.resolve().is_relative_to(ROOT/'tests'):
    parser.error('fresh tests/evidence-* output required')
args.output = args.output.resolve().relative_to(ROOT)
args.collection = args.collection.resolve().relative_to(ROOT)
args.output.mkdir(exist_ok=False)
source = (args.collection/'candidate/src/hs_backend_gles.c').read_text()
compile_command = json.loads((args.collection/'candidate-compile-hs_backend_gles.json').read_text())['argv']
link_command = json.loads((args.collection/'candidate-link-True.json').read_text())['argv']
results = []


def command(argv, label):
    result = subprocess.run(list(map(str, argv)), text=True, capture_output=True, timeout=180, cwd=ROOT)
    record = {'argv': list(map(str, argv)), 'status': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    (args.output/(label+'.json')).write_text(json.dumps(record, indent=2))
    return result


old_buffer_condition = '                if (!draw_setup || draw_vbo != vbo) {'
buffer_condition = old_buffer_condition if old_buffer_condition in source else '                } else if (draw_vbo != vbo) {'
buffer_replacement = '                if (false) {' if buffer_condition == old_buffer_condition else '                } else if (false) {'
mutations = [
    ('persistent', 'same', '    bool draw_setup = false;', '    static bool draw_setup = false;'),
    ('texture-transition', 'mixed', '        if (c->op != HS_RC_DRAW && c->op != HS_RC_DRAW_INSTANCE) draw_setup = false;', ''),
    ('buffer-transition', 'alternate', buffer_condition, buffer_replacement)]
for name, scene, old, new in mutations:
    assert source.count(old) == 1
    mutated = source.replace(old, new)
    if name == 'buffer-transition':
        mutated = mutated.replace('    GLuint draw_vbo = 0;\n', '').replace('                draw_vbo = vbo;\n', '')
    path = args.output/(name+'.c'); path.write_text(mutated)
    obj = args.output/(name+'.o'); binary = args.output/name
    argv = [str(path) if a == str(args.collection/'candidate/src/hs_backend_gles.c') else str(obj) if a == str(args.collection/'candidate/hs_backend_gles.o') else a for a in compile_command]
    assert str(path) in argv and str(obj) in argv
    assert command(argv, name+'-compile').returncode == 0
    argv = [str(obj) if a == str(args.collection/'candidate/hs_backend_gles.o') else str(binary) if a == str(args.collection/'candidate/audit') else a for a in link_command]
    assert str(obj) in argv and str(binary) in argv
    assert command(argv, name+'-link').returncode == 0
    prefix = args.output/(name+'-pixels')
    result = command(['sudo', '-n', binary, scene, 'on', '3', prefix, 'audit'], name+'-run')
    mismatch = False
    if result.returncode == 0:
        mismatch = any((args.output/f'{name}-pixels-{i}.rgba').read_bytes() !=
                       (args.collection/f'audit-{scene}-on-baseline-{i}.rgba').read_bytes() for i in range(4))
    assert result.returncode != 0 or mismatch, (name, 'unsafe cache escaped verification')
    results.append({'mutation': name, 'scene': scene, 'rejected': True, 'status': result.returncode, 'rgba_mismatch': mismatch})
(args.output/'results.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
