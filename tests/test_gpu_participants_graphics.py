"""Exercise normal HSGpu lifecycle/pixels using generated diagnostic copies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

if not __debug__:
    raise RuntimeError('Graphics evidence requires Python assertions enabled')
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
if not args.output.name.startswith('evidence-') or not args.output.resolve().is_relative_to(ROOT/'tests'):
    parser.error('output must be a fresh tests/evidence-* directory')
args.output.mkdir(exist_ok=False)
manifest = {'completed': False, 'commands': [], 'source_sha256': {}}


def edit(source, before, after):
    assert source.count(before) == 1, before
    return source.replace(before, after)


def command(argv, label):
    result = subprocess.run(list(map(str, argv)), text=True, capture_output=True, timeout=120)
    record = {'argv': list(map(str, argv)), 'status': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    (args.output/(label+'.json')).write_text(json.dumps(record, indent=2))
    manifest['commands'].append({'label': label, 'status': result.returncode})
    assert result.returncode == 0, record
    return result.stdout


try:
    command(['make', 'tests/test_fabric_graphics', 'tests/test_comparison_graphics'], 'runtime-build')
    objects = [Path('src')/(name+'.o') for name in ('hs_core', 'hs_nodes', 'hs_gpu', 'hs_fabric', 'hs_async', 'hs_backend_gles', 'hs_ipc', 'hs_text', 'hs_input_stream')]
    for path in [*Path('include').glob('*.h'), *Path('src').glob('*.c')]:
        manifest['source_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for variant, name in [('lifecycle', 'tests/test_fabric_graphics.c'), ('equivalence', 'tests/test_comparison_graphics.c')]:
        source = Path(name).read_text()
        manifest['source_sha256'][name] = hashlib.sha256(source.encode()).hexdigest()
        source = edit(source, 'static HSFabric fabric;', '#define fabric gpu.participants')
        source = edit(source, '    REQUIRE(hs_fabric_init(&fabric));\n', '')
        source = source.replace('hs_fabric_join(&fabric,', 'hs_gpu_participant_join(&gpu,')
        source = source.replace('hs_fabric_submit(&fabric,', 'hs_gpu_participant_submit(&gpu,')
        source = source.replace('hs_fabric_leave(&fabric,', 'hs_gpu_participant_leave(&gpu,')
        source = source.replace('hs_fabric_destroy(&fabric);', '')
        if variant == 'lifecycle':
            source = edit(source, '        while (hs_fabric_step(&fabric, HS_FABRIC_PARTICIPANTS)) { }',
                          '        /* Native service is performed by normal HSGpu calls below. */')
            source = edit(source, '            const f32 black[4] = {0, 0, 0, 1};\n            REQUIRE(hs_fabric_compose(&fabric, &gpu.render, black));',
                          '            hs_gpu_clear(&gpu, v4_make(0, 0, 0, 1));\n            hs_gpu_process(&gpu);')
            source = edit(source, '            hs_gpu_end_frame(&gpu);', '            REQUIRE(hs_gpu_end_frame_checked(&gpu));')
        else:
            source = edit(source, 'static const f32 black[4] = {0, 0, 0, 1};\n', '')
            source = edit(source, '    REQUIRE(hs_fabric_step(&fabric, 8) == 2);', '    REQUIRE(hs_gpu_process(&gpu) > 0);')
            source = edit(source, '        REQUIRE(hs_fabric_compose(&fabric, &gpu.render, black));',
                          '        hs_gpu_clear(&gpu, v4_make(0, 0, 0, 1));\n        hs_gpu_process(&gpu);')
            source = edit(source, '        hs_gpu_end_frame(&gpu);', '        REQUIRE(hs_gpu_end_frame_checked(&gpu));')
        assert 'hs_fabric_step(' not in source and 'hs_fabric_compose(' not in source
        path = args.output/(variant+'.c'); path.write_text(source)
        binary = args.output/variant
        command(['gcc', '-O3', '-Wall', '-Wextra', '-Werror', '-Iinclude', '-Itests', path, *objects,
                 '-lm', '-lGLESv2', '-lgbm', '-ldrm', '-lEGL', '-lpthread', '-Wl,--wrap=eglSwapBuffers', '-o', binary], variant+'-build')
        if variant == 'lifecycle':
            captures = args.output/'captures'; captures.mkdir()
            output = command(['sudo', '-n', 'env', 'NEOGPU_FABRIC_CAPTURE_DIR='+str(captures), binary], variant)
            records = [json.loads(line) for line in output.splitlines() if line.startswith('{')]
            assert any(r.get('presentations') == 18 and r.get('pixel_checks') == 36 and r.get('passed') for r in records)
        else:
            output = command(['sudo', '-n', binary, args.output/'rgba'], variant)
            records = [json.loads(line) for line in output.splitlines() if line.startswith('{')]
            assert any(r.get('record') == 'pixel_equivalence' and r.get('identical') for r in records)
    manifest['completed'] = True
finally:
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
