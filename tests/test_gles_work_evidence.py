"""Compare committed GLES paths or diagnostic frame-local state caches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

if not __debug__:
    raise RuntimeError('GLES evidence requires assertions enabled')
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--baseline', default='2bf623a')
parser.add_argument('--candidate', default='HEAD', help='candidate revision for --production')
parser.add_argument('--production', action='store_true', help='compare unmodified committed revisions; no cache or common repair')
parser.add_argument('--frames', type=int, default=256)
parser.add_argument('--pairs', type=int, default=5)
parser.add_argument('--sanitizers', action='store_true')
args = parser.parse_args()
if not args.output.name.startswith('evidence-') or not args.output.resolve().is_relative_to(ROOT/'tests'):
    parser.error('fresh tests/evidence-* output required')
if not 3 <= args.frames <= 512 or not 1 <= args.pairs <= 9:
    parser.error('frames must be 3..512 and pairs 1..9')
args.output = args.output.resolve().relative_to(ROOT)
args.output.mkdir(exist_ok=False)
manifest = {'completed': False, 'baseline': subprocess.check_output(['git', 'rev-parse', args.baseline], text=True).strip(),
            'source_sha256': {}, 'commands': [], 'runs': [], 'qualification': [], 'pairs': args.pairs, 'frames': args.frames, 'audit_frames': 4,
            'contract': 'Same ordered commands and pixels. GL call counting only in separate audit binaries. Timed frames use normal presentation, with a single separately recorded glFinish at block end. Capture journal reset outside blocks; cold first-use and warm timing separate. No GPU/display FPS claim.'}
manifest['common_remediation'] = 'Both baseline and candidate apply the same connector/EGL teardown repair to frozen source; only candidate changes draw-state setup. Original header is also snapshotted.'
manifest['comparison'] = 'committed-default-path' if args.production else 'diagnostic-cache'
manifest['revisions'] = {'baseline': manifest['baseline'], 'candidate':
    subprocess.check_output(['git', 'rev-parse', args.candidate], text=True).strip() if args.production else manifest['baseline']}
if args.production: manifest['common_remediation'] = 'None: exact committed source for both revisions; no diagnostic cache or header transformation.'
runtime = ('hs_core', 'hs_nodes', 'hs_gpu', 'hs_fabric', 'hs_async', 'hs_backend_gles', 'hs_ipc', 'hs_text', 'hs_input_stream')
scenes = ('single', 'same', 'alternate', 'dirty', 'mixed', 'fallback', 'retained')


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    manifest['source_sha256'][str(path)] = hashlib.sha256(data).hexdigest()


def environment():
    data = {'wall_ns': time.time_ns(), 'affinity': sorted(os.sched_getaffinity(0)), 'sysfs': {}}
    for base, pattern in [('/sys/devices/system/cpu', 'cpu*/cpufreq/scaling_*'), ('/sys/class/thermal', 'thermal_zone*/temp')]:
        for path in sorted(Path(base).glob(pattern)):
            if path.is_file():
                try: data['sysfs'][str(path)] = path.read_text().strip()
                except OSError as error: data['sysfs'][str(path)] = str(error)
    return data


def command(argv, label):
    before = environment()
    result = subprocess.run(list(map(str, argv)), capture_output=True, text=True, timeout=240)
    record = {'argv': list(map(str, argv)), 'status': result.returncode, 'stdout': result.stdout,
              'stderr': result.stderr, 'before': before, 'after': environment()}
    with (args.output/(label+'.json')).open('x') as file: json.dump(record, file, indent=2)
    manifest['commands'].append({'label': label, 'status': result.returncode})
    assert result.returncode == 0, record
    return [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]


def edit(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def candidate(text):
    text = edit(text, 'static GLuint gles_get_vbo_for_buffer(HSGLESBackend* b, HSBuffer* buf, u8 idx) {',
                'static GLuint gles_get_vbo_for_buffer(HSGLESBackend* b, HSBuffer* buf, u8 idx, bool* uploaded) {')
    text = edit(text, '    if (buf->dirty || b->vbo_sizes[idx] != byte_len) {',
                '    if (buf->dirty || b->vbo_sizes[idx] != byte_len) {\n        *uploaded = true;')
    text = edit(text, '                GLuint vbo = gles_get_vbo_for_buffer(b, cpu_buf, buf_idx);',
                '                bool uploaded = false;\n'
                '                GLuint vbo = gles_get_vbo_for_buffer(b, cpu_buf, buf_idx, &uploaded);')
    text = edit(text, '    for (u32 i = 0; i < frame->render->count; i++) {',
                '    /* Frame-local only: external GL access between calls is allowed. */\n'
                '    bool draw_setup = false;\n    GLuint draw_vbo = 0;\n'
                '    for (u32 i = 0; i < frame->render->count; i++) {')
    text = edit(text, '        const HSRenderCmd* c = &frame->render->cmds[i];',
                '        const HSRenderCmd* c = &frame->render->cmds[i];\n'
                '        if (c->op != HS_RC_DRAW && c->op != HS_RC_DRAW_INSTANCE) draw_setup = false;')
    old = '''                glUseProgram(b->program);
                glBindBuffer(GL_ARRAY_BUFFER, vbo);
                glEnableVertexAttribArray(b->pos_loc);
                glVertexAttribPointer(b->pos_loc, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
                glEnableVertexAttribArray(b->color_loc);
                glVertexAttribPointer(b->color_loc, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));'''
    new = '''                if (!draw_setup || uploaded) {
                    glUseProgram(b->program);
                    glBindBuffer(GL_ARRAY_BUFFER, vbo);
                    glEnableVertexAttribArray(b->pos_loc);
                    glVertexAttribPointer(b->pos_loc, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
                    glEnableVertexAttribArray(b->color_loc);
                    glVertexAttribPointer(b->color_loc, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
                } else if (draw_vbo != vbo) {
                    glBindBuffer(GL_ARRAY_BUFFER, vbo);
                    glVertexAttribPointer(b->pos_loc, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
                    glVertexAttribPointer(b->color_loc, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
                }
                /* Keep the original setup sequence after uploads: the first
                 * candidate regressed the dirty-buffer workload. */
                draw_setup = !uploaded;
                draw_vbo = vbo;'''
    return edit(text, old, new)


def cleanup(text):
    text = edit(text, '                gfx->screen_height = conn->modes[0].vdisplay;\n                break;',
                '                gfx->screen_height = conn->modes[0].vdisplay;\n'
                '                drmModeFreeConnector(conn);\n                conn = NULL;\n                break;')
    text = edit(text, '    if (gfx->egl_surface != EGL_NO_SURFACE) {\n        eglDestroySurface',
                '    eglMakeCurrent(gfx->egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);\n'
                '    if (gfx->egl_surface != EGL_NO_SURFACE) {\n        eglDestroySurface')
    text = edit(text, '    if (gfx->gbm_surface) {\n        gbm_surface_destroy',
                '    eglTerminate(gfx->egl_display);\n    eglReleaseThread();\n'
                '    if (gfx->gbm_surface) {\n        gbm_surface_destroy')
    return text


def build(variant, sanitizer=False):
    directory = args.output/(variant+('-asan' if sanitizer else ''))
    directory.mkdir()
    revision = manifest['revisions'][variant]
    headers = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', revision, 'include'], text=True).splitlines()
    names = [name for name in headers if name.endswith('.h')] + ['src/'+name+'.c' for name in runtime]
    for name in names:
        data = subprocess.check_output(['git', 'show', revision+':'+name])
        if not args.production and name == 'include/hs_graphics.h':
            save(args.output/'original'/name, data)
            data = cleanup(data.decode()).encode()
        if not args.production and variant == 'candidate' and name == 'src/hs_backend_gles.c': data = candidate(data.decode()).encode()
        save(directory/name, data)
    flags = ['-O3', '-g', '-march=armv8-a+simd', '-mtune=cortex-a72', '-ffast-math', '-funroll-loops',
             '-Wall', '-Wextra', '-Werror', '-DNDEBUG', '-I'+str(directory/'include'), '-I/usr/include/libdrm']
    if sanitizer: flags += ['-O1', '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
    objects = []
    for name in runtime:
        obj = directory/(name+'.o'); objects.append(obj)
        command(['gcc', *flags, '-c', directory/'src'/(name+'.c'), '-o', obj], directory.name+'-compile-'+name)
    for audit in (False, True):
        obj = directory/('audit.o' if audit else 'timed.o')
        command(['gcc', *flags, *(['-DHS_GLES_AUDIT_COUNTS'] if audit else []), '-c', args.output/'common/test_gles_work.c', '-o', obj], directory.name+'-harness-'+str(audit))
        wraps = ['eglSwapBuffers'] + (['glUseProgram', 'glBindBuffer', 'glEnableVertexAttribArray', 'glVertexAttribPointer', 'glDrawArrays', 'glBufferData'] if audit else [])
        binary = directory/('audit' if audit else 'timed')
        command(['gcc', *flags, obj, *objects, '-lm', '-lGLESv2', '-lgbm', '-ldrm', '-lEGL', '-lpthread',
                 *['-Wl,--wrap='+name for name in wraps], '-o', binary], directory.name+'-link-'+str(audit))
        manifest.setdefault('binary_sha256', {})[str(binary)] = hashlib.sha256(binary.read_bytes()).hexdigest()
    return directory


def run(directory, scene, capture, label, audit=False, sanitizer=False):
    prefix = args.output/label
    argv = ['sudo', '-n']
    if sanitizer: argv += ['env', 'ASAN_OPTIONS=detect_leaks=1:halt_on_error=1', 'UBSAN_OPTIONS=halt_on_error=1']
    argv += [directory/('audit' if audit else 'timed'), scene, capture, args.frames, prefix, 'audit' if audit else 'timed']
    records = command(argv, label)
    lifecycle = [r for r in records if r['record'] == 'lifecycle']; assert len(lifecycle) == 1
    if args.production and directory.name.startswith('candidate'): assert lifecycle[0]['context_released'] is True
    assert len([r for r in records if r['record'] == 'cold']) == 1
    if audit:
        counts = [r for r in records if r['record'] == 'calls']; assert len(counts) == 1
        draws = {'single': 1, 'same': 64, 'alternate': 64, 'dirty': 2, 'mixed': 12, 'fallback': 2, 'retained': 64}[scene]
        assert counts[0]['draw'] == 3*draws
        assert counts[0]['upload'] == (6 if scene == 'dirty' else 0)
        return counts[0]
    blocks = [r for r in records if r['record'] == 'block']; assert len(blocks) == 1
    frames = [r for r in records if r['record'] == 'frame']
    assert len(frames) == args.frames and [r['index'] for r in frames] == list(range(args.frames))
    block = blocks[0]
    assert block['scene'] == scene and block['capture'] == capture and block['frames'] == args.frames
    assert block['journal_count'] == (block['messages'] if capture == 'on' else 0)
    return {**block, 'lifecycle': lifecycle[0], 'median_submit_ns': statistics.median(r['submit_ns'] for r in frames),
            'median_swap_ns': statistics.median(r['swap_ns'] for r in frames),
            'median_total_ns': statistics.median(r['total_ns'] for r in frames),
            'cold': next(r for r in records if r['record'] == 'cold')}


try:
    save(args.output/'driver.py', Path(__file__).read_bytes())
    save(args.output/'common/test_gles_work.c', Path('tests/test_gles_work.c').read_bytes())
    manifest['toolchain'] = {name: subprocess.check_output(argv, text=True) for name, argv in
        [('gcc', ['gcc', '--version']), ('kernel', ['uname', '-a']), ('working_tree', ['git', 'status', '--short'])]}
    binaries = {variant: build(variant) for variant in ('baseline', 'candidate')}
    for scene in scenes:
        for capture in ('off', 'on'):
            entries = {}
            for variant in ('baseline', 'candidate'):
                label = f'audit-{scene}-{capture}-{variant}'
                entries[variant] = run(binaries[variant], scene, capture, label, audit=True)
            for frame in range(4):
                paths = [args.output/f'audit-{scene}-{capture}-{v}-{frame}.rgba' for v in ('baseline', 'candidate')]
                data = [path.read_bytes() for path in paths]; assert data[0] == data[1], paths
            manifest['qualification'].append({'scene': scene, 'capture': capture, 'calls': entries, 'rgba_identical': True})
    if args.sanitizers:
        for variant in ('baseline', 'candidate'):
            directory = build(variant, sanitizer=True)
            for scene in scenes: run(directory, scene, 'on', f'asan-{scene}-{variant}', audit=True, sanitizer=True)
    for scene in scenes:
        for pair in range(1, args.pairs+1):
            order = [(v, c) for c in ('off', 'on') for v in ('baseline', 'candidate')]
            if pair % 2 == 0: order.reverse()
            for variant, capture in order:
                label = f'timed-{scene}-{pair}-{capture}-{variant}'
                block = run(binaries[variant], scene, capture, label)
                manifest['runs'].append({**block, 'variant': variant, 'pair': pair, 'label': label})
    manifest['completed'] = True
finally:
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
