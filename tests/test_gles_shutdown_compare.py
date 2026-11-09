"""Compare bounded shutdown candidates without weakening complete release."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

if not __debug__:
    raise RuntimeError('Shutdown comparison requires assertions enabled')
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--baseline', default='3aaab60')
parser.add_argument('--pairs', type=int, default=7)
parser.add_argument('--frames', type=int, default=32)
parser.add_argument('--experiment', choices=('close-only', 'early-rmfb', 'restore-crtc', 'flip-restore'), default='close-only')
args = parser.parse_args()
if not args.output.name.startswith('evidence-') or not args.output.resolve().is_relative_to(ROOT/'tests'):
    parser.error('fresh tests/evidence-* output required')
if not 3 <= args.frames <= 512 or not 3 <= args.pairs <= 31: parser.error('invalid frames/pairs')
args.output = args.output.resolve().relative_to(ROOT); args.output.mkdir(exist_ok=False)
manifest = {'completed': False, 'baseline': subprocess.check_output(['git', 'rev-parse', args.baseline], text=True).strip(),
            'source_sha256': {}, 'commands': [], 'runs': [], 'pairs': args.pairs, 'frames': args.frames,
            'contract': 'Full EGL teardown and final fd close in both variants. Candidate removes only explicit drmDropMaster before last close. No delayed cleanup. Reacquisition and ownership-probe duration measured separately and combined with shutdown.'}
variants = ('baseline', 'close_only') if args.experiment == 'close-only' else ('baseline', 'owned_fb', 'rmfb_early')
if args.experiment == 'restore-crtc': variants = ('baseline', 'saved_state', 'restore_crtc')
if args.experiment == 'flip-restore': variants = ('baseline', 'saved_state', 'flip_restore')
manifest['experiment'] = args.experiment; manifest['variants'] = list(variants)
if args.experiment == 'early-rmfb':
    manifest['contract'] = 'Exact baseline, per-instance tracked-fb control, and same control with early explicit RmFB. All EGL cleanup, master drop and final close remain. No delayed cleanup; combined shutdown and ownership-probe duration is also measured.'
if args.experiment == 'restore-crtc':
    manifest['contract'] = 'Exact baseline, saved-display-state control, and same control restoring original CRTC/connectors before cleanup. All EGL cleanup and final close remain. Capture/query failures fall back to baseline cleanup. No delayed cleanup; ownership-probe duration is measured separately and combined.'
if args.experiment == 'flip-restore':
    manifest['contract'] = 'Same saved-state control using cached connector queries; page-flip restoration only for matching timing/connector/viewport configurations, with event completion before cleanup. Otherwise full SetCrtc restore; all EGL release and final close remain.'
runtime = ('hs_core', 'hs_nodes', 'hs_gpu', 'hs_fabric', 'hs_async', 'hs_backend_gles', 'hs_ipc', 'hs_text', 'hs_input_stream')


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    manifest['source_sha256'][str(path)] = hashlib.sha256(data).hexdigest()


def command(argv, label):
    environment_before = environment()
    before = time.monotonic_ns()
    result = subprocess.run(list(map(str, argv)), text=True, capture_output=True, timeout=240)
    record = {'argv': list(map(str, argv)), 'status': result.returncode, 'stdout': result.stdout,
              'stderr': result.stderr, 'whole_process_ns': time.monotonic_ns()-before,
              'environment_before': environment_before, 'environment_after': environment()}
    with (args.output/(label+'.json')).open('x') as out: json.dump(record, out, indent=2)
    manifest['commands'].append({'label': label, 'status': result.returncode})
    assert result.returncode == 0, record
    return [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]


def environment():
    data = {'wall_ns': time.time_ns(), 'affinity': sorted(os.sched_getaffinity(0)), 'sysfs': {}}
    for base, pattern in [('/sys/devices/system/cpu', 'cpu*/cpufreq/scaling_*'), ('/sys/class/thermal', 'thermal_zone*/temp')]:
        for path in sorted(Path(base).glob(pattern)):
            if path.is_file():
                try: data['sysfs'][str(path)] = path.read_text().strip()
                except OSError as error: data['sysfs'][str(path)] = str(error)
    return data


def transform(text, variant, experiment):
    if variant == 'baseline': return text
    if experiment == 'close-only':
        target = '        drmDropMaster(gfx->drm_fd);\n'; assert text.count(target) == 1
        return text.replace(target, '')
    target = '    bool           vsync;\n'; assert text.count(target) == 1
    text = text.replace(target, target+'    u32            scanout_fb; /* owned on drm_fd */\n    bool           crtc_setup_done;\n')
    target = 'static int crtc_setup_done = 0;\n'; assert text.count(target) == 1
    text = text.replace(target, '')
    text = text.replace('if (!crtc_setup_done)', 'if (!gfx->crtc_setup_done)').replace('        crtc_setup_done = 1;', '        gfx->crtc_setup_done = true;')
    target = '                        drmModeSetCrtc('; assert text.count(target) == 1
    text = text.replace(target, '                        gfx->scanout_fb = fb;\n'+target)
    if experiment in ('restore-crtc', 'flip-restore'):
        target = '    bool           crtc_setup_done;\n'; assert text.count(target) == 1
        text = text.replace(target, target+'    drmModeCrtc*   saved_crtc;\n    u32*           saved_connectors;\n    int            saved_connector_count;\n')
        if experiment == 'flip-restore':
            text = text.replace('    int            saved_connector_count;\n', '    int            saved_connector_count;\n    bool           flip_restore_eligible, restore_done;\n')
            text = text.replace('#include <errno.h>', '#include <errno.h>\n#include <poll.h>\n#include <time.h>')
            text = text.replace('} HSGraphics;', '} HSGraphics;\n\nstatic inline bool hs_graphics_restore_flip(HSGraphics* gfx);')
        helper = '''static inline void hs_graphics_save_scanout(HSGraphics* gfx, drmModeRes* resources, u32 crtc) {
    if (gfx->saved_crtc || resources->count_connectors <= 0) return;
    drmModeCrtc* saved = drmModeGetCrtc(gfx->drm_fd, crtc);
    if (!saved) return;
    u32* connectors = calloc((size_t)resources->count_connectors, sizeof(*connectors));
    if (!connectors) { drmModeFreeCrtc(saved); return; }
    int count = 0; bool valid = true;
    for (int i = 0; i < resources->count_connectors; i++) {
        drmModeConnector* connector = drmModeGetConnector(gfx->drm_fd, resources->connectors[i]);
        if (!connector) { valid = false; break; }
        if (connector->encoder_id) {
            drmModeEncoder* encoder = drmModeGetEncoder(gfx->drm_fd, connector->encoder_id);
            if (!encoder) valid = false;
            else { if (encoder->crtc_id == crtc) connectors[count++] = connector->connector_id; drmModeFreeEncoder(encoder); }
        }
        drmModeFreeConnector(connector);
        if (!valid) break;
    }
    if (valid && count && saved->mode_valid && saved->buffer_id) {
        gfx->saved_crtc = saved; gfx->saved_connectors = connectors; gfx->saved_connector_count = count;
    } else { free(connectors); drmModeFreeCrtc(saved); }
}

'''
        if experiment == 'flip-restore':
            helper = helper.replace('u32 crtc) {', 'u32 crtc, const drmModeModeInfo* mode, u32 connector_id) {')
            helper = helper.replace('drmModeGetConnector(gfx->drm_fd, resources->connectors[i])', 'drmModeGetConnectorCurrent(gfx->drm_fd, resources->connectors[i])')
            fields = ('clock', 'hdisplay', 'hsync_start', 'hsync_end', 'htotal', 'hskew', 'vdisplay', 'vsync_start', 'vsync_end', 'vtotal', 'vscan', 'vrefresh', 'flags')
            check = ' && '.join('saved->mode.'+name+' == mode->'+name for name in fields)
            helper = helper.replace('gfx->saved_connector_count = count;', 'gfx->saved_connector_count = count;\n        gfx->flip_restore_eligible = count == 1 && connectors[0] == connector_id && saved->x == 0 && saved->y == 0 && '+check+';')
            flip_helper = '''static void hs_graphics_restore_flip_done(int fd, unsigned sequence, unsigned sec, unsigned usec, void* data) {
    (void)fd; (void)sequence; (void)sec; (void)usec; *(bool*)data = true;
}
static inline bool hs_graphics_restore_flip(HSGraphics* gfx) {
    if (!gfx->flip_restore_eligible) return false;
    gfx->restore_done = false;
    if (drmModePageFlip(gfx->drm_fd, gfx->saved_crtc->crtc_id, gfx->saved_crtc->buffer_id,
                        DRM_MODE_PAGE_FLIP_EVENT, &gfx->restore_done) != 0) return false;
    drmEventContext context = {.version = 2, .page_flip_handler = hs_graphics_restore_flip_done};
    struct timespec start, now; if (clock_gettime(CLOCK_MONOTONIC, &start)) return false;
    while (!gfx->restore_done) {
        if (clock_gettime(CLOCK_MONOTONIC, &now)) break;
        long long elapsed = (long long)(now.tv_sec-start.tv_sec)*1000000000ll + now.tv_nsec-start.tv_nsec;
        if (elapsed >= 100000000ll) break;
        struct pollfd p = {.fd = gfx->drm_fd, .events = POLLIN};
        int result = poll(&p, 1, (int)((100000000ll-elapsed+999999ll)/1000000ll));
        if (result < 0 && errno == EINTR) continue;
        if (result <= 0 || (p.revents & (POLLERR|POLLHUP|POLLNVAL))) break;
        if ((p.revents & POLLIN) && drmHandleEvent(gfx->drm_fd, &context) != 0) break;
    }
    return gfx->restore_done;
}

'''
            # Define before finish; saved-state control must not acquire an unused static callback.
            if variant == 'flip_restore':
                target = 'static inline void hs_graphics_finish(HSGraphics* gfx) {\n    if (!gfx->initialized) return;'
                assert text.count(target) == 1
                text = text.replace(target, flip_helper+target)
            else:
                text = text.replace('\nstatic inline bool hs_graphics_restore_flip(HSGraphics* gfx);', '')
        target = 'static inline void hs_graphics_present(HSGraphics* gfx) {\n    if (!gfx->initialized) return;'; assert text.count(target) == 1
        text = text.replace(target, helper+target)
        target = '            if (conn && crtc_idx >= 0) {'; assert text.count(target) == 1
        extra = ', &conn->modes[0], conn->connector_id' if experiment == 'flip-restore' else ''
        text = text.replace(target, target+'\n                hs_graphics_save_scanout(gfx, resources, resources->crtcs[crtc_idx]'+extra+');')
    if variant == 'rmfb_early':
        target = 'static inline void hs_graphics_finish(HSGraphics* gfx) {\n    if (!gfx->initialized) return;'
        assert text.count(target) == 1
        text = text.replace(target, target+'\n    if (gfx->scanout_fb && drmModeRmFB(gfx->drm_fd, gfx->scanout_fb) == 0) gfx->scanout_fb = 0;')
    start = text.index('static inline void hs_graphics_finish(HSGraphics* gfx) {')
    end = text.index('static inline HSGLTexture* hs_graphics_create_texture', start)
    finish = text[start:end]; target = '    gfx->initialized = 0;'; assert finish.count(target) == 1
    finish = finish.replace(target, '    gfx->scanout_fb = 0;\n'+target)
    if experiment in ('restore-crtc', 'flip-restore'):
        if variant in ('restore_crtc', 'flip_restore'):
            target = '    if (!gfx->initialized) return;'; assert finish.count(target) == 1
            finish = finish.replace(target, target+'''
    if (gfx->scanout_fb && gfx->saved_crtc) {
        drmModeCrtc* saved = gfx->saved_crtc;
        (void)drmModeSetCrtc(gfx->drm_fd, saved->crtc_id, saved->buffer_id, saved->x, saved->y,
                           gfx->saved_connectors, gfx->saved_connector_count, &saved->mode);
    }
''')
            if variant == 'flip_restore':
                finish = finish.replace('    if (gfx->scanout_fb && gfx->saved_crtc) {', '    if (gfx->scanout_fb && gfx->saved_crtc && !hs_graphics_restore_flip(gfx)) {')
        finish = finish.replace('    gfx->scanout_fb = 0;', '''    if (gfx->saved_crtc) drmModeFreeCrtc(gfx->saved_crtc);
    free(gfx->saved_connectors); gfx->saved_crtc = NULL; gfx->saved_connectors = NULL; gfx->saved_connector_count = 0;
    gfx->scanout_fb = 0;''')
    return text[:start]+finish+text[end:]


def inject(harness):
    harness = harness.replace('#include <unistd.h>', '#include <unistd.h>\n#include <dirent.h>\n#include <fcntl.h>\n#include <errno.h>')
    helper = '''static u32 mode_sets, mode_failures, saved_fb, matched_restores;
static u32 flip_attempts, flip_submitted, flip_completed, flip_target;
static int flip_error;
static void* flip_data;
static void (*original_flip_handler)(int, unsigned, unsigned, unsigned, void*);
static void audited_flip_handler(int fd, unsigned sequence, unsigned sec, unsigned usec, void* data) {
    REQUIRE(data == flip_data && flip_submitted == 1 && flip_completed == 0);
    flip_completed++;
    original_flip_handler(fd, sequence, sec, usec, data);
}
extern int __real_drmModePageFlip(int, u32, u32, u32, void*);
int __wrap_drmModePageFlip(int fd, u32 crtc, u32 fb, u32 flags, void* data) {
    flip_attempts++; flip_target = fb; flip_data = data;
    REQUIRE(saved_fb && fb == saved_fb && flags == DRM_MODE_PAGE_FLIP_EVENT && data);
    int result = __real_drmModePageFlip(fd, crtc, fb, flags, data);
    flip_error = result ? errno : 0;
    if (!result) flip_submitted++;
    return result;
}
extern int __real_drmHandleEvent(int, drmEventContextPtr);
int __wrap_drmHandleEvent(int fd, drmEventContextPtr context) {
    REQUIRE(context && context->version == 2 && context->page_flip_handler);
    drmEventContext audited = *context;
    original_flip_handler = context->page_flip_handler;
    audited.page_flip_handler = audited_flip_handler;
    return __real_drmHandleEvent(fd, &audited);
}
extern drmModeCrtc* __real_drmModeGetCrtc(int, u32);
drmModeCrtc* __wrap_drmModeGetCrtc(int fd, u32 crtc) {
    drmModeCrtc* value = __real_drmModeGetCrtc(fd, crtc);
    if (value) saved_fb = value->buffer_id;
    return value;
}
extern int __real_drmModeSetCrtc(int, u32, u32, u32, u32, u32*, int, drmModeModeInfoPtr);
int __wrap_drmModeSetCrtc(int fd, u32 crtc, u32 fb, u32 x, u32 y, u32* connectors, int count, drmModeModeInfoPtr mode) {
    int result = __real_drmModeSetCrtc(fd, crtc, fb, x, y, connectors, count, mode);
    mode_sets++; if (result) mode_failures++;
    else if (mode_sets > 1 && saved_fb && fb == saved_fb) matched_restores++;
    return result;
}
static int fd_count(void) {
    DIR* directory = opendir("/proc/self/fd"); REQUIRE(directory);
    int own = dirfd(directory), count = 0; struct dirent* entry;
    while ((entry = readdir(directory))) {
        if (entry->d_name[0] == '.') continue;
        if (atoi(entry->d_name) != own) count++;
    }
    REQUIRE(closedir(directory) == 0); return count;
}
'''
    assert harness.count('static HSGpu gpu;') == 1
    harness = harness.replace('static HSGpu gpu;', helper+'static HSGpu gpu;')
    harness = harness.replace('    u64 init_start = ns();', '    int initial_fds = fd_count();\n    u64 init_start = ns();')
    marker = '    u64 init_ns = ns() - init_start;'
    replacement = marker+'''
    int graphics_fd = ((HSGraphics*)backend.ctx)->drm_fd;
    char link[64], device[256];
    REQUIRE(snprintf(link, sizeof(link), "/proc/self/fd/%d", graphics_fd) < (int)sizeof(link));
    ssize_t device_length = readlink(link, device, sizeof(device)-1); REQUIRE(device_length > 0);
    device[device_length] = 0;
'''
    assert harness.count(marker) == 1; harness = harness.replace(marker, replacement)
    marker = '    return 0;\n}'
    assert harness.count(marker) == 1
    resource = '''    errno = 0;
    bool fd_closed = fcntl(graphics_fd, F_GETFD) == -1 && errno == EBADF;
    int remaining_fds = fd_count();
    REQUIRE(fd_closed && remaining_fds == initial_fds);
    u64 probe_start = ns(); int probe_fd = open(device, O_RDWR | O_CLOEXEC); REQUIRE(probe_fd >= 0);
    REQUIRE(drmSetMaster(probe_fd) == 0 && drmIsMaster(probe_fd) == 1);
    u64 reacquire_ns = ns() - probe_start;
    REQUIRE(drmDropMaster(probe_fd) == 0 && close(probe_fd) == 0);
    u64 ownership_probe_ns = ns() - probe_start;
    REQUIRE(fd_count() == initial_fds);
    printf("{\\"record\\":\\"shutdown_resources\\",\\"initial_fds\\":%d,\\"remaining_fds\\":%d,\\"fd_closed\\":true,\\"reacquire_ns\\":%" PRIu64 ",\\"ownership_probe_ns\\":%" PRIu64 "}\\n",
           initial_fds, remaining_fds, reacquire_ns, ownership_probe_ns);
    printf("{\\"record\\":\\"mode_audit\\",\\"mode_sets\\":%u,\\"mode_failures\\":%u,\\"saved_fb\\":%u,\\"matched_restores\\":%u}\\n", mode_sets, mode_failures, saved_fb, matched_restores);
    printf("{\\"record\\":\\"flip_audit\\",\\"attempts\\":%u,\\"submitted\\":%u,\\"completed\\":%u,\\"target\\":%u,\\"error\\":%d}\\n", flip_attempts, flip_submitted, flip_completed, flip_target, flip_error);
'''
    return harness.replace(marker, resource+marker)


try:
    manifest['toolchain'] = {name: subprocess.check_output(argv, text=True) for name, argv in
        [('gcc', ['gcc', '--version']), ('kernel', ['uname', '-a']), ('working_tree', ['git', 'status', '--short'])]}
    save(args.output/'driver.py', Path(__file__).read_bytes())
    harness = subprocess.check_output(['git', 'show', manifest['baseline']+':tests/test_gles_work.c']).decode()
    save(args.output/'harness.c', inject(harness).encode())
    headers = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', manifest['baseline'], 'include'], text=True).splitlines()
    names = [n for n in headers if n.endswith('.h')] + ['src/'+n+'.c' for n in runtime]
    for variant in variants:
        directory = args.output/variant
        for name in names:
            data = subprocess.check_output(['git', 'show', manifest['baseline']+':'+name])
            if name == 'include/hs_graphics.h': data = transform(data.decode(), variant, args.experiment).encode()
            save(directory/name, data)
        flags = ['-O3', '-g', '-march=armv8-a+simd', '-mtune=cortex-a72', '-ffast-math', '-funroll-loops',
                 '-Wall', '-Wextra', '-Werror', '-DNDEBUG', '-I'+str(directory/'include'), '-I/usr/include/libdrm']
        command(['gcc', *flags, args.output/'harness.c', *[directory/'src'/(n+'.c') for n in runtime],
                 '-Wl,--wrap=eglSwapBuffers,--wrap=drmModeGetCrtc,--wrap=drmModeSetCrtc,--wrap=drmModePageFlip,--wrap=drmHandleEvent', '-lm', '-lGLESv2', '-lgbm', '-ldrm', '-lEGL', '-lpthread', '-o', directory/'benchmark'], 'build-'+variant)
        manifest.setdefault('binary_sha256', {})[str(directory/'benchmark')] = hashlib.sha256((directory/'benchmark').read_bytes()).hexdigest()
    for scene in ('single', 'retained'):
        for pair in range(1, args.pairs+1):
            if len(variants) == 2: order = variants if pair % 2 else tuple(reversed(variants))
            else:
                shift = (pair - 1) % len(variants); order = variants[shift:] + variants[:shift]
            for variant in order:
                label = f'{scene}-{pair}-{variant}'
                records = command(['sudo', '-n', args.output/variant/'benchmark', scene, 'on', args.frames, args.output/label, 'timed'], label)
                lifecycle = [r for r in records if r['record'] == 'lifecycle']; assert len(lifecycle) == 1
                resources = [r for r in records if r['record'] == 'shutdown_resources']; assert len(resources) == 1
                assert lifecycle[0]['context_released'] is True and resources[0]['fd_closed'] is True
                mode = next(r for r in records if r['record'] == 'mode_audit')
                assert mode['mode_failures'] == 0
                if variant == 'restore_crtc': assert mode['mode_sets'] == 2 and mode['matched_restores'] == 1
                flip = next(r for r in records if r['record'] == 'flip_audit')
                if variant == 'flip_restore':
                    assert 0 <= flip['completed'] <= flip['submitted'] <= flip['attempts'] <= 1
                    if flip['attempts']: assert flip['target'] == mode['saved_fb']
                    if flip['completed']: assert mode['mode_sets'] == 1 and mode['matched_restores'] == 0
                    else: assert mode['mode_sets'] == 2 and mode['matched_restores'] == 1
                manifest['runs'].append({'scene': scene, 'pair': pair, 'variant': variant, 'lifecycle': lifecycle[0], 'mode_audit': mode,
                    'flip_audit': flip, 'resources': resources[0], 'block': next(r for r in records if r['record'] == 'block')})
    manifest['completed'] = True
finally:
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
