"""Qualify and measure frozen-baseline fabric candidates in diagnostic copies."""
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
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--frames', type=int, default=200000)
parser.add_argument('--candidate', choices=('hint', 'gate', 'gate-hint', 'split'), default='split')
parser.add_argument('--baseline', default='2f72c91', help='git revision for frozen baseline source and held-lock fixture')
parser.add_argument('--reference', type=Path, default=Path('tests/evidence-comparison-final/reference'))
args = parser.parse_args()
assert args.frames > 0
assert args.output.name.startswith('evidence-') and args.output.resolve().is_relative_to(ROOT/'tests')
args.output.mkdir(exist_ok=False)
(args.output/'driver.py').write_bytes(Path(__file__).read_bytes())
manifest = {'completed': False, 'commands': [], 'runs': [], 'sources': {}}
manifest['candidate'] = args.candidate
manifest['revision'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
manifest['baseline_revision'] = subprocess.check_output(['git', 'rev-parse', args.baseline], text=True).strip()
manifest['common_source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [Path(__file__), *Path('include').glob('*.h'), Path('tests/comparison_native.c'),
              Path('tests/comparison_rigtorp.cc'), Path('tests/test_comparison_transport.c'), Path('tests/test_fabric.c')]}


def environment():
    files = [*Path('/sys/devices/system/cpu').glob('cpu*/cpufreq/scaling_cur_freq'),
             *Path('/sys/devices/system/cpu').glob('cpu*/cpufreq/scaling_governor'),
             *Path('/sys/class/thermal').glob('thermal_zone*/temp')]
    result = {'wall_ns': time.time_ns(), 'affinity': sorted(os.sched_getaffinity(0)), 'sysfs': {}}
    for path in files:
        try: result['sysfs'][str(path)] = path.read_text().strip()
        except OSError as error: result['sysfs'][str(path)] = str(error)
    return result


def run(argv, label, env=None):
    before = environment()
    result = subprocess.run(list(map(str, argv)), capture_output=True, text=True, timeout=180, env=env)
    record = {'argv': list(map(str, argv)), 'status': result.returncode,
              'stdout': result.stdout, 'stderr': result.stderr,
              'environment_before': before, 'environment_after': environment()}
    (args.output/(label+'.json')).write_text(json.dumps(record, indent=2))
    manifest['commands'].append({'label': label, 'status': result.returncode})
    assert result.returncode == 0, record
    return result.stdout


def edit(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


try:
    original = {name: Path(name).read_text() for name in ('src/hs_fabric.c', 'include/hs_fabric.h')}
    baseline = {name: subprocess.check_output(['git', 'show', manifest['baseline_revision']+':'+name], text=True) for name in original}
    source = baseline['src/hs_fabric.c']
    header = edit(baseline['include/hs_fabric.h'], '    bool active, closing;',
                  '    /* Atomic hint only; all authoritative state remains mutex-protected. */\n'
                  '    u32 work_hint;\n    bool active, closing;')
    source = edit(source, 'if (result == HS_FABRIC_OK) slot->closing = true;',
                  'if (result == HS_FABRIC_OK) {\n        slot->closing = true;\n'
                  '        __atomic_store_n(&slot->work_hint, 1, __ATOMIC_RELEASE);\n    }')
    source = edit(source, '            slot->admitted = frame->sequence;',
                  '            slot->admitted = frame->sequence;\n'
                  '            __atomic_store_n(&slot->work_hint, 1, __ATOMIC_RELEASE);')
    source = edit(source, '        HSParticipantSlot* slot = &fabric->slots[i];\n        if (pthread_mutex_trylock(&slot->mutex) != 0) continue;',
                  '        HSParticipantSlot* slot = &fabric->slots[i];\n'
                  '        if (!__atomic_load_n(&slot->work_hint, __ATOMIC_ACQUIRE)) continue;\n'
                  '        if (pthread_mutex_trylock(&slot->mutex) != 0) continue;')
    source = edit(source, '        pthread_mutex_unlock(&slot->mutex);\n    }\n    return events;',
                  '        /* Same mutex as publishers: clearing cannot lose a publication. */\n'
                  '        __atomic_store_n(&slot->work_hint,\n'
                  '                         slot->active && (slot->count || slot->closing), __ATOMIC_RELEASE);\n'
                  '        pthread_mutex_unlock(&slot->mutex);\n    }\n    return events;')
    if args.candidate in ('gate', 'gate-hint'):
        header = edit(header if args.candidate == 'gate-hint' else baseline['include/hs_fabric.h'], '    pthread_mutex_t mutex;',
                      '    /* Bounded atomic try-lock; no blocking or retry loop. */\n    u32 gate;')
        source = source if args.candidate == 'gate-hint' else baseline['src/hs_fabric.c']
        begin = source.index('bool hs_fabric_init(')
        end = source.index('HSFabricResult hs_fabric_join(', begin)
        source = source[:begin] + '''_Static_assert(__atomic_always_lock_free(sizeof(u32), 0), "Slot gate must be lock-free");

static int slot_trylock(HSParticipantSlot* slot) {
    if (__atomic_load_n(&slot->gate, __ATOMIC_RELAXED)) return 1;
    u32 expected = 0;
    return !__atomic_compare_exchange_n(&slot->gate, &expected, 1, false,
                                        __ATOMIC_ACQUIRE, __ATOMIC_RELAXED);
}

static void slot_unlock(HSParticipantSlot* slot) {
    __atomic_store_n(&slot->gate, 0, __ATOMIC_RELEASE);
}

bool hs_fabric_init(HSFabric* fabric) {
    if (!fabric) return false;
    memset(fabric, 0, sizeof(*fabric));
    return true;
}

void hs_fabric_destroy(HSFabric* fabric) { (void)fabric; }

''' + source[end:]
        source = source.replace('pthread_mutex_trylock(&slot->mutex)', 'slot_trylock(slot)')
        source = source.replace('pthread_mutex_unlock(&slot->mutex)', 'slot_unlock(slot)')
    if args.candidate == 'split':
        header = edit(baseline['include/hs_fabric.h'], '    u32 head, count;',
                      '    /* Monotonic modulo-2^32 ring cursors; never reset at join. */\n    u32 read_index, write_index;')
        header = edit(header, '#define HS_FABRIC_QUEUE 4', '#define HS_FABRIC_QUEUE 4\n#define HS_FABRIC_SPLIT_PUBLICATION 1')
        source = baseline['src/hs_fabric.c']
        source = edit(source, '            slot->closing = false;',
                      '            __atomic_store_n(&slot->closing, false, __ATOMIC_RELEASE);')
        source = edit(source, '            slot->head = slot->count = 0;\n', '')
        source = edit(source, 'return slot->closing ? HS_FABRIC_CLOSING : HS_FABRIC_OK;',
                      'return __atomic_load_n(&slot->closing, __ATOMIC_RELAXED) ? HS_FABRIC_CLOSING : HS_FABRIC_OK;')
        source = edit(source, 'if (result == HS_FABRIC_OK) slot->closing = true;',
                      'if (result == HS_FABRIC_OK) __atomic_store_n(&slot->closing, true, __ATOMIC_RELEASE);')
        source = edit(source, '    if (result == HS_FABRIC_OK) {\n        if (slot->admitted',
                      '    if (result == HS_FABRIC_OK) {\n'
                      '        u32 read = __atomic_load_n(&slot->read_index, __ATOMIC_ACQUIRE);\n'
                      '        u32 write = __atomic_load_n(&slot->write_index, __ATOMIC_RELAXED);\n        if (slot->admitted')
        source = edit(source, 'else if (slot->count == HS_FABRIC_QUEUE)', 'else if ((u32)(write - read) >= HS_FABRIC_QUEUE)')
        source = edit(source, '&slot->queue[(slot->head + slot->count) % HS_FABRIC_QUEUE]', '&slot->queue[write % HS_FABRIC_QUEUE]')
        source = edit(source, '            slot->count++;\n', '')
        source = edit(source, '            slot->admitted = frame->sequence;',
                      '            slot->admitted = frame->sequence;\n'
                      '            __atomic_store_n(&slot->write_index, write + 1, __ATOMIC_RELEASE);')
        begin = source.index('u32 hs_fabric_step(')
        end = source.index('bool hs_fabric_compose(', begin)
        source = source[:begin] + '''/* Cursor wrap preserves ring positions because capacity divides 2^32. */
_Static_assert((HS_FABRIC_QUEUE & (HS_FABRIC_QUEUE - 1)) == 0, "Ring capacity must be a power of two");
_Static_assert(__atomic_always_lock_free(sizeof(u32), 0), "Ring cursors must be lock-free");

u32 hs_fabric_step(HSFabric* fabric, u32 budget) {
    if (!fabric || !budget) return 0;
    u32 events = 0;
    for (u32 visited = 0; visited < HS_FABRIC_PARTICIPANTS && events < budget; visited++) {
        u32 i = fabric->cursor;
        fabric->cursor = (i + 1) % HS_FABRIC_PARTICIPANTS;
        HSParticipantSlot* slot = &fabric->slots[i];
        u32 read = __atomic_load_n(&slot->read_index, __ATOMIC_RELAXED);
        u32 write = __atomic_load_n(&slot->write_index, __ATOMIC_ACQUIRE);
        if (read != write) {
            HSParticipantFrame* source = &slot->queue[read % HS_FABRIC_QUEUE];
            HSParticipantView* view = &fabric->views[i];
            /* Generation is published before write_index and cannot change
             * until this same consumer retires the drained generation. */
            view->owner = (HSParticipant){i, slot->generation};
            view->active = true;
            view->frame.sequence = source->sequence;
            view->frame.count = source->count;
            memcpy(view->frame.draws, source->draws, source->count * sizeof(HSRenderCmd));
            __atomic_store_n(&slot->read_index, read + 1, __ATOMIC_RELEASE);
            fabric->applied++;
            events++;
        } else if (__atomic_load_n(&slot->closing, __ATOMIC_ACQUIRE)) {
            if (pthread_mutex_trylock(&slot->mutex) != 0) continue;
            /* Recheck after excluding submit/leave/join. Closing rejects new
             * admissions, and retirement never overtakes accepted frames. */
            if (slot->active && __atomic_load_n(&slot->closing, __ATOMIC_RELAXED) &&
                __atomic_load_n(&slot->read_index, __ATOMIC_RELAXED) ==
                __atomic_load_n(&slot->write_index, __ATOMIC_ACQUIRE)) {
                fabric->views[i].active = false;
                slot->active = false;
                __atomic_store_n(&slot->closing, false, __ATOMIC_RELEASE);
                fabric->retired++;
                events++;
            }
            pthread_mutex_unlock(&slot->mutex);
        }
    }
    return events;
}

''' + source[end:]
    flags = ['-O3', '-g', '-march=armv8-a+simd', '-mtune=cortex-a72', '-ffast-math',
             '-funroll-loops', '-Wall', '-Wextra', '-Werror', '-DNDEBUG']
    reference = args.reference/'include'
    assert hashlib.sha256((reference/'rigtorp/SPSCQueue.h').read_bytes()).hexdigest() == '62d9f783862666be5c7ca3c8fbcec2e45d9920167c4b9e689bda4957f66dd86e'
    assert hashlib.sha256((args.reference/'LICENSE').read_bytes()).hexdigest() == '21d7e737765bc0c986ee78f16a3e25ee3cb5ad424feece274b64a10654c31912'
    for variant in ('baseline', 'candidate'):
        directory = args.output/variant
        directory.mkdir()
        for name, text in [('hs_fabric.c', source if variant == 'candidate' else baseline['src/hs_fabric.c']),
                           ('hs_fabric.h', header if variant == 'candidate' else baseline['include/hs_fabric.h'])]:
            (directory/name).write_text(text)
            manifest['sources'][str(directory/name)] = hashlib.sha256(text.encode()).hexdigest()
        includes = ['-I'+str(directory), '-Iinclude', '-Itests']
        test = directory/'test_fabric.c'
        text = subprocess.check_output(['git', 'show', manifest['baseline_revision']+':tests/test_fabric.c'], text=True)
        if variant == 'candidate' and args.candidate in ('gate', 'gate-hint'):
            text = edit(text, 'pthread_mutex_lock(&fabric.slots[0].mutex) == 0',
                        '__atomic_exchange_n(&fabric.slots[0].gate, 1, __ATOMIC_ACQUIRE) == 0')
            text = edit(text, 'pthread_mutex_unlock(&fabric.slots[0].mutex) == 0',
                        '(__atomic_store_n(&fabric.slots[0].gate, 0, __ATOMIC_RELEASE), true)')
        test.write_text(text)
        run(['gcc', *flags, *includes, test, directory/'hs_fabric.c', '-lpthread', '-o', directory/'fabric'], variant+'-build-tests')
        run([directory/'fabric'], variant+'-tests')
        targeted = [Path('tests/test_bedrock_hint.c')]
        if variant == 'candidate' and args.candidate == 'split': targeted.append(Path('tests/test_bedrock_split.c'))
        for probe in targeted:
            label = variant+'-'+probe.stem
            run(['gcc', *flags, *includes, probe, directory/'hs_fabric.c', '-lpthread', '-o', directory/probe.stem], label+'-build')
            run([directory/probe.stem], label)
        if variant == 'candidate':
            for sanitizer in ('address,undefined', 'thread'):
                label = 'candidate-'+sanitizer.replace(',', '-')
                run(['gcc', *flags, '-O1', '-fno-omit-frame-pointer', '-fsanitize='+sanitizer, *includes,
                     test, directory/'hs_fabric.c', '-lpthread', '-o', directory/label], label+'-build')
                env = {**os.environ, 'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1',
                       'UBSAN_OPTIONS': 'halt_on_error=1', 'TSAN_OPTIONS': 'halt_on_error=1'}
                run([directory/label], label, env)
                for probe in targeted:
                    probe_label = label+'-'+probe.stem
                    run(['gcc', *flags, '-O1', '-fno-omit-frame-pointer', '-fsanitize='+sanitizer, *includes,
                         probe, directory/'hs_fabric.c', '-lpthread', '-o', directory/probe_label], probe_label+'-build')
                    run([directory/probe_label], probe_label, env)
        objects = []
        for name in [str(directory/'hs_fabric.c'), 'tests/comparison_native.c', 'tests/comparison_rigtorp.cc', 'tests/test_comparison_transport.c']:
            cpp = name.endswith('.cc')
            obj = directory/(Path(name).stem+'.o')
            run(['g++' if cpp else 'gcc', *flags, *includes,
                 *(['-std=c++23', '-isystem', reference] if cpp else []), '-c', name, '-o', obj], variant+'-compile-'+obj.stem)
            objects.append(obj)
        run(['g++', *objects, '-lpthread', '-o', directory/'comparison'], variant+'-link')
        run([directory/'comparison', '--selftest'], variant+'-selftest')
    for producers in (1, 2, 8):
        for draws in (1, 64):
            for pair in range(1, 4):
                order = [('baseline', 'neogpu'), ('candidate', 'neogpu'), ('baseline', 'rigtorp')]
                if pair % 2 == 0: order.reverse()
                for variant, impl in order:
                    label = f'run-{producers}-{draws}-{pair}-{variant}-{impl}'
                    output = run([args.output/variant/'comparison', impl, producers, args.frames, draws], label)
                    records = [json.loads(line) for line in output.splitlines()]
                    summary = next(r for r in records if r['record'] == 'summary')
                    assert summary['accepted'] == summary['applied'] == producers*args.frames
                    summary.update(variant=variant, implementation=impl, pair=pair,
                                   fps=summary['applied']*1e9/summary['elapsed_ns'])
                    manifest['runs'].append(summary)
    assert all(Path(name).read_text() == text for name, text in original.items())
    manifest['completed'] = True
finally:
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
