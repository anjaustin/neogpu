"""Package the cost diagnostics into a compact, hash-verified archive.

Bulk raw files (verification RGBA frames, perf .data, built shims) stay on the
host; this archive keeps the manifests, reduced reports, per-call CSVs, traces,
sources, and a SHA-256 for every included file. Diagnostic tooling only.
"""
import hashlib
import tarfile
from pathlib import Path

root = Path('.').resolve()
assert (root/'tests').is_dir() and (root/'.git').is_dir()

DIRS = [
    'tests/evidence-cost-calls-2',
    'tests/evidence-cost-frames-2',
    'tests/evidence-cost-frames-3',   # failed partial run, preserved deliberately
    'tests/evidence-cost-frames-4',
    'tests/evidence-cost-profile-3',
    'tests/evidence-cost-modeset-1',
    'tests/evidence-cost-modeset-2',
    'tests/evidence-cost-modeset-3',
    'tests/evidence-cost-modeset-4',
    'tests/evidence-cost-sweep',
    'tests/evidence-e2e-regress-c54d140',
    'tests/evidence-e2e-regress-sweep',
]
DIRS += [str(p) for p in sorted(Path('tests').glob('evidence-cost-breakdown-*'))]
FILES = [
    'docs/audit/COST_BREAKDOWN_RESULTS.md',
    'docs/audit/GLES_COST_P0.md',
    'tests/gles_call_timer.c',
    'tests/test_gles_call_timer_run.py',
    'tests/test_gles_cost_calls_report.py',
    'tests/test_gles_frame_cost.c',
    'tests/test_gles_frame_cost_run.py',
    'tests/test_gles_cpu_profile.py',
    'tests/test_gles_cost_profile_report.py',
    'tests/test_gles_cost_trace_report.py',
    'tests/package_cost_catalog.py',
    'tests/test_gles_modeset_ab.py',
    'tests/test_gles_sweep_ab.py',
    'tests/panel_mode_probe.c',
]
SKIP_SUFFIXES = ('.verify.rgba', '.rgba', '.data', '.so')
SKIP_EXACT = {'tests/test_gles_frame_cost', 'tests/test_gles_frame_cost_profile',
              'tests/panel_mode_probe'}

selected = []
for directory in DIRS:
    for path in sorted(Path(directory).rglob('*')):
        if not path.is_file():
            continue
        if path.name.endswith(SKIP_SUFFIXES) or str(path) in SKIP_EXACT:
            continue
        selected.append(path)
for name in FILES:
    path = Path(name)
    assert path.is_file(), name
    selected.append(path)
selected = sorted(set(selected))
assert selected

sums = Path('tests/evidence-cost-catalog.SHA256SUMS')
sums.write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p}\n'
                        for p in selected))
archive = Path('tests/evidence-cost-catalog.tar.gz')
with tarfile.open(archive, 'w:gz') as tar:
    tar.add(sums, arcname=str(sums))
    for path in selected:
        tar.add(path, arcname=str(path))

print('included files:', len(selected) + 1)
print('archive bytes:', archive.stat().st_size)
print('archive sha256:', hashlib.sha256(archive.read_bytes()).hexdigest())
print('excluded suffixes:', ' '.join(SKIP_SUFFIXES))