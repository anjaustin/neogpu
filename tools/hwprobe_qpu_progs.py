"""Assemble QPU measurement programs to raw little-endian u64 files for
tools/hwprobe_qpu. Instrument only: uses Idein's py-videocore6 (GPL-2) from a
separate venv, and the .bin outputs (which include its example kernels) stay
out of the repository; nothing from it enters NeoGPU. Programs are generated
with code_offset 0; the harness places each at a 4 KiB-aligned offset, which
keeps the examples' alignment conditions (pos % 512) true.

Every program here follows the two measured QPU rules (docs/benchmarks/
HARDWARE_MAP_2026-11-15.md): a batch of TMU loads ends with a thread switch
before ldtmu, and a program never ends with TMU writes in flight. The
exceptions are deliberate evidence: memset*_ref (unmodified, reproduces the
hang), done_nowt, and ld_delay / ld_tmuau (the load forms that hang).

Usage: VC6_EXAMPLES=<py-videocore6>/examples <venv python> hwprobe_qpu_progs.py OUTDIR
(make hwprobe-progs does this; VC6_EXAMPLES defaults to
~/src-ref/py-videocore6/examples)"""
import os, sys, struct
sys.path.insert(0, os.environ.get('VC6_EXAMPLES', os.path.expanduser('~/src-ref/py-videocore6/examples')))
from videocore6.assembler import *  # noqa
from summation import qpu_summation
from memset import qpu_memset
from scopy import qpu_scopy


def end_seq():
    nop(sig=thrsw); nop(sig=thrsw); nop(); nop(); nop(sig=thrsw); nop(); nop(); nop()


@qpu
def qpu_end(asm):
    end_seq()


@qpu
def qpu_done(asm):
    # unif0 = address; every lane writes 1 there, then waits for the write.
    nop(sig=ldunifrf(rf0))
    mov(tmud, 1)
    mov(tmua, rf0)
    tmuwt()
    end_seq()


@qpu
def qpu_done_nowt(asm):
    # as done, without the tmuwt: the shader ends with its write in flight.
    nop(sig=ldunifrf(rf0))
    mov(tmud, 1)
    mov(tmua, rf0)
    end_seq()


@qpu
def snip_barrier(asm):
    barrierid(syncb, sig=thrsw)
    nop(); nop()


@qpu
def snip_tmuwt(asm):
    tmuwt()


def before_end(words, snippet):
    # splice a snippet in front of the 8-instruction end sequence
    return words[:-8] + snippet + words[-8:]


@qpu
def qpu_alu(asm, *, inner):
    # unif0 = loop count. Each body instruction is a dual issue: one fadd on
    # the add ALU and one fmul on the mul ALU, 16 lanes each, accumulators
    # only (no register-file read ports).
    nop(sig=ldunifrf(rf0))
    mov(r0, 1.0)
    mov(r1, 1.0)
    mov(r2, 1.0)
    mov(r3, 1.0)
    mov(r4, 1)
    with loop as l:
        for i in range(inner):
            fadd(r0, r0, r1).fmul(r2, r2, r3)
        sub(rf0, rf0, r4, cond='pushz')
        l.b(cond='na0')
        nop(); nop(); nop()
    end_seq()


@qpu
def qpu_gather(asm, *, num_qpus, k):
    # out[e] = table[idx[e]] for 4-byte elements. Uniforms: passes, idx base,
    # table base, out base. Twin of hs_qpu_kernel_gather (src/hs_qpu_kernels.c),
    # instruction for instruction. In pass q, batch b's load i reads
    #   base + (q * num_qpus + b) * 64k + 64i + 4 * lane,
    # so each batch owns a contiguous 64k-byte run per pass (the fix for the
    # k = 3..5 slowdown; see QPU_ENCODER.md).
    # Measured rule (ld_* programs): a batch of TMU loads must end with a
    # thread-switch signal before its results are read, or the GPU hangs.
    nop(sig=ldunifrf(rf0))
    nop(sig=ldunifrf(rf1))
    nop(sig=ldunifrf(rf2))
    nop(sig=ldunifrf(rf3))
    if num_qpus == 1:
        mov(rf5, 0)
    else:
        tidx(r0)
        shr(r0, r0, 2)
        band(rf5, r0, 0b1111)
    shl(r0, rf5, 6)
    nop().umul24(r0, r0, k)
    eidx(r1)
    shl(r1, r1, 2)
    add(r0, r0, r1)
    add(rf1, rf1, r0)
    add(rf3, rf3, r0)
    mov(r0, 1)
    shl(rf4, r0, 6)
    mov(r0, k)
    nop().umul24(r0, r0, num_qpus - 1)
    shl(rf6, r0, 6)
    # guard: passes == 0 jumps to the end (the loop tests after its body)
    mov(null, rf0, cond='pushz')
    b(R.done, cond='a0')
    nop()
    nop()
    nop()
    with loop as l:
        for i in range(k):
            if i == k - 1:
                mov(tmua, rf1, sig=thrsw).add(rf1, rf1, rf4)
            else:
                mov(tmua, rf1).add(rf1, rf1, rf4)
        nop()
        nop()
        for i in range(k):
            nop(sig=ldtmu(r0))
            shl(r0, r0, 2)
            if i == k - 1:
                add(tmua, r0, rf2, sig=thrsw)
            else:
                add(tmua, r0, rf2)
        nop()
        nop()
        for i in range(k):
            nop(sig=ldtmu(r1))
            mov(tmud, r1)
            mov(tmua, rf3).add(rf3, rf3, rf4)
        add(rf1, rf1, rf6)
        add(rf3, rf3, rf6)
        sub(rf0, rf0, 1, cond='pushz')
        l.b(cond='na0')
        nop()
        nop()
        nop()
    L.done
    tmuwt()  # never end with TMU writes in flight (see FINDINGS.md)
    end_seq()


@qpu
def qpu_ldtest(asm, *, form):
    # 1 QPU, 1 pass: out[lane] = idx[lane]. Uniforms: passes(unused), idx, table, out
    # [, tmu config, branch rewind]. Tests which load form the TMU accepts.
    nop(sig=ldunifrf(rf0))
    nop(sig=ldunifrf(rf1))
    nop(sig=ldunifrf(rf2))
    nop(sig=ldunifrf(rf3))
    eidx(r0)
    shl(r0, r0, 2)
    add(rf1, rf1, r0)
    add(rf3, rf3, r0)
    if form == 'thrsw':
        mov(tmua, rf1, sig=thrsw)
        nop()
        nop()
        nop(sig=ldtmu(r0))
    elif form == 'delay':
        mov(tmua, rf1)
        nop()
        nop()
        nop(sig=ldtmu(r0))
    elif form == 'tmuau':
        nop(sig=thrsw)
        nop()
        nop()
        mov(tmuau, rf1)
        nop(sig=ldtmu(r0))
    mov(tmud, r0)
    mov(tmua, rf3)
    tmuwt()
    end_seq()


out = sys.argv[1]
progs = {
    'end': assemble(qpu_end),
    'done': assemble(qpu_done),
    'alu64': assemble(qpu_alu, inner=64),
    'sum8': assemble(qpu_summation, num_qpus=8, unroll_shift=2, code_offset=0),
    'sum1': assemble(qpu_summation, num_qpus=1, unroll_shift=2, code_offset=0),
    # memset: the reference kernel ends with TMU writes in flight, which makes
    # the next job's tmuwt hang the GPU. The default adds tmuwt (same
    # bandwidth, measured); _barrier adds the syncb barrier scopy uses; _ref
    # is unmodified and reproduces the hang.
    **{f'memset{nq}{suf}': (before_end(assemble(qpu_memset, num_qpus=nq, unroll_shift=5, code_offset=0), snip)
                            if snip else assemble(qpu_memset, num_qpus=nq, unroll_shift=5, code_offset=0))
       for nq in (1, 8)
       for suf, snip in (('', assemble(snip_tmuwt)), ('_barrier', assemble(snip_barrier)), ('_ref', None))},
    'done_nowt': assemble(qpu_done_nowt),
    **{f'gather{nq}_k{k}': assemble(qpu_gather, num_qpus=nq, k=k) for nq in (1, 8) for k in (1, 2, 4, 8)},
    # TMU load forms: only ld_thrsw completes; the other two hang the GPU
    'ld_thrsw': assemble(qpu_ldtest, form='thrsw'),
    'ld_delay': assemble(qpu_ldtest, form='delay'),
    'ld_tmuau': assemble(qpu_ldtest, form='tmuau'),
    'scopy8': assemble(qpu_scopy, num_qpus=8, unroll_shift=0, code_offset=0),
    'scopy1': assemble(qpu_scopy, num_qpus=1, unroll_shift=0, code_offset=0),
}
for name, words in progs.items():
    with open(f'{out}/{name}.bin', 'wb') as f:
        f.write(struct.pack(f'<{len(words)}Q', *words))
    print(f'{name}: {len(words)} instructions')
