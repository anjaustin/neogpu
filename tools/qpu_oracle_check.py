"""Check NeoGPU's QPU encoder (src/hs_qpu.c) word for word against an
independent assembler, Idein's py-videocore6 (GPL-2), used here as an oracle
only: it runs in its own venv and nothing from it enters NeoGPU.

Usage (make qpu-encode-check runs this):
  <venv python> tools/qpu_oracle_check.py NEODIR ORACLEDIR

NEODIR is what tests/test_qpu_encode wrote: corpus.txt, our programs, and
br_*.bin branch loops. ORACLEDIR is build/qpu from make hwprobe-progs (the
same programs assembled by the oracle). Exit status 1 on any mismatch.
"""
import os
import struct
import sys

from videocore6.assembler import *  # noqa: F401,F403  (oracle syntax)

FIELDS = [('mul_op', 58, 6), ('sig', 53, 5), ('cond', 46, 7), ('mm', 45, 1), ('ma', 44, 1), ('waddr_m', 38, 6),
          ('waddr_a', 32, 6), ('add_op', 24, 8), ('mul_b', 21, 3), ('mul_a', 18, 3), ('add_b', 15, 3),
          ('add_a', 12, 3), ('raddr_a', 6, 6), ('raddr_b', 0, 6)]


def diff(neo, ref):
    out = []
    for n, s, b in FIELDS:
        a, r = (neo >> s) & ((1 << b) - 1), (ref >> s) & ((1 << b) - 1)
        if a != r:
            out.append(f'{n} neo={a} ref={r}')
    return ', '.join(out)


def known_divergence(neo, ref):
    """fmov (mul op 14) of a small immediate (sig 15): the oracle encodes the
    float input modifier abs (mul_b 0) for any immediate, NeoGPU none (1).
    Measured on hardware: docs/benchmarks/QPU_ENCODER.md."""
    return ((neo ^ ref) == 1 << 21 and (neo >> 58) & 63 == 14 and (neo >> 53) & 31 == 15
            and (neo >> 21) & 7 == 1 and (ref >> 21) & 7 == 0)


# Exceptions that mean the corpus line itself is broken (a printer bug in
# tests/test_qpu_encode.c), not that the oracle judged the instruction
# illegal. They fail the check whatever NeoGPU did.
HARNESS_ERRORS = (SyntaxError, NameError, TypeError, AttributeError)

# NeoGPU refuses faddnf with identical inputs (HS_QPU_E_ORDER, -11): operand
# order is the only thing that tells faddnf from fadd, and identical inputs
# have one order, so the oracle silently emits fadd.
KNOWN_NEO_REJECT = '-11'


def oracle(src):
    @qpu
    def p(asm):
        exec(src)
    return assemble(p)


def words(path):
    with open(path, 'rb') as f:
        data = f.read()
    return list(struct.unpack(f'<{len(data) // 8}Q', data))


def main():
    neo, ref = sys.argv[1], sys.argv[2]
    bad = 0

    # 1. instruction corpus
    n_ok = n_rej_both = 0
    mism, neo_only_rej, oracle_rej, known, known_rej, harness = [], [], [], [], [], []
    with open(os.path.join(neo, 'corpus.txt')) as f:
        for line in f:
            head, src = line.rstrip('\n').split('\t', 1)
            try:
                w = oracle(src)
                assert len(w) == 1
                w = w[0]
            except HARNESS_ERRORS as e:
                harness.append((src, f'{type(e).__name__}: {e}'))
                continue
            except Exception as e:  # noqa: BLE001
                w = None
                err = f'{type(e).__name__}: {e}'
            if head.startswith('REJECT'):
                if w is None:
                    n_rej_both += 1
                elif head.split()[1] == KNOWN_NEO_REJECT:
                    known_rej.append(src)
                else:
                    neo_only_rej.append((src, head))
            elif w is None:
                oracle_rej.append((src, err))
            elif int(head, 16) == w:
                n_ok += 1
            elif known_divergence(int(head, 16), w):
                known.append((src, int(head, 16), w))
            else:
                mism.append((src, int(head, 16), w))
    print(f'corpus: {n_ok} identical, {len(mism)} different, {len(oracle_rej)} the oracle rejects, '
          f'{n_rej_both} rejected by both, {len(neo_only_rej)} rejected by NeoGPU only, '
          f'{len(known) + len(known_rej)} known divergences, {len(harness)} broken corpus lines')
    for src in known_rej:
        print(f'  KNOWN DIVERGENCE {src}: NeoGPU refuses faddnf with identical inputs; the oracle emits fadd')
    for src, err in harness:
        print(f'  BROKEN CORPUS LINE {src}: {err[:100]}')
    for src, a, r in known:
        print(f'  KNOWN DIVERGENCE {src}: neo {a:016x} ref {r:016x} (oracle sets fmov input modifier abs on '
              f'small immediates; NeoGPU encodes none, as every float op does)')
    for src, a, r in mism[:60]:
        print(f'  DIFF {src}\n       neo {a:016x} ref {r:016x}: {diff(a, r)}')
    for src, err in oracle_rej[:30]:
        print(f'  ORACLE REJECTS {src}: {err[:100]}')
    for src, head in neo_only_rej[:30]:
        print(f'  NEO ONLY REJECTS ({head}) {src}')
    bad += len(mism) + len(oracle_rej) + len(neo_only_rej) + len(harness)

    # 2. our programs, as assembled by the oracle in tools/hwprobe_qpu_progs.py
    progs = ['end', 'done', 'done_nowt', 'alu64', 'ld_thrsw', 'ld_delay', 'ld_tmuau'] + \
        [f'gather{q}_k{k}' for q in (1, 8) for k in (1, 2, 4, 8)]
    for name in progs:
        a, r = words(os.path.join(neo, name + '.bin')), words(os.path.join(ref, name + '.bin'))
        if a == r:
            print(f'program {name}: identical ({len(a)} instructions)')
        else:
            bad += 1
            print(f'program {name}: DIFFERENT (neo {len(a)}, oracle {len(r)} instructions)')
            for i, (x, y) in enumerate(zip(a, r)):
                if x != y:
                    print(f'  [{i}] neo {x:016x} ref {y:016x}: {diff(x, y)}')

    # 3. branch loops
    names = {0: 'always', 2: 'a0', 3: 'na0', 4: 'alla', 5: 'anyna', 6: 'anya', 7: 'allna'}
    nb = nbad = 0
    for c, cn in names.items():
        for n in (0, 1, 5, 300):
            for ub in range(3):
                tail = '' if ub == 0 else f'.unif_addr(absolute={ub == 2})'
                src = (f"with loop as l:\n    for _ in range({n}): nop()\n    l.b(cond='{cn}'){tail}\n"
                       f"    nop(); nop(); nop()")
                a, r = words(os.path.join(neo, f'br_{c}_{n}_{ub}.bin')), oracle(src)
                nb += 1
                if a != r:
                    nbad += 1
                    print(f'  BRANCH DIFF cond={cn} n={n} ub={ub}: neo {a[n]:016x} ref {r[n]:016x}')
            # forward: branch over its delay slots and n nops to a later label
            for ub in range(3):
                tail = '' if ub == 0 else f'.unif_addr(absolute={ub == 2})'
                src = (f"b(R.fwd, cond='{cn}'){tail}\nnop(); nop(); nop()\nfor _ in range({n}): nop()\n"
                       f"L.fwd\nnop()")
                a, r = words(os.path.join(neo, f'brf_{c}_{n}_{ub}.bin')), oracle(src)
                nb += 1
                if a != r:
                    nbad += 1
                    print(f'  FORWARD BRANCH DIFF cond={cn} n={n} ub={ub}: neo {a[0]:016x} ref {r[0]:016x}')
    print(f'branches: {nb - nbad} of {nb} programs identical (backward loops and forward jumps)')
    bad += nbad
    print('ALL IDENTICAL' if not bad else f'{bad} MISMATCHES')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
