#!/usr/bin/env python3
"""Mutation tests for the P0-5 red team: break hs_mailbox (or the batch number
in the gather kernel) one way at a time and check that a test notices.

Usage (repo root, performance governor set by the caller):
    python3 tests/redteam-p0-5/mutants.py <evidence.txt> [name ...]
Each mutant is a one-place source edit, built into
tests/redteam-p0-5/mutants/<name>/ and run under the GPU lock. "caught" means
the test command exited non-zero. m0 is the unmodified control and must pass.
Mutants are chosen so that the broken program stays harmless on the GPU (no
unbounded loops, no out-of-range addresses).
"""
import os, subprocess, sys

ROOT = "tests/redteam-p0-5/mutants"
RT = "cases 1"
RT8 = "cases 8"
BESIDE = "beside"  # run while tests/test_08_raycast_orbit renders

MUTANTS = [
    ("m0_control", None, None, None, RT, "no change: must pass"),
    ("m1_no_gather_size_limit", "src/hs_mailbox.c",
     "a[4] > 1 || a[6] || n > HS_MB_MAX_GATHER_CHUNKS", "a[4] > 1 || a[6]",
     RT, "send accepts a gather over the size limit"),
    ("m2_no_chain_unit_limit", "src/hs_mailbox.c",
     "a[4] > 16384 || n * ((uint64_t)a[4] + 4) + 4 > MAX_UNITS", "a[4] > 16384",
     RT, "send accepts a chain over the unit limit"),
    ("m3_no_overlap_check", "src/hs_mailbox.c",
     "        return !overlap(a[1], bytes, a[3], bytes);\n    }\n    return 0; /* HS_MB_OP_STOP",
     "        return 1;\n    }\n    return 0; /* HS_MB_OP_STOP",
     RT, "send accepts a chain whose output overlaps its input"),
    ("m4_wait_unsent", "src/hs_mailbox.c",
     "if (!mb->dev || (int32_t)(seq - mb->seq) > 0) return -EINVAL; /* never sent */", "if (!mb->dev) return -EINVAL;",
     RT, "wait on a never-sent sequence spins to its timeout"),
    ("m5_unsigned_ack_compare", "src/hs_mailbox.c",
     "acked += (int32_t)(ack_of(mb, b) - seq) >= 0;", "acked += ack_of(mb, b) >= seq;",
     RT, "acknowledgement compare breaks at the 32-bit wrap"),
    ("m6_no_lost_lease_resync", "src/hs_mailbox.c",
     "if (mb->submitted != d) mb->submitted = d, mb->lost_leases++;", "(void)d;",
     RT, "killed leases are never replaced"),
    ("m7_no_idle_stop", "src/hs_mailbox.c",
     "int idle = t - t_act > mb->idle_ns;", "int idle = 0 * (t - t_act > mb->idle_ns);",
     RT, "leases keep running after the caller goes quiet"),
    ("m18_no_age_yield", "src/hs_mailbox.c",
     "if (!aging && !idle && t - t_change > mb->max_lease_ns) {", "if (0 && !aging && !idle && t - t_change > mb->max_lease_ns) {",
     RT, "no lease time bound"),
    ("m8_no_lease_end_at_ring_wrap", "src/hs_mailbox.c",
     "    hs_qpu_branch(p, HS_QPU_B_A0, done, 0, 0); /* the ring wrapped: end the lease */",
     "    nop(p); /* mutant */",
     RT, "a lease re-reads a slot it already read (stale request)"),
    ("m9_no_idx_reload", "src/hs_mailbox.c",
     "    hs_qpu_label_bind(p, l);\n    op2(p, RF(33));\n    budget_sub(p, I(1));\n    load(p, RF(33), R(0));",
     "    hs_qpu_label_bind(p, l);\n    budget_sub(p, I(1));\n    load(p, RF(33), R(0));",
     RT, "gather indices are not reloaded: stale indices"),
    ("m10_no_result_publish", "src/hs_mailbox.c",
     "    mov(p, HS_QPU_TMUAU, RF(39));\n    op(p, HS_QPU_A_ADD, RF(39), RF(39), RF(22));",
     "    op(p, HS_QPU_A_ADD, RF(39), RF(39), RF(22));",
     RT, "results are not published before the acknowledgement"),
    ("m11_mailbox_batch_from_tidx", "src/hs_mailbox.c",
     "    mov(p, RF(9), RF(0)); /* the hardware's batch number (workgroup x), before rf0 is overwritten */",
     "    op(p, HS_QPU_A_TIDX, R(2), X, X);\n    op(p, HS_QPU_A_SHR, RF(9), R(2), I(2));",
     BESIDE + " busy 8 6 100 1", "mailbox batch number from the thread index, renderer running"),
    ("m12_kernel_batch_from_tidx", "src/hs_qpu_kernels.c",
     "    if (batches != 1) emit(p, hs_qpu_a(HS_QPU_A_BAND, RF(5), RF(0), I(15)));\n    for (int i = 0; i < 4; i++) ldunifrf(p, i);",
     "    for (int i = 0; i < 4; i++) ldunifrf(p, i);\n    if (batches != 1) {\n        emit(p, hs_qpu_a(HS_QPU_A_TIDX, R(0), X, X));\n"
     "        emit(p, hs_qpu_a(HS_QPU_A_SHR, R(0), R(0), I(2)));\n        emit(p, hs_qpu_a(HS_QPU_A_BAND, RF(5), R(0), I(15)));\n    }",
     BESIDE + " jobcheck 8 6", "gather kernel batch number from the thread index, renderer running"),
    ("m14_stop_resent", "src/hs_mailbox.c",
     "        if (!mb->stop_sent) { /* a stop request is sent once: after it, leases end without polling */",
     "        if (1) {",
     RT, "a retried stop sends a second stop request that no lease will serve"),
    ("m15_unsigned_lease_count", "src/hs_mailbox.c",
     "if ((int32_t)(mb->submitted - hs_mailbox_leases_done(mb, 0)) >= (int32_t)mb->qdepth) return 0;",
     "if (mb->submitted - hs_mailbox_leases_done(mb, 0) >= mb->qdepth) return 0;",
     RT8, "lease counts that differ between batches stop all submission (8 batches; 1 batch recovers by resync)"),
    ("m16_table_first_word_only", "src/hs_mailbox.c",
     "!overlap(a[2], tbytes, a[3], bytes);", "!overlap(a[2], 4, a[3], bytes);",
     RT, "an output inside the table is accepted"),
    ("m17_age_yield_soft", "src/hs_mailbox.c",
     "    br(p, HS_QPU_B_ANYNA, done, 0);\n", "    nop(p); /* mutant */\n",
     "pipe 1 6", "the age yield does not end a lease that always has requests pending"),
    ("m19_per_batch_slot_cap", "src/hs_mailbox.c",
     ["    load(p, RF(16), RF(14));\n    load(p, RF(6), R(1));\n",
      "    hs_qpu_branch(p, HS_QPU_B_A0, done, 0, 0); /* the ring wrapped: end the lease */"],
     ["    load(p, RF(16), RF(14));\n    load(p, RF(6), R(1));\n    op(p, HS_QPU_A_ADD, RF(10), RF(19), I(1));\n",
      "    flags(p, HS_QPU_A_SUB, RF(10), RF(10), I(1), HS_QPU_PF_PUSHZ);\n    hs_qpu_branch(p, HS_QPU_B_A0, done, 0, 0);"],
     RT8, "the earlier rule (a lease takes `slots` requests per batch): a slot another batch cached is run stale"),
    ("m13_no_stall_yield", "src/hs_mailbox.c",
     "else if (!mb->yielding && t - t_partial > STALL_NS) {", "else if (0) {",
     BESIDE + " busy 8 6 100 0", "informational: 8 batches beside a renderer without the stall yield (compare p99)"),
]

OBJS = ["src/hs_mailbox", "src/hs_qpu", "src/hs_qpu_kernels", "src/hs_v3d"]


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)


def main():
    out = open(sys.argv[1], "a")
    want = set(sys.argv[2:])

    def log(line):
        print(line, flush=True)
        out.write(line + "\n")
        out.flush()

    caught = missed = 0
    for name, path, old, new, cmd, what in MUTANTS:
        if want and name not in want:
            continue
        d = f"{ROOT}/{name}"
        os.makedirs(d, exist_ok=True)
        objs = []
        for o in OBJS:
            src = o + ".c"
            if src == path:
                text = open(src).read()
                edits = list(zip(old, new)) if isinstance(old, list) else [(old, new)]
                missing = [o for o, _ in edits if text.count(o) != 1]
                if missing:
                    log(f"{name}: EDIT DID NOT APPLY ({len(missing)} of {len(edits)} edits)")
                    objs = None
                    break
                for o, n_ in edits:
                    text = text.replace(o, n_)
                open(f"{d}/mutant.c", "w").write(text)
                r = sh(f"gcc -O2 -Wall -Wextra -Iinclude -c {d}/mutant.c -o {d}/mutant.o")
                if r.returncode:
                    log(f"{name}: BUILD FAILED {r.stderr[:300]}")
                    objs = None
                    break
                objs.append(f"{d}/mutant.o")
            else:
                objs.append(o + ".o")
        if objs is None:
            missed += 1
            continue
        r = sh(f"gcc -O2 -Wall -Wextra -ffp-contract=off -Iinclude tests/redteam-p0-5/rt_mailbox.c {' '.join(objs)} -lpthread -o {d}/rt_mailbox")
        if r.returncode:
            log(f"{name}: LINK FAILED {r.stderr[:300]}")
            missed += 1
            continue
        if cmd.startswith(BESIDE):
            run = f"sudo flock build/gpu.lock tests/redteam-p0-5/coexist.sh {name} {d}/rt_mailbox {cmd[len(BESIDE) + 1:]}"
        else:
            run = f"sudo flock build/gpu.lock timeout 600 {d}/rt_mailbox {cmd}"
        r = sh(run)
        lines = r.stdout.strip().split("\n")
        failed_cases = [l[:300] for l in lines if '"ok":false' in l]
        rc_line = [l for l in lines if l.startswith("# ") and " rc=" in l]
        rc = int(rc_line[0].rsplit("rc=", 1)[1]) if rc_line else r.returncode
        is_caught = rc != 0
        if name == "m0_control":
            verdict = "control passes" if not is_caught else "CONTROL FAILED"
            missed += is_caught
        elif "informational" in what:
            verdict = "informational"
        else:
            verdict = "caught" if is_caught else "NOT CAUGHT"
            caught += is_caught
            missed += not is_caught
        log(f"{name}: {verdict} (rc={rc}) [{what}] cmd=[{cmd}]")
        for l in failed_cases[:4]:
            log(f"    {l}")
        if cmd.startswith(BESIDE):
            for l in lines:
                if '"busy"' in l or '"jobcheck"' in l or "renderer:" in l:
                    log(f"    {l[:330]}")
    log(f"mutants caught: {caught}; problems (not caught / build / control): {missed}")
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
