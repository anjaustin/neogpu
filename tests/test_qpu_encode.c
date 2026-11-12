/* test_qpu_encode: NeoGPU's QPU encoder (src/hs_qpu.c) against the programs
 * we run on the GPU and against an independent assembler.
 *
 *   tests/test_qpu_encode OUTDIR
 *
 * 1. Rebuilds every QPU program NeoGPU owns (tools/hwprobe_qpu_progs.py has the
 *    reference-assembler versions) into OUTDIR/<name>.bin.
 * 2. Checks the default rule enforcement: the two load forms that hung the
 *    GPU and a store left in flight must be rejected, the measured-good forms
 *    accepted.
 * 3. Writes OUTDIR/corpus.txt: one instruction per line, "HEX<TAB>source"
 *    with the source in the reference assembler's syntax, and branch loop
 *    programs OUTDIR/br_<cond>_<n>.bin.
 * tools/qpu_oracle_check.py compares 1 and 3 with the reference assembler.
 * Exit status is nonzero if any step here fails. */
#include "hs_qpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define R(n) HS_QPU_R(n)
#define RF(n) HS_QPU_RF(n)
#define X HS_QPU_NOREG
#define I(v) hs_qpu_imm_int(v)

static const char* outdir;
static int failures;

static void check(int cond, const char* what) {
    printf("%s %s\n", cond ? "ok  " : "FAIL", what);
    if (!cond) failures++;
}

static void emit(hs_qpu_prog* p, hs_qpu_alu x) { hs_qpu_emit(p, &x); }

static int save(const char* name, hs_qpu_prog* p) {
    int n = hs_qpu_prog_finish(p);
    if (n < 0) {
        printf("FAIL %s: error %d at instruction %u\n", name, n, p->rule_at);
        failures++;
        return n;
    }
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", outdir, name);
    FILE* f = fopen(path, "wb");
    if (!f || fwrite(p->code, 8, (size_t)n, f) != (size_t)n) {
        perror(path);
        exit(1);
    }
    fclose(f);
    return n;
}

/* ---- the programs (same order and form as tools/hwprobe_qpu_progs.py) ---- */
static void ldunifrf(hs_qpu_prog* p, int rf) { emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDUNIFRF, RF(rf))); }
static void nop(hs_qpu_prog* p) { emit(p, hs_qpu_alu_nop()); }

static void prog_done(hs_qpu_prog* p, int wait) {
    ldunifrf(p, 0);
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUD, I(1), X));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(0), X));
    if (wait) emit(p, hs_qpu_a(HS_QPU_A_TMUWT, X, X, X));
    hs_qpu_emit_end(p);
}

static void prog_alu(hs_qpu_prog* p, int inner) {
    ldunifrf(p, 0);
    for (int i = 0; i < 4; i++) emit(p, hs_qpu_a(HS_QPU_A_MOV, R(i), hs_qpu_imm_float(1.0f), X));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, R(4), I(1), X));
    int l = hs_qpu_label_new(p);
    hs_qpu_label_bind(p, l);
    for (int i = 0; i < inner; i++)
        emit(p, hs_qpu_m(hs_qpu_a(HS_QPU_A_FADD, R(0), R(0), R(1)), HS_QPU_M_FMUL, R(2), R(2), R(3)));
    hs_qpu_alu s = hs_qpu_a(HS_QPU_A_SUB, RF(0), RF(0), R(4));
    s.af = HS_QPU_PF_PUSHZ;
    emit(p, s);
    hs_qpu_branch(p, HS_QPU_B_NA0, l, 0, 0);
    nop(p);
    nop(p);
    nop(p);
    hs_qpu_emit_end(p);
}

static void prog_gather(hs_qpu_prog* p, int nq, int k) {
    for (int i = 0; i < 4; i++) ldunifrf(p, i);
    int shift = nq == 1 ? 0 : 3;
    if (nq == 1) {
        emit(p, hs_qpu_a(HS_QPU_A_MOV, RF(5), I(0), X));
    } else {
        emit(p, hs_qpu_a(HS_QPU_A_TIDX, R(0), X, X));
        emit(p, hs_qpu_a(HS_QPU_A_SHR, R(0), R(0), I(2)));
        emit(p, hs_qpu_a(HS_QPU_A_BAND, RF(5), R(0), I(15)));
    }
    emit(p, hs_qpu_a(HS_QPU_A_SHL, R(0), RF(5), I(4)));
    emit(p, hs_qpu_a(HS_QPU_A_EIDX, R(1), X, X));
    emit(p, hs_qpu_a(HS_QPU_A_ADD, R(0), R(0), R(1)));
    emit(p, hs_qpu_a(HS_QPU_A_SHL, R(0), R(0), I(2)));
    emit(p, hs_qpu_a(HS_QPU_A_ADD, RF(1), RF(1), R(0)));
    emit(p, hs_qpu_a(HS_QPU_A_ADD, RF(3), RF(3), R(0)));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, R(0), I(1), X));
    emit(p, hs_qpu_a(HS_QPU_A_SHL, RF(4), R(0), I(6 + shift)));
    int l = hs_qpu_label_new(p);
    hs_qpu_label_bind(p, l);
    for (int i = 0; i < k; i++) {
        hs_qpu_alu x = hs_qpu_m(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(1), X), HS_QPU_M_ADD, RF(1), RF(1), RF(4));
        emit(p, i == k - 1 ? hs_qpu_s(x, HS_QPU_SIG_THRSW, X) : x);
    }
    nop(p);
    nop(p);
    for (int i = 0; i < k; i++) {
        emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDTMU, R(0)));
        emit(p, hs_qpu_a(HS_QPU_A_SHL, R(0), R(0), I(2)));
        hs_qpu_alu x = hs_qpu_a(HS_QPU_A_ADD, HS_QPU_TMUA, R(0), RF(2));
        emit(p, i == k - 1 ? hs_qpu_s(x, HS_QPU_SIG_THRSW, X) : x);
    }
    nop(p);
    nop(p);
    for (int i = 0; i < k; i++) {
        emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDTMU, R(1)));
        emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUD, R(1), X));
        emit(p, hs_qpu_m(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(3), X), HS_QPU_M_ADD, RF(3), RF(3), RF(4)));
    }
    hs_qpu_alu s = hs_qpu_a(HS_QPU_A_SUB, RF(0), RF(0), I(1));
    s.af = HS_QPU_PF_PUSHZ;
    emit(p, s);
    hs_qpu_branch(p, HS_QPU_B_NA0, l, 0, 0);
    nop(p);
    nop(p);
    nop(p);
    emit(p, hs_qpu_a(HS_QPU_A_TMUWT, X, X, X));
    hs_qpu_emit_end(p);
}

enum { LD_THRSW, LD_DELAY, LD_TMUAU };
static void prog_ldtest(hs_qpu_prog* p, int form) {
    for (int i = 0; i < 4; i++) ldunifrf(p, i);
    emit(p, hs_qpu_a(HS_QPU_A_EIDX, R(0), X, X));
    emit(p, hs_qpu_a(HS_QPU_A_SHL, R(0), R(0), I(2)));
    emit(p, hs_qpu_a(HS_QPU_A_ADD, RF(1), RF(1), R(0)));
    emit(p, hs_qpu_a(HS_QPU_A_ADD, RF(3), RF(3), R(0)));
    if (form == LD_THRSW) {
        emit(p, hs_qpu_s(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(1), X), HS_QPU_SIG_THRSW, X));
        nop(p);
        nop(p);
    } else if (form == LD_DELAY) {
        emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(1), X));
        nop(p);
        nop(p);
    } else {
        emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_THRSW, X));
        nop(p);
        nop(p);
        emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUAU, RF(1), X));
    }
    emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDTMU, R(0)));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUD, R(0), X));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(3), X));
    emit(p, hs_qpu_a(HS_QPU_A_TMUWT, X, X, X));
    hs_qpu_emit_end(p);
}

/* out[lane] = fmov(imm). For the fmov-immediate divergence: NeoGPU encodes
 * the float input modifier none; the oracle encodes abs (mul_b 0). The abs
 * form is made by clearing that one field, to run both on the GPU. */
static void prog_fmov_imm(hs_qpu_prog* p, int32_t imm) {
    for (int i = 0; i < 4; i++) ldunifrf(p, i);
    emit(p, hs_qpu_a(HS_QPU_A_EIDX, R(0), X, X));
    emit(p, hs_qpu_a(HS_QPU_A_SHL, R(0), R(0), I(2)));
    emit(p, hs_qpu_a(HS_QPU_A_ADD, RF(3), RF(3), R(0)));
    emit(p, hs_qpu_m(hs_qpu_alu_nop(), HS_QPU_M_FMOV, R(1), I(imm), X));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUD, R(1), X));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(3), X));
    emit(p, hs_qpu_a(HS_QPU_A_TMUWT, X, X, X));
    hs_qpu_emit_end(p);
}

static uint64_t buf[4096];

static void build_programs(void) {
    hs_qpu_prog p;
    hs_qpu_prog_init(&p, buf, 4096);
    hs_qpu_emit_end(&p);
    save("end", &p);
    hs_qpu_prog_init(&p, buf, 4096);
    prog_done(&p, 1);
    save("done", &p);
    hs_qpu_prog_init(&p, buf, 4096);
    hs_qpu_prog_rules(&p, 0); /* evidence: ends with its write in flight */
    prog_done(&p, 0);
    save("done_nowt", &p);
    hs_qpu_prog_init(&p, buf, 4096);
    prog_alu(&p, 64);
    save("alu64", &p);
    for (int nq = 1; nq <= 8; nq *= 8)
        for (int k = 1; k <= 8; k *= 2) {
            char name[32];
            snprintf(name, sizeof(name), "gather%d_k%d", nq, k);
            hs_qpu_prog_init(&p, buf, 4096);
            prog_gather(&p, nq, k);
            save(name, &p);
        }
    hs_qpu_prog_init(&p, buf, 4096);
    prog_fmov_imm(&p, -16);
    save("fmov_imm_none", &p);
    p.code[7] &= ~(7ull << 21); /* the oracle's form: input modifier abs */
    save("fmov_imm_abs", &p);
    static const char* LD[3] = {"ld_thrsw", "ld_delay", "ld_tmuau"};
    for (int f = 0; f < 3; f++) {
        hs_qpu_prog_init(&p, buf, 4096);
        if (f != LD_THRSW) hs_qpu_prog_rules(&p, 0); /* evidence: these forms hang the GPU */
        prog_ldtest(&p, f);
        save(LD[f], &p);
    }
}

static void rule_tests(void) {
    hs_qpu_prog p;
    int r;
    hs_qpu_prog_init(&p, buf, 4096);
    prog_ldtest(&p, LD_DELAY);
    r = hs_qpu_prog_finish(&p);
    check(r == HS_QPU_E_RULE_LOAD, "rule 1 rejects a load read with no thrsw (ld_delay, which hung the GPU)");
    hs_qpu_prog_init(&p, buf, 4096);
    prog_ldtest(&p, LD_TMUAU);
    r = hs_qpu_prog_finish(&p);
    check(r == HS_QPU_E_RULE_LOAD, "rule 1 rejects a thrsw issued before the load (ld_tmuau, which hung the GPU)");
    hs_qpu_prog_init(&p, buf, 4096);
    prog_done(&p, 0);
    r = hs_qpu_prog_finish(&p);
    check(r == HS_QPU_E_RULE_END, "rule 2 rejects a program that ends with a TMU write in flight");
    hs_qpu_prog_init(&p, buf, 4096);
    prog_ldtest(&p, LD_THRSW);
    check(hs_qpu_prog_finish(&p) > 0, "rules accept the measured-good load form (ld_thrsw)");
    int all = 1;
    for (int nq = 1; nq <= 8; nq *= 8)
        for (int k = 1; k <= 8; k *= 2) {
            hs_qpu_prog_init(&p, buf, 4096);
            prog_gather(&p, nq, k);
            all &= hs_qpu_prog_finish(&p) > 0;
        }
    check(all, "rules accept all 8 gather programs (0 hangs on hardware)");
    /* a branch to a label never bound */
    hs_qpu_prog_init(&p, buf, 4096);
    int l = hs_qpu_label_new(&p);
    hs_qpu_branch(&p, HS_QPU_B_ALWAYS, l, 0, 0);
    check(hs_qpu_prog_finish(&p) == HS_QPU_E_LABEL, "branch to an unbound label is an error");
    /* three distinct register-file reads in one instruction */
    hs_qpu_alu x = hs_qpu_m(hs_qpu_a(HS_QPU_A_ADD, R(0), RF(1), RF(2)), HS_QPU_M_ADD, R(1), RF(3), R(0));
    uint64_t w;
    check(hs_qpu_pack_alu(&x, &w) == HS_QPU_E_RADDR, "three register-file reads in one instruction are an error");
    x = hs_qpu_a(HS_QPU_A_ADD, R(0), RF(1), RF(2));
    x.mul.op = HS_QPU_M_ADD;
    x.mul.dst = R(1);
    x.mul.a = R(0);
    x.mul.b = I(3);
    check(hs_qpu_pack_alu(&x, &w) == HS_QPU_E_RADDR, "two rf reads plus a small immediate are an error");
    check(hs_qpu_imm_int(16) == HS_QPU_NOREG && hs_qpu_imm_float(3.0f) == HS_QPU_NOREG,
          "values with no small-immediate encoding are refused");
}

/* ---- corpus in the reference assembler's syntax ---- */
static FILE* corpus;
static int ncorpus, nrejected;

static void reg_str(hs_qpu_reg r, char* s, size_t n) {
    unsigned k = r >> 8, i = r & 0xFF;
    if (k == 0) snprintf(s, n, "r%u", i);
    else if (k == 1) snprintf(s, n, "rf%u", i);
    else if (k == 2) snprintf(s, n, "%s", hs_qpu_magic_name(i));
    else if (i < 16) snprintf(s, n, "%u", i);
    else if (i < 32) snprintf(s, n, "%d", (int)i - 32);
    else {
        float f = 1.0f;
        for (int e = (int)i - 40; e > 0; e--) f *= 2;
        for (int e = (int)i - 40; e < 0; e++) f /= 2;
        snprintf(s, n, "%.10g", (double)f);
        if (!strpbrk(s, ".e")) strncat(s, ".0", n - strlen(s) - 1); /* keep it a float in the oracle */
    }
}
static const char* COND_NAME[5] = {"", "ifa", "ifb", "ifna", "ifnb"};
static const char* FLAG_NAME[16] = {"",      "pushz", "pushn", "pushc", "andz",  "andnz", "nornz", "norz",
                                    "andn",  "andnn", "nornn", "norn",  "andc",  "andnc", "nornc", "norc"};

static void part_str(char* s, size_t n, const char* name, hs_qpu_reg d, hs_qpu_reg a, hs_qpu_reg b, int c, int f,
                     const char* sig) {
    char args[160] = "", t[32];
    int first = 1;
    hs_qpu_reg regs[3] = {d, a, b};
    for (int i = 0; i < 3; i++) {
        if (regs[i] == X) continue;
        reg_str(regs[i], t, sizeof(t));
        size_t l = strlen(args);
        snprintf(args + l, sizeof(args) - l, "%s%s", first ? "" : ", ", t);
        first = 0;
    }
    const char* cs = c ? COND_NAME[c] : f ? FLAG_NAME[f] : 0;
    if (cs) {
        size_t l = strlen(args);
        snprintf(args + l, sizeof(args) - l, "%scond='%s'", first ? "" : ", ", cs);
        first = 0;
    }
    if (sig && *sig) {
        size_t l = strlen(args);
        snprintf(args + l, sizeof(args) - l, "%ssig=%s", first ? "" : ", ", sig);
    }
    snprintf(s, n, "%s(%s)", name, args);
}

static void sig_str(const hs_qpu_alu* x, char* s, size_t n) {
    static const struct {
        unsigned bit;
        const char* name;
        int dst;
    } S[] = {{HS_QPU_SIG_THRSW, "thrsw", 0},     {HS_QPU_SIG_LDUNIF, "ldunif", 0},   {HS_QPU_SIG_LDTMU, "ldtmu", 1},
             {HS_QPU_SIG_LDVARY, "ldvary", 1},   {HS_QPU_SIG_LDUNIFRF, "ldunifrf", 1}, {HS_QPU_SIG_LDTLB, "ldtlb", 1},
             {HS_QPU_SIG_LDTLBU, "ldtlbu", 1},   {HS_QPU_SIG_WRTMUC, "wrtmuc", 0},   {HS_QPU_SIG_UCB, "ucb", 0},
             {HS_QPU_SIG_LDUNIFA, "ldunifa", 0}, {HS_QPU_SIG_LDUNIFARF, "ldunifarf", 1}};
    char parts[6][32];
    int np = 0;
    for (size_t i = 0; i < sizeof(S) / sizeof(S[0]); i++) {
        if (!(x->sig & S[i].bit)) continue;
        if (S[i].dst) {
            char t[16];
            reg_str(x->sig_dst, t, sizeof(t));
            snprintf(parts[np++], 32, "%s(%s)", S[i].name, t);
        } else {
            snprintf(parts[np++], 32, "%s", S[i].name);
        }
    }
    s[0] = 0;
    if (np == 1) snprintf(s, n, "%s", parts[0]);
    if (np > 1) {
        size_t l = 0;
        l += (size_t)snprintf(s + l, n - l, "[");
        for (int i = 0; i < np; i++) l += (size_t)snprintf(s + l, n - l, "%s%s", i ? ", " : "", parts[i]);
        snprintf(s + l, n - l, "]");
    }
}

static void corpus_add(hs_qpu_alu x) {
    char sig[96], a[256], m[300];
    sig_str(&x, sig, sizeof(sig));
    /* nop prints as nop(); an operation with no sources and a null
     * destination (tmuwt, vpmwt) prints without one */
    hs_qpu_reg ad = x.add.dst;
    if (x.add.op == HS_QPU_A_NOP || (x.add.a == X && (ad == HS_QPU_NULL || ad == X))) ad = X;
    part_str(a, sizeof(a), hs_qpu_add_name(x.add.op), ad, x.add.a, x.add.b, x.ac, x.af, sig);
    m[0] = 0;
    if (x.mul.op != HS_QPU_M_NOP) {
        char t[256];
        hs_qpu_reg md = x.mul.dst == X ? HS_QPU_NULL : x.mul.dst;
        part_str(t, sizeof(t), hs_qpu_mul_name(x.mul.op), md, x.mul.a, x.mul.b, x.mc, x.mf, 0);
        snprintf(m, sizeof(m), ".%s", t);
    }
    uint64_t w;
    int e = hs_qpu_pack_alu(&x, &w);
    if (e) {
        fprintf(corpus, "REJECT %d\t%s%s\n", e, a, m);
        nrejected++;
    } else {
        fprintf(corpus, "%016llx\t%s%s\n", (unsigned long long)w, a, m);
        ncorpus++;
    }
}

static void build_corpus(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/corpus.txt", outdir);
    corpus = fopen(path, "w");
    if (!corpus) {
        perror(path);
        exit(1);
    }
    const hs_qpu_reg two[][3] = {{R(0), R(1), R(2)},     {R(0), R(1), R(0)},   {RF(10), RF(11), RF(12)},
                                 {R(3), RF(5), R(4)},    {RF(6), RF(7), I(3)}, {R(5), I(-1), RF(1)},
                                 {RF(2), RF(2), RF(2)},  {R(1), R(2), R(1)},   {R(2), RF(9), RF(9)},
                                 {HS_QPU_TMUD, R(0), I(15)}, {R(0), I(1), I(1)}};
    const hs_qpu_reg one[][2] = {{R(0), R(1)}, {RF(3), RF(4)}, {HS_QPU_TMUA, R(2)}, {R(1), I(2)}, {RF(63), R(5)}};
    /* every add op with each operand pattern its kind takes */
    for (int op = 0; op < HS_QPU_A_COUNT; op++) {
        int nsrc = op <= 0 ? 0 : op <= HS_QPU_A_VFMAX ? 2 : op <= HS_QPU_A_UTOF ? 1 : 0;
        if (op == HS_QPU_A_NOP) {
            corpus_add(hs_qpu_alu_nop());
        } else if (nsrc == 2) {
            for (size_t i = 0; i < sizeof(two) / sizeof(two[0]); i++)
                corpus_add(hs_qpu_a(op, two[i][0], two[i][1], two[i][2]));
            corpus_add(hs_qpu_a(op, R(0), hs_qpu_imm_float(0.5f), R(1)));
        } else if (nsrc == 1) {
            for (size_t i = 0; i < sizeof(one) / sizeof(one[0]); i++) corpus_add(hs_qpu_a(op, one[i][0], one[i][1], X));
        } else {
            corpus_add(hs_qpu_a(op, R(0), X, X));
            corpus_add(hs_qpu_a(op, RF(9), X, X));
            if (op == HS_QPU_A_TMUWT || op == HS_QPU_A_VPMWT) corpus_add(hs_qpu_a(op, X, X, X));
            if (op == HS_QPU_A_BARRIERID) corpus_add(hs_qpu_a(op, HS_QPU_SYNCB, X, X));
        }
    }
    /* every mul op, as the only operation and dual-issued */
    for (int op = 1; op < HS_QPU_M_COUNT; op++) {
        int one_src = op == HS_QPU_M_MOV || op == HS_QPU_M_FMOV;
        for (size_t i = 0; i < sizeof(two) / sizeof(two[0]); i++) {
            hs_qpu_alu x = hs_qpu_m(hs_qpu_alu_nop(), op, two[i][0], two[i][1], one_src ? X : two[i][2]);
            corpus_add(x);
        }
        corpus_add(hs_qpu_m(hs_qpu_a(HS_QPU_A_ADD, R(0), RF(1), R(2)), op, R(3), RF(1), one_src ? X : RF(4)));
        corpus_add(hs_qpu_m(hs_qpu_a(HS_QPU_A_FADD, RF(8), R(1), R(0)), op, RF(9), R(2), one_src ? X : I(7)));
    }
    /* destinations: every magic register through mov on both ALUs */
    for (unsigned w = 0; w < 64; w++) {
        if (!hs_qpu_magic_name(w)) continue;
        corpus_add(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_MAGIC(w), R(0), X));
        corpus_add(hs_qpu_m(hs_qpu_alu_nop(), HS_QPU_M_MOV, HS_QPU_MAGIC(w), R(1), X));
    }
    for (int rf = 0; rf < 64; rf += 9) corpus_add(hs_qpu_a(HS_QPU_A_ADD, RF(rf), RF(63 - rf), I(rf % 16)));
    /* every small immediate */
    for (int v = -16; v < 16; v++) corpus_add(hs_qpu_a(HS_QPU_A_ADD, R(0), R(1), I(v)));
    for (int e = -8; e <= 7; e++) {
        float f = 1.0f;
        for (int i = e; i > 0; i--) f *= 2;
        for (int i = e; i < 0; i++) f /= 2;
        corpus_add(hs_qpu_a(HS_QPU_A_FADD, R(0), R(1), hs_qpu_imm_float(f)));
    }
    /* conditions and flags: every single field, then every legal pairing */
    for (int ac = 0; ac <= 4; ac++)
        for (int mc = 0; mc <= 4; mc++)
            for (int af = 0; af <= 15; af++)
                for (int mf = 0; mf <= 15; mf++) {
                    int nset = (ac > 0) + (mc > 0) + (af > 0) + (mf > 0);
                    if (nset == 0 || nset > 2) continue;
                    /* the oracle's syntax takes one cond= per ALU, so a
                     * condition and a flag on the same ALU cannot be written */
                    if ((ac && af) || (mc && mf)) continue;
                    hs_qpu_alu x = hs_qpu_m(hs_qpu_a(HS_QPU_A_ADD, R(0), R(1), R(2)), HS_QPU_M_ADD, R(3), R(1), R(2));
                    x.ac = (uint8_t)ac;
                    x.mc = (uint8_t)mc;
                    x.af = (uint8_t)af;
                    x.mf = (uint8_t)mf;
                    corpus_add(x);
                }
    /* signals: every combination of the single-bit signals, with and without
     * a small immediate, destinations in an accumulator and the register file */
    static const unsigned SIGS[] = {HS_QPU_SIG_THRSW,  HS_QPU_SIG_LDUNIF, HS_QPU_SIG_LDTMU,   HS_QPU_SIG_LDVARY,
                                    HS_QPU_SIG_LDUNIFRF, HS_QPU_SIG_LDTLB, HS_QPU_SIG_LDTLBU, HS_QPU_SIG_WRTMUC,
                                    HS_QPU_SIG_UCB,    HS_QPU_SIG_LDUNIFA, HS_QPU_SIG_LDUNIFARF};
    const unsigned ns = sizeof(SIGS) / sizeof(SIGS[0]);
    for (unsigned mask = 1; mask < (1u << ns); mask++) {
        if (__builtin_popcount(mask) > 3) continue;
        unsigned sig = 0;
        for (unsigned i = 0; i < ns; i++)
            if (mask & (1u << i)) sig |= SIGS[i];
        hs_qpu_alu x = hs_qpu_alu_nop();
        x.sig = (uint16_t)sig;
        uint64_t w;
        x.sig_dst = (sig & HS_QPU_SIG_LDUNIFRF) || (sig & HS_QPU_SIG_LDUNIFARF) ? RF(7) : R(3);
        if (hs_qpu_pack_alu(&x, &w) == 0) {
            corpus_add(x);
            x.sig_dst = RF(7);
            corpus_add(x);
            hs_qpu_alu y = hs_qpu_a(HS_QPU_A_SHL, R(0), R(0), I(2));
            y.sig = x.sig;
            y.sig_dst = x.sig_dst;
            corpus_add(y);
        }
    }
    /* register-file read sharing between the two ALUs */
    corpus_add(hs_qpu_m(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(1), X), HS_QPU_M_ADD, RF(1), RF(1), RF(4)));
    corpus_add(hs_qpu_m(hs_qpu_a(HS_QPU_A_ADD, RF(1), RF(4), RF(1)), HS_QPU_M_SUB, RF(4), RF(1), RF(4)));
    corpus_add(hs_qpu_m(hs_qpu_a(HS_QPU_A_ADD, R(0), RF(2), I(5)), HS_QPU_M_ADD, R(1), RF(2), I(5)));
    corpus_add(hs_qpu_m(hs_qpu_a(HS_QPU_A_FADD, R(0), R(0), R(1)), HS_QPU_M_FMUL, R(2), R(2), R(3)));
    fclose(corpus);
    printf("corpus: %d instructions written, %d rejected by the encoder (listed as REJECT)\n", ncorpus, nrejected);

    /* branch loops: n nops then a branch back, for every condition and both
     * uniform-stream forms */
    static const int BC[] = {HS_QPU_B_ALWAYS, HS_QPU_B_A0, HS_QPU_B_NA0, HS_QPU_B_ALLA, HS_QPU_B_ANYNA,
                             HS_QPU_B_ANYA,   HS_QPU_B_ALLNA};
    static const int BN[] = {0, 1, 5, 300};
    for (size_t c = 0; c < sizeof(BC) / sizeof(BC[0]); c++)
        for (size_t i = 0; i < sizeof(BN) / sizeof(BN[0]); i++)
            for (int ub = 0; ub < 3; ub++) {
                hs_qpu_prog p;
                hs_qpu_prog_init(&p, buf, 4096);
                int l = hs_qpu_label_new(&p);
                hs_qpu_label_bind(&p, l);
                for (int k = 0; k < BN[i]; k++) nop(&p);
                hs_qpu_branch(&p, BC[c], l, ub > 0, ub == 2);
                nop(&p);
                nop(&p);
                nop(&p);
                char name[64];
                snprintf(name, sizeof(name), "br_%d_%d_%d", BC[c], BN[i], ub);
                save(name, &p);
            }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s OUTDIR\n", argv[0]);
        return 2;
    }
    outdir = argv[1];
    build_programs();
    rule_tests();
    build_corpus();
    printf("%s\n", failures ? "FAILED" : "all encoder self-checks passed");
    return failures ? 1 : 0;
}
