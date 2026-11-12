/* hs_qpu: instruction encoder for the Raspberry Pi 4 GPU (Broadcom V3D 4.2,
 * "VideoCore VI") QPU. Dependency-free: no Mesa, no libdrm, no allocation.
 *
 * A QPU instruction is 64 bits. An ALU instruction carries one add-ALU and
 * one mul-ALU operation (dual issue), a signal set, and condition/flag
 * fields. A branch is its own format. Encodings follow the measured layout in
 * docs/benchmarks/QPU_ENCODER.md and are checked word for word against an
 * independent assembler (tests/test_qpu_encode.c, tools/qpu_oracle_check.py).
 *
 * Programs built with hs_qpu_prog enforce two measured hardware rules by
 * default (docs/benchmarks/HARDWARE_MAP_2026-11-15.md); breaking either hung
 * the GPU:
 *   1. every TMU load that ldtmu reads has a thrsw signal at or after the
 *      load request and at least two instructions before the ldtmu;
 *   2. a program does not end with TMU writes in flight (tmuwt first).
 * The check follows instructions in program order (a loop body is checked as
 * written) and assumes one ldtmu per load (the default TMU configuration).
 * Only TMUA/TMUAU general loads and stores are tracked.
 */
#ifndef HS_QPU_H
#define HS_QPU_H

#include <stdint.h>

/* ---- operands ----------------------------------------------------------
 * Accumulators r0..r5 and register-file entries rf0..rf63 can be read and
 * written. Magic registers (TMU, TLB, sync, ...) are write-only. Small
 * immediates are read-only and use the instruction's raddr_b slot. */
typedef uint16_t hs_qpu_reg;
#define HS_QPU_R(n) ((hs_qpu_reg)(0x000u | (unsigned)(n)))
#define HS_QPU_RF(n) ((hs_qpu_reg)(0x100u | (unsigned)(n)))
#define HS_QPU_MAGIC(w) ((hs_qpu_reg)(0x200u | (unsigned)(w)))
#define HS_QPU_SMALL(i) ((hs_qpu_reg)(0x300u | (unsigned)(i)))
#define HS_QPU_NOREG ((hs_qpu_reg)0xFFFFu)

#define HS_QPU_R0 HS_QPU_R(0)
#define HS_QPU_R1 HS_QPU_R(1)
#define HS_QPU_R2 HS_QPU_R(2)
#define HS_QPU_R3 HS_QPU_R(3)
#define HS_QPU_R4 HS_QPU_R(4)
#define HS_QPU_R5 HS_QPU_R(5)
#define HS_QPU_NULL HS_QPU_MAGIC(6)
#define HS_QPU_TLB HS_QPU_MAGIC(7)
#define HS_QPU_TLBU HS_QPU_MAGIC(8)
#define HS_QPU_UNIFA HS_QPU_MAGIC(9)
#define HS_QPU_TMUL HS_QPU_MAGIC(10)
#define HS_QPU_TMUD HS_QPU_MAGIC(11)
#define HS_QPU_TMUA HS_QPU_MAGIC(12)
#define HS_QPU_TMUAU HS_QPU_MAGIC(13)
#define HS_QPU_VPM HS_QPU_MAGIC(14)
#define HS_QPU_VPMU HS_QPU_MAGIC(15)
#define HS_QPU_SYNC HS_QPU_MAGIC(16)
#define HS_QPU_SYNCU HS_QPU_MAGIC(17)
#define HS_QPU_SYNCB HS_QPU_MAGIC(18)
#define HS_QPU_TMUC HS_QPU_MAGIC(32)
#define HS_QPU_TMUS HS_QPU_MAGIC(33)
#define HS_QPU_TMUT HS_QPU_MAGIC(34)
#define HS_QPU_TMUR HS_QPU_MAGIC(35)
#define HS_QPU_TMUI HS_QPU_MAGIC(36)
#define HS_QPU_TMUB HS_QPU_MAGIC(37)
#define HS_QPU_TMUDREF HS_QPU_MAGIC(38)
#define HS_QPU_TMUOFF HS_QPU_MAGIC(39)
#define HS_QPU_TMUSCM HS_QPU_MAGIC(40)
#define HS_QPU_TMUSF HS_QPU_MAGIC(41)
#define HS_QPU_TMUSLOD HS_QPU_MAGIC(42)
#define HS_QPU_TMUHS HS_QPU_MAGIC(43)
#define HS_QPU_TMUHSCM HS_QPU_MAGIC(44)
#define HS_QPU_TMUHSF HS_QPU_MAGIC(45)
#define HS_QPU_TMUHSLOD HS_QPU_MAGIC(46)
#define HS_QPU_R5REP HS_QPU_MAGIC(55)

/* Small immediates: integers -16..15 and the floats 2^-8..2^7. Returns
 * HS_QPU_NOREG when the value has no encoding. hs_qpu_imm_bits takes the raw
 * 32-bit pattern (an int, or a float's bits). */
hs_qpu_reg hs_qpu_imm_int(int32_t v);
hs_qpu_reg hs_qpu_imm_float(float f);
hs_qpu_reg hs_qpu_imm_bits(uint32_t bits);

/* ---- operations -------------------------------------------------------- */
typedef enum {
    HS_QPU_A_NOP = 0, /* no operands, no destination */
    /* two sources */
    HS_QPU_A_FADD, HS_QPU_A_FADDNF, HS_QPU_A_VFPACK, HS_QPU_A_ADD, HS_QPU_A_SUB, HS_QPU_A_FSUB,
    HS_QPU_A_IMIN, HS_QPU_A_IMAX, HS_QPU_A_UMIN, HS_QPU_A_UMAX, HS_QPU_A_SHL, HS_QPU_A_SHR,
    HS_QPU_A_ASR, HS_QPU_A_ROR, HS_QPU_A_FMIN, HS_QPU_A_FMAX, HS_QPU_A_VFMIN, HS_QPU_A_BAND,
    HS_QPU_A_BOR, HS_QPU_A_BXOR, HS_QPU_A_FCMP, HS_QPU_A_VFMAX,
    /* one source */
    HS_QPU_A_MOV, /* bor src, src */
    HS_QPU_A_BNOT, HS_QPU_A_NEG, HS_QPU_A_FLAPUSH, HS_QPU_A_FLBPUSH, HS_QPU_A_FLPOP,
    HS_QPU_A_SETMSF, HS_QPU_A_SETREVF, HS_QPU_A_FROUND, HS_QPU_A_FTOIN, HS_QPU_A_FTRUNC,
    HS_QPU_A_FTOIZ, HS_QPU_A_FFLOOR, HS_QPU_A_FTOUZ, HS_QPU_A_FCEIL, HS_QPU_A_FTOC, HS_QPU_A_FDX,
    HS_QPU_A_FDY, HS_QPU_A_ITOF, HS_QPU_A_CLZ, HS_QPU_A_UTOF,
    /* no source, destination */
    HS_QPU_A_TIDX, HS_QPU_A_EIDX, HS_QPU_A_LR, HS_QPU_A_VFLA, HS_QPU_A_VFLNA, HS_QPU_A_VFLB,
    HS_QPU_A_VFLNB, HS_QPU_A_MSF, HS_QPU_A_REVF, HS_QPU_A_IID, HS_QPU_A_SAMPID, HS_QPU_A_BARRIERID,
    HS_QPU_A_TMUWT, HS_QPU_A_VPMWT,
    HS_QPU_A_COUNT
} hs_qpu_add_op;

typedef enum {
    HS_QPU_M_NOP = 0,
    HS_QPU_M_ADD, HS_QPU_M_SUB, HS_QPU_M_UMUL24, HS_QPU_M_VFMUL, HS_QPU_M_SMUL24, HS_QPU_M_MULTOP,
    HS_QPU_M_FMUL,
    HS_QPU_M_MOV, HS_QPU_M_FMOV,
    HS_QPU_M_COUNT
} hs_qpu_mul_op;

/* signals (combine with |); not every combination exists in hardware */
enum {
    HS_QPU_SIG_THRSW = 1u << 0,
    HS_QPU_SIG_LDUNIF = 1u << 1,
    HS_QPU_SIG_LDTMU = 1u << 2,    /* destination: sig_dst */
    HS_QPU_SIG_LDVARY = 1u << 3,   /* destination: sig_dst */
    HS_QPU_SIG_LDUNIFRF = 1u << 4, /* destination: sig_dst */
    HS_QPU_SIG_SMIMM = 1u << 5,    /* set automatically by a small-immediate operand */
    HS_QPU_SIG_LDTLB = 1u << 6,    /* destination: sig_dst */
    HS_QPU_SIG_LDTLBU = 1u << 7,   /* destination: sig_dst */
    HS_QPU_SIG_WRTMUC = 1u << 8,
    HS_QPU_SIG_UCB = 1u << 9,
    HS_QPU_SIG_LDUNIFA = 1u << 10,
    HS_QPU_SIG_LDUNIFARF = 1u << 11, /* destination: sig_dst */
};

/* per-ALU write condition */
enum { HS_QPU_COND_NONE = 0, HS_QPU_COND_IFA, HS_QPU_COND_IFB, HS_QPU_COND_IFNA, HS_QPU_COND_IFNB };

/* per-ALU flag push (1..3) or update (4..15) */
enum {
    HS_QPU_PF_NONE = 0, HS_QPU_PF_PUSHZ, HS_QPU_PF_PUSHN, HS_QPU_PF_PUSHC,
    HS_QPU_UF_ANDZ, HS_QPU_UF_ANDNZ, HS_QPU_UF_NORNZ, HS_QPU_UF_NORZ,
    HS_QPU_UF_ANDN, HS_QPU_UF_ANDNN, HS_QPU_UF_NORNN, HS_QPU_UF_NORN,
    HS_QPU_UF_ANDC, HS_QPU_UF_ANDNC, HS_QPU_UF_NORNC, HS_QPU_UF_NORC,
};

/* branch conditions */
enum {
    HS_QPU_B_ALWAYS = 0, HS_QPU_B_A0 = 2, HS_QPU_B_NA0 = 3, HS_QPU_B_ALLA = 4,
    HS_QPU_B_ANYNA = 5, HS_QPU_B_ANYA = 6, HS_QPU_B_ALLNA = 7,
};

/* one ALU instruction; start from hs_qpu_alu_nop() */
typedef struct {
    struct {
        uint8_t op; /* hs_qpu_add_op */
        hs_qpu_reg dst, a, b;
    } add;
    struct {
        uint8_t op; /* hs_qpu_mul_op */
        hs_qpu_reg dst, a, b;
    } mul;
    uint16_t sig;
    hs_qpu_reg sig_dst;
    uint8_t ac, mc; /* HS_QPU_COND_* for the add / mul write */
    uint8_t af, mf; /* HS_QPU_PF_* / HS_QPU_UF_* for the add / mul result */
} hs_qpu_alu;

hs_qpu_alu hs_qpu_alu_nop(void);

/* builders: x = hs_qpu_a(HS_QPU_A_ADD, d, a, b), then optionally
 * hs_qpu_m(x, ...) for the mul ALU and hs_qpu_s(x, signals, signal_dst).
 * Unused operands are HS_QPU_NOREG; HS_QPU_NOREG as a destination is null. */
static inline hs_qpu_alu hs_qpu_a(int op, hs_qpu_reg d, hs_qpu_reg a, hs_qpu_reg b) {
    hs_qpu_alu x = hs_qpu_alu_nop();
    x.add.op = (uint8_t)op;
    x.add.dst = d;
    x.add.a = a;
    x.add.b = b;
    return x;
}
static inline hs_qpu_alu hs_qpu_m(hs_qpu_alu x, int op, hs_qpu_reg d, hs_qpu_reg a, hs_qpu_reg b) {
    x.mul.op = (uint8_t)op;
    x.mul.dst = d;
    x.mul.a = a;
    x.mul.b = b;
    return x;
}
static inline hs_qpu_alu hs_qpu_s(hs_qpu_alu x, unsigned sig, hs_qpu_reg sig_dst) {
    x.sig = (uint16_t)(x.sig | sig);
    if (sig_dst != HS_QPU_NOREG) x.sig_dst = sig_dst;
    return x;
}

/* error codes (negative) */
enum {
    HS_QPU_OK = 0,
    HS_QPU_E_OP = -1,       /* unknown operation */
    HS_QPU_E_OPERAND = -2,  /* operand missing, extra, or of a kind the slot cannot take */
    HS_QPU_E_RADDR = -3,    /* more than two register-file reads, or rf + immediate clash */
    HS_QPU_E_SIG = -4,      /* signal combination with no encoding */
    HS_QPU_E_COND = -5,     /* condition/flag combination with no encoding */
    HS_QPU_E_FULL = -6,     /* program buffer or label table full */
    HS_QPU_E_LABEL = -7,    /* branch to an unbound or unknown label */
    HS_QPU_E_RANGE = -8,    /* branch offset out of range */
    HS_QPU_E_RULE_LOAD = -9, /* ldtmu reads a load with no thrsw between (rule 1) */
    HS_QPU_E_RULE_END = -10, /* program ends with TMU writes in flight (rule 2) */
};

int hs_qpu_pack_alu(const hs_qpu_alu* in, uint64_t* out);
/* rel_bytes: target minus (branch address + 32), the branch's 3 delay slots
 * included. ub/ub_abs: also set the uniform stream (relative or absolute). */
int hs_qpu_pack_branch(int cond, int32_t rel_bytes, int ub, int ub_abs, uint64_t* out);

/* ---- programs ---------------------------------------------------------- */
#define HS_QPU_MAX_LABELS 64
#define HS_QPU_MAX_FIXUPS 256
typedef struct {
    uint64_t* code;
    uint32_t n, cap;
    int err; /* first error, sticky */
    int32_t label_at[HS_QPU_MAX_LABELS];
    uint32_t nlabels;
    struct {
        uint32_t at;
        uint16_t label;
        uint8_t cond, ub, ub_abs;
    } fix[HS_QPU_MAX_FIXUPS];
    uint32_t nfix;
    /* rule tracking (hs_qpu_prog_rules) */
    int rules;
    uint32_t loads[16], nloads; /* issue index of outstanding TMU loads, oldest first */
    uint32_t thrsw[16], nthrsw; /* recent thrsw indices */
    uint32_t tmud_pending;      /* TMUD writes since the last TMUA */
    int store_pending;          /* TMU stores not yet waited for */
    uint32_t rule_at;           /* instruction that broke a rule */
} hs_qpu_prog;

void hs_qpu_prog_init(hs_qpu_prog* p, uint64_t* buf, uint32_t cap);
/* rules are on after init; 0 turns them off (only for deliberate evidence) */
void hs_qpu_prog_rules(hs_qpu_prog* p, int on);
void hs_qpu_emit(hs_qpu_prog* p, const hs_qpu_alu* in);
int hs_qpu_label_new(hs_qpu_prog* p);
void hs_qpu_label_bind(hs_qpu_prog* p, int label);
void hs_qpu_branch(hs_qpu_prog* p, int cond, int label, int ub, int ub_abs);
/* thread end: the 8-instruction sequence that ends a QPU program */
void hs_qpu_emit_end(hs_qpu_prog* p);
/* resolve branches and close the rule check; returns the instruction count
 * or a negative error (p->rule_at names the instruction for rule errors) */
int hs_qpu_prog_finish(hs_qpu_prog* p);

/* names as the reference assembler spells them (for listings and tests) */
const char* hs_qpu_add_name(int op);
const char* hs_qpu_mul_name(int op);
const char* hs_qpu_magic_name(unsigned waddr);

#endif
