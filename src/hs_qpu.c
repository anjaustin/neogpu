/* hs_qpu: V3D 4.2 QPU instruction encoder. See include/hs_qpu.h and
 * docs/benchmarks/QPU_ENCODER.md for the measured instruction layout. */
#include "hs_qpu.h"

#include <string.h>

/* ---- ALU word layout (bit positions) ---- */
#define SH_MUL_OP 58 /* 6 bits */
#define SH_SIG 53    /* 5 bits */
#define SH_COND 46   /* 7 bits; holds the destination of a writing signal */
#define SH_MM 45     /* mul destination is magic/accumulator */
#define SH_MA 44     /* add destination is magic/accumulator */
#define SH_WADDR_M 38
#define SH_WADDR_A 32
#define SH_ADD_OP 24 /* 8 bits */
#define SH_MUL_B 21  /* 3-bit input muxes */
#define SH_MUL_A 18
#define SH_ADD_B 15
#define SH_ADD_A 12
#define SH_RADDR_A 6 /* 6 bits */
#define SH_RADDR_B 0

#define MUX_RA 6 /* register file via raddr_a */
#define MUX_RB 7 /* register file or small immediate via raddr_b */

#define F32_UNPACK_NONE 1 /* float input modifier "none" in an opcode's unpack bits */

enum kind {
    K_NONE,  /* no sources; mux_a and mux_b are part of the opcode */
    K_ONE,   /* one source in mux_a; mux_b is part of the opcode */
    K_TWO,   /* two sources */
    K_FTWO,  /* two float sources; opcode carries both unpack fields */
    K_FORD0, /* K_FTWO, commutative pair: operands ordered low first (fadd, fmin) */
    K_FORD1, /* K_FTWO, commutative pair: operands ordered high first (faddnf, fmax) */
    K_MOV,   /* one source read through both muxes (bor x, x) */
};

typedef struct {
    const char* name;
    uint8_t opcode, kind, mux_a, mux_b;
} op_desc;

static const op_desc ADD_OPS[HS_QPU_A_COUNT] = {
    [HS_QPU_A_NOP] = {"nop", 187, K_NONE, 0, 0},
    [HS_QPU_A_FADD] = {"fadd", 0, K_FORD0, 0, 0},
    [HS_QPU_A_FADDNF] = {"faddnf", 0, K_FORD1, 0, 0},
    [HS_QPU_A_VFPACK] = {"vfpack", 53, K_TWO, 0, 0},
    [HS_QPU_A_ADD] = {"add", 56, K_TWO, 0, 0},
    [HS_QPU_A_SUB] = {"sub", 60, K_TWO, 0, 0},
    [HS_QPU_A_FSUB] = {"fsub", 64, K_FTWO, 0, 0},
    [HS_QPU_A_IMIN] = {"imin", 120, K_TWO, 0, 0},
    [HS_QPU_A_IMAX] = {"imax", 121, K_TWO, 0, 0},
    [HS_QPU_A_UMIN] = {"umin", 122, K_TWO, 0, 0},
    [HS_QPU_A_UMAX] = {"umax", 123, K_TWO, 0, 0},
    [HS_QPU_A_SHL] = {"shl", 124, K_TWO, 0, 0},
    [HS_QPU_A_SHR] = {"shr", 125, K_TWO, 0, 0},
    [HS_QPU_A_ASR] = {"asr", 126, K_TWO, 0, 0},
    [HS_QPU_A_ROR] = {"ror", 127, K_TWO, 0, 0},
    [HS_QPU_A_FMIN] = {"fmin", 128, K_FORD0, 0, 0},
    [HS_QPU_A_FMAX] = {"fmax", 128, K_FORD1, 0, 0},
    [HS_QPU_A_VFMIN] = {"vfmin", 176, K_TWO, 0, 0},
    [HS_QPU_A_BAND] = {"band", 181, K_TWO, 0, 0},
    [HS_QPU_A_BOR] = {"bor", 182, K_TWO, 0, 0},
    [HS_QPU_A_BXOR] = {"bxor", 183, K_TWO, 0, 0},
    [HS_QPU_A_FCMP] = {"fcmp", 192, K_FTWO, 0, 0},
    [HS_QPU_A_VFMAX] = {"vfmax", 240, K_TWO, 0, 0},
    [HS_QPU_A_MOV] = {"mov", 182, K_MOV, 0, 0},
    [HS_QPU_A_BNOT] = {"bnot", 186, K_ONE, 0, 0},
    [HS_QPU_A_NEG] = {"neg", 186, K_ONE, 0, 1},
    [HS_QPU_A_FLAPUSH] = {"flapush", 186, K_ONE, 0, 2},
    [HS_QPU_A_FLBPUSH] = {"flbpush", 186, K_ONE, 0, 3},
    [HS_QPU_A_FLPOP] = {"flpop", 186, K_ONE, 0, 4},
    [HS_QPU_A_SETMSF] = {"setmsf", 186, K_ONE, 0, 6},
    [HS_QPU_A_SETREVF] = {"setrevf", 186, K_ONE, 0, 7},
    [HS_QPU_A_FROUND] = {"fround", 245, K_ONE, 0, 0},
    [HS_QPU_A_FTOIN] = {"ftoin", 245, K_ONE, 0, 3},
    [HS_QPU_A_FTRUNC] = {"ftrunc", 245, K_ONE, 0, 4},
    [HS_QPU_A_FTOIZ] = {"ftoiz", 245, K_ONE, 0, 7},
    [HS_QPU_A_FFLOOR] = {"ffloor", 246, K_ONE, 0, 0},
    [HS_QPU_A_FTOUZ] = {"ftouz", 246, K_ONE, 0, 3},
    [HS_QPU_A_FCEIL] = {"fceil", 246, K_ONE, 0, 4},
    [HS_QPU_A_FTOC] = {"ftoc", 246, K_ONE, 0, 7},
    [HS_QPU_A_FDX] = {"fdx", 247, K_ONE, 0, 0},
    [HS_QPU_A_FDY] = {"fdy", 247, K_ONE, 0, 4},
    [HS_QPU_A_ITOF] = {"itof", 252, K_ONE, 0, 0},
    [HS_QPU_A_CLZ] = {"clz", 252, K_ONE, 0, 3},
    [HS_QPU_A_UTOF] = {"utof", 252, K_ONE, 0, 4},
    [HS_QPU_A_TIDX] = {"tidx", 187, K_NONE, 1, 0},
    [HS_QPU_A_EIDX] = {"eidx", 187, K_NONE, 2, 0},
    [HS_QPU_A_LR] = {"lr", 187, K_NONE, 3, 0},
    [HS_QPU_A_VFLA] = {"vfla", 187, K_NONE, 4, 0},
    [HS_QPU_A_VFLNA] = {"vflna", 187, K_NONE, 5, 0},
    [HS_QPU_A_VFLB] = {"vflb", 187, K_NONE, 6, 0},
    [HS_QPU_A_VFLNB] = {"vflnb", 187, K_NONE, 7, 0},
    [HS_QPU_A_MSF] = {"msf", 187, K_NONE, 0, 2},
    [HS_QPU_A_REVF] = {"revf", 187, K_NONE, 1, 2},
    [HS_QPU_A_IID] = {"iid", 187, K_NONE, 2, 2},
    [HS_QPU_A_SAMPID] = {"sampid", 187, K_NONE, 3, 2},
    [HS_QPU_A_BARRIERID] = {"barrierid", 187, K_NONE, 4, 2},
    [HS_QPU_A_TMUWT] = {"tmuwt", 187, K_NONE, 5, 2},
    /* special functions: add-ALU operations with a destination on V3D 4.2
     * (encodings read from Mesa's compiled code for tests/test_08, checked
     * on the hardware by tools/p0_6_sfu_probe). Writing x to the magic
     * registers 48..53 does nothing here. */
    [HS_QPU_A_RECIP] = {"recip", 186, K_ONE, 0, 5},
    [HS_QPU_A_RSQRT] = {"rsqrt", 188, K_ONE, 0, 3},
    [HS_QPU_A_SIN] = {"sin", 188, K_ONE, 0, 6},
    [HS_QPU_A_VPMWT] = {"vpmwt", 187, K_NONE, 6, 2},
};

static const op_desc MUL_OPS[HS_QPU_M_COUNT] = {
    [HS_QPU_M_NOP] = {"nop", 15, K_NONE, 0, 4},
    [HS_QPU_M_ADD] = {"add", 1, K_TWO, 0, 0},
    [HS_QPU_M_SUB] = {"sub", 2, K_TWO, 0, 0},
    [HS_QPU_M_UMUL24] = {"umul24", 3, K_TWO, 0, 0},
    [HS_QPU_M_VFMUL] = {"vfmul", 4, K_TWO, 0, 0},
    [HS_QPU_M_SMUL24] = {"smul24", 9, K_TWO, 0, 0},
    [HS_QPU_M_MULTOP] = {"multop", 10, K_TWO, 0, 0},
    [HS_QPU_M_FMUL] = {"fmul", 16, K_FTWO, 0, 0},
    [HS_QPU_M_MOV] = {"mov", 15, K_ONE, 0, 7},
    [HS_QPU_M_FMOV] = {"fmov", 14, K_ONE, 0, 1}, /* mux_b 1: float input "none" (measured) */
};

const char* hs_qpu_add_name(int op) { return op >= 0 && op < HS_QPU_A_COUNT ? ADD_OPS[op].name : "?"; }
const char* hs_qpu_mul_name(int op) { return op >= 0 && op < HS_QPU_M_COUNT ? MUL_OPS[op].name : "?"; }

const char* hs_qpu_magic_name(unsigned w) {
    static const char* const LO[19] = {"r0",  "r1",   "r2",    "r3",  "r4",   "r5",  "null",
                                       "tlb", "tlbu", "unifa", "tmul", "tmud", "tmua", "tmuau",
                                       "vpm", "vpmu", "sync",  "syncu", "syncb"};
    static const char* const TMU[15] = {"tmuc",   "tmus",    "tmut",  "tmur",    "tmui",
                                        "tmub",   "tmudref", "tmuoff", "tmuscm", "tmusf",
                                        "tmuslod", "tmuhs",  "tmuhscm", "tmuhsf", "tmuhslod"};
    if (w < 19) return LO[w];
    if (w >= 32 && w < 47) return TMU[w - 32];
    if (w == 55) return "r5rep";
    return 0;
}

/* ---- small immediates ---- */
hs_qpu_reg hs_qpu_imm_bits(uint32_t bits) {
    int32_t v = (int32_t)bits;
    if (v >= 0 && v <= 15) return HS_QPU_SMALL(v);
    if (v >= -16 && v <= -1) return HS_QPU_SMALL(32 + v);
    /* floats 2^-8 .. 2^7: sign 0, mantissa 0, exponent 119..134 */
    if ((bits & 0x807FFFFFu) == 0) {
        uint32_t e = bits >> 23;
        if (e >= 119 && e <= 134) return HS_QPU_SMALL(32 + (e - 119));
    }
    return HS_QPU_NOREG;
}
hs_qpu_reg hs_qpu_imm_int(int32_t v) { return hs_qpu_imm_bits((uint32_t)v); }
hs_qpu_reg hs_qpu_imm_float(float f) {
    uint32_t b;
    memcpy(&b, &f, 4);
    return hs_qpu_imm_bits(b);
}

/* ---- signals: the 5-bit field is an index into the combinations the
 * hardware implements (measured). Index 23 (rotate) is not supported here. */
#define S_T HS_QPU_SIG_THRSW
#define S_U HS_QPU_SIG_LDUNIF
#define S_M HS_QPU_SIG_LDTMU
#define S_V HS_QPU_SIG_LDVARY
#define S_S HS_QPU_SIG_SMIMM
#define S_W HS_QPU_SIG_WRTMUC
static const int32_t SIG_TABLE[32] = {
    0,         S_T,       S_U,         S_T | S_U,  S_M,         S_T | S_M,  S_U | S_M,           S_T | S_U | S_M,
    S_V,       S_T | S_V, S_U | S_V,   S_T | S_U | S_V, HS_QPU_SIG_LDUNIFRF, S_T | HS_QPU_SIG_LDUNIFRF,
    S_S | S_V, S_S,       HS_QPU_SIG_LDTLB, HS_QPU_SIG_LDTLBU, S_W, S_T | S_W, S_V | S_W, S_T | S_V | S_W,
    HS_QPU_SIG_UCB, -1,  HS_QPU_SIG_LDUNIFA, HS_QPU_SIG_LDUNIFARF, -1, -1, -1, -1, -1, S_S | S_M,
};
#define SIG_WRITERS \
    (HS_QPU_SIG_LDTMU | HS_QPU_SIG_LDVARY | HS_QPU_SIG_LDUNIFRF | HS_QPU_SIG_LDTLB | HS_QPU_SIG_LDTLBU | \
     HS_QPU_SIG_LDUNIFARF)

hs_qpu_alu hs_qpu_alu_nop(void) {
    hs_qpu_alu a;
    memset(&a, 0, sizeof(a));
    a.add.op = HS_QPU_A_NOP;
    a.mul.op = HS_QPU_M_NOP;
    a.add.dst = a.add.a = a.add.b = HS_QPU_NOREG;
    a.mul.dst = a.mul.a = a.mul.b = HS_QPU_NOREG;
    a.sig_dst = HS_QPU_NOREG;
    return a;
}

/* destination -> (magic flag, waddr) */
static int dst_enc(hs_qpu_reg d, unsigned* magic, unsigned* waddr) {
    if (d == HS_QPU_NOREG) d = HS_QPU_NULL;
    unsigned k = d >> 8, n = d & 0xFF;
    if (k == 0 && n <= 5) { *magic = 1; *waddr = n; return 0; }
    if (k == 1 && n <= 63) { *magic = 0; *waddr = n; return 0; }
    if (k == 2 && hs_qpu_magic_name(n)) { *magic = 1; *waddr = n; return 0; }
    return HS_QPU_E_OPERAND;
}

typedef struct {
    int ra, rb; /* register-file index or -1 */
    int rb_imm; /* raddr_b holds a small immediate */
} raddrs;

static int src_mux(raddrs* r, hs_qpu_reg s, unsigned* mux) {
    unsigned k = s >> 8, n = s & 0xFF;
    if (s == HS_QPU_NOREG) return HS_QPU_E_OPERAND;
    if (k == 0 && n <= 5) { *mux = n; return 0; }
    if (k == 1 && n <= 63) {
        if (r->ra == (int)n) { *mux = MUX_RA; return 0; }
        if (!r->rb_imm && r->rb == (int)n) { *mux = MUX_RB; return 0; }
        if (r->ra < 0) { r->ra = (int)n; *mux = MUX_RA; return 0; }
        if (r->rb < 0) { r->rb = (int)n; *mux = MUX_RB; return 0; }
        return HS_QPU_E_RADDR;
    }
    if (k == 3 && n < 48) {
        if (r->rb < 0) { r->rb = (int)n; r->rb_imm = 1; *mux = MUX_RB; return 0; }
        if (r->rb_imm && r->rb == (int)n) { *mux = MUX_RB; return 0; }
        return HS_QPU_E_RADDR;
    }
    return HS_QPU_E_OPERAND;
}

/* operation -> opcode and muxes */
static int op_enc(const op_desc* d, hs_qpu_reg a, hs_qpu_reg b, raddrs* r, unsigned* opcode, unsigned* ma,
                  unsigned* mb) {
    int e;
    *opcode = d->opcode;
    switch (d->kind) {
    case K_NONE:
        if (a != HS_QPU_NOREG || b != HS_QPU_NOREG) return HS_QPU_E_OPERAND;
        *ma = d->mux_a;
        *mb = d->mux_b;
        return 0;
    case K_ONE:
        if (b != HS_QPU_NOREG) return HS_QPU_E_OPERAND;
        if ((e = src_mux(r, a, ma))) return e;
        *mb = d->mux_b;
        return 0;
    case K_MOV:
        if (b != HS_QPU_NOREG) return HS_QPU_E_OPERAND;
        if ((e = src_mux(r, a, ma))) return e;
        *mb = *ma;
        return 0;
    case K_TWO:
        if ((e = src_mux(r, a, ma)) || (e = src_mux(r, b, mb))) return e;
        return 0;
    case K_FTWO:
    case K_FORD0:
    case K_FORD1: {
        if ((e = src_mux(r, a, ma)) || (e = src_mux(r, b, mb))) return e;
        /* The pair (fadd, faddnf) shares an opcode, as does (fmin, fmax):
         * the hardware tells them apart by operand order. */
        if (d->kind != K_FTWO) {
            /* identical inputs have one order only: fine for fadd and fmin,
             * and fmax(x, x) == fmin(x, x); faddnf would silently become fadd */
            if (*ma == *mb && d->opcode == ADD_OPS[HS_QPU_A_FADDNF].opcode && d->kind == K_FORD1) return HS_QPU_E_ORDER;
            int ordered_high = F32_UNPACK_NONE * 8 + *ma > F32_UNPACK_NONE * 8 + *mb;
            if ((d->kind == K_FORD0 && ordered_high) || (d->kind == K_FORD1 && !ordered_high)) {
                unsigned t = *ma;
                *ma = *mb;
                *mb = t;
            }
        }
        *opcode = d->opcode + (F32_UNPACK_NONE << 2) + F32_UNPACK_NONE;
        return 0;
    }
    }
    return HS_QPU_E_OP;
}

static int cond_enc(const hs_qpu_alu* in, unsigned* cond) {
    unsigned ac = in->ac, mc = in->mc, af = in->af, mf = in->mf;
    if (ac > 4 || mc > 4 || af > 15 || mf > 15) return HS_QPU_E_COND;
    if (!ac && !mc) {
        if (af && mf) return HS_QPU_E_COND;
        *cond = af ? af : mf ? 16 | mf : 0;
    } else if (ac && !mc) {
        if (af || mf > 3) return HS_QPU_E_COND;
        *cond = 32 | (ac - 1) << 2 | mf;
    } else if (!ac && mc) {
        if (mf) return HS_QPU_E_COND;
        /* add push: 48..63; add update shares the 64.. block with ac + mc */
        *cond = af <= 3 ? 48 | (mc - 1) << 2 | af : 64 | (mc - 1) << 4 | af;
    } else {
        if (af || mf) return HS_QPU_E_COND;
        *cond = 64 | (mc - 1) << 4 | (ac - 1);
    }
    return 0;
}

static int is_null(hs_qpu_reg d) { return d == HS_QPU_NOREG || d == HS_QPU_NULL; }
static int is_tmu_reg(hs_qpu_reg d) {
    unsigned k = d >> 8, w = d & 0xFF;
    return k == 2 && ((w >= 10 && w <= 13) || (w >= 32 && w <= 46));
}
/* the destinations an instruction writes: add, mul, and a writing signal */
static int writes(const hs_qpu_alu* in, hs_qpu_reg out[3]) {
    int n = 0;
    if (!is_null(in->add.dst)) out[n++] = in->add.dst;
    if (!is_null(in->mul.dst)) out[n++] = in->mul.dst;
    if ((in->sig & SIG_WRITERS) && !is_null(in->sig_dst)) out[n++] = in->sig_dst;
    return n;
}
/* one instruction may not write a register twice, nor issue two TMU writes:
 * the order between them would be undefined */
static int write_conflicts(const hs_qpu_alu* in) {
    hs_qpu_reg w[3];
    int n = writes(in, w), tmu = 0;
    for (int i = 0; i < n; i++) {
        tmu += is_tmu_reg(w[i]);
        for (int j = i + 1; j < n; j++)
            if (w[i] == w[j]) return HS_QPU_E_WRITE;
    }
    return tmu > 1 ? HS_QPU_E_WRITE : 0;
}

int hs_qpu_pack_alu(const hs_qpu_alu* in, uint64_t* out) {
    if (in->add.op >= HS_QPU_A_COUNT || in->mul.op >= HS_QPU_M_COUNT) return HS_QPU_E_OP;
    raddrs r = {-1, -1, 0};
    unsigned aop, amux_a, amux_b, mop, mmux_a, mmux_b, ama, awa, mma, mwa, cond = 0;
    int e;
    if ((e = op_enc(&ADD_OPS[in->add.op], in->add.a, in->add.b, &r, &aop, &amux_a, &amux_b))) return e;
    if ((e = op_enc(&MUL_OPS[in->mul.op], in->mul.a, in->mul.b, &r, &mop, &mmux_a, &mmux_b))) return e;
    if ((e = dst_enc(in->add.dst, &ama, &awa)) || (e = dst_enc(in->mul.dst, &mma, &mwa))) return e;
    if ((e = write_conflicts(in))) return e;

    uint32_t sig = in->sig & ~(uint32_t)HS_QPU_SIG_SMIMM;
    if (r.rb_imm) sig |= HS_QPU_SIG_SMIMM;
    unsigned sig_idx = 32;
    for (unsigned i = 0; i < 32; i++)
        if (SIG_TABLE[i] == (int32_t)sig) { sig_idx = i; break; }
    if (sig_idx == 32) return HS_QPU_E_SIG;

    if (sig & SIG_WRITERS) {
        unsigned m, w;
        if (in->ac || in->mc || in->af || in->mf) return HS_QPU_E_COND;
        if (in->sig_dst == HS_QPU_NOREG || (e = dst_enc(in->sig_dst, &m, &w))) return HS_QPU_E_OPERAND;
        cond = m << 6 | w;
    } else if ((e = cond_enc(in, &cond))) {
        return e;
    }

    *out = (uint64_t)mop << SH_MUL_OP | (uint64_t)sig_idx << SH_SIG | (uint64_t)cond << SH_COND |
           (uint64_t)mma << SH_MM | (uint64_t)ama << SH_MA | (uint64_t)mwa << SH_WADDR_M |
           (uint64_t)awa << SH_WADDR_A | (uint64_t)aop << SH_ADD_OP | (uint64_t)mmux_b << SH_MUL_B |
           (uint64_t)mmux_a << SH_MUL_A | (uint64_t)amux_b << SH_ADD_B | (uint64_t)amux_a << SH_ADD_A |
           (uint64_t)(r.ra < 0 ? 0 : r.ra) << SH_RADDR_A | (uint64_t)(r.rb < 0 ? 0 : r.rb) << SH_RADDR_B;
    return 0;
}

/* ---- branches ----
 * bit 57 marks a branch; the byte offset is split: bits 3..23 at 35..55,
 * bits 24..31 at 24..31. cond 32..34, msfign 21..22, bdu 15..17, ub 14,
 * bdi 12..13 (1 = relative). */
int hs_qpu_pack_branch(int cond, int32_t rel, int ub, int ub_abs, uint64_t* out) {
    if (cond < 0 || cond > 7 || cond == 1) return HS_QPU_E_COND;
    if (rel & 7) return HS_QPU_E_RANGE;
    uint32_t off = (uint32_t)rel;
    unsigned bdu = ub && ub_abs ? 0 : 1;
    *out = 1ull << 57 | (uint64_t)((off >> 3) & 0x1FFFFF) << 35 | (uint64_t)cond << 32 |
           (uint64_t)((off >> 24) & 0xFF) << 24 | (uint64_t)bdu << 15 | (uint64_t)(ub ? 1 : 0) << 14 | 1ull << 12;
    return 0;
}

/* ---- programs ---- */
void hs_qpu_prog_init(hs_qpu_prog* p, uint64_t* buf, uint32_t cap) {
    memset(p, 0, sizeof(*p));
    p->code = buf;
    p->cap = cap;
    p->rules = 1;
    p->last_branch = -8;
}
void hs_qpu_prog_rules(hs_qpu_prog* p, int on) { p->rules = on; }

static void fail_at(hs_qpu_prog* p, int e, uint32_t at) {
    if (!p->err) {
        p->err = e;
        p->rule_at = at;
    }
}
/* a rule violation: always recorded; an error now if the rules are on, and
 * at hs_qpu_prog_finish if they are on by then */
static void violate(hs_qpu_prog* p, int e, uint32_t at) {
    if (!p->viol) {
        p->viol = e;
        p->viol_at = at;
    }
    if (p->rules) fail_at(p, e, at);
}

/* rule tracking, in program order, for the instruction about to be at p->n */
static void track(hs_qpu_prog* p, const hs_qpu_alu* in) {
    uint32_t idx = p->n;
    /* ldtmu reads the oldest outstanding load (before this instruction's writes) */
    if (in->sig & HS_QPU_SIG_LDTMU) {
        int ok = 0;
        if (p->nloads) {
            uint32_t l = p->loads[0];
            memmove(p->loads, p->loads + 1, (p->nloads - 1) * sizeof(p->loads[0]));
            p->nloads--;
            for (uint32_t i = 0; i < p->nthrsw; i++)
                if (p->thrsw[i] >= l && p->thrsw[i] + 3 <= idx) ok = 1;
        }
        if (!ok) violate(p, HS_QPU_E_RULE_LOAD, idx);
    }
    if (in->sig & HS_QPU_SIG_THRSW) {
        if (p->nthrsw == 16) {
            memmove(p->thrsw, p->thrsw + 1, 15 * sizeof(p->thrsw[0]));
            p->nthrsw--;
        }
        p->thrsw[p->nthrsw++] = idx;
    }
    /* at most one TMU write per instruction (hs_qpu_pack_alu), so slot order
     * cannot matter; a writing signal's destination counts like an ALU's */
    hs_qpu_reg w[3];
    int n = writes(in, w);
    for (int i = 0; i < n; i++) {
        if (!is_tmu_reg(w[i])) continue;
        if (w[i] == HS_QPU_TMUD) {
            p->tmud_pending++;
        } else if (w[i] == HS_QPU_TMUA || w[i] == HS_QPU_TMUAU) {
            if (p->tmud_pending) {
                p->store_pending = 1;
                p->tmud_pending = 0;
            } else if (p->nloads < 16) {
                p->loads[p->nloads++] = idx;
            } else {
                fail_at(p, HS_QPU_E_FULL, idx);
            }
        } else {
            violate(p, HS_QPU_E_RULE_UNCHECKED, idx);
        }
    }
    /* what clears a pending store: tmuwt, or the measured barrier form */
    if (in->add.op == HS_QPU_A_TMUWT ||
        (in->add.op == HS_QPU_A_BARRIERID && in->add.dst == HS_QPU_SYNCB && (in->sig & HS_QPU_SIG_THRSW)))
        p->store_pending = 0;
    /* branch delay slots: no thrsw (conservative) */
    if ((int32_t)idx - p->last_branch <= 3 && (in->sig & HS_QPU_SIG_THRSW)) fail_at(p, HS_QPU_E_BRANCH, idx);
}

void hs_qpu_emit(hs_qpu_prog* p, const hs_qpu_alu* in) {
    uint64_t w;
    int e = hs_qpu_pack_alu(in, &w);
    if (e) {
        fail_at(p, e, p->n);
        return;
    }
    if (p->n >= p->cap) {
        fail_at(p, HS_QPU_E_FULL, p->n);
        return;
    }
    track(p, in);
    p->code[p->n++] = w;
}

int hs_qpu_label_new(hs_qpu_prog* p) {
    if (p->nlabels >= HS_QPU_MAX_LABELS) {
        fail_at(p, HS_QPU_E_FULL, p->n);
        return -1;
    }
    p->label_at[p->nlabels] = -1;
    return (int)p->nlabels++;
}
void hs_qpu_label_bind(hs_qpu_prog* p, int label) {
    /* a label binds once: rebinding would silently move earlier branches */
    if (label < 0 || (uint32_t)label >= p->nlabels || p->label_at[label] >= 0) {
        fail_at(p, HS_QPU_E_LABEL, p->n);
        return;
    }
    p->label_at[label] = (int32_t)p->n;
}
void hs_qpu_branch(hs_qpu_prog* p, int cond, int label, int ub, int ub_abs) {
    if (label < 0 || (uint32_t)label >= p->nlabels) {
        fail_at(p, HS_QPU_E_LABEL, p->n);
        return;
    }
    if (cond < 0 || cond > 7 || cond == 1 || (ub != 0 && ub != 1) || (ub_abs != 0 && ub_abs != 1)) {
        fail_at(p, HS_QPU_E_COND, p->n);
        return;
    }
    if ((int32_t)p->n - p->last_branch <= 3) { /* a branch in another's delay slots */
        fail_at(p, HS_QPU_E_BRANCH, p->n);
        return;
    }
    if (p->n >= p->cap || p->nfix >= HS_QPU_MAX_FIXUPS) {
        fail_at(p, HS_QPU_E_FULL, p->n);
        return;
    }
    p->fix[p->nfix].at = p->n;
    p->fix[p->nfix].label = (uint16_t)label;
    p->fix[p->nfix].cond = (uint8_t)cond;
    p->fix[p->nfix].ub = (uint8_t)ub;
    p->fix[p->nfix].ub_abs = (uint8_t)ub_abs;
    p->nfix++;
    p->last_branch = (int32_t)p->n;
    p->code[p->n++] = 0; /* patched in hs_qpu_prog_finish */
}

void hs_qpu_emit_end(hs_qpu_prog* p) {
    static const uint8_t THR[8] = {1, 1, 0, 0, 1, 0, 0, 0};
    for (int i = 0; i < 8; i++) {
        hs_qpu_alu a = hs_qpu_alu_nop();
        if (THR[i]) a.sig = HS_QPU_SIG_THRSW;
        hs_qpu_emit(p, &a);
    }
}

int hs_qpu_prog_finish(hs_qpu_prog* p) {
    if (!p->err && p->nfix && p->n - (uint32_t)p->last_branch - 1 < 3) /* last branch lacks delay slots */
        fail_at(p, HS_QPU_E_BRANCH, (uint32_t)p->last_branch);
    for (uint32_t i = 0; i < p->nfix && !p->err; i++) {
        uint32_t l = p->fix[i].label;
        if (l >= p->nlabels || p->label_at[l] < 0) {
            fail_at(p, HS_QPU_E_LABEL, p->fix[i].at);
            break;
        }
        int64_t rel = ((int64_t)p->label_at[l] - (int64_t)p->fix[i].at - 4) * 8;
        if (rel < INT32_MIN || rel > INT32_MAX) {
            fail_at(p, HS_QPU_E_RANGE, p->fix[i].at);
            break;
        }
        int e = hs_qpu_pack_branch(p->fix[i].cond, (int32_t)rel, p->fix[i].ub, p->fix[i].ub_abs, &p->code[p->fix[i].at]);
        if (e) fail_at(p, e, p->fix[i].at);
    }
    /* rule 2, plus the conservative end checks: nothing left in flight */
    if (p->store_pending || p->tmud_pending || p->nloads) {
        if (!p->viol) {
            p->viol = HS_QPU_E_RULE_END;
            p->viol_at = p->n;
        }
    }
    if (!p->err && p->rules && p->viol) fail_at(p, p->viol, p->viol_at);
    return p->err ? p->err : (int)p->n;
}
