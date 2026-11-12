/* hs_qpu_kernels: see include/hs_qpu_kernels.h */
#include "hs_qpu_kernels.h"

#define R(n) HS_QPU_R(n)
#define RF(n) HS_QPU_RF(n)
#define X HS_QPU_NOREG
#define I(v) hs_qpu_imm_int(v)

static void emit(hs_qpu_prog* p, hs_qpu_alu x) { hs_qpu_emit(p, &x); }
static void nop(hs_qpu_prog* p) { emit(p, hs_qpu_alu_nop()); }
static void ldunifrf(hs_qpu_prog* p, int rf) { emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDUNIFRF, RF(rf))); }

int hs_qpu_kernel_store_one(hs_qpu_prog* p) {
    ldunifrf(p, 0);
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUD, I(1), X));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(0), X));
    emit(p, hs_qpu_a(HS_QPU_A_TMUWT, X, X, X));
    hs_qpu_emit_end(p);
    return hs_qpu_prog_finish(p);
}

int hs_qpu_kernel_gather(hs_qpu_prog* p, int batches, int k) {
    if ((batches != 1 && batches != 8) || (k != 1 && k != 2 && k != 4 && k != 8)) return HS_QPU_E_OPERAND;
    /* rf0 passes, rf1 idx address, rf2 table, rf3 out; rf4 stride, rf5 batch,
     * rf6 per-pass skip */
    for (int i = 0; i < 4; i++) ldunifrf(p, i);
    if (batches == 1) {
        emit(p, hs_qpu_a(HS_QPU_A_MOV, RF(5), I(0), X));
    } else {
        emit(p, hs_qpu_a(HS_QPU_A_TIDX, R(0), X, X));
        emit(p, hs_qpu_a(HS_QPU_A_SHR, R(0), R(0), I(2)));
        emit(p, hs_qpu_a(HS_QPU_A_BAND, RF(5), R(0), I(15)));
    }
    /* Layout: in pass q, batch b's load i reads
     *   base + (q * batches + b) * 64k + 64i + 4 * lane,
     * so each batch owns a contiguous 64k-byte run per pass. With k >= 4 it
     * owns whole 256-byte memory lines.
     * The earlier interleaved layout (batch b at 64b, stride 64 * batches)
     * shared each line between 4 QPUs. On sequential indices with k = 3..5
     * the QPUs drift apart beyond the ~16-32 KiB window in which the L2T
     * fetches a shared line once, and traffic grew to 2.4-2.8x: 19.4 ms
     * instead of 7.8 ms at k = 4.
     * Root-caused and measured 2026-11-16: QPU_ENCODER.md, tools/gather_anom.c. */
    emit(p, hs_qpu_a(HS_QPU_A_SHL, R(0), RF(5), I(6)));
    emit(p, hs_qpu_m(hs_qpu_alu_nop(), HS_QPU_M_UMUL24, R(0), R(0), I(k))); /* 64k * batch */
    emit(p, hs_qpu_a(HS_QPU_A_EIDX, R(1), X, X));
    emit(p, hs_qpu_a(HS_QPU_A_SHL, R(1), R(1), I(2)));
    emit(p, hs_qpu_a(HS_QPU_A_ADD, R(0), R(0), R(1)));
    emit(p, hs_qpu_a(HS_QPU_A_ADD, RF(1), RF(1), R(0)));
    emit(p, hs_qpu_a(HS_QPU_A_ADD, RF(3), RF(3), R(0)));
    emit(p, hs_qpu_a(HS_QPU_A_MOV, R(0), I(1), X));
    emit(p, hs_qpu_a(HS_QPU_A_SHL, RF(4), R(0), I(6))); /* 64 B between a batch's loads */
    emit(p, hs_qpu_a(HS_QPU_A_MOV, R(0), I(k), X));
    emit(p, hs_qpu_m(hs_qpu_alu_nop(), HS_QPU_M_UMUL24, R(0), R(0), I(batches - 1)));
    emit(p, hs_qpu_a(HS_QPU_A_SHL, RF(6), R(0), I(6))); /* skip the other batches' runs: 64k(batches-1) */
    int l = hs_qpu_label_new(p);
    hs_qpu_label_bind(p, l);
    /* k index loads; the last carries the thrsw rule 1 needs */
    for (int i = 0; i < k; i++) {
        hs_qpu_alu x = hs_qpu_m(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(1), X), HS_QPU_M_ADD, RF(1), RF(1), RF(4));
        emit(p, i == k - 1 ? hs_qpu_s(x, HS_QPU_SIG_THRSW, X) : x);
    }
    nop(p);
    nop(p);
    /* k gathers: table + 4 * idx */
    for (int i = 0; i < k; i++) {
        emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDTMU, R(0)));
        emit(p, hs_qpu_a(HS_QPU_A_SHL, R(0), R(0), I(2)));
        hs_qpu_alu x = hs_qpu_a(HS_QPU_A_ADD, HS_QPU_TMUA, R(0), RF(2));
        emit(p, i == k - 1 ? hs_qpu_s(x, HS_QPU_SIG_THRSW, X) : x);
    }
    nop(p);
    nop(p);
    /* k stores */
    for (int i = 0; i < k; i++) {
        emit(p, hs_qpu_s(hs_qpu_alu_nop(), HS_QPU_SIG_LDTMU, R(1)));
        emit(p, hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUD, R(1), X));
        emit(p, hs_qpu_m(hs_qpu_a(HS_QPU_A_MOV, HS_QPU_TMUA, RF(3), X), HS_QPU_M_ADD, RF(3), RF(3), RF(4)));
    }
    emit(p, hs_qpu_a(HS_QPU_A_ADD, RF(1), RF(1), RF(6)));
    emit(p, hs_qpu_a(HS_QPU_A_ADD, RF(3), RF(3), RF(6)));
    hs_qpu_alu s = hs_qpu_a(HS_QPU_A_SUB, RF(0), RF(0), I(1));
    s.af = HS_QPU_PF_PUSHZ;
    emit(p, s);
    hs_qpu_branch(p, HS_QPU_B_NA0, l, 0, 0);
    nop(p);
    nop(p);
    nop(p);
    emit(p, hs_qpu_a(HS_QPU_A_TMUWT, X, X, X)); /* rule 2 */
    hs_qpu_emit_end(p);
    return hs_qpu_prog_finish(p);
}
