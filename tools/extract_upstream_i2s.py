#!/usr/bin/env python3
"""Extract upstream BitNet's I2_S producer and consumer for reference testing.

Upstream BitNet's own I2_S code is the authority on the byte layout. This
script pulls the two functions that define it straight out of the
vendored source at build time, so the reference test always runs against
the real implementation rather than a copy of it that could drift.

    extract_upstream_i2s.py <ggml-bitnet-mad.cpp> <out.c>

The two functions are:

    quantize_i2_s()             the producer -- writes the bytes
    ggml_vec_dot_i2_i8_s_1x1()   the consumer -- reads them

They are compiled as C. The only ggml symbols they reference are stubbed
out; neither stub affects the bytes or the arithmetic.

Upstream BitNet is MIT licensed. See experimental/BitNet/LICENSE.
"""
import sys

PROLOGUE = """/* GENERATED FILE -- do not edit.
 *
 * Extracted verbatim by tools/extract_upstream_i2s.py from
 * experimental/BitNet/src/ggml-bitnet-mad.cpp (Microsoft BitNet, MIT).
 *
 * It holds upstream's own I2_S producer and consumer, so a test can
 * check our layout conversion against the reference implementation
 * rather than against a copy of it.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/* The two ggml symbols the extracted functions reference. Neither
 * affects the bytes that are written or the arithmetic that is done. */
#define GGML_TYPE_I2_S 36
static size_t ggml_row_size(int type, int64_t n) { (void)type; return (size_t)n; }

/* upstream's own block size, verbatim from the preamble of
 * experimental/BitNet/src/ggml-bitnet-mad.cpp */
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
#define QK_I2_S 128
#elif defined(__ARM_NEON)
#define QK_I2_S 64
#endif

#if defined(__ARM_NEON)

"""

EPILOGUE = """

#else  /* !__ARM_NEON -- upstream's I2_S code has an ARM build only */

/* Stubs so the reference test compiles and skips cleanly elsewhere. */
static size_t quantize_i2_s(const float * src, void * dst, int64_t nrow,
                            int64_t n_per_row, const float * quant_weights) {
    (void)src; (void)dst; (void)nrow; (void)n_per_row; (void)quant_weights;
    return 0;
}
static void ggml_vec_dot_i2_i8_s_1x1(int n, float * s, size_t bs,
                                      const void * vx, size_t bx,
                                      const void * vy, size_t by, int nrc) {
    (void)n; (void)s; (void)bs; (void)vx; (void)bx; (void)vy; (void)by; (void)nrc;
}

#endif /* __ARM_NEON */
"""


def extract_function(src, signature):
    """Return the whole function starting at `signature`, brace-matched."""
    i = src.index(signature)
    j = src.index("{", i)
    depth = 0
    k = j
    while True:
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    return src[i:k + 1]


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: extract_upstream_i2s.py <in.cpp> <out.c>\n")
        return 2

    with open(sys.argv[1], "r", encoding="utf-8", errors="replace") as fh:
        src = fh.read()

    try:
        quantize = extract_function(src, "size_t quantize_i2_s(")
        dot = extract_function(src, "void ggml_vec_dot_i2_i8_s_1x1(")
    except ValueError as exc:
        sys.stderr.write("extract_upstream_i2s: %s\n" % exc)
        return 1

    with open(sys.argv[2], "w", encoding="utf-8") as fh:
        fh.write(PROLOGUE)
        fh.write(quantize)
        fh.write("\n\n")
        fh.write(dot)
        fh.write(EPILOGUE)

    return 0


if __name__ == "__main__":
    sys.exit(main())
