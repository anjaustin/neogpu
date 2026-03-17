# Ternary GEMM QPU shader
# ======================
# Uses vc4asm from /tmp/vc4asm/build/vc4asm
#
# To compile:
#   /tmp/vc4asm/build/vc4asm -V -o ternary.bin ternary.qasm

# Entry point
mov r0, elem_num
mov ra0, unif

# Simple pass-through for now
mov r1, ra0

# End
thrend
nop
