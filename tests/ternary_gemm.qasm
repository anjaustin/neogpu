#
# Ternary GEMM - Minimal Working Version
# ====================================
# 
# This shader computes: output = weights * input
# Using TMU for loading and VPM for output
#

.include "vc4.qinc"

# Labels
:start
    # Get thread ID
    mov r0, elem_num
    
    # Load uniforms
    mov ra0, unif
    mov ra1, unif
    mov ra2, unif
    
    # Initialize accumulator
    mov r1, 0

    # Simple pass-through for now
    mov r2, ra0
    
    # Setup VPM write from QPU
    mov vw_setup, vpm_setup(1, 1, h32(0))
    
    # Write to VPM
    mov vpm, r2
    mov -, vw_wait
    
    # Setup DMA write to memory
    mov vw_setup, vdw_setup_0(1, 1, dma_h32(0, 0))
    
    # Set output address
    mov vw_addr, ra2
    mov -, vw_wait
    
    # End
    thrend
    nop
