#!/bin/bash
# Usage: tests/redteam-p0-5/traced.sh <trace-file> <cmd...>   (as root, inside wrap.sh)
# Runs cmd while reset_trace.bt records the kernel's V3D events to <trace-file>.
tr=$1; shift
bpftrace -q tests/redteam-p0-5/reset_trace.bt > "$tr" 2> "$tr.err" &
bp=$!
for i in $(seq 100); do grep -q '^# t_ns' "$tr" 2>/dev/null && break; sleep 0.1; done
"$@"; rc=$?
sleep 0.3
kill -INT $bp; wait $bp
exit $rc
