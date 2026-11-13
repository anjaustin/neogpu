#!/bin/bash
# Sum on-CPU ns (schedstat field 1) of every thread on the system, grouped as
# kworker / irq / probe / other. Usage: p0_1_redteam_schedstat.sh
for t in /proc/[0-9]*/task/[0-9]*; do
  c=$(cat $t/comm 2>/dev/null) || continue; s=$(cut -d" " -f1 $t/schedstat 2>/dev/null) || continue
  case "$c" in kworker/u*) g=kworker_unbound;; kworker/*) g=kworker_percpu;; irq/*) g=irqthread;; neogpu_kworker*) g=neogpu_kworker;; *) g=other;; esac
  echo "$g $s"
done | awk '{a[$1]+=$2} END{for(k in a) printf "%s %d\n", k, a[k]}' | sort
