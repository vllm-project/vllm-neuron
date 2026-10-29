#!/bin/bash
# usage: enqueue_mr.sh <profile-dir> <cellname> <phase:session-idx>...   e.g. in1024_bs1_mr in1024_bs1_mr prefill:0 decode:1
W=/mnt/nvme/ds41_profiling_opt6; prof=$1; P=$W/profiles/$1; cell=$2; shift 2
for d in $P/output/i-*_pid_*; do
  pid=${d##*_pid_}
  r=$(grep -o "Worker_TP[0-9]*_EP[0-9]* pid=$pid)" $P/server.log | head -1 | sed 's/Worker_TP\([0-9]*\)_.*/\1/')
  for ps in "$@"; do ph=${ps%%:*}; si=${ps##*:}
    echo "$prof pid_$pid $si ${cell}_${ph}_r$r" >> $W/queue.txt
  done
done
tail -20 $W/queue.txt
