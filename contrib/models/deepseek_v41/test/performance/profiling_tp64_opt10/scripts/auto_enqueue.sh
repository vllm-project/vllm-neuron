#!/bin/bash
# watches chain.log; when a capture ENDs, appends its sessions to queue_a.txt (several queue.sh workers share it)
W=/mnt/nvme/ds41_profiling_opt10; n=0; done_list="in1024_bs1_r0"
add() { echo "$*" >> $W/queue_a.txt; }
mr() {  # <profile> <cell> <phase:session>...
  local prof=$1 cell=$2; shift 2
  for d in $W/profiles/$prof/output/i-*_pid_*; do
    pid=${d##*_pid_}
    r=$(grep -o "Worker_TP[0-9]*_EP[0-9]* pid=$pid)" $W/profiles/$prof/server.log | head -1 | sed 's/Worker_TP\([0-9]*\)_.*/\1/')
    for ps in "$@"; do add "$prof pid_$pid ${ps##*:} ${cell}_mr_${ps%%:*}_r$r"; done
  done
}
while true; do
  for name in in1024_bs1_r0 in1024_bs10_r0 in8192_bs1_r0 in8192_bs10_r0 in8192_bs1_c2 in8192_bs10_c2 in1024_bs1_mr in8192_bs1_mr in1024_bs10_mr in8192_bs10_mr in8192_bs10_d2; do
    echo "$done_list" | grep -qw $name && continue
    grep -q "END $name " $W/chain.log 2>/dev/null || continue
    done_list="$done_list $name"
    cell=${name%_*}
    case $name in
      *_r0) add "$name - 0 ${cell}_prefill"
            if [ "$cell" = in8192_bs10 ]; then add "$name - 1 ${cell}_decode_s2"; else add "$name - 1 ${cell}_decode"; fi ;;
      *_c2) add "$name - 0 ${cell}_prefill_c4" ;;
      in1024_bs1_mr) mr $name $cell prefill:0 decode:1 ;;
      in8192_bs1_mr) mr $name $cell decode:1 ;;
      *_mr) mr $name $cell decode:0 ;;
      *_d2) add "$name - 0 ${cell}_decode" ;;
    esac
    echo "[$(date +%T)] enqueued $name" >> $W/process.log
  done
  grep -q "CHAIN DONE" $W/chain.log 2>/dev/null && [ $(echo $done_list | wc -w) -ge 11 ] && break
  sleep 30
done
echo STOP >> $W/queue_a.txt
echo "[$(date +%T)] auto_enqueue done" >> $W/process.log
