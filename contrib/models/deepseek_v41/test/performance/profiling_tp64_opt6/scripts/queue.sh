#!/bin/bash
# processes lines "<profile-dir> <pid-glob|-> <session-idx> <run_name>" of queue.txt in order; "STOP" ends it
W=/mnt/nvme/ds41_profiling_opt6; n=0
while true; do
  line=$(sed -n "$((n+1))p" $W/queue.txt 2>/dev/null)
  if [ -z "$line" ]; then sleep 20; continue; fi
  [ "$line" = "STOP" ] && break
  set -- $line; pid=$2; [ "$pid" = "-" ] && pid=""
  $W/scripts/process.sh "$1" "$pid" "$3" "$4"
  n=$((n+1))
done
echo "[$(date +%T)] queue stopped" >> $W/process.log
