#!/bin/bash
# usage: queue.sh <queue file> <cores>: processes lines "<profile-dir> <pid-glob|-> <session-idx> <run_name>" in order; "STOP" ends it
W=/mnt/nvme/ds41_profiling_opt10; Q=$1; export INGEST_CORES=$2; n=0
while true; do
  line=$(sed -n "$((n+1))p" $Q 2>/dev/null)
  if [ -z "$line" ]; then sleep 20; continue; fi
  [ "$line" = "STOP" ] && break
  set -- $line; pid=$2; [ "$pid" = "-" ] && pid=""
  $W/scripts/process.sh "$1" "$pid" "$3" "$4"
  n=$((n+1))
done
echo "[$(date +%T)] queue $Q stopped" >> $W/process.log
