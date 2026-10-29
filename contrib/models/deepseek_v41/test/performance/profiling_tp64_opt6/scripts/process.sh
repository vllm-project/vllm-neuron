#!/bin/bash
# usage: process.sh <profile-dir> <pid-glob|""> <session-idx> <run_name>   (ingest -> analyze6 -> timeline6)
W=/mnt/nvme/ds41_profiling_opt6; PY=/mnt/nvme/ds41_profiling/venv/bin/python
echo "[$(date +%T)] ingest $4" >> $W/process.log
$W/scripts/ingest.sh "$1" "$2" "$3" "$4" >> $W/process.log 2>&1
nice $PY $W/scripts/analyze6.py "$4" >> $W/process.log 2>&1 && { case "$4" in *_mr_*) ;; *) nice $PY $W/scripts/timeline6.py "$4" >> $W/process.log 2>&1;; esac; }
echo "[$(date +%T)] done $4" >> $W/process.log
