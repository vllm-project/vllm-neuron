#!/bin/bash
# usage: process.sh <profile-dir> <pid-glob|""> <session-idx> <run_name>   (ingest -> analyze6 -> timeline6)
# several queue workers may read the same queue file: a run is processed by the worker that claims it first
W=/mnt/nvme/ds41_profiling_opt10; PY=/mnt/nvme/ds41_profiling/venv/bin/python
mkdir $W/claims/$4 2>/dev/null || exit 0
echo "[$(date +%T)] ingest $4 (port ${INGEST_PORT:-3022}, cores $INGEST_CORES)" >> $W/process.log
$W/scripts/ingest.sh "$1" "$2" "$3" "$4" >> $W/process.log 2>&1
nice $PY $W/scripts/analyze6.py "$4" >> $W/process.log 2>&1 && { case "$4" in *_mr_*) ;; *) nice $PY $W/scripts/timeline6.py "$4" >> $W/process.log 2>&1;; esac; }
echo "[$(date +%T)] done $4" >> $W/process.log
