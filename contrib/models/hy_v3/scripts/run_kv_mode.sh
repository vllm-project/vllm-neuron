#!/bin/bash
# One BF16-KV run and one FP8-KV run of validate_fp8_kv.py in the same mode on
# a reduced checkpoint, then compare_kv_runs.py. Logs and dumps go to <out_dir>.
#
# Usage: run_kv_mode.sh <fp8_ckpt> <profile.safetensors> <m1|m2|m3> <out_dir>
# Env:   NUM_BLOCKS (default 1024), PY (python of the serving venv),
#        FORCE_FALLBACK=1 adds an FP8 run with HY3_FP8_KV_FORCE_UNPACKED_DECODE=1
#        (packed cache read through the unpacked kernel + K write-back)
set -uo pipefail
CKPT=$1 PROFILE=$2 MODE=$3 OUT=$4
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python3}
mkdir -p "$OUT"
export NUM_BLOCKS=${NUM_BLOCKS:-1024}
export NEURON_SKIP_EFA_AFFINITY=${NEURON_SKIP_EFA_AFFINITY:-1}
for kv in auto fp8; do
  echo "=== $MODE kv=$kv $(date -u +%FT%TZ)"
  KV_DTYPE=$kv HY3_KV_SCALE_PROFILE=$PROFILE \
    $PY "$HERE/validate_fp8_kv.py" "$CKPT" "$OUT/$MODE.$kv.pt" "$MODE" > "$OUT/$MODE.$kv.log" 2>&1
  echo "exit $? ; $(grep -a '### ' "$OUT/$MODE.$kv.log" | tail -3)"
done
$PY "$HERE/compare_kv_runs.py" "$OUT/$MODE.auto.pt" "$OUT/$MODE.fp8.pt" | tee "$OUT/$MODE.compare.txt"
rc=${PIPESTATUS[0]}
if [ "${FORCE_FALLBACK:-0}" = 1 ]; then
  echo "=== $MODE kv=fp8 forced unpacked decode $(date -u +%FT%TZ)"
  KV_DTYPE=fp8 HY3_KV_SCALE_PROFILE=$PROFILE HY3_FP8_KV_FORCE_UNPACKED_DECODE=1 \
    $PY "$HERE/validate_fp8_kv.py" "$CKPT" "$OUT/$MODE.fp8fb.pt" "$MODE" > "$OUT/$MODE.fp8fb.log" 2>&1
  $PY "$HERE/compare_kv_runs.py" "$OUT/$MODE.auto.pt" "$OUT/$MODE.fp8fb.pt" | tee "$OUT/$MODE.fp8fb.compare.txt"
  rc=$(( rc | PIPESTATUS[0] ))
fi
exit $rc
