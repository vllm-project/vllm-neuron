#!/bin/bash
# BF16-KV vs FP8-KV acceptance run against a live server (same steps for both).
#
# Usage: kv_acceptance.sh <out_dir> [phases]   phases (default "mem bench math stab"):
#   mem     KV pool lines from the server log + a neuron-monitor memory snapshot
#   bench   PR load matrix: inputs 1024/4096/8192/16000 x concurrency 1/4/15, 256 out
#   math    full MATH-500, greedy (temperature 0), no_think, max 4096 tokens
#   math09  full MATH-500 at the PR's temperature 0.9 (seeded requests)
#   stab    30 min sustained load with prefix reuse (75% shared 8k prompts, c=8)
# Env: SERVE_LOG (server log), MODEL (tokenizer / ckpt), BASE_URL, CLIENT_PY
#      (python with openai + math-verify + datasets), STAB_PROMPTS (default 900),
#      PAUSE_PATTERN (pgrep -f pattern of host jobs, e.g. a CPU calibration, that
#      are SIGSTOPped during the latency phases bench / stab and resumed after)
set -uo pipefail
OUT=$1
PHASES=${2:-"mem bench math stab"}
HERE=$(cd "$(dirname "$0")" && pwd)
BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
CLIENT_PY=${CLIENT_PY:-python3}
mkdir -p "$OUT"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$OUT/acceptance.log"; }
quiet() { [ -n "${PAUSE_PATTERN:-}" ] && pkill -STOP -f "$PAUSE_PATTERN" && log "paused: $PAUSE_PATTERN"; true; }
loud() { [ -n "${PAUSE_PATTERN:-}" ] && pkill -CONT -f "$PAUSE_PATTERN" && log "resumed: $PAUSE_PATTERN"; true; }
trap loud EXIT

until [ "$(curl -s -m 3 "$BASE_URL/health" -o /dev/null -w '%{http_code}')" = 200 ]; do sleep 15; done
log "server healthy; phases: $PHASES"

for ph in $PHASES; do
  case $ph in
  mem)
    grep -a -E "KV cache size|Maximum concurrency|num_gpu_blocks|KV scale profile|kv_cache_dtype|fp8_packed" \
      "$SERVE_LOG" | sort -u > "$OUT/kv_pool_log.txt"
    timeout 30 neuron-monitor 2>/dev/null | head -n 2 > "$OUT/neuron_monitor.json"
    log "mem: $(grep -a -m1 'KV cache size' "$OUT/kv_pool_log.txt")"
    ;;
  bench)
    quiet; log "bench start"
    OUT_DIR="$OUT/bench" MODEL="$MODEL" BASE_URL="$BASE_URL" \
      bash "$HERE/bench_matrix.sh" "1024 4096 8192 16000" "1 4 15" 256 > "$OUT/bench.log" 2>&1
    log "bench done"; loud
    ;;
  math)
    log "MATH-500 greedy start"
    $CLIENT_PY -I "$HERE/eval_math500.py" no_think "$OUT/math500_greedy.jsonl" --temperature 0 \
      --concurrency 32 --max-tokens 4096 --base-url "$BASE_URL/v1" > "$OUT/math500_greedy.log" 2>&1
    log "$(grep '=== MATH-500' "$OUT/math500_greedy.log")"
    ;;
  math09)
    log "MATH-500 temp 0.9 start"
    $CLIENT_PY -I "$HERE/eval_math500.py" no_think "$OUT/math500_t09.jsonl" --temperature 0.9 \
      --seed 1234 --concurrency 32 --max-tokens 4096 --base-url "$BASE_URL/v1" > "$OUT/math500_t09.log" 2>&1
    log "$(grep '=== MATH-500' "$OUT/math500_t09.log")"
    ;;
  stab)
    quiet; log "stability start"
    before=$(grep -a -c -i -E "error|exception|nan" "$SERVE_LOG")
    vllm bench serve --backend openai --base-url "$BASE_URL" --model hy3 --tokenizer "$MODEL" \
      --dataset-name random --random-prefix-len 6144 --random-input-len 2048 --random-output-len 256 \
      --random-range-ratio 0 --ignore-eos --num-prompts "${STAB_PROMPTS:-900}" --seed 42 \
      --max-concurrency 8 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
      --save-result --result-dir "$OUT" --result-filename stability.json > "$OUT/stability.log" 2>&1
    after=$(grep -a -c -i -E "error|exception|nan" "$SERVE_LOG")
    curl -s "$BASE_URL/metrics" | grep -E "^vllm:(prefix_cache_(hits|queries)|num_preemptions|request_success)_total" \
      > "$OUT/stability_metrics.txt"
    log "stability done: $(grep -E 'Successful|Failed' "$OUT/stability.log" | tr '\n' ' ') server-log error/nan lines +$((after - before))"
    loud
    ;;
  esac
done
log "all phases done"
