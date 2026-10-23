#!/bin/bash
# Standard serving benchmark for contrib models: `vllm bench serve` (random
# dataset) over an input-length x output-length x concurrency matrix, against a
# server that is already running. One JSON per cell, named
# in<ISL>_out<OSL>_c<C>.json; summarize.py turns one or more result dirs into
# tables. See README.md for the standard matrix and how to size the server.
#
# Usage:
#   SERVED_NAME=hy3 TOKENIZER=tencent/Hy3-FP8 LABEL=hy3-ep64 \
#     MAX_MODEL_LEN=9472 MAX_CONC=32 bash contrib/benchmark/bench_matrix.sh
#
# Env (defaults give the standard matrix):
#   SERVED_NAME    --served-model-name of the server (required)
#   TOKENIZER      tokenizer path or HF id (required)
#   LABEL          model / configuration tag, saved in every JSON and used as the
#                  default result dir (default: SERVED_NAME)
#   ISLS           input lengths            (default: "1024 8192")
#   OSLS           output lengths           (default: "256 1024")
#   CONCS          concurrencies            (default: "1 4 8 16 32")
#   MAX_CONC       skip concurrencies above this, e.g. the server's max_num_seqs
#   MAX_MODEL_LEN  skip cells whose longest request would not fit
#   INPUT_RATIO    input-length jitter, +/- fraction (default: 0.03)
#   OUTPUT_RATIO   output-length jitter (default: 0, so TPOT is over exact OSL)
#   NUM_WARMUPS    warmup requests per cell (default: the cell's concurrency)
#   BASE_URL       (default: http://127.0.0.1:8000)
#   OUT_DIR        (default: ./results/bench/$LABEL)
#   SEED_SALT      added to every cell's seed (default: 0)
#   EXTRA_METADATA space-separated KEY=VALUE pairs saved in every JSON
#                  (e.g. "tp=64 ep=64 sdk=2.33 backend=neuron_native")
set -u
: "${SERVED_NAME:?set SERVED_NAME to the --served-model-name of the server}"
: "${TOKENIZER:?set TOKENIZER to the tokenizer path or HF id}"
LABEL=${LABEL:-$SERVED_NAME}
ISLS=${ISLS:-1024 8192}
OSLS=${OSLS:-256 1024}
CONCS=${CONCS:-1 4 8 16 32}
INPUT_RATIO=${INPUT_RATIO:-0.03}
OUTPUT_RATIO=${OUTPUT_RATIO:-0}
BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
OUT_DIR=${OUT_DIR:-./results/bench/$LABEL}
mkdir -p "$OUT_DIR"

# Identical input lengths across a concurrent burst are not representative of
# real traffic, so the input length is jittered by default. The output length is
# exact (with --ignore-eos), so every request decodes the same number of tokens.
RATIO="{\"input\": $INPUT_RATIO, \"output\": $OUTPUT_RATIO}"
first=1
for IN in $ISLS; do
  for OUT in $OSLS; do
    if [ -n "${MAX_MODEL_LEN:-}" ]; then
      longest=$(python3 -c "import math; print(math.ceil($IN * (1 + $INPUT_RATIO)) + math.ceil($OUT * (1 + $OUTPUT_RATIO)))")
      if [ "$longest" -gt "$MAX_MODEL_LEN" ]; then
        echo "=== skip in${IN}_out${OUT}: up to $longest tokens > MAX_MODEL_LEN=$MAX_MODEL_LEN"
        continue
      fi
    fi
    for C in $CONCS; do
      if [ -n "${MAX_CONC:-}" ] && [ "$C" -gt "$MAX_CONC" ]; then
        continue
      fi
      N=$(( C * 4 )); [ $N -lt 8 ] && N=8
      tag="in${IN}_out${OUT}_c${C}"
      # Distinct prompts per cell (no prefix-cache reuse across cells), identical
      # across servers so different models / configurations see the same requests.
      SEED=$(( IN * 1000 + C + ${SEED_SALT:-0} ))
      ready=()
      [ $first = 1 ] && ready=(--ready-check-timeout-sec 600)
      first=0
      echo "=== $LABEL $tag (num_prompts=$N)"
      # shellcheck disable=SC2086
      vllm bench serve \
        --backend openai \
        --base-url "$BASE_URL" \
        --model "$SERVED_NAME" \
        --tokenizer "$TOKENIZER" \
        --dataset-name random \
        --random-input-len "$IN" \
        --random-output-len "$OUT" \
        --random-range-ratio "$RATIO" \
        --ignore-eos \
        --num-prompts $N \
        --num-warmups "${NUM_WARMUPS:-$C}" \
        --seed $SEED \
        --max-concurrency "$C" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,90,99 \
        ${ready[@]+"${ready[@]}"} \
        --label "$LABEL" \
        --metadata label="$LABEL" isl="$IN" osl="$OUT" concurrency="$C" \
                   input_ratio="$INPUT_RATIO" output_ratio="$OUTPUT_RATIO" ${EXTRA_METADATA:-} \
        --save-result --result-dir "$OUT_DIR" --result-filename "$tag.json" \
        2>&1 | grep -E "Successful|Failed|Benchmark duration|throughput|TTFT|TPOT|ITL" | grep -E "Mean|Median|P99|throughput|Successful|Failed|duration"
    done
  done
done
