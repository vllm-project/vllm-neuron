#!/bin/bash
# Usage: bench_matrix.sh "<input_lens>" "<concurrencies>" [output_len=256]
# Runs `vllm bench serve` (random dataset) for each (input_len, concurrency) pair.
# Env: OUT_DIR (default ./results/bench/new), MODEL (tokenizer path / HF id),
#      BASE_URL (default http://127.0.0.1:8000), SEED_SALT.
OUT_DIR=${OUT_DIR:-./results/bench/new}
mkdir -p $OUT_DIR
OUTPUT_LEN=${3:-256}
for IN in $1; do
  for C in $2; do
    N=$(( C * 4 )); [ $N -lt 8 ] && N=8
    tag="in${IN}_out${OUTPUT_LEN}_c${C}"
    # Distinct prompts per cell (no prefix-cache reuse across cells), identical
    # across servers so baseline and DCP see the same requests.
    SEED=$(( IN * 1000 + C + ${SEED_SALT:-0} ))
    echo "=== $tag (num_prompts=$N)"
    vllm bench serve \
      --backend openai \
      --base-url "${BASE_URL:-http://127.0.0.1:8000}" \
      --model hy3 \
      --tokenizer "${MODEL:-tencent/Hy3-FP8}" \
      --dataset-name random \
      --random-input-len $IN \
      --random-output-len $OUTPUT_LEN \
      --random-range-ratio 0 \
      --ignore-eos \
      --num-prompts $N \
      --seed $SEED \
      --max-concurrency $C \
      --percentile-metrics ttft,tpot,itl,e2el \
      --save-result --result-dir $OUT_DIR --result-filename $tag.json \
      2>&1 | grep -E "Successful|Benchmark duration|throughput|TTFT|TPOT|ITL" | grep -E "Mean|Median|P99|throughput|Successful|duration"
  done
done
