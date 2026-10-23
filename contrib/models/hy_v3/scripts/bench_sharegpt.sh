#!/bin/bash
# Usage: bench_sharegpt.sh "<concurrencies>"
# `vllm bench serve` on ShareGPT V3 with the dataset's output lengths
# (ignore_eos), 16 x concurrency requests (min 32), a fixed seed per cell.
# vLLM keeps only ShareGPT samples with prompt <= 1024 and prompt + output <= 2048
# tokens, so this covers short requests only; use bench_fixed.sh for long inputs.
# Env: SHAREGPT (path to ShareGPT_V3_unfiltered_cleaned_split.json, from the
#      anon8231489123/ShareGPT_Vicuna_unfiltered HF dataset), OUT_DIR, MODEL, BASE_URL.
OUT_DIR=${OUT_DIR:-./results/bench/sharegpt}
mkdir -p $OUT_DIR
for C in $1; do
  N=$(( C * 16 )); [ $N -lt 32 ] && N=32
  tag="sharegpt_c${C}"
  echo "=== $tag (num_prompts=$N)"
  vllm bench serve \
    --backend openai \
    --base-url "${BASE_URL:-http://127.0.0.1:8000}" \
    --model hy3 \
    --tokenizer "${MODEL:-tencent/Hy3-FP8}" \
    --dataset-name sharegpt \
    --dataset-path "${SHAREGPT:?set SHAREGPT to ShareGPT_V3_unfiltered_cleaned_split.json}" \
    --ignore-eos \
    --num-prompts $N \
    --seed $(( 1000 + C )) \
    --max-concurrency $C \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-dir $OUT_DIR --result-filename $tag.json \
    2>&1 | grep -E "Successful|Benchmark duration|throughput|TTFT|TPOT|ITL" | grep -E "Mean|Median|P99|throughput|Successful|duration"
done
