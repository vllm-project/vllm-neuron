#!/bin/bash
# Usage: bench_fixed.sh "<input_lens>" "<concurrencies>" [output_len=256]
# `vllm bench serve` over the fixed real-text prompts in bench_data/prompts
# (gen_bench_prompts.py), one cell per (input_len, concurrency) pair; the same
# matrix as bench_matrix.sh, but every run sends the same requests.
# Env: OUT_DIR (default ./results/bench/new), MODEL (tokenizer path / HF id),
#      BASE_URL (default http://127.0.0.1:8000), PROMPT_DIR (default bench_data/prompts).
HERE=$(cd "$(dirname "$0")" && pwd)
PROMPT_DIR=${PROMPT_DIR:-$HERE/../bench_data/prompts}
OUT_DIR=${OUT_DIR:-./results/bench/new}
mkdir -p $OUT_DIR
OUTPUT_LEN=${3:-256}
TMP=$(mktemp -d)
trap 'rm -rf $TMP' EXIT
for IN in $1; do
  for C in $2; do
    src=$PROMPT_DIR/in${IN}_c${C}.jsonl.gz
    [ -f $src ] || { echo "missing $src (generate it with gen_bench_prompts.py)"; exit 1; }
    # The custom dataset only reads plain .jsonl.
    P=$TMP/in${IN}_c${C}.jsonl
    gunzip -c $src > $P
    N=$(wc -l < $P)
    tag="in${IN}_out${OUTPUT_LEN}_c${C}"
    echo "=== $tag (num_prompts=$N)"
    vllm bench serve \
      --backend openai \
      --base-url "${BASE_URL:-http://127.0.0.1:8000}" \
      --model hy3 \
      --tokenizer "${MODEL:-tencent/Hy3-FP8}" \
      --dataset-name custom \
      --dataset-path $P \
      --skip-chat-template \
      --disable-shuffle \
      --custom-output-len $OUTPUT_LEN \
      --ignore-eos \
      --num-prompts $N \
      --max-concurrency $C \
      --percentile-metrics ttft,tpot,itl,e2el \
      --save-result --result-dir $OUT_DIR --result-filename $tag.json \
      2>&1 | grep -E "Successful|Benchmark duration|throughput|TTFT|TPOT|ITL" | grep -E "Mean|Median|P99|throughput|Successful|duration"
  done
done
