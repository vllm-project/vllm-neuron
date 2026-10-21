#!/bin/bash
# Prefix-cache benchmark: fixed total input length, a shared prefix of varying size.
#
# Usage: bench_prefix_cache.sh "<total_input_lens>" "<prefix_percents>" "<concurrencies>" [output_len=256]
#   e.g. bench_prefix_cache.sh "8192 16000" "0 25 50 75 90" "1 8"
# Every request of a cell shares one random prefix (the first request fills the
# cache); a distinct seed per cell keeps cells from hitting each other's
# prefixes. The measured hit rate comes from the server's prefix-cache counters
# (/metrics) before and after each cell and is written next to the result JSON.
# Env: OUT_DIR (default ./results/bench/prefix_cache), MODEL (tokenizer),
#      BASE_URL (default http://127.0.0.1:8000), NUM_PROMPTS (default 16).
set -uo pipefail
OUT_DIR=${OUT_DIR:-./results/bench/prefix_cache}
BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
OUTPUT_LEN=${4:-256}
mkdir -p "$OUT_DIR"

counter() {  # sum of a Prometheus counter over all label sets
  curl -s "$BASE_URL/metrics" | awk -v n="$1" '$1 ~ "^"n"(_total)?([{]|$)" {s += $NF} END {printf "%d", s}'
}

for TOTAL in $1; do
  for PCT in $2; do
    PREFIX=$(( TOTAL * PCT / 100 ))
    SUFFIX=$(( TOTAL - PREFIX ))
    for C in $3; do
      tag="in${TOTAL}_prefix${PCT}_c${C}"
      q0=$(counter vllm:prefix_cache_queries); h0=$(counter vllm:prefix_cache_hits)
      echo "=== $tag (prefix $PREFIX + random $SUFFIX tokens)"
      vllm bench serve \
        --backend openai \
        --base-url "$BASE_URL" \
        --model hy3 \
        --tokenizer "${MODEL:-tencent/Hy3-FP8}" \
        --dataset-name random \
        --random-prefix-len $PREFIX \
        --random-input-len $SUFFIX \
        --random-output-len $OUTPUT_LEN \
        --random-range-ratio 0 \
        --ignore-eos \
        --num-prompts "${NUM_PROMPTS:-16}" \
        --seed $(( TOTAL * 1000 + PCT * 10 + C + 7 )) \
        --max-concurrency $C \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result --result-dir "$OUT_DIR" --result-filename $tag.json \
        2>&1 | grep -E "Median TTFT|Median TPOT|Output token throughput|Total token throughput"
      q1=$(counter vllm:prefix_cache_queries); h1=$(counter vllm:prefix_cache_hits)
      python3 - "$OUT_DIR/$tag.json" $((q1 - q0)) $((h1 - h0)) $PREFIX <<'EOF'
import json, sys
path, q, h, prefix = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
d = json.load(open(path))
d["prefix_len"] = prefix
d["prefix_cache_queried_tokens"] = q
d["prefix_cache_hit_tokens"] = h
d["prefix_cache_hit_rate"] = h / q if q else 0.0
json.dump(d, open(path, "w"))
print(f"measured hit rate {d['prefix_cache_hit_rate']:.1%} ({h}/{q} tokens)")
EOF
    done
  done
done
