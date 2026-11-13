#!/bin/bash
# Usage: bench_prefix_fixed.sh "<concurrencies>" [output_len=256]
# Prefix-cache benchmark over the fixed real-text shared-prefix prompts in
# bench_data/prefix (gen_prefix_prompts.py): per concurrency C, 10 x C requests
# that share one prefix (in85000_p78336: 78336 of 85000 tokens, 92.2%).
# Before each cell one warm-up request (same prefix, its own suffix, not
# measured) fills the cache, so every measured request hits the prefix and
# misses its suffix. All requests of a run carry one cache_salt (SALT), which
# keeps a rerun on the same server from hitting the previous run's blocks.
# The measured hit rate comes from the server's prefix-cache counters (/metrics)
# around each cell and is written into the result JSON.
# Env: OUT_DIR (default ./results/bench/prefix_fixed), MODEL (tokenizer),
#      BASE_URL (default http://127.0.0.1:8000), PROMPT_DIR (default bench_data/prefix),
#      CELL (default in85000_p78336), SALT (default run-<unix time>).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PROMPT_DIR=${PROMPT_DIR:-$HERE/../bench_data/prefix}
OUT_DIR=${OUT_DIR:-./results/bench/prefix_fixed}
BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
CELL=${CELL:-in85000_p78336}
SALT=${SALT:-run-$(date +%s)}
OUTPUT_LEN=${2:-256}
mkdir -p "$OUT_DIR"
TMP=$(mktemp -d)
trap 'rm -rf $TMP' EXIT

counter() {  # sum of a Prometheus counter over all label sets
  curl -s "$BASE_URL/metrics" | awk -v n="$1" '$1 ~ "^"n"(_total)?([{]|$)" {s += $NF} END {printf "%d", s}'
}

for C in $1; do
  src=$PROMPT_DIR/${CELL}_c${C}.jsonl.gz
  [ -f "$src" ] || { echo "missing $src (generate it with gen_prefix_prompts.py)"; exit 1; }
  tag="${CELL}_out${OUTPUT_LEN}_c${C}"
  # prefix + suffix -> warm-up request body and the custom-dataset .jsonl.
  python3 - "$src" "$TMP/warmup.json" "$TMP/$tag.jsonl" "$SALT" <<'EOF'
import gzip, json, sys
src, warm, out, salt = sys.argv[1:]
with gzip.open(src, "rt") as f:
    rows = [json.loads(line) for line in f]
prefix = rows[0]["prefix"]
with open(warm, "w") as f:
    json.dump({"model": "hy3", "prompt": prefix + rows[1]["suffix"], "max_tokens": 1,
               "cache_salt": salt}, f)
with open(out, "w") as f:
    for r in rows[2:]:
        f.write(json.dumps({"prompt": prefix + r["suffix"]}, ensure_ascii=False) + "\n")
EOF
  N=$(wc -l < "$TMP/$tag.jsonl")
  echo "=== $tag (warm-up + $N requests, cache_salt=$SALT)"
  curl -sf "$BASE_URL/v1/completions" -H "Content-Type: application/json" \
    -d @"$TMP/warmup.json" > /dev/null || { echo "warm-up request failed"; exit 1; }
  q0=$(counter vllm:prefix_cache_queries); h0=$(counter vllm:prefix_cache_hits)
  vllm bench serve \
    --backend openai \
    --base-url "$BASE_URL" \
    --model hy3 \
    --tokenizer "${MODEL:-tencent/Hy3-FP8}" \
    --dataset-name custom \
    --dataset-path "$TMP/$tag.jsonl" \
    --skip-chat-template \
    --disable-shuffle \
    --custom-output-len $OUTPUT_LEN \
    --ignore-eos \
    --num-prompts $N \
    --max-concurrency $C \
    --ready-check-timeout-sec 0 \
    --extra-body "{\"cache_salt\": \"$SALT\"}" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-dir "$OUT_DIR" --result-filename $tag.json \
    2>&1 | grep -E "Successful|Benchmark duration|throughput|TTFT|TPOT|ITL" | grep -E "Mean|Median|P99|throughput|Successful|duration"
  q1=$(counter vllm:prefix_cache_queries); h1=$(counter vllm:prefix_cache_hits)
  python3 - "$OUT_DIR/$tag.json" $((q1 - q0)) $((h1 - h0)) "$SALT" <<'EOF'
import json, sys
path, q, h, salt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
with open(path) as f:
    d = json.load(f)
d["cache_salt"] = salt
d["prefix_cache_queried_tokens"] = q
d["prefix_cache_hit_tokens"] = h
d["prefix_cache_hit_rate"] = h / q if q else 0.0
with open(path, "w") as f:
    json.dump(d, f)
print(f"measured hit rate {d['prefix_cache_hit_rate']:.1%} ({h}/{q} tokens)")
EOF
done
