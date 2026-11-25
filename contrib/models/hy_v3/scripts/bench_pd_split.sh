#!/bin/bash
# Usage: bench_pd_split.sh "<concurrencies>" [decode_output_len=256]
# Prefill-only / decode-only split benchmark on the fixed shared-prefix prompts
# in bench_data/prefix (in85000_p78336: 85000 tokens, 92.2% shared prefix),
# against one running server (scripts/serve_256k.sh).
#
#   P (prefill only): output 1 token, so TTFT = queueing + prefill. Per
#     concurrency C its own file (own prefix), 10 x C requests, 10 warm-ups.
#   D (decode only): the C=8 file (one prefix, 80 requests) sent twice with
#     the same command; the first pass fills the prefix cache, the second hits
#     ~100% and leaves only the last-token re-compute before decoding.
#
# Flags follow the GPU P/D test (openai-chat, server-side chat template,
# --num-warmups, temperature 0.9 for D). Additions: one cache_salt per cell so
# runs and cells never hit each other's blocks, and the measured hit rate from
# /metrics around each run, written into the result JSON.
# On Neuron prefill and decode never share a step, so the D second pass still
# stalls decode on each arrival's last-token prefill: read its ITL as the pure
# decode step (TPOT - ITL is that stall). With MTP each streamed chunk is one
# step of 1..k+1 tokens, so ITL is the step time and TPOT the per-token time;
# the MTP acceptance (draft / accepted tokens from /metrics) is recorded too.
# Env: OUT_DIR (default ./results/bench/pd_split), MODEL (tokenizer path / HF id),
#      BASE_URL (default http://127.0.0.1:8000), PROMPT_DIR (default bench_data/prefix),
#      CELL (default in85000_p78336), SALT (default run-<unix time>).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PROMPT_DIR=${PROMPT_DIR:-$HERE/../bench_data/prefix}
OUT_DIR=${OUT_DIR:-./results/bench/pd_split}
BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
CELL=${CELL:-in85000_p78336}
SALT=${SALT:-run-$(date +%s)}
MODEL=${MODEL:-tencent/Hy3-FP8}
D_OUTPUT_LEN=${2:-256}
mkdir -p "$OUT_DIR/P" "$OUT_DIR/D"
TMP=$(mktemp -d)
trap 'rm -rf $TMP' EXIT

counter() {  # sum of a Prometheus counter over all label sets
  curl -s "$BASE_URL/metrics" | awk -v n="$1" '$1 ~ "^"n"(_total)?([{]|$)" {s += $NF} END {printf "%d", s}'
}

to_jsonl() {  # prefix + suffix of every request (warm-up row included) -> custom .jsonl
  python3 - "$1" "$2" <<'EOF'
import gzip, json, sys
with gzip.open(sys.argv[1], "rt") as f:
    rows = [json.loads(line) for line in f]
with open(sys.argv[2], "w") as f:
    for r in rows[1:]:
        f.write(json.dumps({"prompt": rows[0]["prefix"] + r["suffix"]}, ensure_ascii=False) + "\n")
EOF
}

bench() {  # bench <out.json> <salt> <vllm bench serve args...>; records hit rate and MTP acceptance
  local out=$1 salt=$2; shift 2
  local q0 h0 q1 h1 s0 d0 a0 s1 d1 a1
  q0=$(counter vllm:prefix_cache_queries); h0=$(counter vllm:prefix_cache_hits)
  s0=$(counter vllm:spec_decode_num_drafts); d0=$(counter vllm:spec_decode_num_draft_tokens)
  a0=$(counter vllm:spec_decode_num_accepted_tokens)
  vllm bench serve --backend openai-chat \
    --base-url "$BASE_URL" --endpoint /v1/chat/completions \
    --model "$MODEL" --served-model-name hy3 --tokenizer "$MODEL" --trust-remote-code \
    --dataset-name custom --skip-chat-template --request-rate inf --ignore-eos \
    --extra-body "{\"cache_salt\": \"$salt\"}" --metric-percentiles 50,99 \
    --save-result --result-dir "$(dirname "$out")" --result-filename "$(basename "$out")" \
    "$@" 2>&1 | grep -E "Successful|duration|P50 (TTFT|TPOT|ITL)|P99 TTFT|Request throughput"
  q1=$(counter vllm:prefix_cache_queries); h1=$(counter vllm:prefix_cache_hits)
  s1=$(counter vllm:spec_decode_num_drafts); d1=$(counter vllm:spec_decode_num_draft_tokens)
  a1=$(counter vllm:spec_decode_num_accepted_tokens)
  python3 - "$out" $((q1 - q0)) $((h1 - h0)) "$salt" $((s1 - s0)) $((d1 - d0)) $((a1 - a0)) <<'EOF'
import json, sys
path, q, h, salt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
steps, drafts, accepted = (int(x) for x in sys.argv[5:8])
with open(path) as f:
    d = json.load(f)
d["cache_salt"] = salt
d["prefix_cache_queried_tokens"] = q
d["prefix_cache_hit_tokens"] = h
d["prefix_cache_hit_rate"] = h / q if q else 0.0
if drafts:
    d["spec_decode_steps"] = steps
    d["spec_decode_draft_tokens"] = drafts
    d["spec_decode_accepted_tokens"] = accepted
    d["spec_decode_acceptance_rate"] = accepted / drafts
    d["spec_decode_tokens_per_step"] = 1 + accepted / steps
with open(path, "w") as f:
    json.dump(d, f)
print(f"measured hit rate (incl. warm-ups) {d['prefix_cache_hit_rate']:.1%} ({h}/{q} tokens)")
if drafts:
    print(f"MTP acceptance (incl. warm-ups) {accepted / drafts:.1%}, "
          f"{1 + accepted / steps:.2f} tokens per step")
EOF
}

for C in $1; do
  src=$PROMPT_DIR/${CELL}_c${C}.jsonl.gz
  [ -f "$src" ] || { echo "missing $src (generate it with gen_prefix_prompts.py)"; exit 1; }
  to_jsonl "$src" "$TMP/p_c$C.jsonl"
  echo "=== P ${CELL}_c$C ($((10 * C)) requests, output 1)"
  bench "$OUT_DIR/P/c$C.json" "$SALT-P-c$C" \
    --dataset-path "$TMP/p_c$C.jsonl" --custom-output-len 1 \
    --num-prompts $((10 * C)) --max-concurrency $C --num-warmups 10 \
    --percentile-metrics ttft,e2el
done

to_jsonl "$PROMPT_DIR/${CELL}_c8.jsonl.gz" "$TMP/replay.jsonl"
for C in $1; do
  for pass in 1 2; do
    echo "=== D ${CELL} replay c$C pass $pass (80 requests, output $D_OUTPUT_LEN)"
    bench "$OUT_DIR/D/c${C}_pass$pass.json" "$SALT-D-c$C" \
      --dataset-path "$TMP/replay.jsonl" --temperature 0.9 --seed 0 \
      --custom-output-len $D_OUTPUT_LEN --num-prompts 80 --max-concurrency $C \
      --num-warmups 2 --percentile-metrics ttft,tpot,itl,e2el
  done
done
