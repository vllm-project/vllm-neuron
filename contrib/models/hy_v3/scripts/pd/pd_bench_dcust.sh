#!/bin/bash
# pd_bench_dcust.sh <out_dir> [concurrencies="1 2 4 6 8"] [output_len=1200]
# The customer's GPU "D 单测" (BL-M2: replay, D hits the cache and only decodes) through the PD
# proxy: 84836-token-class inputs, ~1.2K outputs, max(40, 10 x C) requests per C, mean / p99.
#   1. warm pass: all 80 requests of the c8 shared-prefix file (one 78336-token prefix + 80
#      suffixes), output 1, so both the prefill and the decode node cache every prompt in full.
#   2. per C: the first max(40, 10 x C) of those requests, same cache_salt (~100% hit on both
#      nodes, nothing to transfer), output OUTPUT_LEN tokens (--ignore-eos), 2 warm-ups.
# Flags as the PR's bench_pd_split.sh (openai-chat, server-side chat template, temperature 0.9,
# seed 0, --request-rate inf). Counters from the prefill / decode servers' /metrics.
# Env: CODE, MODEL, PROXY, P_URL / D_URL (or P_HOST / D_HOST), SALT, CELL (in85000_p78336),
#      PROMPTS_DIR (default $CODE/contrib/models/hy_v3/bench_data/prefix).
set -uo pipefail
OUT=${1:?usage: pd_bench_dcust.sh <out_dir> [concurrencies] [output_len]}
CONCS=${2:-1 2 4 6 8}
OUTPUT_LEN=${3:-1200}
HERE=$(cd "$(dirname "$0")" && pwd)
CODE=${CODE:-/mnt/nvme/src/pr73}
MODEL=${MODEL:-/mnt/nvme/models/Hy3-FP8}
PROXY=${PROXY:-http://127.0.0.1:8000}
P_URL=${P_URL:-http://${P_HOST:-127.0.0.1}:8100}
D_URL=${D_URL:-http://${D_HOST:-127.0.0.1}:8200}
CELL=${CELL:-in85000_p78336}
PROMPTS=${PROMPTS_DIR:-$CODE/contrib/models/hy_v3/bench_data/prefix}
SALT=${SALT:-dcust-$(date +%s)}
mkdir -p "$OUT"
TMP=$(mktemp -d); trap 'rm -rf $TMP' EXIT

counter() { curl -s "$1/metrics" | awk -v n="$2" '$1 ~ "^"n"(_total)?([{]|$)" {s += $NF} END {printf "%d", s}'; }
snap() {
  eval "$1_pq=$(counter $P_URL vllm:prefix_cache_queries) $1_ph=$(counter $P_URL vllm:prefix_cache_hits)"
  eval "$1_dq=$(counter $D_URL vllm:prefix_cache_queries) $1_dh=$(counter $D_URL vllm:prefix_cache_hits)"
  eval "$1_sd=$(counter $D_URL vllm:spec_decode_num_drafts) $1_sa=$(counter $D_URL vllm:spec_decode_num_accepted_tokens) $1_st=$(counter $D_URL vllm:spec_decode_num_draft_tokens)"
}
bench() {  # bench <out.json> <vllm bench serve args...>
  local out=$1; shift
  snap a
  vllm bench serve --backend openai-chat --base-url "$PROXY" --endpoint /v1/chat/completions \
    --model "$MODEL" --served-model-name hy3 --tokenizer "$MODEL" --trust-remote-code \
    --dataset-name custom --skip-chat-template --disable-shuffle --request-rate inf --ignore-eos \
    --ready-check-timeout-sec 0 --extra-body "{\"cache_salt\": \"$SALT\"}" --metric-percentiles 50,99 \
    --temperature 0.9 --seed 0 --save-result --result-dir "$(dirname "$out")" --result-filename "$(basename "$out")" \
    "$@" 2>&1 | grep -E "Successful|Failed|duration|Mean (TTFT|TPOT)|P99 (TTFT|TPOT)|Request throughput|Output token"
  snap b
  python3 - "$out" $((b_pq - a_pq)) $((b_ph - a_ph)) $((b_dq - a_dq)) $((b_dh - a_dh)) \
    $((b_sd - a_sd)) $((b_sa - a_sa)) $((b_st - a_st)) <<'EOF'
import json, sys
path = sys.argv[1]
pq, ph, dq, dh, sd, sa, st = map(int, sys.argv[2:])
d = json.load(open(path))
d.update(prefill_hit_rate=ph / pq if pq else 0.0, decode_hit_rate=dh / dq if dq else 0.0)
if sd:
    d.update(spec_tokens_per_step=1 + sa / sd, spec_accept_rate=sa / st if st else 0.0)
dur, n = d["duration"], d["completed"]
d["qpm"] = 60 * n / dur
d["tpm"] = 60 * (d["total_input_tokens"] + d["total_output_tokens"]) / dur
json.dump(d, open(path, "w"))
print(f"hit rate prefill {d['prefill_hit_rate']:.1%}, decode local {d['decode_hit_rate']:.1%}"
      + (f"; MTP {d['spec_tokens_per_step']:.2f} tokens/step" if sd else "")
      + f"; QPM {d['qpm']:.1f}, TPM {d['tpm']:,.0f}")
EOF
}

for u in $P_URL/health $D_URL/health; do curl -sf "$u" >/dev/null || { echo "not up: $u"; exit 1; }; done
for i in $(seq 60); do
  code=$(curl -s -m 120 -o /dev/null -w '%{http_code}' "$PROXY/v1/completions" -H "Content-Type: application/json" \
    -d '{"model": "hy3", "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0}')
  [ "$code" = 200 ] && break; sleep 10
done
[ "$code" = 200 ] || { echo "proxy not ready: HTTP $code"; exit 1; }
python3 - "$PROMPTS/${CELL}_c8.jsonl.gz" "$TMP/all.jsonl" <<'EOF'
import gzip, json, sys
with gzip.open(sys.argv[1], "rt") as f:
    rows = [json.loads(line) for line in f]
with open(sys.argv[2], "w") as f:
    for r in rows[2:]:  # the 80 measured requests of the c8 file
        f.write(json.dumps({"prompt": rows[0]["prefix"] + r["suffix"]}, ensure_ascii=False) + "\n")
EOF
N_ALL=$(wc -l < "$TMP/all.jsonl")
echo "$(date +%T) warm pass: $N_ALL requests, output 1 (salt $SALT)"
bench "$OUT/warm.json" --dataset-path "$TMP/all.jsonl" --custom-output-len 1 --num-prompts "$N_ALL" \
  --max-concurrency 8 --percentile-metrics ttft
for C in $CONCS; do
  N=$((10 * C)); [ $N -lt 40 ] && N=40; [ $N -gt "$N_ALL" ] && N=$N_ALL
  echo "=== D c$C ($N requests, output $OUTPUT_LEN)"
  bench "$OUT/c$C.json" --dataset-path "$TMP/all.jsonl" --custom-output-len "$OUTPUT_LEN" \
    --num-prompts $N --max-concurrency $C --num-warmups 2 --percentile-metrics ttft,tpot,itl,e2el
done
