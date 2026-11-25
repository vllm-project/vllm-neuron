#!/bin/bash
# pd_bench_split.sh <out_dir> [concurrencies="1 4 8"] [decode_output_len=256]
# ../bench_pd_split.sh (the "GPU P/D test" flags) through the PD proxy, so 1P1D can
# be read against the single-instance split and the customer's GPU numbers:
#   P (prefill only): output 1 token, per C its own file (own prefix), 10 x C requests, 10
#     warm-ups. Through PD this is prefill + KV transfer + the decode node's first token.
#   D (decode only): the c8 file (one prefix, 80 requests) sent twice per C with the same salt;
#     pass 1 is ~91% hit (the customer case), pass 2 ~100% (prefill and decode both hit, nothing
#     to transfer). temperature 0.9, seed 0.
# Same flags as the PR's script: openai-chat on /v1/chat/completions with the server-side chat
# template, --request-rate inf, a cache_salt per cell. The proxy has no /metrics, so the counters
# come from the prefill (hit rate) and decode (local hits, MTP acceptance) servers.
# Env: CODE (default: the checkout holding this script), MODEL, PROXY, P_URL / D_URL (or P_HOST / D_HOST), SALT, CELL (in85000_p78336),
#      P_ONLY (1: skip the D part), PROMPTS_DIR (default $CODE/contrib/models/hy_v3/bench_data/prefix).
set -uo pipefail
OUT=${1:?usage: pd_bench_split.sh <out_dir> [concurrencies] [decode_output_len]}
CONCS=${2:-1 4 8}
D_OUTPUT_LEN=${3:-256}
CODE=${CODE:-$(cd "$(dirname "$0")/../../../../.." && pwd)}
PROMPTS_DIR=${PROMPTS_DIR:-}
MODEL=${MODEL:-tencent/Hy3-FP8}
PROXY=${PROXY:-http://127.0.0.1:8000}
P_URL=${P_URL:-http://${P_HOST:-127.0.0.1}:8100}
D_URL=${D_URL:-http://${D_HOST:-127.0.0.1}:8200}
CELL=${CELL:-in85000_p78336}
SALT=${SALT:-split-$(date +%s)}
PROMPTS=${PROMPTS_DIR:-$CODE/contrib/models/hy_v3/bench_data/prefix}
mkdir -p "$OUT/P" "$OUT/D"
TMP=$(mktemp -d); trap 'rm -rf $TMP' EXIT

counter() { curl -s "$1/metrics" | awk -v n="$2" '$1 ~ "^"n"(_total)?([{]|$)" {s += $NF} END {printf "%d", s}'; }
snap() {
  eval "$1_pq=$(counter $P_URL vllm:prefix_cache_queries) $1_ph=$(counter $P_URL vllm:prefix_cache_hits)"
  eval "$1_dq=$(counter $D_URL vllm:prefix_cache_queries) $1_dh=$(counter $D_URL vllm:prefix_cache_hits)"
  eval "$1_sd=$(counter $D_URL vllm:spec_decode_num_drafts) $1_sa=$(counter $D_URL vllm:spec_decode_num_accepted_tokens) $1_st=$(counter $D_URL vllm:spec_decode_num_draft_tokens)"
}
to_jsonl() {  # every request of the file (warm-up row included) -> custom .jsonl
  python3 - "$1" "$2" <<'EOF'
import gzip, json, sys
with gzip.open(sys.argv[1], "rt") as f:
    rows = [json.loads(line) for line in f]
with open(sys.argv[2], "w") as f:
    for r in rows[1:]:
        f.write(json.dumps({"prompt": rows[0]["prefix"] + r["suffix"]}, ensure_ascii=False) + "\n")
EOF
}
bench() {  # bench <out.json> <salt> <vllm bench serve args...>
  local out=$1 salt=$2; shift 2
  snap a
  vllm bench serve --backend openai-chat --base-url "$PROXY" --endpoint /v1/chat/completions \
    --model "$MODEL" --served-model-name hy3 --tokenizer "$MODEL" --trust-remote-code \
    --dataset-name custom --skip-chat-template --request-rate inf --ignore-eos \
    --ready-check-timeout-sec 0 --extra-body "{\"cache_salt\": \"$salt\"}" --metric-percentiles 50,99 \
    --save-result --result-dir "$(dirname "$out")" --result-filename "$(basename "$out")" \
    "$@" 2>&1 | grep -E "Successful|Failed|duration|P50 (TTFT|TPOT|ITL)|P99 (TTFT|ITL)|Request throughput|Output token"
  snap b
  python3 - "$out" "$salt" $((b_pq - a_pq)) $((b_ph - a_ph)) $((b_dq - a_dq)) $((b_dh - a_dh)) \
    $((b_sd - a_sd)) $((b_sa - a_sa)) $((b_st - a_st)) <<'EOF'
import json, sys
path, salt = sys.argv[1], sys.argv[2]
pq, ph, dq, dh, sd, sa, st = map(int, sys.argv[3:])
d = json.load(open(path))
d.update(cache_salt=salt, prefill_prefix_queried=pq, prefill_prefix_hits=ph,
         prefill_hit_rate=ph / pq if pq else 0.0, decode_prefix_queried=dq, decode_prefix_hits=dh,
         decode_hit_rate=dh / dq if dq else 0.0)
if sd:
    d.update(spec_drafts=sd, spec_accepted=sa, spec_draft_tokens=st,
             spec_accept_rate=sa / st if st else 0.0, spec_tokens_per_step=1 + sa / sd)
json.dump(d, open(path, "w"))
print(f"hit rate (incl. warm-ups): prefill {d['prefill_hit_rate']:.1%}, decode local {d['decode_hit_rate']:.1%}"
      + (f"; MTP {d['spec_tokens_per_step']:.2f} tokens/step" if sd else ""))
EOF
}

for u in $P_URL/health $D_URL/health; do curl -sf "$u" >/dev/null || { echo "not up: $u"; exit 1; }; done
# The Mooncake proxy answers 503 until it has the prefill engine id: one request end to end first.
for i in $(seq 60); do
  code=$(curl -s -m 120 -o "$TMP/smoke.json" -w '%{http_code}' "$PROXY/v1/completions" -H "Content-Type: application/json" \
    -d '{"model": "hy3", "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0}')
  [ "$code" = 200 ] && break; sleep 10
done
[ "$code" = 200 ] || { echo "proxy smoke request failed: HTTP $code"; exit 1; }
echo "$(date +%T) smoke ok"
for C in $CONCS; do
  to_jsonl "$PROMPTS/${CELL}_c$C.jsonl.gz" "$TMP/p_c$C.jsonl"
  echo "=== P ${CELL}_c$C ($((10 * C)) requests, output 1)"
  bench "$OUT/P/c$C.json" "$SALT-P-c$C" --dataset-path "$TMP/p_c$C.jsonl" --custom-output-len 1 \
    --num-prompts $((10 * C)) --max-concurrency $C --num-warmups 10 --percentile-metrics ttft,e2el
done
[ "${P_ONLY:-0}" = 1 ] && exit 0
to_jsonl "$PROMPTS/${CELL}_c8.jsonl.gz" "$TMP/replay.jsonl"
for C in $CONCS; do
  for pass in 1 2; do
    echo "=== D ${CELL} replay c$C pass $pass (80 requests, output $D_OUTPUT_LEN)"
    bench "$OUT/D/c${C}_pass$pass.json" "$SALT-D-c$C" --dataset-path "$TMP/replay.jsonl" \
      --temperature 0.9 --seed 0 --custom-output-len $D_OUTPUT_LEN --num-prompts 80 \
      --max-concurrency $C --num-warmups 2 --percentile-metrics ttft,tpot,itl,e2el
  done
done
