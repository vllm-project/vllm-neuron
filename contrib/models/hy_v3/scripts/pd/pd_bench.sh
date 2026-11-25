#!/bin/bash
# pd_bench.sh <out_dir> [concurrencies="1 4 8"] [output_len=1200]: the 85k / 92% shared-prefix
# customer case through the PD proxy, same prompts and procedure as
# ../bench_prefix_fixed.sh (bench_data/prefix/in85000_p78336: 78336 of 85000 tokens shared; one
# warm-up request per cell, then 10 x C requests; a new cache_salt per run), plus a needle check
# first. The proxy has no /metrics, so the prefix-cache counters are read on both servers:
# the prefill hit rate (P) is the customer's 92%; the decode side's own hits (D) are tokens it
# did not have to pull over NIXL.
# Requests go to /v1/chat/completions with the server-side chat template, temperature 0.9 and
# seed 0, as in pd_bench_split.sh and pd_bench_dcust.sh.
# Works with either proxy of pd_proxy.sh (NIXL toy proxy or Mooncake).
# Env: CODE (vllm-neuron checkout, default: the one holding this script), MODEL (tokenizer), PROXY (http://127.0.0.1:8000),
#      P_URL (http://P_HOST:8100), D_URL (http://D_HOST:8200), SALT, MTP (1: also record the
#      decode side's spec-decode acceptance counters), PROMPTS_DIR (default bench_data/prefix),
#      P_INSTANCES (prefill instances behind the proxy, default 1: one warm-up request each; the
#      prefill hit rate is then read on P_URL's instance only).
set -uo pipefail
OUT=${1:?usage: pd_bench.sh <out_dir> [concurrencies] [output_len]}
CONCS=${2:-1 4 8}
OUTPUT_LEN=${3:-1200}
CODE=${CODE:-$(cd "$(dirname "$0")/../../../../.." && pwd)}
MODEL=${MODEL:-tencent/Hy3-FP8}
PROXY=${PROXY:-http://127.0.0.1:8000}
P_URL=${P_URL:-http://${P_HOST:-127.0.0.1}:8100}
D_URL=${D_URL:-http://${D_HOST:-127.0.0.1}:8200}
SALT=${SALT:-pd-$(date +%s)}
PROMPTS=${PROMPTS_DIR:-$CODE/contrib/models/hy_v3/bench_data/prefix}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
TMP=$(mktemp -d); trap 'rm -rf $TMP' EXIT

counter() {  # counter <base url> <metric>: sum over label sets
  curl -s "$1/metrics" | awk -v n="$2" '$1 ~ "^"n"(_total)?([{]|$)" {s += $NF} END {printf "%d", s}'
}
snap() {  # snap <prefix>: P/D prefix-cache and D spec-decode counters into shell vars
  eval "$1_pq=$(counter $P_URL vllm:prefix_cache_queries) $1_ph=$(counter $P_URL vllm:prefix_cache_hits)"
  eval "$1_dq=$(counter $D_URL vllm:prefix_cache_queries) $1_dh=$(counter $D_URL vllm:prefix_cache_hits)"
  eval "$1_sd=$(counter $D_URL vllm:spec_decode_num_drafts) $1_sa=$(counter $D_URL vllm:spec_decode_num_accepted_tokens) $1_st=$(counter $D_URL vllm:spec_decode_num_draft_tokens)"
}

for u in $P_URL/health $D_URL/health; do
  curl -sf "$u" >/dev/null || { echo "not up: $u"; exit 1; }
done
# The proxy: one 1-token request end to end. This works for both proxies (the Mooncake one has no
# health route and answers 503 until it has read the prefill engine id from the bootstrap server)
# and is the first KV transfer, so a broken transfer shows up here and not in the middle of a run.
for i in $(seq 60); do
  code=$(curl -s -m 120 -o "$TMP/smoke.json" -w '%{http_code}' "$PROXY/v1/completions" -H "Content-Type: application/json" \
    -d '{"model": "hy3", "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0}')
  [ "$code" = 200 ] && break
  sleep 10
done
[ "$code" = 200 ] || { echo "proxy smoke request failed: HTTP $code"; cat "$TMP/smoke.json"; exit 1; }
echo "$(date +%T) smoke through the proxy: $(python3 -c 'import json,sys; print(repr(json.load(open(sys.argv[1]))["choices"][0]["text"]))' "$TMP/smoke.json")"
echo "$(date +%T) needle through the proxy:"
python3 "$HERE/needle_check.py" hy3 "$MODEL" "${PROXY##*:}" 2>&1 | grep -E "needle|NEEDLE"

for C in $CONCS; do
  src=$PROMPTS/in85000_p78336_c${C}.jsonl.gz
  [ -f "$src" ] || { echo "missing $src"; exit 1; }
  tag=in85000_p78336_out${OUTPUT_LEN}_c${C}
  python3 - "$src" "$TMP/warmup.json" "$TMP/$tag.jsonl" "$SALT" <<'EOF'
import gzip, json, sys
src, warm, out, salt = sys.argv[1:]
with gzip.open(src, "rt") as f:
    rows = [json.loads(line) for line in f]
prefix = rows[0]["prefix"]
with open(warm, "w") as f:
    json.dump({"model": "hy3", "messages": [{"role": "user", "content": prefix + rows[1]["suffix"]}],
               "max_tokens": 1, "cache_salt": salt}, f)
with open(out, "w") as f:
    for r in rows[2:]:
        f.write(json.dumps({"prompt": prefix + r["suffix"]}, ensure_ascii=False) + "\n")
EOF
  N=$(wc -l < "$TMP/$tag.jsonl")
  echo "=== $tag (warm-up + $N requests, cache_salt=$SALT)"
  # One warm-up per prefill instance (P_INSTANCES): the proxy round-robins, and an instance without
  # the shared prefix cold-prefills 85k tokens for its first request (P99 TTFT 12 s instead of 5).
  for i in $(seq "${P_INSTANCES:-1}"); do
    curl -sf "$PROXY/v1/chat/completions" -H "Content-Type: application/json" -d @"$TMP/warmup.json" >/dev/null \
      || { echo "warm-up failed"; exit 1; }
  done
  snap a
  vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --base-url "$PROXY" \
    --model "$MODEL" --served-model-name hy3 --tokenizer "$MODEL" --trust-remote-code \
    --dataset-name custom --dataset-path "$TMP/$tag.jsonl" --skip-chat-template --disable-shuffle \
    --temperature 0.9 --seed 0 \
    --custom-output-len "$OUTPUT_LEN" --ignore-eos --num-prompts "$N" --max-concurrency "$C" \
    --ready-check-timeout-sec 0 --extra-body "{\"cache_salt\": \"$SALT\"}" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir "$OUT" --result-filename "$tag.json" \
    2>&1 | grep -E "Successful|Failed|duration|throughput|TTFT|TPOT|ITL" | grep -E "Mean|Median|P99|throughput|Successful|Failed|duration"
  snap b
  python3 - "$OUT/$tag.json" "$SALT" $((b_pq - a_pq)) $((b_ph - a_ph)) $((b_dq - a_dq)) $((b_dh - a_dh)) \
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
print(f"prefill hit rate {d['prefill_hit_rate']:.1%} ({ph}/{pq}); decode local hits {d['decode_hit_rate']:.1%} ({dh}/{dq})"
      + (f"; MTP {d['spec_tokens_per_step']:.2f} tokens/step" if sd else ""))
EOF
done
