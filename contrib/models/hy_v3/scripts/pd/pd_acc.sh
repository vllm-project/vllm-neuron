#!/bin/bash
# pd_acc.sh <out_dir>: the PR's long-context accuracy checks through the PD proxy (for the sparse
# decode: run once dense and once with HY3_SPARSE_DECODE, then compare).
#   needle.py run bench_data/needle.jsonl.gz   12 pass keys, 16k-85k tokens x depth 0.1/0.5/0.9
#   sparse_decode_eval.py run                  18 long prompts (10 x 85k shared prefix, 4 x 32k,
#                                              4 x 64k), greedy 64-token continuations
# One request at a time, so for the sparse path set HY3_SPARSE_MIN_BATCH=1 on the decode node
# (by default steps of fewer than 4 requests stay dense).
# Compare: python3 $CODE/contrib/models/hy_v3/scripts/sparse_decode_eval.py compare dense.json sparse.json
# Env: CODE (vllm-neuron checkout, default: the one holding this script), PROXY (http://127.0.0.1:8000).
set -uo pipefail
OUT=${1:?usage: pd_acc.sh <out_dir>}
CODE=${CODE:-$(cd "$(dirname "$0")/../../../../.." && pwd)}
PROXY=${PROXY:-http://127.0.0.1:8000}
S=$CODE/contrib/models/hy_v3/scripts
mkdir -p "$OUT"
for i in $(seq 60); do
  code=$(curl -s -m 120 -o /dev/null -w '%{http_code}' "$PROXY/v1/completions" -H "Content-Type: application/json" \
    -d '{"model": "hy3", "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0}')
  [ "$code" = 200 ] && break; sleep 10
done
[ "$code" = 200 ] || { echo "proxy not ready: HTTP $code"; exit 1; }
echo "$(date +%T) needle (12 prompts)"
python3 "$S/needle.py" run "$CODE/contrib/models/hy_v3/bench_data/needle.jsonl.gz" "$PROXY" 2>&1 | tee "$OUT/needle.log" | tail -15
echo "$(date +%T) sparse_decode_eval run (18 prompts)"
python3 "$S/sparse_decode_eval.py" run "$OUT/eval.json" "$PROXY" 2>&1 | tail -20
echo "$(date +%T) done"
