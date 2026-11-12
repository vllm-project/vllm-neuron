#!/bin/bash
# agentx_trn2.sh <out_dir> <C> [ignored]: InferenceX AgentX through the PD proxy, as a run_pd.sh BENCH
# (BENCH=agentx_trn2). Same settings as gpu/agentx.sh: agentx-harness 1.0.13, the
# semianalysis_cc_traces_weka_062126_256k corpus, 3600 s per point after the cache warm-up, failed
# request threshold 0.10. Sessions carry X-Session-ID, so the proxy keeps each session on one
# prefill instance. /metrics is scraped from both prefill instances (8100, 8101) and the decode.
#
# The client needs its own venv (agentx-harness bundles a forked aiperf) on Python 3.11-3.13, e.g.
#   python3 -m venv ~/agentx-venv && ~/agentx-venv/bin/pip install agentx-harness==1.0.13 'datasets>=4.7.0'
# Env: PROXY (http://127.0.0.1:8000), D_URL (http://127.0.0.1:8200), AGENTX (~/agentx-venv/bin/agentx),
#      MODEL (tokenizer path, /mnt/nvme/models/Hy3-FP8), DURATION (3600), P_PORTS ("8100 8101"; empty: no
#      prefill /metrics, e.g. for a single server).
set -uo pipefail
OUT=$1; C=$2
PROXY=${PROXY:-http://127.0.0.1:8000}; D_URL=${D_URL:-http://127.0.0.1:8200}
mkdir -p "$OUT"
metrics=()
for p in ${P_PORTS-8100 8101}; do metrics+=("http://127.0.0.1:$p/metrics"); done
metrics+=("$D_URL/metrics")
# Smoke test: a streamed request must come back as text/event-stream with content.
echo "$(date +%T) smoke:"
curl -s -m 600 -D - "$PROXY/v1/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"hy3","messages":[{"role":"user","content":"What is 2+3?"}],"max_tokens":8,"stream":true}' \
  | grep -iE "content-type|content\"" | sed -n 1,3p
AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=true AIPERF_FAILED_REQUEST_THRESHOLD=0.10 \
  "${AGENTX:-$HOME/agentx-venv/bin/agentx}" profile --url "$PROXY" --endpoint /v1/chat/completions \
  --model hy3 --tokenizer "${MODEL:-/mnt/nvme/models/Hy3-FP8}" --concurrency "$C" \
  --benchmark-duration "${DURATION:-3600}" --server-metrics "${metrics[@]}" \
  --public-dataset semianalysis_cc_traces_weka_062126_256k \
  --output-artifact-dir "$OUT/aiperf_artifacts" --ui simple > "$OUT/agentx.log" 2>&1
echo "exit $?" >> "$OUT/agentx.log"
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('submission_valid:', d['metadata']['submission_valid'])" \
  "$OUT/aiperf_artifacts/profile_export_aiperf.json" 2>&1
tail -1 "$OUT/agentx.log"
