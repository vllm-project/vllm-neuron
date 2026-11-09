#!/bin/bash
# agentx.sh <out_dir> <concurrency>: InferenceX AgentX (agentic trace replay) against one running
# OpenAI-compatible server or the PD proxy, with InferenceX's settings:
#   agentx-harness 1.0.13 (infx/bench/agentic/requirements.txt), corpus
#   semianalysis_cc_traces_weka_062126_256k (Hy3's max context is 262144; the uncapped corpus has
#   longer traces), 3600 s profile, failed-request threshold 0.10.
# AgentX measures one concurrency per server start: restart the server (and the proxy) between
# points and use a fresh <out_dir>. Check metadata.submission_valid in
# <out_dir>/aiperf_artifacts/profile_export_aiperf.json.
# The client needs its own venv (agentx-harness bundles a forked aiperf) on Python 3.11-3.13, e.g.
#   /opt/pytorch/bin/python3 -m venv ~/agentx-venv && ~/agentx-venv/bin/pip install agentx-harness==1.0.13 'datasets>=4.7.0'
# Env: URL (http://127.0.0.1:8000), METRICS_URL ($URL/metrics; for PD, the decode server's),
#      MODEL_NAME (hy3), TOKENIZER (/opt/dlami/nvme/models/Hy3-FP8), AGENTX (~/agentx-venv/bin/agentx),
#      DATASET, DURATION (3600; below 900 needs --unsafe-override and is not a valid submission).
set -euo pipefail
OUT=${1:?usage: agentx.sh <out_dir> <concurrency>}
C=${2:?usage: agentx.sh <out_dir> <concurrency>}
URL=${URL:-http://127.0.0.1:8000}
DURATION=${DURATION:-3600}
EXTRA=(); [ "$DURATION" -lt 900 ] && EXTRA+=(--unsafe-override)
mkdir -p "$OUT"
AIPERF_FAILED_REQUEST_THRESHOLD=0.10 "${AGENTX:-$HOME/agentx-venv/bin/agentx}" profile \
  --url "$URL" --endpoint /v1/chat/completions \
  --model "${MODEL_NAME:-hy3}" --tokenizer "${TOKENIZER:-/opt/dlami/nvme/models/Hy3-FP8}" \
  --concurrency "$C" --benchmark-duration "$DURATION" \
  --server-metrics "${METRICS_URL:-$URL/metrics}" \
  --public-dataset "${DATASET:-semianalysis_cc_traces_weka_062126_256k}" \
  --output-artifact-dir "$OUT/aiperf_artifacts" --ui simple "${EXTRA[@]}" 2>&1 | tee "$OUT/agentx.log"
python3 - "$OUT/aiperf_artifacts" <<'PY'
import json, sys
d = json.load(open(f"{sys.argv[1]}/profile_export_aiperf.json"))
print("submission_valid:", d.get("metadata", {}).get("submission_valid"))
PY
