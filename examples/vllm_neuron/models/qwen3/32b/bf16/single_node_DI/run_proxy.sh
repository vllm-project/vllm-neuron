#!/bin/bash
# Qwen3-32B single-node disaggregated inference — PROXY router.
#
# Launch this only after BOTH the prefill and decode servers report
# "Application startup complete". Routes client requests to the appropriate
# server over loopback.

set -x

PROXY_SCRIPT="$(dirname "$0")/../../../../../vllm/disaggregated_inference/toy_proxy_server.py"

echo "Starting proxy router on 127.0.0.1:8000"
echo "  Prefill: 127.0.0.1:8100"
echo "  Decode:  127.0.0.1:8200"

python3 "$PROXY_SCRIPT" \
    --port 8000 \
    --host 0.0.0.0 \
    --prefiller-host 127.0.0.1 --prefiller-port 8100 \
    --decoder-host 127.0.0.1 --decoder-port 8200
