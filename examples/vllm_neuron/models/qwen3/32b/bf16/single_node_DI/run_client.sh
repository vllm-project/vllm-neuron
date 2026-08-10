#!/bin/bash
# Qwen3-32B single-node disaggregated inference — CLIENT (validation request).
#
# Run this after the proxy is up. Sends requests to the proxy (port 8000).

set -x

PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-32B}"

curl -i "http://$PROXY_HOST:8000/health"

curl "http://$PROXY_HOST:8000/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL_ID"'",
        "prompt": "The capital of France is ",
        "max_tokens": 16
    }'
