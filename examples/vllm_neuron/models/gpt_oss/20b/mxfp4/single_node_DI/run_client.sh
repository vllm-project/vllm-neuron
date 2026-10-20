#!/bin/bash
# gpt-oss 20B single-node disaggregated inference — CLIENT (validation request).
#
# Run this after the proxy is up. Sends requests TO THE PROXY (port 8000), not
# to the prefill or decode servers directly. Single-node, so the proxy is on
# the same host (127.0.0.1). See docs/tutorials/tutorial-gpt-oss.md.

set -x

PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
MODEL_ID="${MODEL_ID:-openai/gpt-oss-20b}"

# Health check against the proxy (expect HTTP/1.1 200 OK).
curl -i "http://$PROXY_HOST:8000/health"

# Sample completion.
curl "http://$PROXY_HOST:8000/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL_ID"'",
        "prompt": "The capital of France is ",
        "max_tokens": 16
    }'
