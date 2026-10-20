#!/bin/bash
# gpt-oss disaggregated-inference target config — CLIENT (validation request).
#
# Run this after the proxy is up. Sends requests TO THE PROXY (port 8000), not
# to the prefill or decode servers directly.
#
# Mirrors "Deploy with disaggregated inference" -> "6. Validate the deployment"
# in the gpt-oss tutorial: docs/tutorials/tutorial-gpt-oss.md

set -x

# Host running the proxy (the prefill instance in this recipe).
PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
MODEL_ID="${MODEL_ID:-openai/gpt-oss-120b}"

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
