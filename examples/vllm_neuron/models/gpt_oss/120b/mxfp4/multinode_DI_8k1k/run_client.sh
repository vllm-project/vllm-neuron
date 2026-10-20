#!/bin/bash
# GPT-OSS 120B 8K/1K 6P1D: a health check and one completion through the proxy.
#
# Run this after the proxy is up, on the node that runs the proxy. Send the
# requests TO THE PROXY (port 8000), not to the prefill or decode servers.
# Set PROXY_URL to reach a proxy on another host.

set -euo pipefail
# shellcheck source=env_common.sh
source "$(dirname "$0")/env_common.sh"

PROXY_URL="${PROXY_URL:-http://localhost:$PROXY_PORT}"

# Expect HTTP/1.1 200 OK. The proxy serves /healthcheck, not /health.
curl -i "$PROXY_URL/healthcheck"
echo

# Expect an OpenAI-compatible JSON payload with a coherent choices[0].text.
curl "$PROXY_URL/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"model": "'"$MODEL_ID"'", "prompt": "The capital of France is ", "max_tokens": 16}'
echo
