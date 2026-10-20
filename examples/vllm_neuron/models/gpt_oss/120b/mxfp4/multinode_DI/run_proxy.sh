#!/bin/bash
# gpt-oss disaggregated-inference target config — PROXY router.
#
# Launch this only after BOTH the decode and prefill servers report
# "Application startup complete". Run it on the prefill instance (or any
# instance that can reach both). The proxy accepts client requests and
# coordinates the prefill -> decode handoff.
#
# Mirrors "Deploy with disaggregated inference" -> "4. Launch the proxy router"
# in the gpt-oss tutorial: docs/tutorials/tutorial-gpt-oss.md
#
# Set PREFILL_HOST and DECODE_HOST to the instances' addresses. If you tiled
# multiple prefill servers, pass all their ports to --prefiller-port.

set -x

PREFILL_HOST="${PREFILL_HOST:-127.0.0.1}"
DECODE_HOST="${DECODE_HOST:-127.0.0.1}"

PROXY_SCRIPT="$(dirname "$0")/../../../../../vllm/disaggregated_inference/toy_proxy_server.py"

echo "Starting proxy router on port 8000"
echo "  Prefill: $PREFILL_HOST:8100"
echo "  Decode:  $DECODE_HOST:8200"

python3 "$PROXY_SCRIPT" \
    --port 8000 \
    --host 0.0.0.0 \
    --prefiller-host "$PREFILL_HOST" --prefiller-port 8100 \
    --decoder-host "$DECODE_HOST" --decoder-port 8200
