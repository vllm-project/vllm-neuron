#!/bin/bash
# gpt-oss 20B single-node disaggregated inference — PROXY router.
#
# Launch this only after BOTH the prefill and decode servers report
# "Application startup complete". Single-node DI: prefill, decode, and this
# proxy all run on the same host, so the proxy routes over loopback (127.0.0.1).
#
# The proxy accepts client requests on port 8000 and coordinates the
# prefill -> decode handoff. See docs/tutorials/tutorial-gpt-oss.md.

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
