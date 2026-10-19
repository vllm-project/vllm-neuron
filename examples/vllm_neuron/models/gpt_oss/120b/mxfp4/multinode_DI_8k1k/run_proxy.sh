#!/bin/bash
# GPT-OSS 120B 8K/1K 6P1D: the PROXY router.
#
# Run this on the PREFILL node (node A) after the decode server and all 6
# prefill replicas report "Application startup complete". The proxy accepts
# client requests on port 8000. It sends each prompt to one prefill replica
# (round robin), then hands the request to the decode server, which pulls
# the KV cache over NIXL. The log goes to $SERVER_LOG_DIR/proxy.log.
#
# The proxy script is toy_proxy_server.py from the vLLM Neuron examples tree.
# It is a minimal example router, and not a production router. Set
# PROXY_SCRIPT if you copied this directory somewhere else.

set -euo pipefail
# shellcheck source=env_common.sh
source "$(dirname "$0")/env_common.sh"
require_hosts
mkdir -p "$SERVER_LOG_DIR"

if [ ! -f "$PROXY_SCRIPT" ]; then
    echo "The proxy script is not at $PROXY_SCRIPT. Set PROXY_SCRIPT to the path of toy_proxy_server.py from the vLLM Neuron examples." >&2
    exit 1
fi

PREFILL_HOSTS=()
PREFILL_PORTS=()
for ((i = 0; i < NUM_PREFILL; i++)); do
    PREFILL_HOSTS+=("$PREFILL_HOST")
    PREFILL_PORTS+=("$((PREFILL_BASE_PORT + i))")
done

echo "Proxy on port $PROXY_PORT: prefill ports ${PREFILL_PORTS[*]} on $PREFILL_HOST, decode $DECODE_HOST:$DECODE_PORT" | tee "$SERVER_LOG_DIR/proxy.log"

# 0.0.0.0 accepts a client on another host. Set PROXY_BIND_HOST=127.0.0.1 to
# accept a client on this node only.
python3 "$PROXY_SCRIPT" \
    --port "$PROXY_PORT" \
    --host "${PROXY_BIND_HOST:-0.0.0.0}" \
    --prefiller-hosts "${PREFILL_HOSTS[@]}" \
    --prefiller-ports "${PREFILL_PORTS[@]}" \
    --decoder-hosts "$DECODE_HOST" \
    --decoder-ports "$DECODE_PORT" 2>&1 | tee -a "$SERVER_LOG_DIR/proxy.log"
