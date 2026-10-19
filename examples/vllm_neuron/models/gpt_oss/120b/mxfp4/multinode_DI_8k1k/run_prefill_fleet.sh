#!/bin/bash
# GPT-OSS 120B 8K/1K 6P1D: all 6 prefill replicas, one after the other.
#
# Run this on the PREFILL node (node A). It starts replica 0 and waits until
# the replica answers /health. Then it starts replicas 1 to 5 the same way.
# The first replica fills the compile cache, so the other five start from it.
# Each replica writes its log to $SERVER_LOG_DIR/prefill_<i>.log and keeps
# running after this script exits.

set -euo pipefail
# shellcheck source=env_common.sh
source "$(dirname "$0")/env_common.sh"
require_hosts
mkdir -p "$SERVER_LOG_DIR"

# Poll /health of one port until it answers, or until the engine-ready
# timeout passes.
wait_for_health() {
    local port="$1"
    local deadline
    deadline=$(( $(date +%s) + VLLM_ENGINE_READY_TIMEOUT_S ))
    until curl -sf "http://localhost:$port/health" > /dev/null 2>&1; do
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "The server on port $port did not answer /health in $VLLM_ENGINE_READY_TIMEOUT_S s. Read the logs in $SERVER_LOG_DIR." >&2
            exit 1
        fi
        sleep 15
    done
}

for ((i = 0; i < NUM_PREFILL; i++)); do
    nohup "$EXAMPLE_DIR/run_prefill.sh" "$i" > /dev/null 2>&1 &
    echo "Started prefill replica $i (PID $!). Log: $SERVER_LOG_DIR/prefill_$i.log"
    wait_for_health $((PREFILL_BASE_PORT + i))
    echo "Prefill replica $i answers /health on port $((PREFILL_BASE_PORT + i))."
done
echo "All $NUM_PREFILL prefill replicas are up. Start the proxy with run_proxy.sh."
