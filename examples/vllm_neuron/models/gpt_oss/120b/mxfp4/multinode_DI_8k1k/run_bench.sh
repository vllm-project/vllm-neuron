#!/bin/bash
# GPT-OSS 120B 8K/1K 6P1D: the benchmark CLIENT.
#
# Run this on the node that runs the proxy (node A). It drives the proxy with
# two legs:
#   1. a warmup leg of 10 x CONCURRENCY prompts. The first leg after a server
#      launch is not a valid measurement. Discard it.
#   2. the measured leg of 100 x CONCURRENCY prompts. Read its JSON result.
#
# Both legs send random prompts of INPUT_LEN tokens, ask for OUTPUT_LEN tokens
# with --ignore-eos, and keep CONCURRENCY requests in flight. The results go
# to $BENCH_RESULT_DIR. Set PROXY_URL to reach a proxy on another host.

set -euo pipefail
# shellcheck source=env_common.sh
source "$(dirname "$0")/env_common.sh"

PROXY_URL="${PROXY_URL:-http://localhost:$PROXY_PORT}"
mkdir -p "$BENCH_RESULT_DIR"

# The benchmark client opens its HTTP session with trust_env, so aiohttp looks
# up ~/.netrc before it sends each request. When $HOME is on a network file
# system, each lookup can take hundreds of milliseconds. The client then sends
# requests late, and the measured throughput drops with no change on the
# servers. Point the lookup at /dev/null. Set BENCH_NETRC to use your own file.
BENCH_NETRC="${BENCH_NETRC:-/dev/null}"

bench_leg() {
    local num_prompts="$1" result_file="$2"
    echo "Benchmark leg: $num_prompts prompts, $INPUT_LEN in / $OUTPUT_LEN out, concurrency $CONCURRENCY -> $BENCH_RESULT_DIR/$result_file"
    NETRC="$BENCH_NETRC" vllm bench serve \
        --base-url "$PROXY_URL" \
        --model "$MODEL_ID" \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --random-range-ratio 0 \
        --num-prompts "$num_prompts" \
        --max-concurrency "$CONCURRENCY" \
        --request-rate inf \
        --ignore-eos \
        --num-warmups "$CONCURRENCY" \
        --ready-check-timeout-sec 0 \
        --percentile-metrics tpot,ttft \
        --metric-percentiles 90,99 \
        --save-result \
        --disable-shuffle \
        --result-dir "$BENCH_RESULT_DIR" \
        --result-filename "$result_file"
}

# Leg 1: warm up. Discard this result.
bench_leg $((CONCURRENCY * 10)) bench_warmup.json
# Leg 2: measure.
bench_leg $((CONCURRENCY * 100)) bench_clean.json

echo
echo "Measured leg: $BENCH_RESULT_DIR/bench_clean.json"
python3 - "$BENCH_RESULT_DIR/bench_clean.json" <<'EOF'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    result = json.load(fh)
for key in (
    "completed",
    "duration",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "output_throughput",
    "total_token_throughput",
):
    print(f"{key:>24}: {result.get(key)}")
EOF
