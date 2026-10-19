#!/bin/bash
# GPT-OSS 120B 8K/1K 6P1D: ONE prefill replica (kv_producer).
#
# Run this on the PREFILL node (node A), one time for each replica index:
#     ./run_prefill.sh 0        # then 1, 2, 3, 4, 5
# run_prefill_fleet.sh starts all six replicas one after the other.
#
# Replica i binds NeuronCores 4i to 4i+3 at NEURON_LOGICAL_NC_CONFIG=2. It
# serves HTTP on port 8100 + i and the NIXL side channel on 5559 + 100 * i.
# The log goes to $SERVER_LOG_DIR/prefill_<i>.log.
#
# Each replica is its own TP4 DP1 EP2 server, and not one wide DP server.
# Expert parallelism cannot share data-parallel ranks on the prefill graph.

set -euo pipefail
# shellcheck source=env_common.sh
source "$(dirname "$0")/env_common.sh"
require_hosts
mkdir -p "$SERVER_LOG_DIR"

INDEX="${1:-${REPLICA:-0}}"
if [ "$INDEX" -lt 0 ] || [ "$INDEX" -ge "$NUM_PREFILL" ]; then
    echo "The replica index must be 0 to $((NUM_PREFILL - 1)). Got '$INDEX'." >&2
    exit 1
fi

PORT=$((PREFILL_BASE_PORT + INDEX))
NIXL_PORT=$((PREFILL_BASE_NIXL + INDEX * 100))
LOW=$((INDEX * PREFILL_TP))
HIGH=$((LOW + PREFILL_TP - 1))
LOG="$SERVER_LOG_DIR/prefill_$INDEX.log"

# The prefill role runs the standard per-layer path at lnc=2.
export NEURON_LOGICAL_NC_CONFIG=2
export NEURON_VISIBLE_DEVICES="${LOW}-${HIGH}"
export VLLM_NIXL_SIDE_CHANNEL_PORT="$NIXL_PORT"

# A distinct engine_id for each server, so the NIXL connector tracks each
# handshake by itself.
ENGINE_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"

# Replica i takes rank i in the KV transfer group. The decode server takes
# the rank after the last replica, so the group has NUM_PREFILL + 1 members.
KV_TRANSFER_CONFIG=$(cat <<EOF
{"kv_connector": "NeuronNixlConnector", "kv_role": "kv_producer",
 "kv_buffer_device": "cuda", "engine_id": "$ENGINE_ID",
 "kv_ip": "$PREFILL_HOST", "kv_rank": $INDEX,
 "kv_parallel_size": $((NUM_PREFILL + 1)),
 "kv_connector_extra_config": {"backends": ["LIBFABRIC"],
   "prefill_ips": ["$PREFILL_HOST"], "decode_ips": ["$DECODE_HOST"]}}
EOF
)

# The prefill role sets no KV budget override. The worker sizes the KV cache
# from one HBM stack, and at lnc=2 one logical core owns a whole stack.
ADDITIONAL_CONFIG=$(cat <<EOF
{"nixl_side_channel_port": $NIXL_PORT,
 "neuron_config": {"quantization": "mxfp4", "fp8_packed_kv": true,
   "ep_degree": $PREFILL_EP_DEGREE,
   "kv_segment_size_buckets": [$INPUT_LEN],
   "num_batched_tokens_buckets": [$INPUT_LEN],
   "num_seqs_buckets": [1]}}
EOF
)

echo "Prefill replica $INDEX: $MODEL_ID on port $PORT, cores $NEURON_VISIBLE_DEVICES, TP$PREFILL_TP DP1 EP$PREFILL_EP_DEGREE, NIXL side channel $NIXL_PORT" | tee "$LOG"

vllm serve "$MODEL_ID" \
    --tensor-parallel-size "$PREFILL_TP" \
    --data-parallel-size 1 \
    --enable-expert-parallel \
    --distributed-executor-backend mp \
    --max-num-batched-tokens "$INPUT_LEN" \
    --max-num-seqs 1 \
    --max-logprobs 0 \
    --max-model-len "$MAX_MODEL_LEN" \
    --block-size "$BLOCK_SIZE" \
    --dtype bfloat16 \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --no-enable-log-requests \
    --no-disable-hybrid-kv-cache-manager \
    --hf-overrides '{"quantization_config": {}}' \
    --port "$PORT" \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    --additional-config "$ADDITIONAL_CONFIG" 2>&1 | tee -a "$LOG"
