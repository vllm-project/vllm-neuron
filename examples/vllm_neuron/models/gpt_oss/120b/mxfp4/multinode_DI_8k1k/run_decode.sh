#!/bin/bash
# GPT-OSS 120B 8K/1K 6P1D: the DECODE server (kv_consumer).
#
# Run this on the DECODE node (node B). The server takes every NeuronCore of
# the node at NEURON_LOGICAL_NC_CONFIG=1: 128 logical cores = TP8 x DP16,
# with expert parallelism over all 128 ranks (one expert for each rank).
#
# vLLM Neuron fuses the whole decode graph into one megakernel when the
# quantization is mxfp4, the platform is Trn3, NEURON_LOGICAL_NC_CONFIG is 1,
# and the server is the kv_consumer of a disaggregated deployment. No flag
# turns it on.
#
# The log goes to $SERVER_LOG_DIR/decode.log.
#
# Option (environment variable):
#   MEGAKERNEL_SCOPE  the kernel-call scope of the schedule. "auto" (default)
#                     takes the value of the active compile route.

set -euo pipefail
# shellcheck source=env_common.sh
source "$(dirname "$0")/env_common.sh"
require_hosts
mkdir -p "$SERVER_LOG_DIR"

# -------------------------------------------------------- decode-only env
# The megakernel needs one logical NeuronCore for each physical core. The
# prefill role stays at 2.
export NEURON_LOGICAL_NC_CONFIG=1
# Keep the source tensors of a collective off shared HBM. A toolchain that
# requires this stops the trace with "At LNC=1 a collective's src tensors
# must not be on shared_hbm" when the variable is absent.
export VLLM_NEURON_MEGA_COLLECTIVE_PRIVATE_SRC=1
# Keep 4 batches in flight in the async engine. The host work of a step then
# overlaps the device.
export VLLM_NEURON_BATCH_QUEUE_DEPTH=4
# Let the XU worker back off from its completion poll. The poll takes host
# cores from the workers and widens the arrival skew at the 128-way barrier.
export NEURON_RT_XU_WORKER_ENABLE_LOW_LATENCY_COMPLETION=0
export VLLM_NIXL_SIDE_CHANNEL_PORT="$DECODE_NIXL_PORT"

# -------------------------------------------------------- megakernel tuning
# Each value is off by default. Without them, the server gives correct output
# at a much larger step time, with no warning.
#
# 1. The DMA-order schedule, written for the kernel-call scope of the decode
#    graph. See prepare_megakernel_schedule.py. When this file is set, vLLM
#    Neuron also sets the kernel options that the schedule was made with.
SCHEDULE_FILE="$(python3 "$EXAMPLE_DIR/prepare_megakernel_schedule.py" \
    --k "${MEGAKERNEL_SCOPE:-auto}" --source "$MEGAKERNEL_SCHEDULE" \
    --out-dir "$MEGAKERNEL_SCHEDULE_DIR")"
export VLLM_NEURON_MEGA_DMA_ORDER_CONSTRAINTS_FILE="$SCHEDULE_FILE"
# 2. Build the token embedding inside the kernel (needs embedding_dp_size 1).
export VLLM_NEURON_MEGA_FUSE_EMBED=1
# 3. Gather the rows of the fused sampling tail with ONE all_gather over the
#    128-rank lm_head group.
export VLLM_NEURON_TAIL_FLAT_ROW_GATHER=1

# A distinct engine_id for each server, so the NIXL connector tracks each
# handshake by itself.
ENGINE_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"

# The decode server takes the rank after the last prefill replica.
KV_TRANSFER_CONFIG=$(cat <<EOF
{"kv_connector": "NeuronNixlConnector", "kv_role": "kv_consumer",
 "kv_buffer_device": "cuda", "engine_id": "$ENGINE_ID",
 "kv_ip": "$PREFILL_HOST", "kv_rank": $NUM_PREFILL,
 "kv_parallel_size": $((NUM_PREFILL + 1)),
 "kv_connector_extra_config": {"backends": ["LIBFABRIC"],
   "prefill_ips": ["$PREFILL_HOST"], "decode_ips": ["$DECODE_HOST"]}}
EOF
)

# embedding_dp_size 1:  the kernel builds the token embedding itself.
# lm_head_dp_size 16:   the lm_head is replicated over the 16 DP ranks.
# gpt_oss_tail_hops:    ONE 128-rank all-to-all in the fused tail.
ADDITIONAL_CONFIG=$(cat <<EOF
{"neuron_config": {"quantization": "mxfp4", "fp8_packed_kv": true,
   "embedding_dp_size": 1, "lm_head_dp_size": $DECODE_DP,
   "ep_degree": $DECODE_EP_DEGREE, "gpt_oss_tail_hops": [128],
   "kv_segment_size_buckets": [$INPUT_LEN],
   "num_batched_tokens_buckets": [$INPUT_LEN],
   "num_seqs_buckets": [$DECODE_MAX_NUM_SEQS]}}
EOF
)

{
    echo "Decode server: $MODEL_ID on port $DECODE_PORT, TP$DECODE_TP DP$DECODE_DP EP$DECODE_EP_DEGREE at lnc=1, max_num_seqs $DECODE_MAX_NUM_SEQS"
    echo "DMA-order schedule: $SCHEDULE_FILE"
    echo "Log: $SERVER_LOG_DIR/decode.log"
} | tee "$SERVER_LOG_DIR/decode.log"

vllm serve "$MODEL_ID" \
    --tensor-parallel-size "$DECODE_TP" \
    --data-parallel-size "$DECODE_DP" \
    --enable-expert-parallel \
    --distributed-executor-backend mp \
    --optimization-level 2 \
    --max-num-batched-tokens "$INPUT_LEN" \
    --max-num-seqs "$DECODE_MAX_NUM_SEQS" \
    --max-logprobs 0 \
    --max-model-len "$MAX_MODEL_LEN" \
    --block-size "$BLOCK_SIZE" \
    --dtype bfloat16 \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --no-enable-log-requests \
    --no-disable-hybrid-kv-cache-manager \
    --hf-overrides '{"quantization_config": {}}' \
    --port "$DECODE_PORT" \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    --additional-config "$ADDITIONAL_CONFIG" 2>&1 | tee -a "$SERVER_LOG_DIR/decode.log"
