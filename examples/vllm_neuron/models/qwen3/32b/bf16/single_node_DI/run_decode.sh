#!/bin/bash
# Qwen3-32B single-node disaggregated inference — DECODE server (kv_consumer).
#
# Single-node 1P1D on trn2.48xlarge (64 cores). This server binds cores 16-31;
# the prefill server binds cores 0-15.
# KV cache transfers over NixL / LIBFABRIC on loopback.
#
# Topology: TP8 DP2 — larger decode TP for throughput; DP2 for decode batch.
#
# Launch order: start prefill and decode simultaneously; start the proxy
# (run_proxy.sh) only after BOTH report "Application startup complete".

set -x

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-32B}"

export VLLM_NEURON_COMPILATION_TIMEOUT=2400
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_ENGINE_READY_TIMEOUT_S=1800

export VLLM_NIXL_SIDE_CHANNEL_HOST=0.0.0.0
export VLLM_NIXL_SIDE_CHANNEL_PORT=5659

export NEURON_VISIBLE_DEVICES="16-31"

echo "Starting single-node DI decode server (kv_consumer): $MODEL_ID"
echo "  Port: 8200, Cores: 16-31, TP8 DP2"

vllm serve "$MODEL_ID" \
    --tensor-parallel-size 8 \
    --data-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 4 \
    --max-logprobs 0 \
    --no-disable-hybrid-kv-cache-manager \
    --port 8200 \
    --kv-transfer-config '{"kv_connector": "NixlConnector", "kv_role": "kv_consumer", "kv_buffer_device": "cuda", "kv_connector_extra_config": {"backends": ["LIBFABRIC"]}}' \
    --additional-config '{
        "neuron_config": {
            "kv_segment_size_buckets": [8192],
            "num_batched_tokens_buckets": [8192],
            "num_seqs_buckets": [4]
        }
    }'
