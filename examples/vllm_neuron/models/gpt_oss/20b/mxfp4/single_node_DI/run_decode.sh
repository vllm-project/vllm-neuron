#!/bin/bash
# gpt-oss 20B single-node disaggregated inference — DECODE server (kv_consumer).
#
# Single-node 1P1D: prefill and decode share one 64-core Trn3 instance. This
# server binds the second 32 cores (32-63); the prefill server binds cores 0-3.
# KV cache transfers over NixL / LIBFABRIC on loopback (127.0.0.1).
#
# Topology: TP8 DP4 EP32 (1 expert/rank), global decode batch 16 (4 x DP4).
# See docs/tutorials/tutorial-gpt-oss.md for the flag walkthrough.
#
# Launch order: this decode server and the prefill server can start at the same
# time; start the proxy (run_proxy.sh) only after BOTH report
# "Application startup complete".

set -x

MODEL_ID="${MODEL_ID:-openai/gpt-oss-20b}"

# gpt-oss compilation/startup run longer than the vLLM defaults.
export VLLM_NEURON_COMPILATION_TIMEOUT=2400
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_ENGINE_READY_TIMEOUT_S=1800

# NixL side channel: listen on all interfaces (loopback is sufficient single-node).
# The decode side-channel port must differ from the prefill port.
export VLLM_NIXL_SIDE_CHANNEL_HOST=0.0.0.0
export VLLM_NIXL_SIDE_CHANNEL_PORT=5659

# Second half of the 64 cores; prefill uses cores 0-3.
export NEURON_VISIBLE_DEVICES="32-63"

echo "Starting single-node DI decode server (kv_consumer): $MODEL_ID"
echo "  Port: 8200, Cores: 32-63, TP8 DP4 EP32"

vllm serve "$MODEL_ID" \
    --tensor-parallel-size 8 \
    --data-parallel-size 4 \
    --enable-expert-parallel \
    --optimization-level 2 \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 4 \
    --max-logprobs 0 \
    --no-disable-hybrid-kv-cache-manager \
    --port 8200 \
    --hf-overrides '{"quantization_config": {}}' \
    --kv-transfer-config '{"kv_connector": "NeuronNixlConnector", "kv_role": "kv_consumer", "kv_buffer_device": "cuda", "kv_connector_extra_config": {"backends": ["LIBFABRIC"]}}' \
    --additional-config '{
        "neuron_config": {
            "quantization": "mxfp4",
            "embedding_dp_size": 4,
            "lm_head_dp_size": 4,
            "kv_segment_size_buckets": [8192],
            "num_batched_tokens_buckets": [8192],
            "num_seqs_buckets": [4]
        }
    }'
