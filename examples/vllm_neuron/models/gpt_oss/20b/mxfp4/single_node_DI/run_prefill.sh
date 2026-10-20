#!/bin/bash
# gpt-oss 20B single-node disaggregated inference — PREFILL server (kv_producer).
#
# Single-node 1P1D: prefill and decode share one 64-core Trn3 instance. This
# server binds four cores (0-3); the decode server binds cores 32-63.
# KV cache transfers over NixL / LIBFABRIC on loopback (127.0.0.1).
#
# Topology: attention TP4 (DP1), MoE EP2 — the same prefill sharding as the
# 120B recipe. See docs/tutorials/tutorial-gpt-oss.md for the flag walkthrough.
#
# Launch order: this prefill server and the decode server can start at the same
# time; start the proxy (run_proxy.sh) only after BOTH report
# "Application startup complete".

set -x

MODEL_ID="${MODEL_ID:-openai/gpt-oss-20b}"

# gpt-oss compilation/startup run longer than the vLLM defaults.
export VLLM_NEURON_COMPILATION_TIMEOUT=2400
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_ENGINE_READY_TIMEOUT_S=1800

# NixL side channel: listen on all interfaces (loopback is sufficient single-node).
# The prefill side-channel port must differ from the decode port.
export VLLM_NIXL_SIDE_CHANNEL_HOST=0.0.0.0
export VLLM_NIXL_SIDE_CHANNEL_PORT=5559

# Prefill uses four cores; decode takes the second half (32-63).
export NEURON_VISIBLE_DEVICES="0-3"

echo "Starting single-node DI prefill server (kv_producer): $MODEL_ID"
echo "  Port: 8100, Cores: 0-3, TP4 DP1 EP2"

vllm serve "$MODEL_ID" \
    --tensor-parallel-size 4 \
    --data-parallel-size 1 \
    --enable-expert-parallel \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 1 \
    --no-disable-hybrid-kv-cache-manager \
    --port 8100 \
    --hf-overrides '{"quantization_config": {}}' \
    --kv-transfer-config '{"kv_connector": "NeuronNixlConnector", "kv_role": "kv_producer", "kv_buffer_device": "cuda", "kv_connector_extra_config": {"backends": ["LIBFABRIC"]}}' \
    --additional-config '{
        "neuron_config": {
            "quantization": "mxfp4",
            "ep_degree": 2,
            "kv_segment_size_buckets": [8192],
            "num_batched_tokens_buckets": [8192],
            "num_seqs_buckets": [1]
        }
    }'
