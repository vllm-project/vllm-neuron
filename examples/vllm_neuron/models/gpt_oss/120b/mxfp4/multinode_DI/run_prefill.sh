#!/bin/bash
# gpt-oss disaggregated-inference target config — PREFILL server (kv_producer).
#
# Run this on the PREFILL instance. Binds four cores: attention TP4 (DP1), MoE
# EP2. The prefill configuration is identical for gpt-oss 20B and 120B — only
# MODEL_ID changes.
#
# Mirrors "Deploy with disaggregated inference" -> "3. Launch the prefill server"
# in the gpt-oss tutorial: docs/tutorials/tutorial-gpt-oss.md
#
# A single TP4 prefill server uses only 4 of the 64 cores. For best performance,
# tile up to 16 TP4 prefill servers (a 16P1D topology) by launching this script
# once per 4-core slice with a distinct NEURON_VISIBLE_DEVICES, --port, and
# VLLM_NIXL_SIDE_CHANNEL_PORT, then list every prefill port on the proxy. See the
# "Scale prefill to fully utilize the instance" section of the tutorial.

set -x

MODEL_ID="${MODEL_ID:-openai/gpt-oss-120b}"

# gpt-oss compilation and multi-node startup run longer than the vLLM defaults.
export VLLM_NEURON_COMPILATION_TIMEOUT=2400
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_ENGINE_READY_TIMEOUT_S=5400

# NixL side channel: listen on all interfaces so the decode peer can reach it
# over EFA. The prefill side-channel port must differ from the decode port.
export VLLM_NIXL_SIDE_CHANNEL_HOST=0.0.0.0
export VLLM_NIXL_SIDE_CHANNEL_PORT=5559

export NEURON_VISIBLE_DEVICES="0-3"

echo "Starting DI prefill server (kv_producer): $MODEL_ID"
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
