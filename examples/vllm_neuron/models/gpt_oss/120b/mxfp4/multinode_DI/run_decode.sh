#!/bin/bash
# gpt-oss disaggregated-inference target config — DECODE server (kv_consumer).
#
# Run this on the DECODE instance. Binds all 64 cores.
#   gpt-oss 120B: TP8 DP8 EP64 (2 experts/rank), global decode batch 32 (4 x DP8)
#   gpt-oss  20B: TP8 DP4 EP32 (1 expert/rank),  global decode batch 16 (4 x DP4)
#
# Mirrors "Deploy with disaggregated inference" -> "2. Launch the decode server"
# in the gpt-oss tutorial: docs/tutorials/tutorial-gpt-oss.md
#
# Launch order: this decode server and the prefill server can start at the same
# time; start the proxy (run_proxy.sh) only after BOTH report
# "Application startup complete".

set -x

MODEL_ID="${MODEL_ID:-openai/gpt-oss-120b}"

# gpt-oss compilation and multi-node startup run longer than the vLLM defaults.
export VLLM_NEURON_COMPILATION_TIMEOUT=2400
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_ENGINE_READY_TIMEOUT_S=5400

# NixL side channel: listen on all interfaces so the prefill peer can reach it
# over EFA. The decode side-channel port must differ from the prefill port.
export VLLM_NIXL_SIDE_CHANNEL_HOST=0.0.0.0
export VLLM_NIXL_SIDE_CHANNEL_PORT=5659

export NEURON_VISIBLE_DEVICES="0-63"

echo "Starting DI decode server (kv_consumer): $MODEL_ID"
echo "  Port: 8200, Cores: 0-63, TP8 DP8 EP64 (120B)"

vllm serve "$MODEL_ID" \
    --tensor-parallel-size 8 \
    --data-parallel-size 8 \
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
            "embedding_dp_size": 8,
            "lm_head_dp_size": 8,
            "kv_segment_size_buckets": [8192],
            "num_batched_tokens_buckets": [8192],
            "num_seqs_buckets": [4]
        }
    }'
