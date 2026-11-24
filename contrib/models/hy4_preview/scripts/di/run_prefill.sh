#!/bin/bash
# Hy4-preview disaggregated inference: the prefill server (kv_producer), one trn2.48xlarge.
#     MODEL=<dir>/Hy4-preview-neuron-fp8 bash run_prefill.sh
# Needs the contrib plugin installed (pip install -e contrib/models/hy4_preview).
set -x
MODEL="${MODEL:?set MODEL to the converted checkpoint}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# The XLA compile route: the default native one compiles the decode graphs once per rank.
export VLLM_NEURON_BACKEND="${VLLM_NEURON_BACKEND:-vllm_neuron}"
export VLLM_ENGINE_READY_TIMEOUT_S=14400
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export NEURON_RT_EXEC_TIMEOUT=1200
# The address the decode side connects to for the NIXL handshake: this host's, not 0.0.0.0.
export VLLM_NIXL_SIDE_CHANNEL_HOST="${VLLM_NIXL_SIDE_CHANNEL_HOST:-$(hostname -I | cut -d' ' -f1)}"
export VLLM_NIXL_SIDE_CHANNEL_PORT=5559
export VLLM_NEURON_SKIP_DECODE_WARMUP=1   # this server never decodes
vllm serve "$MODEL" \
    --tensor-parallel-size 64 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.8 \
    --no-enable-prefix-caching \
    --distributed-timeout-seconds 14400 \
    --port 8100 \
    --kv-transfer-config '{"kv_connector": "Hy4NixlConnector", "kv_connector_module_path": "hy4_preview.kv_connector", "kv_role": "kv_producer", "kv_buffer_device": "cuda", "kv_connector_extra_config": {"backends": ["LIBFABRIC"]}}' \
    --additional-config '{"neuron_config": {"quantization": "bf16", "num_batched_tokens_buckets": [2048], "num_seqs_buckets": [1]}}'
