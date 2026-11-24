#!/bin/bash
# Hy4-preview disaggregated inference: the decode server (kv_consumer), one trn2.48xlarge.
#     MODEL=<dir>/Hy4-preview-neuron-fp8 bash run_decode.sh
set -x
MODEL="${MODEL:?set MODEL to the converted checkpoint}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
SEQ_BUCKETS="${SEQ_BUCKETS:-[1, 4]}"
# The XLA compile route: the default native one compiles the decode graphs once per rank.
export VLLM_NEURON_BACKEND="${VLLM_NEURON_BACKEND:-vllm_neuron}"
export VLLM_ENGINE_READY_TIMEOUT_S=14400
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export NEURON_RT_EXEC_TIMEOUT=1200
export VLLM_NIXL_SIDE_CHANNEL_HOST=0.0.0.0
export VLLM_NIXL_SIDE_CHANNEL_PORT=5659
export VLLM_NEURON_SKIP_PREFILL_WARMUP=1  # the KV arrives from the prefill server
vllm serve "$MODEL" \
    --tensor-parallel-size 64 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens 2048 \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization 0.8 \
    --no-enable-prefix-caching \
    --distributed-timeout-seconds 14400 \
    --port 8200 \
    --kv-transfer-config '{"kv_connector": "Hy4NixlConnector", "kv_connector_module_path": "hy4_preview.kv_connector", "kv_role": "kv_consumer", "kv_buffer_device": "cuda", "kv_connector_extra_config": {"backends": ["LIBFABRIC"]}}' \
    --additional-config "{\"neuron_config\": {\"quantization\": \"bf16\", \"num_batched_tokens_buckets\": [2048], \"num_seqs_buckets\": $SEQ_BUCKETS}}"
