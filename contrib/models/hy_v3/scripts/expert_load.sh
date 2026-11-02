#!/bin/bash
# Server that records the routed experts of every token (for scripts/expert_load.py):
# the latency config (no prefix caching, prefill buckets 512..16384, BF16 KV,
# plain decode) with a neuron_config tensor_capture of the `moe_route.<layer>`
# tensors that HYV3MoE.forward captures. Each rank writes its tokens' top-8
# expert ids per layer and forward pass under CAPTURE_DIR (prefill: its 1/64
# sequence shard; decode: every token). Drive it with expert_load_client.py.
# Env: MODEL, PORT, CAPTURE_DIR (default ./route_captures).
set -euo pipefail
export NEURON_SKIP_EFA_AFFINITY=${NEURON_SKIP_EFA_AFFINITY:-1}
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=${KV_CAP:-0.45}
exec vllm serve "${MODEL:-tencent/Hy3-FP8}" \
    --served-model-name hy3 \
    --tensor-parallel-size 64 \
    --enable-expert-parallel \
    --max-model-len 16384 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 16384 \
    --block-size 32 \
    --no-enable-prefix-caching \
    --port "${PORT:-8000}" \
    --additional-config "{\"neuron_config\": {
        \"num_batched_tokens_buckets\": [512, 1024, 2048, 4096, 8192, 16384],
        \"num_seqs_buckets\": [1, 8],
        \"tensor_capture\": {\"modules\": [], \"capture_dir\": \"${CAPTURE_DIR:-./route_captures}\"}}}"
