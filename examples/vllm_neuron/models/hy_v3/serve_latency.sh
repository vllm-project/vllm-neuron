#!/bin/bash
# Latency-oriented config: no prefix caching / segmented prefill, so prefill can use
# fine-grained buckets (no 1k->2048 padding) and a 16k prompt runs in one step.
# EP=<n> sets neuron_config.ep_degree (default: full EP=64; EP=32 is the REPORT.md §11 experiment).
export NEURON_SKIP_EFA_AFFINITY=${NEURON_SKIP_EFA_AFFINITY:-1}
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=${KV_CAP:-0.45}
exec vllm serve "${MODEL:-tencent/Hy3-FP8}" \
    --served-model-name hy3 \
    --tensor-parallel-size 64 \
    --enable-expert-parallel \
    --max-model-len 16384 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 16384 \
    --block-size 32 \
    --no-enable-prefix-caching \
    --port 8000 \
    --additional-config '{"neuron_config": {'"${EP:+\"ep_degree\": $EP,}"'
        "num_batched_tokens_buckets": [512, 1024, 2048, 4096, 8192, 16384],
        "num_seqs_buckets": [1, 4],
        "decode_context_length_buckets": [2048, 4096, 8192]
    }}'
