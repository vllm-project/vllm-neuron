#!/bin/bash
# Serve tencent/Hy3-FP8 on a trn2.48xlarge (TP=64, full EP).
#
# Usage: serve.sh <max_model_len> <max_num_seqs> '<num_seqs_buckets json>' '<decode_ctx_buckets json>'
# Env:
#   MODEL   checkpoint path or HF id (default: tencent/Hy3-FP8)
#   DCP     decode context parallel size, 1 or 8 (default: 1)
#   KV_CAP  VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION (default: 0.45; 0.6 OOMed at runtime)
#   PORT    API port (default: 8000)
#
# Examples (configs used in REPORT.md):
#   B (long ctx):   ./serve.sh 16384 32  '[1, 4, 16, 32]'            '[4096, 8192]'
#   C (big batch):  ./serve.sh 2048  128 '[1, 4, 16, 32, 64, 128]'   '[1024]'
#   DCP=8:     DCP=8 ./serve.sh 32768 64  '[1, 4, 16, 32, 64]'        '[4096, 8192, 16384]'
set -euo pipefail
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=${KV_CAP:-0.45}
export NEURON_SKIP_EFA_AFFINITY=${NEURON_SKIP_EFA_AFFINITY:-1}
exec vllm serve "${MODEL:-tencent/Hy3-FP8}" \
    --served-model-name hy3 \
    --tensor-parallel-size 64 \
    --enable-expert-parallel \
    --max-model-len "$1" \
    --max-num-seqs "$2" \
    --max-num-batched-tokens 2048 \
    --block-size 32 \
    --enable-prefix-caching \
    --decode-context-parallel-size "${DCP:-1}" \
    --cp-kv-cache-interleave-size 32 \
    --port "${PORT:-8000}" \
    --additional-config "{\"neuron_config\": {
        \"num_batched_tokens_buckets\": [2048],
        \"kv_segment_size_buckets\": [2048],
        \"num_seqs_buckets\": $3,
        \"decode_context_length_buckets\": $4
    }}"
