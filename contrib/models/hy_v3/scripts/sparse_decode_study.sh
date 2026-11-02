#!/bin/bash
# Server for the EXPERIMENTAL block-sparse decode accuracy study: BF16 KV, no
# DCP, plain decode, one request at a time, max_model_len 86016, prefix caching
# (chunked 4096-token prefill). HY3_SPARSE_DECODE=<fraction> turns the sparse
# selection on (unset: dense, the reference); only the decode NEFF changes.
# Drive it with scripts/needle.py run and scripts/sparse_decode_eval.py run.
set -euo pipefail
export NEURON_SKIP_EFA_AFFINITY=${NEURON_SKIP_EFA_AFFINITY:-1}
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=${KV_CAP:-0.45}
# CAPTURE_DIR=<dir>: also save the selection diagnostics (tensor_capture).
CAPTURE=""
if [ -n "${CAPTURE_DIR:-}" ]; then
    CAPTURE=", \"tensor_capture\": {\"modules\": [], \"capture_dir\": \"$CAPTURE_DIR\"}"
fi
exec vllm serve "${MODEL:-tencent/Hy3-FP8}" \
    --served-model-name hy3 \
    --tensor-parallel-size 64 \
    --enable-expert-parallel \
    --max-model-len 86016 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 4096 \
    --block-size 32 \
    --enable-prefix-caching \
    --port "${PORT:-8000}" \
    --additional-config '{"neuron_config": {
        "num_batched_tokens_buckets": [512, 4096],
        "kv_segment_size_buckets": [2048],
        "num_seqs_buckets": [1]'"$CAPTURE"'}}'
