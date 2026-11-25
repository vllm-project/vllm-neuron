#!/bin/bash
# Latency-oriented config for a single request: no prefix caching (flash-attention
# prefill), fine-grained prefill buckets (no 1k -> 2048 padding) and a 16k prompt
# runs in one prefill step.
# Env: MODEL (default tencent/Hy3-FP8), EP=<n> sets neuron_config.ep_degree
#      (default: full EP=64), KV_CAP, PORT, MTP_K (MTP speculative tokens per
#      step; default off; needs a vllm-neuron with MTP support),
#      HY3_DECODE_HITS_MAX_T (with MTP_K default 4 x (MTP_K + 1)),
#      KV_DTYPE=fp8 (FP8 KV with packed K; KV scales from HY3_KV_SCALE_PROFILE).
set -euo pipefail
export NEURON_SKIP_EFA_AFFINITY=${NEURON_SKIP_EFA_AFFINITY:-1}
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=${KV_CAP:-0.45}
SPEC_ARGS=()
if [ -n "${MTP_K:-}" ]; then
    SPEC_ARGS=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $MTP_K}")
    # A verify step has up to max-num-seqs x (MTP_K + 1) tokens; their routed
    # experts stay few enough for moe_tkg_hits_batch, which beats moe_tkg there
    # (concurrency 4, k=3: 17.1 vs 19.1 ms TPOT).
    export HY3_DECODE_HITS_MAX_T=${HY3_DECODE_HITS_MAX_T:-$((4 * (MTP_K + 1)))}
fi
KV_DTYPE=${KV_DTYPE:-auto}
PACKED_NC=""
if [ "$KV_DTYPE" != auto ]; then
    PACKED_NC='"fp8_packed_kv": true,'
fi
exec vllm serve "${MODEL:-tencent/Hy3-FP8}" "${SPEC_ARGS[@]}" \
    --served-model-name hy3 \
    --tensor-parallel-size 64 \
    --enable-expert-parallel \
    --max-model-len 16384 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 16384 \
    --block-size 32 \
    --no-enable-prefix-caching \
    --kv-cache-dtype "$KV_DTYPE" \
    --port "${PORT:-8000}" \
    --additional-config '{"neuron_config": {'"${EP:+\"ep_degree\": $EP,}$PACKED_NC"'
        "num_batched_tokens_buckets": [512, 1024, 2048, 4096, 8192, 16384],
        "num_seqs_buckets": [1, 4],
        "decode_context_length_buckets": [2048, 4096, 8192]
    }}'
