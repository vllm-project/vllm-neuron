#!/bin/bash
# Device-profiling server for the 85k shared-prefix workload: the serve_85k.sh
# FP8 KV + MTP configuration (no DCP) with the fewest NEFFs
# that the workload uses, so the profiler's buffers fit next to the KV:
# prefill buckets 512 / 4096, batch buckets 1 / 8, and no decode context
# buckets (one 86016 NEFF per batch bucket; every request here is ~85k).
# Drive it with scripts/profile_85k_client.py.
# Env: MODEL, PORT, MTP_K (default 3; empty for plain decode), KV_CAP (0.49),
#      HY3_KV_SCALE_PROFILE (default kv_profiles/Hy3-FP8_mtp_neuron.safetensors),
#      PROFILE_RANKS (JSON list of TP ranks, default 8 spread over EP),
#      PROFILE_DIR (default ./neuron_profiles).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export NEURON_SKIP_EFA_AFFINITY=${NEURON_SKIP_EFA_AFFINITY:-1}
# The device profiler allocates ~2.9 GB per profiled core when a session
# starts (NEURON_RT_ENABLE_DMA_TRACING=0 does not shrink it), which does not
# fit next to the KV of 8 x 85k: KV_CAP 0.49 (568k tokens) leaves room and
# runs 6 such requests at once.
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=${KV_CAP:-0.49}
export HY3_KV_SCALE_PROFILE=${HY3_KV_SCALE_PROFILE:-$HERE/../kv_profiles/Hy3-FP8_mtp_neuron.safetensors}
MTP_K=${MTP_K-3}
SPEC_ARGS=()
if [ -n "$MTP_K" ]; then
    SPEC_ARGS=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $MTP_K}")
    export HY3_DECODE_HITS_MAX_T=${HY3_DECODE_HITS_MAX_T:-$((4 * (MTP_K + 1)))}
fi
RANKS=${PROFILE_RANKS:-[0, 9, 18, 27, 36, 45, 54, 63]}
exec vllm serve "${MODEL:-tencent/Hy3-FP8}" "${SPEC_ARGS[@]}" \
    --served-model-name hy3 \
    --tensor-parallel-size 64 \
    --enable-expert-parallel \
    --max-model-len 86016 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 4096 \
    --block-size 32 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8 \
    --port "${PORT:-8000}" \
    --profiler-config '{"profiler": "cuda"}' \
    --additional-config "{\"neuron_config\": {
        \"fp8_packed_kv\": true,
        \"num_batched_tokens_buckets\": [512, 4096],
        \"kv_segment_size_buckets\": [2048],
        \"num_seqs_buckets\": [1, 8]},
      \"neuron_profiler\": {
        \"activities\": [\"device_profile\", \"system_profile\"],
        \"neuron_cores\": $RANKS,
        \"output_dir\": \"${PROFILE_DIR:-./neuron_profiles}\"}}"
