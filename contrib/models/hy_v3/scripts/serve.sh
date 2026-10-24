#!/bin/bash
# Serve tencent/Hy3-FP8 on a trn2.48xlarge (TP=64, full EP), prefix caching on.
#
# Usage: serve.sh <max_model_len> <max_num_seqs> '<num_seqs_buckets json>' '<decode_ctx_buckets json>'
# Env:
#   MODEL            checkpoint path or HF id (default: tencent/Hy3-FP8)
#   PREFILL_BUCKETS  prefill (Q) buckets, JSON list (default: [2048]). The largest
#                    one is also --max-num-batched-tokens.
#   KV_SEGMENTS      kv_segment_size_buckets, JSON list (default: [2048])
#   DCP              decode context parallel size, 1 or 8 (default: 1)
#   KV_CAP           VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION (default: 0.45)
#   PORT             API port (default: 8000)
#   EP               neuron_config.ep_degree (default: full EP=64)
#   KV_DTYPE         --kv-cache-dtype: auto (BF16, default) or fp8. FP8 needs a KV scale
#                    profile (HY3_KV_SCALE_PROFILE, see calibrate_kv_scales.py) and uses
#                    the packed K layout ("fp8_packed_kv"; FP8_PACKED=0 turns it off)
#   NUM_BLOCKS       --num-gpu-blocks-override (e.g. the same KV token count for BF16/FP8)
#   EXTRA_ARGS       appended to the vllm serve command line (e.g. --no-async-scheduling)
#
# Configs used in the README:
#   (max_model_len is always an implicit decode context bucket)
#   B (long ctx):    ./serve.sh 16384 32  '[1, 4, 16, 32]'          '[4096, 8192]'
#   C (big batch):   ./serve.sh 2048  128 '[1, 4, 16, 32, 64, 128]' '[1024]'
#   DCP=8:     DCP=8 ./serve.sh 32768 64  '[1, 4, 16, 32, 64]'      '[4096, 8192, 16384]'
#              (FP8 KV with DCP > 1: PREFILL_BUCKETS <= 2048; see README "DCP prefill buckets")
#   B+multi-bucket:  PREFILL_BUCKETS='[512, 1024, 2048, 4096, 8192]' ./serve.sh 16384 32 ...
#
# Requires the contrib package: pip install -e contrib/models/hy_v3
set -euo pipefail
PREFILL_BUCKETS=${PREFILL_BUCKETS:-[2048]}
MAX_BATCHED=$(python3 -c "import json,sys; print(max(json.loads(sys.argv[1])))" "$PREFILL_BUCKETS")
KV_DTYPE=${KV_DTYPE:-auto}
PACKED_NC=""
if [ "$KV_DTYPE" != auto ] && [ "${FP8_PACKED:-1}" = 1 ]; then
    PACKED_NC='"fp8_packed_kv": true,'
fi
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=${KV_CAP:-0.45}
# Needed on instances without EFA devices.
export NEURON_SKIP_EFA_AFFINITY=${NEURON_SKIP_EFA_AFFINITY:-1}
exec vllm serve "${MODEL:-tencent/Hy3-FP8}" \
    --served-model-name hy3 \
    --tensor-parallel-size 64 \
    --enable-expert-parallel \
    --max-model-len "$1" \
    --max-num-seqs "$2" \
    --max-num-batched-tokens "$MAX_BATCHED" \
    --block-size 32 \
    --enable-prefix-caching \
    --decode-context-parallel-size "${DCP:-1}" \
    --cp-kv-cache-interleave-size 32 \
    --kv-cache-dtype "$KV_DTYPE" \
    ${NUM_BLOCKS:+--num-gpu-blocks-override "$NUM_BLOCKS"} \
    --port "${PORT:-8000}" \
    ${EXTRA_ARGS:-} \
    --additional-config "{\"neuron_config\": {${EP:+\"ep_degree\": $EP,} $PACKED_NC
        \"num_batched_tokens_buckets\": $PREFILL_BUCKETS,
        \"kv_segment_size_buckets\": ${KV_SEGMENTS:-[2048]},
        \"num_seqs_buckets\": $3,
        \"decode_context_length_buckets\": $4
    }}"
