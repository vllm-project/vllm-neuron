#!/bin/bash
# One node, no PD, with decode context parallelism: TP=64, DCP=8 + MTP k=3 + sparse decode, FP8 KV.
#   max_model_len 262144. DCP=8 shards the KV by 32-token blocks over the 8 ranks that hold the same
#   KV head, so the KV pool holds ~4M tokens instead of ~0.5M (the TP=64 pool replicates each of
#   the 8 KV heads 8x). MTP under DCP needs VLLM_NEURON_DCP_SPEC_DECODE=1 (span-aware DCP decode).
#   Sparse decode 0.0625 on per-block K summaries (each DCP rank summarizes its own blocks; the
#   selection uses Q gathered over the DCP group); dense prefill; dense MTP draft (HY3_SPARSE_DRAFT=0).
#   Prefill buckets up to 4096 (with DCP the 8192 step does not compile, NCC_IGCA037).
#   KV_CAP 0.40: prefill and decode graphs share the HBM of one instance.
#
# Do NOT run DCP prefill with NEURON_RT_CC_ALG_TYPES=^INTRA_RDH_ALG (serve_agentx.sh's default):
# with it the DCP prefill collectives give wrong results on the full model (garbage output above
# ~512 prompt tokens, needle 0/12); without it needle 12/12. It is fine on a PD decode-only node.
#
# Usage: serve_agentx_dcp.sh, then scripts/pd/agentx_trn2.sh against it (see serve_agentx.sh):
#   bash scripts/serve_agentx_dcp.sh                                   # port 8000
#   SEQS=16 SEQ_BUCKETS='[1, 2, 4, 8, 16]' bash scripts/serve_agentx_dcp.sh
#   MTP_K= bash scripts/serve_agentx_dcp.sh                           # no MTP
#   HY3_SPARSE_DECODE= bash scripts/serve_agentx_dcp.sh               # dense decode
# Env: everything serve.sh takes (MODEL, PORT, KV_CAP, EXTRA_ARGS, ...), plus MAX_LEN, SEQS,
#      SEQ_BUCKETS and the HY3_* switches below.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

export VLLM_NEURON_BACKEND=${VLLM_NEURON_BACKEND:-neuron_native} NEURON_RT_EXEC_TIMEOUT=${NEURON_RT_EXEC_TIMEOUT:-1200}
export VLLM_NEURON_BARRIER_TIMEOUT=${VLLM_NEURON_BARRIER_TIMEOUT:-43200} VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-14400}
export NEURON_LIBTORCH_PARALLEL_TRACE_WORKERS=${NEURON_LIBTORCH_PARALLEL_TRACE_WORKERS:-2}
# Cold compile: ~100 NEFFs, ~70 min with 4 workers. The NEFF cache lock timeout must outlast one
# compile, or a waiter breaks a running compile's lock and compiles the same NEFF again.
export VLLM_NEURON_PARALLEL_COMPILE_WORKERS=${VLLM_NEURON_PARALLEL_COMPILE_WORKERS:-4}
export TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT=${TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT:-7200}
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
if [ -n "${NEURON_RT_CC_ALG_TYPES:-}" ]; then
  echo "serve_agentx_dcp.sh: NEURON_RT_CC_ALG_TYPES=$NEURON_RT_CC_ALG_TYPES gives wrong DCP prefill results; unset it" >&2
  exit 1
fi

export KV_DTYPE=${KV_DTYPE:-fp8} DCP=${DCP:-8} MTP_K=${MTP_K-3}
if [ -n "$MTP_K" ]; then
  export VLLM_NEURON_DCP_SPEC_DECODE=1
  export HY3_KV_SCALE_PROFILE=${HY3_KV_SCALE_PROFILE:-$HERE/../kv_profiles/Hy3-FP8_mtp_neuron.safetensors}
else
  unset MTP_K
  export HY3_KV_SCALE_PROFILE=${HY3_KV_SCALE_PROFILE:-$HERE/../kv_profiles/Hy3-FP8_neuron.safetensors}
fi
export HY3_SPARSE_DECODE=${HY3_SPARSE_DECODE-0.0625}
if [ -n "$HY3_SPARSE_DECODE" ]; then
  export HY3_SPARSE_SELECT_EVERY=${HY3_SPARSE_SELECT_EVERY:-8} HY3_SPARSE_DRAFT=${HY3_SPARSE_DRAFT:-0}
else
  unset HY3_SPARSE_DECODE
fi
export HY3_DECODE_MOE_FUSED=${HY3_DECODE_MOE_FUSED:-1} HY3_DECODE_HITS=${HY3_DECODE_HITS:-0}
export HY3_DECODE_HITS_MAX_T=${HY3_DECODE_HITS_MAX_T:-16}
SEQS=${SEQS:-8}
export VLLM_NEURON_MAX_KV_CONCURRENT=${VLLM_NEURON_MAX_KV_CONCURRENT:-$((SEQS + 4))}
export KV_CAP=${KV_CAP:-0.40} PREFILL_BUCKETS=${PREFILL_BUCKETS:-'[512, 1024, 2048, 4096]'}
exec "$HERE/serve.sh" "${MAX_LEN:-262144}" "$SEQS" "${SEQ_BUCKETS:-[1, 2, 4, 8]}"
