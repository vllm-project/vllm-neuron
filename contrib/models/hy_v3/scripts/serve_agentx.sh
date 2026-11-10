#!/bin/bash
# InferenceX AgentX on one node, no PD: one server does both prefill and decode.
#   max_model_len 262144 (Hy3's max_position_embeddings; AgentX traces reach about 250k tokens),
#   FP8 KV, MTP k=3, sparse decode 0.0625 / sparse prefill 0.25, prefix caching.
#   Prefill buckets 512 .. 8192 (as the PD prefill role). Decode as the PD decode role:
#   batch buckets up to 16, decode context buckets 98304 (+ the implicit 262144), selection every
#   8 layers, fused decode MoE, runtime collectives without INTRA_RDH.
#   KV_CAP 0.45: the 8192-token prefill step needs the HBM a larger KV pool would take.
#
# Usage: serve_agentx.sh, then scripts/pd/agentx_trn2.sh against it:
#   bash scripts/serve_agentx.sh                                   # TP=64, port 8000
#   TP=32 CORES=0-31 PORT=8001 bash scripts/serve_agentx.sh         # half a node
#   PROXY=http://127.0.0.1:8000 D_URL=http://127.0.0.1:8000 P_PORTS= \
#     bash scripts/pd/agentx_trn2.sh results/agentx/single_c8 8     # restart the server per C
# Env: everything serve.sh takes (MODEL, PORT, TP, CORES, KV_CAP, EXTRA_ARGS, ...), plus MAX_LEN,
#      SEQS, SEQ_BUCKETS, CTX_BUCKETS and the HY3_* switches below (each can be overridden).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

# Native route, with the TP=64 compile/host-memory settings (8 trace / compile workers per rank
# runs a fresh trn2 out of host memory) and timeouts long enough for a cold compile.
export VLLM_NEURON_BACKEND=${VLLM_NEURON_BACKEND:-neuron_native} NEURON_RT_EXEC_TIMEOUT=${NEURON_RT_EXEC_TIMEOUT:-1200}
export VLLM_NEURON_BARRIER_TIMEOUT=${VLLM_NEURON_BARRIER_TIMEOUT:-43200} VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-14400}
export NEURON_LIBTORCH_PARALLEL_TRACE_WORKERS=${NEURON_LIBTORCH_PARALLEL_TRACE_WORKERS:-2}
export VLLM_NEURON_PARALLEL_COMPILE_WORKERS=${VLLM_NEURON_PARALLEL_COMPILE_WORKERS:-2}
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
# EFA instances need the runtime's EFA affinity (serve.sh defaults it to off for instances without EFA).
if fi_info -p efa >/dev/null 2>&1; then export NEURON_SKIP_EFA_AFFINITY=${NEURON_SKIP_EFA_AFFINITY:-0}; fi

export KV_DTYPE=${KV_DTYPE:-fp8} MTP_K=${MTP_K:-3}
export HY3_KV_SCALE_PROFILE=${HY3_KV_SCALE_PROFILE:-$HERE/../kv_profiles/Hy3-FP8_mtp_neuron.safetensors}
export HY3_SPARSE_DECODE=${HY3_SPARSE_DECODE:-0.0625} HY3_SPARSE_PREFILL=${HY3_SPARSE_PREFILL:-0.25}
export HY3_SPARSE_SELECT_EVERY=${HY3_SPARSE_SELECT_EVERY:-8}
export HY3_DECODE_MOE_FUSED=${HY3_DECODE_MOE_FUSED:-1} HY3_DECODE_HITS=${HY3_DECODE_HITS:-0}
export HY3_DECODE_HITS_MAX_T=${HY3_DECODE_HITS_MAX_T:-$((4 * (MTP_K + 1)))}
export NEURON_RT_CC_ALG_TYPES=${NEURON_RT_CC_ALG_TYPES:-^INTRA_RDH_ALG}
# The scheduler's worst-case admission (KV tokens / max_model_len) would admit 1-2 requests at
# 262144; block allocation still bounds what runs.
SEQS=${SEQS:-16}
export VLLM_NEURON_MAX_KV_CONCURRENT=${VLLM_NEURON_MAX_KV_CONCURRENT:-$((SEQS + 4))}
export KV_CAP=${KV_CAP:-0.45} PREFILL_BUCKETS=${PREFILL_BUCKETS:-'[512, 1024, 2048, 4096, 8192]'}
exec "$HERE/serve.sh" "${MAX_LEN:-262144}" "$SEQS" "${SEQ_BUCKETS:-[1, 4, 8, 16]}" "${CTX_BUCKETS:-[98304]}"
