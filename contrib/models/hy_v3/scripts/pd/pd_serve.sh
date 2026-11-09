#!/bin/bash
# pd_serve.sh <prefill|decode|decode-bench>: one side of a Hy3 1P1D disaggregated
# setup, on top of contrib/models/hy_v3/scripts/serve.sh.
#
#   prefill       kv_producer on PORT 8100: prefix cache, prefill NEFFs only
#                 (VLLM_NEURON_SKIP_DECODE_WARMUP=1)
#   decode        kv_consumer on PORT 8200: pulls the KV over NIXL, decode NEFFs only
#                 (VLLM_NEURON_SKIP_PREFILL_WARMUP=1)
#   decode-bench  kv_consumer with NeuronDecodeBenchConnector: no prefill server, the prompt
#                 is taken as already in the cache (KV left uninitialized, so outputs are
#                 garbage and MTP acceptance is meaningless): the dedicated-decode step time
#
# Config is serve_85k.sh's (the 85k / 92% shared-prefix customer case, no DCP):
# max_model_len MAX_LEN (default 86016; 87040 for 85k in + 1k out), prefill buckets 512..4096, prefix caching, block size 32.
#
# Env:
#   CODE        vllm-neuron checkout with contrib/models/hy_v3 (default: the one holding this script)
#   PKG         optional extra PYTHONPATH entry (e.g. a dir with a hy_v3_neuron link to another checkout)
#   MODEL       checkpoint (default tencent/Hy3-FP8)
#   KV_DTYPE    auto (BF16, default) | fp8 (needs HY3_KV_SCALE_PROFILE; with MTP the _mtp_ profile)
#   MTP_K       MTP draft tokens on BOTH sides (the prefill side has to fill the draft layer's KV)
#   CONNECTOR   NeuronNixlConnector (default) | NeuronMooncakeConnector (vllm-project/vllm-neuron#90,
#               mooncake-transfer-engine-non-cuda >= 0.3.14, protocol efa; prefill pushes the KV
#               to decode; use pd_proxy.sh with CONNECTOR=NeuronMooncakeConnector) | NixlConnector
#   BACKEND     NIXL backend, default LIBFABRIC (EFA). Needs an EFA instance for device memory.
#   MC_BOOTSTRAP_PORT  Mooncake bootstrap server on the prefill side (default 8998)
#   SEQS, SEQ_BUCKETS, CTX_BUCKETS, KV_CAP   override the per-role defaults below (KV_CAP: decode
#               and decode-bench only)
#   P_KV_CAP, P_PREFILL_BUCKETS   the prefill role's KV budget (0.45) and prefill buckets
#               ([512 .. 8192]); PREFILL_BUCKETS overrides the buckets on either role
#   SPLIT       1: single-node split, prefill on NeuronCores 0-31 and decode on 32-63 (TP=32,
#               EP=32 each), side-channel ports 5557 / 5558. Default: one role per node, TP=64.
#   P_HALF      prefill only, 0 | 1: this prefill instance takes half the node (logical NeuronCores
#               0-31 or 32-63, TP=32 / EP=32), API port 8100 + P_HALF, NIXL side channel
#               5557 + 2 * P_HALF, Mooncake bootstrap 8998 + P_HALF. Two such instances on one
#               node prefill twice as many requests at once (run_pd.sh P_INSTANCES=2).
#   PORT, NIXL_PORT   override the role's API / NIXL side-channel port
#   D_DP        decode only: attention DP with D_DP replicas of TP=64/D_DP each (vLLM
#               --data-parallel-size), experts over all 64 ranks (EP=64). SEQS / SEQ_BUCKETS
#               are then per replica.
#   P_DCP, D_DCP  decode-context-parallel size for that role only (serve.sh DCP), e.g. D_DCP=8:
#               prefill without DCP (20% faster) into a DCP=8 decode (8x the KV per rank, 1/8 of
#               the history read per step). Set D_DCP on BOTH nodes: the prefill side needs it for
#               the Mooncake block map (remote_dcp_size). No MTP on either side then (the draft
#               layer's KV must exist on both). NeuronNixlConnector: TP -> DCP natively; NeuronMooncakeConnector: prefill DCP 1 -> decode DCP D only.
set -euo pipefail
ROLE=${1:?usage: pd_serve.sh <prefill|decode|decode-bench>}
CODE=${CODE:-$(cd "$(dirname "$0")/../../../../.." && pwd)}
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=${MODEL:-tencent/Hy3-FP8}
KV_DTYPE=${KV_DTYPE:-auto}
CONNECTOR=${CONNECTOR:-NeuronNixlConnector}
BACKEND=${BACKEND:-LIBFABRIC}

export PYTHONPATH=$CODE:$HERE${PKG:+:$PKG} PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
# Mooncake EFA: bound the concurrent MR registrations (default: one thread per buffer, ~10k at
# once per node), a suspect for the rare unusable-buffer failures (see README).
export MC_MAX_CONCURRENT_REG_MR=${MC_MAX_CONCURRENT_REG_MR:-8}
# Register each HBM buffer only on its topology-local NICs (2 per Neuron device on trn2): transfers
# use only those anyway, and "all" puts every rank's buffers on all 16 NICs (8x the NIC page-table
# load). The default since Mooncake v0.3.14 (kvcache-ai/Mooncake#5102); explicit for older wheels.
export MC_EFA_NIC_SELECTION=${MC_EFA_NIC_SELECTION:-local}
export VLLM_NEURON_BACKEND=neuron_native NEURON_RT_EXEC_TIMEOUT=1200
export VLLM_NEURON_BARRIER_TIMEOUT=${VLLM_NEURON_BARRIER_TIMEOUT:-43200} VLLM_ENGINE_READY_TIMEOUT_S=14400
# Not 0.0.0.0: the prefill side also advertises this address in kv_transfer_params, and a decode
# told 0.0.0.0 handshakes with itself ("Remote NIXL agent engine ID mismatch").
export VLLM_NIXL_SIDE_CHANNEL_HOST=${VLLM_NIXL_SIDE_CHANNEL_HOST:-$(hostname -I | awk '{print $1}')}
# 64 ranks x the default 8 trace / 8 compile workers each: the decode side's fork of its trace
# pool failed with ENOMEM on a fresh trn2 (2 TB host). 2 each, as for every TP=64 native run.
export NEURON_LIBTORCH_PARALLEL_TRACE_WORKERS=${NEURON_LIBTORCH_PARALLEL_TRACE_WORKERS:-2}
export VLLM_NEURON_PARALLEL_COMPILE_WORKERS=${VLLM_NEURON_PARALLEL_COMPILE_WORKERS:-2}
# EFA instances need the runtime's EFA affinity (serve.sh defaults this to 1 for instances without EFA).
if fi_info -p efa >/dev/null 2>&1; then export NEURON_SKIP_EFA_AFFINITY=0; else export NEURON_SKIP_EFA_AFFINITY=1; fi

# Per role: the prefill side keeps the shared prefix and the running prompts; the decode side
# keeps every running request at max_model_len (the scheduler admits by worst case).
if [ "$KV_DTYPE" = fp8 ]; then d_seqs=16 d_buckets='[1, 4, 8, 10, 12, 16]' d_ctx='[32768]' d_cap=0.35
else d_seqs=4 d_buckets='[1, 4]' d_ctx='[8192, 32768, 65536]' d_cap=0.65; fi
[ -n "${MTP_K:-}" ] && [ "$KV_DTYPE" != fp8 ] && d_cap=0.61
# FP8 decode: KV_CAP 0.35 holds 400k tokens. With the 78k shared prefix, 16 requests use about 210k,
# but the scheduler's worst-case cap (KV tokens / max_model_len) would admit 4. It is overridden below
# (VLLM_NEURON_MAX_KV_CONCURRENT); block allocation still bounds what runs.
# HY3_SPARSE_DECODE / HY3_SPARSE_PREFILL (set the summaries' owner, HY3_SPARSE_DECODE, on BOTH nodes:
# prefill writes the K summaries, the connector moves them; HY3_SPARSE_PREFILL only acts on the prefill node):
# as serve_85k.sh, only the max_model_len decode bucket (the summaries do not fit next to the
# 32768-bucket decode NEFFs). Needs NeuronMooncakeConnector or NeuronNixlConnector with the
# CONNECTOR_EXTRA_KV_TENSORS support of vllm-project/vllm-neuron#90.
if [ -n "${HY3_SPARSE_DECODE:-}${HY3_SPARSE_PREFILL:-}" ]; then
  d_ctx=""
  [ "$CONNECTOR" = NixlConnector ] && { echo "HY3_SPARSE_* in PD needs NeuronMooncakeConnector or NeuronNixlConnector (vllm-project/vllm-neuron#90): the K summaries move as extra regions" >&2; exit 1; }
  grep -q CONNECTOR_EXTRA_KV_TENSORS "$CODE/contrib/models/hy_v3/src/model.py" || { echo "$CODE: hy_v3 without CONNECTOR_EXTRA_KV_TENSORS" >&2; exit 1; }
fi
case "$ROLE" in
  prefill)
    kv_role=kv_producer port=8100 nixl_port=5557
    export VLLM_NEURON_SKIP_DECODE_WARMUP=1
    # 8192-token prefill steps (+3% alone, and what the MoE map-pad fix needs to reach 1.37 req/s
    # at C=8) need the HBM that a smaller KV pool leaves: P_KV_CAP 0.45 holds 514k tokens, the
    # running prefills and the prefix cache. (8192 with KV_CAP 0.65 fails to allocate.)
    pb=${P_PREFILL_BUCKETS:-'[512, 1024, 2048, 4096, 8192]'}
    seqs=${SEQS:-16} buckets=${SEQ_BUCKETS:-'[1, 4, 8, 16]'} ctx=${CTX_BUCKETS:-} cap=${P_KV_CAP:-0.45}
    export VLLM_NEURON_MAX_KV_CONCURRENT=${VLLM_NEURON_MAX_KV_CONCURRENT:-16} ;;
  decode)
    kv_role=kv_consumer port=8200 nixl_port=5557
    export VLLM_NEURON_SKIP_PREFILL_WARMUP=1
    seqs=${SEQS:-$d_seqs} buckets=${SEQ_BUCKETS:-$d_buckets} ctx=${CTX_BUCKETS:-$d_ctx} cap=${KV_CAP:-$d_cap}
    if [ "$KV_DTYPE" = fp8 ]; then
      export VLLM_NEURON_MAX_KV_CONCURRENT=${VLLM_NEURON_MAX_KV_CONCURRENT:-$((seqs + 4))}
      export HY3_SPARSE_SELECT_EVERY=${HY3_SPARSE_SELECT_EVERY:-8} HY3_MOE_STATIC_DGE=${HY3_MOE_STATIC_DGE:-1}
    fi ;;
  decode-bench)
    kv_role=kv_consumer port=8200 nixl_port=5557
    export VLLM_NEURON_SKIP_PREFILL_WARMUP=1
    CONNECTOR=NeuronDecodeBenchConnector
    seqs=${SEQS:-$d_seqs} buckets=${SEQ_BUCKETS:-$d_buckets} ctx=${CTX_BUCKETS:-$d_ctx} cap=${KV_CAP:-$d_cap} ;;
  *) echo "role must be prefill, decode or decode-bench" >&2; exit 1 ;;
esac

extra=""
# vllm-neuron 0.29 reads NEURON_VISIBLE_DEVICES as logical NeuronCores (64 on a trn2.48xlarge at
# LNC=2) and needs one per local rank: a TP=32 half is 0-31 or 32-63.
if [ "${SPLIT:-0}" = 1 ]; then
  if [ "$ROLE" = prefill ]; then export NEURON_VISIBLE_DEVICES=0-31; nixl_port=5557
  else export NEURON_VISIBLE_DEVICES=32-63; nixl_port=5558; fi
  extra="--tensor-parallel-size 32"   # argparse: the last occurrence wins over serve.sh's 64
  export EP=${EP:-32}
elif [ -n "${P_HALF:-}" ]; then
  [ "$ROLE" = prefill ] || { echo "P_HALF is for the prefill role" >&2; exit 1; }
  case "$P_HALF" in 0|1) ;; *) echo "P_HALF must be 0 or 1" >&2; exit 1 ;; esac
  export NEURON_VISIBLE_DEVICES=$((32 * P_HALF))-$((32 * P_HALF + 31))
  port=$((8100 + P_HALF)) nixl_port=$((5557 + 2 * P_HALF))
  export MC_BOOTSTRAP_PORT=${MC_BOOTSTRAP_PORT:-$((8998 + P_HALF))}
  extra="--tensor-parallel-size 32"
  export EP=${EP:-32}
fi
export VLLM_NIXL_SIDE_CHANNEL_PORT=${NIXL_PORT:-$nixl_port}
if [ "$ROLE" != prefill ] && [ "${D_DP:-1}" != 1 ]; then
  grep -q "self.cross_dp_ep" "$CODE/contrib/models/hy_v3/src/model.py" || { echo "$CODE: hy_v3 without attention DP" >&2; exit 1; }
  extra="$extra --tensor-parallel-size $((64 / D_DP)) --data-parallel-size $D_DP"
  export EP=${EP:-64}
  # Per replica: a c=8 load is ~1 request per replica.
  seqs=${SEQS:-4} buckets=${SEQ_BUCKETS:-'[1, 4]'}
fi
if [ "$ROLE" = prefill ]; then role_dcp=${P_DCP:-}; else role_dcp=${D_DCP:-}; fi
[ "${D_DCP:-1}${P_DCP:-1}" != 11 ] && [ -n "${MTP_K:-}" ] && { echo "MTP does not run with DCP; unset MTP_K on both sides" >&2; exit 1; }
if [ -n "$role_dcp" ] && [ "$role_dcp" != 1 ]; then
  [ -n "${MTP_K:-}" ] && { echo "MTP does not run with DCP" >&2; exit 1; }
  [ "$CONNECTOR" = NeuronMooncakeConnector ] && [ "$ROLE" = prefill ] && { echo "NeuronMooncakeConnector: prefill DCP must be 1" >&2; exit 1; }
  export DCP=$role_dcp
fi

# Compact JSON (no spaces): serve.sh appends EXTRA_ARGS unquoted, so it is word-split.
if [ "$CONNECTOR" = NeuronDecodeBenchConnector ]; then
  kvt='{"kv_connector":"NeuronDecodeBenchConnector","kv_connector_module_path":"vllm_neuron.vllm.kv_connector.neuron_decode_bench_connector","kv_role":"kv_consumer"}'
elif [ "$CONNECTOR" = NeuronMooncakeConnector ]; then
  # vllm-project/vllm-neuron#90 (not merged yet: apply it to CODE).
  [ -f "$CODE/vllm_neuron/vllm/kv_connector/neuron_mooncake_connector.py" ] || { echo "$CODE has no NeuronMooncakeConnector: apply vllm-project/vllm-neuron#90" >&2; exit 1; }
  mc_mod=vllm_neuron.vllm.kv_connector.neuron_mooncake_connector
  export VLLM_MOONCAKE_BOOTSTRAP_PORT=${MC_BOOTSTRAP_PORT:-8998}
  # Prefill into a DCP decode: the prefill side maps its blocks onto the decode's DCP shards.
  mc_dcp=""; [ "$ROLE" = prefill ] && [ "${D_DCP:-1}" != 1 ] && mc_dcp=",\"remote_dcp_size\":$D_DCP"
  kvt="{\"kv_connector\":\"NeuronMooncakeConnector\",\"kv_connector_module_path\":\"$mc_mod\",\"kv_role\":\"$kv_role\",\"kv_connector_extra_config\":{\"mooncake_protocol\":\"efa\",\"num_workers\":${MC_WORKERS:-10}$mc_dcp}}"
elif [ "$CONNECTOR" = NeuronNixlConnector ]; then
  kvt="{\"kv_connector\":\"NeuronNixlConnector\",\"kv_connector_module_path\":\"vllm_neuron.vllm.kv_connector.neuron_nixl_connector\",\"kv_role\":\"$kv_role\",\"kv_buffer_device\":\"cuda\",\"kv_connector_extra_config\":{\"backends\":[\"$BACKEND\"]}}"
else
  kvt="{\"kv_connector\":\"$CONNECTOR\",\"kv_role\":\"$kv_role\",\"kv_buffer_device\":\"cuda\",\"kv_connector_extra_config\":{\"backends\":[\"$BACKEND\"]}}"
fi

echo "$(date +%T) pd_serve $ROLE: port ${PORT:-$port}, nixl ${VLLM_NIXL_SIDE_CHANNEL_PORT}, kv $KV_DTYPE, mtp ${MTP_K:-off}, seqs $seqs $buckets, ctx ${ctx:-none}, KV_CAP $cap, efa_skip $NEURON_SKIP_EFA_AFFINITY, split ${SPLIT:-0}, p_half ${P_HALF:-none}, dcp ${DCP:-1}"
echo "kv-transfer-config: $kvt"
[ -n "${MTP_K:-}" ] && export HY3_DECODE_HITS_MAX_T=${HY3_DECODE_HITS_MAX_T:-$((4 * (MTP_K + 1)))}
set -f  # the JSON's [ ] must not glob when EXTRA_ARGS is word-split
export KV_DTYPE KV_CAP=$cap PORT=${PORT:-$port} MODEL PREFILL_BUCKETS=${PREFILL_BUCKETS:-${pb:-'[512, 1024, 2048, 4096]'}}
export EXTRA_ARGS="$extra ${EXTRA_ARGS:-} --kv-transfer-config $kvt"
if [ -n "$ctx" ]; then
  exec bash -f "$CODE/contrib/models/hy_v3/scripts/serve.sh" "${MAX_LEN:-86016}" "$seqs" "$buckets" "$ctx"
else
  exec bash -f "$CODE/contrib/models/hy_v3/scripts/serve.sh" "${MAX_LEN:-86016}" "$seqs" "$buckets"
fi
