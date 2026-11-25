#!/bin/bash
# run_pd_best.sh <label> prefill-node|decode-node [KEY=VALUE ...]
# run_pd.sh with the tuned 2P1D configuration for the 85k / 92% shared-prefix case
# (README "Recommended PD configuration"):
#   both nodes: FP8 KV, MTP k=3, max_model_len 87040, sparse decode 0.0625, sparse prefill 0.25
#   prefill node: two TP=32 prefill instances (P_INSTANCES=2), bench pd_bench_final.sh
#   decode node: TP=64, selection every 8 layers, fused decode MoE (no static DGE), batch buckets up
#     to 16, runtime collective algorithm without INTRA_RDH
# Start the decode node first; the prefill node waits for it, then runs the bench.
# Extra KEY=VALUE arguments are appended and override these (e.g. P_CONCS="1 2 4" D_CONCS=8).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LABEL=${1:?usage: run_pd_best.sh <label> prefill-node|decode-node [KEY=VALUE ...]}
ROLE=${2:?usage: run_pd_best.sh <label> prefill-node|decode-node [KEY=VALUE ...]}
shift 2
CODE=$(cd "$HERE/../../../../.." && pwd)
COMMON=(
  CONNECTOR=NeuronMooncakeConnector KV_DTYPE=fp8 MTP_K=3 MAX_LEN=87040
  HY3_KV_SCALE_PROFILE=$CODE/contrib/models/hy_v3/kv_profiles/Hy3-FP8_mtp_neuron.safetensors
  HY3_SPARSE_DECODE=0.0625 HY3_SPARSE_PREFILL=0.25
)
case $ROLE in
  prefill-node)
    ROLE_ARGS=(P_INSTANCES=2 BENCH=pd_bench_final) ;;
  decode-node)
    ROLE_ARGS=(
      HY3_SPARSE_SELECT_EVERY=8 HY3_DECODE_MOE_FUSED=1 HY3_DECODE_HITS=0
      KV_CAP=0.35 SEQS=16 "SEQ_BUCKETS=[1, 2, 4, 8, 10, 12, 16]" VLLM_NEURON_MAX_KV_CONCURRENT=20
      NEURON_RT_CC_ALG_TYPES=^INTRA_RDH_ALG
    ) ;;
  *) echo "role must be prefill-node or decode-node"; exit 1 ;;
esac
exec bash "$HERE/run_pd.sh" "$LABEL" "$ROLE" "${COMMON[@]}" "${ROLE_ARGS[@]}" "$@"
