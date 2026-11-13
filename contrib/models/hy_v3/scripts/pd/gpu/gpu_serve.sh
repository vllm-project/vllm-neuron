#!/bin/bash
# gpu_serve.sh <prefill|decode>: one side of the Hy3-FP8 1P1D GPU reference on a p5.48xlarge
# (8x H100, 32 EFA): vLLM 0.29.0 (image from ./Dockerfile), TP=8, the customer's reference
# single-node flags, vLLM's MooncakeConnector over EFA. Prefill on port 8100 (Mooncake bootstrap
# 8998), decode on 8200. Container name hy3-<role>; logs: docker logs. Run the proxy and the
# benches (../pd_bench_final.sh) from the same image with --network host.
# Env: IMAGE (hy3-efa-029, the image built from ./Dockerfile), MAX_LEN (87040),
#      EXTRA (more vllm serve flags), MODEL_DIR (/opt/dlami/nvme/models), OFFLOAD_BYTES (host-DRAM KV offload).
set -euo pipefail
ROLE=${1:?usage: gpu_serve.sh <prefill|decode>}
case "$ROLE" in
  prefill) kv_role=kv_producer port=8100 ;;
  decode) kv_role=kv_consumer port=8200 ;;
  *) echo "role must be prefill or decode" >&2; exit 1 ;;
esac
docker rm -f hy3-$ROLE >/dev/null 2>&1 || true
KVT="{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"$kv_role\",\"kv_connector_extra_config\":{\"mooncake_protocol\":\"efa\"}}"
# OFFLOAD_BYTES=<n>: also offload KV to host DRAM (vLLM SimpleCPUOffloadConnector, write-through) through a
# MultiConnector, as InferenceX's vLLM PD AgentX recipes do. 1200000000000 (1.2 TB) on p5.
if [ -n "${OFFLOAD_BYTES:-}" ]; then
  KVT="{\"kv_connector\":\"MultiConnector\",\"kv_role\":\"$kv_role\",\"kv_connector_extra_config\":{\"connectors\":[$KVT,{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$OFFLOAD_BYTES,\"lazy_offload\":false}}]}}"
  # vLLM 0.29.0 asserts in MultiConnector's Prometheus metrics; mount the patched file (see its header).
  MOUNTS="-v $(cd "$(dirname "$0")" && pwd)/vllm029_multi_connector.py:/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:ro"
fi
docker run -d --name hy3-$ROLE --gpus all --privileged --ipc=host --network host \
  -v ${MODEL_DIR:-/opt/dlami/nvme/models}:/models --device /dev/gdrdrv ${MOUNTS:-} \
  -e VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm \
  -e VLLM_MOONCAKE_BOOTSTRAP_PORT=8998 \
  -e FI_PROVIDER=efa -e FI_EFA_USE_DEVICE_RDMA=1 -e FI_EFA_FORK_SAFE=1 \
  ${IMAGE:-hy3-efa-029} /models/Hy3-FP8 --served-model-name hy3 \
  --tensor-parallel-size 8 \
  --tool-call-parser hy_v3 --enable-auto-tool-choice --reasoning-parser hy_v3 \
  --max-model-len ${MAX_LEN:-87040} \
  --port $port \
  --kv-transfer-config "$KVT" ${EXTRA:-}
echo "started hy3-$ROLE on port $port"
