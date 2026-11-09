#!/bin/bash
# gpu_single_serve.sh: Hy3-FP8 on one 8-GPU node (H100 / B200) for the AgentX and single-node
# references: vLLM 0.29.0 (image from ./Dockerfile, or vllm/vllm-openai:v0.29.0 since no KV
# connector is used), TP=8, FP8 KV, MTP k=3, the customer's reference flags, port 8000.
# Container name hy3-single; logs: docker logs hy3-single.
# Env: IMAGE (hy3-efa-029), MAX_LEN (262144 for AgentX; 87040 for the 85k benches),
#      MODEL_DIR (/opt/dlami/nvme/models), EXTRA (more vllm serve flags).
set -euo pipefail
docker rm -f hy3-single >/dev/null 2>&1 || true
docker run -d --name hy3-single --gpus all --ipc=host --network host \
  -v ${MODEL_DIR:-/opt/dlami/nvme/models}:/models \
  -e VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm \
  ${IMAGE:-hy3-efa-029} /models/Hy3-FP8 --served-model-name hy3 \
  --tensor-parallel-size 8 \
  --tool-call-parser hy_v3 --enable-auto-tool-choice --reasoning-parser hy_v3 \
  --max-model-len ${MAX_LEN:-262144} --port 8000 \
  --kv-cache-dtype fp8 --speculative-config '{"method":"mtp","num_speculative_tokens":3}' ${EXTRA:-}
echo "started hy3-single on port 8000"
