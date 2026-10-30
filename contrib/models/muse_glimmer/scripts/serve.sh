#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Serve meta-models/Muse-Glimmer-30B (BF16, text + image) on one trn2.3xlarge, TP=4.
# For video, set "video": 1 in --limit-mm-per-prompt (see README).
#
#   pip install -e contrib/models/muse_glimmer   # once: registers the Neuron model in every vLLM process
#   MODEL=/path/to/Muse-Glimmer-30B PORT=8000 bash contrib/models/muse_glimmer/scripts/serve.sh
#
# Add swap before the first start (the compiler can use a lot of host memory for the TP=4 graphs):
#   sudo fallocate -l 64G /swapfile && sudo chmod 600 /swapfile && sudo mkswap --force /swapfile && sudo swapon /swapfile
set -euo pipefail

MODEL=${MODEL:-meta-models/Muse-Glimmer-30B}
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}

export NEURON_SKIP_EFA_AFFINITY=${NEURON_SKIP_EFA_AFFINITY:-1}   # trn2.3xlarge has no EFA
# The default KV budget (0.30) leaves no device memory for the compiled graphs.
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=${VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION:-0.15}

exec vllm serve "$MODEL" \
  --tensor-parallel-size 4 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_MODEL_LEN" \
  --no-enable-prefix-caching \
  --max-logprobs 0 \
  --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --mm-processor-kwargs '{"max_image_tokens": 1024}' \
  --additional-config '{
    "neuron_config": {
      "num_batched_tokens_buckets": [512, 1024, 2048, 4096, 8192],
      "num_seqs_buckets": [1, 2, 4, 8, 16],
      "decode_context_length_buckets": [4096],
      "on_device_sampling_config": {"all_greedy": true}},
    "vision_neuron_config": {
      "vision_attention_block_size": 4096,
      "num_vision_tokens_buckets": [4096],
      "encoder_cache_num_blocks": 16,
      "tp_size": 4}}' \
  --port "$PORT"
