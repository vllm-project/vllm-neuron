#!/bin/bash
# OpenAI-compatible server for Hy4-preview (HYV4ForCausalLM) on a trn2.48xlarge (TP=64), native
# PyTorch route. The defaults are sized for 8K-token inputs and 1K-token outputs
# (contrib/benchmark's 8192 / 1024 cells):
#
#   bash contrib/models/hy4_preview/scripts/serve.sh <dir>/<checkpoint>-neuron-fp8
#
# Prerequisite (once): pip install -e contrib/models/hy4_preview
#
# Env:
#   SERVED_NAME     --served-model-name (default: the checkpoint dir name)
#   MAX_MODEL_LEN   default 9472: an 8K input with +3% jitter (8435) plus 1024
#                   output tokens, rounded up to a multiple of 128 (at most 32768)
#   MAX_NUM_SEQS    default 4. The latent cache is replicated on every rank, so the
#                   KV cache holds about 4 requests of 9472 tokens at GMU 0.8; the
#                   server's "Maximum concurrency for <MAX_MODEL_LEN> tokens per
#                   request" log line is the real limit.
#   DECODE_BUCKETS  default [1, 4]: one decode graph per batch size
#   CHUNK           prefill chunk, default 2048 (also --max-num-batched-tokens);
#                   longer prompts are prefilled in chunks
#   GMU             --gpu-memory-utilization, default 0.8
#   PORT            default 8000
set -eu
MODEL=${1:?usage: serve.sh <checkpoint dir> [extra vllm serve args...]}
shift
CHUNK=${CHUNK:-2048}
export VLLM_NEURON_BACKEND=${VLLM_NEURON_BACKEND:-neuron_native}
export NEURON_SKIP_EFA_AFFINITY=${NEURON_SKIP_EFA_AFFINITY:-1}   # instances without EFA
# The native route compiles every rank's graphs (64 here); each neuronx-cc job
# takes 40-50 GB of host memory. vllm-neuron's default of 8 ran a 2 TB host out
# of memory on the 32K graphs; 6 leaves headroom, while 2 left a cold compile
# unfinished after 7 h. Ranks finish their graphs at different times and wait
# for the rest in gloo barriers, so both the distributed timeout and
# vllm-neuron's own barrier timeout (default 3600 s, which a cold 9472 compile
# outlasted) cover a cold compile. Also give the first 64-rank executions more
# than the runtime's 30 s.
export VLLM_NEURON_PARALLEL_COMPILE_WORKERS=${VLLM_NEURON_PARALLEL_COMPILE_WORKERS:-6}
export VLLM_NEURON_BARRIER_TIMEOUT=${VLLM_NEURON_BARRIER_TIMEOUT:-43200}
export NEURON_RT_EXEC_TIMEOUT=${NEURON_RT_EXEC_TIMEOUT:-1200}
exec vllm serve "$MODEL" \
  --served-model-name "${SERVED_NAME:-$(basename "${MODEL%/}")}" \
  --tensor-parallel-size 64 \
  --max-model-len "${MAX_MODEL_LEN:-9472}" \
  --max-num-batched-tokens "$CHUNK" \
  --max-num-seqs "${MAX_NUM_SEQS:-4}" \
  --gpu-memory-utilization "${GMU:-0.8}" \
  --no-enable-prefix-caching \
  --distributed-timeout-seconds 43200 \
  --additional-config "{\"neuron_config\": {\"quantization\": \"bf16\",
      \"num_batched_tokens_buckets\": [$CHUNK],
      \"num_seqs_buckets\": ${DECODE_BUCKETS:-[1, 4]},
      \"on_device_sampling_config\": {\"all_greedy\": true}}}" \
  --port "${PORT:-8000}" "$@"
