#!/bin/bash
# OpenAI-compatible server for the Qwen3.5 contrib port (text only), on the native
# PyTorch route. The defaults are sized for 8K-token inputs and 1K-token outputs at
# up to 32 concurrent requests (contrib/benchmark's 8192 / 1024 cells):
#
#   bash contrib/models/qwen3_5/scripts/serve.sh /path/to/Qwen3.5-27B
#
# Prerequisite (once): pip install -e contrib/models/qwen3_5, so vllm serve's
# processes register the architectures.
#
# Env (each has a per-checkpoint default, see below):
#   SERVED_NAME       --served-model-name (default: the checkpoint dir name)
#   MAX_MODEL_LEN     default 9472: an 8K input with +3% jitter (8435) plus 1024
#                     output tokens, rounded up to a multiple of 128
#   MAX_NUM_SEQS      default 32 (30 for 397B-A17B, its KV limit at 9472)
#   PREFILL_BUCKETS   default [8448, 9472]. Prefill is single-shot (DeltaNet cannot
#                     continue from a chunk), so the last bucket must be
#                     MAX_MODEL_LEN; 8448 keeps an 8K prompt from being padded to it.
#                     Add smaller buckets for shorter prompts, e.g.
#                     [1024, 2048, 4096, 8448, 9472]; each is one more compile.
#   DECODE_BUCKETS    default [1, 4, 8, 16, MAX_NUM_SEQS]: one graph per batch size
#   TP, EP_DEGREE     tensor / expert parallel degree
#   GMU               --gpu-memory-utilization
#   PORT              default 8000
#
# Concurrency: the server admits a request only while every running one could
# grow to MAX_MODEL_LEN. Its log line "Maximum concurrency for 9,472 tokens per
# request: X" is the real limit; the GMU defaults put X above 32 on trn2.48xlarge.
set -eu
MODEL=${1:?usage: serve.sh <checkpoint dir> [extra vllm serve args...]}
shift
name=$(basename "${MODEL%/}")
# seqs / decode buckets: the KV cache's concurrency at MAX_MODEL_LEN 9472 bounds
# them (397B-A17B holds 30.82 requests); the last bucket must equal seqs.
case "$name" in
  *397B*) tp=64; ep=8; gmu=0.7;  seqs=30 ;;   # moe_intermediate 1024 needs EP for tp_degree <= 8
  *35B*)  tp=4;  ep=;  gmu=0.85; seqs=32 ;;
  *27B*)  tp=4;  ep=;  gmu=0.9;  seqs=32 ;;
  *)      tp=4;  ep=;  gmu=;     seqs=32 ;;   # 2B: vLLM's default leaves KV for 100+ requests
esac
TP=${TP:-$tp}; EP_DEGREE=${EP_DEGREE:-$ep}; GMU=${GMU:-$gmu}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-9472}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-$seqs}
PREFILL_BUCKETS=${PREFILL_BUCKETS:-[8448, $MAX_MODEL_LEN]}
DECODE_BUCKETS=${DECODE_BUCKETS:-[1, 4, 8, 16, $MAX_NUM_SEQS]}

export VLLM_NEURON_BACKEND=${VLLM_NEURON_BACKEND:-neuron_native}
export NEURON_SKIP_EFA_AFFINITY=${NEURON_SKIP_EFA_AFFINITY:-1}   # instances without EFA
if [ "$TP" -ge 64 ]; then
  # The native route compiles the decode graphs once per rank: cap neuronx-cc's
  # parallel jobs per rank (8 run a 2 TB host out of memory), and give the first
  # 64-rank executions more than the runtime's 30 s.
  export VLLM_NEURON_PARALLEL_COMPILE_WORKERS=${VLLM_NEURON_PARALLEL_COMPILE_WORKERS:-2}
  export NEURON_RT_EXEC_TIMEOUT=${NEURON_RT_EXEC_TIMEOUT:-1200}
fi

ep_args=(); ep_cfg=""
if [ -n "$EP_DEGREE" ]; then ep_args=(--enable-expert-parallel); ep_cfg="\"ep_degree\": $EP_DEGREE,"; fi
gmu_args=(); [ -n "$GMU" ] && gmu_args=(--gpu-memory-utilization "$GMU")

exec vllm serve "$MODEL" \
  --served-model-name "${SERVED_NAME:-$name}" \
  --tensor-parallel-size "$TP" "${ep_args[@]}" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_MODEL_LEN" \
  --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  "${gmu_args[@]}" \
  --additional-config "{\"neuron_config\": {$ep_cfg
      \"quantization\": \"bf16\",
      \"num_batched_tokens_buckets\": $PREFILL_BUCKETS,
      \"num_seqs_buckets\": $DECODE_BUCKETS,
      \"on_device_sampling_config\": {\"all_greedy\": true}}}" \
  --port "${PORT:-8000}" "$@"
