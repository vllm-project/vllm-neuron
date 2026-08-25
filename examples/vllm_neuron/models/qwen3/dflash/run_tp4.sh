#!/bin/bash
# Qwen3 4B with DFlash on a single Trn2 chip (trn2.3xlarge), TP4, BF16.
#
# Qwen3 4B is dense, which is what makes DFlash pay at low concurrency: verifying
# a block costs almost the same as verifying one (measured 24.7ms for 1 token vs
# 26.8ms for 4). At concurrency 4 the 16-token block stops paying -- see README.

set -euo pipefail
set -x

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
DRAFT_MODEL_ID="${DRAFT_MODEL_ID:-z-lab/Qwen3-4B-DFlash-b16}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
BLOCK_SIZE=32

# This drafter's proposal block is 16, so it fills 15 masked slots per pass.
# num_speculative_tokens is fixed by training, not a tuning knob.
NUM_SPEC_TOKENS=15

# Size the KV cache to what this configuration can use. vLLM otherwise grows it
# into all spare HBM, and DFlash's context write goes through qkv_proj's
# in-kernel cache path, whose cost scales with the ALLOCATED cache rather than
# with the slots written.
NUM_BLOCKS="${NUM_BLOCKS:-$(( (MAX_MODEL_LEN * MAX_NUM_SEQS) / BLOCK_SIZE + 176 ))}"

# trn2.3xlarge has no EFA device; the affinity hint is a performance
# optimization, not a correctness requirement, and workers abort without this.
export NEURON_SKIP_EFA_AFFINITY="${NEURON_SKIP_EFA_AFFINITY:-1}"
export VLLM_NEURON_COMPILATION_TIMEOUT="${VLLM_NEURON_COMPILATION_TIMEOUT:-3600}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}"

echo "Starting Qwen3 4B with DFlash: target=$MODEL_ID draft=$DRAFT_MODEL_ID"
echo "  One Trn2 chip, BF16, TP4, ${NUM_SPEC_TOKENS} speculative tokens, ${NUM_BLOCKS} KV blocks"

vllm serve "$MODEL_ID" \
    --port "$PORT" \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --num-gpu-blocks-override "$NUM_BLOCKS" \
    --no-async-scheduling \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --disable-hybrid-kv-cache-manager \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"${DRAFT_MODEL_ID}\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS}}" \
    --additional-config '{"neuron_config": {"quantization": "bf16"}}'
