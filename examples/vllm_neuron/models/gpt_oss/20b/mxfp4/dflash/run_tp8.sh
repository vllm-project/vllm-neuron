#!/bin/bash
# GPT-OSS 20B with DFlash on a Trn2 48xlarge using TP8 and BF16.
#
#
# DFlash does not use MXFP4 despite this script's location in the mxfp4/ tree,
# and is not yet supported on Trn3.

set -euo pipefail
set -x

MODEL_ID="${MODEL_ID:-openai/gpt-oss-20b}"
DRAFT_MODEL_ID="${DRAFT_MODEL_ID:-z-lab/gpt-oss-20b-DFlash}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
BLOCK_SIZE=32

# Size the KV cache to what this configuration can actually use. By default
# vLLM grows it into all spare HBM (~434k tokens here), but DFlash's context
# write goes through qkv_proj's in-kernel cache path, whose cost scales with
# the ALLOCATED cache rather than with the slots written -- it reshapes and
# copies back the whole per-layer tensor on every decode step. Right-sizing
# the allocation measured 23.5 -> 40.9 tok/s at concurrency 1.
NUM_BLOCKS="${NUM_BLOCKS:-$(( (MAX_MODEL_LEN * MAX_NUM_SEQS) / BLOCK_SIZE + 176 ))}"

# trn2.3xlarge has no EFA device; the EFA CPU-affinity hint is a performance
# optimization, not a correctness requirement, and the workers abort without
# this. Harmless on instances that do have EFA.
export NEURON_SKIP_EFA_AFFINITY="${NEURON_SKIP_EFA_AFFINITY:-1}"
export VLLM_NEURON_COMPILATION_TIMEOUT="${VLLM_NEURON_COMPILATION_TIMEOUT:-3600}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}"

echo "Starting GPT-OSS 20B with DFlash: target=$MODEL_ID draft=$DRAFT_MODEL_ID"
echo "  Trn2 48xlarge, BF16, TP8, 7 speculative tokens, ${NUM_BLOCKS} KV blocks"

vllm serve "$MODEL_ID" \
    --port "$PORT" \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --num-gpu-blocks-override "$NUM_BLOCKS" \
    --no-async-scheduling \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --disable-hybrid-kv-cache-manager \
    --hf-overrides '{"quantization_config": {}}' \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"${DRAFT_MODEL_ID}\",\"num_speculative_tokens\":7}" \
    --additional-config '{"neuron_config": {"quantization": "bf16"}}'
