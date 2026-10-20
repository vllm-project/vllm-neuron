#!/bin/bash
# gpt-oss single-instance (non-DI) server — TP8.
#
# The simplest way to stand gpt-oss up: one server serving the whole model with
# tensor parallelism. Use it for functional validation, development, and light
# workloads. For the best throughput and latency under load, use the
# disaggregated-inference target config in ../multinode_DI/ instead.
#
# Mirrors the "Deploy on a single instance (non-DI)" section of the gpt-oss
# tutorial: docs/tutorials/tutorial-gpt-oss.md
#
# Defaults to gpt-oss 120B. For gpt-oss 20B, set MODEL_ID="openai/gpt-oss-20b";
# the same command works on Trn3 (MXFP4) or Trn2 (change "quantization" to "bf16").

set -x

MODEL_ID="${MODEL_ID:-openai/gpt-oss-120b}"

# Environment variables (per the tutorial's single-instance section):
# compilation and execution timeouts for gpt-oss (120B cold compile is large).
export VLLM_NEURON_COMPILATION_TIMEOUT=1200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200

echo "Starting single-instance (non-DI) gpt-oss server: $MODEL_ID"
echo "  Port: 8000, TP8"

vllm serve "$MODEL_ID" \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 4 \
    --hf-overrides '{"quantization_config": {}}' \
    --additional-config '{
        "neuron_config": {
            "quantization": "mxfp4",
            "kv_segment_size_buckets": [8192],
            "num_batched_tokens_buckets": [8192],
            "num_seqs_buckets": [4]
        }
    }'
