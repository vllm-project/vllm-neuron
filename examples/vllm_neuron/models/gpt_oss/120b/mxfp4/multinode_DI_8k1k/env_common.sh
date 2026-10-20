#!/bin/bash
# GPT-OSS 120B, 8K/1K, 6P1D disaggregated inference with the decode
# megakernel. Shared settings for every script in this directory.
#
# Each run_*.sh script sources this file. Export PREFILL_HOST and DECODE_HOST
# before you run a script. Every other value has a default.
#
# The values give the tested configuration: 6 prefill replicas, 1 decode
# server, 8,192 input tokens, 1,024 output tokens, and 128 concurrent
# requests. Keep them to run the tested configuration.

# The directory of this file. The scripts find the schedule file, the helper
# and the proxy from here, so the directory is self-contained.
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXAMPLE_DIR

# ------------------------------------------------------------------ Addresses
# The private IP of the prefill node (node A). It runs the 6 prefill
# replicas, the proxy and the benchmark client.
export PREFILL_HOST="${PREFILL_HOST:-<prefill-host>}"
# The private IP of the decode node (node B). It runs the decode server.
export DECODE_HOST="${DECODE_HOST:-<decode-host>}"

# Stop a server script when the two addresses are not set.
require_hosts() {
    if [[ "$PREFILL_HOST" == "<"* || "$DECODE_HOST" == "<"* ]]; then
        echo "PREFILL_HOST or DECODE_HOST is not set. Set both to the private IPs of the two nodes, then run the script again." >&2
        exit 1
    fi
}

# The model: a Hugging Face model ID, or a local path to the checkpoint.
export MODEL_ID="${MODEL_ID:-openai/gpt-oss-120b}"

# ------------------------------------------------------------------ Files
# The benchmark results, and the server logs.
export BENCH_RESULT_DIR="${BENCH_RESULT_DIR:-$HOME/gpt-oss-120b-8k1k-results}"
export SERVER_LOG_DIR="${SERVER_LOG_DIR:-$BENCH_RESULT_DIR/logs}"

# The DMA-order schedule of the decode megakernel. run_decode.sh writes a
# copy for the decode graph to MEGAKERNEL_SCHEDULE_DIR. The path of the copy
# is part of the compile-cache key, so keep one directory. A new directory
# makes the decode server compile its graph again.
export MEGAKERNEL_SCHEDULE="${MEGAKERNEL_SCHEDULE:-$EXAMPLE_DIR/gpt_oss_120b_megakernel_dma_order.json}"
export MEGAKERNEL_SCHEDULE_DIR="${MEGAKERNEL_SCHEDULE_DIR:-$HOME/.cache/vllm-neuron/megakernel_schedule}"

# The DI proxy of the vLLM Neuron examples tree.
export PROXY_SCRIPT="${PROXY_SCRIPT:-$EXAMPLE_DIR/../../../../../vllm/disaggregated_inference/toy_proxy_server.py}"

# ------------------------------------------------------------------ Shape
# These values are the tested configuration of the megakernel decode. See the
# tutorial, section "Configuration reference".
export INPUT_LEN=8192
export OUTPUT_LEN=1024
export MAX_MODEL_LEN=10240
export BLOCK_SIZE=128
export KV_CACHE_DTYPE=fp8

# Six prefill replicas keep one decode server busy.
export NUM_PREFILL=6
export PREFILL_TP=4
export PREFILL_EP_DEGREE=2

export DECODE_TP=8
export DECODE_DP=16
export DECODE_EP_DEGREE=128
# The partition axis caps the per-rank decode batch at 128 * TP / EP = 8.
export DECODE_MAX_NUM_SEQS=8
# 128 requests in flight fill the decode server: 8 for each of the 16 ranks.
export CONCURRENCY=$((DECODE_MAX_NUM_SEQS * DECODE_DP))

# ------------------------------------------------------------------ Ports
export PROXY_PORT=8000
# Replica i serves HTTP on 8100 + i.
export PREFILL_BASE_PORT=8100
export DECODE_PORT=8200
# Replica i uses the NIXL side-channel port 5559 + 100 * i.
export PREFILL_BASE_NIXL=5559
# The decode side-channel port is past the last prefill port. Replicas
# stride by 100.
export DECODE_NIXL_PORT=$((PREFILL_BASE_NIXL + 32 * 100))

# ------------------------------------------- Environment that BOTH roles need
# The compile route. libtorch-neuronx-lite reads this variable itself, and it
# selects its native-route setup only for an explicit value. Set vllm_neuron
# for the XLA route.
export VLLM_NEURON_BACKEND="${VLLM_NEURON_BACKEND:-neuron_native}"
if [ "$VLLM_NEURON_BACKEND" != "vllm_neuron" ]; then
    # The native route needs the graph-capture backend off.
    export NEURON_LIBTORCH_DISABLE_GRAPH_CAPTURE_BACKEND=1
    # Turn off the vocab-sharding FX pass. The per-rank vocab offset then
    # stays out of the graph.
    export TORCH_NEURONX_VOCAB_SHARDING_SPMD_DISABLE=1
fi
# Packed-FP8 KV writes benign out-of-bound skip notifications. Without this
# setting, the notifications stop the decode server on a long run.
export NEURON_RT_DBG_VECTOR_DGE_SKIP_NOTIF=1
# The NIXL progress thread polls with no delay by default and takes a core
# from the worker. A delay gives that host time back to the step.
export VLLM_NEURON_NIXL_PROGRESS_THREAD_DELAY_US=250
# The NIXL side channel must accept the KV handshake from the other node.
export VLLM_NIXL_SIDE_CHANNEL_HOST="${VLLM_NIXL_SIDE_CHANNEL_HOST:-0.0.0.0}"
# The compile of one prefill graph takes several minutes. The default limit
# for one graph is 600 s.
export NEURON_LIBTORCH_COMPILATION_TIMEOUT=2400
# vLLM waits 600 s for the engine by default. A server that compiles its
# graphs needs more time.
export VLLM_ENGINE_READY_TIMEOUT_S=14400
