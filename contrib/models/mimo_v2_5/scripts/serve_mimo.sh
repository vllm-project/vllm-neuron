#!/bin/bash
# MiMo-V2.5 / MiMo-V2.5-Pro single-instance server — TP64 with expert parallelism.
#
# V2.5: 48 layers / 256 routed experts / top-8 / hidden 4096. At TP=64 with EP=64
# each rank owns 4 experts, which is what the MoE kernels are sized against below.
# Pro:  70 layers / 384 routed experts / hidden 6144, 6 experts per rank. Its
# experts stay fp8 on device (MIMO_V2_5_EXPERT_DTYPE=auto resolves to fp8 for Pro
# and bf16 for V2.5; set EXPERT_DTYPE to force one):
#
#   MODEL=XiaomiMiMo/MiMo-V2.5-Pro SERVED_NAME=MiMo-V2.5-Pro bash serve_mimo.sh
#
# Prerequisite (once): install this folder as a package so its
# vllm.general_plugins entry point registers MiMoV2ForCausalLM in every process
# vllm serve starts -- no change to vllm_neuron/model/registry.py:
#
#   pip install -e contrib/models/mimo_v2_5
#
# Text decoder only. --hf-overrides nulls two keys of the checkpoint config:
#   * quantization_config: vLLM would derive quantization="fp8" from it and
#     reject that against NeuronPlatform.supported_quantization. The weight
#     loaders dequantize the FP8 tensors themselves and decide fp8-vs-bf16 by
#     probing the checkpoint for *_scale_inv keys, not from this config.
#   * vision_config: vLLM's MimoV2ModelArchConfigConvertor rewrites
#     architectures to MiMoV2OmniForCausalLM whenever it is truthy, which would
#     route past this port. A null value still satisfies the platform's
#     hasattr(hf_config, "vision_config") gate, so an inert
#     encoder_neuron_config gets built; MiMoV2ForCausalLM.from_configs absorbs
#     it. Its bucket is pinned to the encoder block size (2048) below: the
#     auto-generated one is next_pow2(SEQ), which for SEQ < 2048 is below the
#     block, and the engine core's second config pass then rejects it as an
#     explicit bucket ("Largest bucket (1024) is smaller than
#     encoder_attention_block_size (2048)"). Nothing ever runs the encoder.
#
# The NEFF cache key bakes in (max_num_seqs, max_model_len), so changing BS or
# SEQ forces a fresh compile (~5-10 min). Keep them as env knobs rather than
# editing inline, and let bench.sh drive load against an already-warm server.
#
# MODEL takes a hub repo ID or a local path -- vllm resolves either. The default
# pulls the public checkpoint (~294 GiB) into HF_HOME on first start, which makes
# the first launch look like it is hanging before compilation even begins. To
# fetch only what a text-only server reads (skipping the vision/audio assets this
# port never builds) and serve from disk instead:
#
#   hf download XiaomiMiMo/MiMo-V2.5 --local-dir ./MiMo-V2.5 \
#       --include 'config.json' 'configuration_mimo_v2.py' 'modeling_mimo_v2.py' \
#                 'generation_config.json' 'tokenizer*' 'vocab.json' 'merges.txt' \
#                 'model*.safetensors' 'model.safetensors.index.json'
#   MODEL=./MiMo-V2.5 bash serve_mimo.sh
#
# --max-num-batched-tokens must equal --max-model-len: it does NOT follow
# --max-model-len on its own, validate_num_batched_tokens_buckets requires the
# last num_batched_tokens_bucket to equal it, and on 0.29 any value below
# max_model_len turns on segmented prefill by default, which this port does
# not implement (the factory rejects it at load).
#
# Defaults are sized for 8K-token inputs and 1K-token outputs at the highest
# concurrency the KV cache holds (contrib/benchmark's 8192 / 1024 cells):
#   SEQ=9472             an 8K input with +3% jitter (8435) + 1024 output tokens,
#                        rounded up to a multiple of 128
#   BS                   max_num_seqs: 12 for Pro, 20 otherwise (the KV limit)
#   PREFILL_BUCKETS      [8448, $SEQ]: the last bucket must equal SEQ (no
#                        segmented prefill); 8448 keeps an 8K prompt from
#                        padding to SEQ. Add buckets for shorter prompts, e.g.
#                        [1024, 2048, 4096, 8448, 9472]; each is a compile.
#   DECODE_BUCKETS       one decode graph per batch size, ending at BS
#                        (Pro [1, 4, 8, 12], otherwise [1, 4, 8, 16, 20])
# The server holds int(X) requests, X from its "Maximum concurrency for
# <SEQ> tokens per request" log line.
#
# Usage:
#   bash serve_mimo.sh
#   BS=1 SEQ=512 PREFILL_BUCKETS='[512]' DECODE_BUCKETS='[1]' bash serve_mimo.sh

set -x

MODEL="${MODEL:-XiaomiMiMo/MiMo-V2.5}"
SERVED_NAME="${SERVED_NAME:-MiMo-V2.5}"
# auto | bf16 | fp8 (routed experts); read by the workers.
export MIMO_V2_5_EXPERT_DTYPE="${EXPERT_DTYPE:-${MIMO_V2_5_EXPERT_DTYPE:-auto}}"
# BS defaults to what the KV cache holds at SEQ 9472 (the server's "Maximum
# concurrency" line): Pro 12, Flash / V2.5 20. A larger BS only adds decode
# graphs the cache cannot fill, and their device memory: Pro at BS 32 failed to
# load its graphs ("Could not load the model status=4 message=Allocation Failure").
case "$MODEL" in
  *Pro*) bs_default=12; buckets_default="[1, 4, 8, 12]" ;;
  *)     bs_default=20; buckets_default="[1, 4, 8, 16, 20]" ;;
esac
BS="${BS:-$bs_default}"
SEQ="${SEQ:-9472}"
PREFILL_BUCKETS="${PREFILL_BUCKETS:-[8448, $SEQ]}"
DECODE_BUCKETS="${DECODE_BUCKETS:-$buckets_default}"
PORT="${PORT:-8000}"

# Mandatory on this instance: the Neuron runtime otherwise probes for an EFA
# device per rank and every rank dies with "No EFA device found at
# /sys/bus/pci/devices/.../infiniband".
export NEURON_SKIP_EFA_AFFINITY=1

# The native compile route (the default); set explicitly so the Precompile Cache
# is on and a warm start skips the trace.
export VLLM_NEURON_BACKEND=${VLLM_NEURON_BACKEND:-neuron_native}

# The native route compiles every rank's graphs (64 here); each neuronx-cc job
# takes 40-50 GB of host memory, and a cold compile runs for hours with fewer
# jobs (Pro with 4: more than 4 h). 8 (vllm-neuron's default) fits a 2 TB host
# at SEQ 9472. Ranks that finish first wait in a barrier whose timeout
# (vllm-neuron's default 3600 s) a cold compile can outlast. Also give the first
# 64-rank executions more than the runtime's 30 s default, which otherwise looks
# like a collective hang.
export VLLM_NEURON_PARALLEL_COMPILE_WORKERS=${VLLM_NEURON_PARALLEL_COMPILE_WORKERS:-8}
export VLLM_NEURON_BARRIER_TIMEOUT=${VLLM_NEURON_BARRIER_TIMEOUT:-43200}
export NEURON_RT_EXEC_TIMEOUT=${NEURON_RT_EXEC_TIMEOUT:-1200}

echo "Starting $SERVED_NAME server: $MODEL (BS=$BS, SEQ=$SEQ, TP=64, EP=64, experts=$MIMO_V2_5_EXPERT_DTYPE)"

vllm serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --max-model-len "$SEQ" \
    --max-num-seqs "$BS" \
    --max-num-batched-tokens "$SEQ" \
    --tensor-parallel-size 64 \
    --enable-expert-parallel \
    --no-enable-prefix-caching \
    --disable-hybrid-kv-cache-manager \
    --trust-remote-code \
    --hf-overrides '{"quantization_config": null, "vision_config": null}' \
    --additional-config "{
        \"neuron_config\": {
            \"quantization\": \"bf16\",
            \"ep_degree\": 64,
            \"num_batched_tokens_buckets\": $PREFILL_BUCKETS,
            \"num_seqs_buckets\": $DECODE_BUCKETS,
            \"on_device_sampling_config\": {\"all_greedy\": true}
        },
        \"encoder_neuron_config\": {\"num_encoder_input_buckets\": [2048]}
    }" \
    --port "$PORT"
