#!/bin/bash
# MiMo-V2.5 single-instance server — TP64 with expert parallelism, BF16.
#
# 48 layers / 256 routed experts / top-8 / hidden 4096. At TP=64 with EP=64 each
# rank owns 4 experts, which is what the MoE kernels are sized against below.
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
# Usage:
#   bash serve_mimo.sh                 # BS=32, SEQ=1024
#   BS=1 SEQ=512 bash serve_mimo.sh

set -x

MODEL="${MODEL:-XiaomiMiMo/MiMo-V2.5}"
BS="${BS:-32}"
SEQ="${SEQ:-1024}"
PORT="${PORT:-8000}"

# Mandatory on this instance: the Neuron runtime otherwise probes for an EFA
# device per rank and every rank dies with "No EFA device found at
# /sys/bus/pci/devices/.../infiniband".
export NEURON_SKIP_EFA_AFFINITY=1

echo "Starting MiMo-V2.5 server: $MODEL (BS=$BS, SEQ=$SEQ, TP=64, EP=64)"

vllm serve "$MODEL" \
    --served-model-name MiMo-V2.5 \
    --max-model-len "$SEQ" \
    --max-num-seqs "$BS" \
    --max-num-batched-tokens "$SEQ" \
    --tensor-parallel-size 64 \
    --enable-expert-parallel \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --hf-overrides '{"quantization_config": null, "vision_config": null}' \
    --additional-config "{
        \"neuron_config\": {
            \"quantization\": \"bf16\",
            \"ep_degree\": 64,
            \"num_batched_tokens_buckets\": [$SEQ],
            \"num_seqs_buckets\": [$BS],
            \"on_device_sampling_config\": {\"all_greedy\": true}
        },
        \"encoder_neuron_config\": {\"num_encoder_input_buckets\": [2048]}
    }" \
    --port "$PORT"
