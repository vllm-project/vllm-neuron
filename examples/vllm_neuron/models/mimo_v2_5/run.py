# SPDX-License-Identifier: Apache-2.0
"""Offline inference example for MiMo-V2.5 (mimo_v2_5).

48 layers / hidden 4096 / 256 experts top-8, hybrid attention: 9 full-attention
layers (0, 5, 11, 17, 23, 29, 35, 41, 47) and 39 sliding-window layers
(window 128). Asymmetric head dims (Q/K 192, V 128) and asymmetric KV heads
(full 4, SWA 8).

Runtime is BF16. The released checkpoint stores its matmul weights as 128x128
blockwise FP8-e4m3 with ``*.weight_scale_inv`` companions; the weight loaders
dequantize host-side at load time, so NO separate BF16 conversion pass is
needed -- point ``--model-checkpoint`` straight at the HF checkpoint.

EP config: ``tensor_parallel_size=64`` (the full trn2.48xlarge: 16 devices x 4
NeuronCores) with ``enable_expert_parallel`` and ``ep_degree=64``, giving
tp_sub = 64/64 = 1 (pure EP) and 4 experts per rank. 64 Q heads divide evenly
by 64; the 4-and-8 KV head counts are replicated across ranks.

      python examples/vllm_neuron/models/mimo_v2_5/run.py

Defaults to the public ``XiaomiMiMo/MiMo-V2.5`` repo (~294 GiB, downloaded to
HF_HOME on first use). Pass ``--model-checkpoint`` a local path to reuse an
existing copy. The vision/audio towers are not built by this port, so
``assets/``, ``audio_tokenizer/`` and ``preprocessor_config.json`` are dead
weight in a text-only download -- see serve_mimo.sh for an allow-listed
``hf download`` that skips them.

Attention runs eager (fp32 scores), not through a fused kernel: MiMo's 192-wide
Q/K heads exceed the 128 head_dim cap in flash_attention / attention_decode /
segmented_attention. The MoE does use the NKI kernels (moe_cte for prefill,
moe_tkg for decode).
"""

import argparse

from vllm import LLM, SamplingParams


def _text_only_overrides(config):
    """Drop the config keys that would route this text port elsewhere.

    Both keys must be DELETED, not set to None:

    * ``quantization_config`` -- vLLM derives ``ModelConfig.quantization`` from
      its ``quant_method`` and then checks it against
      ``NeuronPlatform.supported_quantization``, which does not list "fp8", so
      startup aborts before the model is built. The FP8 bytes never reach the
      framework's quant machinery: this port's weight loaders dequantize to bf16
      on the host, and they pick their branch by probing the checkpoint for
      ``*_scale_inv`` keys (``MiMoV2ForCausalLM._detect_quantized``), not by
      reading this config.
    * ``vision_config`` -- vLLM's ``MimoV2ModelArchConfigConvertor`` rewrites
      ``architectures`` to ``["MiMoV2OmniForCausalLM"]`` whenever a vision config
      is present, routing past this port's registry entry. And
      ``NeuronPlatform.check_and_update_config`` gates the vision path on
      ``hasattr(hf_config, "vision_config")``, so a None-valued attribute still
      builds a ``vision_neuron_config`` and sends ``load_model`` down the
      multimodal ``from_configs(text_neuron_config=..., vision_neuron_config=...)``
      branch. Only ``delattr`` clears both checks.
    """
    for key in ("quantization_config", "vision_config"):
        if hasattr(config, key):
            delattr(config, key)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="XiaomiMiMo/MiMo-V2.5",
        help="Hub repo ID or local path to the HF checkpoint (FP8 on disk; "
        "dequantized at load time). Passed through to vLLM unchanged, so a "
        "hub ID downloads to HF_HOME on first use.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=64)
    parser.add_argument("--ep-degree", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=1,
        help="Max concurrent sequences. bs=1 is the bringup path.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="'@@'-separated prompt list overriding the defaults.",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--prefill-bucket",
        type=int,
        default=None,
        help=(
            "Prefill token bucket, i.e. num_batched_tokens_buckets. Defaults to "
            "--max-model-len (single-shot prefill of the whole window). Setting "
            "it smaller decouples the prefill graph's token count from the "
            "KV-cache/block-table sizing, which is how the SEQ=1024 warmup "
            "fault was localized."
        ),
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help=(
            "Wrap each prompt in the checkpoint's chat template instead of "
            "feeding it as a raw completion. MiMo-V2.5 is instruction/thinking "
            "tuned, so raw continuations drift into reasoning-mode text and are "
            "not a fair coherence check; use this to exercise it as intended."
        ),
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help=(
            "With --chat, leave the template's reasoning mode on. Off by "
            "default because the template defaults enable_thinking=True, which "
            "spends the whole token budget inside the reasoning block and "
            "leaves the visible answer empty at small --max-tokens."
        ),
    )
    args = parser.parse_args()

    # The runner requires the last prefill bucket to equal max_num_batched_tokens,
    # so shrinking the prefill graph means lowering that cap too -- which is
    # exactly the point: KV cache and block tables stay sized by max_model_len
    # while the prefill NEFF traces fewer tokens.
    prefill_bucket = args.prefill_bucket or args.max_model_len

    llm = LLM(
        model=args.model_checkpoint,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=prefill_bucket,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_expert_parallel=True,
        # APC is on by default in vLLM V1, and the Neuron runner only supports it
        # alongside segmented prefill (max_num_batched_tokens in
        # {512, 1024, ...}). This port prefills single-shot, so turn APC off.
        enable_prefix_caching=False,
        # MiMo ships its config/modeling as custom repo code (model_type
        # "mimo_v2" is not in upstream transformers), so HF needs this to build
        # the PretrainedConfig that MiMoV2Config.from_configs consumes.
        trust_remote_code=True,
        # Strip the quant/vision config keys; see _text_only_overrides.
        hf_overrides=_text_only_overrides,
        # The released checkpoint is multimodal (vision/audio sub-configs); this
        # port covers the TEXT decoder, so zero out the per-prompt mm limits so
        # the runner does not demand a vision_neuron_config.
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
        additional_config={
            "neuron_config": {
                # BF16 runtime; the FP8 disk format is handled by the loaders.
                "quantization": "bf16",
                "ep_degree": args.ep_degree,
                "num_batched_tokens_buckets": [prefill_bucket],
                "num_seqs_buckets": [args.max_num_seqs],
                "on_device_sampling_config": {"all_greedy": "true"},
            }
        },
    )

    if args.prompts:
        prompts = args.prompts.split("@@")
    else:
        prompts = [
            "The capital of France is",
            "1 2 3 4 5 6 7 8 9",
            "Once upon a time, there was a",
            "def fibonacci(n):",
        ]
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens, temperature=0.0, top_p=1.0
    )

    if args.chat:
        tok = llm.get_tokenizer()
        rendered = [
            tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.thinking,
            )
            for p in prompts
        ]
        outputs = llm.generate(rendered, sampling_params)
    else:
        outputs = llm.generate(prompts, sampling_params)

    for src, o in zip(prompts, outputs):
        print(repr(src), "->", repr(o.outputs[0].text))


if __name__ == "__main__":
    main()
