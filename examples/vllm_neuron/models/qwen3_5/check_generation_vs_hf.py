#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Compare greedy generation on the Neuron device against HuggingFace on CPU.

A layer-by-layer float32 comparison on CPU cannot show what bf16 on device plus
many decode steps does to the output, because it only covers one prefill. So
this generates the same tokens both ways and reports where they diverge. Two
numbers matter:

* **prefix length** — how many leading tokens match exactly. Divergence in the
  first few tokens is a bug; divergence later is expected, because bf16 and an
  independent implementation will eventually disagree on a near-tie and greedy
  decoding then amplifies it.
* **token match rate** — the upstream port this work derives from published
  53/80 tokens (66%) with 3/5 prompts exact, so that is the level to compare
  against rather than 100%.

HF runs in float32 on CPU, which is slow: budget a couple of minutes per prompt
at 32 tokens. Run the device side first (it writes a JSON), then the HF side, so
the device is free while HF grinds.

Usage:

    # 1. on the device
    python examples/vllm_neuron/models/qwen3_5/check_generation_vs_hf.py \
        --model <path-to-checkpoint>/Qwen3.5-2B --side neuron
    # 2. on CPU, then compare
    python examples/vllm_neuron/models/qwen3_5/check_generation_vs_hf.py \
        --model <path-to-checkpoint>/Qwen3.5-2B --side hf
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "1200")
os.environ.setdefault("VLLM_NEURON_COMPILATION_TIMEOUT", "1800")

PROMPTS = [
    "The capital of France is",
    "I am gonna keep counting forever, 1 2 3 4 5",
    "def fibonacci(n):",
    "Once upon a time, there was a",
    "The three primary colours are",
]

DEFAULT_JSON = "/tmp/qwen35_generation.json"


# One image prompt for --vl. Kept to a single item so the vision bucket stays
# small: at 224x224 with patch 16 this is a 14x14 grid, 196 raw tokens.
VL_QUESTION = "Describe this image in detail."
VL_IMAGE_ASSET = "cherry_blossom"
VL_IMAGE_SIZE = 224


def mac_threshold_override(model: str) -> dict:
    """``{"hlo2tensorizer_options": ""}`` for a dense checkpoint, ``{}`` for MoE.

    That override suppresses the runner's ``--modular-flow-mac-threshold=10``.
    The dense graphs need it suppressed -- with the flag, neuronx-cc fails
    codegen on their decode graph (NCC_IBTN006 on a pftranspose copy) -- and the
    sparse ones need it kept, because the flag exists for exactly the NKI kernels
    the MoE block calls. Keyed off the checkpoint rather than a command-line flag
    because the reason is structural and a flag would be forgotten.
    """
    from transformers import AutoConfig

    text_config = AutoConfig.from_pretrained(model).text_config
    if getattr(text_config, "num_experts", None):
        return {}
    return {"hlo2tensorizer_options": ""}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--side", choices=("neuron", "hf"), required=True)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--json", default=DEFAULT_JSON)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument(
        "--vl",
        action="store_true",
        help="compare a single-image prompt instead of the text-only set",
    )
    parser.add_argument("--vision-bucket", type=int, default=256)
    # None keeps vLLM's default. 27B needs it held down to ~0.65 on a 4-core
    # instance: HBM is 24 GB per logical core, not pooled, and the block planner
    # fills whatever budget is left over until neuronx-cc rejects the graph.
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    # float32 is the honest reference. Fall back to bfloat16 only when the
    # checkpoint does not fit in host RAM in float32; that weakens the comparison
    # symmetrically -- both sides then round the same way -- and the check this
    # script exists for, "is the first token right", survives it.
    parser.add_argument(
        "--expert-parallel",
        action="store_true",
        help="shard the experts across ranks instead of each expert's "
        "intermediate dimension. Optional for 35B-A3B and required for "
        "397B-A17B; see run.py for why the degree matters.",
    )
    parser.add_argument(
        "--ep-degree",
        type=int,
        default=None,
        help="expert-parallel degree; requires --expert-parallel. Defaults to "
        "the world size.",
    )
    parser.add_argument("--hf-dtype", default="fp32", choices=("fp32", "bf16"))
    return parser.parse_args()


def engine_extras(args) -> dict:
    extras = {}
    if args.gpu_memory_utilization is not None:
        extras["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.expert_parallel:
        extras["enable_expert_parallel"] = True
    return extras


def ep_neuron_config(args) -> dict:
    """``ep_degree`` for the neuron_config, validated.

    vLLM's ``enable_expert_parallel`` is a bool and its degree is the world size,
    so the plugin carries an explicit degree on NeuronConfig instead. Unset, it
    resolves to the world size -- pure EP, ``tp_degree = 1``.
    """
    if args.ep_degree is None:
        return {}
    if not args.expert_parallel:
        raise SystemExit("--ep-degree requires --expert-parallel")
    return {"ep_degree": args.ep_degree}


def build_vl_prompt(model_path: str):
    """The chat prompt and the PIL image, from the model's own processor.

    Both sides go through the same ``AutoProcessor``, which is also what vLLM's
    frontend uses for this architecture — so the pixel values and the placeholder
    expansion match, and a divergence means the model, not preprocessing.
    """
    from transformers import AutoProcessor
    from vllm.assets.image import ImageAsset

    processor = AutoProcessor.from_pretrained(model_path)
    image = ImageAsset(VL_IMAGE_ASSET).pil_image.resize(
        (VL_IMAGE_SIZE, VL_IMAGE_SIZE)
    )
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": VL_QUESTION}],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return processor, prompt, image


def run_neuron_vl(args) -> None:
    from vllm import LLM, SamplingParams

    _processor, prompt, image = build_vl_prompt(args.model)
    llm = LLM(
        model=args.model,
        **engine_extras(args),
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=1,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 1, "video": 0},
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                "num_batched_tokens_buckets": [args.max_model_len],
                "num_seqs_buckets": [1],
                "on_device_sampling_config": {"all_greedy": True},
                **mac_threshold_override(args.model),
            },
            "vision_neuron_config": {
                "num_vision_tokens_buckets": [args.vision_bucket],
                "vision_attention_block_size": args.vision_bucket,
            },
        },
    )
    outputs = llm.generate(
        [{"prompt": prompt, "multi_modal_data": {"image": [image]}}],
        SamplingParams(max_tokens=args.tokens, temperature=0.0),
    )
    payload = {
        "tokens": args.tokens,
        "vl": True,
        "results": [
            {
                "prompt": VL_QUESTION,
                "token_ids": list(outputs[0].outputs[0].token_ids),
                "text": outputs[0].outputs[0].text,
            }
        ],
    }
    with open(args.json, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {args.json}")
    print(f"  {VL_QUESTION!r} -> {payload['results'][0]['text']!r}")


def run_hf_vl_and_compare(args) -> int:
    """HF's own VL generation on CPU, against the device's tokens."""
    import torch
    from transformers import AutoModelForImageTextToText

    with open(args.json) as handle:
        payload = json.load(handle)
    if not payload.get("vl"):
        raise SystemExit(f"{args.json} was written by the text-only side; rerun "
                         f"--side neuron --vl")

    processor, prompt, image = build_vl_prompt(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float32
    ).eval()

    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.tokens,
            min_new_tokens=args.tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    hf_ids = generated[0, inputs["input_ids"].shape[1] :].tolist()[: args.tokens]
    device_ids = payload["results"][0]["token_ids"][: args.tokens]

    tokenizer = processor.tokenizer
    matched = sum(a == b for a, b in zip(device_ids, hf_ids))
    prefix = 0
    for a, b in zip(device_ids, hf_ids):
        if a != b:
            break
        prefix += 1
    print(f"comparing {len(hf_ids)} greedy tokens for a single-image prompt\n")
    print(f"  prefix {prefix}, {matched}/{len(hf_ids)} tokens matched")
    print(f"  neuron: {tokenizer.decode(device_ids)!r}")
    print(f"  hf    : {tokenizer.decode(hf_ids)!r}")
    if prefix == 0:
        print("\nFIRST TOKEN WRONG — the vision merge or prefill is wrong, not rounding")
        return 1
    return 0


def run_neuron(args) -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        **engine_extras(args),
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=len(PROMPTS),
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                "num_batched_tokens_buckets": [args.max_model_len],
                "num_seqs_buckets": [len(PROMPTS)],
                "on_device_sampling_config": {"all_greedy": True},
                **mac_threshold_override(args.model),
                **ep_neuron_config(args),
            },
        },
    )
    outputs = llm.generate(
        PROMPTS, SamplingParams(max_tokens=args.tokens, temperature=0.0)
    )
    payload = {
        "tokens": args.tokens,
        "results": [
            {
                "prompt": prompt,
                "token_ids": list(out.outputs[0].token_ids),
                "text": out.outputs[0].text,
            }
            for prompt, out in zip(PROMPTS, outputs)
        ],
    }
    with open(args.json, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {args.json}")
    for entry in payload["results"]:
        print(f"  {entry['prompt']!r} -> {entry['text']!r}")


def run_hf_and_compare(args) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(args.json) as handle:
        payload = json.load(handle)
    if payload["tokens"] < args.tokens:
        raise SystemExit(
            f"{args.json} only has {payload['tokens']} tokens per prompt"
        )

    hf_dtype = torch.float32 if args.hf_dtype == "fp32" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=hf_dtype).eval()

    total_tokens = 0
    total_matched = 0
    exact_prompts = 0
    bad_first: list[str] = []
    print(f"comparing {args.tokens} greedy tokens per prompt, HF in {args.hf_dtype}\n")

    for entry in payload["results"]:
        prompt = entry["prompt"]
        device_ids = entry["token_ids"][: args.tokens]
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.tokens,
                min_new_tokens=args.tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        hf_ids = generated[0, inputs.input_ids.shape[1] :].tolist()[: args.tokens]

        matched = sum(a == b for a, b in zip(device_ids, hf_ids))
        prefix = 0
        for a, b in zip(device_ids, hf_ids):
            if a != b:
                break
            prefix += 1
        total_tokens += len(hf_ids)
        total_matched += matched
        exact = prefix == len(hf_ids)
        exact_prompts += int(exact)
        if prefix == 0:
            bad_first.append(prompt)

        flag = "EXACT" if exact else f"prefix {prefix}"
        print(f"  {flag:11s} {matched}/{len(hf_ids)} tokens   {prompt!r}")
        if not exact:
            print(f"      neuron: {tokenizer.decode(device_ids)!r}")
            print(f"      hf    : {tokenizer.decode(hf_ids)!r}")

    rate = 100.0 * total_matched / max(total_tokens, 1)
    print(
        f"\ntotal {total_matched}/{total_tokens} tokens ({rate:.1f}%), "
        f"{exact_prompts}/{len(payload['results'])} prompts exact"
    )
    print("upstream port's published bar: 53/80 tokens (66%), 3/5 prompts exact")

    # A first-token mismatch is the failure mode this check exists to catch: it
    # means the prefill is wrong, not that bf16 lost a near-tie deep into decode.
    if bad_first:
        print(f"\nFIRST TOKEN WRONG for {len(bad_first)} prompt(s): {bad_first}")
        print("that is a prefill bug, not rounding — investigate before trusting output")
        return 1
    return 0


def main() -> int:
    args = parse_args()
    if args.side == "neuron":
        (run_neuron_vl if args.vl else run_neuron)(args)
        return 0
    return (run_hf_vl_and_compare if args.vl else run_hf_and_compare)(args)


if __name__ == "__main__":
    sys.exit(main())
