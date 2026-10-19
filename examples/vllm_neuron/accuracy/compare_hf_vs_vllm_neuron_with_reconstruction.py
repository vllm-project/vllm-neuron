# SPDX-License-Identifier: Apache-2.0
"""
Compare HF vs vLLM Neuron with tensor reconstruction.

Captures intermediate tensors from both HF (CPU) and vLLM Neuron, then
compares them using two-way and three-way metrics. The reconstruction
function converts per-rank Neuron tensors back to a single comparable
tensor (handles TP all-reduce, bucket padding).

Currently supports dense TP-only models (Llama, etc.) with no sequence
parallelism or MoE. All ranks produce identical outputs after all-reduce,
so reconstruction simply uses rank 0.

TODO: Add GPT-OSS support. GPT-OSS uses sequence parallelism where each
rank holds a different seq chunk, requiring SP gather + hidden-dim
unshuffle for reconstruction.

Usage:
    # Full three-way comparison
    python compare_hf_vs_vllm_neuron_with_reconstruction.py \\
        --model meta-llama/Llama-3.2-1B-Instruct --three-way --tp-size 8

    # Step-by-step (useful when HF is slow on CPU):
    python compare_hf_vs_vllm_neuron_with_reconstruction.py \\
        --model meta-llama/Llama-3.2-1B-Instruct --hf-only
    python compare_hf_vs_vllm_neuron_with_reconstruction.py \\
        --model meta-llama/Llama-3.2-1B-Instruct --neuron-only
    python compare_hf_vs_vllm_neuron_with_reconstruction.py \\
        --model meta-llama/Llama-3.2-1B-Instruct --compare-only
"""

import argparse
import json
import os
import re
import shutil
from typing import List

import torch

from vllm_neuron.accuracy.tensor_capture import (
    ModelCapture,
    CaptureWriter,
    TensorRegistry,
)
from vllm_neuron.accuracy.tensor_alignment_utils import (
    align_and_truncate_hidden,
    align_decode_captures,
    hf_reference_reconstruction,
)
from vllm_neuron.accuracy.tensor_compare import (
    compare_captures_two_way,
    compare_captures_three_way,
    print_comparison_report,
    print_three_way_report,
)

LLAMA_MODULES = [
    "model.embed_tokens",
    "model.layers.0-15.input_layernorm",
    "model.layers.0-15.post_attention_layernorm",
    "model.layers.0-15.self_attn",
    "model.layers.0-15.mlp",
    "model.norm",
    "lm_head",
]
MAX_TOKENS = 3
KV_SEGMENT_SIZE = 2048
# First prompt is short; second is a multi-shot chain (implicit string concat).
DEFAULT_PROMPTS = [
    "What is 2+2?",
    "Q: 15 trees, workers plant more, now 21. How many planted? A: 6\n"
    "Q: 3 cars, 2 more arrive. How many total? A: 5\n"
    "Q: 9 computers, 5 more each day Mon-Thu. How many now? A: 29\n"
    "Q: A robe takes 2 bolts blue fiber and half that white. How many total?",
]


def _get_num_layers(model_path: str) -> int:
    """Read num_hidden_layers from the model's config.json."""
    config_path = os.path.join(model_path, "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            return json.load(f)["num_hidden_layers"]
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(
        model_path, trust_remote_code=True
    ).num_hidden_layers


# ---------------------------------------------------------------------------
# Reconstruction functions
# ---------------------------------------------------------------------------


def llama_reconstruct(
    rank_tensors: List[torch.Tensor],
    module_name: str,
    phase: str = "prefill",
    positions: List[int] = None,
) -> torch.Tensor:
    """Reconstruction for dense TP-only models (Llama, etc.) with no SP.

    No sequence parallelism: all ranks are identical after all-reduce.
    Use rank 0 only. Strip bucket padding via positions for both phases.
    For prefill, Neuron pads to bucket size (e.g., 2048) but only the first
    N tokens are real. For decode, batch dim is padded to max_num_seqs.

    Note: The function signature (rank_tensors, module_name, phase, positions)
    is the ReconstructionFn protocol. All custom reconstruction functions must
    accept these same parameters, even if some are unused for a given model.
    """
    from vllm_neuron.accuracy.tensor_alignment_utils import count_real_tokens

    tensor = rank_tensors[0]
    if positions:
        real = count_real_tokens(positions)
        if tensor.shape[0] > real:
            return tensor[:real]
    return tensor


def llama_execution_order(num_layers: int) -> List[str]:
    """Module names in execution order for Llama (dense, no MoE)."""
    order = ["model_embed_tokens"]
    for i in range(num_layers):
        layer = f"model_layers_{i}"
        order.extend(
            [
                f"{layer}_input_layernorm",
                f"{layer}_self_attn",
                f"{layer}_post_attention_layernorm",
                f"{layer}_mlp",
            ]
        )
    order.extend(["model_norm", "lm_head"])
    return order


# ---------------------------------------------------------------------------
# Capture logic
# ---------------------------------------------------------------------------


def get_base_dir(model_id: str, output_dir: str = None) -> str:
    if output_dir:
        return output_dir
    model_short = model_id.split("/")[-1]
    return f"/tmp/tensor_compare_{model_short}_{os.environ.get('USER', 'default')}"


def run_hf_capture(
    model_id: str,
    capture_dir: str,
    dtype: torch.dtype,
    label: str,
    prompts: list,
    modules: list,
):
    if os.path.exists(capture_dir):
        shutil.rmtree(capture_dir)
    TensorRegistry.reset_instance()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n[HF {label}] Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="cpu", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    capture = ModelCapture(model=model, modules=modules, capture_dir=capture_dir)
    writer = CaptureWriter(capture_dir, dp_rank=0)
    writer.enable()

    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt")
        print(f"  Prompt {i}: {inputs['input_ids'].shape[1]} tokens")
        generated_ids = inputs["input_ids"].clone()
        with torch.inference_mode():
            for step in range(MAX_TOKENS):
                capture._registry.clear()
                model_output = model(input_ids=generated_ids)
                captured = capture._registry.get_all()
                positions = torch.arange(generated_ids.shape[1])
                if captured:
                    capture_names = [n for n, _ in captured]
                    captures = [t for _, t in captured]
                    writer.write(
                        tuple(captures),
                        capture_names,
                        req_ids=[str(i)],
                        positions=positions,
                        is_prefill=(step == 0),
                    )
                next_token = torch.argmax(model_output.logits[0, -1, :]).item()
                generated_ids = torch.cat(
                    [generated_ids, torch.tensor([[next_token]])], dim=1
                )
                print(f"    Step {step}: {tokenizer.decode([next_token])}")
    del model, capture
    print(f"  Saved to {capture_dir}")


def run_neuron_capture(
    model_id: str,
    capture_dir: str,
    prompts: list,
    modules: list,
    max_model_len: int = 8192,
    hf_overrides: dict = None,
    kv_segment_size: int = KV_SEGMENT_SIZE,
    tp_size: int = 16,
    max_num_seqs: int = 2,
):
    if os.path.exists(capture_dir):
        shutil.rmtree(capture_dir)

    from vllm import LLM, SamplingParams

    print(f"\n[Neuron] Loading {model_id} (max_model_len={max_model_len})...")
    llm = LLM(
        model=model_id,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=kv_segment_size,
        tensor_parallel_size=tp_size,
        enable_prefix_caching=False,
        hf_overrides=hf_overrides or {},
        additional_config={
            "neuron_config": {
                "kv_segment_size_buckets": [kv_segment_size],
                "num_batched_tokens_buckets": [kv_segment_size],
                "num_seqs_buckets": [max_num_seqs],
                "on_device_sampling_config": {
                    "all_greedy": True,
                },
                "tensor_capture": {
                    "modules": modules,
                    "capture_dir": capture_dir,
                },
            }
        },
    )

    params = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    tokenizer = llm.get_tokenizer()
    for i, prompt in enumerate(prompts):
        print(f"  Prompt {i}: {len(tokenizer(prompt)['input_ids'])} tokens")
        outputs = llm.generate([prompt], params)
        print(f"    Output: {outputs[0].outputs[0].text[:80]}")

    del llm
    print(f"  Saved to {capture_dir}")


def run_compare(
    model_id: str,
    num_layers: int,
    output_dir: str = None,
):
    base_dir = get_base_dir(model_id, output_dir)
    fp32_dir = f"{base_dir}/hf_fp32"
    bf16_dir = f"{base_dir}/hf_bf16"
    neuron_dir = f"{base_dir}/neuron"

    for d, label in [(fp32_dir, "FP32"), (bf16_dir, "BF16"), (neuron_dir, "Neuron")]:
        if not os.path.exists(d):
            print(f"Missing {label} captures at {d}")
            return

    from vllm_neuron.accuracy.tensor_io import read as tensor_io_read

    fp32 = tensor_io_read(fp32_dir)
    bf16 = tensor_io_read(bf16_dir)
    neuron = tensor_io_read(neuron_dir)

    fp32 = align_decode_captures(fp32, neuron)
    bf16 = align_decode_captures(bf16, neuron)

    reconstruct_fn = llama_reconstruct
    module_order = llama_execution_order(num_layers)

    # Two-way: direct FP32 CPU vs BF16 Neuron comparison (static threshold)
    two_way_kwargs = dict(
        ref_reconstruction_fn=hf_reference_reconstruction,
        test_reconstruction_fn=reconstruct_fn,
        module_order=module_order,
    )
    two_way_prefill = compare_captures_two_way(
        fp32,
        neuron,
        phase="prefill",
        alignment_fn=align_and_truncate_hidden,
        **two_way_kwargs,
    )
    print_comparison_report(two_way_prefill, label1="HF FP32", label2="Neuron")

    two_way_decode = compare_captures_two_way(
        fp32,
        neuron,
        phase="decode",
        alignment_fn=align_and_truncate_hidden,
        **two_way_kwargs,
    )
    print_comparison_report(two_way_decode, label1="HF FP32", label2="Neuron")

    # Three-way: FP32 baseline vs BF16 CPU vs BF16 Neuron (dynamic threshold)
    three_way_kwargs = dict(
        reference_reconstruction_fn=hf_reference_reconstruction,
        target_reconstruction_fn=reconstruct_fn,
        module_order=module_order,
    )
    prefill_results = compare_captures_three_way(
        fp32,
        bf16,
        neuron,
        phase="prefill",
        alignment_fn=align_and_truncate_hidden,
        **three_way_kwargs,
    )
    print_three_way_report(
        prefill_results, label_expected="HF BF16", label_actual="Neuron"
    )

    decode_results = compare_captures_three_way(
        fp32,
        bf16,
        neuron,
        phase="decode",
        alignment_fn=align_and_truncate_hidden,
        **three_way_kwargs,
    )
    print_three_way_report(
        decode_results, label_expected="HF BF16", label_actual="Neuron"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare HF vs vLLM Neuron with tensor reconstruction"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Pipe-separated prompts (default: built-in short + long)",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--three-way", action="store_true", help="Full run (default)")
    mode.add_argument("--hf-only", action="store_true", help="Capture HF only")
    mode.add_argument("--neuron-only", action="store_true", help="Capture Neuron only")
    mode.add_argument(
        "--compare-only", action="store_true", help="Compare existing captures"
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Max model length for Neuron (default: 8192)",
    )
    parser.add_argument(
        "--kv-segment-size",
        type=int,
        default=KV_SEGMENT_SIZE,
        help=f"KV segment size for Neuron (default: {KV_SEGMENT_SIZE})",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=8,
        help="Tensor parallel size (default: 8)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=2,
        help="Max number of sequences / decode batch size (default: 2)",
    )
    parser.add_argument(
        "--hf-overrides",
        type=str,
        default="{}",
        help="JSON string of hf_overrides for vLLM (default: '{}')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for captures (default: /tmp/tensor_compare_<model>_<user>)",
    )
    args = parser.parse_args()

    num_layers = _get_num_layers(args.model)
    hf_overrides = json.loads(args.hf_overrides)
    prompts = (
        [p.strip() for p in args.prompts.split("|")]
        if args.prompts
        else DEFAULT_PROMPTS
    )
    base_dir = get_base_dir(args.model, args.output_dir)

    modules = [re.sub(r"0-\d+", f"0-{num_layers - 1}", m) for m in LLAMA_MODULES]

    # Default to three-way if no mode specified
    run_all = not (args.hf_only or args.neuron_only or args.compare_only)

    print("=" * 60)
    print("Tensor Compare with Reconstruction")
    print(f"Model: {args.model} ({num_layers} layers)")
    print(f"Output: {base_dir}")
    print("=" * 60)

    if args.hf_only or run_all:
        run_hf_capture(
            args.model,
            f"{base_dir}/hf_fp32",
            torch.float32,
            "FP32",
            prompts,
            modules=modules,
        )
        run_hf_capture(
            args.model,
            f"{base_dir}/hf_bf16",
            torch.bfloat16,
            "BF16",
            prompts,
            modules=modules,
        )

    if args.neuron_only or run_all:
        run_neuron_capture(
            args.model,
            f"{base_dir}/neuron",
            prompts,
            modules=modules,
            max_model_len=args.max_model_len,
            hf_overrides=hf_overrides,
            kv_segment_size=args.kv_segment_size,
            tp_size=args.tp_size,
            max_num_seqs=args.max_num_seqs,
        )

    if args.compare_only or run_all:
        run_compare(
            args.model,
            num_layers,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
