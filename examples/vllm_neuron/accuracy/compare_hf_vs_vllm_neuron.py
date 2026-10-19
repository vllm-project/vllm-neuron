# SPDX-License-Identifier: Apache-2.0
"""
Compare HuggingFace transformers vs vLLM Neuron captures.

Two comparison modes:

1. Two-way: Direct comparison of HF BF16 vs vLLM Neuron BF16 with static threshold.
   Requires --threshold to determine pass/fail.

2. Three-way (--three-way): Compares both HF BF16 and vLLM Neuron BF16 against HF FP32.
   No threshold needed - uses input-adaptive thresholds based on expected
   precision loss. Passes if vLLM Neuron error is within 1.5x of HF BF16 error.
   This isolates implementation bugs from expected quantization behavior.

See also: docs/design/accuracy/tensor_compare_design.md

Usage:
    # Two-way: requires threshold
    python compare_hf_vs_vllm_neuron.py --model meta-llama/Llama-3.1-8B-Instruct --threshold 0.01

    # Three-way: no threshold needed
    python compare_hf_vs_vllm_neuron.py --model meta-llama/Llama-3.1-8B-Instruct --three-way
"""

import argparse
import os
import shutil

import torch

# NOTE: import vllm BEFORE transformers to avoid Neuron class double-registration
from vllm import LLM, SamplingParams  # noqa: E402

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from vllm_neuron.accuracy import (
    ModelCapture,
    TensorRegistry,
    compare_tensors,
    compare_tensors_three_way,
    print_comparison_report,
    print_three_way_report,
)
from vllm_neuron.accuracy.tensor_io import (
    CapturedForwardPass,
    ForwardPassMetadata,
    read as tensor_io_read,
    write as tensor_io_write,
)

DEFAULT_MODULES = ["model.layers.0", "model.layers.1"]
DEFAULT_PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "Water boils at",
    "The speed of light is approximately",
]


def run_hf_capture(
    model_path: str,
    capture_dir: str,
    modules: list,
    prompts: list,
    max_tokens: int,
    dtype: torch.dtype,
):
    """Capture intermediate tensors from HuggingFace model.

    Writes captures in the same format as CaptureWriter (dp0/ subdirectory
    with decode_b1_N naming) so tensor_io.read() can load them.
    """
    if os.path.exists(capture_dir):
        shutil.rmtree(capture_dir)

    dtype_str = "FP32" if dtype == torch.float32 else "BF16"
    print(f"\nCapturing HF {dtype_str}...")

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    TensorRegistry.reset_instance()
    capture = ModelCapture(model=model, modules=modules, capture_dir=capture_dir)
    capture._registry.configure(enabled=True)
    TensorRegistry._instance = capture._registry

    dp0_dir = os.path.join(capture_dir, "dp0")
    os.makedirs(dp0_dir, exist_ok=True)
    step_counter = 0

    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt")
        generated_ids = inputs["input_ids"].clone()

        with torch.inference_mode():
            for step in range(max_tokens):
                capture._registry.clear()
                model_out = model(input_ids=generated_ids)
                captured = capture._registry.get_all()

                if captured:
                    tensors = {}
                    for name, tensor in captured:
                        if isinstance(tensor, torch.Tensor):
                            tensors[name] = {0: tensor}
                    # Use decode_b1_N naming to match vLLM CaptureWriter format
                    fwd = CapturedForwardPass(
                        dir_name=f"decode_b1_{step_counter}",
                        metadata=ForwardPassMetadata(
                            req_ids=[f"prompt_{i}"],
                            positions=[generated_ids.shape[1] - 1],
                            dp_rank=0,
                        ),
                        tensors=tensors,
                    )
                    tensor_io_write(dp0_dir, [fwd])
                    step_counter += 1

                logits = (
                    model_out.logits if hasattr(model_out, "logits") else model_out[0]
                )
                next_token = torch.argmax(logits[0, -1, :]).item()
                generated_ids = torch.cat(
                    [generated_ids, torch.tensor([[next_token]])], dim=1
                )

        text = tokenizer.decode(generated_ids[0, inputs["input_ids"].shape[1] :])
        print(f"  Prompt {i}: {text[:40]}...")

    del model


def run_vllm_neuron_capture(
    model_path: str,
    capture_dir: str,
    modules: list,
    prompts: list,
    max_tokens: int,
    tp_size: int,
    max_model_len: int,
    kv_segment_size: int,
):
    """Capture intermediate tensors from vLLM Neuron model."""
    if os.path.exists(capture_dir):
        shutil.rmtree(capture_dir)

    print("\nCapturing vLLM Neuron BF16...")

    llm = LLM(
        model=model_path,
        max_model_len=max_model_len,
        max_num_batched_tokens=kv_segment_size,
        tensor_parallel_size=tp_size,
        max_num_seqs=1,
        additional_config={
            "neuron_config": {
                "on_device_sampling_config": {},
                "kv_segment_size_buckets": [kv_segment_size],
                "num_batched_tokens_buckets": [kv_segment_size],
                "num_seqs_buckets": [1],
                "tensor_capture": {"modules": modules, "capture_dir": capture_dir},
            }
        },
    )
    params = SamplingParams(temperature=0, max_tokens=max_tokens)
    for i, prompt in enumerate(prompts):
        outputs = llm.generate([prompt], params)
        print(f"  Prompt {i}: {outputs[0].outputs[0].text[:40]}...")
    del llm


def compare_captures_rank0(dir1, dir2):
    """Compare rank-0 tensors from two capture directories.

    Matches forward passes by dir_name (decode_b1_N) and compares
    rank-0 tensors for each module.
    """
    passes1 = {fwd.dir_name: fwd for fwd in tensor_io_read(dir1)}
    passes2 = {fwd.dir_name: fwd for fwd in tensor_io_read(dir2)}

    # Match decode passes only (skip prefill since HF doesn't have it)
    decode1 = {k: v for k, v in passes1.items() if k.startswith("decode_")}
    decode2 = {k: v for k, v in passes2.items() if k.startswith("decode_")}
    common = sorted(set(decode1) & set(decode2))

    results = []
    for dir_name in common:
        fwd1, fwd2 = decode1[dir_name], decode2[dir_name]
        common_modules = sorted(set(fwd1.tensors) & set(fwd2.tensors))
        for mod in common_modules:
            t1 = fwd1.tensors[mod].get(0)
            t2 = fwd2.tensors[mod].get(0)
            if t1 is not None and t2 is not None:
                results.append(compare_tensors(t1, t2, name=f"{dir_name}/{mod}"))
    return results


def compare_captures_rank0_three_way(baseline_dir, expected_dir, actual_dir):
    """Three-way comparison of rank-0 tensors."""
    base = {f.dir_name: f for f in tensor_io_read(baseline_dir)}
    exp = {f.dir_name: f for f in tensor_io_read(expected_dir)}
    act = {f.dir_name: f for f in tensor_io_read(actual_dir)}

    decode_b = {k: v for k, v in base.items() if k.startswith("decode_")}
    decode_e = {k: v for k, v in exp.items() if k.startswith("decode_")}
    decode_a = {k: v for k, v in act.items() if k.startswith("decode_")}
    common = sorted(set(decode_b) & set(decode_e) & set(decode_a))

    results = []
    for dn in common:
        mods = sorted(
            set(decode_b[dn].tensors)
            & set(decode_e[dn].tensors)
            & set(decode_a[dn].tensors)
        )
        for mod in mods:
            tb = decode_b[dn].tensors[mod].get(0)
            te = decode_e[dn].tensors[mod].get(0)
            ta = decode_a[dn].tensors[mod].get(0)
            if tb is not None and te is not None and ta is not None:
                results.append(
                    compare_tensors_three_way(tb, te, ta, name=f"{dn}/{mod}")
                )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--modules", type=str, default=None)
    parser.add_argument(
        "--prompts", type=str, default=None, help="Pipe-separated prompts"
    )
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument(
        "--threshold", type=float, default=None, help="Required for two-way comparison"
    )
    parser.add_argument(
        "--three-way", action="store_true", help="Three-way: no threshold needed"
    )
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--kv-segment-size", type=int, default=2048)
    args = parser.parse_args()

    if args.kv_segment_size > args.max_model_len:
        parser.error(
            f"--kv-segment-size ({args.kv_segment_size}) must not exceed --max-model-len ({args.max_model_len})"
        )

    if not args.three_way and args.threshold is None:
        parser.error(
            "--threshold is required for two-way comparison (or use --three-way)"
        )

    modules = args.modules.split(",") if args.modules else DEFAULT_MODULES
    if args.prompts:
        prompts = [p.strip() for p in args.prompts.split("|")]
    else:
        prompts = DEFAULT_PROMPTS[: args.num_prompts]

    base_dir = f"/tmp/vllm_neuron_hf_compare_{os.environ.get('USER', 'default')}"

    print("=" * 60)
    print(
        f"COMPARE: HF vs vLLM Neuron ({'Three-Way' if args.three_way else 'Two-Way'})"
    )
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Prompts: {len(prompts)}")

    if args.three_way:
        fp32_dir = f"{base_dir}/hf_fp32"
        bf16_dir = f"{base_dir}/hf_bf16"
        vllm_dir = f"{base_dir}/vllm_neuron"

        run_hf_capture(
            args.model, fp32_dir, modules, prompts, args.max_tokens, torch.float32
        )
        run_hf_capture(
            args.model, bf16_dir, modules, prompts, args.max_tokens, torch.bfloat16
        )
        run_vllm_neuron_capture(
            args.model,
            vllm_dir,
            modules,
            prompts,
            args.max_tokens,
            args.tp_size,
            args.max_model_len,
            args.kv_segment_size,
        )

        results = compare_captures_rank0_three_way(fp32_dir, bf16_dir, vllm_dir)
        print_three_way_report(
            {"all": {0: results}},
            label_baseline="HF FP32",
            label_expected="HF BF16",
            label_actual="vLLM Neuron BF16",
        )
    else:
        hf_dir = f"{base_dir}/hf_bf16"
        vllm_dir = f"{base_dir}/vllm_neuron"

        run_hf_capture(
            args.model, hf_dir, modules, prompts, args.max_tokens, torch.bfloat16
        )
        run_vllm_neuron_capture(
            args.model,
            vllm_dir,
            modules,
            prompts,
            args.max_tokens,
            args.tp_size,
            args.max_model_len,
            args.kv_segment_size,
        )

        results = compare_captures_rank0(hf_dir, vllm_dir)
        print_comparison_report(
            {"all": {0: results}},
            threshold=args.threshold,
            label1="HF BF16",
            label2="vLLM Neuron BF16",
        )


if __name__ == "__main__":
    main()
