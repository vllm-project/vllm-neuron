# SPDX-License-Identifier: Apache-2.0
"""
Tensor Capture Example: Capture intermediate tensors from vLLM Neuron model.

See also: docs/design/accuracy/tensor_capture_design.md

Usage:
    python run_tensor_capture.py --model meta-llama/Llama-3.1-8B-Instruct
    python run_tensor_capture.py --model meta-llama/Llama-3.1-8B-Instruct --modules "model.layers.0-7"
"""

import argparse
import json
import os

from vllm import LLM, SamplingParams

DEFAULT_MODULES = ["model.layers.0", "model.layers.1", "lm_head"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--modules", type=str, default=None, help="Comma-separated modules to capture"
    )
    parser.add_argument("--capture-dir", type=str, default="/tmp/vllm_neuron_captures")
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=10)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--kv-segment-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--hf-overrides",
        type=str,
        default=None,
        help="JSON string for hf_overrides (e.g. '{\"quantization_config\": {}}')",
    )
    args = parser.parse_args()

    if args.kv_segment_size > args.max_model_len:
        parser.error(
            f"--kv-segment-size ({args.kv_segment_size}) must not exceed --max-model-len ({args.max_model_len})"
        )

    modules = args.modules.split(",") if args.modules else DEFAULT_MODULES
    hf_overrides = json.loads(args.hf_overrides) if args.hf_overrides else None

    print(f"Model: {args.model}")
    print(f"Capture modules: {modules}")
    print(f"Capture dir: {args.capture_dir}")

    kwargs = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.kv_segment_size,
        tensor_parallel_size=args.tp_size,
        max_num_seqs=args.batch_size,
        additional_config={
            "neuron_config": {
                "on_device_sampling_config": {},
                "kv_segment_size_buckets": [args.kv_segment_size],
                "num_batched_tokens_buckets": [args.kv_segment_size],
                "num_seqs_buckets": [args.batch_size],
                "tensor_capture": {
                    "modules": modules,
                    "capture_dir": args.capture_dir,
                },
            }
        },
    )
    if hf_overrides:
        kwargs["hf_overrides"] = hf_overrides

    llm = LLM(**kwargs)

    try:
        outputs = llm.generate(
            [args.prompt], SamplingParams(temperature=0, max_tokens=args.max_tokens)
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"Output: {outputs[0].outputs[0].text}")

        # Show captured files
        print(f"\nCaptures saved to: {args.capture_dir}")
        for root, dirs, files in os.walk(args.capture_dir):
            level = root.replace(args.capture_dir, "").count(os.sep)
            indent = "  " * level
            print(f"{indent}{os.path.basename(root)}/")
            for f in sorted(files)[:5]:
                print(f"{indent}  {f}")
            if len(files) > 5:
                print(f"{indent}  ... and {len(files) - 5} more")
    finally:
        del llm


if __name__ == "__main__":
    main()
