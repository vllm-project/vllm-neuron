# SPDX-License-Identifier: Apache-2.0
"""
Tensor capture example for Eagle3 speculative decoding.

Captures tensors from both target and draft models to separate subdirectories.

See also: docs/design/accuracy/tensor_capture_design.md

Usage:
    python run_tensor_capture_eagle3.py --model-checkpoint meta-llama/Llama-3.1-8B-Instruct --draft-model-checkpoint yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
    python run_tensor_capture_eagle3.py --model-checkpoint meta-llama/Llama-3.1-8B-Instruct --draft-model-checkpoint yuhuili/EAGLE3-LLaMA3.1-Instruct-8B --modules "model.layers.0-31,lm_head"
"""

import argparse
import os

from vllm import LLM, SamplingParams

DEFAULT_CAPTURE_MODULES = ["model.layers.0", "model.norm", "lm_head"]
DEFAULT_CAPTURE_DIR = "/tmp/vllm_neuron_captures_eagle3"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--draft-model-checkpoint",
        type=str,
        default="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    )
    parser.add_argument("--max-tokens", type=int, default=10)
    parser.add_argument(
        "--modules", type=str, default=None, help="Comma-separated modules to capture"
    )
    parser.add_argument("--capture-dir", type=str, default=DEFAULT_CAPTURE_DIR)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    args = parser.parse_args()

    capture_modules = (
        args.modules.split(",") if args.modules else DEFAULT_CAPTURE_MODULES
    )

    llm = LLM(
        enable_prefix_caching=False,
        model=args.model_checkpoint,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        speculative_config={
            "method": "eagle3",
            "model": args.draft_model_checkpoint,
            "num_speculative_tokens": 3,
        },
        additional_config={
            "neuron_config": {
                "on_device_sampling_config": {"temperature": "0"},
                "num_batched_tokens_buckets": [args.max_model_len],
                "num_seqs_buckets": [1],
                "tensor_capture": {
                    "modules": capture_modules,
                    "capture_dir": args.capture_dir,
                },
            }
        },
    )

    try:
        prompts = ["The capital of France is"]
        sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
        outputs = llm.generate(prompts, sampling_params)

        for output in outputs:
            print(f"Prompt: {output.prompt}")
            print(f"Generated: {output.outputs[0].text}")

        # Print capture summary
        print(f"\n=== Captures saved to {args.capture_dir} ===")
        for subdir in ["target", "draft"]:
            path = os.path.join(args.capture_dir, subdir)
            if os.path.exists(path):
                steps = sorted(os.listdir(path))
                print(f"{subdir}/: {len(steps)} steps")
                for step in steps[:3]:
                    files = os.listdir(os.path.join(path, step))
                    print(f"  {step}: {len(files)} files")
    finally:
        del llm


if __name__ == "__main__":
    main()
