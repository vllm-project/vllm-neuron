#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline inference example for dense Qwen3 MXFP8 checkpoints on Neuron.

Use this with a checkpoint produced by ``build_mxfp8_checkpoint.py`` to do a
quick qualitative smoke run on the compressed model.

Examples:
    python examples/vllm_neuron/models/qwen3/run_mxfp8.py
    python examples/vllm_neuron/models/qwen3/run_mxfp8.py \
        --model-checkpoint /models/Qwen3-32B-mxfp8 \
        --tp-size 8 \
        --prompt "What is the capital of France? Answer in one short sentence."
"""

from __future__ import annotations

import argparse
import os

from vllm import LLM, SamplingParams

os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "1200")
os.environ.setdefault("NEURON_LIBTORCH_COMPILATION_TIMEOUT", "1200")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="Qwen3-32B-mxfp8",
        help=(
            "Path to the compressed MXFP8 checkpoint directory. "
            "This should point at the output of build_mxfp8_checkpoint.py."
        ),
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=8,
        help="Tensor parallel size. Qwen3-32B dense requires tp>=8.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="Maximum model length for the example engine.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=1,
        help="Maximum concurrent sequences for the example engine.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=8192,
        help="Maximum batched tokens and default bucket size for the example engine.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Maximum generated tokens per prompt.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        nargs="+",
        default=None,
        help=(
            "Prompt to run. Repeat this flag to provide multiple prompts. "
            "If omitted, the script uses a small built-in smoke set."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = (
        [" ".join(prompt_tokens) for prompt_tokens in args.prompt]
        if args.prompt
        else [
            "What is the capital of France? Answer in one short sentence.",
            "Finish this sequence: 1, 1, 2, 3, 5, 8,",
        ]
    )

    llm = LLM(
        enable_prefix_caching=False,
        model=args.model_checkpoint,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tp_size,
        additional_config={
            "neuron_config": {
                "quantization": "mxfp8",
                "on_device_sampling_config": {
                    "all_greedy": "true",
                },
                "num_batched_tokens_buckets": [args.max_num_batched_tokens],
                "num_seqs_buckets": [args.max_num_seqs],
            }
        },
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        generated_text = output.outputs[0].text if output.outputs else ""
        print(f"PROMPT: {output.prompt}")
        print(f"OUTPUT: {generated_text}")
        print()


if __name__ == "__main__":
    main()
