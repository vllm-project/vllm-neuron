# SPDX-License-Identifier: Apache-2.0
"""Offline inference for Tencent Hy4-preview on one trn2.48xlarge.

Needs the Trn2 fp8 checkpoint, written once from the published MXFP8 one::

    python examples/vllm_neuron/models/hy_v4/convert_mxfp8_to_neuron_fp8.py \\
        --src <dir>/Hy4-preview-FP8 --dst <dir>/Hy4-preview-neuron-fp8

then::

    python examples/vllm_neuron/models/hy_v4/run.py --model <dir>/Hy4-preview-neuron-fp8

All 64 logical cores, pure TP: one attention head per rank, every expert
sharded 64 ways on its intermediate dim. See README.md for why, and for the
2048-token context limit.
"""

import argparse
import os

os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "3600")
os.environ.setdefault("VLLM_NEURON_COMPILATION_TIMEOUT", "14400")

from vllm import LLM, SamplingParams

PROMPTS = [
    "The capital of France is",
    "I am gonna keep counting forever, 1 2 3 4 5",
    "def fibonacci(n):",
    "Once upon a time, there was a",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=64)
    # Past index_topk = 2048 tokens Hy4's attention is sparse (DSA), which is
    # not implemented yet; up to it, dense attention is exact. The model
    # refuses a longer max_model_len.
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=32)
    # Weights are ~15 GiB of each logical core's 24; the KV planner spends
    # ``24 * gmu - weights`` and does not count activations.
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--optlevel", type=int, default=None, choices=[0, 1, 2, 3])
    parser.add_argument(
        "--chat",
        action="store_true",
        help="wrap each prompt in the checkpoint's chat template",
    )
    args = parser.parse_args()

    extra = {}
    if args.optlevel is not None:
        from vllm.config.vllm import OptimizationLevel

        extra["optimization_level"] = OptimizationLevel(args.optlevel)

    llm = LLM(
        model=args.model,
        **extra,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        # Equal to max_model_len, which is what turns chunked prefill off on
        # this plugin: prefill attends within the tokens it is given and does
        # not read a cached prefix, so a prompt split into chunks would lose
        # attention to its own earlier chunks.
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        # Prefill attends within the prompt only (no cached prefix is read).
        enable_prefix_caching=False,
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                "num_batched_tokens_buckets": [args.max_model_len],
                "num_seqs_buckets": [args.max_num_seqs],
                "on_device_sampling_config": {"all_greedy": True},
                # No NKI kernel is called by this model yet, which is the case
                # Qwen3.5's dense checkpoints need the mac threshold suppressed
                # for (see that example's mac_threshold_override).
                "hlo2tensorizer_options": "",
            },
        },
    )

    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    if args.chat:
        conversations = [[{"role": "user", "content": p}] for p in PROMPTS]
        outputs = llm.chat(conversations, sampling_params)
    else:
        outputs = llm.generate(PROMPTS, sampling_params)
    for prompt, output in zip(PROMPTS, outputs):
        print(f"Prompt:    {prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}\n")


if __name__ == "__main__":
    main()
