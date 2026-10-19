# SPDX-License-Identifier: Apache-2.0
"""Convert a generative Llama3 checkpoint into an embedding model and embed text.

Demonstrates the generic pooling adapter: a plain ``*ForCausalLM`` checkpoint run
with ``runner="pooling"`` is converted in place (lm_head dropped, pooler attached)
and returns embedding vectors — no embedding-specific checkpoint required.

Usage:
    python examples/vllm_neuron/models/llama3/run_embedding.py
    python examples/vllm_neuron/models/llama3/run_embedding.py --model-checkpoint meta-llama/Llama-3.1-8B-Instruct
"""

import argparse

from vllm import LLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Path to a generative Llama3 checkpoint",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=8,
    )
    args = parser.parse_args()

    # runner="pooling" runs this generative checkpoint through the generic
    # pooling adapter (drops lm_head, LAST-token pool + L2 normalize) and is
    # required for the offline embed() API (LLM.encode checks runner_type ==
    # "pooling"). Pooling is prefill-only — no sampling / decode-warmup config.
    llm = LLM(
        enable_prefix_caching=False,
        model=args.model_checkpoint,
        runner="pooling",
        max_model_len=256,
        max_num_seqs=4,
        tensor_parallel_size=args.tensor_parallel_size,
        additional_config={
            "neuron_config": {
                "num_batched_tokens_buckets": [256],
                "num_seqs_buckets": [4],
            }
        },
    )

    prompts = [
        "The capital of France is Paris.",
        "def fibonacci(n): return n if n < 2 else fib(n-1) + fib(n-2)",
        "Retrieval-augmented generation grounds answers in retrieved documents.",
        "Once upon a time, there was a robot.",
    ]

    outputs = llm.embed(prompts)

    for prompt, output in zip(prompts, outputs):
        vec = output.outputs.embedding
        print(
            f"dim={len(vec)}  first4={[round(x, 4) for x in vec[:4]]}  | {prompt[:48]}"
        )


if __name__ == "__main__":
    main()
