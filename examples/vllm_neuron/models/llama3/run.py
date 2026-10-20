# SPDX-License-Identifier: Apache-2.0
import argparse

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default="auto",
        choices=["auto", "fp8"],
        help="Data type for KV cache storage",
    )
    args = parser.parse_args()

    # If downloading checkpoint from HF (default behavior), consider setting the download_dir param below
    # to download the checkpoint to SSD (this may speed up the download). For example, set
    # download_dir="/path/to/local/ssd"
    llm = LLM(
        enable_prefix_caching=False,
        model=args.model_checkpoint,
        max_model_len=256,
        max_num_seqs=4,
        tensor_parallel_size=8,
        kv_cache_dtype=args.kv_cache_dtype,
        additional_config={
            "neuron_config": {
                "on_device_sampling_config": {
                    "all_greedy": "true",
                },
                "num_batched_tokens_buckets": [256],
                "num_seqs_buckets": [4],  # Warmup decode batch sizes
            }
        },
    )

    token_prompts = [
        "I am gonna keep counting forever, 1 2 3 4 5 ",
        "The capital of France is ",
        "Once upon a time, there was a ",
        "def fibonacci(n):",
    ]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0, top_p=1.0)

    outputs = llm.generate(token_prompts, sampling_params)

    print(outputs)


if __name__ == "__main__":
    main()
