# SPDX-License-Identifier: Apache-2.0
import argparse

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Path to the target model checkpoint",
    )
    parser.add_argument(
        "--draft-model-checkpoint",
        type=str,
        default="RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3",
        help="Path to the draft model checkpoint",
    )
    args = parser.parse_args()

    # TODO enable BS>1 eagle spec after we support decode batch bucket padding.
    # otherwise will fail due to recompilation triggered by input shape change in runtime.
    llm = LLM(
        enable_prefix_caching=False,
        model=args.model_checkpoint,
        tensor_parallel_size=8,
        max_model_len=256,
        max_num_seqs=1,
        speculative_config={
            "method": "eagle3",
            "model": args.draft_model_checkpoint,
            "num_speculative_tokens": 3,
        },
        additional_config={
            "neuron_config": {
                "on_device_sampling_config": {
                    "all_greedy": "true",
                },
                "num_batched_tokens_buckets": [256],
                "num_seqs_buckets": [1],  # Warmup decode batch sizes
            }
        },
    )

    prompts = [
        "I am gonna keep counting forever, 1 2 3 4 5 ",
    ]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0, top_p=1.0)

    outputs = llm.generate(prompts, sampling_params)

    print(outputs)


if __name__ == "__main__":
    main()
