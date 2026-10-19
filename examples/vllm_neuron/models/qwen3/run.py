# SPDX-License-Identifier: Apache-2.0
import argparse
import os

from vllm import LLM, SamplingParams

os.environ["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = "1200"
os.environ["NEURON_LIBTORCH_COMPILATION_TIMEOUT"] = "1200"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="Qwen/Qwen3-32B",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=32,
        help="Tensor parallel size. Qwen3-32B runs at tp>=8 (validated at "
        "tp8/tp16/tp32); higher TP gives more throughput headroom.",
    )
    args = parser.parse_args()

    # If downloading checkpoint from HF (default behavior), consider setting the download_dir param below
    # to download the checkpoint to SSD (this may speed up the download). For example, set
    # download_dir="/path/to/local/ssd"
    llm = LLM(
        enable_prefix_caching=False,
        model=args.model_checkpoint,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        max_num_seqs=4,
        tensor_parallel_size=args.tp_size,
        additional_config={
            "neuron_config": {
                "on_device_sampling_config": {
                    "all_greedy": "true",
                },
                "num_batched_tokens_buckets": [4096],
                "num_seqs_buckets": [4],
            }
        },
    )

    prompts = [
        "I am gonna keep counting forever, 1 2 3 4 5 ",
        "The capital of France is ",
        "Once upon a time, there was a ",
        "def fibonacci(n):",
    ]
    sampling_params = SamplingParams(max_tokens=50, temperature=0.0, top_p=1.0)

    outputs = llm.generate(prompts, sampling_params)

    print(outputs)


if __name__ == "__main__":
    main()
