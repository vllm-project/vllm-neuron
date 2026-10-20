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
        default="Qwen/Qwen3-235B-A22B",
        help="Path to the model checkpoint",
    )
    args = parser.parse_args()

    # If downloading checkpoint from HF (default behavior), consider setting the download_dir param below
    # to download the checkpoint to SSD (this may speed up the download). For example, set
    # download_dir="/path/to/local/ssd"
    llm = LLM(
        model=args.model_checkpoint,
        max_model_len=16384,
        max_num_batched_tokens=16384,
        max_num_seqs=64,
        tensor_parallel_size=64,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                "on_device_sampling_config": {},
                "num_batched_tokens_buckets": [4096, 16384],
                "num_seqs_buckets": [1, 64],
            }
        },
    )

    token_prompts = [
        "I am gonna keep counting forever, 1 2 3 4 5 ",
        "The capital of France is ",
        "Once upon a time, there was a ",
        "def fibonacci(n):",
    ]

    sampling_params = SamplingParams(max_tokens=50, temperature=0.0, top_p=1.0)

    outputs = llm.generate(token_prompts, sampling_params)

    print(outputs)


if __name__ == "__main__":
    main()
