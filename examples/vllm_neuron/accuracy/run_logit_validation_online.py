# SPDX-License-Identifier: Apache-2.0
"""
Logit Validation Example — Online Serving

Demonstrates three-way logit validation (FP32 baseline → BF16 expected → Neuron target)
using a running vLLM server via the OpenAI API.

Note: async_scheduling must be disabled because logprobs are not returned in
async scheduling mode with on-device sampling.

See also: docs/design/accuracy/logit_validation_design.md

Usage:
    # Launch its own server (via the shared LocalServer), validate, then stop it:
    python run_logit_validation_online.py --model meta-llama/Llama-3.1-8B-Instruct --tp-size 8

    # Validate against an already-running server:
    python run_logit_validation_online.py --model meta-llama/Llama-3.1-8B-Instruct --server-url http://localhost:8000
"""

import argparse
import gc
import json
from typing import List

import torch

# NOTE: import vllm BEFORE transformers to avoid Neuron class double-registration
import vllm  # noqa: F401
from vllm import LLM  # noqa: F401, E402 — triggers full Neuron plugin init

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from openai import OpenAI

# Works both when run as a script (``python .../run_logit_validation_online.py``,
# where sys.path[0] is this dir → sibling import) and when imported as part of
# the ``examples`` package (package-absolute import).
try:
    from examples.vllm_neuron.accuracy.server_utils import get_server  # noqa: E402
except ModuleNotFoundError:
    from server_utils import get_server  # noqa: E402

from vllm_neuron.accuracy.logit_validation import (
    multi_prompt_logit_validation,
)

SEQUENCE_LENGTH = 16
TARGET_DTYPE = torch.bfloat16

PROMPTS = [
    "I am gonna keep counting forever, 1 2 3 4 5",
    "The capital of France is",
    "In machine learning, a neural network",
]


def compute_goldens(model_checkpoint, target_dtype, output_length, prompts):
    """Compute FP32 baseline + dtype expected logits on CPU."""
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids_list = []
    fp32_logits_list = []
    dtype_logits_list = []

    # FP32 baseline (autoregressive)
    fp32_model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    fp32_model.eval()

    for prompt in prompts:
        input_ids = tokenizer(
            [prompt], return_tensors="pt", padding=True, truncation=True
        )["input_ids"]
        input_ids_list.append(input_ids.tolist())

        all_logits = []
        current = input_ids.clone()
        with torch.inference_mode():
            for _ in range(output_length):
                out = fp32_model(current, return_dict=True)
                next_logits = out.logits[:, -1, :]
                all_logits.append(next_logits)
                current = torch.cat(
                    [current, torch.argmax(next_logits, dim=-1, keepdim=True)], dim=1
                )
        fp32_logits_list.append(torch.stack(all_logits, dim=0))

    del fp32_model
    gc.collect()

    # dtype expected (teacher-forced from FP32 tokens)
    dtype_model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        torch_dtype=target_dtype,
        device_map="cpu",
        trust_remote_code=True,
    )
    dtype_model.eval()

    for i, prompt in enumerate(prompts):
        input_ids = tokenizer(
            [prompt], return_tensors="pt", padding=True, truncation=True
        )["input_ids"]
        teacher_seq = fp32_logits_list[i].argmax(dim=2)

        all_logits = []
        current = input_ids.clone()
        with torch.inference_mode():
            for step in range(teacher_seq.shape[0]):
                out = dtype_model(current, return_dict=True)
                all_logits.append(out.logits[:, -1, :].float())
                current = torch.cat([current, teacher_seq[step].unsqueeze(1)], dim=1)
        dtype_logits_list.append(torch.stack(all_logits, dim=0))

    del dtype_model
    gc.collect()

    return {
        "input_ids": input_ids_list,
        "fp32_logits": fp32_logits_list,
        "dtype_logits": dtype_logits_list,
    }


def create_online_generate_fn(
    server_url: str, model_name: str, tokenizer, sequence_length: int
):
    """Create generate function for online serving (OpenAI API with logprobs)."""
    client = OpenAI(api_key="EMPTY", base_url=f"{server_url}/v1")
    model_vocab_size = tokenizer.vocab_size

    def generate_fn(input_ids: List[List[int]]) -> torch.Tensor:
        batch_logits = []
        for ids in input_ids:
            response = client.completions.create(
                model=model_name,
                prompt=ids,
                max_tokens=sequence_length,
                temperature=0.0,
                logprobs=model_vocab_size,
            )

            sequence_logits = []
            if (
                response.choices[0].logprobs
                and response.choices[0].logprobs.top_logprobs
            ):
                for top_logprobs_dict in response.choices[0].logprobs.top_logprobs:
                    token_ids = []
                    logprob_values = []
                    for token, logprob in top_logprobs_dict.items():
                        if token.startswith("token_id:"):
                            token_id = int(token.split(":")[1])
                            token_ids.append(token_id)
                            logprob_values.append(logprob)
                    actual_max_id = max(token_ids) if token_ids else 0
                    token_tensor = torch.full(
                        (max(model_vocab_size, actual_max_id + 1),), -float("inf")
                    )
                    indices = torch.tensor(token_ids, dtype=torch.long)
                    values = torch.tensor(logprob_values)
                    token_tensor[indices] = values
                    sequence_logits.append(token_tensor)

            if sequence_logits:
                batch_logits.append(torch.stack(sequence_logits, dim=0))

        return torch.stack(batch_logits, dim=1)

    return generate_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--kv-segment-size", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help=(
            "Base URL of an already-running vLLM server to validate against, "
            "e.g. http://localhost:8000 . If omitted, this script launches its "
            "own 'vllm serve' (via the shared LocalServer) and stops it when done."
        ),
    )
    args = parser.parse_args()

    if args.kv_segment_size > args.max_model_len:
        parser.error(
            f"--kv-segment-size ({args.kv_segment_size}) must not exceed --max-model-len ({args.max_model_len})"
        )

    print("\n" + "=" * 60)
    print("MULTI-PROMPT VALIDATION (online)")
    print("=" * 60)

    # Step 1: Compute HF reference goldens on CPU
    goldens = compute_goldens(args.model, TARGET_DTYPE, SEQUENCE_LENGTH, PROMPTS)

    # Step 2: Get a vLLM server exposing raw logits (--max-logprobs -1,
    # --logprobs-mode raw_logits, --return-tokens-as-token-ids). With
    # --server-url this points at an operator-managed server; otherwise the
    # shared LocalServer launches (and later stops) its own 'vllm serve'.
    def build_serve_cmd(model: str, port: int) -> str:
        additional_config = json.dumps(
            {
                "neuron_config": {
                    "on_device_sampling_config": {},
                    "kv_segment_size_buckets": [args.kv_segment_size],
                    "num_batched_tokens_buckets": [args.kv_segment_size],
                    "num_seqs_buckets": [args.max_num_seqs],
                }
            }
        )
        return (
            f"vllm serve {model}"
            f" --tensor-parallel-size {args.tp_size}"
            f" --max-model-len {args.max_model_len}"
            f" --max-num-batched-tokens {args.kv_segment_size}"
            f" --max-num-seqs {args.max_num_seqs}"
            f" --max-logprobs -1"
            f" --logprobs-mode raw_logits"
            f" --no-async-scheduling"
            f" --no-enable-prefix-caching"
            f" --return-tokens-as-token-ids"
            f" --port {port}"
            f" --additional-config '{additional_config}'"
        )

    server = get_server(args.model, build_serve_cmd, server_url=args.server_url)
    print("Validating against server at:", server.base_url)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        # Step 3: Validate
        generate_fn = create_online_generate_fn(
            server.base_url, args.model, tokenizer, SEQUENCE_LENGTH
        )

        prompts_input_ids = [
            [list(ids) for ids in prompt] for prompt in goldens["input_ids"]
        ]

        result = multi_prompt_logit_validation(
            prompts_input_ids=prompts_input_ids,
            generate_fn=generate_fn,
            prompts_expected_logits=goldens["dtype_logits"],
            prompts_baseline_logits=goldens["fp32_logits"],
            colorize=True,
        )
    finally:
        server.stop()

    print(f"\nMulti-prompt validation: {'PASSED' if result.passed else 'FAILED'}")
