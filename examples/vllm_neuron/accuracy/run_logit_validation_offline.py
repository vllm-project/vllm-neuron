# SPDX-License-Identifier: Apache-2.0
"""
Logit Validation Example — Offline Serving

Demonstrates three-way logit validation (FP32 baseline → BF16 expected → Neuron target)
using the vLLM LLM class (offline serving).

Note: async_scheduling must be disabled because logprobs are not returned in
async scheduling mode with on-device sampling.

See also: docs/design/accuracy/logit_validation_design.md

Usage:
    python run_logit_validation_offline.py --model meta-llama/Llama-3.1-8B-Instruct --tp-size 32
"""

import argparse
import gc
from typing import List

import torch

# NOTE: import vllm BEFORE transformers to avoid Neuron class double-registration
from vllm import LLM, SamplingParams  # noqa: E402

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

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


def create_generate_fn(llm: LLM, output_length: int):
    """Create generate function for offline serving."""
    model_vocab_size = llm.get_tokenizer().vocab_size

    def generate_fn(input_ids: List[List[int]]) -> torch.Tensor:
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=output_length,
            logprobs=model_vocab_size,
            detokenize=False,
        )
        prompts = [{"prompt_token_ids": ids} for ids in input_ids]
        outputs = llm.generate(prompts, sampling_params)

        batch_logits = []
        for output in outputs:
            sequence_logits = []
            for completion in output.outputs:
                if not completion.logprobs:
                    continue
                for logprob_dict in completion.logprobs:
                    actual_max_id = max(logprob_dict.keys())
                    token_tensor = torch.full(
                        (max(model_vocab_size, actual_max_id + 1),), -float("inf")
                    )
                    for tid, lp in logprob_dict.items():
                        token_tensor[tid] = lp.logprob
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
    args = parser.parse_args()

    if args.kv_segment_size > args.max_model_len:
        parser.error(
            f"--kv-segment-size ({args.kv_segment_size}) must not exceed --max-model-len ({args.max_model_len})"
        )

    print("\n" + "=" * 60)
    print("MULTI-PROMPT VALIDATION (offline)")
    print("=" * 60)

    # Step 1: Compute HF reference goldens on CPU
    goldens = compute_goldens(args.model, TARGET_DTYPE, SEQUENCE_LENGTH, PROMPTS)

    # Step 2: Create LLM
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.kv_segment_size,
        tensor_parallel_size=args.tp_size,
        max_num_seqs=args.max_num_seqs,
        max_logprobs=-1,
        logprobs_mode="raw_logits",
        enable_prefix_caching=False,
        async_scheduling=False,
        additional_config={
            "neuron_config": {
                "on_device_sampling_config": {},
                "kv_segment_size_buckets": [args.kv_segment_size],
                "num_batched_tokens_buckets": [args.kv_segment_size],
                "num_seqs_buckets": [args.max_num_seqs],
            }
        },
    )

    # Step 3: Validate
    generate_fn = create_generate_fn(llm, SEQUENCE_LENGTH)

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

    print(f"\nMulti-prompt validation: {'PASSED' if result.passed else 'FAILED'}")
