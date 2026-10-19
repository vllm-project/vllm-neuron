# SPDX-License-Identifier: Apache-2.0
"""
Build a Llama3 checkpoint with KV tensor scales for FP8 KV cache integration using llm-compressor

Produces a checkpoint with q_scale/k_scale/v_scale tensors in safetensors.
The quantization_config is stripped from config.json since vLLM Neuron doesn't
use vLLM's GPU quantization path.

Requirements:
    pip install llmcompressor

Usage:
    python examples/vllm_neuron/models/llama3/quantize_kv_fp8.py
    python examples/vllm_neuron/models/llama3/quantize_kv_fp8.py --model meta-llama/Llama-3.2-1B-Instruct
"""

import argparse

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIB_SAMPLES = 512
MAX_SEQ_LEN = 2048
STRATEGY = "tensor"


def process_and_tokenize(example, tokenizer):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(
        text,
        padding=False,
        max_length=MAX_SEQ_LEN,
        truncation=True,
        add_special_tokens=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--num-calib-samples", type=int, default=NUM_CALIB_SAMPLES)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{args.num_calib_samples}]")
    ds = ds.shuffle(seed=42).map(
        lambda ex: process_and_tokenize(ex, tokenizer),
        remove_columns=ds.column_names,
    )

    fp8_args = QuantizationArgs(num_bits=8, type="float", strategy=STRATEGY)
    recipe = QuantizationModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=["LlamaAttention"],
                input_activations=fp8_args,
            )
        },
        kv_cache_scheme=fp8_args,
    )

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQ_LEN,
        num_calibration_samples=args.num_calib_samples,
    )

    save_dir = f"{args.model.rstrip('/').split('/')[-1]}-kvattn-fp8-{STRATEGY}"
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)

    print(f"Saved to {save_dir}")


if __name__ == "__main__":
    main()
