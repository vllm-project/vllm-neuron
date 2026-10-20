# SPDX-License-Identifier: Apache-2.0
"""
Tensor Capture Example: Capture intermediate tensors from Qwen3-VL vision model.

Captures tensors from both the text decoder (language_model.layers.*) and the
vision encoder (visual.blocks.*) during a multimodal inference run.

Usage:
    python run_tensor_capture_qwen3_vl.py \
        --model Qwen/Qwen3-VL-32B-Instruct \
        --image /path/to/image.jpg \
        --tp-size 16
"""

import argparse
import os
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--capture-dir", type=str, default="/tmp/vllm_neuron_captures_qwen3vl"
    )
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--tp-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image file (required).",
    )
    args = parser.parse_args()

    # Resolve image path
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).resize((448, 448))

    # Capture both vision encoder blocks and text decoder layers
    modules = [
        "visual.blocks.0",
        "visual.blocks.1",
        "visual.merger",
        "language_model.layers.0",
        "language_model.layers.1",
        "lm_head",
    ]

    print(f"Model: {args.model}")
    print(f"Capture modules: {modules}")
    print(f"Capture dir: {args.capture_dir}")
    print(f"Image: {image_path}")

    os.environ["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = "1200"
    os.environ["NEURON_LIBTORCH_COMPILATION_TIMEOUT"] = "1200"

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.batch_size,
        tensor_parallel_size=args.tp_size,
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                "num_batched_tokens_buckets": [args.max_num_batched_tokens],
                "num_seqs_buckets": [args.batch_size],
                "on_device_sampling_config": {
                    "all_greedy": True,
                },
                "tensor_capture": {
                    "modules": modules,
                    "capture_dir": args.capture_dir,
                },
            },
            "encoder_neuron_config": {
                "num_encoder_input_buckets": [4096],
                "encoder_attention_block_size": 2048,
            },
        },
    )

    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    # Build vision prompt using the model's chat template
    processor = AutoProcessor.from_pretrained(args.model)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image briefly."},
            ],
        },
    ]
    vision_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    try:
        # Text-only prompt to exercise decoder capture
        print("\n--- Text-only inference ---")
        outputs = llm.generate(
            ["The capital of France is "],
            sampling_params,
        )
        for output in outputs:
            print(f"Prompt: {output.prompt!r}")
            print(f"Output: {output.outputs[0].text!r}")

        # Vision prompt to exercise both encoder and decoder capture
        print("\n--- Vision inference ---")
        outputs = llm.generate(
            [{"prompt": vision_prompt, "multi_modal_data": {"image": image}}],
            sampling_params,
        )
        for output in outputs:
            print(f"Output: {output.outputs[0].text!r}")

        # Show captured files
        print(f"\nCaptures saved to: {args.capture_dir}")
        for root, dirs, files in os.walk(args.capture_dir):
            level = root.replace(args.capture_dir, "").count(os.sep)
            indent = "  " * level
            print(f"{indent}{os.path.basename(root)}/")
            for f in sorted(files)[:5]:
                print(f"{indent}  {f}")
            if len(files) > 5:
                print(f"{indent}  ... and {len(files) - 5} more")
    finally:
        del llm


if __name__ == "__main__":
    main()
