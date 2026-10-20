# SPDX-License-Identifier: Apache-2.0
"""Offline multimodal inference example for Qwen3-VL-32B on Neuron.

Demonstrates single-image, multi-image, video, and text-only generation
using the vLLM offline API with Qwen3-VL-32B-Instruct.

Usage:
    python examples/vllm_neuron/models/qwen3_vl/run.py \
        --model-checkpoint <path-to-checkpoint>/Qwen_Qwen3-VL-32B-Instruct

    # Video
    python examples/vllm_neuron/models/qwen3_vl/run.py \
        --model-checkpoint <path-to-checkpoint>/Qwen_Qwen3-VL-32B-Instruct \
        --video

    # Text-only (no images passed)
    python examples/vllm_neuron/models/qwen3_vl/run.py \
        --model-checkpoint <path-to-checkpoint>/Qwen_Qwen3-VL-32B-Instruct \
        --text-only
"""

import argparse
import os

os.environ["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = "1200"
os.environ["NEURON_LIBTORCH_COMPILATION_TIMEOUT"] = "1200"

from transformers import AutoProcessor

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset

SINGLE_IMAGE_QUESTION = "Describe this image in detail."
MULTI_IMAGE_QUESTION = "Compare these two images. What is different about them?"
VIDEO_QUESTION = "Describe what happens in this video."

# baby_reading at 8 frames has video_grid_thw [4, 22, 40] = 3520 unmerged
# vision tokens; the vision bucket must be >= that count.
VIDEO_NUM_FRAMES = 8
VIDEO_VISION_BUCKET = 4096


def create_single_image_input(processor: AutoProcessor) -> dict:
    """Single image + text prompt."""
    image = ImageAsset("cherry_blossom").pil_image.resize((640, 320))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": SINGLE_IMAGE_QUESTION},
            ],
        },
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": prompt, "multi_modal_data": {"image": [image]}}


def create_multi_image_input(processor: AutoProcessor) -> dict:
    """Multiple images + text prompt."""
    images = [
        ImageAsset("stop_sign").pil_image.resize((448, 448)),
        ImageAsset("cherry_blossom").pil_image.resize((448, 448)),
    ]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {
                    "type": "text",
                    "text": MULTI_IMAGE_QUESTION,
                },
            ],
        },
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": prompt, "multi_modal_data": {"image": images}}


def create_video_input(processor: AutoProcessor) -> dict:
    """Single video + text prompt.

    Qwen3-VL needs the video frames and their metadata together, so
    multi_modal_data carries the ``(ndarray, metadata)`` tuple the vLLM
    video parser expects (the metadata drives the per-frame timestamp
    tokens). The video placeholder span is non-contiguous, so it
    exercises the is_embed-aware vision-position mapping in the model runner.
    """
    asset = VideoAsset("baby_reading", num_frames=VIDEO_NUM_FRAMES)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": VIDEO_QUESTION},
            ],
        },
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {
        "prompt": prompt,
        "multi_modal_data": {"video": (asset.np_ndarrays, asset.metadata)},
    }


def create_text_only_inputs() -> list[str]:
    """Plain text prompts (no images passed at inference time)."""
    return [
        "I am gonna keep counting forever, 1 2 3 4 5 ",
        "The capital of France is ",
        "Once upon a time, there was a ",
        "def fibonacci(n):",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="Qwen/Qwen3-VL-32B-Instruct",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Run a video prompt instead of images",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Run text-only prompts (skip vision)",
    )
    args = parser.parse_args()

    # A single video item needs a larger vision block than the image demo.
    vision_bucket = VIDEO_VISION_BUCKET if args.video else 2048

    llm = LLM(
        model=args.model_checkpoint,
        max_model_len=32768,
        max_num_batched_tokens=4096,
        max_num_seqs=8,
        tensor_parallel_size=16,
        additional_config={
            "neuron_config": {
                "quantization": "bf16",
                "num_batched_tokens_buckets": [4096],
                "num_seqs_buckets": [8],
                "on_device_sampling_config": {
                    "all_greedy": True,
                },
            },
            "encoder_neuron_config": {
                "num_encoder_input_buckets": [vision_bucket],
                "encoder_attention_block_size": vision_bucket,
            },
        },
    )

    sampling_params = SamplingParams(max_tokens=50, temperature=0.0)

    if args.text_only:
        prompts = create_text_only_inputs()
        outputs = llm.generate(prompts, sampling_params)
        for prompt, output in zip(prompts, outputs):
            print(f"Prompt: {prompt!r}")
            print(f"Generated: {output.outputs[0].text!r}")
            print()
    elif args.video:
        processor = AutoProcessor.from_pretrained(args.model_checkpoint)
        outputs = llm.generate([create_video_input(processor)], sampling_params)
        print(f"Prompt: {VIDEO_QUESTION!r}")
        print(f"Generated: {outputs[0].outputs[0].text!r}")
        print()
    else:
        processor = AutoProcessor.from_pretrained(args.model_checkpoint)
        questions = [SINGLE_IMAGE_QUESTION, MULTI_IMAGE_QUESTION]
        inputs = [
            create_single_image_input(processor),
            create_multi_image_input(processor),
        ]
        outputs = llm.generate(inputs, sampling_params)
        for question, output in zip(questions, outputs):
            print(f"Prompt: {question!r}")
            print(f"Generated: {output.outputs[0].text!r}")
            print()


if __name__ == "__main__":
    main()
