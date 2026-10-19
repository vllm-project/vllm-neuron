#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Encoder Cache Analysis: compare vision encoder outputs between HF and vLLM.

Generates reference vision embeddings on CPU using HuggingFace, then runs
vLLM on Neuron and compares encoder cache outputs (the fat tensors stored
after the vision encoder).

This validates that the Neuron-compiled vision encoder produces embeddings
numerically close to the HuggingFace reference.

See also: docs/design/accuracy/kv_cache_analysis_design.md

Naming matches ``logit_validation``:
- expected: HF reference encoder outputs (required)
- actual: vLLM Neuron encoder cache (target under test)
- baseline: optional FP32 ground truth for three-way comparison

Two-way mode (--two-way): compares BF16 HF vs vLLM using cosine, L-inf, L2.
Three-way mode (default): FP32 baseline + BF16 expected vs vLLM actual.

The script is model-agnostic: it uses the HuggingFace processor for image
preprocessing and reads vision config parameters (e.g., spatial_merge_size)
from the model config rather than hardcoding model-specific values.

Usage — Two-way (BF16 HF vs vLLM):
    NEURON_VISIBLE_DEVICES=0-31 python run_encoder_cache_analysis.py \
        --model Qwen/Qwen3-VL-32B-Instruct --two-way \
        --tp-size 32 --max-model-len 8192 \
        --output-dir enc_two_way

Usage — Three-way (FP32 + BF16 HF vs vLLM):
    NEURON_VISIBLE_DEVICES=0-31 python run_encoder_cache_analysis.py \
        --model Qwen/Qwen3-VL-32B-Instruct \
        --tp-size 32 --max-model-len 8192 \
        --output-dir enc_three_way

Usage — Load pre-saved goldens (skip CPU generation):
    python run_encoder_cache_analysis.py \
        --model Qwen/Qwen3-VL-32B-Instruct \
        --goldens-path enc_three_way/goldens.pkl \
        --tp-size 32 --output-dir enc_rerun
"""

import argparse
import gc
import json
import logging
import os
import pickle

import torch

# NOTE: import vllm BEFORE transformers to avoid Neuron class double-registration
from vllm import LLM, SamplingParams  # noqa: E402

from transformers import AutoConfig, AutoProcessor  # noqa: E402

from vllm.assets.image import ImageAsset

from vllm_neuron.accuracy.encoder_cache_analysis import (
    compare_encoder_caches_by_index,
    enable_encoder_cache_snapshot,
    extract_hf_encoder_outputs,
    extract_vllm_encoder_cache,
    cleanup_encoder_cache_snapshot,
    print_encoder_cache_report,
)

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_INPUTS = [
    [("stop_sign", (640, 320))],
    [("cherry_blossom", (640, 320))],
    [("stop_sign", (448, 448)), ("cherry_blossom", (448, 448))],
]

QUESTION = "Describe what you see in these images in detail."


# =============================================================================
# Image & prompt helpers
# =============================================================================


def load_images(image_specs):
    """Load PIL images from vLLM assets with specified sizes."""
    return [ImageAsset(name).pil_image.resize(size) for name, size in image_specs]


def build_prompt(images, processor):
    """Build a multimodal prompt for the model."""
    placeholders = [{"type": "image"} for _ in images]
    messages = [
        {
            "role": "user",
            "content": [*placeholders, {"type": "text", "text": QUESTION}],
        },
    ]
    formatted = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": formatted, "multi_modal_data": {"image": images}}


def preprocess_images_for_hf(images, processor):
    """Preprocess images using the HuggingFace processor.

    Returns:
        Dict of processor outputs (model-specific keys like pixel_values,
        image_grid_thw, pixel_attention_mask, etc.)
    """
    placeholders = [{"type": "image"} for _ in images]
    messages = [
        {
            "role": "user",
            "content": [*placeholders, {"type": "text", "text": QUESTION}],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=images,
        return_tensors="pt",
        padding=True,
    )
    return inputs


def get_spatial_merge_size(model_config):
    """Read spatial_merge_size from the model's vision config.

    Returns None if the model doesn't use spatial merging (e.g., Pixtral,
    LLaVA), meaning the encoder output is already per-image.
    """
    vision_config = getattr(model_config, "vision_config", None)
    if vision_config is None:
        return None
    return getattr(vision_config, "spatial_merge_size", None)


def split_embeds_by_image(embeds_tensor, processor_inputs, spatial_merge_size):
    """Split a flat embeddings tensor into per-image tensors.

    Uses model-specific metadata from the processor to determine how many
    tokens belong to each image.

    Args:
        embeds_tensor: [total_merged_tokens, dim] flat embeddings.
        processor_inputs: Dict of processor outputs (contains image_grid_thw
            for Qwen-VL models, or pixel_attention_mask for others).
        spatial_merge_size: Spatial merge factor from the model's vision
            config. None means no merging (each image is a separate entry).

    Returns:
        List of tensors, one per image.
    """
    image_grid_thw = processor_inputs.get("image_grid_thw")
    if image_grid_thw is not None and spatial_merge_size is not None:
        merge_factor = spatial_merge_size**2
        tokens_per_image = image_grid_thw.prod(dim=1).tolist()
        merged_per_image = [t // merge_factor for t in tokens_per_image]
        return list(embeds_tensor.split(merged_per_image, dim=0))

    # Fallback for models without image_grid_thw: assume embeddings are
    # already split (one entry per image in encoder cache).
    logger.warning(
        "No image_grid_thw found in processor inputs. "
        "Returning embeddings as a single entry."
    )
    return [embeds_tensor]


# =============================================================================
# Golden generation
# =============================================================================


def generate_hf_encoder_goldens(
    model_id,
    image_inputs,
    baseline_dtype=torch.float32,
    expected_dtype=torch.bfloat16,
):
    """Generate encoder output goldens from HF model on CPU.

    Model-agnostic: uses the HF processor for preprocessing and passes
    model-specific kwargs (e.g., image_grid_thw) to the vision encoder.

    Args:
        model_id: HuggingFace model ID.
        image_inputs: List of image specs [(name, (w, h)), ...] per prompt.
        baseline_dtype: Dtype for baseline. None for two-way.
        expected_dtype: Dtype for expected reference.

    Returns:
        List of dicts with per-prompt goldens.
    """
    from transformers import AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    goldens = []
    three_way = baseline_dtype is not None

    for i, image_specs in enumerate(image_inputs):
        images = load_images(image_specs)
        inputs = preprocess_images_for_hf(images, processor)
        goldens.append(
            {
                "image_specs": image_specs,
                "images": images,
                "processor_inputs": inputs,
            }
        )

    # Build vision encoder kwargs from processor inputs (model-agnostic)
    def _build_encoder_kwargs(processor_inputs):
        """Extract vision-encoder-specific kwargs from processor outputs."""
        kwargs = {}
        pixel_values = processor_inputs.get("pixel_values")
        image_grid_thw = processor_inputs.get("image_grid_thw")
        return pixel_values, image_grid_thw, kwargs

    if three_way:
        print(f"Generating {baseline_dtype} baseline encoder goldens on CPU...")
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=baseline_dtype,
            device_map="cpu",
            trust_remote_code=True,
        )
        model.eval()
        for i, g in enumerate(goldens):
            pixel_values, image_grid_thw, kwargs = _build_encoder_kwargs(
                g["processor_inputs"]
            )
            embeds = extract_hf_encoder_outputs(
                model,
                pixel_values,
                image_grid_thw=image_grid_thw,
                dtype=baseline_dtype,
                **kwargs,
            )
            g["baseline_embeds"] = embeds
            print(f"  image set {i}: baseline embeds shape={embeds.shape}")
        del model
        gc.collect()

    print(f"Generating {expected_dtype} expected encoder goldens on CPU...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=expected_dtype,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    for i, g in enumerate(goldens):
        pixel_values, image_grid_thw, kwargs = _build_encoder_kwargs(
            g["processor_inputs"]
        )
        embeds = extract_hf_encoder_outputs(
            model,
            pixel_values,
            image_grid_thw=image_grid_thw,
            dtype=expected_dtype,
            **kwargs,
        )
        g["expected_embeds"] = embeds
        print(f"  image set {i}: expected embeds shape={embeds.shape}")
    del model
    gc.collect()

    return goldens


# =============================================================================
# vLLM helpers
# =============================================================================


def create_vllm(model_id, tp_size, max_model_len, neuron_config=None):
    """Create vLLM LLM instance for vision model."""
    nc = neuron_config or {}
    nc.setdefault("on_device_sampling_config", {})
    nc.setdefault("num_batched_tokens_buckets", [max_model_len])
    nc.setdefault("num_seqs_buckets", [4])

    return LLM(
        model=model_id,
        max_model_len=max_model_len,
        tensor_parallel_size=tp_size,
        max_num_seqs=4,
        enable_prefix_caching=False,
        async_scheduling=False,
        additional_config={
            "neuron_config": nc,
            "encoder_neuron_config": {
                "num_encoder_input_buckets": [2048, 4096],
                "encoder_attention_block_size": 2048,
            },
        },
    )


# =============================================================================
# Per-prompt analysis
# =============================================================================


def run_image_set(llm, processor, golden, output_prefix, idx, spatial_merge_size):
    """Run encoder cache comparison for one image set.

    Args:
        llm: vLLM LLM instance.
        processor: HF processor for building prompts.
        golden: Dict with expected/baseline embeddings and image data.
        output_prefix: Path prefix for output files.
        idx: Image set index.
        spatial_merge_size: From model config (None if not applicable).

    Returns:
        Dict with metrics, or None on failure.
    """
    images = golden["images"]
    prompt_data = build_prompt(images, processor)

    # Enable snapshot and generate
    enable_encoder_cache_snapshot(llm)
    params = SamplingParams(temperature=0, max_tokens=32)
    llm.generate([prompt_data], params)

    # Extract encoder cache (contains all entries used by this request)
    enc_cache = extract_vllm_encoder_cache(llm)
    if not enc_cache:
        print(f"  WARNING: no encoder cache extracted for image set {idx}")
        return None

    # Split HF reference into per-image using processor metadata
    processor_inputs = golden["processor_inputs"]
    expected_per_image = split_embeds_by_image(
        golden["expected_embeds"], processor_inputs, spatial_merge_size
    )

    actual_per_image = list(enc_cache.values())

    # Compare by positional index (HF doesn't produce mm_hashes, so
    # index-based matching is the correct approach for HF-vs-vLLM).
    # For vLLM-vs-vLLM comparison use compare_encoder_caches (hash-keyed).
    baseline_per_image = None
    if "baseline_embeds" in golden:
        baseline_per_image = split_embeds_by_image(
            golden["baseline_embeds"], processor_inputs, spatial_merge_size
        )

    metrics = compare_encoder_caches_by_index(
        expected_per_image, actual_per_image, baseline_list=baseline_per_image
    )

    # Save metrics
    metrics_out = []
    for i, m in enumerate(metrics):
        metrics_out.append(
            {
                "image_idx": i,
                "linf": m.linf,
                "l2": m.l2,
                "cos_sim": m.cos_sim,
                "num_tokens": m.num_tokens,
                "embedding_dim": m.embedding_dim,
            }
        )

    with open(f"{output_prefix}.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print_encoder_cache_report(metrics)

    return {
        "metrics": metrics,
        "num_images": len(metrics),
    }


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Encoder Cache Analysis: vision encoder comparison on Neuron",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tp-size", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument(
        "--goldens-path",
        type=str,
        default=None,
        help="Load pre-saved goldens.pkl (skip CPU generation)",
    )
    parser.add_argument(
        "--two-way",
        action="store_true",
        help="Two-way mode: skip baseline, compare expected vs actual only",
    )
    parser.add_argument(
        "--neuron-config",
        type=str,
        default=None,
        help="JSON string for neuron_config overrides",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    baseline_dtype = None if args.two_way else torch.float32

    # Read spatial_merge_size from the model's vision config
    model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    spatial_merge_size = get_spatial_merge_size(model_config)
    if spatial_merge_size is not None:
        print(f"Model spatial_merge_size: {spatial_merge_size}")

    # --- Goldens ---
    goldens_path = args.goldens_path or os.path.join(args.output_dir, "goldens.pkl")
    if os.path.exists(goldens_path):
        print(f"Loading goldens from {goldens_path}")
        with open(goldens_path, "rb") as f:
            goldens = pickle.load(f)
    else:
        goldens = generate_hf_encoder_goldens(
            args.model, DEFAULT_IMAGE_INPUTS, baseline_dtype=baseline_dtype
        )
        save_path = os.path.join(args.output_dir, "goldens.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(goldens, f)
        print(f"Saved goldens to {save_path}")

    # --- vLLM ---
    neuron_config = json.loads(args.neuron_config) if args.neuron_config else None

    print(f"\nStarting vLLM (TP={args.tp_size}, max_model_len={args.max_model_len})...")
    llm = create_vllm(args.model, args.tp_size, args.max_model_len, neuron_config)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    # --- Per-image-set analysis ---
    all_results = []
    for i, golden in enumerate(goldens):
        print(f"\n{'=' * 60}")
        print(f"Image set {i}: {golden['image_specs']}")
        print(f"{'=' * 60}")

        prefix = os.path.join(args.output_dir, f"encoder_set_{i}")
        result = run_image_set(llm, processor, golden, prefix, i, spatial_merge_size)
        if result:
            all_results.append({"idx": i, **result})

    # --- Cleanup ---
    cleanup_encoder_cache_snapshot(llm)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for r in all_results:
        metrics = r["metrics"]
        max_linf = max(m.linf for m in metrics) if metrics else 0
        max_l2 = max(m.l2 for m in metrics) if metrics else 0
        avg_cos = sum(m.cos_sim for m in metrics) / len(metrics) if metrics else 0
        print(
            f"  set_{r['idx']}: images={r['num_images']}, "
            f"max_linf={max_linf:.4f}, max_l2={max_l2:.4f}, "
            f"avg_cos={avg_cos:.6f}"
        )


if __name__ == "__main__":
    main()
