# SPDX-License-Identifier: Apache-2.0
"""Encoder (vision) cache analysis for accuracy debugging.

Compares vision encoder outputs (fat tensors) between a HuggingFace reference
and vLLM Neuron. Validates that the Neuron-compiled vision encoder produces
embeddings numerically close to the reference.

Usage::

    from vllm_neuron.accuracy.encoder_cache_analysis import (
        enable_encoder_cache_snapshot,
        extract_vllm_encoder_cache,
        extract_hf_encoder_outputs,
        compare_encoder_caches_by_index,
        print_encoder_cache_report,
    )

    enable_encoder_cache_snapshot(llm)
    llm.generate(...)
    enc_cache = extract_vllm_encoder_cache(llm)
    hf_embeds = extract_hf_encoder_outputs(model, pixel_values, image_grid_thw)
    metrics = compare_encoder_caches_by_index(hf_per_image, vllm_per_image)
    print_encoder_cache_report(metrics)
"""

import io
import logging

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Union

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from vllm import LLM

logger = logging.getLogger(__name__)


@dataclass
class EncoderCacheMetrics:
    """Per-image encoder cache comparison metrics.

    Metrics are computed over the full embedding tensor for one image.
    L-inf and L2 are relative (normalized by reference magnitude),
    matching the convention in ``HeadMetrics`` for KV cache analysis.
    """

    linf: (
        float  # Relative L-inf: max|diff| / max|ref| (global across all tokens & dims)
    )
    l2: float  # Relative L2: ||diff||_2 / ||ref||_2
    cos_sim: float  # Mean per-token cosine similarity
    num_tokens: int  # Number of vision tokens in this image
    embedding_dim: int  # Embedding dimension (fat_dim for Qwen3-VL)


EncoderCacheComparisonResult = List[EncoderCacheMetrics]


# =============================================================================
# VLLM EXTRACTION
# =============================================================================


def extract_vllm_encoder_cache(llm: "LLM") -> Dict[str, torch.Tensor]:
    """Extract the encoder cache (encoder embeddings) from vLLM.

    Calls get_encoder_cache() on rank 0 worker and deserializes the
    tensors. Returns only the entries from the most recent forward pass
    (not stale LRU entries from prior requests).

    Call ``enable_encoder_cache_snapshot(llm)`` before ``generate()``
    to activate per-request snapshotting.

    Args:
        llm: vLLM LLM instance.

    Returns:
        Dict[mm_hash, tensor] where tensor is [num_tokens, fat_dim].

    Example:
        >>> enable_encoder_cache_snapshot(llm)
        >>> llm.generate(...)
        >>> enc_cache = extract_vllm_encoder_cache(llm)
    """
    import numpy as np

    results = llm.collective_rpc("get_encoder_cache")
    if not results or not results[0]:
        return {}

    # Each TP rank holds the whole encoder output, because the cache stores
    # the post-projector embedding, after the encoder's collectives. So rank 0
    # has the complete output.
    rank0_data = results[0]
    cache = {}
    for mm_hash, data in rank0_data.items():
        if isinstance(data, bytes):
            with io.BytesIO(data) as buf:
                cache[mm_hash] = torch.load(buf, weights_only=True).float()
        elif isinstance(data, torch.Tensor):
            cache[mm_hash] = data.float()
        elif isinstance(data, np.ndarray):
            cache[mm_hash] = torch.from_numpy(data).float()
    return cache


def enable_encoder_cache_snapshot(llm: "LLM") -> None:
    """Enable encoder cache snapshotting on all workers.

    Lightweight call — no serialization, just enables snapshotting so
    the next forward pass captures encoder outputs. Must be called
    before the generate() whose encoder cache you want to extract.

    Example:
        >>> enable_encoder_cache_snapshot(llm)
        >>> llm.generate(...)
        >>> enc = extract_vllm_encoder_cache(llm)
    """
    llm.collective_rpc("enable_encoder_cache_snapshot")


def cleanup_encoder_cache_snapshot(llm: "LLM") -> None:
    """Release encoder cache snapshot memory on all workers.

    Call after encoder cache analysis is complete to free memory.

    Example:
        >>> cleanup_encoder_cache_snapshot(llm)
    """
    llm.collective_rpc("clear_encoder_cache_snapshot")


# =============================================================================
# HF EXTRACTION
# =============================================================================


def extract_hf_encoder_outputs(
    model: "PreTrainedModel",
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor = None,
    dtype: torch.dtype = None,
    **visual_kwargs,
) -> torch.Tensor:
    """Extract vision encoder outputs from a HuggingFace model.

    Runs the vision model (patch embed + transformer + merger) and returns
    the output embeddings as a tensor matching the format stored in vLLM's
    encoder cache.

    For Qwen2/3-VL: the visual encoder returns
    ``(main_embeds, [deepstack_0, ...])`` and this function concatenates
    them along dim=-1 to form the fat tensor ``[tokens, hidden * (1+N)]``.

    For models without deepstack (e.g., Pixtral, LLaVA): returns the raw
    encoder output directly.

    Args:
        model: HuggingFace model with a ``visual`` or ``vision_model`` attr.
        pixel_values: Preprocessed pixel values (shape depends on model).
        image_grid_thw: [num_images, 3] — required for Qwen-VL models.
        dtype: Optional dtype to cast model to before running.
        **visual_kwargs: Additional keyword arguments passed to the vision
            encoder (e.g., ``grid_thw`` for Qwen-VL).

    Returns:
        Tensor of shape [total_tokens, embed_dim] (or fat_dim for deepstack).

    Example:
        >>> # Qwen3-VL
        >>> embeds = extract_hf_encoder_outputs(model, pv, image_grid_thw=grid)
        >>> # Pixtral / LLaVA
        >>> embeds = extract_hf_encoder_outputs(model, pv)
    """
    model.eval()

    visual = getattr(model, "visual", None) or getattr(model, "vision_model", None)
    if visual is None:
        raise ValueError("Model does not have a 'visual' or 'vision_model' attribute.")

    input_dtype = dtype if dtype is not None else next(visual.parameters()).dtype

    # Build kwargs for the visual encoder
    kwargs = {}
    if image_grid_thw is not None:
        kwargs["grid_thw"] = image_grid_thw
    kwargs.update(visual_kwargs)

    with torch.inference_mode():
        result = visual(pixel_values.to(input_dtype), **kwargs)

    if isinstance(result, tuple) and len(result) == 2:
        main_embeds, deepstack_list = result
        if isinstance(deepstack_list, list) and deepstack_list:
            fat_tensor = torch.cat([main_embeds] + deepstack_list, dim=-1)
            return fat_tensor.cpu().float()
        return main_embeds.cpu().float()
    elif isinstance(result, torch.Tensor):
        return result.cpu().float()
    else:
        raise ValueError(f"Unexpected visual encoder output type: {type(result)}")


# =============================================================================
# COMPARISON
# =============================================================================


def compare_encoder_caches(
    expected_embeds: Dict[str, torch.Tensor],
    actual_embeds: Dict[str, torch.Tensor],
    baseline_embeds: Dict[str, torch.Tensor] = None,
) -> Dict[str, EncoderCacheMetrics]:
    """Compare encoder cache embeddings keyed by mm_hash.

    Use this for **vLLM-vs-vLLM** comparison where both sides produce
    mm_hashes (e.g., comparing two quantization configs). A key mismatch
    indicates a hashing/preprocessing bug and should be flagged.

    For **HF-vs-vLLM** comparison, use ``compare_encoder_caches_by_index``
    instead (HF doesn't produce mm_hashes).

    Two-way (no baseline): compares actual vs expected directly.
    Three-way (with baseline): errors measured relative to baseline.

    Args:
        expected_embeds: Reference encoder outputs per mm_hash.
            Each value is [num_tokens, dim].
        actual_embeds: Target encoder outputs per mm_hash.
        baseline_embeds: Optional ground-truth for three-way comparison.

    Returns:
        Dict[mm_hash, EncoderCacheMetrics] with comparison metrics.

    Example:
        >>> result = compare_encoder_caches(vllm_bf16_cache, vllm_fp8_cache)
        >>> result['hash123'].linf  # relative L-inf
        0.1234
    """
    has_baseline = baseline_embeds is not None
    ref_embeds = baseline_embeds if has_baseline else expected_embeds

    common_keys = set(ref_embeds.keys()) & set(actual_embeds.keys())
    if not common_keys:
        common_keys = set(expected_embeds.keys()) & set(actual_embeds.keys())

    results = {}
    for key in sorted(common_keys):
        ref = ref_embeds.get(key, expected_embeds.get(key))
        actual = actual_embeds[key]

        if ref is None:
            continue

        results[key] = _compute_encoder_metrics(ref, actual)

    return results


def compare_encoder_caches_by_index(
    expected_list: List[torch.Tensor],
    actual_list: List[torch.Tensor],
    baseline_list: List[torch.Tensor] = None,
) -> List[EncoderCacheMetrics]:
    """Compare encoder cache embeddings by positional index (image order).

    Use this for **HF-vs-vLLM** comparison where HF doesn't produce
    mm_hashes. Images are matched by their order in the request.

    For **vLLM-vs-vLLM** comparison where both sides have mm_hashes,
    use ``compare_encoder_caches`` instead.

    Args:
        expected_list: List of expected embeddings per image [num_tokens, dim].
        actual_list: List of actual embeddings per image.
        baseline_list: Optional ground-truth per image.

    Returns:
        List[EncoderCacheMetrics] with one entry per image.

    Example:
        >>> metrics = compare_encoder_caches_by_index(hf_per_image, vllm_per_image)
        >>> metrics[0].linf  # relative L-inf for first image
        0.1234
    """
    has_baseline = baseline_list is not None
    ref_list = baseline_list if has_baseline else expected_list

    if len(ref_list) != len(actual_list):
        logger.warning(
            f"Image count mismatch: expected {len(ref_list)}, "
            f"actual {len(actual_list)}. Comparing min({len(ref_list)}, {len(actual_list)})."
        )
    n = min(len(ref_list), len(actual_list))
    results = []

    for i in range(n):
        ref = ref_list[i]
        actual = actual_list[i]

        if ref.shape[0] != actual.shape[0]:
            logger.warning(
                f"Image {i} token count mismatch: expected {ref.shape[0]}, "
                f"actual {actual.shape[0]}. Using min."
            )

        results.append(_compute_encoder_metrics(ref, actual))

    return results


def _compute_encoder_metrics(
    ref: torch.Tensor, actual: torch.Tensor
) -> EncoderCacheMetrics:
    """Compute comparison metrics between reference and actual embeddings."""
    ref = ref.float()
    actual = actual.float()

    min_len = min(ref.shape[0], actual.shape[0])
    ref = ref[:min_len]
    actual = actual[:min_len]

    cos_sims = torch.nn.functional.cosine_similarity(ref, actual, dim=-1)
    cos_sim = cos_sims.mean().item()

    ref_max = ref.abs().max().clamp(min=1e-10)
    linf = ((ref - actual).abs().max() / ref_max).item()

    ref_norm = ref.norm().clamp(min=1e-10)
    l2 = ((ref - actual).norm() / ref_norm).item()

    return EncoderCacheMetrics(
        linf=linf,
        l2=l2,
        cos_sim=cos_sim,
        num_tokens=min_len,
        embedding_dim=ref.shape[-1],
    )


# =============================================================================
# REPORTING
# =============================================================================


def print_encoder_cache_report(
    results: Union[Dict[str, EncoderCacheMetrics], List[EncoderCacheMetrics]],
) -> None:
    """Print encoder cache comparison report.

    Example:
        >>> print_encoder_cache_report(metrics)
    """
    if isinstance(results, dict):
        items = list(results.items())
    else:
        items = [(f"image_{i}", m) for i, m in enumerate(results)]

    if not items:
        print("No encoder cache results")
        return

    print(f"\n{'=' * 60}")
    print("ENCODER CACHE COMPARISON")
    print(f"{'=' * 60}")
    print(
        f"  {'Key':<20} {'Tokens':<8} {'Dim':<6} {'L-inf':<12} {'L2':<12} {'CosSim':<10}"
    )
    print(f"  {'-' * 20} {'-' * 8} {'-' * 6} {'-' * 12} {'-' * 12} {'-' * 10}")
    for key, m in items:
        short_key = key[:20] if len(key) > 20 else key
        print(
            f"  {short_key:<20} {m.num_tokens:<8} {m.embedding_dim:<6} "
            f"{m.linf:<12.2e} {m.l2:<12.2e} {m.cos_sim:<10.6f}"
        )
