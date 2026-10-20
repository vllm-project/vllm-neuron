# SPDX-License-Identifier: Apache-2.0
"""FP8 KV cache simulation mechanism for golden reference computation.

Patches HuggingFace models to simulate FP8 quantization noise in the KV cache,
matching the behavior of the vLLM Neuron FP8 KV cache kernel. Used by
:func:`vllm_neuron.accuracy.goldens.reference_logits.generate_three_way_reference_logits`
when ``kv_cache_dtype="fp8"`` — the fp8 teacher-forced *generator* that runs
inside this context manager lives there as ``generate_teacher_forced_logits_fp8``;
this module owns only the scale loading and the cache-patch mechanism.
"""

import json
import os
from contextlib import AbstractContextManager
from typing import Optional

import torch
from safetensors import safe_open


def load_kv_scales(
    model_checkpoint: str,
    num_layers: int,
    quant_scale_paths: Optional[dict] = None,
) -> list[tuple[float, float]]:
    """Load per-layer (k_scale, v_scale) for quantizing KV cache from a checkpoint.

    Args:
        model_checkpoint: Path to the model checkpoint directory.
        num_layers: Number of layers in the model.
        quant_scale_paths: Optional dict mapping scale names to tensor key
            templates with a {layer} placeholder, e.g.
            {"k_scale": "model.layers.{layer}.self_attn.k_scale",
             "v_scale": "model.layers.{layer}.self_attn.v_scale"}
            If None, all scales default to 1.0 (no calibration).

    Returns:
        list[tuple[float, float]]: Per-layer list of (k_scale, v_scale) pairs,
            length equal to num_layers.
    """
    if not quant_scale_paths:
        return [(1.0, 1.0)] * num_layers

    index_path = os.path.join(model_checkpoint, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        weight_map = {}
        for fname in os.listdir(model_checkpoint):
            if fname.endswith(".safetensors"):
                with safe_open(
                    os.path.join(model_checkpoint, fname),
                    framework="pt",
                    device="cpu",
                ) as f:
                    for key in f.keys():
                        weight_map[key] = fname

    def _load_scale(key):
        if key not in weight_map:
            raise ValueError(
                f"Scale key '{key}' not found in checkpoint '{model_checkpoint}'"
            )
        path = os.path.join(model_checkpoint, weight_map[key])
        with safe_open(path, framework="pt", device="cpu") as f:
            return f.get_tensor(key).item()

    k_path = quant_scale_paths["k_scale"]
    v_path = quant_scale_paths["v_scale"]
    scales = []
    for i in range(num_layers):
        k = _load_scale(k_path.format(layer=i))
        v = _load_scale(v_path.format(layer=i))
        scales.append((k, v))
    return scales


def hf_simulate_fp8_kv_cache(
    kv_scales: list[tuple[float, float]],
) -> AbstractContextManager[None]:
    """Context manager that patches ``DynamicCache.update`` to simulate an FP8 KV cache.

    Wraps ``transformers.DynamicCache.update`` so that each layer's K/V tensors
    are round-tripped through FP8 quantization (per the layer's scales) *before*
    they are returned for use in attention, matching the vLLM Neuron FP8 KV-cache
    kernel's numerical behavior. The original ``update`` is restored on exit
    (including on exception). Use it around an HF reference pass that reads from a
    ``DynamicCache`` (e.g. :func:`~vllm_neuron.accuracy.goldens.reference_logits.generate_teacher_forced_logits_fp8`)
    so the produced logits reflect FP8 KV noise.

    Args:
        kv_scales: Per-layer ``(k_scale, v_scale)`` pairs indexed by ``layer_idx``,
            as returned by :func:`load_kv_scales`.

    Yields:
        None. The patch is active only for the duration of the ``with`` block.

    Usage:
        with hf_simulate_fp8_kv_cache(kv_scales):
            run_hf_model()
    """
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        from vllm_neuron.utils.dtype_utils import FP8_CLAMP_MAX
        from transformers import DynamicCache

        original_update = DynamicCache.update

        def _fp8_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            k_scale, v_scale = kv_scales[layer_idx]
            key_states = _fp8_round_trip(key_states, k_scale, FP8_CLAMP_MAX)
            value_states = _fp8_round_trip(value_states, v_scale, FP8_CLAMP_MAX)
            return original_update(
                self, key_states, value_states, layer_idx, cache_kwargs
            )

        DynamicCache.update = _fp8_update
        try:
            yield
        finally:
            DynamicCache.update = original_update

    return _ctx()


def _fp8_round_trip(
    tensor: torch.Tensor, scale: float, clamp_max: float
) -> torch.Tensor:
    """Quantize tensor to FP8 and dequantize back, simulating KV cache noise.

    Uses reciprocal-scale arithmetic to match the model's actual computation:
    quantize via (tensor * recip_scale), dequantize via (tensor / recip_scale),
    where recip_scale = 1/bf16(scale) to match bf16 precision of the model.
    """
    dtype = tensor.dtype
    recip_scale = 1.0 / torch.tensor([scale], dtype=torch.bfloat16)
    return (
        (tensor * recip_scale)
        .clamp(-clamp_max, clamp_max)
        .to(torch.float8_e4m3fn)
        .to(dtype)
    ) / recip_scale
