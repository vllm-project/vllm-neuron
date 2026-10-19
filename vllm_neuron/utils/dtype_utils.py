# SPDX-License-Identifier: Apache-2.0
import os
import torch
from vllm.config import ModelConfig


QUANTIZED_KV_CACHE_DTYPES = ["fp8", "fp8_e4m3"]
_SUPPORTED_KV_CACHE_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "fp8": torch.float8_e4m3fn,
    "fp8_e4m3": torch.float8_e4m3fn,
}

# TRN2 uses e4m3 (with inf), max finite = 240.
# TRN3 uses e4m3fn (no inf), max finite = 448.
_FP8_E4M3_MAX = 240.0
_FP8_E4M3FN_MAX = 448.0


def _resolve_fp8_clamp_max() -> float:
    from libtorch_neuronx_lite.compile.platform import get_platform_target

    try:
        target = get_platform_target()
    except RuntimeError:
        # No platform override and no NRT (bare CPU run).
        # CPU supports finite FP8 dtypes, so use the e4m3fn max.
        if os.environ.get("VLLM_NEURON_CPU_MODE") == "1":
            return _FP8_E4M3FN_MAX
        raise

    if target.startswith("trn3"):
        return _FP8_E4M3FN_MAX
    else:
        return _FP8_E4M3_MAX


# Resolved once at import time so it's a constant during torch.compile tracing.
FP8_CLAMP_MAX: float = _resolve_fp8_clamp_max()


def kv_cache_dtype_str_to_dtype(
    kv_cache_dtype: str, model_config: ModelConfig
) -> torch.dtype:
    """Convert a KV cache dtype string to a torch dtype.

    Unlike upstream vLLM which maps FP8 types to torch.uint8, this uses
    native torch FP8 dtypes (e.g. torch.float8_e4m3fn) so that standard
    PyTorch operations correctly interpret the tensor values. Once all FP8
    KV cache operations are being done within kernels, we can use the upstream
    vLLM one.

    Args:
        kv_cache_dtype: Dtype string (e.g. "auto", "fp8_e4m3", "bfloat16").
            "auto" resolves to the model's configured dtype.
        model_config: Model configuration to read dtype from when "auto".

    Returns:
        The corresponding torch.dtype.

    Raises:
        ValueError: If kv_cache_dtype is not supported on Neuron.
    """
    if kv_cache_dtype == "auto":
        return model_config.dtype

    if kv_cache_dtype not in _SUPPORTED_KV_CACHE_DTYPES:
        raise ValueError(
            f"Unsupported kv_cache_dtype '{kv_cache_dtype}' on Neuron. "
            f"Supported values: 'auto', {sorted(_SUPPORTED_KV_CACHE_DTYPES.keys())}"
        )

    return _SUPPORTED_KV_CACHE_DTYPES[kv_cache_dtype]


def validate_fp8_segmented_supported(kv_is_fp8: bool, fp8_packed: bool) -> None:
    """Reject the FP8 KV configs segmented prefill cannot read.

    ``attention_segmented_cte`` cannot read a non-packed FP8 K cache. Its
    ``load_kv_cache`` casts/transposes K on load and asserts up front that a
    dtype-mismatched (FP8 cache vs BF16 SBUF) K cache is only legal with
    ``k_pre_transposed`` or ``fp8_packed`` ("FP8 KV cache (dtype mismatch)
    requires k_pre_transposed=True"); the vLLM wrapper passes neither for a
    standard-layout cache, so the assert fires. (On older kernels the same
    config instead surfaced as a ``dma_transpose`` 2-byte-dtype error in the
    transpose-on-load step — same root cause, different symptom.) Non-packed
    FP8 is only valid for full prefill, where ``flash_attention`` reads dense
    bf16 K/V. Models call this at the top of their segmented-prefill branch so
    the unsupported combination fails fast with a clear message instead of a
    cryptic NKI kernel abort downstream. (Models with no packed-FP8 KV path,
    e.g. llama3, pass ``fp8_packed=False`` so any FP8 KV cache raises.)
    """
    if kv_is_fp8 and not fp8_packed:
        raise NotImplementedError(
            "FP8 segmented prefill requires a packed KV cache; "
            "attention_segmented_cte cannot read a non-packed FP8 K cache. "
            "Non-packed FP8 is supported for full prefill only."
        )
