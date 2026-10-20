# SPDX-License-Identifier: Apache-2.0
"""HuggingFace reference-model loading for accuracy validation.

Load a HuggingFace model for reference logit generation. This module owns only
the *loading* concern; reference-logit generation and golden orchestration live
in :mod:`vllm_neuron.accuracy.goldens.reference_logits`, and the FP8 KV-cache
simulation mechanism in :mod:`vllm_neuron.accuracy.goldens.fp8_kv_golden`.

Shared by the accuracy debugger prompt plugins
(:mod:`vllm_neuron.accuracy.accuracy_debugger.prompt_plugins`) and the
full-model logit tests.
"""

from typing import Any, Optional

import torch


def init_hf_model(
    model_checkpoint: str,
    dtype: torch.dtype,
    config: Optional[Any] = None,
    attn_implementation: str | None = "sdpa",
    eager_attn_fallback: bool = False,
) -> Any:
    """Load a HuggingFace model as-is for reference logit generation.

    This is the clean, general reference loader: it selects the right AutoModel
    class for the architecture, loads via ``from_pretrained`` (which dequantizes
    standard quantized checkpoints such as MXFP4), prefers SDPA for
    memory-efficient attention, and casts to the target dtype. It applies **no**
    model-specific overrides.

    Callers that need model-specific workarounds — e.g. an LLM-Compressor MXFP8
    (compressed-tensors) CPU-dequant bypass, or a GPT-OSS non-SDPA
    chunked-attention patch — supply a richer loader that wraps this function and
    layers those overrides on top. ``generate_three_way_reference_logits`` accepts
    such a loader via its ``model_loader`` parameter.

    Requires 'accelerate' to be installed for quantized models:
        pip install accelerate

    Args:
        model_checkpoint: HuggingFace model ID or local path.
        dtype: Target dtype for inference.
        config: Optional PretrainedConfig to override the checkpoint's config
            (e.g., for layer truncation via num_hidden_layers).
        attn_implementation: Attention backend passed to ``from_pretrained``.
            Defaults to ``"sdpa"`` (memory-efficient, avoids S×S materialization).
            Pass ``None`` to let transformers choose its default backend — needed
            by non-SDPA models (e.g. GPT-OSS) whose loader override retries here
            after patching eager attention.
        eager_attn_fallback: When ``True``, a load that fails because the model
            rejects SDPA (``ValueError``/``TypeError``) is retried once with
            ``attn_implementation="eager"``. Defaults to ``False`` so the
            exception propagates unchanged — a caller that layers its own
            non-SDPA handling (e.g. a chunked-attention patch, which keeps the
            O(S·chunk) memory optimization) relies on the exception to trigger
            it, and swallowing it here would suppress that. Set ``True`` to load
            non-SDPA models directly when no such override is layered on top.

    Returns:
        An ``nn.Module`` in eval mode on CPU.

    Example:
        >>> model = init_hf_model("meta-llama/Llama-3.1-8B", torch.float32)
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    # Select the AutoModel class from the config. Read the transformers registry
    # for speech models; do not guess from the architecture name.
    # - Speech models (Qwen3-ASR, Whisper): use AutoModelForSpeechSeq2Seq
    # - ForCausalLM models (Llama, GPT-OSS): use AutoModelForCausalLM
    # - ForConditionalGeneration models (Qwen3-VL, LLaVA): use AutoModelForImageTextToText
    if config is not None:
        _cfg = config
    else:
        _cfg = AutoConfig.from_pretrained(model_checkpoint, trust_remote_code=True)
    arch = (getattr(_cfg, "architectures", None) or [""])[0]
    model_type = getattr(_cfg, "model_type", "")

    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES as _SPEECH_MODELS,
    )

    if model_type in _SPEECH_MODELS:
        from transformers import AutoModelForSpeechSeq2Seq

        model_cls = AutoModelForSpeechSeq2Seq
    elif "ForConditionalGeneration" in arch:
        from transformers import AutoModelForImageTextToText

        model_cls = AutoModelForImageTextToText
    else:
        model_cls = AutoModelForCausalLM

    kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": True}
    if config is not None:
        kwargs["config"] = config
    # Only pin attn_implementation when requested; None lets transformers decide.
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    try:
        model = model_cls.from_pretrained(model_checkpoint, **kwargs)
    except (ValueError, TypeError):
        # Some architectures (e.g. GPT-OSS) reject SDPA. Re-raise unless the
        # caller opted into the eager fallback: a caller that layers its own
        # non-SDPA handling (e.g. a chunked-attention patch) relies on this
        # exception to trigger it, so swallowing it here would suppress that.
        if not eager_attn_fallback:
            raise
        kwargs["attn_implementation"] = "eager"
        attn_implementation = "eager"
        model = model_cls.from_pretrained(model_checkpoint, **kwargs)
    if dtype != torch.bfloat16:
        model = model.to(dtype)
    model.eval()
    print(
        f"[RefModel] Loaded with attn_implementation={attn_implementation}, dtype={dtype}"
    )
    return model
