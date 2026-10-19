# SPDX-License-Identifier: Apache-2.0
"""Generic pooling adapters for neuron ``*ForCausalLM`` models.

Neuron analog of vLLM's ``model_executor/models/adapters.py``: any neuron
generative model becomes a pooling model with zero per-model code, converted at
one seam (``NeuronModelRunner.load_model`` when ``runner_type == "pooling"``).

Conversion works on the built *instance* (``__class__`` swap), not the class. It
drops lm_head and attaches a ``DispatchPooler``; the head is ``delattr``'d after
the base builds it, and the neuron loader iterates ``named_parameters()`` so the
deleted head's mapping is inert.
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)

# Generative class-name suffixes, mirroring upstream ``_GENERATE_SUFFIXES``.
_GENERATE_SUFFIXES = (
    "ForCausalLM",
    "ForConditionalGeneration",
    "ChatModel",
    "LMHeadModel",
)

# (base-class, task-suffix) -> pooling subclass, so repeated conversions reuse
# one dynamically-created type.
_POOLING_CLASS_CACHE: dict[tuple[type, str], type] = {}


def as_neuron_pooling_model(model: nn.Module, convert_type: str = "embed") -> nn.Module:
    """Convert a built neuron ``*ForCausalLM`` instance into a pooling model.

    Retypes ``model`` in place (``__class__`` swap), drops the generative
    head/sampler, and attaches the pooler for ``convert_type``. Returns the same
    object. Call after the instance is built and before ``load_weights``.

    Args:
        model: A built concrete neuron ``*ForCausalLM`` with a ``.model``
            backbone, ``.lm_head``, ``load_weights``, ``get_kv_spec``, and
            ``bind_kv_cache``.
        convert_type: Pooling task (vLLM ``ConvertType``): ``"embed"``
            (implemented) or ``"classify"`` (path only). Defaults to ``"embed"``.

    Returns:
        The same instance, retyped. Returned unchanged if already a pooling model.

    Raises:
        NotImplementedError: ``convert_type="classify"``, or a model with no
            ``.model`` backbone.
        ValueError: unknown ``convert_type``, missing ``pooler_config``, or fp8 KV.

    Example:
        >>> model = SomeForCausalLM.from_configs(hf_config, neuron_config)
        >>> model = as_neuron_pooling_model(model, convert_type="embed")
        >>> model.is_pooling_model
        True
    """
    if getattr(model, "is_pooling_model", False):
        return model

    converter = _TASK_CONVERTERS.get(convert_type)
    if converter is None:
        raise ValueError(
            f"convert_type={convert_type!r} is not a supported pooling task; "
            f"expected one of {sorted(_TASK_CONVERTERS)}."
        )
    orig_name = type(model).__name__
    converted = converter(model)
    logger.info(
        "Converted %s to pooling model (convert_type=%s)", orig_name, convert_type
    )
    return converted


def _pooling_name(orig_name: str, task_suffix: str) -> str:
    """Derive the pooling class name from a generative class name.

    >>> _pooling_name("Qwen3ForCausalLM", "ForEmbedding")
    'Qwen3ForEmbedding'
    """
    for suffix in _GENERATE_SUFFIXES:
        if orig_name.endswith(suffix):
            return orig_name[: -len(suffix)] + task_suffix
    return orig_name + task_suffix


def _make_pooling_subclass(
    base_cls: type[nn.Module], task_suffix: str
) -> type[nn.Module]:
    """Build and cache a pooling subclass of a concrete neuron model class."""
    cache_key = (base_cls, task_suffix)
    cached = _POOLING_CLASS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    class NeuronModelForPooling(base_cls):
        is_pooling_model = True

        @torch.no_grad()
        def forward(
            self,
            input_ids: torch.LongTensor,
            positions: torch.Tensor,
            inputs_embeds: torch.Tensor | None = None,
            is_token_ids: torch.Tensor | None = None,
            attn_metadata: object | None = None,
            rank: torch.Tensor | None = None,
            **kwargs,
        ) -> torch.Tensor:
            # Call the backbone directly, passing only kwargs it declares
            # (_backbone_params is None for a **kwargs backbone -> pass all).
            candidates = {
                "attn_metadata": attn_metadata,
                "rank": rank,
                "inputs_embeds": inputs_embeds,
                "is_token_ids": is_token_ids,
            }
            call_kwargs = (
                candidates
                if self._backbone_params is None
                else {k: v for k, v in candidates.items() if k in self._backbone_params}
            )
            out = self.model(input_ids, positions, **call_kwargs)
            return out[0] if isinstance(out, tuple) else out

    NeuronModelForPooling.__name__ = _pooling_name(base_cls.__name__, task_suffix)
    NeuronModelForPooling.__qualname__ = NeuronModelForPooling.__name__
    _POOLING_CLASS_CACHE[cache_key] = NeuronModelForPooling
    return NeuronModelForPooling


def _retype_and_strip_head(model: nn.Module, task_suffix: str) -> VllmConfig:
    """Retype ``model`` to a pooling subclass, drop the head, reject fp8 KV."""
    if not hasattr(model, "model"):
        raise NotImplementedError(
            f"{type(model).__name__} has no '.model' backbone; generic pooling "
            "conversion supports single-tower text *ForCausalLM only."
        )

    model.__class__ = _make_pooling_subclass(type(model), task_suffix)

    # Drop the generative head.
    for attr in ("lm_head", "sampler"):
        if hasattr(model, attr):
            delattr(model, attr)

    # Record backbone-accepted kwargs for forward() to filter against; None means
    # the backbone takes **kwargs, so pass everything.
    sig = inspect.signature(model.model.forward)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        model._backbone_params = None
    else:
        model._backbone_params = frozenset(sig.parameters)

    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()

    # Pooling is prefill-only and never loads fp8 k/v scales. Checking after the
    # swap is safe: load_model has no except around the seam, so this raise aborts
    # the load and discards the half-converted instance — do not reorder.
    cache_dtype = getattr(vllm_config.cache_config, "cache_dtype", None)
    if (cache_dtype or "").startswith("fp8"):
        raise ValueError(
            f"kv_cache_dtype={cache_dtype!r} is not supported for the pooling "
            "model (prefill-only, no FP8 KV cache); use the default (auto/bf16)."
        )
    return vllm_config


def _as_embed(model: nn.Module) -> nn.Module:
    """Convert to an embedding model: attach ``DispatchPooler.for_embedding``."""
    from vllm.model_executor.layers.pooler import DispatchPooler

    vllm_config = _retype_and_strip_head(model, "ForEmbedding")

    # raise, not assert: asserts are stripped under -O.
    pooler_config = vllm_config.model_config.pooler_config
    if pooler_config is None:
        raise ValueError(
            "pooler_config is None — an embedding model requires the pooling "
            "runner (launch with --runner pooling / --convert embed, or a "
            "checkpoint whose sentence-transformers modules.json resolves to a "
            "pooling model)."
        )
    model.pooler = DispatchPooler.for_embedding(pooler_config)
    return model


def _as_classify(model: nn.Module) -> nn.Module:
    """Convert to a classification model. NOT IMPLEMENTED — path only.

    TODO(pooling-classify): unlike embed, classify needs a model-level ``score``
    head (``hidden_size -> num_labels``, absent from the generative checkpoint)
    and ``DispatchPooler.for_seq_cls(..., classifier=self.score)``.
    """
    raise NotImplementedError(
        "convert_type='classify' is not yet supported on neuron; needs a "
        "model-level score head the generative checkpoint lacks. See "
        "vllm_neuron/model/pooling_adapter.py::_as_classify (TODO)."
    )


# convert_type -> converter.
_TASK_CONVERTERS = {
    "embed": _as_embed,
    "classify": _as_classify,
}
