# SPDX-License-Identifier: Apache-2.0
"""
Jina Embeddings v5 (EuroBERT encoder) config
============================================

<-- MODEL-SPECIFIC: Map the EuroBERT encoder fields onto
``JinaEmbeddingsV5Config``. The EuroBERT layer uses the same parameter set as a
Llama layer: RMSNorm, rotate-half RoPE, separate Q/K/V/O projections without
bias, and a SiLU gated MLP. The attention is bidirectional, and the model has
no LM head.

``JinaEmbeddingsV5Config`` holds the subset of the plugin's Llama config
fields that the encoder layers read, so that this folder does not import
another model's package.
"""

from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class JinaEmbeddingsV5Config:
    """Architecture fields of the EuroBERT encoder.

    The defaults are the values of ``jinaai/jina-embeddings-v5-text-nano``.
    Build the config with :func:`jina_v5_config_from_hf`.
    """

    vocab_size: int = 128256
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    head_dim: int | None = 64
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    rope_parameters: dict = field(
        default_factory=lambda: {"rope_type": "default", "rope_theta": 1000000.0}
    )
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16

    # Framework config
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


def _config_dict(hf_config: PretrainedConfig | dict) -> dict[str, Any]:
    if isinstance(hf_config, PretrainedConfig):
        config_dict = hf_config.to_dict()
        # to_dict() can drop attributes that the config class sets at runtime.
        for key in ("rope_parameters", "rope_scaling", "rope_theta", "head_dim"):
            if key not in config_dict and getattr(hf_config, key, None) is not None:
                config_dict[key] = getattr(hf_config, key)
        return config_dict
    return dict(hf_config)


def normalize_rope_parameters(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Return the ``rope_parameters`` of a EuroBERT config.

    EuroBERT uses standard rotate-half RoPE. A ``rope_scaling`` of ``None``, of
    type ``default``, or of type ``linear`` with factor 1.0 gives the same
    frequencies. Any other scaling is not implemented.

    Args:
        config_dict: The HF config as a dict.

    Returns:
        A dict with ``rope_type`` set to ``"default"`` and ``rope_theta``.

    Raises:
        NotImplementedError: The config has a RoPE scaling that changes the
            frequencies.

    Example:
        >>> normalize_rope_parameters({"rope_theta": 1e6, "rope_scaling": None})
        {'rope_type': 'default', 'rope_theta': 1000000.0}
    """
    rope = config_dict.get("rope_parameters") or config_dict.get("rope_scaling") or {}
    rope = dict(rope)
    rope_type = rope.get("rope_type", rope.get("type")) or "default"
    factor = float(rope.get("factor", 1.0) or 1.0)
    if rope_type == "linear" and factor == 1.0:
        rope_type = "default"
    if rope_type != "default":
        raise NotImplementedError(
            f"RoPE type {rope_type!r} with factor {factor} is not supported for "
            "Jina Embeddings v5. Use a checkpoint with default RoPE."
        )
    rope_theta = rope.get("rope_theta", config_dict.get("rope_theta"))
    if rope_theta is None:
        raise ValueError(
            "The config has no rope_theta value. Add rope_theta to config.json."
        )
    return {"rope_type": "default", "rope_theta": float(rope_theta)}


def _resolve_dtype(config_dict: dict[str, Any], dtype: torch.dtype | None):
    if dtype is not None:
        return dtype
    try:
        from vllm.config import get_current_vllm_config

        model_config = get_current_vllm_config().model_config
    except Exception:  # noqa: BLE001 - no vLLM config outside a server
        model_config = None
    if model_config is not None and isinstance(
        getattr(model_config, "dtype", None), torch.dtype
    ):
        # The checkpoint config states float32, but the weights are BF16. Use
        # the dtype that vLLM resolved from --dtype.
        return model_config.dtype
    value = config_dict.get("torch_dtype") or config_dict.get("dtype")
    if isinstance(value, str):
        value = getattr(torch, value)
    return value if isinstance(value, torch.dtype) else torch.bfloat16


def jina_v5_config_from_hf(
    hf_config: PretrainedConfig | dict,
    neuron_config: NeuronConfig | None = None,
    dtype: torch.dtype | None = None,
) -> JinaEmbeddingsV5Config:
    """Build a ``JinaEmbeddingsV5Config`` for the Jina Embeddings v5 encoder.

    Args:
        hf_config: The HF ``JinaEmbeddingsV5Config`` or its dict.
        neuron_config: The Neuron framework config.
        dtype: The parameter dtype. ``None`` uses the vLLM model dtype, then
            the dtype in the HF config.

    Returns:
        The ``JinaEmbeddingsV5Config`` for the encoder layers.

    Example:
        >>> cfg = jina_v5_config_from_hf(hf_config, None, torch.bfloat16)
        >>> cfg.num_hidden_layers
        12
    """
    config_dict = _config_dict(hf_config)
    num_heads = config_dict["num_attention_heads"]
    hidden_size = config_dict["hidden_size"]
    head_dim = config_dict.get("head_dim") or hidden_size // num_heads
    return JinaEmbeddingsV5Config(
        vocab_size=config_dict["vocab_size"],
        hidden_size=hidden_size,
        intermediate_size=config_dict["intermediate_size"],
        num_hidden_layers=config_dict["num_hidden_layers"],
        num_attention_heads=num_heads,
        num_key_value_heads=config_dict.get("num_key_value_heads") or num_heads,
        head_dim=head_dim,
        max_position_embeddings=config_dict["max_position_embeddings"],
        rms_norm_eps=config_dict["rms_norm_eps"],
        rope_parameters=normalize_rope_parameters(config_dict),
        tie_word_embeddings=False,
        torch_dtype=_resolve_dtype(config_dict, dtype),
        neuron_config=neuron_config,
    )
