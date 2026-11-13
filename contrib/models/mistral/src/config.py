# SPDX-License-Identifier: Apache-2.0
"""
Mistral Config
======================

Adapted from ``vllm_neuron/model/llama3/config.py``. Defaults match
Mistral-7B-v0.3 (dense, GQA 32Q/8KV, SiLU, untied embeddings, default RoPE
with rope_theta 1e6, no sliding window).

<-- MODEL-SPECIFIC: Differences from Llama 3.1 8B:
  * vocab_size 32768 (Llama: 128256)
  * rope_theta 1e6 with default RoPE (Llama: 5e5 with "llama3" scaling)
  * max_position_embeddings 32768 (Llama: 131072)
  * Mistral's config.json has no ``head_dim`` key, so ``head_dim`` defaults to
    None and is derived as hidden_size // num_attention_heads. The Llama
    config defaults it to 64, which would be wrong for Mistral on any path
    that does not go through transformers' MistralConfig.
  * ``sliding_window`` is parsed so the factory can reject checkpoints that
    use it (e.g. Mistral-7B-v0.1). v0.3 sets it to null.

Not carried over from the Llama config (this port is BF16 only and is not
used as an Eagle3 draft model): ``quant_spec`` and the Eagle3 draft fields
(``draft_vocab_size``, ``norm_before_residual``, ``norm_before_fc``,
``unpadded_hidden_size``, ``target_hidden_pad_groups``,
``use_draft_mlp_groups``, nested ``transformer_layer_config``).
"""

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class MistralConfig:
    # <-- MODEL-SPECIFIC: Architecture parameters (Mistral-7B-v0.3 defaults)
    vocab_size: int = 32768
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    # None => derived in __post_init__ (see module docstring).
    head_dim: int | None = None
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-5
    rope_parameters: dict = field(
        default_factory=lambda: {"rope_type": "default", "rope_theta": 1000000.0}
    )
    # Mistral v0.3 uses full attention. Non-None values are rejected by the
    # factory because this implementation has no sliding window.
    sliding_window: int | None = None
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16

    # Framework config
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig = None
    ):
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            config_dict = hf_config.to_dict()
            if hasattr(hf_config, "torch_dtype") and hf_config.torch_dtype is not None:
                config_dict["torch_dtype"] = hf_config.torch_dtype
        else:
            config_dict = hf_config

        # BF16 only: fail at construction instead of loading FP8 weights
        # into BF16 parameters.
        if config_dict.get("quantization_config") is not None:
            raise NotImplementedError(
                "Quantized Mistral checkpoints are not supported by the Mistral "
                "contrib port."
            )

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

        # For raw-dict configs, fold top-level rope_theta/rope_scaling into
        # rope_parameters (they'd otherwise be dropped). No-op on the normal
        # AutoConfig path, where transformers has already folded them.
        if "rope_parameters" not in filtered_dict:
            rope_scaling = config_dict.get("rope_scaling")
            rope_theta = config_dict.get("rope_theta")
            if rope_scaling or rope_theta is not None:
                rope_parameters = dict(rope_scaling) if rope_scaling else {}
                if rope_theta is not None:
                    rope_parameters.setdefault("rope_theta", rope_theta)
                rope_parameters.setdefault("rope_type", "default")
                filtered_dict["rope_parameters"] = rope_parameters

        if "torch_dtype" in filtered_dict and isinstance(
            filtered_dict["torch_dtype"], str
        ):
            filtered_dict["torch_dtype"] = getattr(torch, filtered_dict["torch_dtype"])

        if neuron_config is not None:
            filtered_dict["neuron_config"] = neuron_config

        return cls(**filtered_dict)
