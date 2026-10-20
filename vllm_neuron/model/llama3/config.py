# SPDX-License-Identifier: Apache-2.0
"""
Llama Config
======================

<-- MODEL-SPECIFIC: All fields in this config are Llama-specific.
When porting to a new model, update all fields to match the target architecture.
"""

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

from .quantization import QuantizationSpec


@dataclass
class LlamaConfig:
    # <-- MODEL-SPECIFIC: Architecture parameters
    vocab_size: int = 128256
    draft_vocab_size: int | None = None
    norm_before_residual: bool = False
    norm_before_fc: bool = False
    hidden_size: int = 2048
    unpadded_hidden_size: int | None = None
    # Layout of the target's padded hidden dimension, for an Eagle3 draft that
    # consumes target hidden states. 1 means the padding sits at the tail; a
    # larger value means the target deinterleaves the padded axis by that many
    # groups (MXFP4 GPT-OSS uses 4). The proposer sets this from the target
    # model. See ``to_target_hidden_layout`` in ``eagle3_model.py``.
    target_hidden_pad_groups: int = 1
    intermediate_size: int = 8192
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 64
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_parameters: dict = field(
        default_factory=lambda: {"rope_type": "default", "rope_theta": 500000.0}
    )
    tie_word_embeddings: bool = True
    torch_dtype: torch.dtype = torch.bfloat16

    # Framework config
    neuron_config: NeuronConfig | None = None

    # Set on a draft config so its MLP reads the draft MLP groups, not the
    # target's.
    use_draft_mlp_groups: bool = False

    # Quantization spec parsed from the HuggingFace ``quantization_config``
    # (populated by :meth:`from_configs`). ``None`` means "not quantized".
    # Modeling code should query this via
    # ``quant_spec.get_scheme_for_module(prefix)`` to decide weight dtypes
    # and kernel calls on a per-module basis.
    quant_spec: QuantizationSpec | None = field(default=None)

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.unpadded_hidden_size is None:
            self.unpadded_hidden_size = self.hidden_size

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

        # Handle RedHat Eagle3 nested config: model params are inside
        # transformer_layer_config, with draft_vocab_size at the top level
        if (
            "transformer_layer_config" in config_dict
            and "hidden_size" not in filtered_dict
        ):
            tlc = config_dict["transformer_layer_config"]
            if isinstance(tlc, dict):
                for k, v in tlc.items():
                    if k in field_names and k not in filtered_dict:
                        filtered_dict[k] = v
            # Preserve top-level fields that are outside transformer_layer_config
            for key in ("draft_vocab_size", "norm_before_residual", "norm_before_fc"):
                if key in config_dict and key in field_names:
                    filtered_dict[key] = config_dict[key]

        if "torch_dtype" in filtered_dict and isinstance(
            filtered_dict["torch_dtype"], str
        ):
            filtered_dict["torch_dtype"] = getattr(torch, filtered_dict["torch_dtype"])

        if neuron_config is not None:
            filtered_dict["neuron_config"] = neuron_config

        # Parse optional quantization config. Absent => not quantized.
        # Present and recognized => attach a QuantizationSpec.
        # Present but unsupported => raise (surfaced to the user, not silently
        # treated as bf16).
        filtered_dict["quant_spec"] = QuantizationSpec.from_hf_quantization_config(
            config_dict.get("quantization_config")
        )

        return cls(**filtered_dict)
