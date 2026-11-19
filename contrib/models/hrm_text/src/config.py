# SPDX-License-Identifier: Apache-2.0
"""
HRM-Text Config
===============

<-- MODEL-SPECIFIC: All fields in this config are HRM-Text-specific.

HRM-Text has two transformer stacks (H and L) of ``num_layers_per_stack``
layers. The recurrence runs ``H_cycles * (L_cycles + 1)`` stack passes. Each
pass of a layer writes a different KV cache slot, so the model has
``num_layers_per_stack * H_cycles * (L_cycles + 1)`` KV slots.
"""

import json
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

from .compat import hoist_rope_parameters


@dataclass
class HrmTextConfig:
    # <-- MODEL-SPECIFIC: Architecture parameters
    vocab_size: int = 65536
    hidden_size: int = 1536
    intermediate_size: int = 4096
    num_layers_per_stack: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    head_dim: int = 128
    H_cycles: int = 2
    L_cycles: int = 3
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    embedding_scale: float = 1.0
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    # The checkpoint config has no torch_dtype. The checkpoint tensors are BF16.
    torch_dtype: torch.dtype = torch.bfloat16

    # Framework config
    neuron_config: NeuronConfig | None = None

    @property
    def stack_passes(self) -> int:
        """Number of stack passes in one forward: H_cycles * (L_cycles + 1)."""
        return self.H_cycles * (self.L_cycles + 1)

    @property
    def num_kv_slots(self) -> int:
        """Number of KV cache slots: one slot for each layer in each stack pass."""
        return self.num_layers_per_stack * self.stack_passes

    def cycle_offset(self, high_cycle_idx: int, low_cycle_idx: int) -> int:
        """KV slot offset of a stack pass.

        ``low_cycle_idx == L_cycles`` selects the H pass of the high cycle.
        This is the same slot rule as the HF reference.

        Example::

            >>> HrmTextConfig().cycle_offset(1, 3)
            112
        """
        return (
            high_cycle_idx * (self.L_cycles + 1) + low_cycle_idx
        ) * self.num_layers_per_stack

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
            config_dict = dict(hf_config)

        # The raw config.json gives the per-stack layer count in
        # num_hidden_layers. The HF config class moves it to
        # num_layers_per_stack and sets num_hidden_layers to the slot count.
        if config_dict.get("num_layers_per_stack") is None:
            config_dict["num_layers_per_stack"] = config_dict.get(
                "num_hidden_layers", cls.num_layers_per_stack
            )

        config_dict = hoist_rope_parameters(config_dict)
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {
            k: v for k, v in config_dict.items() if k in field_names and v is not None
        }

        if "torch_dtype" not in filtered_dict and config_dict.get("dtype") is not None:
            filtered_dict["torch_dtype"] = config_dict["dtype"]
        if "torch_dtype" in filtered_dict and isinstance(
            filtered_dict["torch_dtype"], str
        ):
            filtered_dict["torch_dtype"] = getattr(torch, filtered_dict["torch_dtype"])

        if neuron_config is not None:
            filtered_dict["neuron_config"] = neuron_config

        return cls(**filtered_dict)
