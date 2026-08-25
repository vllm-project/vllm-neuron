# SPDX-License-Identifier: Apache-2.0
"""Neuron-side configuration for DFlash Qwen3 draft checkpoints."""

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class DFlashConfig:
    vocab_size: int = 201088
    draft_vocab_size: int | None = None
    hidden_size: int = 2880
    unpadded_hidden_size: int | None = None
    target_hidden_size: int | None = None
    intermediate_size: int = 7680
    num_hidden_layers: int = 8
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 64
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    attention_bias: bool = True
    block_size: int = 8
    dflash_config: dict = field(default_factory=dict)
    # Populated from the checkpoint; no default RoPE flavour, because the
    # drafters differ (GPT-OSS's uses YaRN, Llama 3.1's plain RoPE) and
    # silently inheriting one model's scaling would mis-encode every position
    # for the other.
    rope_parameters: dict = field(default_factory=dict)
    num_target_layers: int | None = None
    torch_dtype: torch.dtype = torch.bfloat16
    neuron_config: NeuronConfig | None = None

    def __post_init__(self) -> None:
        if self.unpadded_hidden_size is None:
            self.unpadded_hidden_size = self.hidden_size
        if self.target_hidden_size is None:
            self.target_hidden_size = self.unpadded_hidden_size
        if self.draft_vocab_size is None:
            self.draft_vocab_size = self.vocab_size

    @property
    def target_layer_ids(self) -> list[int]:
        return list(self.dflash_config.get("target_layer_ids", []))

    @property
    def rope_type(self) -> str:
        """RoPE flavour this checkpoint was trained with.

        A checkpoint with ``rope_scaling: null`` means plain RoPE; transformers
        writes that as an absent or None rope_type.
        """
        return self.rope_parameters.get("rope_type") or "default"

    @property
    def mask_token_id(self) -> int:
        mask_token_id = self.dflash_config.get("mask_token_id")
        if mask_token_id is None:
            raise ValueError(
                "DFlash checkpoint config must define dflash_config.mask_token_id"
            )
        return int(mask_token_id)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig | dict | str,
        neuron_config: NeuronConfig | None = None,
    ) -> "DFlashConfig":
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            config_dict = hf_config.to_dict()
            if getattr(hf_config, "torch_dtype", None) is not None:
                config_dict["torch_dtype"] = hf_config.torch_dtype
        else:
            config_dict = dict(hf_config)

        # Transformers names this field rope_scaling; Neuron model configs use
        # rope_parameters and carry rope_theta inside it.
        rope_parameters = dict(
            config_dict.get("rope_parameters") or config_dict.get("rope_scaling") or {}
        )
        rope_parameters.setdefault(
            "rope_theta", config_dict.get("rope_theta", 150000.0)
        )
        if rope_parameters:
            config_dict["rope_parameters"] = rope_parameters

        field_names = set(cls.__dataclass_fields__)
        filtered = {
            key: value for key, value in config_dict.items() if key in field_names
        }
        dtype = filtered.get("torch_dtype")
        if isinstance(dtype, str):
            filtered["torch_dtype"] = getattr(torch, dtype)
        if neuron_config is not None:
            filtered["neuron_config"] = neuron_config
        return cls(**filtered)
