# SPDX-License-Identifier: Apache-2.0
"""
Nanbeige Config
===============

<-- MODEL-SPECIFIC: All fields in this config are Nanbeige-specific.

Nanbeige4.2 is a Llama-style dense decoder with one addition: the model runs
the full stack of physical decoder layers ``num_loops`` times. The final
RMSNorm runs after each loop when ``skip_loop_final_norm`` is False.
"""

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.config_utils import hoist_rope_parameters
from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class NanbeigeConfig:
    # <-- MODEL-SPECIFIC: Architecture parameters (Nanbeige4.2-3B defaults)
    vocab_size: int = 166144
    hidden_size: int = 3072
    intermediate_size: int = 10752
    # Number of physical decoder layers. Each loop runs all of them.
    num_hidden_layers: int = 22
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-5
    rope_theta: float = 70000000.0
    rope_scaling: dict | None = None
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16

    # <-- MODEL-SPECIFIC: Loop parameters
    num_loops: int = 1
    skip_loop_final_norm: bool = False
    loop_loss_weights: list = field(default_factory=list)

    # Framework config
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_loops is None or self.num_loops < 1:
            self.num_loops = 1

    @property
    def num_logical_layers(self) -> int:
        """Return the number of KV cache slots (physical layers x loops)."""
        return self.num_hidden_layers * self.num_loops

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

        # Transformers 5 writes "dtype" in place of "torch_dtype".
        if config_dict.get("torch_dtype") is None and config_dict.get("dtype"):
            config_dict["torch_dtype"] = config_dict["dtype"]

        # The HF remote code can expand num_hidden_layers to physical x loops
        # for its own KV slots. It then records the physical depth in
        # num_physical_layers. Use the physical depth here.
        if config_dict.get("num_physical_layers"):
            config_dict["num_hidden_layers"] = config_dict["num_physical_layers"]

        config_dict = hoist_rope_parameters(config_dict)
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

        if filtered_dict.get("loop_loss_weights") is None:
            filtered_dict.pop("loop_loss_weights", None)

        if "torch_dtype" in filtered_dict and isinstance(
            filtered_dict["torch_dtype"], str
        ):
            filtered_dict["torch_dtype"] = getattr(torch, filtered_dict["torch_dtype"])

        if neuron_config is not None:
            filtered_dict["neuron_config"] = neuron_config

        return cls(**filtered_dict)
