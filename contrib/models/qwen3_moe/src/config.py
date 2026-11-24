# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-MoE Config
================

<-- MODEL-SPECIFIC: Qwen3-30B-A3B defaults.
48 layers, 32 Q heads / 4 KV heads, head_dim=128, hidden=2048.
Each layer has a sparse MoE block: 128 experts, top-8, expert intermediate 768.
The attention, the norms and the RoPE are the same as Qwen3 dense, so the
config extends the ``Qwen3Config`` of the vLLM Neuron plugin
(``vllm_neuron.model.qwen3.config``) with the MoE fields.
"""

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.model.qwen3.config import Qwen3Config


@dataclass
class Qwen3MoeConfig(Qwen3Config):
    # <-- MODEL-SPECIFIC: Qwen3-30B-A3B architecture parameters
    vocab_size: int = 151936
    hidden_size: int = 2048
    # The dense MLP size. Qwen3-30B-A3B has no dense MLP layer, so the model
    # does not use this value.
    intermediate_size: int = 6144
    num_hidden_layers: int = 48
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    max_position_embeddings: int = 40960

    # <-- MODEL-SPECIFIC: MoE parameters
    num_experts: int = 128
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 768
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    mlp_only_layers: list[int] = field(default_factory=list)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig | dict | str,
        neuron_config: NeuronConfig = None,
    ):
        """Build the config from an HF config object, a dict or a config.json path.

        Three keys move before the ``Qwen3Config`` conversion:

        - ``num_experts``: transformers 5 stores it as ``num_local_experts``
          (``attribute_map``), so ``to_dict()`` lacks it. Without the copy,
          every checkpoint would silently get the 128-expert default.
        - ``rope_theta``: transformers 5 keeps it in ``rope_parameters``, and
          ``Qwen3Config`` reads the top-level key. Without the copy, a
          checkpoint with another theta would silently get the 1e6 default.
        - ``torch_dtype``: the implementation computes in BF16. The factory
          has already refused a checkpoint stored in another dtype.

        Example:
            >>> cfg = Qwen3MoeConfig.from_configs(
            ...     {"num_experts": 8, "rope_parameters": {"rope_theta": 5e5}}
            ... )
            >>> cfg.num_experts, cfg.rope_theta, cfg.torch_dtype
            (8, 500000.0, torch.bfloat16)
        """
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            config_dict = hf_config.to_dict()
            # transformers 5 stores some fields under another name and maps
            # the old name with ``attribute_map``, so ``to_dict()`` lacks the
            # field. Read each missing field through the attribute instead.
            for name in cls.__dataclass_fields__:
                if name in ("neuron_config", "torch_dtype"):
                    continue
                if config_dict.get(name) is None:
                    value = getattr(hf_config, name, None)
                    if value is not None:
                        config_dict[name] = value
        else:
            config_dict = dict(hf_config)
        if config_dict.get("num_experts") is None and config_dict.get(
            "num_local_experts"
        ):
            # A ``to_dict()`` of the transformers 5 config.
            config_dict["num_experts"] = config_dict["num_local_experts"]

        rope_parameters = config_dict.get("rope_parameters") or {}
        if config_dict.get("rope_theta") is None and "rope_theta" in rope_parameters:
            config_dict["rope_theta"] = rope_parameters["rope_theta"]
        config_dict["torch_dtype"] = torch.bfloat16
        # A key that is present with the value None (transformers 5 writes
        # ``rope_theta: None``, for example) keeps the default. A None head_dim
        # stays: ``Qwen3Config`` derives it from the hidden size.
        config_dict = {
            k: v for k, v in config_dict.items() if v is not None or k == "head_dim"
        }

        config = super().from_configs(config_dict, neuron_config)
        if config.mlp_only_layers is None:
            config.mlp_only_layers = []
        return config
