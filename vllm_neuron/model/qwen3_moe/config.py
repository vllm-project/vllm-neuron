# SPDX-License-Identifier: Apache-2.0
"""
Qwen3 MoE Configuration
================================
"""

import json
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class Qwen3MoeConfig:
    """
    Configuration for Qwen3 MoE model.

    Architecture overview:
    - Transformer decoder with MoE (Mixture of Experts) feed-forward layers
    - GQA (Grouped Query Attention) with 32 Q heads and 4 KV heads
    - Standard RoPE (theta=1,000,000) for position encoding
    - SiLU (SwiGLU) activation in expert feed-forward
    - Pre-attention and pre-MLP RMSNorm
    - No attention sinks, no sliding window, no attention bias
    """

    # ── Model architecture ───────────────────────────────────────────────
    vocab_size: int = 151936
    hidden_size: int = 2048
    num_hidden_layers: int = 48
    num_attention_heads: int = 32  # Q heads
    num_key_value_heads: int = 4  # KV heads
    head_dim: int = 128
    intermediate_size: int = 6144  # Dense MLP intermediate (unused for MoE layers)
    moe_intermediate_size: int = 768  # Per-expert intermediate size
    rms_norm_eps: float = 1e-6
    torch_dtype: torch.dtype = torch.bfloat16

    # ── MoE configuration ────────────────────────────────────────────────
    num_experts: int = 128  # Total number of experts per layer
    num_experts_per_tok: int = 8  # Top-k experts per token
    norm_topk_prob: bool = True  # Normalize routing probabilities

    # ── RoPE settings ────────────────────────────────────────────────────
    # Both spellings are accepted because the key was renamed underneath us.
    # transformers 4.x exposed a top-level ``rope_theta`` plus ``rope_scaling``;
    # transformers 5.x (shipped with vllm-neuron 0.24) folds both into a single
    # ``rope_parameters`` dict and silently redirects writes to ``rope_scaling``
    # into it, so a config built only from ``rope_scaling`` reads back as None.
    # Reading just one spelling therefore fails silently on one of the two
    # versions: YaRN would appear to be enabled and do nothing.
    #
    # __post_init__ normalises whichever arrived into ``rope_scaling``, and
    # lifts ``rope_theta`` out of the dict when that is where it lives.
    #
    # Qwen publishes this for 131,072-token context:
    #   {"rope_type": "yarn", "factor": 4.0,
    #    "original_max_position_embeddings": 32768}
    # beta_fast / beta_slow are not in Qwen's published config; the YaRN paper
    # defaults (32 / 1) are used when absent.
    rope_theta: float = 1000000.0
    rope_scaling: dict | None = None
    rope_parameters: dict | None = None

    # ── Special tokens ───────────────────────────────────────────────────
    pad_token_id: int | None = None
    bos_token_id: int = 151643
    eos_token_id: int = 151645

    # ── Sequence settings ────────────────────────────────────────────────
    max_position_embeddings: int = 40960

    # ── Framework config (not model-specific) ────────────────────────────
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        self._normalise_rope()

    def _normalise_rope(self) -> None:
        """Collapse rope_parameters / rope_scaling into one representation.

        After this runs, ``rope_theta`` holds the base frequency and
        ``rope_scaling`` holds the scaling spec (or None when unscaled), no
        matter which transformers version produced the source config.
        """
        params = self.rope_parameters or self.rope_scaling or {}

        # transformers 5.x keeps the base frequency inside the dict. Only take
        # it from there if present, so an explicitly-passed rope_theta wins on
        # 4.x where the dict carries no base frequency.
        if "rope_theta" in params:
            self.rope_theta = float(params["rope_theta"])

        rope_type = params.get("rope_type") or params.get("type")
        if rope_type in (None, "default"):
            # "default" means plain RoPE; carrying the dict forward would make
            # the model think a scaling method was requested.
            self.rope_scaling = None
        else:
            self.rope_scaling = dict(params)

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        """Create config from HuggingFace config + NeuronConfig."""
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            if (
                hasattr(hf_config, "quantization_config")
                and hf_config.quantization_config is None
            ):
                delattr(hf_config, "quantization_config")
                config_dict = hf_config.to_dict()
                hf_config.quantization_config = None
            else:
                config_dict = hf_config.to_dict()
        else:
            config_dict = hf_config

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

        if "torch_dtype" in filtered_dict and isinstance(
            filtered_dict["torch_dtype"], str
        ):
            filtered_dict["torch_dtype"] = getattr(torch, filtered_dict["torch_dtype"])

        filtered_dict["neuron_config"] = neuron_config

        return cls(**filtered_dict)
