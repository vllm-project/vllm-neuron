# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-VL Config
======================

Nested multimodal config: top-level Qwen3VLConfig composes separate
Qwen3VLTextConfig and Qwen3VLVisionConfig, each with its own NeuronConfig.

Fields are derived from the HuggingFace Qwen3-VL config.json which has
nested text_config and vision_config sub-objects.
"""

import json
from dataclasses import dataclass, field, fields

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig, EncoderNeuronConfig


def _from_hf_sub_config(cls, hf_sub_config, neuron_config=None):
    """Shared factory logic for building a sub-config from an HF config sub-object.

    Filters the HF config dict to only fields defined on the target dataclass,
    coerces torch_dtype strings, and attaches the neuron_config.
    """
    if isinstance(hf_sub_config, PretrainedConfig):
        config_dict = hf_sub_config.to_dict()
        if (
            hasattr(hf_sub_config, "torch_dtype")
            and hf_sub_config.torch_dtype is not None
        ):
            config_dict["torch_dtype"] = hf_sub_config.torch_dtype
    elif isinstance(hf_sub_config, dict):
        config_dict = hf_sub_config
    else:
        raise TypeError(f"Unsupported config type: {type(hf_sub_config)}")

    field_names = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in config_dict.items() if k in field_names}

    # HF config.json uses "dtype" but our dataclass uses "torch_dtype"
    if (
        "torch_dtype" not in filtered
        and "dtype" in config_dict
        and "torch_dtype" in field_names
    ):
        filtered["torch_dtype"] = config_dict["dtype"]

    if "torch_dtype" in filtered and isinstance(filtered["torch_dtype"], str):
        filtered["torch_dtype"] = getattr(torch, filtered["torch_dtype"])

    if neuron_config is not None:
        filtered["neuron_config"] = neuron_config

    return cls(**filtered)


@dataclass
class Qwen3VLTextConfig:
    """Text decoder config extracted from hf_config.text_config.

    Fields match the HuggingFace Qwen3VLTextConfig. Defaults are from
    the Qwen3-VL-32B checkpoint.
    """

    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    head_dim: int = 128
    hidden_act: str = "silu"
    hidden_size: int = 5120
    intermediate_size: int = 25600
    max_position_embeddings: int = 262144
    num_attention_heads: int = 64
    num_hidden_layers: int = 64
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-6
    rope_parameters: dict = field(
        default_factory=lambda: {
            "rope_type": "default",
            "rope_theta": 5000000.0,
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
        }
    )
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16
    vocab_size: int = 151936

    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

    @classmethod
    def from_hf_config(cls, hf_text_config, neuron_config: NeuronConfig = None):
        """Build text config from the text_config sub-object of HF config."""
        return _from_hf_sub_config(cls, hf_text_config, neuron_config)


@dataclass
class Qwen3VLVisionConfig:
    """Vision encoder config extracted from hf_config.vision_config.

    Fields match the HuggingFace Qwen3VLVisionConfig. Defaults are from
    the Qwen3-VL-32B checkpoint.
    """

    deepstack_visual_indexes: list[int] | None = None
    depth: int = 27
    hidden_act: str = "gelu_pytorch_tanh"
    hidden_size: int = 1152
    in_channels: int = 3
    intermediate_size: int = 4304
    num_heads: int = 16
    num_position_embeddings: int = 2304
    out_hidden_size: int = 5120
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2

    neuron_config: EncoderNeuronConfig | None = None

    def __post_init__(self):
        if self.deepstack_visual_indexes is None:
            self.deepstack_visual_indexes = [8, 16, 24]

    @classmethod
    def from_hf_config(
        cls, hf_vision_config, neuron_config: EncoderNeuronConfig = None
    ):
        """Build vision config from the vision_config sub-object of HF config."""
        return _from_hf_sub_config(cls, hf_vision_config, neuron_config)


@dataclass
class Qwen3VLConfig:
    """Top-level multimodal config composing text and vision sub-configs.

    Each sub-config carries its own NeuronConfig because the text decoder
    and vision encoder may need different parallelism / compilation settings.
    """

    text_config: Qwen3VLTextConfig | None = None
    vision_config: Qwen3VLVisionConfig | None = None

    # Top-level fields from HF config
    image_token_id: int = 151655
    tie_word_embeddings: bool = False
    video_token_id: int = 151656
    vision_end_token_id: int = 151653
    vision_start_token_id: int = 151652

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig | dict | str,
        text_neuron_config: NeuronConfig = None,
        encoder_neuron_config: EncoderNeuronConfig = None,
    ):
        """Top-level factory: decompose HF config and build nested vllm-neuron config.

        Args:
            hf_config: HuggingFace PretrainedConfig (or path/dict) with nested
                text_config and vision_config.
            text_neuron_config: NeuronConfig for the text decoder.
            encoder_neuron_config: NeuronConfig for the vision encoder.
        """
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
            hf_text = config_dict["text_config"]
            hf_vision = config_dict["vision_config"]
            top_level = config_dict
        elif isinstance(hf_config, PretrainedConfig):
            hf_text = hf_config.text_config
            hf_vision = hf_config.vision_config
            top_level = hf_config.to_dict()
        elif isinstance(hf_config, dict):
            hf_text = hf_config["text_config"]
            hf_vision = hf_config["vision_config"]
            top_level = hf_config
        else:
            raise TypeError(f"Unsupported hf_config type: {type(hf_config)}")

        text_config = Qwen3VLTextConfig.from_hf_config(hf_text, text_neuron_config)
        vision_config = Qwen3VLVisionConfig.from_hf_config(
            hf_vision, encoder_neuron_config
        )

        # tie_word_embeddings lives at the top level in HF config.json,
        # propagate it to the text sub-config so weight loading can find it.
        tie_word_embeddings = top_level.get("tie_word_embeddings", False)
        text_config.tie_word_embeddings = tie_word_embeddings

        return cls(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=top_level.get("image_token_id", 151655),
            tie_word_embeddings=tie_word_embeddings,
            video_token_id=top_level.get("video_token_id", 151656),
            vision_end_token_id=top_level.get("vision_end_token_id", 151653),
            vision_start_token_id=top_level.get("vision_start_token_id", 151652),
        )
