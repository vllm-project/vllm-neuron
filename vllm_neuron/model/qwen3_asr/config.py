# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-ASR Config
======================

Nested multimodal config: top-level Qwen3ASRConfig composes separate
Qwen3ASRTextConfig and Qwen3ASRAudioConfig, each with its own NeuronConfig.

"""

import json
import os
from dataclasses import dataclass, fields

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.config_utils import hoist_rope_parameters
from vllm_neuron.model.neuron_config import EncoderNeuronConfig, NeuronConfig
from vllm_neuron.model.qwen3_asr.utils.audio_preprocessing import (
    _ATTN_SEQ_ALIGN,
    compute_window_tokens,
    per_chunk_post_cnn_len,
)


def _get(obj, key, default=None):
    """Read a field from an HF config that may be a dict or a PretrainedConfig."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


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

    config_dict = hoist_rope_parameters(config_dict)
    field_names = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in config_dict.items() if k in field_names}

    # HF config.json uses "dtype" but our dataclass uses "torch_dtype".
    # A null dtype must not overwrite the dataclass default. The Qwen3-ASR
    # text_config and audio_config set dtype=null; only thinker_config sets
    # bfloat16. Keep the bfloat16 default in that case.
    if (
        "torch_dtype" not in filtered
        and config_dict.get("dtype") is not None
        and "torch_dtype" in field_names
    ):
        filtered["torch_dtype"] = config_dict["dtype"]

    if "torch_dtype" in filtered and isinstance(filtered["torch_dtype"], str):
        filtered["torch_dtype"] = getattr(torch, filtered["torch_dtype"])

    if neuron_config is not None:
        filtered["neuron_config"] = neuron_config

    return cls(**filtered)


@dataclass
class Qwen3ASRTextConfig:
    """Text decoder config extracted from thinker_config.text_config.

    Fields match the HuggingFace Qwen3ASRTextConfig. Defaults are from
    the Qwen3-ASR-1.7B checkpoint.
    """

    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = None
    eos_token_id: int = None
    head_dim: int = 128
    hidden_act: str = "silu"
    hidden_size: int = 2048
    initializer_range: float = 0.02
    intermediate_size: int = 6144
    max_position_embeddings: int = 65536
    model_type: str = "qwen3"
    num_attention_heads: int = 16
    num_hidden_layers: int = 28
    num_key_value_heads: int = 8
    pad_token_id: int = None
    rms_norm_eps: float = 1e-06
    rope_theta: float = 1000000
    rope_scaling: dict | None = None
    tie_word_embeddings: bool = True
    use_cache: bool = True
    vocab_size: int = 151936
    torch_dtype: torch.dtype = torch.bfloat16

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
class Qwen3ASRAudioConfig:
    """Audio encoder config extracted from thinker_config.audio_config.

    Fields match the HuggingFace Qwen3ASRAudioEncoderConfig. Defaults are
    from the Qwen3-ASR-1.7B checkpoint.
    """

    activation_dropout: float = 0.0
    activation_function: str = "gelu"
    attention_dropout: float = 0.0
    conv_chunksize: int = 500
    d_model: int = 1024
    downsample_hidden_size: int = 480
    dropout: float = 0.0
    encoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    encoder_layers: int = 24
    head_dim: int | None = (
        None  # derived in __post_init__: d_model // encoder_attention_heads
    )
    initializer_range: float = 0.02
    # Sinusoid position-embedding length (per-chunk post-CNN time steps)
    max_position_embeddings: int | None = None
    # Post-CNN tokens in one full inference window (104 for Qwen3-ASR-1.7B)
    window_tokens: int | None = None  # derived in __post_init__
    max_source_positions: int = 1500
    model_type: str = "qwen3_asr_audio_encoder"
    n_window: int = 50
    n_window_infer: int = 800
    num_mel_bins: int = 128
    output_dim: int = 2048
    scale_embedding: bool = False
    torch_dtype: torch.dtype = torch.bfloat16

    # The 2048 default encoder_attention_block_size is a poor fit for audio
    # (see __post_init__). Set an audio value through encoder_neuron_config.
    neuron_config: EncoderNeuronConfig | None = None

    def __post_init__(self):
        # head_dim is not an independent knob: HF Qwen3ASRAudioAttention has no
        # separate head_dim projection, so it must satisfy
        # head_dim * num_heads == d_model (= 64 * 16 = 1024).
        if self.head_dim is None:
            self.head_dim = self.d_model // self.encoder_attention_heads
        if self.head_dim * self.encoder_attention_heads != self.d_model:
            raise ValueError(
                f"head_dim ({self.head_dim}) * encoder_attention_heads "
                f"({self.encoder_attention_heads}) must equal d_model "
                f"({self.d_model}); the audio encoder has no separate head_dim "
                f"projection."
            )

        # Sinusoid length is fully determined by n_window (per-chunk post-CNN
        # time steps): three (k=3, s=2, p=1) convs over 2*n_window mel frames
        # (=> 13 for n_window=50). It is not a free parameter — the sinusoid
        # position-embedding buffer is sized to it.
        length = per_chunk_post_cnn_len(self.n_window)
        if self.max_position_embeddings is None:
            self.max_position_embeddings = length
        elif self.max_position_embeddings != length:
            raise ValueError(
                f"max_position_embeddings ({self.max_position_embeddings}) must "
                f"equal the per-chunk post-CNN length derived from n_window "
                f"({length}); the sinusoid buffer is sized to it."
            )

        num_mel_frames_per_chunk = self.n_window * 2
        if self.n_window_infer % num_mel_frames_per_chunk != 0:
            raise ValueError(
                f"n_window_infer ({self.n_window_infer}) must be a multiple of "
                f"num_mel_frames_per_chunk ({num_mel_frames_per_chunk} = 2 * "
                f"n_window); an inference window must hold a whole number of "
                f"base chunks."
            )

        # Post-CNN tokens in one full inference window
        window_tokens = compute_window_tokens(
            self.max_position_embeddings, self.n_window, self.n_window_infer
        )
        if self.window_tokens is None:
            self.window_tokens = window_tokens
        elif self.window_tokens != window_tokens:
            raise ValueError(
                f"window_tokens ({self.window_tokens}) must equal the value "
                f"derived from max_position_embeddings, n_window and "
                f"n_window_infer ({window_tokens})."
            )

        # Validate attention block size
        if self.neuron_config is not None:
            block_size = self.neuron_config.encoder_attention_block_size
            if block_size % _ATTN_SEQ_ALIGN != 0:
                raise ValueError(
                    f"encoder_attention_block_size ({block_size}) must be a "
                    f"multiple of {_ATTN_SEQ_ALIGN}; the attention kernel reads "
                    f"the bounds with a per-block stride of "
                    f"{_ATTN_SEQ_ALIGN} * ceil(block_size / {_ATTN_SEQ_ALIGN})."
                )
            # Attention is block-local, so a block must hold at least one whole window
            if block_size < self.window_tokens:
                raise ValueError(
                    f"encoder_attention_block_size ({block_size}) must be at "
                    f"least window_tokens ({self.window_tokens}); one inference "
                    f"window cannot span two attention blocks."
                )

    @classmethod
    def from_hf_config(cls, hf_audio_config, neuron_config: EncoderNeuronConfig = None):
        """Build audio config from the audio_config sub-object of HF config."""
        return _from_hf_sub_config(cls, hf_audio_config, neuron_config)


@dataclass
class Qwen3ASRConfig:
    """Top-level multimodal config composing text and audio sub-configs.

    The HF config.json nests audio_config and text_config under
    `thinker_config`. from_configs unwraps it.

    Each sub-config carries its own NeuronConfig because the text decoder and
    audio encoder may need different parallelism / compilation settings.
    """

    text_config: Qwen3ASRTextConfig | None = None
    audio_config: Qwen3ASRAudioConfig | None = None

    # Token IDs from thinker_config
    audio_token_id: int = 151676
    audio_start_token_id: int = 151669
    audio_end_token_id: int = 151670
    tie_word_embeddings: bool = True

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig | dict | str,
        text_neuron_config: NeuronConfig = None,
        audio_neuron_config: EncoderNeuronConfig = None,
    ):
        """Top-level factory: decompose HF config and build nested vllm-neuron config.

        Args:
            hf_config: HuggingFace PretrainedConfig (or path/dict) with a
                `thinker_config` holding nested text_config and audio_config.
            text_neuron_config: NeuronConfig for the text decoder.
            audio_neuron_config: NeuronConfig for the audio encoder.
        """
        if isinstance(hf_config, (str, bytes)):
            # Accept either a config.json file or a checkpoint directory (like
            # AutoConfig.from_pretrained). Callers commonly pass the checkpoint
            # dir returned by get_model_checkpoint.
            path = os.fspath(hf_config)
            if os.path.isdir(path):
                path = os.path.join(path, "config.json")
            with open(path) as f:
                hf_config = json.load(f)
        if not isinstance(hf_config, (dict, PretrainedConfig)):
            raise TypeError(f"Unsupported hf_config type: {type(hf_config)}")

        thinker = _get(hf_config, "thinker_config")
        if thinker is None:
            raise ValueError("Qwen3-ASR config is missing thinker_config")

        hf_text = _get(thinker, "text_config")
        hf_audio = _get(thinker, "audio_config")

        text_config = Qwen3ASRTextConfig.from_hf_config(hf_text, text_neuron_config)
        audio_config = Qwen3ASRAudioConfig.from_hf_config(hf_audio, audio_neuron_config)

        # tie_word_embeddings lives on the thinker text_config; propagate it to
        # the text sub-config so weight loading can find it.
        tie_word_embeddings = _get(hf_text, "tie_word_embeddings", True)
        text_config.tie_word_embeddings = tie_word_embeddings

        return cls(
            text_config=text_config,
            audio_config=audio_config,
            audio_token_id=_get(thinker, "audio_token_id", 151676),
            audio_start_token_id=_get(thinker, "audio_start_token_id", 151669),
            audio_end_token_id=_get(thinker, "audio_end_token_id", 151670),
            tie_word_embeddings=tie_word_embeddings,
        )
