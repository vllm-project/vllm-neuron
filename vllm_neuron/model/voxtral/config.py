# SPDX-License-Identifier: Apache-2.0
"""Voxtral config.

Voxtral composes a Whisper-style audio encoder, a 2-layer audio adapter, and a
Llama text decoder. The plugin reuses ``LlamaConfig`` for the decoder.

vLLM builds the HF config in one of two formats, with different audio field
names:

- ``config_format="hf"`` (``config.json``): ``hidden_size``, ``num_hidden_layers``,
  ``num_attention_heads``, ``intermediate_size``.
- ``config_format="mistral"`` (``params.json``, the default for Mistral repos):
  ``d_model``, ``encoder_layers``, ``encoder_attention_heads``,
  ``encoder_ffn_dim``, ``downsample_factor``, ``hop_length``, ``window_size``,
  ``global_log_mel_max``.

``VoxtralAudioConfig.from_hf_config`` reads both formats.
"""

from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.llama3.config import LlamaConfig
from vllm_neuron.model.neuron_config import EncoderNeuronConfig, NeuronConfig

# One 30 s window: 3000 mel frames -> 1500 encoder frames -> 375 audio tokens.
SAMPLING_RATE = 16000
CHUNK_LENGTH_S = 30


def _get(obj, *names, default=None):
    """Return the first attribute in ``names`` that ``obj`` sets to a value."""
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if value is not None:
            return value
    return default


@dataclass
class VoxtralAudioConfig:
    """Audio encoder and adapter config. Defaults are Voxtral-Mini-3B-2507."""

    d_model: int = 1280
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    num_mel_bins: int = 128
    max_source_positions: int = 1500
    downsample_factor: int = 4
    hop_length: int = 160
    window_size: int = 400
    sampling_rate: int = SAMPLING_RATE
    global_log_mel_max: float | None = None
    text_hidden_size: int = 3072
    torch_dtype: torch.dtype = torch.bfloat16
    neuron_config: EncoderNeuronConfig | None = None

    def __post_init__(self):
        if self.d_model % self.encoder_attention_heads:
            raise ValueError(
                f"d_model ({self.d_model}) is not a multiple of "
                f"encoder_attention_heads ({self.encoder_attention_heads})."
            )
        self.head_dim = self.d_model // self.encoder_attention_heads
        if self.max_source_positions % self.downsample_factor:
            raise ValueError(
                f"max_source_positions ({self.max_source_positions}) is not a multiple "
                f"of downsample_factor ({self.downsample_factor})."
            )
        # Audio tokens that one 30 s window gives (375 for Voxtral-Mini).
        self.tokens_per_window = self.max_source_positions // self.downsample_factor
        # Mel frames in one 30 s window (2 x max_source_positions, conv2 stride 2).
        self.mel_frames_per_window = 2 * self.max_source_positions
        self.samples_per_window = CHUNK_LENGTH_S * self.sampling_rate

    @classmethod
    def from_hf_config(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: EncoderNeuronConfig | None = None,
    ) -> "VoxtralAudioConfig":
        audio = hf_config.audio_config
        text = hf_config.text_config
        d_model = _get(audio, "d_model", "hidden_size")
        ffn = _get(audio, "encoder_ffn_dim", "intermediate_size")
        # The HF format has no downsample_factor. The adapter input width is
        # downsample_factor * d_model, and Voxtral sets it equal to the encoder
        # FFN width.
        downsample = _get(audio, "downsample_factor", default=ffn // d_model)
        dtype = _get(hf_config, "torch_dtype", "dtype", default=torch.bfloat16)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        return cls(
            d_model=d_model,
            encoder_layers=_get(audio, "encoder_layers", "num_hidden_layers"),
            encoder_attention_heads=_get(
                audio, "encoder_attention_heads", "num_attention_heads"
            ),
            encoder_ffn_dim=ffn,
            num_mel_bins=_get(audio, "num_mel_bins", default=128),
            max_source_positions=_get(audio, "max_source_positions", default=1500),
            downsample_factor=downsample,
            hop_length=_get(audio, "hop_length", default=160),
            window_size=_get(audio, "window_size", default=400),
            sampling_rate=_get(audio, "sampling_rate", default=SAMPLING_RATE),
            global_log_mel_max=_get(audio, "global_log_mel_max"),
            text_hidden_size=text.hidden_size,
            torch_dtype=dtype,
            neuron_config=neuron_config,
        )


def _model_dtype(hf_config: PretrainedConfig) -> torch.dtype | None:
    """Return the top-level torch dtype of ``hf_config``, or None if it is unset.

    Example:
        >>> _model_dtype(PretrainedConfig(dtype="bfloat16"))
        torch.bfloat16
    """
    dtype = _get(hf_config, "torch_dtype", "dtype")
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    return dtype if isinstance(dtype, torch.dtype) else None


@dataclass
class VoxtralConfig:
    """Top-level Voxtral config: Llama text decoder + audio encoder."""

    text_config: LlamaConfig
    audio_config: VoxtralAudioConfig

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config: EncoderNeuronConfig | None = None,
    ) -> "VoxtralConfig":
        text_config = LlamaConfig.from_configs(
            hf_config.text_config, text_neuron_config
        )
        # Use the model dtype (the top-level config, which --dtype sets) for the
        # decoder too. For a local Mistral-format checkpoint, vLLM infers the
        # text_config dtype from every *.safetensors file in the directory.
        # Voxtral's HF shards hold one fp32 tensor, so text_config says float32
        # while the model runs in bfloat16.
        model_dtype = _model_dtype(hf_config)
        if model_dtype is not None:
            text_config.torch_dtype = model_dtype
        return cls(
            text_config=text_config,
            audio_config=VoxtralAudioConfig.from_hf_config(
                hf_config, encoder_neuron_config
            ),
        )
