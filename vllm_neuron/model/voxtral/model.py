# SPDX-License-Identifier: Apache-2.0
"""Voxtral: Whisper-style audio encoder + audio adapter + Llama text decoder.

Upstream vLLM owns the Voxtral processor and the prompt format. The processor
gives ``audio_arrays``: one raw 16 kHz waveform for each clip, zero-padded to a
multiple of 30 s. For each 30 s window, ``embed_multimodal`` does these steps:

1. Compute the log-mel spectrogram on the CPU.
2. Run the encoder and the audio adapter on the encoder TP group.
3. Write ``tokens_per_window`` (375) embeddings into the on-device encoder cache.

The Llama decoder then merges the cache rows into the audio placeholder
positions during prefill. Set ``encoder_attention_block_size`` to
``tokens_per_window``, so one cache block holds one window.

The tower is ``audio_tower``, the same name as the HF checkpoint and the other
plugin audio model. The adapter uses the upstream vLLM names
``audio_language_adapter.w_in`` and ``w_out``. Weights load from the HF-format
checkpoint files (``model.safetensors.index.json``).
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.models.voxtral import (
    VoxtralDummyInputsBuilder,
    VoxtralMultiModalProcessor,
    VoxtralProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

import vllm_neuron.functional as NF
from vllm_neuron.model.interfaces import SupportsEncoderBudget
from vllm_neuron.model.llama3.model import LlamaForCausalLM
from vllm_neuron.model.neuron_config import EncoderNeuronConfig, NeuronConfig
from vllm_neuron.model.voxtral.config import VoxtralAudioConfig, VoxtralConfig
from vllm_neuron.parallel.neuron_parallel_state import get_neuron_encoder_tp_group
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    set_weight_loader,
    sharding_weight_loader,
)

# Checkpoint prefixes of the HF-format Voxtral checkpoint.
_ENCODER_PREFIX = "audio_tower"
_ADAPTER_PREFIX = "multi_modal_projector"
_TEXT_PREFIX = "language_model."


def _gelu(x: torch.Tensor) -> torch.Tensor:
    # <-- MODEL-SPECIFIC: tanh-form GELU. The maximum error against the erf form
    # is 4.7e-4, which is less than the bf16 resolution. The erf form lowers to
    # a slow polynomial on Neuron.
    return x * torch.sigmoid(1.5957691216057308 * (x + 0.044715 * x * x * x))


# =============================================================================
# Log-mel spectrogram (CPU)
# =============================================================================


def _slaney_mel_filters(
    num_mel_bins: int, window_size: int, sampling_rate: int
) -> torch.Tensor:
    """Slaney-scale, Slaney-normalized mel filter bank, ``[n_freq, num_mel_bins]``.

    The values are equal to ``transformers.audio_utils.mel_filter_bank`` with
    ``norm="slaney", mel_scale="slaney"``, which upstream Voxtral uses.
    """

    def hz_to_mel(f):
        f = np.asarray(f, dtype=np.float64)
        log_region = 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (
            math.log(6.4) / 27.0
        )
        return np.where(f >= 1000.0, log_region, 3.0 * f / 200.0)

    def mel_to_hz(m):
        m = np.asarray(m, dtype=np.float64)
        log_region = 1000.0 * np.exp((math.log(6.4) / 27.0) * (m - 15.0))
        return np.where(m >= 15.0, log_region, 200.0 * m / 3.0)

    n_freq = window_size // 2 + 1
    fft_freqs = np.linspace(0, sampling_rate // 2, n_freq)
    hz_points = mel_to_hz(
        np.linspace(hz_to_mel(0.0), hz_to_mel(sampling_rate / 2), num_mel_bins + 2)
    )
    slopes = hz_points[None, :] - fft_freqs[:, None]
    widths = np.diff(hz_points)
    down = -slopes[:, :-2] / widths[:-1]
    up = slopes[:, 2:] / widths[1:]
    filters = np.maximum(0.0, np.minimum(down, up))
    enorm = 2.0 / (hz_points[2 : num_mel_bins + 2] - hz_points[:num_mel_bins])
    return torch.tensor(filters * enorm[None, :], dtype=torch.float32)


class VoxtralLogMel:
    """Whisper log-mel spectrogram, the same values as upstream Voxtral.

    Upstream computes the mel on the GPU. The plugin computes it on the CPU in
    ``embed_multimodal``. The processor pads each clip with zeros to a multiple
    of 30 s. A frame that is fully in the zero padding always gets the clamp
    floor, so this class does not compute those frames. The computed part still
    ends with at least ``window_size`` zeros, so the values match upstream. A
    shorter matmul can round the last fp32 bit differently, depending on the
    BLAS thread count.
    """

    def __init__(self, config: VoxtralAudioConfig):
        self.window_size = config.window_size
        self.hop_length = config.hop_length
        self.global_log_mel_max = config.global_log_mel_max
        # Always make the tables on the CPU. The runner builds the model under a
        # meta-device context, and the mel runs on CPU tensors.
        with torch.device("cpu"):
            self.filters = _slaney_mel_filters(
                config.num_mel_bins, config.window_size, config.sampling_rate
            )
            self.window = torch.hann_window(config.window_size)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """Return the ``[num_mel_bins, len(audio) // hop_length]`` log-mel."""
        audio = audio.float()
        # Whisper drops the last STFT frame.
        num_frames = audio.shape[-1] // self.hop_length
        nonzero = torch.nonzero(audio)
        last = int(nonzero[-1]) + 1 if nonzero.numel() else 0
        keep = min(
            audio.shape[-1],
            ((last + self.window_size) // self.hop_length + 1) * self.hop_length,
        )
        stft = torch.stft(
            audio[:keep],
            self.window_size,
            self.hop_length,
            window=self.window,
            return_complex=True,
        )
        log_spec = torch.clamp(self.filters.T @ stft.abs() ** 2, min=1e-10).log10()
        if self.global_log_mel_max is not None:
            floor = torch.tensor(self.global_log_mel_max - 8.0)
        else:
            floor = torch.maximum(log_spec.max(), torch.tensor(-10.0)) - 8.0
        log_spec = torch.maximum(log_spec, floor)

        out = torch.empty(log_spec.shape[0], num_frames, device="cpu")
        computed = min(log_spec.shape[1], num_frames)
        out[:, :computed] = log_spec[:, :computed]
        if computed < num_frames:
            out[:, computed:] = torch.maximum(torch.tensor(-10.0), floor)
        return (out + 4.0) / 4.0


# =============================================================================
# Audio encoder (Whisper encoder) and audio adapter
# =============================================================================


def _qkv_weight_loader(heads_per_rank: int, head_dim: int) -> SafetensorsWeightLoader:
    """Fuse the q/k/v weights of this rank into one ``[d_model, 3 * hd]`` param."""
    hd = heads_per_rank * head_dim

    def transform(slices, rank):
        parts = [s[rank * hd : (rank + 1) * hd] for s in slices]
        return torch.cat(parts, dim=0).T

    return SafetensorsWeightLoader(transform=transform)


def _qkv_bias_loader(heads_per_rank: int, head_dim: int) -> SafetensorsWeightLoader:
    """Fuse the q/v biases of this rank. Whisper has no k bias, so use zeros."""
    hd = heads_per_rank * head_dim

    def transform(slices, rank):
        q = slices[0][rank * hd : (rank + 1) * hd]
        v = slices[1][rank * hd : (rank + 1) * hd]
        return torch.cat([q, torch.zeros_like(q), v])

    return SafetensorsWeightLoader(transform=transform)


def _transpose_loader() -> SafetensorsWeightLoader:
    return SafetensorsWeightLoader(transform=lambda slices, rank: slices[0][:].T)


class VoxtralEncoderLayer(nn.Module):
    """One Whisper encoder layer, tensor parallel over heads and FFN width."""

    def __init__(self, config: VoxtralAudioConfig):
        super().__init__()
        # >>> PARALLELISM: encoder TP group (separate from the text TP group) <<<
        self.tp_group = get_neuron_encoder_tp_group()
        self.world_size = self.tp_group.world_size
        if config.encoder_attention_heads % self.world_size:
            raise ValueError(
                f"encoder_attention_heads ({config.encoder_attention_heads}) is not "
                f"a multiple of the encoder TP size ({self.world_size}). Change "
                "encoder_neuron_config.tp_size."
            )
        self.d_model = config.d_model
        self.head_dim = config.head_dim
        self.heads_per_rank = config.encoder_attention_heads // self.world_size
        self.ffn_per_rank = config.encoder_ffn_dim // self.world_size
        hd = self.heads_per_rank * self.head_dim
        dtype = config.torch_dtype

        def param(*shape):
            return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)

        self.self_attn_layer_norm = nn.LayerNorm(self.d_model, eps=1e-5, dtype=dtype)
        self.final_layer_norm = nn.LayerNorm(self.d_model, eps=1e-5, dtype=dtype)
        self.qkv_weight = param(self.d_model, 3 * hd)
        self.qkv_bias = param(3 * hd)
        self.out_proj_weight = param(hd, self.d_model)
        self.out_proj_bias = param(self.d_model)
        self.fc1_weight = param(self.d_model, self.ffn_per_rank)
        self.fc1_bias = param(self.ffn_per_rank)
        self.fc2_weight = param(self.ffn_per_rank, self.d_model)
        self.fc2_bias = param(self.d_model)

        # >>> PARALLELISM: shard heads (attention) and the FFN width (MLP) <<<
        set_weight_loader(
            self.qkv_weight, _qkv_weight_loader(self.heads_per_rank, self.head_dim)
        )
        set_weight_loader(
            self.qkv_bias, _qkv_bias_loader(self.heads_per_rank, self.head_dim)
        )
        for weight, shard_dim, size in (
            (self.out_proj_weight, 0, hd),
            (self.fc1_weight, 1, self.ffn_per_rank),
            (self.fc2_weight, 0, self.ffn_per_rank),
        ):
            set_weight_loader(
                weight,
                sharding_weight_loader(
                    shard_dim=shard_dim,
                    shard_size=size,
                    num_shards=self.world_size,
                    is_storage_transposed=True,
                ),
            )
        set_weight_loader(
            self.fc1_bias,
            sharding_weight_loader(
                shard_dim=0, shard_size=self.ffn_per_rank, num_shards=self.world_size
            ),
        )

    def _all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.world_size > 1:
            self.tp_group.all_reduce(x)
        return x

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        windows, seq, _ = h.shape
        hd = self.heads_per_rank * self.head_dim
        qkv = self.self_attn_layer_norm(h) @ self.qkv_weight + self.qkv_bias
        q, k, v = qkv[..., :hd], qkv[..., hd : 2 * hd], qkv[..., 2 * hd :]

        def heads(t):
            return (
                t.view(windows, seq, self.heads_per_rank, self.head_dim)
                .permute(0, 2, 1, 3)
                .reshape(windows * self.heads_per_rank, seq, self.head_dim)
            )

        # <-- MODEL-SPECIFIC: bidirectional attention inside each 30 s window.
        attn = NF.flash_attention(
            heads(q),
            heads(k),
            heads(v),
            scale=self.head_dim**-0.5,
            causal_mask=False,
            tp_q=True,
            tp_k=True,
            tp_out=False,
        )
        attn = (
            attn.view(windows, self.heads_per_rank, seq, self.head_dim)
            .permute(0, 2, 1, 3)
            .reshape(windows, seq, hd)
        )
        # >>> PARALLELISM: row-parallel projections; add the bias after the reduce <<<
        h = h + self._all_reduce(attn @ self.out_proj_weight) + self.out_proj_bias
        x = _gelu(self.final_layer_norm(h) @ self.fc1_weight + self.fc1_bias)
        return h + self._all_reduce(x @ self.fc2_weight) + self.fc2_bias


class VoxtralAudioLanguageAdapter(nn.Module):
    """2-layer adapter from stacked encoder frames to the text hidden size."""

    def __init__(self, config: VoxtralAudioConfig):
        super().__init__()
        in_dim = config.d_model * config.downsample_factor
        dtype = config.torch_dtype
        self.w_in = nn.Parameter(
            torch.empty(in_dim, config.text_hidden_size, dtype=dtype),
            requires_grad=False,
        )
        self.w_out = nn.Parameter(
            torch.empty(config.text_hidden_size, config.text_hidden_size, dtype=dtype),
            requires_grad=False,
        )
        set_weight_loader(self.w_in, _transpose_loader())
        set_weight_loader(self.w_out, _transpose_loader())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _gelu(x @ self.w_in) @ self.w_out


class VoxtralAudioTower(nn.Module):
    """Whisper encoder + audio adapter. Writes embeddings into the encoder cache."""

    def __init__(self, config: VoxtralAudioConfig):
        super().__init__()
        self.config = config
        self.tp_group = get_neuron_encoder_tp_group()
        dtype = config.torch_dtype
        d = config.d_model

        def param(*shape):
            return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)

        self.conv1_weight = param(d, config.num_mel_bins, 3)
        self.conv1_bias = param(d)
        self.conv2_weight = param(d, d, 3)
        self.conv2_bias = param(d)
        self.embed_positions = param(config.max_source_positions, d)
        self.layers = nn.ModuleList(
            [VoxtralEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(d, eps=1e-5, dtype=dtype)
        self.audio_language_adapter = VoxtralAudioLanguageAdapter(config)

    @classmethod
    def build_weight_mappings(cls, num_layers: int) -> dict[str, str | list[str]]:
        """Map each parameter name to its key(s) in the HF-format checkpoint."""
        e, a = _ENCODER_PREFIX, _ADAPTER_PREFIX
        mappings: dict[str, str | list[str]] = {
            "conv1_weight": f"{e}.conv1.weight",
            "conv1_bias": f"{e}.conv1.bias",
            "conv2_weight": f"{e}.conv2.weight",
            "conv2_bias": f"{e}.conv2.bias",
            "embed_positions": f"{e}.embed_positions.weight",
            "layer_norm.weight": f"{e}.layer_norm.weight",
            "layer_norm.bias": f"{e}.layer_norm.bias",
            "audio_language_adapter.w_in": f"{a}.linear_1.weight",
            "audio_language_adapter.w_out": f"{a}.linear_2.weight",
        }
        for i in range(num_layers):
            lp, cp = f"layers.{i}", f"{e}.layers.{i}"
            mappings[f"{lp}.qkv_weight"] = [
                f"{cp}.self_attn.{n}_proj.weight" for n in ("q", "k", "v")
            ]
            mappings[f"{lp}.qkv_bias"] = [
                f"{cp}.self_attn.q_proj.bias",
                f"{cp}.self_attn.v_proj.bias",
            ]
            mappings[f"{lp}.out_proj_weight"] = f"{cp}.self_attn.out_proj.weight"
            mappings[f"{lp}.out_proj_bias"] = f"{cp}.self_attn.out_proj.bias"
            for ln in ("self_attn_layer_norm", "final_layer_norm"):
                for p in ("weight", "bias"):
                    mappings[f"{lp}.{ln}.{p}"] = f"{cp}.{ln}.{p}"
            for fc in ("fc1", "fc2"):
                mappings[f"{lp}.{fc}_weight"] = f"{cp}.{fc}.weight"
                mappings[f"{lp}.{fc}_bias"] = f"{cp}.{fc}.bias"
        return mappings

    def load_weights(self, checkpoint_path: str, device: torch.device) -> None:
        checkpoint = SafetensorsCheckpoint(checkpoint_path)
        state_dict = checkpoint.load_sharded(
            rank=self.tp_group.rank_in_group,
            world_size=self.tp_group.world_size,
            model=self,
            mappings=self.build_weight_mappings(self.config.encoder_layers),
            device=device,
        ).state_dict
        # The checkpoint keeps some tensors (for example embed_positions) in fp32.
        # Cast every tensor to the model dtype, so the graph has one dtype.
        state_dict = {k: v.to(self.config.torch_dtype) for k, v in state_dict.items()}
        missing, unexpected = self.load_state_dict(
            state_dict, strict=False, assign=True
        )
        if missing or unexpected:
            raise RuntimeError(
                f"Voxtral audio weights do not match the checkpoint. Missing: "
                f"{sorted(missing)}. Unexpected: {sorted(unexpected)}."
            )

    def forward(
        self,
        input_features: torch.Tensor,
        encoder_cache_buffer: torch.Tensor,
        write_block_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Encode ``[windows, num_mel_bins, frames]`` mel into the encoder cache.

        Each window gives ``tokens_per_window`` rows. The rows go to the cache
        block ``write_block_ids[i]``. The buffer is aliased in place.
        """
        cfg = self.config
        x = input_features.to(cfg.torch_dtype)
        x = _gelu(F.conv1d(x, self.conv1_weight, self.conv1_bias, padding=1))
        x = _gelu(F.conv1d(x, self.conv2_weight, self.conv2_bias, stride=2, padding=1))
        h = x.permute(0, 2, 1) + self.embed_positions
        for layer in self.layers:
            h = layer(h)
        h = self.layer_norm(h)
        # <-- MODEL-SPECIFIC: stack downsample_factor adjacent frames per token.
        h = h.reshape(
            h.shape[0], cfg.tokens_per_window, cfg.d_model * cfg.downsample_factor
        )
        out = self.audio_language_adapter(h)
        if write_block_ids.shape[0] != out.shape[0]:
            raise ValueError(
                f"write_block_ids has {write_block_ids.shape[0]} ids for "
                f"{out.shape[0]} windows. Give one cache block id for each window."
            )
        encoder_cache_buffer.index_put_(
            (write_block_ids,), out.to(encoder_cache_buffer.dtype)
        )
        return encoder_cache_buffer


# =============================================================================
# Composed model
# =============================================================================


# Keep upstream vLLM's processor: the worker registers this class under the
# upstream architecture name, which replaces the upstream registration.
@MULTIMODAL_REGISTRY.register_processor(
    VoxtralMultiModalProcessor,
    info=VoxtralProcessingInfo,
    dummy_inputs=VoxtralDummyInputsBuilder,
)
class VoxtralForConditionalGeneration(LlamaForCausalLM, SupportsEncoderBudget):
    """Voxtral: the plugin Llama text decoder plus the Neuron audio tower."""

    # Text weights are under language_model.*. The audio tower loads its own weights.
    checkpoint_prefix = _TEXT_PREFIX
    own_weight_prefixes = ("audio_tower.",)

    # Bare flags, not the SupportsTranscription protocol: the upstream Voxtral
    # class owns the prompt methods, so this class only advertises the task.
    supports_transcription: bool = True
    supports_transcription_only: bool = False

    def __init__(self, config: VoxtralConfig):
        super().__init__(config.text_config)
        self.audio_config = config.audio_config
        encoder_config = config.audio_config.neuron_config
        build_encoder = encoder_config is not None and not getattr(
            encoder_config, "mm_language_model_only", False
        )
        self.audio_tower = (
            VoxtralAudioTower(config.audio_config) if build_encoder else None
        )
        self._log_mel = VoxtralLogMel(config.audio_config)

    @classmethod
    def from_configs(
        cls,
        hf_config,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config: EncoderNeuronConfig | None = None,
    ) -> "VoxtralForConditionalGeneration":
        return cls(
            VoxtralConfig.from_configs(
                hf_config, text_neuron_config, encoder_neuron_config
            )
        )

    # ------------------------------------------------------------------
    # SupportsEncoderBudget: buckets are windows (one cache block each).
    # ------------------------------------------------------------------
    @classmethod
    def get_max_encoder_embed_budget(
        cls, hf_config, encoder_neuron_config: EncoderNeuronConfig
    ) -> int | None:
        buckets = encoder_neuron_config.num_encoder_input_buckets
        if not buckets:
            return None
        return (
            max(buckets)
            * VoxtralAudioConfig.from_hf_config(hf_config).tokens_per_window
        )

    @classmethod
    def get_max_encoder_blocks_per_request(
        cls, hf_config, encoder_neuron_config: EncoderNeuronConfig
    ) -> int:
        buckets = encoder_neuron_config.num_encoder_input_buckets
        return max(buckets) if buckets else 0

    # ------------------------------------------------------------------
    # SupportsEncoderWarmup
    # ------------------------------------------------------------------
    def get_encoder(self) -> nn.Module | None:
        return self.audio_tower

    def build_encoder_synthetic_inputs(
        self,
        bucket: int,
        encoder_neuron_config: EncoderNeuronConfig,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        cfg = self.audio_config
        # One cache block holds one window, so a bucket of N windows writes N
        # blocks. The generic warmup sizes write_block_ids from
        # bucket / encoder_attention_block_size (token units), which gives one id
        # here. Give the real shape, so warmup compiles the graph that serving runs.
        return {
            "input_features": torch.zeros(
                bucket,
                cfg.num_mel_bins,
                cfg.mel_frames_per_window,
                dtype=cfg.torch_dtype,
                device=device,
            ),
            "write_block_ids": torch.zeros(bucket, dtype=torch.int64, device=device),
        }

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------
    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None = None
    ) -> None:
        super().load_weights(checkpoint_path, device, cache_dir)
        if self.audio_tower is not None:
            # Load on the CPU, like the other encoder models. The runner moves the
            # model to the device later.
            self.audio_tower.load_weights(checkpoint_path, torch.device("cpu"))

    # ------------------------------------------------------------------
    # Multimodal path
    # ------------------------------------------------------------------
    def embed_multimodal(
        self,
        audio_arrays: list[torch.Tensor] | torch.Tensor | None = None,
        encoder_cache=None,
        mm_hashes: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Encode the audio clips and write the embeddings into the encoder cache.

        Args:
            audio_arrays: One 1-D waveform for each clip, from the upstream Voxtral
                processor. Each length is a multiple of 30 s.
            encoder_cache: EncoderCacheBlocks (buffer, allocate, scratch_block_id).
            mm_hashes: One identifier for each clip, in clip order.
        """
        if audio_arrays is None:
            raise ValueError(
                "embed_multimodal needs audio_arrays. Use the upstream Voxtral processor."
            )
        if isinstance(audio_arrays, torch.Tensor):
            audio_arrays = (
                list(audio_arrays.unbind(0))
                if audio_arrays.dim() > 1
                else [audio_arrays]
            )
        cfg = self.audio_config
        mels, block_ids = [], []
        for mm_hash, wav in zip(mm_hashes, audio_arrays, strict=True):
            wav = wav.reshape(-1).float().cpu()
            windows = max(1, math.ceil(wav.numel() / cfg.samples_per_window))
            wav = F.pad(wav, (0, windows * cfg.samples_per_window - wav.numel()))
            mel = self._log_mel(wav)[:, : windows * cfg.mel_frames_per_window]
            mels.extend(
                mel.reshape(
                    cfg.num_mel_bins, windows, cfg.mel_frames_per_window
                ).unbind(1)
            )
            block_ids.extend(
                encoder_cache.allocate(mm_hash, [cfg.tokens_per_window] * windows)
            )

        buckets = sorted(cfg.neuron_config.num_encoder_input_buckets)
        bucket = next((b for b in buckets if b >= len(mels)), None)
        if bucket is None:
            raise ValueError(
                f"The batch has {len(mels)} audio windows, but the largest encoder "
                f"bucket is {buckets[-1]}. Add a larger num_encoder_input_buckets "
                "value, or send less audio in one step."
            )
        pad = bucket - len(mels)
        features = torch.stack(mels + [torch.zeros_like(mels[0])] * pad)
        write_ids = torch.tensor(
            block_ids + [encoder_cache.scratch_block_id] * pad, dtype=torch.int64
        )
        device = next(self.audio_tower.parameters()).device
        # Cast on the CPU. The device copy does not convert the dtype.
        self.audio_tower(
            features.to(cfg.torch_dtype).contiguous().to(device),
            encoder_cache.buffer,
            write_ids.to(device),
        )
