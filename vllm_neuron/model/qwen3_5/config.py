# SPDX-License-Identifier: Apache-2.0
"""Configs for Qwen3.5-2B (``model_type: qwen3_5``).

Qwen3.5 is a **hybrid** decoder: of its 24 layers, 18 are recurrent gated
DeltaNet ("linear_attention") and 6 are full GQA attention, interleaved
``[linear, linear, linear, full] x 6``. That single fact drives most of the port:
the DeltaNet layers keep a fixed-size recurrent state plus a small conv window
instead of a growing KV cache, so they need a different cache spec, a different
warmup story and a different notion of "context length" from every other model in
this plugin.

Fields are read from the checkpoint rather than hard-coded, and anything this
implementation has not been validated against raises instead of being silently
approximated — the failure mode otherwise is fluent-but-wrong output, which costs
far more to debug than an early exception.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

# <-- MODEL-SPECIFIC: the two entries HF uses in text_config.layer_types.
LINEAR_ATTENTION = "linear_attention"
FULL_ATTENTION = "full_attention"


def _dtype_of(cfg: PretrainedConfig, default: torch.dtype) -> torch.dtype:
    raw = getattr(cfg, "dtype", None) or getattr(cfg, "torch_dtype", None)
    if raw is None:
        return default
    if isinstance(raw, torch.dtype):
        return raw
    return getattr(torch, str(raw).replace("torch.", ""))


@dataclass
class Qwen3_5TextConfig:
    """The text decoder: hybrid DeltaNet + GQA.

    Attention (the 6 full layers):
        hidden_size 2048, 8 query heads, 2 KV heads, head_dim **256**.
        ``attn_output_gate`` adds a per-head sigmoid gate on the attention output,
        so ``q_proj`` emits twice the query width (gate half + query half).
        Rotary is **partial** (``partial_rotary_factor`` 0.25 -> 64 of 256 dims)
        and mRoPE-interleaved.

    Linear attention (the 18 DeltaNet layers):
        16 key heads and 16 value heads at head_dim 128, a depthwise conv of
        width ``linear_conv_kernel_dim`` over q/k/v, and a delta-rule recurrent
        state accumulated in ``ssm_dtype`` (float32 in this checkpoint).
    """

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    layer_types: tuple[str, ...]

    # Full-attention layers
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    attn_output_gate: bool
    partial_rotary_factor: float
    rope_theta: float
    mrope_section: tuple[int, ...]
    mrope_interleaved: bool

    # Linear-attention (DeltaNet) layers
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    ssm_dtype: torch.dtype

    vocab_size: int
    rms_norm_eps: float
    tie_word_embeddings: bool
    max_position_embeddings: int
    torch_dtype: torch.dtype

    # Attached by ``Qwen3_5Config.from_configs``; the model reads on-device
    # sampling and logprob settings off it.
    neuron_config: object | None = None

    @property
    def rotary_dim(self) -> int:
        """Rotary is applied to only the first ``rotary_dim`` dims of each head."""
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def full_attention_layers(self) -> tuple[int, ...]:
        return tuple(
            i for i, t in enumerate(self.layer_types) if t == FULL_ATTENTION
        )

    @property
    def linear_attention_layers(self) -> tuple[int, ...]:
        return tuple(
            i for i, t in enumerate(self.layer_types) if t == LINEAR_ATTENTION
        )

    @property
    def conv_dim(self) -> int:
        """Channels the depthwise conv1d runs over: q, k and v concatenated.

        The conv is applied to the projected q/k/v of the DeltaNet block, so its
        channel count is the sum of their widths, not the model hidden size.
        """
        return (
            self.linear_num_key_heads * self.linear_key_head_dim * 2
            + self.linear_num_value_heads * self.linear_value_head_dim
        )

    def state_shapes(self, tp_size: int) -> tuple[tuple[int, ...], ...]:
        """Per-rank ``(conv_state_shape, recurrent_state_shape)`` for one layer.

        Delegated to vLLM's own ``MambaStateShapeCalculator`` rather than derived
        here, deliberately. vLLM sizes the state *pages* from the same helper (via
        ``Platform._align_hybrid_block_size`` ->
        ``model_cls.get_mamba_state_shape_from_config``), so re-deriving the
        shapes would risk a layout that disagrees with the pages the planner
        allocated — a silent memory-aliasing bug rather than an error. A
        hand-rolled version of this got the conv state transposed
        (``[conv_dim, kernel-1]`` instead of ``[kernel-1, conv_dim]``) and
        unsharded.

        At TP=4 this checkpoint gives conv ``(3, 1536)`` and recurrent
        ``(4, 128, 128)``: the 16 value heads shard 4-per-rank, and only
        ``kernel - 1`` conv columns are carried between steps because the current
        token supplies the last one.
        """
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateShapeCalculator,
        )

        return tuple(
            MambaStateShapeCalculator.gated_delta_net_state_shape(
                tp_size,
                self.linear_num_key_heads,
                self.linear_num_value_heads,
                self.linear_key_head_dim,
                self.linear_value_head_dim,
                self.linear_conv_kernel_dim,
                0,  # num_spec: the MTP head is not wired up
            )
        )

    def state_dtypes(self) -> tuple[torch.dtype, torch.dtype]:
        """``(conv_dtype, recurrent_dtype)``.

        The conv window holds activations so it follows the model dtype, while the
        recurrent state accumulates over the whole sequence and uses
        ``mamba_ssm_dtype`` (float32 in this checkpoint) to stop the delta rule
        drifting.
        """
        return (self.torch_dtype, self.ssm_dtype)

    @classmethod
    def from_hf(cls, text_cfg: PretrainedConfig) -> Qwen3_5TextConfig:
        rope = getattr(text_cfg, "rope_parameters", None) or {}
        if not isinstance(rope, dict):
            rope = dict(rope)

        rope_type = rope.get("rope_type", "default")
        if rope_type != "default":
            raise NotImplementedError(
                f"Qwen3.5 rope_type {rope_type!r} is not implemented; only "
                f"'default' (plain mRoPE, no scaling) has been validated."
            )
        if not rope.get("mrope_interleaved", False):
            raise NotImplementedError(
                "Qwen3.5 with mrope_interleaved=False is not implemented; this "
                "checkpoint interleaves the mRoPE sections and the two layouts "
                "are not interchangeable."
            )

        layer_types = tuple(getattr(text_cfg, "layer_types"))
        unknown = set(layer_types) - {LINEAR_ATTENTION, FULL_ATTENTION}
        if unknown:
            raise NotImplementedError(
                f"unknown Qwen3.5 layer types {sorted(unknown)}; only "
                f"{LINEAR_ATTENTION!r} and {FULL_ATTENTION!r} are implemented."
            )
        if len(layer_types) != text_cfg.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(layer_types)} entries but "
                f"num_hidden_layers is {text_cfg.num_hidden_layers}"
            )

        # <-- MODEL-SPECIFIC: mlp_only_layers is a Qwen3-Next/MoE knob. This
        # checkpoint leaves it empty and the dense MLP is used on every layer.
        mlp_only = tuple(getattr(text_cfg, "mlp_only_layers", ()) or ())
        if mlp_only:
            raise NotImplementedError(
                f"mlp_only_layers={list(mlp_only)} is not implemented; this "
                f"port assumes every layer has both a mixer and an MLP."
            )

        if not getattr(text_cfg, "attn_output_gate", False):
            raise NotImplementedError(
                "Qwen3.5 without attn_output_gate is not implemented; the "
                "q_proj shape depends on it (it emits gate and query halves)."
            )

        ssm_dtype = getattr(text_cfg, "mamba_ssm_dtype", "float32")
        ssm_dtype = (
            ssm_dtype
            if isinstance(ssm_dtype, torch.dtype)
            else getattr(torch, str(ssm_dtype).replace("torch.", ""))
        )

        return cls(
            hidden_size=text_cfg.hidden_size,
            intermediate_size=text_cfg.intermediate_size,
            num_hidden_layers=text_cfg.num_hidden_layers,
            layer_types=layer_types,
            num_attention_heads=text_cfg.num_attention_heads,
            num_key_value_heads=text_cfg.num_key_value_heads,
            head_dim=text_cfg.head_dim,
            attn_output_gate=True,
            partial_rotary_factor=float(rope.get("partial_rotary_factor", 1.0)),
            rope_theta=float(rope.get("rope_theta", 10000.0)),
            mrope_section=tuple(rope.get("mrope_section", ())),
            mrope_interleaved=True,
            linear_num_key_heads=text_cfg.linear_num_key_heads,
            linear_num_value_heads=text_cfg.linear_num_value_heads,
            linear_key_head_dim=text_cfg.linear_key_head_dim,
            linear_value_head_dim=text_cfg.linear_value_head_dim,
            linear_conv_kernel_dim=text_cfg.linear_conv_kernel_dim,
            ssm_dtype=ssm_dtype,
            vocab_size=text_cfg.vocab_size,
            rms_norm_eps=text_cfg.rms_norm_eps,
            tie_word_embeddings=bool(
                getattr(text_cfg, "tie_word_embeddings", False)
            ),
            max_position_embeddings=text_cfg.max_position_embeddings,
            torch_dtype=_dtype_of(text_cfg, torch.bfloat16),
        )


@dataclass
class Qwen3_5VisionConfig:
    """The ViT tower. Not used by the text-only path; kept for the VL stage.

    Unlike Qwen3-VL this checkpoint has ``deepstack_visual_indexes == []``, so
    there are no deepstack side-outputs and the merged embedding is plain
    ``out_hidden_size``-wide.
    """

    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    in_channels: int
    patch_size: int
    spatial_merge_size: int
    temporal_patch_size: int
    num_position_embeddings: int
    out_hidden_size: int
    deepstack_visual_indexes: tuple[int, ...]
    torch_dtype: torch.dtype
    # The plugin's vision encoder reads bucket/block settings off this; attached
    # by ``Qwen3_5Config.from_configs``.
    neuron_config: object | None = None

    @classmethod
    def from_hf(cls, vision_cfg: PretrainedConfig) -> Qwen3_5VisionConfig:
        deepstack = tuple(
            getattr(vision_cfg, "deepstack_visual_indexes", ()) or ()
        )
        if deepstack:
            raise NotImplementedError(
                f"deepstack_visual_indexes={list(deepstack)} is not implemented "
                f"for Qwen3.5; this checkpoint has none, so the merged vision "
                f"embedding is a single out_hidden_size-wide tensor."
            )
        return cls(
            depth=vision_cfg.depth,
            hidden_size=vision_cfg.hidden_size,
            intermediate_size=vision_cfg.intermediate_size,
            num_heads=vision_cfg.num_heads,
            in_channels=vision_cfg.in_channels,
            patch_size=vision_cfg.patch_size,
            spatial_merge_size=vision_cfg.spatial_merge_size,
            temporal_patch_size=vision_cfg.temporal_patch_size,
            num_position_embeddings=vision_cfg.num_position_embeddings,
            out_hidden_size=vision_cfg.out_hidden_size,
            deepstack_visual_indexes=(),
            torch_dtype=_dtype_of(vision_cfg, torch.bfloat16),
        )


@dataclass
class Qwen3_5Config:
    """Top-level config: text decoder, optional vision tower, token ids."""

    text_config: Qwen3_5TextConfig
    vision_config: Qwen3_5VisionConfig | None
    image_token_id: int | None
    video_token_id: int | None
    vision_start_token_id: int | None
    vision_end_token_id: int | None
    neuron_config: object | None = None
    vision_neuron_config: object | None = None
    extras: dict = field(default_factory=dict)

    @classmethod
    def from_hf(
        cls,
        hf_config: PretrainedConfig,
        *,
        include_vision: bool = True,
    ) -> Qwen3_5Config:
        text_cfg = getattr(hf_config, "text_config", None)
        if text_cfg is None:
            raise ValueError("Qwen3.5 config is missing text_config")

        vision_cfg = getattr(hf_config, "vision_config", None)
        vision = (
            Qwen3_5VisionConfig.from_hf(vision_cfg)
            if include_vision and vision_cfg is not None
            else None
        )

        return cls(
            text_config=Qwen3_5TextConfig.from_hf(text_cfg),
            vision_config=vision,
            image_token_id=getattr(hf_config, "image_token_id", None),
            video_token_id=getattr(hf_config, "video_token_id", None),
            vision_start_token_id=getattr(hf_config, "vision_start_token_id", None),
            vision_end_token_id=getattr(hf_config, "vision_end_token_id", None),
        )

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: object | None = None,
        vision_neuron_config: object | None = None,
        *,
        include_vision: bool | None = None,
    ) -> Qwen3_5Config:
        """Entry point used by the factory: HF config plus the runner's NeuronConfigs.

        The vision tower is built only when the runner supplied a
        ``VisionNeuronConfig``, which it does exactly when the engine was
        configured to serve images or video. A text-only launch therefore pays
        neither the tower's weights nor its compile time. (The runner only
        creates that config when ``additional_config["vision_neuron_config"]``
        exists, and the platform only creates *that* when some modality has a
        non-zero per-prompt limit.) Pass ``include_vision`` explicitly to
        override, which the vision checks do.
        """
        if include_vision is None:
            include_vision = vision_neuron_config is not None
        config = cls.from_hf(hf_config, include_vision=include_vision)
        config.text_config.neuron_config = text_neuron_config
        config.neuron_config = text_neuron_config
        config.vision_neuron_config = vision_neuron_config
        if config.vision_config is not None:
            config.vision_config.neuron_config = vision_neuron_config
        return config
