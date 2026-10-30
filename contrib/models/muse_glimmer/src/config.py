# SPDX-License-Identifier: Apache-2.0
"""
MuseGlimmer Config
==================

Nested multimodal config: ``MuseGlimmerConfig`` composes a
``MuseGlimmerTextConfig`` (decoder) and a ``MuseGlimmerVisionConfig``
(perception encoder), each carrying its own NeuronConfig.

Silent-config-drop guard: every architecture-defining text field that has no
safe default (sliding window, layer pattern, per-layer RoPE on/off, QK scale,
post-norm eps, logit multiplier, softcap) is *required* to be present in the
HF config. A missing key raises instead of silently falling back to a
dataclass default. ``validate()`` then asserts the structural invariants of
the architecture, and -- for the released 52-layer geometry, or when
``MUSE_GLIMMER_STRICT_CONFIG=1`` -- the exact released values.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, fields

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig, EncoderNeuronConfig

logger = logging.getLogger(__name__)

# Values of the released checkpoint. Checked when strict validation is active.
REFERENCE_TEXT_VALUES = {
    "hidden_size": 6656,
    "intermediate_size": 19968,
    "num_hidden_layers": 52,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "vocab_size": 202048,
    "sliding_window": 2048,
    "qk_scale_factor": 3.87,
    "output_multiplier": 0.19611613513818404,
    "final_logit_softcapping": 20.0,
    "post_norm_eps": 1e-8,
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0,
}

# Text fields that must be present in the HF config (no silent default).
_REQUIRED_TEXT_KEYS = (
    "sliding_window",
    "layer_types",
    "layer_rope_theta",
    "qk_scale_factor",
    "output_multiplier",
    "final_logit_softcapping",
    "post_norm_eps",
    "rms_norm_eps",
    "rope_parameters",
)

SLIDING = "sliding_attention"
FULL = "full_attention"


def _to_dict(hf_sub_config) -> dict:
    if isinstance(hf_sub_config, PretrainedConfig):
        return hf_sub_config.to_dict()
    if isinstance(hf_sub_config, dict):
        return dict(hf_sub_config)
    raise TypeError(f"Unsupported config type: {type(hf_sub_config)}")


def _filter_fields(cls, config_dict: dict) -> dict:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in config_dict.items() if k in names}


def _strict_requested(text_cfg: "MuseGlimmerTextConfig") -> bool:
    env = os.environ.get("MUSE_GLIMMER_STRICT_CONFIG")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    # Default: strict for the released geometry, structural-only otherwise
    # (truncated test checkpoints).
    return (
        text_cfg.num_hidden_layers == REFERENCE_TEXT_VALUES["num_hidden_layers"]
        and text_cfg.hidden_size == REFERENCE_TEXT_VALUES["hidden_size"]
    )


@dataclass
class MuseGlimmerTextConfig:
    """Text decoder config extracted from ``hf_config.text_config``."""

    vocab_size: int = 202048
    hidden_size: int = 6656
    intermediate_size: int = 19968
    num_hidden_layers: int = 52
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 128
    hidden_activation: str = "silu"
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    bos_token_id: int | None = 200000
    eos_token_id: int | list[int] | None = 200001

    # Architecture-defining fields (required in the HF config).
    sliding_window: int = 2048
    layer_types: list[str] = field(default_factory=list)
    layer_rope_theta: list[float] = field(default_factory=list)
    qk_scale_factor: float = 3.87
    output_multiplier: float = 0.19611613513818404
    final_logit_softcapping: float = 20.0
    post_norm_eps: float = 1e-8
    rope_parameters: dict = field(
        default_factory=lambda: {"rope_theta": 500000.0, "rope_type": "default"}
    )

    torch_dtype: torch.dtype = torch.bfloat16
    neuron_config: NeuronConfig | None = None

    # ── derived helpers ────────────────────────────────────────────────
    @property
    def rope_theta(self) -> float:
        return float(self.rope_parameters["rope_theta"])

    def is_sliding(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == SLIDING

    def uses_rope(self, layer_idx: int) -> bool:
        # HF only uses layer_rope_theta as an on/off switch (0 = NoPE); the
        # angle base is always the global rope_parameters["rope_theta"].
        return bool(self.layer_rope_theta[layer_idx])

    @property
    def attention_scale(self) -> float:
        """Softmax scale with the Q multiplier folded in (exact algebra).

        HF: softmax((qk_scale * norm(q)) . norm(k) * head_dim**-0.5). The Q
        multiplier is applied after the weightless QK-norm and before RoPE;
        RoPE is linear, so folding it into the softmax scale is exact and avoids
        rounding 3.87 to bf16 (3.875).
        """
        return float(self.qk_scale_factor) * float(self.head_dim) ** -0.5

    @classmethod
    def from_hf_config(cls, hf_text_config, neuron_config: NeuronConfig = None):
        config_dict = _to_dict(hf_text_config)
        missing = [k for k in _REQUIRED_TEXT_KEYS if config_dict.get(k) is None]
        if missing:
            raise ValueError(
                "MuseGlimmer text config is missing required field(s) "
                f"{missing}. These have no safe default; refusing to fall back "
                "silently (see silent-config-drops)."
            )
        filtered = _filter_fields(cls, config_dict)
        # Compute dtype. BF16 in production; MUSE_GLIMMER_COMPUTE_DTYPE=float32 is a
        # CPU-mode test knob for exact structural comparison against HF fp32
        # (pair it with LLM(dtype="float32") so the KV cache matches).
        dt_name = os.environ.get("MUSE_GLIMMER_COMPUTE_DTYPE", "bfloat16")
        if dt_name not in ("bfloat16", "float32"):
            raise ValueError(f"MUSE_GLIMMER_COMPUTE_DTYPE={dt_name!r} not in (bfloat16, float32)")
        filtered["torch_dtype"] = getattr(torch, dt_name)
        filtered["layer_types"] = list(config_dict["layer_types"])
        filtered["layer_rope_theta"] = [float(t) for t in config_dict["layer_rope_theta"]]
        filtered["rope_parameters"] = dict(config_dict["rope_parameters"])
        if neuron_config is not None:
            filtered["neuron_config"] = neuron_config
        cfg = cls(**filtered)
        cfg.validate()
        return cfg

    def validate(self, strict: bool | None = None) -> None:
        """Load-time asserts for every special field (raise on mismatch)."""
        n = self.num_hidden_layers
        errs: list[str] = []
        if len(self.layer_types) != n:
            errs.append(f"len(layer_types)={len(self.layer_types)} != num_hidden_layers={n}")
        if len(self.layer_rope_theta) != n:
            errs.append(
                f"len(layer_rope_theta)={len(self.layer_rope_theta)} != num_hidden_layers={n}"
            )
        bad_types = sorted(set(self.layer_types) - {SLIDING, FULL})
        if bad_types:
            errs.append(f"unknown layer_types {bad_types}")
        for i, (lt, th) in enumerate(zip(self.layer_types, self.layer_rope_theta)):
            # The released model ties NoPE to the full-attention layers.
            if (lt == FULL) != (th == 0):
                errs.append(f"layer {i}: layer_type={lt} but layer_rope_theta={th}")
            if th not in (0.0, self.rope_theta):
                errs.append(
                    f"layer {i}: layer_rope_theta={th} differs from the global "
                    f"rope_theta={self.rope_theta}; per-layer bases are not supported "
                    "(HF uses it only as an on/off switch)"
                )
        if not (isinstance(self.sliding_window, int) and self.sliding_window > 0):
            errs.append(f"sliding_window must be a positive int, got {self.sliding_window!r}")
        if self.rope_parameters.get("rope_type", "default") != "default":
            errs.append(f"unsupported text rope_type {self.rope_parameters.get('rope_type')!r}")
        if not self.qk_scale_factor > 0:
            errs.append(f"qk_scale_factor must be > 0, got {self.qk_scale_factor}")
        if not self.output_multiplier > 0:
            errs.append(f"output_multiplier must be > 0, got {self.output_multiplier}")
        if not self.final_logit_softcapping > 0:
            errs.append(
                f"final_logit_softcapping must be > 0, got {self.final_logit_softcapping}"
            )
        if not 0 < self.post_norm_eps < 1e-3:
            errs.append(f"post_norm_eps out of range: {self.post_norm_eps}")
        if self.hidden_activation != "silu":
            errs.append(f"hidden_activation must be silu, got {self.hidden_activation!r}")
        if self.attention_bias:
            errs.append("attention_bias=True is not supported")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            errs.append("num_attention_heads must be a multiple of num_key_value_heads")
        if self.tie_word_embeddings:
            errs.append("tie_word_embeddings=True is not supported (lm_head is untied)")

        if strict is None:
            strict = _strict_requested(self)
        if strict:
            for key, want in REFERENCE_TEXT_VALUES.items():
                got = self.rope_theta if key == "rope_theta" else getattr(self, key)
                if isinstance(want, float):
                    ok = abs(float(got) - want) <= 1e-12 * max(1.0, abs(want))
                else:
                    ok = got == want
                if not ok:
                    errs.append(f"[strict] {key}={got!r}, expected {want!r}")
            for i, lt in enumerate(self.layer_types):
                want_lt = FULL if i % 4 == 3 else SLIDING
                if lt != want_lt:
                    errs.append(f"[strict] layer {i}: layer_type={lt}, expected {want_lt}")
        if errs:
            raise ValueError(
                "MuseGlimmer text config failed load-time validation:\n  "
                + "\n  ".join(errs)
            )
        logger.info(
            "MuseGlimmer text config OK (strict=%s): layers=%d sliding=%d full=%d "
            "window=%d qk_scale=%s attn_scale=%.6f post_norm_eps=%g mult=%s "
            "softcap=%s rope_theta=%s",
            strict,
            n,
            sum(t == SLIDING for t in self.layer_types),
            sum(t == FULL for t in self.layer_types),
            self.sliding_window,
            self.qk_scale_factor,
            self.attention_scale,
            self.post_norm_eps,
            self.output_multiplier,
            self.final_logit_softcapping,
            self.rope_theta,
        )


@dataclass
class MuseGlimmerVisionConfig:
    """Perception encoder config extracted from ``hf_config.vision_config``."""

    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_hidden_layers: int = 50
    num_attention_heads: int = 16
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    patch_size: int = 14
    patch_temporal: int = 2
    merge_size: int = 2
    pos_emb_height: int = 32
    pos_emb_width: int = 32
    max_position_embeddings: int = 1024
    layer_types: list[str] = field(default_factory=list)
    rope_parameters: dict = field(
        default_factory=lambda: {"rope_theta": 10000.0, "rope_type": "axial"}
    )

    neuron_config: EncoderNeuronConfig | None = None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def spatial_merge_size(self) -> int:
        return self.merge_size

    @property
    def rope_theta(self) -> float:
        return float(self.rope_parameters.get("rope_theta", 10000.0))

    @property
    def patch_dim(self) -> int:
        return self.patch_temporal * 3 * self.patch_size**2

    @classmethod
    def from_hf_config(cls, hf_vision_config, neuron_config: EncoderNeuronConfig = None):
        config_dict = _to_dict(hf_vision_config)
        filtered = _filter_fields(cls, config_dict)
        if neuron_config is not None:
            filtered["neuron_config"] = neuron_config
        if not filtered.get("layer_types"):
            # HF MuseGlimmerVisionConfig.__post_init__ default (raw JSON may omit it).
            n = int(filtered.get("num_hidden_layers", cls.num_hidden_layers))
            filtered["layer_types"] = [
                "full_attention" if (i + 1) % 4 == 0 or i == n - 1 else "window_attention"
                for i in range(n)
            ]
        if filtered.get("rope_parameters") is None:
            filtered.pop("rope_parameters", None)
        cfg = cls(**filtered)
        bad = sorted(set(cfg.layer_types) - {"window_attention", "full_attention"})
        if bad:
            raise ValueError(f"unknown vision layer_types {bad}")
        if cfg.pos_emb_height != cfg.pos_emb_width:
            raise ValueError("non-square vision position table is not supported (HF assumes square)")
        if cfg.hidden_act != "gelu":
            raise ValueError(f"vision hidden_act must be gelu (erf), got {cfg.hidden_act!r}")
        # transformers remaps the raw "default" rope_type to "axial" for the
        # vision tower (MuseGlimmerVisionConfig.default_rope_type). A raw JSON
        # parse sees "default"; both mean the same 2D axial RoPE here.
        rope_type = cfg.rope_parameters.get("rope_type", "axial")
        if rope_type not in ("axial", "default"):
            raise ValueError(f"unsupported vision rope_type {rope_type!r}")
        if cfg.layer_types and len(cfg.layer_types) != cfg.num_hidden_layers:
            raise ValueError(
                f"vision len(layer_types)={len(cfg.layer_types)} != "
                f"num_hidden_layers={cfg.num_hidden_layers}"
            )
        return cfg


@dataclass
class MuseGlimmerConfig:
    """Top-level multimodal config composing text and vision sub-configs."""

    text_config: MuseGlimmerTextConfig | None = None
    vision_config: MuseGlimmerVisionConfig | None = None

    image_token_id: int = 200092
    video_token_id: int = 200091
    out_hidden_size: int = 6144  # pre-adapter merge width (4 x vision hidden)
    projector_hidden_size: int = 4096
    projector_hidden_act: str = "gelu"

    # Text-decoder views read by the runner's spec-decode path (``model.config``):
    # the DFlash proposer sizes its context fc from ``hidden_size`` and the async
    # spec-decode decorator validates tokens against ``vocab_size``.
    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size

    @property
    def num_hidden_layers(self) -> int:
        return self.text_config.num_hidden_layers

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig | dict | str,
        text_neuron_config: NeuronConfig = None,
        encoder_neuron_config: EncoderNeuronConfig = None,
    ):
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                top_level = json.load(f)
            hf_text, hf_vision = top_level["text_config"], top_level["vision_config"]
        elif isinstance(hf_config, PretrainedConfig):
            hf_text, hf_vision = hf_config.text_config, hf_config.vision_config
            top_level = hf_config.to_dict()
        elif isinstance(hf_config, dict):
            top_level = hf_config
            hf_text, hf_vision = top_level["text_config"], top_level["vision_config"]
        else:
            raise TypeError(f"Unsupported hf_config type: {type(hf_config)}")

        text_config = MuseGlimmerTextConfig.from_hf_config(hf_text, text_neuron_config)
        vision_config = MuseGlimmerVisionConfig.from_hf_config(
            hf_vision, encoder_neuron_config
        )
        return cls(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=top_level.get("image_token_id", 200092),
            video_token_id=top_level.get("video_token_id", 200091),
            out_hidden_size=top_level.get("out_hidden_size", 6144),
            projector_hidden_size=top_level.get("projector_hidden_size", 4096),
            projector_hidden_act=top_level.get("projector_hidden_act", "gelu"),
        )
