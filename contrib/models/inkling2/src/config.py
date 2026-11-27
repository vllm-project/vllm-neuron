# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini configuration: the Hugging Face config classes the image's transformers lacks, and the dimensions
one tensor-parallel rank serves."""

from __future__ import annotations

from dataclasses import dataclass, field

from transformers import PretrainedConfig


class Inkling2LoopTextHFConfig(PretrainedConfig):
    """The ``text_config`` of ``inkling2_loop``, as stored. Only the keys vLLM reads are named; the rest stay plain
    attributes. ``sliding_window`` is deliberately not set: the model applies its window itself (``model.py``)."""

    model_type = "inkling2_loop_text"

    def __init__(self, **kwargs):
        kwargs.setdefault("max_position_embeddings", kwargs.get("model_max_length", 1048576))
        super().__init__(**kwargs)


class Inkling2LoopHFConfig(PretrainedConfig):
    """``inkling2_loop`` (``Inkling2LoopForConditionalGeneration``). The checkpoint's ``quantization_config``
    (ModelOpt NVFP4) is dropped: the experts are dequantized to bf16 at load, so vLLM must not quantize."""

    model_type = "inkling2_loop"
    sub_configs = {"text_config": Inkling2LoopTextHFConfig}

    def __init__(self, text_config=None, vision_config=None, mtp_config=None, **kwargs):
        kwargs.pop("quantization_config", None)
        if isinstance(text_config, dict) or text_config is None:
            text_config = Inkling2LoopTextHFConfig(**(text_config or {}))
        self.text_config = text_config
        self.vision_config = vision_config
        self.mtp_config = mtp_config
        super().__init__(**kwargs)

    def get_text_config(self, *args, **kwargs):
        return self.text_config


def register_hf_configs() -> None:
    from transformers import AutoConfig

    for cls in (Inkling2LoopTextHFConfig, Inkling2LoopHFConfig):
        try:
            AutoConfig.register(cls.model_type, cls, exist_ok=True)
        except ValueError:  # already registered by a newer transformers
            pass


@dataclass
class Inkling2Config:
    """The text decoder's dimensions and constants, and this rank's share (``tp_size``)."""

    hidden_size: int = 2304
    num_layers: int = 24
    vocab_size: int = 201024
    unpadded_vocab_size: int = 200058
    num_heads: int = 24
    num_kv_heads: int = 8
    head_dim: int = 128
    d_rel: int = 16
    rel_extent: int = 1024
    window: int = 1024
    log_floor: int = 8192
    log_alpha: float = 0.1
    eps: float = 1e-6
    qk_clip: float = 22.627416997969522
    rel_clip: float = 8.0
    block_layers: int = 6
    num_experts: int = 128
    top_k: int = 8
    moe_inter: int = 1152
    route_scale: float = 8.0
    gate_cap: float = 4.0
    up_cap: float = 25.0
    final_cap: float = 30.0
    loop_start: int = 6
    loop_end: int = 18
    passes: int = 3
    gate_rank: int = 64
    full_layers: tuple = field(default_factory=lambda: (5, 11, 17, 23))
    tp_size: int = 1

    @classmethod
    def from_hf(cls, hf, tp_size: int = 1) -> "Inkling2Config":
        t = hf.get_text_config() if hasattr(hf, "get_text_config") else hf
        g = (lambda k, d: t.get(k, d)) if isinstance(t, dict) else (lambda k, d: getattr(t, k, d))
        loop = g("loop_config", None) or {}
        n = g("num_hidden_layers", 24)
        local = set(g("local_layer_ids", [i for i in range(n) if (i + 1) % 6]))
        return cls(
            hidden_size=g("hidden_size", 2304), num_layers=n, vocab_size=g("vocab_size", 201024),
            unpadded_vocab_size=g("unpadded_vocab_size", 200058), num_heads=g("num_attention_heads", 24),
            num_kv_heads=g("num_key_value_heads", 8), head_dim=g("head_dim", 128), d_rel=g("d_rel", 16),
            rel_extent=g("rel_extent", 1024), window=g("sliding_window_size", 1024),
            log_floor=g("log_scaling_n_floor", 8192), log_alpha=g("log_scaling_alpha", 0.1),
            eps=g("rms_norm_eps", 1e-6), qk_clip=g("qk_l2_clip_threshold", 22.627416997969522),
            rel_clip=g("rel_l2_clip_threshold", 8.0), block_layers=g("n_layers_per_attn_res_block", 6),
            num_experts=g("n_routed_experts", 128), top_k=g("num_experts_per_tok", 8),
            moe_inter=g("intermediate_size", 1152), route_scale=g("route_scale", 8.0),
            gate_cap=g("mlp_gate_softcap", 4.0), up_cap=g("mlp_up_softcap", 25.0),
            final_cap=g("final_logit_softcapping", 30.0), loop_start=loop.get("core_start", 6),
            loop_end=loop.get("core_end", 18), passes=loop.get("num_passes", 3), gate_rank=loop.get("gate_rank", 64),
            full_layers=tuple(i for i in range(n) if i not in local), tp_size=tp_size)

    # -- layer structure --

    def is_full(self, L: int) -> bool:
        return L in self.full_layers

    def is_boundary(self, L: int) -> bool:
        return L % self.block_layers == 0

    @property
    def loop_layers(self) -> int:
        return self.loop_end - self.loop_start

    @property
    def cache_layers(self) -> int:
        """One KV and conv-state cache per layer visit: the core layers once per pass."""
        return self.num_layers + (self.passes - 1) * self.loop_layers

    # -- this rank --

    @property
    def q_heads_per_rank(self) -> int:
        return self.num_heads // self.tp_size

    @property
    def kv_heads_per_rank(self) -> int:
        return self.num_kv_heads // self.tp_size

    @property
    def inter_per_rank(self) -> int:
        return self.moe_inter // self.tp_size

    @property
    def lm_rows_per_rank(self) -> int:
        return -(-self.unpadded_vocab_size // self.tp_size)

    def problems(self) -> list[str]:
        out = []
        if self.num_kv_heads % self.tp_size or self.num_heads % self.tp_size:
            out.append(f"{self.num_heads} query / {self.num_kv_heads} KV heads do not split over TP {self.tp_size}")
        if self.moe_inter % self.tp_size:
            out.append(f"expert intermediate {self.moe_inter} does not split over TP {self.tp_size}")
        if self.vocab_size % self.tp_size:
            out.append(f"vocab {self.vocab_size} does not split over TP {self.tp_size}")
        if self.loop_start % self.block_layers or self.loop_end % self.block_layers:
            out.append("the loop core must start and end on attention-residual block boundaries")
        return out
