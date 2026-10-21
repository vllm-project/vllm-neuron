# SPDX-License-Identifier: Apache-2.0
"""
MiMo-V2.5 Config
================

Text-only decoder config for MiMoV2ForCausalLM (``model_type="mimo_v2"``).

Serves both released checkpoints of this architecture. Every geometry field
below is read from ``config.json``; the defaults are MiMo-V2.5's, which is the
only reason they look specific:

  field                                 V2.5        V2.5-Pro
  hidden_size                           4096        6144
  num_hidden_layers (full + SWA)        48 (9+39)   70 (10+60)
  num_attention_heads (full and SWA)    64          128
  num_key_value_heads (full / SWA)      4 / 8       8 / 8
  n_routed_experts (top-8, I=2048)      256         384
  attention_value_scale                 0.707       0.612

Pro is ~1.02T parameters; its routed experts do not fit a trn2.48xlarge in
BF16, so the experts' runtime dtype is a mode (:meth:`resolve_expert_dtype`).

The V2.5 checkpoint is multimodal (it carries ``vision_config`` /
``audio_config`` / ``processor_config``; Pro carries none of them), and both
carry a 3-layer ``model.mtp.layers.*`` MTP head, but this port covers the TEXT
decoder only. The extra sub-configs are dropped by
the ``field_names`` filter in :meth:`MiMoV2Config.from_configs`, and the
corresponding checkpoint tensors are simply never referenced by
``load_weights`` (``load_sharded_pipelined`` only pulls the keys the model's
parameters map to).

Architecture (V2.5 numbers; 48 layers, 256 experts, hidden 4096):
  - HYBRID attention: ``hybrid_layer_pattern[i] == 0`` -> FULL attention,
    ``== 1`` -> SWA (window 128). Note the polarity: 0 is full, and the full
    layers are 0, 5, 11, 17, 23, 29, 35, 41, 47 (9 of 48).
  - Asymmetric head dims: Q/K ``head_dim=192``, V ``v_head_dim=128``.
  - Asymmetric KV heads: full layers 4 KV heads, SWA layers 8.
  - Partial RoPE: ``rope_dim = int(head_dim * partial_rotary_factor) = 64``
    (only the first 64 of 192 dims are rotated).
  - Dual RoPE base: ``rope_theta=1e7`` on full layers,
    ``swa_rope_theta=1e4`` on SWA layers.
  - Attention sink bias on SWA layers ONLY (``add_swa_attention_sink_bias``),
    one scalar per Q head (64 wide).
  - ``attention_value_scale=0.707`` applied to V BEFORE the KV-cache write.
  - MoE on layers 1..47 (``moe_layer_freq[0] == 0`` -> layer 0 is dense);
    sigmoid router with ``noaux_tc`` group top-k selection.
  - Plain RMSNorm (``weight * x``, no ``1 +`` fold), eps from
    ``layernorm_epsilon``.
"""

import json
import os
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

# Runtime dtype of the routed experts: "auto" (default), "bf16" or "fp8".
EXPERT_DTYPE_ENV = "MIMO_V2_5_EXPERT_DTYPE"
EXPERT_DTYPES = ("auto", "bf16", "fp8")
# "auto" picks fp8 when this rank's routed experts would exceed this many GiB
# in BF16. A trn2 logical core has 24 GiB of HBM. V2.5 at TP=64/EP=64 is
# 8.81 GiB (stays BF16); Pro is 29.11 GiB (cannot fit, goes fp8).
BF16_EXPERT_BUDGET_GIB = 12.0


@dataclass
class MiMoV2Config:
    # ── Backbone ──────────────────────────────────────────────────────────
    vocab_size: int = 152576
    hidden_size: int = 4096
    intermediate_size: int = 16384
    num_hidden_layers: int = 48
    max_position_embeddings: int = 1048576
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16

    # HF spells the norm epsilon ``layernorm_epsilon`` (not ``rms_norm_eps``).
    layernorm_epsilon: float = 1e-5

    # ── Full attention geometry ───────────────────────────────────────────
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int = 192
    v_head_dim: int = 128

    # ── SWA attention geometry ────────────────────────────────────────────
    swa_num_attention_heads: int = 64
    swa_num_key_value_heads: int = 8
    swa_head_dim: int = 192
    swa_v_head_dim: int = 128
    sliding_window: int = 128

    # ── Hybrid / RoPE / sink ──────────────────────────────────────────────
    # 0 == FULL attention, 1 == SWA (see module docstring).
    hybrid_layer_pattern: list[int] = field(default_factory=list)
    partial_rotary_factor: float = 0.334
    rope_theta: float = 1e7
    swa_rope_theta: float = 1e4
    attention_bias: bool = False
    attention_value_scale: float = 0.707
    add_full_attention_sink_bias: bool = False
    add_swa_attention_sink_bias: bool = True

    # ── MoE ───────────────────────────────────────────────────────────────
    # moe_layer_freq[i] truthy -> layer i is MoE, else dense MLP.
    moe_layer_freq: list[int] = field(default_factory=list)
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 256
    num_experts_per_tok: int = 8
    n_shared_experts: int | None = None
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    # HF stores ``null`` here and multiplies by 1.0; keep the same default.
    routed_scaling_factor: float | None = None

    # ── Framework ─────────────────────────────────────────────────────────
    neuron_config: NeuronConfig | None = None

    # The released base checkpoint is fp8-e4m3 blockwise (128x128) on disk with
    # ``dtype: bfloat16`` as the COMPUTE dtype. The weight loaders dequantize to
    # bf16 at load time, so runtime is pure BF16 and this field is informational
    # (kept so ``from_configs`` does not choke on the HF key and so the loaders
    # can read the block shape rather than hard-coding 128x128).
    quantization_config: dict | None = None

    # Shard count the fused-QKV weights are PRE-SHARDED into on disk (the
    # checkpoint index's ``metadata.tp_size``; 4 for the released base model).
    # This cannot be inferred from the SWA layers' shapes -- every section there
    # is already 128-divisible, so disk_tp 1, 2 and 4 all yield 116 scale rows
    # while laying the head rows out differently -- so it must be carried
    # explicitly. ``None`` means "infer per tensor", which works only for a
    # checkpoint whose every layer has a ceil-padded grid.
    qkv_disk_tp: int | None = None

    # Routed-expert runtime dtype, one of EXPERT_DTYPES. The env var
    # MIMO_V2_5_EXPERT_DTYPE overrides it; see resolve_expert_dtype. Not an HF
    # key, so it only reaches here through hf_overrides or that env var.
    expert_dtype: str = "auto"

    def __post_init__(self):
        if not self.hybrid_layer_pattern:
            # No pattern -> treat every layer as full attention.
            self.hybrid_layer_pattern = [0] * self.num_hidden_layers
        if not self.moe_layer_freq:
            self.moe_layer_freq = [1] * self.num_hidden_layers
        if len(self.hybrid_layer_pattern) != self.num_hidden_layers:
            raise ValueError(
                f"hybrid_layer_pattern has {len(self.hybrid_layer_pattern)} "
                f"entries but num_hidden_layers={self.num_hidden_layers}"
            )
        if len(self.moe_layer_freq) != self.num_hidden_layers:
            raise ValueError(
                f"moe_layer_freq has {len(self.moe_layer_freq)} entries but "
                f"num_hidden_layers={self.num_hidden_layers}"
            )

    # ── Derived helpers ───────────────────────────────────────────────────

    @property
    def rms_norm_eps(self) -> float:
        """Alias so generic framework code can read the norm eps."""
        return self.layernorm_epsilon

    @property
    def rope_dim(self) -> int:
        """Rotated width of each head (64 for the released config).

        HF: ``rope_dim = int(head_dim * partial_rotary_factor)``. Both the full
        and SWA variants have ``head_dim == 192``, so one value serves both.
        """
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def swa_rope_dim(self) -> int:
        return int(self.swa_head_dim * self.partial_rotary_factor)

    def is_swa_layer(self, layer_idx: int) -> bool:
        return self.hybrid_layer_pattern[layer_idx] == 1

    def is_moe_layer(self, layer_idx: int) -> bool:
        return bool(self.moe_layer_freq[layer_idx])

    @property
    def weight_block_size(self) -> tuple[int, int]:
        """FP8 blockwise quantization tile, ``(128, 128)`` for this release."""
        qc = self.quantization_config or {}
        wbs = qc.get("weight_block_size") or [128, 128]
        return int(wbs[0]), int(wbs[1])

    @property
    def is_fp8_checkpoint(self) -> bool:
        # V2.5 spells ``store_dtype: "fp8"``; Pro omits the key (same scheme).
        # V2.6 says ``"mxfp4"``: its routed experts are MXFP4, everything else
        # quantized is still 128x128 FP8, so it is an FP8 checkpoint too.
        qc = self.quantization_config or {}
        return qc.get("quant_method") == "fp8" and qc.get("store_dtype", "fp8") in (
            "fp8",
            "mxfp4",
        )

    @property
    def num_moe_layers(self) -> int:
        return sum(1 for i in range(self.num_hidden_layers) if self.is_moe_layer(i))

    def expert_bytes_per_rank(
        self, num_local_experts: int, intermediate_per_rank: int, bytes_per_elem: int
    ) -> int:
        """Weight bytes of this rank's routed experts over every MoE layer.

        ``gate_up [E_local, H, 2I]`` plus ``down [E_local, I, H]``, at
        ``bytes_per_elem`` (2 for bf16, 1 for fp8; fp8's per-channel scales are
        counted separately by :meth:`expert_scale_bytes_per_rank`).
        """
        per_expert = 3 * self.hidden_size * intermediate_per_rank
        return self.num_moe_layers * num_local_experts * per_expert * bytes_per_elem

    def expert_scale_bytes_per_rank(
        self, num_local_experts: int, intermediate_per_rank: int
    ) -> int:
        """fp32 scales of the fp8 layout: ``[E_local, 1, 2I]`` + ``[E_local, 1, H]``."""
        per_expert = 2 * intermediate_per_rank + self.hidden_size
        return self.num_moe_layers * num_local_experts * per_expert * 4

    def resolve_expert_dtype(
        self, num_local_experts: int, intermediate_per_rank: int
    ) -> str:
        """``"bf16"`` or ``"fp8"`` for the routed experts on this layout.

        Precedence: ``MIMO_V2_5_EXPERT_DTYPE`` env, then ``self.expert_dtype``,
        then the auto rule: fp8 iff the BF16 experts of one rank exceed
        ``BF16_EXPERT_BUDGET_GIB``. Pure (no logging), so every MoE layer and
        the backbone can call it and get the same answer.
        """
        mode = os.environ.get(EXPERT_DTYPE_ENV) or self.expert_dtype or "auto"
        mode = mode.strip().lower()
        if mode not in EXPERT_DTYPES:
            raise ValueError(
                f"{EXPERT_DTYPE_ENV}/expert_dtype must be one of {EXPERT_DTYPES}, "
                f"got {mode!r}"
            )
        if mode != "auto":
            return mode
        bf16_bytes = self.expert_bytes_per_rank(
            num_local_experts, intermediate_per_rank, 2
        )
        return "fp8" if bf16_bytes > BF16_EXPERT_BUDGET_GIB * 2**30 else "bf16"

    @property
    def fp8_ignored_layers(self) -> set[str]:
        """Module prefixes stored unquantized (bf16) despite the fp8 config.

        For this release that is every ``self_attn.o_proj``.
        """
        qc = self.quantization_config or {}
        return set(qc.get("ignored_layers") or [])

    # ── Construction ──────────────────────────────────────────────────────

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig | dict | str,
        neuron_config: NeuronConfig = None,
    ) -> "MiMoV2Config":
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            config_dict = hf_config.to_dict()
            if getattr(hf_config, "torch_dtype", None) is not None:
                config_dict["torch_dtype"] = hf_config.torch_dtype
        else:
            config_dict = dict(hf_config)

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in field_names}

        # HF 4.57 writes the compute dtype under ``dtype``; older exports use
        # ``torch_dtype``. Accept either.
        if "torch_dtype" not in filtered and "dtype" in config_dict:
            filtered["torch_dtype"] = config_dict["dtype"]
        if isinstance(filtered.get("torch_dtype"), str):
            filtered["torch_dtype"] = getattr(torch, filtered["torch_dtype"])

        # ``sliding_window_size`` is the alias the released config also carries;
        # prefer ``sliding_window`` but fall back so either spelling works.
        if "sliding_window" not in filtered and "sliding_window_size" in config_dict:
            filtered["sliding_window"] = config_dict["sliding_window_size"]

        if neuron_config is not None:
            filtered["neuron_config"] = neuron_config

        return cls(**filtered)
