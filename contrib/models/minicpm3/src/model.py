# SPDX-License-Identifier: Apache-2.0
"""
MiniCPM3 BF16 Implementation
====================================

Implementation of MiniCPM3 (openbmb/MiniCPM3-4B) for the Neuron backend.
Adapted from the plugin's dense BF16 models (Llama 3 and the Qwen3-VL text
decoder; TP + SP).

MiniCPM3 uses multi-head latent attention (MLA):

    latent_q = q_a_layernorm(q_a_proj(x))                  [q_lora_rank]
    c, k_pe  = split(kv_a_proj_with_mqa(x))                [kv_lora_rank], [rope]
    c        = kv_a_layernorm(c)
    q        = q_b_proj(latent_q)       per head: [nope, rope]
    k_nope, v = kv_b_proj(c)            per head: [nope], [v]
    k        = [k_nope, k_pe]           k_pe is shared by all heads
    q_rope, k_pe get RoPE (rotate_half, LongRoPE frequencies)

This port maps MLA onto the standard fused Neuron attention kernels:

1. A replicated "latent" projection (NF.qkv_proj) computes
   [latent_q, c, k_pe] from x in one matmul. The torch code applies the two
   low-rank RMSNorms. The result X' has width ``latent_dim`` (padded to 256).
2. One fused W_qkv [latent_dim, 3 * heads * d_head] maps X' to per-head
   Q, K and V. W_qkv holds q_b_proj, kv_b_proj and an identity copy of k_pe.
   Thus the per-head K and V are standard multi-head K and V, and the KV
   cache holds them as usual (kv_heads == q_heads).
3. The head dimension is padded to ``d_head`` = 128. The permuted layout puts
   the two rotary halves at [0, rope/2) and [64, 64 + rope/2). Thus the
   rotate_half pairs (i, i + 64) of a full-width RoPE rotate the rotary part
   and leave the other channels unchanged (cos = 1, sin = 0). The zero pad
   channels do not change Q.K or the output, because the O projection has
   zero rows at the pad channels. The softmax scale stays
   (qk_nope + qk_rope) ** -0.5.

Prefill uses NF.qkv_proj, NF.flash_attention (or NF.segmented_attention) and
NF.o_proj. Decode uses the fused NF.attention_decode megakernel with a
per-head block table, as in Qwen3-VL.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code.
  # <-- MODEL-SPECIFIC: ...    MiniCPM3-specific.

PARALLELISM SHARDING:
  TP: Attention heads sharded (Q, K, V, O). The latent projection and the
      low-rank norms are replicated. The MLP intermediate dim is sharded.
      The embedding and the LM head are sharded on the vocab dim.
"""

import logging
import math
from dataclasses import dataclass

import torch
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
from vllm_neuron.functional.attention.attention_decode import (
    _can_use_attention_block_kernel,
)
from vllm_neuron.model.kv_cache import (
    KVSpec,
    LayerSpec,
    build_per_head_slot_mapping,
    paged_slot_indices,
)
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    set_weight_loader,
    sharding_weight_loader,
)

from transformers import PretrainedConfig
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.sampler import Sampler

import vllm_neuron.nn as neuron_nn
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding

from .checkpoints import SafetensorsCheckpoint
from .compat import build_per_head_block_table
from .config import MiniCPM3Config

logger = logging.getLogger(__name__)

# The Neuron partition count. The fused attention kernels need a hidden dim
# that is a multiple of this value, and a head dim that is not larger.
_PMAX = 128
# The decode RoPE kernel writes the second rotary half at partition offset
# d_head // 2. This offset must be a multiple of 32.
_PARTITION_ALIGN = 32


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


# =============================================================================
# Section 1: Head layout of the fused MLA projection
# <-- MODEL-SPECIFIC
# =============================================================================


@dataclass(frozen=True)
class MiniCPM3HeadLayout:
    """Channel layout of one padded attention head and of the latent input X'.

    Attributes:
        d_head: Padded head dimension of Q, K and V in the KV cache.
        half_d: d_head // 2. RoPE pairs channel i with channel i + half_d.
        q_index: [qk_nope + qk_rope] Destination channel of each HF Q/K channel.
            HF order per head is [nope, rope].
        rope_index: [qk_rope] Destination channel of each HF rotary channel.
        nope_index: [qk_nope] Destination channel of each HF non-rotary channel.
        v_dim: Number of real V channels. They go to channels [0, v_dim).
        q_lora_rank: Width of latent_q in X'.
        kv_lora_rank: Width of c in X'.
        qk_rope_head_dim: Width of k_pe in X'.
        latent_dim: Padded width of X' (multiple of 256).
    """

    d_head: int
    half_d: int
    q_index: torch.Tensor
    rope_index: torch.Tensor
    nope_index: torch.Tensor
    v_dim: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    latent_dim: int

    @property
    def c_offset(self) -> int:
        """Return the start of c in X'."""
        return self.q_lora_rank

    @property
    def k_pe_offset(self) -> int:
        """Return the start of k_pe in X'."""
        return self.q_lora_rank + self.kv_lora_rank


def minicpm3_head_layout(config: MiniCPM3Config) -> MiniCPM3HeadLayout:
    """Compute the padded head layout and the latent layout for MiniCPM3.

    Put the first rotary half at [0, rope/2) and the second rotary half at
    [half_d, half_d + rope/2). Put the non-rotary channels after each rotary
    half. The value of half_d is a multiple of 32 that holds one rotary half
    and one half of the non-rotary channels.

    Args:
        config: MiniCPM3 config.

    Returns:
        The head layout.

    Example:
        >>> layout = minicpm3_head_layout(MiniCPM3Config())
        >>> layout.d_head, layout.latent_dim
        (128, 1280)
    """
    nope = config.qk_nope_head_dim
    rope = config.qk_rope_head_dim
    if rope % 2 != 0 or nope % 2 != 0:
        raise ValueError(
            f"qk_rope_head_dim ({rope}) and qk_nope_head_dim ({nope}) must be even."
        )
    rope_half = rope // 2
    nope_half = nope // 2
    half_d = _round_up(rope_half + nope_half, _PARTITION_ALIGN)
    half_d = max(half_d, _round_up((config.v_head_dim + 1) // 2, _PARTITION_ALIGN))
    d_head = 2 * half_d
    if d_head > _PMAX:
        raise ValueError(
            f"The padded MLA head dim ({d_head}) is larger than {_PMAX}. "
            f"This port supports qk_nope_head_dim + qk_rope_head_dim <= 128."
        )

    # The server builds the model under a meta device context. Put the index
    # tensors on the CPU explicitly, because the weight builders read them.
    rope_index = torch.tensor(
        [r if r < rope_half else half_d + (r - rope_half) for r in range(rope)],
        dtype=torch.long,
        device="cpu",
    )
    nope_index = torch.tensor(
        [
            rope_half + j if j < nope_half else half_d + rope_half + (j - nope_half)
            for j in range(nope)
        ],
        dtype=torch.long,
        device="cpu",
    )
    q_index = torch.cat([nope_index, rope_index])
    # The decode megakernel splits X' into 128-row tiles over the 2 cores of
    # LNC2. Use an even tile count, so pad X' to a multiple of 2 * 128.
    latent_dim = _round_up(config.q_lora_rank + config.kv_lora_rank + rope, 2 * _PMAX)
    return MiniCPM3HeadLayout(
        d_head=d_head,
        half_d=half_d,
        q_index=q_index,
        rope_index=rope_index,
        nope_index=nope_index,
        v_dim=config.v_head_dim,
        q_lora_rank=config.q_lora_rank,
        kv_lora_rank=config.kv_lora_rank,
        qk_rope_head_dim=rope,
        latent_dim=latent_dim,
    )


def build_latent_proj_weight(
    q_a_weight: torch.Tensor,
    kv_a_weight: torch.Tensor,
    layout: MiniCPM3HeadLayout,
) -> torch.Tensor:
    """Build the replicated latent projection weight [H, latent_dim].

    Args:
        q_a_weight: HF q_a_proj.weight [q_lora_rank, H].
        kv_a_weight: HF kv_a_proj_with_mqa.weight [kv_lora_rank + rope, H].
        layout: Head layout.

    Returns:
        Weight in [in, out] storage. Columns past k_pe are zero.

    Example:
        >>> w = build_latent_proj_weight(q_a, kv_a, minicpm3_head_layout(cfg))
    """
    hidden = q_a_weight.shape[1]
    out = torch.zeros(
        hidden, layout.latent_dim, dtype=q_a_weight.dtype, device=q_a_weight.device
    )
    out[:, : layout.q_lora_rank] = q_a_weight.t()
    out[:, layout.c_offset : layout.k_pe_offset + layout.qk_rope_head_dim] = (
        kv_a_weight.t()
    )
    return out


def build_fused_qkv_weight(
    q_b_weight: torch.Tensor,
    kv_b_weight: torch.Tensor,
    layout: MiniCPM3HeadLayout,
    num_heads: int,
    qk_nope_head_dim: int,
) -> torch.Tensor:
    """Build the fused per-rank W_qkv [latent_dim, 3 * num_heads * d_head].

    The rows of q_b_weight and kv_b_weight must hold the heads of this rank.

    Args:
        q_b_weight: HF q_b_proj.weight rows [num_heads * (nope + rope), q_lora].
        kv_b_weight: HF kv_b_proj.weight rows [num_heads * (nope + v), kv_lora].
        layout: Head layout.
        num_heads: Number of heads of this rank.
        qk_nope_head_dim: Non-rotary Q/K head dim.

    Returns:
        Weight in [in, out] storage. The output columns are [Q | K | V], each
        [num_heads, d_head].

    Example:
        >>> w = build_fused_qkv_weight(q_b, kv_b, layout, num_heads=10, qk_nope_head_dim=64)
    """
    dtype = q_b_weight.dtype
    device = q_b_weight.device
    d = layout.d_head
    rows = layout.latent_dim
    c0, c1 = layout.c_offset, layout.k_pe_offset
    q_lora = layout.q_lora_rank

    q_b = q_b_weight.reshape(num_heads, -1, q_lora)  # [nh, nope + rope, q_lora]
    kv_b = kv_b_weight.reshape(num_heads, -1, layout.kv_lora_rank)  # [nh, nope+v, kv]

    q_index = layout.q_index.to(device)
    nope_index = layout.nope_index.to(device)
    rope_index = layout.rope_index.tolist()

    q_sec = torch.zeros(rows, num_heads, d, dtype=dtype, device=device)
    q_sec[:q_lora, :, q_index] = q_b.permute(2, 0, 1)

    k_sec = torch.zeros(rows, num_heads, d, dtype=dtype, device=device)
    k_sec[c0:c1, :, nope_index] = kv_b[:, :qk_nope_head_dim, :].permute(2, 0, 1)
    # k_pe is shared by all heads: copy each rotary channel of X' into each head.
    for r in range(layout.qk_rope_head_dim):
        k_sec[c1 + r, :, rope_index[r]] = 1.0

    v_sec = torch.zeros(rows, num_heads, d, dtype=dtype, device=device)
    v_sec[c0:c1, :, : layout.v_dim] = kv_b[:, qk_nope_head_dim:, :].permute(2, 0, 1)

    return torch.cat(
        [
            q_sec.reshape(rows, num_heads * d),
            k_sec.reshape(rows, num_heads * d),
            v_sec.reshape(rows, num_heads * d),
        ],
        dim=1,
    )


def build_o_proj_weight(
    o_weight_cols: torch.Tensor, layout: MiniCPM3HeadLayout, num_heads: int
) -> torch.Tensor:
    """Build the padded per-rank O projection weight [num_heads * d_head, H].

    Args:
        o_weight_cols: HF o_proj.weight columns of this rank [H, num_heads * v].
        layout: Head layout.
        num_heads: Number of heads of this rank.

    Returns:
        Weight in [in, out] storage, with zero rows at the pad channels.

    Example:
        >>> w = build_o_proj_weight(o_cols, layout, num_heads=10)
    """
    hidden = o_weight_cols.shape[0]
    w = o_weight_cols.t().reshape(num_heads, layout.v_dim, hidden)
    out = torch.zeros(
        num_heads,
        layout.d_head,
        hidden,
        dtype=o_weight_cols.dtype,
        device=o_weight_cols.device,
    )
    out[:, : layout.v_dim, :] = w
    return out.reshape(num_heads * layout.d_head, hidden)


# =============================================================================
# Section 2: RMS Normalization
# <-- MODEL-SPECIFIC: HF MiniCPM3RMSNorm (fp32 math, cast, then weight multiply)
# =============================================================================


class MiniCPM3RMSNorm(nn.Module):
    """RMS normalization with the HF MiniCPM3 cast order."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        # Cast to the weight dtype (bf16). The residual stream is fp32, and the
        # projections and the collectives after the norm use bf16.
        return self.weight * x.to(self.weight.dtype)


# =============================================================================
# Section 3: Rotary Position Embedding (LongRoPE in the padded head layout)
# <-- MODEL-SPECIFIC
# =============================================================================


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotate_half RoPE. x: [N, T, D], cos/sin: [T, D]."""
    return x * cos.unsqueeze(0) + _rotate_half(x) * sin.unsqueeze(0)


class MiniCPM3RotaryEmbedding(nn.Module):
    """LongRoPE for the rotary channels of the padded head layout.

    The forward pass returns half-width cos/sin [T, half_d]. Entries
    [0, rope/2) hold the LongRoPE values. The other entries hold cos = 1 and
    sin = 0, so RoPE does not change the non-rotary channels.
    """

    def __init__(self, config: MiniCPM3Config, layout: MiniCPM3HeadLayout):
        super().__init__()
        rope = config.qk_rope_head_dim
        self.rope_half = rope // 2
        self.half_d = layout.half_d
        self.rope_theta = config.rope_theta

        ext_factors = None
        attention_factor = 1.0
        if config.rope_short_factor is not None:
            original = config.rope_original_max_position_embeddings
            if (
                config.max_position_embeddings > original
                and config.rope_long_factor != config.rope_short_factor
            ):
                raise NotImplementedError(
                    "MiniCPM3 LongRoPE with different long and short factors is "
                    "not supported. Set max_position_embeddings <= "
                    "original_max_position_embeddings."
                )
            ext_factors = list(config.rope_short_factor)
            factor = config.max_position_embeddings / original
            if factor > 1.0:
                attention_factor = math.sqrt(1 + math.log(factor) / math.log(original))
        self.ext_factors = ext_factors
        self.attention_factor = attention_factor

        # Compute the frequencies once on the CPU and keep them as buffers.
        # The model builds under a meta device context, so give the device
        # explicitly. A constant that the forward pass makes from a Python
        # list does not give correct values on the device.
        # A pad entry has frequency 0 and scale 1, so it gives cos = 1, sin = 0.
        inv_freq = torch.zeros(self.half_d, dtype=torch.float32, device="cpu")
        inv_freq[: self.rope_half] = self._compute_inv_freq(torch.device("cpu"))
        scale = torch.ones(self.half_d, dtype=torch.float32, device="cpu")
        scale[: self.rope_half] = attention_factor
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("rope_scale", scale, persistent=False)

    def _compute_inv_freq(self, device: torch.device) -> torch.Tensor:
        rope = 2 * self.rope_half
        base = self.rope_theta ** (
            torch.arange(0, rope, 2, dtype=torch.int64, device=device).float() / rope
        )
        if self.ext_factors is not None:
            ext = torch.tensor(self.ext_factors, dtype=torch.float32, device=device)
            return 1.0 / (ext * base)
        return 1.0 / base

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute half-width cos/sin for the given positions.

        Returns:
            cos, sin: both of shape [T, half_d]
        """
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)
        scale = self.rope_scale.to(device=device, dtype=torch.float32)
        freqs = position_ids.to(device=device).float()[:, None] * inv_freq[None, :]
        cos = freqs.cos() * scale[None, :]
        sin = freqs.sin() * scale[None, :]
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


# =============================================================================
# Section 4: Attention (MLA mapped onto fused multi-head kernels)
# =============================================================================


class MiniCPM3Attention(nn.Module):
    """MLA attention with TP head sharding.

    >>> PARALLELISM: TP <<<
    - The latent projection and the low-rank norms are replicated.
    - Q/K/V/O heads are sharded across TP ranks (kv_heads == q_heads).
    - Prefill: latent proj on the SP slice -> all-gather X' -> QKV proj ->
      flash attention -> O proj -> reduce-scatter.
    - Decode: latent proj -> fused megakernel -> TP all-reduce.
    """

    def __init__(self, config: MiniCPM3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.scaling = config.qk_head_dim**-0.5

        self.layout = minicpm3_head_layout(config)
        self.head_dim = self.layout.d_head
        self.latent_dim = self.layout.latent_dim

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        if self.num_attention_heads % self.world_size != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) is not divisible "
                f"by the TP size ({self.world_size}). Use a TP size that divides it."
            )
        self.num_attention_heads_per_rank = self.num_attention_heads // self.world_size
        # MLA expands K and V per head, so the KV head count equals the Q head count.
        self.num_key_value_heads_per_rank = self.num_attention_heads_per_rank

        size_per_rank = self.num_attention_heads_per_rank * self.head_dim
        self.qkv_split_indices = [size_per_rank, 2 * size_per_rank]

        # <-- MODEL-SPECIFIC: replicated latent projection and low-rank norms
        self.latent_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.latent_dim, dtype=self.dtype)
        )
        self.q_a_layernorm = MiniCPM3RMSNorm(config.q_lora_rank, 1e-6, self.dtype)
        self.kv_a_layernorm = MiniCPM3RMSNorm(config.kv_lora_rank, 1e-6, self.dtype)

        # >>> PARALLELISM: per-rank fused QKV and O projection <<<
        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.latent_dim, 3 * size_per_rank, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(size_per_rank, self.hidden_size, dtype=self.dtype)
        )

        # KV caches bound externally via bind_kv_cache()
        self.k_cache = None
        self.v_cache = None

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """Attach the loaders that build the fused MLA weights from the checkpoint."""
        layout = self.layout
        nh = self.num_attention_heads_per_rank
        nope = self.qk_nope_head_dim
        q_rows = nope + layout.qk_rope_head_dim
        kv_rows = nope + layout.v_dim

        def latent_transform(slices, rank):
            return build_latent_proj_weight(slices[0][:], slices[1][:], layout)

        def qkv_transform(slices, rank):
            q_b = slices[0][rank * nh * q_rows : (rank + 1) * nh * q_rows, :]
            kv_b = slices[1][rank * nh * kv_rows : (rank + 1) * nh * kv_rows, :]
            return build_fused_qkv_weight(q_b, kv_b, layout, nh, nope)

        def o_transform(slices, rank):
            cols = slices[0][
                :, rank * nh * layout.v_dim : (rank + 1) * nh * layout.v_dim
            ]
            return build_o_proj_weight(cols, layout, nh)

        set_weight_loader(
            self.latent_proj_weight, SafetensorsWeightLoader(latent_transform)
        )
        set_weight_loader(self.qkv_proj_weight, SafetensorsWeightLoader(qkv_transform))
        set_weight_loader(self.o_proj_weight, SafetensorsWeightLoader(o_transform))

    # -- Latent input X' -------------------------------------------------------

    def compute_latent(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute X' = [q_a_norm(latent_q), kv_a_norm(c), k_pe, 0] for [T, H] input."""
        latent = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.latent_proj_weight,
            bias=None,
        ).squeeze(0)
        c0, c1 = self.layout.c_offset, self.layout.k_pe_offset
        latent_q = self.q_a_layernorm(latent[:, :c0])
        c = self.kv_a_layernorm(latent[:, c0:c1])
        return torch.cat([latent_q, c, latent[:, c1:]], dim=-1)

    # -- Forward dispatch -----------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ):
        """Dispatch to prefill or decode path based on metadata."""
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]

        hidden_states = hidden_states.to(self.dtype)
        x_latent = self.compute_latent(hidden_states)

        if max_query_len <= decode_token_threshold:
            return self.forward_decode(
                x_latent, positions, position_embeddings, attn_metadata
            )

        # >>> PARALLELISM: All-gather X' from SP before attention <<<
        # X' is narrower than the hidden states, so gather X' and not the input.
        if self.world_size > 1:
            x_latent = self.tp_group.all_gather(x_latent, dim=0)
        return self.forward_prefill(
            x_latent, positions, position_embeddings, attn_metadata
        )

    # -- Prefill path ---------------------------------------------------------

    def forward_prefill(
        self,
        x_latent: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        """Prefill: fused QKV projection, RoPE, cache write and flash attention."""
        tokens = x_latent.shape[0]
        nh = self.num_attention_heads_per_rank

        # -- Step 1: QKV projection (NKI qkv kernel) --------------------------
        qkv = NF.qkv_proj(
            hidden=x_latent.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,
        ).squeeze(0)
        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
        q = q.view(tokens, nh, self.head_dim).transpose(0, 1)
        k = k.view(tokens, nh, self.head_dim).transpose(0, 1)
        v = v.view(tokens, nh, self.head_dim).transpose(0, 1)

        # -- Step 2: RoPE on the full padded head -----------------------------
        cos_half, sin_half = position_embeddings
        cos = torch.cat([cos_half, cos_half], dim=-1)
        sin = torch.cat([sin_half, sin_half], dim=-1)
        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)

        # -- Step 3: Update KV cache ------------------------------------------
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
        kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

        # The runner marks a pad slot with -1. Send each pad write to the null
        # block, so that the scatter stays in range on the device.
        block_indices, position_indices = paged_slot_indices(
            slot_mapping, block_size, self.k_cache.shape[0]
        )

        k_flat = k.reshape(-1, self.head_dim).to(self.k_cache.dtype)
        v_flat = v.reshape(-1, self.head_dim).to(self.v_cache.dtype)

        head_indices_for_put = torch.arange(
            nh, dtype=torch.long, device=x_latent.device
        ).repeat_interleave(slot_mapping.shape[0])
        block_indices_for_put = block_indices.repeat(nh)
        position_indices_for_put = position_indices.repeat(nh)

        self.k_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            k_flat,
        )
        self.v_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            v_flat,
        )

        # -- Step 4: Attention (NKI attention_cte / segmented kernels) --------
        if kv_segment_size:
            attn_output = NF.segmented_attention(
                q,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                block_tables=block_table,
                prior_tokens=cached_seq_len,
                block_size=block_size,
                kv_segment_size=kv_segment_size,
                scale=self.scaling,
                tp_q=True,
                tp_out=True,
            )  # [Nh, D, T]
        else:
            attn_output = NF.flash_attention(
                q.transpose(1, 2),  # [Nh, D, T]
                k.transpose(1, 2),  # [Nh, D, T]
                v,  # [Nh, T, D]
                scale=self.scaling,
                tp_q=False,
                tp_out=True,
            )  # [Nh, D, T]

        # -- Step 5: Output projection (NKI output_projection kernel) ---------
        attn_output = NF.o_proj(attn_output.unsqueeze(0), self.o_proj_weight, None)
        attn_output = attn_output.squeeze(0)  # [T, H]

        # >>> PARALLELISM: Reduce-scatter to return to SP layout <<<
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        return attn_output.contiguous()

    # -- Decode path ----------------------------------------------------------

    def forward_decode(
        self,
        x_latent: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object,
    ):
        """Decode: fused attention megakernel (QKV + RoPE + attention + O proj)."""
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        B = block_table.shape[0]
        tokens = x_latent.shape[0]
        S_decode = tokens // B
        assert tokens == B * S_decode
        nkh = self.num_key_value_heads_per_rank

        X = x_latent.view(B, S_decode, self.latent_dim)

        # RoPE for the megakernel: [T, half_d] -> [half_d, B, S]
        cos_half, sin_half = position_embeddings
        half_d = self.layout.half_d
        cos_kernel = (
            cos_half.view(B, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )
        sin_kernel = (
            sin_half.view(B, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )

        pos_ids_kernel = positions.view(B, S_decode).to(torch.float32)

        # Pass the 4D cache [num_blocks, nkh, block_size, d_head]. The kernel
        # needs a per-head 3D block table [B, nkh, num_blocks] and per-head
        # write slots. The torch fallback accepts only the shared 2D table and
        # the plain slot mapping. Use the same gate as NF.attention_decode, so
        # that the two inputs agree with the path that the wrapper selects.
        # The kernel reads the write slots as uint32. vLLM's slot mapping is
        # int64, and an int64 slot tensor makes the runtime reject the decode
        # graph.
        block_table = block_table.to(torch.int32)
        per_head_table = build_per_head_block_table(block_table, nkh)
        if _can_use_attention_block_kernel(
            X=X, V_cache=self.v_cache, active_blocks_table=per_head_table
        ):
            active_blocks_table = per_head_table
            kv_cache_update_idx = build_per_head_slot_mapping(
                slot_mapping, block_size, nkh, B
            ).to(torch.uint32)
        else:
            active_blocks_table = block_table
            kv_cache_update_idx = slot_mapping.view(B, S_decode).to(torch.uint32)

        output = NF.attention_decode(
            X=X,
            W_qkv=self.qkv_proj_weight,
            rmsnorm_QK_pre_rope_enabled=False,
            rmsnorm_QK_post_rope_enabled=False,
            cos=cos_kernel,
            sin=sin_kernel,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=active_blocks_table,
            K_cache=self.k_cache,
            V_cache=self.v_cache,
            pos_ids=pos_ids_kernel,
            swa_start_pos_ids=None,
            softmax_scale=self.scaling,
            sink=None,
            update_cache=True,
            kv_cache_update_idx=kv_cache_update_idx,
            W_out=self.o_proj_weight,
            bias_out=None,
            transposed_out=False,
            out_in_sb=False,
        )

        # >>> PARALLELISM: TP all-reduce after megakernel <<<
        if self.world_size > 1:
            self.tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 5: MLP (Dense SwiGLU)
# =============================================================================


class MiniCPM3MLP(nn.Module):
    """Dense SwiGLU MLP with TP intermediate sharding.

    >>> PARALLELISM: TP <<<
    - Prefill: all-gather before -> compute -> reduce-scatter after (SP)
    - Decode: compute -> all-reduce
    """

    def __init__(self, config: MiniCPM3Config):
        super().__init__()

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        if config.intermediate_size % self.world_size != 0:
            raise ValueError(
                f"intermediate_size ({config.intermediate_size}) is not divisible "
                f"by the TP size ({self.world_size}). Use a TP size that divides it."
            )
        self.hidden_size = config.hidden_size
        self.intermediate_size_per_rank = config.intermediate_size // self.world_size

        self.gate_proj_weight = nn.Parameter(
            torch.empty(
                config.hidden_size,
                self.intermediate_size_per_rank,
                dtype=config.torch_dtype,
            )
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(
                config.hidden_size,
                self.intermediate_size_per_rank,
                dtype=config.torch_dtype,
            )
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(
                self.intermediate_size_per_rank,
                config.hidden_size,
                dtype=config.torch_dtype,
            )
        )

        gate_up_loader = sharding_weight_loader(
            shard_dim=1,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.world_size,
            is_storage_transposed=True,
        )
        down_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.world_size,
            is_storage_transposed=True,
        )
        set_weight_loader(self.gate_proj_weight, gate_up_loader)
        set_weight_loader(self.up_proj_weight, gate_up_loader)
        set_weight_loader(self.down_proj_weight, down_loader)

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        # >>> PARALLELISM: All-gather from SP for full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        # <-- MODEL-SPECIFIC: down(silu(gate(x)) * up(x)) through the NKI MLP kernel
        output = NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
        )

        # >>> PARALLELISM: Combine TP shards <<<
        if self.world_size > 1:
            if is_prefill:
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                self.tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 6: Decoder Layer
# <-- MODEL-SPECIFIC: muP residual scale scale_depth / sqrt(num_layers)
# =============================================================================


class MiniCPM3DecoderLayer(nn.Module):
    """x -> RMSNorm -> MLA -> scaled residual -> RMSNorm -> MLP -> scaled residual."""

    def __init__(self, config: MiniCPM3Config, layer_idx: int):
        super().__init__()
        self.input_layernorm = MiniCPM3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = MiniCPM3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.self_attn = MiniCPM3Attention(config, layer_idx=layer_idx)
        self.mlp = MiniCPM3MLP(config)
        self.layer_idx = layer_idx
        self.residual_scale = config.residual_scale

    def _is_decode(self, attn_metadata) -> bool:
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]
        return max_query_len <= decode_token_threshold

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        is_decode = self._is_decode(attn_metadata)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
        )
        # Keep the residual stream in fp32. The muP residual scale makes each
        # branch small against the residual, so a bf16 add loses the low bits
        # of the branch in each of the 2 * num_layers adds.
        hidden_states = residual + hidden_states.float() * self.residual_scale

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, is_prefill=not is_decode)
        hidden_states = residual + hidden_states.float() * self.residual_scale

        return hidden_states


# =============================================================================
# Section 7: Model Backbone
# =============================================================================


class MiniCPM3Model(nn.Module):
    """MiniCPM3 transformer backbone.

    >>> PARALLELISM: SP (Sequence Parallelism) <<<
    During prefill each TP rank holds T / world_size tokens between the
    layers. After the final norm, an all-gather gives the full sequence.
    """

    def __init__(self, config: MiniCPM3Config):
        super().__init__()
        self.config = config

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        if config.vocab_size % self.world_size != 0:
            raise ValueError(
                f"vocab_size ({config.vocab_size}) is not divisible by the TP size "
                f"({self.world_size}). Use a TP size that divides it."
            )

        # >>> PARALLELISM: Vocab-sharded embedding <<<
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )
        self.scale_emb = config.scale_emb

        self.layers = nn.ModuleList(
            [
                MiniCPM3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = MiniCPM3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = MiniCPM3RotaryEmbedding(
            config, self.layers[0].self_attn.layout
        )

        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=False,
            ),
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        # >>> PARALLELISM: VocabDimShardedEmbedding handles SP internally <<<
        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=is_prefill, rank=rank
        )
        # <-- MODEL-SPECIFIC: muP embedding scale. The residual stream starts
        # here, so apply the scale in fp32.
        hidden_states = hidden_states.float() * self.scale_emb

        # RoPE uses the positions of all tokens (MLA gathers X' before RoPE).
        position_embeddings = self.rotary_emb(
            positions, device=hidden_states.device, dtype=self.config.torch_dtype
        )

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
            )

        hidden_states = self.norm(hidden_states)

        # >>> PARALLELISM: SP - all-gather to reconstruct full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return hidden_states


# =============================================================================
# Section 8: Language Model Head
# <-- MODEL-SPECIFIC: tied embeddings and muP logit scale
# =============================================================================


class MiniCPM3ForCausalLM(nn.Module):
    """MiniCPM3 model with a column-parallel LM head (tied to the embedding)."""

    def __init__(self, config: MiniCPM3Config):
        super().__init__()
        self.config = config
        self.model = MiniCPM3Model(config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        self.logits_scaling = config.logits_scaling

        self.on_device_sampling_config = (
            config.neuron_config.on_device_sampling_config
            if config.neuron_config
            else None
        )
        debug_logits_enabled = (
            config.neuron_config is not None
            and config.neuron_config.debug_logits_dir is not None
        )
        self._gather_logits = (
            config.neuron_config is not None and config.neuron_config.max_logprobs != 0
        ) or debug_logits_enabled

        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=config.vocab_size // self.world_size,
                num_shards=self.world_size,
                is_storage_transposed=False,
            ),
        )

        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=self.tp_group.device_group,
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is not None:
            raise ValueError(
                "MiniCPM3 does not support input embeddings. Send token IDs."
            )

        positions = positions.to(torch.int32)

        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        T = input_ids.shape[0]

        # >>> PARALLELISM: SP length validation <<<
        if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(
                f"Prompt Length ({T}) must be > world_size ({self.world_size}) for SP."
            )

        hidden_states = self.model(
            input_ids, positions, attn_metadata=attn_metadata, rank=rank
        )

        hidden_states_for_logits = torch.index_select(
            hidden_states, dim=0, index=sampling_positions
        )
        # <-- MODEL-SPECIFIC: muP logit scale (hidden_size / dim_model_base)
        hidden_states_for_logits = hidden_states_for_logits / self.logits_scaling

        logits = self.lm_head(hidden_states_for_logits)

        if self.on_device_sampling_config is None:
            return logits

        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )

        # >>> PARALLELISM: Gather sharded logits for logprobs computation <<<
        gathered_logits = None
        if self._gather_logits:
            if self.tp_group is not None:
                gathered_logits = self.tp_group.all_gather(logits, dim=1)
            else:
                gathered_logits = logits

        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler

            return rejection_sampler(spec_decode_metadata, sampled_tokens)

        return sampled_tokens, gathered_logits

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = MiniCPM3Config.from_configs(hf_config, neuron_config)
        return cls(config)

    # -- KV Cache Management --------------------------------------------------
    # >>> PARALLELISM: KV spec uses per-rank head counts (TP-sharded) <<<

    def get_kv_spec(self):
        layers = []
        for i, layer in enumerate(self.model.layers):
            layers.append(
                LayerSpec(
                    name=f"layers.{i}.self_attn",
                    num_kv_heads=layer.self_attn.num_key_value_heads_per_rank,
                    head_size=layer.self_attn.head_dim,
                    dtype=layer.self_attn.dtype,
                    sliding_window_size=None,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor, torch.Tensor]]):
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            if layer_name not in kv_caches:
                raise RuntimeError(
                    f"KV cache for layer {layer_name} is not initialized. "
                    f"Allocate the KV cache before the first forward pass."
                )
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]

    # -- Weight Loading -------------------------------------------------------

    def checkpoint_mappings(self) -> dict[str, str | list[str]]:
        """Return the parameter name -> checkpoint key(s) map of this model."""
        mappings: dict[str, str | list[str]] = {}
        for layer_id in range(len(self.model.layers)):
            prefix = f"model.layers.{layer_id}"
            attn = f"{prefix}.self_attn"
            mappings[f"{attn}.latent_proj_weight"] = [
                f"{attn}.q_a_proj.weight",
                f"{attn}.kv_a_proj_with_mqa.weight",
            ]
            mappings[f"{attn}.qkv_proj_weight"] = [
                f"{attn}.q_b_proj.weight",
                f"{attn}.kv_b_proj.weight",
            ]
            mappings[f"{attn}.o_proj_weight"] = f"{attn}.o_proj.weight"
            mappings[f"{attn}.q_a_layernorm.weight"] = f"{attn}.q_a_layernorm.weight"
            mappings[f"{attn}.kv_a_layernorm.weight"] = f"{attn}.kv_a_layernorm.weight"
            mappings[f"{prefix}.input_layernorm.weight"] = (
                f"{prefix}.input_layernorm.weight"
            )
            mappings[f"{prefix}.post_attention_layernorm.weight"] = (
                f"{prefix}.post_attention_layernorm.weight"
            )
            mappings[f"{prefix}.mlp.gate_proj_weight"] = (
                f"{prefix}.mlp.gate_proj.weight"
            )
            mappings[f"{prefix}.mlp.up_proj_weight"] = f"{prefix}.mlp.up_proj.weight"
            mappings[f"{prefix}.mlp.down_proj_weight"] = (
                f"{prefix}.mlp.down_proj.weight"
            )
        mappings["model.embed_tokens.weight"] = "model.embed_tokens.weight"
        mappings["model.norm.weight"] = "model.norm.weight"
        # Tied embeddings: load lm_head from the embed_tokens checkpoint key.
        mappings["lm_head.weight"] = (
            "model.embed_tokens.weight"
            if self.config.tie_word_embeddings
            else "lm_head.weight"
        )
        return mappings

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load weights from the checkpoint (.safetensors or pytorch_model*.bin).

        The published MiniCPM3-4B checkpoint has only ``pytorch_model.bin``.
        The plugin's 0.29 loader reads only ``.safetensors`` files, so this
        model uses the loader in ``checkpoints.py`` of this folder, which adds
        the ``pytorch_model*.bin`` fallback.
        """
        mappings = self.checkpoint_mappings()
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, mappings, device
        ).state_dict

        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        self.load_state_dict(rank_sharded, strict=False, assign=True)
