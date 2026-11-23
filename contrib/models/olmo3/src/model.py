# SPDX-License-Identifier: Apache-2.0
"""
OLMo-3 BF16 Implementation
==========================

Annotated implementation of OLMo-3 for the native Neuron backend.
Adapted from the Qwen3 dense template.

Supported parallelism: TP, SP.
Reference model: allenai/Olmo-3-7B-Instruct (dense, 32Q/32KV MHA, head_dim=128).

Differences from Qwen3:
  - Post-norm decoder layer: ``h = h + norm(attn(h))``, then
    ``h = h + norm(mlp(h))``. No input norm exists.
  - q_norm and k_norm are RMSNorm over the full projection width (4096), not
    over one head. Under TP each rank holds a slice of that width, so the sum
    of squares goes through one TP all-reduce.
  - Three sliding-window layers (window 4096), then one full-attention layer.
  - YaRN RoPE (with attention_factor) on the full-attention layers. Default
    RoPE on the sliding-window layers.
  - The vocabulary (100278) does not divide by the TP degree. The LM head
    pads the vocabulary and masks the pad columns.

KV cache: every layer declares a full-attention cache. The sliding-window
layers apply the window with the attention mask. Thus the scheduler does not
trim the block table, and the cache layout is the same for each layer.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    OLMo-3-specific. Change when porting.
"""

import logging
import math

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec, paged_slot_indices
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
)

from .config import FULL_ATTENTION, SLIDING_ATTENTION, Olmo3Config

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1: RMS Normalization
# <-- MODEL-SPECIFIC: HF Olmo3RMSNorm. Compute in fp32, then multiply by gamma.
# =============================================================================


class Olmo3RMSNorm(nn.Module):
    """RMS Normalization (HF Olmo3RMSNorm semantics)."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


# =============================================================================
# Section 2: Rotary Position Embedding
# <-- MODEL-SPECIFIC: Default RoPE on sliding-window layers, YaRN on
#     full-attention layers. rotate_half layout. The attention factor
#     multiplies cos and sin (HF semantics).
# =============================================================================


def _yarn_parameters(
    rope_theta: float, dim: int, rope_scaling: dict, max_position_embeddings: int
) -> tuple[float, float, float, float]:
    """Return the YaRN factor, ramp limits (low, high) and attention factor.

    This follows the YaRN definition in HF ``_compute_yarn_parameters``. The
    function returns Python floats only, so no tensor goes into the graph as
    a constant.
    """
    factor = rope_scaling["factor"]
    original_max = rope_scaling.get(
        "original_max_position_embeddings", max_position_embeddings
    )
    attention_factor = rope_scaling.get("attention_factor")
    mscale = rope_scaling.get("mscale")
    mscale_all_dim = rope_scaling.get("mscale_all_dim")

    def get_mscale(scale: float, m: float = 1.0) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * m * math.log(scale) + 1.0

    if attention_factor is None:
        if mscale and mscale_all_dim:
            attention_factor = get_mscale(factor, mscale) / get_mscale(
                factor, mscale_all_dim
            )
        else:
            attention_factor = get_mscale(factor)

    beta_fast = rope_scaling.get("beta_fast") or 32
    beta_slow = rope_scaling.get("beta_slow") or 1
    truncate = rope_scaling.get("truncate", True)

    def find_correction_dim(num_rotations: float) -> float:
        return (dim * math.log(original_max / (num_rotations * 2 * math.pi))) / (
            2 * math.log(rope_theta)
        )

    low = find_correction_dim(beta_fast)
    high = find_correction_dim(beta_slow)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    return float(factor), float(low), float(high), float(attention_factor)


class Olmo3RotaryEmbedding(nn.Module):
    """RoPE for one OLMo-3 layer type.

    <-- MODEL-SPECIFIC: ``layer_type`` selects YaRN (full-attention layers,
    when ``rope_scaling`` is YaRN) or the default RoPE (sliding-window layers).

    The module computes the cos and sin tables one time on the CPU in fp32,
    with the HF math, for each position up to ``max_position_embeddings``.
    The forward pass gathers the rows for the positions. Thus the device runs
    no ``pow``, ``cos`` or ``sin``, and the values are the same as on the CPU.
    The tables are non-persistent buffers on the CPU, so the meta default
    device of the runner has no effect. The runner moves them to the device.
    """

    def __init__(self, config: Olmo3Config, layer_type: str):
        super().__init__()
        self.head_dim = config.head_dim
        self.layer_type = layer_type
        self.rope_theta = float(config.rope_theta)
        self.max_positions = int(config.max_position_embeddings)
        rope_scaling = config.rope_scaling or {}
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        # (factor, low, high) for YaRN. None for the default RoPE.
        self.yarn = None
        if layer_type == FULL_ATTENTION and rope_type == "yarn":
            factor, low, high, attention_scaling = _yarn_parameters(
                self.rope_theta,
                self.head_dim,
                rope_scaling,
                config.max_position_embeddings,
            )
            self.yarn = (factor, low, high)
        elif layer_type == FULL_ATTENTION and rope_type != "default":
            raise ValueError(
                f"OLMo-3 rope_type={rope_type!r} is not supported. "
                f"Use 'default' or 'yarn'."
            )
        else:
            attention_scaling = 1.0
        self.attention_scaling = attention_scaling

        cos_half, sin_half = self._build_tables(torch.device("cpu"))
        self.register_buffer("cos_half", cos_half, persistent=False)
        self.register_buffer("sin_half", sin_half, persistent=False)

    def _compute_inv_freq(self, device: torch.device) -> torch.Tensor:
        dim = self.head_dim
        pos_freqs = self.rope_theta ** (
            torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim
        )
        inv_freq = 1.0 / pos_freqs
        if self.yarn is None:
            return inv_freq
        factor, low, high = self.yarn
        ramp = torch.clamp(
            (torch.arange(dim // 2, dtype=torch.float32, device=device) - low)
            / (high - low),
            0,
            1,
        )
        extrapolation_factor = 1 - ramp
        inv_freq_interpolation = 1.0 / (factor * pos_freqs)
        return (
            inv_freq_interpolation * (1 - extrapolation_factor)
            + inv_freq * extrapolation_factor
        )

    def _build_tables(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the half-width cos and sin tables, [max_positions, Hd/2], fp32.

        The tables include the attention factor (HF semantics).
        """
        inv_freq = self._compute_inv_freq(device)[:, None]  # [Hd/2, 1]
        positions = torch.arange(
            self.max_positions, dtype=torch.float32, device=device
        )[None, :]  # [1, P]
        freqs = (inv_freq @ positions).transpose(0, 1)  # [P, Hd/2]
        cos = freqs.cos() * self.attention_scaling
        sin = freqs.sin() * self.attention_scaling
        return cos.contiguous(), sin.contiguous()

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin, [T, Hd], for the flat ``position_ids``.

        ``device`` is not used. The tables are already on the device of the
        module.
        """
        index = position_ids.reshape(-1).to(torch.int32)
        index = torch.clamp(index, 0, self.max_positions - 1)
        cos = torch.index_select(self.cos_half, 0, index)  # [T, Hd/2]
        sin = torch.index_select(self.sin_half, 0, index)
        cos = torch.cat((cos, cos), dim=-1)  # [T, Hd]
        sin = torch.cat((sin, sin), dim=-1)
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


# =============================================================================
# Section 3: Attention
# Mixed: PARALLELISM (TP head sharding, SP, collectives) +
#        MODEL-SPECIFIC (full-width QK-norm, sliding window, per-type RoPE)
# =============================================================================


class Olmo3Attention(nn.Module):
    """Multi-head attention with TP head sharding and full-width QK-norm.

    >>> PARALLELISM: TP <<<
    - Q/K/V heads sharded across TP ranks (no KV replication: MHA)
    - Prefill: all-gather (fp32) → torch QKV (fp32) → global QK-norm and
      RoPE (fp32) → cache write (bf16) → torch attention (fp32), or the
      segmented attention kernel → torch O proj (fp32) → fp32 reduce-scatter
    - Decode: torch QKV (fp32) → global QK-norm and RoPE (fp32) → torch
      attention over the paged cache and O proj (fp32) → fp32 all-reduce
    - The activations stay fp32. The weights and the KV cache stay bf16.

    <-- MODEL-SPECIFIC:
    - q_norm / k_norm over the full width (num_heads * head_dim). Each rank
      holds its slice of gamma. One TP all-reduce gives the global sum of
      squares for q and k.
    - Sliding window 4096 on the sliding-window layers, applied by the mask.
    """

    def __init__(self, config: Olmo3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = config.head_dim**-0.5
        self.is_sliding = config.is_sliding(layer_idx)
        self.layer_type = SLIDING_ATTENTION if self.is_sliding else FULL_ATTENTION
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.eps = config.rms_norm_eps

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: Head sharding <<<
        # <-- MODEL-SPECIFIC: The full-width k_norm needs one distinct KV head
        # slice on each rank, so KV replication is not supported.
        if (
            self.num_attention_heads % self.world_size != 0
            or self.num_key_value_heads % self.world_size != 0
        ):
            raise ValueError(
                f"TP degree {self.world_size} does not divide the head counts "
                f"(q={self.num_attention_heads}, kv={self.num_key_value_heads}). "
                f"Use a TP degree that divides both head counts."
            )
        self.num_attention_heads_per_rank = self.num_attention_heads // self.world_size
        self.num_key_value_heads_per_rank = self.num_key_value_heads // self.world_size
        if self.num_attention_heads_per_rank % 2 != 0 and self.world_size > 1:
            raise ValueError(
                f"TP degree {self.world_size} gives an odd number of Q heads on "
                f"each rank ({self.num_attention_heads_per_rank}). The decode "
                f"kernel needs an even number. Use a smaller TP degree."
            )
        self.num_kv_replicas = 1
        self.num_key_value_groups = (
            self.num_attention_heads_per_rank // self.num_key_value_heads_per_rank
        )

        # >>> PARALLELISM: QKV weight shapes <<<
        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = self.num_key_value_heads_per_rank * self.head_dim
        qkv_size = q_size + 2 * kv_size
        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_size = qkv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]
        # Full widths for the global RMSNorm mean.
        self.q_full_width = self.num_attention_heads * self.head_dim
        self.k_full_width = self.num_key_value_heads * self.head_dim

        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, qkv_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(q_size, self.hidden_size, dtype=self.dtype)
        )

        # <-- MODEL-SPECIFIC: Full-width QK-norm, gamma sharded with the heads.
        self.q_norm = Olmo3RMSNorm(q_size, self.eps, self.dtype)
        self.k_norm = Olmo3RMSNorm(kv_size, self.eps, self.dtype)

        # KV caches bound externally
        self.k_cache = None
        self.v_cache = None

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """>>> PARALLELISM: TP sharding of QKV, O-proj and QK-norm weights. <<<
        <-- MODEL-SPECIFIC: Separate Q, K, V (no bias). HF stores transposed.
        """
        set_weight_loader(
            self.qkv_proj_weight,
            fused_qkv_weight_loader(
                q_size=self.q_size,
                kv_size=self.kv_size,
                shard_dim=1,
                num_shards=self.world_size,
                is_storage_transposed=True,
                num_kv_replicas=self.num_kv_replicas,
            ),
        )
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.q_size,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )
        set_weight_loader(
            self.q_norm.weight,
            sharding_weight_loader(
                shard_dim=0, shard_size=self.q_size, num_shards=self.world_size
            ),
        )
        set_weight_loader(
            self.k_norm.weight,
            sharding_weight_loader(
                shard_dim=0, shard_size=self.kv_size, num_shards=self.world_size
            ),
        )

    # ── QK-norm over the full width ───────────────────────────────────────

    def _global_qk_norm_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        out_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the full-width RMSNorm and RoPE to the local q and k slices.

        q and k have the shape [..., width], with T tokens in the leading
        dimensions. cos and sin have the shape [T, Dh]. The norm and RoPE run
        in fp32. The result has the dtype ``out_dtype`` (the default is the
        model dtype). A bf16 RoPE rounds after each multiply and add. On the
        YaRN layers, this error makes the decode attention less accurate than
        the HF BF16 reference.

        Example::

            q, k = attn._global_qk_norm_rope(q, k, cos, sin, torch.float32)

        >>> PARALLELISM: One TP all-reduce of the stacked q/k sum of squares. <<<
        """
        q32 = q.to(torch.float32)
        k32 = k.to(torch.float32)
        sum_sq = torch.stack(
            [q32.pow(2).sum(-1), k32.pow(2).sum(-1)], dim=0
        )  # [2, ...]
        if self.world_size > 1:
            sum_sq = self.tp_group.all_reduce(sum_sq)
        q_inv = torch.rsqrt(sum_sq[0] / self.q_full_width + self.eps).unsqueeze(-1)
        k_inv = torch.rsqrt(sum_sq[1] / self.k_full_width + self.eps).unsqueeze(-1)
        q32 = self.q_norm.weight * (q32 * q_inv)
        k32 = self.k_norm.weight * (k32 * k_inv)

        # <-- MODEL-SPECIFIC: rotate_half RoPE. cos and sin include the YaRN
        # attention factor.
        cos32 = cos.to(torch.float32).unsqueeze(1)  # [T, 1, Dh]
        sin32 = sin.to(torch.float32).unsqueeze(1)
        q_heads = q32.reshape(-1, self.num_attention_heads_per_rank, self.head_dim)
        k_heads = k32.reshape(-1, self.num_key_value_heads_per_rank, self.head_dim)
        q_heads = q_heads * cos32 + _rotate_half(q_heads) * sin32
        k_heads = k_heads * cos32 + _rotate_half(k_heads) * sin32
        out_dtype = self.dtype if out_dtype is None else out_dtype
        return (
            q_heads.reshape(q.shape).to(out_dtype),
            k_heads.reshape(k.shape).to(out_dtype),
        )

    # ── Forward dispatch ──────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ):
        """>>> PARALLELISM: Dispatch to prefill or decode. <<<"""
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]

        # The activations stay fp32. The projections upcast the bf16 weights.
        hidden_states = hidden_states.to(torch.float32)
        if max_query_len <= decode_token_threshold:
            return self.forward_decode(
                hidden_states, positions, position_embeddings, attn_metadata
            )
        # >>> PARALLELISM: All-gather from SP <<<
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return self.forward_prefill(
            hidden_states, positions, position_embeddings, attn_metadata
        )

    # ── Prefill path ──────────────────────────────────────────────────────

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        hidden_states = hidden_states.to(torch.float32)
        tokens, _ = hidden_states.shape

        # Step 1: QKV projection in torch, in fp32. On the device, the full
        # model was less accurate than HF BF16 from layer 20, but each layer
        # passed alone. The bf16 activations gave most of the error. Thus the
        # activations stay fp32, and only the weights and the KV cache are bf16.
        qkv = torch.matmul(hidden_states, self.qkv_proj_weight.float())
        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)

        # Step 2: Full-width QK-norm, then RoPE (rotate_half), in fp32.
        cos, sin = position_embeddings  # [T, Dh], attention factor included
        q, k = self._global_qk_norm_rope(q, k, cos, sin, torch.float32)

        q = q.view(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(
            0, 1
        )  # [Nh, T, Dh]
        k = k.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )
        v = v.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )

        # Step 3: Update the KV cache.
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
        kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

        block_indices, position_indices = paged_slot_indices(
            slot_mapping, block_size, self.k_cache.shape[0]
        )
        k_flat = k.reshape(-1, self.head_dim).to(self.k_cache.dtype)
        v_flat = v.reshape(-1, self.head_dim).to(self.v_cache.dtype)

        head_indices_for_put = torch.arange(
            self.num_key_value_heads_per_rank,
            dtype=torch.long,
            device=hidden_states.device,
        ).repeat_interleave(slot_mapping.shape[0])
        block_indices_for_put = block_indices.repeat(self.num_key_value_heads_per_rank)
        position_indices_for_put = position_indices.repeat(
            self.num_key_value_heads_per_rank
        )
        self.k_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            k_flat,
        )
        self.v_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            v_flat,
        )

        # Step 4: Attention. The sliding-window layers pass the window only
        # when it can mask a key. A window that is not smaller than the
        # longest possible context masks no key, so the result does not
        # change when the layer passes no window. This keeps a window that is
        # larger than the bucket (4096 against 2048) out of the kernel window
        # mask path.
        if kv_segment_size:
            max_context = block_table.shape[-1] * block_size
        else:
            max_context = tokens
        sliding_window = self.sliding_window
        if sliding_window is not None and sliding_window >= max_context:
            sliding_window = None

        if kv_segment_size:
            # The segmented kernel reads the bf16 cache, so give it a bf16 q.
            attn_output = NF.segmented_attention(
                q.to(self.dtype),
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                block_tables=block_table,
                prior_tokens=cached_seq_len,
                block_size=block_size,
                kv_segment_size=kv_segment_size,
                scale=self.scaling,
                tp_q=True,
                tp_out=True,
                sliding_window=sliding_window,
            )
            # [Nh, Dh, T] -> [T, Nh * Dh]
            attn_flat = attn_output.permute(2, 0, 1).reshape(tokens, -1)
        else:
            # Torch attention in fp32. The flash attention kernel takes bf16
            # q, k and v, and it rounds the probabilities to bf16. The key j
            # is valid for the query i when j <= i, and, with a window, when
            # j > i - window.
            k = k.repeat_interleave(self.num_key_value_groups, dim=0)
            v = v.repeat_interleave(self.num_key_value_groups, dim=0)
            idx = torch.arange(tokens, dtype=torch.int32, device=hidden_states.device)
            valid = idx.view(1, tokens) <= idx.view(tokens, 1)  # [T_q, T_k]
            if sliding_window is not None:
                valid = valid & (
                    idx.view(1, tokens) > idx.view(tokens, 1) - sliding_window
                )
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
            scores = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            attn_output = torch.matmul(probs, v)  # [Nh, T, Dh]
            # [Nh, T, Dh] -> [T, Nh * Dh]
            attn_flat = attn_output.transpose(0, 1).reshape(tokens, -1)

        # Step 5: Output projection in fp32. The NKI o_proj kernel has no
        # output dtype parameter, and it rounds each rank partial to bf16.
        # The bf16 reduce-scatter then adds 16 rounded partials. Thus keep the
        # partials and the sum in fp32. The weight rows are in (Nh, Dh) order.
        attn_output = torch.matmul(attn_flat.float(), self.o_proj_weight.float())

        # >>> PARALLELISM: Reduce-scatter to SP layout <<<
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        # The output stays fp32. The post-attention norm reads it in fp32.
        return attn_output.contiguous()

    # ── Decode path ───────────────────────────────────────────────────────

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object,
    ):
        """>>> PARALLELISM: Torch decode attention with TP. <<<

        <-- MODEL-SPECIFIC: The QKV projection (torch, fp32), the full-width
        QK-norm and RoPE run first. Then a torch attention reads the paged KV
        cache and computes the scores, the softmax, the PV product and the O
        projection in fp32.

        The decode megakernel rounds the probabilities and the O-projection
        partial of each rank to bf16. The bf16 reference keeps more precision
        in these steps. The decode path computes every generated token (and,
        in disaggregated serving, the last prompt token too), so this error is
        in each validated logit. Thus the decode path uses fp32 here.
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        assert tokens == B * S_decode, (
            f"Decode token count ({tokens}) is not a multiple of the batch size "
            f"({B}). Use a decode bucket that is a multiple of the batch size."
        )

        hidden_states = hidden_states.to(torch.float32)
        nh = self.num_attention_heads_per_rank
        nkh = self.num_key_value_heads_per_rank
        Dh = self.head_dim

        # Step 1: QKV projection in torch, in fp32. Then the full-width QK-norm
        # and RoPE in fp32. q, k and v stay fp32. The cache write casts k and
        # v to the cache dtype. The decode input is the raw residual, because
        # the layer has no input norm. The decode path computes each scored
        # token. Thus this path uses no NKI kernel.
        qkv = torch.matmul(hidden_states, self.qkv_proj_weight.float())
        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
        cos, sin = position_embeddings  # [B * S, Dh], attention factor included
        q, k = self._global_qk_norm_rope(q, k, cos, sin, torch.float32)

        # Step 2: Gather the prior K/V of each sequence from the paged cache.
        # A cache with one KV head on each rank can be 3D. Clamp the block
        # indices, because a pad entry of the block table is -1.
        k_cache = self.k_cache if self.k_cache.dim() == 4 else self.k_cache.unsqueeze(1)
        v_cache = self.v_cache if self.v_cache.dim() == 4 else self.v_cache.unsqueeze(1)
        num_blocks = k_cache.shape[0]
        blocks_per_seq = block_table.shape[-1]
        S_ctx = blocks_per_seq * block_size
        flat_idx = torch.clamp(block_table.to(torch.int64), 0, num_blocks - 1)
        flat_idx = flat_idx.reshape(-1)  # [B * blocks]
        k_prior = torch.index_select(k_cache, 0, flat_idx)
        v_prior = torch.index_select(v_cache, 0, flat_idx)
        # [B * blocks, nkh, bs, Dh] -> [B, nkh, S_ctx, Dh]
        k_prior = (
            k_prior.view(B, blocks_per_seq, nkh, block_size, Dh)
            .permute(0, 2, 1, 3, 4)
            .reshape(B, nkh, S_ctx, Dh)
        )
        v_prior = (
            v_prior.view(B, blocks_per_seq, nkh, block_size, Dh)
            .permute(0, 2, 1, 3, 4)
            .reshape(B, nkh, S_ctx, Dh)
        )

        # Step 3: Masks. A prior key is valid when its position is before the
        # first active token. A sliding-window layer also masks the keys
        # before (pos - window + 1). The active tokens use a causal mask.
        assert positions is not None
        pos = positions.view(B, S_decode).to(torch.int32)  # [B, S]
        key_pos = torch.arange(S_ctx, dtype=torch.int32, device=pos.device)
        prior_valid = key_pos.view(1, 1, S_ctx) < pos[:, :1].unsqueeze(-1)
        if self.sliding_window is not None:
            window_start = pos.unsqueeze(-1) - self.sliding_window  # [B, S, 1]
            prior_valid = prior_valid & (key_pos.view(1, 1, S_ctx) > window_start)
        else:
            prior_valid = prior_valid.expand(B, S_decode, S_ctx)
        active_idx = torch.arange(S_decode, dtype=torch.int32, device=pos.device)
        active_valid = active_idx.view(1, S_decode) <= active_idx.view(S_decode, 1)
        active_valid = active_valid.unsqueeze(0).expand(B, S_decode, S_decode)
        valid = torch.cat([prior_valid, active_valid], dim=-1)  # [B, S, S_ctx + S]

        # Stale cache rows can hold any value. Set the V rows at and after
        # the first active position to zero, so that a masked NaN cannot go
        # into the PV product. The rows before it hold written values.
        v_row_used = key_pos.view(1, S_ctx) < pos[:, :1]  # [B, S_ctx]
        v_prior = torch.where(
            v_row_used.view(B, 1, S_ctx, 1), v_prior, torch.zeros_like(v_prior)
        )

        # Step 4: Attention in fp32. [B, heads, S, Dh] layout.
        q_heads = q.reshape(B, S_decode, nh, Dh).transpose(1, 2).float()
        k_active = k.reshape(B, S_decode, nkh, Dh).transpose(1, 2)
        v_active = v.reshape(B, S_decode, nkh, Dh).transpose(1, 2)
        keys = torch.cat([k_prior.float(), k_active.float()], dim=2)
        values = torch.cat([v_prior.float(), v_active.float()], dim=2)
        if self.num_key_value_groups > 1:
            keys = keys.repeat_interleave(self.num_key_value_groups, dim=1)
            values = values.repeat_interleave(self.num_key_value_groups, dim=1)
        scores = torch.matmul(q_heads, keys.transpose(-2, -1)) * self.scaling
        scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        attn = torch.matmul(probs, values)  # [B, nh, S, Dh]

        # Step 5: O projection in fp32. The weight rows are in (Nh, Dh) order.
        attn_flat = attn.transpose(1, 2).reshape(tokens, nh * Dh)
        output = torch.matmul(attn_flat, self.o_proj_weight.float())

        # Step 6: Write the new K/V into the paged cache. Same layout as the
        # prefill path: [nkh, T, Dh], head-major.
        block_indices, position_indices = paged_slot_indices(
            slot_mapping, block_size, num_blocks
        )
        k_flat = (k_active.transpose(0, 1).reshape(nkh, tokens, Dh).reshape(-1, Dh)).to(
            self.k_cache.dtype
        )
        v_flat = (v_active.transpose(0, 1).reshape(nkh, tokens, Dh).reshape(-1, Dh)).to(
            self.v_cache.dtype
        )
        head_indices = torch.arange(
            nkh, dtype=torch.long, device=hidden_states.device
        ).repeat_interleave(slot_mapping.shape[0])
        block_indices = block_indices.repeat(nkh)
        position_indices = position_indices.repeat(nkh)
        k_cache.index_put_((block_indices, head_indices, position_indices), k_flat)
        v_cache.index_put_((block_indices, head_indices, position_indices), v_flat)

        # >>> PARALLELISM: TP all-reduce <<<
        # Add the 16 partial outputs in fp32. The output stays fp32.
        if self.world_size > 1:
            output = self.tp_group.all_reduce(output)
        return output


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# =============================================================================
# Section 4: Dense MLP
# Mixed: PARALLELISM (TP intermediate sharding, SP collectives) +
#        MODEL-SPECIFIC (SiLU gate/up/down, no bias)
# =============================================================================


class Olmo3MLP(nn.Module):
    """Dense MLP with TP intermediate sharding.

    >>> PARALLELISM: TP <<<
    - Prefill: all-gather → torch MLP (fp32) → fp32 reduce-scatter (SP)
    - Decode: torch MLP (fp32) → fp32 all-reduce
    - The activations stay fp32. The weights stay bf16.
    """

    def __init__(self, config: Olmo3Config):
        super().__init__()
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        if config.intermediate_size % self.world_size != 0:
            raise ValueError(
                f"TP degree {self.world_size} does not divide intermediate_size "
                f"{config.intermediate_size}. Use a TP degree that divides it."
            )
        self.intermediate_size_per_rank = config.intermediate_size // self.world_size
        self.dtype = config.torch_dtype

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
        # The post-norm layers send the raw residual to the MLP. The NKI MLP
        # kernels round each rank partial to BF16, and the BF16 collective then
        # adds 16 rounded partials. HF rounds the full sum one time. On the
        # device, the prefill MLP kernel was less accurate than the HF BF16
        # reference. Thus both paths use torch matmuls. They do the activation
        # and the down projection in fp32, and they reduce in fp32. The decode
        # path computes each scored token (in disaggregated serving the first
        # token included). Thus the decode path needs the same precision.
        # The activations stay fp32, and the matmuls upcast the bf16 weights.
        # The full model was less accurate than HF BF16 with bf16 activations,
        # but each layer passed alone. The output stays fp32.
        hidden_states = hidden_states.to(torch.float32)
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        gate = torch.matmul(hidden_states, self.gate_proj_weight.float())
        up = torch.matmul(hidden_states, self.up_proj_weight.float())
        intermediate = nn.functional.silu(gate) * up
        output = torch.matmul(intermediate, self.down_proj_weight.float())
        if self.world_size > 1:
            if is_prefill:
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                output = self.tp_group.all_reduce(output)
        return output


# =============================================================================
# Section 5: Decoder Layer
# <-- MODEL-SPECIFIC: post-norm (norm on the sublayer output, then residual)
# =============================================================================


class Olmo3DecoderLayer(nn.Module):
    """Single OLMo-3 decoder layer.

    Architecture: h = h + post_attention_layernorm(attn(h));
                  h = h + post_feedforward_layernorm(mlp(h))

    The sublayer outputs are fp32. The backbone sends an fp32 ``h``, so the
    residual adds occur in fp32.
    """

    def __init__(self, config: Olmo3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.self_attn = Olmo3Attention(config, layer_idx=layer_idx)
        self.mlp = Olmo3MLP(config)
        self.post_attention_layernorm = Olmo3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_feedforward_layernorm = Olmo3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )

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
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(hidden_states, is_prefill=not is_decode)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# =============================================================================
# Section 6: Model Backbone
# Mixed: PARALLELISM (SP embedding/chunking) + MODEL-SPECIFIC (layer stack, norm)
# =============================================================================


class Olmo3Model(nn.Module):
    """OLMo-3 transformer backbone.

    >>> PARALLELISM: SP during prefill — embed → chunk → layers → all-gather. <<<
    """

    def __init__(self, config: Olmo3Config):
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: Vocab-sharded embedding (pads the last shard) <<<
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=False,
                pad_shard=True,
            ),
        )

        self.layers = nn.ModuleList(
            [
                Olmo3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Olmo3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )

        # <-- MODEL-SPECIFIC: One RoPE table for each layer type.
        self.rotary_emb = nn.ModuleDict(
            {
                SLIDING_ATTENTION: Olmo3RotaryEmbedding(config, SLIDING_ATTENTION),
                FULL_ATTENTION: Olmo3RotaryEmbedding(config, FULL_ATTENTION),
            }
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
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
        hidden_states = NF.merge_prompt_embeds(
            hidden_states, inputs_embeds, is_token_ids
        )
        # Keep the residual stream in fp32 between the layers. A bf16 stream
        # rounds after each of the 64 residual adds. Each layer is more
        # accurate than HF BF16 when it runs alone, but the full model was
        # 2.3x less accurate. Thus the rounding error adds up across the
        # layers. The sublayers also take fp32 input and give fp32 output.
        hidden_states = hidden_states.to(torch.float32)

        # Keep cos and sin in fp32. The attention applies RoPE in fp32.
        position_embeddings = {
            layer_type: rotary(
                positions, device=hidden_states.device, dtype=torch.float32
            )
            for layer_type, rotary in self.rotary_emb.items()
        }

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                positions=positions,
                position_embeddings=position_embeddings[decoder_layer.layer_type],
                attn_metadata=attn_metadata,
            )

        # The norm runs in fp32. Round its output to bf16 one time, the same
        # as the HF final norm.
        hidden_states = self.norm(hidden_states).to(self.config.torch_dtype)

        # >>> PARALLELISM: SP — all-gather the full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return hidden_states


# =============================================================================
# Section 7: LM Head + Weight Loading
# Mixed: PARALLELISM (column-parallel LM head, vocab padding) +
#        MODEL-SPECIFIC (weight mappings)
# =============================================================================


class Olmo3ForCausalLM(nn.Module):
    """OLMo-3 model with an untied LM head.

    >>> PARALLELISM: Column-parallel LM head. <<<
    The vocabulary pads up to a multiple of the TP degree. The pad rows of
    the LM head weight are zero. The forward pass sets the pad logits to the
    lowest value before sampling and removes them from the returned logits.
    """

    def __init__(self, config: Olmo3Config):
        super().__init__()
        self.config = config
        self.model = Olmo3Model(config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

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

        # >>> PARALLELISM: Pad the vocabulary to a multiple of the TP degree <<<
        self.vocab_size = config.vocab_size
        self.vocab_size_per_rank = math.ceil(config.vocab_size / self.world_size)
        self.padded_vocab_size = self.vocab_size_per_rank * self.world_size

        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            self.padded_vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.vocab_size_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=False,
                pad_shard=True,
            ),
        )

        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=self.tp_group.device_group,
            )

    def _mask_local_pad_logits(
        self, logits: torch.Tensor, rank: torch.Tensor | None
    ) -> torch.Tensor:
        """Set the pad columns of a vocab-sharded logits tensor to the lowest value."""
        if self.padded_vocab_size == self.vocab_size:
            return logits
        local_idx = torch.arange(
            self.vocab_size_per_rank, device=logits.device, dtype=torch.int64
        )
        if rank is None:
            offset = self.rank * self.vocab_size_per_rank
        else:
            offset = rank.to(torch.int64).reshape(-1)[0] * self.vocab_size_per_rank
        is_pad = (local_idx + offset) >= self.vocab_size
        return logits.masked_fill(is_pad.unsqueeze(0), torch.finfo(logits.dtype).min)

    def _pad_logit_mask(self, logit_mask: torch.Tensor | None) -> torch.Tensor | None:
        """Pad a full-vocab structured-output mask to the padded vocabulary."""
        if logit_mask is None or logit_mask.shape[-1] == self.padded_vocab_size:
            return logit_mask
        pad = self.padded_vocab_size - logit_mask.shape[-1]
        return torch.nn.functional.pad(logit_mask, (0, pad), value=0)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
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
                f"Prompt length ({T}) is not a multiple of the TP degree "
                f"({self.world_size}), or it is not larger than the TP degree. "
                f"Use a prefill bucket that is a multiple of the TP degree."
            )

        hidden_states = self.model(
            input_ids,
            positions,
            attn_metadata=attn_metadata,
            rank=rank,
            inputs_embeds=inputs_embeds,
            is_token_ids=is_token_ids,
        )

        if (
            not is_prefill
            and max_query_len == 1
            and spec_decode_metadata is None
            and hidden_states.shape[0] == sampling_positions.shape[0]
        ):
            hidden_states_for_logits = hidden_states
        else:
            hidden_states_for_logits = torch.index_select(
                hidden_states, dim=0, index=sampling_positions
            )

        logits = self.lm_head(hidden_states_for_logits)

        # ── No on-device sampling: logits are full (padded) vocab ─────────
        if self.on_device_sampling_config is None:
            return logits[..., : self.vocab_size]

        # ── On-device sampling: logits are vocab-sharded ──────────────────
        logits = self._mask_local_pad_logits(logits, rank)
        sampled_tokens = self.sampler(
            logits,
            sampling_params,
            logit_mask=self._pad_logit_mask(logit_mask),
            tp_rank=rank,
        )

        gathered_logits = None
        if self._gather_logits:
            if self.world_size > 1:
                gathered_logits = self.tp_group.all_gather(logits, dim=1)
            else:
                gathered_logits = logits
            gathered_logits = gathered_logits[..., : self.vocab_size]

        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler

            return rejection_sampler(spec_decode_metadata, sampled_tokens)

        return sampled_tokens, gathered_logits

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = Olmo3Config.from_configs(hf_config, neuron_config)
        return cls(config)

    # ── KV Cache ──────────────────────────────────────────────────────────

    def get_kv_spec(self):
        # <-- MODEL-SPECIFIC: Full-attention spec for every layer. The
        # sliding-window layers apply the window with the attention mask, so
        # the runner does not trim their block tables.
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
                raise KeyError(
                    f"KV cache for layer {layer_name} is not initialized. "
                    f"Bind a cache for each decoder layer."
                )
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]

    # ── Weight Loading ────────────────────────────────────────────────────

    def get_weight_mappings(self) -> dict[str, str | list[str]]:
        """Return the map from model parameter names to HF checkpoint keys."""
        return olmo3_weight_mappings(len(self.model.layers))

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load and shard the OLMo-3 HF checkpoint weights.

        >>> PARALLELISM: Weight loaders handle TP sharding and vocab padding. <<<
        """
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, self.get_weight_mappings(), device
        ).state_dict

        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        self.load_state_dict(rank_sharded, strict=False, assign=True)

    def load_weights_lite(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Skip weight loading during CPU compile. No scale constants exist."""
        return None


def olmo3_weight_mappings(num_layers: int) -> dict[str, str | list[str]]:
    """Return the map from Olmo3ForCausalLM parameter names to HF checkpoint keys."""
    mappings = {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "lm_head.weight": "lm_head.weight",
        "model.norm.weight": "model.norm.weight",
    }
    for layer_id in range(num_layers):
        prefix = f"model.layers.{layer_id}"
        mappings[f"{prefix}.self_attn.qkv_proj_weight"] = [
            f"{prefix}.self_attn.q_proj.weight",
            f"{prefix}.self_attn.k_proj.weight",
            f"{prefix}.self_attn.v_proj.weight",
        ]
        for name in (
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "post_attention_layernorm.weight",
            "post_feedforward_layernorm.weight",
        ):
            mappings[f"{prefix}.{name}"] = f"{prefix}.{name}"
        mappings[f"{prefix}.self_attn.o_proj_weight"] = (
            f"{prefix}.self_attn.o_proj.weight"
        )
        for proj in ("gate_proj", "up_proj", "down_proj"):
            mappings[f"{prefix}.mlp.{proj}_weight"] = f"{prefix}.mlp.{proj}.weight"
    return mappings
