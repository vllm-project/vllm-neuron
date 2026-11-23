# SPDX-License-Identifier: Apache-2.0
"""
Bloom BF16 Implementation
=========================

Implementation of Bloom (for example bigscience/bloomz-560m) for the native
Neuron backend. The structure follows the plugin's dense Qwen3 model.

Supported parallelism: TP, SP.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    Bloom-specific. Change when porting.

ALIBI ENCODING (MODEL-SPECIFIC):
  Bloom adds the bias ``slope[h] * key_position`` to each attention score.
  The Neuron attention kernels have no bias input. This module puts the bias
  into the head dimension, so the kernels compute it exactly:

  - The hidden input gets 3 extra columns: ``1``, ``a = pos // 64`` and
    ``b = pos % 64``. bfloat16 holds ``a`` and ``b`` exactly. The input is
    zero-padded to ``H + bloom_extra_columns(H)`` columns. That width is a
    multiple of 256, because the decode kernel shards the input over the 2
    logical cores in groups of 128 columns.
  - Each head is padded from ``head_dim`` to 128 lanes.
  - The extra Q lanes hold ``64 * t_k`` and ``t_k`` for k = 0, 1, 2. The
    three ``t_k`` terms are bfloat16 values with the sum ``slope / scale``.
  - The extra K lanes hold ``a`` and ``b``. The KV cache stores them, so
    decode and segmented prefill also get the bias.
  - The extra V lanes and the matching rows of the output projection are 0.

  Then ``scale * (q' . k') = scale * (q . k) + slope * key_position``.
  The decode kernel applies RoPE. This module gives cos = 1 and sin = 0, which
  is an exact identity. Prefill does not apply RoPE.

PARALLELISM SHARDING:
  Attention heads are sharded. The MLP intermediate dimension is sharded.
  The ALiBi slopes of a rank go into the QKV weight of that rank at load time.
"""

import logging
import math

import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from nkilib.core.utils.common_types import ActFnType
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec, paged_slot_indices
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.neuron_utils import can_run_kernel
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    set_weight_loader,
    sharding_weight_loader,
)

from .compat import build_per_head_block_table, scatter_new_kv
from .config import BLOOM_COMPUTE_DTYPE, BloomConfig
from .factory import validate_max_model_len

logger = logging.getLogger(__name__)

# <-- MODEL-SPECIFIC: Padded head width and the extra hidden columns.
BLOOM_PADDED_HEAD_DIM = 128
# Extra input columns: 1, a = pos // 64, b = pos % 64, then zero padding.
_NUM_POSITION_COLUMNS = 3
# At LNC 2, the decode kernel needs (input width / 128) to be even.
_INPUT_WIDTH_ALIGNMENT = 2 * 128
_POSITION_BASE = 64
# Number of bfloat16 terms in the split of slope / scale.
_NUM_SLOPE_TERMS = 3
# The ALiBi encoding uses 2 lanes for each slope term.
BLOOM_ALIBI_LANES = 2 * _NUM_SLOPE_TERMS
# The prefill QKV kernel accepts a fused output width of 4096 or less
# (see _can_use_qkv_kernel in functional/attention/qkv.py).
_QKV_KERNEL_MAX_WIDTH = 4096
_QKV_COLUMN_ALIGNMENT = 128


# =============================================================================
# Section 1: ALiBi helpers
# <-- MODEL-SPECIFIC: Bloom ALiBi slopes and the head-dimension encoding
# =============================================================================


def build_alibi_slopes(
    num_heads: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Compute the Bloom ALiBi slopes in float32 on ``device``.

    This is the same algorithm as the HF Bloom ``build_alibi_tensor``. The
    device is explicit, so a ``meta`` default device context has no effect.

    Example:
        >>> build_alibi_slopes(16)[0].item()
        0.7071067690849304
    """
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        dtype=torch.float32,
        device=device,
    )
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32, device=device)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            dtype=torch.float32,
            device=device,
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(
            1, 1 + 2 * num_remaining_heads, 2, dtype=torch.int32, device=device
        )
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    return slopes


def split_slope_terms(values: torch.Tensor) -> torch.Tensor:
    """Split float32 values into 3 bfloat16 terms with the same sum.

    Returns a float32 tensor with shape ``[*values.shape, 3]``. Each term is
    exact in bfloat16.

    Example:
        >>> t = split_slope_terms(torch.tensor([5.656854]))
        >>> bool(torch.allclose(t.sum(-1), torch.tensor([5.656854])))
        True
    """
    remainder = values.to(torch.float32)
    terms = []
    for _ in range(_NUM_SLOPE_TERMS):
        term = remainder.to(torch.bfloat16).to(torch.float32)
        terms.append(term)
        remainder = remainder - term
    return torch.stack(terms, dim=-1)


def bloom_extra_columns(hidden_size: int) -> int:
    """Return the number of extra input columns for the ALiBi encoding.

    The result is the smallest count that holds the 3 position columns and
    makes ``hidden_size + count`` a multiple of 256.

    Example:
        >>> bloom_extra_columns(1024)
        256
        >>> bloom_extra_columns(1152)
        128
    """
    width = hidden_size + _NUM_POSITION_COLUMNS
    aligned = -(-width // _INPUT_WIDTH_ALIGNMENT) * _INPUT_WIDTH_ALIGNMENT
    return aligned - hidden_size


def qkv_column_chunks(fused_width: int) -> list[tuple[int, int]]:
    """Split the fused QKV columns into kernel-sized chunks.

    The result has the smallest number of equal chunks. Each chunk width is
    a multiple of 128 and is 4096 or less, so that each chunk is in the
    limits of the prefill QKV kernel. A width of 4096 or less gives one chunk.

    Example:
        >>> qkv_column_chunks(3072)
        [(0, 3072)]
        >>> qkv_column_chunks(6144)
        [(0, 3072), (3072, 6144)]
    """
    if fused_width % _QKV_COLUMN_ALIGNMENT != 0:
        raise ValueError(
            f"Fused QKV width {fused_width} is not a multiple of "
            f"{_QKV_COLUMN_ALIGNMENT}. Use a padded head width of 128."
        )
    columns = fused_width // _QKV_COLUMN_ALIGNMENT
    num_chunks = -(-fused_width // _QKV_KERNEL_MAX_WIDTH)
    while columns % num_chunks != 0:
        num_chunks += 1
    width = fused_width // num_chunks
    return [(i * width, (i + 1) * width) for i in range(num_chunks)]


def augment_hidden(
    hidden_states: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """Append the ALiBi position columns to the hidden states.

    The result is ``[x, 1, pos // 64, pos % 64, 0, ...]`` with the width
    ``H + bloom_extra_columns(H)``.

    Example:
        >>> x = torch.zeros(2, 4, dtype=torch.bfloat16)
        >>> augment_hidden(x, torch.tensor([0, 65])).shape
        torch.Size([2, 256])
    """
    positions = positions.reshape(-1).to(torch.int32)
    tokens = hidden_states.shape[0]
    dtype = hidden_states.dtype
    ones = torch.ones(tokens, 1, dtype=dtype, device=hidden_states.device)
    high = torch.div(positions, _POSITION_BASE, rounding_mode="floor")
    low = positions - high * _POSITION_BASE
    extra = bloom_extra_columns(hidden_states.shape[-1])
    zeros = torch.zeros(
        tokens,
        extra - _NUM_POSITION_COLUMNS,
        dtype=dtype,
        device=hidden_states.device,
    )
    return torch.cat(
        [
            hidden_states,
            ones,
            high.view(-1, 1).to(dtype),
            low.view(-1, 1).to(dtype),
            zeros,
        ],
        dim=-1,
    )


_GELU_TANH_SCALE = math.sqrt(2.0 / math.pi)
_GELU_TANH_CUBIC = 0.044715


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """GeLU with the tanh approximation, written with elementwise ops.

    This is the formula of ``torch.nn.functional.gelu(x, approximate="tanh")``
    and of the HF Bloom MLP, computed in float32 and rounded once to the input
    dtype. The model calls it instead of ``torch.nn.functional.gelu`` because
    the Neuron XLA overrides replace that function with a wrapper whose inner
    builtin Dynamo cannot trace.

    Example:
        >>> x = torch.linspace(-4, 4, 9)
        >>> ref = torch.nn.functional.gelu(x, approximate="tanh")
        >>> bool(torch.allclose(gelu_tanh(x), ref, atol=1e-6))
        True
    """
    xf = x.to(torch.float32)
    inner = _GELU_TANH_SCALE * (xf + _GELU_TANH_CUBIC * xf * xf * xf)
    return (0.5 * xf * (1.0 + torch.tanh(inner))).to(x.dtype)


def _cast_loader(loader: SafetensorsWeightLoader) -> SafetensorsWeightLoader:
    """Cast the output of a loader to the compute dtype."""
    return SafetensorsWeightLoader(
        transform=lambda slices, rank: loader.load(slices, rank).to(BLOOM_COMPUTE_DTYPE)
    )


def bloom_qkv_weight_loader(
    num_heads: int,
    head_dim: int,
    hidden_size: int,
    num_shards: int,
    scale: float,
) -> SafetensorsWeightLoader:
    """Build the loader of the augmented, padded QKV weight.

    The loader reads 2 slices: ``query_key_value.weight`` ``[3H, H]`` and
    ``query_key_value.bias`` ``[3H]``. HF Bloom interleaves the fused rows
    per head: row ``(g * 3 + t) * head_dim + d``, where t = 0, 1, 2 is q, k, v.

    The result has the shape ``[H + bloom_extra_columns(H), 3 * heads_per_rank
    * 128]``. The
    columns are all Q heads, then all K heads, then all V heads.
    """
    heads_per_rank = num_heads // num_shards
    padded = BLOOM_PADDED_HEAD_DIM
    in_dim = hidden_size + bloom_extra_columns(hidden_size)

    # Make each tensor inside transform, on the device of the checkpoint slice.
    # The model runner builds the model (and this loader) in a meta device
    # context, so a tensor that the builder makes has no data.
    def transform(slices, rank: int) -> torch.Tensor:
        assert len(slices) == 2, "The Bloom QKV loader needs the weight and the bias"
        weight = (
            slices[0][:].to(torch.float32).view(num_heads, 3, head_dim, hidden_size)
        )
        bias = slices[1][:].to(torch.float32).view(num_heads, 3, head_dim)

        first = (rank % num_shards) * heads_per_rank
        local_weight = weight[first : first + heads_per_rank]  # [g, 3, hd, H]
        local_bias = bias[first : first + heads_per_rank]  # [g, 3, hd]

        device = weight.device
        slopes = build_alibi_slopes(num_heads, device=device)
        out = torch.zeros(
            in_dim, 3, heads_per_rank, padded, dtype=torch.float32, device=device
        )
        # Linear weight is [out, in]. The parameter is [in, out].
        out[:hidden_size, :, :, :head_dim] = local_weight.permute(3, 1, 0, 2)
        out[hidden_size, :, :, :head_dim] = local_bias.permute(1, 0, 2)

        # ALiBi: score += scale * q_extra . k_extra = slope * key_position.
        terms = split_slope_terms(slopes[first : first + heads_per_rank] / scale)
        for k in range(_NUM_SLOPE_TERMS):
            lane_high = head_dim + 2 * k
            lane_low = lane_high + 1
            out[hidden_size, 0, :, lane_high] = _POSITION_BASE * terms[:, k]
            out[hidden_size, 0, :, lane_low] = terms[:, k]
            out[hidden_size + 1, 1, :, lane_high] = 1.0
            out[hidden_size + 2, 1, :, lane_low] = 1.0

        return out.reshape(in_dim, 3 * heads_per_rank * padded).to(BLOOM_COMPUTE_DTYPE)

    return SafetensorsWeightLoader(transform=transform)


def bloom_o_proj_weight_loader(
    num_heads: int, head_dim: int, hidden_size: int, num_shards: int
) -> SafetensorsWeightLoader:
    """Build the loader of the padded output projection weight.

    The loader reads ``dense.weight`` ``[H, H]``. The result has the shape
    ``[heads_per_rank * 128, H]``. The rows of the padded lanes are 0.
    """
    heads_per_rank = num_heads // num_shards
    padded = BLOOM_PADDED_HEAD_DIM

    def transform(slices, rank: int) -> torch.Tensor:
        assert len(slices) == 1, "The Bloom O loader needs one slice"
        first = (rank % num_shards) * heads_per_rank
        cols = slice(first * head_dim, (first + heads_per_rank) * head_dim)
        weight = slices[0][:, cols].to(torch.float32)  # [H, g * hd]
        weight = weight.T.reshape(heads_per_rank, head_dim, hidden_size)
        out = torch.zeros(
            heads_per_rank,
            padded,
            hidden_size,
            dtype=torch.float32,
            device=weight.device,
        )
        out[:, :head_dim, :] = weight
        return out.reshape(heads_per_rank * padded, hidden_size).to(BLOOM_COMPUTE_DTYPE)

    return SafetensorsWeightLoader(transform=transform)


def bloom_row_bias_loader(shard_size: int, num_shards: int) -> SafetensorsWeightLoader:
    """Build the loader of a sharded bias with the shape ``[1, shard_size]``."""

    def transform(slices, rank: int) -> torch.Tensor:
        assert len(slices) == 1, "The Bloom bias loader needs one slice"
        start = (rank % num_shards) * shard_size
        bias = slices[0][start : start + shard_size]
        return bias.reshape(1, shard_size).to(BLOOM_COMPUTE_DTYPE)

    return SafetensorsWeightLoader(transform=transform)


def _replicated_loader() -> SafetensorsWeightLoader:
    return SafetensorsWeightLoader(
        transform=lambda slices, rank: slices[0][:].to(BLOOM_COMPUTE_DTYPE)
    )


# =============================================================================
# Section 2: Layer Normalization
# <-- MODEL-SPECIFIC: Bloom uses LayerNorm with bias on every norm
# =============================================================================


class BloomLayerNorm(nn.Module):
    """LayerNorm with weight and bias. The statistics use float32."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        self.variance_epsilon = eps
        set_weight_loader(self.weight, _replicated_loader())
        set_weight_loader(self.bias, _replicated_loader())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        out = F.layer_norm(
            hidden_states.to(torch.float32),
            self.weight.shape,
            self.weight.to(torch.float32),
            self.bias.to(torch.float32),
            self.variance_epsilon,
        )
        return out.to(input_dtype)


# =============================================================================
# Section 3: Attention
# Mixed: PARALLELISM (TP head sharding, SP, collectives) +
#        MODEL-SPECIFIC (MHA, QKV bias, O bias, ALiBi encoding)
# =============================================================================


class BloomAttention(nn.Module):
    """Multi-head attention with ALiBi in the padded head dimension.

    >>> PARALLELISM: TP <<<
    - Heads are sharded across TP ranks. Bloom is MHA, so each rank holds
      the same number of Q and KV heads.
    - Prefill: all-gather -> QKV proj -> flash attention -> O proj -> reduce-scatter
    - Decode: fused megakernel -> all-reduce
    - The O bias is replicated. Each rank adds it after the collective.
    """

    def __init__(self, config: BloomConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.padded_head_dim = BLOOM_PADDED_HEAD_DIM
        self.augmented_hidden_size = config.hidden_size + bloom_extra_columns(
            config.hidden_size
        )
        # <-- MODEL-SPECIFIC: HF Bloom scales by 1 / sqrt(head_dim). The padded
        # head does not change the scale.
        self.scaling = config.head_dim**-0.5

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.num_attention_heads_per_rank = self.num_attention_heads // self.world_size
        # MHA: one KV head for each Q head.
        self.num_key_value_heads_per_rank = self.num_attention_heads_per_rank

        q_size = self.num_attention_heads_per_rank * self.padded_head_dim
        self.q_size = q_size
        self.kv_size = q_size
        self.qkv_split_indices = [q_size, 2 * q_size]
        # TP 1 gives a fused width of 6144, which is more than the QKV kernel
        # accepts. Prefill then calls the kernel once for each column chunk.
        self.qkv_chunks = qkv_column_chunks(3 * q_size)

        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.augmented_hidden_size, 3 * q_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(q_size, self.hidden_size, dtype=self.dtype)
        )
        self.o_proj_bias = nn.Parameter(torch.empty(self.hidden_size, dtype=self.dtype))

        # KV caches bound externally via bind_kv_cache()
        self.k_cache = None
        self.v_cache = None

        set_weight_loader(
            self.qkv_proj_weight,
            bloom_qkv_weight_loader(
                num_heads=self.num_attention_heads,
                head_dim=self.head_dim,
                hidden_size=self.hidden_size,
                num_shards=self.world_size,
                scale=self.scaling,
            ),
        )
        set_weight_loader(
            self.o_proj_weight,
            bloom_o_proj_weight_loader(
                num_heads=self.num_attention_heads,
                head_dim=self.head_dim,
                hidden_size=self.hidden_size,
                num_shards=self.world_size,
            ),
        )
        set_weight_loader(self.o_proj_bias, _replicated_loader())

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        """Dispatch to the prefill path or the decode path."""
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]

        if max_query_len <= decode_token_threshold:
            return self.forward_decode(hidden_states, positions, attn_metadata)

        # >>> PARALLELISM: All-gather from SP before attention <<<
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return self.forward_prefill(hidden_states, positions, attn_metadata)

    # -- Prefill path ---------------------------------------------------------

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        """Prefill: QKV proj, KV cache write, flash attention and O proj."""
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        hidden_states = hidden_states.to(self.dtype)
        tokens = hidden_states.shape[0]
        heads = self.num_attention_heads_per_rank
        padded = self.padded_head_dim

        # -- Step 1: QKV projection of the augmented input --------------------
        # <-- MODEL-SPECIFIC: The QKV bias and the ALiBi lanes come from the
        # extra input columns. No RoPE.
        x_aug = augment_hidden(hidden_states, positions).unsqueeze(0)
        if len(self.qkv_chunks) == 1:
            qkv = NF.qkv_proj(
                hidden=x_aug, qkv_weights=self.qkv_proj_weight, bias=None
            ).squeeze(0)
        else:
            # Each column chunk is an independent projection of the same
            # input. The concatenation gives the same Q, K, V column order.
            qkv = torch.cat(
                [
                    NF.qkv_proj(
                        hidden=x_aug,
                        qkv_weights=self.qkv_proj_weight[:, start:end].contiguous(),
                        bias=None,
                    )
                    for start, end in self.qkv_chunks
                ],
                dim=-1,
            ).squeeze(0)

        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
        q = q.reshape(tokens, heads, padded).transpose(0, 1)  # [Nh, T, 128]
        k = k.reshape(tokens, heads, padded).transpose(0, 1)
        v = v.reshape(tokens, heads, padded).transpose(0, 1)

        # -- Step 2: KV cache write -------------------------------------------
        # >>> PARALLELISM: Cache is per-rank (TP sharded heads) <<<
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
        kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

        block_indices, position_indices = paged_slot_indices(
            slot_mapping, block_size, self.k_cache.shape[0]
        )
        head_indices_for_put = torch.arange(
            heads, dtype=torch.long, device=hidden_states.device
        ).repeat_interleave(slot_mapping.shape[0])
        block_indices_for_put = block_indices.repeat(heads)
        position_indices_for_put = position_indices.repeat(heads)

        self.k_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            k.reshape(-1, padded).to(self.k_cache.dtype),
        )
        self.v_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            v.reshape(-1, padded).to(self.v_cache.dtype),
        )

        # -- Step 3: Attention ------------------------------------------------
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
            )  # [Nh, 128, T]
        else:
            attn_output = NF.flash_attention(
                q.transpose(1, 2),  # [Nh, 128, T]
                k.transpose(1, 2),  # [Nh, 128, T]
                v,  # [Nh, T, 128]
                scale=self.scaling,
                tp_q=False,
                tp_out=True,
            )  # [Nh, 128, T]

        # -- Step 4: Output projection ----------------------------------------
        attn_output = NF.o_proj(
            attn_output.unsqueeze(0), self.o_proj_weight, None
        ).squeeze(0)  # [T, H]

        # >>> PARALLELISM: Reduce-scatter to return to SP layout <<<
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        # <-- MODEL-SPECIFIC: Replicated O bias after the collective.
        return (attn_output + self.o_proj_bias).contiguous()

    # -- Decode path ----------------------------------------------------------

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: object,
    ) -> torch.Tensor:
        """Decode: fused megakernel on the augmented input.

        On a Neuron device, the call uses a per-head 3D block table, which
        the kernel needs for more than one KV head. The torch fallback does not
        support that table, so the CPU path uses the 2D table and an in-place
        cache write.
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        max_blocks_per_seq = attn_metadata[layer_name]["max_blocks_per_seq"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        B = block_table.shape[0]
        tokens = hidden_states.shape[0]
        S_decode = tokens // B
        assert tokens == B * S_decode, (
            f"Decode token count ({tokens}) is not a multiple of the batch size ({B})."
        )

        hidden_states = hidden_states.to(self.dtype)
        S_ctx = max_blocks_per_seq * block_size
        nkh = self.num_key_value_heads_per_rank

        X = augment_hidden(hidden_states, positions).view(
            B, S_decode, self.augmented_hidden_size
        )

        # <-- MODEL-SPECIFIC: Identity rotation (cos = 1, sin = 0). Bloom has
        # no RoPE. The ALiBi lanes must stay unchanged.
        half_d = self.padded_head_dim // 2
        cos_kernel = torch.ones(
            half_d, B, S_decode, dtype=self.dtype, device=hidden_states.device
        )
        sin_kernel = torch.zeros(
            half_d, B, S_decode, dtype=self.dtype, device=hidden_states.device
        )

        attention_mask = NF.gen_attention_decode_mask(
            pos_ids=positions.view(1, B * S_decode).to(torch.float32),
            bs=B,
            q_head=self.num_attention_heads_per_rank,
            s_active=S_decode,
            s_prior=S_ctx,
            start_pos=None,
            block_len=block_size,
        )

        common_args = dict(
            X=X,
            X_hidden_dim_actual=self.augmented_hidden_size,
            rmsnorm_X_enabled=False,
            W_qkv=self.qkv_proj_weight,
            bias_qkv=None,
            rmsnorm_QK_pre_rope_enabled=False,
            rmsnorm_QK_post_rope_enabled=False,
            cos=cos_kernel,
            sin=sin_kernel,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            K_cache=self.k_cache,
            V_cache=self.v_cache,
            attention_mask=attention_mask,
            softmax_scale=self.scaling,
            sink=None,
            W_out=self.o_proj_weight,
            bias_out=None,
            transposed_out=False,
            out_in_sb=False,
        )

        if can_run_kernel(hidden_states):
            # >>> PARALLELISM: Fused megakernel with TP-sharded weights <<<
            # More than one KV head reaches the kernel only with a per-head
            # 3D block table (see qwen3 forward_decode).
            output, K_new, V_new = NF.attention_decode(
                active_blocks_table=build_per_head_block_table(block_table, nkh).to(
                    torch.int32
                ),
                update_cache=False,
                **common_args,
            )
            scatter_new_kv(
                self.k_cache,
                self.v_cache,
                K_new,
                V_new,
                slot_mapping,
                block_size,
                nkh,
                self.padded_head_dim,
            )
        else:
            # Torch fallback (CPU mode): 2D table, in-place cache write.
            output = NF.attention_decode(
                active_blocks_table=block_table.to(torch.int32),
                update_cache=True,
                kv_cache_update_idx=slot_mapping.view(B, S_decode).to(torch.uint32),
                **common_args,
            )

        output = output.reshape(tokens, self.hidden_size)

        # >>> PARALLELISM: TP all-reduce after megakernel <<<
        if self.world_size > 1:
            self.tp_group.all_reduce(output)

        # <-- MODEL-SPECIFIC: Replicated O bias after the collective.
        return output + self.o_proj_bias


# =============================================================================
# Section 4: MLP
# Mixed: PARALLELISM (TP intermediate sharding, SP collectives) +
#        MODEL-SPECIFIC (non-gated GeLU tanh MLP with biases)
# =============================================================================


class BloomMLP(nn.Module):
    """Non-gated MLP: dense_4h_to_h(gelu_tanh(dense_h_to_4h(x))).

    >>> PARALLELISM: TP <<<
    - The up projection and its bias are sharded on the intermediate dim.
    - The down bias is replicated. Each rank adds it after the collective.
    """

    def __init__(self, config: BloomConfig):
        super().__init__()

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.hidden_size = config.hidden_size
        self.intermediate_size_per_rank = config.intermediate_size // self.world_size

        self.up_proj_weight = nn.Parameter(
            torch.empty(
                config.hidden_size,
                self.intermediate_size_per_rank,
                dtype=config.torch_dtype,
            )
        )
        self.up_proj_bias = nn.Parameter(
            torch.empty(1, self.intermediate_size_per_rank, dtype=config.torch_dtype)
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(
                self.intermediate_size_per_rank,
                config.hidden_size,
                dtype=config.torch_dtype,
            )
        )
        self.down_proj_bias = nn.Parameter(
            torch.empty(config.hidden_size, dtype=config.torch_dtype)
        )

        set_weight_loader(
            self.up_proj_weight,
            _cast_loader(
                sharding_weight_loader(
                    shard_dim=1,
                    shard_size=self.intermediate_size_per_rank,
                    num_shards=self.world_size,
                    is_storage_transposed=True,
                )
            ),
        )
        set_weight_loader(
            self.up_proj_bias,
            bloom_row_bias_loader(self.intermediate_size_per_rank, self.world_size),
        )
        set_weight_loader(
            self.down_proj_weight,
            _cast_loader(
                sharding_weight_loader(
                    shard_dim=0,
                    shard_size=self.intermediate_size_per_rank,
                    num_shards=self.world_size,
                    is_storage_transposed=True,
                )
            ),
        )
        set_weight_loader(self.down_proj_bias, _replicated_loader())

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        # >>> PARALLELISM: All-gather from SP for full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        if can_run_kernel(hidden_states):
            # <-- MODEL-SPECIFIC: skip_gate gives act(up(x) + up_b) @ down. The
            # gate weight and the gate bias are not used, so the up tensors
            # fill those arguments. The decode (TKG) kernel reads the dtype of
            # its bias tile from gate_b, and it fails when only up_b is set.
            # The CTE kernel does not apply gate_b when skip_gate is set.
            output = NF.mlp(
                hidden_states,
                self.up_proj_weight,
                self.up_proj_weight,
                self.down_proj_weight,
                skip_gate=True,
                act_fn=ActFnType.GELU_Tanh_Approx,
                gate_b=self.up_proj_bias,
                up_b=self.up_proj_bias,
            )
        else:
            # Torch path (CPU mode). NF.mlp's torch fallback calls
            # torch.nn.functional.gelu, which Dynamo cannot trace when the
            # Neuron XLA overrides replace it, so the model computes the same
            # tanh-approximation GeLU with elementwise ops instead.
            up = hidden_states @ self.up_proj_weight + self.up_proj_bias
            output = gelu_tanh(up) @ self.down_proj_weight

        # >>> PARALLELISM: Combine TP shards <<<
        if self.world_size > 1:
            if is_prefill:
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                self.tp_group.all_reduce(output)

        return output + self.down_proj_bias


# =============================================================================
# Section 5: Decoder Layer
# =============================================================================


class BloomDecoderLayer(nn.Module):
    """Bloom block: x + attn(ln1(x)), then h + mlp(ln2(h)).

    <-- MODEL-SPECIFIC: apply_residual_connection_post_layernorm is False,
    so the residual is the input of each LayerNorm.
    """

    def __init__(self, config: BloomConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = BloomLayerNorm(
            config.hidden_size, config.layer_norm_epsilon, config.torch_dtype
        )
        self.post_attention_layernorm = BloomLayerNorm(
            config.hidden_size, config.layer_norm_epsilon, config.torch_dtype
        )
        self.self_attn = BloomAttention(config, layer_idx=layer_idx)
        self.mlp = BloomMLP(config)
        self.layer_idx = layer_idx

    def _is_decode(self, attn_metadata) -> bool:
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]
        return max_query_len <= decode_token_threshold

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        is_decode = self._is_decode(attn_metadata)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, is_prefill=not is_decode)
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Section 6: Model Backbone
# =============================================================================


class BloomModel(nn.Module):
    """Bloom backbone: embedding, embedding LayerNorm, blocks and ln_f.

    >>> PARALLELISM: SP (Sequence Parallelism) <<<
    During prefill, the embedding scatters the tokens across TP ranks. The
    model all-gathers the result after ln_f.
    """

    def __init__(self, config: BloomConfig):
        super().__init__()
        self.config = config

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: Vocab-sharded embedding <<<
        self.word_embeddings = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.word_embeddings.weight,
            _cast_loader(
                sharding_weight_loader(
                    shard_dim=0,
                    shard_size=self.word_embeddings.vocab_size_per_rank,
                    num_shards=self.world_size,
                    is_storage_transposed=False,
                )
            ),
        )

        # <-- MODEL-SPECIFIC: LayerNorm directly after the embedding
        self.word_embeddings_layernorm = BloomLayerNorm(
            config.hidden_size, config.layer_norm_epsilon, config.torch_dtype
        )

        self.layers = nn.ModuleList(
            [BloomDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )

        self.ln_f = BloomLayerNorm(
            config.hidden_size, config.layer_norm_epsilon, config.torch_dtype
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

        hidden_states = self.word_embeddings(
            input_ids, scatter_tokens=is_prefill, rank=rank
        )
        hidden_states = self.word_embeddings_layernorm(hidden_states)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                positions=positions,
                attn_metadata=attn_metadata,
            )

        hidden_states = self.ln_f(hidden_states)

        # >>> PARALLELISM: SP - all-gather to reconstruct full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return hidden_states


# =============================================================================
# Section 7: Language Model Head
# =============================================================================


def _validate_tp(config: BloomConfig, world_size: int) -> None:
    """Check that the TP degree gives a legal head and vocab sharding."""
    heads = config.num_attention_heads
    if heads % world_size != 0:
        raise ValueError(
            f"Bloom has {heads} attention heads. This number is not divisible by "
            f"the tensor parallel size {world_size}. Use a divisor of {heads}."
        )
    if (heads // world_size) % 2 != 0:
        raise ValueError(
            f"Bloom with tensor parallel size {world_size} gives "
            f"{heads // world_size} heads for each rank. The decode kernel needs "
            "an even number. Use a smaller tensor parallel size."
        )
    for name, value in (
        ("vocab_size", config.vocab_size),
        ("intermediate_size", config.intermediate_size),
    ):
        if value % world_size != 0:
            raise ValueError(
                f"Bloom {name} {value} is not divisible by the tensor parallel "
                f"size {world_size}. Use a divisor of {value}."
            )


def _current_max_model_len() -> int | None:
    """Return the max_model_len of the current vLLM config, or None."""
    try:
        from vllm.config import get_current_vllm_config

        return get_current_vllm_config().model_config.max_model_len
    except Exception:
        return None


class BloomForCausalLM(nn.Module):
    """Bloom model with a tied, column-parallel language model head."""

    def __init__(self, config: BloomConfig):
        super().__init__()
        self.config = config

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        _validate_tp(config, self.world_size)
        validate_max_model_len(_current_max_model_len())

        self.model = BloomModel(config)

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

        # <-- MODEL-SPECIFIC: Tied head. load_weights reads lm_head.weight from
        # the word_embeddings key. Bloom has no lm_head bias.
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
            _cast_loader(
                sharding_weight_loader(
                    shard_dim=0,
                    shard_size=config.vocab_size // self.world_size,
                    num_shards=self.world_size,
                    is_storage_transposed=False,
                )
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
                "Bloom does not support inputs_embeds. Give input_ids instead."
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
                f"Prompt length ({T}) must be larger than world_size "
                f"({self.world_size}) and divisible by it for SP."
            )

        hidden_states = self.model(
            input_ids, positions, attn_metadata=attn_metadata, rank=rank
        )
        hidden_states_for_logits = torch.index_select(
            hidden_states, dim=0, index=sampling_positions
        )
        logits = self.lm_head(hidden_states_for_logits)

        if self.on_device_sampling_config is None:
            return logits

        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )

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
    def from_configs(cls, hf_config, neuron_config: NeuronConfig | None):
        config = BloomConfig.from_configs(hf_config, neuron_config)
        return cls(config)

    # -- KV Cache Management --------------------------------------------------

    def get_kv_spec(self) -> KVSpec:
        # <-- MODEL-SPECIFIC: The cache stores the padded 128-lane heads, which
        # include the ALiBi position lanes of K.
        layers = []
        for i, layer in enumerate(self.model.layers):
            layers.append(
                LayerSpec(
                    name=f"layers.{i}.self_attn",
                    num_kv_heads=layer.self_attn.num_key_value_heads_per_rank,
                    head_size=layer.self_attn.padded_head_dim,
                    dtype=layer.self_attn.dtype,
                    sliding_window_size=None,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            if layer_name not in kv_caches:
                raise RuntimeError(
                    f"KV cache for layer {layer_name} is not initialized. Allocate "
                    "the KV cache before the forward pass."
                )
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]

    # -- Weight Loading -------------------------------------------------------

    def checkpoint_mappings(self, prefix: str = "") -> dict[str, str | list[str]]:
        """Map each model parameter to its HF Bloom checkpoint key(s).

        ``prefix`` is ``"transformer."`` or ``""``, as the checkpoint stores it.
        """
        p = prefix
        mappings: dict[str, str | list[str]] = {
            "model.word_embeddings.weight": f"{p}word_embeddings.weight",
            "model.word_embeddings_layernorm.weight": (
                f"{p}word_embeddings_layernorm.weight"
            ),
            "model.word_embeddings_layernorm.bias": (
                f"{p}word_embeddings_layernorm.bias"
            ),
            "model.ln_f.weight": f"{p}ln_f.weight",
            "model.ln_f.bias": f"{p}ln_f.bias",
        }
        for i in range(len(self.model.layers)):
            src = f"{p}h.{i}"
            dst = f"model.layers.{i}"
            for norm in ("input_layernorm", "post_attention_layernorm"):
                for field in ("weight", "bias"):
                    mappings[f"{dst}.{norm}.{field}"] = f"{src}.{norm}.{field}"
            mappings[f"{dst}.self_attn.qkv_proj_weight"] = [
                f"{src}.self_attention.query_key_value.weight",
                f"{src}.self_attention.query_key_value.bias",
            ]
            mappings[f"{dst}.self_attn.o_proj_weight"] = (
                f"{src}.self_attention.dense.weight"
            )
            mappings[f"{dst}.self_attn.o_proj_bias"] = (
                f"{src}.self_attention.dense.bias"
            )
            mappings[f"{dst}.mlp.up_proj_weight"] = f"{src}.mlp.dense_h_to_4h.weight"
            mappings[f"{dst}.mlp.up_proj_bias"] = f"{src}.mlp.dense_h_to_4h.bias"
            mappings[f"{dst}.mlp.down_proj_weight"] = f"{src}.mlp.dense_4h_to_h.weight"
            mappings[f"{dst}.mlp.down_proj_bias"] = f"{src}.mlp.dense_4h_to_h.bias"

        # <-- MODEL-SPECIFIC: Tied embeddings
        if self.config.tie_word_embeddings:
            mappings["lm_head.weight"] = f"{p}word_embeddings.weight"
        else:
            mappings["lm_head.weight"] = "lm_head.weight"
        return mappings

    @staticmethod
    def detect_checkpoint_prefix(tensor_names) -> str:
        """Return ``"transformer."`` if the checkpoint keys use that prefix."""
        if "transformer.word_embeddings.weight" in tensor_names:
            return "transformer."
        if "word_embeddings.weight" in tensor_names:
            return ""
        raise RuntimeError(
            "The checkpoint has no Bloom word_embeddings.weight key. Use a Bloom "
            "safetensors checkpoint."
        )

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load and shard the HF Bloom checkpoint. A missing key raises."""
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        prefix = self.detect_checkpoint_prefix(checkpoint.get_tensor_names())
        mappings = self.checkpoint_mappings(prefix)

        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, mappings, device
        ).state_dict

        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        self.load_state_dict(rank_sharded, strict=False, assign=True)
