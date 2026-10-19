# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL vision encoder (BF16).

Complete vision transformer implementation: PatchEmbed → Blocks × depth → Merger.
Produces main embeddings and deepstack features at specified intermediate layers.

libtorch_neuronx_lite patches F.gelu/nn.GELU with a C extension that torch.compile
(Dynamo) cannot trace through. This module provides erf-based equivalents
decorated with @torch.compiler.allow_in_graph.

All tensors flow as 3D ``[num_blocks, block_size, hidden_size]`` through the
vision encoder (block packing layout). Each block is processed independently
in attention via bounds masking. After the merger, the fat tensor (main +
deepstack) is scatter-written directly into the on-device encoder cache buffer
via index_put_.

Supported parallelism: TP (vision-specific TP group, decoupled from text model).

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    Qwen3-VL vision-specific. Change when porting.

PARALLELISM SHARDING:
  TP:  Attention heads sharded, MLP intermediate sharded, merger MLP sharded.
       Vision TP group is independent of text model TP group.
       No SP — vision encoder processes full sequence on all TP ranks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm_neuron.functional as NF
from vllm_neuron.parallel.neuron_parallel_state import get_neuron_encoder_tp_group
from vllm_neuron.utils.weight_loader import set_weight_loader, sharding_weight_loader
from vllm_neuron.utils.dynamo_graph import allow_in_graph_cacheable

from .weight_loaders import vis_qkv_bias_loader, vis_qkv_weight_loader

# The sequence-packed attention kernel loads bounds in tiles of this size and
# reads ceil(block_size / align) full tiles per block, so the vision attention
# sequence length must be a multiple of it or the NEFF bakes an out-of-bounds
# DMA that the runtime rejects at load. The vision encoder pads its sequence up
# to this alignment internally (see Qwen3VLVisionModel.forward).
_ATTN_SEQ_ALIGN = 128


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


@allow_in_graph_cacheable
def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU activation via erf, traceable by torch.compile on Neuron."""
    return 0.5 * x * (1.0 + torch.erf(x / 1.4142135623730951))


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class Qwen3VLVisionMLP(nn.Module):
    """Vision transformer MLP block with tensor-parallel sharding.

    Architecture: ``fc1(x) → GELU → fc2(x)``

    Parallelism:
    - fc1 weight is sharded on the intermediate dim (dim=1) across TP ranks.
    - fc2 weight is sharded on the intermediate dim (dim=0) across TP ranks.
    - TP all-reduce after fc2 matmul, then fc2 bias is added.

    Args:
        hidden_size: Input and output dimension.
        intermediate_size: Hidden dimension of the MLP.
        dtype: Weight dtype. Default: torch.bfloat16.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()

        # >>> PARALLELISM: Vision TP group setup <<<
        self.tp_group = get_neuron_encoder_tp_group()
        self.world_size = self.tp_group.world_size

        # >>> PARALLELISM: Intermediate dim sharded across vision TP <<<
        self.intermediate_size_per_rank = intermediate_size // self.world_size

        self.fc1_weight = nn.Parameter(
            torch.empty(hidden_size, self.intermediate_size_per_rank, dtype=dtype)
        )
        self.fc1_bias = nn.Parameter(
            torch.empty(self.intermediate_size_per_rank, dtype=dtype)
        )
        self.fc2_weight = nn.Parameter(
            torch.empty(self.intermediate_size_per_rank, hidden_size, dtype=dtype)
        )
        self.fc2_bias = nn.Parameter(torch.empty(hidden_size, dtype=dtype))

        self._setup_weight_loaders()

    def _setup_weight_loaders(self) -> None:
        """>>> PARALLELISM: TP sharding of MLP weights. <<<"""
        set_weight_loader(
            self.fc1_weight,
            sharding_weight_loader(
                shard_dim=1,
                shard_size=self.intermediate_size_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )
        set_weight_loader(
            self.fc1_bias,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.intermediate_size_per_rank,
                num_shards=self.world_size,
            ),
        )
        set_weight_loader(
            self.fc2_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.intermediate_size_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MLP forward: fc1 → GELU → fc2 → all-reduce → + bias.

        Matmul broadcasts over leading dims, so this works with any batch
        shape (e.g., ``[num_blocks, block_size, hidden_size]``).

        Args:
            hidden_states: ``[..., hidden_size]`` — arbitrary leading dims.

        Returns:
            ``[..., hidden_size]`` — same leading dims as input.
        """
        output = hidden_states @ self.fc1_weight + self.fc1_bias
        output = gelu(output)  # <-- MODEL-SPECIFIC: erf GELU for Neuron traceability
        output = output @ self.fc2_weight

        # >>> PARALLELISM: All-reduce across vision TP ranks <<<
        if self.world_size > 1:
            self.tp_group.all_reduce(output)

        output = output + self.fc2_bias

        return output


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Interleaved (rotate_half) RoPE: split into second/first half, negate first."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Qwen3VLVisionAttention(nn.Module):
    """Vision transformer self-attention with 2D RoPE and bounds masking.

    Architecture:
    - Fused QKV projection (with bias) via raw weight parameters
    - 2D RoPE applied to Q and K (interleaved rotate_half style)
    - Bidirectional attention with bounds masking (no causal mask)
    - Output projection (with bias) via raw weight parameters + TP all-reduce

    Parallelism:
    - QKV weight sharded via vis_qkv_weight_loader (interleaved head sharding)
    - Output projection weight sharded on input dim, all-reduced across TP ranks
    - No SP — vision encoder processes full sequence on all TP ranks
    - No GQA — num_heads Q = num_heads K = num_heads V

    Args:
        num_heads: Total number of attention heads.
        head_dim: Dimension per attention head.
        hidden_size: Total hidden dimension (num_heads * head_dim).
        dtype: Weight dtype. Default: torch.bfloat16.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.scale = head_dim**-0.5
        self.dtype = dtype

        # >>> PARALLELISM: Vision TP group setup <<<
        self.tp_group = get_neuron_encoder_tp_group()
        self.world_size = self.tp_group.world_size

        # >>> PARALLELISM: Head sharding calculation <<<
        assert num_heads % self.world_size == 0, (
            f"num_heads {num_heads} must be divisible by tp world_size {self.world_size}"
        )
        self.num_heads_per_rank: int = num_heads // self.world_size

        # >>> PARALLELISM: QKV weight shapes (per-rank sharded heads) <<<
        qkv_size_per_rank = 3 * self.num_heads_per_rank * self.head_dim
        self.qkv_weight = nn.Parameter(
            torch.empty(hidden_size, qkv_size_per_rank, dtype=dtype)
        )
        self.qkv_bias = nn.Parameter(torch.empty(qkv_size_per_rank, dtype=dtype))

        # >>> PARALLELISM: O-proj input sharded across TP ranks <<<
        o_proj_in = self.num_heads_per_rank * self.head_dim
        self.proj_weight = nn.Parameter(
            torch.empty(o_proj_in, hidden_size, dtype=dtype)
        )
        self.proj_bias = nn.Parameter(torch.empty(hidden_size, dtype=dtype))

        self._setup_weight_loaders()

    def _setup_weight_loaders(self) -> None:
        """>>> PARALLELISM: Weight loaders handle TP sharding of checkpoint tensors. <<<"""
        set_weight_loader(
            self.qkv_weight,
            vis_qkv_weight_loader(
                self.num_heads_per_rank, self.head_dim, self.hidden_size
            ),
        )
        set_weight_loader(
            self.qkv_bias,
            vis_qkv_bias_loader(
                self.num_heads_per_rank, self.head_dim, self.hidden_size
            ),
        )
        set_weight_loader(
            self.proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.num_heads_per_rank * self.head_dim,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        bound_min: torch.Tensor,
        bound_max: torch.Tensor,
    ) -> torch.Tensor:
        """Attention forward with per-block bounds masking.

        Uses block packing layout: each block is processed independently via
        bounds masking. The batch dimension (dim 0) is num_blocks — blocks are
        treated as independent sequences for attention.

        Args:
            hidden_states: ``[num_blocks, block_size, hidden_size]``.
            cos: ``[num_blocks, block_size, head_dim]`` — per-block RoPE cosine.
            sin: ``[num_blocks, block_size, head_dim]`` — per-block RoPE sine.
            bound_min: ``[num_blocks, block_size, 1]`` int32 — inclusive lower bound.
            bound_max: ``[num_blocks, block_size, 1]`` int32 — exclusive upper bound.

        Returns:
            ``[num_blocks, block_size, hidden_size]``
        """
        num_blocks, block_size, _ = hidden_states.shape
        hidden_states = hidden_states.to(self.dtype)

        # >>> PARALLELISM: Each rank computes only its sharded Q/K/V heads <<<
        # qkv: [num_blocks, block_size, 3 * heads_per_rank * head_dim]
        qkv = NF.qkv_proj(
            hidden=hidden_states,
            qkv_weights=self.qkv_weight,
            bias=self.qkv_bias.unsqueeze(0),
        )

        hd = self.num_heads_per_rank * self.head_dim
        q = qkv[:, :, :hd]
        k = qkv[:, :, hd : 2 * hd]
        v = qkv[:, :, 2 * hd :]

        # [num_blocks, block_size, heads_per_rank * head_dim]
        # → [num_blocks, block_size, heads_per_rank, head_dim]
        q = q.view(num_blocks, block_size, self.num_heads_per_rank, self.head_dim)
        k = k.view(num_blocks, block_size, self.num_heads_per_rank, self.head_dim)
        v = v.view(num_blocks, block_size, self.num_heads_per_rank, self.head_dim)

        # <-- MODEL-SPECIFIC: 2D RoPE (interleaved rotate_half style)
        # cos/sin: [num_blocks, block_size, head_dim] → [num_blocks, block_size, 1, head_dim]
        cos = cos.to(dtype=hidden_states.dtype)
        sin = sin.to(dtype=hidden_states.dtype)
        cos_sq = cos.unsqueeze(2)
        sin_sq = sin.unsqueeze(2)
        q = q * cos_sq + _rotate_half(q) * sin_sq
        k = k * cos_sq + _rotate_half(k) * sin_sq

        # [num_blocks, block_size, heads_per_rank, head_dim]
        # → [num_blocks * heads_per_rank, block_size, head_dim]
        q = q.permute(0, 2, 1, 3).reshape(
            num_blocks * self.num_heads_per_rank, block_size, self.head_dim
        )
        k = k.permute(0, 2, 1, 3).reshape(
            num_blocks * self.num_heads_per_rank, block_size, self.head_dim
        )
        v = v.permute(0, 2, 1, 3).reshape(
            num_blocks * self.num_heads_per_rank, block_size, self.head_dim
        )

        # <-- MODEL-SPECIFIC: Bidirectional attention with per-block bounds masking
        # bound_min/max: [num_blocks, block_size, 1]
        # → [num_blocks * heads_per_rank, block_size, 1] (repeat per head)
        attn_bound_min = bound_min.repeat_interleave(self.num_heads_per_rank, dim=0)
        attn_bound_max = bound_max.repeat_interleave(self.num_heads_per_rank, dim=0)

        attn_output = NF.flash_attention(
            q,
            k,
            v,
            scale=self.scale,
            causal_mask=False,
            tp_q=True,
            tp_k=True,
            bound_min=attn_bound_min,
            bound_max=attn_bound_max,
        )

        # >>> PARALLELISM: O-proj + all-reduce across vision TP ranks <<<
        # [num_blocks * heads_per_rank, block_size, head_dim]
        # → [num_blocks, heads_per_rank, head_dim, block_size] for NF.o_proj
        attn_output = attn_output.view(
            num_blocks, self.num_heads_per_rank, block_size, self.head_dim
        ).permute(0, 1, 3, 2)  # [num_blocks, N, D, S]

        output = NF.o_proj(attn_output, self.proj_weight)  # [num_blocks, S, H]

        if self.world_size > 1:
            self.tp_group.all_reduce(output)

        output = output + self.proj_bias

        return output


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------


class Qwen3VLVisionBlock(nn.Module):
    """Single vision transformer block: norm1 → attention → norm2 → MLP.

    Standard pre-norm ViT block layout with residual connections.

    Args:
        num_heads: Total number of attention heads.
        head_dim: Dimension per attention head.
        hidden_size: Vision hidden dimension.
        intermediate_size: MLP intermediate dimension.
        dtype: Weight dtype. Default: torch.bfloat16.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, dtype=dtype)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, dtype=dtype)
        self.attn = Qwen3VLVisionAttention(num_heads, head_dim, hidden_size, dtype)
        self.mlp = Qwen3VLVisionMLP(hidden_size, intermediate_size, dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        bound_min: torch.Tensor,
        bound_max: torch.Tensor,
    ) -> torch.Tensor:
        """Block forward with pre-norm residual connections.

        Args:
            hidden_states: ``[num_blocks, block_size, hidden_size]``
            cos: ``[num_blocks, block_size, head_dim]`` — per-block RoPE cosine.
            sin: ``[num_blocks, block_size, head_dim]`` — per-block RoPE sine.
            bound_min: ``[num_blocks, block_size, 1]`` int32
            bound_max: ``[num_blocks, block_size, 1]`` int32

        Returns:
            ``[num_blocks, block_size, hidden_size]``
        """
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cos, sin, bound_min, bound_max
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------


class Qwen3VLVisionPatchEmbed(nn.Module):
    """Convert raw pixel patches to patch embeddings via Conv3d.

    Input:  ``[1, seq_len, C * temporal_patch_size * patch_size * patch_size]``
    Output: ``[1, seq_len, vis_hidden_size]``

    Args:
        in_channels: Number of input image channels (e.g. 3 for RGB).
        temporal_patch_size: Temporal patch size (e.g. 2).
        patch_size: Spatial patch size (e.g. 16).
        hidden_size: Vision encoder hidden dimension (output channels).
        dtype: Weight dtype. Default: torch.bfloat16.
    """

    def __init__(
        self,
        in_channels: int = 3,
        temporal_patch_size: int = 2,
        patch_size: int = 16,
        hidden_size: int = 1152,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.embed_dim = hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
            dtype=dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project flattened patches to vis_hidden_size.

        Supports block packing layout: input has shape
        ``[num_blocks, block_size, patch_dim]``.

        Args:
            hidden_states: ``[num_blocks, block_size, C * temporal_patch_size * patch_size²]``

        Returns:
            ``[num_blocks, block_size, vis_hidden_size]``
        """
        target_dtype = self.proj.weight.dtype
        num_blocks, block_size, _ = hidden_states.shape
        hidden_states = hidden_states.reshape(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(hidden_states.to(dtype=target_dtype)).view(
            num_blocks, block_size, self.embed_dim
        )


# ---------------------------------------------------------------------------
# Patch Merger
# ---------------------------------------------------------------------------


class Qwen3VLVisionPatchMerger(nn.Module):
    """Spatial patch merger with two-layer MLP projection.

    Merges ``spatial_merge_size²`` adjacent patches by concatenating their
    hidden states, then projects through ``fc1 → GELU → fc2`` to produce
    tokens of dimension ``out_hidden_size``.

    Two variants controlled by ``use_postshuffle_norm``:
    - ``False`` (main merger): norm on ``[hidden_size]`` *before* spatial reshape
    - ``True`` (deepstack mergers): norm on ``[merged_hidden_size]`` *after* spatial reshape

    Parallelism:
    - fc1 weight sharded on output dim (ColumnParallel style) across TP ranks.
    - fc2 weight sharded on input dim (RowParallel style) across TP ranks.
    - TP all-reduce after fc2.
    - fc2 bias added after all-reduce (not sharded).

    Args:
        hidden_size: Vision encoder hidden dimension.
        out_hidden_size: Text model hidden dimension (output of merger).
        spatial_merge_size: Number of patches merged per spatial dimension.
        use_postshuffle_norm: If True, apply LayerNorm after spatial reshape.
        dtype: Weight dtype. Default: torch.bfloat16.
    """

    def __init__(
        self,
        hidden_size: int,
        out_hidden_size: int,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.use_postshuffle_norm = use_postshuffle_norm
        self.spatial_merge_unit = spatial_merge_size**2
        self.merged_hidden_size = hidden_size * self.spatial_merge_unit

        # >>> PARALLELISM: Vision TP group setup <<<
        self.tp_group = get_neuron_encoder_tp_group()
        self.world_size = self.tp_group.world_size

        # >>> PARALLELISM: Merged hidden dim sharded across vision TP <<<
        self.merged_hidden_per_rank = self.merged_hidden_size // self.world_size

        norm_size = self.merged_hidden_size if use_postshuffle_norm else hidden_size
        self.norm = nn.LayerNorm(norm_size, eps=1e-6, dtype=dtype)

        self.fc1_weight = nn.Parameter(
            torch.empty(
                self.merged_hidden_size, self.merged_hidden_per_rank, dtype=dtype
            )
        )
        self.fc1_bias = nn.Parameter(
            torch.empty(self.merged_hidden_per_rank, dtype=dtype)
        )
        self.fc2_weight = nn.Parameter(
            torch.empty(self.merged_hidden_per_rank, out_hidden_size, dtype=dtype)
        )
        self.fc2_bias = nn.Parameter(torch.empty(out_hidden_size, dtype=dtype))

        self._setup_weight_loaders()

    def _setup_weight_loaders(self) -> None:
        """>>> PARALLELISM: TP sharding of merger MLP weights. <<<"""
        set_weight_loader(
            self.fc1_weight,
            sharding_weight_loader(
                shard_dim=1,
                shard_size=self.merged_hidden_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )
        set_weight_loader(
            self.fc1_bias,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.merged_hidden_per_rank,
                num_shards=self.world_size,
            ),
        )
        set_weight_loader(
            self.fc2_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.merged_hidden_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Merge spatial patches and project to text hidden dimension.

        Supports block packing layout: input is 3D
        ``[num_blocks, block_size, hidden_size]`` where ``block_size`` is
        divisible by ``spatial_merge_size²``.

        Args:
            hidden_states: ``[num_blocks, block_size, hidden_size]``

        Returns:
            ``[num_blocks, merged_block_size, out_hidden_size]``
            where ``merged_block_size = block_size // spatial_merge_size²``.
        """
        num_blocks, block_size, _ = hidden_states.shape
        assert block_size % self.spatial_merge_unit == 0, (
            f"block_size {block_size} must be divisible by spatial_merge_unit "
            f"{self.spatial_merge_unit} to avoid cross-block token merging"
        )

        # <-- MODEL-SPECIFIC: Spatial merge reshape (pre/post-shuffle norm variants)
        # Flatten blocks for spatial merge: [num_blocks * block_size, hidden]
        # then group merge_size² adjacent patches into merged tokens.
        if self.use_postshuffle_norm:
            x = hidden_states.reshape(-1, self.merged_hidden_size)
            x = self.norm(x)
        else:
            x = self.norm(hidden_states).reshape(-1, self.merged_hidden_size)

        x = (x @ self.fc1_weight) + self.fc1_bias
        x = gelu(x)

        x = x @ self.fc2_weight

        # >>> PARALLELISM: All-reduce across vision TP ranks <<<
        if self.world_size > 1:
            self.tp_group.all_reduce(x)

        x = x + self.fc2_bias

        # Reshape back to [num_blocks, merged_block_size, out_hidden_size]
        merged_block_size = x.shape[0] // num_blocks
        return x.view(num_blocks, merged_block_size, -1)


# ---------------------------------------------------------------------------
# On-device position embedding (bilinear interpolation)
# ---------------------------------------------------------------------------


class Qwen3VLVisionPosEmbed(nn.Module):
    """Neuron-device computation: given padded interpolation indices and weights
    (precomputed on CPU), performs the embedding lookup and bilinear weighted sum.
    This runs as part of the compiled static graph on the Neuron device.
    """

    def __init__(self, vision_config, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.pos_embed = nn.Embedding(
            vision_config.num_position_embeddings,
            vision_config.hidden_size,
            dtype=dtype,
        )

    def forward(
        self, pos_emb_idx: torch.Tensor, pos_emb_weight: torch.Tensor
    ) -> torch.Tensor:
        """Lookup and blend 4 bilinear corners on device.

        Args:
            pos_emb_idx: ``[4, num_blocks, block_size]`` int32.
            pos_emb_weight: ``[4, num_blocks, block_size]`` bf16.

        Returns:
            ``[num_blocks, block_size, hidden_size]`` — interpolated embeddings.
            Padded positions (weight=0) produce zeros.
        """
        # Indices transferred as int32 to minimize CPU→device transfer; nn.Embedding requires int64
        embeds = self.pos_embed(pos_emb_idx.long())  # [4, N, B, H]
        weighted = embeds * pos_emb_weight.unsqueeze(-1)  # [4, N, B, H]
        return weighted.sum(dim=0)  # [N, B, H]


# ---------------------------------------------------------------------------
# Vision Model (top-level)
# ---------------------------------------------------------------------------


class Qwen3VLVisionModel(nn.Module):
    """Complete Qwen3-VL vision encoder (ViT + mergers).

    Architecture::

        pixel_values → PatchEmbed → + pos_embeds
        → Block_0 → Block_1 → ... → Block_{depth-1}
                      ↓ (at deepstack_visual_indexes)
                      DeepstackMerger_i → deepstack_features[i]
        → Merger → main_embeddings

        Returns: (main_embeddings, deepstack_features)

    Args:
        config: Vision config object with attributes: hidden_size,
            intermediate_size, out_hidden_size, num_heads, depth,
            patch_size, temporal_patch_size, in_channels,
            spatial_merge_size, deepstack_visual_indexes.
        dtype: Weight dtype. Default: torch.bfloat16.
    """

    def __init__(self, config, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.config = config
        self.deepstack_visual_indexes = (
            getattr(config, "deepstack_visual_indexes", None) or []
        )
        head_dim = config.hidden_size // config.num_heads

        self.pos_embed = Qwen3VLVisionPosEmbed(config, dtype=dtype)

        # >>> PARALLELISM: Vision DP group for block scatter/gather <<<
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_encoder_dp_group,
        )

        self.dp_group = get_neuron_encoder_dp_group()
        self.dp_size = self.dp_group.world_size if self.dp_group else 1

        self.patch_embed = Qwen3VLVisionPatchEmbed(
            in_channels=config.in_channels,
            temporal_patch_size=config.temporal_patch_size,
            patch_size=config.patch_size,
            hidden_size=config.hidden_size,
            dtype=dtype,
        )

        self.blocks = nn.ModuleList(
            [
                Qwen3VLVisionBlock(
                    num_heads=config.num_heads,
                    head_dim=head_dim,
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    dtype=dtype,
                )
                for _ in range(config.depth)
            ]
        )

        self.merger = Qwen3VLVisionPatchMerger(
            hidden_size=config.hidden_size,
            out_hidden_size=config.out_hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            use_postshuffle_norm=False,
            dtype=dtype,
        )

        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(
                    hidden_size=config.hidden_size,
                    out_hidden_size=config.out_hidden_size,
                    spatial_merge_size=config.spatial_merge_size,
                    use_postshuffle_norm=True,
                    dtype=dtype,
                )
                for _ in range(len(self.deepstack_visual_indexes))
            ]
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        pos_emb_idx: torch.Tensor,
        pos_emb_weight: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        bound_min: torch.Tensor,
        bound_max: torch.Tensor,
        encoder_cache_buffer: torch.Tensor,
        write_block_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Full vision encoder forward with cache-write.

        All input shapes are STATIC (determined by bucket selection at compile
        time). Variable image content is handled via data values (padding zeros,
        bounds masking), not shape changes.

        Scatter-writes the merged fat tensor directly into the encoder cache
        buffer at positions specified by write_block_ids via index_put_.

        Args:
            pixel_values: ``[num_blocks, block_size, patch_dim]``
            pos_emb_idx: ``[4, num_blocks, block_size]`` int32
            pos_emb_weight: ``[4, num_blocks, block_size]`` bf16
            cos: ``[num_blocks, block_size, head_dim]``
            sin: ``[num_blocks, block_size, head_dim]``
            bound_min: ``[num_blocks, block_size, 1]`` int32
            bound_max: ``[num_blocks, block_size, 1]`` int32
            encoder_cache_buffer: ``[num_cache_blocks, cache_block_size, fat_dim]``
                Pre-allocated cache buffer on device (input-output alias).
            write_block_ids: ``[num_ve_blocks]`` int64
                Maps VE output block i to cache buffer block index.

        Returns:
            Updated encoder_cache_buffer (same tensor, aliased).
        """
        # >>> PARALLELISM: DP scatter — each rank takes its slice of blocks <<<
        if self.dp_size > 1:
            assert pixel_values.shape[0] % self.dp_size == 0, (
                f"num_blocks ({pixel_values.shape[0]}) must be divisible by "
                f"dp_size ({self.dp_size})"
            )
            blocks_per_rank = pixel_values.shape[0] // self.dp_size
            dp_offset = (
                self.dp_group.rank_in_group_tensor.to(torch.int64) * blocks_per_rank
            )
            dp_indices = (
                torch.arange(
                    blocks_per_rank, device=pixel_values.device, dtype=torch.int64
                )
                + dp_offset
            )
            pixel_values = pixel_values.index_select(0, dp_indices)
            pos_emb_idx = pos_emb_idx.index_select(1, dp_indices)
            pos_emb_weight = pos_emb_weight.index_select(1, dp_indices)
            cos = cos.index_select(0, dp_indices)
            sin = sin.index_select(0, dp_indices)
            bound_min = bound_min.index_select(0, dp_indices)
            bound_max = bound_max.index_select(0, dp_indices)

        # Pad the attention sequence dim up to a multiple of _ATTN_SEQ_ALIGN (see
        # the constant for why). Pad rows carry bound_min == bound_max == 0 so
        # they attend to nothing, and their sequence positions sit past every real
        # frame's bound_max so no real query attends them. The pad merged rows are
        # sliced off after the merger, before the cache write.
        real_block_size = pixel_values.shape[1]
        merge_unit = self.merger.spatial_merge_unit
        padded_block_size = (
            (real_block_size + _ATTN_SEQ_ALIGN - 1) // _ATTN_SEQ_ALIGN
        ) * _ATTN_SEQ_ALIGN
        seq_pad = padded_block_size - real_block_size
        real_merged = real_block_size // merge_unit
        if seq_pad > 0:
            pixel_values = F.pad(pixel_values, (0, 0, 0, seq_pad))
            cos = F.pad(cos, (0, 0, 0, seq_pad))
            sin = F.pad(sin, (0, 0, 0, seq_pad))
            bound_min = F.pad(bound_min, (0, 0, 0, seq_pad))
            bound_max = F.pad(bound_max, (0, 0, 0, seq_pad))
            # pos_emb_idx / pos_emb_weight are [4, num_blocks, block_size]: pad
            # the trailing block_size dim.
            pos_emb_idx = F.pad(pos_emb_idx, (0, seq_pad))
            pos_emb_weight = F.pad(pos_emb_weight, (0, seq_pad))

        # Step 1: Patch embedding + position embedding
        # [num_blocks, block_size, patch_dim] → [num_blocks, block_size, hidden]
        hidden_states = self.patch_embed(pixel_values)
        pos_embeds = self.pos_embed(pos_emb_idx, pos_emb_weight)
        hidden_states = hidden_states + pos_embeds

        # Step 2: Transformer blocks with deepstack collection
        deepstack_features: list[torch.Tensor] = []
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, cos, sin, bound_min, bound_max)

            if layer_idx in self.deepstack_visual_indexes:
                ds_idx = self.deepstack_visual_indexes.index(layer_idx)
                ds_merged = self.deepstack_merger_list[ds_idx](hidden_states)
                deepstack_features.append(ds_merged)

        # Step 3: Final merger
        # [num_blocks, block_size, H] → [num_blocks, merged_bs, D]
        main_merged = self.merger(hidden_states)
        # Drop the merged rows produced from the sequence padding so the output
        # matches the real (unpadded) cache_block_size the runner allocated.
        if seq_pad > 0:
            main_merged = main_merged[:, :real_merged]
            deepstack_features = [ds[:, :real_merged] for ds in deepstack_features]
        # >>> PARALLELISM: DP gather — combine blocks from all DP ranks <<<
        if self.dp_size > 1:
            main_merged = self.dp_group.all_gather(main_merged, dim=0)
            deepstack_features = [
                self.dp_group.all_gather(ds, dim=0) for ds in deepstack_features
            ]

        # Construct fat tensor in block layout [num_ve_blocks, merged_bs, fat_dim]
        if deepstack_features:
            ds_stacked = torch.stack(deepstack_features, dim=0)
            N = ds_stacked.shape[0]
            ds_cat = ds_stacked.permute(1, 2, 0, 3).reshape(
                main_merged.shape[0], main_merged.shape[1], N * main_merged.shape[2]
            )
            fat_blocks = torch.cat([main_merged, ds_cat], dim=-1)
        else:
            fat_blocks = main_merged

        # Scatter-write into encoder cache buffer
        encoder_cache_buffer.index_put_((write_block_ids,), fat_blocks)
        return encoder_cache_buffer

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    @staticmethod
    def build_weight_mappings(
        depth: int, deepstack_visual_indexes: list[int]
    ) -> dict[str, str]:
        """Build parameter name → checkpoint key mappings for the full vision model."""
        mappings: dict[str, str] = {}

        mappings["pos_embed.pos_embed.weight"] = "model.visual.pos_embed.weight"
        mappings["patch_embed.proj.weight"] = "model.visual.patch_embed.proj.weight"
        mappings["patch_embed.proj.bias"] = "model.visual.patch_embed.proj.bias"

        for i in range(depth):
            prefix = f"blocks.{i}"
            ckpt_prefix = f"model.visual.blocks.{i}"

            mappings[f"{prefix}.attn.qkv_weight"] = f"{ckpt_prefix}.attn.qkv.weight"
            mappings[f"{prefix}.attn.qkv_bias"] = f"{ckpt_prefix}.attn.qkv.bias"
            mappings[f"{prefix}.attn.proj_weight"] = f"{ckpt_prefix}.attn.proj.weight"
            mappings[f"{prefix}.attn.proj_bias"] = f"{ckpt_prefix}.attn.proj.bias"

            mappings[f"{prefix}.mlp.fc1_weight"] = (
                f"{ckpt_prefix}.mlp.linear_fc1.weight"
            )
            mappings[f"{prefix}.mlp.fc1_bias"] = f"{ckpt_prefix}.mlp.linear_fc1.bias"
            mappings[f"{prefix}.mlp.fc2_weight"] = (
                f"{ckpt_prefix}.mlp.linear_fc2.weight"
            )
            mappings[f"{prefix}.mlp.fc2_bias"] = f"{ckpt_prefix}.mlp.linear_fc2.bias"

            mappings[f"{prefix}.norm1.weight"] = f"{ckpt_prefix}.norm1.weight"
            mappings[f"{prefix}.norm1.bias"] = f"{ckpt_prefix}.norm1.bias"
            mappings[f"{prefix}.norm2.weight"] = f"{ckpt_prefix}.norm2.weight"
            mappings[f"{prefix}.norm2.bias"] = f"{ckpt_prefix}.norm2.bias"

        mappings["merger.fc1_weight"] = "model.visual.merger.linear_fc1.weight"
        mappings["merger.fc1_bias"] = "model.visual.merger.linear_fc1.bias"
        mappings["merger.fc2_weight"] = "model.visual.merger.linear_fc2.weight"
        mappings["merger.fc2_bias"] = "model.visual.merger.linear_fc2.bias"
        mappings["merger.norm.weight"] = "model.visual.merger.norm.weight"
        mappings["merger.norm.bias"] = "model.visual.merger.norm.bias"

        for j in range(len(deepstack_visual_indexes)):
            prefix = f"deepstack_merger_list.{j}"
            ckpt_prefix = f"model.visual.deepstack_merger_list.{j}"
            mappings[f"{prefix}.fc1_weight"] = f"{ckpt_prefix}.linear_fc1.weight"
            mappings[f"{prefix}.fc1_bias"] = f"{ckpt_prefix}.linear_fc1.bias"
            mappings[f"{prefix}.fc2_weight"] = f"{ckpt_prefix}.linear_fc2.weight"
            mappings[f"{prefix}.fc2_bias"] = f"{ckpt_prefix}.linear_fc2.bias"
            mappings[f"{prefix}.norm.weight"] = f"{ckpt_prefix}.norm.weight"
            mappings[f"{prefix}.norm.bias"] = f"{ckpt_prefix}.norm.bias"

        return mappings

    def load_weights(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        *,
        cpu_mode: bool = True,
    ) -> None:
        """Load weights from a HuggingFace safetensors checkpoint."""
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_encoder_tp_group,
        )
        from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

        tp_group = get_neuron_encoder_tp_group()
        tp_rank = tp_group.rank_in_group
        tp_size = tp_group.world_size

        mappings = self.build_weight_mappings(
            self.config.depth, self.deepstack_visual_indexes
        )
        checkpoint = SafetensorsCheckpoint(checkpoint_path)

        if cpu_mode:
            sd = checkpoint.load_sharded(
                rank=tp_rank,
                world_size=tp_size,
                model=self,
                mappings=mappings,
                device=device,
            ).state_dict
        else:
            sd = checkpoint.load_sharded_pipelined(
                rank=tp_rank,
                world_size=tp_size,
                model=self,
                mappings=mappings,
                device=device,
            ).state_dict

        self.load_state_dict(sd, strict=False, assign=True)
