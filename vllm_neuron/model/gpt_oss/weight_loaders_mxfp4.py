# SPDX-License-Identifier: Apache-2.0
"""
MXFP4 Weight Loaders for running GptOss on trn3

Weight loaders that transform HuggingFace checkpoint tensors into the tiled/transposed
layout required by MXFP4 quantized NKI kernels on Neuron hardware.

Below is the list of weight loaders: TODO

Weights that need shuffle_hidden_dim (from mx_layout_transform.py):
"""

import math

import torch

from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    ShardSpec,
    get_shard,
    get_shard_deinterleaved,
    get_shuffled_shard,
    pad_to_shape,
    expert_parallel_tensor_dim_loader,
)

PMAX = 128  # NOTE: nl.tile_size.pmax does not have correct value until trace time
Q_WIDTH = 4  # (quantization block width)
Q_HEIGHT = 8  # (quantization block height)


# =============================================================================
# Non-MLP Weight Loaders
# =============================================================================


def o_proj_weight_loader(
    shard_size: int,
    expected_shard_shape: tuple[int, ...],
    num_shards: int,
) -> SafetensorsWeightLoader:
    """
    Loads o_proj weight with transpose for QKV kernel.

    Input: [H, D] (RowParallelLinear layout)
    Output: [D_shard, H_padded] (transposed for QKV kernel)
    """
    inner_loader = shuffling_weight_loader(
        shuffle_dim=0,
        expected_shard_shape=(
            expected_shard_shape[1],
            expected_shard_shape[0],
        ),  # [H, D_shard]
        shard_spec=ShardSpec(dim=1, size=shard_size, num_shards=num_shards),
    )

    def transform(slices: list, rank: int):
        return inner_loader.load(slices, rank).T

    return SafetensorsWeightLoader(transform=transform)


def shuffling_weight_loader(
    shuffle_dim: int,
    expected_shard_shape: tuple[int],
    shard_spec: ShardSpec | None = None,
    scale: float | None = None,
) -> SafetensorsWeightLoader:
    """
    Optionally shards, pads, then shuffles the provided shuffle_dim.

    Shuffle transform: [H] -> [H//4, 4] -> [4, H//4] -> [H]

    Use for: embed_tokens, lm_head, q/k/v_proj, layernorms, router.weight

    Args:
        shuffle_dim: Dimension to shuffle
        expected_shard_shape: Expected shape after sharding and padding
        shard_spec: Optional ShardSpec for sharding. If None, no sharding is performed.
        scale: Factor to divide tensor by (typically world_size for row-parallel biases)
    """
    assert shuffle_dim >= 0, (
        f"Only positive dimensions are supported but got {shuffle_dim=}"
    )
    if shard_spec is not None:
        assert shard_spec.dim >= 0, (
            f"Only positive dimensions are supported but got {shard_spec.dim=}"
        )

    def transform(slices: list, rank: int):
        assert len(slices) == 1, (
            "Expected a single tensor to be transformed by weight loader"
        )
        slice_obj = slices[0]

        # Materialize shard needed by rank as a tensor
        if shard_spec:
            if shuffle_dim == shard_spec.dim:
                # Same dimension: use get_shuffled_shard with padded size
                padded_dim_size = (
                    expected_shard_shape[shard_spec.dim] * shard_spec.num_shards
                )
                tensor = get_shuffled_shard(
                    slice_obj, shard_spec.dim, shard_spec.size, rank, padded_dim_size
                )
                if scale is not None:
                    tensor = tensor / scale
                return tensor
            else:
                tensor = get_shard(
                    slice_obj,
                    shard_spec.dim,
                    shard_spec.size,
                    shard_spec.num_shards,
                    rank,
                )
        else:
            tensor = slice_obj[:]

        if scale is not None:
            tensor = tensor / scale

        tensor = pad_to_shape(tensor, expected_shard_shape)

        tensor = _shuffle_hidden_dim(tensor, dim=shuffle_dim)

        return tensor

    return SafetensorsWeightLoader(transform=transform)


def _swizzle_h_index(H: int) -> torch.Tensor:
    """Swizzle slot -> natural H index: h512*512 + 4*p + q (the rmsnorm_mx_prefill order).

    Mirrors nkilib's swizzle_h_index. The fused rmsnorm_mx_prefill kernel matmuls the
    router weight against swizzle-transposed activations, so row h of the natural weight
    must be placed at swizzle slot h_swz, i.e. W_swz[s] = W_natural[swizzle[s]].
    """
    assert H % 512 == 0, f"H must be a multiple of 512 for swizzle, got {H}"
    num_h512 = H // 512
    h512 = torch.arange(num_h512).reshape(num_h512, 1, 1)
    q = torch.arange(Q_WIDTH).reshape(1, Q_WIDTH, 1)
    p = torch.arange(PMAX).reshape(1, 1, PMAX)
    return (h512 * 512 + Q_WIDTH * p + q).reshape(-1)


def swizzle_router_weight_loader(
    expected_shard_shape: tuple[int, ...],
) -> SafetensorsWeightLoader:
    """Load the gpt-oss router weight in the rmsnorm_mx_prefill kernel's swizzle-H order.

    For the FUSED prefill path. The checkpoint router weight is [E, H] in natural H order;
    this pads H then permutes the H axis by _swizzle_h_index so that ``.T`` yields the
    [H, E] swizzle-H weight the kernel expects. Replaces the standard
    ``shuffling_weight_loader(shuffle_dim=1)`` (model 4-way interleave) used by the
    torch-router path — the two H orders differ for H > 512.
    """

    def transform(slices: list, rank: int):
        assert len(slices) == 1, "Expected a single router-weight slice"
        tensor = slices[0][:]  # [E, H], no shard (router is replicated)
        tensor = pad_to_shape(tensor, expected_shard_shape)
        H = tensor.shape[1]
        perm = _swizzle_h_index(H).to(tensor.device)
        return tensor.index_select(1, perm).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def natural_hidden_weight_loader(
    expected_shard_shape: tuple[int, ...],
) -> SafetensorsWeightLoader:
    """Load a per-hidden weight (e.g. layernorm gamma) in NATURAL H order (no shuffle).

    For the FUSED prefill path, where the kernel consumes gamma in natural H order
    (it indexes hidden directly). Pads to shape but skips ``_shuffle_hidden_dim`` so the
    weight stays in checkpoint H order. (The standard loader applies the model 4-way
    interleave, which the fused kernel does NOT want.) There is deliberately no
    ``shuffle_dim`` parameter — this loader never shuffles.
    """

    def transform(slices: list, rank: int):
        assert len(slices) == 1, "Expected a single weight slice"
        tensor = slices[0][:]
        return pad_to_shape(tensor, expected_shard_shape)

    return SafetensorsWeightLoader(transform=transform)


def fused_qkv_shuffling_weight_loader(
    q_shard_size: int,
    kv_shard_size: int,
    expected_shard_shape: tuple[int, ...],
    num_kv_replicas: int = 1,
    attention_dp_size: int = 1,
    kv_needs_a2a: bool = False,
) -> SafetensorsWeightLoader:
    """
    Fuses separate Q, K, V tensors, shards, and shuffles.

    Input: 3 slices [Q, K, V] each with shapes:
        - Q: [q_size, H]
        - K: [kv_size, H]
        - V: [kv_size, H]
    Output: Sharded fused QKV with shuffled H dim

    Args:
        q_shard_size: Size of Q shard per rank
        kv_shard_size: Size of K/V shard per rank (after accounting for replicas)
        expected_shard_shape: Expected shape after sharding [q_shard_size + 2*kv_shard_size, H_padded]
        num_kv_replicas: KV replication factor for GQA (e.g., TP=16 with 8 KV heads -> 2 replicas)
        attention_dp_size: Attention DP degree. When > 1, KV rank is derived from
            the TP position within the interleaved effective rank.
        kv_needs_a2a: When True, KV heads are sharded across all effective ranks
            (TP * attention_dp) so each DP rank projects a different KV subset.
    """

    def transform(slices: list, rank: int):
        assert len(slices) == 3, "Expected [Q, K, V] slices"
        q_slice, k_slice, v_slice = slices

        # Input shapes:
        # Q: [q_size, H]
        # K: [kv_size, H]
        # V: [kv_size, H]

        if kv_needs_a2a:
            kv_rank = rank // num_kv_replicas
        elif attention_dp_size > 1:
            tp_rank = rank // attention_dp_size
            kv_rank = tp_rank // num_kv_replicas
        else:
            kv_rank = rank // num_kv_replicas

        # Extract Q shard: [q_shard_size, H]
        q_start = rank * q_shard_size
        q_tensor = q_slice[q_start : q_start + q_shard_size, :]

        # Extract K shard: [kv_shard_size, H]
        k_start = kv_rank * kv_shard_size
        k_tensor = k_slice[k_start : k_start + kv_shard_size, :]

        # Extract V shard: [kv_shard_size, H]
        v_start = kv_rank * kv_shard_size
        v_tensor = v_slice[v_start : v_start + kv_shard_size, :]

        # Transpose and fuse shards: [H, q_shard_size + 2*kv_shard_size]
        tensor = torch.cat([q_tensor.T, k_tensor.T, v_tensor.T], dim=1)

        # Pad: [H_padded, q_shard_size + 2*kv_shard_size]
        tensor = pad_to_shape(tensor, expected_shard_shape)

        # Shuffle H dim (dim 0): [H_padded, q_shard_size + 2*kv_shard_size]
        tensor = _shuffle_hidden_dim(tensor, dim=0)

        return tensor

    return SafetensorsWeightLoader(transform=transform)


def fused_qkv_bias_loader(
    q_shard_size: int,
    kv_shard_size: int,
    num_kv_replicas: int = 1,
    attention_dp_size: int = 1,
    kv_needs_a2a: bool = False,
) -> SafetensorsWeightLoader:
    """
    Fuses separate Q, K, V bias tensors and shards (no shuffle needed for bias).

    Input: 3 slices [Q, K, V] biases with shapes [q_size], [kv_size], [kv_size]
    Output: Sharded fused QKV bias [q_shard_size + 2*kv_shard_size]
    """

    def transform(slices: list, rank: int):
        assert len(slices) == 3, "Expected [Q, K, V] slices"
        q_slice, k_slice, v_slice = slices

        if kv_needs_a2a:
            kv_rank = rank // num_kv_replicas
        elif attention_dp_size > 1:
            tp_rank = rank // attention_dp_size
            kv_rank = tp_rank // num_kv_replicas
        else:
            kv_rank = rank // num_kv_replicas

        q_tensor = q_slice[rank * q_shard_size : (rank + 1) * q_shard_size]
        k_tensor = k_slice[kv_rank * kv_shard_size : (kv_rank + 1) * kv_shard_size]
        v_tensor = v_slice[kv_rank * kv_shard_size : (kv_rank + 1) * kv_shard_size]

        return torch.cat([q_tensor, k_tensor, v_tensor], dim=0)

    return SafetensorsWeightLoader(transform=transform)


# =============================================================================
# MLP Weight Loaders (MXFP4 Gate/Up + Down Projections)
# =============================================================================


def mxfp4_gate_up_blocks_loader(
    shard_spec: ShardSpec,
    expected_shape: tuple[int, ...],
) -> SafetensorsWeightLoader:
    """
    Full MXFP4 transform for gate_up_proj blocks.

    Input (from checkpoint):
        blocks: [E, 2I, H//32, 16[x2]] - packed FP4 weights

    Transform:
        1. Shard blocks
        2. Pad blocks
        3. Deinterleave gate/up (even/odd along 2I dim)
        4. Transpose H <-> I
        5. Pack FP4 x2 -> x4
        6. Tile for NKI kernel access patterns
        7. Permute dimensions
        8. Concatenate gate + up

    Output: [E, 128_H, 2, H//512, TP*I[x4H]]

    Args:
        tp_degree: Tensor parallel degree
        E_size: Number of experts
        H_size: Hidden dimension size
        I_size: Intermediate dimension size
        shard_spec: ShardSpec for sharding blocks
        expected_shape: Expected shape of blocks after sharding and padding
    """

    def transform(slices: list, rank: int):
        assert len(slices) == 1, "Expected [blocks] slice"

        # Original Checkpoint - [E, 2I, H/32, 16[x2]]

        # Deinterleave + Shard - [E, 2I/TP, H/32, 16[x2]]
        # - Original checkpoint is stored as [gate0, up0, gate1, up1]
        blocks_gate, blocks_up = get_shard_deinterleaved(
            slices[0], shard_spec.dim, shard_spec.size, shard_spec.num_shards, rank
        )

        # Pad and Concat - [E, Pad(2I/TP), Pad(H/32), 16[x2]]
        per_tensor_shape = [
            expected_shape[0],
            expected_shape[1] // 2,
            expected_shape[2],
            expected_shape[3],
        ]
        blocks_gate = pad_to_shape(blocks_gate, per_tensor_shape)
        blocks_up = pad_to_shape(blocks_up, per_tensor_shape)
        blocks = torch.cat([blocks_gate, blocks_up], dim=1)

        # View (remains contiguous) - [E, Pad(2I/TP), Pad(H/2[x2])]
        blocks = blocks.view(*blocks.shape[:2], -1)

        # Pack - [E, Pad(2I/TP), Pad(H/4[x4])]
        # - We need to store FP4 in groups of 4 (as opposed to groups of 2 in checkpoint) for our hardware
        blocks = _pack_fp4_x4_uint16(blocks)

        # Transpose - [E, Pad(H/4), Pad(2I/TP)[x4]]
        blocks = blocks.transpose(1, 2)

        # Tile Weight - [E, Pad(H/4), Pad(2I/TP)[x4]]
        blocks = _tile_gate_up_blocks(blocks)

        return blocks.contiguous()

    return SafetensorsWeightLoader(transform=transform)


def mxfp4_gate_up_scale_loader(
    shard_spec: ShardSpec,
    expected_shape: tuple[int, ...],
) -> SafetensorsWeightLoader:
    """
    MXFP4 transform for gate_up_proj scales.

    Input: [E, 2I, H//32]
    Output: [E, 16_H, 2, H//512, I_shard]

    Args:
        shard_spec: ShardSpec for sharding scales
        expected_shape: Expected shape after sharding and padding [E, 2I_shard, H//32]
    """

    def transform(slices: list, rank: int):
        assert len(slices) == 1, "Expected [scales] slice"

        # Original Checkpoint - [E, 2I, H/32]

        # Deinterleave + Shard
        scales_gate, scales_up = get_shard_deinterleaved(
            slices[0], shard_spec.dim, shard_spec.size, shard_spec.num_shards, rank
        )

        # Pad each separately then concat
        per_tensor_shape = (
            expected_shape[0],
            expected_shape[1] // 2,
            expected_shape[2],
        )
        scales_gate = pad_to_shape(scales_gate, per_tensor_shape)
        scales_up = pad_to_shape(scales_up, per_tensor_shape)
        scales = torch.cat([scales_gate, scales_up], dim=1)

        # Transpose - [E, H/32, 2I_shard]
        scales = scales.transpose(1, 2)

        # Tile
        scales = _tile_gate_up_scale(scales)

        return scales.contiguous()

    return SafetensorsWeightLoader(transform=transform)


def mxfp4_gate_up_bias_loader(
    shard_spec: ShardSpec,
    expected_shape: tuple[int, ...],
    hidden_act_bias: float = 0.0,
) -> SafetensorsWeightLoader:
    """
    MXFP4 transform for gate_up_proj bias.

    Input: [E, 2I]
    Output: [E, 2I_shard] (tiled/permuted)

    Args:
        shard_spec: ShardSpec for sharding bias
        expected_shape: Expected shape after sharding and padding [E, 2I_shard]
        hidden_act_bias: Bias to add to up_proj portion
    """

    def transform(slices: list, rank: int):
        assert len(slices) == 1, "Expected [bias] slice"

        # Deinterleave + Shard - [E, I_shard] each
        b_gate, b_up = get_shard_deinterleaved(
            slices[0], shard_spec.dim, shard_spec.size, shard_spec.num_shards, rank
        )

        # Track actual size before padding (to avoid adding bias to padded region)
        actual_I_shard = b_up.shape[1]

        # Pad each separately then concat
        per_tensor_shape = (expected_shape[0], expected_shape[1] // 2)
        b_gate = pad_to_shape(b_gate, per_tensor_shape)
        b_up = pad_to_shape(b_up, per_tensor_shape)

        # Add hidden_act_bias to up_proj (avoid padded region)
        if hidden_act_bias != 0.0:
            b_up[:, :actual_I_shard].add_(
                torch.tensor([hidden_act_bias], dtype=torch.float32)
            )

        bias = torch.cat([b_gate, b_up], dim=1)  # [E, 2I_shard]

        # Tile
        bias = _tile_gate_up_bias(bias)

        return bias.contiguous()

    return SafetensorsWeightLoader(transform=transform)


def mxfp4_down_weight_loader(
    shard_spec: ShardSpec,
    expected_shape: tuple[int, ...],
) -> SafetensorsWeightLoader:
    """
    Full MXFP4 transform for down_proj weights.

    Input (from checkpoint):
        blocks: [E, H, I//32, 16[x2]] - packed FP4 weights

    Output: [E, 128_I, I_shard//512, H[x4I]]

    Args:
        shard_spec: ShardSpec for sharding (along I dim, which is dim 2 in checkpoint)
        expected_shape: Expected shape after sharding and padding [E, H, I_shard//32, 16]
    """

    def transform(slices: list, rank: int):
        assert len(slices) == 1, "Expected [blocks] slice"

        # Checkpoint: [E, H, I/32, 16[x2]]
        # Shard along I dim (dim 2)
        blocks = get_shard(
            slices[0], shard_spec.dim, shard_spec.size, shard_spec.num_shards, rank
        )
        blocks = pad_to_shape(blocks, expected_shape)

        E_size, H_size, _, _ = blocks.shape

        # View: [E, H, I/4, 2[x2]]
        blocks = blocks.view(E_size, H_size, -1, 2)

        # Transpose: [E, I/4, H, 2[x2]]
        blocks = blocks.transpose(1, 2).contiguous()

        # Pack: [E, I/4, H[x4]]
        blocks = _pack_fp4_x4_uint16(blocks)
        blocks = blocks.view(E_size, -1, H_size)

        # Tile
        blocks = _tile_down_blocks(blocks)

        return blocks.contiguous()

    return SafetensorsWeightLoader(transform=transform)


def mxfp4_down_scale_loader(
    shard_spec: ShardSpec,
    expected_shape: tuple[int, ...],
) -> SafetensorsWeightLoader:
    """
    MXFP4 transform for down_proj scales.

    Input: [E, H, I//32]
    Output: [E, 16_I, I_shard//512, H]

    Args:
        shard_spec: ShardSpec for sharding (along I dim, which is dim 2 in checkpoint)
        expected_shape: Expected shape after sharding and padding [E, H, I_shard//32]
    """

    def transform(slices: list, rank: int):
        assert len(slices) == 1, "Expected [scales] slice"

        # Checkpoint: [E, H, I/32]
        # Shard along I dim (dim 2)
        scales = get_shard(
            slices[0], shard_spec.dim, shard_spec.size, shard_spec.num_shards, rank
        )
        scales = pad_to_shape(scales, expected_shape)

        # Transpose: [E, I/32, H]
        scales = scales.transpose(1, 2)

        # Tile
        scales = _tile_down_scale(scales)

        return scales.contiguous()

    return SafetensorsWeightLoader(transform=transform)


def mxfp4_down_bias_loader(
    expected_shape: tuple[int, ...],
    scale: float | None = None,
) -> SafetensorsWeightLoader:
    """
    Transform for down_proj bias: pad, shuffle H dim.

    Input: [E, H]
    Output: [E, H_padded] (shuffled)

    Args:
        expected_shape: Expected shape after padding [E, H_padded]
    """

    def transform(slices: list, rank: int):
        assert len(slices) == 1, "Expected [bias] slice"

        # Checkpoint: [E, H]
        bias = slices[0][:]

        # Scale (for all-reduce averaging)
        if scale is not None:
            bias = bias / scale

        # Pad
        bias = pad_to_shape(bias, expected_shape)

        # Shuffle H dim
        bias = _shuffle_hidden_dim(bias, dim=-1)

        return bias.contiguous()

    return SafetensorsWeightLoader(transform=transform)


# =============================================================================
# Helper Functions
# =============================================================================


def unshuffle_hidden_dim_last(tensor: torch.Tensor) -> torch.Tensor:
    """Inverse of the model 4-way hidden shuffle on the LAST dim: [H] -> [4, H/4] -> [H/4, 4] -> [H].

    Recovers natural H order from the model's interleaved layout (the inverse of
    ``_shuffle_hidden_dim``). Used at the MoE entry on activations for the fused
    rmsnorm_mx_prefill path, which expects plain natural-H hidden states.
    """
    *pre, H = tensor.shape
    assert H % 4 == 0, f"H must be divisible by 4 to unshuffle, got H={H}"
    return tensor.reshape(*pre, 4, H // 4).transpose(-2, -1).reshape(*pre, H)


def _shuffle_hidden_dim(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Shuffle hidden dimension: [H] -> [H/4, 4] -> [4, H/4] -> [H]

    Example Steps:
    1. [H]      - [1, 2, 3, 4, 5, 6, 7]
    2. [H/4, 4] - [[1, 2], [3, 4], [5, 6], [7, 8]]
    3. [4, H/4] - [1, 3, 5, 7], [2, 4, 6, 8]]
    4. [H]      - [1, 3, 5, 7, 2, 4, 6, 8]

    Args:
        tensor: Input tensor
        dim: Dimension to shuffle (supports negative indexing)

    Returns:
        Tensor with shuffled dimension
    """
    if dim < 0:
        dim = dim + tensor.ndim
    pre_shape, H = tensor.shape[:dim], tensor.shape[dim]
    if dim < tensor.ndim - 1:
        post_shape = tensor.shape[dim + 1 :]
    else:
        post_shape = []
    assert H % 4 == 0, (
        f"H dim (dim={dim=}) in shape {tensor.shape=} must be divisible by 4 to shuffle H dim, got H={H=}"
    )
    tensor = (
        tensor.view(*pre_shape, H // 4, 4, *post_shape)
        .transpose(dim, dim + 1)
        .reshape(*pre_shape, H, *post_shape)
    )
    return tensor


def _pack_fp4_x4_uint16(X: torch.Tensor):
    """
    Assumes:
    - X.dtype == uint8
    - uint8 represents fp4_x2
    - dim to pack is final dim

    Example with binary values used to show underlying data in memory
    > Input - torch.tensor([00000001, 00100011, 01000101, 01100111], dtype=torch.uint8)
    > Output - torch.tensor([0000000100100011, 0100010101100111], dtype=torch.uint16)

    Note: this layout is pre-swizzled if we unpack and place all four packed values along another dim.
    - Example: if we have [128, 16] and pack this to [128, 8], we can unpack this to [128, 8, 4] and transpose / reshape to [512, 8] which is swizzled.
    """

    assert X.dtype == torch.uint8, f"expected uint8, got {X.dtype}"

    # [..., D] -> [..., D / 2]
    repacked = X.view(torch.uint16)

    return repacked


def _tile_gate_up_blocks(w_gate_up):
    """
    Docstring for _tile_gate_up_blocks

    2I = 6144
    2I / TP8 = 768

    H = 3072
    H / 4 = 768

    Input - [E, Pad(H/4), Pad(2I/TP)[x4]]
    """

    E_size, H_div_4, _2I_shard = w_gate_up.shape
    num_H_tiles, q_blocks_per_H_tile = _get_h_tiling_shard_i(H_size=H_div_4 * 4)
    num_I_tiles, q_blocks_per_I_tile = _get_i_tiling_shard_i(
        I_shard_size=_2I_shard // 2
    )
    q_packed_H = Q_WIDTH // 4  # We pack 4 fp4 values in uint16
    _128_H = q_blocks_per_H_tile * Q_HEIGHT

    w_gate_up = w_gate_up.reshape(
        [
            E_size,
            num_H_tiles,
            q_blocks_per_H_tile,
            Q_HEIGHT,
            2,
            num_I_tiles,
            q_blocks_per_I_tile,
            Q_HEIGHT,
            Q_WIDTH,
            q_packed_H,
        ]
    )

    # Permute weight
    (
        E,
        N_H_TILES,
        Q_BLOCK_H_TILE,
        Q_HEIGHT_H,
        gate_up_concat,
        NUM_I_TILES,
        Q_BLOCK_I_TILE,
        Q_HEIGHT_I,
        Q_WIDTH_I,
        Q_PACKED_H,
    ) = list(range(w_gate_up.ndim))
    w_gate_up = w_gate_up.permute(
        E,
        Q_BLOCK_H_TILE,
        Q_HEIGHT_H,
        gate_up_concat,
        N_H_TILES,
        NUM_I_TILES,
        Q_WIDTH_I,
        Q_BLOCK_I_TILE,
        Q_HEIGHT_I,
        Q_PACKED_H,
    )

    w_gate_up = w_gate_up.reshape(
        E_size, _128_H, 2, num_H_tiles, -1
    )  # [E, 128_H, 2, H // 512, TP * I[x4H]]

    return w_gate_up


def _tile_gate_up_scale(s_gate_up):
    """
    Tile gate_up scales for MXFP4 kernel.

    Input: [E, H/32, 2*I_shard] where 2*I_shard = [gate_shard, up_shard] concatenated
    Output: [E, 16_H, 2, H//512, I_shard]
    """
    E_size, H_div_32, _2I_shard = s_gate_up.shape

    num_H_tiles, q_blocks_per_H_tile = _get_h_tiling_shard_i(H_size=H_div_32 * 32)
    num_I_tiles, q_blocks_per_I_tile = _get_i_tiling_shard_i(
        I_shard_size=_2I_shard // 2
    )

    s_gate_up = s_gate_up.reshape(
        [
            E_size,
            num_H_tiles,
            q_blocks_per_H_tile,
            2,  # gate/up
            num_I_tiles,
            q_blocks_per_I_tile,
            Q_HEIGHT,
            Q_WIDTH,
        ]
    )

    (
        E,
        N_H_TILES,
        Q_BLOCK_H_TILE,
        gate_up_concat,
        NUM_I_TILES,
        Q_BLOCK_I_TILE,
        Q_HEIGHT_I,
        Q_WIDTH_I,
    ) = range(s_gate_up.ndim)

    s_gate_up = s_gate_up.permute(
        E,
        Q_BLOCK_H_TILE,
        gate_up_concat,
        N_H_TILES,
        NUM_I_TILES,
        Q_WIDTH_I,
        Q_BLOCK_I_TILE,
        Q_HEIGHT_I,
    )

    scale = s_gate_up.reshape(
        E_size, q_blocks_per_H_tile, 2, num_H_tiles, -1
    )  # [E, 16_H, 2, H // 512, I_shard]
    return scale


def _tile_gate_up_bias(b_gate_up):
    """
    Tile gate_up bias for MXFP4 kernel.

    Input: [E, 2*I_shard] where first half is gate, second half is up
    Output: [E, 2*I_shard] (tiled/permuted)

    Reference permutation (from mx_layout_transform.py):
    E, shard, Q_BLOCK_I_TILE, Q_HEIGHT_I, gate_up_concat, NUM_I_TILES, Q_WIDTH_I

    Since we already sharded, we don't have shard dim. The permutation becomes:
    E, Q_BLOCK_I_TILE, Q_HEIGHT_I, gate_up_concat, NUM_I_TILES, Q_WIDTH_I
    """
    E_size, _2I_shard = b_gate_up.shape
    I_shard = _2I_shard // 2

    num_I_tiles, q_blocks_per_I_tile = _get_i_tiling_shard_i(I_shard_size=I_shard)

    b_gate_up = b_gate_up.reshape(
        [
            E_size,
            2,  # gate_up_concat
            num_I_tiles,
            q_blocks_per_I_tile,
            Q_HEIGHT,
            Q_WIDTH,
        ]
    )

    # Indices: E=0, gate_up_concat=1, NUM_I_TILES=2, Q_BLOCK_I_TILE=3, Q_HEIGHT_I=4, Q_WIDTH_I=5
    # Permute to match reference (minus shard dim): E, Q_BLOCK_I_TILE, Q_HEIGHT_I, gate_up_concat, NUM_I_TILES, Q_WIDTH_I
    b_gate_up = b_gate_up.permute(0, 3, 4, 1, 2, 5)

    bias = b_gate_up.reshape(E_size, -1)  # [E, 2I_shard]
    return bias


def _tile_down_blocks(w_down):
    """
    Tile down_proj blocks for MXFP4 kernel (with H dim shuffling).

    Input: [E, I_shard/4, H[x4]]
    Output: [E, 128_I, I_shard//512, H[x4I]]
    """
    E_size, I_div_4, H_size = w_down.shape
    I_shard = I_div_4 * 4

    num_I_tiles, q_blocks_per_I_tile = _get_i_tiling_shard_i(I_shard_size=I_shard)
    q_packed = Q_WIDTH // 4

    w_down = w_down.reshape(
        [
            E_size,
            num_I_tiles,
            q_blocks_per_I_tile,
            Q_HEIGHT,
            H_size // 4,
            4,
            q_packed,
        ]
    )
    # E=0, NUM_I_TILES=1, Q_BLOCK_I_TILE=2, Q_HEIGHT_I=3, NUM_H_BLOCKS=4, H_BLOCK=5, Q_PACKED_I=6
    # Permute to: E, Q_BLOCK_I_TILE, Q_HEIGHT_I, NUM_I_TILES, H_BLOCK, NUM_H_BLOCKS, Q_PACKED_I
    w_down = w_down.permute(0, 2, 3, 1, 5, 4, 6)
    w_down = w_down.reshape(
        E_size, q_blocks_per_I_tile * Q_HEIGHT, num_I_tiles, H_size * q_packed
    )

    return w_down


def _tile_down_scale(s_down):
    """
    Tile down_proj scales for MXFP4 kernel (with H dim shuffling).

    Input: [E, I_shard/32, H]
    Output: [E, 16_I, I_shard//512, H]
    """
    E_size, I_div_32, H_size = s_down.shape
    I_shard = I_div_32 * 32

    num_I_tiles, q_blocks_per_I_tile = _get_i_tiling_shard_i(I_shard_size=I_shard)

    s_down = s_down.reshape(
        [
            E_size,
            num_I_tiles,
            q_blocks_per_I_tile,
            H_size // 4,
            4,
        ]
    )
    # E=0, NUM_I_TILES=1, Q_BLOCK_I_TILE=2, NUM_H_BLOCKS=3, H_BLOCK=4
    # Permute to: E, Q_BLOCK_I_TILE, NUM_I_TILES, H_BLOCK, NUM_H_BLOCKS
    s_down = s_down.permute(0, 2, 1, 4, 3)
    s_down = s_down.reshape(E_size, q_blocks_per_I_tile, num_I_tiles, H_size)

    return s_down


def _get_h_tiling_shard_i(H_size: int):
    """
    Calculate the number of H tiles and blocks per H tile for hidden dimension tiling.

    Computes tiling parameters for the H (hidden) dimension based on hardware constraints.
    Uses fixed tile size of 512 elements and quantization block dimensions.

    Args:
        H_size (int): Total size of the H (hidden) dimension

    Returns:
        tuple: A tuple containing:
            - num_H_tiles (int): Number of tiles in the H dimension (H_size // (pmax * q_width))
            - q_blocks_per_H_tile (int): Number of quantized blocks per H tile (512 // (q_width * q_height))
    """
    num_H_tiles = H_size // (PMAX * Q_WIDTH)
    q_blocks_per_H_tile = 512 // (Q_WIDTH * Q_HEIGHT)
    return num_H_tiles, q_blocks_per_H_tile


def _get_i_tiling_shard_i(I_shard_size: int):
    """
    Calculate the number of I tiles and blocks per I tile for tensor parallel sharding.

    This function determines the tiling parameters for the I (input) dimension when sharding
    across tensor parallel ranks. It handles two cases:
    1. When per-rank I size is > 512: Tiles of size 512 are used
    2. When per-rank I size is <= 512: A single tile is used

    Args:
        tp_degree (int): Tensor parallel degree - number of ranks to shard across
        I_size (int): Total size of the I dimension before sharding

    Returns:
        tuple: A tuple containing:
            - num_I_tiles (int): Number of tiles in the I dimension
            - q_blocks_per_I_tile (int): Number of quantized blocks per I tile

    Raises:
        AssertionError: If per-rank I size is not divisible by required block size
                       or if padding would be needed (not implemented)

    Examples:
        >>> get_i_tiling_shard_i(tp_degree=4, I_size=1536)  # per_rank_I_size=384
        (1, 12)  # Single tile with 12 blocks
        >>> get_i_tiling_shard_i(tp_degree=2, I_size=1024)  # per_rank_I_size=512
        (1, 16)  # Single tile with 16 blocks
    """
    mx_block_size = Q_WIDTH * Q_HEIGHT
    if I_shard_size > 512:
        assert I_shard_size % 512 == 0, (
            f"Unsupported I_shard_size {I_shard_size} must be divisible by 512"
        )
        num_I_tiles = int(math.ceil(I_shard_size / 512.0))
        q_blocks_per_I_tile, q_blocks_per_I_tile_padding = divmod(
            512, Q_WIDTH * Q_HEIGHT
        )
        assert q_blocks_per_I_tile_padding == 0, (
            "Padding is required for q_blocks_per_I_tile dim, not implemented yet"
        )
    else:
        assert I_shard_size % mx_block_size == 0, (
            f"Unsupported I_shard_size {I_shard_size}, must be divisible by {mx_block_size}"
        )
        num_I_tiles = 1
        q_blocks_per_I_tile, q_blocks_per_I_tile_padding = divmod(
            I_shard_size, Q_WIDTH * Q_HEIGHT
        )
        assert q_blocks_per_I_tile_padding == 0, (
            "Padding is required for q_blocks_per_I_tile dim, not implemented yet"
        )
    return num_I_tiles, q_blocks_per_I_tile


# =============================================================================
# Expert Parallel Weight Loaders (MxFP4)
# =============================================================================


def expert_parallel_mxfp4_gate_up_blocks_loader(
    local_expert_indices: list,
    shard_spec: ShardSpec,
    expected_shape: tuple[int, ...],
) -> SafetensorsWeightLoader:
    """Expert parallel version of mxfp4_gate_up_blocks_loader.

    Args:
        local_expert_indices: List of expert indices assigned to this rank
        shard_spec: ShardSpec for tensor parallel sharding
        expected_shape: Expected shape after sharding and padding (for all experts)

    Returns:
        SafetensorsWeightLoader for expert parallel gate_up blocks
    """
    # Create the original loader
    original_loader = mxfp4_gate_up_blocks_loader(
        shard_spec=shard_spec,
        expected_shape=expected_shape,
    )

    # Wrap it with expert parallel logic
    return expert_parallel_tensor_dim_loader(local_expert_indices, original_loader)


def expert_parallel_mxfp4_gate_up_scale_loader(
    local_expert_indices: list,
    shard_spec: ShardSpec,
    expected_shape: tuple[int, ...],
) -> SafetensorsWeightLoader:
    """Expert parallel version of mxfp4_gate_up_scale_loader.

    Args:
        local_expert_indices: List of expert indices assigned to this rank
        shard_spec: ShardSpec for tensor parallel sharding
        expected_shape: Expected shape after sharding and padding (for all experts)

    Returns:
        SafetensorsWeightLoader for expert parallel gate_up scales
    """
    # Create the original loader
    original_loader = mxfp4_gate_up_scale_loader(
        shard_spec=shard_spec,
        expected_shape=expected_shape,
    )

    # Wrap it with expert parallel logic
    return expert_parallel_tensor_dim_loader(local_expert_indices, original_loader)


def expert_parallel_mxfp4_gate_up_bias_loader(
    local_expert_indices: list,
    shard_spec: ShardSpec,
    expected_shape: tuple[int, ...],
    hidden_act_bias: float = 0.0,
) -> SafetensorsWeightLoader:
    """Expert parallel version of mxfp4_gate_up_bias_loader.

    Args:
        local_expert_indices: List of expert indices assigned to this rank
        shard_spec: ShardSpec for tensor parallel sharding
        expected_shape: Expected shape after sharding and padding (for all experts)
        hidden_act_bias: Bias to add to up_proj portion

    Returns:
        SafetensorsWeightLoader for expert parallel gate_up biases
    """
    # Create the original loader
    original_loader = mxfp4_gate_up_bias_loader(
        shard_spec=shard_spec,
        expected_shape=expected_shape,
        hidden_act_bias=hidden_act_bias,
    )

    # Wrap it with expert parallel logic
    return expert_parallel_tensor_dim_loader(local_expert_indices, original_loader)


def expert_parallel_mxfp4_down_weight_loader(
    local_expert_indices: list,
    shard_spec: ShardSpec,
    expected_shape: tuple[int, ...],
) -> SafetensorsWeightLoader:
    """Expert parallel version of mxfp4_down_weight_loader.

    Args:
        local_expert_indices: List of expert indices assigned to this rank
        shard_spec: ShardSpec for tensor parallel sharding
        expected_shape: Expected shape after sharding and padding (for all experts)

    Returns:
        SafetensorsWeightLoader for expert parallel down weights
    """
    # Create the original loader
    original_loader = mxfp4_down_weight_loader(
        shard_spec=shard_spec,
        expected_shape=expected_shape,
    )

    # Wrap it with expert parallel logic
    return expert_parallel_tensor_dim_loader(local_expert_indices, original_loader)


def expert_parallel_mxfp4_down_scale_loader(
    local_expert_indices: list,
    shard_spec: ShardSpec,
    expected_shape: tuple[int, ...],
) -> SafetensorsWeightLoader:
    """Expert parallel version of mxfp4_down_scale_loader.

    Args:
        local_expert_indices: List of expert indices assigned to this rank
        shard_spec: ShardSpec for tensor parallel sharding
        expected_shape: Expected shape after sharding and padding (for all experts)

    Returns:
        SafetensorsWeightLoader for expert parallel down scales
    """
    # Create the original loader
    original_loader = mxfp4_down_scale_loader(
        shard_spec=shard_spec,
        expected_shape=expected_shape,
    )

    # Wrap it with expert parallel logic
    return expert_parallel_tensor_dim_loader(local_expert_indices, original_loader)


def expert_parallel_mxfp4_down_bias_loader(
    local_expert_indices: list,
    expected_shape: tuple[int, ...],
    scale: float | None = None,
) -> SafetensorsWeightLoader:
    """Expert parallel version of mxfp4_down_bias_loader.

    Args:
        local_expert_indices: List of expert indices assigned to this rank
        expected_shape: Expected shape after padding (for all experts)
        scale: Scale factor for all-reduce averaging

    Returns:
        SafetensorsWeightLoader for expert parallel down biases
    """
    # Create the original loader
    original_loader = mxfp4_down_bias_loader(
        expected_shape=expected_shape,
        scale=scale,
    )

    # Wrap it with expert parallel logic
    return expert_parallel_tensor_dim_loader(local_expert_indices, original_loader)
