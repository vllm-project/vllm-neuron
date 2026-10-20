# SPDX-License-Identifier: Apache-2.0
"""STATIC_MX Weight Loaders for Llama3.

Clean named-function API for the Llama3 STATIC_MX weight loaders.

Each public function returns a :class:`SafetensorsWeightLoader` that
internally uses the canonical ``fused_qkv_weight_loader`` or
``sharding_weight_loader`` as the base sharder, then applies the
appropriate MX pack transform from :mod:`.weight_pack_mx_fp8`.

No 240/448 downscale - STATIC_MX is OCP-native (max=448).
"""

import math

import torch


def _shuffle_hidden_dim(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Shuffle hidden dimension: [H] -> [H/4, 4] -> transpose -> [H]."""
    if dim < 0:
        dim = dim + tensor.ndim
    pre_shape, H = tensor.shape[:dim], tensor.shape[dim]
    post_shape = tensor.shape[dim + 1 :] if dim < tensor.ndim - 1 else []
    assert H % 4 == 0
    return (
        tensor.view(*pre_shape, H // 4, 4, *post_shape)
        .transpose(dim, dim + 1)
        .reshape(*pre_shape, H, *post_shape)
    )


from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    fused_qkv_weight_loader,
    pad_to_shape,
    sharding_weight_loader,
    with_rank_override,
)

from .weight_pack_mx_fp8 import (
    mx_shuffle_o_proj,
    qkv_weight_pack_mx_interleaved,
)

_PMAX = 128
_Q_WIDTH = 4
_TILE_SIZE = _PMAX * _Q_WIDTH  # 512
_FP8_DTYPE = torch.float8_e4m3fn


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_scalar_from_slice(slice_obj) -> torch.Tensor:
    """Read a scalar (or 1-element) tensor from a safetensors slice as fp32."""
    shape = slice_obj.get_shape()
    raw = slice_obj[()] if len(shape) == 0 else slice_obj[:]
    flat = raw.to(torch.float32).reshape(-1)
    assert flat.numel() == 1
    return flat


def _canonical_qkv_loader(
    *,
    q_size: int,
    kv_size: int,
    num_shards: int,
    num_kv_replicas: int,
    attention_dp_size: int,
    attention_dp_rank: int,
    kv_sharded_across_attention_dp: bool,
) -> SafetensorsWeightLoader:
    """Canonical QKV sharder - shared by all MX QKV loaders."""
    return fused_qkv_weight_loader(
        q_size=q_size,
        kv_size=kv_size,
        shard_dim=1,
        num_shards=num_shards,
        is_storage_transposed=True,
        num_kv_replicas=num_kv_replicas,
        attention_dp_size=attention_dp_size,
        attention_dp_rank=attention_dp_rank,
        kv_sharded_across_attention_dp=kv_sharded_across_attention_dp,
    )


# ===========================================================================
# QKV loaders
# ===========================================================================


def fused_qkv_weight_loader_mxfp8_cte(
    q_size: int,
    kv_size: int,
    num_shards: int,
    num_kv_replicas: int = 1,
    *,
    attention_dp_size: int = 1,
    attention_dp_rank: int = 0,
    kv_sharded_across_attention_dp: bool = False,
) -> SafetensorsWeightLoader:
    """QKV weight loader for CTE prefill -> ``[H//4, I, 4]`` fp8 (MX_INTERLEAVED)."""
    base = _canonical_qkv_loader(
        q_size=q_size,
        kv_size=kv_size,
        num_shards=num_shards,
        num_kv_replicas=num_kv_replicas,
        attention_dp_size=attention_dp_size,
        attention_dp_rank=attention_dp_rank,
        kv_sharded_across_attention_dp=kv_sharded_across_attention_dp,
    )
    base_transform = base.transform
    return SafetensorsWeightLoader(
        transform=lambda slices, rank: qkv_weight_pack_mx_interleaved(
            base_transform(slices, rank)
        )
    )


def fused_qkv_weight_loader_mxfp8_tkg(
    q_size: int,
    kv_size: int,
    num_shards: int,
    num_kv_replicas: int = 1,
    *,
    attention_dp_size: int = 1,
    attention_dp_rank: int = 0,
    kv_sharded_across_attention_dp: bool = False,
) -> SafetensorsWeightLoader:
    """QKV weight loader for TKG decode -> ``[H//4, I, 4]`` fp8."""
    base = _canonical_qkv_loader(
        q_size=q_size,
        kv_size=kv_size,
        num_shards=num_shards,
        num_kv_replicas=num_kv_replicas,
        attention_dp_size=attention_dp_size,
        attention_dp_rank=attention_dp_rank,
        kv_sharded_across_attention_dp=kv_sharded_across_attention_dp,
    )
    base_transform = base.transform

    def transform(slices, rank):
        w = base_transform(slices, rank)  # [H, I] fp8
        H, I = w.shape
        return w.reshape(H // _Q_WIDTH, _Q_WIDTH, I).permute(0, 2, 1).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def fused_qkv_weight_scale_loader_mxfp8() -> SafetensorsWeightLoader:
    """Load Q/K/V per-tensor weight scales as ``[1, 3]`` fp32."""

    def transform(slices, rank):
        assert len(slices) == 3
        scalars = [_read_scalar_from_slice(s) for s in slices]
        return torch.stack(scalars, dim=0).reshape(1, 3).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def fused_qkv_input_scale_loader_mxfp8() -> SafetensorsWeightLoader:
    """Load Q/K/V input scale as ``[1, 1]`` fp32 (asserts all three equal)."""

    def transform(slices, rank):
        assert len(slices) == 3
        scalars = [_read_scalar_from_slice(s) for s in slices]
        for idx, sc in enumerate(scalars[1:], start=1):
            assert torch.allclose(scalars[0], sc, atol=1e-6), (
                f"QKV input scales must be identical; "
                f"got Q={scalars[0].item()} vs idx{idx}={sc.item()}"
            )
        return scalars[0].reshape(1, 1).contiguous()

    return SafetensorsWeightLoader(transform=transform)


# ===========================================================================
# O-proj loaders
# ===========================================================================


def o_proj_weight_loader_mxfp8_cte(
    shard_size: int,
    num_shards: int,
    rank_override: int | None = None,
) -> SafetensorsWeightLoader:
    """O-proj weight for CTE prefill (MX byte-shuffle, fp8)."""
    base = sharding_weight_loader(
        shard_dim=0,
        shard_size=shard_size,
        num_shards=num_shards,
        is_storage_transposed=True,
    )
    if rank_override is not None:
        base = with_rank_override(base, rank=rank_override)
    base_transform = base.transform

    def transform(slices, rank):
        return mx_shuffle_o_proj(base_transform(slices, rank))

    return SafetensorsWeightLoader(transform=transform)


def o_proj_weight_loader_mxfp8_tkg(
    shard_size: int,
    num_shards: int,
    rank_override: int | None = None,
) -> SafetensorsWeightLoader:
    """O-proj weight for TKG decode -> ``[ND//4, H, 4]`` fp8."""
    base = sharding_weight_loader(
        shard_dim=0,
        shard_size=shard_size,
        num_shards=num_shards,
        is_storage_transposed=True,
    )
    if rank_override is not None:
        base = with_rank_override(base, rank=rank_override)
    base_transform = base.transform

    def transform(slices, rank):
        w = base_transform(slices, rank)  # [ND, H] fp8
        nd, h = w.shape
        return w.reshape(nd // _Q_WIDTH, _Q_WIDTH, h).permute(0, 2, 1).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def o_proj_weight_scale_loader_mxfp8() -> SafetensorsWeightLoader:
    """Load o_proj weight scale, broadcast to ``[128, 1]`` fp32."""

    def transform(slices, rank):
        assert len(slices) == 1
        return _read_scalar_from_slice(slices[0]).expand(_PMAX, 1).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def o_proj_input_scale_loader_mxfp8() -> SafetensorsWeightLoader:
    """Load o_proj input scale, broadcast to ``[128, 1]`` fp32."""

    def transform(slices, rank):
        assert len(slices) == 1
        return _read_scalar_from_slice(slices[0]).expand(_PMAX, 1).contiguous()

    return SafetensorsWeightLoader(transform=transform)


# ===========================================================================
# MLP loaders
# ===========================================================================


def mlp_gate_up_weight_loader_mxfp8_cte(
    intermediate_size_per_rank: int,
    hidden_size: int,
    tp_size: int,
    tp_rank: int,
) -> SafetensorsWeightLoader:
    """MLP gate/up for CTE prefill (H_X4_INNERMOST swizzle).

    INNERMOST packs the 4 H-quad elements as the innermost (adjacent) H values,
    matching the layout a DMA transpose of pre-quantized fp8 hidden produces, so
    the kernel takes the ``mlpp_has_dma_xpose`` fast path (DMA transpose on load,
    no PE ``nc_transpose``). Final shape ``[128_H, H/512, I/512, 4_I, 128_I, 4_H]``.
    """
    n_h512_tile = hidden_size // _TILE_SIZE
    base = with_rank_override(
        sharding_weight_loader(
            shard_dim=1,
            shard_size=intermediate_size_per_rank,
            num_shards=tp_size,
            is_storage_transposed=True,
        ),
        rank=tp_rank,
    )
    base_transform = base.transform

    def transform(slices, rank):
        tensor = base_transform(slices, rank)
        n_i512_tile = math.ceil(intermediate_size_per_rank / _TILE_SIZE)
        padded_i = n_i512_tile * _TILE_SIZE
        if tensor.shape[1] < padded_i:
            tensor = pad_to_shape(tensor, (hidden_size, padded_i))
        # H_X4_INNERMOST: [H, I] viewed as [H/512, 128_H, 4_H, I/512, 128_I, 4_I]
        # -> permute(1, 0, 3, 5, 4, 2) -> [128_H, H/512, I/512, 4_I, 128_I, 4_H]
        tensor = tensor.reshape(
            n_h512_tile, _PMAX, _Q_WIDTH, n_i512_tile, _PMAX, _Q_WIDTH
        )
        return tensor.permute(1, 0, 3, 5, 4, 2).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def mlp_gate_up_weight_loader_mxfp8_tkg(
    intermediate_size_per_rank: int,
    hidden_size: int,
    tp_size: int,
    tp_rank: int,
) -> SafetensorsWeightLoader:
    """MLP gate/up for TKG decode -> ``[128_H, H/512, I/512, 4_I, 128_I, 4_H]`` fp8."""
    n_h512_tile = hidden_size // _TILE_SIZE
    n_i512_tile = math.ceil(intermediate_size_per_rank / _TILE_SIZE)
    base = with_rank_override(
        sharding_weight_loader(
            shard_dim=1,
            shard_size=intermediate_size_per_rank,
            num_shards=tp_size,
            is_storage_transposed=True,
        ),
        rank=tp_rank,
    )
    base_transform = base.transform

    def transform(slices, rank):
        tensor = base_transform(slices, rank)  # [H, I_shard] fp8
        # Pad I to multiple of 512 if needed
        padded_i = n_i512_tile * _TILE_SIZE
        if tensor.shape[1] < padded_i:
            tensor = torch.nn.functional.pad(
                tensor.view(torch.uint8),
                (0, padded_i - tensor.shape[1]),
            ).view(_FP8_DTYPE)
        # H_X4_MIDDLE 6D layout:
        # [H, I] -> [H/512, 4_H, 128_H, I/512, 128_I, 4_I]
        # -> permute(2, 0, 3, 5, 4, 1) -> [128_H, H/512, I/512, 4_I, 128_I, 4_H]
        tensor = tensor.reshape(
            n_h512_tile, _Q_WIDTH, _PMAX, n_i512_tile, _PMAX, _Q_WIDTH
        )
        return tensor.permute(2, 0, 3, 5, 4, 1).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def mlp_down_weight_loader_mxfp8_cte(
    intermediate_size_per_rank: int,
    hidden_size: int,
    tp_size: int,
    tp_rank: int,
) -> SafetensorsWeightLoader:
    """MLP down for CTE prefill (down swizzle)."""
    base = with_rank_override(
        sharding_weight_loader(
            shard_dim=0,
            shard_size=intermediate_size_per_rank,
            num_shards=tp_size,
            is_storage_transposed=True,
        ),
        rank=tp_rank,
    )
    base_transform = base.transform

    def transform(slices, rank):
        tensor = base_transform(slices, rank)
        n_i512_tile = math.ceil(intermediate_size_per_rank / _TILE_SIZE)
        padded_i = n_i512_tile * _TILE_SIZE
        if tensor.shape[0] < padded_i:
            tensor = pad_to_shape(tensor, (padded_i, hidden_size))
        tensor = tensor.reshape(n_i512_tile, _PMAX, _Q_WIDTH, hidden_size)
        return tensor.permute(1, 0, 3, 2).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def mlp_down_weight_loader_mxfp8_tkg(
    intermediate_size_per_rank: int,
    hidden_size: int,
    tp_size: int,
    tp_rank: int,
) -> SafetensorsWeightLoader:
    """MLP down for TKG decode -> ``[128, I/512, H]`` uint32."""
    base = with_rank_override(
        sharding_weight_loader(
            shard_dim=0,
            shard_size=intermediate_size_per_rank,
            num_shards=tp_size,
            is_storage_transposed=True,
        ),
        rank=tp_rank,
    )
    base_transform = base.transform

    def transform(slices, rank):
        tensor = base_transform(slices, rank)  # [I_shard, H] fp8
        n_i512_tile = math.ceil(intermediate_size_per_rank / _TILE_SIZE)
        padded_i = n_i512_tile * _TILE_SIZE
        if tensor.shape[0] < padded_i:
            tensor = pad_to_shape(tensor, (padded_i, hidden_size))
        # Kernel expects [128_I, I/512, H, 4_I] fp8
        tensor = tensor.reshape(n_i512_tile, _PMAX, _Q_WIDTH, hidden_size)
        return tensor.permute(1, 0, 3, 2).contiguous()  # [PMAX, n_i512, H, Q_WIDTH]

    return SafetensorsWeightLoader(transform=transform)


def mlp_weight_scale_loader_mxfp8() -> SafetensorsWeightLoader:
    """Scalar fp32 -> ``[128, 1]`` fp32."""

    def transform(slices, rank):
        assert len(slices) == 1
        return _read_scalar_from_slice(slices[0]).expand(_PMAX, 1).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def mlp_input_scale_loader_mxfp8() -> SafetensorsWeightLoader:
    """Scalar fp32 -> ``[128, 1]`` fp32."""

    def transform(slices, rank):
        assert len(slices) == 1
        return _read_scalar_from_slice(slices[0]).expand(_PMAX, 1).contiguous()

    return SafetensorsWeightLoader(transform=transform)


# ===========================================================================
# Public attach entry points (called by model_mx_fp8.py)
# ===========================================================================

from vllm_neuron.utils.weight_loader import set_weight_loader


def attach_attention_loaders(
    module,
    *,
    q_size: int,
    kv_size: int,
    world_size: int,
    num_kv_replicas: int,
    attention_dp_size: int = 1,
    attention_dp_rank: int = 0,
    kv_needs_a2a: bool = False,
    num_attention_heads: int,
    head_dim: int,
) -> None:
    """Attach MX attention weight loaders to module params."""
    ddp = attention_dp_size
    effective_q_rank = attention_dp_rank + module.tp_group.rank_in_group * ddp
    o_shard_size = (num_attention_heads * head_dim) // (world_size * ddp)

    set_weight_loader(
        module.qkv_proj_weight,
        fused_qkv_weight_loader_mxfp8_cte(
            q_size=q_size,
            kv_size=kv_size,
            num_shards=world_size,
            num_kv_replicas=num_kv_replicas,
            attention_dp_size=ddp,
            attention_dp_rank=attention_dp_rank,
            kv_sharded_across_attention_dp=kv_needs_a2a,
        ),
    )
    set_weight_loader(module.qkv_weight_scale, fused_qkv_weight_scale_loader_mxfp8())
    set_weight_loader(module.qkv_input_scale, fused_qkv_input_scale_loader_mxfp8())

    set_weight_loader(
        module.o_proj_weight,
        o_proj_weight_loader_mxfp8_cte(
            shard_size=o_shard_size,
            num_shards=world_size * ddp,
            rank_override=effective_q_rank,
        ),
    )
    set_weight_loader(module.o_weight_scale, o_proj_weight_scale_loader_mxfp8())
    set_weight_loader(module.o_input_scale, o_proj_input_scale_loader_mxfp8())


def attach_mlp_loaders(
    module,
    *,
    intermediate_size_per_rank: int,
    mlp_tp_size: int,
    mlp_tp_rank: int,
    hidden_size: int,
) -> None:
    """Attach MX MLP weight loaders to module params."""
    set_weight_loader(
        module.gate_proj_weight,
        mlp_gate_up_weight_loader_mxfp8_cte(
            intermediate_size_per_rank=intermediate_size_per_rank,
            hidden_size=hidden_size,
            tp_size=mlp_tp_size,
            tp_rank=mlp_tp_rank,
        ),
    )
    set_weight_loader(
        module.up_proj_weight,
        mlp_gate_up_weight_loader_mxfp8_cte(
            intermediate_size_per_rank=intermediate_size_per_rank,
            hidden_size=hidden_size,
            tp_size=mlp_tp_size,
            tp_rank=mlp_tp_rank,
        ),
    )
    set_weight_loader(
        module.down_proj_weight,
        mlp_down_weight_loader_mxfp8_cte(
            intermediate_size_per_rank=intermediate_size_per_rank,
            hidden_size=hidden_size,
            tp_size=mlp_tp_size,
            tp_rank=mlp_tp_rank,
        ),
    )
    set_weight_loader(module.gate_weight_scale, mlp_weight_scale_loader_mxfp8())
    set_weight_loader(module.up_weight_scale, mlp_weight_scale_loader_mxfp8())
    set_weight_loader(module.down_weight_scale, mlp_weight_scale_loader_mxfp8())
    set_weight_loader(module.gate_up_input_scale, mlp_input_scale_loader_mxfp8())
    set_weight_loader(module.down_input_scale, mlp_input_scale_loader_mxfp8())


def attach_attention_loaders_tkg(
    module,
    *,
    q_size: int,
    kv_size: int,
    world_size: int,
    num_kv_replicas: int,
    attention_dp_size: int = 1,
    attention_dp_rank: int = 0,
    kv_needs_a2a: bool = False,
    num_attention_heads: int,
    head_dim: int,
) -> None:
    """Attach MX TKG attention weight loaders (x4-packed uint32)."""
    ddp = attention_dp_size
    effective_q_rank = attention_dp_rank + module.tp_group.rank_in_group * ddp
    o_shard_size = (num_attention_heads * head_dim) // (world_size * ddp)

    set_weight_loader(
        module.qkv_proj_weight,
        fused_qkv_weight_loader_mxfp8_tkg(
            q_size=q_size,
            kv_size=kv_size,
            num_shards=world_size,
            num_kv_replicas=num_kv_replicas,
            attention_dp_size=ddp,
            attention_dp_rank=attention_dp_rank,
            kv_sharded_across_attention_dp=kv_needs_a2a,
        ),
    )
    set_weight_loader(module.qkv_weight_scale, fused_qkv_weight_scale_loader_mxfp8())
    set_weight_loader(module.qkv_input_scale, fused_qkv_input_scale_loader_mxfp8())

    set_weight_loader(
        module.o_proj_weight,
        o_proj_weight_loader_mxfp8_tkg(
            shard_size=o_shard_size,
            num_shards=world_size * ddp,
            rank_override=effective_q_rank,
        ),
    )
    set_weight_loader(module.o_weight_scale, o_proj_weight_scale_loader_mxfp8())
    set_weight_loader(module.o_input_scale, o_proj_input_scale_loader_mxfp8())


def attach_mlp_loaders_tkg(
    module,
    *,
    intermediate_size_per_rank: int,
    mlp_tp_size: int,
    mlp_tp_rank: int,
    hidden_size: int,
) -> None:
    """Attach MX TKG MLP weight loaders (x4-packed uint32)."""
    set_weight_loader(
        module.gate_proj_weight,
        mlp_gate_up_weight_loader_mxfp8_tkg(
            intermediate_size_per_rank=intermediate_size_per_rank,
            hidden_size=hidden_size,
            tp_size=mlp_tp_size,
            tp_rank=mlp_tp_rank,
        ),
    )
    set_weight_loader(
        module.up_proj_weight,
        mlp_gate_up_weight_loader_mxfp8_tkg(
            intermediate_size_per_rank=intermediate_size_per_rank,
            hidden_size=hidden_size,
            tp_size=mlp_tp_size,
            tp_rank=mlp_tp_rank,
        ),
    )
    set_weight_loader(
        module.down_proj_weight,
        mlp_down_weight_loader_mxfp8_tkg(
            intermediate_size_per_rank=intermediate_size_per_rank,
            hidden_size=hidden_size,
            tp_size=mlp_tp_size,
            tp_rank=mlp_tp_rank,
        ),
    )
    set_weight_loader(module.gate_weight_scale, mlp_weight_scale_loader_mxfp8())
    set_weight_loader(module.up_weight_scale, mlp_weight_scale_loader_mxfp8())
    set_weight_loader(module.down_weight_scale, mlp_weight_scale_loader_mxfp8())
    set_weight_loader(module.gate_up_input_scale, mlp_input_scale_loader_mxfp8())
    set_weight_loader(module.down_input_scale, mlp_input_scale_loader_mxfp8())
