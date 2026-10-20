# SPDX-License-Identifier: Apache-2.0
"""
Static FP8 weight loaders for Llama3 (TRN2 + TRN3 fallback).

Mirrors the public API of :mod:`weight_loaders_mx_fp8` so the model
file can pick which loader implementation to call based on a runtime
flag (platform / dims / spec-decode availability).

Every function in this module is named identically to its counterpart
in :mod:`weight_loaders_mx_fp8`:

  * :func:`attach_attention_loaders`
  * :func:`attach_mlp_loaders`

These are the only public entry points; everything else is internal.
"""

from __future__ import annotations

import torch
from torch import nn

from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
    with_rank_override,
)


# ---------------------------------------------------------------------------
# Static-FP8 constants
# ---------------------------------------------------------------------------

# Static-FP8 weight dtype.
_FP8_DTYPE = torch.float8_e4m3fn

# Partition dimension size the kernels expect for pre-broadcast scales.
_PMAX = 128

# Number of per-projection scales fused into the QKV weight_scale tensor.
_QKV_FUSED = 3


# ---------------------------------------------------------------------------
# Legacy fp8 range compensation (STATIC fp8 path on trn2 only)
# ---------------------------------------------------------------------------
#
# Trn2's nkilib STATIC kernels hardcode SBUF dtype to legacy
# ``nl.float8_e4m3`` (max 240, no NaN). ModelOpt checkpoints use OCP
# ``float8_e4m3fn`` (max 448), so on trn2 we downscale weight bytes into
# the legacy range and compensate the dequant scale by the inverse factor.
# Trn3 supports ``float8_e4m3fn`` natively — no downscale needed.

_FP8_E4M3FN_MAX = 448.0
_FP8_E4M3_MAX = 240.0
_FP8_WEIGHT_DOWNSCALE = _FP8_E4M3_MAX / _FP8_E4M3FN_MAX
_FP8_SCALE_COMPENSATION = _FP8_E4M3FN_MAX / _FP8_E4M3_MAX


def _needs_downscale() -> bool:
    """Return True on trn2 where the STATIC kernel uses legacy 240-range fp8."""
    from libtorch_neuronx_lite.compile.platform import get_platform_target

    return get_platform_target() == "trn2"


def _downscale_fp8_weight(w_fp8: torch.Tensor) -> torch.Tensor:
    """Scale a ModelOpt-calibrated fp8 weight into legacy ``nl.float8_e4m3`` range."""
    return (
        (w_fp8.float() * _FP8_WEIGHT_DOWNSCALE)
        .clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
        .to(_FP8_DTYPE)
    )


def _wrap_with_fp8_downscale(
    loader: SafetensorsWeightLoader,
) -> SafetensorsWeightLoader:
    """Conditionally apply 240/448 downscale (trn2 only)."""
    if not _needs_downscale():
        return loader
    base_transform = loader.transform or (lambda slices, rank: slices[0][:])

    def transform(slices, rank):
        return _downscale_fp8_weight(base_transform(slices, rank))

    return SafetensorsWeightLoader(transform=transform)


def _read_scalar_from_slice(slice_obj) -> torch.Tensor:
    """Read a rank-0/rank-1 scalar fp32 from a ``PySafeSlice``."""
    shape = slice_obj.get_shape()
    raw = slice_obj[()] if len(shape) == 0 else slice_obj[:]
    flat = raw.to(torch.float32).reshape(-1)
    assert flat.numel() == 1, (
        f"_read_scalar_from_slice expects a scalar tensor, got shape "
        f"{tuple(shape)} with {flat.numel()} elements"
    )
    return flat


def _broadcast_scalar_scale_loader(
    is_weight_scale: bool,
) -> SafetensorsWeightLoader:
    """Scalar fp32 -> ``[128, 1]`` fp32 broadcast.

    On trn2, weight scales are pre-multiplied by ``_FP8_SCALE_COMPENSATION``
    to compensate the 240/448 weight downscale. On trn3 no compensation.
    """

    def transform(slices, rank):
        mult = (
            _FP8_SCALE_COMPENSATION if (is_weight_scale and _needs_downscale()) else 1.0
        )
        assert len(slices) == 1, (
            f"_broadcast_scalar_scale_loader expects 1 slice, got {len(slices)}"
        )
        scalar = _read_scalar_from_slice(slices[0])
        return (scalar * mult).expand(_PMAX, 1).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def _fused_qkv_weight_scale_loader() -> SafetensorsWeightLoader:
    """Stack three Q/K/V scalar weight scales into a ``[128, 3]`` fp32 tensor."""

    def transform(slices, rank):
        mult = _FP8_SCALE_COMPENSATION if _needs_downscale() else 1.0
        assert len(slices) == _QKV_FUSED, (
            f"_fused_qkv_weight_scale_loader expects {_QKV_FUSED} slices, "
            f"got {len(slices)}"
        )
        cols = [_read_scalar_from_slice(s) * mult for s in slices]
        stacked = torch.stack(cols, dim=0).reshape(1, _QKV_FUSED)
        return stacked.expand(_PMAX, _QKV_FUSED).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def _fused_qkv_input_scale_loader() -> SafetensorsWeightLoader:
    """Collapse three Q/K/V input scales (which must agree) to one ``[128, 1]``."""

    def transform(slices, rank):
        assert len(slices) == _QKV_FUSED, (
            f"_fused_qkv_input_scale_loader expects {_QKV_FUSED} slices, "
            f"got {len(slices)}"
        )
        scalars = [_read_scalar_from_slice(s) for s in slices]
        for idx, sc in enumerate(scalars[1:], start=1):
            assert torch.equal(scalars[0], sc), (
                "_fused_qkv_input_scale_loader expects identical "
                f"input_scale across Q/K/V for fused QKV; got "
                f"Q={scalars[0].item()} vs index {idx}={sc.item()}."
            )
        return scalars[0].expand(_PMAX, 1).contiguous()

    return SafetensorsWeightLoader(transform=transform)


# ---------------------------------------------------------------------------
# Public entry points (mirror weight_loaders_mx_fp8.py)
# ---------------------------------------------------------------------------


def attach_attention_loaders(
    module: nn.Module,
    *,
    q_size: int,
    kv_size: int,
    world_size: int,
    num_kv_replicas: int,
    attention_dp_size: int,
    attention_dp_rank: int,
    kv_needs_a2a: bool,
    num_attention_heads: int,
    head_dim: int,
) -> None:
    """Attach static-FP8 loaders to the LlamaAttention module's params/buffers.

    Expects ``module`` to expose:
      * ``qkv_proj_weight`` / ``o_proj_weight``  (nn.Parameter, fp8)
      * ``qkv_weight_scale`` / ``qkv_input_scale``  (buffer, fp32)
      * ``o_weight_scale`` / ``o_input_scale``      (buffer, fp32)
    """
    ddp = attention_dp_size
    effective_q_rank = attention_dp_rank + module.tp_group.rank_in_group * ddp
    o_shard_size = (num_attention_heads * head_dim) // (world_size * ddp)

    set_weight_loader(
        module.qkv_proj_weight,
        _wrap_with_fp8_downscale(
            fused_qkv_weight_loader(
                q_size=q_size,
                kv_size=kv_size,
                shard_dim=1,
                num_shards=world_size,
                is_storage_transposed=True,
                num_kv_replicas=num_kv_replicas,
                attention_dp_rank=attention_dp_rank,
                attention_dp_size=ddp,
                kv_sharded_across_attention_dp=kv_needs_a2a,
            )
        ),
    )
    set_weight_loader(module.qkv_weight_scale, _fused_qkv_weight_scale_loader())
    set_weight_loader(module.qkv_input_scale, _fused_qkv_input_scale_loader())

    set_weight_loader(
        module.o_proj_weight,
        _wrap_with_fp8_downscale(
            with_rank_override(
                sharding_weight_loader(
                    shard_dim=0,
                    shard_size=o_shard_size,
                    num_shards=world_size * ddp,
                    is_storage_transposed=True,
                ),
                rank=effective_q_rank,
            )
        ),
    )
    set_weight_loader(
        module.o_weight_scale,
        _broadcast_scalar_scale_loader(is_weight_scale=True),
    )
    set_weight_loader(
        module.o_input_scale,
        _broadcast_scalar_scale_loader(is_weight_scale=False),
    )


def attach_mlp_loaders(
    module: nn.Module,
    *,
    intermediate_size_per_rank: int,
    mlp_tp_size: int,
    mlp_tp_rank: int,
    hidden_size: int | None = None,
) -> None:
    """Attach static-FP8 loaders to the LlamaMLP module's params/buffers.

    ``hidden_size`` is accepted but unused (kept for API parity with
    :func:`weight_loaders_mx_fp8.attach_mlp_loaders`).

    Expects ``module`` to expose:
      * ``gate_proj_weight`` / ``up_proj_weight`` / ``down_proj_weight``
      * ``gate_weight_scale`` / ``up_weight_scale`` / ``down_weight_scale``
      * ``gate_up_input_scale`` / ``down_input_scale``
    """
    gate_up_loader = sharding_weight_loader(
        shard_dim=1,
        shard_size=intermediate_size_per_rank,
        num_shards=mlp_tp_size,
        is_storage_transposed=True,
    )
    down_loader = sharding_weight_loader(
        shard_dim=0,
        shard_size=intermediate_size_per_rank,
        num_shards=mlp_tp_size,
        is_storage_transposed=True,
    )
    gate_up_loader = with_rank_override(gate_up_loader, rank=mlp_tp_rank)
    down_loader = with_rank_override(down_loader, rank=mlp_tp_rank)
    gate_up_loader = _wrap_with_fp8_downscale(gate_up_loader)
    down_loader = _wrap_with_fp8_downscale(down_loader)

    set_weight_loader(module.gate_proj_weight, gate_up_loader)
    set_weight_loader(module.up_proj_weight, gate_up_loader)
    set_weight_loader(module.down_proj_weight, down_loader)

    set_weight_loader(
        module.gate_weight_scale,
        _broadcast_scalar_scale_loader(is_weight_scale=True),
    )
    set_weight_loader(
        module.up_weight_scale,
        _broadcast_scalar_scale_loader(is_weight_scale=True),
    )
    set_weight_loader(
        module.down_weight_scale,
        _broadcast_scalar_scale_loader(is_weight_scale=True),
    )
    set_weight_loader(
        module.gate_up_input_scale,
        _broadcast_scalar_scale_loader(is_weight_scale=False),
    )
    set_weight_loader(
        module.down_input_scale,
        _broadcast_scalar_scale_loader(is_weight_scale=False),
    )
