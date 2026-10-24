# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-to-parameter transforms for the static per-tensor FP8 Hy3 checkpoint.

The checkpoint is calibrated for OCP ``float8_e4m3fn`` (max 448). The STATIC
kernels use the 240-capped ``float8_e4m3`` on TRN2 (and ``moe_tkg`` defaults
to it everywhere). Both formats share exponent bias 7, so every value with
``|x| <= 240`` has the same encoding. Every FP8 weight is therefore halved at
load time (an exact exponent shift for normal values) and the matching weight
scale is doubled. Input (activation) scales are doubled as well so calibrated
activations quantize to ``|x/s| <= 224`` instead of clipping at 240.
"""

import torch

from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

FP8_DTYPE = torch.float8_e4m3fn

# Partition dimension the STATIC kernels expect for pre-broadcast scales.
PMAX = 128

# <-- STATIC-FP8: halving is exact for normal fp8 values and moves the
# 448-range checkpoint into the 240-range kernel dtype.
FP8_WEIGHT_DOWNSCALE = 0.5
FP8_SCALE_COMPENSATION = 1.0 / FP8_WEIGHT_DOWNSCALE


def read_scalar_from_slice(slice_obj) -> torch.Tensor:
    """Read a scalar from a ``PySafeSlice`` as a length-1 fp32 tensor (rank-0 safe)."""
    shape = slice_obj.get_shape()
    raw = slice_obj[()] if len(shape) == 0 else slice_obj[:]
    flat = raw.to(torch.float32).reshape(-1)
    assert flat.numel() == 1, (
        f"expected a scalar tensor, got shape {tuple(shape)} with {flat.numel()} elements"
    )
    return flat


def halve_fp8(w: torch.Tensor) -> torch.Tensor:
    assert w.dtype == FP8_DTYPE, f"expected a float8_e4m3fn weight, got {w.dtype}"
    return (w.float() * FP8_WEIGHT_DOWNSCALE).to(FP8_DTYPE)


def with_fp8_halving(loader: SafetensorsWeightLoader) -> SafetensorsWeightLoader:
    base = loader.transform or (lambda slices, rank: slices[0][:])

    def transform(slices, rank):
        return halve_fp8(base(slices, rank))

    return SafetensorsWeightLoader(transform=transform)


def scalar_scale_loader(num_slices: int = 1, cols: int = 1) -> SafetensorsWeightLoader:
    """Scalar scale(s) -> ``[128, cols]`` fp32, multiplied by the FP8 compensation.

    ``num_slices == cols``: one column per checkpoint scalar (fused QKV weight
    scale). ``cols == 1`` with several slices: the scalars must agree (shared
    activation scale of a fused projection) and a single column is returned.
    """

    def transform(slices, rank):
        assert len(slices) == num_slices, (
            f"expected {num_slices} slices, got {len(slices)}"
        )
        scalars = [read_scalar_from_slice(s) for s in slices]
        if cols == 1:
            for s in scalars[1:]:
                assert torch.equal(scalars[0], s), (
                    "fused projections must share one static input scale, got "
                    f"{[v.item() for v in scalars]}"
                )
            scalars = scalars[:1]
        stacked = torch.cat(scalars).reshape(1, cols) * FP8_SCALE_COMPENSATION
        return stacked.expand(PMAX, cols).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def _row_shard(slice_obj, start: int, size: int, dim: int) -> torch.Tensor:
    """Read ``[start, start+size)`` along ``dim`` of a 2-D slice, zero-padding past the end."""
    total = slice_obj.get_shape()[dim]
    end = min(start + size, total)
    if dim == 0:
        t = slice_obj[start:end, :]
    else:
        t = slice_obj[:, start:end]
    if end - start < size:
        pad_shape = list(t.shape)
        pad_shape[dim] = size - (end - start)
        t = torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype)], dim=dim)
    return t


def padded_mlp_loader(
    is_down: bool, shard_size: int, padded_size: int, tp_rank: int
) -> SafetensorsWeightLoader:
    """TP-shard a dense gate/up ([I, H]) or down ([H, I]) weight and zero-pad I.

    Returns ``[H, padded]`` for gate/up and ``[padded, H]`` for down, halved
    into the kernel fp8 range.
    """

    def transform(slices, rank):
        assert len(slices) == 1
        w = _row_shard(slices[0], tp_rank * shard_size, shard_size, 1 if is_down else 0)
        w = w.T  # gate/up: [H, shard], down: [shard, H]
        if shard_size < padded_size:
            pad_dim = 0 if is_down else 1
            pad_shape = list(w.shape)
            pad_shape[pad_dim] = padded_size - shard_size
            w = torch.cat([w, torch.zeros(pad_shape, dtype=w.dtype)], dim=pad_dim)
        return halve_fp8(w.contiguous())

    return SafetensorsWeightLoader(transform=transform)


def expert_gate_up_loader(
    num_experts: int, i_shard: int, i_rank: int
) -> SafetensorsWeightLoader:
    """``[gate_0..gate_{L-1}, up_0..up_{L-1}]`` ([I, H] each) -> ``[L, H, 2, I_tp]``."""

    def transform(slices, rank):
        assert len(slices) == 2 * num_experts
        experts = []
        for e in range(num_experts):
            g = slices[e][i_rank * i_shard : (i_rank + 1) * i_shard, :]
            u = slices[num_experts + e][i_rank * i_shard : (i_rank + 1) * i_shard, :]
            experts.append(torch.stack([g.T, u.T], dim=1))  # [H, 2, I_tp]
        return halve_fp8(torch.stack(experts).contiguous())

    return SafetensorsWeightLoader(transform=transform)


def expert_down_loader(
    num_experts: int, i_shard: int, i_rank: int
) -> SafetensorsWeightLoader:
    """``[down_0..down_{L-1}]`` ([H, I] each) -> ``[L, I_tp, H]``."""

    def transform(slices, rank):
        assert len(slices) == num_experts
        experts = [s[:, i_rank * i_shard : (i_rank + 1) * i_shard].T for s in slices]
        return halve_fp8(torch.stack(experts).contiguous())

    return SafetensorsWeightLoader(transform=transform)


def expert_scale_loader(
    num_experts: int, num_groups: int, shape_fn
) -> SafetensorsWeightLoader:
    """Per-expert scalar weight scales -> kernel layout.

    ``slices`` is grouped by projection (``[gate_0..gate_{L-1}, up_0..]``).
    ``shape_fn`` maps the ``[L, num_groups]`` fp32 scale matrix to the layout
    the kernel wants.
    """

    def transform(slices, rank):
        assert len(slices) == num_experts * num_groups
        vals = (
            torch.stack([read_scalar_from_slice(s)[0] for s in slices])
            .view(num_groups, num_experts)
            .T
        )  # [L, groups]
        return shape_fn(vals * FP8_SCALE_COMPENSATION).contiguous()

    return SafetensorsWeightLoader(transform=transform)
