# SPDX-License-Identifier: Apache-2.0
"""
GLM weight loaders
==================

<-- MODEL-SPECIFIC: Checkpoint transforms for GLM.

1. Partial interleaved RoPE to contiguous RoPE.
   HF GLM rotates the pairs (2i, 2i+1) of the first ``rotary_dim`` dimensions
   of each head with frequency i. The other dimensions pass through. The Neuron
   attention kernels use the contiguous ("rotate_half") layout over the full
   head, where dimension j pairs with dimension j + head_dim / 2.

   The loader permutes the rows of each Q and K head (weight and bias) so that
   source dimension 2i goes to position i, and source dimension 2i + 1 goes to
   position i + head_dim / 2. The pass-through dimensions fill the remaining
   positions in order. The rotary frequency of those positions is zero
   (cos = 1, sin = 0), so they pass through unchanged.

   The same permutation applies to Q and K, so each Q.K dot product is
   unchanged. V and the output projection do not change. The KV cache holds K
   in the permuted layout. Only this model reads the cache.

2. Fused gate/up projection.
   The checkpoint stores ``mlp.gate_up_proj.weight`` with shape
   [2 * intermediate, hidden]: the gate rows first, then the up rows. The
   loaders split it into the separate gate and up parameters of ``NF.mlp``.
"""

from typing import TYPE_CHECKING

import torch

from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

if TYPE_CHECKING:
    from safetensors import PySafeSlice


def glm_rope_permutation(head_dim: int, rotary_dim: int) -> torch.Tensor:
    """Return the per-head row permutation from HF GLM layout to contiguous layout.

    ``new_head[d] = old_head[perm[d]]`` for each d in ``range(head_dim)``.

    Example::

        >>> glm_rope_permutation(8, 4).tolist()
        [0, 2, 4, 5, 1, 3, 6, 7]
    """
    if rotary_dim % 2 != 0 or head_dim % 2 != 0:
        raise ValueError(
            f"rotary_dim ({rotary_dim}) and head_dim ({head_dim}) must be even."
        )
    half_d = head_dim // 2
    half_r = rotary_dim // 2
    if half_r > half_d:
        raise ValueError(
            f"rotary_dim ({rotary_dim}) is larger than head_dim ({head_dim})."
        )
    perm = [0] * head_dim
    for i in range(half_r):
        perm[i] = 2 * i
        perm[half_d + i] = 2 * i + 1
    pass_src = iter(range(rotary_dim, head_dim))
    for d in list(range(half_r, half_d)) + list(range(half_d + half_r, head_dim)):
        perm[d] = next(pass_src)
    # Set the CPU device explicitly. The V2 runner builds the model in a
    # ``torch.device("meta")`` context, and the constructor makes the loaders.
    # A meta index makes the loaded Q/K/V tensors meta tensors without data.
    return torch.tensor(perm, dtype=torch.long, device="cpu")


def glm_kernel_inv_freq(
    head_dim: int, rotary_dim: int, rope_theta: float, device=None
) -> torch.Tensor:
    """Return ``head_dim // 2`` inverse frequencies for the contiguous layout.

    The first ``rotary_dim // 2`` entries are the HF GLM frequencies. The other
    entries are zero, so their positions pass through (cos = 1, sin = 0).

    Example::

        >>> [round(f, 4) for f in glm_kernel_inv_freq(8, 4, 10000.0).tolist()]
        [1.0, 0.01, 0.0, 0.0]
    """
    half_d = head_dim // 2
    half_r = rotary_dim // 2
    # Use only out-of-place operations. The model calls this function in the
    # compiled forward on the device. An in-place slice write into a new
    # tensor is a lowering that no other model in this repository uses there.
    rotary_inv_freq = 1.0 / (
        rope_theta
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    if half_r == half_d:
        return rotary_inv_freq
    pass_through = torch.zeros(half_d - half_r, dtype=torch.float32, device=device)
    return torch.cat((rotary_inv_freq, pass_through))


def _permute_head_rows(
    tensor: torch.Tensor, head_dim: int, perm: torch.Tensor
) -> torch.Tensor:
    """Permute dim 0 inside each head of a [num_heads * head_dim, ...] tensor."""
    rows = tensor.shape[0]
    if rows % head_dim != 0:
        raise ValueError(
            f"Row count ({rows}) is not a multiple of head_dim ({head_dim})."
        )
    rest = tensor.shape[1:]
    view = tensor.reshape(rows // head_dim, head_dim, *rest)
    return view[:, perm].reshape(rows, *rest)


def _row_slice(slice_obj: "PySafeSlice", start: int, end: int) -> torch.Tensor:
    ndim = len(slice_obj.get_shape())
    if ndim == 1:
        return slice_obj[start:end]
    if ndim == 2:
        return slice_obj[start:end, :]
    raise ValueError(f"Expected a 1-D or 2-D checkpoint tensor, got {ndim} dims.")


def glm_fused_qkv_loader(
    q_size: int,
    kv_size: int,
    num_shards: int,
    num_kv_replicas: int,
    head_dim: int,
    rotary_dim: int,
    is_bias: bool,
) -> SafetensorsWeightLoader:
    """Fuse the Q, K and V checkpoint tensors for one TP rank and apply the RoPE permutation.

    The checkpoint stores weights as [out_features, hidden] and biases as
    [out_features]. The loader returns a [hidden, q_size + 2 * kv_size] weight,
    or a [1, q_size + 2 * kv_size] bias.

    Example::

        loader = glm_fused_qkv_loader(
            q_size=256, kv_size=128, num_shards=16, num_kv_replicas=8,
            head_dim=128, rotary_dim=64, is_bias=False,
        )
    """
    perm = glm_rope_permutation(head_dim, rotary_dim)

    def transform(slices: list["PySafeSlice"], rank: int) -> torch.Tensor:
        if len(slices) != 3:
            raise ValueError(
                f"glm_fused_qkv_loader needs the [Q, K, V] slices, got {len(slices)}."
            )
        local_rank = rank % num_shards
        q_rank = local_rank
        kv_rank = local_rank // num_kv_replicas
        parts = []
        for idx, (slice_obj, size, shard_rank) in enumerate(
            zip(slices, [q_size, kv_size, kv_size], [q_rank, kv_rank, kv_rank])
        ):
            part = _row_slice(slice_obj, shard_rank * size, (shard_rank + 1) * size)
            if idx < 2:
                # Q and K get the RoPE permutation. V keeps its layout.
                part = _permute_head_rows(part, head_dim, perm)
            parts.append(part)
        if is_bias:
            fused = torch.cat(parts, dim=0).unsqueeze(0)
        else:
            fused = torch.cat([p.T for p in parts], dim=1)
        if fused.device.type == "meta":
            raise RuntimeError(
                "The fused QKV tensor is on the meta device and has no data. "
                "Make each index tensor of the loader a CPU tensor."
            )
        return fused

    return SafetensorsWeightLoader(transform=transform)


def glm_gate_up_loader(
    intermediate_size: int,
    shard_size: int,
    num_shards: int,
    is_up: bool,
) -> SafetensorsWeightLoader:
    """Take the gate or up rows of the fused gate_up tensor for one TP rank.

    Returns a [hidden, shard_size] tensor.

    Example::

        gate = glm_gate_up_loader(13696, 856, 16, is_up=False)
        up = glm_gate_up_loader(13696, 856, 16, is_up=True)
    """

    def transform(slices: list["PySafeSlice"], rank: int) -> torch.Tensor:
        if len(slices) != 1:
            raise ValueError(f"glm_gate_up_loader needs one slice, got {len(slices)}.")
        slice_obj = slices[0]
        shape = slice_obj.get_shape()
        if len(shape) != 2 or shape[0] != 2 * intermediate_size:
            raise ValueError(
                f"gate_up_proj has shape {list(shape)}. "
                f"Expected [{2 * intermediate_size}, hidden]."
            )
        base = intermediate_size if is_up else 0
        start = base + (rank % num_shards) * shard_size
        return slice_obj[start : start + shard_size, :].T

    return SafetensorsWeightLoader(transform=transform)
