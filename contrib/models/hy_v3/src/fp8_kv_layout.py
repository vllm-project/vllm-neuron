# SPDX-License-Identifier: Apache-2.0
"""Packed FP8 K-cache layout helpers (local copies, so the model only uses public APIs).

``NF.attention_decode(fp8_packed=True)`` reads K in the packed layout

    unpacked: [..., block_len, d_head]
    packed:   [..., block_len // 2, d_head, 2]   (dim -1: 0 = even position, 1 = odd)

``unswizzle_packed_k`` / ``swizzle_packed_k`` convert between the two.
``packed_decode_viable`` decides, per compiled decode bucket, whether the
packed kernel can read the cache directly. These mirror the vllm_neuron
attention helpers of the same purpose (test/unit/test_fp8_kv_layout.py checks
them against those when they are importable).
"""

import math

import torch

from vllm_neuron.utils.hardware_config import get_logical_nc_config

from .weight_loaders import PMAX


def unswizzle_packed_k(k_packed: torch.Tensor) -> torch.Tensor:
    """[..., block_len // 2, d_head, 2] -> [..., block_len, d_head] (dtype unchanged)."""
    *lead, half, d_head, two = k_packed.shape
    if two != 2:
        raise ValueError(f"packed K last dim must be 2, got {two}")
    return k_packed.transpose(-1, -2).reshape(*lead, half * 2, d_head).contiguous()


def swizzle_packed_k(k_unpacked: torch.Tensor) -> torch.Tensor:
    """[..., block_len, d_head] -> [..., block_len // 2, d_head, 2] (dtype unchanged)."""
    *lead, block_len, d_head = k_unpacked.shape
    if block_len % 2:
        raise ValueError(f"block_len must be even to pack, got {block_len}")
    return (
        k_unpacked.reshape(*lead, block_len // 2, 2, d_head).transpose(-1, -2).contiguous()
    )


def packed_decode_viable(block_len: int, bs: int, q_head: int, s_active: int, s_prior: int) -> bool:
    """Whether the packed decode kernel can read this bucket's cache directly.

    The packed layout pairs two positions, so the block length the kernel
    actually uses must stay >= 2. The kernel needs the blocks per batch to be a
    multiple of ``n_prgs * PMAX`` (``n_prgs`` = LNC degree when it shards
    s_prior, else 1) and shrinks the block to the largest divisor of block_len
    that gets there, so small per-rank contexts (s_prior) end at 1. s_prior is
    sharded when the batch is not; batch sharding wins when ``bs % lnc == 0``
    and (``bs * q_head * s_active >= PMAX`` or ``s_prior <= lnc * PMAX``).
    """
    if block_len <= 0 or s_prior <= 0:
        return False
    lnc = get_logical_nc_config()
    batch_sharded = bs % lnc == 0 and (bs * q_head * s_active >= PMAX or s_prior <= lnc * PMAX)
    n_prgs = lnc if (not batch_sharded and s_prior >= lnc * PMAX) else 1
    bucket_len = (s_prior // block_len) * block_len
    multiple = n_prgs * PMAX
    kernel_block_len = (
        block_len if bucket_len % multiple else math.gcd(block_len, bucket_len // multiple)
    )
    return kernel_block_len >= 2
