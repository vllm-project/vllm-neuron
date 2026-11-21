# SPDX-License-Identifier: Apache-2.0
"""Decode attention over a block KV cache with one KV head, with the nkilib attention_tkg kernel.

This op does the attention part of a decode step only. The caller does the QKV
projection, the norms, RoPE, the KV cache write and the output projection. Use it
when the fused decode block (`attention_decode`) does not fit the model, for
example for head_dim 256 or for an output gate.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from nkilib.core.attention.attention_tkg import attention_tkg
from nkilib.core.attention.attention_tkg_utils import AttnTKGConfig
from nkilib.core.utils.allocator import create_auto_alloc_manager
from nkilib.core.utils.logging import Logger
from torch import Tensor

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.hardware_config import get_logical_nc_config
from vllm_neuron.utils.neuron_utils import can_run_kernel

_P_MAX = 128
_MAX_D_HEAD = 512
# With LNC2 the kernel can split s_prior between the 2 cores. Each half must be
# a multiple of 128 tokens, so pad the table to a multiple of 256 tokens.
_S_PRIOR_MULTIPLE = 256


@nki.jit
def _attention_decode_block_kv_kernel(
    q, k_active, v_active, k_cache, v_cache, active_mask, pos_ids, block_table
):
    """Call attention_tkg for one decode token for each row.

    q is [d, B * H] (scaled), k_active is [d, B], v_active is [B, 1, 1, d], the
    caches are [num_blocks, block_len, d], active_mask is [1, B, H, 1] uint8,
    pos_ids is [B, 1] float32 and block_table is [B, nb] int32 (-1 = skip).
    The output is [B, H, d, 1].
    """
    d, BH = q.shape
    B = k_active.shape[1]
    H = BH // B
    s_prior = block_table.shape[1] * k_cache.shape[1]
    cfg = AttnTKGConfig(
        bs=B,
        q_head=H,
        s_active=1,
        curr_sprior=s_prior,
        full_sprior=s_prior,
        d_head=d,
        block_len=k_cache.shape[1],
        tp_k_prior=True,
        strided_mm1=False,
        use_pos_id=True,
        fuse_rope=False,
        use_gpsimd_sb2sb=True,
        qk_in_sb=True,
        k_out_in_sb=False,
        out_in_sb=False,
    )
    sbm = create_auto_alloc_manager(logger=Logger("SBM"))
    sbm.open_scope()
    # A block KV cache needs q and k in SBUF. Above 128, the head dimension is in
    # 128-row tiles side by side on the free axis.
    d_tile = min(d, _P_MAX)
    n_d = (d + _P_MAX - 1) // _P_MAX
    q_sb = sbm.alloc_stack((d_tile, n_d * BH), dtype=q.dtype, buffer=nl.sbuf)
    k_sb = sbm.alloc_stack((d_tile, n_d * B), dtype=k_active.dtype, buffer=nl.sbuf)
    for i_d in range(n_d):
        nisa.dma_copy(
            q_sb[:, i_d * BH : (i_d + 1) * BH], q[i_d * d_tile : (i_d + 1) * d_tile, :]
        )
        nisa.dma_copy(
            k_sb[:, i_d * B : (i_d + 1) * B],
            k_active[i_d * d_tile : (i_d + 1) * d_tile, :],
        )
    out = nl.ndarray((B, H, d, 1), dtype=q.dtype, buffer=nl.shared_hbm)
    attn_out, _ = attention_tkg(
        q_sb,
        k_sb,
        v_active,
        k_cache,
        v_cache,
        active_mask,
        out,
        cfg,
        sbm,
        rope_pos_ids=pos_ids,
        active_blocks_table=block_table,
    )
    sbm.close_scope()
    return attn_out


def kernel_block_table(
    block_table: Tensor, positions: Tensor, block_len: int
) -> Tensor:
    """Make the block table that the kernel reads.

    The kernel skips an entry of -1. Set -1 for a null entry (0 or less) and for a
    block that starts after the position of the row, so that the kernel reads only
    blocks with real tokens. Pad the table with -1 to a multiple of 256 tokens.

    Args:
        block_table: [B, nb] block IDs. 0 or -1 marks an unused entry.
        positions: [B] position of the new token of each row.
        block_len: Number of tokens in one block.

    Returns:
        An int32 tensor [B, nb_padded].

    Example:
        >>> t = kernel_block_table(torch.tensor([[3, 5, 0]]), torch.tensor([70]), 64)
        >>> t.shape, t[0, :3].tolist()
        (torch.Size([1, 4]), [3, 5, -1])
    """
    B, nb = block_table.shape
    unit = max(1, _S_PRIOR_MULTIPLE // block_len)
    nb_padded = (nb + unit - 1) // unit * unit
    table = block_table
    if nb_padded > nb:
        pad = torch.full(
            (B, nb_padded - nb), -1, dtype=block_table.dtype, device=block_table.device
        )
        table = torch.cat([block_table, pad], dim=1)
    blk = torch.arange(nb_padded, device=block_table.device).unsqueeze(0)
    dead = (table <= 0) | (blk * block_len > positions.reshape(B, 1))
    return torch.where(dead, torch.full_like(table, -1), table).to(torch.int32)


def can_use_attention_decode_block_kv(q: Tensor, k_cache: Tensor) -> bool:
    """Check if the attention_tkg kernel can run this decode attention.

    Args:
        q: [B, H, d] query of one decode token for each row.
        k_cache: [num_blocks, block_len, d] or [num_blocks, 1, block_len, d].

    Returns:
        True if the kernel supports the shapes and the device.

    Example:
        >>> can_use_attention_decode_block_kv(torch.randn(6, 6, 256), torch.randn(9, 1, 64, 256))
        False
    """
    if not can_run_kernel(q):
        return False
    d = q.shape[-1]
    if d > _MAX_D_HEAD or (d > _P_MAX and d % _P_MAX != 0):
        return False
    if k_cache.dim() == 4 and k_cache.shape[1] != 1:
        return False
    return q.dtype == torch.bfloat16 and k_cache.dtype == torch.bfloat16


def _attention_decode_block_kv_torch(
    q: Tensor,
    k_active: Tensor,
    v_active: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    table: Tensor,
    positions: Tensor,
    softmax_scale: float,
) -> Tensor:
    B, H, d = q.shape
    block_len = k_cache.shape[-2]
    ids = table.clamp_min(0).long()
    nb = ids.shape[1]
    k = k_cache.reshape(-1, block_len, d)[ids].reshape(B, nb * block_len, d).float()
    v = v_cache.reshape(-1, block_len, d)[ids].reshape(B, nb * block_len, d).float()
    valid = (table.long() >= 0).repeat_interleave(block_len, dim=1)
    valid &= torch.arange(nb * block_len, device=q.device).unsqueeze(
        0
    ) < positions.reshape(B, 1)
    # A masked slot can hold any value. Zero it, so that 0 * NaN cannot occur.
    k = torch.where(valid.unsqueeze(-1), k, torch.zeros_like(k))
    v = torch.where(valid.unsqueeze(-1), v, torch.zeros_like(v))
    qs = q.float() * softmax_scale
    scores = torch.einsum("bhd,bsd->bhs", qs, k)
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    active = (qs * k_active.float().unsqueeze(1)).sum(-1, keepdim=True)
    weights = torch.softmax(torch.cat([scores, active], dim=-1), dim=-1)
    values = torch.cat([v, v_active.float().unsqueeze(1)], dim=1)
    return torch.einsum("bhs,bsd->bhd", weights, values).to(q.dtype).reshape(B, H * d)


def attention_decode_block_kv(
    q: Tensor,
    k_active: Tensor,
    v_active: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    block_table: Tensor,
    positions: Tensor,
    softmax_scale: float,
) -> Tensor:
    """Do decode attention for one token in each row over a block KV cache.

    Each row attends to the cached tokens at positions 0 to positions[b] - 1 and
    to its new token (k_active, v_active). The cache can already hold the new
    token at positions[b], because the op uses k_active for that position.

    There is one KV head, which all H query heads use. The nkilib attention_tkg
    kernel runs when `can_use_attention_decode_block_kv` is True. Otherwise a
    torch implementation runs.

    Every value in the caches must be finite when the kernel runs. The kernel
    can read a cached value with an attention weight of 0, also from a block
    that is not in the table of the row, and 0 * NaN is NaN. The torch
    implementation sets the masked values to 0.

    Args:
        q: [B, H, d] query, not scaled.
        k_active: [B, d] key of the new token.
        v_active: [B, d] value of the new token.
        k_cache: [num_blocks, block_len, d] or [num_blocks, 1, block_len, d].
        v_cache: Same shape as k_cache.
        block_table: [B, nb] block IDs. 0 or -1 marks an unused entry.
        positions: [B] position of the new token of each row.
        softmax_scale: Scale of q @ k^T, for example head_dim ** -0.5.

    Returns:
        [B, H * d] attention output, in the dtype of q.

    Example:
        >>> B, H, d, L = 2, 4, 128, 64
        >>> kc = torch.randn(5, L, d)
        >>> out = attention_decode_block_kv(
        ...     torch.randn(B, H, d), torch.randn(B, d), torch.randn(B, d), kc, kc.clone(),
        ...     torch.tensor([[1, 2], [3, 4]]), torch.tensor([70, 10]), d ** -0.5)
        >>> out.shape
        torch.Size([2, 512])
    """
    B, H, d = q.shape
    block_len = k_cache.shape[-2]
    table = kernel_block_table(block_table, positions, block_len)
    if not can_use_attention_decode_block_kv(q, k_cache):
        return _attention_decode_block_kv_torch(
            q, k_active, v_active, k_cache, v_cache, table, positions, softmax_scale
        )
    num_blocks = k_cache.shape[0]
    out = wrap_nki(_attention_decode_block_kv_kernel)[get_logical_nc_config()](
        (q * softmax_scale).permute(2, 0, 1).reshape(d, B * H).contiguous(),
        k_active.t().contiguous(),
        v_active.reshape(B, 1, 1, d).contiguous(),
        k_cache.reshape(num_blocks, block_len, d),
        v_cache.reshape(num_blocks, block_len, d),
        torch.ones(1, B, H, 1, dtype=torch.uint8, device=q.device),
        positions.to(torch.float32).reshape(B, 1),
        table,
    )
    return out.reshape(B, H * d)
