# SPDX-License-Identifier: Apache-2.0
"""In-place paged KV-cache scatter for models without a fused attention kernel.

``Tensor.index_put_`` on a cache parameter is not an in-place write under
neuronx-cc: XLA has no in-place scatter, so it lowers to a full-tensor
``scatter`` that materializes a fresh copy of the ENTIRE cache pool (plus, in
practice, an identity ``transpose`` the compiler fails to elide). Writing one
token's 384 bytes of K therefore costs reads and writes of the whole pool.

The FX aliasing pass can rewrite the LAST write per cache parameter into a true
in-place update, but vLLM hands the same raw cache tensor to several layers at
once (``for layer_name in tensor.shared_by`` in ``initialize_kv_cache``), so
those layers' writes chain on one placeholder and every intermediate copy
survives. On MiMo-V2.5 that is 96 writes over 18 buffers -- ~124 GB of pool
traffic per decode step.

This kernel sidesteps the lowering: it DMAs the new rows straight into the cache
buffer with an indirect (``vector_offset``) scatter, and RETURNS the cache
tensors so the NKI compiler reports ``operand_output_aliases``. That puts the
write on the aliasing pass's NKI path, which chains ``k0 -> k1 -> ... -> kN``
across every write to a shared buffer instead of only aliasing the last one.

Models whose head dim fits the fused kernel's 128 cap should keep using
``NF.attention_decode(update_cache=True)``; this exists for the eager-attention
models (e.g. MiMo's 192-wide Q/K) that the fused path rejects.
"""

import os

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import oob_mode
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

from vllm_neuron.nki.nki_hop import can_run_kernel, wrap_nki

# Row index meaning "drop this row". ``oob_mode.skip`` discards indirect-DMA
# offsets that land outside the destination, and int32 max exceeds any real
# cache row count by orders of magnitude, so padding / freed slots
# (``slot_mapping < 0``) are mapped to it. Kept inside int32 so the index
# tensor needs no unsigned dtype, which the torch device layer handles poorly.
_SKIP_ROW = 0x7FFFFFFF


def _scatter_rows(cache_flat, new_rows, idx_tile, start, tile_n, d):
    """Stage one tile of ``new_rows`` into SBUF, then indirect-DMA it to *cache_flat*.

    Factored out and called twice rather than looped over ``(k, v)`` pairs
    because NKI's parser rejects tuple unpacking in a ``for`` target
    ("expecting simple variable").
    """
    tile = nl.ndarray((tile_n, d), dtype=new_rows.dtype, buffer=nl.sbuf)
    nisa.dma_copy(tile, new_rows[nl.ds(start, tile_n), :])
    nisa.dma_copy(
        dst=cache_flat.ap(
            pattern=[[d, tile_n], [1, d]],
            offset=0,
            vector_offset=idx_tile,
            indirect_dim=0,
        ),
        src=tile,
        oob_mode=oob_mode.skip,
    )


@nki.jit
def _kv_cache_scatter_kernel(
    k_cache: Tensor,
    v_cache: Tensor,
    k_new: Tensor,
    v_new: Tensor,
    row_idx: Tensor,
):
    """Scatter ``k_new``/``v_new`` into the K/V caches at rows ``row_idx``.

    The caches arrive 4D and are flattened HERE rather than by the caller: the
    aliasing pass only rewires a write whose FX shape matches the placeholder
    exactly, so a pre-call ``reshape`` would silently disable the write-chain
    serialization this kernel exists to enable.

    Args:
        k_cache: ``[num_blocks, num_kv_heads, block_size, d]``, written in place.
        v_cache: Same shape as *k_cache*.
        k_new: ``[N, d]`` rows to write.
        v_new: ``[N, d]`` rows to write.
        row_idx: ``[N, 1]`` destination row in the (block, head, position)
            flattening; ``_SKIP_ROW`` drops the row.

    Returns:
        ``(k_cache, v_cache)``. Returning them is what makes the NKI compiler
        emit ``operand_output_aliases`` -- a kernel that writes an input but
        returns nothing reports no aliases AND no outputs, so the write is
        dropped from the graph entirely.
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("kv_cache_write", (0, 1), 2)

    num_blocks = k_cache.shape[0]
    nkh = k_cache.shape[1]
    blk = k_cache.shape[2]
    d = k_cache.shape[3]
    n_rows = k_new.shape[0]

    k_flat = k_cache.reshape((num_blocks * nkh * blk, d))
    v_flat = v_cache.reshape((num_blocks * nkh * blk, d))

    tile_sz = nl.tile_size.pmax
    for start in range(0, n_rows, tile_sz):
        tile_n = min(tile_sz, n_rows - start)

        idx_tile = nl.ndarray((tile_n, 1), dtype=row_idx.dtype, buffer=nl.sbuf)
        nisa.dma_copy(idx_tile, row_idx[nl.ds(start, tile_n)])

        # Split K and V across the two logical cores (same split
        # ``_update_block_cache_vectorized`` uses) so the two scatters overlap.
        if n_prgs == 1 or prg_id == 0:
            _scatter_rows(v_flat, v_new, idx_tile, start, tile_n, d)
        if n_prgs == 1 or prg_id == 1:
            _scatter_rows(k_flat, k_new, idx_tile, start, tile_n, d)

    return k_cache, v_cache


def _can_use_kernel(k_cache: Tensor, v_cache: Tensor, k: Tensor) -> bool:
    """Whether the NKI scatter applies; otherwise fall back to ``index_put_``."""
    if not can_run_kernel(k):
        return False

    # Diagnostic escape hatch: force the (functionally equivalent, far slower)
    # index_put_ write. Narrower than VLLM_NEURON_DISABLE_NKI_KERNELS, which
    # would also drop the MoE/attention kernels -- this isolates "is a wrong
    # answer coming from the cache write, or from somewhere else?" while leaving
    # the rest of the model on its normal path.
    if os.environ.get("VLLM_NEURON_KV_WRITE_FORCE_TORCH") == "1":
        return False

    # The scatter itself is dtype-agnostic (it moves whole rows), but the SBUF
    # staging tile is typed. Restrict to the float types this is exercised
    # with; FP8 caches use a packed layout that needs read-modify-write.
    if k_cache.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if k_cache.dtype != v_cache.dtype or k.dtype != k_cache.dtype:
        return False

    # Both caches must be the canonical 4D paged buffer of the same shape, so
    # one row index serves both.
    if k_cache.dim() != 4 or k_cache.shape != v_cache.shape:
        return False

    return True


def write_paged_kv_cache(
    k_cache: Tensor,
    v_cache: Tensor,
    k: Tensor,
    v: Tensor,
    slot_mapping: Tensor,
    block_size: int,
    num_kv_heads: int,
) -> tuple[Tensor, Tensor]:
    """Write post-RoPE K/V into a paged KV cache at ``slot_mapping``.

    ``k``/``v`` are ``[num_kv_heads * n_tokens, d]`` in HEAD-MAJOR order (head
    0's tokens, then head 1's, ...), i.e. what ``[nkh, T, d].reshape(-1, d)``
    gives. Both must already be at the cache's width and dtype: padding V from
    ``v_head_dim`` up to a shared cache ``head_size`` is the caller's job, since
    only the caller knows the two widths.

    Args:
        k_cache: ``[num_blocks, num_kv_heads, block_size, d]``, written in place.
        v_cache: Same shape as *k_cache*.
        k: ``[num_kv_heads * n_tokens, d]`` rows to write.
        v: ``[num_kv_heads * n_tokens, d]`` rows to write.
        slot_mapping: ``[n_tokens]`` global slot per token; negative entries
            (padding / freed rows) are dropped.
        block_size: Slots per block.
        num_kv_heads: KV heads held by this rank.

    Returns:
        ``(k_cache, v_cache)``. On the kernel path these are the kernel's
        aliased outputs; callers MUST rebind their cache references to them so
        the aliasing pass can thread the write through to downstream readers.
    """
    n_tokens = slot_mapping.shape[0]

    if not _can_use_kernel(k_cache, v_cache, k):
        # ``index_put_`` mutates in place, so returning the inputs gives this
        # path the same contract as the kernel's.
        block_indices = (slot_mapping // block_size).repeat(num_kv_heads)
        position_indices = (slot_mapping % block_size).repeat(num_kv_heads)
        head_indices = torch.arange(
            num_kv_heads, dtype=torch.long, device=k.device
        ).repeat_interleave(n_tokens)
        k_cache.index_put_((block_indices, head_indices, position_indices), k)
        v_cache.index_put_((block_indices, head_indices, position_indices), v)
        return k_cache, v_cache

    _, nkh, blk, _ = k_cache.shape

    # Flat row in the (block, head, position) flattening:
    #   row = block * (nkh * blk) + head * blk + pos
    # slot_mapping is (block * block_size + pos) in a head-independent frame, so
    # split it before re-weighting the block stride by nkh.
    slots = slot_mapping.to(torch.int32)
    base = (slots // block_size) * (nkh * blk) + (slots % block_size)
    head_stride = (
        torch.arange(nkh, dtype=torch.int32, device=slots.device) * blk
    ).repeat_interleave(n_tokens)  # head-major, matching k/v row order
    rows = base.repeat(nkh) + head_stride

    # Negative slots are padding / freed rows. Redirect them to the skip
    # sentinel; left alone, the divmod above would fold them onto a live row.
    rows = torch.where(
        slots.repeat(nkh) < 0, torch.full_like(rows, _SKIP_ROW), rows
    )

    wrapped = wrap_nki(_kv_cache_scatter_kernel)
    return wrapped[2](
        k_cache=k_cache,
        v_cache=v_cache,
        k_new=k,
        v_new=v,
        row_idx=rows.view(-1, 1),
    )
