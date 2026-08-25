# SPDX-License-Identifier: Apache-2.0
import math
from typing import Optional

import nki
import torch
from nkilib.core.attention.gen_mask_tkg import gen_mask_tkg_hbm
from torch import Tensor

# QK-swap mask layout: newer nkilib ``attention_block_tkg`` selects a transposed,
# column-tiled MM1 ("QK-swap") path for certain decode shapes and then expects
# the pre-generated mask as ``[B, q_heads, S_tkg, S_ctx]`` instead of the legacy
# ``[S_ctx, B, q_heads, S_tkg]`` produced by ``gen_attention_decode_mask``. The
# decision is a kernel internal; this module (which already owns the nkilib mask
# layout contract) imports the predicate so callers don't have to. Guarded so
# older nkilib without the predicate keeps the legacy layout.
try:
    import inspect as _inspect

    from nkilib.core.attention.attention_tkg_utils import (
        is_qk_swapped as _nki_is_qk_swapped,
    )

    _QK_SWAP_PARAMS = frozenset(_inspect.signature(_nki_is_qk_swapped).parameters)
except ImportError:
    _nki_is_qk_swapped = None
    _QK_SWAP_PARAMS = frozenset()


def maybe_transpose_mask_for_qk_swap(
    attention_mask: Optional[Tensor],
    d_head: int,
    is_fp8_kv: bool = False,
    fp8_packed: bool = False,
) -> Optional[Tensor]:
    """Transpose a pre-generated decode mask to the kernel's QK-swap layout when
    ``attention_block_tkg`` will select that path for the mask's shape.

    Input is the legacy ``[S_ctx, B, q_heads, S_tkg]`` mask produced by
    ``gen_attention_decode_mask``. When the kernel's own ``is_qk_swapped``
    predicate is True for this shape, returns the transposed
    ``[B, q_heads, S_tkg, S_ctx]`` layout the kernel validates against; otherwise
    returns the mask unchanged. On nkilib without the predicate this is a no-op.

    Shape-driven inputs (bs, q_head, s_active, s_prior) are read from the mask's
    own dims — exactly the values the kernel validates against. ``lnc=2`` and
    ``fuse_rope=False`` match how ``attention_block_tkg`` invokes the predicate
    for the decode path; ``d_head``/fp8 flags only gate the predicate's early
    returns. kwargs are filtered to the installed signature so nkilib parameter
    drift (e.g. added ``p_max``/``kv_heads``) doesn't break the call.
    """
    if (
        _nki_is_qk_swapped is None
        or attention_mask is None
        or attention_mask.dim() != 4
    ):
        return attention_mask
    s_ctx, bs, q_head, s_active = (int(d) for d in attention_mask.shape)
    kwargs = {
        "bs": bs,
        "q_head": q_head,
        "d_head": d_head,
        "s_active": s_active,
        "curr_sprior": s_ctx,
        "lnc": 2,
        "p_max": P_MAX,
        "is_block_kv": True,
        # nkilib's is_qk_swapped() KV-dtype arg is ``is_2byte_kv``: True iff the
        # raw KV element is a 2-byte type (bf16/fp16), False for 1-byte fp8. It
        # is the inverse of the caller's fp8 flag, hence ``not is_fp8_kv``.
        "is_2byte_kv": not is_fp8_kv,
        "fp8_packed": fp8_packed,
        "fuse_rope": False,
    }
    swapped = _nki_is_qk_swapped(
        **{k: v for k, v in kwargs.items() if k in _QK_SWAP_PARAMS}
    )
    if not swapped:
        return attention_mask
    # [S_ctx, B, q_heads, S_tkg] -> [B, q_heads, S_tkg, S_ctx]; contiguous so the
    # kernel's DMA sees the expected strides (mask is small vs weights).
    return attention_mask.permute(1, 2, 3, 0).contiguous()


jitted_gen_mask = nki.jit()(gen_mask_tkg_hbm)

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.utils.neuron_utils import can_run_kernel

P_MAX = 128


def gen_attention_decode_mask(
    pos_ids: Tensor,
    bs: int,
    q_head: int,
    s_active: int,
    s_prior: int,
    start_pos: Optional[Tensor] = None,
    block_len: int = 0,
    local_filled_slots: Optional[Tensor] = None,
    dcp_active_mask: Optional[Tensor] = None,
    active_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Generate an attention mask for token-generation (TKG).

    Automatically selects between the NKI ``gen_mask_tkg_hbm`` kernel and a
    PyTorch fallback based on hardware constraints and tensor placement.

    Args:
        pos_ids:      [1, bs * s_active]  Position IDs per-batchline.
        bs:           Batch size.
        q_head:       Number of query heads.
        s_active:     Active (new-token) sequence length.
        s_prior:      Prior (KV-cache) sequence length (must be divisible by 128
                      for the NKI path).
        start_pos:    Optional [1, bs * s_active]  Per-query SWA window start.
        block_len:    Block length for block-KV cache (0 = flat cache).
        local_filled_slots: Filled cache-slot count for DCP interleaved caches
                      (drives a prior mask, slots < threshold valid). Scalar = one
                      threshold for all lines; ``[bs]``/``[1, bs]`` = per-request.
        dcp_active_mask: Per-rank active-token gate for DCP (scalar or ``[bs]``),
                      broadcast the same way as ``local_filled_slots``.
        active_mask:  Optional [s_active, bs, q_head, s_active] active-token
                      mask. Defaults to the causal staircase. DFlash passes an
                      all-ones mask so every query in a proposal block can see
                      the whole block (bidirectional block diffusion).

    Returns:
        Tensor of shape [s_prior, bs, q_head, s_active] with values in {0, 1}.
    """
    if s_prior % P_MAX != 0:
        raise ValueError(f"s_prior ({s_prior}) must be divisible by {P_MAX}")

    # DCP interleaved-cache prior mask. Block-KV stores this rank's owned prior
    # tokens tile-shuffled, so gate by each slot's sequential position (block-KV
    # iota), not a flat arange; active token sits unshuffled at s_prior-1.
    if local_filled_slots is not None:
        # Build via the gen_mask_tkg_hbm kernel (SBUF → HBM), same as the non-CP
        # path: local_filled becomes pos_ids so the kernel's prior mask
        # (iota < min_pos) equals (iota < local_filled), and the per-rank owner
        # gate folds into active_mask. The torch build below is the fallback for
        # when the kernel cannot run (can_run_kernel False: CPU without the NKI
        # simulator, or kernels disabled). The NKI-simulator lane runs the kernel.
        if (
            s_active == 1
            and block_len > 0
            and _can_use_kernel(pos_ids, bs, s_active, s_prior)
        ):
            kernel_pos_ids = local_filled_slots.to(torch.float32).reshape(1, -1)
            if kernel_pos_ids.shape[1] == 1 and bs > 1:
                kernel_pos_ids = kernel_pos_ids.expand(1, bs)
            dcp_mask = torch.ones(
                s_active,
                bs,
                q_head,
                s_active,
                dtype=torch.float32,
                device=pos_ids.device,
            )
            if dcp_active_mask is not None:
                dcp_mask = dcp_mask * dcp_active_mask.to(torch.float32).reshape(
                    1, -1, 1, 1
                )
            return _gen_mask_via_kernel(
                kernel_pos_ids.contiguous(),
                bs,
                q_head,
                s_active,
                s_prior,
                None,
                block_len,
                dcp_mask.contiguous(),
            )

        if block_len > 0:
            block_len = _resize_block_len(block_len, bs, q_head, s_active, s_prior)
            seq_pos = _build_block_kv_iota(
                s_prior, block_len, pos_ids.device, torch.float32
            ).view(s_prior, 1)
        else:
            seq_pos = torch.arange(
                s_prior, device=pos_ids.device, dtype=torch.float32
            ).view(s_prior, 1)
        slot_idx = torch.arange(
            s_prior, device=pos_ids.device, dtype=torch.float32
        ).view(s_prior, 1)
        # [s_prior, 1] vs [1, bs_lf] broadcasts to a per-request threshold; a
        # scalar local_filled_slots collapses bs_lf to 1 and reproduces the
        # single-threshold mask exactly.
        lf = local_filled_slots.to(torch.float32).reshape(1, -1)
        prior_mask = (seq_pos < lf).float()
        active_slots = (slot_idx >= (s_prior - s_active)).float()
        if dcp_active_mask is not None:
            active_slots = active_slots * dcp_active_mask.to(torch.float32).reshape(
                1, -1
            )
        mask = (prior_mask + active_slots).clamp(max=1.0)
        # [s_prior, bs_lf, 1, 1] → full shape. bs_lf is 1 (broadcast) or bs.
        mask = (
            mask.view(s_prior, mask.shape[1], 1, 1)
            .expand(s_prior, bs, q_head, s_active)
            .contiguous()
        )
        return mask

    if active_mask is None:
        # Build causal mask over the two s_active dimensions (dim 0 and dim 3)
        causal = torch.triu(
            torch.ones(s_active, s_active, dtype=torch.float32, device=pos_ids.device)
        )

        # Broadcast to [s_active, bs, q_head, s_active]
        active_mask = (
            causal[:, None, None, :].expand(s_active, bs, q_head, s_active).contiguous()
        )
    else:
        expected = (s_active, bs, q_head, s_active)
        if tuple(active_mask.shape) != expected:
            raise ValueError(
                f"active_mask must have shape {expected}, "
                f"got {tuple(active_mask.shape)}"
            )
        active_mask = active_mask.to(torch.float32).contiguous()

    # SWA + s_active > 1: NKI mask kernel now supports per-token start bounds.
    if _can_use_kernel(pos_ids, bs, s_active, s_prior):
        return _gen_mask_via_kernel(
            pos_ids,
            bs,
            q_head,
            s_active,
            s_prior,
            start_pos,
            block_len,
            active_mask,
        )

    return _torch_gen_attention_decode_mask_impl(
        pos_ids,
        bs=bs,
        q_head=q_head,
        s_active=s_active,
        s_prior=s_prior,
        start_pos=start_pos,
        block_len=block_len,
        active_mask=active_mask,
    )


def _gen_mask_via_kernel(
    pos_ids: Tensor,
    bs: int,
    q_head: int,
    s_active: int,
    s_prior: int,
    start_pos: Optional[Tensor],
    block_len: int,
    active_mask: Tensor,
) -> Tensor:
    """Dispatch to the gen_mask_tkg_hbm NKI kernel (mask built in SBUF → HBM)."""
    wrapped = wrap_nki(jitted_gen_mask)
    pos_ids = pos_ids.to(torch.float32)

    lnc = 2
    if s_prior % (2 * P_MAX) != 0:
        lnc = 1

    return wrapped[lnc](
        pos_ids,
        bs=bs,
        q_head=q_head,
        s_active=s_active,
        s_prior=s_prior,
        start_pos_hbm=start_pos,
        block_len=block_len,
        active_mask=active_mask,
    )


def _can_use_kernel(
    pos_ids: Tensor,
    bs: int,
    s_active: int,
    s_prior: int,
) -> bool:
    """
    Check if the NKI kernel can be used.

    Returns False when any NKI kernel constraint is violated or the tensors
    live on the CPU without the NKI simulator enabled.
    """
    if not can_run_kernel(pos_ids):
        return False

    return True


def _resize_block_len(
    block_len: int,
    bs: int,
    q_head: int,
    s_active: int,
    s_prior: int,
) -> int:
    """Apply the same block_len resize that the NKI kernel applies internally.

    Matches ``gen_mask_tkg_hbm``: sprior_n_prgs is LNC only when
    ``is_s_prior_sharded(bs, q_head, s_active, s_prior, P_MAX)`` is True,
    otherwise 1. s_prior sharding requires s_prior >= 2*P_MAX AND batch is
    not sharded (batch sharding takes priority; it kicks in when
    bs % 2 == 0 AND (bs*q_head*s_active >= P_MAX OR s_prior <= 2*P_MAX)).

    NOTE: the threshold must match the kernel's is_batch_sharded literal
    (currently s_prior <= 2*P_MAX); if it drifts the decode mask breaks at 256.
    """
    if block_len <= 0:
        return block_len

    lnc = 2
    batch_sharded = (bs % lnc == 0) and (
        bs * q_head * s_active >= P_MAX or s_prior <= 2 * P_MAX
    )
    sprior_sharded = (not batch_sharded) and s_prior >= lnc * P_MAX
    sprior_n_prgs = lnc if sprior_sharded else 1

    num_blocks_per_batch = s_prior // block_len
    bucket_len = num_blocks_per_batch * block_len
    min_multiple = sprior_n_prgs * P_MAX
    if bucket_len % min_multiple != 0:
        return block_len

    new_block_len, _ = _resize_cache_block_len_for_attention_tkg_kernel(
        num_blocks_per_batch, block_len, sprior_n_prgs, P_MAX
    )
    return new_block_len


def _resize_cache_block_len_for_attention_tkg_kernel(
    num_blocks_per_batch: int, block_len: int, n_prgs: int, p_max: int
):
    """
    Block KV in token gen attention requires number of blocks per batch to be a multiple of (lnc * p_max).
    This allows loading p_max blocks onto SBUF partitions in parallel.
    If the block count is not divisible by (lnc * p_max), we will reduce block_len to increase num_blocks_per_batch.
    As long as the bucket_len is divisible by lnc * p_max, there is always a block_len (min. 1) that satisfies the requirement.

    Args:
      num_blocks_per_batch: Number of blocks in each batch. Generally the second dimension of the active blocks table.
      block_len: The size of each block.
      n_prgs: Sharding level.
      p_max: Maximum number of partitions.
      full_sprior: Maximum KV cache capacity (bucket size). Used for warning suggestions.

    NOTE: This function is borrowed from NKI due to them having prints which aren't dynamo safe.
    """

    bucket_len = num_blocks_per_batch * block_len
    min_multiple = n_prgs * p_max

    # Find the greatest multiple of block_len that also divides the maximum block length.
    reduced_blk_len = math.gcd(block_len, bucket_len // min_multiple)
    resize_factor = block_len // reduced_blk_len

    return reduced_blk_len, resize_factor


def _build_block_kv_iota(s_prior: int, block_len: int, device, dtype) -> Tensor:
    """
    Build a [s_prior, 1, 1, 1] iota tensor that mirrors the NKI kernel's
    block-KV index layout.

    The HBM output of the kernel is [n_tile, P_MAX, ...] which is reshaped
    to [s_prior, ...].  For a linear output index ``i``:

        tile_idx  = i // P_MAX
        partition = i  % P_MAX
        fold_idx  = tile_idx  // block_len
        blk_off   = tile_idx  %  block_len
        position  = fold_idx * P_MAX * block_len + partition * block_len + blk_off

    This matches the kernel's ``nisa.iota`` with
    ``pattern=[[1, block_len]], channel_multiplier=block_len``.
    """
    i = torch.arange(s_prior, device=device, dtype=dtype)
    tile_idx = i // P_MAX
    partition = i % P_MAX
    fold_idx = tile_idx // block_len
    blk_off = tile_idx % block_len
    iota = fold_idx * P_MAX * block_len + partition * block_len + blk_off
    return iota.view(s_prior, 1, 1, 1)


def _seq_pos_to_linear_idx(pos: int, block_len: int) -> int:
    """
    Inverse of the block-KV iota: given a sequential position, return the
    linear index in the kernel's output layout.

        fold_idx   = pos // (P_MAX * block_len)
        within     = pos %  (P_MAX * block_len)
        partition  = within // block_len
        blk_off    = within %  block_len
        tile_idx   = fold_idx * block_len + blk_off
        linear_idx = tile_idx * P_MAX + partition
    """
    fold_idx = pos // (P_MAX * block_len)
    within = pos % (P_MAX * block_len)
    partition = within // block_len
    blk_off = within % block_len
    tile_idx = fold_idx * block_len + blk_off
    return tile_idx * P_MAX + partition


def _build_seq_to_linear_map(s_prior: int, block_len: int) -> Tensor:
    """Build a [s_prior] tensor mapping sequential position → linear output index.

    ``result[seq_pos]`` gives the linear index in the kernel output where
    sequential position ``seq_pos`` is stored.
    """
    seq_positions = torch.arange(s_prior)
    fold_idx = seq_positions // (P_MAX * block_len)
    within = seq_positions % (P_MAX * block_len)
    partition = within // block_len
    blk_off = within % block_len
    tile_idx = fold_idx * block_len + blk_off
    return tile_idx * P_MAX + partition


def _torch_gen_attention_decode_mask_impl(
    pos_ids: Tensor,
    bs: int,
    q_head: int,
    s_active: int,
    s_prior: int,
    start_pos: Optional[Tensor] = None,
    block_len: int = 0,
    active_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    PyTorch reference implementation of attention mask generation for TKG.

    Produces output in the same block-KV shuffled layout as the NKI kernel
    (``gen_mask_tkg_hbm``).

    The prior mask is **uniform** across all active tokens — it uses the
    minimum pos_id (and maximum start_pos) per batch element, since the
    KV cache is only filled up to the earliest active token.  The per-token
    causal staircase among active tokens is captured entirely by
    ``active_mask``.  Even for ``s_active == 1`` the active mask is needed
    because the ``iota < pos_id`` comparison does not include the active
    token's own position (``pos == pos`` is not ``< pos``).

    Args:
        pos_ids:     [1, bs * s_active]  Position IDs (end positions, exclusive).
        bs:          Batch size.
        q_head:      Number of query heads.
        s_active:    Active (new-token) sequence length.
        s_prior:     Prior (KV-cache) sequence length.
        start_pos:   Optional [1, bs * s_active]  Per-query SWA window start
                     (inclusive).  ``None`` → standard causal mask.
        block_len:   Block length for block-KV cache.  Must be > 0.
        active_mask: [s_active, bs, q_head, s_active] active-token mask
                     loaded into the positions corresponding to the last
                     ``s_active`` sequence slots.  **Always required**.

    Returns:
        Tensor of shape [s_prior, bs, q_head, s_active] with values in {0, 1}.

    Raises:
        NotImplementedError: If ``block_len <= 0``.
        ValueError: If ``active_mask is None``.
    """
    if block_len <= 0:
        raise NotImplementedError(
            f"Block KV cache (block_len={block_len}) is not supported "
            "in the PyTorch fallback path; block_len must be > 0"
        )

    device = pos_ids.device
    dtype = torch.float32

    # Apply the same block_len resize the kernel does internally
    block_len = _resize_block_len(block_len, bs, q_head, s_active, s_prior)

    # Build the block-KV shuffled iota: [s_prior, 1, 1, 1]
    iota = _build_block_kv_iota(s_prior, block_len, device, dtype)

    pos_2d = pos_ids.view(bs, s_active)  # [bs, s_active]
    min_pos = pos_2d.min(dim=1, keepdim=True).values  # [bs, 1]
    min_pos = min_pos.unsqueeze(0).unsqueeze(2)  # [1, bs, 1, 1]

    if start_pos is None:
        # Standard causal: prior positions < min_pos are valid
        mask = (iota < min_pos).to(dtype)
    else:
        # SWA: per-token start, conservative end (min_pos).
        #
        # Each active token gets its own window start so it sees its full
        # sliding window in the prior cache. The end stays at min_pos
        # (conservative) to avoid exposing stale cache positions — active
        # K/V is at end-of-buffer, not at real positions.
        #
        # Cross-active-token attention is handled by the active overlay
        # (appended below), not by the prior mask.
        start_2d = start_pos.view(bs, s_active)  # [bs, s_active]
        per_start = start_2d.unsqueeze(0).unsqueeze(2)  # [1, bs, 1, s_active]

        ge_start = iota >= per_start  # [s_prior, bs, 1, s_active]
        lt_end = iota < min_pos  # [s_prior, bs, 1, 1] → broadcasts

        normal = ge_start & lt_end
        wrap = ge_start | lt_end

        is_wrap = per_start > min_pos  # [1, bs, 1, s_active]
        mask = torch.where(is_wrap, wrap, normal).to(dtype)

    # Broadcast across q_head: [s_prior, bs, 1, s_active] → full shape
    mask = mask.expand(s_prior, bs, q_head, s_active).contiguous()

    # Overlay active mask at the shuffled positions corresponding to the
    # last s_active sequential slots.
    for k in range(s_active):
        seq_pos = s_prior - s_active + k
        lin_idx = _seq_pos_to_linear_idx(seq_pos, block_len)
        mask[lin_idx, :, :, :] = active_mask[k, :, :, :]

    return mask
