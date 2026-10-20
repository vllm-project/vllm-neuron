# SPDX-License-Identifier: Apache-2.0
"""Shared sequence-parallel decode prologue stages for the Qwen3-32B a2a composers.

Both composers (norm + MLP in ``qwen3_32b_sp_norm_mlp.py``, norm + attention in
``qwen3_32b_sp_norm_attn.py``) trace these helpers inline — there is no
standalone entry point. Per boundary they chain::

    carrier load -> all-to-all + fp32 PE reduce + residual add -> carrier store
    (RMSNorm + quantize from the norm stage helpers)
    pack -> ONE SBUF all-gather -> unpack

The fp32 reduction runs as one matmul against a constant reducer matrix, which
folds the rank-partial sum and the ``pre_shuffle_h`` layout transpose into the
same PE pass; the residual add consumes the PSUM result directly.

Layouts
-------
``H`` is the hidden size, ``T_s`` this rank's token shard, ``T = T_s * W`` the
gathered count at tensor-parallel degree ``W``. ``P = 128`` is the partition
size, ``Q_WIDTH = 4`` the MXFP8 pack width, ``TILE = 512`` one contraction tile.

- residual carrier: bf16 HBM ``[T_s, H]``. FLAT form = the SBUF layout dumped
  linearly (flat element ``p * T_s * (H // P) + t * (H // P) + i`` is
  ``state[p, t, i]``), moved with ONE contiguous DMA per hop; NATURAL form =
  token-major, used only at the chain ends, with ``pre_shuffle_h`` folded into
  the DMA access pattern (4 strided DMAs).
- in-SBUF working state: bf16 ``[P, T_s, H // P]``, pre-shuffled order
  (``state[p, t, i]`` holds ``x[t, q * (H // 4) + t512 * P + p]`` with
  ``i = q * (H // TILE) + t512``).
- gather payload: the fp8x4 words and uint8 E8M0 scales share one uint8 buffer
  ``[P, T_s * n512 * 5]`` (4 activation bytes then 1 scale byte per element,
  ``(t512, t)`` order); the gather grows it rank-major and the unpack returns
  the hidden-major ``[P, H // TILE, T]`` pair the in-tree MLP kernel and
  nkilib's MX QKV projection consume.

Port notes (vs ``allgather_sb2sb`` and ``gpt_oss_decode_sp_residual_rms``)
--------------------------------------------------------------------------
The gather is ported from nkilib's experimental ``allgather_sb2sb`` and the
reduce stage from the all-to-all variant of ``gpt_oss_decode_sp_residual_rms``.
The deliberate changes on top of those sources:

1. One packed payload. ``allgather_sb2sb`` moves one tensor of one dtype; here
   the fp8x4 activation and its E8M0 scales share a single uint8 buffer so ONE
   all-gather carries both, and one strided read per payload undoes the
   rank-major growth.
2. Qwen3-32B geometry throughout: ``pre_shuffle_h`` H order, the reducer-matrix
   matmul that fuses the fp32 rank sum with the layout transpose, and the
   residual add on the reduced shard in the same launch.
3. Flat residual carrier. Between composers the carrier crosses HBM in the SBUF
   layout dumped linearly (one contiguous DMA per hop,
   ``sp_load/store_shuffled_flat``); the natural-order forms run only where
   torch produces or consumes the tensor (chain ends).
"""

import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl


_P = 128
_TILE = 512
_Q_WIDTH = 4


__all__ = [
    "sp_gather_replicated",
]


def sp_gather_replicated(q_shard, s_shard, t_shard: int, h: int, tp_degree: int, rg):
    """Pack + SBUF all-gather + unpack of the replicated prologue: the gather
    stage of ``sp_norm_quant_gather_replicated``, callable on its own so a
    composer can issue independent DMAs (weight prefetch) between the norm
    stage and this collective.

    Args:
        q_shard: float8_e4m3fn_x4 SBUF ``[128, H // 512, T_s]`` quantized shard.
        s_shard: uint8 SBUF ``[128, H // 512, T_s]`` E8M0 scales.
        t_shard: This rank's token shard ``T_s``.
        h: Hidden size ``H``.
        tp_degree: Tensor-parallel degree, which is the gather width.
        rg: The ``ncc.ReplicaGroup`` of the tensor-parallel mesh.

    Returns:
        ``(q_full, s_full)``: full-H SBUF tensors ``[128, H // 512, T]``.

    Example:
        >>> # traced inside a sequence-parallel composer; not called directly
    """
    t_full = t_shard * tp_degree
    n_h512 = h // _TILE

    shard_free = t_shard * n_h512

    # Stage 2: pack both payloads into ONE buffer (single dtype per all_gather; 4 activation bytes per element, 1 E8M0 scale byte).
    pack_bytes = (shard_free * _Q_WIDTH) + shard_free
    pack_words = pack_bytes // _Q_WIDTH
    gath_bytes = pack_bytes * tp_degree
    gath_words = pack_words * tp_degree
    assert pack_bytes % _Q_WIDTH == 0, (
        f"packed row of {pack_bytes} bytes must be a multiple of {_Q_WIDTH}, so the "
        f"activation region stays 4-byte aligned and the uint32 access pattern "
        f"covers a whole number of elements."
    )

    # The collective reads and writes the NATIVE uint8 tensors, never a cast.
    pack = nl.ndarray((_P, pack_bytes), dtype=nl.uint8, buffer=nl.sbuf, name="sp_pack")
    gathered = nl.ndarray(
        (_P, gath_bytes), dtype=nl.uint8, buffer=nl.sbuf, name="sp_gathered"
    )

    # Pack token-inner (t512, t) on the FREE axis (no PE transpose); reach payloads via ap(dtype=), never view.
    nisa.tensor_copy(
        dst=pack.ap([[pack_words, _P], [1, shard_free]], dtype=nl.uint32),
        src=q_shard.view(nl.uint32).reshape((_P, shard_free)),
    )
    nisa.tensor_copy(
        dst=pack.ap([[pack_bytes, _P], [1, shard_free]], offset=shard_free * _Q_WIDTH),
        src=s_shard.reshape((_P, shard_free)),
    )

    # collective_dim=1 pairs the gather with SBUF (dim 0 pairs with HBM).
    ncc.all_gather(dsts=[gathered], srcs=[pack], replica_group=rg, collective_dim=1)

    # Stage 2b: unpack; global token t = r * t_shard + j sits at rank offset r * pack_words, each pattern walking (p, t512, r, j).
    q_full = nl.ndarray(
        (_P, n_h512, t_full),
        dtype=nl.float8_e4m3fn_x4,
        buffer=nl.sbuf,
        name="sp_q_full",
    )
    s_full = nl.ndarray(
        (_P, n_h512, t_full), dtype=nl.uint8, buffer=nl.sbuf, name="sp_s_full"
    )
    nisa.tensor_copy(
        dst=q_full.view(nl.uint32).reshape((_P, n_h512, tp_degree, t_shard)),
        src=gathered.ap(
            [
                [gath_words, _P],
                [t_shard, n_h512],
                [pack_words, tp_degree],
                [1, t_shard],
            ],
            dtype=nl.uint32,
        ),
    )
    nisa.tensor_copy(
        dst=s_full.reshape((_P, n_h512, tp_degree, t_shard)),
        src=gathered.ap(
            [
                [gath_bytes, _P],
                [t_shard, n_h512],
                [pack_bytes, tp_degree],
                [1, t_shard],
            ],
            offset=shard_free * _Q_WIDTH,
        ),
    )
    return q_full, s_full


def _make_cross_partition_reducer(p_in: int, rows_out: int, dtype):
    """Build the tiled-identity reduce/transpose matrix in SBUF: column j is 1
    on every partition p with ``p % rows_out == j``, so as the moving operand
    of ``nc_matmul`` it reduces the ``p_in`` all-to-all input partitions into
    ``rows_out`` token rows while the same matmul transposes the hidden chunk
    onto the partitions. Ported from the ``gpt_oss_decode_sp_residual_rms``
    reference: the residue is generated on the free axis with
    iota and moved onto the partitions with DVE stream transposes, because
    tensor_scalar has no mod and iota is affine in the partition index.

    Args:
        p_in: All-to-all input partitions (``tp_degree * T_s``).
        rows_out: Token rows kept per rank (``T_s``).
        dtype: Reducer dtype (matches the partials).

    Returns:
        ``[p_in, rows_out]`` SBUF tensor.

    Example:
        >>> # reducer = _make_cross_partition_reducer(64, 8, nl.bfloat16)
    """
    n_groups = -(-p_in // rows_out)
    n_res = n_groups * rows_out
    tile = 32

    resid_free = nl.ndarray((1, n_res), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(
        dst=resid_free,
        pattern=[[0, n_groups], [1, rows_out]],
        offset=0,
        channel_multiplier=0,
    )

    resid = nl.ndarray((p_in, 1), dtype=nl.float32, buffer=nl.sbuf)
    for base in range(0, p_in, tile):
        hi = base + tile
        if hi > p_in:
            hi = p_in
        nisa.nc_transpose(
            dst=resid[base:hi, :],
            data=resid_free[:, base:hi],
            engine=nisa.engine.vector,
        )

    cols = nl.ndarray((p_in, rows_out), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=cols, pattern=[[1, rows_out]], offset=0, channel_multiplier=0)
    reducer = nl.ndarray((p_in, rows_out), dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=reducer, data=cols, op0=nl.equal, operand0=resid)
    return reducer


def sp_a2a_reduce_residual(partials, residual, tp_degree: int, rg):
    """All-to-all the un-reduced block partials, reduce on the PE, add the
    residual, and return the state in the pre-shuffled kernel layout,
    replacing the torch reduce-scatter + residual add + ``pre_shuffle_h`` +
    transposing load of one block boundary with one in-kernel region. Ported
    from the all-to-all variant of ``gpt_oss_decode_sp_residual_rms``
    with one change: the reduce/transpose matmuls read STRIDE-4
    stationary chunks, so the transpose lands directly in the
    ``pre_shuffle_h`` layout ``state[p, tok, q * n512 + b] =
    x[tok, b * 512 + p * 4 + q]`` and the existing swizzle + ``quantize_mx``
    consume it with no extra pass; the reduction accumulates in fp32 PSUM,
    which the torch reduce-scatter path does not. Per rank at ``T = 64``,
    ``H = 5120``: one HBM all-to-all of ``[64, 5120]`` bf16, 40 ``nc_matmul``
    of ``[64, 128] x [64, T_s]``, and 8 eviction adds.

    Args:
        partials: bf16 HBM ``[T, H]`` un-reduced block partials, ``T = tp_degree * T_s``; every rank holds every token.
        residual: bf16 SBUF ``[128, T_s, H // 128]`` pre-shuffled residual (see ``sp_load_natural_shuffled``).
        tp_degree: Tensor-parallel degree.
        rg: The ``ncc.ReplicaGroup`` of the tensor-parallel mesh.

    Returns:
        bf16 SBUF ``[128, T_s, H // 128]`` state (reduced + residual), pre-shuffled.

    Example:
        >>> # traced inside the a2a composer; not called directly
    """
    t_full, h = partials.shape
    t_shard = t_full // tp_degree
    n512 = h // _TILE
    h1 = h // _P
    assert t_full <= _P, (
        f"the reducer matmul takes at most {_P} stationary partitions, got T={t_full}."
    )
    assert h % _TILE == 0, f"H={h} must be a multiple of {_TILE}."

    # SBUF-to-SBUF all_to_all is not available, so the collective uses private HBM buffers.
    a2a_src = nl.ndarray((t_full, h), dtype=partials.dtype, buffer=nl.private_hbm)
    a2a_dst = nl.ndarray((t_full, h), dtype=partials.dtype, buffer=nl.private_hbm)
    nisa.dma_copy(dst=a2a_src, src=partials)
    ncc.all_to_all(srcs=[a2a_src], dsts=[a2a_dst], replica_group=rg, collective_dim=0)

    # After the a2a, partition s * T_s + j holds rank s's partial for this rank's token j.
    data = nl.ndarray(
        (t_full, h), dtype=partials.dtype, buffer=nl.sbuf, name="a2a_data"
    )
    nisa.dma_copy(dst=data, src=a2a_dst, dge_mode=nisa.dge_mode.none)

    reducer = _make_cross_partition_reducer(t_full, t_shard, partials.dtype)

    # Shuffled position i = q * n512 + b covers natural columns b * 512 + p * 4 + q: a stride-4 stationary read at offset b * 512 + q.
    n_banks = 8
    chunks_per_bank = h1 // n_banks
    state = nl.ndarray(
        (_P, t_shard, h1), dtype=nl.bfloat16, buffer=nl.sbuf, name="a2a_state"
    )
    banks = []
    for bank in range(n_banks):
        banks.append(
            nl.ndarray(
                (_P, chunks_per_bank, t_shard),
                dtype=nl.float32,
                buffer=nl.psum,
                name=f"a2a_psum{bank}",
            )
        )
    data_flat = data.reshape((t_full, h))
    for bank in range(n_banks):
        for c in range(chunks_per_bank):
            i = bank * chunks_per_bank + c
            q = i // n512
            b = i % n512
            nisa.nc_matmul(
                dst=banks[bank][:, c, :],
                stationary=data_flat.ap([[h, t_full], [4, _P]], offset=b * _TILE + q),
                moving=reducer,
            )
        cols = slice(bank * chunks_per_bank, (bank + 1) * chunks_per_bank)
        nisa.tensor_tensor(
            dst=state[:, :, cols],
            data1=banks[bank].permute((0, 2, 1)),
            data2=residual[:, :, cols],
            op=nl.add,
        )
    return state


def sp_load_natural_shuffled(src, t: int, h: int, name: str):
    """Load a NATURAL-order HBM ``[T, H]`` tensor into the pre-shuffled SBUF
    layout ``[128, T, H // 128]`` with 4 strided DMAs (one per x4 lane): the
    a2a composer's residual stream stays in natural order on the torch side,
    so the load folds ``pre_shuffle_h`` into the DMA access pattern instead of
    a torch permutation.

    Args:
        src: HBM ``[T, H]`` tensor in natural order.
        t: Token count.
        h: Hidden size.
        name: SBUF tensor name.

    Returns:
        SBUF ``[128, T, H // 128]`` in the pre-shuffled layout.

    Example:
        >>> # traced inside the a2a composer; not called directly
    """
    n512 = h // _TILE
    h1 = h // _P
    out = nl.ndarray((_P, t, h1), dtype=src.dtype, buffer=nl.sbuf, name=name)
    for q in range(_Q_WIDTH):
        nisa.dma_copy(
            dst=out.ap([[t * h1, _P], [h1, t], [1, n512]], offset=q * n512),
            src=src.ap([[_Q_WIDTH, _P], [h, t], [_TILE, n512]], offset=q),
        )
    return out


def sp_load_shuffled_flat(src, t: int, h: int, name: str):
    """Load a FLAT pre-shuffled HBM ``[T, H]`` carrier into the SBUF layout
    ``[128, T, H // 128]`` with ONE contiguous DMA. The carrier is
    partition-major (flat element ``p * T * (H // 128) + tok * (H // 128) + i``
    is ``state[p, tok, i]``); a composer that receives the residual from
    another composer uses this load instead of ``sp_load_natural_shuffled``,
    moving ``T * H // 128`` contiguous elements per partition at bandwidth
    rate instead of the 2-byte element rate of the strided form.

    Args:
        src: HBM ``[T, H]`` carrier in the flat pre-shuffled layout.
        t: Token count.
        h: Hidden size.
        name: SBUF tensor name.

    Returns:
        SBUF ``[128, T, H // 128]`` in the pre-shuffled layout.

    Example:
        >>> # traced inside the a2a composer; not called directly
    """
    h1 = h // _P
    out = nl.ndarray((_P, t, h1), dtype=src.dtype, buffer=nl.sbuf, name=name)
    nisa.dma_copy(
        dst=out.ap([[t * h1, _P], [1, t * h1]], offset=0),
        src=src.ap([[t * h1, _P], [1, t * h1]], offset=0),
    )
    return out


def sp_store_shuffled_flat(state, dst, t: int, h: int):
    """Store a pre-shuffled SBUF ``[128, T, H // 128]`` tensor into a FLAT
    HBM ``[T, H]`` carrier with ONE contiguous DMA — the reverse of
    ``sp_load_shuffled_flat``. The carrier crosses the torch boundary as an
    opaque ``[T, H]`` bf16 tensor only the next composer reads; use this
    between composers, and ``sp_store_shuffled_natural`` only where torch
    consumes the residual (the last layer).

    Args:
        state: SBUF ``[128, T, H // 128]`` in the pre-shuffled layout.
        dst: HBM ``[T, H]`` carrier, flat pre-shuffled layout.
        t: Token count.
        h: Hidden size.

    Example:
        >>> # traced inside the a2a composer; not called directly
    """
    h1 = h // _P
    nisa.dma_copy(
        dst=dst.ap([[t * h1, _P], [1, t * h1]], offset=0),
        src=state.ap([[t * h1, _P], [1, t * h1]], offset=0),
    )


def sp_store_shuffled_natural(state, dst, t: int, h: int):
    """Store a pre-shuffled SBUF ``[128, T, H // 128]`` tensor into a
    NATURAL-order HBM ``[T, H]`` tensor with 4 strided DMAs — the reverse of
    ``sp_load_natural_shuffled``; the a2a composer returns the updated
    residual stream to torch in natural order.

    Args:
        state: SBUF ``[128, T, H // 128]`` in the pre-shuffled layout.
        dst: HBM ``[T, H]`` tensor, natural order.
        t: Token count.
        h: Hidden size.

    Example:
        >>> # traced inside the a2a composer; not called directly
    """
    n512 = h // _TILE
    h1 = h // _P
    for q in range(_Q_WIDTH):
        nisa.dma_copy(
            dst=dst.ap([[_Q_WIDTH, _P], [h, t], [_TILE, n512]], offset=q),
            src=state.ap([[t * h1, _P], [h1, t], [1, n512]], offset=q * n512),
        )
