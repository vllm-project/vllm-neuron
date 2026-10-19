# SPDX-License-Identifier: Apache-2.0
"""Qwen3-32B MXFP8 decode (single-token-generation) MLP NKI kernel.

This kernel runs the Qwen3-32B decode MLP with hardware MXFP8 quantization. It
fuses the gate projection, the SiLU activation, the elementwise multiply, and the
down projection into one kernel::

    out = silu(x @ Wg.T) * (x @ Wu.T) @ Wd.T

It shards the work across the cores of an LNC group (one core at LNC1, two at
LNC2 -- see the sharding paragraph below). The tensor layouts and shapes are
specialized for Qwen3-32B MXFP8 decode (``H = 5120`` and per-rank
``I = 3200/1600/800`` at TP8/16/32); the layout contract itself is general, but
the constants and weight/scale packing assume this model.

The projection weights are pre-quantized to MXFP8 and stream from HBM. The kernel
prefetches the weight tiles into a rotating ring of SBUF buffers up front, so the
DMA of each tile overlaps the matmuls that read earlier tiles (DMA engine in
parallel with the PE engine). It also prefetches the first down-projection weight
tiles during the gate/up projections, so those loads hide under the gate/up
compute. Set ``weight_ring_depth`` to ``0`` or ``1`` to disable streaming and load
each full weight once.

The kernel takes a pre-normalized activation by default; the caller applies the
``post_attention_layernorm`` RMSNorm before the call. Pass ``gamma`` to fuse that
RMSNorm into the kernel instead: the kernel then reads the raw activation and
normalizes over the full H dimension before quantization.

The fused norm serves the decode shape family only. The decoder layer routes a
token count above ``TKG_BS_SEQLEN_THRESHOLD`` (96) to the prefill CTE kernel,
which has no norm stage, so this path sees ``T <= 96``.

The kernel shards the MLP across the LNC cores (``num_shards = nl.num_programs``).
The core count must divide ``H // TILE`` exactly; the kernel asserts this, because
an inexact split silently drops H elements from the gate and up projections.

At 2 cores with 2 or more I 512-tiles, gate/up shard the **I (output)** axis: each
core contracts the full H over its own column slab, so the slab is complete and
nothing is reduced, and one ``sendrecv`` PLACES the peer's quantized slab
(PORT CHANGE #10). Otherwise gate/up shard the H contraction and a cross-core
``sendrecv`` + add combines the partials before SiLU.

down is the same in both modes: each core contracts the full I and writes a
disjoint H slice, so it needs no reduce. At grid 1 the kernel reduces to a single
core. With the norm fused every core reduces over the full H.

Layouts
-------
``H`` is the hidden size, ``I`` the (per-shard) intermediate size, ``T`` the
token count. ``P = 128`` is the partition size and ``Q_WIDTH = 4`` is the MXFP8
pack width, so ``TILE = 512`` is one contraction tile.

- ``hidden``: bf16 ``[P, H // TILE, T_pad, Q_WIDTH]`` in the H-partitioned
  x4 decode layout. ``hidden[p, t512, t, q]`` holds the logical activation
  ``x[t, t512 * TILE + p * Q_WIDTH + q]``. ``T_pad`` is ``T`` rounded up to a
  multiple of ``Q_WIDTH``. This is the same H order the gate/up weights use, so
  the matmul contracts matching elements.
- ``hidden_q`` / ``hidden_s``: the optional pre-quantized form of ``hidden``.
  ``hidden_q`` is float8_e4m3fn_x4 SBUF ``[P, H // TILE, T]`` and ``hidden_s`` is
  uint8 SBUF ``[P, H // TILE, T]``. They hold the MXFP8 activation and its E8M0
  block scale in the same H order as ``hidden``. A caller in the same kernel
  gives this pair in place of ``hidden``. ``T`` must be a multiple of
  ``Q_WIDTH``.
- ``gate_w`` / ``up_w``: uint32 ``[P, H // TILE, I]`` MXFP8 weights (4 fp8 bytes
  packed per uint32 along the H contraction). Reinterpreted to
  ``float8_e4m3fn_x4`` in the kernel. I is the free axis, and the matmul reads it
  one 128-wide lane at a time, so I pads to a multiple of ``P``, not of ``TILE``.
- ``gate_s`` / ``up_s``: uint8 ``[P, H // TILE, I]`` E8M0 block scales (one
  scale per 32 contraction elements = 8 partitions x Q_WIDTH). The host
  pre-scatters the 16 real scale rows into the first 4 partitions of each
  32-partition quadrant, so the kernel loads all partitions with one DMA.
- ``down_w``: uint32 ``[P, ceil(I / TILE), H]`` MXFP8 weights (4 fp8 bytes packed
  per uint32 along the I contraction). I is the contraction axis here, so it pads
  to a whole ``TILE``. The gate/up I extent can therefore be shorter than
  ``ceil(I / TILE) * TILE``, and the last 512-tile then holds 1 to 3 valid lanes.
  The kernel zeroes the other lanes of the intermediate.
- ``down_s``: uint8 ``[P, I // TILE, H]`` E8M0 block scales (same host
  pre-scatter as the gate/up scale).
- output: bf16 ``[T, H]``.

Port notes (vs nkilib ``mlp_tkg_quad_fp8_mx``)
----------------------------------------------
This kernel is ported from the nkilib MX decode MLP. The nkilib source already
had weight streaming, cross-projection down prefetch, and LNC H-sharding. The
deliberate logic changes here are marked inline with ``PORT CHANGE`` comments:

1. Decode-only scope. Dropped bias, clamp, fused-add, dequant-scale, and the
   B*S tiling. The RMSNorm stage is kept, but it is optional and reshaped:
   - It runs only when the caller passes ``gamma``. Without ``gamma`` the caller
     normalizes and the kernel starts at the quantize.
   - The fused path serves decode shapes only, so ``T <= 96`` (the decoder layer
     sends a larger token count to the CTE kernel, which has no norm stage).
   - nkilib keeps the H reduction on the free axis of an ``[H0, T, H1]`` tile and
     reduces it with one ``tensor_reduce``. This kernel holds the x4 decode
     layout ``[P, H // TILE, T_pad, Q_WIDTH]``, where H spans a partition axis
     and two free axes, so it reduces the ``Q_WIDTH`` lane, then folds the
     ``H // TILE`` blocks with a pairwise tree, then reduces the partitions with
     the same all-ones matmul nkilib uses. ``tensor_reduce`` cannot take the
     block axis directly: it only reduces trailing dimensions.
   - Under LNC sharding nkilib shards the norm input as well and exchanges
     partial sums. This kernel instead loads the full H on every core, so the
     norm needs no collective; each core emits only its own H shard. That trades
     a duplicated activation DMA for one less cross-core reduce.
   - nkilib also has a TH variant for large T. Out of scope here: the fused path
     is capped at 96 tokens.
2. Scale layout. The host pre-scatters the 16-row E8M0 scale into 128
   partitions, so the kernel loads it with one DMA. nkilib scatters in-kernel
   with per-quadrant DMAs. See ``_load_weight_scale``.
3. Gate/up weight ring. Prefetch all ring tiles up front (the scheduler does
   not hoist an in-loop DMA). nkilib DMAs each tile inside the matmul loop.
4. Inline LNC sharding read from the launch grid (nkilib uses a constants
   helper). Same H-shard; grid 1 collapses to a single core.
5. Hoist the down-scale load ahead of gate/up so its DMA overlaps their
   matmuls (nkilib loads it inside the down projection).
6. Stream the down weight in shape-derived chunks near 14 KiB per DMA (nkilib
   streams one H128 tile per DMA, too small to hide).
7. Combine the gate and up cross-core reduces into one ``sendrecv`` + add
   (nkilib reduces each projection separately).
8. Swap the down-projection operands: the intermediate is stationary and the
   weight is moving, so PSUM holds ``[T, H]`` and the ``nc_transpose`` tail
   disappears (nkilib keeps the weight stationary and transposes).
9. Fold the weight scales. The MX scale start offset takes one of {0, 4, 8, 12},
   so a factor of 4 can leave the free axis and enter the partition axis. gate/up
   fold the sub-lane index and down folds the tile index, which cuts the scale
   bytes and the scale SBUF by about 3.6x. The kernel selects this layout from the
   scale's partition count, so the pre-scattered layout still works.

   Each folded scale slice must reach the LAST partition. A short partition
   extent compiles clean AND passes the NKI CPU simulator, because the simulator
   does not model the scale start offset (NKI-1467). Only a device run catches
   it, and it gives about 97% relative error there. The device accuracy target in
   ci/test-manifest-core.yaml is the gate.
10. Shard gate/up on the I (output) axis, not the H contraction, and exchange the
   quantized intermediate. This deletes the cross-core bf16 sum-reduce, halves the
   SiLU / multiply / quantize width, and cuts the exchange payload 5.6x (917,504 ->
   163,840 B at TP8 LNC2). down is unchanged. The exchange sits between this core's
   own down matmuls and the peer's, so the matmuls cover the rendezvous.
11. Optional pre-quantized activation. The ``hidden_q`` / ``hidden_s`` pair lets a
   caller in the same kernel give an MXFP8 activation that it quantized itself.
   The kernel then skips the ``hidden`` HBM load and its own quantize step.
   nkilib always quantizes the activation inside the MLP kernel.
"""

from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

# Tiling constants. The partition size comes from nl.tile_size so the kernel
# tracks the hardware (nl.tile_size.pmax is a plain int, safe to read at import).
# The rest are MXFP8 format constants (Q_WIDTH, MX group) or are derived; they do
# not vary with the hardware. _PSUM_BMAX mirrors nl.tile_size.psum_num_banks,
# which is a backend-gated property (not readable at import), so keep it here.
_P = nl.tile_size.pmax  # partition size (128)
_Q_WIDTH = 4  # fp8 bytes packed per x4 element (MXFP8 format)
_TILE = _P * _Q_WIDTH  # 512, one contraction tile
_Q_HEIGHT = 8  # partitions per MX scale block
_MX_GROUP = _Q_HEIGHT * _Q_WIDTH  # 32 elements share one E8M0 scale
_SCALES_PER_TILE = _TILE // _MX_GROUP  # 16 scale rows per contraction tile
_SBUF_QUADRANT = 32  # partitions per SBUF quadrant (scale scatter step)
_N_QUADRANTS = _P // _SBUF_QUADRANT  # 4
_ROWS_PER_QUADRANT = _SCALES_PER_TILE // _N_QUADRANTS  # 4
# The MX scale start offset must satisfy ``pidx % 4 == 0 and pidx < 16``, so it
# takes one of {0, 4, 8, 12}: exactly 4 slots. That is why exactly a factor of 4
# may leave the scale's free axis and enter its partition axis.
_SCALE_FOLD_SLOTS = 4
_FOLD_ROWS_PER_QUADRANT = _SCALE_FOLD_SLOTS * _ROWS_PER_QUADRANT  # 16
# Partition extent of a folded (pack64) scale in HBM. The kernel reads this from
# the tensor to select the folded path, so the host layout and the kernel cannot
# disagree silently.
_SCALE_FOLD_ROWS = _N_QUADRANTS * _FOLD_ROWS_PER_QUADRANT  # 64
_PSUM_BMAX = 8  # PSUM bank count (== nl.tile_size.psum_num_banks)
# Widest bf16 free extent of one PSUM bank, in elements. A PSUM bank holds 2 KiB
# per partition. PORT CHANGE #8 makes the down PSUM tile [t_pad, w_cols], so
# w_cols is now bounded by the bank and not only by the DMA chunk size. Keep the
# matmul output group at or under this, and keep it a multiple of _P so a group
# boundary never splits a 128-lane output tile.
_PSUM_BANK_BF16_MAX = 1024
# Weight-streaming ring depth for gate/up. The kernel streams each H512 weight
# tile into a slot of a rotating SBUF ring. Slot s is reused every ``depth``
# tiles. When the depth is less than the tile count, the reuse DMA must wait for
# the matmul that still reads slot s (a write-after-read hazard), so its weight
# lands late and the matmuls stall. A decode profile showed exactly this: at
# grid 2 the gate/up shard has n_h512 = 5 tiles but the depth was 4, so tile 4
# reused slot 0 and the gate matmuls stalled ~6.6 us waiting for its late DMA.
#
# Set the depth to 5 so every gate/up tile gets its own buffer (the kernel
# clamps to ``min(depth, n_h512)``), so all 5 weight DMAs prefetch with no reuse
# hazard. 5 is the maximum useful value at grid 2 (one buffer per tile); a larger
# value clamps back to 5. One gate/up ring slot is I*4 bytes ~= 14 KiB, so the
# depth-5 ring is ~70 KiB; with the down chunk ring (~43 KiB) and the scales
# (~52 KiB) the peak stays near 165 KiB, under the partition budget.
# TODO(reusability): the ring depths are hardcoded for the Qwen3-32B shapes. To
# reuse this kernel for other shapes, compute them from the SBUF free space (see
# the QKV-CTE kernel), with a safety margin. Pairs with the SbufManager TODO.
_RING_DEPTH_DEFAULT = 5
# PORT CHANGE (vs nkilib) #6: nkilib streams one H128 down tile per DMA. That
# per-tile DMA is small (n_i512 * 128 * 4 bytes ~= 3.5 KiB) and the tiny decode
# matmuls cannot hide its latency, which makes the down projection sparse. Group
# enough H128 output tiles into one DMA to keep the transfer near the gate/up
# per-H512 chunk size (~14 KiB), so DMA latency amortizes over more matmuls.
# ``n_i512`` varies with the tensor-parallel shard: the original fixed four tiles
# makes 14 KiB at TP8 (seven I512 tiles), but only 8 KiB / 4 KiB at TP16 / TP32.
# All chunks are prefetched up front (one buffer per chunk), so the down weight is
# fully resident and their DMAs overlap the gate/up matmuls.
_DOWN_CHUNK_TARGET_BYTES = 14 * 1024


def _div_ceil(a: int, b: int) -> int:
    """Return the ceiling of ``a / b`` for positive integers.

    Local copy of ``nkilib.core.utils.kernel_helpers.div_ceil``; kept here so the
    in-tree kernel does not import nkilib.

    Args:
        a: Dividend.
        b: Divisor.

    Returns:
        The smallest integer ``>= a / b``.

    Example:
        >>> _div_ceil(3200, 512)
        7
    """
    return (a + b - 1) // b


def _i_lanes_in_tile(i_size: int, i_tile: int) -> int:
    """Return the valid 128-wide I lanes of one 512-tile.

    The gate/up weight pads its I axis to a multiple of 128, not 512, because the
    matmul consumes I 128 columns at a time. The last tile can therefore hold 1,
    2, 3 or 4 valid lanes. The caller must skip the other lanes and must zero
    them, because the 4 lanes of one x4 element share one E8M0 scale.

    Args:
        i_size: Padded I extent of the gate/up weight (a multiple of 128).
        i_tile: 512-tile index.

    Returns:
        Valid lane count in ``[1, 4]``.

    Example:
        >>> [_i_lanes_in_tile(3200, t) for t in range(7)]  # TP8
        [4, 4, 4, 4, 4, 4, 1]
    """
    remaining = i_size - i_tile * _TILE
    if remaining >= _TILE:
        return _Q_WIDTH
    return remaining // _P


def _down_chunk_tiles(n_i512: int) -> int:
    """Return H128 tiles per down-weight DMA for a near-14-KiB transfer.

    One H128 tile contains ``n_i512 * 128`` x4 elements, or
    ``n_i512 * 128 * 4`` bytes. Keeping the transfer near the TP8-proven 14 KiB
    preserves the latency-amortization rationale as the per-rank intermediate
    width changes. The caller trims the last chunk to the columns that remain in
    its H shard, so a chunk is never larger than the shard.

    Args:
        n_i512: Padded 512-wide intermediate tile count.

    Returns:
        Positive number of H128 output tiles in one DMA.

    Example:
        >>> _down_chunk_tiles(7)  # TP8: 7 * 128 * 4 bytes per H128 tile
        4
    """
    return _div_ceil(_DOWN_CHUNK_TARGET_BYTES, n_i512 * _P * _Q_WIDTH)


def _down_mm_group_cols(chunk_cols: int) -> int:
    """Return the down matmul output group width, in H columns.

    One group is one PSUM tile of shape ``[t_pad, w_cols]`` (PORT CHANGE #8), so
    two bounds apply. One bf16 PSUM bank holds ``_PSUM_BANK_BF16_MAX`` elements
    per partition. And a group stays inside one DMA chunk, because that chunk
    holds the weight the matmul reads. Take the widest legal value, and keep it a
    multiple of ``_P`` so a group boundary never splits a 128-lane output tile.

    Widest is best on both counts. A TKG matmul costs a fixed issue overhead plus
    its moving operand, so fewer and longer matmuls cost less for the same MAC
    count. The widest group also gives the FEWEST groups, so it gives the caller
    the best chance that ``n_down_groups <= _PSUM_BMAX`` holds and the
    exchange-cover order stays available (PORT CHANGE #10). No narrower width can
    recover that property.

    This replaces a 512 constant that was tuned on the Qwen3-32B TP8 shape. TP8
    gives ``chunk_cols == 512``, so the function still returns 512 there. TP16 and
    TP32 have a smaller per-rank intermediate width, so ``_down_chunk_tiles``
    gives them wider chunks, and they now get a wider group with no retune.

    Args:
        chunk_cols: H columns in one down-weight DMA chunk. A positive multiple
            of ``_P``.

    Returns:
        Group width in H columns: a positive multiple of ``_P``, at or under one
        bf16 PSUM bank.

    Example:
        >>> _down_mm_group_cols(512)  # TP8: the DMA chunk is the binding bound
        512
        >>> _down_mm_group_cols(1792)  # TP32: the PSUM bank is the binding bound
        1024
    """
    cols = _PSUM_BANK_BF16_MAX
    if chunk_cols < cols:
        cols = chunk_cols
    cols = (cols // _P) * _P
    assert cols > 0, (
        f"down matmul group width is {cols} for a chunk of {chunk_cols} columns. "
        f"The chunk must hold at least {_P} columns."
    )
    return cols


def _load_weight_scale(weight_scale, n_tile: int, width: int, name: str):
    """Load a pre-scattered MXFP8 weight scale into SBUF with one full DMA.

    The host pre-scatters the E8M0 scale into the 128-partition layout the MX
    matmul reads. The 16 real rows sit at the first 4 partitions of each
    32-partition quadrant (see ``_scatter_scale`` in the torch reference). So the
    kernel loads all 128 partitions with one DMA. This removes the per-quadrant
    scatter DMA that wrote only 4 of every 32 partitions. That narrow-partition
    write skewed the load across the LNC2 cores.

    TODO(scale-packing): the MX matmul can take several block scales packed into
    one operand instead of one scale per matmul. Packing the scales here would cut
    the number of scale reads / matmul issues. See the NKI mxfp-matmul deep dive:
    https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/deep-dives/mxfp-matmul.html#packing-scale-values

    Args:
        weight_scale: uint8 HBM ``[128, n_tile, width]`` pre-scattered E8M0 scale.
        n_tile: Number of contraction tiles.
        width: Output width (``I`` for gate/up, ``H`` for down).
        name: SBUF tensor name.

    Returns:
        uint8 SBUF ``[128, n_tile, width]`` scale in the matmul partition layout.

    Example:
        >>> # called inside the kernel; not a standalone doctest
    """
    # PORT CHANGE (vs nkilib) #2: nkilib scatters the 16-row scale into 128
    # partitions in-kernel with per-quadrant DMAs. The host pre-scatters here, so
    # load the whole [128, ...] scale with one DMA.
    scale_sb = nl.ndarray(
        (_P, n_tile, width), dtype=nl.uint8, buffer=nl.sbuf, name=name
    )
    nisa.dma_copy(dst=scale_sb, src=weight_scale, dge_mode=nisa.dge_mode.hwdge)
    return scale_sb


def _load_folded_weight_scale(weight_scale, n_tile: int, width: int, name: str):
    """Load a pack64-folded MXFP8 weight scale into the matmul partition layout.

    PORT CHANGE (vs nkilib) #9. The host folds a factor of 4 out of the free axis
    and into the partition axis, so the HBM tensor holds 64 real rows in place of
    128 rows of which 112 are host-written zero. Quadrant ``q`` owns HBM rows
    ``[16q, 16q+16)`` and SBUF partitions ``[32q, 32q+16)``, so one DMA for each
    quadrant moves 16 contiguous partitions.

    A 16-partition destination is the point. nkilib's in-kernel scatter wrote 4 of
    every 32 partitions, and that narrow write is what PORT CHANGE #2 replaced with
    a host pre-scatter. The fold makes the write 4x wider, so the per-quadrant DMA
    is no longer narrow and the pre-scatter's 8x HBM inflation is unnecessary.

    Partitions ``[32q+16, 32q+32)`` stay unwritten. No matmul reads them: from a
    start offset of ``4 * slot`` the ISA reads partitions
    ``4 * slot + 32k + {0, 1, 2, 3}``, and ``slot <= 3``, so every read lands
    inside ``[32q, 32q+16)``.

    Args:
        weight_scale: uint8 HBM ``[64, n_tile, width]`` folded E8M0 scale.
        n_tile: Number of tiles on the middle axis.
        width: Free extent (``n_i512 * 128`` for gate/up, ``H`` for down).
        name: SBUF tensor name.

    Returns:
        uint8 SBUF ``[128, n_tile, width]`` scale in the matmul partition layout.

    Example:
        >>> # called inside the kernel; not a standalone doctest
    """
    scale_sb = nl.ndarray(
        (_P, n_tile, width), dtype=nl.uint8, buffer=nl.sbuf, name=name
    )
    for q in range(_N_QUADRANTS):
        dst0 = q * _SBUF_QUADRANT
        src0 = q * _FOLD_ROWS_PER_QUADRANT
        nisa.dma_copy(
            dst=scale_sb[dst0 : dst0 + _FOLD_ROWS_PER_QUADRANT],
            src=weight_scale.slice(0, src0, src0 + _FOLD_ROWS_PER_QUADRANT),
            dge_mode=nisa.dge_mode.hwdge,
        )
    return scale_sb


def _exchange_intermediate_slots(xchg, shard_id: int):
    """Place the peer core's intermediate slot beside this core's slot.

    Each core sends its own slot into the peer's slot of the same number, so both
    cores end up holding slot ``c`` = core ``c``'s slab. A placement, not a
    reduction: the I-shard makes each slab complete over the full H.

    Args:
        xchg: uint32 SBUF ``[128, num_shards, slot_words]`` exchange buffer.
        shard_id: This core's rank in the LNC group (0 or 1).

    Example:
        >>> # called inside the kernel; not a standalone doctest
    """
    nisa.sendrecv(
        src=xchg[:, shard_id],
        dst=xchg[:, 1 - shard_id],
        send_to_rank=(1 - shard_id),
        recv_from_rank=(1 - shard_id),
        pipe_id=0,
    )


def _interleave_copy(dst, src, tile_idx: int):
    """Copy ``src`` to ``dst``, alternating the engine by ``tile_idx`` parity.

    In-tree analog of NKL's ``interleave_copy``: even tiles use the Scalar engine
    (``nisa.activation`` copy) and odd tiles use the Vector engine
    (``nisa.tensor_copy``), so a run of per-tile copies spreads across both engines
    and overlaps instead of serializing on one.

    Args:
        dst: Destination SBUF tensor for this tile.
        src: Source tensor (SBUF or PSUM).
        tile_idx: Tile index; its parity picks the engine.

    Example:
        >>> # called inside the kernel; not a standalone doctest
    """
    if tile_idx % 2 == 0:
        nisa.activation(dst=dst, op=nl.copy, data=src)
    else:
        nisa.tensor_copy(dst=dst, src=src)


def _quantize_mx_3d(x3, n_tiles: int, t_pad: int, name: str):
    """Quantize a ``[128, n_tiles, t_pad, Q_WIDTH]`` bf16 tensor to MXFP8.

    ``quantize_mx`` packs every 4 free elements (the trailing Q_WIDTH axis) into
    one x4 element and emits one E8M0 scale per 32-element block.

    Args:
        x3: bf16 SBUF ``[128, n_tiles, t_pad, Q_WIDTH]`` tensor to quantize.
        n_tiles: Number of 512-tiles on the free axis.
        t_pad: Padded token count.
        name: Tensor name prefix.

    Returns:
        ``(qtz3, scale3)``: float8_e4m3fn_x4 ``[128, n_tiles, t_pad]`` and uint8
        ``[128, n_tiles, t_pad]``.

    Example:
        >>> # called inside the kernel; not a standalone doctest
    """
    x_flat = x3.reshape((_P, n_tiles * t_pad * _Q_WIDTH))
    qtz = nl.ndarray(
        (_P, n_tiles * t_pad),
        dtype=nl.float8_e4m3fn_x4,
        buffer=nl.sbuf,
        name=f"{name}_qtz",
    )
    scale = nl.ndarray(
        (_P, n_tiles * t_pad), dtype=nl.uint8, buffer=nl.sbuf, name=f"{name}_scale"
    )
    nisa.quantize_mx(dst=qtz, src=x_flat, dst_scale=scale)
    # Reshape into new names. The standalone NKI front end (the kernel-dev sim
    # harness) rejects rebinding a name to its own reshape (`x = x.reshape(...)`);
    # the @nki.jit path accepts it. Use new names to satisfy both. Verify and
    # relax once the sim harness is wired up.
    qtz3 = qtz.reshape((_P, n_tiles, t_pad))
    scale3 = scale.reshape((_P, n_tiles, t_pad))
    return qtz3, scale3


# The sequence-parallel composers issue the first ring slots per projection
# BEFORE the collectives. Two slots was the spill-free bound measured on the
# pre-rework kernel; the grid compile gates re-verify it on this one.
_SP_PREFETCH_EARLY_SLOTS = 2


def _preload_mlp_weights(
    gate_w,
    up_w,
    down_w,
    gate_s,
    up_s,
    down_s,
    weight_ring_depth: int,
):
    """Issue the MLP scale DMAs and the first gate/up ring slots NOW.

    The sequence-parallel composer calls this BEFORE its collectives, so the
    weights stream on the DGE queues while the all-to-all and the all-gather
    run. The composer is LNC1-only, so this loads the FULL tensors at grid 1:
    ``_mlp_tkg_impl`` shards nothing there (``shard_on_i`` is LNC=2-only).
    The loader helpers and the ring-slot geometry mirror the impl; a mismatch
    is silent garbage, so change both together.

    Args:
        gate_w / up_w / down_w / gate_s / up_s / down_s: The kernel weight
            and scale tensors, as ``_mlp_tkg_impl`` takes them.
        weight_ring_depth: Weight ring depth (see ``_mlp_tkg_impl``).

    Returns:
        ``(gate_scale_sb, up_scale_sb, down_scale_sb, gate_slots, up_slots)``
        for ``_mlp_tkg_impl``'s ``preloaded_weights``.

    Example:
        >>> # traced inside the sequence-parallel composers; not called directly
    """
    num_shards = nl.num_programs(axes=0)
    assert num_shards == 1, (
        f"_preload_mlp_weights supports LNC=1 only, got {num_shards} programs. "
        f"A grid-2 caller must re-add the impl's shard slicing here first."
    )
    n_h512 = gate_w.shape[1]
    i_size = gate_w.shape[2]
    h_size = down_w.shape[2]
    n_i512 = _div_ceil(i_size, _TILE)
    folded_scale = gate_s.shape[0] == _SCALE_FOLD_ROWS
    n_down_scale_buf = _div_ceil(n_i512, _SCALE_FOLD_SLOTS)

    if folded_scale:
        gate_scale_sb = _load_folded_weight_scale(gate_s, n_h512, n_i512 * _P, "gate_s")
        up_scale_sb = _load_folded_weight_scale(up_s, n_h512, n_i512 * _P, "up_s")
        down_scale_sb = _load_folded_weight_scale(
            down_s, n_down_scale_buf, h_size, "down_s"
        )
    else:
        gate_scale_sb = _load_weight_scale(gate_s, n_h512, i_size, "gate_s")
        up_scale_sb = _load_weight_scale(up_s, n_h512, i_size, "up_s")
        down_scale_sb = _load_weight_scale(down_s, n_i512, h_size, "down_s")

    # First ring slots per projection, only when the projection streams.
    # stream=False is resident mode (weight_ring_depth 0/1): the projection
    # loads the whole weight in one DMA inside the kernel, so there is no ring
    # to seed early — the scales above still preload either way. When
    # streaming, seed only _SP_PREFETCH_EARLY_SLOTS (2) ring slots, not all of
    # them: more spilled SBUF (~40 KiB/partition measured), and 2 slots plus
    # the scales already fill the ~15-25 us collective window this prefetch
    # exists to overlap; the projection streams the rest under its matmuls.
    stream = weight_ring_depth >= 2 and n_h512 > 1
    gate_slots = None
    up_slots = None
    if stream:
        _depth = weight_ring_depth
        if n_h512 < _depth:
            _depth = n_h512
        n_early = _SP_PREFETCH_EARLY_SLOTS
        if _depth < n_early:
            n_early = _depth
        gate_slots = []
        up_slots = []
        for _name, _w, _slots in (
            ("gate", gate_w, gate_slots),
            ("up", up_w, up_slots),
        ):
            for _k in range(n_early):
                _buf = nl.ndarray(
                    (_P, 1, i_size),
                    dtype=nl.uint32,
                    buffer=nl.sbuf,
                    name=f"{_name}_wring{_k}",
                )
                nisa.dma_copy(
                    dst=_buf,
                    src=_w[:, _k : _k + 1, :],
                    dge_mode=nisa.dge_mode.hwdge,
                )
                _slots.append(_buf)

    return (gate_scale_sb, up_scale_sb, down_scale_sb, gate_slots, up_slots)


def _gate_or_up_projection(
    hidden_qtz,
    hidden_scale,
    weight_u32,
    weight_scale,
    n_h512: int,
    i_size: int,
    t_pad: int,
    ring_depth: int,
    name: str,
    out=None,
    folded_scale: bool = False,
    preloaded_slots=None,
):
    """Run one gate or up projection with streamed MXFP8 weights.

    Computes ``x @ W.T`` for the MXFP8 quantized activation and weight. The
    output keeps the I dimension on the partition axis, in the x4 layout the
    caller's SiLU / multiply and the MX quantize consume before the down
    projection.

    The CALLER picks the sharding mode and passes the matching slices; this
    function only contracts what it gets. Two modes exist:

    - I-shard (PORT CHANGE #10, the mode the kernel uses at LNC2). Each core
      takes the full ``n_h512`` H tiles and its own slice of I, so its output is
      already complete. The caller reduces nothing.
    - H-shard. Each core takes its own ``n_h512`` H tiles and the full I, so its
      output is a partial full-I sum. The caller sum-reduces the gate and the up
      partials across the cores in one combined cross-core rendezvous before
      SiLU.

    Args:
        hidden_qtz: float8_e4m3fn_x4 SBUF ``[128, n_h512, t_pad]`` activation
            (this core's H tiles).
        hidden_scale: uint8 SBUF ``[128, n_h512, t_pad]`` activation scale.
        weight_u32: uint32 HBM ``[128, n_h512, i_size]`` MXFP8 weight slice.
        weight_scale: uint8 SBUF weight scale for the same slice. The layout
            follows ``folded_scale``.
        n_h512: Number of H contraction tiles to contract.
        i_size: Intermediate width of this slice (padded to a multiple of
            ``TILE``).
        t_pad: Token count padded to a multiple of ``Q_WIDTH``.
        ring_depth: Weight ring depth. ``0`` or ``1`` disables streaming.
        name: Tensor name prefix.
        out: Optional SBUF buffer to write the output into directly. When given,
            the function writes in-place and returns ``out``, so the caller avoids
            a copy. When ``None``, a new ``[128, n_i512, t_pad, Q_WIDTH]`` buffer
            is allocated and returned.
        folded_scale: Read ``weight_scale`` in the folded (pack64) layout instead
            of the pre-scattered one (PORT CHANGE #9). The fold moves a factor of
            4 off the scale's free axis and onto its partition axis, so the buffer
            holds 4 contraction tiles and the loop picks the tile with a partition
            start offset. ``False`` reads one scale buffer per tile at partition
            0. The caller selects the layout from the scale's partition count.

    Returns:
        bf16 SBUF ``[128, n_i512, t_pad, Q_WIDTH]`` projection output.

    Example:
        >>> # called inside the kernel; not a standalone doctest
    """
    assert i_size % _P == 0, (
        f"gate/up I extent ({i_size}) must be a multiple of {_P}: the matmul "
        f"consumes I one {_P}-wide lane at a time."
    )
    n_i512 = _div_ceil(i_size, _TILE)
    # One PSUM bank per I512 tile (see out_psum below); the bank count is the cap.
    assert n_i512 <= _PSUM_BMAX, (
        f"gate/up needs {n_i512} PSUM banks (one per I512 tile) but only "
        f"{_PSUM_BMAX} exist."
    )
    stream = ring_depth >= 2 and n_h512 > 1
    # The standalone NKI front end rejects a ternary / min() expression here, so
    # compute the ring depth with plain statements.
    _ring_depth = 0
    if stream:
        _ring_depth = ring_depth
        if n_h512 < _ring_depth:
            _ring_depth = n_h512

    # Non-streaming path: load the full weight once into SBUF.
    weight_sb = None
    if not stream:
        weight_sb_u32 = nl.ndarray(
            (_P, n_h512, i_size), dtype=nl.uint32, buffer=nl.sbuf
        )
        nisa.dma_copy(dst=weight_sb_u32, src=weight_u32, dge_mode=nisa.dge_mode.hwdge)
        weight_sb = weight_sb_u32.view(nl.float8_e4m3fn_x4)

    # Streaming path: a rotating ring of per-H512-tile weight buffers. Each list
    # element holds exactly one H512 weight chunk. Build the list with a plain
    # loop; the front end rejects list comprehensions.
    weight_ring = []
    if stream:
        # SP weight prefetch: the composer issued the first slots' DMAs
        # BEFORE the sequence-parallel collectives, so those buffers arrive
        # under the collective instead of after it. Reuse them as the head
        # of the ring and allocate only the remaining slots.
        _n_pre = 0
        if preloaded_slots is not None:
            for _buf in preloaded_slots:
                if _n_pre < _ring_depth:
                    weight_ring.append(_buf)
                    _n_pre = _n_pre + 1
        for _d in range(_n_pre, _ring_depth):
            weight_ring.append(
                nl.ndarray(
                    (_P, 1, i_size),
                    dtype=nl.uint32,
                    buffer=nl.sbuf,
                    name=f"{name}_wring{_d}",
                )
            )
        # PORT CHANGE (vs nkilib) #3: nkilib DMAs each weight tile inside the
        # matmul loop. Prefetch the first _ring_depth tiles up front, so the DMA
        # engine fills the ring immediately. In the decode profile the scheduler
        # did not hoist a DMA emitted inside the matmul loop, so without this the
        # tiles trickled in one at a time and the matmuls stalled ~1 us on each.
        # When _ring_depth == n_h512 (grid 2: 5 tiles), this prefetches the whole
        # weight up front.
        for _k in range(_n_pre, _ring_depth):
            nisa.dma_copy(
                dst=weight_ring[_k],
                src=weight_u32[:, _k : _k + 1, :],
                dge_mode=nisa.dge_mode.hwdge,
            )

    # Write into the caller's buffer when given (the combined gu buffer), so the
    # caller does not copy the result out. Otherwise allocate a standalone output.
    if out is None:
        out = nl.ndarray(
            (_P, n_i512, t_pad, _Q_WIDTH),
            dtype=nl.bfloat16,
            buffer=nl.sbuf,
            name=f"{name}_out",
        )

    # One PSUM bank per I512 tile, accumulated across the H contraction tiles.
    out_psum = []
    for _i in range(n_i512):
        out_psum.append(
            nl.ndarray(
                (_P, _Q_WIDTH, t_pad),
                dtype=nl.bfloat16,
                buffer=nl.psum,
                name=f"{name}_psum{_i}",
            )
        )

    for i_h512 in range(n_h512):
        if stream:
            # The first _ring_depth tiles are prefetched above. Stream the rest
            # (only when the ring wraps, i.e. _ring_depth < n_h512) into the
            # reused buffer just before its matmul.
            cur_buf = weight_ring[i_h512 % _ring_depth]
            if i_h512 >= _ring_depth:
                nisa.dma_copy(
                    dst=cur_buf,
                    src=weight_u32[:, i_h512 : i_h512 + 1, :],
                    dge_mode=nisa.dge_mode.hwdge,
                )
            cur_w = cur_buf.view(nl.float8_e4m3fn_x4)
            w_h_idx = 0
        else:
            cur_w = weight_sb
            w_h_idx = i_h512

        for i_i512 in range(n_i512):
            # The last tile can hold fewer than 4 valid lanes, because I pads to a
            # multiple of 128. Skip the lanes that hold no weight: the caller
            # zeroed them, and a matmul on them would read past the weight.
            for i_mm in range(_i_lanes_in_tile(i_size, i_i512)):
                i_off = i_i512 * _TILE + i_mm * _P
                if folded_scale:
                    # The sub-lane index moved to the partition axis, so all 4
                    # sub-blocks of this tile share one 128-wide free slice and
                    # differ only in the start offset. The slice must reach the
                    # last partition, because the ISA derives the other quadrants
                    # from the start offset.
                    p_base = _ROWS_PER_QUADRANT * i_mm  # 0, 4, 8, 12
                    w_scale_ap = weight_scale[
                        nl.ds(p_base, _P - p_base),
                        i_h512,
                        i_i512 * _P : (i_i512 + 1) * _P,
                    ]
                else:
                    w_scale_ap = weight_scale.slice(1, i_h512, i_h512 + 1).slice(
                        2, i_off, i_off + _P
                    )
                nisa.nc_matmul_mx(
                    dst=out_psum[i_i512][:, i_mm, :],
                    stationary=cur_w[:, w_h_idx, i_off : i_off + _P],
                    moving=hidden_qtz.slice(1, i_h512, i_h512 + 1).slice(2, 0, t_pad),
                    stationary_scale=w_scale_ap,
                    moving_scale=hidden_scale.slice(1, i_h512, i_h512 + 1).slice(
                        2, 0, t_pad
                    ),
                )

    # Copy each PSUM bank to the output SBUF, transposing the ``[I, Q_WIDTH, T]``
    # PSUM layout into the ``[I, T, Q_WIDTH]`` down-projection layout. Copy only
    # the lanes a matmul wrote: the rest of the bank is uninitialized, and the
    # caller already zeroed those lanes of ``out``.
    for i_i512 in range(n_i512):
        lanes = _i_lanes_in_tile(i_size, i_i512)
        nisa.tensor_copy(
            dst=out[:, i_i512, :, 0:lanes],
            src=out_psum[i_i512].ap(
                [[_Q_WIDTH * t_pad, _P], [1, t_pad], [t_pad, lanes]]
            ),
        )

    # Each core wrote its partial full-I output over its H shard into ``out``. The
    # caller sum-reduces the gate and up partials across cores in one combined
    # cross-core rendezvous (see ``_mlp_tkg_impl``).
    return out


def _rmsnorm_full(
    x_sb: nl.ndarray,
    gamma_sb: nl.ndarray,
    eps: float,
    n_h512: int,
    t_pad: int,
    h_size: int,
    h512_off: int,
    n_h512_shard: int,
    name: str,
) -> nl.ndarray:
    """Apply RMSNorm over the full hidden dimension, and return one H shard.

    The sum of squares needs every H element, so the reduction reads all of
    ``x_sb``. The gamma multiply and the output cover only
    ``[h512_off, h512_off + n_h512_shard)``, which is the H shard that this core
    projects. The caller uses the result directly as its MLP input.
    """
    sq = nl.ndarray(
        (_P, n_h512, t_pad, _Q_WIDTH),
        dtype=nl.float32,
        buffer=nl.sbuf,
        name=f"{name}_sq",
    )
    nisa.activation(dst=sq, op=nl.square, data=x_sb)
    sq_q = nl.ndarray(
        (_P, n_h512, t_pad), dtype=nl.float32, buffer=nl.sbuf, name=f"{name}_sqq"
    )
    nisa.tensor_reduce(sq_q, nl.add, sq, axis=3)

    # Multiply by gamma here, before the norm factor exists. This operation reads
    # x_sb and gamma_sb only, so it depends on nothing after the reduce above. It
    # is the largest Vector operation of the norm, and this position overlaps it
    # with the block-sum tree, the all-ones matmul on PE, and the rsqrt on the
    # Activation engine. The nkilib norm makes the same move
    # (mlp_tkg_rmsnorm.py:560-574).
    #
    # One multiply covers the whole shard. gamma broadcasts along the token axis,
    # so the shard needs no loop. A per-block loop into one shared temporary makes
    # every block wait for the previous block on a write-after-read hazard.
    xg = nl.ndarray(
        (_P, n_h512_shard, t_pad, _Q_WIDTH),
        dtype=nl.float32,
        buffer=nl.sbuf,
        name=f"{name}_xg",
    )
    x_shard = x_sb.slice(1, h512_off, h512_off + n_h512_shard)
    gamma_b = (
        gamma_sb.slice(1, h512_off, h512_off + n_h512_shard)
        .expand_dim(dim=2)
        .broadcast(dim=2, size=t_pad)
    )
    nisa.tensor_tensor(dst=xg, data1=x_shard, data2=gamma_b, op=nl.multiply)

    partial = nl.ndarray(
        (_P, t_pad), dtype=nl.float32, buffer=nl.sbuf, name=f"{name}_part"
    )
    # Sum the 512-blocks with a pairwise tree. tensor_reduce cannot take axis 1,
    # because the reduce axis must be the trailing dimension of the tile: it
    # asserts "expected axis=(2,) (or a shorter trailing subset). Got axis=(1,)".
    # A linear chain costs n_h512 - 1 dependent Vector operations. The tree needs
    # only ceil(log2(n_h512)) levels, and each level is independent.
    width = n_h512
    while width > 1:
        half = width // 2
        nisa.tensor_tensor(
            dst=sq_q.slice(1, 0, half),
            data1=sq_q.slice(1, 0, half),
            data2=sq_q.slice(1, width - half, width),
            op=nl.add,
        )
        width -= half
    nisa.tensor_copy(dst=partial, src=sq_q[:, 0, :])

    # tensor_reduce cannot reduce the partition axis. The all-ones matmul writes
    # the complete per-token sum to every partition for the following broadcast.
    ones = nl.ndarray((_P, _P), dtype=nl.float32, buffer=nl.sbuf, name=f"{name}_ones")
    nisa.memset(dst=ones, value=1.0)
    full_sum = nl.ndarray(
        (_P, t_pad), dtype=nl.float32, buffer=nl.psum, name=f"{name}_sum"
    )
    # Pass accumulate=False. This matmul is the first and the only write to this
    # PSUM location, and PSUM does not start at zero. The default accumulate=None
    # makes the compiler infer overwrite or accumulate. An inference that selects
    # accumulate adds the sum of squares onto stale PSUM data. The rsqrt then
    # gives a large error or a NaN in every batch slot at the same decode step.
    nisa.nc_matmul(stationary=ones, moving=partial, dst=full_sum, accumulate=False)

    inv = nl.ndarray((_P, t_pad), dtype=nl.float32, buffer=nl.sbuf, name=f"{name}_inv")
    nisa.activation(dst=inv, op=nl.rsqrt, data=full_sum, scale=1.0 / h_size, bias=eps)

    out = nl.ndarray(
        (_P, n_h512_shard, t_pad, _Q_WIDTH),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
        name=f"{name}_out",
    )
    inv_b = inv.expand_dim(dim=2).broadcast(dim=2, size=_Q_WIDTH)
    for i in range(n_h512_shard):
        # Each step reads and writes its own block, so the blocks stay independent.
        nisa.tensor_tensor(
            dst=out[:, i, :, :],
            data1=xg[:, i, :, :],
            data2=inv_b,
            op=nl.multiply,
        )
    return out


def _mlp_tkg_impl(
    hidden: Optional[nl.ndarray],
    gate_w: nl.ndarray,
    up_w: nl.ndarray,
    down_w: nl.ndarray,
    gate_s: nl.ndarray,
    up_s: nl.ndarray,
    down_s: nl.ndarray,
    weight_ring_depth: int,
    hidden_q: Optional[nl.ndarray] = None,
    hidden_s: Optional[nl.ndarray] = None,
    gamma: nl.ndarray = None,
    eps: float = 1e-6,
    preloaded_weights: Optional[tuple] = None,
) -> nl.ndarray:
    """Trace the Qwen3-32B MXFP8 decode MLP. See :func:`qwen3_32b_mxfp8_mlp_tkg`.

    This is the plain (undecorated) body. The ``@nki.jit`` entry point wraps it.
    ``weight_ring_depth`` is a trace-time Python int, so a separate entry point
    with a different value compiles a distinct kernel (streaming vs resident).

    The activation arrives in one of two forms. Give ``hidden`` for the bf16 form:
    the kernel loads it from HBM and quantizes it. Give ``hidden_q`` and
    ``hidden_s`` for the pre-quantized form: the kernel reads the MXFP8 tile
    directly from SBUF and does no load and no quantize. Set ``hidden`` to
    ``None`` in the second form.

    Args:
        hidden: bf16 ``[128, H // 512, T_pad, 4]`` activation. It is raw when
            ``gamma`` is set, otherwise the caller has already normalized it.
            ``None`` with the pre-quantized form.
        gate_w: uint32 ``[128, H // 512, I]`` MXFP8 gate weight.
        up_w: uint32 ``[128, H // 512, I]`` MXFP8 up weight.
        down_w: uint32 ``[128, I // 512, H]`` MXFP8 down weight (zero-padded).
        gate_s: uint8 ``[128, H // 512, I]`` pre-scattered E8M0 gate scale.
        up_s: uint8 ``[128, H // 512, I]`` pre-scattered E8M0 up scale.
        down_s: uint8 ``[128, I // 512, H]`` pre-scattered E8M0 down scale.
        weight_ring_depth: Weight ring depth. ``0`` or ``1`` disables streaming.
        hidden_q: float8_e4m3fn_x4 SBUF ``[128, H // 512, T]`` MXFP8 activation.
            Give it in place of ``hidden``. ``T`` must be a multiple of 4.
        hidden_s: uint8 SBUF ``[128, H // 512, T]`` E8M0 activation scale.
            Required with ``hidden_q``.
        gamma: Optional bf16 RMSNorm weight in the activation's x4 H layout. It is
            rejected with ``hidden_q``: a pre-quantized activation is normalized
            already, and a second norm would scale it twice.
        eps: RMSNorm epsilon when ``gamma`` is given.

    Returns:
        bf16 ``[T, H]`` MLP output in HBM.

    Example:
        >>> # traced through the @nki.jit entry point; not called directly
    """
    # PORT CHANGE (vs nkilib) #11: accept a pre-quantized activation, all-gathered
    # in SBUF by the fused norm kernel. nkilib always quantizes in the MLP kernel.
    pre_quantized = hidden_q is not None
    if pre_quantized:
        # A pre-quantized activation is normalized already. A fused norm here would
        # scale it twice, and it has no bf16 left to scale.
        assert gamma is None, (
            "gamma is not supported with hidden_q: the sequence-parallel caller "
            "normalizes the activation before it quantizes, so the fused RMSNorm "
            "of this kernel must stay off."
        )
        assert hidden_s is not None, (
            "hidden_s is required with hidden_q: the MX matmul reads the E8M0 "
            "activation scale next to the quantized activation."
        )
        assert tuple(hidden_s.shape) == tuple(hidden_q.shape), (
            f"hidden_s {tuple(hidden_s.shape)} must match hidden_q "
            f"{tuple(hidden_q.shape)}: one E8M0 scale per x4 element."
        )
        p, n_h512, t = hidden_q.shape
    else:
        assert hidden is not None, (
            "no activation: give hidden, or give hidden_q and hidden_s."
        )
        p, n_h512, t, q = hidden.shape
    h_size = n_h512 * _TILE
    i_size = gate_w.shape[2]
    # The gate/up weight pads I to a multiple of 128, so the last 512-tile can be
    # partial. The down weight keeps whole 512-tiles, and n_i512 counts those.
    n_i512 = _div_ceil(i_size, _TILE)
    h1 = h_size // _P
    t_pad = _div_ceil(t, _Q_WIDTH) * _Q_WIDTH
    # WARNING: the caller owns hidden_q/hidden_s, so this kernel zeros no pad
    # column. An uninitialized E8M0 byte is 255 (inf) and silently corrupts the tile.
    if pre_quantized:
        assert t % _Q_WIDTH == 0, (
            f"pre-quantized token count ({t}) must be a multiple of {_Q_WIDTH}. "
            f"Pad the token axis before the quantize step."
        )

    # LNC sharding. Read the grid and shard the H (hidden) dimension across the
    # cores, matching the nkilib decode MLP. gate/up shard the H contraction
    # (each core computes a partial, then a cross-core add reduces them). down
    # shards the H output (each core writes a disjoint H slice, no reduce). At
    # grid 1 this reduces to the single-core path (num_shards=1, shard_id=0). The
    # front end rejects a ternary, so use plain statements.
    # PORT CHANGE (vs nkilib) #4: derive the shard inline from the launch grid
    # (nkilib uses a constants helper). Same H-shard behavior.
    grid_ndim = nl.program_ndim()
    num_shards = 1
    shard_id = 0
    if grid_ndim != 0:
        num_shards = nl.num_programs(axes=0)
        shard_id = nl.program_id(axis=0)
    # The kernel splits the H contraction over the cores in 512-element blocks, so
    # the core count must divide the block count exactly. An inexact split
    # truncates the result without any error: n_h512_shard floors, so gate/up read
    # only num_shards * n_h512_shard blocks, while h_per_shard keeps the down
    # projection writing the full H. Two cores and H=1536 give n_h512=3 and
    # n_h512_shard=1, so gate/up drop the last 512 activation elements and the
    # output still looks complete. A zero shard is the same defect at the limit,
    # where nl.ndarray reports only "shape must be a tuple of positive integers,
    # got: (128, 0, 4, 4)". Qwen3-32B has H=5120, so it always divides by 1 or 2.
    assert n_h512 % num_shards == 0, (
        f"H={h_size} gives {n_h512} block(s) of {_TILE}, and the launch grid has "
        f"{num_shards} core(s). The core count must divide the block count "
        f"exactly, or the gate and up projections drop "
        f"{(n_h512 % num_shards) * _TILE} element(s) of H. Use an H that is a "
        f"multiple of {num_shards * _TILE}, or launch the kernel on one core."
    )
    n_h512_shard = n_h512 // num_shards
    h1_shard = h1 // num_shards
    h_per_shard = h_size // num_shards
    # Select the scale layout from the tensor the caller passed: PORT CHANGE #9
    # folds a factor of 4 off the free axis, so a folded scale has 64 partitions and
    # a pre-scattered one 128. A mismatch is silent garbage, so assert the whole
    # shape. Not a tuning knob: the host picks it from _target_supports_scale_pidx()
    # because the partition-offset scale AP is trn3/gen4 only, and this kernel also
    # serves TP16/TP32 where CI can schedule trn2. Same values, fewer bytes.
    folded_scale = gate_s.shape[0] == _SCALE_FOLD_ROWS
    gu_scale_width = n_i512 * _P
    n_down_scale_buf = _div_ceil(n_i512, _SCALE_FOLD_SLOTS)
    if folded_scale:
        assert (
            gate_s.shape[2] == gu_scale_width
            and up_s.shape[0] == _SCALE_FOLD_ROWS
            and up_s.shape[2] == gu_scale_width
        ), (
            f"A folded gate/up scale must be [{_SCALE_FOLD_ROWS}, n_h512, "
            f"{gu_scale_width}]. Got gate_s={gate_s.shape} up_s={up_s.shape}. "
            f"Rebuild the scale with fold_scale_gateup_tkg."
        )
        assert (
            down_s.shape[0] == _SCALE_FOLD_ROWS and down_s.shape[1] == n_down_scale_buf
        ), (
            f"A folded down scale must be [{_SCALE_FOLD_ROWS}, "
            f"{n_down_scale_buf}, H]. Got {down_s.shape}. Rebuild it with "
            f"fold_scale_down_tkg."
        )
    else:
        assert (
            gate_s.shape[0] == _P and up_s.shape[0] == _P and down_s.shape[0] == _P
        ), (
            f"A pre-scattered scale must have {_P} partitions. Got "
            f"gate_s={gate_s.shape} up_s={up_s.shape} down_s={down_s.shape}."
        )

    # PORT CHANGE (vs nkilib) #8 makes the intermediate the stationary operand of
    # the down projection, so its free axis is the token axis. A stationary free
    # size must not exceed the partition size. The caller enforces this already
    # (_INTREE_MAX_DECODE_TOKENS is 128 in model_mxfp8.py), and t_pad is a
    # multiple of _Q_WIDTH by construction, so this only pins the contract.
    assert t_pad <= _P, (
        f"The down projection makes the intermediate stationary, so the padded "
        f"token count must not exceed {_P}. Got t_pad={t_pad} from T={t}. Lower "
        f"the decode token count, or restore the weight-stationary down loop."
    )
    h_off = shard_id * h_per_shard
    h512_off = shard_id * n_h512_shard

    # PORT CHANGE (vs nkilib) #10: gate/up shard the I (output) axis, not the H
    # contraction, so each core owns a complete column slab. SiLU, the multiply and
    # the quantize then run on half the columns and nothing is reduced: one
    # sendrecv PLACES the peer's quantized slab. down is unchanged.
    #
    # The split must land on a 512-tile boundary. One intermediate MX group is
    # 8 partitions x 4 free elements, and out[p, i512, t, q] holds I index
    # i512*512 + q*128 + p, so a group spans all 4 lanes of one tile. Each core
    # quantizes its own slab, so a lane-boundary split would need a group across
    # both cores. Tile counts need NOT match: nl.program_id is trace-time, so the
    # kernel traces once per core.
    #
    # Give the extra tile to the LAST core -- the last tile is ragged, so 3/4 tiles
    # gives 12/13 lanes (the floor) where 4/3 gives 16/9.
    #
    # Neither half of the test below is removable. num_shards == 2: sendrecv is
    # point to point, so 1 core has no peer and the H-shard branch is its only
    # correct form. n_i512 >= num_shards: production shapes give 7/4/2 tiles at
    # TP8/16/32 so it always holds, but the reduced i_size=512 test shape needs it.
    shard_on_i = num_shards == 2 and n_i512 >= num_shards
    i512_counts = []
    i512_offs = []
    if shard_on_i:
        _i_base = n_i512 // num_shards
        _i_rem = n_i512 % num_shards
        _i_off = 0
        for _c in range(num_shards):
            _i_cnt = _i_base
            if _c >= num_shards - _i_rem:
                _i_cnt = _i_base + 1
            i512_counts.append(_i_cnt)
            i512_offs.append(_i_off)
            _i_off = _i_off + _i_cnt
    # gate/up geometry: the H contraction extent, and this core's I output slab.
    if shard_on_i:
        n_h512_gu = n_h512
        h512_off_gu = 0
        n_i512_local = i512_counts[shard_id]
        i512_off = i512_offs[shard_id]
        i_col_off = i512_off * _TILE
        # The last global tile can be partial, so clamp the slab to the real I.
        i_col_end = (i512_off + n_i512_local) * _TILE
        if i_col_end > i_size:
            i_col_end = i_size
        i_local = i_col_end - i_col_off
    else:
        n_h512_gu = n_h512_shard
        h512_off_gu = h512_off
        n_i512_local = n_i512
        i512_off = 0
        i_col_off = 0
        i_local = i_size

    # TODO(sbuf-mgr): the SBUF tensors below use plain nl.ndarray auto-allocation.
    # A follow-up should route them through nkilib's SbufManager
    # (nkilib.core.utils.allocator) with explicit stack scopes to free/reuse SBUF
    # precisely and avoid false anti-dependencies. That is an allocator rewrite
    # that must be validated on device, so it is deferred out of this PR.
    #
    # Move the activation into SBUF and pad the token axis. The fused path first
    # loads the full H so each core can reduce RMSNorm independently, then keeps
    # only this core's H shard for the MLP projections.
    # H512 is dim 1 of hidden_q/hidden_s, so the pre-quantized slice matches the
    # bf16 form (the I-shard reads the full H: h512_off_gu is 0, n_h512_gu is n_h512).
    if pre_quantized:
        inp_qtz3 = hidden_q.slice(1, h512_off_gu, h512_off_gu + n_h512_gu)
        inp_scale3 = hidden_s.slice(1, h512_off_gu, h512_off_gu + n_h512_gu)
    elif gamma is not None:
        full_sb = nl.ndarray(
            (_P, n_h512, t_pad, _Q_WIDTH), dtype=nl.bfloat16, buffer=nl.sbuf
        )
        gamma_sb = nl.ndarray((_P, n_h512, _Q_WIDTH), dtype=nl.bfloat16, buffer=nl.sbuf)
        # Issue the small gamma DMA first. Its latency then hides behind the much
        # larger activation DMA below.
        nisa.dma_copy(dst=gamma_sb, src=gamma, dge_mode=nisa.dge_mode.hwdge)
        if t_pad != t:
            nisa.memset(dst=full_sb, value=0.0)
        nisa.dma_copy(
            dst=full_sb[:, :, 0:t, :], src=hidden, dge_mode=nisa.dge_mode.hwdge
        )
        # The norm writes this core's H shard, so it is the MLP input already. A
        # copy out of a full-H buffer would move the whole tensor at one shard.
        hidden_sb = _rmsnorm_full(
            full_sb,
            gamma_sb,
            eps,
            n_h512,
            t_pad,
            h_size,
            h512_off_gu,
            n_h512_gu,
            "rmsnorm",
        )
    else:
        hidden_sb = nl.ndarray(
            (_P, n_h512_gu, t_pad, _Q_WIDTH), dtype=nl.bfloat16, buffer=nl.sbuf
        )
        if t_pad != t:
            nisa.memset(dst=hidden_sb, value=0.0)
        hidden_shard = hidden.slice(1, h512_off_gu, h512_off_gu + n_h512_gu)
        nisa.dma_copy(
            dst=hidden_sb[:, :, 0:t, :],
            src=hidden_shard,
            dge_mode=nisa.dge_mode.hwdge,
        )

    # Quantize the activation to MXFP8. The I-shard reads the full H here, so this
    # covers every H tile; the H-shard reads only this core's tiles. The
    # pre-quantized path already has the same tiles in SBUF.
    if not pre_quantized:
        inp_qtz3, inp_scale3 = _quantize_mx_3d(hidden_sb, n_h512_gu, t_pad, "inp")

    # Slice the gate/up weights and scales to this core's shard. The I-shard cuts
    # the I (output) axis and keeps the full H; the H-shard cuts the H
    # (contraction) axis and keeps the full I.
    # Build ONLY the slice the active scale layout needs. NkiTensor.slice checks
    # the bounds when it builds the slice and not when a DMA reads it, so a slice
    # of the inactive layout fails the trace: the folded free axis holds one
    # 128-wide slot for each 512-tile (n_i512 * 128 wide) and the pre-scattered
    # one holds every column (i_size wide).
    if shard_on_i:
        gate_w_shard = gate_w.slice(2, i_col_off, i_col_off + i_local)
        up_w_shard = up_w.slice(2, i_col_off, i_col_off + i_local)
        if folded_scale:
            _s0 = i512_off * _P
            _s1 = (i512_off + n_i512_local) * _P
        else:
            _s0 = i_col_off
            _s1 = i_col_off + i_local
        gate_s_shard = gate_s.slice(2, _s0, _s1)
        up_s_shard = up_s.slice(2, _s0, _s1)
    else:
        gate_w_shard = gate_w.slice(1, h512_off, h512_off + n_h512_shard)
        up_w_shard = up_w.slice(1, h512_off, h512_off + n_h512_shard)
        gate_s_shard = gate_s.slice(1, h512_off, h512_off + n_h512_shard)
        up_s_shard = up_s.slice(1, h512_off, h512_off + n_h512_shard)
    _pre_gate_slots = None
    _pre_up_slots = None
    if preloaded_weights is not None:
        # The sequence-parallel composer issued these loads BEFORE its
        # collectives (see _preload_mlp_weights); consume them here.
        (
            gate_scale_sb,
            up_scale_sb,
            _pre_down_scale_sb,
            _pre_gate_slots,
            _pre_up_slots,
        ) = preloaded_weights
    elif folded_scale:
        gate_scale_sb = _load_folded_weight_scale(
            gate_s_shard, n_h512_gu, n_i512_local * _P, "gate_s"
        )
        up_scale_sb = _load_folded_weight_scale(
            up_s_shard, n_h512_gu, n_i512_local * _P, "up_s"
        )
    else:
        gate_scale_sb = _load_weight_scale(gate_s_shard, n_h512_gu, i_local, "gate_s")
        up_scale_sb = _load_weight_scale(up_s_shard, n_h512_gu, i_local, "up_s")

    # Slice the down weight/scale to this core's H (output) shard.
    down_w_shard = down_w.slice(2, h_off, h_off + h_per_shard)
    down_s_shard = down_s.slice(2, h_off, h_off + h_per_shard)

    # PORT CHANGE (vs nkilib) #5: nkilib loads the down scale inside the down
    # projection (after gate/up). Load it now: it does not depend on the gate/up
    # result, so its DMA overlaps the gate/up matmuls instead of stalling the
    # up-to-down handoff.
    if preloaded_weights is not None:
        down_scale_sb = _pre_down_scale_sb
    elif folded_scale:
        down_scale_sb = _load_folded_weight_scale(
            down_s_shard, n_down_scale_buf, h_per_shard, "down_s"
        )
    else:
        down_scale_sb = _load_weight_scale(down_s_shard, n_i512, h_per_shard, "down_s")

    # Cross-projection prefetch: allocate the down-weight chunk ring and start
    # loading the first chunks now. They do not depend on the gate/up result, so
    # the DMAs overlap the gate/up matmuls. Each chunk holds enough H128 output
    # tiles to keep its DMA near the TP8-proven 14 KiB target, so the latency
    # hides under more matmuls at TP8, TP16, and TP32.
    down_stream = weight_ring_depth >= 2 and h1_shard > 1
    down_chunk_tiles = _down_chunk_tiles(n_i512)
    down_ring = []
    n_down_chunks = _div_ceil(h1_shard, down_chunk_tiles)
    chunk_cols = down_chunk_tiles * _P
    n_prefetched_chunks = 0
    down_ring_depth = 0
    if down_stream:
        # Prefetch the WHOLE down weight up front: one chunk buffer per output
        # chunk, so every chunk is resident and the loop never streams. Their
        # DMAs still overlap the gate/up matmuls. This raises the SBUF peak (the
        # full down shard is resident); validated on device for the Qwen3-32B TP8
        # shapes. Pairs with the ring-depth-from-SBUF TODO for reuse.
        down_ring_depth = n_down_chunks
        n_prefetched_chunks = n_down_chunks
        for _d in range(down_ring_depth):
            # Size the last chunk to the columns that remain. h_per_shard need not
            # be a multiple of chunk_cols, and a full-size last buffer holds
            # columns that no matmul reads. That wastes SBUF (1024 columns, or 8
            # KiB per partition, at TP32 and LNC2).
            _cols = chunk_cols
            if (_d + 1) * chunk_cols > h_per_shard:
                _cols = h_per_shard - _d * chunk_cols
            down_ring.append(
                nl.ndarray((_P, n_i512, _cols), dtype=nl.uint32, buffer=nl.sbuf)
            )
        for c in range(n_prefetched_chunks):
            h_src = c * chunk_cols
            w_cols = chunk_cols
            if h_src + w_cols > h_per_shard:
                w_cols = h_per_shard - h_src
            nisa.dma_copy(
                dst=down_ring[c][:, :, 0:w_cols],
                src=down_w_shard[:, :, h_src : h_src + w_cols],
                dge_mode=nisa.dge_mode.hwdge,
            )

    # Down-projection matmul groups: one group is one PSUM tile. The matmul width is
    # capped independently of the DMA chunk, because _DOWN_CHUNK_TARGET_BYTES was
    # tuned as a DMA size and gives 1792 columns at TP32 -- 3584 B per partition
    # against a 2 KiB bank. Derive the width from the bank bound and this shape's
    # chunk, so a new TP degree needs no retune.
    down_mm_group_cols = _down_mm_group_cols(chunk_cols)
    # Group list: (chunk index, H column offset in the shard, offset inside the
    # chunk, column count). Built up front because the I-shard needs the group
    # count before it decides how to order the matmuls.
    down_groups = []
    for _ic in range(n_down_chunks):
        _cs = _ic * chunk_cols
        _cw = chunk_cols
        if _cs + _cw > h_per_shard:
            _cw = h_per_shard - _cs
        for _go in range(0, _cw, down_mm_group_cols):
            _wc = down_mm_group_cols
            if _wc > _cw - _go:
                _wc = _cw - _go
            down_groups.append((_ic, _cs + _go, _go, _wc))
    n_down_groups = len(down_groups)

    # Cover the exchange: run every group's OWN I tiles, then exchange, then the
    # peer tiles. Needs one live PSUM tile per group, so it only applies when the
    # group count fits the bank count; otherwise the first peer matmul waits on the
    # whole rendezvous. At TP8 LNC2 the cover is ~6 us against a ~5.6 us exchange,
    # so the margin is thin -- measure the residual before trusting it.
    cover_exchange = shard_on_i and n_down_groups <= _PSUM_BMAX

    # gate/up write into the two halves of one buffer, so there is no per-projection
    # copy. Under the H-shard ONE rendezvous below sums the halves; under the
    # I-shard each half is already complete, so nothing is reduced.
    gu = nl.ndarray(
        (_P, 2, n_i512_local, t_pad, _Q_WIDTH),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
        name="gu",
    )
    # Zero the lanes of the last 512-tile that hold no weight -- the only
    # uninitialized part of ``gu``. Not just for the down matmul: the 4 lanes of one
    # x4 element share an E8M0 scale, so a stale lane corrupts its neighbour's
    # scale. ``i_local`` is clamped to the real I, so the test holds for both shard
    # modes (under the H-shard both cores zero the same lanes).
    lanes_last = _i_lanes_in_tile(i_local, n_i512_local - 1)
    if lanes_last < _Q_WIDTH:
        nisa.memset(dst=gu[:, :, n_i512_local - 1, :, lanes_last:], value=0.0)
    _gate_or_up_projection(
        inp_qtz3,
        inp_scale3,
        gate_w_shard,
        gate_scale_sb,
        n_h512_gu,
        i_local,
        t_pad,
        weight_ring_depth,
        "gate",
        out=gu[:, 0],
        folded_scale=folded_scale,
        preloaded_slots=_pre_gate_slots,
    )
    _gate_or_up_projection(
        inp_qtz3,
        inp_scale3,
        up_w_shard,
        up_scale_sb,
        n_h512_gu,
        i_local,
        t_pad,
        weight_ring_depth,
        "up",
        out=gu[:, 1],
        folded_scale=folded_scale,
        preloaded_slots=_pre_up_slots,
    )

    # PORT CHANGE (vs nkilib) #7: nkilib reduces gate and up separately (one
    # sendrecv each, inside the projection). Exchange the packed [gate, up] buffer
    # in a single cross-core sendrecv and add, so the two projections share one
    # rendezvous and gu[:, 0] / gu[:, 1] hold the summed gate / up in place.
    # The I-shard (PORT CHANGE #10) needs no reduce here, so it skips this.
    if num_shards > 1 and not shard_on_i:
        gu_recv = nl.ndarray(
            (_P, 2, n_i512_local, t_pad, _Q_WIDTH), dtype=nl.bfloat16, buffer=nl.sbuf
        )
        nisa.sendrecv(
            src=gu,
            dst=gu_recv,
            send_to_rank=(1 - shard_id),
            recv_from_rank=(1 - shard_id),
            pipe_id=0,
        )
        # tensor_tensor allows at most 3 free dims, so add on a flat view (the
        # view shares storage with gu, so gu[:, 0] / gu[:, 1] hold the sum).
        gu_flat = gu.reshape((_P, 2 * n_i512_local * t_pad * _Q_WIDTH))
        gu_recv_flat = gu_recv.reshape((_P, 2 * n_i512_local * t_pad * _Q_WIDTH))
        nisa.tensor_tensor(dst=gu_flat, data1=gu_flat, data2=gu_recv_flat, op=nl.add)

    gate_out = gu[:, 0]
    up_out = gu[:, 1]
    # TODO(LNC1): when num_shards == 1 there is no cross-core reduce, so the gate
    # projection's PSUM result could feed SiLU directly (nisa.activation reads
    # PSUM, writes SBUF), saving the PSUM->SBUF copy. Not done here because the
    # shipping config is TP8 (LNC2), where the gate partial must be reduced before
    # SiLU (silu(a + b) != silu(a) + silu(b)). Revisit for an LNC1 fast path.
    nisa.activation(dst=gate_out, op=nl.silu, data=gate_out)

    # Elementwise multiply: intermediate = silu(gate) * up. Under the H-shard the
    # full I is present on every core (gate/up were reduced), so this is identical
    # on both cores. Under the I-shard it covers only this core's column slab.
    intermediate = gate_out
    nisa.tensor_tensor(dst=intermediate, data1=gate_out, data2=up_out, op=nl.multiply)

    # Quantize the intermediate to MXFP8 for the down projection.
    #
    # The I-shard quantizes into this core's slot of the exchange buffer. That
    # buffer holds the x4 data AND the E8M0 scale of both cores in ONE uint32
    # allocation, so a single sendrecv moves both; a second would pay the ~5.2 us
    # fixed rendezvous cost again. Slot layout, in uint32 words per partition:
    # [0, slot_qtz_words) x4 data, then the E8M0 scale at one BYTE per element.
    #
    # Take both dtype views on the WHOLE tensor and slice afterwards: NkiTensor.view
    # needs a static access and an offset that divides the dtype size. The slot
    # index is the OWNER's rank on both cores, because each core sends its own slot
    # into the peer's identically numbered slot, so the down loop indexes with
    # trace-time constants.
    inter_qtz3 = None
    inter_scale3 = None
    xchg = None
    xchg_x4 = None
    xchg_u8 = None
    _sc0 = 0
    n_sh_max = n_i512_local
    if shard_on_i:
        for _c in range(num_shards):
            if i512_counts[_c] > n_sh_max:
                n_sh_max = i512_counts[_c]
        # sendrecv needs the same shape on both cores, so size the slot to the
        # widest core. A core with fewer tiles leaves its tail unread; the memset
        # below keeps the bytes it sends deterministic.
        slot_qtz_words = n_sh_max * t_pad
        slot_scale_words = _div_ceil(n_sh_max * t_pad, _Q_WIDTH)
        xchg = nl.ndarray(
            (_P, num_shards, slot_qtz_words + slot_scale_words),
            dtype=nl.uint32,
            buffer=nl.sbuf,
            name="inter_xchg",
        )
        xchg_x4 = xchg.view(nl.float8_e4m3fn_x4)
        xchg_u8 = xchg.view(nl.uint8)
        _sc0 = _Q_WIDTH * slot_qtz_words  # byte offset of the scale region
        _n_loc = n_i512_local * t_pad
        if n_i512_local < n_sh_max:
            nisa.memset(dst=xchg[:, shard_id, _n_loc:], value=0)
        nisa.quantize_mx(
            dst=xchg_x4[:, shard_id, 0:_n_loc],
            src=intermediate.reshape((_P, n_i512_local * t_pad * _Q_WIDTH)),
            dst_scale=xchg_u8[:, shard_id, _sc0 : _sc0 + _n_loc],
        )
        if not cover_exchange:
            # No room to hide it: exchange now, before the down matmuls.
            _exchange_intermediate_slots(xchg, shard_id)
    else:
        inter_qtz3, inter_scale3 = _quantize_mx_3d(
            intermediate, n_i512_local, t_pad, "inter"
        )

    # Address every global I512 tile of the intermediate. Building the access
    # patterns here keeps the down loop identical for both shard modes. An access
    # pattern is an address expression and not a read, so forming the peer's
    # pattern before the exchange completes is safe.
    inter_q_aps = []
    inter_s_aps = []
    for _g in range(n_i512):
        if shard_on_i:
            _owner = 0
            for _c in range(num_shards):
                if _g >= i512_offs[_c]:
                    _owner = _c
            _loc = _g - i512_offs[_owner]
            _q0 = _loc * t_pad
            inter_q_aps.append(xchg_x4[:, _owner, _q0 : _q0 + t_pad])
            inter_s_aps.append(xchg_u8[:, _owner, _sc0 + _q0 : _sc0 + _q0 + t_pad])
        else:
            inter_q_aps.append(inter_qtz3.slice(1, _g, _g + 1))
            inter_s_aps.append(inter_scale3[:, _g, :])

    # Non-streaming down weight: load this core's H shard once.
    down_weight_sb = None
    if not down_stream:
        down_weight_u32 = nl.ndarray(
            (_P, n_i512, h_per_shard), dtype=nl.uint32, buffer=nl.sbuf
        )
        nisa.dma_copy(
            dst=down_weight_u32, src=down_w_shard, dge_mode=nisa.dge_mode.hwdge
        )
        down_weight_sb = down_weight_u32.view(nl.float8_e4m3fn_x4)

    # Down projection over near-14-KiB weight chunks. The full I is contracted, so
    # each core produces a disjoint H slice.
    #
    # PORT CHANGE (vs nkilib) #8: the intermediate is the STATIONARY operand and the
    # weight moves, so PSUM holds [T, H] directly and the nc_transpose tail plus its
    # PSUM-to-SBUF copies are gone. Same MAC count, coarser grouping: one matmul
    # covers a chunk of H columns instead of one H128 tile. nkilib cannot reach this
    # on the MX path (use_tkg_down_proj_column_tiling and transposed_out are both
    # asserted off for MX). Legality: nc_matmul_mx needs an even stationary free
    # size <= _P, which the t_pad <= _P assert above pins.
    out = nl.ndarray((t, h_size), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    out_sb = nl.ndarray((t_pad, h_per_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
    # Matmul order over the I contraction tiles. The plain order is one pass over
    # every tile. The I-shard splits it into this core's own tiles and the peer's,
    # and issues the exchange between the two passes (PORT CHANGE #10).
    down_phases = []
    if cover_exchange:
        _own = []
        _peer = []
        for _g in range(n_i512):
            if i512_off <= _g and _g < i512_off + n_i512_local:
                _own.append(_g)
            else:
                _peer.append(_g)
        down_phases.append(_own)
        down_phases.append(_peer)
    else:
        _all = []
        for _g in range(n_i512):
            _all.append(_g)
        down_phases.append(_all)

    # Accumulate in a bf16 PSUM bank (as the original tkg-mlp-mx kernel did). Each
    # tile sums only n_i512 (~5-10) matmuls, so the rounding is small next to the
    # MXFP8 input error. Check this cast first on any accuracy regression (IMT-495).
    # Two phases hold one live tile per group, because a group's accumulation spans
    # both; one phase copies out per group, so the allocator reuses a single bank.
    defer_copy = len(down_phases) > 1
    down_psum = []
    if defer_copy:
        for _gi in range(n_down_groups):
            down_psum.append(
                nl.ndarray(
                    (t_pad, down_groups[_gi][3]), dtype=nl.bfloat16, buffer=nl.psum
                )
            )

    cur_w = None
    for i_phase in range(len(down_phases)):
        if cover_exchange and i_phase == 1:
            # Every matmul above reads only this core's slab, so the rendezvous
            # overlaps them.
            _exchange_intermediate_slots(xchg, shard_id)
        for i_group in range(n_down_groups):
            i_chunk = down_groups[i_group][0]
            h_src = down_groups[i_group][1]
            g_off = down_groups[i_group][2]
            w_cols = down_groups[i_group][3]
            if defer_copy:
                group_psum = down_psum[i_group]
            else:
                group_psum = nl.ndarray(
                    (t_pad, w_cols), dtype=nl.bfloat16, buffer=nl.psum
                )
            if down_stream:
                # All chunks are prefetched above (one buffer for each chunk), so
                # just pick this chunk's buffer -- no in-loop streaming DMA.
                cur_w = down_ring[i_chunk].view(nl.float8_e4m3fn_x4)
            for i_pos in range(len(down_phases[i_phase])):
                i_i512 = down_phases[i_phase][i_pos]
                # Pick the weight operand with a statement: a ternary in the call
                # argument is an unsupported expression for the front end.
                if down_stream:
                    w_tile = cur_w[:, i_i512, g_off : g_off + w_cols]
                else:
                    w_tile = down_weight_sb[:, i_i512, h_src : h_src + w_cols]
                if folded_scale:
                    # The tile index moved to the partition axis: 4 contraction
                    # tiles share one buffer. The slice must reach the last
                    # partition, as in the gate/up loop.
                    p_base = _ROWS_PER_QUADRANT * (i_i512 % _SCALE_FOLD_SLOTS)
                    d_scale_ap = down_scale_sb[
                        nl.ds(p_base, _P - p_base),
                        i_i512 // _SCALE_FOLD_SLOTS,
                        h_src : h_src + w_cols,
                    ]
                else:
                    d_scale_ap = down_scale_sb[:, i_i512, h_src : h_src + w_cols]
                # State the accumulation instead of letting the compiler infer it.
                # The two phases put other groups' matmuls between the writes of
                # one PSUM tile, and PSUM does not start at zero, so an inferred
                # "accumulate" on the first write would add onto stale data.
                acc = True
                if i_phase == 0 and i_pos == 0:
                    acc = False
                nisa.nc_matmul_mx(
                    dst=group_psum,
                    stationary=inter_q_aps[i_i512],
                    moving=w_tile,
                    stationary_scale=inter_s_aps[i_i512],
                    moving_scale=d_scale_ap,
                    accumulate=acc,
                )
            # Interleave the PSUM->SBUF copy across the Scalar and Vector engines
            # per group, so the copies overlap instead of serializing on one.
            if not defer_copy:
                _interleave_copy(out_sb[:, h_src : h_src + w_cols], group_psum, i_group)
    if defer_copy:
        for i_group in range(n_down_groups):
            h_src = down_groups[i_group][1]
            w_cols = down_groups[i_group][3]
            _interleave_copy(
                out_sb[:, h_src : h_src + w_cols], down_psum[i_group], i_group
            )
    nisa.dma_copy(dst=out.slice(1, h_off, h_off + h_per_shard), src=out_sb[0:t, :])
    return out


@nki.jit
def qwen3_32b_mxfp8_mlp_tkg(
    hidden: nl.ndarray,
    gate_w: nl.ndarray,
    up_w: nl.ndarray,
    down_w: nl.ndarray,
    gate_s: nl.ndarray,
    up_s: nl.ndarray,
    down_s: nl.ndarray,
    weight_ring_depth: int = _RING_DEPTH_DEFAULT,
    gamma: nl.ndarray = None,
    eps: float = 1e-6,
) -> nl.ndarray:
    """Compute the Qwen3-32B MXFP8 decode MLP, sharded across the LNC cores.

    See the module docstring for the tensor layouts. The kernel quantizes the
    activation to MXFP8, runs the gate and up projections with streamed weights,
    applies ``silu(gate) * up``, quantizes the result, and runs the down
    projection with streamed weights (with the first tiles prefetched during
    gate/up).

    Pass ``gamma`` to fuse the pre-MLP RMSNorm into the kernel. The parameter
    list must stay identical to ``qwen3_32b_mxfp8_mlp_tkg_torch_ref``, because
    ``nkilib_testing`` asserts that the kernel and the reference take the same
    parameter names. So the pre-quantized ``hidden_q`` / ``hidden_s`` form is on
    ``_mlp_tkg_impl`` only, which an in-kernel caller reaches directly.

    Args:
        hidden: bf16 ``[128, H // 512, T_pad, 4]`` activation. It is raw when
            ``gamma`` is set, otherwise the caller has already normalized it.
        gate_w: uint32 ``[128, H // 512, I]`` MXFP8 gate weight.
        up_w: uint32 ``[128, H // 512, I]`` MXFP8 up weight.
        down_w: uint32 ``[128, I // 512, H]`` MXFP8 down weight (zero-padded).
        gate_s: uint8 ``[128, H // 512, I]`` pre-scattered E8M0 gate scale.
        up_s: uint8 ``[128, H // 512, I]`` pre-scattered E8M0 up scale.
        down_s: uint8 ``[128, I // 512, H]`` pre-scattered E8M0 down scale.
        weight_ring_depth: Weight ring depth. ``0`` or ``1`` disables streaming.
        gamma: Optional bf16 ``[1, H]`` RMSNorm weight in the activation's x4 H
            layout. ``None`` skips the fused norm.
        eps: RMSNorm epsilon. It applies only when ``gamma`` is set.

    Returns:
        bf16 ``[T, H]`` MLP output in HBM.

    Example:
        >>> # Build the packed inputs with the test helpers, then:
        >>> # out = qwen3_32b_mxfp8_mlp_tkg(hidden, gw, uw, dw, gs, us, ds)
        >>> # ... or with the norm fused:
        >>> # out = qwen3_32b_mxfp8_mlp_tkg(hidden, gw, uw, dw, gs, us, ds,
        >>> #                              gamma=gamma_shuffled)
    """
    return _mlp_tkg_impl(
        hidden,
        gate_w,
        up_w,
        down_w,
        gate_s,
        up_s,
        down_s,
        weight_ring_depth,
        gamma=gamma,
        eps=eps,
    )
