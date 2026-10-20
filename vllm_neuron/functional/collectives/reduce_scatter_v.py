# SPDX-License-Identifier: Apache-2.0
"""MoE-combine reduce-scatter, fused as intra-device all-to-all-v + on-chip sum.

`reduce_scatter_v` provides the variable-length sum-reduce + scatter used by the MoE
EP/TP decode combine (the intra-TP reduce-scatter of partial expert-MLP outputs). It is
built on `nki.collectives.all_to_all_v` rather than a dedicated `reduce_scatter_v` NIR op
(not exposed in any reachable NKI wheel):

    1. A2A-v scatter: each source rank `r` sends rank `d` the rows destined for `d`'s
       output partition. Rows are variable-length (real per-(src,dst) token counts) and
       front-packed within a fixed `recv_rows`-tall slot. After the collective, rank `d`
       holds `world_size` slots (one per source), each carrying the SAME tokens in the
       SAME order for `d`.
    2. On-chip sum: the `world_size` slots are reduced element-wise (positional sum) into
       rank `d`'s reduced partition `[recv_rows, H]`.

Both steps run inside a single `@nki.jit` kernel / NEFF: the collective writes a
`shared_hbm` intermediate; the reduction DMAs the per-source slots into SBUF and
accumulates there (LNC-sharded on H, tiled by pmax on rows). There is no torch-level
`[world*N, H]` intermediate and no separate reduction graph.

Reduce semantics require identical output partitioning on all ranks, so every rank passes
the same `recv_row_counts`; with `T` total rows and `TP = world_size` the per-rank slot
is `recv_rows = T // TP` (T must be divisible by TP). The input carries 2 trailing
packed-token-index columns ([N, H+2]) that are not reduced (see `cc_use_intermediate_io`).

The intra-device A2A-v path requires the intra-chip runtime (libnrt 2.x.60026.0,
CR-284747640); the NKI API is unchanged.
"""

from typing import Sequence, Tuple

import torch

import nki
import nki.isa as nisa
import nki.collectives as ncc
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

from vllm.distributed.parallel_state import GroupCoordinator
from vllm_neuron import envs
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

# Trailing packed-token-index columns in the MoE-combine input ([N, H+2]): 2 bf16 cols
# holding a bitcast int32 token id. Not part of the reduced data (only H data cols are
# summed).
_IDX_COLS = 2


def reduce_scatter_v(
    input: torch.Tensor,
    group: GroupCoordinator,
    send_row_counts: Sequence[int],
    recv_row_counts_from: Sequence[int],
    op: str = "add",
    cc_use_intermediate_io: bool = False,
) -> torch.Tensor:
    """MoE-combine reduce-scatter via fused intra-device all-to-all-v + on-chip sum.

    Reduces this rank's ``TP`` per-destination slots of partial expert outputs and scatters
    the reduced partition to each rank. ``T = input.shape[0]`` total rows split evenly into
    ``TP = world_size`` slots of ``recv_rows = T // TP`` rows each (``T`` must be divisible
    by ``TP``); slot ``d`` is this rank's contribution to rank ``d``'s partition, with the
    real per-(src,dst) rows front-packed and the rest zero padding.

    Args:
        input: This rank's contribution, shape ``[T, H+2]``. The last 2 columns are the
            packed token index (bitcast int32) and are NOT reduced. ``T`` is split into
            ``world_size`` contiguous slots of ``T // world_size`` rows.
        group: GroupCoordinator for the ranks the collective executes across (the intra-
            device TP group, e.g. 4 ranks on one Trn2 device).
        send_row_counts: ``send_row_counts[d]`` = real rows this rank sends to dest ``d``
            (rows, not elements). Real rows are front-packed within each fixed
            ``recv_rows`` slot; must be identical on every rank (reduce semantics).
        recv_row_counts_from: ``recv_row_counts_from[s]`` = real rows this rank receives
            from source ``s`` (``recv_counts_known=True``). For this rank's reduction all
            sources carry the SAME tokens in the SAME order, so the reduce is a positional
            sum over the ``world_size`` slots — hence all entries are equal (this rank's
            shared real received count).
        op: Reduction operator. Only ``"add"`` (sum) is supported.
        cc_use_intermediate_io: **Correctness-critical, not just perf.** Selects whether
            the kernel copies ``input`` into an internal ``shared_hbm`` buffer before the
            collective:
              - ``True``: copy first (strips the 2 id cols into an intermediate). Required
                when ``input`` is a NEFF I/O tensor — the compiler forbids a collective
                reading an I/O tensor ("cannot read IO tensors", NCC_INLA001). Use this
                for standalone/test call sites where ``input`` is a graph input.
              - ``False`` (default): feed ``input`` directly as the collective source (no
                strip; the 2 id cols ride along in transport and are not reduced). Valid
                ONLY when ``input`` is a graph intermediate (e.g. the expert-MLP output in
                the model). Faster (saves the strip copy). Passing ``False`` on a NEFF I/O
                input is a compile error, not a silent fallback.

    Returns:
        The reduced partition for this rank, shape ``[recv_rows, H]`` (no id columns).
        Only rows ``[0:real_received_count]`` are valid; the tail ``[real:recv_rows]`` is
        **undefined** (not zeroed) — callers must treat it as padding.

    Example:
        >>> # TP=4, T rows -> recv_rows = T // 4; real per-dest counts e.g. [20,10,0,34].
        >>> out = reduce_scatter_v(x, group=tp_group,
        ...                        send_row_counts=[20, 10, 0, 34],
        ...                        recv_row_counts_from=[C]*4)
    """
    self_rank = group.rank_in_group
    world_size = group.world_size
    group_ranks = tuple(group.ranks)

    T = input.shape[0]
    assert T % world_size == 0, (
        f"reduce_scatter_v: T (input rows) must be divisible by world_size (TP), "
        f"got {T=}, {world_size=}"
    )
    recv_rows = T // world_size

    return _reduce_scatter_v_combine(
        input=input,
        group_ranks=group_ranks,
        self_rank=self_rank,
        world_size=world_size,
        recv_rows=recv_rows,
        send_row_counts=send_row_counts,
        recv_row_counts_from=recv_row_counts_from,
        op=op,
        cc_use_intermediate_io=cc_use_intermediate_io,
    )


def _reduce_scatter_v_combine(
    input,
    group_ranks,
    self_rank,
    world_size,
    recv_rows,
    send_row_counts,
    recv_row_counts_from,
    op,
    cc_use_intermediate_io,
):
    """MoE-combine reduce-scatter: real variable-length a2a-v + positional sum.

    ``input`` is ``[world_size * recv_rows, H+2]`` laid out as per-destination slots (slot
    ``d`` = rows for dest ``d``, real rows grouped at the FRONT, padding after). The 2
    trailing columns (packed token index) are STRIPPED — only the ``H`` data columns are
    sent and reduced. Send/recv counts are the real per-slot row counts (variable, may be
    0), so the collective transports only real rows. After the collective, every source
    slot carries the SAME tokens in the SAME order for this rank, so the reduction is a
    positional sum over the ``world_size`` slots, bounded by this rank's real received
    count. Output is ``[recv_rows, H]`` (no id columns).
    """
    assert not envs.VLLM_NEURON_CPU_MODE, (
        "reduce_scatter_v collective is not supported on CPU mode (no NKI collective sim)"
    )
    assert op == "add", (
        f"reduce_scatter_v only supports op='add' (sum-reduce), got {op=}"
    )
    assert len(send_row_counts) == world_size, (
        f"len(send_row_counts) must == world_size, got "
        f"{len(send_row_counts)=}, {world_size=}"
    )
    # All recv_row_counts_from entries must be equal: every source slot carries this rank's
    # SAME tokens in the SAME order, so the reduction is positional. A non-uniform value would
    # silently produce a wrong sum.
    assert len(set(int(c) for c in recv_row_counts_from)) <= 1, (
        f"recv_row_counts_from entries must all be equal (positional reduce), got "
        f"{list(recv_row_counts_from)=}"
    )
    Hp = input.shape[1]
    H = Hp - _IDX_COLS  # last _IDX_COLS cols are the packed token index (not reduced)

    # Metadata in ELEMENTS, int32 (XLA has no native unsigned; kernel bitcasts). The a2av
    # transports whole rows of width `cc_row`, so counts/displs are in units of cc_row:
    #  - cc_use_intermediate_io=True  (standalone test: `input` is a NEFF IO tensor, which the
    #    compiler forbids as collective I/O -- "cannot read IO tensors", NCC_INLA001): the
    #    kernel STRIPS input[:, :H] into an intermediate cc_input, so cc_row = H.
    #  - cc_use_intermediate_io=False (production: `input` is a graph intermediate = expert
    #    output, not a NEFF IO tensor): the kernel feeds `input` ([N, H+2]) DIRECTLY as the
    #    collective src (no strip), transporting whole H+2-wide rows -> cc_row = Hp = H+2. The
    #    2 id cols ride along (negligible payload) and are never read by the reduction.
    # Displs are the fixed slot starts (real rows front-packed within each recv_rows slot),
    # NOT a packed cumsum. recv_counts_known=True -> row2 is an input we supply.
    cc_row = H if cc_use_intermediate_io else Hp
    send_counts = [int(send_row_counts[d]) * cc_row for d in range(world_size)]
    slot_displs = [d * recv_rows * cc_row for d in range(world_size)]
    recv_counts = [int(recv_row_counts_from[s]) * cc_row for s in range(world_size)]

    # SELF-LOOPBACK SKIP: the a2av's self->self chunk (source == self_rank) is already present
    # in cc_input before the collective, so routing it through the collective is a wasted
    # SB2SB copy. Zero this rank's self send+recv counts so the collective skips the loopback
    # (symmetric: each rank only zeros its OWN diagonal, so no peer sees a count mismatch), and
    # the kernel adds the self contribution directly from cc_input (a load that can overlap the
    # collective). cc_output[self_slot] stays 0 (pre-zeroed) -> the positional sum over slots
    # yields the non-self sum, and +self_sb makes it exact.
    send_counts[self_rank] = 0
    recv_counts[self_rank] = 0
    self_row = (
        self_rank * recv_rows
    )  # row index of this rank's self slot (runtime; scalar_offset)

    counts = torch.tensor(send_counts, dtype=torch.int32, device=input.device)
    sdispls = torch.tensor(slot_displs, dtype=torch.int32, device=input.device)
    rcounts = torch.tensor(recv_counts, dtype=torch.int32, device=input.device)
    rdispls = torch.tensor(slot_displs, dtype=torch.int32, device=input.device)
    metadata = torch.stack([counts, sdispls, rcounts, rdispls])
    self_row_tensor = torch.tensor([[self_row]], dtype=torch.int32, device=input.device)

    # This rank's real received row count (shared across sources). It is RUNTIME (differs per
    # rank: e.g. [20,10,0,34]) — every rank compiles the SAME NEFF, so the reduce bound is
    # passed as a device scalar (n_dyn below) rather than baked into shapes.
    real_rows = int(recv_row_counts_from[0])

    # Static/dynamic split: process the first STATIC_ROWS (64) rows in one STATIC tile via a
    # plain positional sum (exact because cc_output padding is zeroed and only real rows are
    # front-packed, so the 0-count rank sums to 0). The remaining recv_rows-STATIC_ROWS rows
    # are handled in a SINGLE dynamic iteration (n_dyn is 0 or 1): entered only when X exceeds
    # the static chunk, and inside it the remainder is statically unrolled into pmax tiles
    # (e.g. 192 -> 128 + 64). This pays only ONE branch-compare + dynamic-loop overhead,
    # instead of one per pmax tile. Padding rows in the remainder are 0 so the extra summed
    # rows are exact.
    STATIC_ROWS = 64
    n_dyn = 1 if real_rows > STATIC_ROWS else 0
    n_dyn_tensor = torch.tensor([[n_dyn]], dtype=torch.int32, device=input.device)

    wrapped = wrap_nki(_reduce_scatter_v_combine_nki)
    return wrapped[2](
        input=input,
        metadata=metadata,
        n_dyn_tensor=n_dyn_tensor,
        self_row_tensor=self_row_tensor,
        group=group_ranks,
        recv_rows=recv_rows,
        world_size=world_size,
        H=H,
        static_rows=STATIC_ROWS,
        cc_use_intermediate_io=cc_use_intermediate_io,
    )


@nki.jit
def _reduce_scatter_v_combine_nki(
    input: nl.NkiTensor,
    metadata: nl.NkiTensor,
    n_dyn_tensor: nl.NkiTensor,
    self_row_tensor: nl.NkiTensor,
    group: Tuple[int],
    recv_rows: int,
    world_size: int,
    H: int,
    static_rows: int = 64,
    cc_use_intermediate_io: bool = False,
) -> nl.NkiTensor:
    """MoE-combine reduce-scatter kernel: strip ids -> real-count a2a-v -> positional sum.

    Steps:
      1. Strip the trailing _IDX_COLS from ``input`` ([N, H+2]) into the collective src
         ([N, H]) — only real data is transported.
      2. Zero the collective output so the front-packed real rows land on a zeroed slot
         (rows beyond a slot's real count stay 0 → summing them is exact; no oob skip
         needed for correctness).
      3. ``all_to_all_v`` with the passed-in real per-slot counts (recv_counts_known=True).
      4. Positional sum over the ``world_size`` slots, LNC-sharded on H, split into a
         STATIC first tile of ``static_rows`` (64) rows + a DYNAMIC remainder:
           - Static tile: rows [0, static_rows). Since cc_output is zeroed before the
             a2av and each source front-packs only its real rows, a plain positional sum
             over the fixed static_rows is exact for any runtime X (0-count rank -> all
             zeros -> output stays 0). No per-row skip needed.
           - Dynamic remainder: rows [static_rows, recv_rows) handled in a SINGLE dynamic
             iteration (``nl.dynamic_range(n_dyn)``, n_dyn in {0,1}), entered only when
             X > static_rows. Inside it the remainder is statically unrolled into
             compile-time pmax tiles (e.g. 192 -> 128 + 64), so only one branch-compare +
             dynamic overhead is paid.
         Output [recv_rows, H], padded tail stays 0.

    ``H`` is the data width (input width minus _IDX_COLS). ``n_dyn_tensor`` is a [1,1] int32
    device scalar = 1 if X (this rank's real received row count) > static_rows else 0.
    ``self_row_tensor`` is a [1,1] int32 scalar = this rank's self-slot row offset
    (self_rank * recv_rows), used as a scalar_offset for the self-loopback preload.
    """
    if metadata.dtype != nl.uint32:
        metadata = metadata.view(nl.uint32)

    N = world_size * recv_rows
    replica_group = ncc.ReplicaGroup([list(group)])

    Hp = input.shape[1]  # input row stride (= H + _IDX_COLS)
    # cc_row = the collective's transported row width (see wrapper). When cc_use_intermediate_io
    # we strip input[:, :H] into cc_input (row width H); otherwise we feed `input` directly and
    # transport whole H+2-wide rows (row width Hp). cc_output matches that width.
    cc_row = H if cc_use_intermediate_io else Hp
    cc_output = nl.ndarray((N, cc_row), input.dtype, buffer=nl.shared_hbm)
    cc_metadata = nl.ndarray(metadata.shape, metadata.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(cc_metadata, metadata)

    _, n_prgs, prg_id = get_verified_program_sharding_info("reduce_scatter_v", (0, 1))
    kernel_assert(H % n_prgs == 0, f"Expected H divisible by LNC, got {H=} {n_prgs=}")
    H_local = H // n_prgs
    H_off = H_local * prg_id
    P = nl.tile_size.pmax

    # SELF-LOOPBACK PRELOAD (issued FIRST, from the original `input`): this rank's self->self
    # chunk was zeroed out of the a2av metadata (cc_output[self_slot] stays 0), so its own
    # contribution to the positional sum is loaded directly. Reading from `input` (not cc_input)
    # means this DMA depends ONLY on the kernel input — not on the strip loop, the zeroing, or
    # the collective, and it shares NO buffer with the collective's srcs=[cc_input]. So it runs
    # truly in PARALLEL with strip + zero + the collective transport. self_row is runtime
    # (per-rank) -> scalar_offset. Only the static_rows front tile is preloaded here; the >64
    # remainder self-term is loaded inside the dynamic loop. input is [N, H+2]; slicing the H
    # data cols via the H_off..H_off+H_local free-dim range strips the id cols implicitly.
    self_row_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(self_row_sb, self_row_tensor)
    self_static = nl.ndarray((static_rows, H_local), dtype=input.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=self_static,
        src=input.ap(
            pattern=[[Hp, static_rows], [1, H_local]],
            offset=H_off,
            scalar_offset=self_row_sb,
        ),
    )

    # Collective src. NEFF I/O cannot be collective I/O (compiler verifier: "Collective
    # instruction cannot read IO tensors", NCC_INLA001). So:
    #  - cc_use_intermediate_io=True (input IS a NEFF IO tensor, e.g. standalone test): strip
    #    input[:, :H] into an intermediate cc_input [N, H] shared_hbm buffer (also drops ids).
    #  - else (input is a graph intermediate, e.g. production expert output): feed input
    #    DIRECTLY as the collective src -> NO strip copy (saves ~16.8MB read + ~16.8MB write).
    if cc_use_intermediate_io:
        cc_input = nl.ndarray((N, H), input.dtype, buffer=nl.shared_hbm)
        for row0 in range(0, N, P):
            rows = min(P, N - row0)
            tmp = nl.ndarray((rows, H_local), dtype=input.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=tmp, src=input[row0 : row0 + rows, H_off : H_off + H_local]
            )
            nisa.dma_copy(cc_input[row0 : row0 + rows, H_off : H_off + H_local], tmp)
    else:
        cc_input = input

    # Zero ONLY this rank's self slot in cc_output (was: zero all N=1024 rows, ~16MB write, a
    # big chunk of the DMA-bound preamble). Rationale:
    #  - Non-self slots: the a2av writes rows [0:recv_count]; any garbage in the padding rows
    #    [recv_count:recv_rows] only ever lands in the OUTPUT TAIL (rows >= real_rows), which is
    #    don't-care (output is semantically [real_rows, H]). So non-self padding needs no zero.
    #  - Self slot: the a2av SKIPS it (self-loopback opt zeroed its send/recv count), so it is
    #    entirely unwritten. The reduction's positional pack-sum includes it, then adds
    #    self_static separately -> the self slot MUST read 0 or it corrupts the valid out[0:X].
    # So zero just the self slot (recv_rows rows at the runtime self_row offset). memset writes
    # sbuf only, so memset a zero SBUF tile and DMA it across the self-slot rows.
    zero_sb = nl.ndarray((P, H_local), dtype=input.dtype, buffer=nl.sbuf)
    nisa.memset(zero_sb, 0)
    for row0 in range(0, recv_rows, P):
        rows = min(P, recv_rows - row0)
        nisa.dma_copy(
            dst=cc_output.ap(
                pattern=[[cc_row, rows], [1, H_local]],
                offset=row0 * cc_row + H_off,
                scalar_offset=self_row_sb,
            ),
            src=zero_sb[:rows, :],
        )

    ncc.all_to_all_v(
        srcs=[cc_input],
        dsts=[cc_output],
        replica_group=replica_group,
        metadata_tensor=cc_metadata,
        recv_counts_known=True,
        has_rdispls=False,
    )

    # Output [recv_rows, H]. NOT pre-zeroed: the reduction writes rows [0:real_rows] (static
    # tile + dynamic remainder), and rows beyond that are the don't-care tail (output is
    # semantically [real_rows, H]). Dropping the tail memset saves ~4MB write. Callers must
    # treat out[real_rows:] as undefined.
    output_hbm = nl.ndarray((recv_rows, H), input.dtype, buffer=nl.shared_hbm)

    # ---- STATIC first tile: rows [0, static_rows). ----
    # cc_output was fully zeroed before the a2av, and every source front-packs only its X
    # real rows per destination slot — so all padding rows are exact zeros. A plain
    # positional sum over the fixed static_rows is therefore exact for ANY runtime X,
    # including the 0-count rank (X=0 -> all static rows are zero padding -> output stays 0).
    # Rows beyond static_rows are handled by the dynamic remainder below (n_dyn=0 when X<=64).
    #
    # NAIVE sequential sum (each slot is a separate 64-partition add: acc = s0+s1; acc += s2;
    # ... over all world_size slots), then + self (self-loopback term preloaded from `input`).
    # Two levels of DMA/compute overlap:
    #   (1) RANK PIPELINE within an H-chunk: slot s+1's load is issued BEFORE the add of slot
    #       s, so DMA(s+1) overlaps the add.
    #   (2) H-HALF DOUBLE BUFFER across chunks: the next H-chunk's first two slot loads are
    #       issued before this chunk's compute, so its DMA overlaps this chunk's compute.
    # Distinct per-slot SBUF buffers carry no false dependency. (NKI tracer rejects
    # list-of-tiles indexing, so slots are named vars rebound across chunks / pipeline steps.)
    # (slot-load is inlined at each use — NKI forbids inner function definitions.)
    HC = 2048  # free-dim (H) pipeline chunk (e.g. H_local=4096 -> 2 chunks = the two H halves)
    if world_size >= 2 and static_rows <= P and H_local % HC == 0:
        n_chunks = H_local // HC
        w = HC
        SR = static_rows

        # Prefetch chunk 0's first two slots (s0, s1).
        cur0 = nl.ndarray((SR, w), dtype=input.dtype, buffer=nl.sbuf)
        cur1 = nl.ndarray((SR, w), dtype=input.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=cur0,
            src=cc_output.ap(
                pattern=[[cc_row, SR], [1, w]], offset=0 * recv_rows * cc_row + H_off
            ),
        )
        nisa.dma_copy(
            dst=cur1,
            src=cc_output.ap(
                pattern=[[cc_row, SR], [1, w]], offset=1 * recv_rows * cc_row + H_off
            ),
        )

        for i in range(n_chunks):
            h0 = i * HC
            # H-double-buffer: kick off the NEXT chunk's first two slot loads now.
            if i + 1 < n_chunks:
                hn = (i + 1) * HC
                nxt0 = nl.ndarray((SR, w), dtype=input.dtype, buffer=nl.sbuf)
                nxt1 = nl.ndarray((SR, w), dtype=input.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=nxt0,
                    src=cc_output.ap(
                        pattern=[[cc_row, SR], [1, w]],
                        offset=0 * recv_rows * cc_row + H_off + hn,
                    ),
                )
                nisa.dma_copy(
                    dst=nxt1,
                    src=cc_output.ap(
                        pattern=[[cc_row, SR], [1, w]],
                        offset=1 * recv_rows * cc_row + H_off + hn,
                    ),
                )

            # acc = s0 + s1, then rank-pipeline slots 2..world_size-1: issue slot s+1's load
            # before adding slot s so the DMA overlaps the add. `pend` holds the in-flight tile.
            acc = nl.ndarray((SR, w), dtype=input.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=acc, data1=cur0, data2=cur1, op=nl.add)
            if world_size > 2:
                pend = nl.ndarray((SR, w), dtype=input.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=pend,
                    src=cc_output.ap(
                        pattern=[[cc_row, SR], [1, w]],
                        offset=2 * recv_rows * cc_row + H_off + h0,
                    ),
                )
                for s in range(2, world_size):
                    ready = pend
                    if s + 1 < world_size:
                        pend = nl.ndarray((SR, w), dtype=input.dtype, buffer=nl.sbuf)
                        nisa.dma_copy(
                            dst=pend,
                            src=cc_output.ap(
                                pattern=[[cc_row, SR], [1, w]],
                                offset=(s + 1) * recv_rows * cc_row + H_off + h0,
                            ),
                        )
                    nisa.tensor_tensor(dst=acc, data1=acc, data2=ready, op=nl.add)
            # + this rank's own (self-loopback) contribution, preloaded before the collective.
            nisa.tensor_tensor(
                dst=acc, data1=acc, data2=self_static[:, h0 : h0 + w], op=nl.add
            )
            nisa.dma_copy(
                dst=output_hbm.ap(pattern=[[H, SR], [1, w]], offset=H_off + h0),
                src=acc,
            )
            if i + 1 < n_chunks:
                cur0, cur1 = nxt0, nxt1
    else:
        # General fallback: sequential positional sum over the slots.
        acc = nl.ndarray((static_rows, H_local), dtype=input.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=acc,
            src=cc_output.ap(
                pattern=[[cc_row, static_rows], [1, H_local]], offset=H_off
            ),
        )
        for s in range(1, world_size):
            stage = nl.ndarray(
                (static_rows, H_local), dtype=input.dtype, buffer=nl.sbuf
            )
            nisa.dma_copy(
                dst=stage,
                src=cc_output.ap(
                    pattern=[[cc_row, static_rows], [1, H_local]],
                    offset=s * recv_rows * cc_row + H_off,
                ),
            )
            nisa.tensor_tensor(dst=acc, data1=acc, data2=stage, op=nl.add)
        # Add this rank's own (self-loopback) contribution, preloaded before the collective.
        nisa.tensor_tensor(dst=acc, data1=acc, data2=self_static, op=nl.add)
        nisa.dma_copy(
            dst=output_hbm.ap(pattern=[[H, static_rows], [1, H_local]], offset=H_off),
            src=acc,
        )

    # ---- DYNAMIC remainder: rows [static_rows, recv_rows) in a SINGLE dynamic iteration
    # (n_dyn is 0 or 1). Inside the iteration the remainder is statically unrolled into
    # compile-time pmax tiles (e.g. 192 -> 128 + 64), so only ONE branch-compare + dynamic
    # overhead is paid regardless of how many pmax tiles the remainder spans. Tile offsets
    # are compile-time constants, so no runtime row_off register is needed. Padded rows in
    # cc_output are 0 so summing full tiles is exact.
    P = 128  # nl.tile_size.pmax
    n_dyn_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(n_dyn_sb, n_dyn_tensor)
    n_dyn_reg = nisa.register_alloc()
    nisa.register_load(src=n_dyn_sb, dst=n_dyn_reg)

    def _dynamic_remainder_body(_):
        for row0 in range(static_rows, recv_rows, P):
            rows = min(P, recv_rows - row0)
            acc_d = nl.ndarray((rows, H_local), dtype=input.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=acc_d,
                src=cc_output.ap(
                    pattern=[[cc_row, rows], [1, H_local]], offset=row0 * cc_row + H_off
                ),
            )
            for s in range(1, world_size):
                stage = nl.ndarray((rows, H_local), dtype=input.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=stage,
                    src=cc_output.ap(
                        pattern=[[cc_row, rows], [1, H_local]],
                        offset=s * recv_rows * cc_row + row0 * cc_row + H_off,
                    ),
                )
                nisa.tensor_tensor(dst=acc_d, data1=acc_d, data2=stage, op=nl.add)
            # Add this rank's own (self-loopback) rows for this remainder tile, from `input`.
            self_d = nl.ndarray((rows, H_local), dtype=input.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=self_d,
                src=input.ap(
                    pattern=[[Hp, rows], [1, H_local]],
                    offset=row0 * Hp + H_off,
                    scalar_offset=self_row_sb,
                ),
            )
            nisa.tensor_tensor(dst=acc_d, data1=acc_d, data2=self_d, op=nl.add)
            nisa.dma_copy(
                dst=output_hbm.ap(
                    pattern=[[H, rows], [1, H_local]], offset=row0 * H + H_off
                ),
                src=acc_d,
            )

    nl.fori_loop(0, n_dyn_reg, _dynamic_remainder_body)

    return output_hbm
