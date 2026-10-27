"""Paged recurrent-state gather / in-place scatter for the DeltaNet layers.

Each DeltaNet layer reads every live request's state page (conv window +
recurrent state, ~0.8 MB float32 per page on 27B at TP=4) and writes it back
once per step. Written as ``index_select`` / ``index_copy_`` over the
``[pages, rows, 128]`` row view, neuronx-cc lowers the read, inside the full
decode graph, to a gather the sync engine issues as ~37k small software DMAs
per page: 20.6 ms per gather on 27B, so a batch-1 decode step took 621 ms
(53.6 ms on the XLA route).

These kernels move each half-page with ONE DMA, HBM to HBM: the page id is a
scalar offset on the page dimension, read at runtime by the hardware descriptor
generator (``dge_mode.hwdge``). A per-row ``vector_offset`` gather was tried
first and is slower than the torch ops on 35B-A3B (GpSimd generates a
descriptor per 512-byte row). The scatter RETURNS the state tensors so the NKI
compiler reports ``operand_output_aliases`` and the aliasing pass keeps the
write in place (the pattern of MiMo-V2.5's ``kv_cache_write``).

``Qwen3_5GatedDeltaNet._use_state_kernel`` decides where each is used: the
scatter everywhere, the gather on the dense checkpoints only (on the sparse
ones the torch gather is already fast and the kernel's output costs a spilled
HBM transpose per layer).
"""

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import oob_mode
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki


def _dense_page(t, i, rows=None):
    """``rows`` rows starting at row ``i * rows`` of a dense ``[n, rows', R]`` HBM tensor.

    The same 3-D pattern as :func:`_page`. ``rows`` defaults to the tensor's own
    row count, i.e. page ``i``.
    """
    _, t_rows, r = t.shape
    rows = t_rows if rows is None else rows
    return t.ap(pattern=[[rows * r, 1], [r, rows], [1, r]], offset=i * rows * r)


def _page_rows(state, idx_sb, i, row0, nrows):
    """Rows ``[row0, row0 + nrows)`` of page ``idx_sb[i]`` (runtime scalar offset)."""
    _, rows, r = state.shape
    return state.ap(
        pattern=[[rows * r, 1], [r, nrows], [1, r]],
        offset=row0 * r,
        scalar_offset=idx_sb[i : i + 1, 0:1],
        indirect_dim=0,
    )


def _dense_rows(t, i, row0, nrows):
    """Rows ``[row0, row0 + nrows)`` of item ``i`` of a dense ``[n, rows, R]`` HBM tensor."""
    _, rows, r = t.shape
    return t.ap(pattern=[[nrows * r, 1], [r, nrows], [1, r]], offset=(i * rows + row0) * r)


def _page(state, idx_sb, i):
    """``state[idx_sb[i]]`` as a DMA access pattern: page ``i``'s id is a runtime scalar offset."""
    _, rows, r = state.shape
    return state.ap(
        pattern=[[rows * r, 1], [r, rows], [1, r]],
        offset=0,
        scalar_offset=idx_sb[i : i + 1, 0:1],
        indirect_dim=0,
    )


@nki.jit
def _state_gather_kernel(state_a: Tensor, state_b: Tensor, pages: Tensor):
    """Pages ``pages`` of both halves, as one ``[n, 2 * rows, R]`` tensor (A then B per page).

    One output rather than one per half: concatenating the halves in torch costs
    a ``[n, 2 * rows * R]`` float32 copy, which at batch 32 on 27B (26 MB) fails to
    compile (``NCC_IGCA044``).

    Args:
        state_a: ``[num_pages, rows, R]`` float32, the first half of every page.
        state_b: Same shape, the second half.
        pages: ``[n, 1]`` int32 page ids.
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("state_gather", (0, 1), 2)
    _, rows, r = state_a.shape
    n = pages.shape[0]
    out = nl.ndarray((n, 2 * rows, r), dtype=state_a.dtype, buffer=nl.shared_hbm)
    idx_sb = nl.ndarray((n, 1), dtype=pages.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx_sb, src=pages)
    for i in range(n):
        # One half per logical core, so the two copies overlap. Half h of page i
        # is rows [(2i + h) * rows, (2i + h + 1) * rows) of the output.
        if n_prgs == 1 or prg_id == 0:
            nisa.dma_copy(dst=_dense_page(out, 2 * i, rows), src=_page(state_a, idx_sb, i), dge_mode=nisa.dge_mode.hwdge)
        if n_prgs == 1 or prg_id == 1:
            nisa.dma_copy(dst=_dense_page(out, 2 * i + 1, rows), src=_page(state_b, idx_sb, i), dge_mode=nisa.dge_mode.hwdge)
    return out


@nki.jit
def _state_scatter_kernel(
    state_a: Tensor, state_b: Tensor, page_a: Tensor, page_b: Tensor, pages: Tensor
):
    """Write ``page_a`` / ``page_b`` ``[n, rows, R]`` to pages ``pages`` of the halves, in place.

    The halves arrive in their bound ``[num_pages, rows, R]`` shape: the aliasing
    pass only rewires a write whose shape matches the graph input exactly. An
    out-of-range page id drops that page's write.

    Returns:
        ``(state_a, state_b)``. Returning them is what makes the compiler report
        the write as an input/output alias; without it the write is dropped.
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("state_scatter", (0, 1), 2)
    n = pages.shape[0]
    idx_sb = nl.ndarray((n, 1), dtype=pages.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx_sb, src=pages)
    for i in range(n):
        if n_prgs == 1 or prg_id == 0:
            nisa.dma_copy(
                dst=_page(state_a, idx_sb, i), src=_dense_page(page_a, i),
                dge_mode=nisa.dge_mode.hwdge, oob_mode=oob_mode.skip,
            )
        if n_prgs == 1 or prg_id == 1:
            nisa.dma_copy(
                dst=_page(state_b, idx_sb, i), src=_dense_page(page_b, i),
                dge_mode=nisa.dge_mode.hwdge, oob_mode=oob_mode.skip,
            )
    return state_a, state_b


@nki.jit
def _state_scatter_split_kernel(
    state_a: Tensor, state_b: Tensor, conv: Tensor, rec: Tensor, pages: Tensor
):
    """Write ``conv`` ``[n, cr, R]`` and ``rec`` ``[n, rr, R]`` into pages ``pages``, in place.

    A page is ``conv | rec`` laid over half A then half B, so ``conv`` fills A's
    first ``cr`` rows, ``rec`` the rest of A and then the head of B. Writing the
    pieces directly avoids building the ``[n, 2 * rows * R]`` page in torch, whose
    concatenate fails to compile at decode batch 32 on 27B (``NCC_IGCA044``).

    Returns:
        ``(state_a, state_b)``, so the write is reported as an input/output alias.
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("state_scatter_split", (0, 1), 2)
    _, rows, _ = state_a.shape
    n, cr, _ = conv.shape
    rr = rec.shape[1]
    in_a = rows - cr  # rec rows that land in half A
    idx_sb = nl.ndarray((n, 1), dtype=pages.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx_sb, src=pages)
    for i in range(n):
        if n_prgs == 1 or prg_id == 0:
            nisa.dma_copy(
                dst=_page_rows(state_a, idx_sb, i, 0, cr), src=_dense_rows(conv, i, 0, cr),
                dge_mode=nisa.dge_mode.hwdge, oob_mode=oob_mode.skip,
            )
            nisa.dma_copy(
                dst=_page_rows(state_a, idx_sb, i, cr, in_a), src=_dense_rows(rec, i, 0, in_a),
                dge_mode=nisa.dge_mode.hwdge, oob_mode=oob_mode.skip,
            )
        if n_prgs == 1 or prg_id == 1:
            nisa.dma_copy(
                dst=_page_rows(state_b, idx_sb, i, 0, rr - in_a), src=_dense_rows(rec, i, in_a, rr - in_a),
                dge_mode=nisa.dge_mode.hwdge, oob_mode=oob_mode.skip,
            )
    return state_a, state_b


def can_scatter_split(state_a: Tensor, conv_numel: int, rec_numel: int) -> bool:
    """Whether :func:`scatter_state_split` applies: whole rows, ``conv`` in A, ``rec`` spanning A and B."""
    _, rows, r = state_a.shape
    if conv_numel % r or rec_numel % r:
        return False
    cr, rr = conv_numel // r, rec_numel // r
    return 0 < cr < rows and rows - cr < rr <= 2 * rows - cr


def scatter_state_split(
    state_a: Tensor, state_b: Tensor, conv: Tensor, rec: Tensor, pages: Tensor
) -> None:
    """Write ``conv`` ``[n, cr, R]`` and ``rec`` ``[n, rr, R]`` into pages ``pages`` (``[n]``), in place."""
    wrap_nki(_state_scatter_split_kernel)[2](
        state_a=state_a, state_b=state_b, conv=conv, rec=rec, pages=pages.to(torch.int32).view(-1, 1)
    )


def gather_state_pages(state_a: Tensor, state_b: Tensor, pages: Tensor) -> Tensor:
    """``[n, 2 * rows, R]``: pages ``pages`` (``[n]``) of both halves, A then B per page."""
    return wrap_nki(_state_gather_kernel)[2](
        state_a=state_a, state_b=state_b, pages=pages.to(torch.int32).view(-1, 1)
    )


def scatter_state_pages(
    state_a: Tensor, state_b: Tensor, page_a: Tensor, page_b: Tensor, pages: Tensor
) -> None:
    """Write ``[n, rows, R]`` pages into both halves at ``pages`` (``[n]``), in place."""
    wrap_nki(_state_scatter_kernel)[2](
        state_a=state_a, state_b=state_b, page_a=page_a, page_b=page_b,
        pages=pages.to(torch.int32).view(-1, 1),
    )
