# SPDX-License-Identifier: Apache-2.0
"""Row gather and in-place row write with indirect DMA, over a cache viewed as ``[R, width]``.

The GDN state lives in the KV cache (``state.py``), so a decode step reads the batch's state rows out of it and
writes the advanced rows back. Torch indexing on a cache inside a compiled graph lowers to work proportional to the
whole cache; these kernels move only the rows they name, one indirect DMA per tile of 128 rows. An out-of-range
row index makes the write skip the row, which is how padding rows are left out.
"""

import torch
import torch.nn.functional as F

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron import envs
from vllm_neuron.utils.hardware_config import get_logical_nc_config
from vllm_neuron.utils.neuron_utils import can_run_kernel

TILE = 128  # rows per indirect DMA (one per SBUF partition)
# The compiler pins the dynamic-DMA scratch at SBUF [0, 16 KB); the kernels' tiles start above it.
_SBUF_BASE = 17 * 1024
MAX_ROW_BYTES = 8192


def _flat(pool, dtype):
    """``pool`` as a 1-D tensor of ``dtype`` (a reinterpret when the dtypes differ)."""
    total = 1
    for size in pool.shape:
        total *= int(size)
    flat = pool.reshape((total,))
    if dtype is not None and dtype != pool.dtype:
        flat = flat.view(dtype)
        total = flat.shape[0]
    return flat, int(flat.shape[0])


@nki.jit
def _rows_write_kernel(pool, rows, row_index):
    """Write ``rows[128 * t + p]`` to row ``row_index[p, t]`` of ``pool`` viewed as ``[R, width]`` of the rows'
    dtype (the pool's bytes reinterpreted when its dtype differs), in place.

    An index outside ``[0, R)`` writes nothing. Returns ``(pool, done)``: the alias of ``pool`` and a
    ``[128, 1]`` zero token that completes after the writes.
    """
    parts, tiles = row_index.shape
    width = rows.shape[1]
    flat, total = _flat(pool, rows.dtype)
    assert parts == TILE and rows.shape[0] == TILE * tiles  # noqa: S101
    assert total % width == 0  # noqa: S101
    view = flat.reshape((total // width, width))
    done = nl.ndarray((TILE, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    idx_bytes = (tiles * 4 + 63) // 64 * 64
    at_base = _SBUF_BASE + idx_bytes
    buf_base = at_base + 256
    idx = nl.ndarray((TILE, tiles), dtype=nl.int32, buffer=nl.sbuf, name="q38w_idx", address=(0, _SBUF_BASE))
    nisa.dma_copy(dst=idx, src=row_index)
    n_buf = min(2, tiles)
    ats = [nl.ndarray((TILE, 1), dtype=nl.int32, buffer=nl.sbuf, name=f"q38w_at{k}",
                      address=(0, at_base + 64 * k)) for k in range(n_buf)]
    bufs = [nl.ndarray((TILE, width), dtype=rows.dtype, buffer=nl.sbuf, name=f"q38w_rows{k}",
                       address=(0, buf_base + MAX_ROW_BYTES * k)) for k in range(n_buf)]
    for t in range(tiles):
        at, buf = ats[t % 2], bufs[t % 2]
        # the DMA index must come from an engine-written tile, not straight from a DMA load
        nisa.tensor_scalar(dst=at, data=idx[0:TILE, t:t + 1], op0=nl.add, operand0=0)
        nisa.dma_copy(dst=buf, src=rows[nl.ds(t * TILE, TILE), :])
        nisa.dma_copy(dst=view.ap(pattern=[[width, TILE], [1, width]], offset=0, vector_offset=at[0:TILE, 0:1],
                                  indirect_dim=0),
                      src=buf, oob_mode=nisa.oob_mode.skip)
    zero = nl.ndarray((TILE, 1), dtype=nl.int32, buffer=nl.sbuf, name="q38w_zero", address=(0, at_base + 128))
    nisa.tensor_scalar(dst=zero, data=idx[0:TILE, 0:1], op0=nl.multiply, operand0=0)
    nisa.dma_copy(dst=done, src=zero)
    return pool, done


@nki.jit
def _rows_gather_kernel(pool, row_index, width: int, as_f32: bool = False):
    """``out[128 * t + p] = pool_view[row_index[p, t]]`` with ``pool`` viewed as ``[R, width]`` (of fp32 when
    ``as_f32``, the pool's bytes reinterpreted); indices in range."""
    parts, tiles = row_index.shape
    flat, total = _flat(pool, nl.float32 if as_f32 else None)
    assert parts == TILE and total % width == 0  # noqa: S101
    view = flat.reshape((total // width, width))
    out = nl.ndarray((TILE * tiles, width), dtype=flat.dtype, buffer=nl.shared_hbm)
    idx_bytes = (tiles * 4 + 63) // 64 * 64
    at_base = _SBUF_BASE + idx_bytes
    buf_base = at_base + 256
    idx = nl.ndarray((TILE, tiles), dtype=nl.int32, buffer=nl.sbuf, name="q38g_idx", address=(0, _SBUF_BASE))
    nisa.dma_copy(dst=idx, src=row_index)
    n_buf = min(2, tiles)
    ats = [nl.ndarray((TILE, 1), dtype=nl.int32, buffer=nl.sbuf, name=f"q38g_at{k}",
                      address=(0, at_base + 64 * k)) for k in range(n_buf)]
    bufs = [nl.ndarray((TILE, width), dtype=flat.dtype, buffer=nl.sbuf, name=f"q38g_rows{k}",
                       address=(0, buf_base + MAX_ROW_BYTES * k)) for k in range(n_buf)]
    for t in range(tiles):
        at, buf = ats[t % 2], bufs[t % 2]
        nisa.tensor_scalar(dst=at, data=idx[0:TILE, t:t + 1], op0=nl.add, operand0=0)
        nisa.dma_copy(dst=buf, src=view.ap(pattern=[[width, TILE], [1, width]], offset=0,
                                           vector_offset=at[0:TILE, 0:1], indirect_dim=0))
        nisa.dma_copy(dst=out[nl.ds(t * TILE, TILE), :], src=buf)
    return out


def _grid():
    return 1 if envs.VLLM_NEURON_CPU_MODE else get_logical_nc_config()


def untile(index: torch.Tensor, n: int) -> torch.Tensor:
    """The ``[n]`` row numbers of a ``[128, tiles]`` index, in order (host fallbacks only)."""
    return index.t().reshape(-1)[:n]


def gather_rows(pool: torch.Tensor, index: torch.Tensor, n: int, width: int, dtype=None) -> torch.Tensor:
    """``pool`` viewed as ``[R, width]`` of ``dtype`` (default: the pool's; only bf16 to fp32 is a reinterpret);
    ``index`` the ``[128, tiles]`` row numbers (in range) of ``n`` rows, row ``128 * t + p`` at ``index[p, t]``.
    Returns ``[n, width]``. The reinterpret happens inside the kernel: torch never sees a dtype view of the
    pool, which AOTAutograd refuses for a mutated input and neuronx-cc 2.28 failed to compile as a bitcast."""
    as_f32 = dtype == torch.float32 and pool.dtype != torch.float32
    if not can_run_kernel(pool):
        src = pool.view(torch.float32) if as_f32 else pool
        return src.reshape(-1, width).index_select(0, untile(index, n).long())
    out = wrap_nki(_rows_gather_kernel)[_grid()](pool=pool, row_index=index, width=width, as_f32=as_f32)
    return out[:n]


def write_rows(pool: torch.Tensor, values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Write ``values`` ``[n, width]`` to the rows ``index`` names (as in ``gather_rows``) of ``pool`` viewed as
    ``[R, width]``, in place; a row outside ``[0, R)`` is skipped. Returns the kernel's completion token."""
    n, width = values.shape
    total = pool.numel() * pool.element_size() // values.element_size() // width
    if not can_run_kernel(pool):
        rows = untile(index, n).long()
        keep = (rows >= 0) & (rows < total)
        safe = torch.where(keep, rows, 0)
        view = pool.view(values.dtype).view(-1, width)
        view.index_copy_(0, safe, torch.where(keep[:, None], values, view.index_select(0, safe)))
        return torch.zeros((TILE, 1), dtype=torch.int32, device=pool.device)
    pad = index.numel() - n
    _, token = wrap_nki(_rows_write_kernel)[_grid()](pool=pool, rows=F.pad(values, (0, 0, 0, pad)), row_index=index)
    return token
