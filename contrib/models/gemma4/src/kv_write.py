# SPDX-License-Identifier: Apache-2.0
"""Paged KV-cache access with indirect DMA: in-place row write and block gather.

These kernels work on the flat 1-D tensor over a shared KV buffer (see ``README.md`` in this folder for
why the model hands the compiler one flat tensor per buffer and nothing else).

Why kernels and not torch ops: a torch scatter into the cache lowers to work proportional to the whole
cache (in a 6-layer profile, ``scatter`` moved ~1.5 GB of HBM per decode step), while this kernel moves
each new row with one indirect DMA per tile of 128 rows. The kernel alone was not the fix for the slow
decode: the cost came from how the compiler wrote mutated views back into a merged base (see README.md).

Rows are ``head_dim`` elements wide. An out-of-range row index writes nothing, which is how padding rows
are skipped. The write kernel returns a small token that completes after the writes; the gather kernel
adds ``0 * token`` to its indices so the read is ordered after the write.
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

_TILE = 128  # rows per indirect DMA (one per SBUF partition)
# The compiler pins the dynamic-DMA scratch at SBUF [0, 16 KB); user tiles start above it.
_SBUF_BASE = 17 * 1024
_MAX_ROW_BYTES = 8192


def _elem_bytes(dtype) -> int:
    return 1 if dtype in (nl.float8_e4m3fn, nl.float8_e4m3, nl.uint8, nl.int8) else 2


@nki.jit
def _kv_rows_write_kernel(pool, rows, row_index):
    """Write ``rows[128 * t + p]`` to row ``row_index[p, t]`` of ``pool`` viewed as ``[R, width]``.

    Args:
        pool: HBM cache ``[..., width]`` (any leading shape), bf16 or FP8.
        rows: ``[128 * T, width]`` rows in the dtype of ``pool``.
        row_index: ``[128, T]`` int32 row numbers. An index outside ``[0, R)``
            writes nothing, which is how padding rows (and negative slots) are skipped.

    Returns:
        ``(pool, done)``: ``pool`` with the rows written (alias of the input) and a ``[128, 1]``
        zero token that completes after the writes.
    """
    parts, tiles = row_index.shape
    width = rows.shape[1]
    total = 1
    for size in pool.shape:
        total *= int(size)
    assert parts == _TILE and rows.shape[0] == _TILE * tiles  # noqa: S101
    assert total % width == 0  # noqa: S101
    assert width * _elem_bytes(rows.dtype) <= _MAX_ROW_BYTES  # noqa: S101

    view = pool.reshape((total // width, width))
    done = nl.ndarray((_TILE, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    idx_bytes = (tiles * 4 + 63) // 64 * 64
    at_base = _SBUF_BASE + idx_bytes
    buf_base = at_base + 256

    idx = nl.ndarray(
        (_TILE, tiles), dtype=nl.int32, buffer=nl.sbuf, name="kvw_idx", address=(0, _SBUF_BASE)
    )
    nisa.dma_copy(dst=idx, src=row_index)
    n_buf = min(2, tiles)
    ats = [
        nl.ndarray(
            (_TILE, 1), dtype=nl.int32, buffer=nl.sbuf, name=f"kvw_at{k}",
            address=(0, at_base + 64 * k),
        )
        for k in range(n_buf)
    ]
    bufs = [
        nl.ndarray(
            (_TILE, width), dtype=rows.dtype, buffer=nl.sbuf, name=f"kvw_rows{k}",
            address=(0, buf_base + _MAX_ROW_BYTES * k),
        )
        for k in range(n_buf)
    ]
    for t in range(tiles):
        at, buf = ats[t % 2], bufs[t % 2]
        # The DMA index must come from an engine-written tile, not straight from a DMA load.
        nisa.tensor_scalar(dst=at, data=idx[0:_TILE, t : t + 1], op0=nl.add, operand0=0)
        nisa.dma_copy(dst=buf, src=rows[nl.ds(t * _TILE, _TILE), :])
        nisa.dma_copy(
            dst=view.ap(
                pattern=[[width, _TILE], [1, width]],
                offset=0,
                vector_offset=at[0:_TILE, 0:1],
                indirect_dim=0,
            ),
            src=buf,
            oob_mode=nisa.oob_mode.skip,
        )
    # A zero token that exists only once the kernel has run: a later kernel that consumes it
    # starts after every row above is written (ordering without reading the cache alias).
    zero = nl.ndarray(
        (_TILE, 1), dtype=nl.int32, buffer=nl.sbuf, name="kvw_zero", address=(0, at_base + 128)
    )
    nisa.tensor_scalar(dst=zero, data=idx[0:_TILE, 0:1], op0=nl.multiply, operand0=0)
    nisa.dma_copy(dst=done, src=zero)
    return pool, done


def cache_row_index(block_idx, pos_idx, num_kv_heads, heads, block_size):
    """Flat row number of each (head, token) in head-major order (``h * T + t``)."""
    h = torch.arange(num_kv_heads, dtype=block_idx.dtype, device=block_idx.device)
    return (
        (block_idx.view(1, -1) * heads + h.view(-1, 1)) * block_size + pos_idx.view(1, -1)
    ).reshape(-1)


def _no_token(like):
    return torch.zeros((_TILE, 1), dtype=torch.int32, device=like.device)


@nki.jit
def _kv_rows_gather_kernel(pool, row_index, after, width: int):
    """``out[128 * t + p] = pool_view[row_index[p, t]]`` with ``pool`` viewed as ``[R, width]``.

    One indirect DMA per tile of 128 rows. Read-only on ``pool``. Indices must be in range.
    ``after`` is a ``[128, 1]`` int32 token: the kernel adds 0 * after to every index, so it
    cannot start before the kernel that produced ``after`` (a cache write) has finished.
    """
    parts, tiles = row_index.shape
    total = 1
    for size in pool.shape:
        total *= int(size)
    assert parts == _TILE and total % width == 0  # noqa: S101
    assert width * _elem_bytes(pool.dtype) <= _MAX_ROW_BYTES  # noqa: S101

    view = pool.reshape((total // width, width))
    out = nl.ndarray((_TILE * tiles, width), dtype=pool.dtype, buffer=nl.shared_hbm)
    idx_bytes = (tiles * 4 + 63) // 64 * 64
    at_base = _SBUF_BASE + idx_bytes
    buf_base = at_base + 256

    idx = nl.ndarray(
        (_TILE, tiles), dtype=nl.int32, buffer=nl.sbuf, name="kvg_idx", address=(0, _SBUF_BASE)
    )
    nisa.dma_copy(dst=idx, src=row_index)
    edge = nl.ndarray(
        (_TILE, 1), dtype=nl.int32, buffer=nl.sbuf, name="kvg_edge", address=(0, at_base + 128)
    )
    nisa.dma_copy(dst=edge, src=after)
    zero = nl.ndarray(
        (_TILE, 1), dtype=nl.int32, buffer=nl.sbuf, name="kvg_zero", address=(0, at_base + 192)
    )
    nisa.tensor_scalar(dst=zero, data=edge, op0=nl.multiply, operand0=0)
    n_buf = min(2, tiles)
    ats = [
        nl.ndarray(
            (_TILE, 1), dtype=nl.int32, buffer=nl.sbuf, name=f"kvg_at{k}",
            address=(0, at_base + 64 * k),
        )
        for k in range(n_buf)
    ]
    bufs = [
        nl.ndarray(
            (_TILE, width), dtype=pool.dtype, buffer=nl.sbuf, name=f"kvg_rows{k}",
            address=(0, buf_base + _MAX_ROW_BYTES * k),
        )
        for k in range(n_buf)
    ]
    for t in range(tiles):
        at, buf = ats[t % 2], bufs[t % 2]
        nisa.tensor_tensor(dst=at, data1=idx[0:_TILE, t : t + 1], data2=zero, op=nl.add)
        nisa.dma_copy(
            dst=buf,
            src=view.ap(
                pattern=[[width, _TILE], [1, width]],
                offset=0,
                vector_offset=at[0:_TILE, 0:1],
                indirect_dim=0,
            ),
        )
        nisa.dma_copy(dst=out[nl.ds(t * _TILE, _TILE), :], src=buf)
    return out


def _row_width(block_elems: int, elem_bytes: int) -> int:
    width = block_elems
    while width * elem_bytes > _MAX_ROW_BYTES:
        width //= 2
    assert block_elems % width == 0
    return width



def write_kv(flat, k_rows, v_rows, block_idx, pos_idx, num_kv_heads, dims):
    """Write new K and V rows into the flat cache in place; return the ordering token.

    ``flat`` is the 1-D tensor over a shared buffer (K plane, then V plane) and ``dims`` is
    ``(n_blocks, heads, block_size, head_dim)`` of the calling layer. The kernel operand is the
    flat tensor itself, never a slice or a differently shaped view of it: a kernel that mutates a
    view makes AOTAutograd write the view back into its base with a pad + arange + where over
    the whole buffer.

    Args:
        k_rows, v_rows: ``[num_kv_heads * T, head_dim]`` new rows, head-major.
        block_idx, pos_idx: ``[T]`` block and in-block position of each new token (pad slots
            already routed to the null block by ``paged_slot_indices``).

    Returns:
        A ``[128, 1]`` zero token that completes after the writes. Pass it to ``gather_kv``.
    """
    n_blocks, heads, block_size, head_dim = dims
    rows_k = cache_row_index(block_idx, pos_idx, num_kv_heads, heads, block_size)
    rows = torch.cat([rows_k, rows_k + n_blocks * heads * block_size])
    values = torch.cat([k_rows, v_rows])
    if not can_run_kernel(flat):
        flat.view(-1, head_dim).index_copy_(0, rows, values)
        return _no_token(flat)

    n = rows.shape[0]
    tiles = -(-n // _TILE)
    pad = tiles * _TILE - n
    # Padding rows get an out-of-range index, so the kernel skips them.
    index = F.pad(rows, (0, pad), value=flat.numel() // head_dim).to(torch.int32)
    index = index.view(tiles, _TILE).t().contiguous()
    grid = 1 if envs.VLLM_NEURON_CPU_MODE else get_logical_nc_config()
    _, token = wrap_nki(_kv_rows_write_kernel)[grid](
        pool=flat, rows=F.pad(values, (0, 0, 0, pad)), row_index=index
    )
    return token


def gather_kv(flat, block_ids, dims, after=None):
    """Return ``(k_blocks, v_blocks)``, each ``[len(block_ids), heads, block_size, head_dim]``.

    Reads with indirect DMA, ordered after the write that produced ``after``. In a decode graph
    that also writes the cache, read it only through kernels: any torch op that reads the cache
    (even a static slice) costs a whole-buffer copy per step.
    """
    n_blocks, heads, block_size, head_dim = dims
    nb = block_ids.shape[0]
    if not can_run_kernel(flat):
        planes = flat.reshape(2, n_blocks, heads, block_size, head_dim)
        return torch.index_select(planes[0], 0, block_ids), torch.index_select(planes[1], 0, block_ids)

    block_elems = heads * block_size * head_dim
    width = _row_width(block_elems, flat.element_size())
    per_block = block_elems // width
    ids = torch.cat([block_ids, block_ids + n_blocks])
    rows = (
        ids.view(-1, 1) * per_block
        + torch.arange(per_block, dtype=ids.dtype, device=ids.device).view(1, -1)
    ).reshape(-1)
    n = rows.shape[0]
    tiles = -(-n // _TILE)
    index = F.pad(rows, (0, tiles * _TILE - n), value=0).to(torch.int32)
    index = index.view(tiles, _TILE).t().contiguous()
    grid = 1 if envs.VLLM_NEURON_CPU_MODE else get_logical_nc_config()
    if after is None:
        after = _no_token(flat)
    out = wrap_nki(_kv_rows_gather_kernel)[grid](
        pool=flat, row_index=index, after=after, width=width
    )
    out = out[:n].view(2, nb, heads, block_size, head_dim)
    return out[0], out[1]
