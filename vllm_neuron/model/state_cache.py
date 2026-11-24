# SPDX-License-Identifier: Apache-2.0
"""
Generic per-request recurrent-state page helpers
================================================

Pure-torch (no vLLM / Neuron imports). Shared by models that declare state
(``StateLayerSpec``) and the runner that allocates and binds it.

A state page packs several tensors back to back, each starting at a 16-byte aligned
offset. Views into a raw int8 allocation are strided (no copy), so a page can be padded
to a larger stride (vLLM requires page sizes across KV groups to divide evenly).
"""

import math
from dataclasses import dataclass

import torch

ALIGN = 16


def _round_up(n: int, m: int) -> int:
    return (n + m - 1) // m * m


@dataclass(frozen=True)
class StatePageLayout:
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype, ...]
    offsets: tuple[int, ...]  # byte offset of each tensor inside the page
    page_bytes: int  # unpadded content size (multiple of ALIGN)


def page_layout(
    shapes: tuple[tuple[int, ...], ...], dtypes: tuple[torch.dtype, ...]
) -> StatePageLayout:
    offsets, cur = [], 0
    for shape, dtype in zip(shapes, dtypes):
        cur = _round_up(cur, ALIGN)
        offsets.append(cur)
        cur += math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    return StatePageLayout(
        tuple(shapes), tuple(dtypes), tuple(offsets), _round_up(cur, ALIGN)
    )


def padded_state_page_bytes(
    state_page_bytes: int, attn_page_bytes: int, align_pages: int = 1
) -> int:
    """Smallest multiple of the attention page that holds the state page.

    vLLM unifies page sizes across KV groups by scaling the *smaller* pages' block size
    by ``max_page // page``; that only works when the ratio is integral, so the state
    page is padded up to an integral multiple of the attention page.
    """
    return _round_up(state_page_bytes, attn_page_bytes * align_pages)


def paired_half_views(
    raw: torch.Tensor, page_bytes: int, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contiguous ``[num_pages, half_elems]`` views of a raw KV-cache tensor (A, B).

    vLLM's hybrid KV manager shares each raw tensor between one attention layer and
    the state layers of the other groups, with disjoint block ids. The attention cache
    lays the raw tensor out as ``(2, num_blocks, ...)`` (all K, then all V), so block
    ``i`` occupies bytes ``[i*P/2, (i+1)*P/2)`` of each half. State page ``i`` must use
    exactly those bytes, so it is stored as two halves: ``A[i]`` (first half) and
    ``B[i]`` (second half). Both views are contiguous, which the Neuron runtime requires
    for graph inputs (strided device-tensor slices are rejected).
    """
    esize = torch.empty((), dtype=dtype).element_size()
    assert raw.numel() % page_bytes == 0 and (page_bytes // 2) % esize == 0
    num_pages = raw.numel() // page_bytes
    halves = raw.view(dtype).view(2, num_pages, page_bytes // 2 // esize)
    return halves[0], halves[1]


def state_half_elems(layout: StatePageLayout, page_bytes: int) -> tuple[int, int]:
    """(total float32 state elements S, elements per half h); requires all-float32 state."""
    assert all(d == torch.float32 for d in layout.dtypes), (
        "paired-half state must be float32"
    )
    total = sum(math.prod(s) for s in layout.shapes)
    half = page_bytes // 2 // 4
    if (total + 1) // 2 > half or total > 2 * half:
        raise ValueError(
            f"state of {total} f32 elements does not fit two halves of {half}"
        )
    return total, half
