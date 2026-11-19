# SPDX-License-Identifier: Apache-2.0
"""The GDN recurrent state, stored in vLLM's KV cache and addressed by each request's first block.

vLLM Neuron's runner has no recurrent-state (Mamba) cache, so the state is declared as two extra KV-cache layers
(``state.rec`` and ``state.conv``, ``Qwen38ForCausalLM.get_kv_spec``) in the same cache group as the attention
layers. They share the attention layers' block table, so every block a request owns also owns a region of state,
and the region of the request's FIRST block (``block_table[r, 0]``) holds its state. With the prefix cache off
(required, ``factory.py``) no two live requests share a block, so no two share a state region; the vLLM KV
manager budgets the memory and frees it with the request. Block 0 is vLLM's null block, never a request's first
block: padded decode rows read their state from it and their writes are skipped, so it stays zero.

Per rank and per GDN layer a request's state is the temporal state ``[128 k, vh, 128 v]`` fp32 in the kernels'
layout and the conv window ``[Cc, 3]`` bf16. Each state layer has a K and a V plane ``[N, block, width]``; the
first half of the GDN layers lives in the K plane and the second half in the V plane. Inside one block of a plane
the layers' records are consecutive.

Every access is one indirect-DMA kernel over a whole plane (``rows.py``): a decode step gathers the batch's records
of all layers before the first layer and writes the advanced records back after the last.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import rows as R

REC_ROW_BYTES = 8192


def _ceil_div(a, b):
    return -(-a // b)


@dataclass(frozen=True)
class StateLayout:
    """Byte geometry of the two state layers for one rank."""

    gdn_layers: int
    v_heads: int
    conv_channels: int
    block_size: int
    head_k: int = 128
    head_v: int = 128
    conv_window: int = 3

    @property
    def half(self) -> int:
        return self.gdn_layers // 2

    @property
    def rec_elems(self) -> int:
        """fp32 elements of one layer's temporal state for one request."""
        return self.head_k * self.v_heads * self.head_v

    @property
    def conv_elems(self) -> int:
        return self.conv_channels * self.conv_window

    @property
    def rec_width(self) -> int:
        """fp32 elements per DMA row of the temporal state (the widest divisor of a record within 8 KB)."""
        for w in range(min(self.rec_elems, REC_ROW_BYTES // 4), 0, -1):
            if self.rec_elems % w == 0:
                return w
        raise ValueError(self.rec_elems)

    @property
    def conv_width(self) -> int:
        for w in range(min(self.conv_elems, R.MAX_ROW_BYTES // 2), 0, -1):
            if self.conv_elems % w == 0:
                return w
        raise ValueError(self.conv_elems)

    def row_elems(self, which: str) -> int:
        """Elements per DMA row of a plane, in the state's dtype (fp32 temporal, bf16 conv)."""
        return self.rec_width if which == "rec" else self.conv_width

    def plane_rows(self, which: str, plane: torch.Tensor) -> int:
        """DMA rows in a bf16 plane."""
        row_bytes = self.rec_width * 4 if which == "rec" else self.conv_width * 2
        return plane.numel() * plane.element_size() // row_bytes

    def head_size(self, which: str) -> int:
        """The bf16 ``head_size`` of the state layer's spec (one KV head, ``block_size`` tokens) that holds half
        of the layers' records per block in each of its two planes."""
        record_bytes = self.rec_elems * 4 if which == "rec" else self.conv_elems * 2
        row_bytes = self.rec_width * 4 if which == "rec" else self.conv_width * 2
        size = _ceil_div(self.half * record_bytes, 2 * self.block_size)
        # a plane's block must be a whole number of DMA rows, so that row ids are block-aligned
        while (2 * self.block_size * size) % row_bytes:
            size += 1
        return size

    def block_stride(self, which: str) -> int:
        """Rows of one block in a plane viewed as ``[R, width]``."""
        plane_bytes = self.block_size * self.head_size(which) * 2
        row_bytes = self.rec_width * 4 if which == "rec" else self.conv_width * 2
        if plane_bytes % row_bytes:
            raise ValueError(f"a {which} plane block of {plane_bytes} bytes is not a whole number of "
                             f"{row_bytes}-byte rows; pick another block size")
        return plane_bytes // row_bytes

    def record_rows(self, which: str) -> int:
        return self.rec_elems // self.rec_width if which == "rec" else self.conv_elems // self.conv_width


def plane_views(kv: list, which: str):
    """The two planes of a state layer's cache as 1-D bf16 tensors (the runner's dtype), built once outside the
    graph. The fp32 temporal state is reinterpreted inside the row kernels only (``rows.gather_rows``).

    Each plane is the only tensor the model keeps over its bytes: a second, differently shaped view of a buffer
    a kernel writes makes the compiler write the buffer back in full (Gemma 4 contrib README)."""
    out = []
    for plane in kv[:2]:
        assert plane.is_contiguous(), "state planes must be contiguous"
        assert plane.dtype == torch.bfloat16, f"state planes must be bf16, got {plane.dtype}"
        out.append(plane.reshape(-1))
    return out


def record_index(layout: StateLayout, which: str, slots: torch.Tensor, skip: torch.Tensor | None = None,
                 fill: int = 0) -> torch.Tensor:
    """The rows of the batch's records in one plane, ordered (layer, batch row, row), as the row kernels'
    ``[128, tiles]`` int32 index: row ``n`` of the order at ``[n % 128, n // 128]``. Built from a position grid
    in that layout, not by transposing a flat list, so the graph holds no integer transpose. Positions past the
    records, and rows of batch rows with ``skip`` set, get ``fill``."""
    stride = layout.block_stride(which)
    per = layout.record_rows(which)
    batch = slots.shape[0]
    n = layout.half * batch * per
    tiles = -(-n // R.TILE)
    dev = slots.device
    pos = (torch.arange(R.TILE, dtype=torch.int32, device=dev).view(-1, 1)
           + R.TILE * torch.arange(tiles, dtype=torch.int32, device=dev).view(1, -1))
    j = torch.div(pos, batch * per, rounding_mode="floor")
    b = torch.remainder(torch.div(pos, per, rounding_mode="floor"), batch)
    r = torch.remainder(pos, per)
    s = slots.to(torch.int32).index_select(0, b.reshape(-1)).view(R.TILE, tiles)
    rows = s * stride + j * per + r
    out = pos >= n
    if skip is not None:
        out = out | (skip.to(torch.int32).index_select(0, b.reshape(-1)).view(R.TILE, tiles) > 0)
    return torch.where(out, torch.full_like(rows, fill), rows)


class StateBank:
    """The batch's state records of every GDN layer, gathered once per forward and written back once.

    ``slots`` ``[B]`` are the rows' first blocks; a row with a negative slot (padding) reads the null block and
    is not written back."""

    def __init__(self, layout: StateLayout, planes: dict, slots: torch.Tensor, reset: torch.Tensor | None = None):
        self.layout, self.planes = layout, planes
        self.batch = slots.shape[0]
        read = torch.clamp(slots, min=0)
        self.count = {w: layout.half * self.batch * layout.record_rows(w) for w in ("rec", "conv")}
        self.read_index = {w: record_index(layout, w, read) for w in ("rec", "conv")}
        # padding rows (negative slot) index past the plane, which the write kernel skips
        self.write_index = {w: record_index(layout, w, read, skip=slots < 0,
                                            fill=layout.plane_rows(w, planes[w][0])) for w in ("rec", "conv")}
        B, half, L = self.batch, layout.half, layout
        self.rec, self.conv = [], []
        for p in range(2):
            rec = R.gather_rows(planes["rec"][p], self.read_index["rec"], self.count["rec"], L.rec_width,
                                 dtype=torch.float32)
            rec = rec.view(half, B, L.head_k, L.v_heads, L.head_v)
            conv = R.gather_rows(planes["conv"][p], self.read_index["conv"], self.count["conv"], L.conv_width)
            conv = conv.view(half, B, L.conv_channels, L.conv_window)
            if reset is not None:
                # a request's first chunk starts from zero state, whatever its block held before
                rec = rec * reset.to(rec.dtype)
                conv = conv * reset.to(conv.dtype)
            self.rec.append(rec)
            self.conv.append(conv)
        self.new_rec = [[None] * half, [None] * half]
        self.new_conv = [[None] * half, [None] * half]

    def get(self, g: int):
        p, j = divmod(g, self.layout.half)
        return self.conv[p][j], self.rec[p][j]

    def put(self, g: int, conv: torch.Tensor, rec: torch.Tensor) -> None:
        p, j = divmod(g, self.layout.half)
        self.new_conv[p][j], self.new_rec[p][j] = conv, rec

    def publish(self) -> None:
        L = self.layout
        for p in range(2):
            rec = torch.stack(self.new_rec[p]).to(torch.float32).reshape(-1, L.rec_width)
            conv = torch.stack(self.new_conv[p]).reshape(-1, L.conv_width)
            R.write_rows(self.planes["rec"][p], rec, self.write_index["rec"])
            R.write_rows(self.planes["conv"][p], conv.to(self.planes["conv"][p].dtype), self.write_index["conv"])
