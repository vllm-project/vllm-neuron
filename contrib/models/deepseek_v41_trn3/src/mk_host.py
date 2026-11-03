# SPDX-License-Identifier: Apache-2.0
"""Host-side integer bookkeeping for the layer megakernel.

Everything here is index arithmetic over the scheduler's metadata (positions,
block tables, slot mappings); no model arithmetic. The kernel gathers its
window / compressed rows from the paged caches through the flat slot lists
built here and masks empty slots through three per-row columns.

Spans: a list of `(begin, length, start_pos)` over the flat `T` token axis, as
`model._sequence_spans` produces them, plus one block table row per span.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

EMPTY = -1


@dataclass
class AttnIndices:
    """What the kernel takes for one step (CPU int32 tensors unless noted)."""

    row_pos: torch.Tensor      # [T, 1] absolute position of each row
    write_slots: torch.Tensor  # [T, 1] cache slot this step writes each row to
    win_slots: torch.Tensor    # [B*nwin, 1] window-cache slot of every window table row (0 where empty)
    cmp_slots: torch.Tensor    # [B*read_len, 1] compressed-cache slot of every compressed table row
    twin: torch.Tensor         # [T, 1] slot-table row of a row's first window slot
    wv: torch.Tensor           # [T, 1] window validity base: slot j is live iff wv + j >= 0
    cbase: torch.Tensor        # [T, 1] slot-table row of a span's first compressed slot
    nwin: int                  # window rows per span (window + L - 1)
    n_per_span: int            # slot-table rows per span (nwin + read_len)
    clen: torch.Tensor = None       # [T, 1] compress_len of each row ((pos + 1) // ratio)
    grp_pos: torch.Tensor = None    # [B*G, 1] position of each compressed group's first token
    grp_mslots: torch.Tensor = None  # [B*G*ratio, 1] member slots of each group (pooling inputs)
    grp_wslots: torch.Tensor = None  # [B*G, 1] cache slot each group's latent is written to
    groups_per_span: int = 0        # G (padded groups write slot 0, the null block)


def slots_of(block_table: torch.Tensor, positions: torch.Tensor, block_size: int) -> torch.Tensor:
    """Flat cache slots of absolute `positions` (CPU int64) under `block_table`."""
    return block_table[positions // block_size] * block_size + positions % block_size


def build_attn_indices(
    spans: list[tuple[int, int, int]],
    block_tables: list[torch.Tensor],
    slot_mapping: torch.Tensor,
    block_size: int,
    window: int,
    read_len: int = 0,
    ratio: int = 0,
    total_rows: int | None = None,
) -> AttnIndices:
    """Index tensors for `spans` of equal length `L` (decode batch) or one span (prefill).

    Window rows of span `b` are the cache rows at positions
    `start_pos - (window - 1) + i, i < window + L - 1`, clamped to slot 0 where the
    position is negative (those slots are masked by `wv`). Compressed rows, when
    `ratio > 0`, are the `read_len` group positions `min(arange(read_len),
    compress_len - 1) * ratio` (the model's `_compress_kv` read), where
    `compress_len = (start_pos + L) // ratio`.
    """
    lengths = {length for _, length, _ in spans}
    if len(lengths) != 1:
        raise ValueError(f"spans must share one length, got {sorted(lengths)}")
    L = lengths.pop()
    T = total_rows if total_rows is not None else sum(length for _, length, _ in spans)
    nwin = window + L - 1
    n = nwin + (read_len if ratio else 0)
    B = len(spans)
    row_pos = torch.zeros(T, dtype=torch.int64)
    twin = torch.zeros(T, dtype=torch.int64)
    wv = torch.zeros(T, dtype=torch.int64)
    cbase = torch.zeros(T, dtype=torch.int64)
    wtab = torch.zeros(B * nwin, dtype=torch.int64)
    ctab = torch.zeros(max(1, B * (read_len if ratio else 0)), dtype=torch.int64)
    slot_mapping = slot_mapping.reshape(-1).to(torch.int64)
    for b, ((begin, length, start_pos), table) in enumerate(zip(spans, block_tables)):
        table = table.to(torch.int64).clamp_min(0)
        t = torch.arange(length)
        row_pos[begin : begin + length] = start_pos + t
        twin[begin : begin + length] = b * n + t
        wv[begin : begin + length] = start_pos - (window - 1) + t
        cbase[begin : begin + length] = b * n + nwin
        wpos = start_pos - (window - 1) + torch.arange(nwin)
        wtab[b * nwin : (b + 1) * nwin] = slots_of(table, wpos.clamp_min(0), block_size)
        if ratio:
            compress_len = (start_pos + length) // ratio
            gpos = torch.minimum(
                torch.arange(read_len), torch.tensor(max(compress_len - 1, 0))
            ) * ratio
            ctab[b * read_len : (b + 1) * read_len] = slots_of(table, gpos, block_size)
    ws = torch.zeros(T, dtype=torch.int64)
    ws[: slot_mapping.numel()] = slot_mapping[:T]
    i32 = lambda x: x.to(torch.int32).reshape(-1, 1).contiguous()  # noqa: E731
    out = AttnIndices(
        row_pos=i32(row_pos), write_slots=i32(ws), win_slots=i32(wtab), cmp_slots=i32(ctab),
        twin=i32(twin), wv=i32(wv), cbase=i32(cbase), nwin=nwin, n_per_span=n,
    )
    if ratio:
        out.clen = i32((row_pos + 1) // ratio)
        G = L if ratio == 1 else L // ratio + 1
        gpos = torch.zeros(B * G, dtype=torch.int64)
        gm = torch.zeros(B * G * ratio, dtype=torch.int64)
        gw = torch.zeros(B * G, dtype=torch.int64)
        for b, ((begin, length, start_pos), table) in enumerate(zip(spans, block_tables)):
            table = table.to(torch.int64).clamp_min(0)
            if ratio == 1:
                firsts = start_pos + torch.arange(length)
            else:
                g0, g1 = start_pos // ratio, (start_pos + length) // ratio
                firsts = torch.arange(g0, g1) * ratio
            c = firsts.numel()
            gpos[b * G : b * G + c] = firsts
            gw[b * G : b * G + c] = ws[begin : begin + length] if ratio == 1 else slots_of(table, firsts, block_size)
            members = (firsts.unsqueeze(1) + torch.arange(ratio)).reshape(-1)
            gm[b * G * ratio : b * G * ratio + c * ratio] = slots_of(table, members, block_size)
        out.grp_pos, out.grp_mslots, out.grp_wslots, out.groups_per_span = i32(gpos), i32(gm), i32(gw), G
    return out


def replica_groups(world_size: int, group_size: int) -> tuple[tuple[int, ...], ...]:
    """Consecutive rank groups of `group_size`, as the kernel wants them (a constant)."""
    return tuple(
        tuple(range(g * group_size, (g + 1) * group_size))
        for g in range(world_size // group_size)
    )
