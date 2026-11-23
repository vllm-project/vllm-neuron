# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the Neuron KV connectors for registering KV memory with an EFA transport.

Coalesced registration
----------------------
The Neuron driver hands EFA 2 MiB pages for a registration only when the registration is at
least 256 MiB and its length, virtual address and physical address are all 2 MiB-aligned
(``neuron_p2p_register_va`` in aws-neuronx-dkms, ``NEURON_P2P_HUGE_PAGE_SZ_USAGE_THRESHOLD``);
anything else is mapped with 4 KiB pages. The EFA driver takes the page size from that call
(``efa_neuronmem.c``), so every 4 KiB page costs one entry in the NIC's page table.

A KV cache is many buffers of a few hundred MiB or less (a layer's K/V storage, or each K and V
slab), none of which qualifies. At TP=64 on a trn2.48xlarge each EFA NIC serves the ranks of two
Neuron devices, and per-buffer registration exhausts the NIC page tables: ``REG_MR err -12``
(ENOMEM) in dmesg, failed registrations, or buffers a transfer cannot use.

All KV buffers of a rank live inside a few large ``/dev/neuron<N>`` mappings, and the driver
accepts any range inside one mapping. So the connectors register one 2 MiB-aligned span per
mapping instead (grown to 256 MiB where the mapping allows), which takes the 2 MiB pages: one
~15 GiB registration in a fraction of a second instead of hundreds of small ones over minutes.
A span may also cover other allocations in the same mapping; they are registered, never the
target of a transfer. Set ``VLLM_NEURON_KV_COALESCE_REGISTRATION=0`` to register buffer by buffer.

Extra per-layer tensors
-----------------------
A model module may expose ``CONNECTOR_EXTRA_KV_TENSORS = {layer name: tensor}`` for tensors that
are blocks-first, indexed by the same block ids as the layer's KV cache and written by prefill
(for example sparse-attention K summaries). The connectors register each one as one more region
of its layer, so it moves block by block with the layer's K and V. Both sides of a transfer must
expose the same tensors.
"""

from __future__ import annotations

import os
import sys

HUGE_PAGE = 2 << 20
HUGE_PAGE_MIN_REGISTRATION = 256 << 20


def coalesce_enabled() -> bool:
    return os.environ.get("VLLM_NEURON_KV_COALESCE_REGISTRATION", "1") != "0"


def neuron_mappings() -> list[tuple[int, int]]:
    """``[(start, end)]`` of this process's ``/dev/neuron`` mappings."""
    out = []
    with open("/proc/self/maps") as f:
        for line in f:
            if "/dev/neuron" in line:
                start, end = (int(x, 16) for x in line.split()[0].split("-"))
                out.append((start, end))
    return sorted(out)


def coalesce_neuron_spans(
    ptrs: list[int], lens: list[int], mappings: list[tuple[int, int]] | None = None
) -> tuple[list[int], list[int]]:
    """Replace the buffers that live in Neuron device mappings by one span per mapping.

    Each span covers the mapping's buffers, aligned to 2 MiB and grown to 256 MiB when the
    mapping allows, and never leaves the mapping. Buffers outside every Neuron mapping (host
    memory) are returned unchanged.

    Example:
        >>> coalesce_neuron_spans([0x40100000, 0x48000000], [1 << 20, 1 << 20],
        ...                       mappings=[(0x40000000, 0x80000000)])
        ([1073741824], [268435456])
    """
    if mappings is None:
        mappings = neuron_mappings()
    groups: dict[tuple[int, int], list[int]] = {}
    out_ptrs: list[int] = []
    out_lens: list[int] = []
    for ptr, length in zip(ptrs, lens):
        mapping = next((m for m in mappings if m[0] <= ptr and ptr + length <= m[1]), None)
        if mapping is None:
            out_ptrs.append(ptr)
            out_lens.append(length)
            continue
        lo, hi = groups.get(mapping, [ptr, ptr + length])
        groups[mapping] = [min(lo, ptr), max(hi, ptr + length)]
    for (map_lo, map_hi), (lo, hi) in sorted(groups.items()):
        lo = max(map_lo, lo // HUGE_PAGE * HUGE_PAGE)
        hi = min(map_hi, -(-hi // HUGE_PAGE) * HUGE_PAGE)
        if hi - lo < HUGE_PAGE_MIN_REGISTRATION:
            hi = min(map_hi, lo + HUGE_PAGE_MIN_REGISTRATION)
            lo = max(map_lo, hi - HUGE_PAGE_MIN_REGISTRATION)
        out_ptrs.append(lo)
        out_lens.append(hi - lo)
    return out_ptrs, out_lens


def extra_kv_tensors() -> dict:
    """``{layer name: tensor}`` from every loaded module's ``CONNECTOR_EXTRA_KV_TENSORS``."""
    out: dict = {}
    for module in list(sys.modules.values()):
        extra = getattr(module, "CONNECTOR_EXTRA_KV_TENSORS", None)
        if isinstance(extra, dict):
            out.update(extra)
    return out
