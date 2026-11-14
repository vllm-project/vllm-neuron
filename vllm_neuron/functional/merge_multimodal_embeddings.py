# SPDX-License-Identifier: Apache-2.0
"""Scatter encoder embeddings into hidden_states for prefill.

Shared by every encoder model (Qwen3-VL image/video, Qwen3-ASR audio). Deepstack is
optional: a model with no deepstack (N_ds == 0) gets the main scatter only.

Handles sequence-parallel (SP) coordinate remapping and deepstack
construction. Used by the prefill graph to merge on-device encoder
cache blocks into the token embedding sequence.

Two implementations share one entry point, ``merge_multimodal_embeddings``:

* ``_merge_multimodal_embeddings_torch`` -- portable torch reference. Gathers the
  needed cache blocks with a single ``index_select`` and scatters with
  ``index_put_``. Runs on CPU (so the unit tests exercise it) and is the
  default everywhere until the NKI kernel below is validated on device.
* ``_merge_multimodal_embeddings_nki`` -- a NKI kernel that fuses the gather and
  the scatter so the compiler never materializes the ``[B, block_size,
  fat_dim]`` intermediate and never emits the ``stack_concatenate`` /
  per-block ``DMA_DIRECTED2D`` trail. Selected on device via ``can_run_kernel``
  (see the dispatch in ``merge_multimodal_embeddings``).

The NKI kernel is validated against the torch reference by
``test_merge_multimodal_embeddings_nki.py`` — parity is exact (pure data movement) on
both the NKI simulator and Trn3 hardware, across single/multi block,
scratch/padding, all-sentinel, SP rank shift + out-of-shard drop, multiple
deepstack levels, block_size 256/1024, and the no-deepstack case. The CPU unit
tests additionally cover the torch reference path.
"""

from __future__ import annotations

import torch

from vllm_neuron.utils.neuron_utils import can_run_kernel

# NKI is always available in the Neuron build/test environment (same top-level
# import convention as vllm_neuron/functional/**). Kept module-level so the
# kernel is traced once.
import nki
import nki.isa as nisa
import nki.language as nl
from vllm_neuron.utils.hardware_config import get_logical_nc_config
from nki.isa import oob_mode

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.core.utils.kernel_helpers import div_ceil
from nkilib.core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast


def global_to_local_positions(
    positions: torch.Tensor, local_len: int, rank: int | torch.Tensor
) -> torch.Tensor:
    """Remap global batch positions to an SP rank's local coordinates.

    Positions outside this rank's shard [rank*local_len, (rank+1)*local_len)
    become the sentinel value (local_len), which writes to a dummy row
    during index_put_.

    Example:
        >>> import torch
        >>> global_to_local_positions(torch.tensor([4, 5, 9]), local_len=4, rank=1)
        tensor([0, 1, 4])
    """
    local_start = rank * local_len
    local_positions = positions - local_start
    out_of_range = (local_positions < 0) | (local_positions >= local_len)
    return torch.where(out_of_range, local_len, local_positions)


def scatter_with_dummy_row(
    target: torch.Tensor,
    positions: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Scatter values into target at positions using a dummy row for sentinels.

    Positions equal to target.shape[0] write to an appended dummy row
    that is discarded after the scatter. This avoids branching on
    sentinel values inside torch.compile-traced graphs.

    Example:
        >>> import torch
        >>> out = scatter_with_dummy_row(
        ...     torch.zeros(3, 2), torch.tensor([0, 3]), torch.ones(2, 2))
        >>> out[0]  # row 0 written; the position-3 write hits the discarded dummy row
        tensor([1., 1.])
    """
    dummy = torch.zeros(1, target.shape[-1], dtype=target.dtype, device=target.device)
    with_dummy = torch.cat([target, dummy], dim=0)
    with_dummy.index_put_((positions,), values.to(target.dtype))
    return with_dummy[: target.shape[0]]


def merge_multimodal_embeddings(
    hidden_states: torch.Tensor,
    encoder_embedding_buffer: torch.Tensor,
    encoder_block_ids: torch.Tensor,
    encoder_positions: torch.Tensor,
    rank: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Scatter encoder embeddings from cache blocks into hidden_states.

    Dispatches to the NKI kernel on device (when ``can_run_kernel`` is True and
    the shapes are supported) and to the torch reference otherwise -- including
    on CPU, so the unit tests exercise the reference path. Both paths are
    numerically equivalent; see the module docstring for the rationale.

    Args:
        hidden_states: [local_len, hidden_dim] from embed_tokens (SP-sharded).
        encoder_embedding_buffer: [num_blocks, block_size, fat_dim] cache buffer.
        encoder_block_ids: [max_num_encoder_blocks] int32 block indices to gather
            (padded with the scratch block id for unused slots).
        encoder_positions: [max_num_encoder_blocks, block_size] global positions.
        rank: SP rank for coordinate remapping.

    Returns:
        (hidden_states, deepstack_embeds) where:
        - hidden_states: [local_len, hidden_dim] with main embeds scattered in
        - deepstack_embeds: [N_ds, local_len, hidden_dim] or None

    Example:
        >>> import torch
        >>> hidden = torch.zeros(8, 4)                    # [local_len, hidden_dim]
        >>> buffer = torch.arange(16.0).reshape(1, 4, 4)  # one block, no deepstack
        >>> block_ids = torch.zeros(1, dtype=torch.int32)
        >>> positions = torch.tensor([[2, 3, 4, 5]], dtype=torch.int32)
        >>> out, deepstack = merge_multimodal_embeddings(
        ...     hidden, buffer, block_ids, positions, rank=0)
        >>> deepstack is None
        True
    """
    if _can_use_kernel(hidden_states, encoder_embedding_buffer):
        return _merge_multimodal_embeddings_nki_wrapper(
            hidden_states,
            encoder_embedding_buffer,
            encoder_block_ids,
            encoder_positions,
            rank,
        )
    return _merge_multimodal_embeddings_torch(
        hidden_states,
        encoder_embedding_buffer,
        encoder_block_ids,
        encoder_positions,
        rank,
    )


def _num_deepstack_levels(fat_dim: int, hidden_dim: int) -> int:
    return (fat_dim - hidden_dim) // hidden_dim


def _can_use_kernel(
    hidden_states: torch.Tensor, encoder_embedding_buffer: torch.Tensor
) -> bool:
    """Gate the NKI path on device availability and supported shapes.

    Unsupported shapes fall back to the (equivalent) torch reference rather
    than crashing on device, so enabling the kernel can never regress
    correctness for a config it does not handle.
    """
    if not can_run_kernel(hidden_states):
        return False

    # block_size is tiled over the partition dim in pmax (=128) chunks, so any
    # block_size is supported (the real config uses 1024). N_ds == 0 (no
    # deepstack) is also supported: the kernel still gathers/scatters the main
    # embedding and returns None for deepstack.
    if hidden_states.dtype != torch.bfloat16:
        return False

    # The kernel slices deepstack as hidden_dim-wide columns, so fat_dim must be
    # an exact (1 + N_ds) multiple of hidden_dim. The torch reference enforces
    # this implicitly via reshape(-1, N_ds, hidden_dim); guard here so any odd
    # config falls back to the reference (which raises) rather than silently
    # dropping a trailing partial deepstack in the kernel.
    hidden_dim = hidden_states.shape[-1]
    fat_dim = encoder_embedding_buffer.shape[-1]
    if (fat_dim - hidden_dim) % hidden_dim != 0:
        return False
    return True


def _merge_multimodal_embeddings_torch(
    hidden_states: torch.Tensor,
    encoder_embedding_buffer: torch.Tensor,
    encoder_block_ids: torch.Tensor,
    encoder_positions: torch.Tensor,
    rank: int | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Portable reference: single index_select gather + index_put_ scatter."""
    gathered = encoder_embedding_buffer.index_select(0, encoder_block_ids)
    flat_embeds = gathered.reshape(-1, gathered.shape[-1])
    positions_flat = encoder_positions.reshape(-1)

    hidden_dim = hidden_states.shape[-1]
    local_len = hidden_states.shape[0]
    local_positions = global_to_local_positions(positions_flat, local_len, rank)

    # Scatter main encoder embeddings
    main_embeds = flat_embeds[:, :hidden_dim]
    hidden_states = scatter_with_dummy_row(hidden_states, local_positions, main_embeds)

    # Build deepstack in local coordinates
    deepstack_embeds = None
    fat_dim = flat_embeds.shape[-1]
    if fat_dim > hidden_dim:
        N_ds = (fat_dim - hidden_dim) // hidden_dim
        ds_embeds = flat_embeds[:, hidden_dim:].reshape(-1, N_ds, hidden_dim)
        deepstack_embeds = torch.zeros(
            N_ds,
            local_len,
            hidden_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        for n in range(N_ds):
            deepstack_embeds[n] = scatter_with_dummy_row(
                deepstack_embeds[n], local_positions, ds_embeds[:, n, :]
            )

    return hidden_states, deepstack_embeds


def _merge_multimodal_embeddings_nki_wrapper(
    hidden_states: torch.Tensor,
    encoder_embedding_buffer: torch.Tensor,
    encoder_block_ids: torch.Tensor,
    encoder_positions: torch.Tensor,
    rank: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Host-side glue: normalize dtypes and invoke the traced kernel.

    The caller can pass a runtime rank tensor. This keeps the local offset
    identical in every traced graph.
    """
    local_len = hidden_states.shape[0]
    if not isinstance(rank, torch.Tensor):
        raise TypeError(
            "The NKI encoder merge requires a runtime rank tensor. "
            "Pass group.rank_in_group_tensor."
        )
    local_offset = rank.to(torch.int32) * local_len
    hidden_dim = hidden_states.shape[-1]
    fat_dim = encoder_embedding_buffer.shape[-1]
    n_ds = _num_deepstack_levels(fat_dim, hidden_dim)

    # encoder_block_ids / encoder_positions are already int32 (the DMA address
    # pattern needs int32, and the runner constructs them narrow -- see
    # NeuronModelRunner._gather_mm_embeddings). No cast here keeps the graph
    # free of an on-device dtype conversion.
    wrapped = wrap_nki(_merge_multimodal_embeddings_nki)
    # The kernel returns (hidden, deepstack) when n_ds > 0, else just hidden --
    # a compile-time branch on the static n_ds (see the kernel). Unpack to match.
    if n_ds > 0:
        out_hidden, out_ds = wrapped[get_logical_nc_config()](
            hidden_states,
            encoder_embedding_buffer,
            encoder_block_ids,
            encoder_positions,
            local_offset=local_offset,
        )
        return out_hidden, out_ds
    out_hidden = wrapped[get_logical_nc_config()](
        hidden_states,
        encoder_embedding_buffer,
        encoder_block_ids,
        encoder_positions,
        local_offset=local_offset,
    )
    return out_hidden, None


@nki.jit
def _merge_multimodal_embeddings_nki(
    hidden_states,
    buffer,
    block_ids,
    encoder_positions,
    local_offset,
):
    """Fused gather + scatter merge (validated on sim + Trn3; see module docstring).

    For each cache block b and each partition-tile pt of ``pmax`` tokens
    (``block_size`` is tiled because it exceeds the 128-partition limit — the
    real config uses 1024):
      1. gather ``buffer[block_ids[b]]`` rows ``block_ids[b]*block_size + tok``
         for this tile -> SBUF ``[rows, fat_dim]`` with one indirect DMA,
      2. compute local dst rows ``encoder_positions[b, tile] - local_offset``,
      3. scatter the main slice into ``out_hidden`` and each deepstack slice
         into ``out_ds[n]`` with ``oob_mode.skip`` dropping out-of-shard /
         sentinel rows.

    No ``[B, block_size, fat_dim]`` intermediate and no concatenate are ever
    built, which is what removes the ``DMA_DIRECTED2D`` trail.
    """
    num_blocks, block_size, fat_dim = buffer.shape
    local_len, hidden_dim = hidden_states.shape
    B = block_ids.shape[0]
    N_ds = (fat_dim - hidden_dim) // hidden_dim
    pmax = nl.tile_size.pmax
    n_ptiles = div_ceil(block_size, pmax)

    # NOTE(block_size < pmax): correctness is unaffected -- ``rows`` clamps to
    # ``block_size`` below, so a partial tile only touches the live rows. But a
    # block_size < 128 under-utilizes each DMA (only block_size of the 128
    # partitions are moved per packet). If a future config lowers block_size to
    # shrink padding (block_size == 1 removes it entirely; any block_size that
    # divides the merged encoder length has none), this loop leaves throughput on the
    # table. Optimization: when block_size < pmax, pack floor(pmax / block_size)
    # blocks into a single [pmax, fat_dim] DMA packet instead of one packet per
    # block. Not done here because the production config uses block_size 1024.

    # Outputs live in HBM. out_hidden is seeded with the token embeddings (all
    # rows valid); out_ds (only when there are deepstack levels) starts zeroed
    # because the scatter only writes the rows that carry an encoder token.
    out_hidden = nl.ndarray(
        hidden_states.shape, hidden_states.dtype, buffer=nl.shared_hbm
    )
    nisa.dma_copy(out_hidden, hidden_states)  # HBM->HBM (validated on sim + device)

    # Each deepstack level is its OWN standalone HBM tensor so oob_mode.skip
    # bounds dropped/sentinel rows to [0, local_len) per level. A shared
    # [N_ds, local_len, hidden_dim] tensor sliced as out_ds[n] does NOT re-bound
    # the skip check, so an OOB row (index == local_len) would leak into the
    # next level's row 0. Assembled into the [N_ds, ...] output at the end.
    ds_levels = []
    for n in range(N_ds):  # compile-time (N_ds derived from static shapes)
        lvl = nl.ndarray(
            (local_len, hidden_dim), hidden_states.dtype, buffer=nl.shared_hbm
        )
        _zero_hbm(lvl, hidden_dim)
        ds_levels.append(lvl)

    # Flat view of the cache: buffer[b, t, :] == buffer_flat[b*block_size + t, :].
    buffer_flat = buffer.reshape((num_blocks * block_size, fat_dim))

    # Load all block ids into partition 0 once; per-block scalar picked below.
    block_ids_sb = nl.ndarray((1, B), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(block_ids_sb[0, :], block_ids.reshape((1, B)))
    local_offset_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(local_offset_sb[0, :], local_offset)
    local_offset_broadcast = nl.ndarray((pmax, 1), dtype=nl.int32, buffer=nl.sbuf)
    stream_shuffle_broadcast(src=local_offset_sb, dst=local_offset_broadcast)

    for b in range(B):  # B (max_encoder_blocks) is a compile-time constant
        # block_ids[b] * block_size, broadcast across pmax partitions once.
        blk = nl.ndarray((pmax, 1), dtype=nl.int32, buffer=nl.sbuf)
        stream_shuffle_broadcast(src=block_ids_sb[0:1, b : b + 1], dst=blk)
        base = nl.ndarray((pmax, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(data=blk, op0=nl.multiply, operand0=block_size, dst=base)

        for pt in range(n_ptiles):  # tile block_size over the partition dim
            row0 = pt * pmax
            rows = min(pmax, block_size - row0)  # compile-time (block_size static)

            # src_flat[t] = block_ids[b]*block_size + (row0 + t), t in [0, rows).
            # iota fills the free dim and broadcasts across partitions; to vary
            # along the partition dim use channel_multiplier: value(p)=row0 + p.
            t_iota = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.iota(dst=t_iota, pattern=[[0, 1]], offset=row0, channel_multiplier=1)
            src_flat = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=src_flat, data1=base[0:rows, :], data2=t_iota, op=nl.add
            )

            # Gather this tile's rows: [rows, fat_dim].
            blk_sb = nl.ndarray((rows, fat_dim), dtype=buffer.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                src=buffer_flat.ap(
                    pattern=[[fat_dim, rows], [1, fat_dim]],
                    offset=0,
                    vector_offset=src_flat,
                    indirect_dim=0,
                ),
                dst=blk_sb,
                name=f"merge_multimodal_gather_b{b}_pt{pt}",
            )

            # Local destination rows: pos - local_offset. Values outside
            # [0, local_len) (other SP shards + sentinel) are dropped by scatter.
            pos = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(
                pos,
                encoder_positions[b, row0 : row0 + rows]
                .reshape((1, rows))
                .permute((1, 0)),
            )  # transpose-on-load into the partition dim
            nisa.tensor_tensor(
                data1=pos,
                data2=local_offset_broadcast[0:rows, :],
                op=nl.subtract,
                dst=pos,
            )

            # Scatter main embedding (first hidden_dim columns).
            nisa.dma_copy(
                src=blk_sb[:, 0:hidden_dim],
                dst=out_hidden.ap(
                    pattern=[[hidden_dim, rows], [1, hidden_dim]],
                    offset=0,
                    vector_offset=pos,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip,
                name=f"merge_multimodal_scatter_main_b{b}_pt{pt}",
            )

            # Scatter each deepstack level (cols hidden_dim*(1+n):hidden_dim*(2+n)).
            for n in range(N_ds):
                lo = hidden_dim * (1 + n)
                nisa.dma_copy(
                    src=blk_sb[:, lo : lo + hidden_dim],
                    dst=ds_levels[n].ap(
                        pattern=[[hidden_dim, rows], [1, hidden_dim]],
                        offset=0,
                        vector_offset=pos,
                        indirect_dim=0,
                    ),
                    oob_mode=oob_mode.skip,
                    name=f"merge_multimodal_scatter_ds{n}_b{b}_pt{pt}",
                )

    # Match the arity the wrapper unpacks: (hidden, deepstack) with deepstack,
    # else just hidden. Compile-time branch on the static N_ds.
    if N_ds > 0:
        # Assemble the standalone levels into [N_ds, local_len, hidden_dim].
        out_ds = nl.ndarray(
            (N_ds, local_len, hidden_dim), hidden_states.dtype, buffer=nl.shared_hbm
        )
        for n in range(N_ds):
            nisa.dma_copy(out_ds[n], ds_levels[n])
        return out_hidden, out_ds
    return out_hidden


def _zero_hbm(hbm_2d, cols):
    """Zero an HBM ``[rows, cols]`` tensor in pmax-row tiles.

    Runs during kernel setup (before the block loop), so its ``zero_sb`` SBUF
    tile is not co-resident with the block loop's ``blk_sb``. ``memset`` runs
    once before the tile loop; the loop only reads ``zero_sb`` as a DMA source
    and never writes it, so the single memset stays valid for every tile.
    """
    rows = hbm_2d.shape[0]
    pmax = nl.tile_size.pmax
    zero_sb = nl.ndarray((pmax, cols), dtype=hbm_2d.dtype, buffer=nl.sbuf)
    nisa.memset(zero_sb, 0)
    num_tiles = div_ceil(rows, pmax)
    for tile in range(num_tiles):
        tile_rows = min(pmax, rows - tile * pmax)
        nisa.dma_copy(
            hbm_2d[nl.ds(tile * pmax, tile_rows), :],
            zero_sb[0:tile_rows, :],
        )
