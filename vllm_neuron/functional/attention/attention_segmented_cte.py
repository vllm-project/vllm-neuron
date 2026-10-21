# SPDX-License-Identifier: Apache-2.0
import nki
import torch

from typing import Optional
from torch import Tensor

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.parallel.neuron_parallel_state import all_contiguous_replica_groups
from vllm_neuron.utils.neuron_utils import can_run_kernel
from vllm_neuron.utils.bucket_utils import SUPPORTED_KV_SEGMENT_SIZES
import inspect as _inspect

import nki.language as nl
from nki.collectives import ReplicaGroup as _ReplicaGroup
from nkilib.core.attention.attention_kv_parallel_segmented_cte import (
    attention_kv_parallel_segmented_cte as _attention_kv_parallel_segmented_cte_impl,
)
from nkilib.core.attention.attention_segmented_cte import (
    attention_segmented_cte,
)

# Whether the installed nkilib segmented kernel accepts the fp8_packed kwarg.
# The packed read path is on nkilib mainline but may not be in the consumed
# version set yet; gate on the live signature so passing fp8_packed never
# breaks on an older kernel.
# TODO: remove this capability gate (and forward fp8_packed unconditionally)
# once the consumed nkilib image ships the segmented-prefill fp8_packed path.
_SEGMENTED_KERNEL_HAS_FP8_PACKED = (
    "fp8_packed" in _inspect.signature(attention_segmented_cte).parameters
)
_attention_segmented_cte_jit = nki.jit()(attention_segmented_cte)
_wrapped_attention_segmented_cte = wrap_nki(_attention_segmented_cte_jit)


@nki.jit
def _attention_kv_parallel_segmented_cte_wrapper(
    q: nl.ndarray,
    k_cache: nl.ndarray,
    v_cache: nl.ndarray,
    block_tables: nl.ndarray,
    kvp_q_offset: nl.ndarray,
    replica_groups: tuple,
    group_size: int,
    block_size: int,
    seg_size: int,
    scale: float = 1.0,
    global_q_offset: int = 0,
    tp_out: bool = False,
    sliding_window: int = 0,
    kvp_rank_id=None,
    kvp_group_size: int = 0,
    apc_mode: bool = False,
    valid_num_prior_tokens=None,
) -> nl.ndarray:
    """Thin wrapper that converts the tuple-of-tuples partition to ReplicaGroup."""
    rg = _ReplicaGroup([list(g) for g in replica_groups])
    return _attention_kv_parallel_segmented_cte_impl(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        kvp_q_offset=kvp_q_offset,
        replica_groups=rg,
        group_size=group_size,
        block_size=block_size,
        seg_size=seg_size,
        scale=scale,
        global_q_offset=global_q_offset,
        tp_out=tp_out,
        sliding_window=sliding_window,
        kvp_rank_id=kvp_rank_id,
        kvp_group_size=kvp_group_size,
        apc_mode=apc_mode,
        valid_num_prior_tokens=valid_num_prior_tokens,
    )


_wrapped_attention_kv_parallel_segmented_cte = wrap_nki(
    _attention_kv_parallel_segmented_cte_wrapper
)

# Maximum head dimension supported by the NKI kernel (SBUF partition constraint).
_MAX_HEAD_DIM = 128


def _decode_packed_to_unpacked(k_cache: Tensor) -> Tensor:
    """Un-swizzle the decode-packed 5D per-head K cache to standard block layout.

    ``[num_blocks, kv_heads, block_size // 2, head_dim, 2]``  ->
    ``[num_blocks, kv_heads, block_size, head_dim]``

    Two consecutive sequence positions are packed into the trailing size-2 axis
    (pos 2i at [...,0], pos 2i+1 at [...,1]); un-swizzle merges that axis back
    into the block_size dimension. Used only by the CPU fallback, which gathers
    from a standard-layout cache. The on-device kernel reads the 5D per-head
    layout directly (no conversion).
    """
    num_blocks, kv_heads, half, head_dim, two = k_cache.shape
    assert two == 2, f"packed K last dim must be 2, got {two}"
    # [nb, kh, half, d, 2] -> [nb, kh, half, 2, d] -> [nb, kh, block_size, d]
    return (
        k_cache.transpose(3, 4)
        .reshape(num_blocks, kv_heads, half * 2, head_dim)
        .contiguous()
    )


def _can_use_segmented_kernel(q: Tensor) -> bool:
    """
    Check if the segmented attention NKI kernel can be used.

    Returns False if the kernel is unavailable or the device is CPU
    without the NKI simulator enabled.
    """
    if _wrapped_attention_segmented_cte is None:
        return False
    if not can_run_kernel(q):
        return False
    return True


def _torch_segmented_attention_impl(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    prior_tokens: Tensor,
    block_size: int,
    kv_segment_size: int,
    scale: float,
    tp_q: bool = True,
    tp_out: bool = False,
    sliding_window: Optional[int] = None,
    sink: Optional[Tensor] = None,
) -> Tensor:
    """
    Pure PyTorch fallback for segmented attention (CPU mode).

    Gathers K/V from the paged block cache and computes standard scaled
    dot-product attention with a causal + out-of-range mask over prior +
    active tokens.

    This implementation is dynamo-traceable under ``torch.compile(..., fullgraph=True)``:
      - ``prior_tokens`` stays a tensor (no ``.item()``); it broadcasts into
        position masks.
      - Shapes are fully static. Instead of trimming to ``prior_len + S_q``,
        the gather walks the full ``max_blocks_per_seq * block_size`` span;
        positions past ``prior_len + S_q`` are masked out of the softmax.
      - No data-dependent Python branches. ``sliding_window`` and ``sink``
        presence are Python-constant kwargs — dynamo resolves the ``if``
        branches at trace time and produces per-specialization graphs.

    Args:
        q: Query tensor [Nh, S_q, D] (tp_q=True) or [Nh, D, S_q] (tp_q=False)
            where Nh = num Q heads on this TP rank.
        k_cache: Paged key cache [num_blocks, num_kv_heads, block_size, D]
        v_cache: Paged value cache [num_blocks, num_kv_heads, block_size, D]
        block_tables: Block table [B_kv, max_blocks_per_seq].
        prior_tokens: Number of prior cached tokens, shape [Nh, 1].
        block_size: Block size (Python int; static at trace time)
        kv_segment_size: Segment size (unused here; full padded gather is done)
        scale: Attention scaling factor
        tp_q: If True, q is [Nh, S_q, D]; if False, q is [Nh, D, S_q]
        tp_out: If True, output is [Nh, D, S_q]; if False, output is [Nh, S_q, D]
        sliding_window: Window size for local attention. None or 0 means full attention.
        sink: Attention sink bias tensor [Nh, 1]. Appended as extra column to
            attention scores before softmax, then dropped before V matmul.
    """
    if not tp_q:
        q = q.transpose(1, 2)

    Nh, S_q, D = q.shape
    num_kv_heads = k_cache.shape[1]
    max_blocks_per_seq = block_tables.shape[1]
    # Full padded KV length we gather over. Static at trace time.
    padded_kv_len = max_blocks_per_seq * block_size

    # prior_tokens as a scalar tensor. Kept as tensor (no .item()) so dynamo
    # can trace the downstream mask arithmetic.
    prior_len_t = prior_tokens.reshape(-1)[0].to(torch.int64)

    # Gather K/V from the paged block cache across the full block-table span.
    # Unused slots (0 in production, -1 in tests) are clamped to index 0; the
    # in-sequence position mask below zeroes out their softmax contribution.
    bt = block_tables[0]  # [max_blocks_per_seq]
    bt_clamped = bt.clamp_min(0).to(torch.int64)

    # k_cache[bt_clamped]: [max_blocks_per_seq, num_kv_heads, block_size, D]
    k_blocks = k_cache[bt_clamped]
    v_blocks = v_cache[bt_clamped]

    # Flatten blocks into a continuous sequence: [num_kv_heads, padded_kv_len, D]
    k_seq = k_blocks.permute(1, 0, 2, 3).reshape(num_kv_heads, padded_kv_len, D)
    v_seq = v_blocks.permute(1, 0, 2, 3).reshape(num_kv_heads, padded_kv_len, D)

    heads_per_kv = Nh // num_kv_heads
    if heads_per_kv > 1:
        k_seq = k_seq.repeat_interleave(heads_per_kv, dim=0)  # [Nh, padded_kv_len, D]
        v_seq = v_seq.repeat_interleave(heads_per_kv, dim=0)

    # Build a single [S_q, padded_kv_len] boolean mask of *allowed* positions:
    #   1. within sequence       : k_pos < prior_len + S_q
    #   2. causal                : k_pos <= q_pos + prior_len  (q_pos in [0, S_q))
    #   3. sliding window (opt.) : k_pos > q_pos + prior_len - sliding_window
    device = q.device
    q_pos = torch.arange(S_q, device=device, dtype=torch.int64).unsqueeze(1)  # [S_q, 1]
    k_pos = torch.arange(padded_kv_len, device=device, dtype=torch.int64).unsqueeze(
        0
    )  # [1, padded_kv_len]
    total_kv_len_t = prior_len_t + S_q  # scalar tensor

    in_seq = k_pos < total_kv_len_t  # [1, padded_kv_len]
    causal = k_pos <= (q_pos + prior_len_t)  # [S_q, padded_kv_len]
    allowed = in_seq & causal  # [S_q, padded_kv_len]
    if sliding_window is not None and sliding_window > 0:
        sw = k_pos > (q_pos + prior_len_t - sliding_window)
        allowed = allowed & sw

    q_f = q.float()
    k_seq_f = k_seq.float()
    v_seq_f = v_seq.float()

    scores = torch.bmm(q_f, k_seq_f.transpose(1, 2)) * scale

    # Mask out disallowed positions. allowed broadcasts [1, S_q, padded_kv_len].
    scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))

    if sink is not None:
        sink_f = sink.float().reshape(Nh, 1, 1).expand(Nh, S_q, 1)
        scores = torch.cat([scores, sink_f], dim=-1)

    attn_weights = torch.nn.functional.softmax(scores, dim=-1)

    if sink is not None:
        attn_weights = attn_weights[:, :, :-1]

    # Rows where no positions are allowed (e.g. purely padded queries) softmax
    # to NaN; zero them so the matmul contributes nothing.
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    output = torch.bmm(attn_weights, v_seq_f).to(q.dtype)

    if tp_out:
        output = output.transpose(1, 2)  # [Nh, D, S_q]

    return output


def _torch_segmented_attention_cp_impl(
    q: Tensor,
    k_local: Tensor,
    v_local: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    prior_tokens: Tensor,
    block_size: int,
    cp_rank: Tensor,
    cp_world_size: int,
    cp_kv_cache_interleave_size: int,
    cp_group,
    scale: float,
    tp_q: bool = True,
    tp_out: bool = False,
) -> Tensor:
    """
    Pure PyTorch fallback for DCP segmented attention, mirroring decode
    (``attention_decode``).

    Each rank holds its full-TP Q head shard for the FULL query length and a KV
    shard for its interleaved token slice (prior from cache + current
    projection). Q is AllGathered across heads so every rank in the DCP group
    holds all Q heads in the KV-replica set; each rank attends gathered-Q
    against its local KV shard, extracts a partial LSE, corrects across the
    group, and ReduceScatters along heads to return this rank's head shard.

    The causal mask maps Q global positions to local KV global positions:
      - Prior cache slot s → global pos: (s//I)*(W*I) + R*I + (s%I)
      - Current local token j → global pos: prior_global + (j//I)*(W*I) + R*I + (j%I)
      - Query token i → global pos: prior_global + i
    """
    if not tp_q:
        q = q.transpose(1, 2)  # [Nh_local, S, D]

    # ── AllGather Q across heads (mirror decode Stage 6.8) ──
    # [Nh_local, S, D] → [Nh_local * W, S, D]: every rank now holds all Q heads
    # in the KV-replica (DCP) group.
    q = cp_group.all_gather(q.contiguous(), dim=0)

    Nh_gathered, S_total, D = q.shape
    num_kv_heads = k_local.shape[0]
    S_local = k_local.shape[1]  # this rank's owned token count (S / cp_world_size)
    max_blocks_per_seq = block_tables.shape[1]
    padded_kv_len = max_blocks_per_seq * block_size

    if prior_tokens is None:
        prior_local_t = torch.zeros((), dtype=torch.int64, device=q.device)
    else:
        prior_local_t = prior_tokens.reshape(-1)[0].to(torch.int64)
    prior_global_t = prior_local_t * cp_world_size

    device = q.device
    I = cp_kv_cache_interleave_size
    W = cp_world_size
    # cp_rank arrives as a (1, 1) tensor; flatten to a scalar so R broadcasts
    # cleanly into the 1-D position arithmetic below (R * I added to a 1-D
    # arange). A (1, 1) R would add a spurious leading dim and break the
    # torch.cat of prior/current global positions.
    R = cp_rank.reshape(-1)[0]

    # GQA expansion
    heads_per_kv = Nh_gathered // num_kv_heads
    if heads_per_kv > 1:
        k_local_exp = k_local.repeat_interleave(heads_per_kv, dim=0)
        v_local_exp = v_local.repeat_interleave(heads_per_kv, dim=0)
    else:
        k_local_exp = k_local
        v_local_exp = v_local

    # ── Gather prior KV from cache (static padded shape) ──
    bt = block_tables[0].clamp_min(0).to(torch.int64)
    k_blocks = k_cache[bt]
    v_blocks = v_cache[bt]
    k_prior = k_blocks.permute(1, 0, 2, 3).reshape(num_kv_heads, padded_kv_len, D)
    v_prior = v_blocks.permute(1, 0, 2, 3).reshape(num_kv_heads, padded_kv_len, D)

    if heads_per_kv > 1:
        k_prior = k_prior.repeat_interleave(heads_per_kv, dim=0)
        v_prior = v_prior.repeat_interleave(heads_per_kv, dim=0)

    # ── Concatenate local KV: [prior_padded, current_local] ──
    k_full_local = torch.cat(
        [k_prior, k_local_exp], dim=1
    )  # [Nh_gathered, padded+S_local, D]
    v_full_local = torch.cat([v_prior, v_local_exp], dim=1)
    total_local_kv = padded_kv_len + S_local

    # ── Build causal+validity mask [S_total, total_local_kv] ──
    # Q global positions: [prior_global, prior_global + S_total)
    q_global = prior_global_t + torch.arange(S_total, device=device, dtype=torch.int64)

    # Prior KV global positions (interleaved): slot s → (s//I)*(W*I) + R*I + (s%I)
    prior_slot = torch.arange(padded_kv_len, device=device, dtype=torch.int64)
    prior_global_pos = (prior_slot // I) * (W * I) + R * I + (prior_slot % I)

    # Current KV global positions: this rank owns interleaved positions
    # Token j on rank R has global pos: prior_global + (j//I)*(W*I) + R*I + (j%I)
    cur_j = torch.arange(S_local, device=device, dtype=torch.int64)
    cur_global_pos = prior_global_t + (cur_j // I) * (W * I) + R * I + (cur_j % I)

    # Fold validity into positions: invalid prior slots get a position larger than
    # any Q position so the <= comparison naturally returns False.
    # This avoids a broadcast boolean AND which triggers a Neuron compiler bug
    # at certain tensor shapes (e.g. [4096, 5120] & [1, 5120]).
    # TODO: triage the bug(CHRS-985)
    # INVALID_POS > q_global.max() = prior_global_t + S_total - 1
    #            <= prior_local_t * W + S_total - 1 < padded_kv_len * W + S_total
    INVALID_POS = padded_kv_len * W + S_total + 1
    prior_valid_mask = prior_slot < prior_local_t
    prior_global_pos_masked = torch.where(
        prior_valid_mask,
        prior_global_pos,
        torch.full_like(prior_global_pos, INVALID_POS),
    )
    kv_global_pos = torch.cat(
        [prior_global_pos_masked, cur_global_pos]
    )  # [total_local_kv]

    # Combined causal + validity mask in a single comparison
    allowed = kv_global_pos.unsqueeze(0) <= q_global.unsqueeze(
        1
    )  # [S_total, total_local_kv]

    # ── Compute local partial attention ──
    q_f = q.float()
    scores = torch.bmm(q_f, k_full_local.float().transpose(1, 2)) * scale
    scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))

    lse_local = torch.logsumexp(scores, dim=-1)  # [Nh_gathered, S_total]
    w = torch.nn.functional.softmax(scores, dim=-1)
    w = torch.nan_to_num(w, nan=0.0)
    out_local = torch.bmm(w, v_full_local.float())  # [Nh_gathered, S_total, D]

    # ── LSE correction across DCP ranks ──
    all_lse = cp_group.all_gather(lse_local.contiguous(), dim=0)
    all_lse = all_lse.view(W, Nh_gathered, S_total)
    global_lse = torch.logsumexp(all_lse, dim=0)  # [Nh_gathered, S_total]

    # Weight local output and ReduceScatter along heads to combine the partials
    # and return this rank's head shard.
    local_weight = torch.exp(lse_local - global_lse)  # [Nh_gathered, S_total]
    local_weight = torch.nan_to_num(local_weight, nan=0.0)
    weighted_out = out_local * local_weight.unsqueeze(-1)  # [Nh_gathered, S_total, D]

    # ReduceScatter on dim=0 (heads): sums the LSE-weighted partials across DCP
    # ranks, then scatters the gathered heads back so each rank receives the
    # Nh_local heads it owns (mirror decode Stage 7.5a).
    output = cp_group.reduce_scatter(
        weighted_out.contiguous(), dim=0
    )  # [Nh_local, S_total, D]

    output = output.to(q.dtype)
    if tp_out:
        output = output.transpose(1, 2)  # [Nh_local, D, S_total]

    return output


_LNC_SIZE = 2


def _get_cp_seg_size(seq_len: int) -> int:
    """Pick the segment size for the KV-parallel segmented CP kernel.

    The kernel iterates over ``num_q_chunks = seq_len // seg_size`` chunks (see
    ``attention_kv_parallel_segmented_cte``), so any supported segment size that
    evenly divides ``seq_len`` is valid. We return the largest such size to
    minimize the number of chunks (fewer collective/merge iterations).

    Returns 0 if no supported segment size divides ``seq_len``, in which case the
    kernel is ineligible and the caller falls back to the torch impl.
    """
    for size in sorted(SUPPORTED_KV_SEGMENT_SIZES, reverse=True):
        if seq_len >= size and seq_len % size == 0:
            return size
    return 0


def _can_use_segmented_cp_kernel(
    q: Tensor,
    seq_len: int,
    cp_world_size: int,
    cp_kv_cache_interleave_size: int,
    block_size: int,
    Nh_local: int,
) -> bool:
    """Check if the segmented attention CP NKI kernel can be used."""
    if not can_run_kernel(q):
        return False
    # Eligible whenever a supported segment size divides seq_len (the kernel
    # processes seq_len in seg_size chunks), not only when seq_len is itself a
    # supported size.
    if _get_cp_seg_size(seq_len) == 0:
        return False
    if cp_kv_cache_interleave_size != block_size:
        return False
    total_heads = Nh_local * cp_world_size
    if total_heads % _LNC_SIZE != 0:
        return False
    return True


def segmented_attention_cp(
    q: Tensor,
    k_local: Tensor,
    v_local: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    prior_tokens: Tensor,
    block_size: int,
    cp_rank: Tensor,
    cp_world_size: int,
    cp_kv_cache_interleave_size: int,
    cp_group,
    scale: Optional[float] = None,
    tp_q: bool = True,
    tp_out: bool = False,
) -> Tensor:
    """
    Segmented Attention API for DCP, mirroring decode (``attention_decode``).

    Each rank holds its full-TP Q head shard for the full query length and a KV
    shard for its interleaved token slice. Q is AllGathered across heads (so
    every rank holds all Q heads in the KV-replica/DCP group), each rank
    attends gathered-Q × local_KV (prior cache + current local) producing a
    partial output, then LSE correction across the DCP group + ReduceScatter
    along heads returns this rank's Q head shard.

    Each DCP rank caches only its owned tokens (S/DCP per chunk) at interleaved
    positions. The local KV for attention is: prior from cache + current from
    this rank's projection.

    Args:
        q: This rank's head-sharded query for the full length, NOT pre-gathered:
            [Nh_local, S, D] (tp_q=True) or [Nh_local, D, S] (tp_q=False).
            The head AllGather across cp_group happens inside.
        k_local: This rank's current chunk keys [Nh_kv, S_local, D].
        v_local: This rank's current chunk values [Nh_kv, S_local, D].
        k_cache: Paged key cache [num_blocks, num_kv_heads, block_size, D].
        v_cache: Paged value cache [num_blocks, num_kv_heads, block_size, D].
        block_tables: Block table [B_kv, max_blocks_per_seq].
        prior_tokens: Local cached token count [Nh_local, 1] (= global_prior / cp_world_size).
        block_size: Cache block size.
        cp_rank: This rank's CP index (determines owned positions).
        cp_world_size: Total CP ranks (DCP degree).
        cp_kv_cache_interleave_size: Interleave granularity for cache slot mapping.
        cp_group: Process group for DCP communication (AllGather Q/LSE, ReduceScatter).
        scale: Scaling factor. Default: 1/sqrt(d_head).
        tp_q: Query transpose flag.
        tp_out: Output transpose flag.

    Returns:
        This rank's head shard: [Nh_local, D, S] (tp_out=True) or [Nh_local, S, D].
    """
    d_head = q.shape[2] if tp_q else q.shape[1]

    if d_head > _MAX_HEAD_DIM:
        raise ValueError(
            f"head_dim={d_head} exceeds maximum supported head dimension "
            f"({_MAX_HEAD_DIM}). Requires head_dim <= {_MAX_HEAD_DIM}."
        )

    if scale is None:
        scale = 1.0 / (d_head**0.5)

    seq_len = q.shape[1] if tp_q else q.shape[2]
    Nh_local = q.shape[0]
    if _can_use_segmented_cp_kernel(
        q=q,
        seq_len=seq_len,
        cp_world_size=cp_world_size,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        block_size=block_size,
        Nh_local=Nh_local,
    ):
        return _nki_segmented_attention_cp_impl(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            prior_tokens=prior_tokens,
            block_size=block_size,
            cp_rank=cp_rank,
            cp_world_size=cp_world_size,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            cp_group=cp_group,
            scale=scale,
            tp_q=tp_q,
            tp_out=tp_out,
        )

    return _torch_segmented_attention_cp_impl(
        q=q,
        k_local=k_local,
        v_local=v_local,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        prior_tokens=prior_tokens,
        block_size=block_size,
        cp_rank=cp_rank,
        cp_world_size=cp_world_size,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        cp_group=cp_group,
        scale=scale,
        tp_q=tp_q,
        tp_out=tp_out,
    )


def _nki_segmented_attention_cp_impl(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    prior_tokens: Tensor,
    block_size: int,
    cp_rank: Tensor,
    cp_world_size: int,
    cp_kv_cache_interleave_size: int,
    cp_group,
    scale: float,
    tp_q: bool = True,
    tp_out: bool = False,
) -> Tensor:
    """NKI kernel path for DCP segmented attention.

    Uses attention_kv_parallel_segmented_cte which handles:
    1. AllGather Q across ranks (via NKI collectives)
    2. Each rank computes segmented attention on its local KV shard
    3. All-to-all exchange of partial outputs + softmax stats
    4. Merge partials using online softmax
    """
    if not tp_q:
        q = q.transpose(1, 2)

    Nh_local, S, D = q.shape
    lnc_degree = _LNC_SIZE
    group_size = cp_world_size

    # Largest supported segment size that divides S. The kernel iterates
    # num_q_chunks = S // seg_size chunks internally (APC per-chunk offsets keep
    # causal masking correct), so seg_size < S is fully supported and lets the
    # kernel run for sequence lengths beyond the supported-size set.
    seg_size = _get_cp_seg_size(S)

    # global_q_offset = 0: conservative compile-time value. The kernel won't
    # skip masking on any prior tiles (slightly less efficient but correct for
    # all prior counts including APC). The runtime kvp_q_offset tensor drives
    # the actual causal mask computation on hardware.
    global_q_offset = 0

    # Pass the full block_tables — the kernel iterates all blocks and uses
    # kvp_q_offset for causal masking to exclude unfilled positions.
    block_tables_local = block_tables.to(torch.int32)

    # kvp_q_offset: runtime tensor for actual global Q position (causal mask).
    if prior_tokens is not None:
        kvp_q_offset = (prior_tokens.reshape(1, 1) * cp_world_size).to(torch.int32)
        valid_num_prior_tokens = prior_tokens.reshape(1, 1).to(torch.int32)
    else:
        kvp_q_offset = torch.zeros((1, 1), dtype=torch.int32, device=q.device)
        valid_num_prior_tokens = torch.zeros((1, 1), dtype=torch.int32, device=q.device)

    # kvp_rank_id: this rank's index within the KV-parallel group. Callers build
    # cp_rank at shape (1, 1) with a factory op (torch.full), which the kernel
    # consumes directly. We must NOT .reshape/.view it here: cp_rank originates
    # from a constant built on q.device, and during compile that device is meta,
    # so a view op on it fails Dynamo fake-tensor tracing (aten.view on a
    # non-fake meta constant). Keeping it factory-shaped avoids the view entirely.
    kvp_rank_id = cp_rank.to(torch.int32)

    # Pre-scale Q (kernel uses scale=1.0 internally when we pre-scale).
    q_scaled = q * scale

    # The kernel's replica_groups arg is the full partition the CP group belongs
    # to (converted to ReplicaGroup inside the NKI kernel). Passing every
    # sub-group instead of only cp_group.ranks keeps the graph rank-agnostic.
    group_ranks = tuple(tuple(g) for g in all_contiguous_replica_groups(cp_group.ranks))

    # Pass the full k/v cache. The kernel uses k_cache.shape[0] to compute
    # max_prior_tokens and iterates accordingly. The runtime kvp_q_offset
    # ensures correct causal masking regardless of actual prior count.
    result = _wrapped_attention_kv_parallel_segmented_cte[lnc_degree](
        q=q_scaled,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables_local,
        kvp_q_offset=kvp_q_offset,
        replica_groups=group_ranks,
        group_size=group_size,
        block_size=block_size,
        seg_size=seg_size,
        scale=1.0,
        global_q_offset=global_q_offset,
        tp_out=tp_out,
        sliding_window=0,
        kvp_rank_id=kvp_rank_id,
        kvp_group_size=cp_world_size,
        apc_mode=True,
        valid_num_prior_tokens=valid_num_prior_tokens,
    )

    return result


def segmented_attention(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    prior_tokens: Tensor,
    block_size: int,
    kv_segment_size: int,
    scale: Optional[float] = None,
    tp_q: bool = True,
    tp_out: bool = False,
    sliding_window: Optional[int] = None,
    sink: Optional[Tensor] = None,
    fp8_packed: bool = False,
    k_scale: Optional[Tensor] = None,
    v_scale: Optional[Tensor] = None,
) -> Tensor:
    """
    Segmented Attention API using the attention_segmented_cte NKI kernel.

    This implements segmented prefill attention: the query attends to prior cached
    KV in fixed-size segments, enabling efficient chunked prefill with block-based
    KV cache. Computes: softmax(scale * Q @ K_cache^T + mask) @ V_cache

    Input Layouts (controlled by transpose flags):
        q: Query tensor
           - [Nh, S_q, D] when tp_q=True (default)
           - [Nh, D, S_q] when tp_q=False

    Output Layout:
        - [Nh, D, S_q] if tp_out=True
        - [Nh, S_q, D] if tp_out=False (default)

    Dimensions:
        Nh: Number of Q heads on this TP rank
        S_q: Query sequence length
        D: Head dimension (max 128)

    Args:
        q: Query tensor
        k_cache: Block KV cache for keys [num_blocks, num_kv_heads, block_size, D]
        v_cache: Block KV cache for values [num_blocks, num_kv_heads, block_size, D]
        block_tables: Block table mapping sequences to cache blocks [B_kv, max_blocks_per_seq]
        prior_tokens: Number of prior cached tokens, shape [Nh, 1]. Must be multiple of block_size.
        block_size: Size of each block in the KV cache
        kv_segment_size: Segment size for iterative prior KV processing
        scale: Scaling factor for attention scores. Default: 1/sqrt(d_head).
               Must be 1.0 when using sliding_window, prefix caching, or context parallel.
        tp_q: Query transpose flag. True means Q is [Nh, S, D]. Default: True
        tp_out: Output transpose flag. True means output is [Nh, D, S]. Default: False
        sliding_window: Window size for local attention. None or 0 means full attention. Default: None
        sink: Attention sink tensor [Nh, 1] for streaming/infinite context. Default: None
        fp8_packed: When True, k_cache uses the swizzled packed FP8 layout
            [num_blocks, block_size // 2, num_kv_heads * head_dim, 2]; the kernel
            views it as BF16 to DMA-transpose, then reinterprets back to FP8. V
            is never packed. Default: False
        k_scale: FP8 K dequant scale [PMAX, 1], applied in-kernel. Default: None
        v_scale: FP8 V dequant scale [PMAX, 1], applied in-kernel. Default: None

    Returns:
        Output tensor with attention results.

    Raises:
        ValueError: If any kernel constraint is violated:
            - head_dim > 128
            - kv_segment_size not in SUPPORTED_KV_SEGMENT_SIZES
            - kv_segment_size not divisible by block_size
            - sliding_window not divisible by block_size (when set)
            - seqlen_q != kv_segment_size (temporary constraint)
        RuntimeError: If the segmented attention NKI kernel is not available and
            there is no torch fallback implementation.

    Note:
        The segmented kernel does not yet support tp_q=False or tp_out=True natively.
        When these flags are set, transposing is handled at the boundary before/after
        the kernel call.

        fp8_packed: the caller passes the cache in the decode-packed 5D
        per-head layout ([num_blocks, kv_heads, block_size // 2, head_dim, 2])
        — the single shared K cache the runner allocates. As of nki_library
        1.0.14244 the segmented kernel reads this layout directly, so it is
        passed straight through on the kernel path (only the CPU fallback
        un-swizzles it to the standard block layout).
    """
    # The K cache is stored in the decode-packed 5D per-head layout
    # ``[num_blocks, kv_heads, block_size//2, head_dim, 2]`` — the single shared
    # cache the runner allocates. As of nki_library 1.0.14244, the segmented
    # kernel's ``load_kv_cache`` reads this 5D per-head layout DIRECTLY (folds
    # only the head into the row dim, row width head_dim*2), so NO conversion is
    # needed on the kernel path — pass ``k_cache`` straight through. (Older
    # nkilib expected the folded ``[nb, block_size//2, kv_heads*head_dim, 2]``
    # layout via ``_decode_packed_to_segmented_packed``; that is no longer the
    # kernel contract.)
    if not _can_use_segmented_kernel(q):
        if scale is None:
            d_head = q.shape[2] if tp_q else q.shape[1]
            scale = 1.0 / (d_head**0.5)
        # The CPU fallback gathers from a standard-layout cache, so un-swizzle
        # the 5D per-head packed K cache to [nb, kv_heads, block_size, head_dim]
        # first. K-scale dequant is fused into the softmax scale by the caller
        # (matching the on-device path), so the fallback consumes scales the
        # same way regardless of packing.
        if fp8_packed:
            k_cache = _decode_packed_to_unpacked(k_cache)
        return _torch_segmented_attention_impl(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            prior_tokens=prior_tokens,
            block_size=block_size,
            kv_segment_size=kv_segment_size,
            scale=scale,
            tp_q=tp_q,
            tp_out=tp_out,
            sliding_window=sliding_window,
            sink=sink,
        )

    # --- Validate kernel constraints ---
    # These mirror the kernel_assert checks in the NKI kernel

    # Extract dimensions (layout depends on tp_q)
    seqlen_q = q.shape[1] if tp_q else q.shape[2]
    d_head = q.shape[2] if tp_q else q.shape[1]

    # 1. head_dim must fit in a single SBUF partition (128 elements)
    if d_head > _MAX_HEAD_DIM:
        raise ValueError(
            f"head_dim={d_head} exceeds maximum supported head dimension "
            f"({_MAX_HEAD_DIM}). The segmented attention kernel requires "
            f"head_dim <= {_MAX_HEAD_DIM}."
        )

    # 2. kv_segment_size must be a supported size
    if kv_segment_size not in SUPPORTED_KV_SEGMENT_SIZES:
        raise ValueError(
            f"kv_segment_size={kv_segment_size} is not supported. "
            f"Supported sizes: {sorted(SUPPORTED_KV_SEGMENT_SIZES)}."
        )

    # 3. kv_segment_size must be divisible by block_size
    if kv_segment_size % block_size != 0:
        raise ValueError(
            f"kv_segment_size ({kv_segment_size}) must be divisible by "
            f"block_size ({block_size})."
        )

    # 4. sliding_window must be divisible by block_size when set
    if sliding_window is not None and sliding_window > 0:
        if sliding_window % block_size != 0:
            raise ValueError(
                f"sliding_window ({sliding_window}) must be divisible by "
                f"block_size ({block_size})."
            )

    # 5. Query sequence length must equal kv_segment_size.
    #    TODO: This is a temporary constraint. The kernel can be extended to
    #    support seqlen_q != prior_seg_size (e.g., smaller Q attending to a
    #    larger KV segment) once the active-segment tiling logic is decoupled
    #    from the query length.
    if seqlen_q != kv_segment_size:
        raise ValueError(
            f"Query sequence length ({seqlen_q}) must equal "
            f"kv_segment_size ({kv_segment_size}). The segmented kernel "
            f"currently requires seqlen_q == kv_segment_size."
        )

    # Compute default scale if not provided
    if scale is None:
        scale = 1.0 / (d_head**0.5)

    # Segmented kernel does not yet support tp_q=False or tp_out=True natively,
    # so we transpose at the boundary.
    q_seg = q.transpose(1, 2) if not tp_q else q
    q_seg = q_seg * scale

    # Only forward fp8_packed when the installed kernel supports it (the packed
    # read path may not be in the consumed nkilib version set yet). A packed
    # cache with an older kernel cannot be read correctly, so fail loudly rather
    # than silently misread.
    extra_kwargs = {}
    if _SEGMENTED_KERNEL_HAS_FP8_PACKED:
        extra_kwargs["fp8_packed"] = fp8_packed
    elif fp8_packed:
        raise RuntimeError(
            "fp8_packed segmented prefill requires an nkilib "
            "attention_segmented_cte with the fp8_packed parameter, which the "
            "installed kernel does not have. Bump nkilib to a version that "
            "ships the segmented packed FP8 read path."
        )

    result = _wrapped_attention_segmented_cte[2](
        q=q_seg,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        prior_tokens=prior_tokens,
        block_size=block_size,
        prior_seg_size=kv_segment_size,
        scale=1.0,
        tp_q=True,
        tp_out=False,
        sliding_window=sliding_window if sliding_window else None,
        sink=sink,
        num_q_heads=q_seg.shape[0],
        k_scale=k_scale,
        v_scale=v_scale,
        **extra_kwargs,
    )

    if tp_out:
        result = result.transpose(1, 2)

    return result
