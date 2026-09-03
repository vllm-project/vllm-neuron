# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet mixer for Qwen3.5 (18 of its 24 layers).

A DeltaNet layer replaces attention with a **linear** recurrence. Instead of a
KV cache that grows with the sequence it keeps two fixed-size state tensors per
sequence:

* ``conv_state`` ``[kernel - 1, conv_dim]`` — the tail of the depthwise causal
  conv1d window over the projected q/k/v.
* ``recurrent_state`` ``[num_v_heads, head_k_dim, head_v_dim]`` — the delta-rule
  associative memory, accumulated in float32.

The recurrence per token is

    S <- S * exp(g_t)                                (gated decay)
    S <- S + k_t (v_t - S^T k_t) * beta_t            (delta rule)
    o_t = S^T q_t

with ``q``/``k`` L2-normalised and ``q`` scaled by ``1/sqrt(head_k_dim)``.
Prefill evaluates that recurrence in chunks so the bulk of the work becomes
matmuls; decode evaluates a single step directly. Both paths are numerically
matched against ``transformers``' ``torch_chunk_gated_delta_rule`` /
``torch_recurrent_gated_delta_rule``, which are the oracle for this port — see
``examples/vllm_neuron/models/qwen3_5/check_deltanet_vs_hf.py``.

Layout conventions in this file follow the rest of the plugin: weights are
stored transposed (``[in_features, out_features]``) so a forward pass is
``hidden @ weight``, and activations are ``[tokens, hidden]`` with no batch dim.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.distributed.parallel_state import get_tp_group

from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    set_weight_loader,
    sharding_weight_loader,
)

from .config import Qwen3_5TextConfig
from .flags import (
    SEQUENCE_PARALLEL,
    nki_delta_rule_enabled,
    nki_delta_rule_variant,
)

# Chunk length for the prefill recurrence. 64 is what HF's reference kernel
# uses, and the UT-transform inverse below assumes a power of two.
_CHUNK_SIZE = 64


# ---------------------------------------------------------------------------
# Weight loaders
# ---------------------------------------------------------------------------
# ``in_proj_qkv`` and ``conv1d`` are single checkpoint tensors whose output
# dimension is the *concatenation* ``[q | k | v]``. Sharding that dimension
# contiguously would hand rank 1 a mixture of q and k channels, so each segment
# has to be sliced separately and re-concatenated. The depthwise conv1d weight
# needs exactly the same treatment because its channels line up with those of
# ``in_proj_qkv`` one-for-one.


def _segment_shard_loader(
    segment_sizes: tuple[int, ...],
    num_shards: int,
    *,
    is_storage_transposed: bool,
    squeeze_dim: int | None = None,
) -> SafetensorsWeightLoader:
    """Shard each ``[q | k | v]`` segment of one tensor, then re-concatenate.

    Args:
        segment_sizes: Global size of each segment along the checkpoint's dim 0.
            Every entry must divide ``num_shards``.
        num_shards: TP world size.
        is_storage_transposed: Transpose the result, for the plugin's
            ``[in_features, out_features]`` parameter layout.
        squeeze_dim: Drop this dim after slicing. ``conv1d.weight`` is
            ``[conv_dim, 1, kernel]`` and the singleton input-channel dim is
            not wanted.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1, "_segment_shard_loader() takes a single tensor"
        slice_obj = slices[0]
        ndim = len(slice_obj.get_shape())

        parts = []
        offset = 0
        for size in segment_sizes:
            per_rank = size // num_shards
            start = offset + (rank % num_shards) * per_rank
            sel = [slice(None)] * ndim
            sel[0] = slice(start, start + per_rank)
            parts.append(slice_obj[tuple(sel)])
            offset += size

        result = torch.cat(parts, dim=0)
        if squeeze_dim is not None:
            result = result.squeeze(squeeze_dim)
        if is_storage_transposed:
            result = result.T
        return result

    return SafetensorsWeightLoader(transform=transform)


def _stacked_shard_loader(
    num_shards: int,
    *,
    is_storage_transposed: bool,
) -> SafetensorsWeightLoader:
    """Shard several checkpoint tensors on dim 0 and concatenate them.

    Used to fuse ``in_proj_b`` and ``in_proj_a`` (both ``[num_v_heads, hidden]``)
    into one projection, and to fuse the per-head ``dt_bias``/``A_log`` vectors.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        parts = []
        for slice_obj in slices:
            size = slice_obj.get_shape()[0]
            per_rank = size // num_shards
            start = (rank % num_shards) * per_rank
            parts.append(slice_obj[start : start + per_rank])
        result = torch.cat(parts, dim=0)
        return result.T if is_storage_transposed else result

    return SafetensorsWeightLoader(transform=transform)


# ---------------------------------------------------------------------------
# The delta rule
# ---------------------------------------------------------------------------


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalise the last dim, matching HF's ``l2norm`` exactly.

    Note this is ``rsqrt(sum(x^2) + eps)``, not the more common
    ``rsqrt(mean(x^2) + eps)`` of RMSNorm — the two differ by ``sqrt(dim)``.
    """
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _strictly_lower_inverse(a: torch.Tensor, size: int) -> torch.Tensor:
    """``(I - A)^-1`` for strictly lower-triangular ``A``, in ``log2(size)`` steps.

    HF's reference computes this by scalar forward substitution: ``size - 1``
    sequential slice assignments into a shared tensor. That is stable but it is
    also exactly the pattern the NxDI port flags as hitting neuronx-cc codegen
    failures, and 63 dependent steps is a poor shape for the compiler.

    This is the *blocked* version of the same elimination. Keep ``T`` block
    diagonal with blocks of width ``s``, each holding the exact inverse for its
    own diagonal block, and double ``s`` each round. For a block split into an
    upper half ``U`` and a lower half ``L``,

        (I - A)^-1 = [[T_U, 0], [T_L A_LU T_U, T_L]]

    and because ``T`` is block diagonal at width ``s``, ``T_L A_LU T_U`` is just
    the ``(L, U)`` quadrant of ``T @ A @ T``. So one round is two matmuls and a
    mask, giving ``2 * log2(size)`` matmuls in ``log2(size)`` dependent steps —
    12 and 6 respectively at ``size == 64``.

    The obvious alternative, summing the Neumann series by repeated squaring
    (``prod (I + A^(2^i))``), is *not* usable here even though ``A`` is
    nilpotent: with this checkpoint's weights ``|A^16|`` reaches 2.7e6 while the
    true inverse has entries of magnitude 1, so the product loses every
    significant digit to cancellation (measured: 0.57 absolute error in
    float32, against 1.2e-7 for elimination). Every intermediate here is a true
    inverse of a sub-problem, so nothing grows.
    """
    lead = (1,) * (a.dim() - 2)
    rows = torch.arange(size, device=a.device).reshape(size, 1)
    cols = torch.arange(size, device=a.device).reshape(1, size)

    # ``.expand_as`` would give the matmuls below a stride-0 batched operand.
    # Materialise instead: it is a 64x64 tensor, so the copy is free, and a
    # broadcast view feeding a batched matmul is the kind of construct worth not
    # asking a compiler to handle.
    inv = torch.eye(size, dtype=a.dtype, device=a.device).repeat(
        *a.shape[:-2], 1, 1
    )
    width = 1
    while width < size:
        # The (lower-half, upper-half) quadrant of every width-2*width block.
        quadrant = (
            (rows // (2 * width) == cols // (2 * width))
            & (rows // width % 2 == 1)
            & (cols // width % 2 == 0)
        )
        mask = quadrant.to(a.dtype).reshape(*lead, size, size)
        inv = inv + mask * (inv @ a @ inv)
        width *= 2
    return inv


def _nki_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The same recurrence, on a vendored NKI kernel. **Off by default.**

    Two variants are vendored, differing only in how they apply the in-chunk
    ``(I - A)^-1``. Measured with ``probe_nki_deltanet.py``, 4 heads on one rank,
    seq 1024, against the torch path at 0.81 ms/call:

    | variant | inverse | ms | vs torch |
    |---|---|---|---|
    | ``fused`` (default) | Neumann by power-doubling on 32x32 leaves, then Schur composition 32 -> 64 -> 128 | 1.86 | 0.43x |
    | ``legacy`` | forward substitution: 128 sequential steps, each a full 128x128x128 matmul with all but one row masked away | 9.74 | 0.08x |

    Both are *correct* — output within 1.2e-05 of the torch reference, final state
    within 2.5e-06.

    **There is no library involved on either side.** The torch path is not eager
    PyTorch: dynamo traces it to an FX graph (336 ``matmul`` nodes for this
    function), which the plugin's backend lowers to HLO and hands to *neuronx-cc*,
    the same compiler that builds the NKI kernels, running on the same Tensor
    Engine. No BLAS, no hand-tuned library. So the difference is entirely in what
    the two express, and it is not the same difference for the two variants.

    Counting the in-chunk inverse and its application exactly, per 128 tokens of
    one head:

    | | inverse FLOPs | vs torch | time vs torch | achieved throughput |
    |---|---|---|---|---|
    | torch | 16.8 MFLOP | 1.00x | 1.00x | 1.00x |
    | ``fused`` | 7.9 MFLOP | **0.47x** | 2.30x | **0.20x** |
    | ``legacy`` | 536.9 MFLOP | **32x** | 12.02x | **2.66x** |

    Two different failure modes, and neither is "torch has a faster library":

    * ``legacy`` loses on **arithmetic**. Forward substitution issues a full
      128x128x128 matmul per row and keeps one row, so 32x the work. Its hardware
      utilisation is actually 2.7x *better* than the torch path's — it is a
      well-engineered kernel executing a wasteful algorithm.
    * ``fused`` loses on **utilisation**. Its hierarchical inverse is genuinely
      cheaper than the blocked elimination — less than half the FLOPs — and it
      still takes 2.3x the time, i.e. it reaches only a fifth of the throughput.
      That is serialisation: one launch per ``(batch, head)`` walking chunks with
      ``sequential_range``, so 4 launches x 8 chunks = 32 dependent steps at seq
      1024, each too small to fill the engine. The torch graph puts the same work
      into a handful of matmuls batched over all 64 ``(head, chunk)`` pairs, which
      is what lets the compiler keep the Tensor Engine busy.

        HF's factorisation is what allows the hoisting:
    ``v_new = T @ v_beta - (T @ (k_beta * e^g)) @ state`` keeps both applications
    of ``T`` free of ``state``, so they can be computed for every chunk before the
    sequential loop. The kernels solve ``(I - A) v_new = rhs(state)`` with state on
    the right-hand side — same algebra, but it pins the solve inside the loop.

    The reference's *multihead* variant in ``nki_deltanet_fused.py`` batches head
    groups, which is the missing factor of ~4 and could plausibly close the
    remaining 2.3x. It is also the variant the reference documents as numerically
    unstable on real vision embeddings, and this port has working VL, so it is not
    used. Worth noting for anyone who tries: the leaf-Neumann inverse carries about
    1.2e-3 absolute error with this checkpoint's weights (against 1.2e-7 for the
    blocked elimination in ``_strictly_lower_inverse``), yet end-to-end output
    still lands within 1.2e-05 — the recurrence damps it. And measured on real
    vision embeddings from the tower, ``A`` is *not* worse conditioned than on text
    (``|A_leaf^8|`` 4907 vs 7027), because ``k`` is L2-normalised so ``A`` is
    nearly invariant to input magnitude. That matches the reference's own note that
    random vectors at the same std decode cleanly, and means the instability is
    still unexplained — it is not simply the leaf Neumann blowing up.

    Finally, the ceiling: model compute is ~28 ms of a ~109 ms TTFT, so even a free
    delta rule takes only about 20% off. That, not the kernel, is where the gap to
    the reference's 42.2 ms lives.

        Contract (the reference's "legacy_direct" mode): ``query`` arrives
    L2-normalised, this function applies the ``1/sqrt(dk)`` scale, ``key`` is
    L2-normalised, ``g`` is RAW per-token log-decay (the kernel does its own
    cumsum), ``beta`` is ``sigmoid(b)``.
    """
    from vllm_neuron.nki.nki_hop import wrap_nki

    # The two vendored variants differ only in the in-chunk (I - A)^-1:
    #   "fused"  hierarchical — Neumann on 32x32 leaves + Schur composition to
    #            128; normalises q/k and applies the query scale in-kernel
    #   "legacy" forward substitution, 128 sequential full matmuls; the caller
    #            normalises and scales
    variant = nki_delta_rule_variant()
    if variant == "fused":
        from .nki_deltanet_fused import CHUNK_SIZE, deltanet_fused_chunked_fwd
    else:
        from .nki_deltanet import CHUNK_SIZE, deltanet_fused_chunked_fwd

    b, h, t, dk = query.shape
    dv = value.shape[-1]
    device = query.device
    bh = b * h

    # (B, H, T, d) -> (B*H, T, d); g/beta -> (B*H, T, 1) as the kernel wants.
    # The legacy kernel wants the 1/sqrt(dk) query scale applied by the caller (the
    # torch body applies it internally too, so callers pass unscaled queries to
    # every path). The fused kernel applies both the l2-norm and the scale itself,
    # so it gets the query unscaled. Its re-normalisation of an already-normalised
    # q/k is a ~5e-7 no-op, which the probe confirms.
    scaled = query if variant == "fused" else query * (dk**-0.5)
    query_f = scaled.reshape(bh, t, dk).float().contiguous()
    key_f = key.reshape(bh, t, dk).float().contiguous()
    value_f = value.reshape(bh, t, dv).float().contiguous()
    g_f = g.reshape(bh, t).unsqueeze(-1).float().contiguous()
    beta_f = beta.reshape(bh, t).unsqueeze(-1).float().contiguous()
    state_f = initial_state.reshape(bh, dk, dv).float().contiguous()

    # The kernel's three constant masks, shared across launches. Built with torch
    # ops on ``device`` rather than from the kernel module's numpy helpers: a
    # ``torch.tensor(numpy_array)`` inside the traced graph produces a CPU tensor,
    # and the mixed-device call then fails to dispatch with "could not find kernel
    # for HigherOrderOperator nki_kernel_wrapper at dispatch key PrivateUse1".
    ones = torch.ones(CHUNK_SIZE, CHUNK_SIZE, dtype=torch.float32, device=device)
    lower_mask = ones.tril(-1)  # strict lower triangle
    lower_mask_diag = ones.tril(0)  # lower triangle including the diagonal
    identity = torch.eye(CHUNK_SIZE, dtype=torch.float32, device=device)

    kernel = wrap_nki(deltanet_fused_chunked_fwd)
    outputs, states = [], []
    for i in range(bh):
        out_i, state_i = kernel(
            query_f[i],
            key_f[i],
            value_f[i],
            g_f[i],
            beta_f[i],
            state_f[i],
            lower_mask,
            identity,
            lower_mask_diag,
        )
        outputs.append(out_i)
        states.append(state_i)

    output = torch.stack(outputs, dim=0).reshape(b, h, t, dv)
    final_state = torch.stack(states, dim=0).reshape(b, h, dk, dv)
    return output, final_state


def _can_use_nki_delta_rule(query: torch.Tensor, value: torch.Tensor) -> bool:
    """Whether the vendored kernel's fixed geometry matches this call.

    The kernel hard-codes ``P_MAX = CHUNK_SIZE = k_dim = v_dim = 128`` and needs
    the sequence padded to a multiple of 128. Everything else falls back to torch,
    which is the reference implementation the CPU checks validate.
    """
    if not nki_delta_rule_enabled():
        return False
    from vllm_neuron.nki.nki_hop import can_run_kernel

    if not can_run_kernel(query):
        return False

    from .nki_deltanet import CHUNK_SIZE, P_MAX

    _b, _h, t, dk = query.shape
    return dk == P_MAX and value.shape[-1] == P_MAX and t % CHUNK_SIZE == 0


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    chunk_size: int = _CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked gated delta rule for prefill.

    The torch body below is the shipped path. ``VLLM_NEURON_QWEN35_ENABLE_NKI=1``
    switches to the vendored NKI kernel, which is correct but ~10x slower — see
    ``_nki_chunk_gated_delta_rule`` for the numbers and why.

    Args:
        query, key: ``[B, H, T, head_k_dim]``, already L2-normalised. Note this
            function applies the ``1/sqrt(head_k_dim)`` query scale itself, so
            callers pass unscaled queries either way.
        value: ``[B, H, T, head_v_dim]``.
        g: ``[B, H, T]`` log decay per token (<= 0).
        beta: ``[B, H, T]`` update strength in (0, 1).
        initial_state: ``[B, H, head_k_dim, head_v_dim]``.
        chunk_size: Must be a power of two and divide ``T``.

    Returns:
        ``(output [B, H, T, head_v_dim], final_state [B, H, head_k_dim, head_v_dim])``

    The math is HF's ``torch_chunk_gated_delta_rule`` with two changes: the
    ``(I - A)^-1`` loop is replaced (see ``_strictly_lower_inverse``) and the
    per-chunk state carry is left as a Python loop over ``T / chunk_size``,
    which is a compile-time constant here because ``T`` is bucketed.
    """
    # On device, hand this to the NKI kernel; the torch body below stays the
    # reference implementation and the CPU checks keep validating it.
    if _can_use_nki_delta_rule(query, value):
        return _nki_chunk_gated_delta_rule(
            query, key, value, g, beta, initial_state
        )

    b, h, t, dk = query.shape
    dv = value.shape[-1]
    if t % chunk_size:
        raise ValueError(f"sequence length {t} must be a multiple of {chunk_size}")
    num_chunks = t // chunk_size
    bh = b * h
    bhc = bh * num_chunks
    device, dtype = query.device, query.dtype

    # Everything below is deliberately kept at rank 3. neuronx-cc fails codegen
    # on higher-rank matmul chains at these dimensions: the NxDI reference had to
    # collapse its attention to (B*H, S, d) for exactly this reason
    # ("NCC_INLA001: Expected 2D tensor but got 4D AP") and flags its own
    # PyTorch chunked DeltaNet forward as hitting a codegen ICE "with these
    # DeltaNet dimensions", defaulting to NKI kernels instead. The natural
    # formulation here is rank 5 — [b, h, chunks, chunk, dim] — so fold the
    # leading dims into one and index chunks by reshaping rather than slicing.
    def chunked(x: torch.Tensor) -> torch.Tensor:
        """``[b, h, t, d] -> [b*h*chunks, chunk, d]``: every chunk independent."""
        return x.reshape(bhc, chunk_size, x.shape[-1])

    query = chunked(query * (dk**-0.5))
    key_c = chunked(key)
    v_beta = chunked(value * beta.unsqueeze(-1))
    k_beta = chunked(key * beta.unsqueeze(-1))
    value_c = chunked(value)
    g = g.reshape(bhc, chunk_size)

    # Masks as float multiplies rather than tril/triu/masked_fill: one construct
    # instead of three, and no bool tensors in the matmul chain.
    rows = torch.arange(chunk_size, device=device).reshape(chunk_size, 1)
    cols = torch.arange(chunk_size, device=device).reshape(1, chunk_size)
    lower = (rows >= cols).to(dtype)  # i >= j, diagonal included
    strictly_lower = (rows > cols).to(dtype)

    # Cumulative log-decay within each chunk. A matmul with an upper-triangular
    # ones matrix rather than ``torch.cumsum``: exact, and the plugin already
    # prefers that form on Neuron (see ``NF.cumsum``, which falls back to matmul
    # off-device for the same reason). ``chunk_size`` is 64, so this is tiny.
    g = g @ (cols >= rows).to(dtype)

    # Pairwise decay exp(g_i - g_j) for i >= j, zero above the diagonal. The
    # mask is applied before the exp so the upper triangle is exp(0) == 1 rather
    # than exp(+large), then zeroed.
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)) * lower).exp() * lower

    # ``decay_mask`` already zeroes i < j, so the extra mask here only has to
    # drop the diagonal.
    attn = -((k_beta @ key_c.transpose(-1, -2)) * decay_mask) * strictly_lower
    # UT transform: turn the sequential in-chunk dependency into one matrix.
    t_mat = _strictly_lower_inverse(attn, chunk_size)

    value_c = t_mat @ v_beta
    k_cumdecay = t_mat @ (k_beta * g.exp().unsqueeze(-1))

    # Regroup so a chunk index can be taken without slicing a rank-5 tensor:
    # [b*h*chunks, chunk, d] -> chunks x [b*h, chunk, d].
    def per_chunk(x: torch.Tensor) -> list[torch.Tensor]:
        return list(x.reshape(bh, num_chunks, chunk_size, x.shape[-1]).unbind(1))

    query_l = per_chunk(query)
    key_l = per_chunk(key_c)
    value_l = per_chunk(value_c)
    k_cumdecay_l = per_chunk(k_cumdecay)
    decay_l = list(
        decay_mask.reshape(bh, num_chunks, chunk_size, chunk_size).unbind(1)
    )
    g_l = list(g.reshape(bh, num_chunks, chunk_size).unbind(1))

    state = initial_state.reshape(bh, dk, dv).to(dtype)
    outputs = []
    for i in range(num_chunks):
        q_i, k_i, v_i, g_i = query_l[i], key_l[i], value_l[i], g_l[i]
        # Intra-chunk contribution. No causal masking needed: ``decay_l[i]``
        # is already zero above the diagonal.
        attn_i = (q_i @ k_i.transpose(-1, -2)) * decay_l[i]
        v_new = v_i - k_cumdecay_l[i] @ state
        # ... plus what the state carried in from earlier chunks.
        attn_inter = (q_i * g_i.unsqueeze(-1).exp()) @ state
        outputs.append(attn_inter + attn_i @ v_new)

        g_last = g_i[:, -1:]
        state = state * g_last.unsqueeze(-1).exp() + (
            k_i * (g_last - g_i).unsqueeze(-1).exp()
        ).transpose(-1, -2) @ v_new

    output = torch.stack(outputs, dim=1).reshape(b, h, t, dv)
    return output, state.reshape(b, h, dk, dv)


def recurrent_gated_delta_rule_step(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One decode step of the gated delta rule.

    Args:
        query, key: ``[B, H, head_k_dim]``, already L2-normalised.
        value: ``[B, H, head_v_dim]``.
        g, beta: ``[B, H]``.
        state: ``[B, H, head_k_dim, head_v_dim]``.

    Returns:
        ``(output [B, H, head_v_dim], new_state)``
    """
    dk = query.shape[-1]
    query = query * (dk**-0.5)

    state = state * g.exp().unsqueeze(-1).unsqueeze(-1)
    # S^T k_t: contract over the key dim.
    kv_mem = (state * key.unsqueeze(-1)).sum(dim=-2)
    delta = (value - kv_mem) * beta.unsqueeze(-1)
    state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
    output = (state * query.unsqueeze(-1)).sum(dim=-2)
    return output, state


# ---------------------------------------------------------------------------
# The module
# ---------------------------------------------------------------------------


class Qwen3_5GatedDeltaNet(nn.Module):
    """Gated DeltaNet mixer, TP-sharded over value heads.

    Parallelism follows the plugin's attention modules: the input projections
    are column-parallel (heads split across ranks), ``out_proj`` is
    row-parallel, and prefill all-gathers from the sequence-parallel layout on
    entry and reduce-scatters back on exit while decode all-reduces.

    The two state tensors are bound from vLLM's KV cache by ``bind_kv_cache``
    and are updated in place.
    """

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_name = f"layers.{layer_idx}.linear_attn"
        self.dtype = config.torch_dtype
        self.state_dtype = config.ssm_dtype
        self.rms_norm_eps = config.rms_norm_eps
        self.hidden_size = config.hidden_size
        self.conv_kernel_size = config.linear_conv_kernel_dim

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        for name, heads in (
            ("linear_num_key_heads", config.linear_num_key_heads),
            ("linear_num_value_heads", config.linear_num_value_heads),
        ):
            if heads % self.world_size:
                raise ValueError(
                    f"{name}={heads} must be divisible by tp_size={self.world_size}"
                )
        self.num_k_heads = config.linear_num_key_heads // self.world_size
        self.num_v_heads = config.linear_num_value_heads // self.world_size
        if self.num_v_heads % self.num_k_heads:
            raise ValueError(
                f"per-rank value heads ({self.num_v_heads}) must be a multiple of "
                f"per-rank key heads ({self.num_k_heads})"
            )
        self.num_v_groups = self.num_v_heads // self.num_k_heads

        # ``recurrent_state`` is declared to vLLM as
        # ``(num_v_heads, head_v_dim, head_k_dim)`` but this module indexes it as
        # ``(num_v_heads, head_k_dim, head_v_dim)`` — the delta rule's natural
        # order, and the one HF uses. vLLM never reads the contents (its only
        # interest is the page size, and its state-copy hooks move whole blocks),
        # so the interpretation is ours to pick. That is only sound while the two
        # head dims agree, hence:
        if self.head_k_dim != self.head_v_dim:
            raise NotImplementedError(
                f"linear_key_head_dim ({self.head_k_dim}) != linear_value_head_dim "
                f"({self.head_v_dim}); the recurrent-state layout declared to vLLM "
                f"is (num_v_heads, head_v_dim, head_k_dim) and this module's "
                f"(num_v_heads, head_k_dim, head_v_dim) view would no longer match."
            )

        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.qkv_split = (self.key_dim, self.key_dim, self.value_dim)

        global_key_dim = config.linear_num_key_heads * self.head_k_dim
        global_value_dim = config.linear_num_value_heads * self.head_v_dim
        global_v_heads = config.linear_num_value_heads

        self.in_proj_qkv_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.conv_dim, dtype=self.dtype)
        )
        self.in_proj_z_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.value_dim, dtype=self.dtype)
        )
        # ``b`` (update strength) and ``a`` (decay) are both [num_v_heads, hidden]
        # and shard identically, so they ride in one projection.
        self.in_proj_ba_weight = nn.Parameter(
            torch.empty(self.hidden_size, 2 * self.num_v_heads, dtype=self.dtype)
        )
        self.conv1d_weight = nn.Parameter(
            torch.empty(self.conv_dim, self.conv_kernel_size, dtype=self.dtype)
        )
        # Kept in float32: they feed a softplus/exp that decides how fast the
        # state decays, and bf16 rounding there is visible over long sequences.
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32))
        self.norm_weight = nn.Parameter(torch.empty(self.head_v_dim, dtype=self.dtype))
        self.out_proj_weight = nn.Parameter(
            torch.empty(self.value_dim, self.hidden_size, dtype=self.dtype)
        )

        # Bound by ``bind_kv_cache``: one page-major float32 view of this
        # layer's share of the KV pool, [pages, page_floats]. Both states live
        # inside a single page so that block ``b`` stays inside page ``b`` --
        # the attention layer sharing this buffer relies on it. The conv window
        # is logically bf16 but is carried in float32 columns because a
        # page-major view is only contiguous at full page width, hence one dtype
        # for the whole page.
        self.state_pages: torch.Tensor | None = None
        self.conv_numel = (self.conv_kernel_size - 1) * self.conv_dim
        self.rec_numel = self.num_v_heads * self.head_k_dim * self.head_v_dim

        set_weight_loader(
            self.in_proj_qkv_weight,
            _segment_shard_loader(
                (global_key_dim, global_key_dim, global_value_dim),
                self.world_size,
                is_storage_transposed=True,
            ),
        )
        set_weight_loader(
            self.in_proj_z_weight,
            sharding_weight_loader(
                shard_dim=1,
                shard_size=self.value_dim,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )
        set_weight_loader(
            self.in_proj_ba_weight,
            _stacked_shard_loader(self.world_size, is_storage_transposed=True),
        )
        set_weight_loader(
            self.conv1d_weight,
            _segment_shard_loader(
                (global_key_dim, global_key_dim, global_value_dim),
                self.world_size,
                is_storage_transposed=False,
                squeeze_dim=1,
            ),
        )
        for param in (self.dt_bias, self.A_log):
            set_weight_loader(
                param,
                sharding_weight_loader(
                    shard_dim=0,
                    shard_size=global_v_heads // self.world_size,
                    num_shards=self.world_size,
                ),
            )
        set_weight_loader(
            self.out_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.value_dim,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    # ── State plumbing ───────────────────────────────────────────────────

    def bind_state_pages(self, pages: torch.Tensor) -> None:
        """Bind the page-major state view and check the page has room."""
        if pages.dim() != 2 or pages.shape[1] < self.conv_numel + self.rec_numel:
            raise ValueError(
                f"{self.layer_name}: state page view {tuple(pages.shape)} cannot "
                f"hold {self.conv_numel} conv + {self.rec_numel} recurrent floats"
            )
        self.state_pages = pages

    def _read_states(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather ``(conv_window, recurrent_state)`` for the batch's pages."""
        rows = self.state_pages.index_select(0, indices)
        n = rows.shape[0]
        conv = rows[:, : self.conv_numel].reshape(
            n, self.conv_kernel_size - 1, self.conv_dim
        )
        rec = rows[:, self.conv_numel : self.conv_numel + self.rec_numel].reshape(
            n, self.num_v_heads, self.head_k_dim, self.head_v_dim
        )
        return conv, rec

    def _write_states(
        self, indices: torch.Tensor, conv: torch.Tensor, rec: torch.Tensor
    ) -> None:
        """Scatter both states back, as one full-width row per page.

        One ``index_copy_`` rather than two narrow writes: a per-state column
        slice of the page is non-contiguous, and an in-place write through a
        strided view of a bound device tensor is what Neuron rejects. The
        leftover columns are the page padding vLLM added, so zeroing them is
        free.
        """
        n = indices.shape[0]
        parts = [conv.reshape(n, -1).float(), rec.reshape(n, -1).float()]
        pad = self.state_pages.shape[1] - self.conv_numel - self.rec_numel
        if pad:
            parts.append(
                torch.zeros(
                    n, pad, dtype=self.state_pages.dtype, device=indices.device
                )
            )
        self.state_pages.index_copy_(0, indices, torch.cat(parts, dim=1))

    def state_indices(
        self, metadata: dict, num_reqs: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-request read and write page for the batch, as ``(read, write)``.

        The recurrent group's ``mamba_block_size`` equals ``max_model_len``, so
        every sequence owns exactly one block and its state slot is simply that
        block's id.

        Padded batch rows need two different redirects, which is why this returns
        a pair. They must *read* zeros: their output is discarded but their
        logits are not, and the sampler's argmax reduces across the whole tile,
        so a dead row that reads another group's bytes as float32 state hands its
        NaN logits to every live row in the batch. And they must *write*
        somewhere else, since writing the zero page is what would stop it being
        zeros. The runner reserves one page for each, past ``num_blocks``.
        """
        block_table = metadata["block_table_tensor"]
        indices = block_table[:num_reqs, 0].to(torch.long)
        slot_mapping = metadata["slot_mapping"].view(num_reqs, -1)[:, 0]
        sink = self.state_pages.shape[0] - 1
        zero_page = sink - 1
        # ``> 0``, not ``>= 0``: the runner's padding sentinel changed from
        # PAD_SLOT_ID (-1) on 0.21 to NULL_BLOCK_ID (0) on 0.24, and 0 is vLLM's
        # reserved null block, so no live token ever maps there. Bound-check the
        # id too -- a padded batch row's block_table entry is simply stale from
        # an earlier step, and on device an out-of-range index is an out-of-bound
        # indirect DMA rather than a wrapped one.
        live = (slot_mapping > 0) & (indices > 0) & (indices < zero_page)
        return torch.where(
            live,
            indices,
            torch.full_like(indices, zero_page),
        ), torch.where(
            live,
            indices,
            torch.full_like(indices, sink),
        )

    # ── Shared pieces ────────────────────────────────────────────────────

    def _project(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``hidden -> (qkv, z, beta, g)``. ``beta``/``g`` come back float32."""
        hidden_states = hidden_states.to(self.dtype)
        qkv = hidden_states @ self.in_proj_qkv_weight
        z = hidden_states @ self.in_proj_z_weight
        ba = hidden_states @ self.in_proj_ba_weight
        # Plain slices, not ``Tensor.split``: ``split(sizes, dim=-1)`` silently
        # miscompiles on Neuron. Verified with ``probe_device_ops.py`` — the
        # device result is unrelated to the CPU one (relative error 1.2-1.4)
        # while ``t[..., a:b]``, ``torch.chunk`` and reshape-then-index are all
        # exact. This was the model's actual bug.
        num_v = self.num_v_heads
        b, a = ba[..., :num_v], ba[..., num_v:]

        beta = b.float().sigmoid()
        # ``-exp(A_log) * softplus(a + dt_bias)`` is <= 0, so ``exp(g)`` in the
        # recurrence is a decay in (0, 1]. Computed in float32 because
        # ``exp(A_log)`` overflows bf16 for the largest ``A_log`` in this
        # checkpoint.
        g = -self.A_log.exp() * F.softplus(a.float() + self.dt_bias)
        return qkv, z, beta, g

    def _causal_conv(self, qkv: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """Depthwise causal conv over the token dim, as a sum of shifted taps.

        ``out[t, c] = sum_j w[c, j] * x[t - (K - 1) + j, c]``, i.e. exactly what
        ``F.conv1d(..., groups=conv_dim, padding=K-1)[..., :T]`` computes.

        Written out rather than calling the grouped conv on purpose. A 1536-group
        depthwise conv1d is an exotic shape for neuronx-cc, and the NxDI reference
        avoids it too — it unrolls the taps in its cached paths and keeps its one
        ``F.conv1d`` call behind a flag that defaults to off. Four slices and four
        multiplies are trivially compilable, and this form also drops the two
        transposes the conv layout needed.
        """
        padded = F.pad(qkv, (0, 0, self.conv_kernel_size - 1, 0))
        out = padded[: num_tokens] * self.conv1d_weight[:, 0]
        for tap in range(1, self.conv_kernel_size):
            out = out + padded[tap : tap + num_tokens] * self.conv1d_weight[:, tap]
        return out

    @staticmethod
    def _tail_rows(
        rows_in: torch.Tensor, is_real: torch.Tensor, count: int
    ) -> torch.Tensor:
        """The last ``count`` real rows of ``rows_in``, without dynamic indexing.

        The obvious implementation — count the real tokens, then
        ``index_select`` at ``L - count .. L - 1`` — **silently miscompiles on
        Neuron**. Measured with ``probe_device_ops.py``: the device result is
        unrelated to the CPU one (relative error 1.5) while compiling without
        complaint. The index is data-dependent (it comes from a device-side
        ``sum``), and that is the part the compiler gets wrong.

        So select by *arithmetic* instead. With ``r[t] = 1`` for ``t < L``,
        ``r[t] - r[t+1]`` is a one-hot at ``t == L - 1``; shifting that left by
        ``k`` moves it to ``t == L - 1 - k``. Stacking the shifts gives a
        ``[count, T]`` selector matrix, and one matmul picks the rows. Every
        index is a compile-time constant.

        Rows that would fall before the start of the sequence (a prompt shorter
        than ``count``) come out zero on their own: the shift pushes the one-hot
        off the front. That matches a sequence starting from a cold state.
        """
        real = is_real.to(rows_in.dtype)
        # 1 at t == L - 1. F.pad supplies the r[T] == 0 the difference needs.
        onehot_last = real - F.pad(real[1:], (0, 1))
        selector = torch.stack(
            [
                F.pad(onehot_last[count - 1 - j :], (0, count - 1 - j))
                if count - 1 - j
                else onehot_last
                for j in range(count)
            ]
        )
        return selector @ rows_in

    def _split_heads(
        self, mixed: torch.Tensor, leading: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split post-conv ``[*leading, conv_dim]`` into per-head q, k, v.

        Key heads are broadcast up to the value-head count when the model has
        more value heads than key heads (not the case for the 2B checkpoint,
        where both are 16, but the 27B sibling has 48 value heads).
        """
        # Plain slices rather than ``Tensor.split`` — see ``_project``.
        key_dim, value_dim = self.key_dim, self.value_dim
        q = mixed[..., :key_dim]
        k = mixed[..., key_dim : 2 * key_dim]
        v = mixed[..., 2 * key_dim : 2 * key_dim + value_dim]
        q = q.reshape(*leading, self.num_k_heads, self.head_k_dim)
        k = k.reshape(*leading, self.num_k_heads, self.head_k_dim)
        v = v.reshape(*leading, self.num_v_heads, self.head_v_dim)
        if self.num_v_groups > 1:
            # Not ``repeat_interleave``: broadcasting a size-1 axis and folding
            # it into the head dim is exact -- output head ``h * groups + j``
            # reads input head ``h`` either way -- and lowers to a copy rather
            # than a gather. It was first tried as a fix for 27B's
            # NCC_IINAR001 'start_addr_active_channels' failure on a
            # ``pftranspose``, since this path (num_v_groups > 1) goes live for
            # the first time on 27B; it is *not* a fix -- the failing
            # instruction is byte-identical either way and -O2/-O3 is what
            # compiles -- but it is kept for the lowering.
            groups = self.num_v_groups
            expanded = (*leading, self.num_k_heads, groups, self.head_k_dim)
            folded = (*leading, self.num_v_heads, self.head_k_dim)
            q = q.unsqueeze(-2).expand(expanded).reshape(folded)
            k = k.unsqueeze(-2).expand(expanded).reshape(folded)
        return q, k, v

    def _output(
        self, core: torch.Tensor, z: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        """Per-head gated RMSNorm, then the row-parallel output projection.

        ``core`` is ``[num_tokens, num_v_heads, head_v_dim]``. Matches HF's
        ``Qwen3_5RMSNormGated``: normalise in float32, scale by the learned
        weight in the model dtype, then gate by ``silu(z)`` in float32.
        """
        core = core.float()
        variance = core.pow(2).mean(-1, keepdim=True)
        core = core * torch.rsqrt(variance + self.rms_norm_eps)
        core = self.norm_weight * core.to(self.dtype)
        core = core * F.silu(z.reshape_as(core).float())
        core = core.to(self.dtype).reshape(num_tokens, self.value_dim)
        return core @ self.out_proj_weight

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict,
    ) -> torch.Tensor:
        metadata = attn_metadata[self.layer_name]
        if metadata["max_query_len"] <= metadata["decode_token_threshold"]:
            return self.forward_decode(hidden_states, metadata)
        if self.world_size > 1 and SEQUENCE_PARALLEL:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return self.forward_prefill(hidden_states, positions, metadata)

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        metadata: dict,
    ) -> torch.Tensor:
        """Prefill one sequence from a zero state, then persist the final state.

        The plugin pads a prefill to a compiled bucket by appending token id 0
        and repeating the last real position, so ``positions`` no longer
        advances once the real prompt ends. That gives a padding mask for free:
        a token at index ``i`` is real iff ``positions[i] - positions[0] == i``.
        Zeroing ``g`` and ``beta`` at the pads makes them true no-ops — decay
        becomes ``exp(0) == 1`` and the delta update becomes zero — so the state
        this writes out is the state after the *last real* token.
        """
        num_tokens = hidden_states.shape[0]
        device = hidden_states.device

        offsets = torch.arange(num_tokens, device=device, dtype=positions.dtype)
        is_real = (positions - positions[0]) == offsets
        real_f32 = is_real.to(torch.float32).unsqueeze(-1)

        qkv, z, beta, g = self._project(hidden_states)
        # Pads carry a real embedding (token id 0), so they have to be zeroed
        # before the conv, whose kernel would otherwise smear them backwards
        # into the last few real positions.
        qkv = qkv * real_f32.to(qkv.dtype)

        mixed = F.silu(self._causal_conv(qkv, num_tokens))

        q, k, v = self._split_heads(mixed, (num_tokens,))
        beta = beta * real_f32
        g = g * real_f32

        # -> [1, heads, T, dim] for the chunked kernel.
        def to_bht(x: torch.Tensor) -> torch.Tensor:
            return x.unsqueeze(0).transpose(1, 2).float()

        initial_state = torch.zeros(
            1,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            dtype=torch.float32,
            device=device,
        )
        core, final_state = chunk_gated_delta_rule(
            l2norm(to_bht(q)),
            l2norm(to_bht(k)),
            to_bht(v),
            g.unsqueeze(0).transpose(1, 2),
            beta.unsqueeze(0).transpose(1, 2),
            initial_state,
        )
        core = core.squeeze(0).transpose(0, 1)  # [T, heads, head_v_dim]

        # Persist state for the one sequence being prefilled. The conv window is
        # the *pre*-conv q/k/v at the last ``kernel - 1`` real positions; where
        # the prompt is shorter than the window the missing (negative) slots
        # stay zero, matching a sequence that started from a cold state.
        new_conv_state = self._tail_rows(qkv, is_real, self.conv_kernel_size - 1)

        _, write_idx = self.state_indices(metadata, 1)
        self._write_states(write_idx, new_conv_state.unsqueeze(0), final_state)

        output = self._output(core, z, num_tokens)
        if self.world_size > 1:
            if SEQUENCE_PARALLEL:
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                self.tp_group.all_reduce(output)
        return output.contiguous()

    def forward_decode(
        self, hidden_states: torch.Tensor, metadata: dict
    ) -> torch.Tensor:
        """One recurrent step for each of the batch's sequences."""
        num_reqs = metadata["block_table_tensor"].shape[0]
        num_tokens = hidden_states.shape[0]
        if num_tokens != num_reqs:
            raise NotImplementedError(
                f"DeltaNet decode expects one token per request, got "
                f"{num_tokens} tokens for {num_reqs} requests. Multi-token "
                f"decode (speculative decoding) needs the recurrence to be "
                f"stepped once per draft token."
            )

        qkv, z, beta, g = self._project(hidden_states)
        read_idx, write_idx = self.state_indices(metadata, num_reqs)

        # Depthwise conv as a single fused window: the cached tail plus this
        # token, contracted against the kernel.
        conv_prev, state = self._read_states(read_idx)
        conv_window = torch.cat([conv_prev.to(qkv.dtype), qkv.unsqueeze(1)], dim=1)
        mixed = F.silu((conv_window * self.conv1d_weight.t()).sum(dim=1))

        q, k, v = self._split_heads(mixed, (num_reqs,))
        core, new_state = recurrent_gated_delta_rule_step(
            l2norm(q.float()),
            l2norm(k.float()),
            v.float(),
            g,
            beta,
            state,
        )
        # Both states go back in one write, so the conv update waits for the
        # recurrence. Nothing reads either state in between.
        self._write_states(write_idx, conv_window[:, 1:], new_state)

        output = self._output(core, z, num_reqs)
        if self.world_size > 1:
            self.tp_group.all_reduce(output)
        return output
