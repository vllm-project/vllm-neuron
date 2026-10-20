# SPDX-License-Identifier: Apache-2.0
"""a2a sequence-parallel MLP composer for Qwen3-32B dense MXFP8 decode (LNC1).

One launch absorbs the pre-MLP block boundary. From the UN-REDUCED attention
partials and the residual carrier it chains::

    carrier load -> MLP weight prefetch (DMAs stream under the collectives)
    -> all-to-all + fp32 PE reduce + residual add -> carrier store
    -> post-attention RMSNorm -> MX quantize -> packed SBUF all-gather
    -> in-tree decode MLP on the pre-quantized activation

and returns the next UN-REDUCED MLP partials plus the updated carrier; the
model loop closes the last carry with one torch reduce-scatter per forward.

Layouts
-------
``H = 5120``, ``T_s`` this rank's token shard, ``T = T_s * W`` at
tensor-parallel degree ``W`` (release shape: ``T_s = 8``, ``W = 8``, bs64).

- ``partials``: bf16 HBM ``[T, H]`` un-reduced attention block output.
- ``residual``: bf16 HBM ``[T_s, H]`` carrier (FLAT pre-shuffled mid-chain,
  natural at the chain ends — see the prologue module docstring).
- ``gamma_shuffled``: bf16 ``[1, H]`` post-attention RMSNorm weight in
  ``pre_shuffle_h`` order.
- weights/scales: identical to
  ``vllm_neuron/_kernels/mlp/qwen3_32b_mxfp8_mlp_tkg.py`` (uint32
  ``[128, H // 512, I]`` gate/up, ``[128, I // 512, H]`` down, uint8 scales).
- packed gather: uint8 ``[128, 400] -> [128, 3200]`` at the release shape —
  one collective, no HBM round trip.
- returns: bf16 HBM ``[T, H]`` partials + bf16 HBM ``[T_s, H]`` carrier.

Port notes (vs ``gpt_oss_decode_sp_residual_rms``, a2a variant)
---------------------------------------------------------------
Ported from the all-to-all variant of ``gpt_oss_decode_sp_residual_rms``;
the shared stages and their own port notes live in
``qwen3_32b_sp_prologue.py``. The deliberate changes in this composer:

1. Both-boundary carry. The composer returns UN-REDUCED partials for the next
   layer's a2a attention prologue instead of reduce-scattering its own output,
   so the per-boundary torch collectives disappear and ONE reduce-scatter per
   forward remains.
2. Flat pre-shuffled residual carrier between composers (natural order only at
   the chain ends) — one contiguous DMA per hop instead of a 2-byte-element
   strided scatter.
3. MLP weight prefetch before the collectives (``_preload_mlp_weights``: the
   scale DMAs plus the first gate/up ring slots), consumed by the MLP body as
   ``preloaded_weights``.
4. The MLP runs on the pre-quantized activation through ``_mlp_tkg_impl``'s
   ``hidden_q`` / ``hidden_s`` seam (PORT CHANGE #11 of the decode MLP kernel):
   no HBM load and no re-quantize of the gathered tokens.
"""

import nki
import nki.collectives as ncc
import nki.language as nl
from torch import Tensor

from vllm_neuron._kernels.mlp.qwen3_32b_mxfp8_mlp_tkg import (
    _RING_DEPTH_DEFAULT,
    _mlp_tkg_impl,
    _preload_mlp_weights,
)
from vllm_neuron._kernels.qwen3_32b_sp.qwen3_32b_sp_prologue import (
    sp_a2a_reduce_residual,
    sp_gather_replicated,
    sp_load_natural_shuffled,
    sp_load_shuffled_flat,
    sp_store_shuffled_flat,
    sp_store_shuffled_natural,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

_P = 128
_TILE = 512
_Q_WIDTH = 4
# Sequence-parallel decode runs at the validated global token count only
# (bs64, one token per sequence). Perf and accuracy gates cover this shape;
# every other count keeps the all-reduce path.
_SP_GLOBAL_TOKENS = 64

# Tested TP degrees. TP64 is out of scope: ``I/rank = 400`` is under one 512-tile.
_SUPPORTED_TP_DEGREES = (8, 16, 32)

# The MLP output transpose supports at most 128 tokens.
_MAX_GATHERED_TOKENS = 128

__all__ = [
    "qwen3_32b_mxfp8_sp_a2a_mlp_decode",
    "qwen3_sp_a2a_supports",
]


def _resolve_replica_mesh(group: object) -> tuple[tuple[int, ...], ...]:
    """Return the FULL replica mesh of ``group`` as a plain nested tuple, resolved
    eagerly outside the compile scope: ``group.ranks`` alone is only this rank's
    cell, and a single-cell ReplicaGroup is not SPMD consistent, so each rank
    would lower a different graph (same rule as
    ``vllm_neuron/functional/collectives/all_gather_v.py``).

    Args:
        group: The tensor-parallel GroupCoordinator.

    Returns:
        One tuple for each cell of the mesh.

    Example:
        >>> # TP8 x DP2 gives ((0, ..., 7), (8, ..., 15))
    """
    from libtorch_neuronx_lite.parallel.replica_mesh import resolve_full_mesh

    return tuple(tuple(cell) for cell in resolve_full_mesh(group))


def _qwen3_sp_a2a_mlp_body(
    partials: nl.ndarray,
    residual: nl.ndarray,
    gamma: nl.ndarray,
    gate_w: nl.ndarray,
    up_w: nl.ndarray,
    down_w: nl.ndarray,
    gate_s: nl.ndarray,
    up_s: nl.ndarray,
    down_s: nl.ndarray,
    eps: float,
    hidden_actual: int,
    tp_degree: int,
    replica_mesh: tuple,
    residual_in_shuffled: bool = False,
    residual_out_shuffled: bool = False,
):
    """All-to-all composer, the LNC1 form of the sequence-parallel MLP block: it
    absorbs the torch reduce-scatter boundary, so from the UN-REDUCED attention
    partials and the natural-order residual shard one launch runs the all-to-all,
    the fp32 PE reduction fused with the ``pre_shuffle_h`` transpose, the residual
    add, the post-attention RMSNorm, the MX quantize, the packed SBUF all-gather,
    and the MLP; ported from the all-to-all variant of
    ``gpt_oss_decode_sp_residual_rms``.

    Args:
        partials: bf16 HBM ``[T, H]`` un-reduced attention block partials.
        residual: bf16 HBM ``[T_s, H]`` residual shard, NATURAL order.
        gamma: bf16 HBM ``[1, H]`` post-attention RMSNorm gamma, ``pre_shuffle_h`` order.
        gate_w / up_w / down_w / gate_s / up_s / down_s: uint32 MXFP8 weights and uint8
            E8M0 scales, ``[128, H // 512, I]`` gate/up and ``[128, I // 512, H]`` down.
        eps: RMSNorm epsilon.
        hidden_actual: Unpadded hidden size that divides the sum of squares.
        tp_degree: Tensor-parallel degree.
        replica_mesh: Every cell of the tensor-parallel replica mesh.
        residual_in_shuffled: Load the residual carrier in the FLAT pre-shuffled layout.
        residual_out_shuffled: Write the residual carrier pre-shuffled for the next composer.

    Returns:
        ``[mlp_partials, residual_out]``: bf16 HBM ``[T, H]`` partial sums, bf16 HBM
        ``[T_s, H]`` natural-order residual.

    Example:
        >>> # traced by wrap_nki; not called directly
    """
    from vllm_neuron._kernels.norm.qwen3_32b_mxfp8_rmsnorm_quant_tkg import (
        _compute_rmsnorm,
        _load_gamma,
        _swizzle_to_mlp_layout,
    )
    from vllm_neuron._kernels.mlp.qwen3_32b_mxfp8_mlp_tkg import (
        _quantize_mx_3d,
    )

    t_full, h = partials.shape
    t_shard = t_full // tp_degree
    n512 = h // 512
    h1 = h // 128

    cells = []
    for cell in replica_mesh:
        cells.append(list(cell))
    rg = ncc.ReplicaGroup(cells)

    # Small DMAs first (they gate the critical path), then the a2a region.
    gamma_sb = _load_gamma(gamma, h1)
    if residual_in_shuffled:
        residual_sb = sp_load_shuffled_flat(residual, t_shard, h, "a2a_resid")
    else:
        residual_sb = sp_load_natural_shuffled(residual, t_shard, h, "a2a_resid")
    # Prefetch: the MLP weight DMAs stream on the DGE queues under the collective.
    preloaded_weights = _preload_mlp_weights(
        gate_w, up_w, down_w, gate_s, up_s, down_s, _RING_DEPTH_DEFAULT
    )

    state = sp_a2a_reduce_residual(partials, residual_sb, tp_degree, rg)

    # Return the updated residual stream to torch in natural order.
    residual_out = nl.ndarray((t_shard, h), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    if residual_out_shuffled:
        sp_store_shuffled_flat(state, residual_out, t_shard, h)
    else:
        sp_store_shuffled_natural(state, residual_out, t_shard, h)

    # RMSNorm + MX quantize, then the packed gather and the MLP, at grid 1.
    normed = _compute_rmsnorm(state, gamma_sb, eps, hidden_actual, t_shard, h1)
    swizzled = _swizzle_to_mlp_layout(normed, n512, t_shard, h1)
    q_n, s_n = _quantize_mx_3d(swizzled, n512, t_shard, "a2a_norm")
    q_act, s_act = sp_gather_replicated(q_n, s_n, t_shard, h, tp_degree, rg)

    mlp_out = _mlp_tkg_impl(
        None,
        gate_w,
        up_w,
        down_w,
        gate_s,
        up_s,
        down_s,
        _RING_DEPTH_DEFAULT,
        hidden_q=q_act,
        hidden_s=s_act,
        preloaded_weights=preloaded_weights,
    )
    return [mlp_out, residual_out]


# Trace-mode entries for wrap_nki; the plain bodies also serve the kernel tests.
_qwen3_sp_a2a_mlp_trace = nki.jit(mode="trace")(_qwen3_sp_a2a_mlp_body)


def qwen3_sp_a2a_supports(
    hidden_size: int,
    tokens_per_shard: int,
    tp_degree: int,
    lnc: int,
) -> bool:
    """Return True when the LNC1 all-to-all MLP block can run: the GLOBAL token
    count ``T_s * tp_degree`` must equal ``_SP_GLOBAL_TOKENS`` (the validated
    bs64 release shape), the quantize needs whole 4-token groups, the PSUM
    eviction needs ``H // 128`` divisible by the 8 banks, and the composer
    targets LNC=1 only. The reducer matmul also caps the global count at 128
    stationary partitions, but the bs64 pin is the tighter rule.

    Args:
        hidden_size: Hidden size ``H``.
        tokens_per_shard: This rank's decode token shard ``T_s``.
        tp_degree: Tensor-parallel degree.
        lnc: Logical NeuronCore degree of the process.

    Returns:
        True when the a2a path runs, else False.

    Example:
        >>> qwen3_sp_a2a_supports(5120, 8, 8, 1)
        True
        >>> qwen3_sp_a2a_supports(5120, 8, 8, 2)
        False
    """
    if not can_run_kernel():
        return False
    if lnc != 1:
        return False
    if tokens_per_shard * tp_degree != _SP_GLOBAL_TOKENS:
        return False
    if tokens_per_shard % _Q_WIDTH or tokens_per_shard * tp_degree > _P:
        return False
    if hidden_size % _TILE or (hidden_size // _P) % 8:
        return False
    return True


def qwen3_32b_mxfp8_sp_a2a_mlp_decode(
    partials: Tensor,
    residual: Tensor,
    gate_w: Tensor,
    up_w: Tensor,
    down_w: Tensor,
    gate_s: Tensor,
    up_s: Tensor,
    down_s: Tensor,
    *,
    gamma_shuffled: Tensor,
    eps: float,
    tp_group: object,
    residual_in_shuffled: bool = False,
    residual_out_shuffled: bool = False,
) -> tuple[Tensor, Tensor]:
    """Run the LNC1 all-to-all sequence-parallel MLP block: one launch runs the
    all-to-all of the un-reduced attention partials, the fp32 PE reduction fused
    with the layout transpose, the residual add, the post-attention RMSNorm, the
    MX quantize, the packed SBUF all-gather and the MLP; the caller
    reduce-scatters the returned MLP partials, keeps the returned residual stream
    for the next block, and must check ``qwen3_sp_a2a_supports`` before the call.

    Args:
        partials: bf16 ``[T, H]`` un-reduced attention block partial sums.
        residual: bf16 ``[T_s, H]`` residual shard, NATURAL order.
        gate_w / up_w / down_w / gate_s / up_s / down_s: uint32 MXFP8 weights and uint8
            E8M0 scales, ``[128, H // 512, I]`` gate/up and ``[128, I // 512, H]`` down.
        gamma_shuffled: bf16 ``[1, H]`` post-attention RMSNorm gamma, ``pre_shuffle_h`` order.
        eps: RMSNorm epsilon.
        tp_group: The tensor-parallel GroupCoordinator.

    Returns:
        ``(mlp_partials, residual_out)``: bf16 ``[T, H]`` partial sums, bf16
        ``[T_s, H]`` natural-order residual.

    Example:
        >>> # qwen3_32b_mxfp8_sp_a2a_mlp_decode(..., gamma_shuffled=g, eps=1e-6, tp_group=tp)
    """
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from vllm_neuron.utils.hardware_config import get_logical_nc_config

    t_full, hidden_size = partials.shape
    tp_degree = tp_group.world_size
    lnc = get_logical_nc_config()
    assert qwen3_sp_a2a_supports(hidden_size, t_full // tp_degree, tp_degree, lnc), (
        f"the a2a sequence-parallel MLP refuses T={t_full}, H={hidden_size}, "
        f"TP={tp_degree}, LNC={lnc}. Check qwen3_sp_a2a_supports first."
    )

    wrapped = wrap_nki(_qwen3_sp_a2a_mlp_trace)
    result = wrapped[lnc](
        partials=partials,
        residual=residual,
        gamma=gamma_shuffled,
        gate_w=gate_w,
        up_w=up_w,
        down_w=down_w,
        gate_s=gate_s,
        up_s=up_s,
        down_s=down_s,
        eps=eps,
        hidden_actual=hidden_size,
        tp_degree=tp_degree,
        replica_mesh=_resolve_replica_mesh(tp_group),
        residual_in_shuffled=residual_in_shuffled,
        residual_out_shuffled=residual_out_shuffled,
    )
    return result[0], result[1]


# Process-wide dedupe: a per-layer log would print 64 lines for each rank.
_sp_norm_mlp_logged: set[tuple[int, int, int, bool]] = set()
