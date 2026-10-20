# SPDX-License-Identifier: Apache-2.0
"""Functional wrapper for the GPT-OSS decode "mega" giga-kernel (GK).

``gpt_oss_decode_mega_kernel`` fuses the WHOLE decoder stack (``num_layers``
DP x TP attention + EP MoE layers) into one NEFF for the decode step. It takes:

- ``x_shard`` — this rank's SP token shard ``[B/(DP*TP), H]``
- per-layer weights STACKED on a leading ``[L, ...]`` axis (NKI rejects Python
  lists as kernel args; integer-indexing an HBM tensor removes dim 0, so
  ``W[i]`` is layer ``i``'s tensor in its natural per-layer shape)
- shared, layer-invariant attention control (RoPE cos/sin, full + trimmed-SWA
  block tables, active mask, positions, kv-update index)
- two ``ReplicaGroup``s: the TP mesh and the EP mesh
- packed-FP8 KV scales (layer-invariant; ``None`` => bf16/fp16 KV)

It returns the final post-stack SP token shard ``[B/(DP*TP), H]`` and, by default,
the in-place-updated K/V caches for input<->output aliasing
(``VLLM_NEURON_MEGA_KV_UPDATE=0`` disables the KV write-back).

This is a DECODE-ONLY, TRN3-only, MXFP4 kernel; there is no torch fallback here
(prefill keeps the standard per-layer ``model_mxfp4`` path). The vendored kernel
lives under ``nkilib.models.gpt_oss.c128_giga_kernel`` and is only importable in
an environment where the workspace nkilib is installed (e.g. on the Trainium
host via kaena-remote), so the import is guarded and ``can_use_gpt_oss_mega_decode``
returns ``False`` when it is absent.
"""

import logging

import nki
import nki.language as nl
from torch import Tensor

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nki.collectives import ReplicaGroup as _ReplicaGroup

from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

# The mega kernel is vendored under nkilib.models.* and is only present when the
# workspace KaenaNeuronKernelLibrary is installed (device / kaena-remote). Guard
# the import so this module stays importable on a plain CPU host — the perf model
# imports NF at construction time, long before any decode runs.
try:
    from nkilib.models.gpt_oss.c128_giga_kernel.experimental.gpt_oss.gpt_oss_decode_layer_dp_tp_ep import (  # noqa: E501
        gpt_oss_decode_mega_kernel as _gpt_oss_decode_mega_kernel,
    )

    _MEGA_KERNEL_AVAILABLE = True
except Exception as e:  # pragma: no cover - import availability is env-dependent
    _gpt_oss_decode_mega_kernel = None
    _MEGA_KERNEL_AVAILABLE = False
    logger.info(
        "gpt_oss_decode_mega_kernel unavailable (%s); the GK decode path will "
        "raise if selected. This is expected off-device.",
        e,
    )

if _MEGA_KERNEL_AVAILABLE:

    @nki.jit
    def _gpt_oss_mega_decode_wrapper(
        x_shard: nl.ndarray,
        # per-layer weights, stacked [L, ...]
        input_layernorm_weight: nl.ndarray,
        W_qkv: nl.ndarray,
        bias_qkv: nl.ndarray,
        W_out: nl.ndarray,
        bias_out: nl.ndarray,
        sink: nl.ndarray,
        K_cache: nl.ndarray,
        V_cache: nl.ndarray,
        post_attn_layernorm_weight: nl.ndarray,
        router_weights: nl.ndarray,
        router_bias: nl.ndarray,
        expert_gate_up_weights: nl.ndarray,
        expert_down_weights: nl.ndarray,
        expert_gate_up_weights_scale: nl.ndarray,
        expert_down_weights_scale: nl.ndarray,
        expert_gate_up_bias: nl.ndarray,
        expert_down_bias: nl.ndarray,
        swa_start_pos_ids: nl.ndarray,
        # shared attention control
        cos: nl.ndarray,
        sin: nl.ndarray,
        active_blocks_table: nl.ndarray,
        active_blocks_table_swa: nl.ndarray,
        attention_mask: nl.ndarray,
        kv_cache_update_idx: nl.ndarray,
        pos_ids: nl.ndarray,
        pos_ids_swa: nl.ndarray,
        rank_id: nl.ndarray,
        # rank meshes as plain tuples-of-tuples -> ReplicaGroup, in-trace
        tp_replica_ranks: tuple = (),
        ep_replica_ranks: tuple = (),
        num_layers: int = 0,
        tp_size: int = 0,
        ep_size: int = 0,
        B_attn: int = 0,
        S_tkg: int = 0,
        top_k: int = 0,
        eps: float = 1e-5,
        hidden_actual: int = 0,
        softmax_scale: float = 1.0,
        gate_clamp_upper_limit: float = 0.0,
        up_clamp_upper_limit: float = 0.0,
        up_clamp_lower_limit: float = 0.0,
        k_scale: nl.ndarray = None,
        v_scale: nl.ndarray = None,
        kv_cache_update_idx_swa: nl.ndarray = None,
        x_first_full: nl.ndarray = None,
        # Per-buffer HMA KV caches (base-0), passed when the stacked K_cache/V_cache are None.
        # They come BEFORE the fusion inputs, which is the kernel's own order.
        K_cache_0: nl.ndarray = None,
        K_cache_1: nl.ndarray = None,
        K_cache_2: nl.ndarray = None,
        K_cache_3: nl.ndarray = None,
        K_cache_4: nl.ndarray = None,
        K_cache_5: nl.ndarray = None,
        K_cache_6: nl.ndarray = None,
        K_cache_7: nl.ndarray = None,
        K_cache_8: nl.ndarray = None,
        K_cache_9: nl.ndarray = None,
        K_cache_10: nl.ndarray = None,
        K_cache_11: nl.ndarray = None,
        K_cache_12: nl.ndarray = None,
        K_cache_13: nl.ndarray = None,
        K_cache_14: nl.ndarray = None,
        K_cache_15: nl.ndarray = None,
        K_cache_16: nl.ndarray = None,
        K_cache_17: nl.ndarray = None,
        V_cache_0: nl.ndarray = None,
        V_cache_1: nl.ndarray = None,
        V_cache_2: nl.ndarray = None,
        V_cache_3: nl.ndarray = None,
        V_cache_4: nl.ndarray = None,
        V_cache_5: nl.ndarray = None,
        V_cache_6: nl.ndarray = None,
        V_cache_7: nl.ndarray = None,
        V_cache_8: nl.ndarray = None,
        V_cache_9: nl.ndarray = None,
        V_cache_10: nl.ndarray = None,
        V_cache_11: nl.ndarray = None,
        V_cache_12: nl.ndarray = None,
        V_cache_13: nl.ndarray = None,
        V_cache_14: nl.ndarray = None,
        V_cache_15: nl.ndarray = None,
        V_cache_16: nl.ndarray = None,
        V_cache_17: nl.ndarray = None,
        embed_weight: nl.ndarray = None,
        embed_ids: nl.ndarray = None,
        embed_x_shard_ids: nl.ndarray = None,
        inv_freq: nl.ndarray = None,
        rope_concentration: float = 1.0,
    ):
        """Thin @nki.jit shim: rebuild ReplicaGroups from tuple meshes, then call
        the mega kernel. Mirrors ``attention_segmented_cte``'s tuple->ReplicaGroup
        convention (torch/Dynamo cannot carry a ReplicaGroup across the HOP
        boundary, so ranks are passed as static tuples and reconstructed here).

        The MXFP4 expert weights are stored in the model as ``uint16``/``uint8``
        packed tiles; reinterpret the uint16 block tensors as
        ``float4_e2m1fn_x4`` so the kernel's MoE matmul sees the MX dtype (the
        uint8 scales stay uint8). Mirrors ``moe_block_tkg_wrapper``.
        """
        if expert_gate_up_weights.dtype == nl.uint16:
            expert_gate_up_weights = expert_gate_up_weights.view(nl.float4_e2m1fn_x4)
        if expert_down_weights.dtype == nl.uint16:
            expert_down_weights = expert_down_weights.view(nl.float4_e2m1fn_x4)

        # ReplicaGroup accepts the tuple-of-tuples directly (List[List[int]]);
        # do NOT use a list-comprehension here — the NKI frontend rejects it
        # inside @nki.jit ("unsupported expression"). The public wrapper already
        # normalized these to nested int tuples.
        tp_rg = _ReplicaGroup(tp_replica_ranks)
        ep_rg = _ReplicaGroup(ep_replica_ranks)

        return _gpt_oss_decode_mega_kernel(
            x_shard,
            input_layernorm_weight,
            W_qkv,
            bias_qkv,
            W_out,
            bias_out,
            sink,
            K_cache,
            V_cache,
            post_attn_layernorm_weight,
            router_weights,
            router_bias,
            expert_gate_up_weights,
            expert_down_weights,
            expert_gate_up_weights_scale,
            expert_down_weights_scale,
            expert_gate_up_bias,
            expert_down_bias,
            swa_start_pos_ids,
            cos,
            sin,
            active_blocks_table,
            active_blocks_table_swa,
            attention_mask,
            kv_cache_update_idx,
            pos_ids,
            pos_ids_swa,
            rank_id,
            tp_rg,
            ep_rg,
            num_layers,
            tp_size,
            ep_size,
            B_attn,
            S_tkg,
            top_k,
            eps,
            hidden_actual,
            softmax_scale,
            gate_clamp_upper_limit,
            up_clamp_upper_limit,
            up_clamp_lower_limit,
            k_scale,
            v_scale,
            kv_cache_update_idx_swa,
            x_first_full,
            # Pass the per-buffer KV caches BY KEYWORD: the kernel appends them at the END of its
            # signature, after the fusion inputs.
            K_cache_0=K_cache_0,
            K_cache_1=K_cache_1,
            K_cache_2=K_cache_2,
            K_cache_3=K_cache_3,
            K_cache_4=K_cache_4,
            K_cache_5=K_cache_5,
            K_cache_6=K_cache_6,
            K_cache_7=K_cache_7,
            K_cache_8=K_cache_8,
            K_cache_9=K_cache_9,
            K_cache_10=K_cache_10,
            K_cache_11=K_cache_11,
            K_cache_12=K_cache_12,
            K_cache_13=K_cache_13,
            K_cache_14=K_cache_14,
            K_cache_15=K_cache_15,
            K_cache_16=K_cache_16,
            K_cache_17=K_cache_17,
            V_cache_0=V_cache_0,
            V_cache_1=V_cache_1,
            V_cache_2=V_cache_2,
            V_cache_3=V_cache_3,
            V_cache_4=V_cache_4,
            V_cache_5=V_cache_5,
            V_cache_6=V_cache_6,
            V_cache_7=V_cache_7,
            V_cache_8=V_cache_8,
            V_cache_9=V_cache_9,
            V_cache_10=V_cache_10,
            V_cache_11=V_cache_11,
            V_cache_12=V_cache_12,
            V_cache_13=V_cache_13,
            V_cache_14=V_cache_14,
            V_cache_15=V_cache_15,
            V_cache_16=V_cache_16,
            V_cache_17=V_cache_17,
            # Pass the fusion inputs BY KEYWORD: pregen_mask_full and pregen_mask_swa take the next
            # two positions, so a positional argument would land on a mask.
            embed_weight=embed_weight,
            embed_ids=embed_ids,
            embed_x_shard_ids=embed_x_shard_ids,
            inv_freq=inv_freq,
            rope_concentration=rope_concentration,
        )

    # lnc=1 -> grid size 1 (see model_mxfp4_perf / --logical-nc-config=1).
    _wrapped_gpt_oss_mega_decode = wrap_nki(_gpt_oss_mega_decode_wrapper)
else:
    _wrapped_gpt_oss_mega_decode = None


def can_use_gpt_oss_mega_decode(x_shard: Tensor) -> bool:
    """Whether the GK decode kernel can run for this input.

    Requires the vendored kernel to be importable AND a Neuron device (or the
    NKI simulator). There is no torch fallback: the caller must keep the
    standard per-layer decode path for the ``False`` case.
    """
    if not _MEGA_KERNEL_AVAILABLE:
        return False
    return can_run_kernel(x_shard)


def gpt_oss_mega_decode(
    x_shard: Tensor,
    *,
    input_layernorm_weight: Tensor,
    W_qkv: Tensor,
    bias_qkv: Tensor,
    W_out: Tensor,
    bias_out: Tensor,
    sink: Tensor,
    K_cache: Tensor,
    V_cache: Tensor,
    post_attn_layernorm_weight: Tensor,
    router_weights: Tensor,
    router_bias: Tensor,
    expert_gate_up_weights: Tensor,
    expert_down_weights: Tensor,
    expert_gate_up_weights_scale: Tensor,
    expert_down_weights_scale: Tensor,
    expert_gate_up_bias: Tensor,
    expert_down_bias: Tensor,
    swa_start_pos_ids: Tensor,
    cos: Tensor,
    sin: Tensor,
    active_blocks_table: Tensor,
    active_blocks_table_swa: Tensor,
    attention_mask: Tensor,
    kv_cache_update_idx: Tensor,
    pos_ids: Tensor,
    pos_ids_swa: Tensor,
    rank_id: Tensor,
    tp_replica_ranks: tuple,
    ep_replica_ranks: tuple,
    num_layers: int,
    tp_size: int,
    ep_size: int,
    B_attn: int,
    S_tkg: int,
    top_k: int,
    eps: float,
    hidden_actual: int,
    softmax_scale: float,
    gate_clamp_upper_limit: float,
    up_clamp_upper_limit: float,
    up_clamp_lower_limit: float,
    k_scale: Tensor | None = None,
    v_scale: Tensor | None = None,
    kv_cache_update_idx_swa: Tensor | None = None,
    x_first_full: Tensor | None = None,
    k_bufs: list[Tensor] | None = None,
    v_bufs: list[Tensor] | None = None,
    embed_weight: Tensor | None = None,
    embed_ids: Tensor | None = None,
    embed_x_shard_ids: Tensor | None = None,
    inv_freq: Tensor | None = None,
    rope_concentration: float = 1.0,
):
    """Run the fused whole-stack GPT-OSS decode giga-kernel.

    All per-layer weight tensors must be stacked on a leading ``[num_layers, ...]``
    axis (see module docstring); the attention-control tensors are shared across
    layers. ``tp_replica_ranks`` / ``ep_replica_ranks`` are tuples-of-tuples of
    global ranks describing the TP and EP collective meshes.

    ``x_first_full`` is the OPTIONAL pre-gathered first-layer attention input
    ``[B/DP, H]``. The decode hidden states are TP-replicated, so layer 0's
    AllGather[TP] (SP shard -> ``[B/DP, H]``) is redundant; pass the full tensor
    to skip it. ``x_shard`` still carries layer 0's ``[B/(DP*TP), H]`` residual
    slice. ``None`` keeps the layer-0 AllGather.

    ``k_bufs`` / ``v_bufs`` give the HMA KV cache as one BASE-0 tensor for each group,
    in place of the stacked ``K_cache`` / ``V_cache``. Pass the per-group lists and set
    ``K_cache=None`` and ``V_cache=None``; the kernel counts the buffers it receives and
    pairs SWA layer ``2j`` with full layer ``2j+1`` on buffer ``j``. Each buffer then
    indexes from offset 0, which keeps the block-KV DGE ``HWAPOffset`` small. The lists
    pad to the kernel's 18 slots, and the unused slots stay ``None``.

    ``embed_weight`` (this rank's embedding shard) plus ``embed_ids`` (the token
    indices) turn on the FUSED EMBEDDING. The kernel then builds layer 0's input by
    an in-NEFF gather plus a TP collective, so the caller does not compute the hidden
    state and the hidden state does not go to HBM. Pass ``x_first_full=None`` and a
    placeholder ``x_shard`` in that mode. ``None`` keeps the external-hidden path.

    The kernel selects one of two shardings from ``embed_weight``'s last dim.
    VOCAB-dim (``[vocab_per_rank, H]``): ``embed_ids`` are LOCAL vocab indices
    (``-1`` = outside this rank's shard), and the kernel does a reduce_scatter plus
    an all_gather. HIDDEN-dim (``[vocab, H/tp_size]``): ``embed_ids`` are GLOBAL
    token ids, ``embed_x_shard_ids`` gives this rank's token rows, and the kernel
    does ONE hidden-dim all_gather with no cross-rank sum.

    ``inv_freq`` (``[head_dim/2, 1]``) plus ``rope_concentration`` turn on the FUSED
    RoPE. The kernel computes the cos/sin tables on-chip from ``pos_ids``, so the
    caller does not run the host rotary embedding. ``None`` keeps the ``cos``/``sin``
    inputs.

    Returns the final SP token shard ``[B/(DP*TP), H]``. By default the kernel
    writes the KV cache in place and additionally returns the updated
    ``(K_cache, V_cache)`` for input<->output aliasing
    (``VLLM_NEURON_MEGA_KV_UPDATE=0`` disables it).

    Raises:
        RuntimeError: if the vendored kernel is unavailable or the tensors are on
            CPU without the NKI simulator (this is a device-only kernel with no
            torch reference).

    Example:
        >>> # on a Trainium host, with stacked weights + shared control tensors
        >>> out = gpt_oss_mega_decode(x_shard, input_layernorm_weight=iln, ...)
        >>> out.shape
        torch.Size([B_local, H])
    """
    if not can_use_gpt_oss_mega_decode(x_shard):
        raise RuntimeError(
            "gpt_oss_mega_decode requires the vendored GK kernel and a Neuron "
            "device (or NKI simulator). Kernel available="
            f"{_MEGA_KERNEL_AVAILABLE}. There is no torch fallback for the "
            "decode giga-kernel; keep the standard per-layer decode path."
        )

    # Per-buffer HMA KV: pad the provided base-0 buffers to the kernel's fixed
    # slot count (extra slots stay None; the kernel counts non-None buffers).
    _MAX_GK_KV_BUFS = 18
    _k = (list(k_bufs) if k_bufs else []) + [None] * (
        _MAX_GK_KV_BUFS - len(k_bufs or [])
    )
    _v = (list(v_bufs) if v_bufs else []) + [None] * (
        _MAX_GK_KV_BUFS - len(v_bufs or [])
    )
    assert len(_k) == _MAX_GK_KV_BUFS and len(_v) == _MAX_GK_KV_BUFS, (
        f"GK per-buffer KV supports at most {_MAX_GK_KV_BUFS} buffers"
    )

    return _wrapped_gpt_oss_mega_decode[1](
        x_shard,
        input_layernorm_weight,
        W_qkv,
        bias_qkv,
        W_out,
        bias_out,
        sink,
        K_cache,
        V_cache,
        post_attn_layernorm_weight,
        router_weights,
        router_bias,
        expert_gate_up_weights,
        expert_down_weights,
        expert_gate_up_weights_scale,
        expert_down_weights_scale,
        expert_gate_up_bias,
        expert_down_bias,
        swa_start_pos_ids,
        cos,
        sin,
        active_blocks_table,
        active_blocks_table_swa,
        attention_mask,
        kv_cache_update_idx,
        pos_ids,
        pos_ids_swa,
        rank_id,
        tuple(tuple(int(r) for r in g) for g in tp_replica_ranks),
        tuple(tuple(int(r) for r in g) for g in ep_replica_ranks),
        num_layers,
        tp_size,
        ep_size,
        B_attn,
        S_tkg,
        top_k,
        eps,
        hidden_actual,
        softmax_scale,
        gate_clamp_upper_limit,
        up_clamp_upper_limit,
        up_clamp_lower_limit,
        k_scale,
        v_scale,
        kv_cache_update_idx_swa,
        x_first_full,
        *_k,
        *_v,
        embed_weight,
        embed_ids,
        embed_x_shard_ids,
        inv_freq,
        rope_concentration,
    )
