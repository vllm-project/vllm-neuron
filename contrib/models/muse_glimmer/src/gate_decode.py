# SPDX-License-Identifier: Apache-2.0
"""gated decode attention with the output projection inside the NKI megakernel.

``gated_attention_decode`` computes, per decode step and TP rank,

    Xn  = rmsnorm(X) * gamma            (only if rmsnorm_X_enabled; else Xn = X)
    q, k, v = Xn @ W_qkv  (+ QK-norm, RoPE) -> attention over the paged cache (update_cache=False)
    out = (attn * sigmoid(Xn @ W_gate) * gate_dq) @ W_out          -> [B*S, H] (pre all-reduce)

On Neuron (or under NKI_SIMULATOR) it runs ``nki_attn_gate_tkg.attention_block_tkg`` (nkilib 0.7.0 copy with
the gate applied in SBUF before ``output_projection_tkg``). Elsewhere it runs the stock torch reference
(``_torch_attention_decode_impl`` without W_out) followed by the gate and o_proj in torch.
Returns ``(out, K_new, V_new)`` with the same K/V layouts as ``NF.attention_decode(update_cache=False)``.
"""
from __future__ import annotations

from typing import Optional

import nki
import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.core.utils.common_types import QuantizationType

from vllm_neuron.functional.attention.attention_decode import (
    _build_default_active_mask,
    _can_use_attention_block_kernel,
    _torch_attention_decode_impl,
)
from vllm_neuron.functional.attention.attention_decode_mask import gen_attention_decode_mask
from vllm_neuron.utils.hardware_config import get_logical_nc_config
from vllm_neuron.utils.neuron_utils import can_run_kernel

from .kernels.nki_attn_gate_tkg import attention_block_tkg as _gated_block


@nki.jit
def _gated_attention_block_kernel(
    X,
    rmsnorm_X_enabled: bool = False,
    rmsnorm_X_eps: float = None,
    rmsnorm_X_gamma=None,
    W_qkv=None,
    quantization_type_qkv: QuantizationType = QuantizationType.NONE,
    weight_dequant_scale_qkv=None,
    rmsnorm_QK_pre_rope_enabled: bool = False,
    rmsnorm_QK_pre_rope_eps: float = 1e-6,
    cos=None,
    sin=None,
    active_blocks_table=None,
    K_cache=None,
    V_cache=None,
    attention_mask=None,
    softmax_scale: float = None,
    pos_ids=None,
    swa_start_pos_ids=None,
    k_scale=None,
    v_scale=None,
    W_out=None,
    W_gate=None,
    W_gate_scale=None,
    gate_dq: float = 1.0,
):
    return _gated_block(
        X=X,
        X_hidden_dim_actual=None,
        rmsnorm_X_enabled=rmsnorm_X_enabled,
        rmsnorm_X_eps=rmsnorm_X_eps,
        rmsnorm_X_gamma=rmsnorm_X_gamma,
        W_qkv=W_qkv,
        bias_qkv=None,
        quantization_type_qkv=quantization_type_qkv,
        weight_dequant_scale_qkv=weight_dequant_scale_qkv,
        input_dequant_scale_qkv=None,
        rmsnorm_QK_pre_rope_enabled=rmsnorm_QK_pre_rope_enabled,
        rmsnorm_QK_pre_rope_eps=rmsnorm_QK_pre_rope_eps,
        rmsnorm_QK_pre_rope_W_Q=None,
        rmsnorm_QK_pre_rope_W_K=None,
        cos=cos,
        sin=sin,
        rope_contiguous_layout=True,
        rmsnorm_QK_post_rope_enabled=False,
        rmsnorm_QK_post_rope_eps=1e-6,
        rmsnorm_QK_post_rope_W_Q=None,
        rmsnorm_QK_post_rope_W_K=None,
        K_cache_transposed=False,
        active_blocks_table=active_blocks_table,
        K_cache=K_cache,
        V_cache=V_cache,
        attention_mask=attention_mask,
        sink=None,
        update_cache=False,
        kv_cache_update_idx=None,
        W_out=W_out,
        bias_out=None,
        quantization_type_out=QuantizationType.NONE,
        weight_dequant_scale_out=None,
        input_dequant_scale_out=None,
        transposed_out=False,
        out_in_sb=False,
        softmax_scale=softmax_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        sbm=None,
        KVDP=1,
        KVDP_replica_group=None,
        pos_ids=pos_ids,
        swa_start_pos_ids=swa_start_pos_ids,
        S_ctx=None,
        W_gate=W_gate,
        W_gate_scale=W_gate_scale,
        gate_dq=gate_dq,
    )


def gated_attention_decode(
    *,
    X: torch.Tensor,
    W_qkv: torch.Tensor,
    W_gate: torch.Tensor,
    W_out: torch.Tensor,
    K_cache: torch.Tensor,
    V_cache: torch.Tensor,
    active_blocks_table: torch.Tensor,
    softmax_scale: float,
    qk_norm_eps: float,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    pos_ids: Optional[torch.Tensor] = None,
    swa_start_pos_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    rmsnorm_X_enabled: bool = False,
    rmsnorm_X_eps: Optional[float] = None,
    rmsnorm_X_gamma: Optional[torch.Tensor] = None,
    quantization_type_qkv: QuantizationType = QuantizationType.NONE,
    weight_dequant_scale_qkv: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    gate_dq: float = 1.0,
    force_torch: bool = False,
):
    """See module docstring. ``pos_ids`` (fused mask) XOR a full ``attention_mask`` [S_ctx, B, N, S]."""
    B, S, H = X.shape
    d = V_cache.shape[-1]
    nq = W_gate.shape[1] // d
    block_len = V_cache.shape[1] if V_cache.dim() == 3 else V_cache.shape[2]
    s_prior = active_blocks_table.shape[-1] * block_len
    use_kernel = (not force_torch and can_run_kernel(X) and B * S <= 128
                  and _can_use_attention_block_kernel(X=X, V_cache=V_cache, active_blocks_table=active_blocks_table))
    if use_kernel:
        if pos_ids is not None and attention_mask is None:
            attention_mask = _build_default_active_mask(B=B, S_tkg=S, q_heads=nq, device=X.device)
        wrapped = wrap_nki(_gated_attention_block_kernel)
        return wrapped[get_logical_nc_config()](
            X,
            rmsnorm_X_enabled=rmsnorm_X_enabled,
            rmsnorm_X_eps=rmsnorm_X_eps,
            rmsnorm_X_gamma=rmsnorm_X_gamma,
            W_qkv=W_qkv,
            quantization_type_qkv=quantization_type_qkv,
            weight_dequant_scale_qkv=weight_dequant_scale_qkv,
            rmsnorm_QK_pre_rope_enabled=True,
            rmsnorm_QK_pre_rope_eps=qk_norm_eps,
            cos=cos,
            sin=sin,
            active_blocks_table=active_blocks_table,
            K_cache=K_cache,
            V_cache=V_cache,
            attention_mask=attention_mask,
            softmax_scale=softmax_scale,
            pos_ids=pos_ids,
            swa_start_pos_ids=swa_start_pos_ids,
            k_scale=k_scale,
            v_scale=v_scale,
            W_out=W_out,
            W_gate=W_gate,
            gate_dq=float(gate_dq),
        )

    # torch reference (CPU mode): same math, unfused
    if pos_ids is not None:
        attention_mask = gen_attention_decode_mask(
            pos_ids=pos_ids.reshape(1, B * S).to(torch.float32), bs=B, q_head=nq, s_active=S, s_prior=s_prior,
            start_pos=swa_start_pos_ids.reshape(1, B * S).to(torch.float32) if swa_start_pos_ids is not None else None,
            block_len=block_len)
    Xn = X
    if rmsnorm_X_enabled:
        xf = X.to(torch.float32)
        Xn = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + rmsnorm_X_eps)
              * rmsnorm_X_gamma.to(torch.float32)).to(X.dtype)
    o, K_new, V_new = _torch_attention_decode_impl(
        X=Xn, W_qkv=W_qkv, rmsnorm_QK_pre_rope_enabled=True, rmsnorm_QK_pre_rope_eps=qk_norm_eps,
        cos=cos, sin=sin, rope_contiguous_layout=True, active_blocks_table=active_blocks_table,
        K_cache=K_cache, V_cache=V_cache, attention_mask=attention_mask, softmax_scale=softmax_scale,
        update_cache=False, W_out=None)
    attn = o.permute(0, 3, 1, 2).reshape(B * S, nq * d)  # [B, N, d, S] -> [B*S, N*d]
    gate = torch.sigmoid(Xn.reshape(B * S, H) @ W_gate) * gate_dq
    out = (attn * gate).to(X.dtype) @ W_out
    return out, K_new, V_new
