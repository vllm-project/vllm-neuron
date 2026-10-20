# SPDX-License-Identifier: Apache-2.0
import nki

import nki.language as nl
from typing import Optional

from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    RouterActFnType,
)

from nkilib.core.moe_block.moe_block_tkg import moe_block_tkg


# TODO: this wrapper needs to be upstreamed into NKL.
@nki.jit
def moe_block_tkg_wrapper(
    inp: nl.NkiTensor,
    gamma: nl.NkiTensor,
    router_weights: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    shared_expert_gate_w: Optional[nl.NkiTensor] = None,
    shared_expert_up_w: Optional[nl.NkiTensor] = None,
    shared_expert_down_w: Optional[nl.NkiTensor] = None,
    expert_gate_up_weights_scale: Optional[nl.NkiTensor] = None,
    expert_down_weights_scale: Optional[nl.NkiTensor] = None,
    router_bias: Optional[nl.NkiTensor] = None,
    expert_gate_up_bias: Optional[nl.NkiTensor] = None,
    expert_down_bias: Optional[nl.NkiTensor] = None,
    shared_expert_gate_bias: Optional[nl.NkiTensor] = None,
    shared_expert_up_bias: Optional[nl.NkiTensor] = None,
    shared_expert_down_bias: Optional[nl.NkiTensor] = None,
    eps: float = 1e-6,
    top_k: int = 1,
    router_act_fn: RouterActFnType = RouterActFnType.SIGMOID,
    router_pre_norm: bool = True,
    norm_topk_prob: bool = False,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.NO_SCALE,
    hidden_act_fn: ActFnType = ActFnType.SiLU,
    hidden_act_scale_factor: Optional[float] = None,
    hidden_act_bias: Optional[float] = None,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    router_mm_dtype=nl.bfloat16,
    hidden_actual: Optional[int] = None,
    skip_router_logits: bool = False,
    is_all_expert: bool = False,
    rank_id: Optional[nl.NkiTensor] = None,
    residual: Optional[nl.NkiTensor] = None,
    expert_gate_up_input_scale: Optional[nl.NkiTensor] = None,
    expert_down_input_scale: Optional[nl.NkiTensor] = None,
):
    """Thin wrapper around ``moe_block_tkg`` reinterpreting packed
    ``uint16``/``uint32`` MX weights to ``float4_e2m1fn_x4``/``float8_e4m3fn_x4``.

    For STATIC_MX, ``expert_gate_up_input_scale`` and ``expert_down_input_scale``
    forward per-expert FP8 activation dequant scales (``[E_L, 1]`` fp32).
    """
    if expert_gate_up_weights.dtype == nl.uint16:
        expert_gate_up_weights = expert_gate_up_weights.view(nl.float4_e2m1fn_x4)
    if expert_down_weights.dtype == nl.uint16:
        expert_down_weights = expert_down_weights.view(nl.float4_e2m1fn_x4)
    if expert_gate_up_weights.dtype == nl.uint32:
        expert_gate_up_weights = expert_gate_up_weights.view(nl.float8_e4m3fn_x4)
    if expert_down_weights.dtype == nl.uint32:
        expert_down_weights = expert_down_weights.view(nl.float8_e4m3fn_x4)

    return moe_block_tkg(
        inp=inp,
        gamma=gamma,
        router_weights=router_weights,
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        shared_expert_gate_w=shared_expert_gate_w,
        shared_expert_up_w=shared_expert_up_w,
        shared_expert_down_w=shared_expert_down_w,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        router_bias=router_bias,
        expert_gate_up_bias=expert_gate_up_bias,
        expert_down_bias=expert_down_bias,
        shared_expert_gate_bias=shared_expert_gate_bias,
        shared_expert_up_bias=shared_expert_up_bias,
        shared_expert_down_bias=shared_expert_down_bias,
        eps=eps,
        top_k=top_k,
        router_act_fn=router_act_fn,
        router_pre_norm=router_pre_norm,
        norm_topk_prob=norm_topk_prob,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        hidden_act_fn=hidden_act_fn,
        hidden_act_scale_factor=hidden_act_scale_factor,
        hidden_act_bias=hidden_act_bias,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        router_mm_dtype=router_mm_dtype,
        hidden_actual=hidden_actual,
        skip_router_logits=skip_router_logits,
        is_all_expert=is_all_expert,
        rank_id=rank_id,
        residual=residual,
        expert_gate_up_input_scale=expert_gate_up_input_scale,
        expert_down_input_scale=expert_down_input_scale,
    )
