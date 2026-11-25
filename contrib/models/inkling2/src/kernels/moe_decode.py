# SPDX-License-Identifier: Apache-2.0
"""The decode MoE on nkilib's selective-expert ``moe_tkg`` kernel, with Inkling 2's activation.

Inkling 2's experts compute ``4 tanh(g / 4) * sigmoid(g) * 25 tanh(u / 25)`` (a softcapped SwiGLU), which is not one
of ``moe_tkg``'s activations. ``install()`` wraps the kernel's gate-up post-processing
(``nkilib.core.moe.moe_tkg.mlp_tkg_gate_up_projection._post_process_gate_up``): a call made with
``activation_fn=ActFnType.SquaredReLU``, which no MoE in this process uses, takes the softcapped activation; every
other call runs nkilib's code unchanged. The kernel reads only each token's top-k experts, in the layouts this model
keeps them in: ``gate_up [E, H, 2, I_r]``, ``down [E, I_r, H]`` (all experts on every rank, the intermediate dim
split over TP).
"""

from __future__ import annotations

import torch

GATE_CAP, UP_CAP = 4.0, 25.0
MARKER = "SquaredReLU"
_installed = False


def install() -> None:
    global _installed
    if _installed:
        return
    import importlib

    import nki.isa as nisa
    import nki.language as nl
    from nkilib.core.utils.common_types import ActFnType

    # importlib: ``nkilib.core.moe.moe_tkg`` as an attribute is the kernel function, which shadows the package
    gup = importlib.import_module("nkilib.core.moe.moe_tkg.mlp_tkg_gate_up_projection")

    original = gup._post_process_gate_up
    marker = getattr(ActFnType, MARKER)

    def post_process(output, params, dims, sbm, gate_sb_view, up_sb_view, gate_up_recv, use_fused_gate_up_sendrecv,
                     gate_up_sb_fp32, tiles):
        if params.activation_fn != marker or params.skip_gate_proj or dims.num_shards > 1 \
                or params.use_tkg_gate_up_proj_column_tiling:
            return original(output, params, dims, sbm, gate_sb_view, up_sb_view, gate_up_recv,
                            use_fused_gate_up_sendrecv, gate_up_sb_fp32, tiles)
        # one rank per core (no sendrecv); gate and up are fp32 SBUF tiles
        sig = sbm.alloc_stack(tuple(gate_sb_view.shape), dtype=nl.float32, name="inkling2_gate_sigmoid",
                              buffer=nl.sbuf, align=4)
        nisa.activation(dst=sig, op=nl.sigmoid, data=gate_sb_view, scale=1.0)
        nisa.activation(dst=gate_sb_view, op=nl.tanh, data=gate_sb_view, scale=1.0 / GATE_CAP)
        nisa.tensor_tensor(dst=gate_sb_view, data1=gate_sb_view, data2=sig, op=nl.multiply)
        nisa.activation(dst=up_sb_view, op=nl.tanh, data=up_sb_view, scale=1.0 / UP_CAP)
        nisa.tensor_scalar(dst=up_sb_view, data=up_sb_view, op0=nl.multiply, operand0=GATE_CAP * UP_CAP)
        nisa.tensor_tensor(dst=output, data1=gate_sb_view, data2=up_sb_view, op=nl.multiply)
        return tiles

    gup._post_process_gate_up = post_process
    _installed = True


try:  # nkilib ships with the Neuron images; the module stays importable without it (host tests)
    import nki
    from nkilib.core.moe.moe_tkg.moe_tkg import moe_tkg as _moe_tkg
    from nkilib.core.utils.common_types import ActFnType as _ActFnType
    from nkilib.core.utils.common_types import ExpertAffinityScaleMode as _ScaleMode

    @nki.jit
    def inkling2_moe_tkg(hidden_input, expert_gate_up_weights, expert_down_weights, expert_affinities,
                         expert_index):
        """``moe_tkg`` (a plain function in nkilib) as a kernel: selective experts, the marked activation, routing
        weights applied after the down projection."""
        return _moe_tkg(hidden_input=hidden_input, expert_gate_up_weights=expert_gate_up_weights,
                        expert_down_weights=expert_down_weights, expert_affinities=expert_affinities,
                        expert_index=expert_index, is_all_expert=False,
                        expert_affinities_scaling_mode=_ScaleMode.POST_SCALE, activation_fn=_ActFnType.SquaredReLU)
except ImportError:  # pragma: no cover
    inkling2_moe_tkg = None


def moe_decode(wrap, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, gate_up: torch.Tensor,
               down: torch.Tensor) -> torch.Tensor:
    """``x [T, H]`` bf16, ``idx [T, k]``, ``w [T, k]`` fp32 routing weights, ``gate_up [E, H, 2 * I_r]``,
    ``down [E, I_r, H]``. ``wrap`` turns an NKI kernel into a callable (the caller's ``wrap_nki(...)[lnc]``).
    Returns this rank's partial ``[T, H]``. Call ``install()`` first, outside any traced function."""
    E, H, I2 = gate_up.shape
    # fp32: the kernel scales each expert output by its affinity with a float32 scalar operand
    aff = torch.zeros(x.shape[0], E, dtype=torch.float32, device=x.device).scatter(1, idx, w.float())
    return wrap(inkling2_moe_tkg)(x, gate_up.view(E, H, 2, I2 // 2), down, aff, idx.to(torch.int32))
