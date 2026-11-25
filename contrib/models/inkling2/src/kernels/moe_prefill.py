# SPDX-License-Identifier: Apache-2.0
"""The prefill MoE on nkilib's blockwise ``moe_cte`` kernel (``shard_on_block``), with Inkling 2's activation.

The torch prefill MoE (``ops.moe_dense``) runs all 128 experts on every token of a chunk; the blockwise kernel runs
each expert on the blocks of tokens routed to it (top 8 of 128), so it does about 1/16 of the expert matmuls plus
the padding of the last block of each expert. ``install()`` wraps nkilib's ``compute_intermediate_states`` as
``kernels/moe_decode.py`` wraps the decode post-processing: a call made with ``ActFnType.SquaredReLU`` (which no MoE
in this process uses) takes ``4 tanh(g / 4) * sigmoid(g) * 25 tanh(u / 25)``; every other call runs nkilib's code.

Opt-in with ``INKLING2_NKI_MOE_PREFILL=1``; experimental (not yet measured in the server).
"""

from __future__ import annotations

import torch

GATE_CAP, UP_CAP = 4.0, 25.0
BLOCK = 32            # tokens a block: T * top_k / BLOCK + E - 1 blocks (188 for a 256-token chunk, top 8 of 128)
_installed = False


class _Rank0Group:
    """``build_blockwise_mapping`` reads the group's rank tensor even without TP sharding (``tp_degree=1``: every
    rank computes the same mapping over all its experts)."""

    def __init__(self, device):
        self.rank_in_group_tensor = torch.zeros(1, dtype=torch.int32, device=device)


def install() -> None:
    global _installed
    if _installed:
        return
    import importlib

    import nki.isa as nisa
    import nki.language as nl
    from nkilib.core.utils.common_types import ActFnType

    blk = importlib.import_module("nkilib.core.moe.moe_cte.bwmm_shard_on_block")
    original = blk.compute_intermediate_states
    marker = ActFnType.SquaredReLU

    def compute(gate_and_up_proj_states, B, I_TP, dtype, activation_function=ActFnType.SiLU,
                expert_affinity_T_broadcasted=None, gup_scale=None, expert_affinity_multiply_on_I=False, sbm=None,
                intermediate_states_lst=None, skip_gate_proj=False):
        if activation_function != marker or expert_affinity_T_broadcasted is not None or gup_scale is not None \
                or skip_gate_proj or sbm is not None or intermediate_states_lst is not None:
            return original(gate_and_up_proj_states, B, I_TP, dtype, activation_function=activation_function,
                            expert_affinity_T_broadcasted=expert_affinity_T_broadcasted, gup_scale=gup_scale,
                            expert_affinity_multiply_on_I=expert_affinity_multiply_on_I, sbm=sbm,
                            intermediate_states_lst=intermediate_states_lst, skip_gate_proj=skip_gate_proj)
        TILE, PSUM = nl.tile_size.pmax, nl.tile_size.psum_fmax
        n_i = -(-I_TP // TILE)
        n_b = -(-B // PSUM)
        free = gate_and_up_proj_states[0][0][0].shape[-1]
        out = [nl.ndarray((TILE, B), dtype=dtype, buffer=nl.sbuf) for _ in range(n_i)]
        for i in range(n_i):
            n = min(TILE, I_TP - TILE * i)
            sig = nl.ndarray((TILE, free), dtype=nl.float32, buffer=nl.sbuf)
            g = nl.ndarray((TILE, free), dtype=nl.float32, buffer=nl.sbuf)
            u = nl.ndarray((TILE, free), dtype=nl.float32, buffer=nl.sbuf)
            for b in range(n_b):
                gate = gate_and_up_proj_states[0][b][i][0:n, 0:free]
                up = gate_and_up_proj_states[1][b][i][0:n, 0:free]
                nisa.activation(dst=sig[0:n, 0:free], op=nl.sigmoid, data=gate, scale=1.0)
                nisa.activation(dst=g[0:n, 0:free], op=nl.tanh, data=gate, scale=1.0 / GATE_CAP)
                nisa.tensor_tensor(dst=g[0:n, 0:free], data1=g[0:n, 0:free], data2=sig[0:n, 0:free], op=nl.multiply)
                nisa.activation(dst=u[0:n, 0:free], op=nl.tanh, data=up, scale=1.0 / UP_CAP)
                nisa.tensor_scalar(dst=u[0:n, 0:free], data=u[0:n, 0:free], op0=nl.multiply,
                                   operand0=GATE_CAP * UP_CAP)
                nisa.tensor_tensor(dst=out[i][0:n, b * PSUM:b * PSUM + free], data1=g[0:n, 0:free],
                                   data2=u[0:n, 0:free], op=nl.multiply)
        return out

    blk.compute_intermediate_states = compute
    _installed = True


def mapping(idx: torch.Tensor, w: torch.Tensor, E: int):
    """The blockwise routing tables for ``idx [T, k]``, ``w [T, k]``: masked affinities ``[T * E, 1]``, token of each
    block position ``[N * BLOCK]`` (-1 padding), expert of each block ``[N]``, block-valid flags ``[N]``."""
    from vllm_neuron import functional as NF

    T, k = idx.shape
    aff = torch.zeros(T, E, dtype=torch.float32, device=idx.device).scatter(1, idx, w.float())
    masked, pos_to_id, block_to_expert, conditions = NF.build_blockwise_mapping(
        expert_affinities=aff, num_local_experts=E, num_experts_per_token=k, block_size=BLOCK,
        moe_group=_Rank0Group(idx.device), tp_degree=1)
    return masked, pos_to_id.to(torch.int32), block_to_expert.to(torch.int32), conditions


def kernel_args(x, idx, w, gate_up, down):
    from nkilib.core.moe.moe_cte.moe_cte import ActFnType, ExpertAffinityScaleMode, MoECTEImplementation
    import nki.language as nl

    E, H, I2 = gate_up.shape
    masked, pos_to_id, block_to_expert, conditions = mapping(idx, w, E)
    return dict(hidden_states=x, expert_affinities_masked=masked, gate_up_proj_weight=gate_up.view(E, H, 2, I2 // 2),
                down_proj_weight=down, token_position_to_id=pos_to_id, block_to_expert=block_to_expert,
                block_size=BLOCK, implementation=MoECTEImplementation.shard_on_block, conditions=conditions,
                activation_function=ActFnType.SquaredReLU, skip_token=True, is_tensor_update_accumulating=True,
                expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE, compute_dtype=nl.bfloat16)


def moe_prefill(wrap, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, gate_up: torch.Tensor,
                down: torch.Tensor) -> torch.Tensor:
    """``x [T, H]`` bf16, ``idx [T, k]``, ``w [T, k]`` fp32 routing weights, ``gate_up [E, H, 2 * I_r]``,
    ``down [E, I_r, H]``; ``wrap`` turns the kernel into a callable at this process's LNC (vllm-neuron's ``moe_cte``
    wrapper launches it at LNC 2; a rank here is one physical core). Returns this rank's partial ``[T, H]`` fp32.
    Call ``install()`` first, outside any traced function."""
    from vllm_neuron.functional.moe.moe_cte import _torch_compatible_moe_cte_kernel

    out = wrap(_torch_compatible_moe_cte_kernel)(**kernel_args(x, idx, w, gate_up, down))
    return out[:, 0, :x.shape[-1]].float()        # the kernel's output is [T, 2, H + E]
