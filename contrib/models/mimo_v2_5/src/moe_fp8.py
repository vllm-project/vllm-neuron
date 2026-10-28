# SPDX-License-Identifier: Apache-2.0
"""Dense masked routed-expert compute over fp8-resident weights (torch only).

Used by ``MiMoV2SparseMoeBlock`` when the experts are fp8
(``MIMO_V2_5_EXPERT_DTYPE``; auto-selected for MiMo-V2.5-Pro) for BOTH prefill
and decode. The NKI ``moe_cte`` / ``moe_tkg`` paths are not used there:
``moe_tkg`` rejects ``float8_e4m3fn`` weights, and Trn2 has no blockwise-FP8
MoE kernel. Kept in its own torch-only module so the CPU unit test can import
it without NKI or vLLM.

Every local expert runs over every token, and an expert's output is multiplied
by that token's routing affinity for it (zero for unselected experts), i.e.
``moe_tkg(is_all_expert=True, POST_SCALE)``. Like the Hy4 contrib port's
experts, the weights are converted fp8 -> bf16 and the per-output-channel scale
is applied AFTER the matmul, where it factors out of the contraction.
"""

import torch
import torch.nn.functional as F

# Token chunk for the dense compute. The live activations are
# [E_local, chunk, 2I] plus [E_local, chunk, H]; at Pro's E_local=6, I=2048,
# H=6144 and chunk=1024 that is ~50 MB bf16 + ~150 MB fp32. Prefill at longer
# max_model_len is cut into static chunks (T is a static bucket size, so the
# Python loop unrolls into a fixed graph).
DEFAULT_TOKEN_CHUNK = 1024


def experts_dense_fp8(
    hidden_states: torch.Tensor,
    local_affinities: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
    compute_dtype: torch.dtype = torch.bfloat16,
    token_chunk: int = DEFAULT_TOKEN_CHUNK,
) -> torch.Tensor:
    """SwiGLU experts over fp8 weights with per-output-channel scales.

    Args:
        hidden_states: ``[T, H]``.
        local_affinities: ``[T, E_local]`` routing weights for THIS rank's
            experts (already sliced with the traced EP rank).
        gate_up_weight: ``[E_local, H, 2I]`` fp8; gate is ``[..., :I]``, up is
            ``[..., I:]`` (the bf16 layout's ``[E, H, 2, I]`` order).
        gate_up_scale: ``[E_local, 1, 2I]`` fp32.
        down_weight: ``[E_local, I, H]`` fp8.
        down_scale: ``[E_local, 1, H]`` fp32.
        compute_dtype: matmul dtype (bf16).
        token_chunk: max tokens per dense pass, bounding activation memory.

    Returns:
        ``[T, H]`` fp32, this rank's partial sum (the caller reduces).
    """
    num_tokens = hidden_states.shape[0]
    inter = gate_up_weight.shape[-1] // 2
    w_gate_up = gate_up_weight.to(compute_dtype)  # [E, H, 2I]
    w_down = down_weight.to(compute_dtype)  # [E, I, H]
    aff = local_affinities.to(torch.float32).transpose(0, 1).unsqueeze(-1)  # [E,T,1]
    x_all = hidden_states.to(compute_dtype)

    outs = []
    for start in range(0, num_tokens, max(1, token_chunk)):
        stop = min(start + token_chunk, num_tokens)
        x = x_all[start:stop].unsqueeze(0)  # [1, t, H] broadcasts over experts
        gate_up = torch.matmul(x, w_gate_up).float() * gate_up_scale  # [E, t, 2I]
        act = F.silu(gate_up[..., :inter]) * gate_up[..., inter:]  # fp32 [E, t, I]
        # The affinity goes on BEFORE down (down is linear, so this equals
        # POST_SCALE), and unrouted (expert, token) pairs become exact zeros.
        # Dense masked compute runs every local expert on every token, and an
        # expert can overflow on a token it was never routed: Pro's layer 10
        # did, and the old ``down * 0`` turned that inf into NaN, which reached
        # the next router and faulted its gather on device (nrta 1006).
        a = aff[:, start:stop]
        act = torch.where(a > 0, act * a, torch.zeros_like(act))
        down = torch.matmul(act.to(compute_dtype), w_down).float() * down_scale
        outs.append(down.sum(dim=0))  # [t, H]
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
