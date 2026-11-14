# SPDX-License-Identifier: Apache-2.0
"""Hy4's routed + shared MoE, with fp8 experts, in torch.

256 routed experts (top-8) of ``moe_intermediate_size`` 2048 plus one shared
expert of the same width. The routed experts are 744B of the model's 770B
parameters, so they are the one thing that has to stay 8-bit to fit a
trn2.48xlarge: ``convert_mxfp8_to_neuron_fp8.py`` stores them as Trn2-range
fp8 with one fp32 scale per output row.

**Layout: pure tensor parallelism over the experts' intermediate dimension.**
Every rank holds all 256 experts and ``I / world_size`` (32 at world 64) of each
one's columns, and the combine is the same all-reduce the attention output
already pays. That is chosen for bring-up, not speed:

* it needs exactly one prefill and one decode graph -- expert parallelism makes
  ``ep_rank`` a trace constant and multiplies the graph count by ``ep_degree``;
* it does not depend on Trn2 accepting fp8 weights in ``moe_cte`` /
  ``moe_tkg`` (``NF.moe_tkg`` rejects ``float8`` outright today), so nothing
  here is a kernel guess.

The cost is that every rank evaluates all 256 experts on every token and masks
the 248 unselected ones with a zero affinity -- exact, but 32x the routed
FLOPs. At decode that is free (the step reads every local expert's weights
either way); at a 2048-token prefill it is ~0.6 TFLOP per rank per layer. The
kernel path (``moe_cte`` shard_on_block FP8 for prefill, ``moe_tkg`` for decode)
needs ``moe_intermediate_size / tp_degree`` to be a multiple of 128, i.e.
``ep_degree >= 4`` at world 64, and is the next step.

Routing is DeepSeek-V3's, *not* what the fused ``moe_block_tkg`` router does:

    s = sigmoid(x @ Wr.T)                      # fp32, over all 256
    pick top-8 of (s + e_score_correction_bias)  # bias chooses, never weighs
    w = s[picked] / sum(s[picked]) * routed_scaling_factor

The kernel's ``router_bias`` is added to the logits and so changes the weights
as well as the choice; that is why routing runs here and not in a kernel.

The routed SwiGLU is clamped (``gate <= 10``, ``-10 <= up <= 10``); the shared
expert and the dense layer-0 MLP are not.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    set_weight_loader,
    sharding_weight_loader,
)

from .config import HYV4Config

FP8 = torch.float8_e4m3fn


def _as_bytes(t: torch.Tensor) -> torch.Tensor:
    """fp8 -> uint8 view, so CPU permutes and copies never touch fp8 kernels."""
    return t.view(torch.uint8) if t.dtype == FP8 else t


def _expert_gate_up_loader(intermediate: int, shard: int) -> SafetensorsWeightLoader:
    """``[E, 2I, H]`` fp8 -> this rank's ``[H, E, 2, I/tp]`` fp8.

    Rows ``[0, I)`` of each expert are gate and ``[I, 2I)`` are up (contiguous
    halves, ``chunk(2)`` in ``HYV4Experts._apply_gate``). Stored hidden-major so
    the forward is one ``[T, H] @ [H, E * 2 * I_r]`` matmul.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        (source,) = slices
        num_experts, two_i, hidden = source.get_shape()
        if two_i != 2 * intermediate:
            raise ValueError(
                f"gate_up_proj has {two_i} rows, expected {2 * intermediate}"
            )
        start = rank * shard
        gate = _as_bytes(source[:, start : start + shard, :])
        up = _as_bytes(
            source[:, intermediate + start : intermediate + start + shard, :]
        )
        out = torch.stack((gate, up), dim=1)  # [E, 2, I_r, H]
        return out.permute(3, 0, 1, 2).contiguous().view(FP8)

    return SafetensorsWeightLoader(transform=transform)


def _expert_gate_up_scale_loader(
    intermediate: int, shard: int
) -> SafetensorsWeightLoader:
    """``[E, 2I]`` row scales -> ``[1, E * 2 * I/tp]``, in the weight's column order."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        (source,) = slices
        start = rank * shard
        gate = source[:, start : start + shard]
        up = source[:, intermediate + start : intermediate + start + shard]
        return torch.stack((gate, up), dim=1).reshape(1, -1).float()

    return SafetensorsWeightLoader(transform=transform)


def _expert_down_loader(shard: int) -> SafetensorsWeightLoader:
    """``down_proj_t [E, I, H]`` fp8 -> this rank's ``[E, I/tp, H]`` fp8.

    The converter stores the down weight transposed precisely so that this is a
    row slice: sharding HF's ``[E, H, I]`` on its last dim is either ``E * H``
    32-byte reads or every rank reading the whole 3 GiB tensor per layer.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        (source,) = slices
        start = rank * shard
        return source[:, start : start + shard, :]

    return SafetensorsWeightLoader(transform=transform)


class HYV4MLP(nn.Module):
    """Plain SwiGLU, intermediate dim TP-sharded. Dense layer 0 and the shared expert.

    Returns this rank's partial sum; the caller reduces.
    """

    def __init__(self, hidden: int, intermediate: int, world_size: int, dtype):
        super().__init__()
        if intermediate % world_size:
            raise ValueError(
                f"intermediate_size={intermediate} not divisible by {world_size}"
            )
        inter = intermediate // world_size
        self.gate_proj_weight = nn.Parameter(torch.empty(hidden, inter, dtype=dtype))
        self.up_proj_weight = nn.Parameter(torch.empty(hidden, inter, dtype=dtype))
        self.down_proj_weight = nn.Parameter(torch.empty(inter, hidden, dtype=dtype))
        for param in (self.gate_proj_weight, self.up_proj_weight):
            set_weight_loader(
                param,
                sharding_weight_loader(
                    shard_dim=1,
                    shard_size=inter,
                    num_shards=world_size,
                    is_storage_transposed=True,
                ),
            )
        set_weight_loader(
            self.down_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=inter,
                num_shards=world_size,
                is_storage_transposed=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.down_proj_weight.dtype)
        return (
            F.silu(x @ self.gate_proj_weight) * (x @ self.up_proj_weight)
        ) @ self.down_proj_weight


def route(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
) -> torch.Tensor:
    """Dense ``[T, E]`` affinities, zero outside each token's top-k.

    Returned dense rather than as (indices, weights) because the expert compute
    below is dense over experts; built from a one-hot instead of a scatter,
    which keeps the trace free of ``index_put``.
    """
    logits = hidden_states.float() @ router_weight.float().t()
    scores = torch.sigmoid(logits)
    _, idx = torch.topk(scores + correction_bias.float(), top_k, dim=-1)
    weights = torch.gather(scores, 1, idx)
    if norm_topk_prob:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * routed_scaling_factor
    # A comparison, not ``F.one_hot``: the capture backend hands topk's indices
    # back as int32, which ``one_hot`` rejects.
    experts = torch.arange(scores.shape[-1], device=idx.device, dtype=idx.dtype)
    one_hot = (idx.unsqueeze(-1) == experts).to(weights.dtype)  # [T, K, E]
    return (one_hot * weights.unsqueeze(-1)).sum(dim=1)


class HYV4MoE(nn.Module):
    """Router + 256 fp8 experts (dense over experts, masked) + shared expert.

    Takes the *normed* hidden states and returns this rank's partial sum.
    """

    def __init__(self, config: HYV4Config):
        super().__init__()
        self.dtype = config.torch_dtype
        self.world_size = get_tp_group().world_size
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.swiglu_limit = config.swiglu_limit
        intermediate = config.moe_intermediate_size
        if intermediate % self.world_size:
            raise ValueError(
                f"moe_intermediate_size={intermediate} not divisible by "
                f"tp={self.world_size}"
            )
        self.inter_per_rank = intermediate // self.world_size
        h, e, i_r = self.hidden_size, self.num_experts, self.inter_per_rank

        self.experts_gate_up_weight = nn.Parameter(
            torch.empty(h, e, 2, i_r, dtype=FP8), requires_grad=False
        )
        self.experts_gate_up_scale = nn.Parameter(
            torch.empty(1, e * 2 * i_r, dtype=torch.float32), requires_grad=False
        )
        self.experts_down_weight = nn.Parameter(
            torch.empty(e, i_r, h, dtype=FP8), requires_grad=False
        )
        self.experts_down_scale = nn.Parameter(
            torch.empty(e, 1, h, dtype=torch.float32), requires_grad=False
        )
        set_weight_loader(
            self.experts_gate_up_weight, _expert_gate_up_loader(intermediate, i_r)
        )
        set_weight_loader(
            self.experts_gate_up_scale, _expert_gate_up_scale_loader(intermediate, i_r)
        )
        set_weight_loader(self.experts_down_weight, _expert_down_loader(i_r))
        # Per output row, so the scale of an input-dim shard is the whole scale.
        set_weight_loader(
            self.experts_down_scale,
            SafetensorsWeightLoader(
                transform=lambda s, rank: s[0][:].unsqueeze(1).float()
            ),
        )

        # Replicated, as stored: ``[E, H]`` bf16 and ``[E]`` fp32.
        self.router_weight = nn.Parameter(torch.empty(e, h, dtype=self.dtype))
        self.e_score_correction_bias = nn.Parameter(torch.empty(e, dtype=torch.float32))

        self.shared_experts = (
            HYV4MLP(
                h, intermediate * config.n_shared_experts, self.world_size, self.dtype
            )
            if config.n_shared_experts
            else None
        )

    def forward(self, normed: torch.Tensor) -> torch.Tensor:
        normed = normed.to(self.dtype)
        tokens = normed.shape[0]
        e, i_r = self.num_experts, self.inter_per_rank

        affinities = route(
            normed,
            self.router_weight,
            self.e_score_correction_bias,
            self.top_k,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )  # [T, E] fp32

        # gate/up: the per-row scale is per *output* column here, so it applies
        # after the matmul and the weight only needs a dtype convert.
        gate_up = normed @ self.experts_gate_up_weight.to(self.dtype).view(
            self.hidden_size, -1
        )
        gate_up = (gate_up.float() * self.experts_gate_up_scale).view(tokens, e, 2, i_r)
        gate, up = gate_up[:, :, 0, :], gate_up[:, :, 1, :]
        if self.swiglu_limit > 0:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        act = F.silu(gate) * up * affinities.unsqueeze(-1)  # [T, E, I_r]

        # down: the scale is per output channel *per expert*, so it cannot be
        # pulled out of the sum over experts -- dequantise the weight instead.
        # The product is formed in fp32: a bf16 scale alone would cost 0.4% per row.
        down = (self.experts_down_weight.float() * self.experts_down_scale).to(
            self.dtype
        )
        out = act.to(self.dtype).reshape(tokens, e * i_r) @ down.view(
            e * i_r, self.hidden_size
        )

        if self.shared_experts is not None:
            out = out + self.shared_experts(normed)
        return out
