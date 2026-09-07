# SPDX-License-Identifier: Apache-2.0
"""The sparse MLP of ``qwen3_5_moe`` (35B-A3B, 397B-A17B), with TP or EP.

This is the *only* thing that differs from the dense Qwen3.5 checkpoints. The
hybrid DeltaNet/GQA stack, the zero-centred norms, the gated attention output and
the partial interleaved mRoPE in ``model.py`` are all reused unchanged, so this
module is dropped in where ``Qwen3_5MLP`` sits and nothing else moves.

Two vendor kernels do the work, one per phase, and they take the *unnormalised*
residual because each fuses the pre-MLP RMSNorm itself:

* prefill -- ``moe_cte``, blockwise. Tokens are sorted into fixed-size blocks of
  one expert each, so the cost is ``num_blocks x block_size`` rather than
  ``T x top_k``; the padding inside partly-filled blocks is the price. See
  ``BLOCK_SIZE`` for why 128 and not gpt-oss's 256.
* decode -- ``moe_block_tkg``, which fuses norm, router, top-k and both
  projections, and (below a batch threshold) DMAs only the experts that were
  actually selected.

Three things here are easy to get silently wrong:

**The norm's gamma is ``1 + weight``.** ``Qwen3_5RMSNorm`` is zero-centred
(``normalise(x) * (1 + w)``) while the kernels' fused norm is the ordinary
``normalise(x) * gamma``. Passing ``w`` straight through multiplies every
activation by roughly zero and the model emits fluent garbage with no error.

**The shared expert cannot go inside the kernel.** ``moe_block_tkg`` does accept
``shared_expert_{gate,up,down}_w``, but it adds that branch's output to the routed
one internally, and Qwen3.5 scales it first by ``sigmoid(shared_expert_gate(x))``
-- an input-dependent scalar that cannot be folded into the weights. So the shared
expert runs in torch here, which costs one extra RMSNorm per layer (negligible at
``hidden_size`` 2048) and keeps the gate where it belongs.

**Routing is softmax-over-all, then top-k, then L1 renormalise.** Not top-k then
softmax, which is what gpt-oss does and what both kernels default to. The two
disagree numerically because the renormalised softmax-over-256 is not the softmax
over the 8 selected logits.
"""

from __future__ import annotations

import logging

import nki.language as nl
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    RouterActFnType,
)
from vllm_neuron.functional.moe.router import RouterComputationOrder
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    expert_parallel_tensor_dim_loader,
    set_weight_loader,
    sharding_weight_loader,
    with_rank_override,
)

from .config import Qwen3_5TextConfig
from .flags import SEQUENCE_PARALLEL

logger = logging.getLogger(__name__)

# Tokens per expert block in prefill. ``moe_cte`` wants a multiple of 128, and 128
# is the *smallest* legal value, which is the right end to be at: the block count
# is
#     N = ceil((T * top_k - (E - 1)) / block_size) + E - 1
# so with E = 256 the ``+ E - 1`` term dominates and a larger block only inflates
# the padding. At T = 2048, top_k 8: 381 blocks at 128 (33.6% of slots real)
# against 318 at 256 (20.1%). gpt-oss uses 256 because it has 32 experts, where
# the constant term is negligible.
#
# ``E`` here is ``num_local_experts``, so expert parallelism shrinks the constant
# term -- but it does not make prefill cheaper, and it is worth being clear about
# why. ``total_local_tokens`` inside ``build_blockwise_mapping`` is ``T * top_k``
# whatever ``ep_degree`` is: the mapping bounds the blocks worst-case, as though
# every assignment landed on this rank, rather than the ``T * top_k / ep_degree``
# that land there on average. Meanwhile each rank's intermediate width grows from
# ``I / world_size`` to ``I * ep_degree / world_size``. Per-rank work
# ``N * block_size * I_local`` therefore *rises* with ``ep_degree``
# (T = 1024, 35B-A3B on 4 ranks): 318 blocks x 128 wide = 5.21M at ep=1, 191 x 256
# = 6.26M at ep=2, 127 x 512 = 8.32M at ep=4.
#
# Decode is indifferent: the all-expert weight read is
# ``E_local * 3 * I_local * H``, and ``E_local * I_local = E * I / world_size``
# regardless of ``ep_degree``. So EP costs prefill padding, buys no memory, and
# exists here only to make otherwise-illegal topologies legal -- which is exactly
# 397B-A17B's problem. See ``Qwen3_5SparseMoeBlock`` for that.
BLOCK_SIZE = 128

# Above this fraction of *local* experts touched per step, ``moe_block_tkg``
# stops being selective and loads every local expert's weights. At top_k 8 / 256
# experts the crossover is batch 32; below it decode DMAs only the ~8*B experts
# it needs, which is the whole reason a 35B MoE decodes faster than a dense 27B
# here. Under EP the kernel is never selective (see ``forward_decode``), but the
# local expert count is ``num_experts / ep_degree``, so the DMA per rank shrinks
# anyway.
SELECTIVE_LOADING_THRESHOLD = 1.0


def _expert_gate_up_loader(
    intermediate_size: int,
    shard_size: int,
    num_shards: int,
    dtype: torch.dtype = torch.bfloat16,
) -> SafetensorsWeightLoader:
    """HF ``[E, 2I, H]`` -> this rank's ``[E_local, H, 2, I/tp_degree]``.

    The checkpoint stores one fused ``gate_up_proj`` per expert as an ordinary
    linear weight, and ``modeling_qwen3_5_moe`` splits its *output* with
    ``chunk(2, dim=-1)``. So the first ``I`` rows are gate and the next ``I`` are
    up -- contiguous halves, not interleaved the way gpt-oss's checkpoint has
    them. Getting that backwards swaps ``silu(gate) * up`` for
    ``silu(up) * gate``, which is wrong but perfectly stable-looking.

    The kernels want ``[E, H, 2, I_tp]`` with 0 = gate and 1 = up. Written into a
    preallocated buffer one half at a time: the transposed source is 128 MiB per
    half per layer, and materialising the permutation of the whole thing instead
    doubles a peak that four ranks pay simultaneously.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        (source,) = slices
        num_experts, two_i, hidden = source.get_shape()
        if two_i != 2 * intermediate_size:
            raise ValueError(
                f"gate_up_proj has {two_i} rows, expected 2 x {intermediate_size}"
            )
        start = (rank % num_shards) * shard_size

        out = torch.empty((num_experts, hidden, 2, shard_size), dtype=dtype)
        # Row-slices of each expert's matrix, so each read is
        # ``shard_size * hidden`` contiguous elements rather than a strided walk.
        gate = source[:, start : start + shard_size, :]
        out[:, :, 0, :] = gate.transpose(1, 2).to(dtype)
        del gate
        up = source[
            :, intermediate_size + start : intermediate_size + start + shard_size, :
        ]
        out[:, :, 1, :] = up.transpose(1, 2).to(dtype)
        return out

    return SafetensorsWeightLoader(transform=transform)


def _expert_down_loader(
    shard_size: int,
    num_shards: int,
    dtype: torch.dtype = torch.bfloat16,
    chunk_experts: int = 32,
) -> SafetensorsWeightLoader:
    """HF ``[E, H, I]`` -> this rank's ``[E_local, I/tp_degree, H]``.

    Sharded on ``I``, which is the *last* checkpoint dimension: slicing it
    directly would turn one read into ``E * H`` reads of ``shard_size`` elements
    each. Read whole experts in groups instead and slice in memory -- a group of
    32 is 64 MiB, so the peak stays a fraction of the 128 MiB output.

    ``num_experts`` is taken from the source shape, so wrapping this in
    ``expert_parallel_tensor_dim_loader`` (which narrows the source to the local
    expert range) composes without changes here.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        (source,) = slices
        num_experts, hidden, _ = source.get_shape()
        start = (rank % num_shards) * shard_size

        out = torch.empty((num_experts, shard_size, hidden), dtype=dtype)
        for first in range(0, num_experts, chunk_experts):
            last = min(first + chunk_experts, num_experts)
            group = source[first:last]
            out[first:last] = (
                group[:, :, start : start + shard_size].transpose(1, 2).to(dtype)
            )
        return out

    return SafetensorsWeightLoader(transform=transform)


class Qwen3_5SparseMoeBlock(nn.Module):
    """``num_experts`` routed experts, top-k, plus one gated shared expert.

    Two ways to spread the experts over ``world_size`` ranks, chosen by vLLM's
    ``enable_expert_parallel``:

    **TP** (``ep_degree == 1``, the default) -- every rank holds every expert and
    shards each expert's intermediate dimension ``world_size`` ways. This is what
    35B-A3B wants; see ``BLOCK_SIZE`` for the arithmetic, but the short version is
    that EP costs prefill padding and saves nothing.

    **EP** (``ep_degree > 1``) -- every rank holds a disjoint ``1/ep_degree`` of
    the experts at full width, and shards the intermediate dimension only across
    the ``tp_degree = world_size / ep_degree`` ranks in its own partition.
    397B-A17B *requires* this, and not for speed: the fused decode kernel needs
    ``moe_intermediate_size / tp_degree`` to be a multiple of 128, so with
    ``I = 1024`` pure TP cannot exceed 8 ranks -- and 8 ranks cannot hold that
    checkpoint's 725 GiB of weights, which need 31 of this device's 24 GiB cores.
    EP breaks the tie between the two: weights are sharded ``world_size`` ways
    whatever ``ep_degree`` is, so EP buys no memory -- it only makes a wide
    ``world_size`` *legal* by keeping ``tp_degree`` at or below 8.

    Which follows a rule worth stating, because the intuitive choice is wrong:
    **pick the smallest ``ep_degree`` that satisfies the 128-multiple, not the
    largest.** Per-rank prefill work grows with ``ep_degree`` while memory does
    not, so for 397B on 64 ranks the useful setting is ``ep_degree = 8``
    (``tp_degree = 8``, 2.34M block-slots x width at T = 1024), not the
    ``enable_expert_parallel`` default of ``ep_degree = world_size`` (``tp_degree
    = 1``, 9.31M) -- 4x the prefill work for identical footprint. ``ep_degree``
    of 1, 2 or 4 would leave ``I / tp_degree`` at 16, 32 or 64 and fail the
    kernel's guard below.

    Either way each rank produces a *partial sum* over the same physical group --
    a column shard under TP, a subset of each token's experts under EP -- so the
    combine is the same all-reduce (or reduce-scatter under sequence
    parallelism) in both modes.

    Takes the residual **before** the post-attention norm; see the module
    docstring.
    """

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        moe = config.moe
        if moe is None:
            raise ValueError("Qwen3_5SparseMoeBlock needs config.moe")
        if not moe.norm_topk_prob:
            raise NotImplementedError(
                "norm_topk_prob=False is not implemented: the kernels offer "
                "'activation then top-k then renormalise' but not that same "
                "order without the renormalise."
            )

        self.dtype = config.torch_dtype
        self.eps = config.rms_norm_eps
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        self.hidden_size = config.hidden_size
        self.num_experts = moe.num_experts
        self.top_k = moe.num_experts_per_tok
        self.intermediate_size = moe.moe_intermediate_size

        # TP or EP; see the class docstring for which checkpoint wants which.
        # The groups come from the plugin's own parallel state rather than being
        # derived here, because ``ep_tp_group`` is the sub-group the blockwise
        # mapping coordinates over and only the plugin knows how it was built.
        from vllm.config import get_current_vllm_config

        # ``get_current_vllm_config`` raises outside an engine context -- which
        # the offline compile probes and the CPU checks are. EP is off there, and
        # off is the default, so degrade to TP rather than making an engine
        # context a precondition for building this module.
        try:
            parallel_config = get_current_vllm_config().parallel_config
            ep_enabled = bool(parallel_config.enable_expert_parallel)
            dp_size = int(parallel_config.data_parallel_size)
        except (AssertionError, RuntimeError):
            ep_enabled, dp_size = False, 1

        if ep_enabled:
            from vllm_neuron.parallel.neuron_parallel_state import (
                get_neuron_ep_degree,
                get_neuron_ep_rank,
                get_neuron_ep_tp_group,
            )

            self.ep_degree = get_neuron_ep_degree()
            self.ep_rank = get_neuron_ep_rank()
            self.ep_tp_group = get_neuron_ep_tp_group()
            self.tp_degree = self.ep_tp_group.world_size
        else:
            self.ep_degree = 1
            self.ep_rank = 0
            self.ep_tp_group = self.tp_group
            self.tp_degree = self.world_size

        # Experts spread across data-parallel replicas would need each token
        # exchanged between replicas before the kernel and reduced after -- the
        # ``world_size_across_dp`` layout ``resolve_ep_degree`` allows and the
        # plugin's MLP-TP group exists to serve. This block reduces over
        # ``get_tp_group()`` only, so off-replica tokens would be silently
        # dropped rather than raise, hence the guard rather than a best effort.
        #
        # What that costs, concretely: 397B-A17B in bf16 is 725 GiB, and a single
        # tensor-parallel group cannot exceed 32 ranks because it has 32 query
        # heads -- so it lands at 22.7 of each core's 24 GiB with the KV cache in
        # what is left. Sharding the experts across two DP replicas instead would
        # halve that. Worth having; not in this PR.
        if self.ep_degree > 1 and dp_size > 1:
            raise NotImplementedError(
                "Qwen3.5 expert parallelism across data-parallel replicas is "
                "not implemented; keep expert parallelism inside one tensor-"
                "parallel group."
            )
        if self.num_experts % self.ep_degree:
            raise ValueError(
                f"num_experts={self.num_experts} must be divisible by "
                f"ep_degree={self.ep_degree}"
            )
        # Linear placement, matching ``gpt_oss``: EP rank k owns experts
        # ``[k * num_local_experts, (k + 1) * num_local_experts)``.
        self.num_local_experts = self.num_experts // self.ep_degree

        if self.hidden_size % 256:
            raise NotImplementedError(
                f"hidden_size={self.hidden_size} must be a multiple of 256 for "
                f"the fused decode MoE kernel."
            )
        if self.intermediate_size % self.tp_degree:
            raise ValueError(
                f"moe_intermediate_size={self.intermediate_size} must be "
                f"divisible by tp_degree={self.tp_degree}"
            )
        self.inter_per_rank = self.intermediate_size // self.tp_degree
        if self.inter_per_rank % 128:
            raise NotImplementedError(
                f"moe_intermediate_size/tp_degree = {self.inter_per_rank} must "
                f"be a multiple of 128 for the fused decode MoE kernel (and of "
                f"16 for the prefill one). Either reduce "
                f"tensor_parallel_size, or pass enable_expert_parallel=True so "
                f"that tp_degree = tensor_parallel_size / ep_degree is smaller. "
                f"This is the constraint that makes EP mandatory for 397B-A17B."
            )
        # ``enable_expert_parallel`` with no explicit degree resolves to
        # ``ep_degree = world_size``, i.e. ``tp_degree = 1``, which satisfies the
        # guard above but is the expensive end of the range: see ``BLOCK_SIZE``.
        if self.ep_degree > 1 and self.tp_degree == 1:
            smallest = next(
                (
                    g
                    for g in range(2, self.ep_degree)
                    if self.ep_degree % g == 0
                    and (self.intermediate_size * g // self.ep_degree) % 128 == 0
                ),
                None,
            )
            if smallest is not None:
                logger.warning(
                    "ep_degree=%d leaves tp_degree=1. ep_degree=%d holds the "
                    "same bytes per rank (weights shard world_size ways either "
                    "way) but does strictly less blockwise prefill work, because "
                    "per-rank work scales with ep_degree while memory does not. "
                    "Set NeuronConfig.ep_degree=%d to take it.",
                    self.ep_degree,
                    smallest,
                    smallest,
                )

        shared = moe.shared_expert_intermediate_size
        if shared % self.world_size:
            raise ValueError(
                f"shared_expert_intermediate_size={shared} must be divisible by "
                f"tp_size={self.world_size}"
            )
        self.shared_per_rank = shared // self.world_size

        h, i_r = self.hidden_size, self.inter_per_rank

        # Routed experts, in the layout both kernels want. Only this rank's
        # experts under EP; all of them under TP.
        self.experts_gate_up_weight = nn.Parameter(
            torch.empty(self.num_local_experts, h, 2, i_r, dtype=self.dtype)
        )
        self.experts_down_weight = nn.Parameter(
            torch.empty(self.num_local_experts, i_r, h, dtype=self.dtype)
        )
        gate_up_loader = _expert_gate_up_loader(
            intermediate_size=self.intermediate_size,
            shard_size=i_r,
            num_shards=self.tp_degree,
            dtype=self.dtype,
        )
        down_loader = _expert_down_loader(
            shard_size=i_r, num_shards=self.tp_degree, dtype=self.dtype
        )
        if self.ep_degree > 1:
            local_experts = list(
                range(
                    self.ep_rank * self.num_local_experts,
                    (self.ep_rank + 1) * self.num_local_experts,
                )
            )
            # Narrows the source to this rank's expert range *before* the
            # transform above runs, so the other ranks' experts are never read
            # off disk. Both transforms take the expert count from the source
            # shape, so they compose unchanged.
            gate_up_loader = expert_parallel_tensor_dim_loader(
                local_experts, gate_up_loader
            )
            down_loader = expert_parallel_tensor_dim_loader(local_experts, down_loader)
            # The loader is handed the *global* TP rank, but under EP the
            # intermediate-dimension shard index is this rank's position inside
            # its own EP partition. Outermost, so it overrides the rank the
            # wrapper passes down.
            ep_tp_rank = self.ep_tp_group.rank_in_group
            gate_up_loader = with_rank_override(gate_up_loader, rank=ep_tp_rank)
            down_loader = with_rank_override(down_loader, rank=ep_tp_rank)
        set_weight_loader(self.experts_gate_up_weight, gate_up_loader)
        set_weight_loader(self.experts_down_weight, down_loader)

        # Router. Replicated and **global** -- every rank scores every expert,
        # even under EP: the decode kernel computes the full router itself and
        # uses ``rank_id`` to work out which slice of those logits it owns, and
        # prefill slices the affinities down with ``get_local_expert_affinities``
        # after the router runs. It is 1 MiB. Stored in the checkpoint's
        # ``[E, H]`` orientation so the load is an identity; both kernels take
        # ``[H, E]``, hence the ``.T`` below, which the compiler folds into a
        # constant.
        self.router_weight = nn.Parameter(
            torch.empty(self.num_experts, h, dtype=self.dtype)
        )

        # Shared expert: an ordinary SwiGLU MLP, sharded like the dense one.
        s_r = self.shared_per_rank
        self.shared_gate_proj_weight = nn.Parameter(
            torch.empty(h, s_r, dtype=self.dtype)
        )
        self.shared_up_proj_weight = nn.Parameter(torch.empty(h, s_r, dtype=self.dtype))
        self.shared_down_proj_weight = nn.Parameter(
            torch.empty(s_r, h, dtype=self.dtype)
        )
        for param in (self.shared_gate_proj_weight, self.shared_up_proj_weight):
            set_weight_loader(
                param,
                sharding_weight_loader(
                    shard_dim=1,
                    shard_size=s_r,
                    num_shards=self.world_size,
                    is_storage_transposed=True,
                ),
            )
        set_weight_loader(
            self.shared_down_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=s_r,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )
        # The scalar gate on the shared branch. Replicated, ``[1, H]`` as stored.
        self.shared_expert_gate_weight = nn.Parameter(
            torch.empty(1, h, dtype=self.dtype)
        )

    # ── helpers ──────────────────────────────────────────────────────────

    def _shared_expert(self, normed: torch.Tensor) -> torch.Tensor:
        """``sigmoid(gate(x)) * SwiGLU(x)``, as this rank's partial sum over I.

        The gate is computed from the replicated weight, so it is identical on
        every rank; scaling each rank's partial ``down`` output by the same
        scalar and then reducing gives the same answer as scaling the reduced
        sum, and saves having to reduce twice.
        """
        out = (
            F.silu(normed @ self.shared_gate_proj_weight)
            * (normed @ self.shared_up_proj_weight)
        ) @ self.shared_down_proj_weight
        gate = normed @ self.shared_expert_gate_weight.transpose(0, 1)  # [T, 1]
        return out * torch.sigmoid(gate.float()).to(out.dtype)

    def _combine(self, out: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        """Reduce this rank's partial sum over the intermediate dimension."""
        if self.world_size == 1:
            return out
        if is_prefill and SEQUENCE_PARALLEL:
            return self.tp_group.reduce_scatter(out, dim=0).contiguous()
        return self.tp_group.all_reduce(out)

    # ── phases ───────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        is_prefill: bool,
        norm: nn.Module,
        positions: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``hidden_states`` is the residual *before* ``norm``.

        ``norm`` is the layer's ``post_attention_layernorm``: this module needs
        both the normalised activations (for the shared expert, and for the
        prefill kernel which does not fuse a norm) and the raw weight (for the
        decode kernel, which does). Passing the module rather than duplicating
        the parameter keeps one copy in the checkpoint mapping.
        """
        hidden_states = hidden_states.to(self.dtype)
        if is_prefill:
            return self.forward_prefill(hidden_states, norm, positions, rank)
        # ``1 + weight`` because ``Qwen3_5RMSNorm`` is zero-centred, and fp32
        # because that is the dtype the kernel is handed elsewhere in the plugin.
        gamma = (1.0 + norm.weight.float()).unsqueeze(0)
        return self.forward_decode(hidden_states, norm, gamma, rank)

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        norm: nn.Module,
        positions: torch.Tensor | None,
        rank: torch.Tensor | None,
    ) -> torch.Tensor:
        normed = norm(hidden_states)

        # Router on the sequence-parallel shard, before the gather: it is a
        # [T/tp, H] x [H, 256] matmul either way, and gathering [T, 256] scores
        # is cheaper than gathering first.
        expert_affinities = NF.router(
            hidden_states=normed,
            router_weights=self.router_weight.T,
            top_k=self.top_k,
            router_bias=None,
            activation="softmax",
            computation_dtype=torch.float32,
            # softmax over all 256 -> top-8 -> L1 renormalise, which is what
            # ``Qwen3_5MoeTopKRouter`` does. The default order (top-k first)
            # would give different weights.
            router_computation_order=(
                RouterComputationOrder.PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER
            ),
        )

        gathered = self.world_size > 1 and SEQUENCE_PARALLEL
        if gathered:
            expert_affinities = self.tp_group.all_gather(expert_affinities, dim=0)
            normed = self.tp_group.all_gather(normed, dim=0)

        if self.ep_degree > 1:
            # ``build_blockwise_mapping`` wants ``[T, E_local]``, so drop the
            # columns for experts this rank does not hold. Their affinities are
            # not lost -- the rank that owns them contributes those terms, and
            # the combine below sums across ranks.
            local_expert_indices = torch.arange(
                self.ep_rank * self.num_local_experts,
                (self.ep_rank + 1) * self.num_local_experts,
                device=expert_affinities.device,
                dtype=torch.int32,
            )
            expert_affinities = NF.get_local_expert_affinities(
                expert_affinities, local_expert_indices
            )

        # Padding tokens still route somewhere, and blocks that hold nothing but
        # padding are skipped at runtime via ``conditions`` -- worth having,
        # because only a third of the block slots are real to begin with. The
        # mask is only sound while a prefill carries one monotonic sequence,
        # which is the same assumption the causal mask in ``Qwen3_5Attention``
        # already makes.
        padding_mask = None
        if positions is not None and positions.dim() == 1:
            if positions.shape[0] == normed.shape[0]:
                last_real = torch.argmax(positions)
                token_ids = torch.arange(positions.shape[0], device=positions.device)
                padding_mask = token_ids <= last_real

        (
            expert_affinities_masked,
            token_position_to_id,
            block_to_expert,
            conditions,
        ) = NF.build_blockwise_mapping(
            expert_affinities=expert_affinities,
            num_local_experts=self.num_local_experts,
            num_experts_per_token=self.top_k,
            block_size=BLOCK_SIZE,
            # The TP sub-group inside this rank's EP partition: the full
            # tp_group under pure TP, a single-rank group under pure EP.
            moe_group=self.ep_tp_group,
            # Under TP every rank holds every expert and shards I, so the block
            # *construction* is what gets split across ranks (and merged with a
            # max-reduce), not the expert set. Under pure EP this is 1 and each
            # rank builds blocks for its own experts alone.
            tp_degree=self.tp_degree,
            padding_mask=padding_mask,
            # A tensor, not an int: baking the rank in would compile one graph
            # per rank.
            rank=rank,
        )

        out = NF.moe_cte(
            implementation=MoECTEImplementation.shard_on_block,
            conditions=conditions,
            hidden_states=normed,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=self.experts_gate_up_weight,
            down_proj_weight=self.experts_down_weight,
            activation_function=ActFnType.SiLU,
            block_size=BLOCK_SIZE,
            token_position_to_id=token_position_to_id.to(dtype=torch.int32),
            block_to_expert=block_to_expert.to(dtype=torch.int32),
            # POST_SCALE: the affinity multiplies the expert's *output*, matching
            # ``expert_output * top_k_weights`` in ``Qwen3_5MoeExperts``.
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            skip_token=True,
            is_tensor_update_accumulating=True,
            compute_dtype=nl.bfloat16,
        )

        out = out + self._shared_expert(normed)
        return self._combine(out, is_prefill=True)

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        norm: nn.Module,
        gamma: torch.Tensor,
        rank: torch.Tensor | None,
    ) -> torch.Tensor:
        total_tokens = hidden_states.shape[0]
        # Selective loading: below the threshold the kernel DMAs only the experts
        # the batch actually selected. ``rank_id`` tells the kernel which slice
        # of the global expert ids this rank holds -- all of them under pure TP,
        # hence 0 there -- and the kernel only consults it on the all-expert
        # path, so under EP that path is mandatory rather than an optimization
        # choice. The denominator is the *local* expert count, since that is
        # what this rank would have to DMA.
        loaded_fraction = total_tokens * self.top_k / self.num_local_experts
        is_all_expert = (
            loaded_fraction >= SELECTIVE_LOADING_THRESHOLD or self.ep_degree > 1
        )
        rank_id = None
        if is_all_expert:
            rank_id = torch.tensor(
                [[self.ep_rank]], dtype=torch.int32, device=hidden_states.device
            )

        out = NF.moe_block_tkg(
            inp=hidden_states.unsqueeze(0),
            gamma=gamma,
            router_weights=self.router_weight.T,
            expert_gate_up_weights=self.experts_gate_up_weight,
            expert_down_weights=self.experts_down_weight,
            rank_id=rank_id,
            top_k=self.top_k,
            eps=self.eps,
            router_act_fn=RouterActFnType.SOFTMAX,
            # softmax over all logits, then top-k, then renormalise. See the
            # router note in ``forward_prefill``.
            router_pre_norm=True,
            norm_topk_prob=True,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            hidden_act_fn=ActFnType.SiLU,
            is_all_expert=is_all_expert,
            # The shared expert is applied outside, so the kernel must not add
            # one; and its router logits are unused.
            skip_router_logits=True,
        )
        if isinstance(out, tuple):
            out = out[0]

        out = out + self._shared_expert(norm(hidden_states))
        return self._combine(out, is_prefill=False)


__all__ = ["BLOCK_SIZE", "Qwen3_5SparseMoeBlock"]
