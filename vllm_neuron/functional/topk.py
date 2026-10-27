# SPDX-License-Identifier: Apache-2.0
"""
Distributed topk for tensor-parallel inference.

This module provides a distributed topk operation that works across
sharded tensors in a tensor-parallel setting.
"""

import logging
from enum import Enum

import torch
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed._functional_collectives import (
    AsyncCollectiveTensor,
    all_gather_tensor,
)

import nki.language as nl

from libtorch_neuronx_lite.compile.platform import get_platform_target

from nkilib.experimental.topk.gpsimd_topk import (
    create_gpsimd_topk_config,
    gpsimd_topk,
)
from vllm_neuron.functional.vendored_kernels.rotational_topk import (
    create_rotational_topk_config,
    create_topk_config,
    rotational_topk,
)
from libtorch_neuronx_lite.nki.nki_dtype import torch_to_nki_dtype
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron import envs
from vllm_neuron.functional.process_groups import group_ranks_tensor
from vllm_neuron.utils.hardware_config import get_logical_nc_config
from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)


# Common NKI top-k selection seam.
#
# vLLM-Neuron runs top-k under torch.compile and cannot call nkilib's host-side
# ``topk()`` dispatcher inside the graph (it is a host dispatcher, not a
# torch-traceable ``@nki.jit`` kernel, and vllm-neuron also layers its own
# distributed orchestration on top). Instead we mirror nkilib's method dispatch
# at this layer: ``_select_topk_method`` picks a ``TopkMethod`` value, and
# ``_topk_nki`` HOP-wraps the matching ``@nki.jit`` kernel via ``_TOPK_KERNELS``.
#
# The method id is a vllm-neuron-local enum (NOT nkilib's SupportedTopkMethods):
# GPSIMD has no member in nkilib's enum, and the rotational kernel here is the
# vendored one.


class TopkMethod(Enum):
    """vLLM-Neuron-local top-k kernel selector (keys ``_TOPK_KERNELS``)."""

    ROTATIONAL = 0
    GPSIMD = 1


# method -> @nki.jit kernel registry. Each method's compile-time config is built
# by its own helper inside _topk_nki (rotational: _get_rotational_topk_config;
# gpsimd: _get_gpsimd_topk_config), folded to a constant at trace time.
_TOPK_KERNELS = {
    TopkMethod.ROTATIONAL: rotational_topk,
    TopkMethod.GPSIMD: gpsimd_topk,
}


def _select_topk_method(tensor: Tensor, k: int, dim: int) -> TopkMethod | None:
    """Pick the NKI top-k kernel for this (tensor, k, dim), or None for torch.

    Returns the ``TopkMethod`` whose kernel can handle the shape, or ``None`` to
    signal the ``torch.topk`` fallback. This is the single common selection point:
    prefer the faster GpSIMD ``nisa.topk`` kernel where it is supported (gen3,
    bf16, in its shape envelope), else the rotational kernel, else torch. Each
    call site branches only on the returned method, so the preference order lives
    here alone. The kernel is used automatically whenever it can run (following
    the standard vLLM-Neuron kernel pattern); everything else falls to torch.

    Examples:
        >>> import torch
        >>> t = torch.randn(16, 25136, dtype=torch.bfloat16, device="meta")
        >>> _select_topk_method(t, 256, -1)  # meta -> can_run_kernel False -> None
        >>> # on gen3 hw prefers GPSIMD, else ROTATIONAL, else None (torch)
    """
    if _can_use_gpsimd_topk(tensor, k, dim):
        return TopkMethod.GPSIMD
    if _can_use_nki_topk(tensor, k, dim):
        return TopkMethod.ROTATIONAL
    return None


@torch._dynamo.assume_constant_result
def _log_topk_choice(
    kernel: str, n_rows: int, vocab_size: int, k: int, reason: str = ""
) -> None:
    """Emit one greppable line naming the top-k kernel actually used.

    Grep for ``[topk] kernel=`` to see which path ran and whether any fallback
    was taken. Wrapped in ``assume_constant_result`` so it runs eagerly off the
    compiled graph (the call is folded to a constant during tracing) -- it fires
    at trace time, not per decode step, so there is nothing to de-duplicate at
    runtime.

    Args:
        kernel: Kernel selected (e.g. ``rotational-nki`` or ``torch``).
        n_rows: Flattened batch*seq row count.
        vocab_size: Reduction-dim length.
        k: Number of top elements requested.
        reason: Why this path was chosen (e.g. fallback cause); optional.

    Examples:
        >>> _log_topk_choice("rotational-nki", 16, 25136, 256)
        >>> _log_topk_choice("torch", 16, 256, 256, reason="out-of-envelope")
    """
    suffix = f" reason={reason}" if reason else ""
    logger.info(
        "[topk] kernel=%s rows=%d vocab=%d k=%d%s",
        kernel,
        n_rows,
        vocab_size,
        k,
        suffix,
    )


def topk(
    tensor: Tensor,
    k: int,
    dim: int,
    gather_dim: int,
    process_group: ProcessGroup | None = None,
    rank: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Performs distributed topk on sharded tensors using 2-step algorithm.

    Uses the NKI top-k kernel selected by ``_select_topk_method`` where it can
    run, falling back to ``torch.topk`` otherwise.

    Args:
        tensor: Input tensor to perform topk on.
        k: Number of top elements to select.
        dim: Dimension along which to find topk.
        gather_dim: Dimension the tensor is sharded on.
        process_group: Process group for distributed operations.
        rank: Rank of this process within the group, as a scalar tensor
            (dtype int32). Required when ``process_group`` is set and
            ``gather_dim == dim``. Passing rank as a tensor avoids baking
            it as a constant during compilation, which would cause
            redundant per-rank compilations.

    Returns:
        Tuple of (values, indices) with global topk across all shards.

    Example:
        >>> import torch
        >>> import torch.distributed as dist
        >>> from vllm_neuron.functional.topk import topk
        >>>
        >>> # Tensor-parallel inference with TP=2, vocab_size=256
        >>> # Each rank has logits of shape (batch=1, vocab_per_rank=128)
        >>> logits = torch.randn(1, 128)  # Local shard on this rank
        >>> pg = dist.new_group([0, 1])
        >>> rank_t = torch.tensor(dist.get_rank(pg), dtype=torch.int32)
        >>>
        >>> # Compute global top-10 across both ranks
        >>> values, indices = topk(logits, k=10, dim=1, gather_dim=1,
        ...                        process_group=pg, rank=rank_t)
        >>> # Returns shapes (1, 10) with global values and indices
    """
    # Fast path for single device (no process group or tp_degree=1)
    if process_group is None:
        tp_degree = 1
    else:
        tp_degree = torch.distributed.get_world_size(group=process_group)

    # Normalize negative dims. torch 2.13's functional all_gather_tensor
    # (_maybe_view_chunk_cat) miscomputes the view shape for negative
    # gather_dim; dim must move with gather_dim so the `gather_dim == dim`
    # checks below stay consistent.
    dim = dim % tensor.dim()
    gather_dim = gather_dim % tensor.dim()

    if tp_degree == 1:
        method = _select_topk_method(tensor, k, dim)
        if method is not None:
            # Use the selected NKI kernel on Neuron
            return _topk_nki(tensor, k, dim, method)
        else:
            # torch fallback (CPU, kernels disabled, or out of envelope)
            return _topk_torch(tensor, k, dim)

    # Top-k is a pure selection, so it runs in the input dtype: feeding native
    # bf16 lets the bf16-only GpSIMD kernel be selected (an fp32 upcast would
    # force the rotational/torch path). The removed unconditional fp32 upcast was
    # legacy from the deleted NKIv1 kernel; the sampling math's fp32 precision is
    # now preserved explicitly at the sampler (see functional.sampling.sample).

    # Step 1: Local topk on each rank
    local_k = min(k, tensor.shape[gather_dim]) if gather_dim == dim else k
    local_method = _select_topk_method(tensor, local_k, dim)
    if local_method is not None:
        local_value, local_index = _topk_nki(tensor, local_k, dim, local_method)
    else:
        local_value, local_index = _topk_torch(tensor, local_k, dim)

    # Apply sharding offset locally before gather (when gather_dim == dim)
    if gather_dim == dim:
        if rank is None:
            raise ValueError(
                "rank must be provided as a tensor when process_group is set "
                "and gather_dim == dim, to avoid baking rank as a compile-time "
                "constant. Pass torch.tensor(dist.get_rank(group), dtype=torch.int32)."
            )
        offset = rank * tensor.shape[gather_dim]
        local_index = local_index + offset

    # Step 2: Gather from all ranks
    global_values = all_gather_tensor(local_value, gather_dim, group=process_group)
    global_indices = all_gather_tensor(local_index, gather_dim, group=process_group)

    # Step 3: Global topk on gathered results
    global_method = _select_topk_method(global_values, k, dim)
    if global_method is not None:
        values, global_max_local_index = _topk_nki(global_values, k, dim, global_method)
    else:
        values, global_max_local_index = _topk_torch(global_values, k, dim)

    # Convert indices to match global_indices dtype for gather
    global_max_local_index = global_max_local_index.to(dtype=global_indices.dtype)
    # Keep the indirect load in range. Neuron needs an explicit bound for the
    # runtime index.
    global_max_local_index = global_max_local_index.clamp(
        min=0, max=global_indices.shape[dim] - 1
    )
    final_indices = torch.gather(global_indices, dim, global_max_local_index)

    return values, final_indices


def batch_sharded_topk(
    tensor: Tensor,
    k: int,
    dim: int,
    dp_degree: int,
    all2all_group: ProcessGroup,
    gather_group: ProcessGroup,
    rank: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Distributed topk with batch sharding for data-parallel sampling.

    Uses all-to-all to redistribute batch→vocab, performs local topk on larger
    vocab slices, then gathers results for final global topk.

    Flow:
        1. All-to-all: [batch, vocab/TP] → [batch/DP, vocab/TP × DP]
        2. Local topk on each rank's larger vocab slice
        3. Index correction: local indices → global vocab indices
        4. Gather topk results across gather_group
        5. Final topk on gathered results

    Args:
        tensor: Input tensor to perform topk on. Shape: [batch, vocab_per_rank].
        k: Number of top elements to select.
        dim: Dimension along which to find topk (typically -1 for vocab).
        dp_degree: Data-parallel degree (batch sharding factor).
        all2all_group: Process group for batch↔vocab redistribution.
        gather_group: Process group for gathering topk results.
        rank: Runtime rank tensor for the all-to-all fallback.

    Returns:
        Tuple of (values, indices) with global topk across all shards.
        Shapes: [batch/DP, k] for both values and indices.

    Example:
        >>> # TP=8, DP=2, batch=8, vocab_per_rank=16K
        >>> logits = torch.randn(8, 16384)
        >>> values, indices = batch_sharded_topk(
        ...     logits, k=10, dim=-1, process_group=tp_group,
        ...     dp_degree=2, all2all_group=..., gather_group=..., all2all_mesh=...
        ... )
        >>> values.shape, indices.shape  # ([4, 10], [4, 10])
    """
    from vllm_neuron.functional import all_to_all

    # Normalize negative dim: it flows into all_gather_tensor at step 4, whose
    # torch 2.13 _maybe_view_chunk_cat miscomputes the view shape for negative
    # gather dims.
    dim = dim % tensor.dim()

    # Step 1: All-to-all redistribution (batch → vocab)
    tensor = all_to_all(
        tensor,
        split_dim=0,
        concat_dim=dim,
        group=all2all_group,
        rank=rank,
    )

    bs = tensor.shape[0]  # batch/DP
    sharded_size = tensor.shape[dim] // dp_degree

    # No fp32 upcast: run top-k in the native input dtype so the bf16-only GpSIMD
    # kernel can be selected. See the matching note in ``topk()`` -- the legacy
    # NKIv1 fp32 requirement is gone, and the downstream sampling-math fp32
    # precision is preserved explicitly at the sampler, not here.

    # Step 2: Local topk on larger vocab slice
    local_method = _select_topk_method(tensor, k, dim)
    if local_method is not None:
        local_value, local_index = _topk_nki(tensor, k, dim, local_method)
    else:
        local_value, local_index = _topk_torch(tensor, k, dim)

    # Step 3: Index correction (local → global vocab indices)
    # After all-to-all, this rank's vocab is concatenated from the ranks in its
    # all2all group. The shard order matches the rank order in the group, so we
    # just need this rank's all2all group members to map shard ID → global rank.
    # Built from graph-native ops: a torch.tensor(list) here would embed a
    # real constant in the compiled graph, which the native-route StableHLO
    # lowering rejects under FakeTensorMode (see group_ranks_tensor).
    rank_map = group_ranks_tensor(
        all2all_group, device=tensor.device, dtype=torch.int32
    )

    # Decompose the local index. Bound the shard id before it indexes rank_map.
    quotient = (local_index // sharded_size).clamp(min=0, max=rank_map.shape[0] - 1)
    remainder = local_index % sharded_size  # position within shard
    flat_quotient = quotient.flatten()
    shard_rank_id_flat = rank_map[flat_quotient]
    shard_rank_id = shard_rank_id_flat.view(
        quotient.size(0), quotient.size(1)
    )  # Dynamic shape

    # Convert to global vocab index
    global_index = shard_rank_id * sharded_size + remainder

    # Step 4: Gather topk results across gather_group. Values and indices are
    # packed into ONE tensor so a single all-gather moves both. Vocab indices
    # need fp32 to be represented exactly (bf16's 8-bit mantissa cannot hold
    # indices > 256), so the cat promotes the (possibly bf16) values to fp32.
    topk_results = torch.cat([local_value, global_index.float()], dim=0)
    gathered_results = all_gather_tensor(topk_results, dim, group=gather_group)
    global_values, global_indices = torch.split(gathered_results, bs, dim=0)
    global_indices = global_indices.to(torch.int32)
    # Restore the input dtype before the final selection: the cat above promoted
    # to fp32 only to co-transport the indices, and the round-trip is lossless
    # (the values originated in this dtype). This lets the final top-k step select
    # the same kernel as the local step rather than being pinned to fp32.
    global_values = global_values.to(local_value.dtype)

    # Step 5: Final topk on gathered results
    final_method = _select_topk_method(global_values, k, dim)
    if final_method is not None:
        values, final_local_idx = _topk_nki(global_values, k, dim, final_method)
    else:
        values, final_local_idx = _topk_torch(global_values, k, dim)
    # Bound the wide index before the int32 conversion. This prevents an
    # out-of-range value from wrapping before the bound.
    final_local_idx = final_local_idx.clamp(
        min=0, max=global_indices.shape[dim] - 1
    ).to(torch.int32)
    final_indices = torch.gather(global_indices, dim, final_local_idx)

    return values, final_indices


# GpSIMD nisa.topk constraints (mirrored from the kernel's own _validate_gpsimd_topk;
# keep in sync with nkilib.experimental.topk.gpsimd_topk). These are exact
# integer bounds, not a staging envelope, so no dry-run is needed (unlike the
# rotational gate): the kernel's validity asserts live in the @nki.jit body, which
# we cannot run off-graph, but every condition it checks is reproduced here.
_GPSIMD_TOPK_MIN_VOCAB = 8
_GPSIMD_TOPK_MAX_VOCAB = 65536  # exclusive: kernel requires vocab < 65536
_GPSIMD_TOPK_MAX_K = 32768  # exclusive: kernel requires k < 32768
# The nisa.topk ISA op is gen3-only; trn3 is the only gen3 platform-target family
# (trn1/trn1n/trn2/inf2 are pre-gen3). get_platform_target() reports this
# host-side (env override or NRT) without an active NKI backend. Matched by prefix
# (startswith) so trn3 variants are all admitted.
_GPSIMD_TOPK_PLATFORM_PREFIX = "trn3"


# assume_constant_result is REQUIRED here (same reason as _rotational_topk_config_
# compiles): _select_topk_method runs INSIDE the fullgraph=True torch.compile trace,
# and get_platform_target() falls back to torch.classes.neuron.Runtime() -- a
# ScriptClass whose __call__ Dynamo cannot trace ("Unsupported method call ...
# ScriptClass"). Folding the platform check to a constant runs it eagerly off-graph
# (the target is fixed for the process), keeping the untraceable NRT call out of the
# compiled graph. (The NEURON_PLATFORM_TARGET_OVERRIDE env branch alone would trace,
# but the NRT fallback must not reach Dynamo.)
@torch._dynamo.assume_constant_result
def _is_gpsimd_topk_platform() -> bool:
    return get_platform_target().startswith(_GPSIMD_TOPK_PLATFORM_PREFIX)


def _can_use_gpsimd_topk(tensor: Tensor, k: int, dim: int) -> bool:
    """True if the GpSIMD ``nisa.topk`` kernel can handle this (tensor, k, dim).

    The GpSIMD kernel is the preferred NKI top-k path where it applies -- it maps
    each row onto the 16-partition ``nisa.topk`` snake -- but it is narrower than
    the rotational kernel: it is gen3-only (the ``nisa.topk`` ISA op does not exist
    on gen2), bf16-only, and bounded to ``8 <= vocab < 65536`` / ``1 <= k < 32768``
    / ``k < vocab``. Anything outside that falls through to the rotational gate,
    then torch. Reduces over the LAST dimension only.

    ``can_run_kernel`` is the first check (CLAUDE.md kernel-guard rule) so
    ``VLLM_NEURON_DISABLE_NKI_KERNELS`` disables this path too. The gen3 gate uses
    ``get_platform_target()`` rather than ``nisa.get_nc_version()``: the nisa
    version query requires an ACTIVE NKI backend (it raises "No backend set" from
    plain host Python), but this gate runs at host selection time off-graph, so it
    must use the host-safe platform-target lookup instead.

    Examples:
        >>> import torch
        >>> t = torch.randn(16, 25136, dtype=torch.bfloat16, device="meta")
        >>> _can_use_gpsimd_topk(t, 256, -1)  # meta -> can_run_kernel False
        False
    """
    if not can_run_kernel(tensor):
        return False
    if tensor.ndim < 2 or dim not in (-1, tensor.ndim - 1):
        return False
    # GpSIMD is bf16-only (the snake pad sentinel is a bf16 value; the kernel
    # validator rejects anything else). Sampling logits are bf16, so GpSIMD is
    # eligible on the sampling path; anything non-bf16 falls to the rotational gate.
    if tensor.dtype != torch.bfloat16:
        return False
    # gen3-only: the nisa.topk ISA op does not exist pre-gen3. trn3 is the only
    # gen3 platform target; anything else falls through to the rotational gate.
    # Via the assume_constant_result helper so the ScriptClass NRT lookup is folded
    # off-graph (see _is_gpsimd_topk_platform).
    if not _is_gpsimd_topk_platform():
        return False
    vocab_size = tensor.shape[-1]
    if not (_GPSIMD_TOPK_MIN_VOCAB <= vocab_size < _GPSIMD_TOPK_MAX_VOCAB):
        return False
    if not (0 < k < _GPSIMD_TOPK_MAX_K):
        return False
    # Strict k < vocab_size (not the kernel's own k <= vocab): k == vocab_size is
    # the degenerate "sort the whole row" case, which the rotational gate also
    # rejects and torch.topk handles directly -- keep the two NKI gates consistent
    # and leave that case to the torch fallback.
    if k >= vocab_size:
        return False
    return True


def _can_use_nki_topk(tensor: Tensor, k: int, dim: int) -> bool:
    """True if the rotational NKI top-k kernel can handle this (tensor, k, dim).

    The kernel (nkilib.core.topk.rotational_topk) reduces over the LAST
    dimension. Following the standard vLLM-Neuron kernel pattern (cf.
    functional.argmax._can_use_nki_max), the kernel is used automatically
    whenever it can run; everything else falls back to ``torch.topk``.

    The cheap checks below are NECESSARY conditions (runtime availability,
    rank/dim/dtype, and ``0 < k < vocab_size``). They are NOT sufficient: the
    kernel also has a shape-dependent feasibility envelope -- it splits the vocab
    into ``n_stages`` and asserts the concatenated SBUF free dimension
    ``chunk + n_stages * local_k <= 16384``. That bound is non-monotonic in k
    (small k pins ``n_stages`` at the HW minimum, so the per-stage chunk stays
    large and overflows), so it cannot be captured by static vocab/k caps. It is
    decided authoritatively by ``_rotational_topk_config_compiles`` below, which
    dry-runs the kernel's OWN config factories -- the only drift-proof way to
    admit exactly the set the kernel can compile. (Static caps were both unsafe
    and over-conservative: k=2/vocab=32768 passed them but the kernel asserts at
    concatenated_free=16400 > 16384, while the real gpt-oss vocab=201088 was
    needlessly rejected by a 151936 "tested envelope" cap it actually handles.)
    """
    if not can_run_kernel(tensor):
        return False
    if tensor.ndim < 2 or dim not in (-1, tensor.ndim - 1):
        return False
    # Kernel is tested only for bf16/fp32 (its pad sentinel _get_dtype_min has no
    # fp16 branch). Sampling logits are bf16; the distributed paths preserve the
    # input dtype (no longer upcast to fp32), so this admits bf16 directly.
    if tensor.dtype not in (torch.bfloat16, torch.float32):
        return False
    vocab_size = tensor.shape[-1]
    # Strict k < vocab_size: the rotational kernel rejects k == vocab_size for
    # sorted output ("sorted=True is not supported when k == vocab_size"), which
    # arises e.g. for small-vocab models. torch.topk handles k == vocab_size, so
    # leave that (degenerate sort-everything) case to the fallback. NOTE this
    # pre-filter is load-bearing: the dry-run below builds the config (which does
    # NOT check k==vocab_size -- that assert lives in the kernel body) and would
    # otherwise admit k==vocab_size at small vocab. Do not relax to ``k <= vocab``.
    if not (0 < k < vocab_size):
        return False
    n_rows = 1
    for d in tensor.shape[:-1]:
        n_rows *= d
    nki_dtype = getattr(nl, torch_to_nki_dtype(tensor.dtype))
    return _rotational_topk_config_compiles(n_rows, vocab_size, k, nki_dtype)


def _get_topk_num_programs() -> int:
    """Return the LNC grid degree for top-k config builders in this process.

    The SPMD grid must match the logical-NC config the NEFF runs on, else the
    kernel's program sharding leaves output shards unwritten. CPU mode simulates a
    single core, so it pins 1 regardless of the configured degree.
    """
    # The kernel may still reduce to n_prgs=1 internally when BxS == 1.
    return 1 if envs.VLLM_NEURON_CPU_MODE else get_logical_nc_config()


# Authoritative compile-time feasibility gate for the rotational NKI top-k
# kernel: run the kernel's OWN config factories and report whether they build
# without raising. This admits EXACTLY the (n_rows, vocab_size, k) set the kernel
# can compile and cannot drift when nkilib changes HW_PARAMS or its staging cost
# model. Building the two config factories is sufficient: every validity assert
# (kernel_assert) fires inside them, and the rotational kernel now builds its
# constants (rotation matrix, stage offsets) on-device in the kernel body, so
# there is no separate host-side constant-materialization step to run here.
#
# assume_constant_result is REQUIRED, for the same reason _get_rotational_topk_config
# uses it: create_topk_config emits logger.info when BxS == 1, which Dynamo cannot
# trace under fullgraph=True. Folding the returned bool to a constant runs this
# helper eagerly (off-graph) exactly once per distinct (n_rows, vocab_size, k,
# dtype) and keeps it out of the compiled graph. The try/except MUST live inside
# the helper so the AssertionError is consumed eagerly and Dynamo only ever sees a
# clean bool constant. num_programs comes from _get_topk_num_programs() so the
# gate evaluates the same config the real run (_get_rotational_topk_config) builds.
#
# CAVEAT: the kernel signals infeasibility via ``kernel_assert`` (a bare ``assert``).
# Under ``python -O`` / ``PYTHONOPTIMIZE`` asserts are stripped, so the factories
# would NOT raise on an out-of-envelope shape and this gate would wrongly return
# True (the kernel then hard-fails at real compile with no torch fallback). vLLM
# Neuron is not run under -O, but this is the one mode where the gate is unsafe.
@torch._dynamo.assume_constant_result
def _rotational_topk_config_compiles(
    n_rows: int, vocab_size: int, k: int, nki_dtype
) -> bool:
    # Use the VENDORED factories (module-level import) so the feasibility gate
    # dry-runs exactly the config the real run builds (no baseline-vs-vendored
    # drift).
    try:
        topk_config = create_topk_config(
            inp_shape=(n_rows, vocab_size),
            inp_dtype=nki_dtype,
            k=k,
            num_programs=_get_topk_num_programs(),
        )
        create_rotational_topk_config(
            inp_shape=(n_rows, vocab_size), topk_config=topk_config
        )
        return True
    except AssertionError:
        # The kernel signals an out-of-envelope shape via kernel_assert -> a plain
        # AssertionError. That is the ONLY expected "cannot run" signal, so treat it
        # as "fall back to torch.topk". Any OTHER exception (e.g. an ImportError or
        # a TypeError from an nkilib API/signature change) is NOT a feasibility
        # answer -- swallowing it would silently disable the kernel for every shape
        # with no test catching the regression, so let it propagate loudly instead.
        return False


# Build the rotational top-k config (the kernel's compile-time companion object)
# once at trace time and fold it in as a constant. assume_constant_result is
# required because this config-production logic is host-side Python that Dynamo
# cannot (and should not) trace into the graph: the factories run shape/staging
# math and emit progress logs (create_topk_config logs when BxS == 1). None of
# that is a graph operation -- it produces a frozen, shape-determined config.
# Folding it to a constant (same technique as nki_hop.register_kernel_to_torch)
# runs it exactly once per distinct (shape, dtype, k) and keeps it out of the
# compiled graph; without it, fullgraph=True tracing fails on the untraceable
# host code (the logger calls are merely the first thing Dynamo trips over).
#
# The kernel builds its rotation matrix and stage offsets ON-DEVICE in the kernel
# body, so the config needs no host-side constant materialization -- the factory
# output is the complete config the kernel consumes.
@torch._dynamo.assume_constant_result
def _get_rotational_topk_config(n_rows: int, vocab_size: int, k: int, nki_dtype):
    # Uses the VENDORED factories (module-level import); see the import block.
    topk_config = create_topk_config(
        inp_shape=(n_rows, vocab_size),
        inp_dtype=nki_dtype,
        k=k,
        num_programs=_get_topk_num_programs(),  # kernel forces 1 when BxS == 1
    )
    return create_rotational_topk_config(
        inp_shape=(n_rows, vocab_size), topk_config=topk_config
    )


# Build the GpSIMD top-k config once at trace time and fold it in as a constant,
# for the same reasons as _get_rotational_topk_config (host-side shape math that
# Dynamo cannot trace under fullgraph=True). num_programs is the LNC/SPMD grid
# degree the kernel shards BxS over (n_prgs; per_lnc_BxS = ceil(BxS / n_prgs)), not
# a rotational-specific tuning value -- hence the shared constant. The kernel forces
# n_prgs to 1 internally when BxS == 1.
@torch._dynamo.assume_constant_result
def _get_gpsimd_topk_config(n_rows: int, vocab_size: int, k: int, nki_dtype):
    # Uses the nkilib factory (module-level import from nkilib.experimental.topk).
    return create_gpsimd_topk_config(
        inp_shape=(n_rows, vocab_size),
        inp_dtype=nki_dtype,
        k=k,
        num_programs=_get_topk_num_programs(),
    )


# method -> (config-builder, greppable kernel label) for the in-graph HOP dispatch
# in _topk_nki. Adding a method means: register its @nki.jit kernel in
# _TOPK_KERNELS, its config builder here, and admit it in _select_topk_method.
_TOPK_CONFIG_BUILDERS = {
    TopkMethod.ROTATIONAL: (_get_rotational_topk_config, "rotational-nki"),
    TopkMethod.GPSIMD: (_get_gpsimd_topk_config, "gpsimd-nki"),
}


def _topk_nki(
    tensor: Tensor, k: int, dim: int, method: TopkMethod
) -> tuple[Tensor, Tensor]:
    """Top-k via the selected NKI kernel; reduces over the last dim.

    ``method`` is the ``TopkMethod`` chosen by ``_select_topk_method`` at the call
    site; this function HOP-wraps the matching ``@nki.jit`` kernel from
    ``_TOPK_KERNELS`` and builds its config via ``_TOPK_CONFIG_BUILDERS``. Both
    kernels take a single compile-time config object and return (values, uint32
    indices), so the dispatch is uniform. Logs the kernel actually used (grep
    ``[topk] kernel=``).

    Returns sorted-descending (values, int64 indices) to match torch.topk
    (the kernels emit uint32 indices; GpSIMD sorts descending by default).
    """
    if dim < 0:
        dim += tensor.ndim
    assert dim == tensor.ndim - 1, "NKI top-k reduces over the last dim"

    # Callers feed this kernel tensors straight out of all_gather_tensor,
    # which in CPU mode are lazy AsyncCollectiveTensor subclasses. Plain torch
    # ops dispatch through the subclass, but the NKI HOP does not: torch
    # requires an explicit HOP-x-subclass rule and raises NotImplementedError.
    # Materialize before the kernel; on device the XLA collective already
    # returns a plain tensor, so this is a no-op there.
    if isinstance(tensor, AsyncCollectiveTensor):
        tensor = tensor.wait()

    vocab_size = tensor.shape[-1]
    leading_shape = tensor.shape[:-1]
    inp2d = tensor.reshape(-1, vocab_size)
    n_rows = inp2d.shape[0]
    # Split over the LNC programs, an odd row count leaves the last program a one-row tile,
    # which the rotational kernel can return as NaN values with index 0 in some compiles (an
    # MTP verify at batch 1 has 3 rows; it surfaced as token id 0 for the bonus token). Pad
    # with copies of the last row to a multiple of the program count and drop them from the
    # outputs. The kernel's partial-tile fill (cascaded_max_utils.predicated_folded_load)
    # fixes one uninitialized read but did not make that failure go away on its own.
    n_programs = _get_topk_num_programs()
    pad_rows = (-n_rows) % n_programs if n_rows > 1 else 0
    if pad_rows:
        inp2d = torch.cat([inp2d, inp2d[-1:].expand(pad_rows, vocab_size)], dim=0)

    kernel_fn = _TOPK_KERNELS[method]
    config_builder, kernel_label = _TOPK_CONFIG_BUILDERS[method]
    _log_topk_choice(kernel_label, n_rows + pad_rows, vocab_size, k)

    nki_dtype = getattr(nl, torch_to_nki_dtype(inp2d.dtype))
    config = config_builder(n_rows + pad_rows, vocab_size, k, nki_dtype)

    topk_kernel = wrap_nki(kernel_fn)
    values, indices = topk_kernel[n_programs](inp2d, config)
    if pad_rows:
        values, indices = values[:n_rows], indices[:n_rows]

    out_shape = (*leading_shape, k)
    values = values.reshape(out_shape)
    indices = indices.reshape(out_shape).to(torch.int64)
    return values, indices


def _topk_torch(tensor: Tensor, k: int, dim: int) -> tuple[Tensor, Tensor]:
    """
    Compute topk using torch.topk.

    Non-NKI fallback for inputs the rotational top-k kernel cannot handle (see
    ``_can_use_nki_topk``). This is the path taken on CPU, when the kernel runtime
    is disabled, and on Neuron when the input is out of the kernel's envelope --
    fp16, k == vocab_size, or a (vocab, k) whose staging exceeds the kernel's
    concatenated free-dimension HW limit.

    Args:
        tensor: Input tensor.
        k: Number of top elements to select.
        dim: Dimension along which to find topk.

    Returns:
        Tuple of (values, indices) with top-k elements.
    """
    vocab_size = tensor.shape[-1]
    n_rows = 1
    for d in tensor.shape[:-1]:
        n_rows *= d
    # No NKI method selected for this shape (CPU, kernels disabled, or out of
    # the kernel's envelope). Log the fallback so it's visible which calls did
    # not hit an accelerated kernel (grep ``[topk] kernel=torch``).
    _log_topk_choice("torch", n_rows, vocab_size, k, reason="no-nki-method")
    return torch.topk(tensor, k, dim=dim)
