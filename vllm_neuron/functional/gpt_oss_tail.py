# SPDX-License-Identifier: Apache-2.0
"""Fused GPT-OSS decode sampling tail: final RMSNorm + lm_head GEMM + top-k + sampler.

The decode step runs the tail as three separate dispatches -- the final ``GptOssRMSNorm``, the
column-parallel ``lm_head``, and ``Sampler`` (which is a distributed top-k followed by the
sampling math). The ``[T, H] -> [T, V_s]`` boundary in particular goes to HBM and comes straight
back. This module routes all three through ONE NKI kernel, which deletes that boundary. The
kernel keeps in SBUF every intermediate that no collective reads; the collective buffers must stay
in HBM.

The kernel is the CHUNKED tail, vendored at
``functional/vendored_kernels/gpt_oss_tail_chunked.py``. It is the only tail this module supports.
It shards batch ROWS across the lm_head TP group instead of gathering candidates, so it needs ONE
logits exchange where the candidate-gathering tails need two all-gathers. The default ``(8, 16)``
mesh runs that exchange as TWO ``all_to_all`` hops. Each rank then computes only the ``gbs / VTP`` rows
it owns, and the caller puts the row blocks back in batch order with ``chunked_tail_gather``.
That map is the identity at one hop and a permutation for a multi-hop mesh such as ``(8, 16)``,
where a rank-ordered concatenation misplaces the rows of 126 of the 128 ranks. Always apply it: it
costs nothing when it is the identity and is required when it is not.

The tail is ON by default. Set ``VLLM_NEURON_GPT_OSS_TAIL_KERNEL=0`` to take the torch tail back.
That switch is the only environment variable here: with one kernel, a mode name would be a second
spelling of the same state.

Two checks decide eligibility. An unmet check falls back to the unfused path. It does not fail the
compile. ``gpt_oss_tail_precheck`` holds the host-side sampler preconditions, which are the ones
that would otherwise diverge silently, as different tokens. ``gpt_oss_tail_reject_reason`` asserts
the one supported geometry. Both report through one ``[gpt_oss_tail]`` log line, so every fallback
names its cause.

The kernel reads its lm_head shard as ``[H, V_s]``, so the caller must keep a transposed
C-contiguous mirror of it. That mirror is REQUIRED, not an option: see ``gpt_oss_chunked_tail``.
"""

import logging
import math
from enum import Enum
from typing import Optional, Tuple

import torch
from packaging.version import InvalidVersion, Version
from torch import Tensor

import nki
import nki.language as nl

from libtorch_neuronx_lite.compile.platform import get_platform_target
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron import envs
from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)


# The chunked tail is a VENDORED copy in this repo, so it does not need the installed nkilib to
# have gpt_oss_tail or experimental.topk. It needs only nki and nkilib.core.utils.
# Import-guard rather than hard-depend: a missing module must degrade to the torch tail, not break
# model construction.
try:
    from vllm_neuron.functional.vendored_kernels.gpt_oss_tail_chunked import (
        _HW_SORT_AVAILABLE as _CHUNKED_HW_SORT_AVAILABLE,
        compose_hops as _chunked_compose_hops,
        gpt_oss_tail_chunked_megakernel as _chunked_megakernel,
        owned_row_block as _chunked_owned_row_block,
    )

    _CHUNKED_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on nki and the installed nkilib
    _chunked_compose_hops = None
    _chunked_megakernel = None
    _chunked_owned_row_block = None
    _CHUNKED_HW_SORT_AVAILABLE = False
    _CHUNKED_AVAILABLE = False


# The kernel's two selection stages are both nisa.topk, which is gen3-only, and trn3 is the only
# gen3 platform-target family. Mirrors the gate in functional/topk.py.
_TAIL_PLATFORM_PREFIX = "trn3"

# Supported GPT-OSS-120B decode geometry: TP8, lm_head DP16, LNC1 on trn3.
# v_s comes from lm_head.out_features_per_rank, so v_s * world equals _VOCAB.
_VOCAB = 201088
_SUPPORTED_WORLD = 128
_SUPPORTED_V_S = _VOCAB // _SUPPORTED_WORLD
_SUPPORTED_HIDDEN = 3072
_SUPPORTED_HIDDEN_ACTUAL = 2880
_SUPPORTED_K = 256
# Known numerical gap AT this k, not at a rejected one. On a bf16 weight the k=256 INDEX path has
# no tie-aware comparator, so a correct tie swap reads as a large index difference. It applies at
# every shard, 1571 included, and no gate catches it.

# nkilib validates width, p, and k. This gate does not repeat that check because v_s and k are fixed.
# If either changes, validate snake_padded_n(v_s), not v_s.

# Rows equal local batch times lm_head_dp_size. The supported deployment has 128 rows.
# Larger validated kernel batches are different serving configurations, so the gate rejects them.
_SUPPORTED_ROWS = 128

# Draw widths measured on hardware. The caller uses only the first `rows` values.
# Tests require that prefix to contain one distinct value per row.
_MEASURED_DRAW_WIDTH = {8: 16, 128: 256}

# This matches sampling.py::_multinomial when the fused tail is off.
# The caller supplies a draw, so the kernel's three-column fallback is unused.
# A different value changes the sampled quantile without an error.
_KERNEL_DETERMINISTIC_DRAW = 0.5

# World size does not identify the deployment: TP8 x DP16 and TP4 x DP32 both use 128 ranks.
# Only TP8 x DP16 is integrated and measured. EP and embedding DP do not change kernel shapes.
_SUPPORTED_TP = 8
_SUPPORTED_LM_HEAD_DP = 16

# Sorted top-k needs both the nki signature and a compiler BIR feature.
# `_CHUNKED_HW_SORT_AVAILABLE` checks nki; the compiler version is the only available BIR signal.
# Reject unreadable, incomplete, or unknown versions. A false admission can fail warm-up.
# A dev build must report a dotted numeric version.
_MIN_SORTED_TOPK_NEURONX_CC = Version("2.0.280935")


# This probe shells out. `assume_constant_result` runs it once at trace time and keeps the
# subprocess out of the fullgraph trace.
@torch._dynamo.assume_constant_result
def _neuronx_cc_has_sorted_topk_bir() -> bool:
    """True only if the COMPILER's own version is readable, complete, and at or above the release."""
    import re
    import subprocess

    try:
        out = subprocess.run(
            ["neuronx-cc", "--version"], capture_output=True, text=True, timeout=30
        )
    except Exception:  # pragma: no cover - a missing or hanging compiler refuses
        return False
    if out.returncode != 0:
        # A command that failed did not report a version, whatever it printed on the way out.
        return False
    # BOTH streams: the banner has appeared on stderr as well as stdout.
    text = "{}\n{}".format(out.stdout or "", out.stderr or "")
    # Match only the compiler banner. An unanchored version pattern can select Python or NumPy
    # when the compiler line is absent or printed on the other stream.
    m = re.search(r"NeuronX Compiler version\s+(\S+)", text)
    if m is None:
        return False
    # Use PEP 440. Numeric-prefix parsing admits malformed text and prereleases at the floor.
    # Version rejects malformed text and orders prereleases before the floor.
    try:
        return Version(m.group(1)) >= _MIN_SORTED_TOPK_NEURONX_CC
    except InvalidVersion:
        return False


# The decompositions the GATE ADMITS: (8, 16) two hops on any runtime build, and (128,) one hop,
# which needs runtime build 42cf17c9a or later and yields zero NEFFs on an older one.
_SUPPORTED_HOPS = ((8, 16), (128,))
# Embedding DP sizes the EMBEDDING TP group only, and the tail reads the lm_head weight on the
# lm_head TP group, so the gate admits both values. The GK fused embedding needs 1.
_SUPPORTED_EMBEDDING_DP = (1, 16)
_SUPPORTED_EP = 128

# The all-to-all decomposition. (8, 16) is the DEFAULT because it is the only layout THIS GATE admits
# and the one measured to run here: hop 0 stays inside one device and hop 1 takes one core from every
# device. No evidence here establishes that as a universal runtime property, so do not say "every
# runtime": shipped one-hop and (8, 4, 4) results contradict the universal form. ORDER MATTERS -- (16, 8) puts 16 contiguous ranks
# in hop 0, which spans two devices, and is rejected like (128,).
# One 128-rank hop is faster by 2.239 us (0.03415% of engine-active time) and makes the row map the
# identity, but the gate REFUSES it on every runtime: it needs an all-to-all coalesced over a group
# holding several cores on each of several devices, and a measured stack fails
# at warm-up with no mesh subtype for the collective. So setting (128,) only costs the kernel: it
# parses, the gate refuses it, and the tail falls back to torch. Reaching it needs the gate widened
# together with a runtime that executes the collective, not a config change.
_DEFAULT_HOPS = (8, 16)

# ONE program, always: the kernel asserts ``n_prgs == 1``. Every stage after the collective is
# unsharded and carries no core_barrier, so two programs would race on the same buffers.
_TAIL_NUM_PROGRAMS = 1


class TailMode(Enum):
    """Whether the fused tail owns the decode tail.

    ONE value drives all three decisions the caller has to keep consistent -- whether the model
    skips its final RMSNorm, whether the kernel runs, and whether the framework still samples.
    They cannot contradict each other because they all come from this, rather than from separate
    booleans (a "kernel normed and framework also normed" pairing is unrepresentable).

    Members:
        OFF: no fusion. The framework runs norm, lm_head, top-k and sampler.
        CHUNKED: the kernel runs norm, lm_head, the distributed top-k and the sampling math, and
            returns sampled tokens for the row block this rank owns. The framework does not sample.
    """

    OFF = "off"
    CHUNKED = "chunked"

    @property
    def skips_model_final_norm(self) -> bool:
        """True when the kernel applies the final RMSNorm, so the model must not.

        Example:
            >>> TailMode.CHUNKED.skips_model_final_norm
            True
            >>> TailMode.OFF.skips_model_final_norm
            False
        """
        return self is not TailMode.OFF

    @property
    def framework_samples(self) -> bool:
        """True when the framework's own sampler still runs after the tail.

        Example:
            >>> TailMode.OFF.framework_samples
            True
            >>> TailMode.CHUNKED.framework_samples
            False
        """
        return self is TailMode.OFF


# There is ONE kernel, so there is nothing to select: the kill switch alone decides. A mode string
# would be a second spelling of the same state, which is how the old four-mode surface grew, so
# `resolve_tail_mode` reads no mode variable at all.


def resolve_tail_mode() -> TailMode:
    """CHUNKED when the tail is enabled, OFF when it is not.

    Returns:
        ``TailMode.CHUNKED`` if ``VLLM_NEURON_GPT_OSS_TAIL_KERNEL`` is on, else ``TailMode.OFF``.

    Example:
        >>> isinstance(resolve_tail_mode(), TailMode)
        True
    """
    if not envs.VLLM_NEURON_GPT_OSS_TAIL_KERNEL:
        return TailMode.OFF
    return TailMode.CHUNKED


# assume_constant_result is required for the same reason functional/topk.py needs it on
# _is_gpsimd_topk_platform: get_platform_target() falls back to a torch ScriptClass that
# Dynamo cannot trace under fullgraph=True. Folding it to a constant runs it off-graph.
@torch._dynamo.assume_constant_result
def _is_tail_kernel_platform() -> bool:
    return get_platform_target().startswith(_TAIL_PLATFORM_PREFIX)


# This must stay behind assume_constant_result. A bare logger.info inside the traced tail is not a
# graph break, it is a hard failure: Dynamo raises "logging.Logger method not supported for
# non-export cases", and the server exits before it compiles anything. Folding the call to a
# constant runs it off-graph, so it fires once at trace time.
@torch._dynamo.assume_constant_result
def _log_tail_choice(
    used: bool, n_rows: int, v_s: int, k: int, world: int, reason: str, mode: str
) -> None:
    """Emit one greppable line naming whether the fused tail ran.

    Grep for ``[gpt_oss_tail]`` to see the decision and, on a fallback, why.

    Example:
        >>> _log_tail_choice(True, 128, 1571, 256, 128, "eligible", "chunked")
    """
    logger.info(
        "[gpt_oss_tail] fused=%s mode=%s rows=%d v_s=%d k=%d world=%d reason=%s",
        used,
        mode,
        n_rows,
        v_s,
        k,
        world,
        reason,
    )


@torch._dynamo.assume_constant_result
def _log_local_rows(local_count: int, rows_per_rank: int) -> None:
    """Witness that the kernel took the LOCAL-ROWS form (behind assume_constant_result like every traced log)."""
    logger.info(
        "[gpt_oss_tail] chunked local-rows form: kernel params from the replica-local %d rows "
        "(no lm_head-DP gather), %d rows per rank",
        local_count,
        rows_per_rank,
    )


@torch._dynamo.assume_constant_result
def _log_chunked_rows(
    vtp: int, gbs: int, rows_per_rank: int, permuted: bool, hops: Tuple[int, ...] = ()
) -> None:
    """Witness the row map and the hop shape, because rank order does not give batch order.

    The map is a permutation for a multi-hop mesh. A caller that re-assembles by rank order gets a
    silent wrong answer, so this line makes the placement auditable in a serve log. This rank's own
    block is absent on purpose: it comes from a runtime tensor, so it is not known at trace time.

    ``hops`` is here for a second reason. An all_to_all group must be entirely within one device or
    hold exactly one core per device. A hop that is neither has no mesh subtype, and the runtime
    then rejects it during warm-up with an error that names an op type and a byte count and NOT the
    hop shape that caused it. This line is the only place the shape appears before that.

    Example:
        >>> _log_chunked_rows(128, 128, 1, True, (8, 16))
    """
    logger.info(
        "[gpt_oss_tail] chunked vtp=%d gbs=%d rows_per_rank=%d permuted=%s hops=%s hop0_ranks=%s",
        vtp,
        gbs,
        rows_per_rank,
        permuted,
        hops or (vtp,),
        (hops or (vtp,))[0],
    )


def _sampler_veto(sampler) -> Optional[str]:
    """The sampler states that block the fused tail, and the reason each one reports."""
    if sampler is None:
        return None
    if sampler.all_greedy:
        return "all-greedy"
    if getattr(sampler, "capture_topk", False):
        # tensor_capture records the top-k tensors from the torch path
        # (functional/sampling.py). The fused kernel never produces them, so fusing
        # would silently empty the capture -- in exactly the harness someone would reach for
        # to validate this kernel. Refuse rather than hand back a quiet blank.
        return "topk-capture-active"
    if getattr(sampler, "all2all_group", None) is not None:
        # sampling_dp_degree > 1 routes the reference to batch_sharded_topk
        # (functional/sampling.py), a different algorithm with a different row
        # layout. This module wraps the topk() twin, so the S=8 kernel is the right tool
        # there -- not this one.
        return "sampling-dp-degree-gt-1"
    return None


def gpt_oss_tail_precheck(
    is_prefill: bool,
    sampler,
    on_device_sampling_config,
    sampling_params: Optional[Tensor],
    logit_mask: Optional[Tensor],
    spec_decode_metadata,
    rank: Optional[Tensor],
    mode: TailMode = TailMode.CHUNKED,
    gather_logits: bool = False,
) -> Optional[str]:
    """Host-side preconditions for the fused tail; None means "keep going".

    These are the checks that do not depend on shape, so the caller can run them before the
    decoder stack (it must: the fused path needs un-normed hidden states, so the decision
    also controls whether the model applies its final RMSNorm).

    They are separated from ``gpt_oss_tail_reject_reason`` only because they need model
    objects rather than integers -- but they are logged through the SAME line, so a fallback
    always leaves a ``[gpt_oss_tail]`` breadcrumb. Silently returning False here would be
    indistinguishable from "the flag was never set", which is the confusion this module's
    reason strings exist to prevent.

    The sampler-config checks are the load-bearing ones. The kernel's own asserts fail as a
    compile error. These fail SILENTLY instead, as different tokens. NONE trips on the stock
    default config, so the fused path engages there:
      * ``all_greedy`` is the only sampler veto left, and an operator must set it True. It takes a
        distributed argmax over the full vocab and never reaches top-k, so it agrees with the
        kernel only at temperature 0.

    ``no-on-device-sampling`` is a SHAPE reason, not a sampler one. ``lm_head`` is built with
    ``gather_output=not on_device_sampling_config``, so with ODS off the unfused path returns the
    GATHERED ``[T, V]`` logits while the kernel returns sampled tokens for a row block. The kernel
    cannot reproduce that gather.

    Args:
        is_prefill: True on a prefill pass (the kernel is the decode tail).
        sampler: the model's ``Sampler``, or None if on-device sampling is off.
        on_device_sampling_config: the model's ODS config, or None.
        sampling_params: per-request ``[T, 3]`` params, or None.
        logit_mask: structured-output mask, or None.
        spec_decode_metadata: speculative-decode metadata, or None.
        rank: the model's rank tensor, or None.
        mode: the resolved mode. ``OFF`` reports the switch and skips every other check.
        gather_logits: True when the caller must return logits for logprobs or a debug dump. The
            kernel runs with ``emit_logits=False`` and returns tokens only, so there is nothing to
            gather. It reports ``needs-logits`` rather than gathering a None.

    Returns:
        None if every precondition holds, else a short reason string.

    Example:
        The flag is checked before anything else, so with the tail switched off the other
        arguments cannot change the answer -- the precedence a control arm relies on:

        >>> gpt_oss_tail_precheck(True, None, None, None, None, None, None)  # doctest: +SKIP
        'disabled'
    """
    if mode is TailMode.OFF or not envs.VLLM_NEURON_GPT_OSS_TAIL_KERNEL:
        # FIRST, so a flag-off run reports the switch rather than a downstream reason.
        reason = "disabled"
    elif is_prefill:
        reason = "prefill"
    elif on_device_sampling_config is None or sampler is None:
        reason = "no-on-device-sampling"
    elif gather_logits:
        reason = "needs-logits"
    elif (veto := _sampler_veto(sampler)) is not None:
        reason = veto
    elif sampling_params is None:
        reason = "no-sampling-params"
    elif logit_mask is not None:
        reason = "logit-mask"
    elif spec_decode_metadata is not None:
        reason = "spec-decode"
    elif rank is None:
        reason = "no-rank-tensor"
    else:
        return None
    _log_tail_choice(False, -1, -1, -1, -1, reason, mode.value)
    return reason


def chunked_tail_row_order(
    vtp: int, gbs: int, hops: Tuple[int, ...] | None = None
) -> list[int]:
    """Rank order that puts the chunked tail's per-rank outputs back in batch order.

    The kernel gives each rank a block of batch rows, and with the production two-hop mesh that
    map is NOT the identity. A caller that concatenates by rank order misplaces the rows of 126 of
    the 128 ranks, and no shape or dtype check sees it. Concatenate in the order this returns
    instead.

    Args:
        vtp: number of ranks the vocab is split over.
        gbs: global batch size.
        hops: the exchange's hop sizes, or None for a single hop over all ranks.

    Returns:
        Ranks sorted by the first batch row each one owns.

    Example:
        >>> chunked_tail_row_order(4, 8, hops=(2, 2))
        [0, 2, 1, 3]
        >>> chunked_tail_row_order(4, 8)
        [0, 1, 2, 3]
    """
    from vllm_neuron.functional.vendored_kernels.gpt_oss_tail_chunked import (
        owned_row_block,
    )

    return sorted(range(vtp), key=lambda r: owned_row_block(r, vtp, gbs, hops)[0])


def chunked_tail_gather_index(
    vtp: int, gbs: int, hops: Tuple[int, ...] | None = None
) -> list[int]:
    """Index that turns a rank-ordered gather of the chunked tail's outputs into batch order.

    An ``all_gather`` over the lm_head TP group concatenates the per-rank row blocks in RANK
    order. Position ``vtp``-block ``r`` therefore holds the rows ``owned_row_block(r, ...)``, which
    is not block ``r`` of the batch. Element ``i`` of this index is where batch row ``i`` sits in
    that gathered tensor, so one ``index_select`` restores batch order.

    The whole index depends on ``(vtp, gbs, hops)`` only, never on the rank. So it is the SAME host
    constant on every rank, which is what lets the de-permutation live in one shared NEFF -- unlike
    the per-rank input slice, which must come from the runtime rank tensor.

    Args:
        vtp: number of ranks the vocab is split over.
        gbs: global batch size.
        hops: the exchange's hop sizes, or None for a single hop over all ranks.

    Returns:
        A ``gbs``-long permutation of ``range(gbs)``.

    Example:
        >>> chunked_tail_gather_index(4, 8, hops=(2, 2))
        [0, 1, 4, 5, 2, 3, 6, 7]
        >>> chunked_tail_gather_index(4, 8)
        [0, 1, 2, 3, 4, 5, 6, 7]
    """
    from vllm_neuron.functional.vendored_kernels.gpt_oss_tail_chunked import (
        owned_row_block,
    )

    rows_per_rank = gbs // vtp
    index = [0] * gbs
    for r in range(vtp):
        row_start, n_rows = owned_row_block(r, vtp, gbs, hops)
        for j in range(n_rows):
            index[row_start + j] = r * rows_per_rank + j
    return index


def chunked_tail_gather(
    gathered: Tensor, vtp: int, gbs: int, hops: Tuple[int, ...] | None = None
) -> Tensor:
    """Put a rank-ordered gather of the chunked tail's outputs back into batch order.

    Same map as :func:`chunked_tail_gather_index`, evaluated as arithmetic on a traced
    ``arange`` instead of a host list. A constant index tensor does not lift to the device
    during tracing, so it raises gb4315 under parallel trace and fails HLO conversion after
    a ``.to()``; an ``arange`` is an op and lifts. That is what lets a multi-hop exchange
    trace, and a multi-hop exchange is what a runtime without the LNC=1 all-to-all needs.

    Args:
        gathered: the all_gather output, rank-ordered on dim 0.
        vtp: number of ranks the vocab is split over.
        gbs: global batch size.
        hops: the exchange's hop sizes, or None for a single hop over all ranks.

    Returns:
        ``gathered`` in batch order. At one hop the gather is already batch-ordered and
        ``gathered`` is returned unchanged, so no index is built at all.

    Example:
        >>> import torch
        >>> g = torch.arange(8).reshape(8, 1)
        >>> chunked_tail_gather(g, 4, 8, hops=(2, 2)).flatten().tolist()
        [0, 1, 4, 5, 2, 3, 6, 7]
        >>> chunked_tail_gather(g, 4, 8).flatten().tolist()
        [0, 1, 2, 3, 4, 5, 6, 7]
    """
    hop_sizes = (vtp,) if hops is None else tuple(hops)
    if len(hop_sizes) == 1:
        return gathered
    rows_per_rank = gbs // vtp
    # Peel each hop's coordinate off the batch position, then rebuild the rank index from the
    # coordinates. The remainder left after the last hop is the row within the rank's block.
    pos = torch.arange(gbs, dtype=torch.int64, device=gathered.device)
    rank_of_pos, span, stride = None, gbs, 1
    for hop in hop_sizes:
        span //= hop
        term = (pos // span) * stride
        rank_of_pos = term if rank_of_pos is None else rank_of_pos + term
        pos = pos % span
        stride *= hop
    return torch.index_select(gathered, 0, rank_of_pos * rows_per_rank + pos)


def gpt_oss_tail_reject_reason(
    probe: Tensor,
    n_rows: int,
    hidden: int,
    v_s: int,
    k: int,
    world: int,
    hidden_actual: Optional[int],
    *,
    tp: int,
    lm_head_dp: int,
    embedding_dp: int,
    ep: int,
    hops: Tuple[int, ...],
    lm_head_tp_ranks: Tuple[int, ...],
    mode: TailMode = TailMode.CHUNKED,
) -> Optional[str]:
    """Return None if the fused tail can run at this shape, else a reason why it cannot.

    Deliberately takes plain ints rather than the tensors, because the caller has to decide
    BEFORE it runs the decoder stack: the fused path needs hidden states that have *not*
    been through the final RMSNorm, so skipping that norm and calling the kernel are one
    decision, not two. Splitting them would let the two drift and silently double- or
    zero-apply the norm.

    Returns a reason string rather than a bool so a fallback is greppable: a silent
    ``False`` looks identical to "the flag was never set", and the two need different
    fixes. The size conditions mirror ones the kernel asserts. The availability, switch, platform
    and degree conditions do not: they hold a deployment to the one measured configuration.

    Args:
        probe: any tensor on the target device, used only for the ``can_run_kernel``
            device/CPU-mode check (the lm_head weight is the natural choice).
        n_rows: rows the sampler will see (post lm_head-DP gather). This is the kernel's ``gbs``.
        hidden: padded hidden dim H.
        v_s: this rank's vocab shard width.
        k: candidates kept per rank (``max_top_k``).
        world: size of the lm_head TP group, which is the kernel's VTP.
        hidden_actual: unpadded hidden dim, or None if H is unpadded.
        tp: base TP degree (``--tensor-parallel-size``), which the target deployment calls
            attention tp. ``world == tp * lm_head_dp`` holds by construction, so this is the
            degree that tells one decomposition of a 128-rank group from another.
        lm_head_dp: ``neuron_config.lm_head_dp_size``.
        embedding_dp: ``neuron_config.embedding_dp_size``.
        ep: the RESOLVED expert-parallel degree, from ``get_neuron_ep_degree()``. Not
            ``neuron_config.ep_degree``, which stays 1 when the degree is derived.
        hops: the all-to-all hop sizes. Their product must equal ``world``.
        lm_head_tp_ranks: the group's global ranks in group order. Must be 0..VTP-1 in
            order, because the kernel's mesh assumes exactly that.
        mode: the resolved mode. ``OFF`` reports the switch.

    The four degrees are keyword-only and have NO default, on purpose. A default equal to the
    supported value would admit every caller that forgot to pass one, which is the
    over-admission this argument exists to close.

    Example:
        >>> import torch
        >>> probe = torch.zeros(4, 4, device="meta")
        >>> # None means eligible; any string is the reason it fell back. No value is shown
        >>> # because it is environment-dependent: this function tests kernel availability
        >>> # BEFORE the flag, so an installation without the kernel reports
        >>> # 'vendored-chunked-tail-unavailable' where one with it reports 'disabled'.
        >>> reason = gpt_oss_tail_reject_reason(
        ...     probe, 128, 3072, 1571, 256, 128, 2880,
        ...     tp=8, lm_head_dp=16, embedding_dp=16, ep=128,
        ...     hops=(8, 16), lm_head_tp_ranks=tuple(range(128)),
        ... )
    """
    reason = _reject_reason(
        probe,
        n_rows,
        hidden,
        v_s,
        k,
        world,
        hidden_actual,
        tp=tp,
        lm_head_dp=lm_head_dp,
        embedding_dp=embedding_dp,
        ep=ep,
        hops=hops,
        lm_head_tp_ranks=lm_head_tp_ranks,
        mode=mode,
    )
    _log_tail_choice(
        reason is None, n_rows, v_s, k, world, reason or "eligible", mode.value
    )
    return reason


def _reject_reason(
    probe: Tensor,
    n_rows: int,
    hidden: int,
    v_s: int,
    k: int,
    world: int,
    hidden_actual: Optional[int],
    *,
    tp: int,
    lm_head_dp: int,
    embedding_dp: int,
    ep: int,
    hops: Tuple[int, ...],
    lm_head_tp_ranks: Tuple[int, ...],
    mode: TailMode = TailMode.CHUNKED,
) -> Optional[str]:
    """The checks themselves; the public wrapper adds the one log line.

    Every branch here either REFUSES or falls through to ``_degree_reject_reason`` and then
    ``_size_reject_reason``, which holds the only ``return None`` in the trio. So the order
    decides which reason an operator sees first, never whether a shape is admitted.
    """
    if not can_run_kernel(probe):
        return "kernels-unavailable"
    if not _CHUNKED_AVAILABLE:
        # The vendored kernel failed to import, so nki or nkilib.core.utils is missing.
        return "vendored-chunked-tail-unavailable"
    if mode is TailMode.OFF or not envs.VLLM_NEURON_GPT_OSS_TAIL_KERNEL:
        # Test the flag independently of ``mode`` on purpose. The flag is the master switch, so
        # a caller that resolved a mode elsewhere must not be able to bypass it.
        return "disabled"
    if not _neuronx_cc_has_sorted_topk_bir():
        # The nki half can be present while the compiler half is not, and that combination reaches
        # the kernel and dies at warm-up with no fallback. Refuse here instead.
        return "neuronx-cc-too-old-for-sorted-topk"
    if not _CHUNKED_HW_SORT_AVAILABLE:
        # The kernel needs nisa.topk(sorted=True) and REFUSES a sorted request without it -- the
        # software sort was deleted, so there is no degradation inside the kernel and the compile
        # dies after the host has already materialised buffers. Fall back to the torch tail here.
        # This reads the nki half only. The compiler half is checked separately, just above, by
        # _neuronx_cc_has_sorted_topk_bir(), so a compiler-only gap is refused before this point.
        return "hw-sorted-topk-unavailable"
    if not _is_tail_kernel_platform():
        return "not-gen3"
    if world > 0 and n_rows % world != 0:
        # The kernel shards ROWS, not vocab, across the group: it asserts gbs % vtp == 0 and takes
        # rows_per_rank = gbs // vtp. A non-multiple asserts inside the traced kernel, so refuse
        # here and name the divisor. At the supported world this also makes gbs a multiple of 8.
        return f"rows-{n_rows}-not-multiple-of-vtp-{world}"
    if n_rows != _SUPPORTED_ROWS:
        # One concurrency is the target, so one row count is correct. See _SUPPORTED_ROWS.
        return f"rows-{n_rows}-unsupported-expected-{_SUPPORTED_ROWS}"
    degree_reason = _degree_reject_reason(
        world=world,
        tp=tp,
        lm_head_dp=lm_head_dp,
        embedding_dp=embedding_dp,
        ep=ep,
        hops=hops,
        lm_head_tp_ranks=lm_head_tp_ranks,
    )
    if degree_reason is not None:
        return degree_reason
    return _size_reject_reason(hidden=hidden, v_s=v_s, k=k, hidden_actual=hidden_actual)


def _degree_reject_reason(
    *,
    world: int,
    tp: int,
    lm_head_dp: int,
    embedding_dp: int,
    ep: int,
    hops: Tuple[int, ...],
    lm_head_tp_ranks: Tuple[int, ...],
) -> Optional[str]:
    """Refuse any parallelism decomposition that is not the one measured deployment.

    The world size comes first, because it is the product the other degrees decompose: with the
    wrong world, which split produced it is the second question. Then the split itself, which no
    shape in this gate can see -- TP8 x lm_head_dp 16 and TP4 x lm_head_dp 32 agree on the world,
    the rank list, v_s and the row count.

    Each degree names itself and its expected value, so an operator reads WHICH degree to change
    rather than that "the world is wrong" when the world is right.
    """
    if world != _SUPPORTED_WORLD:
        return f"world-{world}-unsupported-expected-{_SUPPORTED_WORLD}"
    if math.prod(hops) != world:
        # The hop sizes must multiply to the lm_head group, which compose_hops asserts INSIDE the
        # traced kernel. Refuse here instead, so a wrong decomposition falls back with a reason
        # rather than raising at trace time.
        return f"hops-product-{math.prod(hops)}-unsupported-expected-{world}"
    if tuple(hops) not in _SUPPORTED_HOPS:
        # A product-correct mesh is not enough. An unsupported mesh dies at warm-up with no torch
        # fallback, so admitting it kills the server rather than degrading it.
        return f"hops-{tuple(hops)}-unsupported-expected-one-of-{_SUPPORTED_HOPS}"
    if tuple(lm_head_tp_ranks) != tuple(range(len(lm_head_tp_ranks))):
        # compose_hops takes only the COUNT and emits replica ids 0..vtp-1, so the kernel's mesh
        # assumes this group IS ranks 0..vtp-1 IN ORDER. At a larger global world the second
        # group's real ranks are absent from that mesh and its all_to_all cannot form. A permuted
        # group starts at 0 and still fails, so the reason names the identity, not the first rank.
        return (
            f"lm-head-ranks-must-equal-0-to-{len(lm_head_tp_ranks) - 1}-in-order-"
            f"got-{lm_head_tp_ranks[0]}-to-{lm_head_tp_ranks[-1]}"
        )
    if tp != _SUPPORTED_TP:
        return f"tp-{tp}-unsupported-expected-{_SUPPORTED_TP}"
    if lm_head_dp != _SUPPORTED_LM_HEAD_DP:
        # Reachable only when the caller's world and lm_head_dp disagree, because the group is
        # BUILT as tp * lm_head_dp: the two come from different sources (the group's rank list
        # and neuron_config), so the gate reads both rather than deriving one.
        return f"lm-head-dp-{lm_head_dp}-unsupported-expected-{_SUPPORTED_LM_HEAD_DP}"
    if embedding_dp not in _SUPPORTED_EMBEDDING_DP:
        return (
            f"embedding-dp-{embedding_dp}-unsupported-expected-one-of-"
            f"{'-or-'.join(str(v) for v in _SUPPORTED_EMBEDDING_DP)}"
        )
    if ep != _SUPPORTED_EP:
        return f"ep-{ep}-unsupported-expected-{_SUPPORTED_EP}"
    return None


def _size_reject_reason(
    *,
    hidden: int,
    v_s: int,
    k: int,
    hidden_actual: Optional[int],
) -> Optional[str]:
    """Refuse anything that is not the one supported geometry.

    This tail ships for one configuration, so the gate ASSERTS that configuration instead of
    sweeping an envelope. Each dimension gets its own reason string, because an operator reading a
    log needs to know WHICH one is wrong.
    """
    if v_s != _SUPPORTED_V_S:
        # v_s is lm_head.out_features_per_rank, so this means the lm_head TP group is not the
        # 128-rank supergroup this tail was measured on.
        return f"v_s-{v_s}-unsupported-expected-{_SUPPORTED_V_S}"
    if hidden != _SUPPORTED_HIDDEN or hidden_actual != _SUPPORTED_HIDDEN_ACTUAL:
        return f"hidden-{hidden}-{hidden_actual}-unsupported-expected-{_SUPPORTED_HIDDEN}-{_SUPPORTED_HIDDEN_ACTUAL}"
    if k != _SUPPORTED_K:
        return f"k-{k}-unsupported-expected-{_SUPPORTED_K}"
    return None


def _prepare_norm_inputs(
    hidden_states: Tensor,
    gamma: Tensor,
    hidden_actual: Optional[int],
    zero_interleaved_pad: bool,
) -> Tuple[Tensor, Tensor]:
    """Shape and dtype prep the kernel's RMSNorm stage needs.

    The pad zeroing below is a correctness step, not a tidy-up.

    Args:
        hidden_states: ``[T, H]`` post-MoE hidden states, before the final RMSNorm.
        gamma: ``[1, H]`` or ``[H]`` final RMSNorm weight.
        hidden_actual: unpadded hidden dim, or None if H is unpadded.
        zero_interleaved_pad: True for the mxfp4 checkpoint, False for bf16.

    Returns:
        ``(hidden_states, gamma)`` as fp32 ``[T, H]`` and 2-D ``[1, H]``.

    Example:
        >>> import torch
        >>> h, g = _prepare_norm_inputs(torch.zeros(2, 4), torch.zeros(4), None, True)
        >>> h.dtype, tuple(g.shape)
        (torch.float32, (1, 4))
    """
    if gamma.ndim == 1:
        gamma = gamma.unsqueeze(0)

    hidden_states = hidden_states.to(torch.float32)

    if hidden_actual is not None and hidden_actual < hidden_states.shape[-1]:
        if not zero_interleaved_pad:
            # bf16 checkpoint: the pad is a CONTIGUOUS tail, and model_bf16's RMSNorm reduces
            # over the [:H_u] prefix only while the kernel sums over all H and divides by H_u.
            # Those agree only if the pad lanes are exactly zero, which the bf16 norm does not
            # assume. Zero them here so the equality holds by construction, not by luck.
            hidden_states = hidden_states.clone()
            hidden_states[..., hidden_actual:] = 0.0
            # mxfp4 needs no such step: its norm already reduces to sum(x^2)/H_u
            # (model_mxfp4.py), and a contiguous zeroing there would hit the wrong lanes,
            # because mxfp4's padding is interleaved.

    return hidden_states, gamma


def _clamp_disabled_top_k(sampling_params: Tensor, k: int) -> Tensor:
    """Rewrite a disabled ``top_k`` to ``k``, so a default request still samples.

    Matches the torch sampler's "top_k <= 0 disables top-k" semantics.

    torch carries an explicit ``& (k > 0)`` term in its mask (functional/sampling.py), so a
    disabled k masks NOTHING. The kernel has no such term. It computes its threshold as
    ``max(min(top_k, k) - 1, 0)``, which lands on index 0 for ``top_k = -1``. Index 0 is the row
    MAX, so the kernel masks every other candidate and collapses the row to argmax.

    -1 is the runner's default (neuron_model_runner.py). So without this clamp the fused path
    de-randomizes every default request, silently. Clamping to k reproduces torch exactly: the
    threshold becomes the k-th of exactly k candidates, and nothing is masked.

    index_select, not basic slicing: a column slice of a [T, 3] tensor is strided, and on
    Neuron a strided view cannot be materialized -- both ``params[:, 0:1]`` fed to torch.where
    and an explicit ``.contiguous()`` on it raise "Expected self.is_contiguous() to be true".
    index_select returns a fresh contiguous tensor, which is why nn/sampler.py already
    uses it to shard the logit_mask. Only hardware shows this; the host gate tests cannot.

    Args:
        sampling_params: ``[T, 3]`` fp32 columns [top_k, top_p, temperature].
        k: the kernel's candidate width, used as the "disabled" replacement.

    Returns:
        ``[T, 3]`` with column 0 clamped.

    Example:
        >>> import torch
        >>> p = torch.tensor([[-1.0, 1.0, 1.0]])
        >>> _clamp_disabled_top_k(p, 256)[0, 0].item()
        256.0
    """
    cols = sampling_params.shape[-1]
    idx_top_k = torch.zeros(1, dtype=torch.int64, device=sampling_params.device)
    idx_rest = torch.arange(1, cols, dtype=torch.int64, device=sampling_params.device)
    top_k = torch.index_select(sampling_params, 1, idx_top_k)
    rest = torch.index_select(sampling_params, 1, idx_rest)
    sampling_params = torch.cat(
        [torch.where(top_k <= 0, torch.full_like(top_k, float(k)), top_k), rest],
        dim=-1,
    )
    return sampling_params


def tail_glue_anchor_zero(anchor: Tensor) -> Tensor:
    """Return a ``[1]`` int32 ZERO whose value depends on ``anchor`` (the GK kernel output), so the row-select
    glue that consumes the sum is scheduled after the kernel instead of hoisted into its last layer.

    Exact: ``x != x`` is 1 for NaN and 0 otherwise, and ``minimum(., 0)`` is 0 in both cases; ``t + 0`` is ``t``
    for every integer and every float except ``-0.0`` -> ``+0.0``, which no sampling parameter distinguishes.
    """
    x = anchor.reshape(-1)[0:1]
    isnan = (x != x).to(torch.int32)
    return torch.minimum(isnan, torch.zeros_like(isnan))


@nki.jit
def _gpt_oss_chunked_tail_nki(
    hidden_states: nl.NkiTensor,
    gamma: nl.NkiTensor,
    lm_head_weight_t: nl.NkiTensor,
    sampling_params: nl.NkiTensor,
    hop_meshes: tuple,
    eps: float = 1e-5,
    hidden_actual: Optional[int] = None,
    zero_interleaved_pad: bool = True,
    k: int = 256,
):
    """Thin @nki.jit wrapper for the VENDORED chunked tail.

    ``hop_meshes`` arrives as a plain tuple of rank-tuples, which is what lets it cross the HOP
    boundary: the kernel builds each ``ncc.ReplicaGroup`` itself. ``emit_logits`` is passed
    explicitly because the kernel DEFAULTS it to True, and returning per-rank logits this caller
    discards would cost a [gbs, V_s] write for nothing.
    """
    return _chunked_megakernel(
        hidden_states=hidden_states,
        gamma=gamma,
        lm_head_weight_t=lm_head_weight_t,
        eps=eps,
        hidden_actual=hidden_actual,
        zero_interleaved_pad=zero_interleaved_pad,
        k=k,
        sampling_params=sampling_params,
        hop_meshes=hop_meshes,
        emit_logits=False,
        deferred_rms=True,
        skip_nan_fold=True,
    )


def gpt_oss_chunked_tail(
    hidden_states: Tensor,
    gamma: Tensor,
    lm_head_weight_hv: Tensor,
    sampling_params: Tensor,
    rank: Tensor,
    lm_head_tp_ranks: Tuple[int, ...],
    deterministic: bool,
    hops: Tuple[int, ...] = _DEFAULT_HOPS,
    k: int = 256,
    eps: float = 1e-5,
    hidden_actual: Optional[int] = None,
    zero_interleaved_pad: bool = True,
    local_rows_rank: Optional[Tensor] = None,
) -> Tensor:
    """Run the VENDORED chunked tail, which shards batch ROWS instead of gathering candidates.

    Eligibility is not re-checked here: the caller must already have got ``None`` from
    ``gpt_oss_tail_reject_reason``.

    LOCAL-ROWS form (``local_rows_rank`` given): ``sampling_params`` holds only THIS REPLICA's
    ``[gbs / n_replicas, 3]`` rows and ``local_rows_rank`` is the stamped TP rank-in-group tensor.
    Valid only for the one-hop identity row map on a contiguous, aligned TP mesh (the CALLER asserts
    the mesh): rank ``r = tp*k + s`` owns exactly rows ``[s*R, (s+1)*R)`` of its own replica ``k``,
    so the lm_head-DP all_gather of the metadata is not needed. ``gbs`` then comes from
    ``hidden_states.shape[0]``; the draw column is still drawn over the WHOLE batch.

    Two things differ structurally from a candidate-gathering tail, and both are why this returns
    a row block rather than the whole token column. The kernel takes NO ``rank``: row ownership is
    positional, so this function must select this rank's rows itself. It reads ``rank`` at runtime,
    from a tensor, because ONE NEFF runs on every rank. And each rank computes only the
    ``gbs // vtp`` rows it owns, so the tokens are NOT identical across the group.

    Args:
        hidden_states: ``[gbs, H]`` post-MoE hidden states, **before** the final RMSNorm. Cast to
            bf16 here: the kernel widens a 16-bit input exactly on chip, so the outputs are
            bitwise those of an fp32 copy of the SAME bf16 data, while the largest input DMA
            halves. A wider input loses its extra precision in this cast, not on chip.
        gamma: ``[1, H]`` or ``[H]`` final RMSNorm weight.
        lm_head_weight_hv: this rank's lm_head shard stored ``[H, V_s]`` and C-contiguous, which
            is the layout this kernel REQUIRES -- not the ``[V_s, H]`` ``ColumnParallelLinear``
            keeps. The mirror is therefore not an option to gate: a caller must derive it. A bare
            ``.T`` view is numerically identical but keeps the original strides, and the DMA
            descriptor is built from strides, so a view costs about 50x on the weight read while
            every correctness test still passes.
        sampling_params: the FULL ``[gbs, 3]`` columns ``[top_k, top_p, temperature]``. A draw
            column is APPENDED here, so the kernel receives ``[gbs/VTP, 4]``. Sliced to
            this rank's owned rows after that append, because the kernel asserts on
            any other row count.
        rank: this rank's index WITHIN the lm_head TP group, as a one-element TENSOR. A host int
            would bake one rank's rows into a NEFF that every rank runs, so the slice below reads
            the rank at runtime.
        lm_head_tp_ranks: the global ranks of that group, in group order. Only its LENGTH is read,
            and that length is VTP. The shard order itself is a contract of the process group.
        hops: the all-to-all decomposition, whose sizes must multiply to VTP. Defaults to
            ``(8, 16)``, the ONLY tuple this gate admits and the one the kernel's own acceptance
            tests use at VTP=128. ``(128,)`` is one hop and would be slightly faster, but this gate
            REFUSES it, and a measured stack failed it at warm-up. No evidence here fixes a runtime
            version boundary for it, so treat the refusal as this gate's policy plus that one
            measurement, not as a version rule.
        k: candidates kept per rank (``max_top_k``, 256 in production).
        eps: RMSNorm epsilon (1e-5 for GPT-OSS).
        hidden_actual: unpadded hidden dim (2880 for GPT-OSS; H is padded to 3072).
        zero_interleaved_pad: True for the mxfp4 checkpoint, False for bf16.

    Returns:
        The ``[gbs/VTP]`` int32 token ids for the row block this rank owns. The kernel gives a
        ``[gbs/VTP, 1]`` uint32 column, and this function drops the column and casts to int32, so
        the shape and the dtype agree with the torch sampler path and with the runners. The caller must
        assemble the full batch column, and MUST NOT concatenate by rank order: the row map is a
        permutation. Gather the blocks, then de-permute with ``chunked_tail_gather``.

    Example:
        >>> tokens = gpt_oss_chunked_tail(  # doctest: +SKIP
        ...     hidden_states=h, gamma=g, lm_head_weight_hv=w_hv, sampling_params=p,
        ...     rank=rank, lm_head_tp_ranks=tuple(range(128)), k=256, hidden_actual=2880,
        ...     deterministic=True,
        ... )
    """
    assert _CHUNKED_AVAILABLE, (
        "gpt_oss_chunked_tail needs the vendored chunked kernel to import; "
        "gpt_oss_tail_reject_reason returns vendored-chunked-tail-unavailable when it does not"
    )
    if not isinstance(rank, Tensor):
        raise ValueError(
            "rank must be a torch.Tensor, because one NEFF runs on every rank. "
            f"Got {type(rank).__name__}."
        )
    vtp = len(lm_head_tp_ranks)
    hop_sizes = (vtp,) if hops is None else tuple(hops)
    local_rows = local_rows_rank is not None
    if local_rows:
        if not isinstance(local_rows_rank, Tensor):
            raise ValueError(
                "local_rows_rank must be a torch.Tensor (the stamped TP rank), because one NEFF "
                f"runs on every rank. Got {type(local_rows_rank).__name__}."
            )
        if len(hop_sizes) != 1:
            raise ValueError(
                "the local-rows form needs the one-hop identity row map (hops == (vtp,)), "
                f"got hops={hop_sizes}: a multi-hop map owns rows of OTHER replicas."
            )
        gbs = int(hidden_states.shape[0])
    else:
        gbs = int(sampling_params.shape[0])
    hidden_states, gamma = _prepare_norm_inputs(
        hidden_states, gamma, hidden_actual, zero_interleaved_pad
    )
    if sampling_params.shape[-1] != 3:
        raise ValueError(
            f"sampling_params must arrive with 3 columns [top_k, top_p, temperature], "
            f"got {sampling_params.shape[-1]}. The draw column is added below, after the clamp."
        )
    sampling_params = _clamp_disabled_top_k(sampling_params, k)
    # ALWAYS append the draw column, so determinism is a VALUE and not a code path. The kernel reads
    # column 3 when present and its own compiled 0.5 when absent, so writing 0.5 here is NUMERICALLY
    # equivalent to omitting the column -- not identical: 4 columns make the kernel read a tensor
    # slice where 3 give it an immediate, so the graph differs even though the value does not.
    rows = gbs
    if deterministic:
        draw = sampling_params.new_full((rows, 1), _KERNEL_DETERMINISTIC_DRAW)
    else:
        # Draw over ALL gbs rows, then let the owned-row select below take this rank's slice.
        # Drawing only rows_per_rank would give every rank the same values, because workers share
        # one model seed -- reproducing the constant draw this column exists to replace.
        # Draw the width recorded in _MEASURED_DRAW_WIDTH and keep the first rows. The property the
        # caller needs is a DISTINCT HEAD; see _MEASURED_DRAW_WIDTH for which images were measured
        # repeating and which were not.
        width = _MEASURED_DRAW_WIDTH.get(rows)
        if width is None:
            raise ValueError(
                f"no measured draw width for rows={rows}; measured pairs are "
                f"{_MEASURED_DRAW_WIDTH}. Only measured widths are admitted, because a generator "
                f"that repeats can share one value across rows at an unmeasured width."
            )
        draw = torch.rand(
            width,
            device=sampling_params.device,
            dtype=sampling_params.dtype,
        )[:rows].unsqueeze(-1)
    if not local_rows:
        sampling_params = torch.cat((sampling_params, draw), dim=-1)
        if sampling_params.shape[-1] != 4:
            raise ValueError(
                f"sampling_params must reach the kernel with 4 columns "
                f"[top_k, top_p, temperature, draw], got {sampling_params.shape[-1]}."
            )

    h_dim, v_s = lm_head_weight_hv.shape
    assert h_dim == hidden_states.shape[1], (
        f"lm_head_weight_hv must be [H, V_s] with H={hidden_states.shape[1]}, "
        f"got {(h_dim, v_s)}; a [V_s, H] shard here makes v_s H and corrupts every id"
    )

    # The row block comes from the RUNTIME rank. Every rank runs the same NEFF, so a host-side
    # owned_row_block(rank, ...) would give every rank rank 0's rows, and nothing would assert: the
    # shapes, the dtypes and the token count all stay correct. Derive it from the rank tensor, so
    # torch.compile cannot extract a scalar. row_starts below is for the log line only.
    rows_per_rank = gbs // vtp
    row_starts = [_chunked_owned_row_block(r, vtp, gbs, hops)[0] for r in range(vtp)]
    _log_chunked_rows(
        vtp,
        gbs,
        rows_per_rank,
        row_starts != [r * rows_per_rank for r in range(vtp)],
        (vtp,) if hops is None else tuple(hops),
    )
    # Evaluate the row map as ARITHMETIC on the rank tensor, and build no table at any hop count. A
    # constant table fails on every trace path, because a constant does not lift to the device but a
    # neighbouring op does: `device=` under parallel trace gives a meta tensor with no storage, which
    # raises gb4315 against the FakeTensor from `rank`; `.to()` passes dynamo and then dies at HLO
    # conversion, where the sibling arange lifts to xla:0 and the moved constant stays meta; and on a
    # real device `.to()` is a mid-graph host-to-device copy that kills the worker during warm-up.
    # owned_row_block is a mixed-radix sum, so the same recurrence over floor-div, mod, mul and add
    # gives the start for any hop tuple. Only `rank` is a tensor; every span and stride is a host int.
    rank_i = rank.to(dtype=torch.int64).reshape(1)
    row_start = None
    span, stride = gbs, 1
    for hop in (vtp,) if hops is None else tuple(hops):
        span //= hop
        term = ((rank_i // stride) % hop) * span
        row_start = term if row_start is None else row_start + term
        stride *= hop
    owned_rows = (
        torch.arange(rows_per_rank, dtype=torch.int64, device=sampling_params.device)
        + row_start
    )
    if local_rows:
        local_count = int(sampling_params.shape[0])
        if rows_per_rank == 0 or local_count % rows_per_rank != 0:
            raise ValueError(
                f"local-rows form: the replica-local {local_count} rows are not a whole number "
                f"of {rows_per_rank}-row rank blocks (gbs={gbs}, vtp={vtp})."
            )
        local_rank_i = local_rows_rank.to(dtype=torch.int64).reshape(1)
        local_rows_idx = (
            torch.arange(
                rows_per_rank, dtype=torch.int64, device=sampling_params.device
            )
            + local_rank_i * rows_per_rank
        )
        kernel_params = torch.cat(
            (
                torch.index_select(sampling_params, 0, local_rows_idx),
                torch.index_select(draw, 0, owned_rows),
            ),
            dim=-1,
        )
        _log_local_rows(local_count, rows_per_rank)
    else:
        kernel_params = torch.index_select(sampling_params, 0, owned_rows)

    outputs = wrap_nki(_gpt_oss_chunked_tail_nki)[_TAIL_NUM_PROGRAMS](
        hidden_states=hidden_states.to(torch.bfloat16),
        gamma=gamma.to(torch.float32),
        lm_head_weight_t=lm_head_weight_hv,
        sampling_params=kernel_params,
        hop_meshes=_chunked_compose_hops(vtp, hops),
        eps=eps,
        hidden_actual=hidden_actual,
        zero_interleaved_pad=zero_interleaved_pad,
        k=k,
    )
    # emit_logits=False, so the token column is the last output at every arity the kernel returns.
    tokens = outputs[-1] if isinstance(outputs, tuple) else outputs
    # Drop the kernel's trailing column. The kernel gives [rows, 1], but the runners index sampled
    # tokens as [rows] and do int(tokens[i]), which raises TypeError on a 1-element list. Index the
    # column, and do not squeeze: a squeeze of a 1-row batch also removes the row axis.
    # Cast to int32. torch has no promotion rule for uint32, so the dtype after later arithmetic
    # is backend-dependent: XLA gives int64 and the native backend keeps uint32.
    return tokens[:, 0].to(torch.int32)
