# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V4.1-Flash (timeshare) on one Trainium 2 device
========================================================

Module names here are the checkpoint's own module paths — `layers.<i>.attn.wq_b`,
`layers.<i>.ffn.shared_experts.w1`, `mtp.<s>.markov_head` — which are also the
reference implementation's (`partition_artifact/vendor/model.py`). The identity
is deliberate: it makes "is this the weight the reference uses here" a string
comparison against `model.safetensors.index.json` rather than an argument about
a renaming scheme, and it keeps `weight_loader.build_weight_mappings` 1:1.

What makes this model unlike the Llama/Qwen3 ports in this repo
--------------------------------------------------------------
Five things, each of which shows up below as its own class:

**Hyper-connections.** The residual stream is `hc_mult = 4` parallel copies.
Every sublayer is wrapped: `hc_mixes` projects the flattened stream to
`(2 + hc_mult) * hc_mult = 24` coefficients, splits them into `pre`/`post`/`comb`
and makes `comb` doubly stochastic with 20 Sinkhorn iterations; `hc_pre`
collapses the copies into the sublayer's input; `hc_post` expands the output
back out and mixes the residual through `comb`. The coefficients a sublayer
computes are consumed by the *next* one, which is why `Block.forward` returns
`ffn_pre` alongside the stream.

**MLA-shaped attention with low-rank Q and O.** `num_key_value_heads = 1` at
`head_dim = 512`: one shared latent KV for all 64 query heads. Q goes through
`wq_a` (5120 -> `q_lora_rank` 1280), a norm, then `wq_b` (1280 -> 64*512). The
output projection is factorised and *grouped*: `wo_a` is block-diagonal over
`o_groups = 8`, each group projecting only its own 8 heads into `o_lora_rank`
1024, then `wo_b` reduces 8*1024 -> 5120. That block-diagonality is why `wo_a`
is an `einsum` and not a `Linear`.

**Two KV sources per layer, concatenated into one attention call.** Every layer
attends over a 128-slot sliding window of raw KV. Layers whose
`compress_ratios` entry is nonzero *also* attend over `index_topk = 512`
compressed positions reaching arbitrarily far back. The compressed KV is
produced by the four layers in `kv_source_layer_ids` and read by the rest; which
512 of those positions a query sees is chosen by the `Indexer` at the eight
layers in `index_source_layer_ids`, two-level, with layer 20 publishing
candidate blocks that the later indexers score inside.

**Engram.** Layers 1 and 14 add an n-gram memory read into the residual stream,
gated by how well the retrieved key matches the stream. The table is
384,006,168 rows of 256 fp8 values — 94.6 GiB each, 3.9x one HBM bank — so it is
never resident; `weight_loader.TimeshareWeightStore.engram` gathers the 24 rows
a token needs.

**DSpark.** Three further blocks stored under `mtp.<stage>`, over their own
128-expert pool at top-3, reading the attention inputs of layers 37/38/39
(`dspark_target_layer_ids`) through `mtp.0.main_proj`.

Timesharing
-----------
9.83 GiB of the 475.24 GiB checkpoint is dense; the rest is 384 experts per
layer (sparsely routed, 6 of 384 per token) and the two engram tables (sparsely
read). So the dense part is resident and sharded 4 ways (~2.5 GiB per rank), and
the experts and engram rows are fetched per forward into fixed staging buffers.
See `weight_loader.py`'s module docstring for the measured census and the
throughput numbers that make the schedule feasible.

Where the arithmetic runs
------------------------
Every matrix multiply in this file goes through a NKI kernel in
`kernels/blockquant.py` — the MLA projections, the block-diagonal `wo_a`, the
routed and shared experts, the engram read-out, the compressor, the indexer, the
router, the unembedding and the hyper-connection coefficients. `nki_linear` is
the single dispatch point and there is no torch alternative on the execution
path; `BlockQuantLinear._reference` is the accuracy oracle and nothing calls it.
Top-k goes through `vllm_neuron.functional.topk`, which is this repo's NKI
rotational top-k.

The two pieces of arithmetic that are not matrix multiplies have their own
kernels for the same reason:

* `kernels/sparse_attn.py` — the attention itself. The QK score, the masked
  softmax with the learned per-head sink and the probability-weighted sum, over
  the gathered slot list every layer of this architecture attends to. Reached
  from `sparse_attn`, which keeps only the gather and the mask comparison.
* `kernels/hc_mixes.py` — the hyper-connection coefficients after their GEMM:
  the stream statistic, the three nonlinearities and the 20-iteration Sinkhorn.
  Reached from `HyperConnections.mixes`, which keeps only the reshapes.

What remains in torch is pointwise glue — the norms, SwiGLU, RoPE, the
quantization round trips, the index arithmetic — which the goal permits but
deprioritizes.

That torch has to be *inside a compiled graph*, which is the fact that shapes
this file. `libtorch_neuronx_lite` registers a **backend fallback** for `aten`
on the `neuron` device, not device kernels: eager ops on a Neuron tensor run on
the host, and the fallback does not even cover a dtype cast
(`RuntimeError: Expected self.dtype() == dst.dtype()` on `x.float()`). Several
constructs are also broken under the `vllm_neuron` compile backend; each
workaround below is commented with the error it removes, and
`Tensor.split([sizes], dim)` is the one to remember because it returns **wrong
numbers** rather than failing.

Status
------
Exercised on the device, against the recorded traces in
`partition_artifact/trace/`: the hyper-connection coefficients including the
20-iteration Sinkhorn (cos 1.000000), the unembedding (0.999999), the engram
read off the 94.6 GiB table (1.000085), the 384-of-6 MoE with 180 real routed
experts (0.999864) and a whole MLA attention block (0.999895). Also exercised:
the module tree at the checkpoint's own shapes and dtypes, the per-rank shard
plan, weight loading, `get_kv_spec`/`bind_kv_cache`, `embed_input_ids`,
`compute_logits`.

A prefill now runs end to end on the device — every layer, the engram lookups,
the streamed experts, the DSpark stages, the head and the sampler. The decode
step does not: it runs out of HBM loading the last of ~115 compiled graphs.
`forward`'s own docstring says so precisely, and
`.autohelix/notes/iter-3.md` §5b-ter has the memory table and the fix. Nothing
here runs a subset of the layers or substitutes an approximation; where it
cannot run it raises and names the reason.
"""

from __future__ import annotations

import functools
import gc
import logging

import torch
import torch.nn.functional as F
from torch import nn

# The functional collectives, imported at module level because dynamo will not
# trace into the import machinery (see `_nki_entry`). `_resolve_group_name` is
# the same private helper `vllm_neuron/functional/collectives/all_to_all.py`
# uses to turn a process group into the name these ops take.
from torch.distributed._functional_collectives import (
    _resolve_group_name as _fc_resolve_group_name,
    all_gather_tensor as _fc_all_gather_tensor,
)
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF

from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig

from .config import DeepseekV41Config

logger = logging.getLogger(__name__)

#: `(2 + hc_mult) * hc_mult` — the width of a sublayer's coefficient projection,
#: covering `pre` (hc), `post` (hc) and `comb` (hc x hc). 24 at hc_mult=4.
def _mix_width(hc_mult: int) -> int:
    return (2 + hc_mult) * hc_mult


# =============================================================================
# Section 1: sharding plan
# =============================================================================
# One declarative entry per buffer, consumed by `shard_parameter`. Making this
# data rather than control flow means the plan can be printed and checked
# against the checkpoint's shapes, which is what `verify_shard_plan` does.
#
# `dim=None` replicates. A block-scaled fp8/fp4 weight and its `e8m0` scale
# shard along the *same* logical axis, but the scale's extent is the weight's
# divided by the 32-wide block — so the plan stores the axis and the divisor
# together and `shard_parameter` does not have to know which tensor is which.


class ShardRule:
    """How one buffer is split across the tensor-parallel ranks.

    Attributes:
        dim: axis to split, or None to replicate.
        reason: why this axis — recorded because for several of these buffers
            the axis is only correct in combination with another one's (heads
            with `attn_sink`, groups with `wo_a`), and a wrong-but-plausible
            axis produces a model that runs and is subtly wrong.
    """

    __slots__ = ("dim", "reason")

    def __init__(self, dim: int | None, reason: str) -> None:
        self.dim = dim
        self.reason = reason

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ShardRule(dim={self.dim}, reason={self.reason!r})"


REPLICATE = ShardRule(None, "no free axis; every rank needs the whole tensor")


# =============================================================================
# Section 2: leaf modules
# =============================================================================


class RMSNorm(nn.Module):
    """`vendor/model.py`'s `RMSNorm`, including its fp32 interior, as a NKI
    kernel.

    `rms_norm_eps` is 1e-20 for this model — small enough that computing the
    variance in bf16 would make the epsilon a no-op and the reciprocal sqrt
    noisy, so the cast to fp32 is load-bearing rather than defensive.

    **Why this is a kernel and not four torch ops.** The torch form was

        variance = x.float().square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * x).to(dtype)

    and `mean(-1)` is a sum over `dim` fp32 values *per row*, which is the one
    reduction in this model whose accumulation order the compiler is free to
    choose differently for a 1-row tensor than for a 6-row or 512-row one.
    A different order is a different last bit in the reciprocal square root,
    and an argmax over 129,280 near-tied logits turns that into a different
    token — which is what makes a speculative verify step (`1 + k` rows where a
    plain decode step has 1) emit different text from the same prompt. This is
    ~258 calls per forward (6 per layer x 43), so it is the dominant term.

    `kernels/rmsnorm.rms_norm` reduces each row in its own partition in
    `dim // chunk` chunks whose width depends on `dim` alone, so a row's output
    does not depend on how many rows the step carried. Checked under
    `nki.simulate` against the torch expression above at `dim` 5120, 1536 and
    128: **bit-identical to torch**, and row 0 of a 1-row call is bit-identical
    to row 0 of 6-row and 130-row calls (`/tmp/iter8/ktest_rms.py`, recorded in
    `.autohelix/notes/iter-8.md` §3).
    """

    def __init__(self, dim: int, eps: float, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.register_buffer("weight", torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.dim)
        # `eps` travels as a tensor because a kernel entry point takes tensors:
        # a Python float would be baked into the kernel's cache key, which is
        # harmless, but `wrap_nki` passes positional tensors through to the
        # kernel and a scalar is not one of them.
        eps = torch.full((1, 1), self.eps, dtype=torch.float32, device=x.device)
        out = _nki_entry("rmsnorm", "rms_norm")(
            flat, self.weight.reshape(1, self.dim), eps
        )
        return out.reshape(shape)


#: fp8 e4m3's range, as `vendor/kernel.py`'s `act_quant_kernel` uses it. 448 is
#: the OCP e4m3 maximum; note the Trainium kernels under `kernels/` may use the
#: legacy nl.float8_e4m3 whose maximum is 240, which is what the compiler's
#: `--experimental-unsafe-fp8e4m3fn-as-fp8e4m3` flag is about.
_FP8_MAX = 448.0


def _pow2(exponent: torch.Tensor) -> torch.Tensor:
    """`2 ** exponent`, elementwise, for an integral-valued float exponent.

    Not `torch.exp2`, which has no lowering on this backend and *aborts* the
    interpreter when the graph runs (a PjRt check failure, not an exception).
    Not `torch.exp(e * ln2)` either: measured at 4.3e-04 of absolute error,
    and a `ue8m0` scale is only useful if it is exactly a power of two, so the
    dequantization stays lossless. `torch.pow` against a tensor of twos lowers
    and is bit-exact over the whole range these scales use (verified against
    `torch.exp2` on the host for exponents -40..20).
    """
    return torch.pow(torch.full_like(exponent, 2.0), exponent)


def act_quant(
    x: torch.Tensor, block: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-wise fp8 activation quantization, as the reference does it.

    `vendor/kernel.py`'s `act_quant` with `scale_fmt = "ue8m0"`: one scale per
    `block` consecutive values along the last axis, taken from that block's
    absolute maximum, **rounded up to a power of two** (that is what `ue8m0`
    means and what `fast_round_scale` computes), with the amax floored at 1e-4
    so an all-zero block does not divide by zero.

    Returns `(quantized, scale)` with `scale` broadcastable against
    `quantized`, so `quantized * scale` is the dequantized activation.
    """
    *lead, width = x.shape
    blocks = width // block
    if width % block:
        raise ValueError(f"act_quant needs a multiple of {block}, got {width}")
    grouped = x.detach().float().reshape(*lead, blocks, block)
    amax = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    # ceil(log2(amax / 448)) as an exponent, then 2 ** that.
    scale = _pow2(torch.ceil(torch.log2(amax / _FP8_MAX)))
    quantized = (grouped / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return (
        quantized.reshape(*lead, width),
        scale.expand(*lead, blocks, block).reshape(*lead, width),
    )


# =============================================================================
# Section 2b: every GEMM in this model, as a NKI kernel
# =============================================================================
# `kernels/blockquant.py` holds four shape-general NKI kernels — the fp8 and fp4
# block-quantized GEMMs, the fp8 one without the activation round trip, and the
# plain bf16 one. Between them they are every matrix multiply this architecture
# performs: the MLA projections, the grouped `wo_a`, the routed and shared
# experts, the engram read-out, the compressor, the indexer, the router and the
# unembedding. There is no torch equivalent on the execution path; `_reference`
# below exists for the accuracy-debugging comparison and nothing calls it.
#
# A NKI kernel cannot be invoked eagerly on this backend — the HOP has no
# PrivateUse1 dispatch —
#
#     NotImplementedError: could not find kernel for HigherOrderOperator
#     nki_kernel_wrapper at dispatch key DispatchKey.PrivateUse1
#
# so each one is reached through its own `torch.compile` entry point, which the
# `vllm_neuron` backend lowers to a NEFF. One entry point per kernel serves every
# shape: dynamo recompiles per (m, k, n) and the compilation cache under
# `~/.cache/neuron_libtorch` makes a repeat free.


#: Mirrors what `NeuronModelRunner.load_model` passes for the whole-model
#: compile, so a kernel reached from here is built the same way as one reached
#: from a single traced graph. The fp8 flag is not optional on Trn2: the kernels
#: write `nl.float8_e4m3fn`, whose legacy hardware form saturates at 240.
_COMPILE_OPTIONS = {
    "alias_meta_to_neuron": True,
    "compiler_args": [
        "--auto-cast=none",
        "-O1",
        "--internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10"
        " --experimental-unsafe-fp8e4m3fn-as-fp8e4m3",
        "--internal-backend-options=--enable-verifier=false"
        " --enable-nested-dynamic-loop",
    ],
}

def _raise_dynamo_limits() -> None:
    """Let dynamo compile as many graph variants as this model has.

    Three separate caps, and **the third one is the one that bites**:

      * `recompile_limit` (8) — per code object. `NeuronModelRunner.load_model`
        raises its old alias `cache_size_limit` for the same reason: this model
        has far more than 8 distinct linear shapes.
      * `cache_size_limit` — the pre-2.9 name of the same thing, kept in sync so
        it does not matter which one torch reads.
      * `accumulated_recompile_limit` (**256**) — the cap *across all code
        objects*, which the runner does not touch. 43 layers' worth of
        `_run_block` variants plus one entry point per kernel shape goes past
        256 during a single prefill, and when it does dynamo does not raise: it
        **permanently skips the frame**, silently, and from then on `_run_block`
        and `Attention.forward` run eagerly. The first symptom is far away from
        the cause — a decode step dies inside `apply_rotary_emb` with

            RuntimeError: Detected non-contiguous slicing for requested Device
            Tensor.

        because `q[..., -rd:]` is fine as an *interior* value of a graph and
        not fine as a graph *input*, which is what it becomes once the frame
        around it is no longer compiled.

    Raising a compile-count cap costs compile time and nothing else; none of
    these changes what the compiled code computes.
    """
    for name in (
        "recompile_limit",
        "cache_size_limit",
        "accumulated_recompile_limit",
        "accumulated_cache_size_limit",
    ):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, 1 << 30)


# =============================================================================
# Tensor-parallel collectives
# =============================================================================
# `GroupCoordinator.all_reduce` reaches `torch.distributed.all_reduce`, which on
# this stack only lowers while dynamo is tracing. Called from a region dynamo
# broke out of — `MoE.forward` is one, because it streams expert weights off
# disk mid-layer — it runs eagerly against a process group that has no backend
# for a Trainium tensor, and the whole step dies with
#
#     RuntimeError: No backend type associated with device type neuron
#
# at `vllm/distributed/parallel_state.py` -> `base_device_communicator` ->
# `torch.distributed.all_reduce`. Measured at TP=4: attention's reductions, which
# are inside `attn_half`'s graph, went through fine and the MoE's did not
# (`.autohelix/notes/iter-5.md` §5).
#
# The functional collectives do lower, because libtorch_neuronx_lite registers
# XLA implementations for them (`overrides/xla_collectives.py`, logged as "Custom
# XLA implementations registered via torch.library.impl"), and
# `vllm_neuron/nn/rpl.py` uses them for exactly this. Compiling the one-op
# wrapper keeps the collective inside a graph wherever it is called from.
#
# They are also *functional*: the result is returned, not written in place. The
# previous form discarded the return value of a `GroupCoordinator.all_reduce`
# that is out-of-place whenever `use_custom_op_call` is set, which would have
# silently dropped the reduction.


def _fc_sum(x: torch.Tensor, group_name: str) -> torch.Tensor:
    """`x` summed across `group_name`'s ranks."""
    out = torch.ops._c10d_functional.all_reduce(x, "sum", group_name)
    return torch.ops._c10d_functional.wait_tensor(out)


def _fc_gather_last(x: torch.Tensor, group_name: str) -> torch.Tensor:
    """Every rank's `x` concatenated along the last axis."""
    return _fc_all_gather_tensor(x, -1, group_name)


def _group_name(tp_group) -> str:
    """The functional-collective name of a `GroupCoordinator`'s device group."""
    return _fc_resolve_group_name(tp_group.device_group)


#: Largest all-reduce payload, in elements, handed to one collective call.
#:
#: **This is a host-memory lever, not a latency one, and it is the one piece of
#: the TP=4 fix that lives in this file.** The measurement is in
#: `.autohelix/hints.md` (2026-10-12 18:47): at four ranks the collectives'
#: host-side comm/proxy buffer is ~18.7 GiB *per rank*, ~112 GiB of system
#: commitment on a 124.8 GiB box, and that — not device HBM — is what wedged
#: the instance twice. `NEURON_RT_CC_ALG_TYPES=INTRA_SINGLE_STEP_MESH_ALG`
#: brings the whole 4-rank run down to ~30 GiB and the engine comes up, but the
#: runtime then asserts `allr_sz <= ch_buf_split_sz`: one all-reduce payload has
#: to fit one split of a channel buffer. `NEURON_RT_CC_CHANNEL_BUFFER_N` and
#: `NEURON_RT_CC_CHUNK_SIZE` do not lift that split, so the payload is what has
#: to come down — which is this constant.
#:
#: 262,144 elements is 1 MiB of fp32. The largest reduction this model performs
#: is the MoE's `[T, 5120]` fp32 at the 512-token prefill bucket, 10.5 MiB, so
#: that one becomes 11 collectives instead of 1; a decode step's `[6, 5120]` is
#: one either way. Each chunk is a separate collective over the same group, and
#: every rank derives the chunk count from the same shape, so the ranks stay in
#: lockstep without any extra synchronization.
#:
#: **Untested at four ranks** — iteration 8 had no budget for a 4-rank run (see
#: `.autohelix/notes/iter-8.md` §4) and this is dead code at TP=1, where
#: `world_size == 1` means no collective runs at all. If a 4-rank run still
#: trips the assert, lower this; the value that matters is the runtime's split
#: size, which it does not report.
_ALLREDUCE_CHUNK_ELEMS = 1 << 18


def tp_all_reduce(x: torch.Tensor, tp_group) -> torch.Tensor:
    """Sum `x` across the tensor-parallel ranks. Returns the reduced tensor.

    Chunked at `_ALLREDUCE_CHUNK_ELEMS`, for the host-memory reason documented
    there. The chunking is exact: a sum over ranks is elementwise, so splitting
    the payload and concatenating the reduced pieces is the same tensor,
    bit for bit, as reducing it whole.
    """
    name = _group_name(tp_group)
    if x.numel() <= _ALLREDUCE_CHUNK_ELEMS:
        return _compiled(_fc_sum)(x, name)
    return _compiled(_fc_sum_chunked)(x, name, _ALLREDUCE_CHUNK_ELEMS)


def _fc_sum_chunked(x: torch.Tensor, group_name: str, chunk: int) -> torch.Tensor:
    """`_fc_sum` over `x` in `chunk`-element pieces, as one compiled step.

    The splitting and the `cat` are *inside* the compiled region on purpose.
    `tp_all_reduce`'s callers include `MoE.forward`, which dynamo has broken out
    of, so a `reshape`/slice written in the caller would be an eager op on a
    Neuron tensor — the backend's fallback, which is where iteration 7 lost a
    run to `Expected self.is_contiguous() to be true` on exactly that pattern
    (iter-7 §2f). In a traced graph they are ordinary views.
    """
    flat = x.reshape(-1)
    parts = [
        _fc_sum(flat[lo : lo + chunk], group_name)
        for lo in range(0, flat.numel(), chunk)
    ]
    return torch.cat(parts).reshape(x.shape)


def tp_all_gather_last(x: torch.Tensor, tp_group) -> torch.Tensor:
    """Gather `x` across the tensor-parallel ranks on the last axis."""
    return _compiled(_fc_gather_last)(x, _group_name(tp_group))


@functools.lru_cache(maxsize=None)
def _nki_entry(module: str, name: str = "kernel"):
    """A compiled entry point for `kernels.<module>.<name>`.

    Cached per kernel, not per shape: dynamo keys its own code cache on the
    input shapes behind this one callable. `recompile_limit` has to be raised
    because the default is 8 and this model has more than 8 distinct linear
    shapes —

        torch._dynamo.exc.FailOnRecompileLimitHit: Hard failure due to
        fullgraph=True

    which is the same reason `NeuronModelRunner.load_model` raises
    `cache_size_limit`.
    """
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from vllm_neuron.envs import get_compile_backend_name

    # Static imports, not `importlib.import_module`: the first call for a given
    # kernel happens *while dynamo is tracing* the sublayer that uses it, and
    # dynamo refuses to trace into the import machinery —
    #
    #     torch._dynamo.exc.Unsupported: Attempted to call function marked as
    #     skipped ... `_gcd_import` in `<frozen importlib._bootstrap>`
    #
    # whereas an import statement it handles.
    from .kernels import (
        blockquant,
        engram_gate,
        hc_attn_out,
        hc_collapse,
        hc_expand,
        hc_mixes,
        index_score,
        rmsnorm,
        sparse_attn as sparse_attn_kernel,
    )

    _raise_dynamo_limits()
    modules = {
        "blockquant": blockquant,
        "engram_gate": engram_gate,
        "hc_attn_out": hc_attn_out,
        "hc_collapse": hc_collapse,
        "hc_expand": hc_expand,
        "hc_mixes": hc_mixes,
        "index_score": index_score,
        "rmsnorm": rmsnorm,
        "sparse_attn": sparse_attn_kernel,
    }
    wrapped = wrap_nki(getattr(modules[module], name))

    def entry(*args):
        return wrapped(*args)

    entry.__name__ = f"nki_{module}_{name}"
    return torch.compile(
        entry,
        backend=get_compile_backend_name(),
        fullgraph=True,
        options=_COMPILE_OPTIONS,
    )


def nki_topk(tensor: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-`k` along the last axis, through this repo's NKI top-k.

    `Tensor.topk` cannot be used on this backend: it lowers to an HLO `sort`
    that neuronx-cc rejects outright —

        %sort.50 = (f32[128,384], s32[128,384]) sort(...) metadata={
            op_type="aten__topk" op_name="Gate[G['gate']]/aten__topk" ...}
        FATAL: neuronx-cc compilation failed with 70.

    `vllm_neuron.functional.topk` exists for exactly this reason and picks
    `nkilib`'s rotational top-k whenever its shape envelope allows. Called with
    no process group: every top-k in this model (MoE routing, the indexer's
    `index_topk`, the candidate blocks) scores a replicated tensor, so there is
    nothing to gather across ranks.
    """
    from vllm_neuron.functional.topk import topk as _topk

    return _topk(tensor, k, -1, -1)


def nki_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    storage: str,
    quantize_activation: bool = True,
) -> torch.Tensor:
    """`x @ weight.T` on the device, through the matching NKI kernel.

    `x` is `[..., in_features]` of any leading shape and either bf16 or fp32;
    `weight` is in its *stored* form — `storage` says which: `"fp8"` e4m3 bytes
    and `"fp4"` packed nibbles with `scale` the e8m0 exponent bytes beside
    them, `"bf16"` as is, `"fp32w"` as is and narrowed inside the kernel. Everything this function
    does around the kernel call is a view — `reshape`, a dtype `view`, a row
    slice — because any real aten op here would be run by the backend's CPU
    fallback.
    """
    k = x.shape[-1]
    flat = x.reshape(-1, k)
    rows = flat.shape[0]
    if storage == "bf16":
        out = _nki_entry("blockquant", "bf16_linear")(flat, weight)
    elif storage == "fp32w":
        out = _nki_entry("blockquant", "fp32w_linear")(flat, weight)
    elif storage == "fp8":
        name = "fp8_linear" if quantize_activation else "fp8_linear_raw"
        out = _nki_entry("blockquant", name)(flat, weight, scale)
    elif storage == "fp4":
        out = _nki_entry("blockquant", "fp4_linear")(flat, weight, scale)
    else:
        raise ValueError(f"unknown storage {storage!r}")
    # The kernels return `ceil(rows/128)*128` rows; the pad rows are zeros.
    return out[:rows].reshape(*x.shape[:-1], out.shape[-1])


class BlockQuantLinear(nn.Module):
    """A linear whose weight is stored block-scaled, as the checkpoint has it.

    Three storage layouts appear, all with `e8m0` (power-of-two) scales:

        fp8  `[out, in]` e4m3 + `[ceil(out/32), ceil(in/32)]`   attention, shared experts
        fp4  `[out, in//2]` two e2m1 per byte + `[out, in/32]`   routed experts
        bf16 `[out, in]`, no scale                               compressor, indexer wk

    The weight is held in the stored form and stays that way: `forward` hands
    the stored bytes and their e8m0 scales to the matching NKI kernel in
    `kernels/blockquant.py`, which decodes them a tile at a time inside the
    kernel. `dequantized()` materializes the full bf16/fp32 weight and is used
    by `_reference` and by the shard-plan checks, not by `forward` — at 4x the
    memory it is exactly what a timeshared model cannot afford.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        storage: str = "fp8",
        dtype: torch.dtype = torch.bfloat16,
        block: int = 32,
        persistent: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.storage = storage
        self.block = block

        # Packed weights are held as raw `uint8`, not as `float8_e4m3fn` /
        # `int8`. The NKI kernels want the bytes, and a `view(torch.uint8)`
        # inside a traced graph is rejected when the graph runs ("Element at
        # index 1 is not an XLA tensor. Got: XLAByteType"), so the
        # reinterpretation happens on the host in `weight_loader.as_bytes`.
        if storage == "fp8":
            specs = [
                ("weight", (out_features, in_features), torch.uint8),
                (
                    "scale",
                    (-(-out_features // block), -(-in_features // block)),
                    torch.uint8,
                ),
            ]
        elif storage == "fp4":
            specs = [
                ("weight", (out_features, in_features // 2), torch.uint8),
                ("scale", (out_features, in_features // block), torch.uint8),
            ]
        elif storage == "bf16":
            specs = [("weight", (out_features, in_features), dtype)]
        else:
            raise ValueError(f"unknown storage {storage!r}")

        #: Kept so `allocate` can materialize a buffer that was built on the
        #: meta device. A staging slot (`persistent=False`) has no checkpoint
        #: key, so nothing assigns it during weight loading and it has to be
        #: allocated explicitly.
        self._specs = specs
        for name, shape, buffer_dtype in specs:
            self.register_buffer(
                name, torch.empty(shape, dtype=buffer_dtype), persistent=persistent
            )

    def allocate(self, device) -> None:
        """Materialize this linear's buffers on `device`."""
        for name, shape, buffer_dtype in self._specs:
            setattr(self, name, torch.zeros(shape, dtype=buffer_dtype, device=device))

    def dequantized(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """The weight in `dtype`.

        `dtype` matters and is not cosmetic: dequantizing to bf16 and *then*
        widening to fp32 for the GEMM rounds the weight twice, which the
        reference's fp8/fp4 GEMMs do not — they consume the stored values and
        fp32 scales directly. Asking for fp32 here is worth about half the
        residual error against the recorded traces.
        """
        from .weight_loader import dequant_block_fp4, dequant_block_fp8

        if self.storage == "fp8":
            return dequant_block_fp8(self.weight, self.scale, self.block, dtype)
        if self.storage == "fp4":
            return dequant_block_fp4(self.weight, self.scale, self.block, dtype)
        return self.weight.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x @ weight.T` in a NKI kernel. The only path; there is no fallback.

        The activation round trip the quantized kernels perform is the
        reference's, not an artifact: `linear()` in `vendor/model.py` runs
        `act_quant` before *both* the fp8 and the fp4 GEMM, and skipping it is
        worth 1.5e-2 of max relative error on a 384-expert MoE layer.
        """
        return nki_linear(
            x, self.weight, getattr(self, "scale", None), self.storage
        )

    def grouped(self, x: torch.Tensor, group: int, width: int) -> torch.Tensor:
        """One block of a block-diagonal projection: rows
        `[group*width, (group+1)*width)` against `x`.

        `wo_a` is block-diagonal over `o_groups`, so the stored `[groups*rank,
        in]` weight is really `groups` independent linears sharing a buffer.
        The row slice takes the matching 32-row band of the scale grid with it.
        The activation is *not* quantized here: the reference runs this
        projection against the dequantized weight with the activation untouched.
        """
        if self.storage != "fp8":
            raise ValueError(f"grouped() is for fp8 storage, not {self.storage!r}")
        if width % self.block:
            raise ValueError(f"group width {width} must be a multiple of {self.block}")
        band = width // self.block
        return nki_linear(
            x,
            self.weight[group * width : (group + 1) * width],
            self.scale[group * band : (group + 1) * band],
            "fp8",
            quantize_activation=False,
        )

    def _reference(self, x: torch.Tensor) -> torch.Tensor:
        """The torch emulation of `forward`, for accuracy comparison only.

        Nothing on the execution path calls this — `forward` goes straight to
        the kernel. It is kept because the accuracy-debugging guide's method is
        to compare a component against a reference implementation, and this is
        the one the recorded traces in `partition_artifact/trace/` were checked
        against (see `.autohelix/notes/iter-1.md` section 3). It differs from
        the kernel in one documented way: the kernels divide the activation by
        224 rather than 448 because Trn2's fp8 saturates at 240, which is worth
        about 3e-3 of max relative error.
        """
        if self.storage == "bf16":
            return F.linear(x, self.weight.to(x.dtype))
        x_q, x_scale = act_quant(x, self.block)
        # `x_q` is fp8; torch will not promote a Float8 type, so the
        # dequantization is explicit.
        activation = x_q.to(torch.float32) * x_scale
        weight = self.dequantized(torch.float32)
        return F.linear(activation, weight).to(x.dtype)


def _round_to_bits(x: torch.Tensor, mantissa: int, min_exp: int) -> torch.Tensor:
    """Round to a float format with `mantissa` explicit bits, arithmetically.

    A `Tensor.to(torch.float8_e4m3fn)` cannot be used inside a traced graph on
    this backend: the cast has no lowering and the graph dies when it runs, with

        FATAL: Unknown custom-call API version enum value: 0
        (API_VERSION_UNSPECIFIED)

    naming the node and nothing else. So the rounding is done the way the
    hardware does it — quantize to the step of the value's own binade, with
    `min_exp` holding the step flat through the subnormal range, and round half
    to even. Exact for every representable magnitude, including zero.
    """
    # The clamp is *before* the log, and that is the whole correctness of this
    # function on the device. `log2(0)` is -inf on the host and the old
    # `log2(x).clamp_min(min_exp)` was fine there, but on this backend it comes
    # out **NaN**, which then propagates through the quantization into the whole
    # attention output — a layer-2 attention came back `cos=nan` with nothing in
    # any log to say why, and layer 0 passed only because its KV latent happened
    # to contain no exact zeros. Clamping the magnitude to the smallest normal
    # first is exactly equivalent (anything below it rounds with the subnormal
    # step either way) and the log never sees a zero.
    magnitude = x.abs().clamp_min(2.0**min_exp)
    exponent = torch.floor(torch.log2(magnitude))
    step = _pow2(exponent - mantissa)
    return torch.round(x / step) * step


#: e4m3: 3 explicit mantissa bits, smallest normal 2**-6.
_FP8_MANTISSA_BITS, _FP8_MIN_EXP = 3, -6
#: e2m1: 1 explicit mantissa bit, smallest normal 2**0.
_FP4_MANTISSA_BITS, _FP4_MIN_EXP = 1, 0


def act_quant_dequant(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """`act_quant(..., inplace=True)`: round-trip `x` through fp8 and return it.

    The KV latent and the DSpark window KV go through this before being cached,
    so the cache holds values that have already lost fp8 precision. Skipping it
    would make attention read a *more* precise KV than the reference does.

    Same arithmetic as `act_quant`, written without a float8 cast so it can be
    traced — see `_round_to_bits`.
    """
    *lead, width = x.shape
    blocks = width // block
    if width % block:
        raise ValueError(f"act_quant needs a multiple of {block}, got {width}")
    grouped = x.float().reshape(*lead, blocks, block)
    amax = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = _pow2(torch.ceil(torch.log2(amax / _FP8_MAX)))
    normalized = (grouped / scale).clamp(-_FP8_MAX, _FP8_MAX)
    rounded = _round_to_bits(normalized, _FP8_MANTISSA_BITS, _FP8_MIN_EXP)
    return (rounded * scale).reshape(*lead, width).to(x.dtype)


def fp4_quant_dequant(
    x: torch.Tensor, block: int = 32, scale_e4m3: bool = False
) -> torch.Tensor:
    """`vendor/kernel.py`'s `fp4_act_quant(..., inplace=True)`: round-trip `x`
    through fp4 and return the dequantized result.

    Two call sites with two different scale formats, which is why the flag
    exists and is not a default: the indexer's keys and queries use 32-wide
    groups with `e8m0` (power-of-two) scales, while the compressed KV latent
    uses 16-wide groups with `e4m3` scales. Each has its own amax floor, taken
    from the kernel — 6 * 2**-9 for e4m3, 6 * 2**-126 for e8m0 — so an all-zero
    group keeps a nonzero scale instead of dividing by zero.
    """
    *lead, width = x.shape
    blocks = width // block
    if width % block:
        raise ValueError(f"fp4 quant needs a multiple of {block}, got {width}")
    grouped = x.float().reshape(*lead, blocks, block)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    if scale_e4m3:
        # The *scale* itself is stored e4m3 here, so it is rounded to e4m3
        # before it is used — rounded arithmetically, because a float8 cast has
        # no lowering in a traced graph (see `_round_to_bits`).
        amax = amax.clamp_min(6 * 2.0**-9)
        scale = _round_to_bits(amax / _FP4_MAX, _FP8_MANTISSA_BITS, _FP8_MIN_EXP)
    else:
        amax = amax.clamp_min(6 * 2.0**-126)
        scale = _pow2(torch.ceil(torch.log2(amax / _FP4_MAX)))
    normalized = (grouped / scale).clamp(-_FP4_MAX, _FP4_MAX)
    rounded = _round_to_bits(normalized, _FP4_MANTISSA_BITS, _FP4_MIN_EXP)
    return (rounded * scale).reshape(*lead, width).to(x.dtype)


#: fp4 `e2m1`'s maximum magnitude.
_FP4_MAX = 6.0

#: The 8 magnitudes `e2m1` can represent, and the midpoints between them, which
#: is what round-to-nearest needs.
_FP4_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_FP4_MIDPOINTS = (_FP4_MAGNITUDES[1:] + _FP4_MAGNITUDES[:-1]) / 2


def _round_to_fp4(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest `e2m1` value by table lookup, keeping the sign.

    The oracle `_round_to_bits(..., _FP4_MANTISSA_BITS, _FP4_MIN_EXP)` is
    checked against; `torch.bucketize` and `torch.copysign` are both untraceable
    on this backend, so this is not on the execution path.
    """
    midpoints = _FP4_MIDPOINTS.to(x.device)
    magnitudes = _FP4_MAGNITUDES.to(x.device)
    bucket = torch.bucketize(x.abs(), midpoints)
    return torch.copysign(magnitudes[bucket], x)


# =============================================================================
# Section 3: attention — latent KV, low-rank grouped output, two KV sources
# =============================================================================


def yarn_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotary frequencies as complex exponentials, one row per position.

    `vendor/model.py`'s `precompute_freqs_cis`. With `original_seq_len > 0` this
    is YaRN: a dimension whose wavelength already fits inside the training
    context keeps its frequency, one far beyond it is divided by `factor`, and
    the `beta_fast`..`beta_slow` band between them is faded across with a linear
    ramp.

    Which of the two regimes a layer is in is not a tuning choice: a layer that
    attends over compressed KV rotates at `compress_rope_theta` *with* YaRN,
    because one latent stands for `compress_ratio` tokens and its positions are
    that much further apart; a pure sliding-window layer has YaRN off and uses
    the base `rope_theta`.
    """
    import math

    freqs = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    if original_seq_len > 0:

        def corrected_dim(rotations: float) -> float:
            return (
                dim
                * math.log(original_seq_len / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = (
            (torch.arange(dim // 2, dtype=torch.float32, device=device) - low)
            / max(high - low, 1e-3)
        ).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    angles = torch.outer(torch.arange(seqlen, device=device), freqs)
    # `(cos, sin)` rather than the reference's `torch.polar` complex tensor.
    # Same rotation, real arithmetic — see `apply_rotary_emb`. neuronx-cc has no
    # complex dtype, so `torch.polar` / `view_as_complex` cannot be traced into
    # a NEFF; keeping the table real is what lets the whole RoPE run on device.
    return torch.cos(angles), torch.sin(angles)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    """Rotate `x`, taking **adjacent element pairs** as complex numbers.

    Accepts `[b, s, d]` and `[b, s, h, d]`. `inverse` conjugates the rotation,
    which is how the attention *output* gets the query's rotation removed again
    — that is what lets one shared KV cache stay in a single rotated form.

    Written with real multiplies instead of the reference's complex ones: for a
    pair `(a, b)` at angle `t`, `(a + ib)(cos t + i sin t)` is
    `(a cos t - b sin t, a sin t + b cos t)`, and conjugating flips the two sin
    terms. Note the *adjacent-pair* convention — this is not the `rotate_half`
    split that Llama and Qwen3 use, and swapping them silently produces
    plausible-looking garbage.
    """
    cos, sin = freqs_cis
    if inverse:
        sin = -sin
    pairs = x.float().unflatten(-1, (-1, 2))
    a, b = pairs[..., 0], pairs[..., 1]
    # `cos`/`sin` are [s, d/2]; `a`/`b` are [b, s, d/2] or [b, s, h, d/2].
    if a.ndim == 3:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    else:
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
    rotated = torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1)
    return rotated.flatten(-2).to(x.dtype)


def slice_freqs(
    freqs_cis: tuple[torch.Tensor, torch.Tensor], index
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take the same positions out of both halves of a `(cos, sin)` table."""
    cos, sin = freqs_cis
    return cos[index], sin[index]


def span_freqs(
    freqs_cis: tuple[torch.Tensor, torch.Tensor],
    start_pos: torch.Tensor,
    length: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`length` rows of the RoPE table starting at `start_pos`, gathered.

    A gather with a *tensor* index rather than `cos[start_pos : start_pos + n]`,
    and the difference is not cosmetic: a slice bakes its offset into the graph,
    so a decode step at a new position is a different FX graph and misses the
    compile cache — ~45 recompiles per token, for graphs whose shapes are all
    identical. With the positions as a tensor the graph is the same at every
    position and only the data changes.

    `start_pos` is the `[1]` device tensor `PagedSpan.pos` carries, **not** a
    Python int, for the same reason one step further on: an int would be folded
    into the graph as `operator.add(167, %arange)` and the graph at position 168
    would again be a different one. See `PagedSpan.pos`.

    Numerically identical to the slice: `index_select` of consecutive indices.
    """
    positions = start_pos + torch.arange(length, device=device)
    return slice_freqs(freqs_cis, positions)


#: An empty top-k slot: a window position before the sequence started, or a
#: compressed position a query cannot reach yet. `sparse_attn` skips it.
EMPTY_SLOT = -1

#: How many queries share one gather. See `sparse_attn`.
#:
#: **This is a compile-time constant, not an accuracy one, and 512 rather than
#: 128 because of what the compiler does with the unrolled loop.** The loop in
#: `sparse_attn` is a Python `range`, so dynamo unrolls it into the graph: at a
#: 128-wide tile an 8192-token prefill segment is 64 copies of
#: gather/mask/kernel-call, which is a 1330-line FX graph with 155 kernel calls,
#: and `neuronx-cc`'s `Tensorizer/MaskPropagation` pass took **>16 minutes**
#: on one of them, single-threaded, with 39 more to go
#: (`.autohelix/notes/iter-5.md` §10c). At 512 it is 16 copies.
#:
#: It cannot change a digit of the output: each tile is a separate call with
#: its own softmax over its own query rows' slots, every reduction inside
#: `kernels/sparse_attn.gathered_attn` is per token, and the kernel derives
#: `heads = q.shape[0] // bias.shape[0]` so it takes the tile width from its
#: arguments and asserts nothing about it.
_ATTN_QUERY_TILE = 512

#: The slot-list granularity `kernels/sparse_attn` works in (its partition
#: width). The caller pads the top-k list up to a multiple of this with empty
#: slots, which costs arithmetic but nothing in accuracy.
_ATTN_SLOT_TILE = 128

#: Bucket for the compressed-KV and index-key reads. `compress_len` grows with
#: the decode position and is a tensor *shape*, so reading exactly that many
#: groups makes every layer recompile once per token. Rounding the read up to a
#: multiple of this changes the shape only on bucket boundaries — 128 matches
#: `kernels/blockquant`'s partition width, so the padded groups cost nothing the
#: kernel was not already going to spend.
#:
#: The padded groups are **not** attended to: `Indexer.forward` masks their
#: score to -1e30 and maps any pick at or beyond `compress_len` to
#: `EMPTY_SLOT`, which `kernels/sparse_attn` removes from the softmax.
_COMPRESS_TILE = 128

#: Above this many KV slots, `_cap_kv_cache_blocks` leaves vLLM's memory-based
#: sizing alone — a backstop against a configuration so large that the cap's own
#: arithmetic is the thing that is wrong.
#:
#: The number it guards has to cover the verifier's own defaults, which are
#: `max_model_len=8224` at `max_num_seqs=4` and so want
#: `(8224/16 + 1) * 4 * 2 = 4,120` blocks, i.e. 65,920 slots. At 32,768 the cap
#: silently did not apply there, and vLLM then sized the cache from free memory
#: — ~15,000 blocks, ~12 GiB of window KV — on top of which this port allocates
#: its own per-slot buffers (compressed KV, index keys, the compressor's pooling
#: inputs, the n-gram history) that vLLM's accounting cannot see, because they
#: are not in `get_kv_spec`. That overruns the bank.
#:
#: Honest about the tradeoff, since `num_gpu_blocks_override` replaces vLLM's
#: figure in *either* direction: what the cap computes is the number of slots
#: the configuration can address, so it can only ever cut a cache that was
#: larger than anything the scheduler could fill. If HBM were small enough that
#: vLLM's own figure came out *below* that ceiling, this would raise it and the
#: failure would move from "cache smaller than max_model_len" to an allocation
#: error. That is a configuration this port cannot serve either way: every cache
#: it keeps — window ring, compressed groups, index keys, pooling inputs — is
#: addressed by absolute position within a sequence's own context, so a sequence
#: needs its whole context resident.
_KV_CAP_MAX_SLOTS = 131072


def _hbm_used_bytes() -> int | None:
    """Device HBM in use, straight from the Neuron runtime's own counters.

    `NeuronWorker._query_runtime_memory_stats` makes exactly this call; it is
    read here so a forward can account for its own allocations per layer rather
    than inferring them from `/tmp/neuron_mem_table_device_*.log`, which is
    written only when an allocation *fails* and carries one number per core.

    The obvious alternatives are both unavailable in this venv, which is worth
    recording because it cost an iteration's planning:
    `libtorch_neuronx_lite._compiler` exposes `empty_cache()` and
    `get_device_properties()`, but its extension fails to import here —
    ``undefined symbol: PyObject_ClearManagedDict`` — and it is the only handle
    on a caching allocator. The execution path does not go through it:
    `lib/libtorchneuron.so` registers `aten::empty` for the neuron device
    directly, so a device tensor is an `nrt_tensor_allocate` and dropping the
    last reference frees it. There is nothing to empty, and these counters are
    the live set.

    Returns None rather than raising: this is instrumentation, and a forward
    must not fail because a counter was unreadable.
    """
    global _HBM_RUNTIME
    try:
        if _HBM_RUNTIME is None:
            _HBM_RUNTIME = torch.classes.neuron.Runtime()
        used, _free = _HBM_RUNTIME.get_vnc_memory_stats()
        return int(used)
    except Exception:  # noqa: BLE001 - instrumentation, never fatal
        return None


#: Cached `torch.classes.neuron.Runtime` handle for `_hbm_used_bytes`.
_HBM_RUNTIME = None

#: How many forwards log their HBM use per layer. After that each forward logs
#: one summary line. The per-layer detail is what identifies *which* step grows
#: the live set; the summary is what shows whether it comes back down.
_HBM_TRACE_FORWARDS = 6

#: `[steps seen, previous reading, previous layer id]` for `_hbm_mark`.
_HBM_STATE = [0, None, -1]

#: Rows below which `_collect_layer_garbage` is not worth its own cost. A
#: decode step's whole stream is a few hundred KiB; a collection costs tens of
#: milliseconds and would be paid 43 times per generated token.
_GC_MIN_ROWS = 64

#: Layers at which `_census_device_tensors` reports. Early, middle and late, so
#: the difference between them is what the layer loop accumulates.
_CENSUS_LAYERS = frozenset({4, 20, 39})


def _collect_layer_garbage() -> None:
    """Free the previous layer's device tensors, including the ones in cycles.

    **This is the fix for the wall every iteration since 3 has hit.** A layer's
    intermediates are dead the moment `_run_block` returns — the stream and the
    `pre_mix` are rebound, everything else goes out of scope — but a measured
    ~90 KiB *per token per layer* was not coming back:

        HBM  11.105 GiB (     +0.0 MiB) at layer 0 of forward 0, 512 rows
        HBM  11.648 GiB (    +46.8 MiB) at layer 4 of forward 0, 512 rows
        HBM  13.430 GiB (    +55.7 MiB) at layer 39 of forward 0, 512 rows

    — +2.33 GiB over 40 layers at a 512-token bucket, monotonic, never
    released. Scaled to the verifier's 9328-token prompt that is **~37 GiB on a
    24 GiB bank**, and it is why the wall did not move when the KV cache halved,
    when tensor parallelism quartered the weights, or when the prefill segment
    halved (`.autohelix/notes/iter-5.md` §6 measured all three): the per-layer
    footprint is proportional to `layers x tokens`, so chunking the prefill
    moves it between steps without reducing it, and every run died at whatever
    was left of the bank.

    Refcounting alone does not collect it because the frames that held those
    tensors are reachable from reference *cycles* — a graph break leaves dynamo
    holding frame state, and a caught exception's traceback holds its frame,
    both of which are cyclic. The tensors are garbage; nothing had asked for
    them to be collected.

    So this asks, once per layer, from host code the compiled graphs have
    already been broken out of. `nrt_tensor_free` really does return the memory:
    the Lite backend registers `aten::empty` for the neuron device directly and
    has no caching allocator in the way (`_hbm_used_bytes`).

    It is not free — a full collection over this process's heap is tens of
    milliseconds — so it is skipped for the small shapes where there is nothing
    worth collecting. A decode step moves ~90 KiB per layer; a prefill segment
    moves hundreds of MiB.
    """
    gc.collect()


def _census_device_tensors(layer_id: int) -> None:
    """Log what is alive on the device, by shape, biggest first.

    The counterpart to `_hbm_used_bytes`: that says how much is held, this says
    what is holding it. Walks the GC's object graph, which every `torch.Tensor`
    is tracked in, so it sees tensors reachable from anywhere — a module
    buffer, a dynamo cache entry, a compiled graph's constants — without
    needing to guess which.

    Expensive (one full heap walk), so it is called at a handful of layers
    rather than all of them.
    """
    sizes: dict[tuple, list[int]] = {}
    for obj in gc.get_objects():
        try:
            if not isinstance(obj, torch.Tensor) or obj.device.type != "neuron":
                continue
            key = (tuple(obj.shape), str(obj.dtype).replace("torch.", ""))
        except Exception:  # noqa: BLE001 - a half-built object, skip it
            continue
        entry = sizes.setdefault(key, [0, 0])
        entry[0] += 1
        entry[1] += obj.numel() * obj.element_size()
    ranked = sorted(sizes.items(), key=lambda kv: -kv[1][1])[:10]
    total = sum(v[1] for v in sizes.values())
    logger.info(
        "deepseek_v41_timeshare: live device tensors at layer %d: %.3f GiB in "
        "%d shapes", layer_id, total / 1024**3, len(sizes),
    )
    for (shape, dtype), (count, nbytes) in ranked:
        logger.info(
            "deepseek_v41_timeshare:   %8.1f MiB  x%-5d %s %s",
            nbytes / 1024**2, count, dtype, shape,
        )


def _hbm_mark(layer_id: int, rows: int) -> None:
    """Log HBM use at one layer of a forward, with the delta since the last.

    A layer id that does not advance means a new forward started, which is how
    the step count is kept without a call in `forward` itself — see
    `MoE._route_plan` for why there must not be one.
    """
    if layer_id <= _HBM_STATE[2]:
        _HBM_STATE[0] += 1
        _HBM_STATE[1] = None
    _HBM_STATE[2] = layer_id
    if _HBM_STATE[0] >= _HBM_TRACE_FORWARDS:
        return
    used = _hbm_used_bytes()
    if used is None:
        return
    previous = _HBM_STATE[1]
    _HBM_STATE[1] = used
    logger.info(
        "deepseek_v41_timeshare: HBM %7.3f GiB (%+9.1f MiB) at layer %d of "
        "forward %d, %d rows",
        used / 1024**3,
        0.0 if previous is None else (used - previous) / 1024**2,
        layer_id, _HBM_STATE[0], rows,
    )


def _compress_read_len(compress_len: int) -> int:
    """How many compressed groups to read for `compress_len` live ones."""
    if compress_len <= 0:
        return 0
    return -(-compress_len // _COMPRESS_TILE) * _COMPRESS_TILE

#: The finite lower bound the reference kernel starts its running max at. A row
#: whose every slot is empty would give exp(-inf - -inf) = NaN with a true -inf;
#: with this bound the sink term overflows to +inf and the row comes out zero,
#: which is the kernel's convention.
#:
#: Every "unreachable" sentinel in this file uses this and `_POS_BOUND` rather
#: than a true infinity, for two reasons beyond the NaN above. A
#: `masked_fill(..., -inf)` has no lowering on this backend (the graph aborts
#: the interpreter when it runs), and an infinity reaching
#: `vllm_neuron.functional.topk`'s rotational NKI kernel — whose pad sentinel is
#: the dtype minimum — came back as NaN from the whole of a layer-2 attention.
#: Finite bounds are also what `vendor/kernel.py` itself uses.
_NEG_BOUND = -1e30

#: `_NEG_BOUND`'s mirror, for pinning a block *in* to a top-k.
_POS_BOUND = 1e30


def _bound(x: torch.Tensor, value: float) -> torch.Tensor:
    """`value` as a 0-d tensor in `x`'s dtype.

    A Python float handed to `masked_fill`, `F.pad(value=...)` or a comparison
    becomes an **f64** device constant, which traces, lowers, and then kills the
    compile in a pass that names no line:

        [ERROR] [NCC_ESPP004] f64 dtype is not supported.
        RuntimeError: neuronx-cc compilation failed with 70.

    Confirmed by parsing the saved HLO: `xla__device_data` f64 constants, one
    per site, with a `convert` and a `broadcast` on top. (The scalar *arithmetic*
    ops — `tensor * 1e30` — are fine; it is the ones that take a `Scalar`
    argument.) `new_full` takes the dtype from the tensor, so the constant is
    born fp32 and nothing has to be converted.
    """
    return x.new_full((), value)


def window_topk_idxs(
    window_size: int, seqlen: int, start_pos: torch.Tensor, device
) -> torch.Tensor:
    """Which rows of the gathered window each query attends to; -1 is empty.

    `vendor/model.py`'s `get_window_topk_idxs`, rewritten for the fixed-size
    paged read in `Attention._window_kv`. That read hands back the
    `window + seqlen - 1` positions ending at `start_pos + seqlen - 1`, i.e.
    starting at `base = start_pos - window + 1`, which is negative early in a
    sequence. A query at offset `t` within the span (absolute position
    `start_pos + t`) sees the `window` rows `t .. t + window - 1`, and a row
    whose underlying position is negative is marked empty.

    **The row count does not depend on `start_pos`**, which is the point: it is
    a tensor *shape*, and a shape that grows with the decode position makes
    dynamo recompile the whole layer once per token. Masking the rows before
    the sequence started costs nothing — `kernels/sparse_attn` gives an empty
    slot a -1e30 bias, so it leaves the softmax.

    `start_pos` is `PagedSpan.pos`, a `[1]` device tensor, so neither the shape
    *nor the graph* depends on the decode position — see `span_freqs`.

    Causality is unchanged: query `start_pos + t` names positions
    `start_pos + t - window + 1 .. start_pos + t`, never a later one. Order
    within a row does not matter to `sparse_attn`, which treats every slot
    independently.
    """
    base = start_pos - (window_size - 1)
    rows = torch.arange(seqlen, device=device).unsqueeze(1) + torch.arange(
        window_size, device=device
    )
    idxs = torch.where(base + rows < 0, EMPTY_SLOT, rows)
    return idxs.int().unsqueeze(0)


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Multi-head attention over gathered slots, with a learned sink.

    `q` is `[b, s, h, d]`; `kv` is `[b, n, d]` and shared by every head (there
    is one latent KV, not one per head); `topk_idxs` is `[b, s, topk]` naming
    the slots each query may see. Semantics are `vendor/kernel.py`'s
    `sparse_attn`, transcribed from the torch equivalent the partition artifact
    ships as `compat/sparse_attn_sm89.py`.

    **The query axis is tiled, and it has to be.** The gather is per query, so
    one `kv[batch, topk_idxs]` over a whole prefill materializes
    `[1, 8192, 640, 512]` bf16 = 5.4 GiB of activations for a single layer —
    which no device here has room for, and which earlier component checks never
    hit because they only ever ran 128-token prompts. Tiling bounds it at 84 MiB
    and changes nothing numerically: every tile is an independent set of rows
    with its own softmax.

    **The arithmetic is `kernels/sparse_attn.gathered_attn`** — the QK score,
    the masked softmax with the per-head sink, and the probability-weighted sum.
    What is left here is the gather itself (`kv[batch, idxs]`, data movement),
    the comparison that builds the mask, and the padding of the slot list to the
    kernel's 128-slot granularity. Padding with *empty* slots is free, because
    an empty slot's bias removes it from the softmax.

    Where the precision is deliberate — see the kernel's module docstring for
    which instruction does which:

      - the QK product takes bf16 operands into an fp32 accumulator, and the
        scale multiplies the *logits* afterwards rather than `q` beforehand, so
        it never passes through a bf16 rounding;
      - the probabilities are rounded to bf16 before the PV product, matching
        the reference kernel's bf16 staging buffer, while the denominator sums
        the fp32 ones;
      - the sink joins the denominator against the final row max.
    """
    b, s, h, d = q.shape
    n = kv.size(1)
    topk = topk_idxs.size(-1)
    if topk == 0 or n == 0:
        return torch.zeros_like(q)

    # The per-tile results are collected and concatenated rather than written
    # into slices of one `empty_like(q)`, and that is a compiler decision.
    # `out[:, lo:hi] = ...` over an uninitialized buffer is the one pattern
    # `neuronx-cc`'s integer-set analysis is worst at: proving a set of partial
    # writes covers a tensor is `IntegerSetAnalysis.cover_whole_tensor`, which
    # is both where the `islpy` version mismatch aborted the compile outright
    # (`__init__.warn_on_unsupported_islpy`) and where `Tensorizer/
    # MaskPropagation` spent **>12 minutes per graph** on this model's prefill
    # shapes (`.autohelix/notes/iter-5.md` §10c). A `cat` is one op with
    # nothing to prove.
    #
    # Numerically identical: the tiles partition `range(0, s)` exactly, each
    # one is a separate kernel call over its own query rows, and concatenating
    # them in order is the same tensor the slice writes produced.
    tiles: list[torch.Tensor] = []

    kpad = -(-topk // _ATTN_SLOT_TILE) * _ATTN_SLOT_TILE
    sink = attn_sink.float().reshape(h, 1)
    scale = q.new_full((1, 1), softmax_scale, dtype=torch.float32)
    batch = torch.arange(b, device=kv.device).view(b, 1, 1)
    for lo in range(0, s, _ATTN_QUERY_TILE):
        hi = min(lo + _ATTN_QUERY_TILE, s)
        t = hi - lo
        idxs = topk_idxs[:, lo:hi].long()
        empty = idxs == EMPTY_SLOT

        gathered = kv[batch, idxs.clamp_min(0)]
        gathered = gathered.masked_fill(empty.unsqueeze(-1), 0)
        # `empty.float() * _NEG_BOUND`, not `torch.where(empty, _NEG_BOUND,
        # 0.0)`. A `where` whose both branches are Python scalars has no tensor
        # operand to take a dtype from, so it comes out **f64** — which traces
        # and lowers and then fails the compile:
        #
        #     [ERROR] [NCC_ESPP004] f64 dtype is not supported.
        #     RuntimeError: neuronx-cc compilation failed with 70.
        #
        # and the `.float()` after it is too late, because the f64 node is
        # already in the graph. The multiply anchors the dtype on a tensor and
        # is exact: a live slot gives `0.0 * -1e30`, an empty one `-1e30`.
        bias = empty.float() * _NEG_BOUND
        if kpad != topk:
            # Both widths the model actually produces are multiples of 128 —
            # `sliding_window` and `sliding_window + index_topk` — so this
            # branch is for a config that changes one of them.
            gathered = torch.cat(
                [gathered, gathered.new_zeros(b, t, kpad - topk, d)], dim=2
            )
            bias = torch.cat(
                [bias, bias.new_full((b, t, kpad - topk), _NEG_BOUND)], dim=2
            )

        tiles.append(
            _nki_entry("sparse_attn", "gathered_attn")(
                q[:, lo:hi].reshape(-1, d),
                gathered.reshape(-1, d),
                bias.reshape(-1, kpad),
                sink,
                scale,
            ).reshape(b, t, h, d)
        )
    return tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=1)


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: torch.Tensor | int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of the two-level top-k: keep the best `topk_blocks` blocks.

    `logits` has unreachable positions already at -inf, which is what makes a
    block score of -inf mean "not reachable yet". The block holding the query's
    newest position is pinned in: it is only partly filled, so it could be
    outscored by an older full block despite holding the most recent tokens.
    """
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=_NEG_BOUND)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last,
        _bound(scores, _POS_BOUND),
    )
    top_values, top_indices = nki_topk(scores, min(topk_blocks, num_blocks))
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top_indices, top_values > _bound(top_values, _NEG_BOUND)
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class PagedSpan:
    """Where one scheduled sequence's KV lives in vLLM's paged allocation.

    The runtime hands the model `slot_mapping` for the tokens of this step and
    a `block_table` row per sequence; a slot is
    `block_id * block_size + offset`, which is also the row index of
    `cache.view(-1, head_size)`. Everything this port caches per position —
    the sliding window, the compressed latents, the index keys and the
    compressor's per-token pooling inputs — is addressed through `slots()`, so
    there is no per-request bookkeeping anywhere and the content lives in
    blocks that prefix caching already tracks. A private contiguous ring would
    be simpler and would silently serve stale KV on a prefix-cache hit.

    `block_table` is one *row* of `block_table_tensor`: row `k` is the `k`-th
    scheduled sequence, which is the `k`-th span from `_sequence_spans`.

    ── Why this object also carries the position, in two forms ──────────────

    This is the single most important thing about the decode path's cost, so it
    is written here rather than at the six call sites that depend on it.

    `start_pos` is a Python int, and dynamo **specializes** on Python ints: a
    graph traced at position 167 carries an `EQUALS_MATCH` guard for 167 and the
    arithmetic folds in as a constant,

        %positions : = call_function[target=operator.add](args = (167, %arange))

    so position 168 is a different FX graph. Measured: a decode step recompiled
    4 graphs *per step, forever* — 79 distinct cache keys over 18 steps, none of
    them ever reused — and since every loaded NEFF reserves DMA rings, that was
    also an unbounded HBM leak. Nothing about the shapes was changing; only the
    constant.

    So the position reaches a compiled graph in exactly two forms, and the split
    is the whole trick:

    * **`pos`** — a `[1]` int64 device tensor, for every use of the position as
      a *value* (RoPE offsets, window bases, which compressed groups are
      reachable). A graph input, so the graph is identical at every position.
    * **`read_lens` / `group_counts`** — Python ints, for the uses that are
      tensor *shapes* and so cannot be tensors. Both are bucketed
      (`_compress_read_len`) or tiny (`group_counts` is 0 or 1 during decode), so
      dynamo's guard on them holds for many consecutive steps instead of one.

    Indexed by `compress_ratio`, which is 0, 1 or 2 (`config.compress_ratios`),
    with slot 0 unused — a tuple index with a constant is the form dynamo
    handles most predictably, and it keeps `Attention` from having to know
    anything about how the plan was built.

    `start_pos` itself stays for the host-side callers (`_engram_hashes`,
    `forward_spec`) and must **not** be read from inside a compiled region.
    """

    __slots__ = (
        "block_table",
        "block_size",
        "slot_mapping",
        "start_pos",
        "length",
        "begin",
        "pos",
        "at_zero",
        "read_lens",
        "group_counts",
    )

    def __init__(
        self,
        block_table: torch.Tensor,
        block_size: int,
        slot_mapping: torch.Tensor,
        start_pos: int,
        length: int,
        pos: torch.Tensor | None = None,
        begin: int = 0,
    ) -> None:
        self.block_table = block_table
        self.block_size = block_size
        self.slot_mapping = slot_mapping
        self.start_pos = start_pos
        self.length = length
        #: Where this sequence's tokens start in the flat `[T]` token axis.
        self.begin = begin
        self.at_zero = start_pos == 0
        self.pos = (
            pos
            if pos is not None
            else torch.tensor(
                [start_pos], dtype=torch.int64, device=slot_mapping.device
            )
        )
        # Per-ratio shape plan. `compress_len` is how many compressed groups
        # exist once this span's tokens are written; `read_len` rounds it up to
        # a `_COMPRESS_TILE` bucket, and `group_counts` is how many groups this
        # span *completes* (the compressor's output row count).
        read_lens, counts = [0, 0, 0], [0, 0, 0]
        for ratio in (1, 2):
            compress_len = (start_pos + length) // ratio
            read_lens[ratio] = _compress_read_len(compress_len)
            counts[ratio] = compress_len - start_pos // ratio
        self.read_lens = tuple(read_lens)
        self.group_counts = tuple(counts)

    def slots(self, positions: torch.Tensor) -> torch.Tensor:
        """Flat cache rows for absolute `positions` in this sequence."""
        return (
            self.block_table[positions // self.block_size] * self.block_size
            + positions % self.block_size
        )

    def write(self, cache: torch.Tensor, values: torch.Tensor) -> None:
        """Write this step's tokens into a `[num_blocks, 1, block_size, dim]`
        cache at the runtime's own slots.

        `index_put_` with a tuple of index tensors, which is the pattern
        `qwen3/model.py` uses and which lowers on this backend; a flat
        `view(-1, dim)[slots] = values` does not survive functionalization as
        an in-place write to the real allocation.
        """
        slot = self.slot_mapping
        cache.index_put_(
            (
                slot // self.block_size,
                torch.zeros_like(slot),
                slot % self.block_size,
            ),
            values,
        )

    def write_flat(self, flat: torch.Tensor, values: torch.Tensor) -> None:
        """Write this step's tokens into a flat `[num_slots, dim]` buffer."""
        flat.index_copy_(0, self.slot_mapping, values)

    def put(
        self, flat: torch.Tensor, positions: torch.Tensor, values: torch.Tensor
    ) -> None:
        """Write `values` into a flat `[num_slots, dim]` buffer at the slots of
        arbitrary absolute `positions` — used for the compressed latents, which
        land at the slot of their group's first token rather than at this
        step's own tokens.
        """
        flat.index_copy_(0, self.slots(positions), values)

    def gather(self, flat: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rows of a flat `[num_slots, dim]` cache for absolute `positions`."""
        return flat.index_select(0, self.slots(positions))


class SharedAttentionState:
    """What attention layers hand down the stack instead of recomputing.

    `vendor/model.py`'s `SharedAttentionRuntime`. Layers run in order and every
    source writes before its consumers read, so one slot each is enough *per
    sequence*: `compress_kv` and `index_k` come from the four
    `kv_source_layer_ids`, `topk_idxs` from the eight `index_source_layer_ids`,
    and `candidates` from the single `candidate_source_layer_id`.

    **One instance per span, never one per batch.** `Block.attn_half` runs the
    sequence loop, so a shared instance has each span overwriting the previous
    one's published index list — and because the published width is derived
    from the span's own position, the consumer then gathers past the end of its
    own window. `forward` builds the list; `.autohelix/notes/iter-4.md` §2a has
    the device error this produced.
    """

    __slots__ = ("compress_kv", "index_k", "topk_idxs", "candidates")

    def __init__(self) -> None:
        self.compress_kv: torch.Tensor | None = None
        self.index_k: torch.Tensor | None = None
        self.topk_idxs: torch.Tensor | None = None
        self.candidates: torch.Tensor | None = None


class Compressor(nn.Module):
    """Pools `compress_ratio` consecutive tokens into one KV latent.

    `vendor/model.py`'s `Compressor`. Above ratio 1 the pooling is a learned
    softmax gate and runs in fp32, which is why `wgate` exists only there; at
    ratio 1 it is a plain projection and the checkpoint has no `wgate` key for
    that layer — a difference the weight mapping already encodes
    (`_attention_keys`).

    Returns the latent *before* RoPE: the indexer needs the unrotated form, so
    `Attention` rotates afterwards.
    """

    def __init__(
        self, config: DeepseekV41Config, layer_id: int, max_batch_size: int = 1
    ) -> None:
        super().__init__()
        del max_batch_size  # the pooling inputs are paged, not per batch row
        self.compress_ratio = config.compress_ratios[layer_id]
        self.head_dim = config.head_dim
        self.norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.wkv = BlockQuantLinear(config.hidden_size, config.head_dim, storage="bf16")
        if self.compress_ratio > 1:
            self.wgate = BlockQuantLinear(
                config.hidden_size, config.head_dim, storage="bf16"
            )
        # Per-token pooling inputs, one row per cache slot. A group may straddle
        # a chunk boundary or two decode steps, so the members have to outlive
        # the call that produced them — and keying them by batch row would be
        # wrong the moment the scheduler reorders the running batch or two
        # requests share a cached prefix. Paged by the token's own slot, the
        # question does not arise. `persistent=False`: nothing in the
        # checkpoint corresponds to them, and `allocate_state` fills them
        # because a module built under `torch.device("meta")` cannot.
        self.register_buffer("pool_kv", None, persistent=False)
        self.register_buffer("pool_score", None, persistent=False)

    def allocate_state(self, device, num_slots: int) -> None:
        """Allocate the per-slot pooling inputs.

        **bf16, not fp32, and that is lossless here**: both buffers hold the
        output of a bf16 GEMM (`wkv`/`wgate` are stored bf16 and
        `kernels/blockquant.bf16_linear` returns bf16), so widening on the way
        in would store zeros in the low 16 bits. The pooling arithmetic that
        consumes them still runs in fp32 — `forward` casts on the read, which is
        where the reference's fp32 pooling actually lives.

        It halves this state from ~940 MiB to ~470 MiB at the slot count vLLM
        allocates, and HBM is the binding constraint on this device (see
        `.autohelix/notes/iter-3.md` §5b-ter).
        """
        if self.compress_ratio > 1:
            self.pool_kv = torch.zeros(
                num_slots, self.head_dim, dtype=torch.bfloat16, device=device
            )
            self.pool_score = torch.zeros(
                num_slots, self.head_dim, dtype=torch.bfloat16, device=device
            )

    def forward(
        self, x: torch.Tensor, start_pos: torch.Tensor, paged: PagedSpan
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Pool `compress_ratio` consecutive tokens into one latent.

        Returns `(latent, first_positions)` where `first_positions` holds the
        absolute position of each returned group's *first* token — which is
        also the position the latent stands for, so the caller's RoPE is just
        `slice_freqs(freqs_cis, first_positions)` with no prefill/decode split.
        Returns `(None, None)` when this call completes no group, which during
        decode is every step but the last of a group. Callers must read that as
        "nothing to publish yet", not "no compressed KV": the cache still holds
        every earlier group.

        Stateless. The reference carries an incomplete group in `kv_state` /
        `score_state`; here the per-token inputs go to `pool_kv`/`pool_score` at
        the token's own slot, so a group that straddles two calls is pooled from
        the cache rather than from carried state.
        """
        _, seqlen, _ = x.shape
        ratio, dtype = self.compress_ratio, x.dtype
        device = x.device
        if ratio == 1:
            # One token per group: nothing to pool, so no gate and no fp32.
            positions = start_pos + torch.arange(seqlen, device=device)
            return self.norm(self.wkv(x)), positions

        # The GEMM output goes in as it comes out — bf16, which is what the
        # buffers are (`allocate_state` says why that is lossless). `write_flat`
        # is `index_copy_`, which is strict about dtype:
        #
        #     RuntimeError: index_copy_(): self and source expected to have the
        #     same dtype, but got (self) Float and (source) BFloat16
        #
        # XLA tolerates the mismatch by converting; CPU does not, so matching
        # them keeps the component harnesses runnable on the host.
        xf = x.float()
        paged.write_flat(self.pool_kv, self.wkv(xf).reshape(seqlen, -1))
        paged.write_flat(self.pool_score, self.wgate(xf).reshape(seqlen, -1))

        # Groups this call completes. Group `g` covers tokens
        # `g*ratio .. g*ratio+ratio-1`, so it is complete once the chunk has
        # reached `start_pos + seqlen - 1`.
        #
        # The *count* has to be a Python int — it is the row count of everything
        # below — and comes off the span's precomputed plan, where it is 0 or 1
        # for a decode step and so guards stably. The *first group's index* is
        # the position as a value, so it stays a tensor: `arange(group_lo,
        # group_hi)` would fold `group_lo` into the graph and recompile this
        # layer every `ratio` steps. See `PagedSpan`.
        count = paged.group_counts[ratio]
        if count <= 0:
            return None, None

        firsts = (start_pos // ratio + torch.arange(count, device=device)) * ratio
        members = firsts.unsqueeze(1) + torch.arange(ratio, device=device)
        slots = paged.slots(members.reshape(-1))
        # fp32 for the pooling itself, which is where the reference's fp32
        # interior actually matters: the softmax over `ratio` members and the
        # weighted sum. The *storage* is bf16 and widening it here is exact —
        # see `allocate_state`.
        kv = self.pool_kv.index_select(0, slots).reshape(count, ratio, -1).float()
        score = (
            self.pool_score.index_select(0, slots).reshape(count, ratio, -1).float()
        )
        pooled = (kv * score.softmax(dim=1)).sum(dim=1).unsqueeze(0)
        return self.norm(pooled.to(dtype)), firsts


class Indexer(nn.Module):
    """Keeps the `index_topk = 512` best compressed positions per query.

    A small side attention: 32 heads of 128 dims score the query against one
    shared key per compressed position, the scores are rectified and then
    combined by `weights_proj`. `owns_k` — true only for a layer that is also a
    KV source — is what decides whether this indexer *produces* the index keys
    or reads the ones its source published (`vendor/model.py`'s `Indexer`).
    """

    def __init__(self, config: DeepseekV41Config, layer_id: int, tp_size: int) -> None:
        super().__init__()
        self.owns_k = layer_id in config.kv_source_layer_ids
        self.compress_ratio = config.compress_ratios[layer_id]
        self.is_candidate_source = layer_id == config.candidate_source_layer_id
        self.uses_candidates = 0 <= config.candidate_source_layer_id < layer_id
        self.n_heads = config.index_n_heads
        self.n_local_heads = config.index_n_heads // tp_size
        self.index_head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = config.index_head_dim**-0.5

        self.wq_b = BlockQuantLinear(
            config.q_lora_rank, self.n_local_heads * config.index_head_dim
        )
        self.weights_proj = BlockQuantLinear(
            config.hidden_size, self.n_local_heads, storage="bf16"
        )
        if self.owns_k:
            self.wk = BlockQuantLinear(
                config.head_dim, config.index_head_dim, storage="bf16"
            )
            self.k_norm = RMSNorm(config.index_head_dim, config.rms_norm_eps)

        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        self.rope_head_dim = config.qk_rope_head_dim
        self.world_size = tp_size
        self.freqs_cis: torch.Tensor | None = None
        self.k_cache: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        latent: torch.Tensor | None,
        group_positions: torch.Tensor | None,
        start_pos: torch.Tensor,
        offset: int,
        shared: SharedAttentionState,
        paged: PagedSpan,
        tp_group=None,
    ) -> torch.Tensor:
        """The `index_topk` compressed positions each query keeps.

        `latent` is this layer's RoPE-free compressed latent, `None` when the
        layer does not compress or its current group is incomplete. An index-key
        owner turns it into index keys *here*, which has to happen before
        `Attention` overwrites that same storage with the RoPE'd, quantized
        values — the ordering is the reason `Indexer` runs before the cache
        write rather than after.
        """
        assert self.freqs_cis is not None
        bsz, seqlen, _ = x.shape
        ratio, rd = self.compress_ratio, self.rope_head_dim

        if self.owns_k:
            # Published whether or not this step completed a group. The cache
            # holds every earlier group — that is the point of it living in the
            # paged allocation — so an owner's store is the index-key store for
            # the whole step. Publishing it only when a new group arrived left
            # `shared.index_k` as None on the decode steps that complete none
            # (every other step at `compress_ratio = 2`), and the gather below
            # then failed inside the compiled block with nothing but
            #
            #     torch._dynamo.exc.Unsupported: Observed exception
            #       Developer debug context: raised exception
            #       AttributeError([ConstantVariable(str: 'index_select')])
            #
            # to say that the tensor it was handed was None.
            shared.index_k = self.k_cache
            if latent is not None:
                # A latent stands for the first token of its group, which is the
                # position `group_positions` carries — so one expression covers
                # both the prefill and the decode case the reference splits.
                freqs = slice_freqs(self.freqs_cis, group_positions)
                k = self.k_norm(self.wk(latent))
                k = torch.cat(
                    [k[..., :-rd], apply_rotary_emb(k[..., -rd:], freqs)], dim=-1
                )
                k = fp4_quant_dequant(k, 32)
                paged.put(self.k_cache, group_positions, k.reshape(k.size(1), -1))

        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim))
        q = torch.cat(
            [
                q[..., :-rd],
                apply_rotary_emb(
                    q[..., -rd:],
                    span_freqs(self.freqs_cis, start_pos, seqlen, x.device),
                ),
            ],
            dim=-1,
        )
        q = fp4_quant_dequant(q, 32)

        # The compressed positions that exist so far, gathered out of the paged
        # index-key cache: group j lives at the slot of token j * ratio.
        #
        # `read_len` is the bucketed Python int off the span's plan (a shape);
        # the clamp bound is the position as a *value*, so it is a tensor. See
        # `PagedSpan` for why mixing those up recompiles every layer every step.
        read_len = paged.read_lens[ratio]
        compress_len = (start_pos + seqlen) // ratio
        group_pos = (
            torch.minimum(
                torch.arange(read_len, device=x.device),
                (compress_len - 1).clamp_min(0),
            )
            * ratio
        )
        index_k = paged.gather(shared.index_k, group_pos).reshape(read_len, -1)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        # The score, the rectification and the weighted head sum, fused:
        # `kernels/index_score.weighted_scores`. The torch form was
        #
        #     index_score = torch.einsum("bshd,btd->bsht", q.float(), k.float())
        #     index_score = (index_score.relu() * weights...).sum(dim=2)
        #
        # whose first line *names* a `[s, heads, read_len]` fp32 intermediate —
        # 11.2 GB for one layer of the verifier's 9,328-token prompt — for a
        # value the second line reduces away. The kernel's live set is one
        # `[heads, tile]` psum tile instead, and it is exact to fp32 against
        # that expression (checked under `nki.simulate`: max relative error
        # 2e-7 at every shape the model produces).
        #
        # Measured honestly: this did **not** reduce HBM use, so neuronx-cc was
        # already fusing the einsum into the reduce. `.autohelix/notes/iter-5.md`
        # §4 has the before/after memory tables. What it buys is not depending
        # on the compiler finding that fusion, the arithmetic in one place, and
        # one NKI call where there were three torch ops.
        #
        # `w` goes in head-major because the weighted sum contracts over heads;
        # the transpose is `[s, 32]`, and doing it here is cheaper than a
        # per-token transpose inside the kernel. Both products contract on
        # `index_head_dim`, which the kernel transposes for.
        index_score = _nki_entry("index_score", "weighted_scores")(
            q.reshape(-1, self.index_head_dim),
            index_k,
            weights.reshape(seqlen, self.n_local_heads).t().contiguous().float(),
        ).reshape(1, seqlen, read_len)
        if self.world_size > 1 and tp_group is not None:
            # Heads are split across ranks and the score sums over heads, so the
            # partial sums have to be added before the top-k — a per-rank top-k
            # would pick from a different score.
            index_score = tp_all_reduce(index_score, tp_group)

        # How many compressed positions a query can see: a group becomes visible
        # once the query has passed its last token. Query `t` of this span sits
        # at absolute position `start_pos + t`, so it can see
        # `(start_pos + t + 1) // ratio` groups — one expression covering the
        # prefill and the decode case, which also removes the groups past
        # `compress_len` that `read_len` added: `arange(read_len) >=
        # compress_lens` is true for every one of them, so a padded group can
        # never win the top-k.
        #
        # This replaces a `if start_pos == 0` split whose `else` branch used the
        # span's *last* position for every query. That is right for a decode
        # step (`seqlen == 1`, the only case it ran in) and would have been
        # wrong for a chunked prefill, which hits the same branch and would have
        # let an early query reach a group it cannot see yet. The tensor form is
        # per query, so the branch is gone and with it the guard on `start_pos`.
        compress_lens = (
            (start_pos + torch.arange(1, seqlen + 1, device=x.device)) // ratio
        ).unsqueeze(-1)
        index_score = index_score.masked_fill(
            torch.arange(read_len, device=x.device) >= compress_lens,
            _bound(index_score, _NEG_BOUND),
        )

        if self.is_candidate_source:
            shared.candidates = select_candidate_blocks(
                index_score,
                compress_lens,
                self.candidate_topk_blocks,
                self.candidate_block_size,
            )
        elif self.uses_candidates:
            # Level two: score with our own weights, but only inside the
            # source's candidate blocks.
            index_score = index_score.masked_fill(
                ~shared.candidates, _bound(index_score, _NEG_BOUND)
            )

        # `read_len`, not `compress_len`: the width of this layer's published
        # top-k list is a tensor shape, so it has to change on bucket
        # boundaries rather than on every decode step. The extra picks come
        # back as `EMPTY_SLOT` from the test below, since their score is
        # -1e30 and their index is at or beyond `compress_lens`.
        topk = min(self.index_topk, read_len)
        # The reference finishes with `.sort(dim=-1)` on the chosen indices.
        # That is not dropped lightly, and it is not an approximation: every
        # consumer of this result is invariant under a permutation of the last
        # axis. `sparse_attn` gathers each named slot independently and sums
        # over them, the `+ offset` and the `< compress_lens` test are
        # elementwise, and the later layers that reuse `shared.topk_idxs` do the
        # same. The reference sorts so its kernel reads the cache in order.
        #
        # It is also not available here: `Tensor.sort` makes neuronx-cc fail
        # (see `nki_topk`) and this repo's `NF.argsort_unstable` is documented
        # for 1-D or `[1, N]` input only — "Vectorization along a batch
        # dimension is not supported" — while this is `[b, s, index_topk]`.
        # If a future NKI attention kernel needs sorted slots, sort there.
        idxs = nki_topk(index_score, topk)[1]
        return torch.where(idxs < compress_lens, idxs + offset, EMPTY_SLOT).int()


class Attention(nn.Module):
    """Latent attention over a sliding window plus, where configured,
    compressed KV — concatenated into one sparse-attention call.

    `compress_ratios[layer_id] > 0` does *not* mean this layer compresses its
    own KV: only the four `kv_source_layer_ids` do, and the rest read that same
    cache. Same for the index: eight `index_source_layer_ids` run an indexer,
    the layers between them reuse the top-k their source published.
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        layer_id: int,
        tp_size: int,
        max_batch_size: int = 1,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_local_heads = config.num_attention_heads // tp_size
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = config.head_dim - config.qk_rope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.n_groups = config.o_groups
        self.n_local_groups = config.o_groups // tp_size
        self.window_size = config.sliding_window
        self.compress_ratio = config.compress_ratios[layer_id]
        self.softmax_scale = config.head_dim**-0.5

        is_backbone = layer_id < config.num_hidden_layers
        self.is_kv_source = is_backbone and layer_id in config.kv_source_layer_ids
        self.is_index_source = is_backbone and layer_id in config.index_source_layer_ids

        # One fp32 logit bias per query head, added inside the softmax.
        self.register_buffer(
            "attn_sink", torch.empty(self.n_local_heads, dtype=torch.float32)
        )

        self.wq_a = BlockQuantLinear(config.hidden_size, config.q_lora_rank)
        self.q_norm = RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.wq_b = BlockQuantLinear(
            config.q_lora_rank, self.n_local_heads * config.head_dim
        )
        # One latent KV for all heads (num_key_value_heads=1), so there is no
        # head axis to shard here and `wkv` is replicated.
        self.wkv = BlockQuantLinear(config.hidden_size, config.head_dim)
        self.kv_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        # Block-diagonal over `o_groups`: each group sees only its own heads, so
        # in_features is per-group and the group axis is the one that shards.
        self.wo_a = BlockQuantLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_local_groups * config.o_lora_rank,
        )
        self.wo_b = BlockQuantLinear(
            self.n_local_groups * config.o_lora_rank, config.hidden_size
        )

        self.compressor = (
            Compressor(config, layer_id, max_batch_size) if self.is_kv_source else None
        )
        self.indexer = (
            Indexer(config, layer_id, tp_size) if self.is_index_source else None
        )
        self.world_size = tp_size
        self.config = config

        # The 128-slot ring of raw KV every layer attends over. Bound by
        # `bind_kv_cache` from the runtime's allocation; `None` until then, so a
        # forward before binding fails on the attribute rather than reading
        # zeros.
        self.window_kv_cache: torch.Tensor | None = None
        # Only a KV source owns these two; everyone else reads the source's.
        self.compress_kv_cache: torch.Tensor | None = None
        self.index_k_cache: torch.Tensor | None = None

        # Which RoPE a layer uses is decided by whether it reads compressed KV,
        # not by taste: a compressed latent stands for `compress_ratio` tokens,
        # so its positions are further apart and it rotates at
        # `compress_rope_theta` with YaRN on. A pure sliding-window layer has
        # YaRN off and uses the base `rope_theta`.
        self.freqs_cis: torch.Tensor | None = None

    def build_freqs_cis(self, max_seq_len: int, device) -> None:
        config = self.config
        if self.compress_ratio:
            original_seq_len = config.original_max_position_embeddings
            base = config.compress_rope_theta
        else:
            original_seq_len, base = 0, config.rope_theta
        self.freqs_cis = yarn_freqs_cis(
            self.rope_head_dim,
            max_seq_len,
            original_seq_len,
            base,
            config.rope_factor,
            config.beta_fast,
            config.beta_slow,
            device=device,
        )
        if self.indexer is not None:
            self.indexer.freqs_cis = self.freqs_cis
            self.indexer.k_cache = self.index_k_cache


    # ── Forward ───────────────────────────────────────────────────────────

    def _window_kv(
        self,
        x: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        start_pos: torch.Tensor,
        paged: PagedSpan,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """This layer's sliding-window K, and the slots every query may see.

        The K stays fp8-rounded, quantized over the whole post-RoPE vector
        including its RoPE tail — so the cache holds values that have already
        been through fp8, and attention must read those rather than the
        pre-quantization ones.

        Written into, and read back out of, vLLM's *paged* allocation
        (`[num_blocks, 1, block_size, head_dim]`) rather than a private
        contiguous ring: see `PagedSpan`. The write goes to the slots the
        runtime assigned this step's tokens; the read gathers the
        `window + seqlen - 1` positions ending at `start_pos + seqlen - 1`,
        which is every position any query in this span can see.

        **That count is deliberately fixed rather than `start_pos`-dependent.**
        Early in a sequence the span reaches back before position 0; those
        positions are clamped to slot 0 on the read and marked empty in
        `window_topk_idxs`, so they contribute nothing. Sizing the read to the
        live history instead would make the gathered tensor's *shape* a function
        of the decode position, and every layer would recompile once per token.
        """
        _, seqlen, _ = x.shape
        win = self.window_size
        rd = self.rope_head_dim
        if self.window_kv_cache is None:
            raise RuntimeError(
                f"layer {self.layer_id} attention has no window KV cache; "
                "bind_kv_cache() must run before forward()."
            )
        kv = self.kv_norm(self.wkv(x))
        kv = torch.cat([kv[..., :-rd], apply_rotary_emb(kv[..., -rd:], freqs_cis)], -1)
        kv = act_quant_dequant(kv, 32)

        paged.write(self.window_kv_cache, kv.reshape(seqlen, self.head_dim))

        span = win + seqlen - 1
        base = start_pos - (win - 1)
        positions = (base + torch.arange(span, device=x.device)).clamp_min(0)
        window_kv = paged.gather(
            self.window_kv_cache.view(-1, self.head_dim), positions
        ).reshape(1, span, self.head_dim)
        return window_kv, window_topk_idxs(win, seqlen, start_pos, x.device)

    def _compress_kv(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: torch.Tensor,
        offset: int,
        shared: SharedAttentionState,
        paged: PagedSpan,
        tp_group,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The shared compressed KV and the compressed slots each query sees.

        Paged like the window (`PagedSpan`), with group `j`'s latent at the slot
        of token `j * compress_ratio`. That is what makes the cache correct
        under prefix caching: a shared prefix produces the same latents in the
        same blocks, and a divergent suffix writes to different blocks because
        the block table diverges.
        """
        bsz, seqlen, _ = x.shape
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        # Shape from the span's bucketed plan, value as a tensor — `PagedSpan`
        # says why the two cannot be the same quantity.
        read_len = paged.read_lens[ratio]
        compress_len = (start_pos + seqlen) // ratio

        latent, group_positions = None, None
        if self.is_kv_source:
            latent, group_positions = self.compressor(x, start_pos, paged)
            shared.compress_kv = self.compress_kv_cache

        # The indexer needs the latent *before* RoPE, so it runs before the
        # cache write below.
        if self.is_index_source:
            # `read_len`, not `compress_len`: `read_len == 0` exactly when
            # `compress_len <= 0`, and it is the bucketed int, so this branch
            # does not read the raw position (see `PagedSpan`).
            if read_len == 0:
                idxs = torch.empty(
                    bsz, seqlen, 0, dtype=torch.int32, device=x.device
                )
            else:
                idxs = self.indexer(
                    x, qr, latent, group_positions, start_pos, offset, shared,
                    paged, tp_group,
                )
            shared.topk_idxs = idxs
        else:
            # Normalized on the way in. An index-source layer publishes int32,
            # but by the time a *later* layer's graph lifts the same tensor in
            # as an input its dtype has become unsigned, and concatenating it
            # with this layer's int32 window list fails at HLO conversion:
            #
            #     RuntimeError: Promotion for uint16, uint32, uint64 types is
            #     not supported, attempted to promote Int and UInt32
            #     While executing %topk_idxs_1 = torch.cat((%topk_idxs,
            #       %l_shared_topk_idxs), dim = -1)
            #
            # An explicit convert is honoured where the implicit promotion is
            # not, and it is free: the values are small non-negative slot
            # numbers or the -1 sentinel.
            idxs = shared.topk_idxs.to(torch.int32)

        if latent is not None:
            freqs = slice_freqs(self.freqs_cis, group_positions)
            rotated = torch.cat(
                [latent[..., :-rd], apply_rotary_emb(latent[..., -rd:], freqs)], -1
            )
            # Compressed KV uses 16-wide groups with e4m3 scales; the indexer
            # above uses 32 with e8m0. Not interchangeable.
            rotated = fp4_quant_dequant(rotated, 16, scale_e4m3=True)
            paged.put(
                self.compress_kv_cache,
                group_positions,
                rotated.reshape(rotated.size(1), -1),
            )

        # Read after the write, so a query sees the groups this call completed.
        # `read_len` rather than `compress_len`: see `_compress_read_len`. The
        # groups past `compress_len` are read from the last live group's slot
        # and are never named by `idxs`, because `Indexer.forward` maps any pick
        # at or beyond `compress_len` to `EMPTY_SLOT`.
        group_pos = (
            torch.minimum(
                torch.arange(read_len, device=x.device),
                (compress_len - 1).clamp_min(0),
            )
            * ratio
        )
        compress_kv = paged.gather(shared.compress_kv, group_pos).reshape(
            1, read_len, -1
        )
        return compress_kv, idxs

    def forward(
        self,
        x: torch.Tensor,
        start_pos: torch.Tensor,
        shared: SharedAttentionState,
        paged: PagedSpan,
        tp_group=None,
    ) -> torch.Tensor:
        """`x` is the normalized sublayer input `[b, s, hidden]`.

        `start_pos` is `paged.pos`, the `[1]` device tensor — not a Python int.
        See `PagedSpan` for why that distinction decides whether a decode step
        costs four NEFF compiles or none.
        """
        bsz, seqlen, _ = x.shape
        freqs_cis = span_freqs(self.freqs_cis, start_pos, seqlen, x.device)
        rd = self.rope_head_dim

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        q = torch.cat([q[..., :-rd], apply_rotary_emb(q[..., -rd:], freqs_cis)], -1)

        kv, topk_idxs = self._window_kv(x, freqs_cis, start_pos, paged)
        if self.compress_ratio:
            compress_kv, compress_idxs = self._compress_kv(
                x, qr, start_pos, kv.size(1), shared, paged, tp_group
            )
            kv = torch.cat([kv, compress_kv], dim=1)
            topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)

        o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        # Remove the query's rotation again, so the cache can stay in one
        # shared rotated form.
        o = torch.cat(
            [o[..., :-rd], apply_rotary_emb(o[..., -rd:], freqs_cis, True)], -1
        )

        # `wo_a` is block-diagonal over groups — each projects only its own
        # heads — so it is `n_local_groups` independent GEMMs sharing one
        # buffer, one `grouped()` kernel call each, rather than a dense einsum
        # against a dequantized `[groups*rank, group_in]` weight.
        o = o.reshape(bsz * seqlen, self.n_local_groups, -1)
        o = torch.cat(
            [
                self.wo_a.grouped(o[:, g], g, self.o_lora_rank)
                for g in range(self.n_local_groups)
            ],
            dim=-1,
        ).reshape(bsz, seqlen, -1)
        out = self.wo_b(o)
        if self.world_size > 1 and tp_group is not None:
            # `wo_b` is row-parallel, so each rank holds a partial sum. The
            # reference reduces in fp32.
            out = tp_all_reduce(out.float(), tp_group)
            out = out.to(x.dtype)
        return out


# =============================================================================
# Section 4: the MoE FFN
# =============================================================================
# `MoE.forward` is the one step that cannot live inside an enclosing graph: it
# reads the router's choice back to the host and streams one expert's 17.9 MiB
# off disk per iteration. Dynamo handles a graph break fine, but after a break
# inside a loop whose *length is data-dependent* it stops building resume
# functions and runs the remainder of the frame eagerly — and eager on this
# backend is the host, which cannot even cast a dtype:
#
#     RuntimeError: Expected self.dtype() == dst.dtype() to be true, but got
#     false ... in torch_dynamo_resume_in_forward_at_1812
#       contribution = expert(flat.index_select(0, rows), route_weights).float()
#
# So the loop stays eager by construction (`MoE.forward` is dynamo-disabled)
# and every piece of arithmetic in it has an explicit compiled entry point
# below. All of them take the weights as *arguments* rather than reading them
# off a module, so one NEFF serves all 43 layers: dynamo guards on parameter
# identity, and a bound method's parameters are part of that identity.


def _compile_fn(fn):
    """`fn` as a device graph. One call, so every compiled entry point in this
    file is built the same way as the whole-model compile the runner does."""
    from vllm_neuron.envs import get_compile_backend_name

    _raise_dynamo_limits()
    return torch.compile(
        fn,
        backend=get_compile_backend_name(),
        fullgraph=True,
        options=_COMPILE_OPTIONS,
    )


@functools.lru_cache(maxsize=None)
def _compiled(fn):
    """`fn` as a device graph, cached per function rather than per shape.

    The same mechanism as `_nki_entry`, for a step that is more than a single
    kernel call. Dynamo keys its own code cache on the input shapes behind this
    one callable, and `~/.cache/neuron_libtorch` makes a repeated shape free.

    Only for *free* functions: a bound method is a fresh object on every
    attribute access and would miss this cache every call — see `_block_entry`.
    """
    return _compile_fn(fn)


def _swiglu_expert(x, weights, w1, s1, w2, s2, w3, s3, storage, limit):
    """One expert's SwiGLU FFN; all three GEMMs through `kernels/blockquant`.

    The clamps are the training ones at `swiglu_limit = 10`: the up branch is
    clamped on both sides, the gate branch only from above. `weights` — the
    routing weight, or None for a shared expert — multiplies *before* `w2`,
    which is what makes a zero routing weight contribute exactly zero, since
    `w2` has no bias.
    """
    dtype = x.dtype
    gate = nki_linear(x, w1, s1, storage).float()
    up = nki_linear(x, w3, s3, storage).float()
    if limit > 0:
        up = torch.clamp(up, min=-limit, max=limit)
        gate = torch.clamp(gate, max=limit)
    y = F.silu(gate) * up
    if weights is not None:
        y = weights * y
    return nki_linear(y.to(dtype), w2, s2, storage)


def _routed_expert_step(
    carry, flat, rows, route_weights, w1, s1, w2, s2, w3, s3, limit
):
    """Gather one expert's rows, run it, scatter-add into the fp32 carry.

    `rows` is padded to a power of two (see `MoE._route_plan`) with entries
    pointing at row 0 and a routing weight of 0, so the pad rows add exactly
    zero and no masking is needed on the way out.
    """
    y = _swiglu_expert(
        flat.index_select(0, rows), route_weights,
        w1, s1, w2, s2, w3, s3, "fp4", limit,
    )
    return carry.index_add(0, rows, y.float())


def _shared_expert_step(carry, flat, w1, s1, w2, s2, w3, s3, limit):
    """The shared expert every token goes through, added to the routed sum."""
    contribution = _swiglu_expert(
        flat, None, w1, s1, w2, s2, w3, s3, "fp8", limit
    )
    return carry + contribution.float()


def _moe_carry(flat):
    """The fp32 accumulator the routed experts scatter into.

    A compiled step of its own because `torch.zeros_like` on a Neuron tensor
    outside a graph goes through the backend's host fallback.
    """
    return torch.zeros_like(flat, dtype=torch.float32)


def _dspark_tap(stream):
    """The DSpark head's view of a target layer: the attention input averaged
    over the hyper-connection copies."""
    return stream.float().mean(dim=1)


def _concat_last(tensors):
    """`torch.cat(..., -1)`, as a compiled step."""
    return torch.cat(tensors, dim=-1)


def _splice_rows(head, tail):
    """`head` rows followed by `tail` rows, as a compiled step."""
    return torch.cat([head, tail], dim=0)


def _select_rows(x, rows):
    """`x.index_select(0, rows)`, as a compiled step."""
    return torch.index_select(x, 0, rows)


def _post_into(out, ffn_out, residual, ffn_post, ffn_comb):
    """`HyperConnections.post`, written into `out` rather than returned.

    A free function so `_compiled` keys it on the function and **one** graph
    serves all 43 layers; the point of writing into `out` is that the one graph
    then retains one `[tokens, hc, dim]` buffer instead of one per layer. See
    `Block.attn_half`'s `out_residual` for the measurement that makes this
    matter. `out` and `residual` are different buffers, and `attn_half` has
    finished before this runs, so nothing here reads a buffer it is writing.
    """
    out.copy_(HyperConnections.post(ffn_out, residual, ffn_post, ffn_comb).squeeze(0))
    return out


def _one_hot_pre_mix(stream, hc):
    """The draft stream's initial hyper-connection coefficients: copy 0 only.

    A compiled step because the slice assignment that writes the one-hot column
    is a strided write, and the backend's eager fallback refuses one
    (`Expected self.is_contiguous() to be true, but got false`).
    """
    pre_mix = stream.new_zeros(
        stream.shape[0], stream.shape[1], hc, dtype=torch.float32
    )
    pre_mix[:, :, 0] = 1.0
    return pre_mix


class _RejectionView:
    """The three fields `vllm_neuron.nn.rejection_sampler` reads off a
    `SpecDecodeMetadata`, carried without the dataclass.

    At module level rather than inside `_rejection_sample`, because a `class`
    statement in a traced frame is a graph break dynamo cannot take:

        torch._dynamo.exc.Unsupported: Attempted to call function marked as
        skipped ... qualname: __build_class__
    """

    __slots__ = ("draft_token_ids", "cu_num_draft_tokens", "max_spec_len")


def _rejection_sample(sampled, draft_token_ids, cu_num_draft_tokens, num_draft_tokens):
    """Verify this step's drafts against the target's own tokens, on the device.

    `vllm_neuron.nn.rejection_sampler`, which is what the Neuron runner expects
    an on-device-sampling model to have already applied when a
    `spec_decode_metadata` is present: it hands back
    `[batch, 1 + max_spec_len]` with `-1` where a draft was rejected, and
    `NeuronModelRunner._parse_rejection_sampling_output` trims on that.

    Greedy acceptance, so **the output is exactly what the target would have
    produced on its own**: draft position `i` is kept only if it equals the
    token the target sampled at position `i - 1`, and the first mismatch takes
    the target's token and stops. That is the whole reason speculation here is
    an optimisation rather than an approximation.

    The sampler reads three fields off `SpecDecodeMetadata`, and they are passed
    as plain tensors and a tuple of ints rather than as the dataclass because
    this is a `_compiled` entry point: dynamo guards an opaque object's
    attributes one by one, where a locally built view traces cleanly.
    `num_draft_tokens` arrives as a tuple so dynamo treats it as a constant —
    it is `(k,) * batch` on every verify step.
    """
    from vllm_neuron.nn.rejection_sampler import rejection_sampler

    view = _RejectionView()
    view.draft_token_ids = draft_token_ids
    view.cu_num_draft_tokens = cu_num_draft_tokens
    view.max_spec_len = max(num_draft_tokens)
    return rejection_sampler(view, sampled.reshape(-1))


def _accepted_tail(accepted):
    """`(last accepted token, how many were accepted)` per sequence.

    `accepted` is the rejection sampler's `[batch, 1 + max_spec_len]`, padded
    with `-1`. The DSpark draft for the next step hangs off the last token this
    step actually committed, so both the seed token and the row of the step the
    draft is anchored at come from this count. Tensors throughout: the count is
    data, and a `.item()` here would be a device sync in the middle of a step.
    """
    valid = (accepted >= 0).sum(dim=-1)
    seeds = accepted.gather(1, (valid - 1).clamp_min(0).unsqueeze(1)).reshape(-1)
    return seeds, valid.to(torch.int64)


def _greedy_rows(logits):
    """`argmax` over the vocabulary, as a compiled step."""
    return logits.argmax(dim=-1)


def _tap_rows(valid, lengths, rows_per_seq: int):
    """Which row of this step each sequence's draft block hangs off.

    `rows_per_seq` is how many of the span's rows the sampler scored: 1 without
    speculation (one logits row per sequence, at its last token) and `1 + k` on
    a verify step, where it equals the span's whole length. `valid` is how many
    tokens the step committed for that sequence — 1, or the rejection sampler's
    accepted count. So the anchor is the last *committed* row:

        no speculation   lengths - 1 + (1 - 1)     = the span's last row
        verify step      lengths - lengths + v - 1 = v - 1, inside the block

    The `minimum` is a guard for a batch where a request carried fewer drafts
    than the padded width, which would otherwise anchor a draft past the
    sequence's own rows. A compiled step rather than inline arithmetic: `_draft`
    is an uncompiled frame, and `sub`/`minimum` there would go to the backend's
    eager fallback.
    """
    return torch.minimum(lengths - rows_per_seq + valid - 1, lengths - 1)


def _moe_narrow(y, dtype):
    """The accumulator back down to the stream's dtype. Also its own step: a
    dtype cast is the one thing the backend's eager fallback does not implement
    at all (`RuntimeError: Expected self.dtype() == dst.dtype()`).
    """
    return y.to(dtype)


#: Smallest row bucket `MoE._route_plan` pads to. `kernels/blockquant` tiles
#: the token axis at the partition width anyway, so anything under this is the
#: same number of instructions.
_EXPERT_ROW_FLOOR = 128


def _gate_step(
    x, weight, bias, bias_vl, image_mask, topk, scoring_func, norm_topk_prob,
    route_scale,
):
    """MoE routing: `sqrtsoftplus` scores, `noaux_tc` selection.

    The correction bias steers *selection* only — `(scores + bias).topk` picks
    the experts and the routing weights are gathered from the unbiased scores
    (`vendor/model.py`'s `Gate`).
    """
    scores = nki_linear(x, weight, None, "bf16").float()
    if scoring_func == "softmax":
        scores = scores.softmax(dim=-1)
    elif scoring_func == "sigmoid":
        scores = scores.sigmoid()
    else:
        scores = F.softplus(scores).sqrt()

    if image_mask is not None:
        bias = torch.where(image_mask.unsqueeze(-1), bias_vl, bias)

    indices = nki_topk(scores + bias, topk)[1]
    weights = scores.gather(1, indices)
    if norm_topk_prob and topk > 1:
        # 1e-20 here is the training constant, not `rms_norm_eps`, even though
        # they happen to be equal in this config.
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights * route_scale, indices


class Gate(nn.Module):
    """MoE routing. `sqrtsoftplus` scores, `noaux_tc` selection.

    The correction bias steers *selection* only: `(scores + bias).topk` picks the
    experts, and the routing weights are gathered from the unbiased scores
    (`vendor/model.py`'s `Gate`). `bias_vl` is the image-span variant, loaded
    but unused on the text path.
    """

    def __init__(self, config: DeepseekV41Config, layer_id: int) -> None:
        super().__init__()
        n_routed, n_activated = _moe_config(config, layer_id)
        self.topk = n_activated
        self.n_routed_experts = n_routed
        self.scoring_func = config.scoring_func
        self.norm_topk_prob = config.norm_topk_prob
        self.route_scale = config.routed_scaling_factor
        self.register_buffer(
            "weight", torch.empty(n_routed, config.hidden_size, dtype=torch.bfloat16)
        )
        self.register_buffer("bias", torch.empty(n_routed, dtype=torch.float32))
        self.register_buffer("bias_vl", torch.empty(n_routed, dtype=torch.float32))

    def forward(
        self, x: torch.Tensor, image_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The router, as its own compiled graph.

        Its own, because `MoE.forward` around it is dynamo-disabled (see the
        section comment above). Nesting is free the other way: when this is
        reached from inside an enclosing traced frame — as it is in the
        component checks — the graph is traced *through* and the kernels become
        nodes of the outer graph.

        The router's GEMM goes through the bf16 NKI kernel. That is not a
        precision concession: `self.weight` is stored bf16, `x` is the bf16
        residual stream, and the kernel accumulates in fp32 — so it computes
        the same thing as the reference's `linear(x.float(), w.float())`
        modulo summation order, with no rounding the reference does not have.
        """
        return _compiled(_gate_step)(
            x,
            self.weight,
            self.bias,
            self.bias_vl,
            image_mask,
            self.topk,
            self.scoring_func,
            self.norm_topk_prob,
            self.route_scale,
        )


class Expert(nn.Module):
    """One SwiGLU FFN. The clamps are the training ones, at `swiglu_limit = 10`:
    the up branch is clamped on both sides, the gate branch only from above.
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        storage: str,
        intermediate_size: int | None = None,
        persistent: bool = True,
    ) -> None:
        super().__init__()
        inter = intermediate_size or config.moe_intermediate_size
        self.swiglu_limit = config.swiglu_limit
        self.w1 = BlockQuantLinear(
            config.hidden_size, inter, storage=storage, persistent=persistent
        )
        self.w2 = BlockQuantLinear(
            inter, config.hidden_size, storage=storage, persistent=persistent
        )
        self.w3 = BlockQuantLinear(
            config.hidden_size, inter, storage=storage, persistent=persistent
        )

    def allocate(self, device) -> None:
        for name in ("w1", "w2", "w3"):
            getattr(self, name).allocate(device)

    def forward(
        self, x: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        return _swiglu_expert(
            x,
            weights,
            self.w1.weight, self.w1.scale,
            self.w2.weight, self.w2.scale,
            self.w3.weight, self.w3.scale,
            self.w1.storage,
            self.swiglu_limit,
        )


def _moe_config(config: DeepseekV41Config, layer_id: int) -> tuple[int, int]:
    """Routed/activated expert counts: 384-of-6 on the backbone, 128-of-3 on a
    DSpark stage (`ModelArgs.get_moe_config`).
    """
    if layer_id < config.num_hidden_layers:
        return config.n_routed_experts, config.num_experts_per_tok
    return config.dspark_n_routed_experts, config.dspark_num_experts_per_tok


class MoE(nn.Module):
    """Top-k routed experts plus one shared expert every token goes through.

    The routed experts are *not* submodules: 384 of them per layer is 6.9 GiB
    of fp4, and 40 layers of that is 276 GiB. They are fetched by id from
    `weight_loader.TimeshareWeightStore` into the staging slots allocated here,
    which is the whole of what "timesharing the HBM" means for this model. The
    gate and the shared expert are resident, because every token needs them.

    `expert_slots` is sized for the worst case a step can route to, not for 384:
    `max_num_seqs` sequences x `num_experts_per_tok` experts, capped at the pool
    size. A decode step with 4 sequences can touch at most 24 of 384.
    """

    def __init__(
        self, config: DeepseekV41Config, layer_id: int, tp_size: int, tp_rank: int
    ) -> None:
        super().__init__()
        n_routed, n_activated = _moe_config(config, layer_id)
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.n_routed_experts = n_routed
        self.n_activated_experts = n_activated
        # Experts are partitioned by id across ranks exactly as the reference
        # does, so a rank runs only the experts it owns and the four partial
        # accumulators are summed.
        self.n_local_experts = n_routed // tp_size
        self.experts_start_idx = tp_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts

        self.gate = Gate(config, layer_id)
        self.shared_experts = Expert(config, storage="fp8")
        # One expert's worth of device memory, reused for every routed expert
        # this rank owns. This single module *is* the timeshare: 384 experts
        # would be 6.9 GiB per layer and 276 GiB over the model, against 24 GiB
        # of HBM per rank. Registered as a child so `.to(device)` places its
        # buffers, and refilled in place by `_stage` rather than reallocated.
        self.expert_slot = Expert(config, storage="fp4", persistent=False)

    @torch._dynamo.disable
    def _stage(self, blob: dict[str, torch.Tensor]) -> Expert:
        """Copy one expert's stored weights into the staging slot.

        `copy_()` into the existing device buffers, not assignment: the buffers
        are what `.to(device)` placed in HBM and what a compiled graph is bound
        to, so rebinding the attribute would move the computation off them.

        Dynamo-disabled because `.to(device)` and `copy_()` are *traceable* aten
        ops, so with this inlined dynamo pulled the mmap'd host blob into the
        surrounding graph as an input and the backend refused it:

            ValueError: Neuron Compilation called with tensors not on neuron.
            Tensor with shape torch.Size([5898240]) is on device cpu

        (5898240 = 2304 x 2560, one expert's packed fp4 `w1`.) The general rule
        this is an instance of: **anything that touches the host is its own
        decorated method, not a line in the middle of a traced one.**
        """
        for name in ("w1", "w2", "w3"):
            linear = getattr(self.expert_slot, name)
            linear.weight.copy_(blob[f"{name}.weight"].to(linear.weight.device))
            linear.scale.copy_(blob[f"{name}.scale"].to(linear.scale.device))
        return self.expert_slot

    @torch._dynamo.disable
    def _route_plan(
        self,
        indices: torch.Tensor,
        weights: torch.Tensor,
        live_rows: int | None,
    ) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
        """Which of this rank's experts to run, over which rows, with what
        weight.

        Host-side, and it has to be: the set of routed experts and the number of
        rows each one got are *outputs of the router*, so no graph can know them.
        Returned as `(expert_id, rows, route_weights)` with the row list padded
        to a power of two (floor 128, which is what `kernels/blockquant` tiles
        the token axis at anyway). Padding keeps the number of distinct shapes
        the expert graph is compiled for down to a handful instead of one per
        step; a pad entry points at row 0 with weight 0, and since `w2` has no
        bias its contribution is exactly zero.

        `live_rows` is where the real tokens stop. The runner sizes `input_ids`
        to the batched-token bucket and leaves the tail at whatever the buffer
        held, so a 20-token prompt in a 512-token bucket has 492 rows of stale
        content — which still route, and an expert nobody needs costs 17.9 MiB
        of disk traffic. At that bucket this is the difference between ~8 experts
        and most of 384.

        The alternative needs no host read at all: run every routed expert over
        all `T` tokens with a zero weight where the token did not pick it. It is
        numerically identical and it is `n_routed_experts / num_experts_per_tok`
        = **64x** the arithmetic — 8.1 PFLOP for one 8192-token prefill against
        129 TFLOP. Keep the host read.

        It is also the per-layer HBM probe (`_hbm_mark`), because it is called
        exactly once per MoE layer from host code dynamo has already broken out
        of. **That placement is the point.** The first attempt put the marks in
        `forward`'s layer loop, which looks like uncompiled host code and is
        not: dynamo partitions that frame into subgraphs around its existing
        breaks, so an extra host call there is an extra break, every cache key
        downstream of it changes, and one of the new fragments turned out to
        crash the compiler outright (`[NCC_ISMP902] Simplifier error:
        is_subset()`). Instrumentation must go where a break already is.
        """
        rows = int(indices.shape[0])
        if rows >= _GC_MIN_ROWS:
            _collect_layer_garbage()
        _hbm_mark(self.layer_id, rows)
        if _HBM_STATE[0] == 0 and self.layer_id in _CENSUS_LAYERS:
            _census_device_tensors(self.layer_id)

        ids = indices if live_rows is None else indices[:live_rows]
        wts = weights if live_rows is None else weights[:live_rows]
        # `.cpu()` before `.float()`, not after. A dtype cast on a Neuron tensor
        # outside a graph is served by the backend's eager fallback, which does
        # not implement conversion at all:
        #
        #     RuntimeError: Expected self.dtype() == dst.dtype() to be true,
        #     but got false
        #
        # The host copy is the cheap part here (`[T, topk]`), and once it is on
        # the host every cast is ordinary torch.
        ids_host = ids.cpu()
        wts_host = wts.cpu().float()
        device = indices.device

        plan = []
        for expert_id in sorted(set(ids_host.flatten().tolist())):
            if not self.experts_start_idx <= expert_id < self.experts_end_idx:
                continue
            rows, slot = torch.where(ids_host == expert_id)
            count = rows.numel()
            if count == 0:
                continue
            bucket = max(_EXPERT_ROW_FLOOR, 1 << (count - 1).bit_length())
            padded_rows = rows.new_zeros(bucket)
            padded_rows[:count] = rows
            padded_weights = wts_host.new_zeros(bucket, 1)
            padded_weights[:count, 0] = wts_host[rows, slot]
            plan.append(
                (
                    expert_id,
                    padded_rows.to(device),
                    padded_weights.to(device),
                )
            )
        return plan

    @torch._dynamo.disable
    def forward(
        self,
        x: torch.Tensor,
        image_mask: torch.Tensor | None,
        weight_store,
        tp_group=None,
        live_rows: int | None = None,
    ) -> torch.Tensor:
        """Route to `num_experts_per_tok` of the pool, plus the shared expert.

        The loop runs only over the experts *this rank owns that this step
        actually routed to*, and fetches each one's weights as it reaches it.
        For a 4-sequence decode step that is at most 24 of 384 experts across
        all ranks — 6 of them here — so the layer's expert traffic is 6% of its
        expert bytes rather than all of them. That streaming is the whole of
        what "timesharing the HBM" means for this model.

        Expert selection is identical on every rank (the gate is replicated and
        scores the whole pool), which is what makes the per-rank id partition
        cover the routed set exactly once. If selection could differ, the
        all-reduce would sum a different set of experts per rank.

        Dynamo-disabled, with every arithmetic step a compiled entry point —
        see the section comment above for the error that forces it.
        """
        if weight_store is None:
            raise RuntimeError(
                f"layer {self.layer_id} MoE has no weight store: the routed "
                "experts are streamed, so load_weights() must have attached "
                "one. Running without it would mean running without the "
                "experts."
            )
        shape = x.shape
        flat = x.reshape(-1, self.hidden_size)
        weights, indices = self.gate(flat, image_mask)
        plan = self._route_plan(indices, weights, live_rows)

        y = _compiled(_moe_carry)(flat)
        limit = self.expert_slot.swiglu_limit
        for expert_id, rows, route_weights in plan:
            slot = self._stage(weight_store.expert(self.layer_id, expert_id))
            y = _compiled(_routed_expert_step)(
                y, flat, rows, route_weights,
                slot.w1.weight, slot.w1.scale,
                slot.w2.weight, slot.w2.scale,
                slot.w3.weight, slot.w3.scale,
                limit,
            )

        if tp_group is not None and self.n_local_experts != self.n_routed_experts:
            # Each rank holds the partial sum over its own experts. The
            # reference reduces in fp32, before the shared expert joins.
            y = tp_all_reduce(y, tp_group)

        shared = self.shared_experts
        y = _compiled(_shared_expert_step)(
            y, flat,
            shared.w1.weight, shared.w1.scale,
            shared.w2.weight, shared.w2.scale,
            shared.w3.weight, shared.w3.scale,
            shared.swiglu_limit,
        )
        return _compiled(_moe_narrow)(y, x.dtype).reshape(shape)


# =============================================================================
# Section 5: Engram
# =============================================================================


def _next_prime(start: int, seen: set[int]) -> int:
    """The smallest prime above `start` not already handed out.

    `vendor/engram.py` uses `sympy.isprime`; this is the same function written
    out, because the whole bucket layout depends on *which* primes come out and
    in what order, and the layout has to be reproducible from this file alone.
    Deterministic Miller-Rabin over the first 12 prime bases is exact below
    3.3e24, which covers the ~1.6e7 range these land in.
    """
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for small in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % small == 0:
            return n == small
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for base in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(base, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Collapse token ids onto a smaller space where tokens that normalize alike
    share an id, so " The", "the" and "THE" hash the same way.

    `vendor/engram.py`'s function verbatim, including the private-use sentinel —
    without it a token that is exactly one space would be stripped to the empty
    string and merge with unrelated tokens.

    The returned size is not just a bounds check: every hash multiplier is
    derived from it, so a mismatch rehashes the whole table silently. That is
    why the caller asserts it against `engram_compressed_vocab_size`.
    """
    from tokenizers import Regex, normalizers

    # A private-use char: without it a token that is exactly one space would
    # be stripped to the empty string and merge with unrelated tokens.
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # A partial UTF-8 byte token: nothing to normalize, so key it by its
            # raw form.
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


class NgramHashState(nn.Module):
    """Maps each position to the hash ids of the n-grams ending there.

    This is the input `Engram` needs and cannot produce: a row index into a
    384,006,168-row table per (n-gram size, head). Ids go through the compressed
    map above, then each position is hashed against the `max_ngram_size - 1 = 3`
    tokens before it, giving `3 * n_heads = 24` ids per position.

    Three details that are easy to lose and change every lookup:

      - Look-back stops at the start of the sequence and at any dead token (an
        image span), filling with `engram_pad_token_id` — so an n-gram never
        spans one.
      - Each (n-gram size, head) pair owns a disjoint prime-sized bucket range,
        with the primes drawn in order from `engram_vocab_size - 1` and never
        reused. The offsets that place those ranges are cumulative, so getting
        the draw order wrong moves every bucket.
      - The hash is a rolling XOR of multiplied ids, and the running value after
        step `i` is the hash of the `(i+1)`-gram. The multipliers come from
        numpy's PCG64 seeded `10007 * layer_id` and are forced odd.

    `cache` carries the compressed ids across the prefill/decode split, which is
    what lets a decode step hash an n-gram reaching back into the prompt. It is
    **slot-addressed**, through `PagedSpan`, exactly like the window KV and the
    compressed latents: one flat `[num_slots]` row per KV slot rather than a
    `[batch, position]` rectangle.

    That is not a detail. A `[batch, position]` cache has to be indexed by a
    request's position *in the running batch*, and this port only ever sees
    `batch == 1` here because `_engram_hashes` hashes one span at a time — so
    with more than one sequence scheduled (the verifier's default is
    `max_num_seqs=4`) every sequence wrote its ids over row 0 and the next one
    hashed against them. Addressing by slot makes the history live in the
    blocks the sequence owns, which also makes it correct under prefix caching:
    a shared prefix is the same ids in the same blocks.
    """

    DEAD = -1

    def __init__(self, config: DeepseekV41Config, tokenizer) -> None:
        super().__init__()
        import numpy as np

        self.layer_ids = tuple(config.engram_layer_ids)
        self.max_ngram_size = config.engram_max_ngram_size
        self.n_heads = config.engram_n_heads

        token_map, vocab_size = build_compressed_token_map(tokenizer)
        if vocab_size != config.engram_compressed_vocab_size:
            raise ValueError(
                f"compressed vocab is {vocab_size}, config says "
                f"{config.engram_compressed_vocab_size}. Every hash multiplier "
                "derives from this number, so a mismatch would rehash the whole "
                "table — refusing rather than looking up the wrong rows."
            )
        self.pad_id = token_map[config.engram_pad_token_id]

        primes, seen = [], set()
        for _ in self.layer_ids:
            per_ngram = []
            for _ in range(self.max_ngram_size - 1):
                sizes, current = [], config.engram_vocab_size - 1
                for _ in range(self.n_heads):
                    current = _next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))

        flat = [[p for per_ngram in layer for p in per_ngram] for layer in primes]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]

        max_long = np.iinfo(np.int64).max
        bound = max(1, (max_long // vocab_size) // 2)
        rows = []
        for layer_id in self.layer_ids:
            generator = np.random.default_rng(10007 * layer_id)
            values = generator.integers(
                low=0, high=bound, size=(self.max_ngram_size,), dtype=np.int64
            )
            rows.append(torch.tensor(values * 2 + 1))

        self.register_buffer("primes", torch.tensor(primes), persistent=False)
        self.register_buffer(
            "offsets", torch.tensor(np.array(offsets)), persistent=False
        )
        self.register_buffer("multipliers", torch.stack(rows), persistent=False)
        self.register_buffer("token_map", torch.tensor(token_map), persistent=False)
        # Allocated by `allocate_state` once the KV slot count is known.
        self.register_buffer("cache", None, persistent=False)

    def allocate_state(self, device, num_slots: int) -> None:
        """The per-slot compressed-id history.

        `DEAD` rather than zeros: an unwritten slot must read as "no token
        here", which blocks the look-back and pads the n-gram, rather than as
        compressed id 0 — which is a real token and would hash a bucket nothing
        put anything in. `int64` because the ids feed the hash multipliers.
        """
        self.cache = torch.full(
            (num_slots,), self.DEAD, dtype=torch.int64, device=device
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        paged: PagedSpan,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns `[B, L, n_engram_layers, n_hash_cols]` row indices.

        `token_mask` is `[B, L]`, False for tokens that take no part in an
        n-gram (image spans). Text-only serving passes None.

        The position comes off `paged.pos` as a tensor. Both uses of it used to
        be Python-int expressions — a `cache[:, start_pos : start_pos + seqlen]`
        slice assignment and an `arange(start_pos, start_pos + seqlen)` — and
        each baked the decode position into this graph as a constant, so the
        hash state recompiled once per token. See `PagedSpan`.
        """
        if self.cache is None:
            raise RuntimeError(
                "the n-gram hash state has no slot cache; "
                "allocate_state() must run before forward()"
            )
        batch, seqlen = input_ids.shape
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, self.DEAD)

        positions = paged.pos + torch.arange(seqlen, device=input_ids.device)
        paged.write_flat(self.cache, compressed.reshape(seqlen))

        positions = positions.unsqueeze(0).expand(batch, seqlen)
        tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.max_ngram_size):
            # Read back through the block table, so the look-back reaches this
            # sequence's own history and no one else's.
            source = self.cache.index_select(
                0, paged.slots((positions - shift).clamp_min(0).reshape(-1))
            ).reshape(batch, seqlen)
            blocked = blocked | (positions < shift) | (source == self.DEAD)
            tokens.append(torch.where(blocked, self.pad_id, source))
        stacked = torch.stack(tokens, dim=-1)

        products = stacked.unsqueeze(2) * self.multipliers
        rolling, hashes = products[..., 0], []
        for i in range(1, self.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets


class Engram(nn.Module):
    """n-gram memory read, gated by its match against the residual stream.

    Layers 1 and 14. `wkv` turns the `(max_ngram_size - 1) * n_heads = 24`
    looked-up rows into one key per hc copy plus one shared value; the gate is a
    normalized dot product of stream against key, per (token, hc copy), with a
    signed square root before the sigmoid to match the training kernel.

    The table is absent from this module on purpose — 94.6 GiB against a 24 GiB
    bank. `weight_loader.TimeshareWeightStore.engram` gathers rows by index and
    dequantizes them; `forward` consumes that result. Skipping the lookup, or
    substituting zeros for it, would change the model, so the store raises
    rather than returning anything if a row is unreachable.
    """

    #: The floor under `|dot|` before the gate's square root, so its derivative
    #: at zero stays finite. `vendor/kernel.py`'s own constant, which
    #: `kernels/engram_gate` bakes in and `_check_kernel_constants` compares
    #: against this.
    CLAMP_VALUE = 1e-6

    def __init__(self, config: DeepseekV41Config, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        self.head_dim = config.engram_head_dim
        self.n_hash_cols = (config.engram_max_ngram_size - 1) * config.engram_n_heads

        self.wkv = BlockQuantLinear(
            self.n_hash_cols * config.engram_head_dim,
            config.hidden_size * (config.hc_mult + 1),
        )
        self.register_buffer(
            "q_weight",
            torch.ones(config.hc_mult, config.hidden_size, dtype=torch.bfloat16),
        )
        self.register_buffer(
            "k_weight",
            torch.ones(config.hc_mult, config.hidden_size, dtype=torch.bfloat16),
        )

    def forward(
        self,
        x: torch.Tensor,
        rows: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`x`: `[B, L, hc, dim]`; `rows`: `[B, L, n_hash_cols, head_dim]` bf16
        from the store; `token_mask`: `[B, L]`, False shuts the gate.
        """
        kv = self.wkv(rows.flatten(-2))
        # Sliced, not `split` — see `HyperConnections.mixes`.
        width = self.hc_mult * self.hidden_size
        key = kv[..., :width]
        value = kv[..., width : width + self.hidden_size]
        # q and k gains are only ever used as a product.
        weight = (self.q_weight.float() * self.k_weight.float()).reshape(1, width)
        # The gate and the gated add: `kernels/engram_gate.gated_add`, which is
        # `engram_short.py`'s arithmetic generalized to a ragged token count.
        # It computes, per (token, hc copy) and over `dim`,
        #
        #     rstd = rsqrt(mean(h^2) + eps) * rsqrt(mean(key^2) + eps)
        #     dot  = sum(h * weight * key) * rstd * dim**-0.5
        #     out  = h + sigmoid(sign(dot) * sqrt(max(|dot|, clamp))) * value
        #
        # — normalized per copy, not jointly over the copies. The torch form
        # this replaces is in the kernel's docstring; it agrees with it to fp32
        # under `nki.simulate` (max relative error 0 at `rows <= 128`, 2e-5
        # above it, which is the bf16 rounding of the result).
        #
        # `sign` is `2 * (dot >= 0) - 1`, which is `torch.where(dot < 0, ...)`
        # and *not* `copysign`: the latter has no lowering in this backend and
        # the graph dies at execution time with "FATAL: Unknown custom-call API
        # version enum value: 0", naming the node but not the reason. The two
        # differ only at negative zero.
        out = _nki_entry("engram_gate", "gated_add")(
            x.reshape(-1, width),
            key.reshape(-1, width),
            value.reshape(-1, self.hidden_size),
            weight,
        ).reshape(x.shape)
        if token_mask is not None:
            # The mask shut the gate, and a shut gate leaves the stream
            # untouched — so selecting `x` is the same value. Nothing on the
            # live path passes a mask (`forward` passes None); it exists for
            # the reference's image spans.
            out = torch.where(token_mask.unsqueeze(-1).unsqueeze(-1), out, x)
        return out


# =============================================================================
# Section 6: hyper-connections
# =============================================================================


class HyperConnections(nn.Module):
    """The `hc_mult = 4` parallel residual streams and the mixing between them.

    One instance per sublayer (`hc_attn_*` and `hc_ffn_*` in the checkpoint).
    `mixes` projects the flattened stream to 24 coefficients and splits them;
    `pre` collapses; `post` expands and mixes the residual in.

    The Sinkhorn normalization is the part that is easy to get subtly wrong, so
    it is written out against `vendor/kernel.py`'s `hc_split_sinkhorn`: 20
    alternating row/column normalizations of the `hc x hc` `comb` block, which
    drive it toward doubly stochastic. 20 is `hc_sinkhorn_iters` from the config
    and is not a convergence tolerance — the reference runs exactly that many,
    so matching it matters bit-for-bit.
    """

    def __init__(self, config: DeepseekV41Config) -> None:
        super().__init__()
        hc = config.hc_mult
        self.hc_mult = hc
        self.iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        self.hidden_size = config.hidden_size

        width = _mix_width(hc)
        self.register_buffer(
            "fn", torch.empty(width, hc * config.hidden_size, dtype=torch.float32)
        )
        self.register_buffer("base", torch.empty(width, dtype=torch.float32))
        self.register_buffer("scale", torch.empty(3, dtype=torch.float32))

    def mixes(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`x`: `[B, L, hc, dim]` -> `(pre [B,L,hc], post [B,L,hc],
        comb [B,L,hc,hc])`.

        Two kernels and no arithmetic in between. `fn` is the one weight the
        checkpoint stores fp32, handed to `kernels/blockquant.fp32w_linear` as
        stored and narrowed inside it — see that kernel for why neither the
        caller nor the traced forward can do the cast — and everything after it
        is `kernels/hc_mixes`: the stream statistic (over the whole flattened
        `hc * dim` copy set, one per token rather than one per copy), the three
        nonlinearities, and the 20-iteration Sinkhorn.

        The three groups are *not* interchangeable and the kernel is written
        against `hc_split_sinkhorn_kernel` in `vendor/kernel.py` rather than
        inferred: `pre` is a sigmoid plus eps, `post` is twice a sigmoid,
        `comb` is a softmax plus eps followed by `iters` column passes and
        `iters - 1` row passes. `_check_kernel_constants` asserts the three
        constants the kernel bakes against this config.
        """
        hc = self.hc_mult
        lead = x.shape[:2]
        rows = x.flatten(2).reshape(-1, hc * self.hidden_size)
        raw = nki_linear(rows, self.fn, None, "fp32w")
        pre, post, comb = _nki_entry("hc_mixes")(
            raw, rows, self.base.reshape(1, -1), self.scale.reshape(1, 3)
        )
        return (
            pre.reshape(*lead, hc),
            post.reshape(*lead, hc),
            comb.reshape(*lead, hc, hc),
        )

    @staticmethod
    def pre(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        """Collapse the copies into one sublayer input, weighted by `pre_mix`.

        `kernels/hc_collapse.py`, the shipped `HyperConnectCollapse` kernel,
        generalized to a ragged row count. The torch equivalent is
        `torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2).to(x.dtype)`; the
        kernel accumulates in fp32 and narrows on the store, which is the same
        arithmetic.
        """
        if x.dim() != 4:
            raise ValueError(f"hc_pre wants [batch, tokens, hc, dim], got {tuple(x.shape)}")
        if pre_mix.dtype is not torch.float32:
            raise TypeError(f"hc_pre wants fp32 coefficients, got {pre_mix.dtype}")
        return _nki_entry("hc_collapse")(x, pre_mix)

    @staticmethod
    def post(
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """Expand the sublayer output back out and mix the residual through
        `comb`. `x`: `[B,L,dim]`, `residual`: `[B,L,hc,dim]`.

        `kernels/hc_attn_out.py`, the shipped `HyperConnectSublayerOut` kernel,
        generalized to a ragged row count. The torch equivalent is

            broadcast = post.unsqueeze(-1) * x.unsqueeze(-2)
            mixed = torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
            (broadcast + mixed).type_as(x)

        and note which `hc` axis of `comb` is summed over: the first one, so
        the kernel reads `comb[i * hc + j]` against copy `i` into output copy
        `j`.
        """
        if post.dtype is not torch.float32 or comb.dtype is not torch.float32:
            raise TypeError(
                f"hc_post wants fp32 coefficients, got {post.dtype}/{comb.dtype}"
            )
        return _nki_entry("hc_attn_out")(x, residual, post, comb)


# =============================================================================
# Section 7: the decoder block
# =============================================================================


class Block(nn.Module):
    """One layer: hyper-connected attention, then hyper-connected MoE FFN.

    The coefficient handoff is the subtlety. Each sublayer's own `mixes` output
    is used by the *next* sublayer, so attention consumes what the previous
    layer's FFN produced, the FFN consumes what this attention produced, and
    `forward` returns the FFN's `pre` for the next layer.
    """

    #: Overridden by `DSparkBlock`, as `vendor/model.py`'s `Block` does it.
    attention_cls = Attention

    def __init__(
        self,
        config: DeepseekV41Config,
        layer_id: int,
        tp_size: int,
        tp_rank: int,
        max_batch_size: int = 1,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = self.attention_cls(config, layer_id, tp_size, max_batch_size)
        self.ffn = MoE(config, layer_id, tp_size, tp_rank)
        self.hc_attn = HyperConnections(config)
        self.hc_ffn = HyperConnections(config)
        self.engram = (
            Engram(config, layer_id) if layer_id in config.engram_layer_ids else None
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: torch.Tensor,
        pre_mix: torch.Tensor,
        image_mask: torch.Tensor | None,
        shared: SharedAttentionState,
        paged: PagedSpan,
        weight_store,
        tp_group=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One layer over the `hc_mult`-copy residual stream.

        `pre_mix` collapses the copies into this block's attention input. The
        handoff is the subtle part and is the reference's: each sub-block's own
        `mixes` output feeds the *next* one, so attention uses what the previous
        layer's FFN produced, the FFN uses what this attention produced, and the
        FFN's `pre` is returned for the next layer.

        Returns `(stream, next_pre_mix)`.
        """
        if int(start_pos) != paged.start_pos:
            raise ValueError(
                f"start_pos {int(start_pos)} disagrees with the PagedSpan's "
                f"{paged.start_pos}; the span owns the position now (it carries "
                "it as a tensor as well as the bucketed shape plan), so the two "
                "have to be the same sequence"
            )
        residual = x
        attn_pre, attn_post, attn_comb = self.hc_attn.mixes(x)
        y = HyperConnections.pre(x, pre_mix)
        y = self.attn_norm(y)
        y = self.attn(y, paged.pos, shared, paged, tp_group)
        x = HyperConnections.post(y, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_ffn.mixes(x)
        y = HyperConnections.pre(x, attn_pre)
        y = self.ffn_norm(y)
        y = self.ffn(y, image_mask, weight_store, tp_group)
        x = HyperConnections.post(y, residual, ffn_post, ffn_comb)
        return x, ffn_pre

    # ── The two halves the flat-batch path compiles ───────────────────────
    # `forward` above is the reference's shape — one `[B, L]` batch at a single
    # `start_pos` — and is what the component checks drive. The engine's path
    # is `DeepseekV41ForCausalLM._run_block`, which has to interleave the MoE's
    # host I/O between the two halves, so the halves are separate methods and
    # each is compiled on its own (`_block_entry`). The arithmetic is the same
    # expression in the same order.

    def attn_half(
        self, stream, pre_mix, slices, paged, shared: list, tp_group,
        out_residual=None, out_normed=None,
    ):
        """Everything from the stream to the MoE's normalized input.

        `out_residual` and `out_normed` are **the same two buffers for every
        layer**, and that is a memory decision rather than a style one. There
        is one of these graphs per block — there has to be, while the layer's
        weights are captured parameters, since dynamo guards on parameter
        identity — and the Neuron executor holds the tensors it last ran with
        from C++, where no Python reference drop can reach them. So each
        layer's graph was keeping its own copy of this function's inputs and
        outputs alive: measured at **92.8 KiB per token per layer**, which is
        36 GiB of a 24 GiB bank for the verifier's 9328-token prompt
        (`.autohelix/notes/iter-5.md` §4). Writing the two big results into
        buffers that every layer shares makes 43 executors retain one storage
        instead of 43.

        They are optional so that `Block.forward` — the reference-shaped path,
        which runs one layer at a time and has no accumulation to avoid — keeps
        working unchanged.

        Attention runs per sequence, because the window and the compressed
        history are per sequence; the hyper-connection coefficients and both
        norms are per token and run over the whole flat batch at once.

        `slices` is `(begin, length)` per span and deliberately carries **no
        start_pos**: this is a compiled entry point, and an int argument is a
        value dynamo specializes on, so a span list with the position in it
        would re-trace this graph at every decode step even though every shape
        in it is unchanged. The position comes off `span.pos` as a tensor
        instead — see `PagedSpan`.

        `shared` is **one `SharedAttentionState` per span**, zipped with
        `slices` and `paged`. It cannot be one instance for the batch: the
        sequence loop is in here, so a single instance would have span N
        overwriting what spans 0..N-1 published and every consumer layer would
        read the last span's index list for all of them. That is not a quiet
        accuracy bug — the published width depends on the span's own position
        (`read_len`), so a 256-wide list read against a 128-row window is an
        out-of-bounds device gather (`nrta status=1006`). See
        `.autohelix/notes/iter-4.md` §2a. `compress_kv`/`index_k` are
        layer-wide cache tensors and so are identical in every instance; only
        the index lists were ever per sequence.
        """
        batched = stream.unsqueeze(0)
        residual = batched
        attn_pre, attn_post, attn_comb = self.hc_attn.mixes(batched)
        normed = self.attn_norm(HyperConnections.pre(batched, pre_mix.unsqueeze(0)))

        # `zeros_like`, not `empty_like`: no `attn` call writes the rows past
        # `live_rows`, so an uninitialized buffer carried allocator garbage into
        # the hyper-connection mix and from there into the router.
        attn_out = torch.zeros_like(normed)
        for (begin, length), span, span_shared in zip(slices, paged, shared):
            attn_out[:, begin : begin + length] = self.attn(
                normed[:, begin : begin + length],
                span.pos,
                span_shared,
                span,
                tp_group,
            )
        batched = HyperConnections.post(attn_out, residual, attn_post, attn_comb)

        residual = batched
        ffn_pre, ffn_post, ffn_comb = self.hc_ffn.mixes(batched)
        normed = self.ffn_norm(HyperConnections.pre(batched, attn_pre))
        if out_residual is None:
            return normed, residual, ffn_pre, ffn_post, ffn_comb
        # Written, not returned: an output aliased to an input is what the
        # backend's `aliasing_output_rewrite` pass maps into `io_map`, so the
        # graph hands back the buffer it was given rather than a fresh
        # allocation. Same mechanism the paged KV writes already use.
        out_residual.copy_(residual)
        out_normed.copy_(normed)
        return out_normed, out_residual, ffn_pre, ffn_post, ffn_comb

    def ffn_half(self, ffn_out, residual, ffn_post, ffn_comb):
        """The MoE's output mixed back into the residual streams."""
        return HyperConnections.post(ffn_out, residual, ffn_post, ffn_comb)


def _block_entry(block: nn.Module, name: str):
    """`block.<name>` as its own compiled graph, one per block.

    **Explicit, rather than leaving it to dynamo's frame interception.** Letting
    dynamo find these frames by itself does work — until it decides not to, at
    which point the symptom is a layout error in a leaf function several calls
    away (see `_raise_dynamo_limits`) and the model has quietly moved to the
    host. Compiling them here means `fullgraph=True` holds the promise: if a
    break ever appears inside a block, the error says so.

    One graph per block, keyed on the module, because dynamo guards on
    parameter identity anyway — a bound method of layer 7 can never reuse layer
    6's graph. Cached on the block's `__dict__` rather than through
    `functools.lru_cache`, because `block.attn_half` builds a *new* bound-method
    object on every attribute access and would miss a function-keyed cache every
    time.
    """
    cache = block.__dict__.setdefault("_compiled_entries", {})
    entry = cache.get(name)
    if entry is None:
        entry = _compile_fn(getattr(block, name))
        cache[name] = entry
    return entry


def dspark_topk_idxs(
    window_rows: int, base: torch.Tensor, bsz: int, block_size: int, device
) -> torch.Tensor:
    """Which slots a draft query attends to. `vendor/model.py`'s
    `get_dspark_topk_idxs`, renumbered for the paged read.

    Every position in the draft block sees the same set: every row of the
    gathered window plus **all** `block_size` draft positions, with no causal
    mask among them — the draft block is scored jointly, which is the point of
    predicting a block at once.

    `base` is the absolute position of the gathered window's first row, which is
    negative early in a sequence: the window read is a fixed `window_rows` long
    so that its shape does not depend on the decode position (see
    `window_topk_idxs` for why that matters), and a row whose position is
    negative is marked empty here.
    """
    if window_rows <= 0:
        raise ValueError("the DSpark draft path only runs during decode")
    matrix = torch.arange(window_rows + block_size, device=device)
    live = (base + matrix >= 0) | (matrix >= window_rows)
    matrix = torch.where(live, matrix, EMPTY_SLOT)
    return matrix.int().view(1, 1, -1).expand(bsz, block_size, -1).contiguous()


class DSparkAttention(Attention):
    """Draft attention: the main stream's KV in the ring, the draft block's own
    KV appended, and no compressed KV at all (`compress_ratio == 0` for every
    `mtp` stage, since `compress_ratios` ends `[..., 0, 0, 0]`).

    The asymmetry worth noticing: `main_x` is the step's *real* tokens, and `x`
    is the `dspark_block_size = 5` draft positions. The real tokens' KV goes
    into the window (so the next step sees it); the draft positions' KV is used
    for this step only and never cached.

    **`start_pos` and `draft_pos` are two different positions, and conflating
    them is wrong as soon as speculation is on.** `start_pos` is where this
    step's real tokens begin — `main_x` has one row per scheduled token, which
    is 1 on a plain decode step, `1 + num_speculative_tokens` on a verify step
    and the whole prompt at prefill — and `draft_pos` is the position of the
    last *accepted* token, which is where the draft block hangs off. In
    `vendor/model.py` the main stream is always one row during decode, so the
    reference has only `start_pos` and derives the draft block's positions as
    `start_pos + seqlen`; that identity breaks under rejection sampling, where
    the step carries `1 + k` rows of which only a prefix was accepted.
    """

    def forward(
        self,
        x: torch.Tensor,
        start_pos: torch.Tensor,
        draft_pos: torch.Tensor,
        main_x: torch.Tensor,
        shared: SharedAttentionState | None = None,
        paged: PagedSpan | None = None,
        tp_group=None,
    ) -> torch.Tensor:
        if self.compress_ratio:
            raise ValueError(
                f"DSpark stage {self.layer_id} has compress_ratio "
                f"{self.compress_ratio}; the draft path has no compressed KV"
            )
        bsz, seqlen, _ = main_x.shape
        win, rd = self.window_size, self.rope_head_dim

        main_freqs = span_freqs(self.freqs_cis, start_pos, seqlen, main_x.device)
        main_kv = self.kv_norm(self.wkv(main_x))
        main_kv = torch.cat(
            [main_kv[..., :-rd], apply_rotary_emb(main_kv[..., -rd:], main_freqs)],
            dim=-1,
        )
        main_kv = act_quant_dequant(main_kv, 32)

        if paged is None:
            raise RuntimeError(
                f"DSpark stage {self.layer_id} needs the paged KV addressing; "
                "forward_spec must pass the PagedSpan it was given"
            )
        # `paged.at_zero`, not `start_pos == 0`: `start_pos` is a tensor now, and
        # a `.item()` here would be a device sync inside the graph. The span
        # carries the prefill/decode split as a bool, which dynamo guards on in
        # two variants rather than one per position (see `PagedSpan`).
        if paged.at_zero:
            # Prefill only seeds the window; the draft stream passes through.
            paged.write(self.window_kv_cache, main_kv.reshape(seqlen, self.head_dim))
            return x

        bsz, block_size, _ = x.shape
        # The draft block starts one past the last accepted token, not one past
        # this step's last row: under speculation the rows after the accepted
        # prefix hold rejected drafts, and drafting from them would place the
        # block at a position the sequence never reached. See the class
        # docstring.
        freqs = span_freqs(self.freqs_cis, draft_pos + 1, block_size, x.device)

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        q = torch.cat(
            [q[..., :-rd], apply_rotary_emb(q[..., -rd:], freqs)], dim=-1
        )
        kv = self.kv_norm(self.wkv(x))
        kv = torch.cat(
            [kv[..., :-rd], apply_rotary_emb(kv[..., -rd:], freqs)], dim=-1
        )
        kv = act_quant_dequant(kv, 32)

        # This step's real tokens' KV joins the window at their own slots, then
        # the window is gathered back out of the paged allocation and the draft
        # block's own KV is concatenated after it.
        #
        # Every row is written, including the ones after the accepted prefix on
        # a verify step: their slots are rewritten by the next step, which
        # starts at the accepted position and is `1 + k` rows long, so it covers
        # every slot this one may have written speculatively. That is the same
        # argument vLLM's own paged KV relies on, and the window *read* below
        # never looks past `draft_pos`, so nothing reads a speculative slot.
        paged.write(self.window_kv_cache, main_kv.reshape(seqlen, self.head_dim))
        # Fixed-size window read, for the same reason as `Attention._window_kv`:
        # a length that grows with `start_pos` is a tensor shape, and a shape
        # that changes every decode step recompiles the stage every step.
        window_rows = win
        base = draft_pos - (win - 1)
        positions = (base + torch.arange(window_rows, device=x.device)).clamp_min(0)
        window_kv = paged.gather(
            self.window_kv_cache.view(-1, self.head_dim), positions
        ).reshape(1, window_rows, self.head_dim)
        topk = dspark_topk_idxs(window_rows, base, bsz, block_size, x.device)
        kv = torch.cat([window_kv, kv], dim=1)

        o = sparse_attn(q, kv, self.attn_sink, topk, self.softmax_scale)
        o = torch.cat(
            [o[..., :-rd], apply_rotary_emb(o[..., -rd:], freqs, True)], dim=-1
        )
        o = o.reshape(bsz * block_size, self.n_local_groups, -1)
        o = torch.cat(
            [
                self.wo_a.grouped(o[:, g], g, self.o_lora_rank)
                for g in range(self.n_local_groups)
            ],
            dim=-1,
        ).reshape(bsz, block_size, -1)
        out = self.wo_b(o)
        if self.world_size > 1 and tp_group is not None:
            out = tp_all_reduce(out.float(), tp_group)
            out = out.to(x.dtype)
        return out


class DSparkBlock(Block):
    """A DSpark stage, stored under `mtp.<stage>`.

    Same hyper-connected structure as a backbone `Block` over a 128-expert pool
    at top-3, with three additions across the three stages: stage 0 projects the
    concatenated attention inputs of layers 37/38/39 down to one hidden
    (`main_proj`/`main_norm`), and the last stage carries the final norm, the
    Markov head (a rank-256 embedding plus its own unembedding, added into the
    draft logits) and the confidence head.
    """

    attention_cls = DSparkAttention

    def __init__(
        self,
        config: DeepseekV41Config,
        stage: int,
        tp_size: int,
        tp_rank: int,
        max_batch_size: int = 1,
    ) -> None:
        layer_id = config.num_hidden_layers + stage
        super().__init__(config, layer_id, tp_size, tp_rank, max_batch_size)
        self.stage = stage
        self.block_size = config.dspark_block_size
        self.noise_token_id = config.dspark_noise_token_id
        self.hc_mult = config.hc_mult
        #: The *shared* top-level embedding and LM head, as the reference wires
        #: them (`self.mtp[-1].embed = self.embed`). Set by the model, not
        #: owned here, so the checkpoint has no `mtp.*.embed` key.
        self.embed_fn = None
        self.head_fn = None

        if stage == 0:
            self.main_proj = BlockQuantLinear(
                config.hidden_size * len(config.dspark_target_layer_ids),
                config.hidden_size,
            )
            self.main_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        if stage == config.num_nextn_predict_layers - 1:
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.markov_head = DSparkMarkovHead(config, tp_size, tp_rank)
            self.confidence_head = DSparkConfidenceHead(config)


    def forward(
        self,
        x: torch.Tensor,
        start_pos: torch.Tensor,
        draft_pos: torch.Tensor,
        pre_mix: torch.Tensor,
        main_x: torch.Tensor,
        shared: SharedAttentionState,
        paged: PagedSpan,
        weight_store,
        tp_group=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One draft stage. Prefill only seeds the window and passes through."""
        if paged.at_zero:
            self.attn(x, start_pos, draft_pos, main_x, shared, paged, tp_group)
            return x, pre_mix

        residual = x
        attn_pre, attn_post, attn_comb = self.hc_attn.mixes(x)
        y = HyperConnections.pre(x, pre_mix)
        y = self.attn_norm(y)
        y = self.attn(y, start_pos, draft_pos, main_x, shared, paged, tp_group)
        x = HyperConnections.post(y, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_ffn.mixes(x)
        y = HyperConnections.pre(x, attn_pre)
        y = self.ffn_norm(y)
        # Drafts are text, so no VL routing bias. 128 experts at top-3 here,
        # not the backbone's 384 at top-6 — `_moe_config` picks that off the
        # layer id.
        y = self.ffn(y, None, weight_store, tp_group)
        x = HyperConnections.post(y, residual, ffn_post, ffn_comb)
        return x, ffn_pre

    def forward_embed(
        self, main_hidden: torch.Tensor, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage 0 only: build the draft stream and the main-stream input.

        The draft block is `[accepted token, noise, noise, ...]` —
        `dspark_noise_token_id` fills the positions the draft has not predicted
        yet, which is how the block is scored jointly in one pass.
        """
        if self.stage != 0:
            raise ValueError("forward_embed belongs to DSpark stage 0")
        main_x = self.main_norm(self.main_proj(main_hidden))
        draft_ids = input_ids.new_full(
            (input_ids.shape[0], self.block_size), self.noise_token_id
        )
        draft_ids[:, 0] = input_ids
        x = self.embed_fn(draft_ids.reshape(-1)).reshape(
            input_ids.shape[0], self.block_size, -1
        )
        x = x.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        return x, main_x

    def forward_head(
        self,
        x: torch.Tensor,
        pre_mix: torch.Tensor,
        input_ids: torch.Tensor,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Last stage only: draft logits, draft tokens and their confidence.

        The loop is sequential on purpose and is not an optimization target: the
        Markov bias for draft position `i + 1` is conditioned on the token just
        sampled at position `i`, so each step needs the previous one's output.
        """
        if self.markov_head is None:
            raise ValueError("forward_head belongs to the last DSpark stage")
        x = HyperConnections.pre(x, pre_mix)
        logits = self.head_fn(self.norm(x).reshape(-1, x.shape[-1])).reshape(
            x.shape[0], self.block_size, -1
        )
        output_ids = input_ids.new_empty(input_ids.shape[0], self.block_size + 1)
        output_ids[:, 0] = input_ids
        markov_embeds = []
        for i in range(self.block_size):
            bias, markov_embed = self.markov_head(output_ids[:, i])
            logits[:, i] = logits[:, i] + bias
            markov_embeds.append(markov_embed)
            output_ids[:, i + 1] = sample_tokens(logits[:, i], temperature)
        confidence = self.confidence_head(x, torch.stack(markov_embeds, dim=1))
        return output_ids, logits, confidence


def sample_tokens(logits: torch.Tensor, temperature: float = 0.0):
    """`vendor/model.py`'s `sample`. Greedy at temperature 0, which is what the
    verifier asks for (`TEMPERATURE = 0.0`).
    """
    if temperature == 0:
        return logits.argmax(dim=-1)
    probabilities = torch.softmax(
        logits / max(temperature, 1e-5), dim=-1, dtype=torch.float32
    )
    return probabilities.div_(
        torch.empty_like(probabilities).exponential_(1)
    ).argmax(dim=-1)


class DSparkMarkovHead(nn.Module):
    """A rank-256 token-conditioned logit bias plus the embedding it reads."""

    def __init__(
        self, config: DeepseekV41Config, tp_size: int, tp_rank: int = 0
    ) -> None:
        super().__init__()
        rank = config.dspark_markov_rank
        self.vocab_size = config.vocab_size
        self.part_vocab_size = config.vocab_size // tp_size
        self.register_buffer(
            "embed_weight",
            torch.empty(self.part_vocab_size, rank, dtype=torch.bfloat16),
        )
        self.register_buffer(
            "head_weight",
            torch.empty(self.part_vocab_size, rank, dtype=torch.bfloat16),
        )
        self.world_size = tp_size
        # This rank's slice of the vocab, and it has to be *this* rank's: both
        # tables are sharded on dim 0 (`_build_shard_plan`), so a hardcoded 0
        # made every rank above the first subtract the wrong offset, look up
        # rows it does not own and mask out the ones it does — a wrong Markov
        # bias on every draft token at TP > 1, which costs acceptance rate
        # without ever being visible at TP = 1. Inert at one rank, where
        # `rank_in_group` is 0 anyway.
        self.rank_in_group = tp_rank

    def forward(self, token_ids: torch.Tensor):
        """A rank-256 embedding of the previous draft token, and the logit bias
        it implies. Both tables are vocab-parallel, like the ones they mirror.
        """
        start = self.rank_in_group * self.part_vocab_size
        end = start + self.part_vocab_size
        local = (token_ids - start).masked_fill(
            (token_ids < start) | (token_ids >= end), 0
        )
        embed = F.embedding(local, self.embed_weight)
        embed = embed.masked_fill(
            ((token_ids < start) | (token_ids >= end)).unsqueeze(-1), 0
        )
        if self.world_size > 1 and self._tp_group is not None:
            embed = tp_all_reduce(embed, self._tp_group)
        logits = nki_linear(embed, self.head_weight, None, "bf16").float()
        if self.world_size > 1 and self._tp_group is not None:
            logits = tp_all_gather_last(logits, self._tp_group)
        return logits, embed

    #: Set by the model, which owns the process group.
    _tp_group = None


class DSparkConfidenceHead(nn.Module):
    """Scores a draft position from its hidden state and its Markov embedding."""

    def __init__(self, config: DeepseekV41Config) -> None:
        super().__init__()
        width = config.hidden_size + config.dspark_markov_rank
        self.register_buffer("proj_weight", torch.empty(1, width, dtype=torch.bfloat16))

    def forward(
        self, hidden: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        joined = torch.cat([hidden, markov_embed], dim=-1)
        return nki_linear(joined, self.proj_weight, None, "bf16").float().squeeze(-1)


# =============================================================================
# Section 8: the model
# =============================================================================


class DeepseekV41ForCausalLM(nn.Module):
    """DeepSeek-V4.1-Flash: embed -> expand to 4 copies -> 40 blocks ->
    collapse -> norm -> head, with the three DSpark stages hanging off the
    attention inputs of layers 37/38/39.
    """

    def __init__(self, config: DeepseekV41Config) -> None:
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config

        tp_group = get_tp_group()
        self.tp_group = tp_group
        self.world_size = tp_group.world_size
        self.rank = tp_group.rank_in_group

        self.hc_mult = config.hc_mult
        self.part_vocab_size = config.vocab_size // self.world_size

        self.register_buffer(
            "embed_weight",
            torch.empty(self.part_vocab_size, config.hidden_size, dtype=torch.bfloat16),
        )
        self.layers = nn.ModuleList(
            [
                Block(
                    config, layer_id, self.world_size, self.rank, self._max_batch_size()
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.register_buffer(
            "head_weight",
            torch.empty(self.part_vocab_size, config.hidden_size, dtype=torch.bfloat16),
        )
        self.mtp = nn.ModuleList(
            [
                DSparkBlock(
                    config, stage, self.world_size, self.rank, self._max_batch_size()
                )
                for stage in range(config.num_nextn_predict_layers)
            ]
        )

        #: Set by `weight_loader.load_weights`. The handle on the streamed
        #: families; `forward` reaches the routed experts and the engram rows
        #: through it.
        self.weight_store = None

        #: Set by `bind_kv_cache`, which is the first hook that runs with the
        #: tokenizer reachable. Produces the row indices `Engram` looks up; see
        #: `NgramHashState` for why it cannot be derived from the weights.
        self.ngram_hash: NgramHashState | None = None

        #: `(rows, dtype, device) -> (stream, residual, normed)`, filled by
        #: `_layer_io`. One set of per-layer I/O buffers shared by every block,
        #: which is what keeps the live set off `layers x tokens`.
        self._layer_io_slots: dict[tuple, tuple[torch.Tensor, ...]] = {}

        #: `(shape, dtype, device) -> buffer`, filled by `_engram_rows`. The
        #: same idea as `_layer_io_slots` applied to the gathered n-gram rows:
        #: both engram layers hand their lookup to the same device buffer, so
        #: the engram graph's retained input binding does not grow with the
        #: number of engram layers.
        self._engram_row_slots: dict[tuple, torch.Tensor] = {}

        # On-device sampling, if the runner asked for it. This is not a
        # preference the model gets to express: `NeuronModelRunner` reads
        # `neuron_config.on_device_sampling_config` itself
        # (`neuron_model_runner.py:488`) and, when it is set, *interprets the
        # model's return value as sampled token ids*. A model that returns
        # logits under that configuration produces a scheduler crash with
        # nothing in it about sampling —
        #
        #     TypeError: 'float' object is not iterable
        #       ... vllm/v1/request.py:232, in append_output_token_ids
        #
        # which is the logits tensor arriving where a token id list belongs.
        # Same contract as `qwen3/model.py`: return `logits` when it is off,
        # `(sampled_tokens, gathered_logits)` when it is on.
        ods = getattr(self.neuron_config, "on_device_sampling_config", None)
        self.on_device_sampling_config = ods
        self.sampler = None
        if ods is not None:
            from vllm_neuron.nn.sampler import Sampler

            self.sampler = Sampler(
                ods, process_group=getattr(tp_group, "device_group", None)
            )

        self._shard_plan = self._build_shard_plan()
        self._check_kernel_constants()

    # ── Construction ──────────────────────────────────────────────────────

    def _check_kernel_constants(self) -> None:
        """The constants the kernels bake in; make sure the config agrees.

        `hc_expand` emits the one-hot `pre_mix` and the broadcast width as part
        of its instruction stream, `hc_mixes` has the Sinkhorn iteration count
        as a loop bound and the two epsilons as instruction immediates, and
        `engram_gate` has the normalization epsilon and the gate's clamp floor
        the same way, so none of them can be a runtime argument. A disagreement
        would silently produce the wrong number of residual streams, the wrong
        number of normalization passes or an unnormalized gate — exactly the
        class of thing `config-matches-official` cannot see.
        """
        from .kernels import engram_gate, hc_expand, hc_mixes

        baked = {
            "hc_mult": (hc_expand.HC, self.config.hc_mult),
            "hc_sinkhorn_iters": (
                hc_mixes.SINKHORN_ITERS,
                self.config.hc_sinkhorn_iters,
            ),
            "hc_eps": (hc_mixes.HC_EPS, self.config.hc_eps),
            "rms_norm_eps": (hc_mixes.NORM_EPS, self.config.rms_norm_eps),
            "rms_norm_eps (engram_gate)": (
                engram_gate.EPS,
                self.config.rms_norm_eps,
            ),
            "engram clamp": (engram_gate.CLAMP, Engram.CLAMP_VALUE),
        }
        for name, (in_kernel, in_config) in baked.items():
            if in_kernel != in_config:
                raise ValueError(
                    f"kernels/ is built for {name}={in_kernel} but this config "
                    f"says {in_config}"
                )

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None = None,
    ) -> DeepseekV41ForCausalLM:
        return cls(DeepseekV41Config.from_configs(hf_config, neuron_config))

    # ── Sharding ──────────────────────────────────────────────────────────

    def _build_shard_plan(self) -> dict[str, ShardRule]:
        """One rule per checkpoint key, keyed by this model's buffer name.

        Built here rather than inline at each buffer so the whole plan is one
        object that can be printed and checked against the checkpoint shapes
        (`verify_shard_plan`). The axes follow `vendor/model.py`'s
        Column/RowParallelLinear choices, which is what keeps the four ranks'
        partial sums addable.
        """
        config = self.config
        plan: dict[str, ShardRule] = {
            "embed.weight": ShardRule(0, "vocab-parallel embedding"),
            "head.weight": ShardRule(0, "vocab-parallel LM head"),
            "norm.weight": REPLICATE,
        }

        def add_attention(prefix: str, layer_id: int) -> None:
            plan[f"{prefix}.attn_sink"] = ShardRule(
                0, "one bias per query head; shards with the heads of wq_b"
            )
            plan[f"{prefix}.q_norm.weight"] = REPLICATE
            plan[f"{prefix}.kv_norm.weight"] = REPLICATE
            # wq_a: 5120 -> 1280, no head axis yet.
            plan[f"{prefix}.wq_a.weight"] = REPLICATE
            plan[f"{prefix}.wq_a.scale"] = REPLICATE
            # wq_b: 1280 -> 64*512. Column-parallel on the head axis.
            plan[f"{prefix}.wq_b.weight"] = ShardRule(0, "column-parallel: query heads")
            plan[f"{prefix}.wq_b.scale"] = ShardRule(0, "follows wq_b.weight dim 0")
            # One shared latent KV for every head: nothing to shard.
            plan[f"{prefix}.wkv.weight"] = REPLICATE
            plan[f"{prefix}.wkv.scale"] = REPLICATE
            # wo_a: block-diagonal over o_groups. Rank r owns groups
            # [r*G/ws, (r+1)*G/ws), i.e. rows [.. * o_lora_rank ..) — the same
            # rows a column-parallel split of out_features gives, which is why
            # the group partition has to agree with the head partition: group g
            # projects heads [8g, 8g+8).
            plan[f"{prefix}.wo_a.weight"] = ShardRule(
                0, "column-parallel over o_groups; must agree with the head split"
            )
            plan[f"{prefix}.wo_a.scale"] = ShardRule(0, "follows wo_a.weight dim 0")
            # wo_b: row-parallel, so each rank holds a partial sum.
            plan[f"{prefix}.wo_b.weight"] = ShardRule(
                1, "row-parallel: the o_groups*o_lora_rank contraction"
            )
            plan[f"{prefix}.wo_b.scale"] = ShardRule(1, "follows wo_b.weight dim 1")

            if layer_id in config.kv_source_layer_ids:
                plan[f"{prefix}.compressor.norm.weight"] = REPLICATE
                plan[f"{prefix}.compressor.wkv.weight"] = REPLICATE
                if config.compress_ratios[layer_id] > 1:
                    plan[f"{prefix}.compressor.wgate.weight"] = REPLICATE
            if layer_id in config.index_source_layer_ids:
                plan[f"{prefix}.indexer.wq_b.weight"] = ShardRule(
                    0, "column-parallel: index heads"
                )
                plan[f"{prefix}.indexer.wq_b.scale"] = ShardRule(
                    0, "follows indexer.wq_b.weight dim 0"
                )
                plan[f"{prefix}.indexer.weights_proj.weight"] = ShardRule(
                    0, "one weight per index head; shards with indexer.wq_b"
                )
                if layer_id in config.kv_source_layer_ids:
                    plan[f"{prefix}.indexer.wk.weight"] = REPLICATE
                    plan[f"{prefix}.indexer.k_norm.weight"] = REPLICATE

        def add_block(prefix: str, layer_id: int) -> None:
            plan[f"{prefix}.attn_norm.weight"] = REPLICATE
            plan[f"{prefix}.ffn_norm.weight"] = REPLICATE
            for name in ("hc_attn", "hc_ffn"):
                plan[f"{prefix}.{name}_fn"] = REPLICATE
                plan[f"{prefix}.{name}_base"] = REPLICATE
                plan[f"{prefix}.{name}_scale"] = REPLICATE
            # The gate scores every expert on every rank: selection has to be
            # identical across ranks for the per-rank expert partition to cover
            # the routed set exactly once.
            plan[f"{prefix}.ffn.gate.weight"] = REPLICATE
            plan[f"{prefix}.ffn.gate.bias"] = REPLICATE
            plan[f"{prefix}.ffn.gate.bias_vl"] = REPLICATE
            # The shared expert runs after the routed all-reduce, on the full
            # hidden state, so it is replicated rather than split.
            for name in ("w1", "w2", "w3"):
                plan[f"{prefix}.ffn.shared_experts.{name}.weight"] = REPLICATE
                plan[f"{prefix}.ffn.shared_experts.{name}.scale"] = REPLICATE
            add_attention(f"{prefix}.attn", layer_id)
            if layer_id in config.engram_layer_ids:
                # The reference's Engram.wkv is a plain Linear (replicated); the
                # `engram_short` kernel instead shards its `ngram` contraction
                # 4 ways and all-reduces the key before forming the gate.
                # Replicated here: 157 MiB per table is affordable, and it keeps
                # the gate — which is nonlinear in the *summed* key — out of the
                # collective.
                plan[f"{prefix}.engram.wkv.weight"] = REPLICATE
                plan[f"{prefix}.engram.wkv.scale"] = REPLICATE
                plan[f"{prefix}.engram.q_weight"] = REPLICATE
                plan[f"{prefix}.engram.k_weight"] = REPLICATE

        for layer_id in range(config.num_hidden_layers):
            add_block(f"layers.{layer_id}", layer_id)
        for stage in range(config.num_nextn_predict_layers):
            add_block(f"mtp.{stage}", config.num_hidden_layers + stage)

        plan["mtp.0.main_proj.weight"] = REPLICATE
        plan["mtp.0.main_proj.scale"] = REPLICATE
        plan["mtp.0.main_norm.weight"] = REPLICATE
        last = config.num_nextn_predict_layers - 1
        plan[f"mtp.{last}.norm.weight"] = REPLICATE
        plan[f"mtp.{last}.markov_head.embed.weight"] = ShardRule(
            0, "vocab-parallel, like the embedding it mirrors"
        )
        plan[f"mtp.{last}.markov_head.head.weight"] = ShardRule(
            0, "vocab-parallel, like the LM head it mirrors"
        )
        plan[f"mtp.{last}.confidence_head.proj.weight"] = REPLICATE
        return plan

    def shard_parameter(
        self, key: str, tensor: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """This rank's slice of one checkpoint tensor, on `device`.

        Raises on a key with no rule rather than guessing: replicating a tensor
        that should have been split, or splitting one that should not, both
        produce a model that runs and is wrong, which is the failure mode worth
        the most noise.
        """
        rule = self._shard_plan.get(key)
        if rule is None:
            raise KeyError(
                f"{key!r} has no sharding rule. Every checkpoint key this model "
                "loads needs one — add it to _build_shard_plan rather than "
                "defaulting to replicate."
            )
        if rule.dim is None or self.world_size == 1:
            return tensor.to(device)

        extent = tensor.shape[rule.dim]
        if extent % self.world_size:
            raise ValueError(
                f"{key}: dim {rule.dim} has extent {extent}, which is not "
                f"divisible by world_size {self.world_size} ({rule.reason})"
            )
        width = extent // self.world_size
        return tensor.narrow(rule.dim, self.rank * width, width).to(device)

    def verify_shard_plan(self, index) -> list[str]:
        """Check the plan against the checkpoint: every key present, every
        sharded axis divisible, every resulting shape matching the buffer it
        will be assigned to. Returns the problems; empty means consistent.

        Exists because the plan is the part of this port that a reviewer cannot
        check by reading — the axes are only right in combination.
        """
        from .weight_loader import build_weight_mappings

        problems: list[str] = []
        buffers = dict(self.named_buffers())
        for name, keys in build_weight_mappings(self).items():
            key = keys if isinstance(keys, str) else keys[0]
            if key not in index:
                problems.append(f"{key}: not in the checkpoint")
                continue
            rule = self._shard_plan.get(name)
            if rule is None:
                problems.append(f"{name}: no sharding rule")
                continue
            shape = list(index.ref(key).shape)
            if rule.dim is not None:
                if shape[rule.dim] % self.world_size:
                    problems.append(
                        f"{key}: dim {rule.dim} extent {shape[rule.dim]} is not "
                        f"divisible by {self.world_size} ({rule.reason})"
                    )
                    continue
                shape[rule.dim] //= self.world_size
            target = buffers.get(_buffer_name(name))
            if target is None:
                problems.append(f"{name}: no buffer named {_buffer_name(name)!r}")
            elif list(target.shape) != shape:
                problems.append(
                    f"{name}: checkpoint shard is {shape}, buffer is "
                    f"{list(target.shape)}"
                )
        return problems

    # ── KV cache ──────────────────────────────────────────────────────────

    def get_kv_spec(self) -> KVSpec:
        """One `LayerSpec` per attention, for the 128-slot sliding window.

        What is and is not in here, because the mapping is a design decision
        rather than a transcription:

        *In* — the window storage. Every one of the 43 attentions (40 backbone +
        3 DSpark) attends over a window of `sliding_window = 128` raw KV
        positions, one latent head of `head_dim = 512`. That is per-layer,
        per-sequence, bounded, and written once per token: exactly what vLLM's
        block manager handles.

        **`sliding_window_size` is deliberately left None, so this is a
        `FullAttentionSpec` and not a `SlidingWindowSpec`.** Declaring the
        window would be describing the attention honestly and *managing* the
        blocks wrongly, because vLLM reads that field as a licence to recycle:

          - `SlidingWindowManager.remove_skipped_blocks` frees every block
            older than the window on each step and overwrites the request's
            entries with the shared **null block**. The window layers would not
            miss them, but this model's own per-slot caches — compressed
            latents, index keys, the compressor's pooling inputs, the n-gram
            ids — live in the same paged space at the same slots, and they are
            read at *every* earlier position, not just the last 128. The sum is
            exact: `get_num_skipped_tokens(2008)` is
            `2008 - sliding_window + 1 = 1881`, so 58 blocks are freed and the
            needle's first 1,856 positions resolve to block 0. No error, just a
            wrong answer, which is the worse failure of the two.
          - `NeuronModelRunner` *trims* a sliding-window group's decode block
            table to `num_swa_blocks` (8, at this window and block size) and
            rebases it by `swa_kv_pos_offset`. `_paged_spans` reads the
            untrimmed clone and so survives that, but there is no reason to
            keep asking for a table the model then has to work around.

        The cost of `FullAttentionSpec` is that a sequence's whole context
        stays resident, which this port needs anyway and `_KV_CAP_MAX_SLOTS`
        already says: every cache it keeps is addressed by absolute position
        within a sequence's own context. The window being 128 is enforced where
        it belongs — in the attention itself, which reads 128 slots back from
        `pos` — not by asking the allocator to throw the rest away.

        *Out* — the compressed KV, the index keys and the compressor's
        per-token pooling inputs. Those are **shared across layers**, not
        per-layer: the four `kv_source_layer_ids` produce them and every other
        layer reads the same cache. A `KVSpec` entry describes storage owned by
        one layer, so expressing a cache that four layers write and 36 read
        would either duplicate it 40 times (40x the memory for one copy of the
        data) or lie about who owns it.

        They are therefore model-owned buffers allocated in `bind_kv_cache` —
        but **paged the same way and addressed by the same `block_table`**, one
        row per slot of the runtime's own allocation rather than one row per
        batch index. That is not a memory decision, it is a correctness one:
        prefix caching is on by default here, so a cache row has to be
        identified by *which tokens* produced it, not by where its request
        happens to sit in this step's running batch. See `PagedSpan`.
        """
        dtype = self.config.torch_dtype
        layers = []
        for layer_id, block in enumerate(self.layers):
            layers.append(
                LayerSpec(
                    name=f"layers.{layer_id}.attn",
                    num_kv_heads=1,
                    head_size=block.attn.head_dim,
                    dtype=dtype,
                    # None on purpose; see this method's docstring. The window
                    # is 128 and the attention enforces it.
                    sliding_window_size=None,
                    chunk_size=None,
                )
            )
        for stage, block in enumerate(self.mtp):
            layers.append(
                LayerSpec(
                    name=f"mtp.{stage}.attn",
                    num_kv_heads=1,
                    head_size=block.attn.head_dim,
                    dtype=dtype,
                    sliding_window_size=None,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        """Bind the runtime's window-KV allocation, and allocate the shared
        compressed/index caches the spec deliberately leaves out.
        """
        attentions = [(f"layers.{i}.attn", b.attn) for i, b in enumerate(self.layers)]
        attentions += [(f"mtp.{s}.attn", b.attn) for s, b in enumerate(self.mtp)]

        for name, attn in attentions:
            if name not in kv_caches:
                raise KeyError(
                    f"KV cache for {name} was not allocated; get_kv_spec() "
                    f"declared it, so this is a runtime/spec mismatch"
                )
            # The window is one latent KV, so the two halves of the runtime's
            # (k, v) pair are the same store read twice: this architecture's
            # "KV" is a single 512-wide vector per position, not a k/v split.
            attn.window_kv_cache = kv_caches[name][0]

        max_len = self._max_sequence_length()
        device = self.norm.weight.device
        # The runtime's own allocation fixes the slot space: a window cache is
        # `[num_blocks, num_kv_heads, block_size, head_size]`, so a slot id runs
        # to `num_blocks * block_size`. Everything this model caches per
        # position is sized the same way and addressed by the same
        # `block_table`, which is what makes it correct under prefix caching
        # (see `PagedSpan`).
        window = self.layers[0].attn.window_kv_cache
        num_slots = window.shape[0] * window.shape[2]
        # What this costs, logged rather than estimated. The runtime's memory
        # table reports one `Tensors` figure per NeuronCore with no breakdown
        # (`.autohelix/notes/iter-5.md` §6), and HBM is the binding constraint
        # on this port, so the two allocations the model can actually account
        # for should say how large they are.
        kv_bytes = sum(
            t.numel() * t.element_size()
            for pair in kv_caches.values()
            for t in pair
        )
        own_bytes = 0
        for name, attn in attentions:
            if attn.is_kv_source:
                attn.compress_kv_cache = torch.zeros(
                    num_slots,
                    attn.head_dim,
                    dtype=self.config.torch_dtype,
                    device=device,
                )
                attn.index_k_cache = torch.zeros(
                    num_slots,
                    self.config.index_head_dim,
                    dtype=self.config.torch_dtype,
                    device=device,
                )
                if attn.compressor is not None:
                    attn.compressor.allocate_state(device, num_slots)
                own_bytes += sum(
                    t.numel() * t.element_size()
                    for t in (attn.compress_kv_cache, attn.index_k_cache)
                )
            # RoPE tables last, so an indexer picks up the index-key cache that
            # was just allocated above.
            attn.build_freqs_cis(max_len, device)

        if self.config.engram_layer_ids and self.ngram_hash is None:
            self.ngram_hash = self._build_ngram_hash(device)
            self.ngram_hash.allocate_state(device, num_slots)

        logger.info(
            "deepseek_v41_timeshare: %d KV slots — vLLM's window caches are "
            "%.2f GiB over %d layers, this model's own per-slot caches "
            "(compressed KV, index keys) are %.2f GiB. Neither is in "
            "`get_kv_spec`'s accounting for the second of those; see "
            "model.get_kv_spec",
            num_slots,
            kv_bytes / 2**30,
            len(kv_caches),
            own_bytes / 2**30,
        )


    def allocate_staging_buffers(self, device: torch.device) -> None:
        """Materialize the per-layer expert staging slots on `device`.

        Called from `load_weights`, which is **before** the runtime's
        `model.to(device)`. It has to be: these buffers carry no checkpoint key,
        so nothing assigns them during loading, and `Module.to()` cannot move a
        meta tensor —

            NotImplementedError: Cannot copy out of meta tensor; no data!

        One expert's worth of HBM per layer (17.9 MiB of fp4 weights plus 1.1
        MiB of scales), refilled per forward by `MoE._stage`. 43 layers of that
        is ~0.8 GiB, against the 276 GiB the 384-per-layer pool would need.
        """
        for block in list(self.layers) + list(self.mtp):
            block.ffn.expert_slot.allocate(device)

    def _build_ngram_hash(self, device) -> NgramHashState:
        """Build the n-gram hash state, which needs the *tokenizer*.

        The compressed token map is a function of the tokenizer's vocabulary —
        every hash multiplier derives from its size — so this is the one part of
        the model that cannot be built from the config and the checkpoint alone.
        The tokenizer path comes from the active vLLM config, which is the same
        directory the weights came from.
        """
        from transformers import AutoTokenizer
        from vllm.config import get_current_vllm_config

        model_config = get_current_vllm_config().model_config
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.tokenizer or model_config.model
        )
        state = NgramHashState(self.config, tokenizer)
        return state.to(device)

    def _max_sequence_length(self) -> int:
        """The longest sequence this engine will serve.

        Read from vLLM's config, not from `max_position_embeddings`. The
        difference is not academic: this model declares 1,048,576 positions, and
        sizing the RoPE tables and the compressed-KV caches for that allocates
        ~11 GiB of RoPE tables and ~11 GiB of caches per rank, which overruns
        the 24 GiB bank and takes the Neuron runtime down with
        `tensor_write: Assertion 'tensor->sto != NULL' failed` — a native
        assertion, with no Python frame to read. `max_model_len` for the
        verifier's run is 8224.

        `NeuronConfig` carries no length field, so this goes through
        `get_current_vllm_config()`, which is the same route
        `qwen3/model.py:_load_kv_cache_scales` uses. Raises rather than guessing:
        a wrong guess here is the failure above.
        """
        from vllm.config import get_current_vllm_config

        try:
            return int(get_current_vllm_config().model_config.max_model_len)
        except Exception as exc:  # noqa: BLE001 - re-raised with the reason
            raise RuntimeError(
                "cannot determine max_model_len from the active vLLM config, "
                "and this model's max_position_embeddings (1,048,576) is far "
                "too large to size buffers with. Set max_model_len on the "
                f"engine. ({type(exc).__name__}: {exc})"
            ) from exc

    def _max_batch_size(self) -> int:
        """The most concurrent sequences, for the compressor's partial-group
        state. From vLLM's scheduler config, same reasoning as above.
        """
        from vllm.config import get_current_vllm_config

        try:
            return int(get_current_vllm_config().scheduler_config.max_num_seqs)
        except Exception:  # noqa: BLE001 - a single sequence is the safe floor
            return 1

    # ── vLLM's generic interface ──────────────────────────────────────────

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Vocab-parallel embedding: look up this rank's rows, zero the rest,
        and all-reduce so every rank ends with the same full embedding.
        """
        start = self.rank * self.part_vocab_size
        end = start + self.part_vocab_size
        local = input_ids - start
        off_rank = (input_ids < start) | (input_ids >= end)
        local = local.masked_fill(off_rank, 0)
        out = F.embedding(local, self.embed_weight)
        out = out.masked_fill(off_rank.unsqueeze(-1), 0)
        if self.world_size > 1:
            out = tp_all_reduce(out, self.tp_group)
        return out

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Vocab-parallel unembedding, gathered to full width.

        The widest GEMM in the model (5120 x 129280 per rank at TP=1) and the
        one that most needs to be on the device, so it goes through the bf16
        NKI kernel. `head_weight` is stored bf16, so an fp32 torch `linear`
        would only differ in summation order.
        """
        logits = nki_linear(hidden_states, self.head_weight, None, "bf16").float()
        if self.world_size > 1:
            logits = tp_all_gather_last(logits, self.tp_group)
        return logits

    # ── Execution ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata: object | None = None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """tokens -> logits, through all 40 layers and the three DSpark stages.

        **How this runs, which is the design and not an accident.** The model
        interleaves compute with host I/O — a layer's routed experts and the
        engram rows for this step are read off disk in the middle of the layer —
        so this method is Python and the compute underneath it is compiled.
        Every host boundary is its own `@torch._dynamo.disable`d method
        (`_sequence_spans`, `_engram_rows`, `MoE._stage`, `MoE._route_plan`),
        which is what `VLLM_NEURON_DEBUG_MODE=1` in `__init__.py` is set for: it
        only relaxes `fullgraph`, and moves nothing to the host.

        **This frame itself is not compiled, and cannot be.** Dynamo says so:

            Explanation: torch.compile detected a graph break in a for/while
            loop. Skipping the frame and falling back to eager, as graph breaks
            in loops are not supported.

        — the layer loop breaks at the engram fetch and at the MoE, so the frame
        is skipped. What makes that *correct* rather than a CPU fallback is that
        dynamo's eval-frame hook still intercepts the frames this one calls, so
        `_run_block`, `embed_input_ids`, `compute_logits`, `block.engram` and
        `_engram_hashes` each compile as their own graph. The one thing that
        rule does not cover is tensor arithmetic written *inline here*, which
        really would run on the host — so there is none: every step below is a
        call, and the four small ones that used to be expressions
        (`_dspark_tap`, `_concat_last`, `_splice_rows`, `_select_rows`) are
        compiled entry points.

        The KV layout is vLLM's own paged allocation: the sliding window, the
        compressed KV, the index keys and the compressor's pooling inputs all
        live in it, addressed through `PagedSpan`, and both attention variants
        are checked against the recorded traces that way (`iter-2.md` §5).

        **Test status, stated precisely, because "the code is here" is not "the
        computation has run".** As of `.autohelix/notes/iter-7.md`, the
        verifier's own four prompts are served end to end on the device and
        judged sound: all 40 decoder layers, the engram lookups off the 94.6 GiB
        table, the MoE with real streamed fp4 experts, the three DSpark stages,
        the head and the on-device sampler — and, since iteration 7, speculative
        decoding through those stages, with 35-44% of the drafts accepted. The
        2008-token needle's access code is retrieved at a distance of ~1,900
        tokens, which is the long-context path (compressed latents, index keys,
        top-512-of-2048 sparse attention) being right rather than merely
        running.

        Two things that cost whole iterations to find and are easy to undo:
        every position-derived Python int folded into the graph as a *constant*,
        so each decode step was a fresh set of graphs (`PagedSpan` documents the
        fix), and the runner's decode block table is *trimmed* for a
        sliding-window group while this port addresses every cache by absolute
        position (`_paged_spans` documents that one).

        Shape of the computation, from `vendor/model.py`'s `Transformer.forward`:

            embed -> expand to hc_mult=4 copies -> 40 x Block -> collapse with
            the last layer's pre_mix -> norm -> head

        with `Engram` added into the stream before layers 1 and 14, and the
        attention *input* of layers 37/38/39 (`dspark_target_layer_ids`)
        collected as `main_hiddens` for the DSpark stages.

        The one structural difference from the reference, and why: the reference
        runs one `[B, L]` batch at a single `start_pos`. vLLM hands over a flat
        `[T]` token axis covering several sequences at different positions. So
        the per-token work (hyper-connection coefficients, norms, MoE routing
        and the experts) stays flat over `[T]` — routing does not care which
        sequence a token came from — while **attention runs per sequence**,
        because each sequence has its own `start_pos`, its own 128-slot ring and
        its own compressed-KV history.

        The sequence loop is *inside* the layer loop, not outside. That is the
        timeshare schedule: a layer's expert weights are staged once and used by
        every sequence, whereas looping sequences outermost would re-stream 276
        GiB per sequence.
        """
        if self.weight_store is None:
            raise RuntimeError(
                "forward() needs the weight store that load_weights attaches: "
                "the routed experts and the engram tables are streamed, not "
                "resident."
            )
        meta = self._attn_meta(attn_metadata)
        spans = self._sequence_spans(positions, meta["slot_mapping"])
        paged = self._paged_spans(spans, meta)
        # `(begin, length)` only — the position is deliberately left out of what
        # the compiled per-layer graph is handed, because dynamo specializes on
        # int arguments and a position in here is a recompile per decode step.
        # It travels as `span.pos`, a tensor. See `PagedSpan`.
        slices = [(begin, length) for begin, length, _ in spans]
        # Where the real tokens stop. Everything after it is buffer padding the
        # runner did not clear; see `_sequence_spans`.
        live_rows = sum(length for _, length, _ in spans)

        # ── embed -> hc_expand ────────────────────────────────────────────
        hidden = self.embed_input_ids(input_ids)
        hidden = NF.merge_prompt_embeds(hidden, inputs_embeds, is_token_ids)
        # `hc_expand`: one hidden state becomes hc_mult copies, with a one-hot
        # pre_mix so the first layer's hc_pre returns copy 0 unchanged.
        # `kernels/hc_expand.py` is the shipped kernel for it — pure DMA, so the
        # copies come out bit-identical — generalized to a ragged row count.
        stream, pre_mix = _nki_entry("hc_expand")(hidden.unsqueeze(0))
        stream, pre_mix = stream.squeeze(0), pre_mix.squeeze(0)

        # ── engram hash ids, once for the whole batch ─────────────────────
        hashes = self._engram_hashes(input_ids, paged)

        # ── the 40 decoder layers ─────────────────────────────────────────
        # One per span, not one per batch: `attn_half` runs the sequence loop
        # inside the compiled graph, so a single instance lets the last span
        # overwrite what the earlier ones published. §2a of
        # `.autohelix/notes/iter-4.md` has the out-of-bounds this caused.
        shared = [SharedAttentionState() for _ in paged]
        main_hiddens: list[torch.Tensor] = []
        target_layers = set(self.config.dspark_target_layer_ids)

        for layer_id, block in enumerate(self.layers):
            if block.engram is not None:
                index = list(self.config.engram_layer_ids).index(layer_id)
                rows = self._engram_rows(layer_id, hashes, index, stream.device)
                # Only the live rows. `hashes` is built per span, so it covers
                # `live_rows` tokens while the stream is the full batched-token
                # bucket — a 158-token prompt in a 512-token bucket gave
                #
                #     RuntimeError: Attempting to broadcast a dimension of
                #     length 158 at -2! ... torch.Size([1, 158, 4]); but
                #     expected shape should be broadcastable to [1, 512, 4]
                #
                # inside the engram's per-(token, copy) gate. Hashing the pad
                # rows instead would be the other fix and costs 3x the random
                # reads off the 94.6 GiB table for rows nothing consumes.
                updated = block.engram(
                    stream[:live_rows].unsqueeze(0), rows, None
                ).squeeze(0)
                if live_rows < stream.shape[0]:
                    stream = _compiled(_splice_rows)(updated, stream[live_rows:])
                else:
                    stream = updated
            if layer_id in target_layers:
                # The DSpark head reads the attention *input* of its target
                # layers, not their output, and averaged over the hc copies.
                main_hiddens.append(_compiled(_dspark_tap)(stream))
            stream, pre_mix = self._run_block(
                block, stream, pre_mix, slices, paged, shared, live_rows
            )

        # ── collapse -> norm -> head ──────────────────────────────────────
        collapsed = HyperConnections.pre(stream.unsqueeze(0), pre_mix.unsqueeze(0))
        collapsed = self.norm(collapsed).squeeze(0)

        selected = collapsed
        if sampling_positions is not None:
            selected = _compiled(_select_rows)(collapsed, sampling_positions)
        logits = self.compute_logits(selected)

        # ── the DSpark stack, on every forward ────────────────────────────
        # The three `mtp.*` stages are part of this model, not an optional
        # accelerator: checkpoint layers 40-42, each a hyper-connected block
        # over its own 128-expert pool at top-3, fed by the attention inputs
        # tapped from layers 37/38/39 above. So they run on every step, and
        # each scheduled sequence gets its own draft — the draft block is per
        # sequence, like the backbone's attention.
        #
        # **They run after the sampler, not before, and that ordering is the
        # whole correctness of the draft.** The first MTP stage is conditioned
        # on the token the step just *committed* — the sample at the sequence's
        # last position, or under speculation the last token the rejection
        # sampler accepted — and the draft block hangs off that token's
        # position. Running the stages before the sampler, as this file did
        # through iteration 6, could only seed them with `input_ids[-1]`: the
        # token fed *in* at the last position rather than the one sampled from
        # it, which puts the whole draft block one position early and makes
        # every draft token a prediction of something already known. The
        # reference (`vendor/model.py`'s `Transformer.forward_spec`) is called
        # from the generation loop with the sampled token, for the same reason.
        #
        # `DSparkProposer` (`drafter.py`) reads `last_draft` and hands it to the
        # scheduler, so these drafts are proposed and verified rather than
        # computed and dropped. Without a speculative config there is nowhere
        # to put them and the step returns the backbone's own tokens — the
        # stages still run, because the alternative is a forward that silently
        # executes 40 of the checkpoint's 43 layers.
        #
        # Each stage writes into *its own* paged allocation — `get_kv_spec`
        # declares `mtp.{stage}.attn` beside the 40 backbone layers — so a draft
        # write cannot disturb the backbone's window cache.
        main_hidden = (
            _compiled(_concat_last)(main_hiddens) if main_hiddens else None
        )

        if self.sampler is None:
            # No on-device sampling: the runner samples on the host, and under a
            # speculative config does the rejection sampling there too
            # (`NeuronModelRunner.sample_tokens`'s CPU branch), which needs the
            # logits of every verify position — so the step returns them. The
            # stages still run, seeded from the greedy token: that is the
            # sampler's own choice at the verifier's `temperature = 0`, and
            # where it is not, a mis-seeded draft costs acceptance rate and
            # never output, because every draft is verified against the target.
            self.last_draft = self._draft(
                _compiled(_greedy_rows)(logits), None, spans, paged, shared,
                main_hidden, sampling_positions is not None,
            )
            return logits

        # The sampler is the repo's own on-device one; `logit_mask` and the
        # per-request `sampling_params` come from the runner. The second
        # element is the logits the runner keeps for its debug/spec paths —
        # already full-width here, since `compute_logits` gathers.
        # Compiled, like every other step: `forward` is an uncompiled frame
        # (see the docstring), so an eager call here would be a host round trip
        # of a `[1, 129280]` logits row and a dtype cast the backend's fallback
        # cannot do. The goal permits sampling on the host; this is simply
        # cheaper and keeps the whole step on the device.
        sampled = _block_entry(self.sampler, "forward")(
            logits, sampling_params, logit_mask, rank
        )

        # Under a speculative config this step is a *verify* step: it was handed
        # `1 + k` tokens per sequence, of which `k` are the previous step's
        # drafts, and the Neuron runner expects an on-device-sampling model to
        # return the rejection sampler's verdict rather than raw samples
        # (`neuron_model_runner.py:8131`, "Model already performed rejection
        # sampling on-device"). Greedy acceptance keeps exactly the prefix the
        # target would have emitted unaided — exact as a *rule*, and measured
        # as 2 of the verifier's 4 prompts byte-identical with speculation on
        # and off, the other two flipping one near-tied token each. The gap is
        # the arithmetic rather than the rule: `1 + k` rows per sequence where
        # a plain decode step has 1, and the torch sum-reductions that are left
        # on the live path (`Gate`'s 384-wide softmax, the compressor's
        # pooling) tile differently at a different row count. `RMSNorm` was the
        # largest of them and is now a row-invariant kernel; see
        # `.autohelix/notes/iter-8.md` §3c-§3d for the measurement and for the
        # one change that would close the rest at once.
        accepted = None
        if spec_decode_metadata is not None:
            accepted = _compiled(_rejection_sample)(
                sampled,
                spec_decode_metadata.draft_token_ids,
                spec_decode_metadata.cu_num_draft_tokens,
                tuple(spec_decode_metadata.num_draft_tokens),
            )

        self.last_draft = self._draft(
            sampled, accepted, spans, paged, shared, main_hidden,
            sampling_positions is not None,
        )
        if accepted is not None:
            return accepted
        return sampled, logits

    # ── Forward helpers ───────────────────────────────────────────────────

    @torch._dynamo.disable
    def _engram_rows(
        self, layer_id: int, hashes: torch.Tensor, index: int, device
    ) -> torch.Tensor:
        """This layer's n-gram rows for every token, off the 94.6 GiB table.

        The hash ids name rows the store gathers and dequantizes — 24 per token
        — so the table is never resident and never materialized, but every
        lookup a token needs does happen. This is the engram half of the
        timeshare.

        Dynamo-disabled for the same reason as `MoE._stage`: the gather happens
        on the host against an mmap, and `rows.to(device)` is a traceable op, so
        inlined it would drag a CPU tensor into the surrounding graph.

        The ids come back to the host *before* the layer slice, because slicing
        a middle axis leaves a strided view and the backend's eager fallback
        cannot flatten one:

            RuntimeError: Expected self.is_contiguous() to be true, but got
            false
              ... weight_loader.py:308, in rows
                  flat_indices = indices.reshape(-1).to(torch.int64)

        `[1, T, n_engram_layers, 24]` of int64 is 196 KiB at the 512-token
        bucket, so the whole-tensor copy is cheaper than the gather it feeds.
        """
        ids = hashes.cpu()[:, :, index, :].contiguous()
        rows = self.weight_store.engram(layer_id, ids)
        # Into a buffer shared by both engram layers rather than a fresh
        # device allocation each time, for the reason in `_layer_io`: the
        # executor retains a binding to the tensors its last call was handed,
        # so a fresh input per engram layer per forward is retained per engram
        # layer. `[1, T, 24, 256]` bf16 is 12 KiB a token, i.e. 96 MiB a layer
        # at the 8192-token prefill segment. Host-side and dynamo-disabled, so
        # no traced code and no compiled graph is touched.
        key = (tuple(rows.shape), rows.dtype, str(device))
        buf = self._engram_row_slots.get(key)
        if buf is None:
            buf = torch.empty(rows.shape, dtype=rows.dtype, device=device)
            self._engram_row_slots[key] = buf
        buf.copy_(rows)
        return buf

    def _attn_meta(self, attn_metadata: object | None) -> dict:
        """The one metadata dict every layer in the KV group shares.

        `attn_metadata` is keyed by layer name and every layer in a cache group
        gets the same dict, so the window layers and the three `mtp` stages all
        read the same `slot_mapping` and `block_table_tensor` — which is what
        lets a draft stage address the backbone's slots.
        """
        if attn_metadata is None:
            raise RuntimeError(
                "forward() needs attn_metadata: the sliding window, the "
                "compressed KV and the index keys all live in vLLM's paged "
                "allocation and are addressed through its slot_mapping and "
                "block_table (see PagedSpan)."
            )
        name = f"layers.{0}.attn"
        try:
            meta = attn_metadata[name]
        except (TypeError, KeyError) as exc:
            raise RuntimeError(
                f"attn_metadata has no entry for {name!r}; get_kv_spec() "
                "declared it, so this is a runtime/spec mismatch "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        return meta

    @torch._dynamo.disable
    def _paged_spans(
        self, spans: list[tuple[int, int, int]], meta: dict
    ) -> list[PagedSpan]:
        """One `PagedSpan` per scheduled sequence.

        Row `k` of the block table is the `k`-th scheduled sequence, which is
        the `k`-th span: the runner builds both from the same request order.

        Dynamo-disabled, like the other host boundaries, because each span's
        `pos` is a host-built `[1]` tensor moved to the device: inside a graph
        that transfer would be a constant again, which is the whole thing
        `PagedSpan.pos` exists to avoid. One `[1]` int64 per sequence per step.

        **`full_block_table_tensor`, not `block_table_tensor`.** The two differ
        on exactly one step kind and that difference was a device out-of-bounds
        for five iterations. `NeuronModelRunner._build_attention_metadata`
        *trims* the decode block table for a sliding-window group
        (`_compute_swa_decode_tensors`): row `k` becomes the `num_swa_blocks`
        blocks that the window covers, gathered from `start_block` onward, and
        `swa_kv_pos_offset = start_block * block_size` is handed over so a
        consumer can rebase its positions into that frame. With
        `sliding_window = 128` and `block_size = 32` the trimmed width is 8.

        Every cache this model addresses — the window, the compressed latents,
        the index keys, the n-gram ids — is indexed by *absolute* position
        (`PagedSpan.slots`), so a trimmed row is the wrong table: at position
        2008, `slots` computes `block_table[62]` against 8 entries and the
        runtime reports

            scatter/gather (indirect memory copy via scalar DGE)
            out-of-bound access ... engine=GPSIMD
            Source location: model.py:2921

        in both the n-gram graph and layer 0's. It is silent below position
        256, because there `start_block` clamps to 0 and the trimmed row *is*
        the first 8 untrimmed entries — which is why every short prompt decodes
        and the 2048-token needle does not.

        The runner provides the untrimmed clone for this exact use ("for use
        cases that need to compute slot indices into the *full* KV cache"), and
        it is present on every path, prefill included, so there is nothing to
        branch on. The `.get` fallback is for a metadata dict that predates it.
        """
        block_size = int(meta["block_size"])
        block_table = meta.get("full_block_table_tensor")
        if block_table is None:
            block_table = meta["block_table_tensor"]
        slot_mapping = meta["slot_mapping"]
        device = slot_mapping.device
        starts = torch.tensor(
            [start_pos for _, _, start_pos in spans], dtype=torch.int64
        ).to(device)
        return [
            PagedSpan(
                block_table=block_table[row],
                block_size=block_size,
                slot_mapping=slot_mapping[begin : begin + length],
                start_pos=start_pos,
                length=length,
                pos=starts[row : row + 1],
                begin=begin,
            )
            for row, (begin, length, start_pos) in enumerate(spans)
        ]

    @torch._dynamo.disable
    def _sequence_spans(
        self, positions: torch.Tensor, slot_mapping: torch.Tensor
    ) -> list[tuple[int, int, int]]:
        """Split the flat token axis into `(start_index, length, start_pos)`.

        vLLM concatenates every scheduled sequence's tokens into one `[T]` axis
        and gives their absolute positions in `positions`. A sequence boundary
        is where `positions` stops increasing by one — a decode step is then `B`
        spans of length 1, and a prefill (chunked or not) is one span per
        sequence whose `start_pos` is that chunk's first position.

        **`positions` is padded, and `slot_mapping` is how you find the real
        tokens.** The runner sizes `input_ids` and `positions` to the
        batched-token bucket and leaves the tail at whatever the buffer held, so
        a 20-token prompt in a 512-token bucket arrives as `0..19` followed by
        492 zeros — and every one of those pad zeros looks like the start of a
        new sequence. That produced hundreds of spans against a one-row block
        table:

            IndexError: index 1 is out of bounds for dimension 0 with size 1

        `slot_mapping` is the authority. For a prefill the runner slices it to
        `total_num_scheduled_tokens`; for a decode step it pads it with
        `NULL_BLOCK_ID` = 0, vLLM's reserved null block
        (`_apply_decode_slot_mapping_padding`). So the live region is
        `slot_mapping` up to its trailing zeros.

        Host-side, over at most `max_num_seqs` entries, and dynamo-disabled:
        inlined, the `tolist()` leaves unbacked symints and the boundary test
        cannot be guarded —

            torch._dynamo.exc.UserError: Could not guard on data-dependent
            expression Ne(u1, u0 + 1)

        — which is literally `values[i] != values[i - 1] + 1`. Behind the
        boundary the result is a plain Python list, which dynamo then treats as
        static and unrolls the loops over.
        """
        slots = slot_mapping.reshape(-1).tolist()
        if not slots:
            return []
        # `live > 1`, not `live > 0`: slot 0 is a real slot — block 0, offset 0 —
        # for the first token of whichever sequence owns block 0. A decode token
        # can never land there (it is never a sequence's first token), but a
        # single-token prefill can, so one token always survives the trim.
        live = len(slots)
        while live > 1 and slots[live - 1] == 0:
            live -= 1
        values = positions.reshape(-1).tolist()[:live]
        spans: list[tuple[int, int, int]] = []
        begin = 0
        for i in range(1, len(values) + 1):
            if i == len(values) or values[i] != values[i - 1] + 1:
                spans.append((begin, i - begin, int(values[begin])))
                begin = i
        return spans

    def _engram_hashes(
        self, input_ids: torch.Tensor, paged: list[PagedSpan]
    ) -> torch.Tensor:
        """n-gram row indices for every token, `[1, T, n_engram_layers, 24]`.

        Hashed per sequence, because the n-gram look-back must not reach across
        a sequence boundary — and `NgramHashState` keeps the compressed ids of
        earlier chunks in its own cache, which is what lets a decode step hash
        an n-gram reaching back into its prompt.

        Driven off the `PagedSpan`s rather than the raw span tuples, so the
        position reaches `NgramHashState.forward` as its `pos` tensor: this
        compiles as its own graph, and a Python int here recompiled it once per
        decode step (see `PagedSpan`).
        """
        if self.ngram_hash is None:
            raise RuntimeError(
                "engram layers "
                f"{list(self.config.engram_layer_ids)} need the n-gram hash "
                "state; bind_kv_cache builds it from the tokenizer."
            )
        per_span = []
        for span in paged:
            length, begin = span.length, span.begin
            ids = input_ids[begin : begin + length].reshape(1, length)
            per_span.append(self.ngram_hash(ids, span))
        return torch.cat(per_span, dim=1)

    @torch._dynamo.disable
    def _layer_io(
        self, stream: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The three per-layer buffers every block shares, for this shape.

        `(stream_out [T, hc, dim], residual [1, T, hc, dim], normed
        [1, T, dim])`. One set per `(rows, dtype, device)`, allocated on first
        use and kept — 720 MiB at an 8192-token segment, against the ~31 GiB
        the per-block graphs were each retaining a private copy of
        (`Block.attn_half`).

        **Dynamo-disabled, and called before anything traced in `_run_block`.**
        `torch.empty` is a traceable op, so inlined it would be pulled into the
        surrounding graph; and an added break *between* traced ops rewrites
        every cache key downstream of it, which is how iteration 5 crashed the
        compiler (`.autohelix/notes/iter-5.md` §5). At the top of `_run_block`
        there is nothing traced before it to split.

        Not a `register_buffer`: these hold no checkpoint state, they are keyed
        by a shape that is only known once the scheduler picks a bucket, and
        `state_dict` has no business carrying them.
        """
        rows = int(stream.shape[0])
        key = (rows, stream.dtype, str(stream.device))
        slots = self._layer_io_slots.get(key)
        if slots is None:
            hc, dim = int(stream.shape[1]), int(stream.shape[2])
            opts = {"dtype": stream.dtype, "device": stream.device}
            slots = (
                torch.empty(rows, hc, dim, **opts),
                torch.empty(1, rows, hc, dim, **opts),
                torch.empty(1, rows, dim, **opts),
            )
            self._layer_io_slots[key] = slots
            logger.info(
                "deepseek_v41_timeshare: allocated the shared per-layer I/O "
                "buffers for %d rows (%.1f MiB, one set for all %d blocks)",
                rows,
                sum(t.numel() * t.element_size() for t in slots) / 1024**2,
                len(self.layers),
            )
        return slots

    def _run_block(
        self,
        block: Block,
        stream: torch.Tensor,
        pre_mix: torch.Tensor,
        slices: list[tuple[int, int]],
        paged: list[PagedSpan],
        shared: list[SharedAttentionState],
        live_rows: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One layer over a flat `[T, hc, dim]` stream covering several
        sequences.

        Three steps: a compiled graph up to the MoE's input, the MoE itself
        (eager Python around compiled entry points, because it streams expert
        weights off disk mid-layer), then a compiled graph for the output mix.
        `Block.attn_half` and `Block.ffn_half` hold the arithmetic; this method
        is the schedule.

        The MoE runs over the whole flat batch, which is what keeps a layer's
        staged expert weights used once per layer rather than once per
        sequence — the timeshare schedule. Looping sequences outermost would
        re-stream 276 GiB per sequence.
        """
        tp_group = self.tp_group if self.world_size > 1 else None
        # The shared per-shape buffers, resolved before anything traced runs so
        # this adds no graph break (see `_layer_io`).
        io_stream, io_residual, io_normed = self._layer_io(stream)
        normed, residual, ffn_pre, ffn_post, ffn_comb = _block_entry(
            block, "attn_half"
        )(stream, pre_mix, slices, paged, shared, tp_group, io_residual, io_normed)
        ffn_out = block.ffn(normed, None, self.weight_store, tp_group, live_rows)
        # `HyperConnections.post` directly, not `_block_entry(block,
        # "ffn_half")`. They compute the same thing — `ffn_half`'s whole body is
        # this call — but `post` reaches the `hc_attn_out` kernel through
        # `_nki_entry`, which is keyed on the *kernel*, so **one** NEFF serves
        # all 43 layers instead of one per layer.
        #
        # That matters for HBM, not for compile time: every loaded NEFF reserves
        # DMA rings, this graph's are tens of MiB (its tensors are
        # `[1, T, hc, dim]`), and 43 of them was ~1 GiB of a 24 GiB bank. See
        # `.autohelix/notes/iter-3.md` §5b-ter. `Block.ffn_half` is kept for the
        # reference-shaped `Block.forward` path and for symmetry with
        # `attn_half`, whose per-block weights leave it no such option.
        #
        # Through `_post_into` rather than `HyperConnections.post` directly, so
        # the next layer's input is the *same tensor* every layer: the stream is
        # the largest of the three things each per-block graph was retaining.
        return _compiled(_post_into)(
            io_stream, ffn_out, residual, ffn_post, ffn_comb
        ), ffn_pre.squeeze(0)

    @torch.no_grad()
    def _draft(
        self,
        sampled: torch.Tensor,
        accepted: torch.Tensor | None,
        spans: list[tuple[int, int, int]],
        paged: list[PagedSpan],
        shared: list[SharedAttentionState],
        main_hidden: torch.Tensor | None,
        gathered: bool,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """One DSpark draft per scheduled sequence, seeded from this step's own
        committed token.

        Two things are worked out here rather than in `forward_spec`, because
        both are per-sequence and both have to stay tensors:

        * **the seed** — the token the draft continues from. Without
          speculation that is the sample at the sequence's last position; with
          it, the *last token the rejection sampler accepted*, which is the one
          the sequence will actually continue from.
        * **the anchor** — which row of this step the draft block hangs off.
          Without speculation the last row (`length - 1`). On a verify step the
          step's rows are `[accepted token, draft_1 ... draft_k]`, so the
          accepted count indexes straight into the block: `valid - 1`. The
          `_tap_rows` has the arithmetic and the guard for a padded batch.

        `gathered` says whether `sampled` has one row per sequence, which it
        does whenever the runner passed `sampling_positions` — every path it
        actually takes — or one row per token.

        **Not `@torch._dynamo.disable`d, and that matters.** `disable` is
        recursive: with it on this method, `forward_spec` and every module it
        calls ran *eagerly*, and the first eager dtype cast inside
        `BlockQuantLinear` died on

            RuntimeError: Expected self.dtype() == dst.dtype() to be true

        — the backend's eager fallback, which implements almost nothing. Like
        `forward`, this frame is one dynamo *skips* (the span loop breaks at the
        MoE's host I/O) and its callees are still intercepted and compiled,
        which is the whole structure `forward`'s docstring describes. The two
        small host-built index tensors are the same pattern, and the same cost,
        as `_paged_spans`' `starts`.
        """
        device = sampled.device
        lengths = torch.tensor(
            [length for _, length, _ in spans], dtype=torch.int64, device=device
        )
        ones = torch.ones(len(spans), dtype=torch.int64, device=device)
        if accepted is not None:
            seeds, valid = _compiled(_accepted_tail)(accepted)
            # The rejection sampler's batch is the runner's *padded* one: on a
            # verify step `_build_spec_decode_metadata` appends fake full-spec
            # requests up to the decode bucket, so with 3 scheduled sequences in
            # a batch of 4 `accepted` has four rows where `spans` has three. The
            # real requests come first, in span order.
            seeds, valid = seeds[: len(spans)], valid[: len(spans)]
            rows_per_seq = int(accepted.shape[1])
        else:
            if gathered:
                picks = torch.arange(len(spans), dtype=torch.int64, device=device)
            else:
                picks = torch.tensor(
                    [begin + length - 1 for begin, length, _ in spans],
                    dtype=torch.int64,
                    device=device,
                )
            seeds = _compiled(_select_rows)(sampled.reshape(-1), picks)
            valid, rows_per_seq = ones, 1
        taps = _compiled(_tap_rows)(valid, lengths, rows_per_seq)
        return [
            self.forward_spec(
                seeds[row : row + 1], taps[row : row + 1],
                main_hidden, meta, span, span_shared,
            )
            for row, (meta, span, span_shared) in enumerate(
                zip(spans, paged, shared)
            )
        ]

    def forward_spec(
        self,
        seed_ids: torch.Tensor,
        tap: torch.Tensor,
        main_hidden: torch.Tensor | None,
        span: tuple[int, int, int],
        paged: PagedSpan,
        shared: SharedAttentionState,
    ):
        """The DSpark draft path over the `mtp.*` stages, for one sequence.

        `vendor/model.py`'s `Transformer.forward_spec`: build the draft stream
        from the committed token and the projected `main_hidden`, run it through
        the three `mtp` stages (each a hyper-connected block over the 128-expert
        pool at top-3), then the shared LM head with the Markov bias folded in
        per draft position, and the confidence head. Returns
        `(output_ids [1, 1 + k], logits, confidence)` — `output_ids[0]` is the
        seed and the rest are the drafts, which is the `[1 + k]` row
        `DSparkProposer.propose` hands the scheduler.

        `main_hidden` is handed over **whole for this span**, not just the
        anchor row, and the reason is the window cache: `DSparkAttention` writes
        one window entry per scheduled token at its own slot, so every real
        token this step saw is in the draft stages' window for later steps. The
        reference does the same thing — its `forward_spec` is called with the
        prompt's full `main_hidden` at `start_pos == 0` and one row per step
        afterwards — and passing only the last row made a prefill broadcast a
        single position's KV across all of the prompt's slots.
        """
        if main_hidden is None:
            raise RuntimeError(
                "the DSpark stages read the attention inputs of layers "
                f"{list(self.config.dspark_target_layer_ids)}; forward() "
                "collected none, so the target-layer tap did not run"
            )
        begin, length, _ = span
        main = main_hidden[begin : begin + length].unsqueeze(0)

        stage0 = self.mtp[0]
        stage0.embed_fn = self.embed_input_ids
        stream, main_x = stage0.forward_embed(main, seed_ids)
        pre_mix = _compiled(_one_hot_pre_mix)(stream, self.hc_mult)

        tp_group = self.tp_group if self.world_size > 1 else None
        # Tensors, like every other position on the compiled path: this step's
        # real tokens start at `paged.pos`, and the draft block hangs off the
        # committed token at `paged.pos + tap`. See `PagedSpan` for why neither
        # may be a Python int, and `DSparkAttention` for why they differ.
        draft_pos = paged.pos + tap
        for stage in self.mtp:
            stream, pre_mix = stage(
                stream, paged.pos, draft_pos, pre_mix, main_x, shared, paged,
                self.weight_store, tp_group,
            )

        last = self.mtp[-1]
        last.head_fn = self.compute_logits
        last.markov_head._tp_group = tp_group
        return last.forward_head(stream, pre_mix, seed_ids)

    # ── Weights ───────────────────────────────────────────────────────────

    def load_weights(
        self,
        checkpoint_path: str,
        device: torch.device,
        cache_dir: str | None = None,
    ) -> None:
        """Load the sharded checkpoint. Delegates to `weight_loader`."""
        from .weight_loader import load_weights as _load_weights

        _raise_dynamo_limits()
        self._allow_request_time_compilation()
        self._cap_kv_cache_blocks()
        _load_weights(self, checkpoint_path, device, cache_dir)

    @staticmethod
    def _cap_kv_cache_blocks() -> None:
        """Cap the KV cache at what this configuration can actually address, so
        the rest of HBM is left for the model's compiled graphs.

        vLLM sizes the KV cache from the memory that is free when it profiles,
        and it profiles *after* the weights load but *before* any of the model's
        NEFFs do. For an ordinary model that is right — the graphs are built
        during warmup, so profiling sees their cost. This port skips warmup
        (`__init__.py`) and, more importantly, loads of order 120 graphs, one
        per block per shape set, whose DMA rings are ~4.4 GiB. Measured on this
        host at `max_model_len=544, max_num_seqs=1`:

            Neuron HBM: 10.60 GiB used, 13.40 GiB free
            GPU KV cache size: 76,250 tokens
            Maximum concurrency for 544 tokens per request: 140.17x

        — a cache 140x larger than the scheduler can ever fill, 3.3 GiB of it,
        and then the graphs have nowhere to go:

            RuntimeError: Failed to schedule neff execution. status=4
            message=Allocation Failure

        So the cap is `max_model_len / block_size + 1` blocks per sequence,
        times the running batch. **It is a ceiling, not a reduction below what
        correctness needs**: one sequence cannot reference more slots than its
        own context, and the compressed caches and index keys are addressed by
        absolute position within that context. What it gives up is cross-request
        prefix *retention* beyond the running batch's own contexts.

        **`cache.block_size` is 16 here and 32 by the time the cache is laid
        out.** The runner logs the resolution — `KV cache block_size resolved:
        cache_config=32, per_group=[32]` — and it happens after this, so
        dividing by what `cache_config` currently says produced about twice the
        slot count the arithmetic named, and the cap barely bit: at the
        verifier's own defaults it took vLLM's 2523 blocks down to 2360, 9
        sequences' worth for a scheduler that runs 4. The block size is now
        taken from `NeuronPlatform.update_block_size_for_backend`'s rule
        instead, which is the one that will be applied, so the cap realizes
        what it says: one full `max_model_len` per sequence, which is the part
        correctness needs.

        `num_gpu_blocks_override` is honoured at
        `vllm/v1/core/kv_cache_utils.py:945`, which runs after
        `determine_available_memory`, which runs after this — the ordering is
        visible in the engine log (weights, then `Neuron HBM:`, then
        `GPU KV cache size:`). An operator who sets the override themselves
        wins: this only fills it in when it is unset.
        """
        from vllm.config import get_current_vllm_config

        try:
            config = get_current_vllm_config()
            cache, model, sched = (
                config.cache_config,
                config.model_config,
                config.scheduler_config,
            )
        except Exception as exc:  # noqa: BLE001 - no active config, nothing to cap
            logger.warning(
                "deepseek_v41_timeshare: no active vLLM config, leaving the KV "
                "cache size alone (%s: %s)", type(exc).__name__, exc
            )
            return
        if cache.num_gpu_blocks_override is not None:
            return

        # The block size to divide by is the one the cache is **laid out** with,
        # which is not the one `cache_config` carries at this point. The rule
        # is `NeuronPlatform.update_block_size_for_backend`'s, applied here
        # rather than guessed: 32 unless the operator named a block size, and
        # that method runs after this one.
        #
        # Dividing by the 16 that is still in `cache_config` made the realized
        # cache twice the size this formula intends. Measured at the verifier's
        # own `max_model_len = 9424, max_num_seqs = 4`:
        #
        #     capping the KV cache at 2360 blocks of 16 (37760 nominal slots)
        #     Overriding num_gpu_blocks=2523 with num_gpu_blocks_override=2360
        #     GPU KV cache size: 85,213 tokens
        #     Maximum concurrency for 9,424 tokens per request: 9.04x
        #
        # — a 6% reduction where the intent was 50%, and 9 sequences' worth of
        # cache for a scheduler that runs 4. That overage is ~2 GB of window KV
        # and per-slot buffers, and it is what the `nrt_tensor_allocate
        # status=4` at an 8192-token prefill segment was spending.
        block = int(getattr(cache, "block_size", None) or 32)
        if not getattr(cache, "user_specified_block_size", False):
            block = 32
        per_sequence = -(-int(model.max_model_len) // block) + 1
        blocks = per_sequence * int(sched.max_num_seqs)
        # Only ever *reduce*. `num_gpu_blocks_override` is applied blindly — it
        # replaces the number vLLM computed from available memory, in either
        # direction — so a configuration whose working set is genuinely large
        # (a long context, or many concurrent sequences) must be left to vLLM's
        # memory-based sizing rather than handed a number that does not fit.
        # `_KV_CAP_MAX_SLOTS` is the line between "this cache is obviously
        # oversized for the configuration" and "I cannot tell".
        if blocks * block > _KV_CAP_MAX_SLOTS:
            logger.info(
                "deepseek_v41_timeshare: leaving the KV cache to vLLM — this "
                "configuration wants %d slots (%d per sequence x %d sequences "
                "x 2), which is not obviously oversized",
                blocks * block, per_sequence, sched.max_num_seqs,
            )
            return
        cache.num_gpu_blocks_override = blocks
        logger.info(
            "deepseek_v41_timeshare: capping the KV cache at %d blocks of %d "
            "(%d slots) — %d per sequence x %d sequences, one full context "
            "each. The rest of HBM is needed for this port's compiled graphs "
            "and its per-slot buffers, which vLLM's own sizing cannot see; see "
            "model._cap_kv_cache_blocks",
            blocks, block, blocks * block, per_sequence, sched.max_num_seqs,
        )

    @staticmethod
    def _allow_request_time_compilation() -> None:
        """Let a request compile a graph that warmup did not build.

        `NeuronModelRunner` wraps every forward in
        `neuron_utils.model_forward_context()`, which is
        `torch.compiler.set_stance("fail_on_recompile")` — on the assumption
        that warmup built every graph the model will ever need:

            RuntimeError: Detected recompile when torch.compile stance is
            'fail_on_recompile'

        For this port that assumption cannot be made true, for two independent
        reasons:

          * warmup is skipped (`__init__.py` says why: for this model it is a
            251 GiB disk read rather than a cheap pre-compile), and
          * **the graph set is input-dependent.** How many experts a step routes
            to, and how many rows each gets, are outputs of the router.
            `MoE._route_plan` buckets the row lists to powers of two precisely
            so that there is a handful of shapes rather than one per step, but a
            handful is not one. The only way to make it static is to run every
            expert over every token, which is 64x the arithmetic.

        It cannot be relaxed from inside `forward`. The stance is checked when
        the *outer* compile of that frame recompiles, which happens before any
        line of the body runs. Both obvious attempts were made and both still
        raise:

          1. `with torch.compiler.set_stance("default")` around the body;
          2. the same plus `@torch._dynamo.disable(recursive=False)` on
             `forward`, which only moves the report to
             `torch/_dynamo/external_utils.py:inner` — the module-call wrapper,
             which guards on input shapes regardless of what `forward` is
             marked.

        What works is rebinding the runner's own reference: it imported the
        helper by name (`from ... import model_forward_context`), so it holds
        its own binding, and this replaces that binding with one that sets the
        `"default"` stance.

        **This relaxes no numerics and moves nothing to the host.** A recompile
        under the default stance is a compile-time cost, and the compilation
        cache under `~/.cache/neuron_libtorch` makes a repeat free.

        It cannot live in `__init__.py`: importing the runner from there is an
        import cycle (`registry` -> this package -> runner -> registry).
        """
        from vllm_neuron.vllm.worker import neuron_model_runner as runner

        if getattr(runner, "_deepseek_v41_request_time_compilation", False):
            return
        runner.model_forward_context = lambda vllm_config: (
            torch.compiler.set_stance("default")
        )
        runner._deepseek_v41_request_time_compilation = True
        logger.info(
            "deepseek_v41_timeshare: model_forward_context relaxed to the "
            "default compile stance — this port's graph set depends on the "
            "router's output, so a request may compile a shape warmup did not"
        )


def _buffer_name(checkpoint_key: str) -> str:
    """Map a checkpoint key to the buffer that holds it.

    Almost all of them are the same string, which is the point of mirroring the
    reference's module paths. The exceptions are the four places where the
    checkpoint's leaf is a module and this model's is a buffer on its parent:
    `embed.weight` / `head.weight` (there is no `embed` submodule, just a
    vocab-parallel buffer), the hyper-connection parameters (`hc_attn_fn` ->
    `hc_attn.fn`), and the DSpark heads.
    """
    direct = {
        "embed.weight": "embed_weight",
        "head.weight": "head_weight",
    }
    if checkpoint_key in direct:
        return direct[checkpoint_key]

    for sublayer in ("hc_attn", "hc_ffn"):
        for suffix in ("fn", "base", "scale"):
            tail = f".{sublayer}_{suffix}"
            if checkpoint_key.endswith(tail):
                return checkpoint_key[: -len(tail)] + f".{sublayer}.{suffix}"

    for tail, replacement in (
        (".markov_head.embed.weight", ".markov_head.embed_weight"),
        (".markov_head.head.weight", ".markov_head.head_weight"),
        (".confidence_head.proj.weight", ".confidence_head.proj_weight"),
        (".main_proj.weight", ".main_proj.weight"),
    ):
        if checkpoint_key.endswith(tail):
            return checkpoint_key[: -len(tail)] + replacement
    return checkpoint_key
