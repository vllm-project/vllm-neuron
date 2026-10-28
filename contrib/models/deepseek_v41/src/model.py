# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V4.1-Flash on 16 Trainium 2 devices
============================================

Module names here are the checkpoint's own module paths — `layers.<i>.attn.wq_b`,
`layers.<i>.ffn.shared_experts.w1`, `mtp.<s>.markov_head` — which are also the
reference implementation's (`partition_artifact/vendor/model.py`). The identity
is deliberate: it makes "is this the weight the reference uses here" a string
comparison against `model.safetensors.index.json` rather than an argument about
a renaming scheme, and it keeps `weight_loader.build_weight_mappings` 1:1.

    dense       ~3.9 GiB a rank   heads/groups 8 ways, vocab 64 ways
    experts      4.3 GiB a rank   by expert id, 6 of 384 a layer
    engram       5.7 GiB a rank   by row, ~6.0M of 384M a table
                ----------------
                ~13.9 GiB of a 24 GiB bank
"""

from __future__ import annotations

import functools
import gc
import logging
import os
import time

import torch
import torch.nn.functional as F
from torch import nn

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
from . import mk_host, mk_model

#: Run every block and the DSpark drafter as one NKI megakernel each
#: (`kernels/layer_kernel.py`, `kernels/draft_kernel.py`, bridged by `mk_model`).
#: `DEEPSEEK_V41_MEGAKERNEL=0` keeps the per-op path for comparison.
_MEGAKERNEL = os.environ.get("DEEPSEEK_V41_MEGAKERNEL", "1") == "1"

logger = logging.getLogger(__name__)

#: `(2 + hc_mult) * hc_mult` — the width of a sublayer's coefficient projection,
#: covering `pre` (hc), `post` (hc) and `comb` (hc x hc). 24 at hc_mult=4.
def _mix_width(hc_mult: int) -> int:
    return (2 + hc_mult) * hc_mult


class ShardRule:
    """How one buffer is split across the tensor-parallel ranks.

    Attributes:
        dim: axis to split, or None to replicate.
        reason: why this axis — recorded because for several of these buffers
            the axis is only correct in combination with another one's (heads
            with `attn_sink`, groups with `wo_a`), and a wrong-but-plausible
            axis produces a model that runs and is subtly wrong.
        align: when set, each rank's slice is rounded *up* to a multiple of this
            and the checkpoint tensor is zero-padded to match. See
            `vocab_shard_width` for why the unembedding needs it; nothing else
            does, and `align=None` leaves the arithmetic exactly as it was.
        group: *which* ranks the axis is split across, `"world"` or `"attn"`.
            Not every axis of this model divides by the world size: `o_groups`
            is 8, so the attention's head and group axes split 8 ways and are
            replicated over the remaining ranks, while the expert and vocab axes
            split across all of them. See
            `config.ATTENTION_TENSOR_PARALLEL_SIZE`. A rule that names the wrong
            group produces a model that loads and is wrong, which is why this is
            spelled per rule rather than inferred from the shape.
    """

    __slots__ = ("dim", "reason", "align", "group")

    def __init__(
        self,
        dim: int | None,
        reason: str,
        align: int | None = None,
        group: str = "world",
    ) -> None:
        self.dim = dim
        self.reason = reason
        self.align = align
        if group not in ("world", "attn"):
            raise ValueError(f"group must be 'world' or 'attn', not {group!r}")
        self.group = group

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ShardRule(dim={self.dim}, reason={self.reason!r}, "
            f"align={self.align}, group={self.group!r})"
        )


def attn_shard(dim: int, reason: str) -> "ShardRule":
    """A rule split across the attention ranks rather than the whole world."""
    return ShardRule(dim, reason, group="attn")


REPLICATE = ShardRule(None, "no free axis; every rank needs the whole tensor")

#: Alignment the unembedding's vocab shard is padded to, and the number is the
#: tensor engine's partition width because that is what the GEMM tiles on.
#: 512: the drafter megakernel tiles its vocab-parallel head and Markov embedding
#: in 512-column chunks, so a shard has to be a multiple of 512 (129280 over 8
#: ranks is 16160 = 126.25 x 128, which it rejects; 16384 it takes). 64 ranks get
#: 2048 either way, and the last rank's pad stays 1792 under both schemes.
VOCAB_SHARD_ALIGN = 512


def vocab_shard_width(vocab_size: int, world_size: int) -> int:
    """Rows of the vocab-parallel embedding and head that one rank holds.

        129280,  1 rank  -> 129536   (129280 = 252.5 x 512, so round up)
        129280,  8 ranks ->  16384   ( 16160 = 31.56 x 512)
        129280, 64 ranks ->   2048   (  2020 =  3.95 x 512)
    """
    exact = -(-vocab_size // world_size)
    return -(-exact // VOCAB_SHARD_ALIGN) * VOCAB_SHARD_ALIGN


class RMSNorm(nn.Module):
    """`vendor/model.py`'s `RMSNorm`, including its fp32 interior, as a NKI
    kernel.

        variance = x.float().square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * x).to(dtype)
    """

    def __init__(self, dim: int, eps: float, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.register_buffer("weight", torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.dim)
        eps = torch.full((1, 1), self.eps, dtype=torch.float32, device=x.device)
        out = _nki_entry("rmsnorm", "rms_norm")(
            flat, self.weight.reshape(1, self.dim), eps
        )
        return out.reshape(shape)


#: fp8 e4m3's range, as `vendor/kernel.py`'s `act_quant_kernel` uses it. 448 is
#: the OCP e4m3 maximum; note the Trainium kernels under `kernels/` may use the
#: legacy nl.float8_e4m3 whose maximum is 240, which is what the compiler's
_FP8_MAX = 448.0


def _pow2(exponent: torch.Tensor) -> torch.Tensor:
    """`2 ** exponent`, elementwise, for an integral-valued float exponent."""
    return torch.pow(torch.full_like(exponent, 2.0), exponent)


def act_quant(
    x: torch.Tensor, block: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-wise fp8 activation quantization, as the reference does it."""
    *lead, width = x.shape
    blocks = width // block
    if width % block:
        raise ValueError(f"act_quant needs a multiple of {block}, got {width}")
    grouped = x.detach().float().reshape(*lead, blocks, block)
    amax = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = _pow2(torch.ceil(torch.log2(amax / _FP8_MAX)))
    quantized = (grouped / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return (
        quantized.reshape(*lead, width),
        scale.expand(*lead, blocks, block).reshape(*lead, width),
    )


#: Mirrors what `NeuronModelRunner.load_model` passes for the whole-model
#: compile, so a kernel reached from here is built the same way as one reached
#: from a single traced graph. The fp8 flag is not optional on Trn2: the kernels
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
    """
    for name in (
        "recompile_limit",
        "cache_size_limit",
        "accumulated_recompile_limit",
        "accumulated_cache_size_limit",
    ):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, 1 << 30)


def _fc_sum(x: torch.Tensor, group_name: str) -> torch.Tensor:
    """`x` summed across `group_name`'s ranks."""
    out = torch.ops._c10d_functional.all_reduce(x, "sum", group_name)
    return torch.ops._c10d_functional.wait_tensor(out)


def _fc_gather_last(x: torch.Tensor, group_name: str) -> torch.Tensor:
    """Every rank's `x` concatenated along the last axis.

    The axis is passed as a non-negative index on purpose. torch 2.13's
    `all_gather_tensor` rearranges the gathered `[world, ...]` result with
    `torch._utils._maybe_view_chunk_cat`, whose view fast path slices
    `shape[1:gather_dim]` and `shape[gather_dim + 1:]` without normalising the
    dim first, so `-1` builds a shape of twice the rank:

        view(FakeTensor(size=(64, 2048)), [1, 131072, 64, 2048]):
        shape '[1, 131072, 64, 2048]' is invalid for input of size 131072
    """
    return _fc_all_gather_tensor(x, x.dim() - 1, group_name)


def _group_name(tp_group) -> str:
    """The functional-collective name of a `GroupCoordinator`'s device group."""
    return _fc_resolve_group_name(tp_group.device_group)


#: Largest all-reduce payload, in elements, handed to one collective call.
_ALLREDUCE_CHUNK_ELEMS = int(
    os.environ.get("DEEPSEEK_V41_ALLREDUCE_CHUNK_ELEMS", 1 << 30)
)


# What this collective costs, measured on device in iteration 3. A forward
# carries ~43 *world-wide* all-reduces (one per MoE layer, two engram layers,
# the logits path) on top of the 8-way attention ones, and until iteration 3
# nobody had priced them — the floorplan says so of itself:
# "no intra-device collective can be timed on this host" (REPORT.md:742-745).
#
# Substituting the 8-rank EP-TP group for the world here, with every shape,
# graph and launch left alone, moved the `in512_bs1` cell by:
#
#     prefill forward, 503 tokens    947.4 ms -> 873.7 ms   (-7.8%)
#     decode step, 6 rows            402.6 ms -> 320.1 ms   (-20.5%)
#
# The saving is the *same ~75 ms at 503 rows and at 6*, so what a 64-rank
# all-reduce costs here is per-call latency and synchronization — ~1.9 ms,
# against ~0.1 ms for 8 consecutive ranks — and not bytes (5.2 MB at 503
# tokens, ~10 µs of wire).
#
# Two consequences, both measured rather than argued:
#
#   * **The cost tracks how many devices a collective spans, not how many
#     ranks.** Factoring the world sum as 8 (EP-TP row: 8 consecutive ranks,
#     two column-adjacent devices, one torus hop) x 8 (EP column: 8 ranks
#     strided across 8 *different* devices) is exact and measured 2.8% SLOWER
#     (974.2 ms), because the column leg costs what the 16-device world leg
#     costs.
#   * So a total sum cannot be made cheap: every rank needs data from every
#     device. The way to stop paying for it is to stop needing it, which is
#     what pipeline parallelism does — and that is now the measured case for
#     TP=8 x PP=8, with `kernels/pp_permute.py` as the stage transport.
#
# The two probes that produced these numbers are
# `.autohelix/notes/iter-3-probes.patch` (apply, measure, revert): they are not
# in the tree because a probe-tree verifier run failed its second phase twice
# while an untouched HEAD passed, and no script constraint is worth that risk
# for a knob that is off. `.autohelix/notes/iter-3.md` §2 and §4 have the
# detail, including why a probe's TPOT is not comparable (nonsense logits
# collapse the drafter's acceptance; convert with `TPOT x (1 + 5 x accept)`).
#: **A latency probe, not a deployment knob.** `DEEPSEEK_V41_PROBE_AR_WIDTH=w`
#: routes every *world-wide* all-reduce on the forward path to the
#: consecutive-rank group of `w` ranks instead of all 64. The sums are then
#: partial — each rank gets w/64 of the shards — so **the tokens are nonsense**
#: and nothing but a latency may be read off a run with this set.
#:
#: It exists because it isolates the one term a TP x PP factorization moves.
#: TP x PP = 64 in every candidate, so per-rank arithmetic and per-rank weight
#: footprint (384 experts x 41 layers / 64 = 246 slots) are identical at TP=4,
#: 8, 16 and 64; what differs is how many ranks — and, iteration 3's result,
#: how many *devices* — each tensor-parallel all-reduce spans. Rank r is on
#: device r // 4, so width 4 is one device, 8 is two column-adjacent devices
#: (the floorplan's scheme-c group, one torus hop), 16 is four devices.
#:
#: Iteration 3 measured width 8 by substituting the EP-TP group: -73.7 ms of a
#: 947.4 ms prefill and -82.5 ms of a 403 ms decode step. This generalizes that
#: probe to any width, so TP=4 x PP=16 and TP=16 x PP=4 can be priced the same
#: way without first building a pipeline.
#:
#: Requires `DEEPSEEK_V41_LANE_WIDTHS` to have created the group. Never leave
#: it set: `verifier` would fail on gibberish.
_PROBE_AR_WIDTH = int(os.environ.get("DEEPSEEK_V41_PROBE_AR_WIDTH", "0") or 0)

#: `[group]` once the model has resolved the probe's lane group, `[None]` when
#: the probe is off. Nothing reads it unless `_PROBE_AR_WIDTH` is set.
_NARROW_AR_GROUP: list = [None]

#: **A pipeline-latency emulation, and the whole of what this port can measure
#: about pipeline parallelism without a stage-aware weight plan.**
#: `DEEPSEEK_V41_PP_EMULATE=p` runs the forward with the *latency structure* of
#: TP=(64/p) x PP=p on all 64 real ranks, real weights and real collectives:
#:
#:   * **blocks are divided between p stages** and a rank executes only its
#:     own stage's — so the critical path is, as in a real pipeline, the sum
#:     over all 40 blocks of *one stage's* cost for that block;
#:   * **each block's expert-parallel all-reduce reduces over the stage**, the
#:     lane group of 64/p consecutive ranks, instead of over the world;
#:   * **each block's routed-expert loop runs p times**, because a rank of a
#:     64/p-wide tensor-parallel group owns `384/(64/p) = 6p` expert slots per
#:     block where a rank of the flat world owns 6. Repeating the plan gives
#:     the right launch count, the right FLOPs and the right HBM read volume —
#:     the same experts are re-read, and they are resident, so a byte from HBM
#:     costs what a byte from HBM costs;
#:   * **each stage boundary moves the hyper-connection stream** to every rank,
#:     including one after the last stage so the vocab-sharded head and the
#:     sampler are reached unchanged.
#:
#: What it does **not** model, in both directions, because honesty about this
#: is the whole value of the number:
#:
#:   * the tokens are **wrong**. Each block's expert sum covers 64/p of the 64
#:     expert shards, and the blocks a rank skips leave their KV unwritten. So
#:     this is a latency probe exactly like `_PROBE_AR_WIDTH`, and `verifier`
#:     must never be run under it.
#:   * the transport is **pessimistic**: two world-wide all-reduces per
#:     boundary (~1.9 ms each, iteration 3) where a real pipeline would use a
#:     neighbour permute — `kernels/pp_permute.py` priced the wire at ~0.4 ms a
#:     boundary at 1024 tokens. So the emulation overstates a real pipeline's
#:     transport by ~3 ms a boundary.
#:   * per-rank **weight footprint** is unchanged (13.16 GiB), where a real
#:     pipeline would hold 1/p of the replicated dense weights and the same
#:     258 expert slots. That is a capacity difference, not a latency one.
#:   * it holds `MAX_NUM_SEQS` at whatever the engine was built for, so it
#:     measures the pipeline **without microbatching** — which is the
#:     configuration this port is actually in, and the one the floorplan's own
#:     `decode_micro_batch: 4` / `pipeline_chunks: true` are designed to avoid.
#:
#: Run it with `DEEPSEEK_V41_NO_SPEC_DECODE=1`. The DSpark taps come off
#: backbone layers 37-39, which land in the last stage, so with speculation on
#: the ranks of other stages would reach `_draft` with no `main_hidden` and
#: take a different path through its collectives — a mismatched collective
#: across 64 ranks is the one failure this deployment must not have.
_PP_EMULATE = int(os.environ.get("DEEPSEEK_V41_PP_EMULATE", "0") or 0)

#: `DEEPSEEK_V41_TRACE_INPUTS=1` logs each forward's inputs on rank 0; see `_trace_inputs`.
_TRACE_INPUTS = os.environ.get("DEEPSEEK_V41_TRACE_INPUTS", "") not in ("", "0")
_TRACE_STEP = [0]
#: `DEEPSEEK_V41_DUMP_DIR=<dir>` saves rank 0's per-layer live rows; see `_dump`.
_DUMP_DIR = os.environ.get("DEEPSEEK_V41_DUMP_DIR", "")
_DUMP_STEP = [0]
_DUMP_LIVE = [0]


def tp_all_reduce(x: torch.Tensor, tp_group) -> torch.Tensor:
    """Sum `x` across the tensor-parallel ranks. Returns the reduced tensor."""
    if _PROBE_AR_WIDTH and _NARROW_AR_GROUP[0] is not None:
        if tp_group.world_size > _PROBE_AR_WIDTH:
            tp_group = _NARROW_AR_GROUP[0]
    name = _group_name(tp_group)
    if x.numel() <= _ALLREDUCE_CHUNK_ELEMS:
        return _compiled(_fc_sum)(x, name)
    return _compiled(_fc_sum_chunked)(x, name, _ALLREDUCE_CHUNK_ELEMS)


def _fc_sum_chunked(x: torch.Tensor, group_name: str, chunk: int) -> torch.Tensor:
    """`_fc_sum` over `x` in `chunk`-element pieces, as one compiled step."""
    flat = x.reshape(-1)
    parts = [
        _fc_sum(flat[lo : lo + chunk], group_name)
        for lo in range(0, flat.numel(), chunk)
    ]
    return torch.cat(parts).reshape(x.shape)


def tp_all_gather_last(x: torch.Tensor, tp_group) -> torch.Tensor:
    """Gather `x` across the tensor-parallel ranks on the last axis."""
    return _compiled(_fc_gather_last)(x, _group_name(tp_group))


#: The plugin's `torch.compile` backend name, resolved once by `_compile_backend`.
_COMPILE_BACKEND: str | None = None


def _compile_backend() -> str:
    """The backend name every `_nki_entry` compiles against, resolved outside a trace.

    `_nki_entry` is `lru_cache`d, so its first call for a kernel happens wherever
    that kernel is first used — inside the target's own `torch.compile`d forward,
    with dynamo inlining this helper. Resolving the name there through
    `vllm_neuron.envs.get_compile_backend_name` reaches lite's `register()`,
    which on `libtorch-neuronx-lite` 2.13 probes `torch._dynamo.lookup_backend`,
    a function dynamo refuses to trace:

        torch._dynamo.exc.Unsupported: Attempted to call function marked as
        skipped ... `lookup_backend` ... file matches MOD_SKIPLIST

    So the name is resolved once, eagerly, from `DeepseekV41ForCausalLM.__init__`
    (plain Python, before any compile) and read back as a module constant from
    then on, which dynamo handles as a string literal.
    """
    global _COMPILE_BACKEND
    if _COMPILE_BACKEND is None:
        from vllm_neuron.envs import get_compile_backend_name

        _COMPILE_BACKEND = get_compile_backend_name()
    return _COMPILE_BACKEND


@functools.lru_cache(maxsize=None)
def _nki_entry(module: str, name: str = "kernel", grid: int = 0):
    """A compiled entry point for `kernels.<module>.<name>`.

        torch._dynamo.exc.FailOnRecompileLimitHit: Hard failure due to
        fullgraph=True

    `grid` is the NKI launch grid. 0 means `wrap_nki`'s own `[]`, which
    `compile_nki` turns into `lnc = 1`: one function, no `lnc_id`, so only one
    of the two physical NeuronCores of a logical core at LNC=2 has code to run.
    A positive `grid` asks for that many program instances —
    `wrap_nki(kernel)[2]` — which is only correct for a kernel that divides its
    own work by `nl.program_id(0)`. A kernel written for one core run at grid 2
    computes the same result twice over the same data, so this is opt-in per
    entry point and the two-core kernels assert on the grid they get.
    """
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .kernels import (
        blockquant,
        compress_pool,
        engram_gate,
        hc_attn_out,
        hc_collapse,
        hc_expand,
        hc_mixes,
        index_score,
        rmsnorm,
        sparse_attn as sparse_attn_kernel,
        swiglu as swiglu_kernel,
    )

    _raise_dynamo_limits()
    modules = {
        "blockquant": blockquant,
        "compress_pool": compress_pool,
        "engram_gate": engram_gate,
        "hc_attn_out": hc_attn_out,
        "hc_collapse": hc_collapse,
        "hc_expand": hc_expand,
        "hc_mixes": hc_mixes,
        "index_score": index_score,
        "rmsnorm": rmsnorm,
        "sparse_attn": sparse_attn_kernel,
        "swiglu": swiglu_kernel,
    }
    wrapped = wrap_nki(getattr(modules[module], name))
    if grid:
        wrapped = wrapped[grid]

    def entry(*args):
        return wrapped(*args)

    # No `grid` in the name: the entry is `lru_cache`d on `(module, name, grid)`
    # so nothing needs it to be distinct, the kernel names already carry
    # `_2core`, and keeping the string identical to the one-core form is what
    # lets a graph iteration 6 compiled still be found in /mnt/nvme/neuron-cache.
    entry.__name__ = f"nki_{module}_{name}"
    return torch.compile(
        entry,
        backend=_compile_backend(),
        fullgraph=True,
        options=_COMPILE_OPTIONS,
    )


def nki_topk(tensor: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-`k` along the last axis, through this repo's NKI top-k.

        %sort.50 = (f32[128,384], s32[128,384]) sort(...) metadata={
            op_type="aten__topk" op_name="Gate[G['gate']]/aten__topk" ...}
        FATAL: neuronx-cc compilation failed with 70.
    """
    from vllm_neuron.functional.topk import topk as _topk

    return _topk(tensor, k, -1, -1)


#: Which block-quant storages run their GEMM across **both** physical cores of
#: every logical NeuronCore. At LNC=2 a logical core is a pair of
#: NeuronCore-v3, and `wrap_nki` launches at grid 1, so a kernel's code only
#: exists for one of the pair: half the machine idles. `fp4_linear_2core`
#: divides its output columns by `nl.program_id(0)` and is launched at grid 2,
#: so the two per-core functions' op counts *sum* to the grid-1 total instead of
#: both doing the whole GEMM (`kernels/blockquant.py:574` records the lowering
#: check that establishes this).
#:
#: Measured on device, same prompts, 64 ranks, against the undivided kernel:
#:
#:     `fp4` (the routed experts)   in512_bs1 TTFT  947.9 -> 881.2 ms  (-7.0%)
#:                                             TPOT  142.8 -> 121.4 ms (-15.0%)
#:                                  in1024_bs1 TTFT 1504.7 -> 1432.1   (-4.8%)
#:                                  in2048_bs1 TTFT 3067.3 -> 2930.5   (-4.5%)
#:
#: and iteration 5 measured 888.2 ms for the same arm at the same settings, a
#: 0.8% agreement across two independent implementations of the split.
#:
#: `fp8` is deliberately **not** in the default. It is worth a further ~-5% and
#: iteration 5 verified it on device, but fp8 appears in every graph holding an
#: attention projection and re-keys ~125 distinct graph shapes per context,
#: measured at >48 min of compile at context 8448 without completing — and it
#: flips no cell that `fp4` has not already flipped. There is no `fp8_*_2core`
#: kernel in this tree for the same reason; iter-5.md §9 and iter-6.md §5/§11
#: hold its implementation notes, including that `nki.simulate` cannot run any
#: fp8 kernel here (`OverflowError` on `E4M3_SGN2` in the simulator's numpy
#: backend, in the undivided kernel too).
#: The storages a `_2core` kernel exists for. Intersected with the request
#: below, so asking for one that has no split kernel is inert rather than a
#: claim in the log that something was split when it was not.
#:
#: `fp8` joined this on 2026-10-20, when `blockquant.fp8_linear_2core` /
#: `fp8_linear_raw_2core` were written back from `iter-5.md` §8-§9 — the
#: iteration that measured them (-5.2% TTFT on top of fp4, TPOT 111.2 ms) was
#: rejected for an unrelated verifier failure and its worktree went with it, so
#: three later reviews recorded `fp4,fp8` as **inert**: the flag was accepted
#: and silently did nothing because this set had only `fp4` in it.
_BQ_2CORE_AVAILABLE = frozenset({"fp4", "fp8"})

#: **`fp4,fp8` since 2026-10-20**, measured on the integrated tree at
#: `in512_bs1`, 64 ranks, against the fp4-only arm this repo shipped:
#:
#:     TTFT   873.2 -> 835.1 ms  (-4.4%)      iteration 5 measured 841.8
#:     TPOT   129.9 -> 116.9 ms  (-10.0%)     iteration 5 measured 111.2
#:     decode   7.03 -> 7.68 tok/s (+9.2%)
#:     acceptance 36.4% -> 36.8%  (unchanged, which is the correctness signal:
#:                a wrong scale slice would collapse draft agreement)
#:
#: Two independent implementations five iterations apart agreeing to within 0.8%
#: on TTFT is the evidence that the split is real rather than a measurement
#: artefact. The compile cost is also real and is the reason this was not the
#: default before: fp8 appears in every graph holding an attention projection
#: and re-keys ~125 graphs a context, ~12 min at context 768.
_BQ_2CORE = _BQ_2CORE_AVAILABLE & frozenset(
    part.strip()
    for part in os.environ.get("DEEPSEEK_V41_BQ_2CORE", "fp4,fp8").split(",")
    if part.strip() and part.strip() != "none"
)

#: Programs in the two-core launch grid. Not a knob: the kernels assert on it.
_TWO_CORE_PROGRAMS = 2

#: Logged once per process, so a server log says which arm ran.
_BQ_2CORE_LOGGED = [False]

#: `DEEPSEEK_V41_BQ_FUSED=fp4` routes the routed-expert GEMMs to
#: `blockquant.fp4_linear_fused`, which decodes the weight in SBUF instead of
#: writing a bf16 copy to private HBM and reading it straight back. The
#: undivided `fp4_linear` moves ~47 MB of HBM per linear for a 5.9 MB weight at
#: `n=2304, k=5120`, so the round trip is the obvious thing to remove — and it
#: is a *fixed* cost per GEMM, which is the term that dominates both the 512-
#: token cell's TTFT and the decode step.
#:
#: **It loses, and by a lot.** Measured at `in512_bs1`, 64 ranks, three cells in
#: one session, nothing else changed:
#:
#:     fp4_linear_2core  (default)   TTFT  876.5 ms   TPOT 131.5 ms   6.93 tok/s
#:     fp4_linear        (iter-4)    TTFT  947.9 ms   TPOT 142.8 ms   6.25 tok/s
#:     fp4_linear_fused              TTFT 1313.3 ms   TPOT 163.2 ms   5.71 tok/s
#:
#: i.e. +50% TTFT and +24% TPOT against the default, and +39% / +14% against the
#: undivided kernel it is a drop-in for — so the round trip it removes is not
#: what the GEMM was spending its time on. `_gemm_fp4_fused` re-decodes the
#: weight inside the k loop and `_fp4_plan` caps the n-chunk by PSUM capacity,
#: so it trades ~47 MB of HBM for a much narrower stationary operand and more
#: decode work per matmul. It is also cheaper to *compile*: this cell's first
#: warm-up request took 24 s against 155 s for the two-core arm.
#:
#: This settles the question iter-6.md §15 named as the first thing to try, and
#: the inconclusive measurement in the kernel's own docstring. Kept reachable by
#: one variable so a later iteration can re-measure it against a *changed*
#: expert step (e.g. one that decodes a weight once for many m-tiles) rather
#: than rediscovering the kernel — but as written, hoisting the dequantization
#: out of the GEMM is not where `F` goes.
#:
#: Not combinable with the two-core split: there is no `fp4_linear_fused_2core`,
#: so asking for fused gives up the split, which is part of why it loses.
_BQ_FUSED = _BQ_2CORE_AVAILABLE & frozenset(
    part.strip()
    for part in os.environ.get("DEEPSEEK_V41_BQ_FUSED", "").split(",")
    if part.strip() and part.strip() != "none"
)


def _log_two_core_plan() -> None:
    """Say once, on the logging rank, which GEMMs use both physical cores.

    Without this there is no way to tell from a server log which arm produced a
    measurement, which iteration 4 had to infer from a compile that only a new
    graph could explain.
    """
    if _BQ_2CORE_LOGGED[0] or not _LOG_RANK[0]:
        return
    _BQ_2CORE_LOGGED[0] = True
    if _BQ_FUSED:
        logger.warning(
            "deepseek_v41: DEEPSEEK_V41_BQ_FUSED=%s — storage %s runs "
            "`fp4_linear_fused` (weight decoded in SBUF, no bf16 round trip "
            "through private HBM) on ONE physical core, which overrides the "
            "two-core split for that storage. Measured HALF AS FAST AGAIN "
            "at TTFT and a quarter slower at TPOT; see nki_linear.",
            os.environ.get("DEEPSEEK_V41_BQ_FUSED", ""),
            ",".join(sorted(_BQ_FUSED)),
        )
    if not _BQ_2CORE:
        logger.info(
            "deepseek_v41: block-quant GEMMs run on ONE physical core per "
            "logical NeuronCore (DEEPSEEK_V41_BQ_2CORE=%r). The measured "
            "default is 'fp4'; see nki_linear.",
            os.environ.get("DEEPSEEK_V41_BQ_2CORE", ""),
        )
        return
    split = _BQ_2CORE - _BQ_FUSED
    if not split:
        return
    logger.info(
        "deepseek_v41: block-quant GEMMs on BOTH physical cores for storage "
        "%s (grid=%d, DEEPSEEK_V41_BQ_2CORE=%r)",
        ",".join(sorted(split)), _TWO_CORE_PROGRAMS,
        os.environ.get("DEEPSEEK_V41_BQ_2CORE", "fp4"),
    )


def _two_core(storage: str, n: int) -> bool:
    """Does this GEMM run split across both physical cores?

    `n` has to divide into `_TWO_CORE_PROGRAMS` shares that are each a whole
    number of 128-row tensor-engine tiles, which is what the kernel asserts. A
    projection width that cannot be halved that way falls back to the undivided
    kernel, so a new shape is slower rather than wrong.
    """
    return storage in _BQ_2CORE and n % (_TWO_CORE_PROGRAMS * 128) == 0


def nki_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    storage: str,
    quantize_activation: bool = True,
    out_fp32: bool = False,
) -> torch.Tensor:
    """`x @ weight.T` on the device, through the matching NKI kernel.

    `out_fp32` (fp8 storage only) returns the fp32 accumulator instead of a
    bf16 rounding of it, for partial sums that are reduced across ranks.
    """
    k = x.shape[-1]
    flat = x.reshape(-1, k)
    rows = flat.shape[0]
    if out_fp32:
        if storage != "fp8" or not quantize_activation:
            raise ValueError("out_fp32 is implemented for the quantizing fp8 path only")
        out = _nki_entry("blockquant", "fp8_linear_f32")(flat, weight, scale)
        return out[:rows].reshape(*x.shape[:-1], out.shape[-1])
    if storage == "bf16":
        out = _nki_entry("blockquant", "bf16_linear")(flat, weight)
    elif storage == "fp32w":
        out = _nki_entry("blockquant", "fp32w_linear")(flat, weight)
    elif storage == "fp8":
        name = "fp8_linear" if quantize_activation else "fp8_linear_raw"
        if _two_core("fp8", int(weight.shape[0])):
            out = _nki_entry(
                "blockquant", f"{name}_2core", _TWO_CORE_PROGRAMS
            )(flat, weight, scale)
        else:
            out = _nki_entry("blockquant", name)(flat, weight, scale)
    elif storage == "fp4":
        if "fp4" in _BQ_FUSED and k % 64 == 0:
            out = _nki_entry("blockquant", "fp4_linear_fused")(flat, weight, scale)
        elif _two_core("fp4", int(weight.shape[0])):
            out = _nki_entry(
                "blockquant", "fp4_linear_2core", _TWO_CORE_PROGRAMS
            )(flat, weight, scale)
        else:
            out = _nki_entry("blockquant", "fp4_linear")(flat, weight, scale)
    else:
        raise ValueError(f"unknown storage {storage!r}")
    return out[:rows].reshape(*x.shape[:-1], out.shape[-1])


class BlockQuantLinear(nn.Module):
    """A linear whose weight is stored block-scaled, as the checkpoint has it.

        fp8  `[out, in]` e4m3 + `[ceil(out/32), ceil(in/32)]`   attention, shared experts
        fp4  `[out, in//2]` two e2m1 per byte + `[out, in/32]`   routed experts
        bf16 `[out, in]`, no scale                               compressor, indexer wk
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
        """The weight in `dtype`."""
        from .weight_loader import dequant_block_fp4, dequant_block_fp8

        if self.storage == "fp8":
            return dequant_block_fp8(self.weight, self.scale, self.block, dtype)
        if self.storage == "fp4":
            return dequant_block_fp4(self.weight, self.scale, self.block, dtype)
        return self.weight.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x @ weight.T` in a NKI kernel. The only path; there is no fallback."""
        return nki_linear(
            x, self.weight, getattr(self, "scale", None), self.storage
        )

    def grouped(self, x: torch.Tensor, group: int, width: int) -> torch.Tensor:
        """One block of a block-diagonal projection: rows
        `[group*width, (group+1)*width)` against `x`.
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
        """The torch emulation of `forward`, for accuracy comparison only."""
        if self.storage == "bf16":
            return F.linear(x, self.weight.to(x.dtype))
        x_q, x_scale = act_quant(x, self.block)
        activation = x_q.to(torch.float32) * x_scale
        weight = self.dequantized(torch.float32)
        return F.linear(activation, weight).to(x.dtype)


def _round_to_bits(x: torch.Tensor, mantissa: int, min_exp: int) -> torch.Tensor:
    """Round to a float format with `mantissa` explicit bits, arithmetically.

        FATAL: Unknown custom-call API version enum value: 0
        (API_VERSION_UNSPECIFIED)
    """
    magnitude = x.abs().clamp_min(2.0**min_exp)
    exponent = torch.floor(torch.log2(magnitude))
    step = _pow2(exponent - mantissa)
    return torch.round(x / step) * step


#: e4m3: 3 explicit mantissa bits, smallest normal 2**-6.
_FP8_MANTISSA_BITS, _FP8_MIN_EXP = 3, -6
#: e2m1: 1 explicit mantissa bit, smallest normal 2**0.
_FP4_MANTISSA_BITS, _FP4_MIN_EXP = 1, 0


def act_quant_dequant(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """`act_quant(..., inplace=True)`: round-trip `x` through fp8 and return it."""
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
    """
    *lead, width = x.shape
    blocks = width // block
    if width % block:
        raise ValueError(f"fp4 quant needs a multiple of {block}, got {width}")
    grouped = x.float().reshape(*lead, blocks, block)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    if scale_e4m3:
        amax = amax.clamp_min(6 * 2.0**-9)
        scale = _round_to_bits(amax / _FP4_MAX, _FP8_MANTISSA_BITS, _FP8_MIN_EXP)
        # An e4m3 scale is not a power of two, and this host's compiled fp32
        # division is a multiply by an approximate reciprocal: one ulp off on
        # ~28% of elements (measured 2026-10-22, trn3.48xlarge). The reference
        # divides exactly, then rounds to e2m1 half-to-even; a bf16 latent over
        # an e4m3 scale lands on an exact tie (x/s = 5.0, 2.5, ...) often, and
        # the ulp decides which way it goes, so the compressed KV came out 1.2%
        # off the reference with 50% errors on tie elements. One Newton step on
        # the quotient restores the exact value and the reference's rounding.
        normalized = grouped / scale
        normalized = normalized - (normalized * scale - grouped) / scale
        normalized = normalized.clamp(-_FP4_MAX, _FP4_MAX)
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
    """Round to the nearest `e2m1` value by table lookup, keeping the sign."""
    midpoints = _FP4_MIDPOINTS.to(x.device)
    magnitudes = _FP4_MAGNITUDES.to(x.device)
    bucket = torch.bucketize(x.abs(), midpoints)
    return torch.copysign(magnitudes[bucket], x)


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
    """Rotary frequencies as complex exponentials, one row per position."""
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
    return torch.cos(angles), torch.sin(angles)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    """Rotate `x`, taking **adjacent element pairs** as complex numbers."""
    cos, sin = freqs_cis
    if inverse:
        sin = -sin
    pairs = x.float().unflatten(-1, (-1, 2))
    a, b = pairs[..., 0], pairs[..., 1]
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
    """`length` rows of the RoPE table starting at `start_pos`, gathered."""
    positions = start_pos + torch.arange(length, device=device)
    return slice_freqs(freqs_cis, positions)


#: An empty top-k slot: a window position before the sequence started, or a
#: compressed position a query cannot reach yet. `sparse_attn` skips it.
EMPTY_SLOT = -1

#: How many queries share one gather. See `sparse_attn`.
_ATTN_QUERY_TILE = 512

#: The slot-list granularity `kernels/sparse_attn` works in (its partition
#: width). The caller pads the top-k list up to a multiple of this with empty
#: slots, which costs arithmetic but nothing in accuracy.
_ATTN_SLOT_TILE = 128

#: Mirrors `kernels/sparse_attn.BIAS_PAD`: how many fp32 columns of the bias row
#: come before the bias proper, carrying `softmax_scale`. Kept here as a literal
#: rather than imported because `sparse_attn` runs inside a dynamo trace and the
_ATTN_BIAS_PAD = 16

#: Bucket for the compressed-KV and index-key reads. `compress_len` grows with
#: the decode position and is a tensor *shape*, so reading exactly that many
#: groups makes every layer recompile once per token. Rounding the read up to a
_COMPRESS_TILE = 128

#: Context-sized floor for the same read, in **tokens**, or `None` to round to
#: `_COMPRESS_TILE` and no further. Set once per engine by
#: `bind_kv_caches` (`_bind_compress_read_ceiling`).
_COMPRESS_READ_CEILING: int | None = None

#: Above this many KV slots, `_cap_kv_cache_blocks` leaves vLLM's memory-based
#: sizing alone — a backstop against a configuration so large that the cap's own
#: arithmetic is the thing that is wrong.
_KV_CAP_MAX_SLOTS = 131072


def _hbm_used_bytes() -> int | None:
    """Device HBM in use, straight from the Neuron runtime's own counters."""
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

#: How many forwards log their HBM use **per layer**. The per-layer detail
#: identifies which step grows the live set; the per-forward summary shows
#: whether it comes back down, and that one is always logged.
_HBM_TRACE_FORWARDS = int(os.environ.get("DEEPSEEK_V41_HBM_TRACE", 0))

#: `[steps seen, previous reading, previous layer id]` for `_hbm_mark`.
_HBM_STATE = [0, None, -1]

#: `[is this the rank that logs?]`, set by the model's `__init__` once the TP
#: group is known. 64 ranks emitting the same per-forward line is 64x the log
#: for 1x the information, and the per-rank facts worth having (which experts,
_LOG_RANK = [True]

#: `[live row count]`, the MoE's real token count while the backbone's layer
#: loop is running, or `None`.
_MOE_LIVE_ROWS: list[int | None] = [None]

#: Rows below which `_collect_layer_garbage` is not worth its own cost. A
#: decode step's whole stream is a few hundred KiB; a collection costs tens of
#: milliseconds and would be paid 43 times per generated token.
_GC_MIN_ROWS = 64

#: Collect every Nth layer rather than every layer.
_GC_STRIDE = max(1, int(os.environ.get("DEEPSEEK_V41_GC_LAYER_STRIDE", 4)))

#: `[collections, seconds]` since the last forward boundary, for `_hbm_mark`'s
#: per-forward line. Instrumentation only.
_GC_STATE = [0, 0.0]

#: HBM, per rank, to spend holding routed experts between steps — the expert
#: cache `MoE` keeps. 0 disables it and restores the single-staging-slot
#: behaviour exactly.
_EXPERT_CACHE_BYTES = int(
    os.environ.get("DEEPSEEK_V41_EXPERT_CACHE_BYTES", 8 << 30)
)

#: HBM, per rank, to spend holding engram table shards. Both tables together, so
#: 6 GiB covers the 5.72 GiB two 64-rank shards come to in bf16.
_ENGRAM_TABLE_BYTES = int(
    os.environ.get(
        "DEEPSEEK_V41_ENGRAM_TABLE_BYTES",
        # Under the pipeline scheme a stage holds at most one table, sharded over
        # its 8 ranks (~23 GiB bf16 a rank); the flat scheme two 64-rank shards.
        (26 << 30) if os.environ.get("DEEPSEEK_V41_SCHEME", "tp64") == "tp8pp8" else (6 << 30),
    )
)

#: True once `weight_loader.load_weights` has established that every weight this
#: rank needs is in HBM. What it gates is the *host-side housekeeping* the
#: streaming design needed and a resident one does not:
_RESIDENT_WEIGHTS = [False]


def host_housekeeping_needed() -> bool:
    """Does the forward have host allocations to clean up after?"""
    if os.environ.get("DEEPSEEK_V41_FORCE_HOST_HOUSEKEEPING") == "1":
        return True
    return not _RESIDENT_WEIGHTS[0]


#: `[hits, misses, admissions]` since the last forward boundary, summed over
#: every `MoE`. Instrumentation only.
_EXPERT_CACHE_STATS = [0, 0, 0]

#: Bytes one expert occupies in its stored form, filled in from the buffers
#: themselves by `allocate_staging_buffers` so the fetched-bytes figure in the
#: per-forward log is measured rather than remembered. 18,800,640 for this
_EXPERT_BYTES = [0]

#: Use counts are halved once the hottest expert in a layer reaches this, so an
#: expert that was popular early cannot hold a slot forever.
_EXPERT_USE_CEILING = 64

#: Split the per-expert cost into "get the weights onto the device" and "launch
#: the GEMM", and report both in `_hbm_mark`'s per-forward line.
_EXPERT_TIMING = os.environ.get("DEEPSEEK_V41_TIME_EXPERTS", "0") != "0"

#: `[fetch+stage, launch, read, host->device]` seconds since the last forward
#: boundary. The first is the sum of the last two plus the cache lookup.
_EXPERT_TIME = [0.0, 0.0, 0.0, 0.0]

#: Layers at which `_census_device_tensors` reports. Early, middle and late, so
#: the difference between them is what the layer loop accumulates.
_CENSUS_LAYERS = frozenset({4, 20, 39})


def _collect_layer_garbage() -> None:
    """Free the previous layer's device tensors, including the ones in cycles.

        HBM  11.105 GiB (     +0.0 MiB) at layer 0 of forward 0, 512 rows
        HBM  11.648 GiB (    +46.8 MiB) at layer 4 of forward 0, 512 rows
        HBM  13.430 GiB (    +55.7 MiB) at layer 39 of forward 0, 512 rows
    """
    if not host_housekeeping_needed():
        return
    start = time.perf_counter()
    gc.collect()
    _GC_STATE[0] += 1
    _GC_STATE[1] += time.perf_counter() - start


def _census_device_tensors(layer_id: int) -> None:
    """Log what is alive on the device, by shape, biggest first."""
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
        "deepseek_v41: live device tensors at layer %d: %.3f GiB in "
        "%d shapes", layer_id, total / 1024**3, len(sizes),
    )
    for (shape, dtype), (count, nbytes) in ranked:
        logger.info(
            "deepseek_v41:   %8.1f MiB  x%-5d %s %s",
            nbytes / 1024**2, count, dtype, shape,
        )


def _hbm_mark(layer_id: int, rows: int) -> None:
    """Log HBM use at one layer of a forward, with the delta since the last."""
    if layer_id <= _HBM_STATE[2]:
        from .weight_loader import EXPERT_STREAM_STATS

        hits, misses, admitted = _EXPERT_CACHE_STATS
        served, stalled, inline = EXPERT_STREAM_STATS
        if (hits or misses) and (_LOG_RANK[0] or misses):
            logger.info(
                "deepseek_v41: forward %d experts %d/%d from HBM "
                "(%.0f%%), %d admitted, ~%.2f GiB fetched; prefetch %d served "
                "(%d stalled, %d inline); gc %d collections in %.2f s"
                "; experts %.1f s staging (%.1f read + %.1f h2d) "
                "+ %.1f s launching",
                _HBM_STATE[0], hits, hits + misses,
                100.0 * hits / max(1, hits + misses), admitted,
                misses * _EXPERT_BYTES[0] / 1024**3,
                served, stalled, inline,
                _GC_STATE[0], _GC_STATE[1],
                _EXPERT_TIME[0], _EXPERT_TIME[2], _EXPERT_TIME[3],
                _EXPERT_TIME[1],
            )
        EXPERT_STREAM_STATS[:] = [0, 0, 0]
        _EXPERT_CACHE_STATS[:] = [0, 0, 0]
        _EXPERT_TIME[:] = [0.0, 0.0, 0.0, 0.0]
        _GC_STATE[:] = [0, 0.0]
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
        "deepseek_v41: HBM %7.3f GiB (%+9.1f MiB) at layer %d of "
        "forward %d, %d rows",
        used / 1024**3,
        0.0 if previous is None else (used - previous) / 1024**2,
        layer_id, _HBM_STATE[0], rows,
    )


def _heap_bytes() -> int:
    """This process's `[heap]` mapping, in bytes, or 0 if it has none."""
    try:
        with open("/proc/self/maps") as handle:
            for line in handle:
                if line.rstrip().endswith("[heap]"):
                    lo, hi = line.split(None, 1)[0].split("-")
                    return int(hi, 16) - int(lo, 16)
    except OSError:  # pragma: no cover - /proc is always there on Linux
        pass
    return 0


_LIBC = None

#: How often `forward`'s layer loop samples (and trims) the host heap. Every
#: 20th of 40 layers: two `malloc_trim` calls and two log lines a forward, which
#: is enough to notice the heap starting to grow and little enough to ignore.
_HEAP_TRIM_EVERY = 20


def host_heap_trim(tag: str, world_size: int) -> None:
    """Instrument — and, where there is any, reclaim — the host heap.

        TDRV:vring_allocate_next_desc  Failed to expand vring page pool
        LLVM ERROR: out of memory                     (neuronx-cc, mid-forward)
        MemoryError                                   (worker, in execute_model)
    """
    global _LIBC
    if world_size <= 1:
        return
    if not host_housekeeping_needed():
        return
    if _LIBC is None:
        import ctypes

        try:
            _LIBC = ctypes.CDLL("libc.so.6")
        except OSError as exc:  # pragma: no cover - glibc is always present
            logger.info("deepseek_v41: malloc_trim unavailable (%s)", exc)
            _LIBC = False
    if not _LIBC:
        return
    before = _heap_bytes()
    try:
        _LIBC.malloc_trim(0)
    except Exception as exc:  # noqa: BLE001 - never fatal; it is only hygiene
        logger.info("deepseek_v41: malloc_trim failed (%s)", exc)
        _LIBC = False
        return
    after = _heap_bytes()
    logger.info(
        "deepseek_v41: host heap %.2f -> %.2f GiB after %s "
        "(malloc_trim returned %.2f GiB)",
        before / 1024**3, after / 1024**3, tag, (before - after) / 1024**3,
    )


def _compress_read_len(compress_len: int, ratio: int = 1) -> int:
    """How many compressed groups to read for `compress_len` live ones."""
    if compress_len <= 0:
        return 0
    want = compress_len
    if _COMPRESS_READ_CEILING:
        want = max(want, -(-_COMPRESS_READ_CEILING // max(ratio, 1)))
    return -(-want // _COMPRESS_TILE) * _COMPRESS_TILE

#: The finite lower bound the reference kernel starts its running max at. A row
#: whose every slot is empty would give exp(-inf - -inf) = NaN with a true -inf;
#: with this bound the sink term overflows to +inf and the row comes out zero,
_NEG_BOUND = -1e30

#: `_NEG_BOUND`'s mirror, for pinning a block *in* to a top-k.
_POS_BOUND = 1e30


def _bound(x: torch.Tensor, value: float) -> torch.Tensor:
    """`value` as a 0-d tensor in `x`'s dtype.

        [ERROR] [NCC_ESPP004] f64 dtype is not supported.
        RuntimeError: neuronx-cc compilation failed with 70.
    """
    return x.new_full((), value)


def window_topk_idxs(
    window_size: int, seqlen: int, start_pos: torch.Tensor, device
) -> torch.Tensor:
    """Which rows of the gathered window each query attends to; -1 is empty."""
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

    tiles: list[torch.Tensor] = []

    kpad = -(-topk // _ATTN_SLOT_TILE) * _ATTN_SLOT_TILE
    sink = attn_sink.float().reshape(h, 1)
    assert b == 1, (
        f"sparse_attn gathers from kv[{b}, n, d]; the kernel takes one table, "
        "and every producer in this model (`Attention._window_kv`, "
        "`DSparkAttention.forward`) reshapes to a leading 1"
    )
    for lo in range(0, s, _ATTN_QUERY_TILE):
        hi = min(lo + _ATTN_QUERY_TILE, s)
        t = hi - lo
        idxs = topk_idxs[:, lo:hi]
        empty = idxs == EMPTY_SLOT

        bias = empty.float() * _NEG_BOUND
        gidx = idxs.clamp_min(0).int()
        if kpad != topk:
            gidx = torch.cat([gidx, gidx.new_zeros(b, t, kpad - topk)], dim=2)
            bias = torch.cat(
                [bias, bias.new_full((b, t, kpad - topk), _NEG_BOUND)], dim=2
            )
        bias = torch.cat(
            [bias.new_full((b, t, _ATTN_BIAS_PAD), softmax_scale), bias], dim=2
        )

        tiles.append(
            _nki_entry("sparse_attn", "gathered_attn_idx")(
                q[:, lo:hi].reshape(-1, d),
                kv.reshape(n, d),
                gidx.reshape(-1, kpad),
                bias.reshape(-1, _ATTN_BIAS_PAD + kpad),
                sink,
            ).reshape(b, t, h, d)
        )
    return tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=1)


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: torch.Tensor | int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of the two-level top-k: keep the best `topk_blocks` blocks."""
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

        %positions : = call_function[target=operator.add](args = (167, %arange))
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
        "host_table",
        "host_slots",
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
        host_table: torch.Tensor | None = None,
        host_slots: torch.Tensor | None = None,
    ) -> None:
        self.block_table = block_table
        self.block_size = block_size
        self.slot_mapping = slot_mapping
        #: Host copies of `block_table` / `slot_mapping` (`_paged_spans` has them before it
        #: moves the table to the device), so the host index work reads neither back.
        self.host_table = host_table
        self.host_slots = host_slots
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
        read_lens, counts = [0, 0, 0], [0, 0, 0]
        for ratio in (1, 2):
            compress_len = (start_pos + length) // ratio
            read_lens[ratio] = _compress_read_len(compress_len, ratio)
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
    """What attention layers hand down the stack instead of recomputing."""

    __slots__ = (
        "compress_kv", "index_k", "topk_idxs", "candidates", "compress_read",
    )

    def __init__(self) -> None:
        self.compress_kv: torch.Tensor | None = None
        self.index_k: torch.Tensor | None = None
        self.topk_idxs: torch.Tensor | None = None
        self.candidates: torch.Tensor | None = None
        #: The rows of `compress_kv` the last `is_kv_source` layer read out,
        #: for the layers behind it that would read exactly the same ones.
        #: See `Attention.reads_shared_compress`.
        self.compress_read: torch.Tensor | None = None


class Compressor(nn.Module):
    """Pools `compress_ratio` consecutive tokens into one KV latent."""

    def __init__(
        self, config: DeepseekV41Config, layer_id: int, max_batch_size: int = 1
    ) -> None:
        super().__init__()
        del max_batch_size
        self.compress_ratio = config.compress_ratios[layer_id]
        self.head_dim = config.head_dim
        self.norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.wkv = BlockQuantLinear(config.hidden_size, config.head_dim, storage="bf16")
        if self.compress_ratio > 1:
            self.wgate = BlockQuantLinear(
                config.hidden_size, config.head_dim, storage="bf16"
            )
        self.register_buffer("pool_kv", None, persistent=False)
        self.register_buffer("pool_score", None, persistent=False)

    def allocate_state(self, device, num_slots: int) -> None:
        """Allocate the per-slot pooling inputs.

        fp32, as the reference keeps its `kv_state` / `score_state`: the
        pooling inputs are the fp32 projections `forward` writes, and rounding
        them to bf16 before the softmax pooling moved the fp4-quantized latent
        ~1.5% off the reference (isolated at layers 2 and 8, 2026-10-22).
        """
        if self.compress_ratio > 1:
            self.pool_kv = torch.zeros(
                num_slots, self.head_dim, dtype=torch.float32, device=device
            )
            self.pool_score = torch.zeros(
                num_slots, self.head_dim, dtype=torch.float32, device=device
            )

    def forward(
        self, x: torch.Tensor, start_pos: torch.Tensor, paged: PagedSpan
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Pool `compress_ratio` consecutive tokens into one latent."""
        _, seqlen, _ = x.shape
        ratio, dtype = self.compress_ratio, x.dtype
        device = x.device
        if ratio == 1:
            positions = start_pos + torch.arange(seqlen, device=device)
            return self.norm(self.wkv(x)), positions

        # The reference promotes `wkv` / `wgate` to fp32 above ratio 1 and pools
        # in fp32 ("the softmax pooling above ratio 1 runs in fp32, so those
        # weights are promoted to fp32 to match", vendor/model.py Compressor).
        # The bf16 NKI GEMM rounded both projections to bf16 first, and the fp4
        # round trip after the pooling turns that 0.4% into flipped e2m1
        # roundings. Plain fp32 matmuls here: [T, 5120] x [5120, 512], small
        # enough for the tensor engine's fp32 path.
        xf = x.float()
        kv_proj = torch.matmul(xf, self.wkv.weight.float().t())
        score_proj = torch.matmul(xf, self.wgate.weight.float().t())
        paged.write_flat(self.pool_kv, kv_proj.reshape(seqlen, -1))
        paged.write_flat(self.pool_score, score_proj.reshape(seqlen, -1))

        count = paged.group_counts[ratio]
        if count <= 0:
            return None, None

        firsts = (start_pos // ratio + torch.arange(count, device=device)) * ratio
        members = firsts.unsqueeze(1) + torch.arange(ratio, device=device)
        slots = paged.slots(members.reshape(-1))
        kv = self.pool_kv.index_select(0, slots).reshape(count, ratio, self.head_dim)
        score = self.pool_score.index_select(0, slots).reshape(
            count, ratio, self.head_dim
        )
        pooled = (kv * torch.softmax(score, dim=1)).sum(dim=1).unsqueeze(0)
        return self.norm(pooled.to(dtype)), firsts


class Indexer(nn.Module):
    """Keeps the `index_topk = 512` best compressed positions per query."""

    def __init__(
        self,
        config: DeepseekV41Config,
        layer_id: int,
        tp_size: int,
        replicas: int = 1,
        attn_group=None,
    ) -> None:
        super().__init__()
        #: The group the index-score reduction runs over; see
        #: `Attention.attn_group`.
        self.attn_group = attn_group
        #: See `Attention.replicas`: the index score sums over index heads, so
        #: a replicated attention makes its all-reduce `replicas x` the sum.
        self.replicas = max(1, replicas)
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
        """The `index_topk` compressed positions each query keeps."""
        assert self.freqs_cis is not None
        bsz, seqlen, _ = x.shape
        ratio, rd = self.compress_ratio, self.rope_head_dim

        if self.owns_k:
            shared.index_k = self.k_cache
            if latent is not None:
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
        index_score = _nki_entry("index_score", "weighted_scores")(
            q.reshape(-1, self.index_head_dim),
            index_k,
            weights.reshape(seqlen, self.n_local_heads).t().contiguous().float(),
        ).reshape(1, seqlen, read_len)
        if self.world_size > 1 and tp_group is not None:
            index_score = tp_all_reduce(
                index_score, self.attn_group or tp_group
            )
            if self.replicas > 1:
                index_score = index_score / self.replicas

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
            index_score = index_score.masked_fill(
                ~shared.candidates, _bound(index_score, _NEG_BOUND)
            )

        topk = min(self.index_topk, read_len)
        idxs = nki_topk(index_score, topk)[1]
        return torch.where(idxs < compress_lens, idxs + offset, EMPTY_SLOT).int()


class Attention(nn.Module):
    """Latent attention over a sliding window plus, where configured,
    compressed KV — concatenated into one sparse-attention call.
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        layer_id: int,
        tp_size: int,
        max_batch_size: int = 1,
        replicas: int = 1,
        attn_group=None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        #: The group this layer's output reduction runs over, or None to use
        #: whatever group the forward is handed. See
        #: `DeepseekV41Model.attn_tp_group`.
        self.attn_group = attn_group
        #: How many ranks hold *this* rank's head shard. 1 when the attention
        #: splits across the whole world; `world_size // o_groups` when the
        #: world is wider than `o_groups` can split, in which case the
        self.replicas = max(1, replicas)
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

        #: Whether this layer reads the compressed KV that an earlier layer
        #: already gathered, instead of gathering the same rows again.
        #:
        #: Only `is_kv_source` layers own a `compress_kv_cache` (see
        #: `bind_kv_cache`), and `_compress_kv` hands the owner's buffer down
        #: the stack as `shared.compress_kv`. The rows it then reads out of that
        #: buffer are `min(arange(read_len), compress_len - 1) * ratio`, and
        #: every term is fixed by `(ratio, start_pos, seqlen)` — so **every
        #: layer behind one `is_kv_source` layer, at that layer's ratio, gathers
        #: byte-for-byte the same tensor**. With `kv_source_layer_ids
        #: [2, 8, 14, 20]` and `compress_ratios` 2 on layers 2-19 and 1 on
        #: 20-39, that is 38 indirect gathers a sequence a forward where 4 are
        #: needed: layers 3-7, 9-13, 15-19 and 21-39 repeat their source's.
        #:
        #: Each repeat is `read_len x head_dim` pulled through the indirect
        #: scatter/gather path — 9.7 MB a layer at the bench's 8448-token
        #: context, since `_COMPRESS_READ_CEILING` pins `read_len` to the
        #: engine's context.
        #:
        #: **It is not, however, a latency win, and this comment is deliberately
        #: not claiming one.** Removing 34 of the 38 gathers — 2.6 GB of
        #: indirect reads out of an 8-forward `in8192_bs1` cell — measured
        #: within +-1.3% on all four batch-1 bench cells, two in each direction
        #: (`.autohelix/notes/iter-9.md` §6 and §8). What it buys is device
        #: working set: 34 fewer `read_len x 576` tables live per sequence per
        #: forward, under a 24 GiB per-rank budget. The predecessor of this
        #: comment, in `__init__.py`, attributed the whole per-forward prefill
        #: cost to this read; that attribution is false, and iteration 9's §9b
        #: measures where the time actually is by halving the forward count.
        #:
        #: Decided here, from the layer topology, rather than by comparing
        #: tensors at runtime: the flag is a module constant, so the two arms in
        #: `_compress_kv` are chosen at trace time and neither adds a guard on
        #: anything a previous graph wrote. The equality of `ratio` is checked
        #: rather than assumed, so a config whose ratio changes inside a
        #: kv-source group gathers again instead of reusing rows at the wrong
        #: stride.
        self.reads_shared_compress = False
        if is_backbone and self.compress_ratio and not self.is_kv_source:
            sources = [s for s in config.kv_source_layer_ids if s < layer_id]
            self.reads_shared_compress = bool(sources) and (
                config.compress_ratios[max(sources)] == self.compress_ratio
            )

        self.register_buffer(
            "attn_sink", torch.empty(self.n_local_heads, dtype=torch.float32)
        )

        self.wq_a = BlockQuantLinear(config.hidden_size, config.q_lora_rank)
        self.q_norm = RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.wq_b = BlockQuantLinear(
            config.q_lora_rank, self.n_local_heads * config.head_dim
        )
        self.wkv = BlockQuantLinear(config.hidden_size, config.head_dim)
        self.kv_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
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
            Indexer(config, layer_id, tp_size, replicas=self.replicas,
                    attn_group=attn_group)
            if self.is_index_source
            else None
        )
        self.world_size = tp_size
        self.config = config

        self.window_kv_cache: torch.Tensor | None = None
        self.compress_kv_cache: torch.Tensor | None = None
        self.index_k_cache: torch.Tensor | None = None

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


    def _window_kv(
        self,
        x: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        start_pos: torch.Tensor,
        paged: PagedSpan,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """This layer's sliding-window K, and the slots every query may see."""
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
        """The shared compressed KV and the compressed slots each query sees."""
        bsz, seqlen, _ = x.shape
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        read_len = paged.read_lens[ratio]
        compress_len = (start_pos + seqlen) // ratio

        latent, group_positions = None, None
        if self.is_kv_source:
            latent, group_positions = self.compressor(x, start_pos, paged)
            shared.compress_kv = self.compress_kv_cache

        if self.is_index_source:
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
            idxs = shared.topk_idxs.to(torch.int32)

        if latent is not None:
            freqs = slice_freqs(self.freqs_cis, group_positions)
            rotated = torch.cat(
                [latent[..., :-rd], apply_rotary_emb(latent[..., -rd:], freqs)], -1
            )
            rotated = fp4_quant_dequant(rotated, 16, scale_e4m3=True)
            paged.put(
                self.compress_kv_cache,
                group_positions,
                rotated.reshape(rotated.size(1), -1),
            )

        if self.reads_shared_compress:
            # The same rows of the same buffer an earlier layer already read.
            # Handed down the stack like `shared.compress_kv` and
            # `shared.topk_idxs` beside it, and for the same reason: it is the
            # one mechanism in this model that is known to carry a computed
            # tensor between two separately compiled block graphs.
            #
            # Correct because nothing writes `shared.compress_kv` between the
            # gather and this read: the only writer is the `paged.put` above,
            # which runs exactly where `latent is not None`, i.e. on the
            # `is_kv_source` layer that also refreshes `shared.compress_read`
            # two lines down.
            compress_kv = shared.compress_read
        else:
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
            shared.compress_read = compress_kv
        return compress_kv, idxs

    #: Whether `forward_spans` below may serve this attention. False on
    #: `DSparkAttention`, whose `forward` takes the draft block's own KV and a
    #: separate main-stream tap and is reached from `DSparkBlock.forward`
    #: rather than from `Block.attn_half`.
    spans_supported = True

    def project_in(
        self, x: torch.Tensor, freqs_cis: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The row-wise half of the query path: the roped `q` and the q LoRA.

        Every op here is row-wise, which is what lets `forward_spans` run it
        once over the concatenated rows of several sequences.
        """
        rd = self.rope_head_dim
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        q = torch.cat([q[..., :-rd], apply_rotary_emb(q[..., -rd:], freqs_cis)], -1)
        return q, qr

    def latent_kv(
        self, x: torch.Tensor, freqs_cis: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """This layer's latent KV for `x`'s own rows — roped and round-tripped
        through fp8, but not yet written to any page. Row-wise.
        """
        rd = self.rope_head_dim
        kv = self.kv_norm(self.wkv(x))
        kv = torch.cat([kv[..., :-rd], apply_rotary_emb(kv[..., -rd:], freqs_cis)], -1)
        return act_quant_dequant(kv, 32)

    def window_slots(
        self,
        kv: torch.Tensor,
        start_pos: torch.Tensor,
        paged: PagedSpan,
        device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Page one sequence's latent KV and gather the window it may see.

        The per-sequence half: it addresses `paged`'s own block table, so
        unlike `project_in`/`latent_kv` it cannot be shared between sequences.
        """
        seqlen = kv.shape[1]
        win = self.window_size
        if self.window_kv_cache is None:
            raise RuntimeError(
                f"layer {self.layer_id} attention has no window KV cache; "
                "bind_kv_cache() must run before forward()."
            )
        paged.write(self.window_kv_cache, kv.reshape(seqlen, self.head_dim))

        span = win + seqlen - 1
        base = start_pos - (win - 1)
        positions = (base + torch.arange(span, device=device)).clamp_min(0)
        window_kv = paged.gather(
            self.window_kv_cache.view(-1, self.head_dim), positions
        ).reshape(1, span, self.head_dim)
        return window_kv, window_topk_idxs(win, seqlen, start_pos, device)

    def project_out(
        self,
        o: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        dtype: torch.dtype,
        tp_group,
    ) -> torch.Tensor:
        """The row-wise half of the output path: the inverse rope, the grouped
        O LoRA, and **one** reduction over the attention group.
        """
        rd = self.rope_head_dim
        bsz, seqlen = o.shape[0], o.shape[1]
        o = torch.cat(
            [o[..., :-rd], apply_rotary_emb(o[..., -rd:], freqs_cis, True)], -1
        )
        # The reference rounds `o` through fp8 (`act_quant(o, 32, ..., inplace=True)`)
        # before the grouped `wo_a` einsum; `grouped()` runs the GEMM raw, so do it here.
        # Measured 2026-10-22: without it layer 0's attention is 3.1% off the reference
        # and the drift compounds to ~30% by the last layer; with it the port's own
        # math replicated on CPU matched the device to 0.5%.
        o = act_quant_dequant(o, 32)
        o = o.reshape(bsz * seqlen, self.n_local_groups, -1)
        o = torch.cat(
            [
                self.wo_a.grouped(o[:, g], g, self.o_lora_rank)
                for g in range(self.n_local_groups)
            ],
            dim=-1,
        ).reshape(bsz, seqlen, -1)
        if self.world_size > 1 and tp_group is not None:
            # fp32 partials into the all-reduce: see `fp8_linear_f32`.
            out = nki_linear(o, self.wo_b.weight, self.wo_b.scale, "fp8", out_fp32=True)
            out = tp_all_reduce(out, self.attn_group or tp_group)
            if self.replicas > 1:
                out = out / self.replicas
            out = out.to(dtype)
        else:
            out = self.wo_b(o)
        return out

    def forward_spans(
        self,
        x: torch.Tensor,
        lengths: list[int],
        paged: list[PagedSpan],
        shared: list[SharedAttentionState],
        tp_group=None,
    ) -> torch.Tensor:
        """`forward`, but over several sequences' rows at once.

        `x` is `[1, sum(lengths), hidden]`, the concatenated normalized rows of
        the sequences `paged` describes, in the same order. The arithmetic is
        identical to calling `forward` once per sequence — every op hoisted out
        of the loop is row-wise, so a row's result does not depend on which
        other rows share the call — but the q/kv/o projections, the two ropes
        and the output all-reduce happen **once** instead of once per sequence.

        That is the whole of the multi-row decode penalty iteration 7 named:
        `attn_half` is one compiled graph, so its Python loop is unrolled into
        it, and at batch B the graph held B copies of this entire method at a
        row count that pads to the 128-row tensor-engine partition anyway. A
        decode step gives each sequence `1 + num_speculative_tokens` rows, so B
        sequences were paying B full tiles for work that fits in one.

        What stays per sequence is what addresses a different block table:
        `window_slots`, `_compress_kv` (the compressor, the indexer and the
        compressed gather) and `sparse_attn`.
        """
        device = x.device
        if len(lengths) == 1:
            freqs_cis = span_freqs(self.freqs_cis, paged[0].pos, lengths[0], device)
        else:
            parts = [
                span_freqs(self.freqs_cis, span.pos, length, device)
                for length, span in zip(lengths, paged)
            ]
            freqs_cis = (
                torch.cat([cos for cos, _ in parts], dim=0),
                torch.cat([sin for _, sin in parts], dim=0),
            )

        q, qr = self.project_in(x, freqs_cis)
        kv_rows = self.latent_kv(x, freqs_cis)

        if self._spans_uniform(lengths, paged):
            o = self._spans_attended(
                x, q, qr, kv_rows, lengths[0], paged, shared, tp_group, device
            )
            return self.project_out(o, freqs_cis, x.dtype, tp_group)

        attended: list[torch.Tensor] = []
        begin = 0
        for length, span, span_shared in zip(lengths, paged, shared):
            rows = slice(begin, begin + length)
            begin += length
            kv, topk_idxs = self.window_slots(
                kv_rows[:, rows], span.pos, span, device
            )
            if self.compress_ratio:
                compress_kv, compress_idxs = self._compress_kv(
                    x[:, rows], qr[:, rows], span.pos, kv.size(1),
                    span_shared, span, tp_group,
                )
                kv = torch.cat([kv, compress_kv], dim=1)
                topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)
            attended.append(
                sparse_attn(
                    q[:, rows], kv, self.attn_sink, topk_idxs, self.softmax_scale
                )
            )
        o = attended[0] if len(attended) == 1 else torch.cat(attended, dim=1)
        return self.project_out(o, freqs_cis, x.dtype, tp_group)

    def _spans_uniform(
        self, lengths: list[int], paged: list[PagedSpan]
    ) -> bool:
        """Whether every span has the same row count *and* the same table
        widths, which is what lets one `sparse_attn` serve them all.

        True in a pure decode step — every sequence contributes
        `1 + num_speculative_tokens` rows and `read_lens` is pinned to the
        engine context by `_COMPRESS_READ_CEILING`, so the window table is
        `window_size + L - 1` rows and the compressed one `read_len` rows for
        every sequence alike. False in a mixed prefill/decode step, where a
        chunk's row count is nothing like a decode row's, and that is the case
        the loop below still serves.
        """
        if len(lengths) < 2 or len(set(lengths)) != 1:
            return False
        ratio = self.compress_ratio
        if ratio and len({span.read_lens[ratio] for span in paged}) != 1:
            return False
        return len({span.block_size for span in paged}) == 1

    def _spans_attended(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        qr: torch.Tensor,
        kv_rows: torch.Tensor,
        length: int,
        paged: list[PagedSpan],
        shared: list[SharedAttentionState],
        tp_group,
        device,
    ) -> torch.Tensor:
        """`sparse_attn` over every span's rows in **one** call.

        The kernel this ends in (`kernels/sparse_attn.gathered_attn_idx`) takes
        a single `[n, head_dim]` table and a `[tokens, topk]` index map, and its
        body is `for t in nl.affine_range(tokens)` — a token's output is a
        function of its own query row, its own gathered slots and its own bias
        and nothing else. So B sequences can share one launch by concatenating
        their tables and shifting sequence `i`'s indices by `i * rows_per`, and
        the result is **bit-identical** to calling it once per sequence: the
        same q row meets the same slot values in the same order.

        That is where the multi-row decode penalty went after iteration 8
        hoisted the projections. At batch B the loop above issues, per layer,
        B window writes, B window gathers, B table concatenations and B kernel
        launches, each over `1 + k` rows against a 128-row partition. This
        issues one of each.

        What stays per sequence is `_compress_kv`, because the compressor and
        the indexer address a different block table per sequence — but on 34 of
        the 40 layers it is a no-op that hands down `shared.compress_read`
        (`Attention.reads_shared_compress`), so the loop that remains is a
        Python loop over cheap state rather than over device work.

        The empty-slot sentinel is **not** shifted: `EMPTY_SLOT` is -1 and
        `sparse_attn` recognises it by equality to build the `-1e30` bias, so an
        offset -1 would silently become a real slot of the previous sequence.
        """
        bsz = len(paged)
        win = self.window_size
        head_dim = self.head_dim
        block_size = paged[0].block_size
        span_rows = win + length - 1
        if self.window_kv_cache is None:
            raise RuntimeError(
                f"layer {self.layer_id} attention has no window KV cache; "
                "bind_kv_cache() must run before forward()."
            )

        # One write for every sequence's own slots. `slot_mapping` is this
        # step's own rows, so concatenating them in span order lines up with
        # `kv_rows`, which `forward_spans` built over the same concatenation.
        slots = torch.cat([span.slot_mapping for span in paged])
        self.window_kv_cache.index_put_(
            (
                slots // block_size,
                torch.zeros_like(slots),
                slots % block_size,
            ),
            kv_rows.reshape(bsz * length, head_dim),
        )

        # One gather for every sequence's window. `PagedSpan.slots` per span is
        # `block_table[pos // block_size] * block_size + pos % block_size`;
        # stacking the block tables turns that into a `gather` over the same
        # integer arithmetic, so the slot list is the same list.
        starts = torch.cat([span.pos for span in paged]).reshape(bsz, 1)
        tables = torch.stack([span.block_table for span in paged])
        window_pos = (
            starts - (win - 1) + torch.arange(span_rows, device=device)
        ).clamp_min(0)
        window_slots = (
            tables.gather(1, window_pos // block_size) * block_size
            + window_pos % block_size
        )
        window_kv = self.window_kv_cache.view(-1, head_dim).index_select(
            0, window_slots.reshape(-1)
        ).reshape(bsz, span_rows, head_dim)

        # One window index map. `window_topk_idxs`' arithmetic with the per-span
        # `start_pos` broadcast over a leading sequence axis.
        rows = torch.arange(length, device=device).unsqueeze(1) + torch.arange(
            win, device=device
        )
        window_idxs = torch.where(
            (starts.reshape(bsz, 1, 1) - (win - 1)) + rows < 0,
            EMPTY_SLOT,
            rows,
        ).int()

        parts: list[torch.Tensor] = []
        idxs: list[torch.Tensor] = []
        begin = 0
        for b, (span, span_shared) in enumerate(zip(paged, shared)):
            sel = slice(begin, begin + length)
            begin += length
            parts.append(window_kv[b : b + 1])
            if not self.compress_ratio:
                idxs.append(window_idxs[b : b + 1])
                continue
            compress_kv, compress_idxs = self._compress_kv(
                x[:, sel], qr[:, sel], span.pos, span_rows,
                span_shared, span, tp_group,
            )
            parts.append(compress_kv)
            idxs.append(
                torch.cat([window_idxs[b : b + 1], compress_idxs], dim=-1)
            )
        kv = torch.cat(parts, dim=1)
        rows_per = kv.shape[1] // bsz
        idx = torch.cat(idxs, dim=0)
        shift = (
            torch.arange(bsz, device=device, dtype=idx.dtype) * rows_per
        ).reshape(bsz, 1, 1)
        idx = torch.where(idx == EMPTY_SLOT, idx, idx + shift)
        return sparse_attn(
            q,
            kv,
            self.attn_sink,
            idx.reshape(1, bsz * length, -1),
            self.softmax_scale,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: torch.Tensor,
        shared: SharedAttentionState,
        paged: PagedSpan,
        tp_group=None,
    ) -> torch.Tensor:
        """`x` is the normalized sublayer input `[b, s, hidden]`.

        Kept as its own body rather than delegating to `forward_spans` with one
        span, deliberately: this is the path every passing verifier run and
        every passing batch-1 bench cell takes, and a graph is keyed by the ops
        dynamo traces. Routing it through the new code would re-key ~120 graphs
        at every context for no arithmetic change, which costs the gate a full
        compile in each cell's warm-up and risks the eleven constraints that
        hold today. The duplication is the price of that, and the two have to
        be kept in step; `forward_spans`' docstring says which ops are shared.
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
        o = torch.cat(
            [o[..., :-rd], apply_rotary_emb(o[..., -rd:], freqs_cis, True)], -1
        )

        # The reference rounds `o` through fp8 (`act_quant(o, 32, ..., inplace=True)`)
        # before the grouped `wo_a` einsum; `grouped()` runs the GEMM raw, so do it here.
        # Measured 2026-10-22: without it layer 0's attention is 3.1% off the reference
        # and the drift compounds to ~30% by the last layer; with it the port's own
        # math replicated on CPU matched the device to 0.5%.
        o = act_quant_dequant(o, 32)
        o = o.reshape(bsz * seqlen, self.n_local_groups, -1)
        o = torch.cat(
            [
                self.wo_a.grouped(o[:, g], g, self.o_lora_rank)
                for g in range(self.n_local_groups)
            ],
            dim=-1,
        ).reshape(bsz, seqlen, -1)
        if self.world_size > 1 and tp_group is not None:
            # fp32 partials into the all-reduce: see `fp8_linear_f32`.
            out = nki_linear(o, self.wo_b.weight, self.wo_b.scale, "fp8", out_fp32=True)
            out = tp_all_reduce(out, self.attn_group or tp_group)
            if self.replicas > 1:
                out = out / self.replicas
            out = out.to(x.dtype)
        else:
            out = self.wo_b(o)
        return out


def _compile_fn(fn):
    """`fn` as a device graph. One call, so every compiled entry point in this
    file is built the same way as the whole-model compile the runner does.

        torch._dynamo.exc.Unsupported: Attempted to call function marked as
        skipped ... Dynamo does not know how to trace the builtin
        `time.perf_counter`
    """
    _raise_dynamo_limits()
    return torch.compile(
        fn,
        backend=_compile_backend(),
        fullgraph=True,
        options=_COMPILE_OPTIONS,
    )


@torch._dynamo.disable
def _graph_break() -> None:
    """A dynamo graph boundary and nothing else — no tensors, no side effects.

        ERROR [job.WalrusDriver.0]: Backend exited with code -11 and stderr:
        [F139] neuronx-cc terminated abnormally

        RuntimeError: Skip inlining `torch.compiler.disable()`d function
        ... <function MoE.forward> ... Hint: Remove the `torch.compiler.disable`
        call
    """
    return None


@functools.lru_cache(maxsize=None)
def _compiled(fn):
    """`fn` as a device graph, cached per function rather than per shape."""
    return _compile_fn(fn)


#: `(limit, ones)` as (1, 1) fp32 device tensors, one pair per (limit, device).
#: `kernels/swiglu` takes both as tensors, and both are loop-invariant — built
#: once here rather than once a layer, so the 40 MoE calls of a forward do not
#: pay 80 tiny host-to-device copies for two constants.
_SWIGLU_CONSTS: dict = {}


def swiglu_consts(limit: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    """The clamp limit and the shared expert's unit routing weight, on
    `device`. Call from eager code: this allocates, so it must not be traced.
    """
    key = (float(limit), str(device))
    hit = _SWIGLU_CONSTS.get(key)
    if hit is None:
        # Built on the host and copied, which is the one way of putting a
        # constant on this device that the rest of this file already relies on
        # (`_route_plan` stages its row lists the same way).
        hit = (
            torch.full((1, 1), float(limit), dtype=torch.float32).to(device),
            torch.ones((1, 1), dtype=torch.float32).to(device),
        )
        _SWIGLU_CONSTS[key] = hit
    return hit


def _swiglu_expert(x, weights, w1, s1, w2, s2, w3, s3, storage, limit):
    """One expert's SwiGLU FFN; all three GEMMs through `kernels/blockquant`
    and the clamped activation through `kernels/swiglu`.

      * the needle case answered correctly and stopped emitting EOS (19 tokens
        `stop` -> 32 tokens `length`), and
      * speculation on and off stopped producing identical tokens — two of the
        three short cases still matched exactly, `smoke-test` diverged after 80
        characters.

    `weights` is a `(rows, 1)` fp32 tensor for a routed expert and the `(1, 1)`
    unit tensor for the shared expert, and `limit` a `(1, 1)` fp32 tensor; both
    come from `swiglu_consts`, because a NKI kernel takes tensors and this
    function is traced.
    """
    dtype = x.dtype
    gate = nki_linear(x, w1, s1, storage)
    up = nki_linear(x, w3, s3, storage)
    y = _nki_entry("swiglu", "swiglu")(gate, up, weights, limit)
    return nki_linear(y.to(dtype), w2, s2, storage)


def _routed_expert_step(
    carry, flat, rows, route_weights, w1, s1, w2, s2, w3, s3, limit
):
    """Gather one expert's rows, run it, scatter-add into the fp32 carry."""
    y = _swiglu_expert(
        flat.index_select(0, rows), route_weights,
        w1, s1, w2, s2, w3, s3, "fp4", limit,
    )
    return carry.index_add(0, rows, y.float())


def _routed_expert_step2(
    carry, flat, limit,
    rows_a, rw_a, w1a, s1a, w2a, s2a, w3a, s3a,
    rows_b, rw_b, w1b, s1b, w2b, s2b, w3b, s3b,
):
    """Two experts in **one** graph execution. Same arithmetic, same order."""
    carry = _routed_expert_step(
        carry, flat, rows_a, rw_a, w1a, s1a, w2a, s2a, w3a, s3a, limit
    )
    return _routed_expert_step(
        carry, flat, rows_b, rw_b, w1b, s1b, w2b, s2b, w3b, s3b, limit
    )


def _routed_expert_step3(
    carry, flat, limit,
    rows_a, rw_a, w1a, s1a, w2a, s2a, w3a, s3a,
    rows_b, rw_b, w1b, s1b, w2b, s2b, w3b, s3b,
    rows_c, rw_c, w1c, s1c, w2c, s2c, w3c, s3c,
):
    """Three experts in one graph execution."""
    carry = _routed_expert_step2(
        carry, flat, limit,
        rows_a, rw_a, w1a, s1a, w2a, s2a, w3a, s3a,
        rows_b, rw_b, w1b, s1b, w2b, s2b, w3b, s3b,
    )
    return _routed_expert_step(
        carry, flat, rows_c, rw_c, w1c, s1c, w2c, s2c, w3c, s3c, limit
    )


#: Routed experts per graph execution. **1 is today's path, exactly.**
#:
#: Why this is the lever it is. Iteration 4 §3 priced one graph execution on
#: this deployment at ~1.12 ms, by measuring a configuration that adds 720 of
#: them to a prefill and dividing. A 503-token prefill runs ~258
#: `_routed_expert_step` executions (6 owned experts x 43 blocks), so the
#: routed-expert path is ~289 ms of a 947 ms prefill, and the 128-row GEMM
#: inside one of them has a roofline of ~154 us. Most of that 1.12 ms is
#: therefore *not* arithmetic, and the only way to stop paying it is to issue
#: fewer executions.
#:
#: `_EXPERT_ROW_FLOOR` is **not** the lever, and its comment says why: the
#: blockquant GEMM tiles the token axis at the 128-wide partition, so a bucket
#: below 128 is the same instruction count. Fusing experts is, because it
#: divides the count.
#:
#: Experts are fused only within one row bucket, so the fused graph has one
#: shape per (bucket, count) rather than one per combination. At 503 tokens
#: every expert lands in the floor bucket of 128 — ~8 of its rows are real —
#: so in practice all of a block's experts fuse together.
_EXPERT_FUSE = int(os.environ.get("DEEPSEEK_V41_EXPERT_FUSE", "1") or 1)

#: `k -> the compiled entry that runs k experts in one execution`.
_FUSED_EXPERT_STEPS = {2: _routed_expert_step2, 3: _routed_expert_step3}

#: So the "fusion is on" warning is logged once rather than 43 times a forward.
_FUSE_LOGGED = [False]


def _shared_expert_step(carry, flat, ones, w1, s1, w2, s2, w3, s3, limit):
    """The shared expert every token goes through, added to the routed sum."""
    contribution = _swiglu_expert(
        flat, ones, w1, s1, w2, s2, w3, s3, "fp8", limit
    )
    return carry + contribution.float()


def _moe_carry(flat):
    """The fp32 accumulator the routed experts scatter into."""
    return torch.zeros_like(flat, dtype=torch.float32)


def _dspark_tap(stream):
    """The DSpark head's view of a target layer: the attention input averaged
    over the hyper-connection copies.
    """
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
    """`HyperConnections.post`, written into `out` rather than returned."""
    out.copy_(HyperConnections.post(ffn_out, residual, ffn_post, ffn_comb).squeeze(0))
    return out


def _one_hot_pre_mix(stream, hc):
    """The draft stream's initial hyper-connection coefficients: copy 0 only."""
    pre_mix = stream.new_zeros(
        stream.shape[0], stream.shape[1], hc, dtype=torch.float32
    )
    pre_mix[:, :, 0] = 1.0
    return pre_mix


class _RejectionView:
    """The three fields `vllm_neuron.nn.rejection_sampler` reads off a
    `SpecDecodeMetadata`, carried without the dataclass.

        torch._dynamo.exc.Unsupported: Attempted to call function marked as
        skipped ... qualname: __build_class__
    """

    __slots__ = ("draft_token_ids", "cu_num_draft_tokens", "max_spec_len")


def _rejection_sample(sampled, draft_token_ids, cu_num_draft_tokens, max_spec_len):
    """Verify this step's drafts against the target's own tokens, on the device.

    `max_spec_len` arrives as an **int**, not as the per-row tuple it used to be
    derived from. The value is identical -- it was always `max(num_draft_tokens)`
    and nothing else in this body read the tuple -- but the tuple was a
    *compile-time constant* to Dynamo, so a graph was keyed on the exact per-row
    draft pattern. At batch 32 that is one graph per pattern, and the patterns
    change whenever a sequence finishes or starts: iteration 10 watched a
    batch-32 cell go quiet for 8 minutes with no completed step the moment
    `Running` fell 32 -> 29, which is this axis recompiling
    (`.autohelix/notes/iter-10.md` §0c). An int keys at most `dspark_block_size
    + 1` = 6 graphs, and a batch losing rows no longer changes it at all.

    Not the constant `dspark_block_size`, which would key exactly one: the
    sampler's `all_accepted = (matches & valid).all(dim=1)` is False for any
    column past a row's own count, so widening the window past the true maximum
    would silently suppress the bonus token of every row that drafted fewer --
    a correctness change disguised as a compile win. The real maximum is the
    widest window every row can fill.
    """
    from vllm_neuron.nn.rejection_sampler import rejection_sampler

    view = _RejectionView()
    view.draft_token_ids = draft_token_ids
    view.cu_num_draft_tokens = cu_num_draft_tokens
    view.max_spec_len = max_spec_len
    return rejection_sampler(view, sampled.reshape(-1))


def _accepted_tail(accepted):
    """`(last accepted token, how many were accepted)` per sequence."""
    valid = (accepted >= 0).sum(dim=-1)
    seeds = accepted.gather(1, (valid - 1).clamp_min(0).unsqueeze(1)).reshape(-1)
    return seeds, valid.to(torch.int64)


_GREEDY_TEMPERATURE_EPS = 1e-5

#: The candidate window for top-k / top-p filtering on the random path (the plugin
#: sampler's `max_top_k` is 256; a request's top-k above this is clamped to it, and a
#: top-p set that needs more candidates keeps the whole vocabulary).
_SAMPLE_TOPK_WINDOW = 256


def _hash32(x: torch.Tensor) -> torch.Tensor:
    """A 32-bit integer mix (Wellons' `lowbias32`) on int32 tensors: a bijection, so
    distinct inputs give distinct outputs. Right shifts are masked to act as logical
    shifts on the signed type; the multiplies wrap modulo 2**32 on both the host and
    the device, so every rank computes the same bits."""
    x = x ^ ((x >> 16) & 0xFFFF)
    x = x * 2146121005                      # 0x7feb352d
    x = x ^ ((x >> 15) & 0x1FFFF)
    x = x * -2073254261                     # 0x846ca68b as int32
    x = x ^ ((x >> 16) & 0xFFFF)
    return x


def _gumbel_noise(seeds: torch.Tensor, width: int) -> torch.Tensor:
    """`[rows, width]` standard Gumbel noise, a deterministic function of the per-row
    `seeds` (int32): u = (23 bits of `hash(hash(seed) ^ column)` + 0.5) / 2**23, strictly
    inside (0, 1) in fp32 (24 bits plus the half step rounded the top value to 1.0, and
    -log(-log(1)) = inf turned a masked -inf logit into NaN, which argmax picks: 6 of
    20000 top-k=5 draws landed outside the top 5 on the host), g = -log(-log(u)).
    Every rank holds the same seeds (drawn on the host) and so the same noise, which
    is what keeps the ranks' sampled tokens identical without a broadcast."""
    col = torch.arange(width, dtype=torch.int32, device=seeds.device).unsqueeze(0)
    h = _hash32(_hash32(seeds.to(torch.int32)).unsqueeze(-1) ^ col)
    u = (((h >> 9) & 0x7FFFFF).to(torch.float32) + 0.5) * (1.0 / 8388608.0)
    return -torch.log(-torch.log(u))


def _sample_full_vocab(full: torch.Tensor, sampling_params: torch.Tensor, seeds: torch.Tensor,
                       filter_topk: bool) -> torch.Tensor:
    """One token a row from the gathered, full-vocabulary logits `full` `[rows, V]`:
    the argmax for greedy rows (temperature < eps), otherwise an exact draw from
    softmax(logits / temperature) restricted by the row's top-k / top-p, by the
    Gumbel-max trick (argmax of logits / T + Gumbel noise). `sampling_params` is
    `[rows, 3]` = (top_k, top_p, temperature), one row per logits row."""
    full = full.float()
    temperature = sampling_params[:, 2]
    greedy = temperature < _GREEDY_TEMPERATURE_EPS
    greedy_ids = full.argmax(dim=-1)
    x = full / torch.where(greedy, torch.ones_like(temperature), temperature).unsqueeze(-1)
    if filter_topk:
        neg = torch.full_like(x[:, :1], float("-inf"))
        vals = torch.topk(x, _SAMPLE_TOPK_WINDOW, dim=-1).values          # sorted, descending
        k = sampling_params[:, 0].to(torch.int64)
        kk = k.clamp(min=1, max=_SAMPLE_TOPK_WINDOW)
        thr_k = torch.where((k <= 0).unsqueeze(-1), neg, vals.gather(-1, (kk - 1).unsqueeze(-1)))
        top_p = sampling_params[:, 1]
        cum = torch.cumsum(torch.exp(vals - torch.logsumexp(x, dim=-1, keepdim=True)), dim=-1)
        need = (cum < top_p.unsqueeze(-1)).sum(dim=-1) + 1
        thr_p = vals.gather(-1, (need.clamp(max=_SAMPLE_TOPK_WINDOW) - 1).unsqueeze(-1))
        thr_p = torch.where(((need > _SAMPLE_TOPK_WINDOW) | (top_p >= 1.0)).unsqueeze(-1), neg, thr_p)
        x = torch.where(x >= torch.maximum(thr_k, thr_p), x, torch.full_like(x, float("-inf")))
    drawn = (x + _gumbel_noise(seeds, x.shape[-1])).argmax(dim=-1)
    return torch.where(greedy, greedy_ids, drawn).to(torch.int32)


def _greedy_override(sampled: torch.Tensor, greedy_ids: torch.Tensor, sampling_params: torch.Tensor) -> torch.Tensor:
    """`sampled` with its greedy rows replaced by the gathered-logits argmax.

    `sampled` holds one token per scheduled row (`1 + draft` rows a request
    under speculation); the greedy flag (temperature column of
    `sampling_params`, one row per request) is repeated over a request's rows.
    """
    greedy = sampling_params[:, 2] < _GREEDY_TEMPERATURE_EPS
    flat = sampled.reshape(-1)
    g = greedy_ids.reshape(-1).to(flat.dtype)
    n = min(int(flat.shape[0]), int(g.shape[0]))
    reqs = int(greedy.shape[0])
    reps = n // reqs if reqs and n % reqs == 0 else 1
    mask = greedy.reshape(-1).repeat_interleave(reps)[:n]
    head = torch.where(mask, g[:n], flat[:n])
    out = torch.cat([head, flat[n:]]) if n < flat.shape[0] else head
    return out.reshape(sampled.shape)


def _greedy_rows(logits):
    """`argmax` over the vocabulary, as a compiled step."""
    return logits.argmax(dim=-1)


def _tap_rows(valid, lengths, rows_per_seq: int):
    """Which row of this step each sequence's draft block hangs off.

        no speculation   lengths - 1 + (1 - 1)     = the span's last row
        verify step      lengths - lengths + v - 1 = v - 1, inside the block
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
    """MoE routing: `sqrtsoftplus` scores, `noaux_tc` selection."""
    # fp32, as the reference's `linear(x.float(), self.weight.float())`: the
    # bf16 NKI GEMM rounds the scores to bf16, which moves every routing
    # weight by ~0.4% in every layer. [T, 5120] x [5120, 384] is small.
    scores = torch.matmul(x.float(), weight.float().t())
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
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights * route_scale, indices


class Gate(nn.Module):
    """MoE routing. `sqrtsoftplus` scores, `noaux_tc` selection."""

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
        """The router, as its own compiled graph."""
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
        limit, ones = swiglu_consts(self.swiglu_limit, x.device)
        return _swiglu_expert(
            x,
            ones if weights is None else weights,
            self.w1.weight, self.w1.scale,
            self.w2.weight, self.w2.scale,
            self.w3.weight, self.w3.scale,
            self.w1.storage,
            limit,
        )


def _moe_config(config: DeepseekV41Config, layer_id: int) -> tuple[int, int]:
    """Routed/activated expert counts: 384-of-6 on the backbone, 128-of-3 on a
    DSpark stage (`ModelArgs.get_moe_config`).
    """
    if layer_id < config.num_hidden_layers:
        return config.n_routed_experts, config.num_experts_per_tok
    return config.dspark_n_routed_experts, config.dspark_num_experts_per_tok


class MoE(nn.Module):
    """Top-k routed experts plus one shared expert every token goes through."""

    def __init__(
        self, config: DeepseekV41Config, layer_id: int, tp_size: int, tp_rank: int
    ) -> None:
        super().__init__()
        n_routed, n_activated = _moe_config(config, layer_id)
        self.layer_id = layer_id
        #: True for the three `mtp.*` stages, whose layer ids continue past the
        #: backbone. `_PP_EMULATE` divides the backbone between stages and
        #: leaves the MTP stages world-wide, so this is what tells them apart.
        self.is_mtp = layer_id >= config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.n_routed_experts = n_routed
        self.n_activated_experts = n_activated
        self.n_local_experts = n_routed // tp_size
        self.experts_start_idx = tp_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts

        self.gate = Gate(config, layer_id)
        self.shared_experts = Expert(config, storage="fp8")
        self.expert_slot = Expert(config, storage="fp4", persistent=False)

        self.cache_slots = nn.ModuleList()
        #: `expert_id -> slot index` for the experts currently resident.
        self._resident: dict[int, int] = {}
        #: Slot indices with nothing in them yet.
        self._free: list[int] = []
        #: True once `load_resident_experts` has put *every* expert this rank
        #: owns in a slot. Then `_expert` can only hit, and a miss is a bug
        #: rather than a fetch — see `_expert`, which raises instead of reading.
        self.resident = False
        #: `expert_id -> times this rank has fetched it`, the admission and
        #: eviction key. Halved at `_EXPERT_USE_CEILING` so early popularity
        #: does not pin a slot for the rest of the run.
        self._uses: dict[int, int] = {}

    def allocate_cache_slots(self, config, device, count: int) -> int:
        """Give this layer `count` HBM slots to hold routed experts.

        A full pool is allocated as **one stacked buffer per tensor**
        (`self.stacked["w1.weight"]` is `[n_local_experts, inter, dim // 2]` and
        so on) with each slot's linears as views into it: the layer megakernel
        (`mk_model.run_layer`) reads the stack, and `_stage`'s `copy_` into a
        view fills it in place.
        """
        count = max(0, min(count, self.n_local_experts))
        slots = []
        self.stacked = None
        if count and count == self.n_local_experts:
            proto = Expert(config, storage="fp4", persistent=False)
            self.stacked = {}
            for name in ("w1", "w2", "w3"):
                for bname, shape, buffer_dtype in getattr(proto, name)._specs:
                    self.stacked[f"{name}.{bname}"] = torch.zeros(
                        (count, *shape), dtype=buffer_dtype, device=device
                    )
            for i in range(count):
                slot = Expert(config, storage="fp4", persistent=False)
                for name in ("w1", "w2", "w3"):
                    linear = getattr(slot, name)
                    for bname, _, _ in linear._specs:
                        setattr(linear, bname, self.stacked[f"{name}.{bname}"][i])
                slots.append(slot)
        else:
            for _ in range(count):
                slot = Expert(config, storage="fp4", persistent=False)
                slot.allocate(device)
                slots.append(slot)
        self.cache_slots = nn.ModuleList(slots)
        self._resident = {}
        self._free = list(range(count))
        return count

    @torch._dynamo.disable
    def load_resident_experts(self, weight_store) -> int:
        """Put every expert this rank owns in its own slot, once, at load time."""
        if len(self.cache_slots) < self.n_local_experts:
            return 0
        for slot, expert_id in enumerate(
            range(self.experts_start_idx, self.experts_end_idx)
        ):
            self._stage(weight_store.expert(self.layer_id, expert_id),
                        self.cache_slots[slot])
            self._resident[expert_id] = slot
        self._free = []
        self.resident = True
        return self.n_local_experts

    @torch._dynamo.disable
    def _expert(self, weight_store, expert_id: int) -> Expert:
        """The device buffers holding `expert_id`, fetching it if they must."""
        if self.resident:
            slot = self._resident.get(expert_id)
            if slot is None:
                raise RuntimeError(
                    f"layer {self.layer_id} routed to expert {expert_id}, which "
                    f"rank {self.experts_start_idx // max(1, self.n_local_experts)} "
                    f"does not own (it owns "
                    f"[{self.experts_start_idx}, {self.experts_end_idx})). The "
                    "expert pool is resident, so this is a routing/partition "
                    "disagreement rather than a cache miss."
                )
            _EXPERT_CACHE_STATS[0] += 1
            return self.cache_slots[slot]

        uses = self._uses.get(expert_id, 0) + 1
        if uses >= _EXPERT_USE_CEILING:
            self._uses = {e: n // 2 for e, n in self._uses.items()}
            uses = self._uses.get(expert_id, 0) + 1
        self._uses[expert_id] = uses
        slot = self._resident.get(expert_id)
        if slot is not None:
            _EXPERT_CACHE_STATS[0] += 1
            return self.cache_slots[slot]

        _EXPERT_CACHE_STATS[1] += 1
        if _EXPERT_TIMING:
            mark = time.perf_counter()
            blob = weight_store.expert(self.layer_id, expert_id)
            read = time.perf_counter()
            slot = self._stage(blob, self._target(expert_id, uses))
            _EXPERT_TIME[2] += read - mark
            _EXPERT_TIME[3] += time.perf_counter() - read
            return slot
        blob = weight_store.expert(self.layer_id, expert_id)
        return self._stage(blob, self._target(expert_id, uses))

    def _target(self, expert_id: int, uses: int) -> Expert:
        """Where to stage a missing expert: a cache slot, or the shared one."""
        if not len(self.cache_slots):
            return self.expert_slot
        if self._free:
            slot = self._free.pop()
        else:
            victim = min(self._resident, key=self._uses.__getitem__)
            if self._uses[victim] >= uses:
                return self.expert_slot
            slot = self._resident.pop(victim)
        self._resident[expert_id] = slot
        _EXPERT_CACHE_STATS[2] += 1
        return self.cache_slots[slot]

    @torch._dynamo.disable
    def _stage(
        self, blob: dict[str, torch.Tensor], slot: Expert | None = None
    ) -> Expert:
        """Copy one expert's stored weights into `slot`, or the staging slot.

            ValueError: Neuron Compilation called with tensors not on neuron.
            Tensor with shape torch.Size([5898240]) is on device cpu
        """
        slot = self.expert_slot if slot is None else slot
        for name in ("w1", "w2", "w3"):
            linear = getattr(slot, name)
            linear.weight.copy_(blob[f"{name}.weight"].to(linear.weight.device))
            linear.scale.copy_(blob[f"{name}.scale"].to(linear.scale.device))
        return slot

    @torch._dynamo.disable
    def _route_plan(
        self,
        indices: torch.Tensor,
        weights: torch.Tensor,
        live_rows: int | None,
        weight_store=None,
    ) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
        """Which of this rank's experts to run, over which rows, with what
        weight.
        """
        rows = int(indices.shape[0])
        if rows >= _GC_MIN_ROWS and self.layer_id % _GC_STRIDE == 0:
            _collect_layer_garbage()
        _hbm_mark(self.layer_id, rows)
        if _HBM_STATE[0] == 0 and self.layer_id in _CENSUS_LAYERS:
            _census_device_tensors(self.layer_id)

        override = _MOE_LIVE_ROWS[0]
        if override is not None:
            live_rows = override
        ids = indices if live_rows is None else indices[:live_rows]
        wts = weights if live_rows is None else weights[:live_rows]
        ids_host = ids.cpu()
        wts_host = wts.cpu().float()
        device = indices.device

        candidates = [
            expert_id
            for expert_id in sorted(set(ids_host.flatten().tolist()))
            if self.experts_start_idx <= expert_id < self.experts_end_idx
        ]
        if weight_store is not None and not self.resident:
            weight_store.prefetch_experts(
                self.layer_id, candidates, self._resident
            )

        plan = []
        for expert_id in candidates:
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

    @staticmethod
    def _expert_chunks(plan: list) -> list[list]:
        """`plan` grouped into the chunks that share one graph execution.

        At `_EXPERT_FUSE == 1` this is `[[e] for e in plan]` — one execution an
        expert, which is what the port has always done — and the caller takes
        the single-expert branch, so the compiled entry point and its graph are
        unchanged.

        Above 1, entries are grouped by **row bucket** first and only then
        chunked, so a fused graph is keyed by `(bucket, chunk length)` instead
        of by every combination of bucket sizes that happened to co-occur. The
        plan is in ascending expert-id order within a bucket, and the fused
        step applies its experts in argument order, so the only reordering is
        between buckets — and the carry is fp32, where a reordered sum of
        disjoint row sets is the same sum.
        """
        if _EXPERT_FUSE <= 1 or len(plan) < 2:
            return [[entry] for entry in plan]
        width = min(_EXPERT_FUSE, max(_FUSED_EXPERT_STEPS))
        if _LOG_RANK[0] and not _FUSE_LOGGED[0]:
            _FUSE_LOGGED[0] = True
            logger.warning(
                "deepseek_v41: DEEPSEEK_V41_EXPERT_FUSE=%d — routed experts "
                "run %d to a graph execution (capped at %d by the fused "
                "entries that exist). Logged once, because otherwise there is "
                "no way to tell from a server log whether fusion was active: "
                "iteration 4 had to infer it from a compile that only a new "
                "graph could explain.",
                _EXPERT_FUSE, width, max(_FUSED_EXPERT_STEPS),
            )
        buckets: dict[int, list] = {}
        for entry in plan:
            buckets.setdefault(int(entry[1].shape[0]), []).append(entry)
        chunks: list[list] = []
        for bucket in sorted(buckets):
            entries = buckets[bucket]
            for start in range(0, len(entries), width):
                chunks.append(entries[start : start + width])
        return chunks

    @torch._dynamo.disable
    def forward(
        self,
        x: torch.Tensor,
        image_mask: torch.Tensor | None,
        weight_store,
        tp_group=None,
        live_rows: int | None = None,
    ) -> torch.Tensor:
        """Route to `num_experts_per_tok` of the pool, plus the shared expert."""
        if weight_store is None and not self.resident:
            raise RuntimeError(
                f"layer {self.layer_id} MoE has no weight store and its expert "
                "pool is not resident, so the routed experts are streamed and "
                "load_weights() must have attached one. Running without it "
                "would mean running without the experts."
            )
        shape = x.shape
        flat = x.reshape(-1, self.hidden_size)
        weights, indices = self.gate(flat, image_mask)
        plan = self._route_plan(indices, weights, live_rows, weight_store)
        if _PP_EMULATE > 1 and not self.is_mtp:
            # A rank of a `64 // p`-wide tensor-parallel group owns
            # `384 / (64/p) = 6p` expert slots in this block where a rank of
            # the flat world owns 6, so its routed-expert loop is p times as
            # long. Repeating the plan reproduces that: the same launch count,
            # the same FLOPs and the same HBM read volume, against the same
            # resident weights. The *sum* is then p times too large as well as
            # being over the wrong shard set — which is why `_PP_EMULATE` is a
            # latency probe and says so.
            plan = list(plan) * _PP_EMULATE

        y = _compiled(_moe_carry)(flat)
        limit, ones = swiglu_consts(self.expert_slot.swiglu_limit, flat.device)
        for chunk in self._expert_chunks(plan):
            mark = time.perf_counter() if _EXPERT_TIMING else 0.0
            slots = [self._expert(weight_store, eid) for eid, _, _ in chunk]
            staged = time.perf_counter() if _EXPERT_TIMING else 0.0
            if len(chunk) == 1:
                (_, rows, route_weights), slot = chunk[0], slots[0]
                y = _compiled(_routed_expert_step)(
                    y, flat, rows, route_weights,
                    slot.w1.weight, slot.w1.scale,
                    slot.w2.weight, slot.w2.scale,
                    slot.w3.weight, slot.w3.scale,
                    limit,
                )
            else:
                args: list = []
                for (_, rows, route_weights), slot in zip(chunk, slots):
                    args += [
                        rows, route_weights,
                        slot.w1.weight, slot.w1.scale,
                        slot.w2.weight, slot.w2.scale,
                        slot.w3.weight, slot.w3.scale,
                    ]
                y = _compiled(_FUSED_EXPERT_STEPS[len(chunk)])(
                    y, flat, limit, *args
                )
            if _EXPERT_TIMING:
                _EXPERT_TIME[0] += staged - mark
                _EXPERT_TIME[1] += time.perf_counter() - staged
        if weight_store is not None and not self.resident:
            weight_store.end_expert_loop()

        if tp_group is not None and self.n_local_experts != self.n_routed_experts:
            y = tp_all_reduce(y, tp_group)

        shared = self.shared_experts
        shared_limit, _ = swiglu_consts(shared.swiglu_limit, flat.device)
        y = _compiled(_shared_expert_step)(
            y, flat, ones,
            shared.w1.weight, shared.w1.scale,
            shared.w2.weight, shared.w2.scale,
            shared.w3.weight, shared.w3.scale,
            shared_limit,
        )
        return _compiled(_moe_narrow)(y, x.dtype).reshape(shape)


def _next_prime(start: int, seen: set[int]) -> int:
    """The smallest prime above `start` not already handed out."""
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
    """
    from tokenizers import Regex, normalizers

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
        self.register_buffer("cache", None, persistent=False)

    def allocate_state(self, device, num_slots: int) -> None:
        """The per-slot compressed-id history, **on the host**.

        The hash is 64-bit integer arithmetic: a compressed id (< 99,092) times
        a multiplier (~1e13) is ~1e18, and the xor-rolled product is reduced
        modulo a 16M prime. Compiled for this host's NeuronCores, int64
        multiply, xor and modulo are evaluated at 32-bit width and wrap, so
        the device returns a residue that is in range but wrong, and every
        engram lookup reads the wrong row. Measured 2026-10-22 on
        trn3.48xlarge (neuronx-cc 2.28): `id * multiplier` came back as the
        int32-wrapped product on 100% of elements, and the layer-1 engram
        output was 79% off the reference while its input was 3% off. So the
        history and the hash stay on the host, where the arithmetic is exact
        and matches the reference bit for bit; the row indices go to the
        device as plain int64 that never exceed 2**29.
        """
        self.cache = torch.full((num_slots,), self.DEAD, dtype=torch.int64)
        self._host = {
            name: getattr(self, name).detach().cpu()
            for name in ("primes", "offsets", "multipliers", "token_map")
        }

    @torch._dynamo.disable
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        paged: PagedSpan,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns `[B, L, n_engram_layers, n_hash_cols]` row indices, computed on
        the host (see `allocate_state`) and moved to `input_ids.device`."""
        if self.cache is None:
            raise RuntimeError(
                "the n-gram hash state has no slot cache; "
                "allocate_state() must run before forward()"
            )
        host = self._host
        device = input_ids.device
        batch, seqlen = input_ids.shape
        ids = input_ids.detach().cpu()
        compressed = host["token_map"][ids]
        if token_mask is not None:
            compressed = torch.where(token_mask.detach().cpu(), compressed, self.DEAD)

        block_size = paged.block_size
        block_table = paged.block_table.detach().cpu()
        slot_mapping = paged.slot_mapping.detach().cpu()
        positions = paged.start_pos + torch.arange(seqlen)
        self.cache.index_copy_(0, slot_mapping[:seqlen], compressed.reshape(seqlen))

        def slots(pos: torch.Tensor) -> torch.Tensor:
            return block_table[pos // block_size] * block_size + pos % block_size

        positions = positions.unsqueeze(0).expand(batch, seqlen)
        tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.max_ngram_size):
            source = self.cache.index_select(
                0, slots((positions - shift).clamp_min(0).reshape(-1))
            ).reshape(batch, seqlen)
            blocked = blocked | (positions < shift) | (source == self.DEAD)
            tokens.append(torch.where(blocked, self.pad_id, source))
        stacked = torch.stack(tokens, dim=-1)

        products = stacked.unsqueeze(2) * host["multipliers"]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % host["primes"][:, i - 1])
        return (torch.cat(hashes, dim=-1) + host["offsets"]).to(device)

    @torch._dynamo.disable
    @torch.no_grad()
    def hash_spans(self, ids: torch.Tensor, paged: list[PagedSpan]) -> torch.Tensor:
        """`forward` for every span of a step in one pass: `ids` is the step's flat `[T]`
        token ids **on the host**, each span reads its rows `[begin, begin + length)`, and
        the result is `[1, rows, n_engram_layers, n_hash_cols]` on the host, the spans'
        rows in order (what `_engram_hashes` concatenated from one `forward` a span).

        The spans' tokens all go into the history before any is read, where `forward`
        writes a span's and reads it before the next span writes. That can only differ
        where one span reads a slot a later one writes, and sequences own disjoint
        pages: only the padded spans (`_paged_spans` points them all at block 0) share
        slots, and their rows are discarded. Uses the spans' host copies of the block
        table and slots (`PagedSpan.host_table` / `host_slots`), so nothing is read back
        from the device."""
        if self.cache is None:
            raise RuntimeError(
                "the n-gram hash state has no slot cache; "
                "allocate_state() must run before forward()"
            )
        host = self._host
        block_size = paged[0].block_size
        if len(paged) == 1:
            span = paged[0]
            rows = ids.reshape(-1)[span.begin : span.begin + span.length]
            slot_mapping = span.host_slots.reshape(-1)[: span.length]
            positions = span.start_pos + torch.arange(span.length)
            tables = span.host_table.reshape(1, -1)
            row_span = torch.zeros(span.length, dtype=torch.int64)
        else:
            flat = ids.reshape(-1)
            rows = torch.cat([flat[p.begin : p.begin + p.length] for p in paged])
            slot_mapping = torch.cat([p.host_slots.reshape(-1)[: p.length] for p in paged])
            positions = torch.cat([p.start_pos + torch.arange(p.length) for p in paged])
            width = max(int(p.host_table.numel()) for p in paged)
            tables = torch.stack([
                F.pad(p.host_table.reshape(-1), (0, width - int(p.host_table.numel())))
                for p in paged
            ])
            row_span = torch.repeat_interleave(
                torch.arange(len(paged)), torch.tensor([p.length for p in paged])
            )
        compressed = host["token_map"][rows]
        self.cache.index_copy_(0, slot_mapping.to(torch.int64), compressed)

        tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.max_ngram_size):
            pos = (positions - shift).clamp_min(0)
            source = self.cache[tables[row_span, pos // block_size] * block_size + pos % block_size]
            blocked = blocked | (positions < shift) | (source == self.DEAD)
            tokens.append(torch.where(blocked, self.pad_id, source))
        stacked = torch.stack(tokens, dim=-1).unsqueeze(0)

        products = stacked.unsqueeze(2) * host["multipliers"]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % host["primes"][:, i - 1])
        return torch.cat(hashes, dim=-1) + host["offsets"]


def _engram_gather(
    hashes: torch.Tensor,
    table: torch.Tensor,
    index: int,
    row_start: torch.Tensor,
    row_width: int,
) -> torch.Tensor:
    """This rank's contribution to the n-gram rows, `[1, T, n_hash_cols, dim]`."""
    ids = hashes[:, :, index, :]
    off_rank = (ids < row_start) | (ids >= row_start + row_width)
    local = (ids - row_start).masked_fill(off_rank, 0)
    out = F.embedding(local, table)
    return out.masked_fill(off_rank.unsqueeze(-1), 0)


class Engram(nn.Module):
    """n-gram memory read, gated by its match against the residual stream."""

    #: The floor under `|dot|` before the gate's square root, so its derivative
    #: at zero stays finite. `vendor/kernel.py`'s own constant, which
    #: `kernels/engram_gate` bakes in and `_check_kernel_constants` compares
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
        #: This rank's rows of the n-gram table, `[row_width, head_dim]` bf16.
        self.register_buffer(
            "table",
            torch.empty(0, config.engram_head_dim, dtype=torch.bfloat16),
            persistent=False,
        )
        #: Global row this rank's shard starts at, as a 0-d device tensor so it
        #: reaches the lookup graph as an *input* and the 64 ranks share one
        #: compile rather than keying 64 — see `_engram_gather`. `row_width` is
        self.row_start: torch.Tensor | None = None
        self.row_width = 0
        #: True once `bind_table` has installed a shard. False means the caller
        #: has to use the streaming path.
        self.resident = False

    def bind_table(self, table: torch.Tensor, row_start: int, row_width: int) -> None:
        """Install this rank's rows of the n-gram table."""
        if table.shape[0] != row_width:
            raise ValueError(
                f"engram layer {self.layer_id}: shard is {table.shape[0]} rows, "
                f"row_width says {row_width}. The width has to be exact — it is "
                "the bound `lookup` masks with, and it has to be the same on "
                "every rank for the graph to be shared."
            )
        if table.shape[1] != self.head_dim:
            raise ValueError(
                f"engram layer {self.layer_id}: shard is {table.shape[1]} wide, "
                f"engram_head_dim is {self.head_dim}"
            )
        self.table = table
        self.row_start = torch.tensor(
            int(row_start), dtype=torch.int64, device=table.device
        )
        self.row_width = int(row_width)
        self.resident = True

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
        width = self.hc_mult * self.hidden_size
        key = kv[..., :width]
        value = kv[..., width : width + self.hidden_size]
        weight = (self.q_weight.float() * self.k_weight.float()).reshape(1, width)
        out = _nki_entry("engram_gate", "gated_add")(
            x.reshape(-1, width),
            key.reshape(-1, width),
            value.reshape(-1, self.hidden_size),
            weight,
        ).reshape(x.shape)
        if token_mask is not None:
            out = torch.where(token_mask.unsqueeze(-1).unsqueeze(-1), out, x)
        return out


class HyperConnections(nn.Module):
    """The `hc_mult = 4` parallel residual streams and the mixing between them."""

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
        """
        hc = self.hc_mult
        lead = x.shape[:2]
        rows = x.flatten(2).reshape(-1, hc * self.hidden_size)
        # fp32, as the reference's `F.linear(x.float(), hc_fn)`: the `fp32w`
        # kernel narrows both operands and its output to bf16, and the three
        # coefficient sets (sigmoids and a Sinkhorn) come straight out of it.
        raw = torch.matmul(rows.float(), self.fn.t())
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
        """Collapse the copies into one sublayer input, weighted by `pre_mix`."""
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

            broadcast = post.unsqueeze(-1) * x.unsqueeze(-2)
            mixed = torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
            (broadcast + mixed).type_as(x)
        """
        if post.dtype is not torch.float32 or comb.dtype is not torch.float32:
            raise TypeError(
                f"hc_post wants fp32 coefficients, got {post.dtype}/{comb.dtype}"
            )
        return _nki_entry("hc_attn_out")(x, residual, post, comb)


class Block(nn.Module):
    """One layer: hyper-connected attention, then hyper-connected MoE FFN."""

    #: Overridden by `DSparkBlock`, as `vendor/model.py`'s `Block` does it.
    attention_cls = Attention

    def __init__(
        self,
        config: DeepseekV41Config,
        layer_id: int,
        tp_size: int,
        tp_rank: int,
        max_batch_size: int = 1,
        attn_tp: int | None = None,
        attn_rank: int | None = None,
        attn_group=None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        attn_tp = tp_size if attn_tp is None else attn_tp
        attn_rank = tp_rank if attn_rank is None else attn_rank
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = self.attention_cls(
            config, layer_id, attn_tp, max_batch_size,
            replicas=1 if attn_group is not None else tp_size // attn_tp,
            attn_group=attn_group,
        )
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
        """One layer over the `hc_mult`-copy residual stream."""
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


    def attn_half(
        self, stream, pre_mix, slices, paged, shared: list, tp_group,
        out_residual=None, out_normed=None,
    ):
        """Everything from the stream to the MoE's normalized input."""
        batched = stream.unsqueeze(0)
        residual = batched
        attn_pre, attn_post, attn_comb = self.hc_attn.mixes(batched)
        normed = self.attn_norm(HyperConnections.pre(batched, pre_mix.unsqueeze(0)))

        attn_out = self._attn_rows(normed, slices, paged, shared, tp_group)
        batched = HyperConnections.post(attn_out, residual, attn_post, attn_comb)

        residual = batched
        ffn_pre, ffn_post, ffn_comb = self.hc_ffn.mixes(batched)
        normed = self.ffn_norm(HyperConnections.pre(batched, attn_pre))
        if out_residual is None:
            return normed, residual, ffn_pre, ffn_post, ffn_comb
        out_residual.copy_(residual)
        out_normed.copy_(normed)
        return out_normed, out_residual, ffn_pre, ffn_post, ffn_comb

    def _attn_rows(self, normed, slices, paged, shared, tp_group):
        """The attention sublayer's output for every row of `normed`.

        Two arms, and which one runs is decided at trace time because `slices`
        is a Python list:

          * **one sequence** — `self.attn(...)` exactly as it has always been
            called. This is the arm the verifier and every batch-1 bench cell
            take, and it is kept on the unchanged `Attention.forward` so their
            graphs keep their keys and their NEFFs.
          * **several** — one `forward_spans` over the concatenated rows, which
            does the row-wise work once instead of once per sequence. See
            `Attention.forward_spans`.

        `_sequence_spans` tiles the live token axis with contiguous spans from
        row 0, so the second arm's rows are `[0, sum(lengths))` and anything
        above that is padding the old loop left at zero — reproduced here by
        the explicit zero tail. The contiguity is checked rather than assumed:
        a non-contiguous plan falls back to the loop, which is slow but right.
        """
        lengths = [length for _, length in slices]
        contiguous = all(
            begin == sum(lengths[:i]) for i, (begin, _) in enumerate(slices)
        )
        if len(slices) > 1 and contiguous and self.attn.spans_supported:
            covered = sum(lengths)
            attn_out = self.attn.forward_spans(
                normed[:, :covered], lengths, paged, shared, tp_group
            )
            total = normed.shape[1]
            if covered < total:
                attn_out = torch.cat(
                    [attn_out, normed.new_zeros(
                        normed.shape[0], total - covered, normed.shape[2]
                    )],
                    dim=1,
                )
            return attn_out

        attn_out = torch.zeros_like(normed)
        for (begin, length), span, span_shared in zip(slices, paged, shared):
            attn_out[:, begin : begin + length] = self.attn(
                normed[:, begin : begin + length],
                span.pos,
                span_shared,
                span,
                tp_group,
            )
        return attn_out

    def ffn_half(self, ffn_out, residual, ffn_post, ffn_comb):
        """The MoE's output mixed back into the residual streams."""
        return HyperConnections.post(ffn_out, residual, ffn_post, ffn_comb)


def _block_entry(block: nn.Module, name: str):
    """`block.<name>` as its own compiled graph, one per block."""
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
    """

    #: `Attention.forward_spans` does not serve this: a draft stage's KV comes
    #: from the main-stream tap and the draft block together, and
    #: `DSparkBlock.forward` calls `self.attn` directly rather than through
    #: `Block.attn_half`. Declared rather than left to inheritance so that a
    #: future caller of `forward_spans` cannot reach it by accident.
    spans_supported = False

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
        if paged.at_zero:
            paged.write(self.window_kv_cache, main_kv.reshape(seqlen, self.head_dim))
            return x

        bsz, block_size, _ = x.shape
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

        paged.write(self.window_kv_cache, main_kv.reshape(seqlen, self.head_dim))
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
        # The reference rounds `o` through fp8 (`act_quant(o, 32, ..., inplace=True)`)
        # before the grouped `wo_a` einsum; `grouped()` runs the GEMM raw, so do it here.
        # Measured 2026-10-22: without it layer 0's attention is 3.1% off the reference
        # and the drift compounds to ~30% by the last layer; with it the port's own
        # math replicated on CPU matched the device to 0.5%.
        o = act_quant_dequant(o, 32)
        o = o.reshape(bsz * block_size, self.n_local_groups, -1)
        o = torch.cat(
            [
                self.wo_a.grouped(o[:, g], g, self.o_lora_rank)
                for g in range(self.n_local_groups)
            ],
            dim=-1,
        ).reshape(bsz, block_size, -1)
        if self.world_size > 1 and tp_group is not None:
            # fp32 partials into the all-reduce: see `fp8_linear_f32`.
            out = nki_linear(o, self.wo_b.weight, self.wo_b.scale, "fp8", out_fp32=True)
            out = tp_all_reduce(out, self.attn_group or tp_group)
            if self.replicas > 1:
                out = out / self.replicas
            out = out.to(x.dtype)
        else:
            out = self.wo_b(o)
        return out


class DSparkBlock(Block):
    """A DSpark stage, stored under `mtp.<stage>`."""

    attention_cls = DSparkAttention

    def __init__(
        self,
        config: DeepseekV41Config,
        stage: int,
        tp_size: int,
        tp_rank: int,
        max_batch_size: int = 1,
        attn_tp: int | None = None,
        attn_rank: int | None = None,
        attn_group=None,
    ) -> None:
        layer_id = config.num_hidden_layers + stage
        super().__init__(
            config, layer_id, tp_size, tp_rank, max_batch_size,
            attn_tp=attn_tp, attn_rank=attn_rank, attn_group=attn_group,
        )
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
        y = self.ffn(y, None, weight_store, tp_group)
        x = HyperConnections.post(y, residual, ffn_post, ffn_comb)
        return x, ffn_pre

    def forward_embed(
        self, main_hidden: torch.Tensor, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage 0 only: build the draft stream and the main-stream input."""
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
        """Last stage only: draft logits, draft tokens and their confidence."""
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
        self.part_vocab_size = vocab_shard_width(config.vocab_size, tp_size)
        self.register_buffer(
            "embed_weight",
            torch.empty(self.part_vocab_size, rank, dtype=torch.bfloat16),
        )
        self.register_buffer(
            "head_weight",
            torch.empty(self.part_vocab_size, rank, dtype=torch.bfloat16),
        )
        self.world_size = tp_size
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
            if logits.shape[-1] != self.vocab_size:
                logits = logits[..., : self.vocab_size]
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


class DeepseekV41ForCausalLM(nn.Module):
    """DeepSeek-V4.1-Flash: embed -> expand to 4 copies -> 40 blocks ->
    collapse -> norm -> head, with the three DSpark stages hanging off the
    attention inputs of layers 37/38/39.
    """

    #: The forward takes `sampling_hint` (all_greedy, no_top_k, no_top_p) from the
    #: Neuron model runner (`NeuronModelRunner` passes it to models that set this) and
    #: samples temperature rows from the full vocabulary (`_random_sampler_entry`).
    accepts_sampling_hint = True

    def __init__(self, config: DeepseekV41Config) -> None:
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config

        # Resolve the NKI entries' compile backend here, in plain Python, so no
        # traced forward has to; see `_compile_backend`.
        _compile_backend()

        tp_group = get_tp_group()
        self.tp_group = tp_group
        self.world_size = tp_group.world_size
        self.rank = tp_group.rank_in_group
        if mk_model.PROF is not None:
            # `DEEPSEEK_V41_HOST_PROF`: every forward timed by part (`mk_model.HostProf`)
            inner = self.forward

            def _timed_forward(*args, **kwargs):
                mk_model.PROF.begin()
                try:
                    return inner(*args, **kwargs)
                finally:
                    drafts = getattr(self, "last_draft", None)
                    if mk_model._HOST_PROF_SYNC and drafts:
                        mk_model.PROF.mark("drafted", drafts[-1][0])
                    mk_model.PROF.end(getattr(self, "_mk_step", None), self.rank)

            self.forward = _timed_forward
            try:
                from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

                mk_model.RPC_PROF.wrap(NeuronModelRunner, self.rank)
            except ImportError:  # pragma: no cover - the plugin is always there when serving
                pass

        from .config import ATTENTION_TENSOR_PARALLEL_SIZE

        self.attn_tp = max(1, min(self.world_size, ATTENTION_TENSOR_PARALLEL_SIZE))
        if self.world_size % self.attn_tp:
            raise ValueError(
                f"world_size {self.world_size} is not a multiple of the "
                f"attention shard degree {self.attn_tp}, so the attention "
                "cannot be replicated evenly and an all-reduce of a head "
                "partial would not be a whole multiple of the true sum. Choose "
                "a tensor_parallel_size divisible by "
                f"{ATTENTION_TENSOR_PARALLEL_SIZE}."
            )
        self.attn_rank = self.rank % self.attn_tp
        self.attn_replicas = self.world_size // self.attn_tp
        self.attn_tp_group = self.tp_group
        self.attn_group_is_world = True
        if self.attn_replicas > 1:
            try:
                from vllm_neuron.parallel.neuron_parallel_state import (
                    get_neuron_ep_tp_group,
                )

                candidate = get_neuron_ep_tp_group()
            except Exception as exc:  # noqa: BLE001 - absence is the normal case
                logger.info(
                    "deepseek_v41: no EP-TP group (%s), so the attention "
                    "reduces over all %d ranks and divides by %d; see "
                    "deepseek_v41.require_attention_group",
                    exc, self.world_size, self.attn_replicas,
                )
            else:
                if candidate.world_size == self.attn_tp:
                    self.attn_tp_group = candidate
                    self.attn_group_is_world = False
                else:
                    # The EP-TP group is 8 consecutive ranks, which is the
                    # right group only at ATTENTION_TENSOR_PARALLEL_SIZE=8.
                    # Below that — TP=4 x PP=16 needs 4, since a 4-rank
                    # tensor-parallel group cannot contain an 8-way attention
                    # shard — a lane group of the right width is the same
                    # shape of membership (consecutive ranks, so one device at
                    # width 4) and is what this reduction wants. Falling
                    # through to the world would turn a one-device collective
                    # into a sixteen-device one, which iteration 3 measured at
                    # ~1.9 ms against ~0.1 ms, 43 times a forward.
                    from vllm_neuron.parallel.neuron_parallel_state import (
                        get_neuron_lane_group,
                    )

                    try:
                        self.attn_tp_group = get_neuron_lane_group(self.attn_tp)
                    except Exception as lane_exc:  # noqa: BLE001
                        logger.warning(
                            "deepseek_v41: the EP-TP group is %d ranks but the "
                            "attention splits %d ways, and no lane group of "
                            "that width exists (%s), so the head-partial "
                            "reduction falls back to the world and divides by "
                            "%d. Set DEEPSEEK_V41_LANE_WIDTHS=%d to give it "
                            "the group it wants.",
                            candidate.world_size, self.attn_tp, lane_exc,
                            self.attn_replicas, self.attn_tp,
                        )
                    else:
                        self.attn_group_is_world = False
                        logger.info(
                            "deepseek_v41: the attention splits %d ways and "
                            "the EP-TP group is %d, so the head-partial "
                            "reduction uses the %d-rank lane group %s",
                            self.attn_tp, candidate.world_size, self.attn_tp,
                            self.attn_tp_group.ranks,
                        )
        _LOG_RANK[0] = self.rank == 0

        #: Pipeline emulation state. `pp_size == 1` is the shipped
        #: configuration and every branch below collapses to today's path:
        #: `stage_group is tp_group`, `block_stage` is all zeros and
        #: `pp_stage` is 0, so every rank owns every block.
        self.pp_size = 1
        self.pp_stage = 0
        self.pp_tp_width = self.world_size
        self.stage_group = self.tp_group
        self.block_stage = [0] * config.num_hidden_layers
        #: Blocks every stage runs, whichever stage owns them. The attention's
        #: compressed KV, its index top-k and its candidate blocks are produced
        #: at `kv_source_layer_ids` / `index_source_layer_ids` and read by every
        #: later block through `SharedAttentionState`, so a stage whose first
        #: compressing block is not a source would reach
        #: `shared.topk_idxs.to(...)` with `None`. Running the twelve source
        #: blocks everywhere seeds that state on every rank and costs the
        #: critical path **nothing**: all 64 ranks execute them at the same
        #: point in the program, concurrently, each with its own stage's
        #: TP-width structure — which is exactly what a real pipeline pays for
        #: the one stage that owns them.
        self.pp_always: set[int] = set()
        if _PP_EMULATE > 1:
            self._init_pp_emulation(_PP_EMULATE)

        _log_two_core_plan()

        if _PROBE_AR_WIDTH:
            from vllm_neuron.parallel.neuron_parallel_state import (
                get_neuron_lane_group,
            )

            _NARROW_AR_GROUP[0] = get_neuron_lane_group(_PROBE_AR_WIDTH)
            logger.warning(
                "deepseek_v41: DEEPSEEK_V41_PROBE_AR_WIDTH=%d — every "
                "world-wide all-reduce now reduces over %d consecutive ranks "
                "(%s), so every sum on the forward path is PARTIAL and the "
                "TOKENS ARE NONSENSE. This is a collective-latency probe; read "
                "the prefill TTFT and the decode step time off it and nothing "
                "else. A probe's TPOT is not comparable to a real run's either "
                "— partial sums collapse the drafter's acceptance, so convert "
                "to step time with TPOT x (1 + 5 x acceptance) first.",
                _PROBE_AR_WIDTH, _PROBE_AR_WIDTH,
                _NARROW_AR_GROUP[0].ranks,
            )
        if self.attn_replicas > 1 and _LOG_RANK[0]:
            logger.info(
                "deepseek_v41: rank %d/%d — attention sharded %d ways "
                "(attn_rank %d) and replicated %dx; experts and vocab sharded "
                "%d ways",
                self.rank, self.world_size, self.attn_tp, self.attn_rank,
                self.attn_replicas, self.world_size,
            )

        self.hc_mult = config.hc_mult
        self.part_vocab_size = vocab_shard_width(config.vocab_size, self.world_size)
        #: Columns of `logits_shard` this rank owns that are *not* real tokens.
        self.vocab_pad = max(
            0, (self.rank + 1) * self.part_vocab_size - config.vocab_size
        )
        #: `[part_vocab_size]` fp32, zero on real columns and `-inf` on the pad:
        #: a buffer built here, on every rank, never a lazily filled per-rank
        #: cache. The forward's tail (norm, logits, the gather, the sampler) is
        #: one dynamo frame, and dynamo guards on every Python-side value it
        #: reads; a dict that only the padded rank fills made that rank, alone,
        #: re-trace the frame on its next prefill-shaped step and recompile its
        #: all-gather graph while the other 63 ranks already sat in the
        #: collective -- the whole engine hung (2026-10-23, verifier short phase).
        #: Built on `allocate_staging_buffers`' device (the module is constructed
        #: on the meta device).
        self.register_buffer("vocab_pad_bias", None, persistent=False)

        #: Where this rank sits (`mk_model.PipePlan.from_groups`): the flat scheme
        #: (one tensor-parallel group of 64) or the pipeline scheme (vLLM's 8 stages
        #: of 8). A stage builds, loads and runs only its own blocks; their MoE is
        #: built with the stage's tensor-parallel rank and width (48 experts a rank)
        #: and the attention is sharded 8 ways within the stage, as it already is in
        #: the flat scheme. The embedding lives on the first stage (and the last,
        #: whose DSpark stages embed the drafted ids); the norm, the head and the
        #: DSpark stages on the last.
        self.mk_plan = mk_model.PipePlan.from_groups(config.num_hidden_layers)
        plan = self.mk_plan
        if plan.pp > 1 and not _MEGAKERNEL:
            raise NotImplementedError("pipeline_parallel_size > 1 runs on the megakernel path only")
        if plan.pp > 1 and plan.tp != self.world_size:
            raise RuntimeError(f"stage width {plan.tp} != tensor-parallel world {self.world_size}")
        self.block_ids: list[int] = plan.owned()
        if plan.is_first or plan.is_last:
            self.register_buffer(
                "embed_weight",
                torch.empty(self.part_vocab_size, config.hidden_size, dtype=torch.bfloat16),
            )
        else:
            self.embed_weight = None
        block_tp, block_rank = plan.tp, plan.stage_rank
        if plan.pp > 1:
            logger.info(
                "deepseek_v41: pipeline scheme -- vLLM runs %d stages x %d tensor-parallel ranks; "
                "this rank (global %d) is rank %d of stage %d, holds blocks %d-%d with %d of the "
                "%d routed experts and %d-way attention, reduces over its stage, and hands its "
                "activations to the next stage through the in-kernel collectives of "
                "kernels/pp_kernels.py",
                plan.pp, plan.tp, plan.rank, plan.stage_rank, plan.stage,
                self.block_ids[0], self.block_ids[-1], config.n_routed_experts // block_tp,
                config.n_routed_experts, self.attn_tp,
            )
        # A `ModuleDict` keyed by the layer id, so the state-dict paths stay
        # `layers.<layer_id>.*` whichever blocks this rank holds.
        self.layers = nn.ModuleDict(
            {
                str(layer_id): Block(
                    config, layer_id, block_tp, block_rank,
                    self._max_batch_size(),
                    attn_tp=self.attn_tp, attn_rank=self.attn_rank,
                    attn_group=None if self.attn_group_is_world
                    else self.attn_tp_group,
                )
                for layer_id in self.block_ids
            }
        )
        self.blocks_by_id: dict[int, Block] = {
            layer_id: self.layers[str(layer_id)] for layer_id in self.block_ids
        }
        #: The kv-source block whose cache rows cross the boundary into this stage
        #: (`mk_model.shipped_source_id`), when another stage owns it: its attention
        #: module stands in for the source here -- the compressor and indexer weights
        #: the consumers' kernel arguments borrow, and the caches the boundary fills
        #: (`bind_kv_cache`). Loaded from the source layer's own checkpoint keys.
        self.src_attn = nn.ModuleDict()
        shipped = mk_model.shipped_source_id(config, self.block_ids[0]) if plan.pp > 1 else None
        if shipped is not None and shipped not in self.blocks_by_id:
            self.src_attn[str(shipped)] = Attention(
                config, shipped, self.attn_tp, self._max_batch_size(),
                replicas=1 if not self.attn_group_is_world else block_tp // self.attn_tp,
                attn_group=None if self.attn_group_is_world else self.attn_tp_group,
            )
        if plan.is_last:
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.register_buffer(
                "head_weight",
                torch.empty(self.part_vocab_size, config.hidden_size, dtype=torch.bfloat16),
            )
            self.mtp = nn.ModuleList(
                [
                    DSparkBlock(
                        config, stage, self.world_size, self.rank,
                        self._max_batch_size(),
                        attn_tp=self.attn_tp, attn_rank=self.attn_rank,
                        attn_group=None if self.attn_group_is_world
                        else self.attn_tp_group,
                    )
                    for stage in range(config.num_nextn_predict_layers)
                ]
            )
        else:
            self.norm = None
            self.head_weight = None
            self.mtp = nn.ModuleList([])

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
        self._engram_row_slots: dict[tuple, torch.Tensor] = {}

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


    def _init_pp_emulation(self, pp_size: int) -> None:
        """Set up `_PP_EMULATE`'s stage map and stage-local group. See the
        `_PP_EMULATE` docstring for what this does and does not model.
        """
        if self.world_size % pp_size:
            raise ValueError(
                f"DEEPSEEK_V41_PP_EMULATE={pp_size} does not divide the world "
                f"size {self.world_size}: TP x PP has to be the whole world, "
                "or the factorization is not of this deployment."
            )
        tp_width = self.world_size // pp_size
        if tp_width < self.attn_tp:
            raise ValueError(
                f"DEEPSEEK_V41_PP_EMULATE={pp_size} leaves a {tp_width}-rank "
                f"tensor-parallel group, narrower than the {self.attn_tp}-way "
                "attention shard, so the attention's own group would straddle "
                "two stages. Lower ATTENTION_TENSOR_PARALLEL_SIZE first."
            )
        self.pp_size = pp_size
        self.pp_tp_width = tp_width
        self.pp_stage = self.rank // tp_width
        n_layers = self.config.num_hidden_layers
        #: Stage of each backbone block, balanced: block i goes to stage
        #: `i * pp / n_layers`, which gives ceil/floor(n_layers/pp) blocks a
        #: stage and keeps consecutive blocks together so a boundary is crossed
        #: `pp - 1` times rather than once a block.
        self.block_stage = [i * pp_size // n_layers for i in range(n_layers)]
        self.pp_always = set(self.config.kv_source_layer_ids) | set(
            self.config.index_source_layer_ids
        )
        if tp_width < self.world_size:
            from vllm_neuron.parallel.neuron_parallel_state import (
                get_neuron_lane_group,
            )

            self.stage_group = get_neuron_lane_group(tp_width)
        counts = [self.block_stage.count(s) for s in range(pp_size)]
        logger.warning(
            "deepseek_v41: DEEPSEEK_V41_PP_EMULATE=%d — emulating TP=%d x "
            "PP=%d. Rank %d is in stage %d and runs blocks %s of %d; each "
            "block's expert all-reduce reduces over %d consecutive ranks (%s) "
            "and its routed-expert loop runs %dx to match a %d-wide "
            "tensor-parallel rank's %d owned slots. Blocks a stage: %s. "
            "THE TOKENS ARE WRONG — every expert sum covers %d of the 64 "
            "shards and skipped blocks leave their KV unwritten. This is a "
            "LATENCY PROBE; never run `verifier` under it.",
            pp_size, tp_width, pp_size, self.rank, self.pp_stage,
            [i for i, s in enumerate(self.block_stage) if s == self.pp_stage],
            n_layers, tp_width, self.stage_group.ranks, pp_size, tp_width,
            self.config.n_routed_experts // tp_width, counts, tp_width,
        )

    def _pp_boundary(
        self, stream: torch.Tensor, pre_mix: torch.Tensor, owner: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move `owner`'s hyper-connection stream to every rank.

        A world-wide sum in which exactly one rank contributes: the lead rank
        of the owning stage sends its values and all 63 others send exact
        zeros, so the sum is the sender's tensor bit for bit. `zeros_like`
        rather than `x * 0.0` on purpose — a non-owning rank's `stream` is
        whatever the last boundary left there, and `0.0 * inf` is `nan`.

        This is the emulation's transport, and it is the pessimistic one: two
        64-rank all-reduces, ~1.9 ms each, against ~0.4 ms for the neighbour
        permute a real pipeline would use (`kernels/pp_permute.py`). A real
        pipeline's figure is therefore *better* than what this measures by
        about 3 ms a boundary.
        """
        lead = self.pp_stage == owner and self.rank % self.pp_tp_width == 0
        send_stream = stream if lead else torch.zeros_like(stream)
        send_mix = pre_mix if lead else torch.zeros_like(pre_mix)
        # One collective, not two. At p = 16 that is 16 world-wide all-reduces
        # a forward instead of 32, and collective *resources* — not time — are
        # what killed the first 16-stage attempt: `ENC:enc_barrier_internal
        # ... failed to get NCCL comm for a replica` while loading a NEFF,
        # three decode steps into the measured pass. Same family as iteration
        # 3 §5's CC-barrier exhaustion.
        flat = torch.cat(
            [send_stream.reshape(-1), send_mix.reshape(-1)], dim=0
        )
        summed = tp_all_reduce(flat, self.tp_group)
        cut = send_stream.numel()
        return (
            summed[:cut].reshape(stream.shape),
            summed[cut:].reshape(pre_mix.shape),
        )

    def _check_kernel_constants(self) -> None:
        """The constants the kernels bake in; make sure the config agrees."""
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


    def _build_shard_plan(self) -> dict[str, ShardRule]:
        """One rule per checkpoint key, keyed by this model's buffer name."""
        config = self.config
        plan: dict[str, ShardRule] = {
            "embed.weight": ShardRule(
                0, "vocab-parallel embedding", align=VOCAB_SHARD_ALIGN
            ),
            "head.weight": ShardRule(
                0, "vocab-parallel LM head", align=VOCAB_SHARD_ALIGN
            ),
            "norm.weight": REPLICATE,
        }

        def add_attention(prefix: str, layer_id: int) -> None:
            plan[f"{prefix}.attn_sink"] = attn_shard(
                0, "one bias per query head; shards with the heads of wq_b"
            )
            plan[f"{prefix}.q_norm.weight"] = REPLICATE
            plan[f"{prefix}.kv_norm.weight"] = REPLICATE
            plan[f"{prefix}.wq_a.weight"] = REPLICATE
            plan[f"{prefix}.wq_a.scale"] = REPLICATE
            plan[f"{prefix}.wq_b.weight"] = attn_shard(0, "column-parallel: query heads")
            plan[f"{prefix}.wq_b.scale"] = attn_shard(0, "follows wq_b.weight dim 0")
            plan[f"{prefix}.wkv.weight"] = REPLICATE
            plan[f"{prefix}.wkv.scale"] = REPLICATE
            plan[f"{prefix}.wo_a.weight"] = attn_shard(
                0, "column-parallel over o_groups; must agree with the head split"
            )
            plan[f"{prefix}.wo_a.scale"] = attn_shard(0, "follows wo_a.weight dim 0")
            plan[f"{prefix}.wo_b.weight"] = attn_shard(
                1, "row-parallel: the o_groups*o_lora_rank contraction"
            )
            plan[f"{prefix}.wo_b.scale"] = attn_shard(1, "follows wo_b.weight dim 1")

            if layer_id in config.kv_source_layer_ids:
                plan[f"{prefix}.compressor.norm.weight"] = REPLICATE
                plan[f"{prefix}.compressor.wkv.weight"] = REPLICATE
                if config.compress_ratios[layer_id] > 1:
                    plan[f"{prefix}.compressor.wgate.weight"] = REPLICATE
            if layer_id in config.index_source_layer_ids:
                plan[f"{prefix}.indexer.wq_b.weight"] = attn_shard(
                    0, "column-parallel: index heads"
                )
                plan[f"{prefix}.indexer.wq_b.scale"] = attn_shard(
                    0, "follows indexer.wq_b.weight dim 0"
                )
                plan[f"{prefix}.indexer.weights_proj.weight"] = attn_shard(
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
            plan[f"{prefix}.ffn.gate.weight"] = REPLICATE
            plan[f"{prefix}.ffn.gate.bias"] = REPLICATE
            plan[f"{prefix}.ffn.gate.bias_vl"] = REPLICATE
            for name in ("w1", "w2", "w3"):
                plan[f"{prefix}.ffn.shared_experts.{name}.weight"] = REPLICATE
                plan[f"{prefix}.ffn.shared_experts.{name}.scale"] = REPLICATE
            add_attention(f"{prefix}.attn", layer_id)
            if layer_id in config.engram_layer_ids:
                plan[f"{prefix}.engram.wkv.weight"] = REPLICATE
                plan[f"{prefix}.engram.wkv.scale"] = REPLICATE
                plan[f"{prefix}.engram.q_weight"] = REPLICATE
                plan[f"{prefix}.engram.k_weight"] = REPLICATE

        for layer_id in range(config.num_hidden_layers):
            add_block(f"layers.{layer_id}", layer_id)
        # The pipeline stage's replica of the kv source it receives rows from
        # (`src_attn`): the source layer's attention rules under the replica's path.
        for sid in self.src_attn:
            add_attention(f"src_attn.{sid}", int(sid))
        for stage in range(config.num_nextn_predict_layers):
            add_block(f"mtp.{stage}", config.num_hidden_layers + stage)

        plan["mtp.0.main_proj.weight"] = REPLICATE
        plan["mtp.0.main_proj.scale"] = REPLICATE
        plan["mtp.0.main_norm.weight"] = REPLICATE
        last = config.num_nextn_predict_layers - 1
        plan[f"mtp.{last}.norm.weight"] = REPLICATE
        plan[f"mtp.{last}.markov_head.embed.weight"] = ShardRule(
            0,
            "vocab-parallel, like the embedding it mirrors",
            align=VOCAB_SHARD_ALIGN,
        )
        plan[f"mtp.{last}.markov_head.head.weight"] = ShardRule(
            0,
            "vocab-parallel, like the LM head it mirrors",
            align=VOCAB_SHARD_ALIGN,
        )
        plan[f"mtp.{last}.confidence_head.proj.weight"] = REPLICATE
        return plan

    def shard_parameter(
        self, key: str, tensor: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """This rank's slice of one checkpoint tensor, on `device`."""
        rule = self._shard_plan.get(key)
        if rule is None:
            raise KeyError(
                f"{key!r} has no sharding rule. Every checkpoint key this model "
                "loads needs one — add it to _build_shard_plan rather than "
                "defaulting to replicate."
            )
        degree, rank = self.shard_group(rule)
        if rule.dim is None or degree == 1:
            return tensor.to(device)

        extent = tensor.shape[rule.dim]
        if rule.align is not None:
            width = vocab_shard_width(extent, degree)
            start = min(rank * width, extent)
            have = max(0, min(width, extent - start))
            shard = tensor.narrow(rule.dim, start, have)
            if have < width:
                pad_shape = list(shard.shape)
                pad_shape[rule.dim] = width - have
                shard = torch.cat(
                    [shard, shard.new_zeros(pad_shape)], dim=rule.dim
                )
            return shard.to(device)

        if extent % degree:
            raise ValueError(
                f"{key}: dim {rule.dim} has extent {extent}, which is not "
                f"divisible by the {rule.group} shard degree {degree} "
                f"({rule.reason})"
            )
        width = extent // degree
        return tensor.narrow(rule.dim, rank * width, width).to(device)

    def shard_group(self, rule: ShardRule) -> tuple[int, int]:
        """`(degree, rank)` for `rule` — which ranks it splits across."""
        if rule.group == "attn":
            return self.attn_tp, self.attn_rank
        return self.world_size, self.rank

    def verify_shard_plan(self, index) -> list[str]:
        """Check the plan against the checkpoint: every key present, every
        sharded axis divisible, every resulting shape matching the buffer it
        will be assigned to. Returns the problems; empty means consistent.
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
            degree, _ = self.shard_group(rule)
            shape = list(index.ref(key).shape)
            if rule.dim is not None and rule.align is not None:
                shape[rule.dim] = vocab_shard_width(shape[rule.dim], degree)
            elif rule.dim is not None:
                if shape[rule.dim] % degree:
                    problems.append(
                        f"{key}: dim {rule.dim} extent {shape[rule.dim]} is not "
                        f"divisible by the {rule.group} shard degree {degree} "
                        f"({rule.reason})"
                    )
                    continue
                shape[rule.dim] //= degree
            target = buffers.get(_buffer_name(name))
            if target is None:
                problems.append(f"{name}: no buffer named {_buffer_name(name)!r}")
            elif list(target.shape) != shape:
                problems.append(
                    f"{name}: checkpoint shard is {shape}, buffer is "
                    f"{list(target.shape)}"
                )
        return problems


    def get_kv_spec(self) -> KVSpec:
        """One `LayerSpec` per attention, for the 128-slot sliding window.

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
        """
        dtype = self.config.torch_dtype
        layers = []
        for layer_id, block in zip(self.block_ids, self.layers.values()):
            layers.append(
                LayerSpec(
                    name=f"layers.{layer_id}.attn",
                    num_kv_heads=1,
                    head_size=block.attn.head_dim,
                    dtype=dtype,
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
        attentions = [(f"layers.{i}.attn", b.attn) for i, b in zip(self.block_ids, self.layers.values())]
        attentions += [(f"mtp.{s}.attn", b.attn) for s, b in enumerate(self.mtp)]

        for name, attn in attentions:
            if name not in kv_caches:
                raise KeyError(
                    f"KV cache for {name} was not allocated; get_kv_spec() "
                    f"declared it, so this is a runtime/spec mismatch"
                )
            attn.window_kv_cache = kv_caches[name][0]

        max_len = self._max_sequence_length()
        self._bind_compress_read_ceiling(max_len)
        device = self.device
        window = next(iter(self.layers.values())).attn.window_kv_cache
        num_slots = window.shape[0] * window.shape[2]
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
            attn.build_freqs_cis(max_len, device)
        # The replica of a kv source another stage owns: the caches the stage boundary
        # scatters this step's rows into (no compressor state: the source stage pools).
        for attn in self.src_attn.values():
            attn.compress_kv_cache = torch.zeros(
                num_slots, attn.head_dim, dtype=self.config.torch_dtype, device=device
            )
            attn.index_k_cache = torch.zeros(
                num_slots, self.config.index_head_dim, dtype=self.config.torch_dtype, device=device
            )
            own_bytes += sum(
                t.numel() * t.element_size() for t in (attn.compress_kv_cache, attn.index_k_cache)
            )

        owns_engram = any(
            self.blocks_by_id[lid].engram is not None
            for lid in self.config.engram_layer_ids if lid in self.blocks_by_id
        )
        if owns_engram and self.ngram_hash is None:
            self.ngram_hash = self._build_ngram_hash(device)
            self.ngram_hash.allocate_state(device, num_slots)

        logger.info(
            "deepseek_v41: %d KV slots — vLLM's window caches are "
            "%.2f GiB over %d layers, this model's own per-slot caches "
            "(compressed KV, index keys) are %.2f GiB. Neither is in "
            "`get_kv_spec`'s accounting for the second of those; see "
            "model.get_kv_spec",
            num_slots,
            kv_bytes / 2**30,
            len(kv_caches),
            own_bytes / 2**30,
        )
        if os.environ.get("DEEPSEEK_V41_CHECK_COLLECTIVES", "") not in ("", "0"):
            self._check_collectives(device)

    @torch._dynamo.disable
    def _check_collectives(self, device) -> None:
        """`DEEPSEEK_V41_CHECK_COLLECTIVES=1`: sum/gather known values over every
        group this port reduces over, and log whether the runtime got them right.
        """
        r = self.rank
        x = torch.full((1024,), float(r), dtype=torch.float32, device=device)
        world_sum = tp_all_reduce(x, self.tp_group).cpu()
        attn_sum = tp_all_reduce(x, self.attn_tp_group).cpu()
        gathered = tp_all_gather_last(x[:4].contiguous(), self.tp_group).cpu()
        world_ranks = list(range(self.world_size))
        attn_ranks = list(getattr(self.attn_tp_group, "ranks", world_ranks))
        exp_world = float(sum(world_ranks))
        exp_attn = float(sum(attn_ranks))
        exp_gather = torch.tensor(
            [float(i) for i in world_ranks for _ in range(4)]
        )
        ok_world = bool((world_sum == exp_world).all())
        ok_attn = bool((attn_sum == exp_attn).all())
        ok_gather = bool((gathered == exp_gather).all())
        logger.warning(
            "deepseek_v41 COLLECTIVES rank=%d attn_group=%s | world all-reduce %s "
            "(got %s expected %s) | attn all-reduce %s (got %s expected %s) | "
            "all-gather %s (got %s)",
            r, attn_ranks, "OK" if ok_world else "WRONG",
            world_sum.unique().tolist(), exp_world,
            "OK" if ok_attn else "WRONG", attn_sum.unique().tolist(), exp_attn,
            "OK" if ok_gather else "WRONG",
            gathered.reshape(-1, 4)[:, 0].tolist() if not ok_gather else "expected",
        )


    def allocate_staging_buffers(self, device: torch.device) -> None:
        """Materialize the per-layer expert staging slots on `device`.

            NotImplementedError: Cannot copy out of meta tensor; no data!
        """
        bias = torch.zeros(self.part_vocab_size, dtype=torch.float32)
        if self.vocab_pad:
            bias[self.part_vocab_size - self.vocab_pad :] = float("-inf")
        self.vocab_pad_bias = bias.to(device)
        blocks = list(self.layers.values()) + list(self.mtp)
        for block in blocks:
            block.ffn.expert_slot.allocate(device)

        per_expert = sum(
            buffer.numel() * buffer.element_size()
            for buffer in blocks[0].ffn.expert_slot.buffers()
        )
        _EXPERT_BYTES[0] = per_expert
        wants = [block.ffn.n_local_experts for block in blocks]
        wanted = sum(wants)
        n_owned = max(1, sum(1 for w in wants if w))
        affordable = _EXPERT_CACHE_BYTES // (per_expert * n_owned)
        allocated = sum(
            block.ffn.allocate_cache_slots(self.config, device, min(want, affordable))
            for block, want in zip(blocks, wants)
        )
        logger.info(
            "deepseek_v41: expert slots %d of %d owned over %d layers "
            "(%.1f MiB each, %.2f GiB of HBM, budget pays for %d a layer) for "
            "rank %d/%d — %s",
            allocated, wanted, len(blocks), per_expert / 1024**2,
            allocated * per_expert / 1024**3, affordable,
            self.rank, self.world_size,
            "pool is RESIDENT" if allocated == wanted else
            "NOT resident: the budget is short, so these layers stream. Raise "
            "DEEPSEEK_V41_EXPERT_CACHE_BYTES or widen the world.",
        )

    def _build_ngram_hash(self, device) -> NgramHashState:
        """Build the n-gram hash state, which needs the *tokenizer*."""
        from transformers import AutoTokenizer
        from vllm.config import get_current_vllm_config

        model_config = get_current_vllm_config().model_config
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.tokenizer or model_config.model
        )
        state = NgramHashState(self.config, tokenizer)
        return state.to(device)

    def _bind_compress_read_ceiling(self, max_len: int) -> None:
        """Pin the compressed-group read to the context, above one rank."""
        global _COMPRESS_READ_CEILING
        if self.world_size <= 1:
            return
        ceiling = int(max_len)
        if ceiling <= 0 or _COMPRESS_READ_CEILING == ceiling:
            return
        _COMPRESS_READ_CEILING = ceiling
        logger.info(
            "deepseek_v41: compressed-group read pinned to the "
            "%d-token context (ratio 1 -> %d groups, ratio 2 -> %d) so every "
            "prefill chunk and decode step share one set of per-layer graphs; "
            "see model._COMPRESS_READ_CEILING",
            ceiling, _compress_read_len(1, 1), _compress_read_len(1, 2),
        )

    def _max_sequence_length(self) -> int:
        """The longest sequence this engine will serve."""
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

    def logits_shard(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """This rank's columns of the logits: `[T, part_vocab_size]`."""
        logits = nki_linear(hidden_states, self.head_weight, None, "bf16").float()
        if self.vocab_pad:
            logits = logits + self.vocab_pad_bias
        return logits

    def gather_logits(self, logits_shard: torch.Tensor) -> torch.Tensor:
        """Every rank's columns, concatenated and trimmed to the real vocabulary."""
        if self.world_size == 1:
            return logits_shard
        gathered = tp_all_gather_last(logits_shard, self.tp_group)
        if gathered.shape[-1] != self.config.vocab_size:
            gathered = gathered[..., : self.config.vocab_size]
        return gathered

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Vocab-parallel unembedding, gathered to full width."""
        return self.gather_logits(self.logits_shard(hidden_states))


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
        sampling_hint: tuple | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """tokens -> logits, through all 40 layers and the three DSpark stages.

            Explanation: torch.compile detected a graph break in a for/while
            loop. Skipping the frame and falling back to eager, as graph breaks
            in loops are not supported.

            embed -> expand to hc_mult=4 copies -> 40 x Block -> collapse with
            the last layer's pre_mix -> norm -> head
        """
        if self.weight_store is None:
            raise RuntimeError(
                "forward() needs the weight store that load_weights attaches: "
                "the routed experts and the engram tables are streamed, not "
                "resident."
            )
        meta = self._attn_meta(attn_metadata)
        rows_per_seq = self._decode_rows_per_seq(meta, int(input_ids.shape[0]))
        live_spans = self._sequence_spans(
            positions, meta["slot_mapping"], rows_per_seq
        )
        live_rows = sum(length for _, length, _ in live_spans)
        spans, bucket_meta = self._bucket_rows(
            live_spans, meta, int(input_ids.shape[0]), rows_per_seq
        )
        paged = self._paged_spans(spans, bucket_meta, len(live_spans))
        if _TRACE_INPUTS:
            self._trace_inputs(input_ids, positions, meta, live_spans, spans)
        live_lengths = [length for _, length, _ in live_spans]
        # `_draft` indexes its per-row tensors by span, so a bucket that added
        # spans has to extend this too. The drafts those rows produce are
        # discarded — `NeuronModelRunner._sanitize_sync_drafts` aligns the row
        # count to the request count — but the lengths have to be in range.
        if len(live_lengths) < len(spans):
            live_lengths = live_lengths + [
                length for _, length, _ in spans[len(live_lengths):]
            ]
        slices = [(begin, length) for begin, length, _ in spans]
        bucket_rows = sum(length for _, length, _ in spans)

        plan = self.mk_plan
        if _DUMP_DIR:
            _DUMP_LIVE[0] = live_rows
            _DUMP_STEP[0] += 1
        # A decode step of the flat scheme embeds its tokens in its first fused launch
        # (`mk_model.fuses_embed`).
        fuse_embed = (_MEGAKERNEL and plan.is_first and inputs_embeds is None and not _DUMP_DIR
                      and mk_model.fuses_embed(self, spans))
        if plan.is_first and not fuse_embed:
            hidden = self.embed_input_ids(input_ids)
            hidden = NF.merge_prompt_embeds(hidden, inputs_embeds, is_token_ids)
            if _DUMP_DIR:
                self._dump("embed", hidden, live_rows)
            stream, pre_mix = _nki_entry("hc_expand")(hidden.unsqueeze(0))
            stream, pre_mix = stream.squeeze(0), pre_mix.squeeze(0)
        else:
            # A later pipeline stage: the stream arrives from the previous stage
            # (`mk_model.run_recv`, below); or the first fused launch embeds.
            stream = pre_mix = None

        # The step's tokens on the host, read back once: the n-gram hash and the fused
        # embedding's id column are both host work on them.
        ids_host = (mk_model.host_ids(input_ids)
                    if _MEGAKERNEL and (self.ngram_hash is not None or fuse_embed) else None)
        # The n-gram ids of the engram blocks this rank holds (the hash state is
        # only built on stages with one).
        hashes = (self._engram_hashes(input_ids, paged, ids_host)
                  if self.ngram_hash is not None else None)

        shared = [SharedAttentionState() for _ in paged]
        main_hiddens: list[torch.Tensor] = []
        target_layers = set(self.config.dspark_target_layer_ids)

        _MOE_LIVE_ROWS[0] = live_rows if live_rows != bucket_rows else None

        if _MEGAKERNEL:
            # One specialized megakernel per block (`mk_kernels`: by phase, block
            # archetype and scheme): the engram read-out, the DSpark tap and every
            # collective happen inside the kernels. See `mk_model.run_blocks`.
            if self.pp_size != 1:
                raise NotImplementedError("the megakernel path serves pp_size == 1")
            T = int(input_ids.shape[0])
            if bucket_rows != T:
                raise NotImplementedError(
                    f"the megakernel path wants the spans to tile the {T} "
                    f"rows of the step, got {bucket_rows}"
                )
            step = mk_model.make_step(spans, paged, self.device, hashes,
                                      decode_rows=1 + int(self.config.dspark_block_size),
                                      stable=plan.scheme == "tp64")
            self._mk_step = step
            host_heap_trim("layers", self.world_size)
            dump = (lambda name, t: self._dump(name, t, live_rows)) if _DUMP_DIR else None
            if not plan.is_first:
                stream, pre_mix = mk_model.run_recv(self, step, T)
            # An all-greedy speculative decode step: the last fused launch also runs the
            # head and the argmax (`mk_model.fuses_head`); its logits are never read
            # (the step returns the rejection sampler's tokens).
            head = (dump is None and not _TRACE_INPUTS and self.sampler is not None
                    and sampling_params is not None and sampling_hint is not None
                    and bool(sampling_hint[0]) and spec_decode_metadata is not None
                    and plan.is_last and mk_model.fuses_head(self, step))
            if fuse_embed:
                step.embed_ids = mk_model.embed_ids(input_ids, ids_host)
            stream, pre_mix = mk_model.run_blocks(self, stream, pre_mix, step, dump=dump, head=head,
                                                  embed=fuse_embed)
            if not plan.is_last:
                # Hand off to the next stage; nothing to sample here (the runner's
                # non-last-stage path ignores the value, which completes with the
                # hand-off).
                self.last_draft = None
                return mk_model.run_send(self, stream, pre_mix, step)
            main_hiddens = step.taps

        for layer_id, block in ([] if _MEGAKERNEL else zip(self.block_ids, self.layers.values())):
            if layer_id % _HEAP_TRIM_EVERY == 0:
                host_heap_trim(f"layer {layer_id}", self.world_size)
            # `owns` is True for every block when pp_size == 1, which is the
            # shipped configuration. Under `_PP_EMULATE` the engram *gather*
            # still runs on every rank — its rows are sharded world-wide and
            # rejoined by a world-wide all-reduce, so all 64 ranks have to
            # reach that collective whichever stage owns the block that
            # consumes it.
            owns = (
                self.pp_size == 1
                or self.block_stage[layer_id] == self.pp_stage
                or layer_id in self.pp_always
            )
            if block.engram is not None:
                index = list(self.config.engram_layer_ids).index(layer_id)
                rows = self._engram_rows(layer_id, hashes, index, stream.device)
                if _DUMP_DIR:
                    self._dump(f"engram{layer_id}_rows", rows.reshape(rows.shape[1], *rows.shape[2:]), live_rows)
                    self._dump(f"engram{layer_id}_hashes", hashes.reshape(hashes.shape[1], *hashes.shape[2:]), live_rows)
                if owns:
                    updated = block.engram(
                        stream[:bucket_rows].unsqueeze(0), rows, None
                    ).squeeze(0)
                    if bucket_rows < stream.shape[0]:
                        stream = _compiled(_splice_rows)(
                            updated, stream[bucket_rows:]
                        )
                    else:
                        stream = updated
            # Tapped on every rank, owner or not. `_draft` raises when
            # `main_hidden` is None and it runs on all 64 ranks, so a tap that
            # only the owning stage collected would send the other stages down
            # a different path through the MTP stages' own collectives — the
            # one thing 64 ranks must not do. A non-owner's tap is of a stale
            # stream, which costs one cheap compiled call and no collective,
            # and the emulation's tokens are wrong anyway.
            if layer_id in target_layers:
                main_hiddens.append(_compiled(_dspark_tap)(stream))
            if owns:
                if _DUMP_DIR and block.engram is not None:
                    self._dump(f"engram{layer_id}", stream, live_rows)
                stream, pre_mix = self._run_block(
                    block, stream, pre_mix, slices, paged, shared, bucket_rows
                )
                if _DUMP_DIR:
                    self._dump(f"layer{layer_id}", stream, live_rows)
                    self._dump(f"premix{layer_id}", pre_mix, live_rows)
            if self.pp_size > 1 and (
                layer_id + 1 == self.config.num_hidden_layers
                or self.block_stage[layer_id + 1] != self.block_stage[layer_id]
            ):
                stream, pre_mix = self._pp_boundary(
                    stream, pre_mix, self.block_stage[layer_id]
                )
        _MOE_LIVE_ROWS[0] = None

        greedy_ids = getattr(self._mk_step, "greedy_ids", None) if _MEGAKERNEL else None
        if greedy_ids is not None:
            sampled = greedy_ids.reshape(-1)
            if sampling_positions is not None:
                sampled = _compiled(_select_rows)(sampled, sampling_positions)
            return self._finish_spec_step(
                sampled, spec_decode_metadata, spans, paged, shared,
                _compiled(_concat_last)(main_hiddens) if main_hiddens else None,
                sampling_positions is not None, live_lengths,
            )

        collapsed = HyperConnections.pre(stream.unsqueeze(0), pre_mix.unsqueeze(0))
        collapsed = self.norm(collapsed).squeeze(0)
        if mk_model.PROF is not None:
            mk_model.PROF.mark("collapsed", collapsed)
        if _DUMP_DIR:
            self._dump("final_normed", collapsed, live_rows)
            self._dump("input_ids", input_ids, live_rows)

        selected = collapsed
        if sampling_positions is not None:
            selected = _compiled(_select_rows)(collapsed, sampling_positions)
        logits_shard = self.logits_shard(selected)
        if mk_model.PROF is not None:
            mk_model.PROF.mark("lshard", logits_shard)
        # A speculative step returns the rejection sampler's tokens, never the gathered
        # logits (`[T, vocab]` fp32 across the world): only gather what is read.
        logits = (self.gather_logits(logits_shard)
                  if spec_decode_metadata is None or self.sampler is None or _TRACE_INPUTS else None)
        if mk_model.PROF is not None:
            mk_model.PROF.mark("head", logits_shard)

        main_hidden = (
            _compiled(_concat_last)(main_hiddens) if main_hiddens else None
        )

        if self.sampler is None:
            self.last_draft = self._draft(
                _compiled(_greedy_rows)(logits), None, spans, paged,
                shared, main_hidden, sampling_positions is not None,
                live_lengths,
            )
            return logits

        if logit_mask is not None and self.vocab_pad:
            full = self.world_size * self.part_vocab_size
            if logit_mask.shape[-1] < full:
                pad_shape = list(logit_mask.shape)
                pad_shape[-1] = full - logit_mask.shape[-1]
                logit_mask = torch.cat(
                    [logit_mask, logit_mask.new_zeros(pad_shape)], dim=-1
                )
        # `sampling_hint` = (all_greedy, no_top_k, no_top_p), host booleans from the
        # runner's sampling metadata (`NeuronModelRunner`): temperature-sampled rows take
        # the full-vocabulary sampler, an all-greedy step keeps the original graph.
        if sampling_params is not None and sampling_hint is not None and not sampling_hint[0]:
            filter_topk = not (sampling_hint[1] and sampling_hint[2])
            seeds = self._sample_seeds(int(logits_shard.shape[0]), logits_shard.device)
            sampled = self._random_sampler_entry(filter_topk)(logits_shard, sampling_params, logit_mask, seeds)
        else:
            sampled = self._sampler_entry()(logits_shard, sampling_params, logit_mask, rank)
        if _TRACE_INPUTS:
            self._trace_sampler(logits_shard, logits, sampled, rank, sampling_params)
        accepted = None
        if spec_decode_metadata is not None:
            accepted = _compiled(_rejection_sample)(
                sampled,
                spec_decode_metadata.draft_token_ids,
                spec_decode_metadata.cu_num_draft_tokens,
                max(spec_decode_metadata.num_draft_tokens),
            )

        if mk_model.PROF is not None:
            mk_model.PROF.mark("sampled", sampled if accepted is None else accepted)
        self.last_draft = self._draft(
            sampled, accepted, spans, paged, shared, main_hidden,
            sampling_positions is not None, live_lengths,
        )
        if accepted is not None:
            return accepted
        return sampled, logits


    def _finish_spec_step(self, sampled, spec_decode_metadata, spans, paged, shared, main_hidden,
                          gathered, live_lengths):
        """A speculative step's end from its target tokens: verify the drafts, draft the
        next block, return the accepted tokens (`forward`'s own tail, after sampling)."""
        accepted = _compiled(_rejection_sample)(
            sampled,
            spec_decode_metadata.draft_token_ids,
            spec_decode_metadata.cu_num_draft_tokens,
            max(spec_decode_metadata.num_draft_tokens),
        )
        if mk_model.PROF is not None:
            mk_model.PROF.mark("sampled", accepted)
        self.last_draft = self._draft(
            sampled, accepted, spans, paged, shared, main_hidden, gathered, live_lengths,
        )
        return accepted

    @torch._dynamo.disable
    def _engram_rows(
        self, layer_id: int, hashes: torch.Tensor, index: int, device
    ) -> torch.Tensor:
        """This layer's n-gram rows for every token, `[1, T, 24, 256]` bf16.

            RuntimeError: Expected self.is_contiguous() to be true, but got
            false
              ... weight_loader.py:308, in rows
                  flat_indices = indices.reshape(-1).to(torch.int64)
        """
        engram = self.blocks_by_id[layer_id].engram
        if engram is not None and engram.resident:
            rows = _compiled(_engram_gather)(
                hashes, engram.table, index, engram.row_start, engram.row_width
            )
            if self.world_size > 1:
                rows = tp_all_reduce(rows, self.tp_group)
        else:
            ids = hashes.cpu()[:, :, index, :].contiguous()
            rows = self.weight_store.engram(layer_id, ids)
        key = (tuple(rows.shape), rows.dtype, str(device))
        buf = self._engram_row_slots.get(key)
        if buf is None:
            buf = torch.empty(rows.shape, dtype=rows.dtype, device=device)
            self._engram_row_slots[key] = buf
        buf.copy_(rows)
        return buf

    def _attn_meta(self, attn_metadata: object | None) -> dict:
        """The one metadata dict every layer in the KV group shares."""
        if attn_metadata is None:
            raise RuntimeError(
                "forward() needs attn_metadata: the sliding window, the "
                "compressed KV and the index keys all live in vLLM's paged "
                "allocation and are addressed through its slot_mapping and "
                "block_table (see PagedSpan)."
            )
        # The first block this rank holds (block 0 on the flat scheme and the first
        # pipeline stage; the stage's first block elsewhere): every block's entry is
        # the same dict.
        name = f"layers.{self.block_ids[0]}.attn"
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
        self,
        spans: list[tuple[int, int, int]],
        meta: dict,
        live_spans: int | None = None,
    ) -> list[PagedSpan]:
        """One `PagedSpan` per scheduled sequence, plus any the bucket added.

            scatter/gather (indirect memory copy via scalar DGE)
            out-of-bound access ... engine=GPSIMD
            Source location: model.py:2921
        """
        if live_spans is None:
            live_spans = len(spans)
        block_size = int(meta["block_size"])
        block_table = meta.get("full_block_table_tensor")
        if block_table is None:
            block_table = meta["block_table_tensor"]
        # vLLM leaves **negative** entries in the rows of `block_table` it has
        # not allocated, and `PagedSpan.slots` turns one of those into
        # `-1 * block_size + pos % block_size` — a negative cache row. Every
        # gather that addresses the paged allocation then reads out of bounds:
        #
        #   TDRV:exec_process_custom_notification  nd13:nc4: Received
        #   notification generated at runtime: scatter/gather (indirect memory
        #   copy via vector DGE) out-of-bound access
        #
        # Measured 2026-10-18 at `max_num_seqs=10`: 142 distinct graphs emitting
        # that, ~89 a second, and the engine stopped making progress with
        # `Running: 10 reqs`. `_bucket_rows` below already does exactly this
        # clamp, but only on the path it takes — which bails out at
        # `len(spans) != 1`, so one sequence was defended and ten were not.
        #
        # A no-op wherever no entry is negative, which is every configuration
        # that passes today. Via the host rather than `clamp_min(0)` on the
        # device, for the reason `_bucket_rows` spells the same way: this runs
        # outside every compiled region, and the payload is a few kB.
        host_table = block_table.cpu().clamp_min(0)
        block_table = host_table.to(block_table.device)
        slot_mapping = meta["slot_mapping"]
        host_slots = slot_mapping.cpu()
        device = slot_mapping.device
        starts = torch.tensor(
            [start_pos for _, _, start_pos in spans], dtype=torch.int64
        ).to(device)
        # A span the bucket added rather than the scheduler (see
        # `_bucket_rows`) gets an **all-zero** block table, so every row it
        # reads and every row it writes resolves into block 0 — the one
        # `vllm/v1/core/block_pool.py`'s `BlockPool.__init__` pops out of the
        # free queue before any request can be served, so no sequence owns it.
        #
        # This is not belt-and-braces. The block-table rows above the live
        # request count are not cleared by vLLM — `condense()` moves a row
        # without clearing its source — so they can hold a *stale duplicate*
        # of a live request's table. A padded span carrying that table would
        # send `_compress_kv`'s `paged.put` into a live sequence's compressed
        # pages, which is a silent corruption of another sequence's KV rather
        # than an error. Zeros cannot alias anything.
        #
        # Built only when the bucket actually added a span, so the path every
        # batch-1 forward takes does not gain a device op it has no use for.
        blank = (
            torch.zeros_like(block_table[0])
            if live_spans < len(spans)
            else None
        )
        host_blank = torch.zeros_like(host_table[0]) if blank is not None else None
        return [
            PagedSpan(
                block_table=block_table[row] if row < live_spans else blank,
                block_size=block_size,
                slot_mapping=slot_mapping[begin : begin + length],
                start_pos=start_pos,
                length=length,
                pos=starts[row : row + 1],
                begin=begin,
                host_table=host_table[row] if row < live_spans else host_blank,
                host_slots=host_slots[begin : begin + length],
            )
            for row, (begin, length, start_pos) in enumerate(spans)
        ]

    @torch._dynamo.disable
    def _bucket_rows(
        self,
        spans: list[tuple[int, int, int]],
        meta: dict,
        total_rows: int,
        rows_per_seq: int | None = None,
    ) -> tuple[list[tuple[int, int, int]], dict]:
        """Round a lone sequence's span up to the whole token bucket, above one
        rank, so a short prefill chunk reuses the graphs a full one built.

            scatter/gather (indirect memory copy via vector DGE)
            out-of-bound access ... engine=GPSIMD
            RuntimeError: nrta status=1006
        """
        if self.world_size <= 1 or total_rows <= 0:
            return spans, meta
        # A decode step pads with whole `rows_per_seq` spans even when only one sequence is
        # left in a multi-sequence bucket. Rounding that lone 6-row span up to the bucket
        # (60 rows at `max_num_seqs=10`) as a prefill chunk would run it through the prefill
        # graphs and give its 54 padded rows positions `start_pos + 6 ..`, past the
        # sequence's end: within 60 positions of `max_model_len` the n-gram hash then
        # indexed past its block table (IndexError in `NgramHashState.forward`, measured
        # 2026-10-27 at 1024 + 240 tokens, context 1280).
        if len(spans) > 1 or rows_per_seq:
            return self._bucket_spans(spans, meta, total_rows, rows_per_seq)
        if len(spans) != 1:
            return spans, meta
        begin, length, start_pos = spans[0]
        if begin != 0 or length >= total_rows:
            return spans, meta
        flat = meta["slot_mapping"].reshape(-1)
        slots = flat[:length].tolist()
        slots.extend([0] * (total_rows - length))
        bucket = dict(meta)
        bucket["slot_mapping"] = torch.tensor(slots, dtype=flat.dtype).to(flat.device)
        for key in ("full_block_table_tensor", "block_table_tensor"):
            table = meta.get(key)
            if table is None:
                continue
            host = table.cpu().clamp_min(0)
            bucket[key] = host.to(table.device)
        return [(0, total_rows, start_pos)], bucket

    @torch._dynamo.disable
    def _bucket_spans(
        self,
        spans: list[tuple[int, int, int]],
        meta: dict,
        total_rows: int,
        rows_per_seq: int | None,
    ) -> tuple[list[tuple[int, int, int]], dict]:
        """Keep the padded spans the runner's `num_seqs` bucket already built,
        instead of letting the live sequence count decide the graph.

        **This is the compile-volume fix, measured 2026-10-20.** A graph here
        is keyed by the span count, 40 layers deep, so a decode step with 32
        live sequences, one with 30 after two retire, one with 29 — each is a
        structurally different graph. One `in1024_bs32` cell spent 12 minutes
        compiling its first 32-span step and then went back to compiling the
        moment `Running` fell to 30; it answered 2 of 32 requests in 23
        minutes, all of it `Compiling...`, with the engine reporting
        `generation throughput: 0.0 tokens/s` in every 10-second window but
        two. Iteration 8 measured the same walk from the other side: ~1600
        compile-cache misses for one batch-10 cell, and 57 distinct new keys
        still appearing on the third run of the same shape.

        `NeuronModelRunner` already rounds the decode batch up to a compiled
        `num_seqs` bucket (`get_decode_padded_batch_size`, default powers of
        two up to `max_num_seqs`) and already writes `NULL_BLOCK_ID` into the
        padded slots (`_apply_decode_slot_mapping_padding`). `_sequence_spans`
        then trims exactly that padding back off, which is what lets the live
        count reach the graph. Keeping it collapses the decode graph set to one
        per bucket: at `max_num_seqs=32` the counts 17..32 all run the 32-span
        graph.

        The padded rows cost almost nothing beyond the attention they go
        through: `_MOE_LIVE_ROWS` already restricts expert selection to the
        live rows (set in `forward` from `live_rows != bucket_rows`), so the
        MoE does not route them, and `_paged_spans` points them at block 0.
        """
        if not rows_per_seq:
            return spans, meta
        live = sum(length for _, length, _ in spans)
        if live != len(spans) * rows_per_seq or live >= total_rows:
            return spans, meta
        if (total_rows - live) % rows_per_seq:
            return spans, meta
        # The padded spans borrow the first live span's `start_pos`, so their
        # `read_lens` and `at_zero` match a real span's exactly — which is what
        # keeps them on the same graph rather than keying another one, and what
        # lets `_spans_uniform` see one uniform batch.
        start_pos = spans[0][2]
        padded = list(spans)
        padded.extend(
            (off, rows_per_seq, start_pos)
            for off in range(live, total_rows, rows_per_seq)
        )
        return padded, meta

    @torch._dynamo.disable
    def _decode_rows_per_seq(self, meta: dict, total_rows: int) -> int | None:
        """Rows one sequence contributes to a decode step, or None if this step
        is not a decode step with several sequences in it.

        `NeuronScheduler` never mixes a prefill chunk with decodes — it holds
        the decoding requests back and raises if both appear in one batch
        (`vllm/core/scheduler.py`'s "Neuron constraint violated: Mixed
        prefill ... and decode") — so a step with more than one request in its
        block table and a row count divisible by that request count is a decode
        step, and every request in it contributed the same `1 + k` rows.

        The `1 + dspark_block_size` ceiling is what keeps a *prefill* step out:
        a prefill chunk is 512 or 1024 rows for the one request the scheduler
        allows, so an accidental divisibility cannot be mistaken for a decode
        shape.
        """
        table = meta.get("full_block_table_tensor")
        if table is None:
            table = meta.get("block_table_tensor")
        if table is None or table.ndim < 1:
            return None
        reqs = int(table.shape[0])
        if reqs < 2 or total_rows <= 0 or total_rows % reqs:
            return None
        rows = total_rows // reqs
        if rows > 1 + int(self.config.dspark_block_size):
            return None
        return rows

    @torch._dynamo.disable
    def _dump(self, name: str, tensor: torch.Tensor, live_rows: int) -> None:
        """`DEEPSEEK_V41_DUMP_DIR`: save the live rows of `tensor` from the first rank of
        each pipeline stage (rank 0 on the flat scheme), one file per forward and name,
        for comparison against a reference. `forward` counts the steps."""
        plan = getattr(self, "mk_plan", None)
        if (plan.stage_rank if plan is not None else self.rank) != 0:
            return
        step = _DUMP_STEP[0]
        os.makedirs(_DUMP_DIR, exist_ok=True)
        torch.save(tensor[:live_rows].detach().cpu(), os.path.join(_DUMP_DIR, f"step{step}_{name}.pt"))

    def _random_sampler_entry(self, filter_topk: bool):
        """The sampler for steps with temperature-sampled rows, as one compiled graph:
        the shards' logits gathered to the full vocabulary, then `_sample_full_vocab`.

        The plugin's vocab-parallel sampler draws from rank 0's own shard only:
        measured 2026-10-25, every one of 1280 tokens sampled at temperature 1.0 had
        an id below 2048 (rank 0's 2048-column shard), at `max_num_seqs` 1 and 10
        alike, and the text came out letter by letter ("S h a k e s p e a r e").
        Greedy steps never come here (`forward`'s `sampling_hint`), so
        their graph and cost are unchanged.
        """
        key = "_random_sampler_fn_topk" if filter_topk else "_random_sampler_fn"
        fn = self.__dict__.get(key)
        if fn is None:
            group_name = _group_name(self.tp_group)

            def sample(logits_shard, sampling_params, logit_mask, seeds):
                full = _fc_gather_last(logits_shard, group_name)
                if logit_mask is not None:
                    full = full.masked_fill(~logit_mask.to(torch.bool), float("-inf"))
                return _sample_full_vocab(full, sampling_params, seeds, filter_topk)

            fn = self.__dict__[key] = _compile_fn(sample)
        return fn

    @torch._dynamo.disable
    def _sample_seeds(self, rows: int, device) -> torch.Tensor:
        """`[rows]` int32 seeds for the random path, from a host generator every rank
        seeds and advances identically (all ranks run the same forwards)."""
        gen = self.__dict__.get("_sample_generator")
        if gen is None:
            gen = self.__dict__["_sample_generator"] = torch.Generator().manual_seed(0x5EED)
        return torch.randint(0, 2**31 - 1, (rows,), generator=gen, dtype=torch.int32).to(device)

    def _sampler_entry(self):
        """The plugin sampler plus the greedy override, as **one** compiled graph
        with exactly the sampler's inputs.

        The plugin's distributed sampler returned rank 0's local argmax instead
        of the global one on this host (measured 2026-10-22: the gathered top
        was token 2581 at 33.75, the sampler chose 671 at 24.0 on every rank),
        so greedy rows take the argmax of the gathered logits instead, from the
        shard, inside the same graph: anything handed across a graph boundary
        here (a narrowed `logits`, a parameter column) tripped the native
        backend's `Expected self.is_contiguous()` input check.
        """
        fn = self.__dict__.get("_sampler_entry_fn")
        if fn is None:
            sampler, group_name = self.sampler, _group_name(self.tp_group)

            def sample(logits_shard, sampling_params, logit_mask, rank):
                sampled = sampler.forward(logits_shard, sampling_params, logit_mask, rank)
                if sampling_params is None:
                    return sampled
                greedy_ids = _fc_gather_last(logits_shard, group_name).argmax(dim=-1)
                return _greedy_override(sampled, greedy_ids, sampling_params)

            fn = self.__dict__["_sampler_entry_fn"] = _compile_fn(sample)
        return fn

    @torch._dynamo.disable
    def _trace_sampler(self, logits_shard, logits, sampled, rank, sampling_params) -> None:
        """`DEEPSEEK_V41_TRACE_INPUTS=1`: what the sampler saw and chose, per rank."""
        ls = logits_shard.detach().float().cpu()
        lv, li = ls[-1].topk(3)
        msg = (f"deepseek_v41 SAMPLER rank={self.rank} part_vocab={self.part_vocab_size} pad={self.vocab_pad} "
               f"rank_arg={None if rank is None else rank.detach().cpu().reshape(-1).tolist()} "
               f"local_top3={[(int(i) + self.rank * self.part_vocab_size, round(float(v), 2)) for v, i in zip(lv, li)]} "
               f"sampled={sampled.detach().cpu().reshape(-1).tolist()[:8]}")
        if self.rank == 0:
            lg = logits.detach().float().cpu()
            gv, gi = lg[-1].topk(5)
            msg += (f" | gathered_top5={[(int(i), round(float(v), 2)) for v, i in zip(gv, gi)]} "
                    f"gathered_shape={tuple(lg.shape)} params={None if sampling_params is None else sampling_params.detach().cpu().reshape(-1).tolist()[:3]}")
        # eager probes of the plugin's distributed top-k (collective: every rank runs them)
        try:
            from vllm_neuron.functional.topk import topk as plugin_topk
            from torch.distributed._functional_collectives import all_gather_tensor
            grp = self.tp_group.device_group
            # Compiled probes (collectives only run inside compiled graphs here):
            # the same all-gather with the ProcessGroup object the plugin's
            # sampler passes, and with the group *name* this port passes.
            row = logits_shard.detach()[-1:].contiguous()           # fp32 already
            lv, li = row.topk(256, dim=-1)
            lv, li = lv.contiguous(), (li + self.rank * self.part_vocab_size).contiguous()
            name = _group_name(self.tp_group)
            g_obj = _compiled(lambda v: all_gather_tensor(v, 1, group=grp))(lv)
            g_name = _compiled(lambda v: all_gather_tensor(v, 1, group=name))(lv)
            gi_name = _compiled(lambda v: all_gather_tensor(v.to(torch.float32), 1, group=name))(li)
            v_e, i_e = _compiled(lambda r: plugin_topk(r, 256, -1, -1, process_group=grp, rank=rank))(row)
            h_obj, h_name, h_gi = g_obj.cpu().reshape(-1), g_name.cpu().reshape(-1), gi_name.cpu().reshape(-1)
            v_h, i_h = v_e.cpu().reshape(-1), i_e.cpu().reshape(-1)
            if self.rank == 0:
                chunks_obj = int((h_obj.reshape(-1, 256).abs().sum(-1) > 0).sum()); chunks_name = int((h_name.reshape(-1, 256).abs().sum(-1) > 0).sum())
                j = int(h_name.argmax())
                msg += (f" | compiled all_gather(group=object): width {h_obj.numel()} nonzero_chunks {chunks_obj} max {float(h_obj.max()):.2f}"
                        f" | all_gather(group=name): width {h_name.numel()} nonzero_chunks {chunks_name} max {float(h_name.max()):.2f} at idx {int(h_gi[j])}"
                        f" | compiled plugin_topk top3={[(int(i), round(float(v), 2)) for v, i in zip(v_h[:3], i_h[:3])]}")
        except Exception as e:  # noqa: BLE001 - diagnostics only
            msg += f" | eager probe failed: {type(e).__name__}: {str(e)[:160]}"
        logger.warning(msg)

    @torch._dynamo.disable
    def _trace_inputs(self, input_ids, positions, meta, live_spans, spans) -> None:
        """`DEEPSEEK_V41_TRACE_INPUTS=1`: log what one forward was handed, on rank 0."""
        if self.rank != 0:
            return
        ids = input_ids.reshape(-1).tolist()
        pos = positions.reshape(-1).tolist()
        slots = meta["slot_mapping"].reshape(-1).tolist()
        table = meta.get("full_block_table_tensor")
        if table is None:
            table = meta.get("block_table_tensor")
        row0 = table[0].tolist()[:12] if table is not None and table.ndim == 2 else None
        _TRACE_STEP[0] += 1
        logger.warning(
            "deepseek_v41 TRACE step=%d rows=%d live_spans=%s spans=%s "
            "ids[:8]=%s ids[-8:]=%s pos[:4]=%s pos[-6:]=%s slots[:4]=%s slots[-6:]=%s "
            "block_size=%s table[0][:12]=%s cached_seq_len=%s kv_segment_size=%s",
            _TRACE_STEP[0], len(ids), live_spans, spans, ids[:8], ids[-8:],
            pos[:4], pos[-6:], slots[:4], slots[-6:], meta.get("block_size"), row0,
            (meta.get("cached_seq_len").reshape(-1).tolist()
             if meta.get("cached_seq_len") is not None else None),
            meta.get("kv_segment_size"),
        )

    @torch._dynamo.disable
    def _sequence_spans(
        self,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        rows_per_seq: int | None = None,
    ) -> list[tuple[int, int, int]]:
        """Split the flat token axis into `(start_index, length, start_pos)`.

            IndexError: index 1 is out of bounds for dimension 0 with size 1

            torch._dynamo.exc.UserError: Could not guard on data-dependent
            expression Ne(u1, u0 + 1)
        """
        slots = slot_mapping.reshape(-1).tolist()
        if not slots:
            return []
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
        if not rows_per_seq:
            return spans
        # Two sequences whose next positions happen to be adjacent — sequence
        # j at `[p, p+k]` and sequence j+1 starting at `p+k+1` — have one
        # unbroken run of positions between them, so the loop above merges
        # them into a single span. `_paged_spans` then indexes `block_table`
        # by span ordinal, so the second sequence's rows would be addressed
        # through the first sequence's block table: its KV write would land in
        # another sequence's pages and its queries would attend to that
        # sequence's history. `_draft` would also emit one row fewer than
        # there are requests, which is the mismatch
        # `NeuronModelRunner._sanitize_sync_drafts` was built to paper over.
        #
        # A decode step gives every request exactly `rows_per_seq` rows, so a
        # longer span is a merge and nothing else, and splitting it back is
        # exact: the positions inside it are consecutive by construction, so
        # the j-th piece starts at `start_pos + j * rows_per_seq`.
        split: list[tuple[int, int, int]] = []
        for begin, length, start_pos in spans:
            if length <= rows_per_seq or length % rows_per_seq:
                split.append((begin, length, start_pos))
                continue
            for off in range(0, length, rows_per_seq):
                split.append((begin + off, rows_per_seq, start_pos + off))
        return split

    @torch._dynamo.disable
    def _engram_hashes(
        self, input_ids: torch.Tensor, paged: list[PagedSpan],
        ids_host: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """n-gram row indices for every token, `[1, T, n_engram_layers, 24]`: on the
        host for the megakernel path (`StepIndices.hash_cols` cuts each layer's columns
        from it), on the device otherwise. One pass over every span
        (`NgramHashState.hash_spans`) from the step's host token ids `ids_host` (read
        back here when not given); one `forward` a span read each span's ids, table and
        slots back from the device and concatenated the spans' rows on it."""
        if self.ngram_hash is None:
            raise RuntimeError(
                "engram layers "
                f"{list(self.config.engram_layer_ids)} need the n-gram hash "
                "state; bind_kv_cache builds it from the tokenizer."
            )
        if ids_host is None:
            ids_host = mk_model.host_ids(input_ids)
        hashes = self.ngram_hash.hash_spans(ids_host, paged)
        return hashes if _MEGAKERNEL else hashes.to(input_ids.device)

    @torch._dynamo.disable
    def _layer_io(
        self, stream: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The three per-layer buffers every block shares, for this shape."""
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
                "deepseek_v41: allocated the shared per-layer I/O "
                "buffers for %d rows (%.1f MiB, one set for all %d blocks)",
                rows,
                sum(t.numel() * t.element_size() for t in slots) / 1024**2,
                len(self.block_ids),
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
        row_count: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One layer over a flat `[T, hc, dim]` stream covering several
        sequences.
        """
        if _MEGAKERNEL:
            return mk_model.run_layer(self, block, stream, pre_mix, self._mk_step)
        # `stage_group` *is* `tp_group` unless `_PP_EMULATE` narrowed it. The
        # attention and the indexer are unaffected either way: both reduce over
        # `self.attn_group or tp_group` and `attn_group` is always set here, so
        # what this selects is the expert-parallel all-reduce inside `MoE`.
        tp_group = self.stage_group if self.world_size > 1 else None
        io_stream, io_residual, io_normed = self._layer_io(stream)
        normed, residual, ffn_pre, ffn_post, ffn_comb = _block_entry(
            block, "attn_half"
        )(stream, pre_mix, slices, paged, shared, tp_group, io_residual, io_normed)
        ffn_out = block.ffn(normed, None, self.weight_store, tp_group, row_count)
        if _DUMP_DIR:
            self._dump(f"normed{block.attn.layer_id}", normed.reshape(-1, normed.shape[-1]), _DUMP_LIVE[0])
            self._dump(f"ffn_out{block.attn.layer_id}", ffn_out.reshape(-1, ffn_out.shape[-1]), _DUMP_LIVE[0])
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
        live_lengths: list[int] | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """One DSpark draft per scheduled sequence, seeded from this step's own
        committed token.

            RuntimeError: Expected self.dtype() == dst.dtype() to be true
        """
        device = sampled.device
        if live_lengths is None:
            live_lengths = [length for _, length, _ in spans]
        lengths = torch.tensor(live_lengths, dtype=torch.int64, device=device)
        ones = torch.ones(len(spans), dtype=torch.int64, device=device)
        if accepted is not None:
            seeds, valid = _compiled(_accepted_tail)(accepted)
            seeds, valid = seeds[: len(spans)], valid[: len(spans)]
            rows_per_seq = int(accepted.shape[1])
        else:
            if gathered:
                picks = torch.arange(len(spans), dtype=torch.int64, device=device)
            else:
                picks = torch.tensor(
                    [
                        begin + live - 1
                        for (begin, _, _), live in zip(spans, live_lengths)
                    ],
                    dtype=torch.int64,
                    device=device,
                )
            seeds = _compiled(_select_rows)(sampled.reshape(-1), picks)
            valid, rows_per_seq = ones, 1
        taps = _compiled(_tap_rows)(valid, lengths, rows_per_seq)
        if _MEGAKERNEL:
            # One launch drafts every sequence of the step (`mk_model.run_draft`).
            if main_hidden is None:
                raise RuntimeError(
                    "the DSpark stages read the attention inputs of layers "
                    f"{list(self.config.dspark_target_layer_ids)}; forward() "
                    "collected none, so the target-layer tap did not run"
                )
            return mk_model.run_draft(self, seeds[: len(spans)], taps[: len(spans)], main_hidden, spans, paged)
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
        """The DSpark draft path over the `mtp.*` stages, for one sequence."""
        if main_hidden is None:
            raise RuntimeError(
                "the DSpark stages read the attention inputs of layers "
                f"{list(self.config.dspark_target_layer_ids)}; forward() "
                "collected none, so the target-layer tap did not run"
            )
        if _MEGAKERNEL:
            return mk_model.run_draft(self, seed_ids, tap, main_hidden, [span], [paged])[0]
        begin, length, _ = span
        main = main_hidden[begin : begin + length].unsqueeze(0)

        stage0 = self.mtp[0]
        stage0.embed_fn = self.embed_input_ids
        stream, main_x = stage0.forward_embed(main, seed_ids)
        pre_mix = _compiled(_one_hot_pre_mix)(stream, self.hc_mult)

        tp_group = self.tp_group if self.world_size > 1 else None
        draft_pos = paged.pos + tap
        split = self.world_size > 1
        for stage in self.mtp:
            args = (
                stream, paged.pos, draft_pos, pre_mix, main_x, shared, paged,
                self.weight_store, tp_group,
            )
            if split:
                _graph_break()
            stream, pre_mix = stage(*args)

        last = self.mtp[-1]
        last.head_fn = self.compute_logits
        last.markov_head._tp_group = tp_group
        if split:
            _graph_break()
        return last.forward_head(stream, pre_mix, seed_ids)


    # ---- megakernel bridge (see mk_model) ---------------------------------------
    _mk_step = None
    _mk_dummies_ = None
    _mk_layers_: dict | None = None
    _mk_freqs_: dict | None = None
    _mk_draft_ = None
    _mk_groups_attn_ = None

    @property
    def device(self) -> torch.device:
        """The device this rank's weights live on (after `load_weights`); every stage
        has `vocab_pad_bias`, which `allocate_staging_buffers` materializes."""
        bias = self.vocab_pad_bias
        if bias is None:
            raise RuntimeError("the model is still on the meta device: load_weights() first")
        return bias.device

    @property
    def _mk_dummies(self):
        if self._mk_dummies_ is None:
            self._mk_dummies_ = mk_model.Dummies.on(self.device)
        return self._mk_dummies_

    def _mk_layer(self, block) -> "mk_model.LayerWeights":
        if self._mk_layers_ is None:
            self._mk_layers_ = {}
        lw = self._mk_layers_.get(block.layer_id)
        if lw is None:
            lw = mk_model.layer_weights(self, block)
            self._mk_layers_[block.layer_id] = lw
        return lw

    def _mk_draft(self):
        if self._mk_draft_ is None:
            self._mk_draft_ = mk_model.draft_weights(self)
        return self._mk_draft_

    def _mk_freqs(self, attn) -> torch.Tensor:
        """`[max_pos, 64]` fp32 (cos | sin) of this attention's RoPE table; the
        compressed layers share one table and the others another."""
        if self._mk_freqs_ is None:
            self._mk_freqs_ = {}
        key = bool(attn.compress_ratio)
        t = self._mk_freqs_.get(key)
        if t is None:
            if attn.freqs_cis is None:
                raise RuntimeError("bind_kv_cache() must run before the megakernel path")
            t = mk_model._freqs_table(attn.freqs_cis, self.device)
            self._mk_freqs_[key] = t
        return t

    def _mk_nearest_source(self, layer_id: int, kind: str):
        """The attention of the latest kv ("kv") or index ("index") source at or before
        `layer_id` (the first one when none precedes it). On a pipeline stage that does
        not own it, the stage's replica of the shipped kv source stands in: it has the
        caches the boundary fills and indexer / compressor modules of the right shapes
        (every kv source is an index source too)."""
        ids = list(self.config.kv_source_layer_ids if kind == "kv" else self.config.index_source_layer_ids)
        before = [s for s in ids if s <= layer_id]
        sid = max(before) if before else min(ids)
        block = self.blocks_by_id.get(sid)
        if block is not None:
            return block.attn
        if self.src_attn:
            return next(iter(self.src_attn.values()))
        raise RuntimeError(
            f"layer {layer_id} needs {kind} source {sid}, which this stage neither owns nor "
            "received a replica of"
        )

    @property
    def _mk_groups_world(self):
        """The expert-reduce group of a backbone block: the world under tp64, the
        stage under tp8pp8."""
        return self.mk_plan.groups_stage()

    @property
    def _mk_groups_full(self):
        """The drafter's group: the whole world under tp64, the (last) stage under
        tp8pp8 -- the same ranks as `_mk_groups_world`."""
        return self.mk_plan.groups_stage()

    @property
    def _mk_groups_attn(self):
        """Consecutive `attn_tp`-rank groups, checked against the group the
        plugin gave the attention reduction."""
        if self._mk_groups_attn_ is None:
            if self.attn_group_is_world:
                if self.attn_replicas > 1:
                    raise NotImplementedError(
                        "the attention is replicated but reduces over the world; the "
                        "megakernel wants an attention group of attn_tp ranks "
                        "(see deepseek_v41.require_attention_group)"
                    )
                self._mk_groups_attn_ = self._mk_groups_world
            else:
                groups = mk_host.replica_groups(self.world_size, self.attn_tp)
                ranks = tuple(int(r) for r in self.attn_tp_group.ranks)
                mine = groups[self.rank // self.attn_tp]
                if ranks != mine:
                    raise NotImplementedError(
                        f"rank {self.rank}: the attention group is {ranks}, not the consecutive "
                        f"{mine} the megakernel's replica groups assume"
                    )
                self._mk_groups_attn_ = groups
        return self._mk_groups_attn_

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

            Neuron HBM: 10.60 GiB used, 13.40 GiB free
            GPU KV cache size: 76,250 tokens
            Maximum concurrency for 544 tokens per request: 140.17x

            RuntimeError: Failed to schedule neff execution. status=4
            message=Allocation Failure
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
                "deepseek_v41: no active vLLM config, leaving the KV "
                "cache size alone (%s: %s)", type(exc).__name__, exc
            )
            return
        if cache.num_gpu_blocks_override is not None:
            return

        block = int(getattr(cache, "block_size", None) or 32)
        if not getattr(cache, "user_specified_block_size", False):
            block = 32
        per_sequence = -(-int(model.max_model_len) // block) + 1
        blocks = per_sequence * int(sched.max_num_seqs)
        if blocks * block > _KV_CAP_MAX_SLOTS:
            logger.info(
                "deepseek_v41: leaving the KV cache to vLLM — this "
                "configuration wants %d slots (%d per sequence x %d sequences "
                "x 2), which is not obviously oversized",
                blocks * block, per_sequence, sched.max_num_seqs,
            )
            return
        cache.num_gpu_blocks_override = blocks
        logger.info(
            "deepseek_v41: capping the KV cache at %d blocks of %d "
            "(%d slots) — %d per sequence x %d sequences, one full context "
            "each. The rest of HBM is needed for this port's compiled graphs "
            "and its per-slot buffers, which vLLM's own sizing cannot see; see "
            "model._cap_kv_cache_blocks",
            blocks, block, blocks * block, per_sequence, sched.max_num_seqs,
        )

    @staticmethod
    def _allow_request_time_compilation() -> None:
        """Let a request compile a graph that warmup did not build.

            RuntimeError: Detected recompile when torch.compile stance is
            'fail_on_recompile'

          * warmup is skipped (`__init__.py` says why: for this model it is a
            251 GiB disk read rather than a cheap pre-compile), and
          * **the graph set is input-dependent.** How many experts a step routes
            to, and how many rows each gets, are outputs of the router.
            `MoE._route_plan` buckets the row lists to powers of two precisely
            so that there is a handful of shapes rather than one per step, but a
            handful is not one. The only way to make it static is to run every
            expert over every token, which is 64x the arithmetic.

          1. `with torch.compiler.set_stance("default")` around the body;
          2. the same plus `@torch._dynamo.disable(recursive=False)` on
             `forward`, which only moves the report to
             `torch/_dynamo/external_utils.py:inner` — the module-call wrapper,
             which guards on input shapes regardless of what `forward` is
             marked.
        """
        from vllm_neuron.vllm.worker import neuron_model_runner as runner

        if getattr(runner, "_deepseek_v41_request_time_compilation", False):
            return
        runner.model_forward_context = lambda vllm_config: (
            torch.compiler.set_stance("default")
        )
        runner._deepseek_v41_request_time_compilation = True
        logger.info(
            "deepseek_v41: model_forward_context relaxed to the "
            "default compile stance — this port's graph set depends on the "
            "router's output, so a request may compile a shape warmup did not"
        )


def _buffer_name(checkpoint_key: str) -> str:
    """Map a checkpoint key to the buffer that holds it."""
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
