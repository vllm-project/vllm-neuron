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

Status
------
Real and exercised: the module tree with every parameter at the checkpoint's own
shape and dtype, the per-rank sharding plan (`shard_parameter`), weight loading,
`get_kv_spec`/`bind_kv_cache`, `embed_input_ids`, `compute_logits`.

Not yet: `forward`. It raises, naming what is left, rather than running a subset
of the layers or substituting an approximation — see the `NotImplementedError`
at the bottom for the remaining work and the order it has to happen in.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn
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
    """`vendor/model.py`'s `RMSNorm`, including its fp32 interior.

    `rms_norm_eps` is 1e-20 for this model — small enough that computing the
    variance in bf16 would make the epsilon a no-op and the reciprocal sqrt
    noisy, so the cast to fp32 is load-bearing rather than defensive.
    """

    def __init__(self, dim: int, eps: float, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.register_buffer("weight", torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * x).to(dtype)


#: fp8 e4m3's range, as `vendor/kernel.py`'s `act_quant_kernel` uses it. 448 is
#: the OCP e4m3 maximum; note the Trainium kernels under `kernels/` may use the
#: legacy nl.float8_e4m3 whose maximum is 240, which is what the compiler's
#: `--experimental-unsafe-fp8e4m3fn-as-fp8e4m3` flag is about.
_FP8_MAX = 448.0


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
    scale = torch.exp2(torch.ceil(torch.log2(amax / _FP8_MAX)))
    quantized = (grouped / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return (
        quantized.reshape(*lead, width),
        scale.expand(*lead, blocks, block).reshape(*lead, width),
    )


class BlockQuantLinear(nn.Module):
    """A linear whose weight is stored block-scaled, as the checkpoint has it.

    Three storage layouts appear, all with `e8m0` (power-of-two) scales:

        fp8  `[out, in]` e4m3 + `[ceil(out/32), ceil(in/32)]`   attention, shared experts
        fp4  `[out, in//2]` two e2m1 per byte + `[out, in/32]`   routed experts
        bf16 `[out, in]`, no scale                               compressor, indexer wk

    The weight is held in the stored form. On device the fp8/fp4 GEMMs in
    `kernels/` consume it directly; `forward` here dequantizes through
    `weight_loader`'s helpers, which is the same arithmetic at 4x the memory and
    is what the accuracy-debugging path compares against.
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
                ("weight", (out_features, in_features), torch.float8_e4m3fn),
                (
                    "scale",
                    (-(-out_features // block), -(-in_features // block)),
                    torch.uint8,
                ),
            ]
        elif storage == "fp4":
            specs = [
                ("weight", (out_features, in_features // 2), torch.int8),
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
        if self.storage == "bf16":
            return F.linear(x, self.weight.to(x.dtype))
        # The quantized GEMMs do not take a bf16 activation: `linear()` in
        # `vendor/model.py` runs `act_quant` first, for fp4 weights as well as
        # fp8 ones, so the activation's own rounding is part of the reference's
        # arithmetic rather than an implementation detail. Emulating it here —
        # instead of multiplying the bf16 activation by a dequantized weight —
        # is worth 1.5e-2 of max relative error on a 384-expert MoE layer.
        x_q, x_scale = act_quant(x, self.block)
        # `x_q` is fp8; torch will not promote a Float8 type, so the
        # dequantization is explicit.
        activation = x_q.to(torch.float32) * x_scale
        weight = self.dequantized(torch.float32)
        return F.linear(activation, weight).to(x.dtype)


def act_quant_dequant(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """`act_quant(..., inplace=True)`: round-trip `x` through fp8 and return it.

    The KV latent and the DSpark window KV go through this before being cached,
    so the cache holds values that have already lost fp8 precision. Skipping it
    would make attention read a *more* precise KV than the reference does.
    """
    quantized, scale = act_quant(x, block)
    return (quantized.to(torch.float32) * scale).to(x.dtype)


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
    grouped = x.detach().float().reshape(*lead, blocks, block)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    if scale_e4m3:
        amax = amax.clamp_min(6 * 2.0**-9)
        scale = (amax / _FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(6 * 2.0**-126)
        scale = torch.exp2(torch.ceil(torch.log2(amax / _FP4_MAX)))
    normalized = (grouped / scale).clamp(-_FP4_MAX, _FP4_MAX)
    rounded = _round_to_fp4(normalized)
    return (rounded * scale).reshape(*lead, width).to(x.dtype)


#: fp4 `e2m1`'s maximum magnitude.
_FP4_MAX = 6.0

#: The 8 magnitudes `e2m1` can represent, and the midpoints between them, which
#: is what round-to-nearest needs.
_FP4_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_FP4_MIDPOINTS = (_FP4_MAGNITUDES[1:] + _FP4_MAGNITUDES[:-1]) / 2


def _round_to_fp4(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest `e2m1` value, keeping the sign."""
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


#: An empty top-k slot: a window position before the sequence started, or a
#: compressed position a query cannot reach yet. `sparse_attn` skips it.
EMPTY_SLOT = -1

#: The finite lower bound the reference kernel starts its running max at. A row
#: whose every slot is empty would give exp(-inf - -inf) = NaN with a true -inf;
#: with this bound the sink term overflows to +inf and the row comes out zero,
#: which is the kernel's convention.
_NEG_BOUND = -1e30


def window_topk_idxs(
    window_size: int, bsz: int, seqlen: int, start_pos: int, device
) -> torch.Tensor:
    """Which sliding-window cache slots each query attends to; -1 is empty.

    `vendor/model.py`'s `get_window_topk_idxs`. The cache is a ring of
    `window_size` slots. Prefill needs one row per query, each seeing its own
    causal window; a decode step has one query that sees the whole ring, oldest
    first. Order within a row does not matter to `sparse_attn`, which treats
    every slot independently.
    """
    if start_pos == 0:
        end = torch.arange(seqlen, device=device).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(
            min(seqlen, window_size), device=device
        )
        idxs = torch.where(idxs > end, EMPTY_SLOT, idxs)
    else:
        oldest = start_pos % window_size + 1
        idxs = torch.cat(
            [
                torch.arange(oldest, window_size, device=device),
                torch.arange(oldest, device=device),
            ]
        )
        idxs = torch.where(idxs > start_pos, EMPTY_SLOT, idxs)
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


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
    ships as `compat/sparse_attn_sm89.py` — including where the precision
    changes, because those are part of the reference's arithmetic:

      - the QK product takes bf16 operands into an fp32 accumulator, and the
        scale multiplies the *logits* afterwards rather than `q` beforehand, so
        it never passes through a bf16 rounding;
      - the probabilities are rounded to bf16 before the PV product, matching
        the kernel's bf16 staging buffer;
      - the sink joins the denominator after the loop, against the final row
        max.
    """
    b, s, h, d = q.shape
    n = kv.size(1)
    topk = topk_idxs.size(-1)
    out = torch.empty_like(q)
    if topk == 0 or n == 0:
        return out.zero_()

    sink = attn_sink.float().view(1, 1, h)
    batch = torch.arange(b, device=kv.device).view(b, 1, 1)
    idxs = topk_idxs.long()
    empty = idxs == EMPTY_SLOT

    gathered = kv[batch, idxs.clamp_min(0)]
    gathered = gathered.masked_fill(empty.unsqueeze(-1), 0).float()

    scores = torch.einsum("bthd,btkd->bthk", q.float(), gathered) * softmax_scale
    scores = scores.masked_fill(empty.unsqueeze(2), -torch.inf)

    scores_max = scores.amax(dim=-1).clamp_min(_NEG_BOUND)
    probabilities = torch.exp(scores - scores_max.unsqueeze(-1))
    denominator = probabilities.sum(dim=-1) + torch.exp(sink - scores_max)
    accumulated = torch.einsum(
        "bthk,btkd->bthd", probabilities.to(torch.bfloat16).float(), gathered
    )
    return (accumulated / denominator.unsqueeze(-1)).to(out.dtype)


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
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last, torch.inf
    )
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class SharedAttentionState:
    """What attention layers hand down the stack instead of recomputing.

    `vendor/model.py`'s `SharedAttentionRuntime`. Layers run in order and every
    source writes before its consumers read, so one slot each is enough:
    `compress_kv` and `index_k` come from the four `kv_source_layer_ids`,
    `topk_idxs` from the eight `index_source_layer_ids`, and `candidates` from
    the single `candidate_source_layer_id`.
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
        self.compress_ratio = config.compress_ratios[layer_id]
        self.head_dim = config.head_dim
        self.norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.wkv = BlockQuantLinear(config.hidden_size, config.head_dim, storage="bf16")
        if self.compress_ratio > 1:
            self.wgate = BlockQuantLinear(
                config.hidden_size, config.head_dim, storage="bf16"
            )
            # The tail of an incomplete group, carried across decode steps.
            # Runtime state, not weights: `persistent=False` keeps them out of
            # the state dict (nothing in the checkpoint corresponds to them) and
            # `allocate_state` fills them on the device, because a module built
            # under `torch.device("meta")` cannot allocate anything here.
            self.state_shape = (max_batch_size, self.compress_ratio, config.head_dim)
            self.register_buffer("kv_state", None, persistent=False)
            self.register_buffer("score_state", None, persistent=False)

    def allocate_state(self, device) -> None:
        if self.compress_ratio > 1:
            self.kv_state = torch.zeros(
                self.state_shape, dtype=torch.float32, device=device
            )
            self.score_state = torch.full(
                self.state_shape, -torch.inf, dtype=torch.float32, device=device
            )

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor | None:
        """Pool `compress_ratio` consecutive tokens into one latent.

        Returns `None` while a group is still filling up, which during decode
        means it yields only every `compress_ratio` steps — the partial group
        waits in `kv_state`/`score_state`. Callers must treat `None` as "nothing
        to publish yet", not as "no compressed KV": the cache still holds every
        earlier group.
        """
        bsz, seqlen, _ = x.shape
        ratio, dtype = self.compress_ratio, x.dtype
        if ratio == 1:
            # One token per group: nothing to pool, so no gate and no fp32.
            return self.norm(self.wkv(x))

        x = x.float()
        kv, score = self.wkv(x), self.wgate(x)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            if remainder:
                kv, self.kv_state[:bsz, :remainder] = kv.split(
                    [cutoff, remainder], dim=1
                )
                score, self.score_state[:bsz, :remainder] = score.split(
                    [cutoff, remainder], dim=1
                )
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % ratio == 0
            slot = start_pos % ratio
            self.kv_state[:bsz, slot] = kv.squeeze(1)
            self.score_state[:bsz, slot] = score.squeeze(1)
            if should_compress:
                kv = (
                    self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)
                ).sum(dim=1, keepdim=True)
        if not should_compress:
            return None
        return self.norm(kv.to(dtype))


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
        start_pos: int,
        offset: int,
        shared: SharedAttentionState,
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
        end_pos = start_pos + seqlen

        if self.owns_k and latent is not None:
            # A latent stands for the first token of its group, so group j takes
            # position j * ratio.
            freqs = (
                slice_freqs(self.freqs_cis, slice(None, seqlen - seqlen % ratio, ratio))
                if start_pos == 0
                else slice_freqs(
                    self.freqs_cis, slice(start_pos + 1 - ratio, start_pos + 2 - ratio)
                )
            )
            k = self.k_norm(self.wk(latent))
            k = torch.cat(
                [k[..., :-rd], apply_rotary_emb(k[..., -rd:], freqs)], dim=-1
            )
            k = fp4_quant_dequant(k, 32)
            self.k_cache[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k
            shared.index_k = self.k_cache

        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim))
        q = torch.cat(
            [
                q[..., :-rd],
                apply_rotary_emb(
                    q[..., -rd:],
                    slice_freqs(self.freqs_cis, slice(start_pos, end_pos)),
                ),
            ],
            dim=-1,
        )
        q = fp4_quant_dequant(q, 32)

        index_k = shared.index_k[:bsz, : end_pos // ratio]
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        index_score = torch.einsum("bshd,btd->bsht", q.float(), index_k.float())
        index_score = (index_score.relu() * weights.float().unsqueeze(-1)).sum(dim=2)
        if self.world_size > 1 and tp_group is not None:
            # Heads are split across ranks and the score sums over heads, so the
            # partial sums have to be added before the top-k — a per-rank top-k
            # would pick from a different score.
            tp_group.all_reduce(index_score)

        # How many compressed positions a query can see: a block becomes visible
        # once the query has passed its last token.
        if start_pos == 0:
            compress_lens = (
                torch.arange(1, seqlen + 1, device=x.device) // ratio
            ).unsqueeze(-1)
            index_score = index_score.masked_fill(
                torch.arange(seqlen // ratio, device=x.device) >= compress_lens,
                -torch.inf,
            )
        else:
            compress_lens = end_pos // ratio

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
            index_score = index_score.masked_fill(~shared.candidates, -torch.inf)

        topk = min(self.index_topk, end_pos // ratio)
        idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
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
        self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """This layer's sliding-window K, and the slots every query may see.

        The K stays fp8-rounded, quantized over the whole post-RoPE vector
        including its RoPE tail — so the cache holds values that have already
        been through fp8, and attention must read those rather than the
        pre-quantization ones.
        """
        bsz, seqlen, _ = x.shape
        win = self.window_size
        rd = self.rope_head_dim
        if self.window_kv_cache is None:
            raise RuntimeError(
                f"layer {self.layer_id} attention has no window KV cache; "
                "bind_kv_cache() must run before forward()."
            )
        if self.window_kv_cache.shape[-2:] != (win, self.head_dim):
            # The known gap, made explicit so it fails here with its own name
            # rather than as a fake-tensor error from inside dynamo's warmup
            # ("expand: attempting to expand a dimension of length 128 -> 32!").
            raise NotImplementedError(
                "the sliding-window KV cache is in vLLM's *paged* layout "
                f"{tuple(self.window_kv_cache.shape)} "
                "([num_blocks, num_kv_heads, block_size, head_size]), and this "
                "attention indexes a contiguous ring "
                f"[batch, {win}, {self.head_dim}] the way the reference does "
                "(`window_kv_cache[:bsz, start_pos % window]`). Bridging the "
                "two is the last piece of the port; see "
                ".autohelix/notes/iter-1.md section 4c for the two candidate "
                "designs and why writing into the paged layout (rather than "
                "allocating a private ring) is the one that keeps prefix "
                "caching sound."
            )
        kv = self.kv_norm(self.wkv(x))
        kv = torch.cat([kv[..., :-rd], apply_rotary_emb(kv[..., -rd:], freqs_cis)], -1)
        kv = act_quant_dequant(kv, 32)

        if start_pos == 0:
            # Prefill attends over this chunk and seeds the ring for decode.
            if seqlen <= win:
                self.window_kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                tail_a, tail_b = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
                self.window_kv_cache[:bsz, cutoff:win] = tail_a
                self.window_kv_cache[:bsz, :cutoff] = tail_b
            window_kv = kv
        else:
            self.window_kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            window_kv = self.window_kv_cache[:bsz]
        return window_kv, window_topk_idxs(win, bsz, seqlen, start_pos, x.device)

    def _compress_kv(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        offset: int,
        shared: SharedAttentionState,
        tp_group,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The shared compressed KV and the compressed slots each query sees."""
        bsz, seqlen, _ = x.shape
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        compress_len = (start_pos + seqlen) // ratio

        latent = None
        if self.is_kv_source:
            latent = self.compressor(x, start_pos)
            shared.compress_kv = self.compress_kv_cache

        # The indexer needs the latent *before* RoPE, so it runs before the
        # cache write below.
        if self.is_index_source:
            if compress_len == 0:
                idxs = torch.empty(
                    bsz, seqlen, 0, dtype=torch.int32, device=x.device
                )
            else:
                idxs = self.indexer(
                    x, qr, latent, start_pos, offset, shared, tp_group
                )
            shared.topk_idxs = idxs
        else:
            idxs = shared.topk_idxs

        if latent is not None:
            freqs = (
                slice_freqs(self.freqs_cis, slice(None, seqlen - seqlen % ratio, ratio))
                if start_pos == 0
                else slice_freqs(
                    self.freqs_cis, slice(start_pos + 1 - ratio, start_pos + 2 - ratio)
                )
            )
            rotated = torch.cat(
                [latent[..., :-rd], apply_rotary_emb(latent[..., -rd:], freqs)], -1
            )
            # Compressed KV uses 16-wide groups with e4m3 scales; the indexer
            # above uses 32 with e8m0. Not interchangeable.
            rotated = fp4_quant_dequant(rotated, 16, scale_e4m3=True)
            self.compress_kv_cache[
                :bsz, start_pos // ratio : start_pos // ratio + rotated.size(1)
            ] = rotated

        # Read after the write, so this does not depend on the slice aliasing
        # the cache.
        return shared.compress_kv[:bsz, :compress_len], idxs

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        shared: SharedAttentionState,
        tp_group=None,
    ) -> torch.Tensor:
        """`x` is the normalized sublayer input `[b, s, hidden]`."""
        bsz, seqlen, _ = x.shape
        freqs_cis = slice_freqs(self.freqs_cis, slice(start_pos, start_pos + seqlen))
        rd = self.rope_head_dim

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        q = torch.cat([q[..., :-rd], apply_rotary_emb(q[..., -rd:], freqs_cis)], -1)

        kv, topk_idxs = self._window_kv(x, freqs_cis, start_pos)
        if self.compress_ratio:
            compress_kv, compress_idxs = self._compress_kv(
                x, qr, start_pos, kv.size(1), shared, tp_group
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
        # heads — hence an einsum and not a Linear. The checkpoint stores it
        # fp8; the reference runs it dequantized to bf16 with no activation
        # quantization, which is what `dequantized()` without `act_quant`
        # reproduces.
        o = o.reshape(bsz, seqlen, self.n_local_groups, -1)
        wo_a = self.wo_a.dequantized(torch.float32).view(
            self.n_local_groups, self.o_lora_rank, -1
        )
        o = torch.einsum("bsgd,grd->bsgr", o.float(), wo_a)
        out = self.wo_b(o.flatten(2).to(x.dtype))
        if self.world_size > 1 and tp_group is not None:
            # `wo_b` is row-parallel, so each rank holds a partial sum. The
            # reference reduces in fp32.
            out = out.float()
            tp_group.all_reduce(out)
            out = out.to(x.dtype)
        return out


# =============================================================================
# Section 4: the MoE FFN
# =============================================================================


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
        scores = F.linear(x.float(), self.weight.float())
        if self.scoring_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.scoring_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()

        bias = self.bias
        if image_mask is not None:
            bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)

        indices = (scores + bias).topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            # 1e-20 here is the training constant, not `rms_norm_eps`, even
            # though they happen to be equal in this config.
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return weights * self.route_scale, indices


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
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        y = F.silu(gate) * up
        if weights is not None:
            y = weights * y
        return self.w2(y.to(dtype))


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

    def _stage(self, blob: dict[str, torch.Tensor]) -> Expert:
        """Copy one expert's stored weights into the staging slot.

        `copy_()` into the existing device buffers, not assignment: the buffers
        are what `.to(device)` placed in HBM and what a compiled graph would be
        bound to, so rebinding the attribute would move the computation off
        them.
        """
        for name in ("w1", "w2", "w3"):
            linear = getattr(self.expert_slot, name)
            linear.weight.copy_(blob[f"{name}.weight"].to(linear.weight.device))
            linear.scale.copy_(blob[f"{name}.scale"].to(linear.scale.device))
        return self.expert_slot

    def forward(
        self,
        x: torch.Tensor,
        image_mask: torch.Tensor | None,
        weight_store,
        tp_group=None,
    ) -> torch.Tensor:
        """Route to `num_experts_per_tok` of the pool, plus the shared expert.

        The loop runs only over the experts *this rank owns that this step
        actually routed to*, and fetches each one's weights as it reaches it.
        For a 4-sequence decode step that is at most 24 of 384 experts across
        all ranks — 6 of them here — so the layer's expert traffic is 6% of its
        expert bytes rather than all of them.

        Expert selection is identical on every rank (the gate is replicated and
        scores the whole pool), which is what makes the per-rank id partition
        cover the routed set exactly once. If selection could differ, the
        all-reduce would sum a different set of experts per rank.
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

        y = torch.zeros_like(flat, dtype=torch.float32)
        mine = [
            expert_id
            for expert_id in indices.unique().tolist()
            if self.experts_start_idx <= expert_id < self.experts_end_idx
        ]
        for expert_id, blob in weight_store.experts(self.layer_id, mine).items():
            expert = self._stage(blob)
            rows, slot = torch.where(indices == expert_id)
            y[rows] += expert(flat[rows], weights[rows, slot, None]).float()

        if tp_group is not None and self.n_local_experts != self.n_routed_experts:
            # Each rank holds the partial sum over its own experts. The
            # reference reduces in fp32, before the shared expert joins.
            tp_group.all_reduce(y)

        y = y + self.shared_experts(flat).float()
        return y.to(x.dtype).reshape(shape)


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
    what lets a decode step hash an n-gram reaching back into the prompt.
    """

    DEAD = -1

    def __init__(
        self,
        config: DeepseekV41Config,
        tokenizer,
        max_batch_size: int,
        max_seq_len: int,
    ) -> None:
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
        self.register_buffer(
            "cache",
            torch.empty(max_batch_size, max_seq_len, dtype=torch.int64),
            persistent=False,
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns `[B, L, n_engram_layers, n_hash_cols]` row indices.

        `token_mask` is `[B, L]`, False for tokens that take no part in an
        n-gram (image spans). Text-only serving passes None.
        """
        batch, seqlen = input_ids.shape
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, self.DEAD)
        self.cache[:batch, start_pos : start_pos + seqlen] = compressed

        positions = torch.arange(
            start_pos, start_pos + seqlen, device=input_ids.device
        ).expand(batch, seqlen)
        tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.max_ngram_size):
            source = self.cache[:batch].gather(1, (positions - shift).clamp_min(0))
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

    def __init__(self, config: DeepseekV41Config, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        self.clamp_value = 1e-6
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
        key, value = kv.split([self.hc_mult * self.hidden_size, self.hidden_size], -1)
        key = key.float().unflatten(-1, (self.hc_mult, self.hidden_size))
        # q and k gains are only ever used as a product.
        weight = self.q_weight.float() * self.k_weight.float()
        h = x.float()
        # Normalized per (token, hc copy) over `dim`, NOT jointly over copies.
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(
            key.square().mean(-1) + self.eps
        )
        dot = (h * weight * key).sum(-1) * rstd * self.hidden_size**-0.5
        gate = torch.sigmoid(
            torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot)
        )
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


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

        The normalization is over the whole flattened `hc * dim` stream — one
        statistic per token, not per copy.
        """
        hc = self.hc_mult
        flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flat, self.fn) * rsqrt

        # Transcribed from `hc_split_sinkhorn_kernel` in `vendor/kernel.py`, not
        # inferred: the three slices get different nonlinearities (`pre` a
        # sigmoid plus eps, `post` *twice* a sigmoid, `comb` nothing), and the
        # Sinkhorn is not a symmetric loop — the first column normalization is
        # outside it, so there are `iters` column passes but only `iters - 1`
        # row passes after the softmax. Getting either of those wrong leaves a
        # model that runs and drifts.
        raw_pre, raw_post, raw_comb = mixes.split([hc, hc, hc * hc], dim=-1)
        base_pre, base_post, base_comb = self.base.split([hc, hc, hc * hc])

        pre = torch.sigmoid(raw_pre * self.scale[0] + base_pre) + self.hc_eps
        post = 2.0 * torch.sigmoid(raw_post * self.scale[1] + base_post)

        comb = (raw_comb * self.scale[2] + base_comb).unflatten(-1, (hc, hc))
        comb = comb.softmax(dim=-1) + self.hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        return pre, post, comb

    @staticmethod
    def pre(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        """Collapse the copies into one sublayer input, weighted by `pre_mix`."""
        return torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2).to(x.dtype)

    @staticmethod
    def post(
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """Expand the sublayer output back out and mix the residual through
        `comb`. `x`: `[B,L,dim]`, `residual`: `[B,L,hc,dim]`.
        """
        broadcast = post.unsqueeze(-1) * x.unsqueeze(-2)
        mixed = torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return (broadcast + mixed).type_as(x)


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
        start_pos: int,
        pre_mix: torch.Tensor,
        image_mask: torch.Tensor | None,
        shared: SharedAttentionState,
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
        residual = x
        attn_pre, attn_post, attn_comb = self.hc_attn.mixes(x)
        y = HyperConnections.pre(x, pre_mix)
        y = self.attn_norm(y)
        y = self.attn(y, start_pos, shared, tp_group)
        x = HyperConnections.post(y, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_ffn.mixes(x)
        y = HyperConnections.pre(x, attn_pre)
        y = self.ffn_norm(y)
        y = self.ffn(y, image_mask, weight_store, tp_group)
        x = HyperConnections.post(y, residual, ffn_post, ffn_comb)
        return x, ffn_pre


def dspark_topk_idxs(
    window_size: int, bsz: int, block_size: int, start_pos: int, device
) -> torch.Tensor:
    """Which slots a draft query attends to. `vendor/model.py`'s
    `get_dspark_topk_idxs`.

    Every position in the draft block sees the same set: the whole ring (as far
    as it has filled) plus **all** `block_size` draft positions, with no causal
    mask among them — the draft block is scored jointly, which is the point of
    predicting a block at once.
    """
    if start_pos <= 0:
        raise ValueError("the DSpark draft path only runs during decode")
    matrix = torch.cat(
        [
            torch.arange(min(window_size, start_pos + 1), device=device),
            window_size + torch.arange(block_size, device=device),
        ]
    )
    return matrix.int().view(1, 1, -1).expand(bsz, block_size, -1).contiguous()


class DSparkAttention(Attention):
    """Draft attention: the main stream's KV in the ring, the draft block's own
    KV appended, and no compressed KV at all (`compress_ratio == 0` for every
    `mtp` stage, since `compress_ratios` ends `[..., 0, 0, 0]`).

    The asymmetry worth noticing: `main_x` is one *real* token, and `x` is the
    `dspark_block_size = 5` draft positions. The real token's KV goes into the
    ring (so the next step sees it); the draft positions' KV is used for this
    step only and never cached.
    """

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        main_x: torch.Tensor,
        shared: SharedAttentionState | None = None,
        tp_group=None,
    ) -> torch.Tensor:
        if self.compress_ratio:
            raise ValueError(
                f"DSpark stage {self.layer_id} has compress_ratio "
                f"{self.compress_ratio}; the draft path has no compressed KV"
            )
        bsz, seqlen, _ = main_x.shape
        win, rd = self.window_size, self.rope_head_dim

        main_freqs = slice_freqs(
            self.freqs_cis, slice(start_pos, start_pos + seqlen)
        )
        main_kv = self.kv_norm(self.wkv(main_x))
        main_kv = torch.cat(
            [main_kv[..., :-rd], apply_rotary_emb(main_kv[..., -rd:], main_freqs)],
            dim=-1,
        )
        main_kv = act_quant_dequant(main_kv, 32)

        if start_pos == 0:
            # Prefill only seeds the ring; the draft stream passes through.
            if seqlen <= win:
                self.window_kv_cache[:bsz, :seqlen] = main_kv
            else:
                cutoff = seqlen % win
                tail_a, tail_b = main_kv[:, -win:].split([win - cutoff, cutoff], 1)
                self.window_kv_cache[:bsz, cutoff:win] = tail_a
                self.window_kv_cache[:bsz, :cutoff] = tail_b
            return x

        bsz, block_size, _ = x.shape
        freqs = slice_freqs(
            self.freqs_cis,
            slice(start_pos + seqlen, start_pos + seqlen + block_size),
        )

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

        topk = dspark_topk_idxs(win, bsz, block_size, start_pos, x.device)
        self.window_kv_cache[:bsz, start_pos % win] = main_kv.squeeze(1)
        kv = torch.cat([self.window_kv_cache[:bsz], kv], dim=1)

        o = sparse_attn(q, kv, self.attn_sink, topk, self.softmax_scale)
        o = torch.cat(
            [o[..., :-rd], apply_rotary_emb(o[..., -rd:], freqs, True)], dim=-1
        )
        o = o.reshape(bsz, block_size, self.n_local_groups, -1)
        wo_a = self.wo_a.dequantized(torch.float32).view(
            self.n_local_groups, self.o_lora_rank, -1
        )
        o = torch.einsum("bsgd,grd->bsgr", o.float(), wo_a)
        out = self.wo_b(o.flatten(2).to(x.dtype))
        if self.world_size > 1 and tp_group is not None:
            out = out.float()
            tp_group.all_reduce(out)
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
            self.markov_head = DSparkMarkovHead(config, tp_size)
            self.confidence_head = DSparkConfidenceHead(config)


    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        pre_mix: torch.Tensor,
        main_x: torch.Tensor,
        shared: SharedAttentionState,
        weight_store,
        tp_group=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One draft stage. Prefill only seeds the ring and passes through."""
        if start_pos == 0:
            self.attn(x, start_pos, main_x, shared, tp_group)
            return x, pre_mix

        residual = x
        attn_pre, attn_post, attn_comb = self.hc_attn.mixes(x)
        y = HyperConnections.pre(x, pre_mix)
        y = self.attn_norm(y)
        y = self.attn(y, start_pos, main_x, shared, tp_group)
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

    def __init__(self, config: DeepseekV41Config, tp_size: int) -> None:
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
        self.rank_in_group = 0

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
            self._tp_group.all_reduce(embed)
        logits = F.linear(embed.float(), self.head_weight.float())
        if self.world_size > 1 and self._tp_group is not None:
            logits = self._tp_group.all_gather(logits, dim=-1)
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
        joined = torch.cat([hidden, markov_embed], dim=-1).float()
        return F.linear(joined, self.proj_weight.float()).squeeze(-1)


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

        self._shard_plan = self._build_shard_plan()

    # ── Construction ──────────────────────────────────────────────────────

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

        *In* — the sliding window. Every one of the 43 attentions (40 backbone +
        3 DSpark) attends over a window of `sliding_window = 128` raw KV
        positions, one latent head of `head_dim = 512`. That is per-layer,
        per-sequence, bounded, and written once per token: exactly what vLLM's
        block manager handles, and `sliding_window_size` is how it is told the
        window never grows.

        *Out* — the compressed KV and the index keys. Those are **shared across
        layers**, not per-layer: the four `kv_source_layer_ids` produce them and
        every other layer reads the same cache. A `KVSpec` entry describes
        storage owned by one layer, so expressing a cache that four layers write
        and 36 read would either duplicate it 40 times (40x the memory for one
        copy of the data) or lie about who owns it. They are therefore
        model-owned buffers sized by `max_model_len / compress_ratio`, allocated
        in `bind_kv_cache` next to the window caches — small enough for that to
        be the right call: 8224 positions at ratio 2 is 2 MiB of latent per
        source layer.
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
                    sliding_window_size=self.config.sliding_window,
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
                    sliding_window_size=self.config.sliding_window,
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
        for name, attn in attentions:
            if attn.is_kv_source:
                ratio = max(attn.compress_ratio, 1)
                attn.compress_kv_cache = torch.zeros(
                    self._max_batch_size(),
                    max_len // ratio,
                    attn.head_dim,
                    dtype=self.config.torch_dtype,
                    device=device,
                )
                attn.index_k_cache = torch.zeros(
                    self._max_batch_size(),
                    max_len // ratio,
                    self.config.index_head_dim,
                    dtype=self.config.torch_dtype,
                    device=device,
                )
                if attn.compressor is not None:
                    attn.compressor.allocate_state(device)
            # RoPE tables last, so an indexer picks up the index-key cache that
            # was just allocated above.
            attn.build_freqs_cis(max_len, device)

        if self.config.engram_layer_ids and self.ngram_hash is None:
            self.ngram_hash = self._build_ngram_hash(
                self._max_batch_size(), max_len, device
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

    def _build_ngram_hash(
        self, max_batch_size: int, max_seq_len: int, device
    ) -> NgramHashState:
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
        state = NgramHashState(self.config, tokenizer, max_batch_size, max_seq_len)
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
            self.tp_group.all_reduce(out)
        return out

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Vocab-parallel unembedding, gathered to full width."""
        logits = F.linear(hidden_states.float(), self.head_weight.float())
        if self.world_size > 1:
            logits = self.tp_group.all_gather(logits, dim=-1)
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

        > **Test status, stated plainly: this method has not yet produced a
        > token.** Every component it calls is checked against
        > `partition_artifact/trace/` — see the table in
        > `.autohelix/notes/iter-1.md`, including a whole decoder layer at
        > cos 1.000000 — but the assembly below, and in particular the
        > `_sequence_spans` bridge from vLLM's flat batch to the reference's
        > per-sequence `start_pos`, has only been reasoned about. Treat a
        > `verifier` failure as information about *this* method, not about the
        > parts.
        >
        > Do not read "the code is here" as "the computation has run". The
        > constraint that decides that is `verifier`.

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
        # Checked up front, before any work, so the failure the engine reports
        # is this sentence rather than whatever dynamo says several hundred
        # traced nodes later.
        self._require_ring_kv_layout()

        spans = self._sequence_spans(input_ids, positions)

        # ── embed -> hc_expand ────────────────────────────────────────────
        hidden = self.embed_input_ids(input_ids)
        hidden = NF.merge_prompt_embeds(hidden, inputs_embeds, is_token_ids)
        # `hc_expand`: one hidden state becomes hc_mult copies, with a one-hot
        # pre_mix so the first layer's hc_pre returns copy 0 unchanged.
        stream = hidden.unsqueeze(1).repeat(1, self.hc_mult, 1)
        pre_mix = stream.new_zeros(stream.shape[0], self.hc_mult, dtype=torch.float32)
        pre_mix[:, 0] = 1.0

        # ── engram hash ids, once for the whole batch ─────────────────────
        hashes = self._engram_hashes(input_ids, spans)

        # ── the 40 decoder layers ─────────────────────────────────────────
        shared = SharedAttentionState()
        main_hiddens: list[torch.Tensor] = []
        target_layers = set(self.config.dspark_target_layer_ids)

        for layer_id, block in enumerate(self.layers):
            if block.engram is not None:
                index = list(self.config.engram_layer_ids).index(layer_id)
                # The hash ids name rows of a 94.6 GiB table; the store gathers
                # just those rows (24 per token) and dequantizes them. This is
                # the engram half of the timeshare: the table is never resident
                # and never materialized, but every lookup a token needs does
                # happen.
                rows = self.weight_store.engram(layer_id, hashes[:, :, index, :])
                stream = block.engram(
                    stream.unsqueeze(0), rows.to(stream.device), None
                ).squeeze(0)
            if layer_id in target_layers:
                # The DSpark head reads the attention *input* of its target
                # layers, not their output, and averaged over the hc copies.
                main_hiddens.append(stream.float().mean(dim=1))
            stream, pre_mix = self._run_block(block, stream, pre_mix, spans, shared)

        # ── collapse -> norm -> head ──────────────────────────────────────
        collapsed = HyperConnections.pre(stream.unsqueeze(0), pre_mix.unsqueeze(0))
        collapsed = self.norm(collapsed).squeeze(0)

        main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
        if spec_decode_metadata is not None:
            # The DSpark draft path. Only reachable under a speculative config;
            # the reference's own `Transformer.forward` likewise collects
            # `main_hiddens` and leaves `forward_spec` to the caller.
            return self.forward_spec(input_ids, main_hidden, spans, shared)

        selected = collapsed
        if sampling_positions is not None:
            selected = torch.index_select(collapsed, 0, sampling_positions)
        return self.compute_logits(selected)

    # ── Forward helpers ───────────────────────────────────────────────────

    def _require_ring_kv_layout(self) -> None:
        """Refuse to run against a KV cache this attention cannot address.

        `get_kv_spec` declares 43 sliding-window layers and the runtime
        allocates them paged — `[num_blocks, num_kv_heads, block_size,
        head_size]` — while `Attention._window_kv` indexes a contiguous
        `[batch, sliding_window, head_dim]` ring, the way the reference does.
        Bridging the two is the last piece of this port.

        Raised here rather than left to fail inside the ring indexing so the
        engine's error names the cause. See `.autohelix/notes/iter-1.md`
        section 5a for the design that closes it, and section 4b for the
        `fullgraph=True` graph-break problem that sits in front of it.
        """
        cache = self.layers[0].attn.window_kv_cache
        if cache is None:
            raise RuntimeError(
                "bind_kv_cache() has not run; forward() has no KV cache."
            )
        expected = (self.config.sliding_window, self.layers[0].attn.head_dim)
        if tuple(cache.shape[-2:]) != expected:
            raise NotImplementedError(
                "this port cannot yet run against vLLM's paged KV layout "
                f"{tuple(cache.shape)}. It indexes a contiguous ring "
                f"[batch, {expected[0]}, {expected[1]}], which is what the "
                "reference uses and what every component check in "
                ".autohelix/notes/iter-1.md was run against. Writing the "
                "sliding window, the compressed KV and the index keys through "
                "`slot_mapping`/`block_table` instead is the remaining work — "
                "see that file, section 5a."
            )

    def _sequence_spans(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> list[tuple[int, int, int]]:
        """Split the flat token axis into `(start_index, length, start_pos)`.

        vLLM concatenates every scheduled sequence's tokens into one `[T]` axis
        and gives their absolute positions in `positions`. A sequence boundary
        is where `positions` stops increasing by one — a decode step is then `B`
        spans of length 1, and a prefill (chunked or not) is one span per
        sequence whose `start_pos` is that chunk's first position.

        Python-level, over at most `max_num_seqs` entries, and it reads
        `positions` on the host. Under `torch.compile` that is a graph break per
        batch shape, which is the cost of the reference's single-`start_pos`
        attention; removing it means rewriting the ring indexing to take
        `positions` as a tensor.
        """
        values = positions.reshape(-1).tolist()
        if not values:
            return []
        spans: list[tuple[int, int, int]] = []
        begin = 0
        for i in range(1, len(values) + 1):
            if i == len(values) or values[i] != values[i - 1] + 1:
                spans.append((begin, i - begin, int(values[begin])))
                begin = i
        return spans

    def _engram_hashes(
        self, input_ids: torch.Tensor, spans: list[tuple[int, int, int]]
    ) -> torch.Tensor:
        """n-gram row indices for every token, `[1, T, n_engram_layers, 24]`.

        Hashed per sequence, because the n-gram look-back must not reach across
        a sequence boundary — and `NgramHashState` keeps the compressed ids of
        earlier chunks in its own cache, which is what lets a decode step hash
        an n-gram reaching back into its prompt.
        """
        if self.ngram_hash is None:
            raise RuntimeError(
                "engram layers "
                f"{list(self.config.engram_layer_ids)} need the n-gram hash "
                "state; bind_kv_cache builds it from the tokenizer."
            )
        per_span = []
        for begin, length, start_pos in spans:
            ids = input_ids[begin : begin + length].reshape(1, length)
            per_span.append(self.ngram_hash(ids, start_pos))
        return torch.cat(per_span, dim=1)

    def _run_block(
        self,
        block: Block,
        stream: torch.Tensor,
        pre_mix: torch.Tensor,
        spans: list[tuple[int, int, int]],
        shared: SharedAttentionState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One layer over a flat `[T, hc, dim]` stream covering several
        sequences.

        The hyper-connection coefficients and both norms are per token, so they
        run over the whole batch at once. Attention runs per sequence, because
        the ring and the compressed history are per sequence. The MoE runs over
        the whole batch, which is what keeps the staged expert weights used once
        per layer rather than once per sequence.
        """
        tp_group = self.tp_group if self.world_size > 1 else None
        batched = stream.unsqueeze(0)

        residual = batched
        attn_pre, attn_post, attn_comb = block.hc_attn.mixes(batched)
        collapsed = HyperConnections.pre(batched, pre_mix.unsqueeze(0))
        normed = block.attn_norm(collapsed)

        attn_out = torch.empty_like(normed)
        for begin, length, start_pos in spans:
            attn_out[:, begin : begin + length] = block.attn(
                normed[:, begin : begin + length], start_pos, shared, tp_group
            )
        batched = HyperConnections.post(attn_out, residual, attn_post, attn_comb)

        residual = batched
        ffn_pre, ffn_post, ffn_comb = block.hc_ffn.mixes(batched)
        collapsed = HyperConnections.pre(batched, attn_pre)
        normed = block.ffn_norm(collapsed)
        ffn_out = block.ffn(normed, None, self.weight_store, tp_group)
        batched = HyperConnections.post(ffn_out, residual, ffn_post, ffn_comb)

        return batched.squeeze(0), ffn_pre.squeeze(0)

    @torch.no_grad()
    def forward_spec(
        self,
        input_ids: torch.Tensor,
        main_hidden: torch.Tensor | None,
        spans: list[tuple[int, int, int]],
        shared: SharedAttentionState,
    ):
        """The DSpark draft path over the `mtp.*` stages.

        `vendor/model.py`'s `Transformer.forward_spec`: build the draft stream
        from the accepted token and the projected `main_hidden`, run it through
        the three `mtp` stages (each a hyper-connected block over the 128-expert
        pool at top-3), then the shared LM head with the Markov bias folded in
        per draft position, and the confidence head.

        Not reachable from the verifier's configuration — it has no speculative
        config, so `spec_decode_metadata` is None, and the reference's own
        `Transformer.forward` likewise only collects `main_hiddens`. Implemented
        because the 128-expert pool and the two heads are part of the
        architecture; checked against the trace's `mtp.*` records rather than by
        running it here.
        """
        if main_hidden is None:
            raise RuntimeError(
                "the DSpark stages read the attention inputs of layers "
                f"{list(self.config.dspark_target_layer_ids)}; forward() "
                "collected none, so the target-layer tap did not run"
            )
        if len(spans) != 1:
            raise NotImplementedError(
                "DSpark drafting is implemented for one sequence per call "
                f"(got {len(spans)} spans). The draft block is per sequence and "
                "the ring index is per sequence; batching it is the same "
                "per-sequence slotting problem as the backbone's, see "
                ".autohelix/notes/iter-1.md section 5a."
            )
        (begin, length, start_pos), = spans

        # The accepted token this draft continues from: the last real token.
        last_id = input_ids[begin + length - 1].reshape(1)
        main = main_hidden[begin + length - 1].reshape(1, 1, -1)

        stage0 = self.mtp[0]
        stage0.embed_fn = self.embed_input_ids
        stream, main_x = stage0.forward_embed(main, last_id)
        pre_mix = stream.new_zeros(
            stream.shape[0], stream.shape[1], self.hc_mult, dtype=torch.float32
        )
        pre_mix[:, :, 0] = 1.0

        tp_group = self.tp_group if self.world_size > 1 else None
        draft_pos = start_pos + length - 1
        for stage in self.mtp:
            stream, pre_mix = stage(
                stream, draft_pos, pre_mix, main_x, shared, self.weight_store,
                tp_group,
            )

        last = self.mtp[-1]
        last.head_fn = self.compute_logits
        last.markov_head._tp_group = tp_group
        return last.forward_head(stream, pre_mix, last_id)

    # ── Weights ───────────────────────────────────────────────────────────

    def load_weights(
        self,
        checkpoint_path: str,
        device: torch.device,
        cache_dir: str | None = None,
    ) -> None:
        """Load the sharded checkpoint. Delegates to `weight_loader`."""
        from .weight_loader import load_weights as _load_weights

        _load_weights(self, checkpoint_path, device, cache_dir)


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
