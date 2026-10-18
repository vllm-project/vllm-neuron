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
VOCAB_SHARD_ALIGN = 128


def vocab_shard_width(vocab_size: int, world_size: int) -> int:
    """Rows of the vocab-parallel embedding and head that one rank holds.

        129280, 1 rank  -> 129280   (129280 = 1010 x 128)
        129280, 2 ranks ->  64640   ( 64640 =  505 x 128)
        129280, 4 ranks ->  32384   ( 32320 =  252.5 x 128, so round up)
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
    """Every rank's `x` concatenated along the last axis."""
    return _fc_all_gather_tensor(x, -1, group_name)


def _group_name(tp_group) -> str:
    """The functional-collective name of a `GroupCoordinator`'s device group."""
    return _fc_resolve_group_name(tp_group.device_group)


#: Largest all-reduce payload, in elements, handed to one collective call.
_ALLREDUCE_CHUNK_ELEMS = int(
    os.environ.get("DEEPSEEK_V41_ALLREDUCE_CHUNK_ELEMS", 1 << 30)
)


def tp_all_reduce(x: torch.Tensor, tp_group) -> torch.Tensor:
    """Sum `x` across the tensor-parallel ranks. Returns the reduced tensor."""
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


@functools.lru_cache(maxsize=None)
def _nki_entry(module: str, name: str = "kernel"):
    """A compiled entry point for `kernels.<module>.<name>`.

        torch._dynamo.exc.FailOnRecompileLimitHit: Hard failure due to
        fullgraph=True
    """
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from vllm_neuron.envs import get_compile_backend_name

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

        %sort.50 = (f32[128,384], s32[128,384]) sort(...) metadata={
            op_type="aten__topk" op_name="Gate[G['gate']]/aten__topk" ...}
        FATAL: neuronx-cc compilation failed with 70.
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
    """`x @ weight.T` on the device, through the matching NKI kernel."""
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
    os.environ.get("DEEPSEEK_V41_ENGRAM_TABLE_BYTES", 6 << 30)
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

    __slots__ = ("compress_kv", "index_k", "topk_idxs", "candidates")

    def __init__(self) -> None:
        self.compress_kv: torch.Tensor | None = None
        self.index_k: torch.Tensor | None = None
        self.topk_idxs: torch.Tensor | None = None
        self.candidates: torch.Tensor | None = None


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
        """Allocate the per-slot pooling inputs."""
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
        """Pool `compress_ratio` consecutive tokens into one latent."""
        _, seqlen, _ = x.shape
        ratio, dtype = self.compress_ratio, x.dtype
        device = x.device
        if ratio == 1:
            positions = start_pos + torch.arange(seqlen, device=device)
            return self.norm(self.wkv(x)), positions

        xf = x.float()
        paged.write_flat(self.pool_kv, self.wkv(xf).reshape(seqlen, -1))
        paged.write_flat(self.pool_score, self.wgate(xf).reshape(seqlen, -1))

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
        pooled = _nki_entry("compress_pool", "softmax_pool")(kv, score).unsqueeze(0)
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
        """`x` is the normalized sublayer input `[b, s, hidden]`."""
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
            out = tp_all_reduce(out.float(), self.attn_group or tp_group)
            if self.replicas > 1:
                out = out / self.replicas
            out = out.to(x.dtype)
        return out


def _compile_fn(fn):
    """`fn` as a device graph. One call, so every compiled entry point in this
    file is built the same way as the whole-model compile the runner does.

        torch._dynamo.exc.Unsupported: Attempted to call function marked as
        skipped ... Dynamo does not know how to trace the builtin
        `time.perf_counter`
    """
    from vllm_neuron.envs import get_compile_backend_name

    _raise_dynamo_limits()
    return torch.compile(
        fn,
        backend=get_compile_backend_name(),
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


def _swiglu_expert(x, weights, w1, s1, w2, s2, w3, s3, storage, limit):
    """One expert's SwiGLU FFN; all three GEMMs through `kernels/blockquant`.

      * the needle case answered correctly and stopped emitting EOS (19 tokens
        `stop` -> 32 tokens `length`), and
      * speculation on and off stopped producing identical tokens — two of the
        three short cases still matched exactly, `smoke-test` diverged after 80
        characters.
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
    """Gather one expert's rows, run it, scatter-add into the fp32 carry."""
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


def _rejection_sample(sampled, draft_token_ids, cu_num_draft_tokens, num_draft_tokens):
    """Verify this step's drafts against the target's own tokens, on the device."""
    from vllm_neuron.nn.rejection_sampler import rejection_sampler

    view = _RejectionView()
    view.draft_token_ids = draft_token_ids
    view.cu_num_draft_tokens = cu_num_draft_tokens
    view.max_spec_len = max(num_draft_tokens)
    return rejection_sampler(view, sampled.reshape(-1))


def _accepted_tail(accepted):
    """`(last accepted token, how many were accepted)` per sequence."""
    valid = (accepted >= 0).sum(dim=-1)
    seeds = accepted.gather(1, (valid - 1).clamp_min(0).unsqueeze(1)).reshape(-1)
    return seeds, valid.to(torch.int64)


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
    """Top-k routed experts plus one shared expert every token goes through."""

    def __init__(
        self, config: DeepseekV41Config, layer_id: int, tp_size: int, tp_rank: int
    ) -> None:
        super().__init__()
        n_routed, n_activated = _moe_config(config, layer_id)
        self.layer_id = layer_id
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
        """Give this layer `count` HBM slots to hold routed experts."""
        count = max(0, min(count, self.n_local_experts))
        slots = []
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

        y = _compiled(_moe_carry)(flat)
        limit = self.expert_slot.swiglu_limit
        for expert_id, rows, route_weights in plan:
            mark = time.perf_counter() if _EXPERT_TIMING else 0.0
            slot = self._expert(weight_store, expert_id)
            staged = time.perf_counter() if _EXPERT_TIMING else 0.0
            y = _compiled(_routed_expert_step)(
                y, flat, rows, route_weights,
                slot.w1.weight, slot.w1.scale,
                slot.w2.weight, slot.w2.scale,
                slot.w3.weight, slot.w3.scale,
                limit,
            )
            if _EXPERT_TIMING:
                _EXPERT_TIME[0] += staged - mark
                _EXPERT_TIME[1] += time.perf_counter() - staged
        if weight_store is not None and not self.resident:
            weight_store.end_expert_loop()

        if tp_group is not None and self.n_local_experts != self.n_routed_experts:
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
        """The per-slot compressed-id history."""
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
        """Returns `[B, L, n_engram_layers, n_hash_cols]` row indices."""
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
        out_residual.copy_(residual)
        out_normed.copy_(normed)
        return out_normed, out_residual, ffn_pre, ffn_post, ffn_comb

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
            out = tp_all_reduce(out.float(), self.attn_group or tp_group)
            if self.replicas > 1:
                out = out / self.replicas
            out = out.to(x.dtype)
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

    def __init__(self, config: DeepseekV41Config) -> None:
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config

        tp_group = get_tp_group()
        self.tp_group = tp_group
        self.world_size = tp_group.world_size
        self.rank = tp_group.rank_in_group

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
                    logger.warning(
                        "deepseek_v41: the EP-TP group is %d ranks but the "
                        "attention splits %d ways, so it is not the group this "
                        "reduction needs; reducing over the world instead",
                        candidate.world_size, self.attn_tp,
                    )
        _LOG_RANK[0] = self.rank == 0
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
        #: `(width, dtype, device) -> bias`, filled by `_vocab_pad_bias`. Same
        #: idea as `_layer_io_slots`: one buffer per shape, so the mask is not
        #: rebuilt per forward and the graph's retained inputs do not grow.
        self._vocab_pad_slots: dict[tuple, torch.Tensor] = {}

        self.register_buffer(
            "embed_weight",
            torch.empty(self.part_vocab_size, config.hidden_size, dtype=torch.bfloat16),
        )
        self.layers = nn.ModuleList(
            [
                Block(
                    config, layer_id, self.world_size, self.rank,
                    self._max_batch_size(),
                    attn_tp=self.attn_tp, attn_rank=self.attn_rank,
                    attn_group=None if self.attn_group_is_world
                    else self.attn_tp_group,
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
                    config, stage, self.world_size, self.rank,
                    self._max_batch_size(),
                    attn_tp=self.attn_tp, attn_rank=self.attn_rank,
                    attn_group=None if self.attn_group_is_world
                    else self.attn_tp_group,
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
        for layer_id, block in enumerate(self.layers):
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
        attentions = [(f"layers.{i}.attn", b.attn) for i, b in enumerate(self.layers)]
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
        device = self.norm.weight.device
        window = self.layers[0].attn.window_kv_cache
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

        if self.config.engram_layer_ids and self.ngram_hash is None:
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


    def allocate_staging_buffers(self, device: torch.device) -> None:
        """Materialize the per-layer expert staging slots on `device`.

            NotImplementedError: Cannot copy out of meta tensor; no data!
        """
        blocks = list(self.layers) + list(self.mtp)
        for block in blocks:
            block.ffn.expert_slot.allocate(device)

        per_expert = sum(
            buffer.numel() * buffer.element_size()
            for buffer in blocks[0].ffn.expert_slot.buffers()
        )
        _EXPERT_BYTES[0] = per_expert
        wanted = sum(block.ffn.n_local_experts for block in blocks)
        affordable = _EXPERT_CACHE_BYTES // (per_expert * max(1, len(blocks)))
        allocated = sum(
            block.ffn.allocate_cache_slots(
                self.config, device, min(block.ffn.n_local_experts, affordable)
            )
            for block in blocks
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
            logits = logits + self._vocab_pad_bias(logits)
        return logits

    def _vocab_pad_bias(self, like: torch.Tensor) -> torch.Tensor:
        """`[part_vocab_size]`, zero on real columns and `-inf` on the pad."""
        key = (like.shape[-1], like.dtype, like.device)
        bias = self._vocab_pad_slots.get(key)
        if bias is None:
            bias = torch.zeros(key[0], dtype=like.dtype, device=like.device)
            bias[key[0] - self.vocab_pad :] = float("-inf")
            self._vocab_pad_slots[key] = bias
        return bias

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
        live_spans = self._sequence_spans(positions, meta["slot_mapping"])
        live_rows = sum(length for _, length, _ in live_spans)
        spans, bucket_meta = self._bucket_rows(
            live_spans, meta, int(input_ids.shape[0])
        )
        paged = self._paged_spans(spans, bucket_meta)
        live_lengths = [length for _, length, _ in live_spans]
        slices = [(begin, length) for begin, length, _ in spans]
        bucket_rows = sum(length for _, length, _ in spans)

        hidden = self.embed_input_ids(input_ids)
        hidden = NF.merge_prompt_embeds(hidden, inputs_embeds, is_token_ids)
        stream, pre_mix = _nki_entry("hc_expand")(hidden.unsqueeze(0))
        stream, pre_mix = stream.squeeze(0), pre_mix.squeeze(0)

        hashes = self._engram_hashes(input_ids, paged)

        shared = [SharedAttentionState() for _ in paged]
        main_hiddens: list[torch.Tensor] = []
        target_layers = set(self.config.dspark_target_layer_ids)

        _MOE_LIVE_ROWS[0] = live_rows if live_rows != bucket_rows else None

        for layer_id, block in enumerate(self.layers):
            if layer_id % _HEAP_TRIM_EVERY == 0:
                host_heap_trim(f"layer {layer_id}", self.world_size)
            if block.engram is not None:
                index = list(self.config.engram_layer_ids).index(layer_id)
                rows = self._engram_rows(layer_id, hashes, index, stream.device)
                updated = block.engram(
                    stream[:bucket_rows].unsqueeze(0), rows, None
                ).squeeze(0)
                if bucket_rows < stream.shape[0]:
                    stream = _compiled(_splice_rows)(updated, stream[bucket_rows:])
                else:
                    stream = updated
            if layer_id in target_layers:
                main_hiddens.append(_compiled(_dspark_tap)(stream))
            stream, pre_mix = self._run_block(
                block, stream, pre_mix, slices, paged, shared, bucket_rows
            )
        _MOE_LIVE_ROWS[0] = None

        collapsed = HyperConnections.pre(stream.unsqueeze(0), pre_mix.unsqueeze(0))
        collapsed = self.norm(collapsed).squeeze(0)

        selected = collapsed
        if sampling_positions is not None:
            selected = _compiled(_select_rows)(collapsed, sampling_positions)
        logits_shard = self.logits_shard(selected)
        logits = self.gather_logits(logits_shard)

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
        sampled = _block_entry(self.sampler, "forward")(
            logits_shard, sampling_params, logit_mask, rank
        )

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
            sampling_positions is not None, live_lengths,
        )
        if accepted is not None:
            return accepted
        return sampled, logits


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
        engram = self.layers[layer_id].engram
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

            scatter/gather (indirect memory copy via scalar DGE)
            out-of-bound access ... engine=GPSIMD
            Source location: model.py:2921
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
    def _bucket_rows(
        self,
        spans: list[tuple[int, int, int]],
        meta: dict,
        total_rows: int,
    ) -> tuple[list[tuple[int, int, int]], dict]:
        """Round a lone sequence's span up to the whole token bucket, above one
        rank, so a short prefill chunk reuses the graphs a full one built.

            scatter/gather (indirect memory copy via vector DGE)
            out-of-bound access ... engine=GPSIMD
            RuntimeError: nrta status=1006
        """
        if self.world_size <= 1 or len(spans) != 1 or total_rows <= 0:
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
    def _sequence_spans(
        self, positions: torch.Tensor, slot_mapping: torch.Tensor
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
        return spans

    def _engram_hashes(
        self, input_ids: torch.Tensor, paged: list[PagedSpan]
    ) -> torch.Tensor:
        """n-gram row indices for every token, `[1, T, n_engram_layers, 24]`."""
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
        row_count: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One layer over a flat `[T, hc, dim]` stream covering several
        sequences.
        """
        tp_group = self.tp_group if self.world_size > 1 else None
        io_stream, io_residual, io_normed = self._layer_io(stream)
        normed, residual, ffn_pre, ffn_post, ffn_comb = _block_entry(
            block, "attn_half"
        )(stream, pre_mix, slices, paged, shared, tp_group, io_residual, io_normed)
        ffn_out = block.ffn(normed, None, self.weight_store, tp_group, row_count)
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
