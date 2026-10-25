# SPDX-License-Identifier: Apache-2.0
"""The bridge from `model.py`'s modules to the megakernels (Trainium2).

    prepare(model)                                   once, after the weights load
    make_step(spans, paged, ...)                     once a forward
    run_layer(model, block, stream, pre_mix, step)   one `Block` = one layer kernel
    run_draft(model, seed, tap, main_hidden, span, paged)   `forward_spec` = one drafter kernel

**Weights.** `prepare` re-lays every fp8 linear the kernels read as its
dequantized bf16 transpose (`kmajor_bf16`: `[in, out]`, the contraction on the
partition axis, the tensor engine's moving-operand layout) and releases the fp8
bytes, so each kernel's weight tile is one plain DMA (see `kernels/mk_gemm`). The
values are the ones the per-op GEMMs compute in-kernel on every call
(`blockquant._decode_e4m3` times the row scale), so nothing about the arithmetic
moves; the per-op path cannot run afterwards, which is why the choice is made once,
at load (`DEEPSEEK_V41_MEGAKERNEL`). The routed experts stay fp4 and k-major, as
the resident pool holds them, restacked once onto a leading expert axis with the
per-op slots left as views of the stack. Everything else a kernel reads -- `fn.T`
and the router weight as fp32, norms as `[1, n]` rows, the RoPE table as one
`[max_pos, 64]` (cos | sin) table -- is built once per block on first use.

**Per forward.** `StepIndices` turns the scheduler's block tables and slot mapping
into the kernels' index columns (`mk_host.build_attn_indices`) from host copies the
forward already made, and carries what the layers hand down the stack: the current
compressed / index-key caches, the latest top-k, the DSpark taps.

**Which kernel.** `layer_kernel.archetype` names a block from its config; the
phase is the step's shape -- at most 128 rows is a decode-shaped step (the
column-split kernel), more is a prefill chunk (the row-split kernel). The kernels
do their own collectives; the replica groups are constants (consecutive ranks).
"""

from __future__ import annotations

import functools
import glob
import hashlib
import inspect
import logging
import os
from dataclasses import dataclass, field

import torch

from . import mk_host
from .kernels import draft_kernel, layer_kernel

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
#: Kernels submitted with nothing on the host waiting on them; see `_throttle`. Each
#: drafter kernel brings two small helper graphs the limiter does not track, so 8
#: keeps the worst case (8 x 3) inside the runtime's 32-deep queue.
_IN_FLIGHT = 8
#: `DEEPSEEK_V41_MK_SYNC=1` waits for every kernel before submitting the next and
#: logs its wall time on the first and last rank -- the way to name a kernel that
#: does not finish.
_MK_SYNC = os.environ.get("DEEPSEEK_V41_MK_SYNC", "") not in ("", "0")
PT = 128


@functools.lru_cache(maxsize=None)
def kernel_version() -> int:
    """A hash of every kernel source file, folded into the kernels' constant
    `flags`: the NKI compile caches key a kernel on its *own* function source,
    so an edit to a helper module would otherwise be served a stale NEFF."""
    h = hashlib.sha256()
    files = sorted(glob.glob(os.path.join(_HERE, "kernels", "*.py")))
    files += [os.path.join(_HERE, "mk_host.py")]
    for path in files:
        with open(path, "rb") as fh:
            h.update(fh.read())
    return int.from_bytes(h.digest()[:4], "little") & 0x7FFFFFFF


@functools.lru_cache(maxsize=None)
def entry(kernel):
    """The compiled entry point of one kernel, launched on both physical cores."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from . import model as M

    M._raise_dynamo_limits()
    wrapped = wrap_nki(kernel)[2]

    def run(*args):
        return wrapped(*args)

    name = getattr(kernel, "__name__", None) or getattr(getattr(kernel, "func", None), "__name__", "k")
    run.__name__ = f"nki_mk_{name}"
    # A code object of its own per kernel: dynamo keeps its compiled entries on the
    # code object, and one shared `run` would make every kernel a recompile of the
    # previous one (`len(args) == 41` guards) and share one cache-size budget.
    run.__code__ = run.__code__.replace(co_name=run.__name__)
    return torch.compile(run, backend=M._compile_backend_name(), fullgraph=True,
                         options=M._COMPILE_OPTIONS)


@functools.lru_cache(maxsize=None)
def params(kernel) -> tuple[str, ...]:
    func = kernel.func if hasattr(kernel, "func") else kernel
    return tuple(inspect.signature(func).parameters)


def _host(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu()


def _dev(t: torch.Tensor, device) -> torch.Tensor:
    return t.contiguous().to(device)


def _row(t: torch.Tensor, device) -> torch.Tensor:
    """A 1-D buffer as the `[1, n]` row the kernels broadcast from."""
    return _dev(_host(t).reshape(1, -1), device)


def _t_f32(t: torch.Tensor, device) -> torch.Tensor:
    """`w.T` as fp32, for the helpers that stage a weight transposed in fp32."""
    return _dev(_host(t).t().float(), device)


def _t_bf16(t: torch.Tensor, device) -> torch.Tensor:
    """`w.T` of a bf16 weight (no quantization), k-major."""
    return _dev(_host(t).t().to(torch.bfloat16), device)


# --------------------------------------------------------------------------- weights


def kmajor_bf16(linear, release: bool = True) -> torch.Tensor:
    """An fp8 `BlockQuantLinear` as the kernels read it: `dequant(w).T`, bf16,
    `[in, out]`, on the weight's device. Converted once on the host and cached on
    the linear (`linear._kb16`); the fp8 buffers are released afterwards."""
    kb = getattr(linear, "_kb16", None)
    if kb is not None:
        return kb
    from .weight_loader import dequant_block_fp8

    assert linear.storage == "fp8", linear.storage
    device = linear.weight.device
    w = dequant_block_fp8(_host(linear.weight), _host(linear.scale), linear.block, torch.float32)
    kb = w.t().contiguous().to(torch.bfloat16).to(device)
    linear._kb16 = kb
    if release:
        linear.weight = None
        linear.scale = None
        if getattr(linear, "row_scale", None) is not None:
            linear.row_scale = None
    return kb


def stacked_experts(ffn) -> dict:
    """The resident expert pool as `[E, k', n]` k-major fp4 / e8m0 stacks (`w1`,
    `s1`, ...), the per-op slots rewired as views of them. Built once."""
    st = getattr(ffn, "_mk_stacked", None)
    if st is not None:
        return st
    if not ffn.resident or len(ffn.cache_slots) != ffn.n_local_experts:
        raise RuntimeError(
            f"layer {ffn.layer_id}: the routed experts are not resident (the pool streams); "
            "the megakernels read every local expert from HBM. Raise "
            "DEEPSEEK_V41_EXPERT_CACHE_BYTES."
        )
    device = ffn.cache_slots[0].w1.weight.device
    st = {}
    for name in ("w1", "w2", "w3"):
        for buf, key in (("weight", "w"), ("scale", "s")):
            parts = [_host(getattr(getattr(slot, name), buf)) for slot in ffn.cache_slots]
            stack = torch.stack(parts).contiguous().to(device)
            st[f"{key}{name[1]}"] = stack
            for i, slot in enumerate(ffn.cache_slots):
                setattr(getattr(slot, name), buf, stack[i])
    ffn._mk_stacked = st
    return st


def _freqs_table(attn, device) -> torch.Tensor:
    """`attn.freqs_cis` as `[max_pos, 64]` fp32 (cos | sin)."""
    cos, sin = attn.freqs_cis
    return _dev(torch.cat([_host(cos).float(), _host(sin).float()], -1), device)


@dataclass
class LayerWeights:
    """One block's static kernel arguments by name, and its archetype."""

    arch: str
    ratio: int
    kv_src: bool
    idx_src: bool
    engram: bool
    tap: bool
    window: int
    named: dict


def prepare(model) -> None:
    """Re-lay every weight the kernels read (see the module docstring). Called once,
    from `load_weights`, so the KV cache is sized against the converted footprint."""
    for block in model.layers:
        _block_weights(model, block)
    draft_weights(model, with_freqs=False)
    logger.info("deepseek_v41: megakernel weights prepared (fp8 linears as bf16 k-major, "
                "expert pools stacked)")


def _block_weights(model, block) -> dict:
    """The weights of `block` that need no runtime state (no caches, no RoPE table)."""
    cached = getattr(block, "_mk_static", None)
    if cached is not None:
        return cached
    device = model.norm.weight.device
    attn, ffn = block.attn, block.ffn
    n = {
        "hca_fn_t": _t_f32(block.hc_attn.fn, device), "hca_base": _row(block.hc_attn.base, device),
        "hca_scale": _row(block.hc_attn.scale, device), "attn_norm_w": _row(block.attn_norm.weight, device),
        "hcf_fn_t": _t_f32(block.hc_ffn.fn, device), "hcf_base": _row(block.hc_ffn.base, device),
        "hcf_scale": _row(block.hc_ffn.scale, device), "ffn_norm_w": _row(block.ffn_norm.weight, device),
        "wq_a": kmajor_bf16(attn.wq_a), "q_norm_w": _row(attn.q_norm.weight, device),
        "wq_b": kmajor_bf16(attn.wq_b), "wkv": kmajor_bf16(attn.wkv),
        "kv_norm_w": _row(attn.kv_norm.weight, device),
        "wo_a": kmajor_bf16(attn.wo_a), "wo_b": kmajor_bf16(attn.wo_b),
        "attn_sink": _dev(_host(attn.attn_sink).float().reshape(-1, 1), device),
        "gate_w_t": _t_f32(ffn.gate.weight, device), "gate_b": _row(ffn.gate.bias, device),
        "expert_lo": torch.tensor([[float(ffn.experts_start_idx)]], dtype=torch.float32, device=device),
        "sh_w1": kmajor_bf16(ffn.shared_experts.w1), "sh_w3": kmajor_bf16(ffn.shared_experts.w3),
        "sh_w2": kmajor_bf16(ffn.shared_experts.w2),
    }
    n.update(stacked_experts(ffn))
    if attn.is_kv_source:
        comp = attn.compressor
        if comp.compress_ratio > 1:
            n["cm_wkv_t"] = _t_f32(comp.wkv.weight, device)
            n["cm_wgate_t"] = _t_f32(comp.wgate.weight, device)
        else:
            n["cm_wkv"] = _t_bf16(comp.wkv.weight, device)
        n["cm_norm_w"] = _row(comp.norm.weight, device)
        n["ix_wk"] = _t_bf16(attn.indexer.wk.weight, device)
        n["ix_knorm_w"] = _row(attn.indexer.k_norm.weight, device)
    if attn.is_index_source:
        n["ix_wq_b"] = kmajor_bf16(attn.indexer.wq_b)
        n["ix_wproj_t"] = _t_f32(attn.indexer.weights_proj.weight, device)
    if block.engram is not None:
        eg = block.engram
        if not eg.resident:
            raise RuntimeError(
                f"layer {block.layer_id}: the engram table is not resident on the device, and "
                "the megakernel reads it there. Raise DEEPSEEK_V41_ENGRAM_TABLE_BYTES."
            )
        n["etable"] = eg.table
        # The ids reach the kernel already local to the shard (`StepIndices.hash_cols`).
        n["erow_start"] = torch.zeros(1, 1, dtype=torch.int32, device=device)
        n["e_wkv"] = kmajor_bf16(eg.wkv)
        n["e_qk"] = _dev((_host(eg.q_weight).float() * _host(eg.k_weight).float()).reshape(1, -1), device)
    block._mk_static = n
    return n


def layer_weights(model, block) -> LayerWeights:
    lw = getattr(block, "_mk_weights", None)
    if lw is not None:
        return lw
    attn = block.attn
    cfg = model.config
    n = dict(_block_weights(model, block))
    n["freqs"] = model._mk_freqs(attn)
    ratio = int(attn.compress_ratio)
    tap = attn.layer_id in set(cfg.dspark_target_layer_ids)
    arch = layer_kernel.archetype(ratio, attn.is_kv_source, attn.is_index_source,
                                  block.engram is not None, tap)
    lw = LayerWeights(arch=arch, ratio=ratio, kv_src=bool(attn.is_kv_source),
                      idx_src=bool(attn.is_index_source), engram=block.engram is not None,
                      tap=tap, window=int(attn.window_size), named=n)
    block._mk_weights = lw
    return lw


# --------------------------------------------------------------------------- per step


@dataclass
class StepIndices:
    """Per-forward index columns and the state the layers hand down the stack."""

    spans: list
    paged: list
    device: torch.device
    hashes: torch.Tensor | None
    tables: list
    slot_mapping: torch.Tensor
    live_rows: int
    T: int = 0
    _by_ratio: dict = field(default_factory=dict)
    _hash_cols: dict = field(default_factory=dict)
    topk: torch.Tensor | None = None
    compress_cache: torch.Tensor | None = None
    index_cache: torch.Tensor | None = None
    taps: list = field(default_factory=list)
    #: Outputs of the layers submitted and not yet known complete, oldest first.
    pending: list = field(default_factory=list)

    def __post_init__(self):
        self.T = sum(length for _, length, _ in self.spans)
        self.block_size = int(self.paged[0].block_size)
        lengths = {length for _, length, _ in self.spans}
        if len(lengths) != 1:
            raise NotImplementedError(
                f"the layer kernels want every span of a step to have the same length, got "
                f"{sorted(lengths)}; the Neuron scheduler does not mix prefill and decode"
            )
        self.live = torch.tensor([[float(self.live_rows)]], dtype=torch.float32, device=self.device)

    def read_len(self, ratio: int) -> int:
        if not ratio:
            return 0
        lens = {int(p.read_lens[ratio]) for p in self.paged}
        if len(lens) != 1:
            raise NotImplementedError(f"spans with different compressed read lengths {sorted(lens)}")
        return lens.pop()

    def for_ratio(self, ratio: int, window: int) -> mk_host.AttnIndices:
        key = (ratio, window)
        idx = self._by_ratio.get(key)
        if idx is None:
            idx = mk_host.build_attn_indices(
                self.spans, self.tables, self.slot_mapping, self.block_size, window,
                self.read_len(ratio), ratio, self.T,
            )
            for name in ("row_pos", "write_slots", "win_slots", "cmp_slots", "twin", "wv", "cbase",
                         "clen", "grp_pos", "grp_mslots", "grp_wslots"):
                t = getattr(idx, name)
                if t is not None:
                    setattr(idx, name, t.to(self.device))
            self._by_ratio[key] = idx
        return idx

    def hash_cols(self, index: int, row_start: torch.Tensor) -> torch.Tensor:
        """This engram layer's `[T, 24]` int32 row ids, **local to this rank's shard**
        (`global - row_start`): the kernel's gather turns ids into fp32 for its range
        mask, and a global id of a 384M-row table is not exact above 2^24, while a
        local one is below the shard width; an off-shard one only has to stay out of
        range, which a rounded value does."""
        t = self._hash_cols.get(index)
        if t is None:
            from . import model as M

            if self.hashes is None:
                raise RuntimeError("an engram layer ran but the step has no n-gram hashes")
            t = M._compiled(_local_ids)(self.hashes, index, row_start)
            self._hash_cols[index] = t
        return t


def _local_ids(hashes, index, row_start):
    """`hashes[0, :, index, :] - row_start` as int32, as a compiled step."""
    return (hashes[0, :, index, :] - row_start).to(torch.int32)


@torch._dynamo.disable
def run_layers(model, stream, pre_mix, spans, paged, hashes, live_rows):
    """Every backbone block of one forward through its layer kernel. Returns
    `(stream, pre_mix, taps)`. Outside dynamo, with the per-step host data (the
    span list, the host block tables and slot lists `_paged_spans` kept) built and
    read only here."""
    from . import model as M

    step = StepIndices(
        spans, paged, stream.device, hashes,
        [span.host_table for span in paged],
        torch.as_tensor([slot for span in paged for slot in span.host_slots], dtype=torch.int64),
        int(live_rows),
    )
    for layer_id, block in enumerate(model.layers):
        if layer_id % M._HEAP_TRIM_EVERY == 0:
            M.host_heap_trim(f"layer {layer_id}", model.world_size)
        stream, pre_mix = run_layer(model, block, stream, pre_mix, step)
    return stream, pre_mix, step.taps


@torch._dynamo.disable
def run_layer(model, block, stream: torch.Tensor, pre_mix: torch.Tensor, step: StepIndices):
    """One backbone block through its archetype's kernel. Returns `(stream, ffn_pre)`."""
    lw = layer_weights(model, block)
    attn = block.attn
    T = int(stream.shape[0])
    B = len(step.spans)
    phase = "decode" if T <= PT else "prefill"
    if phase == "prefill" and B != 1:
        raise NotImplementedError(f"a {T}-row step over {B} sequences: the prefill kernels serve one")
    idx = step.for_ratio(lw.ratio, lw.window)
    hd = attn.head_dim
    n = dict(lw.named)
    n.update(
        stream=stream, pre_mix=pre_mix, row_pos=idx.row_pos, write_slots=idx.write_slots,
        win_slots=idx.win_slots, twin=idx.twin, wv=idx.wv,
        window_kv_cache=attn.window_kv_cache.view(-1, hd), live=step.live,
        groups_attn=model._mk_groups_attn, groups_world=model._mk_groups_world,
        flags=(B, lw.window, 9, kernel_version()),
    )
    if lw.kv_src:
        step.compress_cache = attn.compress_kv_cache.view(-1, hd)
        step.index_cache = attn.index_k_cache.view(-1, attn.index_k_cache.shape[-1])
        n.update(grp_pos=idx.grp_pos, grp_mslots=idx.grp_mslots, grp_wslots=idx.grp_wslots)
        if lw.ratio == 2:
            n.update(pool_kv=attn.compressor.pool_kv, pool_score=attn.compressor.pool_score)
    if lw.ratio:
        if step.compress_cache is None or step.index_cache is None:
            raise RuntimeError(f"layer {attn.layer_id} reads compressed KV before any kv-source layer ran")
        if attn.is_index_source and attn.indexer.uses_candidates:
            # The candidate-block mask keeps the best `candidate_topk_blocks` blocks of
            # `candidate_block_size` groups; it only removes groups when the context
            # holds more than that, which the kernel does not model.
            limit = int(model.config.candidate_topk_blocks) * int(model.config.candidate_block_size)
            if step.read_len(lw.ratio) > limit:
                raise NotImplementedError(
                    f"the context reads {step.read_len(lw.ratio)} compressed groups, more than the "
                    f"{limit} the candidate-block mask keeps; the kernels assume the mask is a no-op"
                )
        if step.topk is None:
            step.topk = torch.zeros(T, int(model.config.index_topk), dtype=torch.int32, device=step.device)
        n.update(cmp_slots=idx.cmp_slots, cbase=idx.cbase, topk_in=step.topk,
                 compress_kv_cache=step.compress_cache)
    if lw.idx_src:
        n.update(index_k_cache=step.index_cache, clen=idx.clen)
    if lw.engram:
        index = list(model.config.engram_layer_ids).index(attn.layer_id)
        n["hash_ids"] = step.hash_cols(index, block.engram.row_start)
    kernel = layer_kernel.KERNELS[(lw.arch, phase)]
    if _MK_SYNC:
        import time

        t0 = time.perf_counter()
        if model.rank in (0, model.world_size - 1):
            logger.warning("deepseek_v41 MK_SYNC rank %d layer %d %s/%s T=%d B=%d submit",
                           model.rank, attn.layer_id, lw.arch, phase, T, B)
    out = entry(kernel)(*[n[name] for name in params(kernel)])
    if _MK_SYNC:
        out[1].cpu()
        if model.rank in (0, model.world_size - 1):
            logger.warning("deepseek_v41 MK_SYNC rank %d layer %d done in %.1f ms", model.rank,
                           attn.layer_id, 1e3 * (time.perf_counter() - t0))
    out_stream, ffn_pre = out[0], out[1]
    k = 3 + ((4 if lw.ratio == 2 else 2) if lw.kv_src else 0)
    if lw.idx_src:
        step.topk = out[k]
        k += 1
    if lw.tap:
        step.taps.append(out[k])
    _throttle(model, ffn_pre)
    return out_stream, ffn_pre


def _throttle(model, small: torch.Tensor) -> None:
    """Keep at most `_IN_FLIGHT` of this bridge's kernels outstanding.

    The runtime's submission queue holds `NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS`
    (32) executions and a forward submits a kernel a layer, then a drafter kernel (and
    its two small helper graphs) a sequence, with nothing on the host waiting on any of
    them -- at ten sequences that overflowed the queue (`Execution Queue Full`,
    in512_bs10). Reading the oldest outstanding kernel's tiny output back blocks until
    that execution, and every earlier one, has completed; the device never drains.
    The list lives on the model so it spans the layers and the drafts of a step."""
    pending = model.__dict__.setdefault("_mk_pending", [])
    pending.append(small)
    while len(pending) > _IN_FLIGHT:
        pending.pop(0).cpu()


# --------------------------------------------------------------------------- drafter


def _draft_meta(seed, tap, pos):
    return torch.stack([seed.reshape(-1)[:1], tap.reshape(-1)[:1], pos.reshape(-1)[:1]]).to(torch.int32).reshape(1, 3)


def _ids_i64(ids):
    """The drafted ids as int64, as a compiled step (a dtype cast is what the
    backend's eager fallback does not implement)."""
    return ids.to(torch.int64)


def draft_weights(model, with_freqs: bool = True) -> dict:
    """The drafter's static arguments by name (built once; the RoPE table, which
    `bind_kv_cache` builds after the weights load, joins on first use)."""
    dw = getattr(model, "_mk_draft_", None)
    if dw is not None:
        if with_freqs and "freqs" not in dw:
            dw["freqs"] = _freqs_table(model.mtp[0].attn, model.norm.weight.device)
        return dw
    device = model.norm.weight.device
    stages = list(model.mtp)
    s0, last = stages[0], stages[-1]

    def stack(get, dtype=None):
        # Through the host: a one-time copy, and no eager op on the device.
        t = torch.stack([_host(get(s)) for s in stages])
        return _dev(t if dtype is None else t.to(dtype), device)

    n = {
        "main_proj": kmajor_bf16(s0.main_proj), "main_norm_w": _row(s0.main_norm.weight, device),
        "embed_w": model.embed_weight,
        "vocab_start": torch.tensor([[int(model.rank * model.part_vocab_size)]], dtype=torch.int32, device=device),
        "hca_base": stack(lambda s: s.hc_attn.base), "hca_scale": stack(lambda s: s.hc_attn.scale),
        "attn_norm_w": stack(lambda s: s.attn_norm.weight),
        "hcf_base": stack(lambda s: s.hc_ffn.base), "hcf_scale": stack(lambda s: s.hc_ffn.scale),
        "ffn_norm_w": stack(lambda s: s.ffn_norm.weight),
        "wq_a": stack(lambda s: kmajor_bf16(s.attn.wq_a)), "q_norm_w": stack(lambda s: s.attn.q_norm.weight),
        "wq_b": stack(lambda s: kmajor_bf16(s.attn.wq_b)),
        "wkv": stack(lambda s: kmajor_bf16(s.attn.wkv)), "kv_norm_w": stack(lambda s: s.attn.kv_norm.weight),
        "wo_a": stack(lambda s: kmajor_bf16(s.attn.wo_a)), "wo_b": stack(lambda s: kmajor_bf16(s.attn.wo_b)),
        "attn_sink": stack(lambda s: s.attn.attn_sink.float().reshape(-1, 1)),
        "gate_b": stack(lambda s: s.ffn.gate.bias),
        "expert_lo": torch.tensor([[float(s0.ffn.experts_start_idx)]], dtype=torch.float32, device=device),
        "live": torch.tensor([[float(model.config.dspark_block_size)]], dtype=torch.float32, device=device),
        "sh_w1": stack(lambda s: kmajor_bf16(s.ffn.shared_experts.w1)),
        "sh_w3": stack(lambda s: kmajor_bf16(s.ffn.shared_experts.w3)),
        "sh_w2": stack(lambda s: kmajor_bf16(s.ffn.shared_experts.w2)),
        "fnorm_w": _row(last.norm.weight, device),
        "head_w": _t_bf16(model.head_weight, device),
        "mk_embed_w": last.markov_head.embed_weight, "mk_head_w": last.markov_head.head_weight,
        "conf_w": last.confidence_head.proj_weight,
    }
    for i, s in enumerate(stages):
        n[f"hca_fn_t{i}"] = _t_f32(s.hc_attn.fn, device)
        n[f"hcf_fn_t{i}"] = _t_f32(s.hc_ffn.fn, device)
        n[f"gate_w_t{i}"] = _t_f32(s.ffn.gate.weight, device)
    exps = [stacked_experts(s.ffn) for s in stages]
    for key in ("w1", "s1", "w3", "s3", "w2", "s2"):
        n[key] = _dev(torch.stack([_host(e[key]) for e in exps]), device)
    # The per-stage copies are views of the stacks from here on, so the drafter's
    # weights are held once.
    for i, s in enumerate(stages):
        for name, lin in (("wq_a", s.attn.wq_a), ("wq_b", s.attn.wq_b), ("wkv", s.attn.wkv),
                          ("wo_a", s.attn.wo_a), ("wo_b", s.attn.wo_b), ("sh_w1", s.ffn.shared_experts.w1),
                          ("sh_w3", s.ffn.shared_experts.w3), ("sh_w2", s.ffn.shared_experts.w2)):
            lin._kb16 = n[name][i]
        st = s.ffn._mk_stacked
        for key in st:
            st[key] = n[key][i]
        for e, slot in enumerate(s.ffn.cache_slots):
            for lname in ("w1", "w2", "w3"):
                lin = getattr(slot, lname)
                lin.weight = n["w" + lname[1]][i, e]
                lin.scale = n["s" + lname[1]][i, e]
    model._mk_draft_ = n
    if with_freqs:
        n["freqs"] = _freqs_table(s0.attn, device)
    return n


@torch._dynamo.disable
def run_draft(model, seed_ids, tap, main_hidden, span, paged):
    """`forward_spec` for one sequence through its phase's drafter kernel:
    `(output_ids [1, K+1] int64, None, confidence [1, K] fp32)`."""
    from . import model as M

    cfg = model.config
    device = main_hidden.device
    host_table, host_slots = paged.host_table, paged.host_slots
    begin, length, start_pos = span
    main = main_hidden[begin : begin + length]
    meta = M._compiled(_draft_meta)(seed_ids, tap, paged.pos)
    n = dict(draft_weights(model))
    K = int(cfg.dspark_block_size)
    stages = list(model.mtp)
    n.update(
        meta=meta, main_hidden=main,
        row_pos=(start_pos + torch.arange(length)).to(torch.int32).reshape(-1, 1).to(device),
        write_slots=torch.as_tensor(host_slots[:length], dtype=torch.int32).reshape(-1, 1).to(device),
        block_table=host_table.to(torch.int32).reshape(-1, 1).contiguous().to(device),
        cache0=stages[0].attn.window_kv_cache.view(-1, stages[0].attn.head_dim),
        cache1=stages[1].attn.window_kv_cache.view(-1, stages[1].attn.head_dim),
        cache2=stages[2].attn.window_kv_cache.view(-1, stages[2].attn.head_dim),
        groups_attn=model._mk_groups_attn, groups_world=model._mk_groups_world,
        flags=(K, int(stages[0].attn.window_size), 9, kernel_version(),
               int(cfg.dspark_noise_token_id), int(cfg.vocab_size), int(cfg.dspark_num_experts_per_tok),
               int(paged.block_size)),
    )
    kernel = draft_kernel.draft_prefill if paged.at_zero else draft_kernel.draft_decode
    if _MK_SYNC and model.rank in (0, model.world_size - 1):
        logger.warning("deepseek_v41 MK_SYNC rank %d draft %s L=%d submit", model.rank,
                       "prefill" if paged.at_zero else "decode", length)
    out = entry(kernel)(*[n[name] for name in params(kernel)])
    if _MK_SYNC:
        out[0].cpu()
        if model.rank in (0, model.world_size - 1):
            logger.warning("deepseek_v41 MK_SYNC rank %d draft done", model.rank)
    output_ids, confidence = out[0], out[1]
    _throttle(model, confidence)
    return M._compiled(_ids_i64)(output_ids), None, confidence.reshape(1, K)
