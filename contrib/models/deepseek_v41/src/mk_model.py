# SPDX-License-Identifier: Apache-2.0
"""The bridge from `model.py`'s modules to the two megakernels.

    run_layer(model, block, stream, pre_mix, step)   one `Block` = one `layer_forward`
    run_draft(model, seed, tap, main_hidden, span, paged)   `forward_spec` = one `draft_forward`

Weights are gathered once per block into the kernel's argument order
(`LayerWeights`), with the few re-layouts the kernels want (`fn.T` and `gate.T`
as fp32, the routed experts stacked on a leading axis) done on the host at first
use. Per forward, `StepIndices` turns the scheduler's block tables and slot
mapping into the kernels' index columns (`mk_host.build_attn_indices`) and carries
the state the layers hand down the stack: the current compressed/index caches,
the latest top-k, the DSpark taps.

The kernels do their own collectives, so nothing here touches a process group;
the replica groups are passed as constants (consecutive ranks).
"""

from __future__ import annotations

import functools
import glob
import hashlib
import os
from dataclasses import dataclass, field

import torch

from . import mk_host
from .kernels import draft_kernel, layer_kernel

_HERE = os.path.dirname(os.path.abspath(__file__))
_IN_FLIGHT = 12
#: `DEEPSEEK_V41_MK_DEBUG=<layer>:<stage>` runs that layer's kernel at the debug
#: `stage` (see `layer_kernel`'s docstring), saves its `dbg` output from rank 0 to
#: `DEEPSEEK_V41_DUMP_DIR/mk_layer<layer>_stage<stage>.pt` and stops the forward.
_MK_DEBUG = os.environ.get("DEEPSEEK_V41_MK_DEBUG", "")


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
def entry(name: str):
    """The compiled entry point of one megakernel, launched on both cores."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from . import model as M

    M._raise_dynamo_limits()
    fn = {"layer": layer_kernel.layer_forward, "draft": draft_kernel.draft_forward}[name]
    wrapped = wrap_nki(fn)[2]

    def run(*args):
        return wrapped(*args)

    run.__name__ = f"nki_mk_{name}"
    return torch.compile(run, backend=M._compile_backend(), fullgraph=True, options=M._COMPILE_OPTIONS)


def _host(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu()


def _dev(t: torch.Tensor, device) -> torch.Tensor:
    return t.contiguous().to(device)


def _row(t: torch.Tensor, device) -> torch.Tensor:
    """A 1-D buffer as the `[1, n]` row the kernels broadcast from."""
    return _dev(_host(t).reshape(1, -1), device)


def _t_f32(t: torch.Tensor, device) -> torch.Tensor:
    """`w.T` as fp32, for the helpers that stage a weight transposed."""
    return _dev(_host(t).t().float(), device)


def _freqs_table(freqs_cis, device) -> torch.Tensor:
    """`(cos, sin)` -> `[max_pos, 64]` fp32 (cos | sin)."""
    cos, sin = freqs_cis
    return _dev(torch.cat([_host(cos).float(), _host(sin).float()], -1), device)


@dataclass
class Dummies:
    """Placeholders for the arguments a layer archetype does not read."""

    i32: torch.Tensor
    u8: torch.Tensor
    bf16: torch.Tensor
    f32: torch.Tensor

    @staticmethod
    def on(device) -> "Dummies":
        return Dummies(
            i32=torch.zeros(1, 1, dtype=torch.int32, device=device),
            u8=torch.zeros(1, 1, dtype=torch.uint8, device=device),
            bf16=torch.zeros(1, 1, dtype=torch.bfloat16, device=device),
            f32=torch.zeros(1, 1, dtype=torch.float32, device=device),
        )


@dataclass
class LayerWeights:
    """One block's weights in `layer_forward`'s argument order, in three runs:
    `hc_attn` (before the index columns), `compress` (between the caches and the
    engram), `moe` (after the engram)."""

    hc_attn: list
    compress: list
    engram: list
    moe: list
    ratio: int
    is_kv_src: bool
    is_idx_src: bool
    has_engram: bool
    tap: bool
    window: int


def _fp8(linear) -> tuple[torch.Tensor, torch.Tensor]:
    assert linear.storage == "fp8", linear.storage
    return linear.weight, linear.scale


def layer_weights(model, block) -> LayerWeights:
    """Gather one block's weights (see `LayerWeights`)."""
    device = model.norm.weight.device
    attn, ffn = block.attn, block.ffn
    cfg = model.config
    layer_id = attn.layer_id
    d = model._mk_dummies
    hc_attn = [
        _t_f32(block.hc_attn.fn, device), _row(block.hc_attn.base, device), _row(block.hc_attn.scale, device),
        _row(block.attn_norm.weight, device),
        _t_f32(block.hc_ffn.fn, device), _row(block.hc_ffn.base, device), _row(block.hc_ffn.scale, device),
        _row(block.ffn_norm.weight, device),
        *_fp8(attn.wq_a), _row(attn.q_norm.weight, device), *_fp8(attn.wq_b), *_fp8(attn.wkv),
        _row(attn.kv_norm.weight, device), *_fp8(attn.wo_a), *_fp8(attn.wo_b),
        _dev(_host(attn.attn_sink).float().reshape(-1, 1), device),
        model._mk_freqs(attn),
    ]
    # Compressor / indexer weights. Layers that are not a source still need
    # tensors of the right rank; they get the nearest source's (unread).
    src = attn if attn.is_kv_source else model._mk_nearest_source(layer_id, "kv")
    isrc = attn if attn.is_index_source else model._mk_nearest_source(layer_id, "index")
    comp = src.compressor
    compress = [
        comp.wkv.weight, _t_f32(comp.wkv.weight, device),
        _t_f32(comp.wgate.weight, device) if comp.compress_ratio > 1 else d.f32,
        _row(comp.norm.weight, device),
    ]
    ix = isrc.indexer
    ixk = (src.indexer if src.indexer is not None and src.indexer.owns_k else model._mk_nearest_source(layer_id, "kv").indexer)
    index = [
        *_fp8(ix.wq_b), _t_f32(ix.weights_proj.weight, device), ixk.wk.weight, _row(ixk.k_norm.weight, device),
    ]
    engram = [d.bf16, d.i32, d.u8, d.u8, d.f32]
    if block.engram is not None:
        eg = block.engram
        if not eg.resident:
            raise RuntimeError(
                f"layer {layer_id}: the engram table is not resident on the device, and the "
                "megakernel reads it there. Raise DEEPSEEK_V41_ENGRAM_TABLE_BYTES."
            )
        engram = [
            # row_start 0: the ids reach the kernel already local to the shard (see
            # `StepIndices.hash_cols`).
            eg.table, torch.zeros(1, 1, dtype=torch.int32, device=device),
            *_fp8(eg.wkv),
            _dev((_host(eg.q_weight).float() * _host(eg.k_weight).float()).reshape(1, -1), device),
        ]
    stacked = getattr(ffn, "stacked", None)
    if stacked is None or not ffn.resident:
        raise RuntimeError(
            f"layer {layer_id}: the routed experts are not resident as one stacked buffer "
            "(the expert pool streams). The megakernel needs them on the device; raise "
            "DEEPSEEK_V41_EXPERT_CACHE_BYTES."
        )
    moe = [
        _t_f32(ffn.gate.weight, device), _row(ffn.gate.bias, device),
        torch.tensor([[float(ffn.experts_start_idx)]], dtype=torch.float32, device=device),
        stacked["w1.weight"], stacked["w1.scale"], stacked["w3.weight"], stacked["w3.scale"],
        stacked["w2.weight"], stacked["w2.scale"],
        *_fp8(ffn.shared_experts.w1), *_fp8(ffn.shared_experts.w3), *_fp8(ffn.shared_experts.w2),
    ]
    return LayerWeights(
        hc_attn=hc_attn, compress=compress + index, engram=engram, moe=moe,
        ratio=int(attn.compress_ratio), is_kv_src=bool(attn.is_kv_source),
        is_idx_src=bool(attn.is_index_source), has_engram=block.engram is not None,
        tap=layer_id in set(cfg.dspark_target_layer_ids), window=int(attn.window_size),
    )


@dataclass
class StepIndices:
    """Per-forward index columns and the state the layers hand down the stack."""

    spans: list
    paged: list
    device: torch.device
    hashes: torch.Tensor | None = None
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
        self.tables = [_host(p.block_table).to(torch.int64) for p in self.paged]
        self.slot_mapping = torch.cat([_host(p.slot_mapping).reshape(-1) for p in self.paged]).to(torch.int64)
        lengths = {length for _, length, _ in self.spans}
        if len(lengths) != 1:
            raise NotImplementedError(
                f"the layer megakernel wants every span of a step to have the same length, got "
                f"{sorted(lengths)} (a mixed prefill/decode step). Serve with max_num_seqs=1."
            )

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

    def hash_cols(self, index: int, row_start: int) -> torch.Tensor:
        """This engram layer's `[T, 24]` int32 row ids, **local to this rank's shard**
        (`global - row_start`). The kernel's gather turns ids into fp32 for the range
        mask, and a global id of this 384M-row table is not exactly representable
        above 2^24; a local id is below the shard width (6M), so it is exact, and an
        out-of-range one only has to keep its sign / stay out of range."""
        key = (index, row_start)
        t = self._hash_cols.get(key)
        if t is None:
            if self.hashes is None:
                raise RuntimeError("an engram layer ran but the step has no n-gram hashes")
            t = (_host(self.hashes)[0, :, index, :].to(torch.int64) - int(row_start)).to(torch.int32)
            t = t.contiguous().to(self.device)
            self._hash_cols[key] = t
        return t


@torch._dynamo.disable
def make_step(spans, paged, device, hashes) -> StepIndices:
    """`StepIndices` for this forward, built outside the frame dynamo traces
    (the index math is host work on CPU copies of the block tables)."""
    return StepIndices(spans, paged, device, hashes)


@torch._dynamo.disable
def run_layer(model, block, stream: torch.Tensor, pre_mix: torch.Tensor, step: StepIndices):
    """One backbone block through `layer_forward`. Returns `(stream, ffn_pre)`.

    Outside dynamo (`torch._dynamo.disable`, as `MoE.forward` is): the index
    columns are host work, and the kernel entry is its own `torch.compile`d
    function, which compiles from here as the per-op steps do."""
    lw = model._mk_layer(block)
    d = model._mk_dummies
    T = int(stream.shape[0])
    B = len(step.spans)
    idx = step.for_ratio(lw.ratio, lw.window)
    attn = block.attn
    if lw.ratio:
        if step.topk is None:
            step.topk = torch.zeros(T, int(model.config.index_topk), dtype=torch.int32, device=step.device)
        topk_in = step.topk
    else:
        topk_in = torch.zeros(T, 1, dtype=torch.int32, device=step.device)
    if lw.is_kv_src:
        step.compress_cache = attn.compress_kv_cache
        step.index_cache = attn.index_k_cache
        if lw.ratio == 2:
            pool_kv, pool_score = attn.compressor.pool_kv, attn.compressor.pool_score
        else:
            pool_kv, pool_score = d.f32, d.f32
        grp = [idx.grp_pos, idx.grp_mslots, idx.grp_wslots]
    else:
        pool_kv, pool_score = d.f32, d.f32
        grp = [d.i32, d.i32, d.i32]
    if lw.ratio and (step.compress_cache is None or step.index_cache is None):
        raise RuntimeError(f"layer {attn.layer_id} reads compressed KV before any kv-source layer ran")
    compress_cache = step.compress_cache if lw.ratio else d.bf16
    index_cache = step.index_cache if lw.ratio else d.bf16
    if lw.is_idx_src and attn.indexer.uses_candidates:
        # The candidate-block mask of layer 20 keeps the best `candidate_topk_blocks`
        # blocks of `candidate_block_size` groups; it only removes groups when the
        # context holds more than that, which the kernel does not model.
        limit = int(model.config.candidate_topk_blocks) * int(model.config.candidate_block_size)
        if step.read_len(lw.ratio) > limit:
            raise NotImplementedError(
                f"the context reads {step.read_len(lw.ratio)} compressed groups, more than the "
                f"{limit} the candidate-block mask keeps; the megakernel assumes the mask is a no-op"
            )
    hash_ids = d.i32
    engram = lw.engram
    if lw.has_engram:
        hash_ids = step.hash_cols(list(model.config.engram_layer_ids).index(attn.layer_id),
                                  int(block.engram.row_start))
    stage = 9
    debug = None
    if _MK_DEBUG:
        dbg_layer, dbg_stage = (int(v) for v in _MK_DEBUG.split(":"))
        if dbg_layer == attn.layer_id:
            stage, debug = dbg_stage, (dbg_layer, dbg_stage)
    flags = (lw.ratio, B, lw.window, stage, int(lw.is_kv_src), int(lw.is_idx_src), int(lw.has_engram),
             int(lw.tap), kernel_version())
    args = [
        stream, pre_mix, *lw.hc_attn,
        idx.row_pos, idx.write_slots, idx.win_slots, idx.cmp_slots, idx.twin, idx.wv, idx.cbase, topk_in,
        attn.window_kv_cache.view(-1, attn.head_dim), compress_cache, pool_kv, pool_score, index_cache,
        *lw.compress[:4], *grp, *lw.compress[4:], idx.clen if lw.ratio else d.i32,
        hash_ids, *engram, *lw.moe,
        model._mk_groups_attn, model._mk_groups_world, flags,
    ]
    out = entry("layer")(*args)
    if debug is not None:
        dump_dir = os.environ.get("DEEPSEEK_V41_DUMP_DIR", ".")
        if model.rank == 0:
            os.makedirs(dump_dir, exist_ok=True)
            path = os.path.join(dump_dir, f"mk_layer{debug[0]}_stage{debug[1]}.pt")
            torch.save({"dbg": out[0].cpu(), "stream_in": stream.cpu(), "pre_mix_in": pre_mix.cpu()}, path)
        raise RuntimeError(f"DEEPSEEK_V41_MK_DEBUG: layer {debug[0]} stage {debug[1]} dumped; stopping")
    out_stream, ffn_pre, topk_out, tap_out = out[0], out[1], out[2], out[3]
    # The runtime's submission queue holds `NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS`
    # (32) executions and a forward submits one per layer with nothing on the host
    # waiting on any of them (the per-op path got a sync a layer from the router's
    # `.cpu()`): keep at most `_IN_FLIGHT` layers outstanding by reading the oldest
    # one's tiny `ffn_pre` back, which blocks until that execution -- and every
    # earlier one on the queue -- has completed. The device never drains: the host
    # is a dozen layers ahead of it.
    step.pending.append(ffn_pre)
    if len(step.pending) > _IN_FLIGHT:
        step.pending.pop(0).cpu()
    if lw.is_idx_src:
        step.topk = topk_out
    if lw.tap:
        step.taps.append(tap_out)
    return out_stream, ffn_pre


def _draft_meta(seed, tap, pos):
    return torch.stack([seed.reshape(-1)[:1], tap.reshape(-1)[:1], pos.reshape(-1)[:1]]).to(torch.int32).reshape(1, 3)


def _ids_i64(ids):
    """The drafted ids as int64, as a compiled step: a dtype cast is the one
    thing the backend's eager fallback does not implement (see `_moe_narrow`)."""
    return ids.to(torch.int64)


def draft_weights(model) -> list:
    """The drafter's weights in `draft_forward`'s argument order (after the index
    columns, before the caches), plus the head's, as two lists."""
    device = model.norm.weight.device
    stages = list(model.mtp)
    s0, last = stages[0], stages[-1]

    def stack(get):
        # Through the host: a one-time copy, and no eager op on the device.
        return _dev(torch.stack([_host(get(s)) for s in stages]), device)

    stack_host = stack

    head = [
        *_fp8(s0.main_proj), _row(s0.main_norm.weight, device), model.embed_weight,
        torch.tensor([[int(model.rank * model.part_vocab_size)]], dtype=torch.int32, device=device),
    ]
    hc = [
        *[_t_f32(s.hc_attn.fn, device) for s in stages],
        stack_host(lambda s: s.hc_attn.base), stack_host(lambda s: s.hc_attn.scale),
        stack(lambda s: s.attn_norm.weight),
        *[_t_f32(s.hc_ffn.fn, device) for s in stages],
        stack_host(lambda s: s.hc_ffn.base), stack_host(lambda s: s.hc_ffn.scale),
        stack(lambda s: s.ffn_norm.weight),
        stack(lambda s: s.attn.wq_a.weight), stack(lambda s: s.attn.wq_a.scale), stack(lambda s: s.attn.q_norm.weight),
        stack(lambda s: s.attn.wq_b.weight), stack(lambda s: s.attn.wq_b.scale),
        stack(lambda s: s.attn.wkv.weight), stack(lambda s: s.attn.wkv.scale), stack(lambda s: s.attn.kv_norm.weight),
        stack(lambda s: s.attn.wo_a.weight), stack(lambda s: s.attn.wo_a.scale),
        stack(lambda s: s.attn.wo_b.weight), stack(lambda s: s.attn.wo_b.scale),
        stack_host(lambda s: s.attn.attn_sink.float().reshape(-1, 1)),
        model._mk_freqs(s0.attn),
    ]
    for s in stages:
        if getattr(s.ffn, "stacked", None) is None or not s.ffn.resident:
            raise RuntimeError("the DSpark stages' routed experts are not resident as stacked buffers")
    moe = [
        *[_t_f32(s.ffn.gate.weight, device) for s in stages],
        stack_host(lambda s: s.ffn.gate.bias),
        torch.tensor([[float(s0.ffn.experts_start_idx)]], dtype=torch.float32, device=device),
        *[stack(lambda s, k=k: s.ffn.stacked[k]) for k in ("w1.weight", "w1.scale", "w3.weight", "w3.scale", "w2.weight", "w2.scale")],
        stack(lambda s: s.ffn.shared_experts.w1.weight), stack(lambda s: s.ffn.shared_experts.w1.scale),
        stack(lambda s: s.ffn.shared_experts.w3.weight), stack(lambda s: s.ffn.shared_experts.w3.scale),
        stack(lambda s: s.ffn.shared_experts.w2.weight), stack(lambda s: s.ffn.shared_experts.w2.scale),
    ]
    tail = [
        _row(last.norm.weight, device), model.head_weight,
        last.markov_head.embed_weight, last.markov_head.head_weight, last.confidence_head.proj_weight,
    ]
    return head + hc + moe, tail


@torch._dynamo.disable
def run_draft(model, seed_ids, tap, main_hidden, span, paged):
    """`forward_spec` for one sequence through `draft_forward`:
    `(output_ids [1, K+1] int64, None, confidence [1, K] fp32)`."""
    from . import model as M

    cfg = model.config
    device = main_hidden.device
    begin, length, start_pos = span
    main = main_hidden[begin : begin + length]
    meta = M._compiled(_draft_meta)(seed_ids, tap, paged.pos)
    table = _host(paged.block_table).to(torch.int32).reshape(-1, 1).contiguous().to(device)
    row_pos = (start_pos + torch.arange(length)).to(torch.int32).reshape(-1, 1).to(device)
    write_slots = _host(paged.slot_mapping).reshape(-1)[:length].to(torch.int32).reshape(-1, 1).contiguous().to(device)
    weights, tail = model._mk_draft()
    K = int(cfg.dspark_block_size)
    stages = list(model.mtp)
    flags = (int(paged.at_zero), K, int(stages[0].attn.window_size), 9, kernel_version(),
             int(cfg.dspark_noise_token_id), int(cfg.vocab_size), int(cfg.dspark_num_experts_per_tok),
             int(paged.block_size))
    caches = [s.attn.window_kv_cache.view(-1, s.attn.head_dim) for s in stages]
    args = [meta, main, row_pos, write_slots, table, *weights, *caches, *tail,
            model._mk_groups_attn, model._mk_groups_world, flags]
    out = entry("draft")(*args)
    output_ids, confidence = out[0], out[1]
    return M._compiled(_ids_i64)(output_ids), None, confidence.reshape(1, K)
