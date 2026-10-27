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
import logging
import os
from dataclasses import dataclass, field

import torch

from . import mk_host
from .kernels import mk_kernels

_HERE = os.path.dirname(os.path.abspath(__file__))
_mk_log = logging.getLogger(__name__)
_IN_FLIGHT = int(os.environ.get("DEEPSEEK_V41_MK_IN_FLIGHT", "12"))
#: Small outputs of the megakernel launches submitted by this process and not yet
#: known complete, oldest first, **across forwards**: the runtime's submission queue
#: holds `NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS` (32) executions, and under
#: pipeline parallelism a stage's host launches batch after batch without ever
#: waiting on the device (the first stage with eight prefills in flight queued 64
#: executions and died with "Failed to schedule neff execution ... Execution Queue
#: Full", 2026-10-24), so the window has to span the steps, not sit inside one.
#: The compiled graphs around the kernels (the forward's tail, the drafter's
#: prologue / epilogue) sit on the same queue untracked: 12 tracked launches plus
#: a tail's worth of graphs stays under 32 now that a step drafts in one launch
#: (ten sequences drafting one by one, 3 untracked graphs each, overflowed at 12
#: on 2026-10-24; the window went to 8 for a while and cost the flat scheme ~2 ms
#: a token).
_pending: list = []


def _track(small: torch.Tensor) -> None:
    """Keep at most `_IN_FLIGHT` launches outstanding: read the oldest one's tiny
    output back, which blocks until that execution -- and every earlier one on the
    queue -- has completed. The device never drains: the host stays a dozen
    launches ahead of it."""
    _pending.append(small)
    if len(_pending) > _IN_FLIGHT:
        _pending.pop(0).cpu()
#: `DEEPSEEK_V41_MK_TRACE=1`: log every megakernel launch on each stage's first rank (which entry,
#: the step's phase and shape), to pin a device hang on one kernel.
_MK_TRACE = os.environ.get("DEEPSEEK_V41_MK_TRACE", "") not in ("", "0")
#: The scheme is read back from vLLM's parallel groups (`PipePlan.from_groups`):
#: "tp64" when the engine runs one tensor-parallel group, "tp8pp8" when it runs
#: `pipeline_parallel_size` stages of `tensor_parallel_size` ranks (`config.SCHEME`
#: sizes the engine arguments). Under "tp8pp8" each stage owns 5 blocks, builds and
#: loads only those, reduces over the stage, and hands its activations to the next
#: stage through `pp_kernels.boundary_body` (a 2-rank collective per tensor-parallel
#: offset); vLLM's scheduler keeps `pipeline_parallel_size` batches in flight.


@dataclass
class PipePlan:
    """Where this rank sits in the deployment scheme and which blocks it owns."""

    scheme: str
    world: int
    rank: int
    tp: int                     # ranks a stage (the world under tp64)
    pp: int                     # stages
    stage: int
    stage_rank: int
    block_stage: list           # stage of each backbone block

    @staticmethod
    def build(scheme: str, world: int, rank: int, n_layers: int, attn_tp: int) -> "PipePlan":
        if scheme == "tp64":
            return PipePlan(scheme, world, rank, world, 1, 0, rank, [0] * n_layers)
        pp = 8
        if world % pp:
            raise ValueError(f"tp8pp8 wants a world divisible by 8, got {world}")
        tp = world // pp
        if tp != attn_tp:
            raise ValueError(f"tp8pp8: a stage is {tp} ranks but the attention shards {attn_tp} ways")
        block_stage = [i * pp // n_layers for i in range(n_layers)]
        return PipePlan(scheme, world, rank, tp, pp, rank // tp, rank % tp, block_stage)

    @staticmethod
    def from_groups(n_layers: int) -> "PipePlan":
        """The plan vLLM's parallel groups imply: a stage is one tensor-parallel group
        (consecutive global ranks), the pipeline rank is the stage. Global rank ids are
        what the in-kernel collectives' replica groups name, so `rank` is global."""
        import torch.distributed as dist
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group

        tp_group, pp_group = get_tp_group(), get_pp_group()
        tp, pp = int(tp_group.world_size), int(pp_group.world_size)
        stage, stage_rank = int(pp_group.rank_in_group), int(tp_group.rank_in_group)
        world = tp * pp
        rank = stage * tp + stage_rank
        if dist.is_initialized() and dist.get_rank() != rank:
            raise RuntimeError(
                f"rank {dist.get_rank()}: vLLM placed it at tensor-parallel rank {stage_rank} of "
                f"pipeline stage {stage}, which is not the consecutive layout "
                f"(stage * tp + tp_rank = {rank}) the kernels' replica groups assume"
            )
        if pp == 1:
            return PipePlan("tp64", world, rank, world, 1, 0, rank, [0] * n_layers)
        scheme = "tp8pp8"
        if (tp, pp) != (8, 8):
            raise ValueError(f"the pipeline scheme is written for 8 stages x 8 ranks, got pp={pp} tp={tp}")
        block_stage = [i * pp // n_layers for i in range(n_layers)]
        return PipePlan(scheme, world, rank, tp, pp, stage, stage_rank, block_stage)

    def owns(self, layer_id: int) -> bool:
        return self.block_stage[layer_id] == self.stage

    def owned(self) -> list:
        """The blocks of this stage, in order."""
        return [i for i, st in enumerate(self.block_stage) if st == self.stage]

    @property
    def is_first(self) -> bool:
        return self.stage == 0

    @property
    def is_last(self) -> bool:
        return self.stage == self.pp - 1

    @property
    def first_block(self) -> int:
        return self.owned()[0]

    @property
    def last_block(self) -> int:
        return self.owned()[-1]

    def ends_stage(self, layer_id: int) -> bool:
        n = len(self.block_stage)
        return layer_id + 1 < n and self.block_stage[layer_id + 1] != self.block_stage[layer_id]

    @property
    def lead_last(self) -> int:
        """The rank that broadcasts the final stream: the first rank of the last stage."""
        return (self.pp - 1) * self.tp

    def groups_stage(self):
        return mk_host.replica_groups(self.world, self.tp)

    def groups_full(self):
        return mk_host.replica_groups(self.world, self.world)

    def boundary_groups(self, parity: int):
        """32 pairs `(8s + t, 8(s+1) + t)` for the sending stages `s` of this parity
        (the odd layout pairs stage 7 with stage 0: a pair where nobody sends)."""
        pairs = []
        for st in range(parity, self.pp, 2):
            nxt = (st + 1) % self.pp
            for t in range(self.tp):
                pairs.append((st * self.tp + t, nxt * self.tp + t))
        return tuple(sorted(pairs))

    def boundary_role(self, sending_stage: int) -> int:
        if self.stage == sending_stage:
            return 1
        if self.stage == sending_stage + 1:
            return 2
        return 0
#: `DEEPSEEK_V41_MK_DEBUG=<layer>:<stage>` runs that layer's kernel at the debug
#: `stage` (see `layer_kernel`'s docstring), saves its `dbg` output from rank 0 to
#: `DEEPSEEK_V41_DUMP_DIR/mk_layer<layer>_stage<stage>.pt` and stops the forward.
_MK_DEBUG = os.environ.get("DEEPSEEK_V41_MK_DEBUG", "")
#: `DEEPSEEK_V41_KCAPTURE=<dir>`: record one block kernel launch -- every argument
#: before the launch and every output after it -- on every rank, to
#: `<dir>/<tag>/L<layer>/rank<rank>.pt`, for standalone kernel work. Armed by
#: `<dir>/ARM.json` (`{"tag", "layers", "phase", "nth"}` plus any of `"start"`,
#: `"min_start"`, `"nspans"`, written by the driver while the server is idle): the
#: `nth` forward of `phase` after the arm whose first span starts at position `start`,
#: whose every span starts at or after `min_start` and which has `nspans` spans (each
#: filter only when given) is captured, at each of `layers`.
_KCAPTURE = os.environ.get("DEEPSEEK_V41_KCAPTURE", "")
_kcap = {"arm": None, "mtime": None, "step": None, "count": 0, "done": set()}


def _kcapture_target(model, layer_id: int, step) -> dict | None:
    """The arm spec when this launch is the one to record, else None."""
    path = os.path.join(_KCAPTURE, "ARM.json")
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    if mtime != _kcap["mtime"]:
        import json

        with open(path) as fh:
            _kcap.update(arm=json.load(fh), mtime=mtime, step=None, count=0, done=set())
    arm = _kcap["arm"]
    starts = [int(s[2]) for s in step.spans]
    if (step.phase != arm["phase"]
            or (arm.get("start") is not None and starts[0] != int(arm["start"]))
            or (arm.get("min_start") is not None and min(starts) < int(arm["min_start"]))
            or (arm.get("nspans") is not None and len(starts) != int(arm["nspans"]))):
        return None
    if _kcap["step"] is not step:
        _kcap["step"] = step
        _kcap["count"] += 1
    if _kcap["count"] != int(arm["nth"]) or layer_id not in arm["layers"] or layer_id in _kcap["done"]:
        return None
    _kcap["done"].add(layer_id)
    return arm


def _kcapture_save(model, arm, kernel, layer_id: int, step, args, out) -> None:
    from .kernels import mk_kernels as _mkk

    names = _mkk._positional_names(_mkk.pkg(model.mk_plan.scheme).layer_kernel.layer_body)
    rec = {
        "kernel": kernel.func.__name__, "names": names, "args": args,
        "outs": [o.detach().cpu() for o in out],
        "meta": {"layer": layer_id, "rank": int(model.rank), "world": int(model.world_size),
                 "phase": step.phase, "spans": [tuple(int(v) for v in s) for s in step.spans],
                 "T": int(step.T), "arm": arm},
    }
    d = os.path.join(_KCAPTURE, arm["tag"], f"L{layer_id:02d}")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"rank{int(model.rank):02d}.pt")
    torch.save(rec, path + ".tmp")
    os.replace(path + ".tmp", path)
    _mk_log.warning("kcapture: %s layer %d rank %d -> %s", kernel.func.__name__, layer_id, model.rank, d)


@functools.lru_cache(maxsize=None)
def kernel_version(kind: str = "block", scheme: str = "tp64") -> int:
    """A hash of the kernel source files a kernel is built from, folded into its
    constant `flags`: the NKI compile caches key a kernel on its *own* function
    source, so an edit to a helper module would otherwise be served a stale NEFF.
    The files are the scheme's own package (`kernels/<scheme>`: each scheme has
    its own copies of every megakernel module, so an edit for one scheme never
    touches the other's NEFFs). `kind` "block": the block kernel and the helper
    modules; "draft": the drafter and the helpers; "pp": the stage boundary and
    the helpers it imports. Each kind leaves out the other kinds' own modules, so
    an edit to the drafter or the boundary does not invalidate the 40 block
    kernels' NEFFs (and vice versa). An optimized block kernel's kind is its package
    ("r2_prefill", "r2_decode", "r1_prefill", "r1_decode": `mk_kernels.OPTIMIZED_KINDS`),
    which has its own helper copies in `kernels/<scheme>/<kind>/`; the hash covers that
    package only (`mk_kernels.version_kind`)."""
    h = hashlib.sha256()
    kdir = os.path.join(_HERE, "kernels", scheme)
    own = {"block": ("layer_kernel.py",), "draft": ("draft_kernel.py",), "pp": ("pp_kernels.py",)}
    if kind not in own and kind not in mk_kernels.OPTIMIZED_KINDS:
        raise ValueError(f"kernel_version: unknown kind {kind!r}")
    if not os.path.isdir(kdir):
        raise ValueError(f"kernel_version: no kernel package for scheme {scheme!r}")
    if kind not in own:
        files = sorted(glob.glob(os.path.join(kdir, kind, "*.py")))
        if not files:
            raise ValueError(f"kernel_version: no {kind} package for scheme {scheme!r}")
    else:
        files = sorted(glob.glob(os.path.join(kdir, "*.py")))
        others = {os.path.join(kdir, name) for k, names in own.items() if k != kind for name in names}
        files = [f for f in files if f not in others]
    files += [os.path.join(_HERE, "mk_host.py")]
    for path in files:
        with open(path, "rb") as fh:
            h.update(fh.read())
    return int.from_bytes(h.digest()[:4], "little") & 0x7FFFFFFF


@functools.lru_cache(maxsize=None)
def entry(kernel):
    """The compiled entry point of one specialized megakernel (`mk_kernels`), launched
    on both cores."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from . import model as M

    M._raise_dynamo_limits()
    wrapped = wrap_nki(kernel)[2]

    def run(*args):
        return wrapped(*args)

    run.__name__ = f"nki_{kernel.func.__name__}"
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


def mx_fp8(weight: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """An fp8 `[n, k]` e4m3 weight with `[n/32, k/32]` e8m0 scales, re-laid for the
    tensor engine's MX matmul (`kernels/<scheme>/mk_mx.py`): `[k/4, n]` int32 words packing
    `k = 4p + j` in byte `j`, and `[k/32, n]` scale bytes (one per k-group and column).
    CPU in, CPU out."""
    n, k = weight.shape
    w = weight.reshape(n, k // 4, 4).permute(1, 0, 2).contiguous().view(torch.int32).reshape(k // 4, n)
    s = scale[: n // 32, : k // 32].t().repeat_interleave(32, 1).contiguous()
    return w, s


def mx_fp4(packed: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """An fp4 `[n, k/2]` e2m1-pair weight with `[n, k/32]` e8m0 scales: `[k/4, n]` int16
    words (nibble `j` = `k = 4p + j`) and `[k/32, n]` scale bytes."""
    n, k2 = packed.shape
    w = packed.reshape(n, k2 // 2, 2).permute(1, 0, 2).contiguous().view(torch.int16).reshape(k2 // 2, n)
    s = scale[:, : k2 // 16].t().contiguous()
    return w, s


def _fp8(linear) -> tuple[torch.Tensor, torch.Tensor]:
    """The MX layout of an fp8 linear's weight, converted once on the host and kept
    on the device beside the original (`linear._mx`)."""
    assert linear.storage == "fp8", linear.storage
    mx = getattr(linear, "_mx", None)
    if mx is None:
        w, s = mx_fp8(_host(linear.weight), _host(linear.scale))
        mx = (w.to(linear.weight.device), s.to(linear.weight.device))
        linear._mx = mx
    return mx


def _experts_mx(ffn) -> dict:
    """The resident expert pool in the MX layout: `w1`/`w3` `[E, dim/4, inter]` int16
    with `[E, dim/32, inter]` scales, `w2` `[E, inter/4, dim]` with `[E, inter/32, dim]`.
    Converted once; the original stacked buffers are released afterwards (the
    megakernel is the only reader of a resident pool), which keeps the device
    footprint where it was."""
    mx = getattr(ffn, "stacked_mx", None)
    if mx is not None:
        return mx
    stacked = ffn.stacked
    device = stacked["w1.weight"].device
    mx = {}
    for name in ("w1", "w2", "w3"):
        ws, ss = [], []
        wh, sh = _host(stacked[f"{name}.weight"]), _host(stacked[f"{name}.scale"])
        for e in range(wh.shape[0]):
            w, s = mx_fp4(wh[e], sh[e])
            ws.append(w)
            ss.append(s)
        mx[f"{name}.weight"] = torch.stack(ws).to(device)
        mx[f"{name}.scale"] = torch.stack(ss).to(device)
    ffn.stacked_mx = mx
    # release the per-op layout: the slot views first, then the stacks
    for slot in ffn.cache_slots:
        for name in ("w1", "w2", "w3"):
            linear = getattr(slot, name)
            linear.weight = None
            linear.scale = None
    ffn.stacked = {k: None for k in stacked}
    return mx


def layer_weights(model, block) -> LayerWeights:
    """Gather one block's weights (see `LayerWeights`)."""
    device = model.device
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
    if (stacked is None and getattr(ffn, "stacked_mx", None) is None) or not ffn.resident:
        raise RuntimeError(
            f"layer {layer_id}: the routed experts are not resident as one stacked buffer "
            "(the expert pool streams). The megakernel needs them on the device; raise "
            "DEEPSEEK_V41_EXPERT_CACHE_BYTES."
        )
    stacked = _experts_mx(ffn)
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
    #: rows a span of a decode step has (1 committed + the draft block)
    decode_rows: int = 6
    T: int = 0
    phase: str = "prefill"
    _by_ratio: dict = field(default_factory=dict)
    _hash_cols: dict = field(default_factory=dict)
    topk: torch.Tensor | None = None
    compress_cache: torch.Tensor | None = None
    index_cache: torch.Tensor | None = None
    taps: list = field(default_factory=list)

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
        self.phase = "decode" if lengths.pop() <= self.decode_rows else "prefill"

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
def make_step(spans, paged, device, hashes, decode_rows=6) -> StepIndices:
    """`StepIndices` for this forward, built outside the frame dynamo traces
    (the index math is host work on CPU copies of the block tables)."""
    return StepIndices(spans, paged, device, hashes, decode_rows=decode_rows)


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
    plan = model.mk_plan
    hash_ids = d.i32
    engram = lw.engram
    # The engram read-out is in-kernel under both schemes: the table is sharded over
    # the ranks of `_mk_groups_world` (the world under tp64, the stage under tp8pp8).
    in_kernel_engram = lw.has_engram
    if in_kernel_engram:
        hash_ids = step.hash_cols(list(model.config.engram_layer_ids).index(attn.layer_id),
                                  int(block.engram.row_start))
    else:
        engram = [d.bf16, d.i32, d.u8, d.u8, d.f32]
    stage = 9
    debug = None
    if _MK_DEBUG:
        dbg_layer, dbg_stage = (int(v) for v in _MK_DEBUG.split(":"))
        if dbg_layer == attn.layer_id:
            stage, debug = dbg_stage, (dbg_layer, dbg_stage)
    arch = (lw.ratio, int(lw.is_kv_src), int(lw.is_idx_src), int(in_kernel_engram), int(lw.tap))
    kver = kernel_version(mk_kernels.version_kind(step.phase, arch, plan.scheme), plan.scheme)
    flags = (lw.ratio, B, lw.window, stage, arch[1], arch[2], arch[3], arch[4], kver)
    args = [
        stream, pre_mix, *lw.hc_attn,
        idx.row_pos, idx.write_slots, idx.win_slots, idx.cmp_slots, idx.twin, idx.wv, idx.cbase, topk_in,
        attn.window_kv_cache.view(-1, attn.head_dim), compress_cache, pool_kv, pool_score, index_cache,
        *lw.compress[:4], *grp, *lw.compress[4:], idx.clen if lw.ratio else d.i32,
        hash_ids, *engram, *lw.moe,
        model._mk_groups_attn, model._mk_groups_world, flags,
    ]
    kernel = mk_kernels.layer(step.phase, arch, plan.scheme)
    if _MK_TRACE and plan.stage_rank == 0:
        # (the kind tells an optimized kernel from the generic body's entry of the same name)
        _mk_log.warning("mk launch %s [%s] layer=%d T=%d B=%d spans=%s", kernel.func.__name__,
                        mk_kernels.version_kind(step.phase, arch, plan.scheme), attn.layer_id, T, B,
                        step.spans)
    kcap = _kcapture_target(model, attn.layer_id, step) if _KCAPTURE else None
    if kcap is not None:
        # Copies taken before the launch: the kernel writes the caches in place.
        cap_args = [a.detach().cpu().clone() if isinstance(a, torch.Tensor) else a for a in args]
    out = entry(kernel)(*args)
    if kcap is not None:
        _kcapture_save(model, kcap, kernel, attn.layer_id, step, cap_args, out)
    if debug is not None:
        dump_dir = os.environ.get("DEEPSEEK_V41_DUMP_DIR", ".")
        if model.rank == 0:
            os.makedirs(dump_dir, exist_ok=True)
            path = os.path.join(dump_dir, f"mk_layer{debug[0]}_stage{debug[1]}.pt")
            torch.save({"dbg": out[0].cpu(), "stream_in": stream.cpu(), "pre_mix_in": pre_mix.cpu()}, path)
        raise RuntimeError(f"DEEPSEEK_V41_MK_DEBUG: layer {debug[0]} stage {debug[1]} dumped; stopping")
    out_stream, ffn_pre, topk_out, tap_out = out[0], out[1], out[2], out[3]
    # A forward submits one execution per layer with nothing on the host waiting on
    # any of them (the per-op path got a sync a layer from the router's `.cpu()`):
    # `_track` bounds the outstanding launches with the tiny `ffn_pre`.
    _track(ffn_pre)
    if lw.is_idx_src:
        step.topk = topk_out
    if lw.tap:
        step.taps.append(tap_out)
    return out_stream, ffn_pre


def _draft_meta(seeds, taps, pos):
    """`[B, 3]` int32 (seed, tap, pos) per sequence, as a compiled step: the seeds
    and taps come out of the rejection sampler on the device, so this stays
    there (no read-back before the draft launch)."""
    return torch.stack([seeds.reshape(-1), taps.reshape(-1), pos.reshape(-1)], dim=-1).to(torch.int32)


def draft_consts(B: int, K: int, ntab_p: int) -> torch.Tensor:
    """`[B*K, 4]` int32 per draft row: (i, b, b * ntab_p, i * B + b) -- the row's
    token index, its sequence, its sequence's base row in the kernel's slot table
    and its markov embedding row (CPU)."""
    r = torch.arange(B * K)
    i, b = r % K, r // K
    return torch.stack([i, b, b * ntab_p, i * B + b], dim=-1).to(torch.int32).contiguous()


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

    def stack8(get):
        """The stages' fp8 linears in the MX layout, stacked: `[3, K/4, N]`, `[3, K/32, N]`."""
        pairs = [_fp8(get(s)) for s in stages]
        return (_dev(torch.stack([_host(w) for w, _ in pairs]), device),
                _dev(torch.stack([_host(sc) for _, sc in pairs]), device))

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
        *stack8(lambda s: s.attn.wq_a), stack(lambda s: s.attn.q_norm.weight),
        *stack8(lambda s: s.attn.wq_b),
        *stack8(lambda s: s.attn.wkv), stack(lambda s: s.attn.kv_norm.weight),
        *stack8(lambda s: s.attn.wo_a),
        *stack8(lambda s: s.attn.wo_b),
        stack_host(lambda s: s.attn.attn_sink.float().reshape(-1, 1)),
        model._mk_freqs(s0.attn),
    ]
    for s in stages:
        if (getattr(s.ffn, "stacked", None) is None and getattr(s.ffn, "stacked_mx", None) is None) or not s.ffn.resident:
            raise RuntimeError("the DSpark stages' routed experts are not resident as stacked buffers")
    mxs = [_experts_mx(s.ffn) for s in stages]
    moe = [
        *[_t_f32(s.ffn.gate.weight, device) for s in stages],
        stack_host(lambda s: s.ffn.gate.bias),
        torch.tensor([[float(s0.ffn.experts_start_idx)]], dtype=torch.float32, device=device),
        *[_dev(torch.stack([_host(m[k]) for m in mxs]), device)
          for k in ("w1.weight", "w1.scale", "w3.weight", "w3.scale", "w2.weight", "w2.scale")],
        *stack8(lambda s: s.ffn.shared_experts.w1), *stack8(lambda s: s.ffn.shared_experts.w3),
        *stack8(lambda s: s.ffn.shared_experts.w2),
    ]
    tail = [
        _row(last.norm.weight, device), model.head_weight,
        last.markov_head.embed_weight, last.markov_head.head_weight, last.confidence_head.proj_weight,
    ]
    return head + hc + moe, tail


def _draft_meta_seq(seed, tap, pos):
    """`[1, 3]` int32 (seed, tap, pos) of one sequence, as a compiled step (the flat
    scheme's drafter prologue, unchanged)."""
    return torch.stack([seed.reshape(-1)[:1], tap.reshape(-1)[:1], pos.reshape(-1)[:1]]).to(torch.int32).reshape(1, 3)


def _run_draft_seq(model, seed_ids, tap, main_hidden, span, paged):
    """The flat scheme's drafter launch for one sequence (`draft_kernel_seq`): the
    path as it was before the pipeline work, so `tp64` keeps its behaviour."""
    from . import model as M

    cfg = model.config
    device = main_hidden.device
    begin, length, start_pos = span
    main = main_hidden[begin : begin + length]
    meta = M._compiled(_draft_meta_seq)(seed_ids, tap, paged.pos)
    _track(meta)
    table = _host(paged.block_table).to(torch.int32).reshape(-1, 1).contiguous().to(device)
    row_pos = (start_pos + torch.arange(length)).to(torch.int32).reshape(-1, 1).to(device)
    write_slots = _host(paged.slot_mapping).reshape(-1)[:length].to(torch.int32).reshape(-1, 1).contiguous().to(device)
    weights, tail = model._mk_draft()
    K = int(cfg.dspark_block_size)
    stages = list(model.mtp)
    flags = (int(paged.at_zero), K, int(stages[0].attn.window_size), 9, kernel_version("draft", model.mk_plan.scheme),
             int(cfg.dspark_noise_token_id), int(cfg.vocab_size), int(cfg.dspark_num_experts_per_tok),
             int(paged.block_size))
    caches = [s.attn.window_kv_cache.view(-1, s.attn.head_dim) for s in stages]
    args = [meta, main, row_pos, write_slots, table, *weights, *caches, *tail,
            model._mk_groups_attn, model._mk_groups_full, flags]
    kernel = mk_kernels.draft("prefill" if paged.at_zero else "decode", model.mk_plan.scheme)
    if _MK_TRACE and model.mk_plan.stage_rank == 0:
        _mk_log.warning("mk launch %s span=%s", kernel.func.__name__, span)
    out = entry(kernel)(*args)
    output_ids, confidence = out[0], out[1]
    _track(confidence)
    ids = M._compiled(_ids_i64)(output_ids)
    _track(ids)
    return ids, None, confidence.reshape(1, K)


@torch._dynamo.disable
def run_draft(model, seeds, taps, main_hidden, spans, paged):
    """`_draft` for every sequence of the step: `[(output_ids [1, K+1] int64, None,
    confidence [1, K] fp32)]`, one per span. The flat scheme launches its drafter
    once per sequence (`_run_draft_seq`, the path as before); the pipeline scheme
    drafts all of them in one launch (below).

    launch: `[(output_ids [1, K+1] int64, None, confidence [1, K] fp32)]`, one per
    span. The kernel drafts the B sequences' blocks side by side (B * K rows of one
    tile), so a step's drafting costs one kernel's collectives however many
    sequences it has -- a 10-sequence batch on the pipeline's last stage drafted
    one sequence after another at ~15 ms each before (2026-10-24)."""
    if model.mk_plan.scheme == "tp64":
        return [
            _run_draft_seq(model, seeds.reshape(-1)[b : b + 1], taps.reshape(-1)[b : b + 1],
                           main_hidden, span, pg)
            for b, (span, pg) in enumerate(zip(spans, paged))
        ]
    from . import model as M

    cfg = model.config
    device = main_hidden.device
    B = len(spans)
    lengths = {length for _, length, _ in spans}
    if len(lengths) != 1:
        raise NotImplementedError(f"the drafter wants spans of one length, got {sorted(lengths)}")
    K = int(cfg.dspark_block_size)
    stages = list(model.mtp)
    win = int(stages[0].attn.window_size)
    ntab_p = -(-(win + K) // 128) * 128          # the kernel's slot-table rows per sequence (128-row tiles)
    pieces = [main_hidden[begin : begin + length] for begin, length, _ in spans]
    main = pieces[0] if B == 1 else torch.cat(pieces)
    row_pos = torch.cat([start_pos + torch.arange(length) for _, length, start_pos in spans])
    row_pos = row_pos.to(torch.int32).reshape(-1, 1).contiguous().to(device)
    write_slots = torch.cat([
        _host(p.slot_mapping).reshape(-1)[:length] for p, (_, length, _) in zip(paged, spans)
    ]).to(torch.int32).reshape(-1, 1).contiguous().to(device)
    tables = [_host(p.block_table).to(torch.int32).reshape(-1) for p in paged]
    nblk = max(int(t.numel()) for t in tables)
    table = torch.stack([
        torch.nn.functional.pad(t, (0, nblk - int(t.numel()))) for t in tables
    ]).reshape(-1, 1).contiguous().to(device)
    pos = paged[0].pos.reshape(-1)[:1] if B == 1 else torch.cat([p.pos.reshape(-1)[:1] for p in paged])
    meta = M._compiled(_draft_meta)(seeds.reshape(-1)[:B], taps.reshape(-1)[:B], pos)
    _track(meta)
    consts = model.__dict__.setdefault("_mk_draft_consts", {})
    rows_c = consts.get((B, K, ntab_p))
    if rows_c is None:
        rows_c = consts[(B, K, ntab_p)] = draft_consts(B, K, ntab_p).to(device)
    weights, tail = model._mk_draft()
    at_zero = int(paged[0].at_zero)
    flags = (at_zero, K, win, 9, kernel_version("draft", model.mk_plan.scheme),
             int(cfg.dspark_noise_token_id), int(cfg.vocab_size), int(cfg.dspark_num_experts_per_tok),
             int(paged[0].block_size))
    caches = [s.attn.window_kv_cache.view(-1, s.attn.head_dim) for s in stages]
    args = [meta, rows_c, main, row_pos, write_slots, table, *weights, *caches, *tail,
            model._mk_groups_attn, model._mk_groups_full, flags]
    # The drafter runs on the ranks that hold the head and the sampler: the flat
    # world under tp64, the last stage under tp8pp8 (`_mk_groups_full` is that
    # stage's groups); it is a distinct entry per scheme, `draft_<phase>_<scheme>`,
    # as every other kernel is.
    kernel = mk_kernels.draft("prefill" if at_zero else "decode", model.mk_plan.scheme)
    if _MK_TRACE and model.mk_plan.stage_rank == 0:
        _mk_log.warning("mk launch %s B=%d spans=%s", kernel.func.__name__, B, spans)
    out = entry(kernel)(*args)
    output_ids, confidence = out[0], out[1]
    _track(confidence)
    ids = M._compiled(_ids_i64)(output_ids)
    _track(ids)
    conf = confidence.reshape(B, K)
    return [(ids[b : b + 1], None, conf[b : b + 1]) for b in range(B)]


@torch._dynamo.disable
@dataclass
class SourceReplica:
    """This stage's replica of a kv-source block another stage owns: the rows the
    source wrote this step arrive through the stage boundary and are scattered into
    these caches, which the stage's consumers of that source then read."""

    layer_id: int
    compress_ratio: int
    window_size: int
    compress_kv_cache: torch.Tensor
    index_k_cache: torch.Tensor
    is_kv_source = True


def shipped_source_id(config, stage_first_block: int):
    """The kv-source block whose cache rows cross the boundary *into* the stage whose
    first block is `stage_first_block`: the latest source before it, or None."""
    srcs = [sid for sid in config.kv_source_layer_ids if sid < stage_first_block]
    return max(srcs) if srcs else None


def topk_width(model, step, upto_layer: int) -> int:
    """Columns of the index top-k the latest index source at or before `upto_layer`
    produced this step: the kernel keeps `min(index_topk, read_len)` of its ratio's
    compressed read (`layer_kernel`: `topk = min(topk_in.shape[1], read_len)`), so the
    width follows the context length (384 at a 608-token context for ratio 2, 512 at
    2104). Both sides of a stage boundary size the top-k payload with this, or their
    kernels would be traced for different shapes and the runtime would refuse the
    pair's collective ("replica group signature mismatch")."""
    # The width is a running minimum along the blocks: an index source keeps
    # `min(width of the top-k it was handed, its own read length)`, so a later
    # source with a longer read (ratio 1 after ratio 2) inherits the narrower width.
    cfg = model.config
    width = int(cfg.index_topk)
    for sid in sorted(cfg.index_source_layer_ids):
        if sid > upto_layer:
            break
        ratio = int(cfg.compress_ratios[sid])
        if ratio:
            width = min(width, int(step.read_len(ratio)))
    return width


def _boundary_args(model, step, src, T, upto_layer):
    """`(topk, cmp_cache, idx_cache, slots)` for a boundary carrying `src`'s rows and
    the top-k of the latest index source at or before `upto_layer`."""
    d = model._mk_dummies
    width = topk_width(model, step, upto_layer)
    topk = step.topk if step.topk is not None else torch.zeros(
        T, width, dtype=torch.int32, device=step.device)
    if int(topk.shape[1]) != width:
        raise RuntimeError(
            f"stage {model.mk_plan.stage}: the index top-k is {int(topk.shape[1])} wide, the "
            f"boundary expects {width} (the latest index source up to block {upto_layer})"
        )
    if src is None:
        return topk, d.bf16, d.bf16, d.i32
    idx = step.for_ratio(int(src.compress_ratio), int(src.window_size))
    return topk, src.compress_kv_cache, src.index_k_cache, idx.grp_wslots


@torch._dynamo.disable
def run_recv(model, step, T):
    """A pipeline stage's entry (every stage but the first): the previous stage's
    stream, pre_mix and index top-k, and the rows the shipped kv source wrote this
    step, scattered into this stage's replica of its caches."""
    plan = model.mk_plan
    cfg = model.config
    src_id = shipped_source_id(cfg, plan.first_block)
    src = model._mk_nearest_source(src_id, "kv") if src_id is not None else None
    stream = torch.zeros(T, cfg.hc_mult, cfg.hidden_size, dtype=torch.bfloat16, device=step.device)
    pre_mix = torch.zeros(T, cfg.hc_mult, dtype=torch.float32, device=step.device)
    topk, cmp_cache, idx_cache, slots = _boundary_args(model, step, src, T, plan.first_block - 1)
    parity = (plan.stage - 1) % 2
    kernel = mk_kernels.boundary(parity)
    if _MK_TRACE and plan.stage_rank == 0:
        _mk_log.warning("mk launch %s (recv) stage=%d T=%d src=%s args=%s", kernel.func.__name__, plan.stage, T, src_id,
                        _arg_sig([stream, pre_mix, topk, cmp_cache, idx_cache, slots]))
    out = entry(kernel)(stream, pre_mix, topk, cmp_cache, idx_cache, slots, _role_tensor(model, 0.0),
                        plan.boundary_groups(parity), (kernel_version("pp", "tp8pp8"),))
    _track(out[1])
    step.topk = out[2]
    if src is not None:
        step.compress_cache, step.index_cache = src.compress_kv_cache, src.index_k_cache
    return out[0], out[1]


@torch._dynamo.disable
def run_send(model, stream, pre_mix, step):
    """A pipeline stage's exit (every stage but the last): hand the stream, pre_mix,
    the current index top-k and the current kv source's rows of this step to the
    next stage. Returns a small device tensor that completes with the hand-off."""
    plan = model.mk_plan
    T = int(stream.shape[0])
    src = model._mk_nearest_source(plan.last_block, "kv")
    topk, cmp_cache, idx_cache, slots = _boundary_args(model, step, src, T, plan.last_block)
    parity = plan.stage % 2
    kernel = mk_kernels.boundary(parity)
    if _MK_TRACE and plan.stage_rank == 0:
        _mk_log.warning("mk launch %s (send) stage=%d T=%d src=%d args=%s", kernel.func.__name__, plan.stage, T, src.layer_id,
                        _arg_sig([stream, pre_mix, topk, cmp_cache, idx_cache, slots]))
    out = entry(kernel)(stream, pre_mix, topk, cmp_cache, idx_cache, slots, _role_tensor(model, 1.0),
                        plan.boundary_groups(parity), (kernel_version("pp", "tp8pp8"),))
    _track(out[1])
    return out[1]


def _arg_sig(tensors) -> list:
    """`(shape, dtype, stride, contiguous)` of each tensor, for the launch trace."""
    return [(tuple(t.shape), str(t.dtype).replace("torch.", ""), tuple(t.stride()), bool(t.is_contiguous()))
            for t in tensors]


def _role_tensor(model, value: float) -> torch.Tensor:
    """`[1, 1]` fp32 on the device: 1.0 on the sending side of a boundary, 0.0 on
    the receiving one (the kernel's runtime role; see `pp_kernels`)."""
    cache = model.__dict__.setdefault("_mk_role_tensors", {})
    t = cache.get(value)
    if t is None:
        t = torch.full((1, 1), float(value), dtype=torch.float32).to(model.device)
        cache[value] = t
    return t


def run_blocks(model, stream, pre_mix, step, dump=None):
    """This rank's backbone blocks through the megakernels (every block under tp64,
    the stage's five under tp8pp8). Returns `(stream, pre_mix)`; `step.taps` holds
    the DSpark taps on the stage that owns the tapped blocks."""
    for layer_id, block in zip(model.block_ids, model.layers.values()):
        stream, pre_mix = run_layer(model, block, stream, pre_mix, step)
        if dump is not None:
            dump(f"layer{layer_id}", stream)
            dump(f"premix{layer_id}", pre_mix)
    return stream, pre_mix
