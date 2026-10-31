# SPDX-License-Identifier: Apache-2.0
"""Host side of the TP8EP8 decode MoE (`kernels/moe_tp8`): each rank's 256-column
chunks of its group's experts, built once at load.

A rank loads the 6 experts it owns (EP64). Its group's other members own the rest
of the 48 experts it needs a chunk of, so the chunks travel through `/dev/shm`, one
layer at a time: every rank writes the 9 chunks of each of its experts to a file in
chunk order, the ranks meet at a barrier, every rank reads the chunk of each group
expert it holds (the groups are the attention groups, `kernels/moe_tp8`) (its own 9th chunks it already has), and they meet again before the
files go. Nothing extra is read from the checkpoint.

The decode copy costs ~113 MB a layer a rank on top of the EP64 pool the prefill
kernels keep reading, so the number of converted layers is budgeted against the HBM
free after the weights, minus the KV cache the engine will ask for and a reserve for
the runtime (`_count`); the count is the minimum over the ranks, so every rank runs
the same kernels.

`DEEPSEEK_V41_MOE_TP8`: `auto` (default, as many layers as fit), `0` (EP64 decode),
or a layer count. The reserve is `DEEPSEEK_V41_MOE_TP8_RESERVE_GIB` (default 2) plus
`DEEPSEEK_V41_MOE_TP8_RESERVE_GIB_PER_K` (default 0.3) a 1024 tokens of context past
the first.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

TP = 8
CW = 256
#: Bytes of one chunk record: w1, w3 `[2560, 256]`, s1, s3 `[160, 256]`, w2 `[128, 5120]`,
#: s2 `[8, 5120]` (k-major fp4 / e8m0).
_PARTS = ("w1", "s1", "w3", "s3", "w2", "s2")
#: KV-cache bytes a slot, all 43 layers' caches together (2.94 GiB for 33 920 slots,
#: measured 2026-10-29), for the budget.
_KV_BYTES_PER_SLOT = 96 * 1024

_SPEC = os.environ.get("DEEPSEEK_V41_MOE_TP8", "auto").strip().lower()
_RESERVE = float(os.environ.get("DEEPSEEK_V41_MOE_TP8_RESERVE_GIB", "2")) * 2**30
#: More of the reserve a 1024 tokens of context past the first: the prefill kernels' own
#: allocations grow with the context they read. Measured 2026-10-30: at 8448 tokens
#: 2.0-2.6 GiB left after the weights and the KV cache ran out (`nrt_tensor_allocate
#: status=4` in an 8k prefill) and 3.85 GiB did not; at 1280 and 2104 tokens 2.2 and
#: 3.2 GiB were enough.
_RESERVE_PER_K = float(os.environ.get("DEEPSEEK_V41_MOE_TP8_RESERVE_GIB_PER_K", "0.3")) * 2**30


def _chunk(st: dict, e: int, k: int) -> dict:
    """Chunk `k` of local expert `e` from the host EP64 stacks `st`."""
    c0, c1 = k * CW, (k + 1) * CW
    return {
        "w1": st["w1"][e, :, c0:c1], "s1": st["s1"][e, :, c0:c1],
        "w3": st["w3"][e, :, c0:c1], "s3": st["s3"][e, :, c0:c1],
        "w2": st["w2"][e, c0 // 2 : c1 // 2, :], "s2": st["s2"][e, c0 // 32 : c1 // 32, :],
    }


def _shapes(st: dict) -> dict:
    return {k: tuple(v.shape) for k, v in _chunk(st, 0, 0).items()}


def _record(ch: dict) -> np.ndarray:
    return np.concatenate([np.ascontiguousarray(ch[p].numpy()).reshape(-1) for p in _PARTS])


#: `pack_records`' record: the parts of one chunk, in this order, as flat bytes.
REC_PARTS = (("w13", (2560, 2 * CW)), ("s13", (160, 2 * CW)), ("w2", (CW // 2, 5120)),
             ("s2", (CW // 32, 5120)))
REC_BYTES = sum(int(np.prod(s)) for _, s in REC_PARTS)


def pack_records(chunks: list, device=None) -> dict:
    """The decode kernel's `w1 .. s2` from a rank's chunks as one contiguous record a
    chunk (`kernels/moe_tp8`: a chunk is staged into a fixed buffer by a few large
    dynamic DMAs, and the fp4 GEMMs read the buffer with static descriptors): `w1`
    `[n, REC_BYTES]` uint8 holds `[w1 | w3]` `[2560, 512]`, `[s1 | s3]` `[160, 512]`,
    `w2` `[128, 5120]` and `s2` `[8, 5120]` back to back; `s1 .. s2` are 1-byte
    placeholders."""
    recs = []
    for ch in chunks:
        parts = {"w13": torch.cat([ch["w1"], ch["w3"]], -1), "s13": torch.cat([ch["s1"], ch["s3"]], -1),
                 "w2": ch["w2"], "s2": ch["s2"]}
        flat = []
        for name, shape in REC_PARTS:
            t = parts[name].contiguous()
            assert tuple(t.shape) == shape, (name, tuple(t.shape), shape)
            flat.append(t.reshape(-1))
        recs.append(torch.cat(flat))
    out = {"w1": torch.stack(recs).contiguous()}
    for p in ("s1", "w3", "s3", "w2", "s2"):
        out[p] = torch.zeros((1, 1, 1), dtype=torch.uint8)
    return {k: (v.to(device) if device is not None else v) for k, v in out.items()}


def _unpack(buf: np.ndarray, shapes: dict) -> dict:
    out, at = {}, 0
    for p in _PARTS:
        n = int(np.prod(shapes[p]))
        out[p] = torch.from_numpy(buf[at : at + n].copy()).view(torch.uint8).reshape(shapes[p])
        at += n
    return out


def _free_hbm() -> int | None:
    try:
        used, free = torch.classes.neuron.Runtime().get_vnc_memory_stats()
        return int(free) if free > 0 else None
    except Exception:  # noqa: BLE001 - no runtime (the unit tests)
        return None


def _engine_shape() -> tuple[int, int]:
    """`(max_model_len, max_num_seqs)` of the engine being built, `(0, 0)` outside one."""
    try:
        from vllm.config import get_current_vllm_config

        cfg = get_current_vllm_config()
        return int(cfg.model_config.max_model_len), int(cfg.scheduler_config.max_num_seqs)
    except Exception:  # noqa: BLE001
        return 0, 0


def _kv_need() -> int:
    ctx, seqs = _engine_shape()
    blocks = -(-(ctx + 32) // 32)
    return blocks * 32 * seqs * _KV_BYTES_PER_SLOT if ctx else 0


def _reserve() -> int:
    ctx, _ = _engine_shape()
    return int(_RESERVE + _RESERVE_PER_K * max(0, ctx - 1024) / 1024)


def _count(model, per_layer: int, n_layers: int, group) -> int:
    if _SPEC in ("0", "off", "none", "false"):
        want = 0
    elif _SPEC in ("", "auto", "1", "on", "all"):
        free = _free_hbm()
        if free is None:
            want = n_layers
        else:
            want = max(0, min(n_layers, int((free - _kv_need() - _reserve()) // per_layer)))
    else:
        want = max(0, min(n_layers, int(_SPEC)))
    t = torch.tensor([want], dtype=torch.int64)
    if group is not None:
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=group)
    return int(t.item())


def build(model, stacked_experts) -> int:
    """Give every converted layer's block `_mk_tp8`: its decode kernel's `w1 .. s2` (the
    `[54, ...]` chunk stacks) and `expert_lo` (`[1, 55]` fp32, each chunk's expert id and
    the -1 of the zero sentinel column). Returns the number of converted layers."""
    from .kernels import moe_tp8

    world, rank = int(model.world_size), int(model.rank)
    layers = list(model.layers)
    if world % TP or not layers:
        return 0
    n_local = int(layers[0].ffn.n_local_experts)
    if int(model.config.n_routed_experts) != world * n_local or moe_tp8.CHUNKS != TP + 1:
        return 0
    group = getattr(model.tp_group, "cpu_group", None)
    nch = TP * n_local + n_local
    st0 = {k: v[:1].cpu() for k, v in stacked_experts(layers[0].ffn).items()}
    shapes = _shapes(st0)
    rec = sum(int(np.prod(s)) for s in shapes.values())
    free0 = _free_hbm()
    count = _count(model, nch * rec, len(layers), group)
    if count == 0:
        logger.info("deepseek_v41: TP8EP8 decode MoE off (DEEPSEEK_V41_MOE_TP8=%s, %s)", _SPEC,
                    "budget" if _SPEC == "auto" else "requested")
        return 0
    members = next(g for g in model._mk_groups_attn if rank in g)
    experts, _cols, owners = moe_tp8.chunk_layout(n_local, rank, members)
    j = list(members).index(rank)
    device = model.norm.weight.device
    lo = torch.tensor([[float(e) for e in experts] + [-1.0]], dtype=torch.float32).to(device)
    root = Path(f"/dev/shm/ds41_tp8_{os.getppid()}")
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    def barrier():
        # an all-reduce of a CPU tensor on the Gloo group (`dist.barrier` asks the
        # accelerator hooks, which the Neuron backend does not register)
        if group is not None:
            torch.distributed.all_reduce(torch.zeros(1), group=group)

    for layer in range(count):
        ffn = layers[layer].ffn
        st = {k: v.cpu() for k, v in stacked_experts(ffn).items()}
        mine = root / f"L{layer}_r{rank}.bin"
        tmp = mine.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            for e in range(n_local):
                for k in range(moe_tp8.CHUNKS):
                    fh.write(_record(_chunk(st, e, k)).tobytes())
        tmp.rename(mine)
        barrier()
        chunks = []
        for c in range(TP * n_local):
            owner, e = owners[c], c % n_local
            if owner == rank:
                chunks.append({k: v.contiguous() for k, v in _chunk(st, e, j).items()})
                continue
            idx = e * moe_tp8.CHUNKS + j
            buf = np.fromfile(root / f"L{layer}_r{owner}.bin", dtype=np.uint8, count=rec,
                              offset=idx * rec)
            chunks.append(_unpack(buf, shapes))
        for e in range(n_local):
            chunks.append({k: v.contiguous() for k, v in _chunk(st, e, moe_tp8.CHUNKS - 1).items()})
        assert len(chunks) == nch
        named = pack_records(chunks, device)
        named["expert_lo"] = lo
        layers[layer]._mk_tp8 = named
        barrier()
        mine.unlink()
    try:
        root.rmdir()
    except OSError:
        pass
    free1 = _free_hbm()
    logger.info("deepseek_v41: TP8EP8 decode MoE on %d of %d layers (%d chunks of %d experts a "
                "rank, %.1f MiB a layer, %.1f s; HBM free %.2f -> %.2f GiB, KV need %.2f GiB, "
                "reserve %.2f GiB) for rank %d/%d", count, len(layers), nch, TP * n_local,
                nch * rec / 2**20, time.perf_counter() - t0, (free0 or 0) / 2**30,
                (free1 or 0) / 2**30, _kv_need() / 2**30, _reserve() / 2**30, rank, world)
    return count
