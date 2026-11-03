# SPDX-License-Identifier: Apache-2.0
"""Every optimized block kernel of the flat scheme (`kernels/tp64/<kind>/`, routed by
`mk_kernels.layer`) against the generic block body specialized to the same (phase, arch,
tp64), on identical inputs, on N ranks.

    python kharness_mp.py 8 test_opt_kernels_diff.py [--kernels r2_decode,r2_src_idx_prefill]
                                                     [--cases 1x6:1040,10x6:8208]

`--kernels`: a comma list of the optimized kernels' packages (`mk_kernels.OPTIMIZED_KINDS`,
`<archetype>_<phase>`: `r1_tap_decode`, `r2_src_idx_prefill`, ...), or `all` (the default).

An optimized kernel must be **bit-identical** to the generic one: same arithmetic, only
scheduling / data movement changed (the bar each was optimized under in its workspace).
The workspaces' recorded cases cover few shapes (prefill T = 1024 / 2048, decode one or ten
6-row spans at 1k / 8k context, plain blocks only); this sweeps the token count, the span
count and the start position (read length), and every archetype: the kv / index sources
(compressor, index scores and top-k), the index-only sources, engram and tap blocks and the
ratio-0 blocks. Inputs are built as `test_layer.py` builds them (random weights per rank, 8
heads and 6 routed experts a rank, the attention group = the world), like
`test_r2_prefill_diff.py`, which this generalizes. Each case: `[rank r target case] PASS`
when every output and every cache is byte-equal.

One deliberate exception, at world < 64: the decode kernels send only the T real rows
through their all-reduces when the payload is small (the r2_decode-based kernels'
`CC2_T_ROWS_MAX`, the r1_decode-based ones' `CC_SMALL_BYTES`) instead of the generic body's 128. The
collective's summation order depends on the message size and the world: at the
deployment's 64 ranks the small payload was measured bit-exact (every bs1 recorded case,
every rank), on 8 ranks it can differ by one bf16 ulp in a few elements. So below 64 ranks
the test runs them with full payloads (`--small-payloads` keeps them, to see that difference).
"""

import argparse

if __name__ == "__main__":
    from kharness_mp import init_rank

    rank, world = init_rank()

import torch  # noqa: E402

from kharness import run_nki  # noqa: E402
from deepseek_v41.config import DeepseekV41Config  # noqa: E402
from deepseek_v41.kernels import mk_kernels  # noqa: E402
from deepseek_v41 import mk_host, model as M  # noqa: E402
from test_layer import make_params, mx4_stack, mx8  # noqa: E402

#: [<spans>x]<tokens per span>:<start position of the first span>. Prefill: odd T, T below
#: one 128-row tile, T not a multiple of the attention step, several spans. Decode: the
#: server's steps (1 committed + 5 drafted rows a sequence, one span a sequence, the
#: engine padding to max_num_seqs spans) at short and long context, other span counts
#: (and T > 128, the kernels' general path), and 1-row spans (no speculation).
DEFAULT_CASES = {
    "prefill": "7:0,23:200,56:2048,129:0,333:500,512:0,777:1300,1500:100,2x300:400",
    "decode": "1x6:1040,10x6:1040,1x6:8208,10x6:8208,2x6:300,5x6:4000,7x6:2000,22x6:1040,1x1:900,10x1:900",
}
OUT_NAMES = ["out_stream", "ffn_pre", "topk_out", "tap_out", "window_kv_cache",
             "compress_kv_cache", "pool_kv", "pool_score", "index_k_cache"]
ERW = 4096   # engram table rows a rank


def parse_cases(spec):
    cases = []
    for item in spec.split(","):
        size, start = item.split(":")
        b, l = (int(v) for v in size.split("x")) if "x" in size else (1, int(size))
        cases.append((b, l, int(start)))
    return cases


def parse_targets(spec):
    """(kind, phase, arch) of the optimized kernels named."""
    kinds = mk_kernels.OPTIMIZED_KINDS if spec == "all" else spec.split(",")
    by_kind = {kind: (phase, arch) for (phase, arch, scheme), kind in mk_kernels._KIND.items()
               if scheme == "tp64"}
    for k in kinds:
        assert k in by_kind, f"unknown kernel {k!r}; known: {', '.join(sorted(by_kind))}"
    return [(k, *by_kind[k]) for k in kinds]


def build_args(cfg, p, B, L, start, world, arch):
    """The kernel's positional arguments (CPU tensors) for one step of archetype `arch` with
    `B` spans of `L` rows, span b starting at `start + 37 b`."""
    ratio, src, isrc, eng, tap = arch
    dim, hc, hd = cfg.hidden_size, cfg.hc_mult, cfg.head_dim
    win, block_size = cfg.sliding_window, 32
    T = B * L
    g0 = torch.Generator().manual_seed(T * 7919 + start + 104729 * ratio + 13 * sum(arch))
    x = (torch.randn(T, hc, dim, generator=g0) * 1.5).to(torch.bfloat16)
    pre_mix = torch.softmax(torch.randn(T, hc, generator=g0), -1)
    nblk = max(16, -(-(start + 37 * (B - 1) + L) // block_size))
    num_slots = max(64, B * nblk) * block_size
    cache = M.act_quant_dequant((torch.randn(num_slots, hd, generator=g0) * 0.3).to(torch.bfloat16), 32)
    spans, tables, slots = [], [], []
    for b in range(B):
        s = start + 37 * b
        spans.append((b * L, L, s))
        table = torch.arange(b * nblk, (b + 1) * nblk)
        tables.append(table)
        slots.append(mk_host.slots_of(table, torch.arange(s, s + L), block_size))
    read_len = 0
    if ratio:
        compress_len = (start + 37 * (B - 1) + L) // ratio
        read_len = max(128, -(-compress_len // 128) * 128)
    idx = mk_host.build_attn_indices(spans, tables, torch.cat(slots), block_size, win, read_len, ratio)
    if not ratio:
        topk = torch.zeros(T, 1, dtype=torch.int32)
    elif isrc:
        # computed by the kernel (its width says how many)
        topk = torch.zeros(T, cfg.index_topk, dtype=torch.int32)
    else:
        topk = torch.full((T, cfg.index_topk), -1, dtype=torch.int64)
        for t in range(T):
            b = t // L
            clen = (spans[b][2] + t - b * L + 1) // ratio
            k = min(cfg.index_topk, clen)
            if k > 0:
                topk[t, :k] = torch.randperm(clen, generator=g0)[:k]
        topk = topk.to(torch.int32).contiguous()
    ccache = M.fp4_quant_dequant((torch.randn(num_slots, hd, generator=g0) * 0.3).to(torch.bfloat16), 16,
                                 scale_e4m3=True)
    kcache = M.fp4_quant_dequant((torch.randn(num_slots, 128, generator=g0) * 0.3).to(torch.bfloat16), 32)
    if src and ratio == 2:
        pool_kv = torch.randn(num_slots, hd, generator=g0)
        pool_score = torch.randn(num_slots, hd, generator=g0)
    else:
        pool_kv, pool_score = torch.zeros(1, 1), torch.zeros(1, 1)
    one = torch.zeros(1, 1, dtype=torch.int32)
    if eng:
        ge = torch.Generator().manual_seed(500 + rank)
        etable = (torch.randn(ERW, 256, generator=ge) * 0.5).to(torch.bfloat16)
        erow_start = torch.tensor([[rank * ERW]], dtype=torch.int32)
        hash_ids = torch.randint(0, world * ERW, (T, 24), generator=g0).to(torch.int32)
        e_args = [hash_ids, etable, erow_start, *mx8(p["e_wkv"], p["e_wkv_s"]), p["e_qk"].reshape(1, -1)]
    else:
        e_args = [one, torch.zeros(1, 1, dtype=torch.bfloat16), one, torch.zeros(1, 1, dtype=torch.uint8),
                  torch.zeros(1, 1, dtype=torch.uint8), torch.zeros(1, 1)]
    groups = mk_host.replica_groups(world, world)
    flags = (ratio, B, win, 9, src, isrc, eng, tap, 0)
    return [
        x, pre_mix,
        p["hca_fn"].t().contiguous(), p["hca_base"].reshape(1, -1), p["hca_scale"].reshape(1, 3), p["attn_norm_w"].reshape(1, -1),
        p["hcf_fn"].t().contiguous(), p["hcf_base"].reshape(1, -1), p["hcf_scale"].reshape(1, 3), p["ffn_norm_w"].reshape(1, -1),
        *mx8(p["wq_a"], p["wq_a_s"]), p["q_norm_w"].reshape(1, -1), *mx8(p["wq_b"], p["wq_b_s"]), *mx8(p["wkv"], p["wkv_s"]), p["kv_norm_w"].reshape(1, -1),
        *mx8(p["wo_a"], p["wo_a_s"]), *mx8(p["wo_b"], p["wo_b_s"]), p["attn_sink"].reshape(-1, 1), p["freqs"],
        idx.row_pos, idx.write_slots, idx.win_slots, idx.cmp_slots, idx.twin, idx.wv, idx.cbase, topk,
        cache, ccache, pool_kv, pool_score, kcache,
        p["cm_wkv"], p["cm_wkv"].t().float().contiguous(), p["cm_wgate"].t().float().contiguous(), p["cm_norm_w"].reshape(1, -1),
        idx.grp_pos if src else one, idx.grp_mslots if src else one, idx.grp_wslots if src else one,
        *mx8(p["ix_wq_b"], p["ix_wq_b_s"]), p["ix_wproj"].t().float().contiguous(), p["ix_wk"], p["ix_knorm_w"].reshape(1, -1),
        idx.clen if ratio else one,
        *e_args,
        p["gate_w"].t().float().contiguous(), p["gate_b"].reshape(1, -1), torch.tensor([[float(p["expert_lo"])]]),
        *mx4_stack(p["w1"], p["s1"]), *mx4_stack(p["w3"], p["s3"]), *mx4_stack(p["w2"], p["s2"]),
        *mx8(p["sh_w1"], p["sh_s1"]), *mx8(p["sh_w3"], p["sh_s3"]), *mx8(p["sh_w2"], p["sh_s2"]),
        groups, groups, flags,
    ]


def launch(kernel, args):
    """Run `kernel` on fresh device copies of the (in-place written) caches."""
    fresh = [a.clone() if isinstance(a, torch.Tensor) else a for a in args]
    return run_nki(kernel, *fresh, grid=2)


def diff(ref, got, prefix, arch):
    """The outputs where `got` is not byte-equal to `ref`, as notes."""
    unwritten = {"topk_out"} if not arch[2] else set()
    unwritten |= {"tap_out"} if not arch[4] else set()
    notes = []
    for name, r, g in zip(OUT_NAMES, ref, got):
        if name in unwritten:
            continue
        if r.shape != g.shape or r.dtype != g.dtype:
            notes.append(f"{prefix}{name} shape/dtype {tuple(g.shape)} {g.dtype} vs {tuple(r.shape)} {r.dtype}")
        elif not torch.equal(r, g):
            err = (g.double() - r.double()).abs()
            notes.append(f"{prefix}{name} max_abs {float(err.max()):.4g} ({int((err > 0).sum())} elements differ)")
    return notes


def main():
    cfg = DeepseekV41Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", default="all",
                    help="comma list of optimized kernels (mk_kernels.OPTIMIZED_KINDS), or all")
    ap.add_argument("--cases", default=None,
                    help="comma list of [<spans>x]<tokens>:<start> (default: per phase, DEFAULT_CASES)")
    ap.add_argument("--small-payloads", action="store_true",
                    help="keep the decode kernels' T-row all-reduce payloads at world < 64 "
                         "(see the module docstring)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="launch each kernel this many times a case and also require every launch "
                         "to reproduce its first (run-to-run determinism of each kernel on its own)")
    args = ap.parse_args()
    if world < 64 and not args.small_payloads:
        import importlib

        for (phase, _, scheme), kind in mk_kernels._KIND.items():
            if phase == "decode" and scheme == "tp64":
                mod = importlib.import_module(f"deepseek_v41.kernels.tp64.{kind}.{kind}")
                if hasattr(mod, "CC2_T_ROWS_MAX"):
                    mod.CC2_T_ROWS_MAX = 0
                if hasattr(mod, "CC_SMALL_BYTES"):
                    mod.CC_SMALL_BYTES = 0
    heads = cfg.num_attention_heads // world if world <= 8 else 8
    ngroups = max(1, cfg.o_groups // world) if world <= 8 else 1
    targets = parse_targets(args.kernels)
    cases = {t[0]: parse_cases(args.cases or DEFAULT_CASES[t[1]]) for t in targets}
    # the rope table must cover every case's last position (the kernels gather it by position)
    last = max(start + 37 * (B - 1) + L for cs in cases.values() for B, L, start in cs)
    p = make_params(cfg, rank, world, heads, ngroups, 6, max(4096, -(-last // 1024) * 1024))
    all_ok = True
    for kind, phase, arch in targets:
        generic = mk_kernels.layer_generic(phase, arch, "tp64")
        optimized = mk_kernels.layer(phase, arch, "tp64")
        assert mk_kernels.version_kind(phase, arch, "tp64") == kind, f"{kind} is not enabled"
        for B, L, start in cases[kind]:
            tag = f"[rank {rank} {kind} B={B} L={L} start={start}]"
            kargs = build_args(cfg, p, B, L, start, world, arch)
            try:
                refs = [launch(generic, kargs) for _ in range(args.repeat)]
                gots = [launch(optimized, kargs) for _ in range(args.repeat)]
            except Exception as exc:  # noqa: BLE001 - report the case and go on
                all_ok = False
                print(f"{tag} ERROR {type(exc).__name__}: {str(exc).splitlines()[0][:300]}", flush=True)
                continue
            notes = diff(refs[0], gots[0], "", arch)
            for i in range(1, args.repeat):
                notes += diff(refs[0], refs[i], f"generic run {i} vs 0: ", arch)
                notes += diff(gots[0], gots[i], f"optimized run {i} vs 0: ", arch)
            all_ok &= not notes
            print(f"{tag} {'PASS' if not notes else 'FAIL'} {'; '.join(notes)}", flush=True)
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
