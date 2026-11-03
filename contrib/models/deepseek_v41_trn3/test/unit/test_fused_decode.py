# SPDX-License-Identifier: Apache-2.0
"""A fused launch of several block kernels (`mk_kernels.fused`) against the same kernels
launched one by one, on identical inputs, on N ranks.

    python kharness_mp.py 8 test_fused_decode.py [--groups r0,r0_e,r2_src_idx,r2]
                                                 [--cases 1x6:1040,10x6:8208]

A group is a run of the model's consecutive blocks (archetype names as
`mk_kernels.arch_name` spells them). The members share what the model's blocks share: a
consumer reads the compressed / index-key caches of the nearest kv source before it and the
top-k of the nearest index source (in the group when there is one, else ones made here), and
each block's output stream is the next one's input. The fused launch must be
**bit-identical** to the separate launches: the same kernels' bodies run, only the launch
boundaries (and the cache copies the model never reads) are gone. Every output and every
cache the group writes is compared byte for byte.

`--embed`: every group's fused launch starts with the token embedding and expand
(`mk_kernels.embed_expand`, random rows over the same small vocabulary), whose stream /
pre_mix the separate launches get from the standalone embedding kernel (itself checked
against the exact lookup).

`--head`: every group's fused launch ends with the greedy head (`mk_kernels.head_greedy`,
random head rows over a `--head-vocab` vocabulary, small enough for one rank's shard at 8
ranks), whose tokens must equal the standalone head kernel's on the separate launches'
last stream / pre_mix.
"""

import argparse
import inspect
import time

if __name__ == "__main__":
    from kharness_mp import init_rank

    rank, world = init_rank()

import torch  # noqa: E402

import kharness as KH  # noqa: E402
import test_opt_kernels_diff as TD  # noqa: E402
from deepseek_v41 import mk_host, model as M  # noqa: E402
from deepseek_v41.config import DeepseekV41Config  # noqa: E402
from deepseek_v41.kernels import mk_kernels  # noqa: E402
from deepseek_v41.kernels.tp64 import head_kernel  # noqa: E402

#: the model's blocks 0-7, 8-15, 20-27 and 36-39 (every archetype; kv / index sources
#: with their consumers, the engram blocks, the taps)
DEFAULT_GROUPS = ("r0,r0_e,r2_src_idx,r2,r2,r2,r2,r2;"
                  "r2_src_idx,r2,r2,r2,r2,r2,r2_src_idx_e,r2;"
                  "r1_src_idx,r1,r1,r1,r1_idx,r1,r1,r1;"
                  "r1_idx,r1_tap,r1_tap,r1_tap")


def main():
    cfg = DeepseekV41Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default=DEFAULT_GROUPS, help="';'-separated groups of archetype names")
    ap.add_argument("--cases", default="1x6:1040,10x6:8208", help="[<spans>x]<tokens>:<start>, decode")
    ap.add_argument("--extra-returns", default="",
                    help="comma list of <member>:<output> also returned by the fused launch (debugging)")
    ap.add_argument("--per-member", action="store_true",
                    help="also return and compare every member's stream / pre_mix / top-k (debugging)")
    ap.add_argument("--head", action="store_true", help="end every fused launch with the greedy head")
    ap.add_argument("--embed", action="store_true", help="start every fused launch with the embedding")
    ap.add_argument("--head-vocab", type=int, default=16000)
    ap.add_argument("--time", type=int, default=0,
                    help="also time this many repeated launches (host time per launch, 4 in flight)")
    args = ap.parse_args()
    TD.rank, TD.world = rank, world
    archs = {}
    for l in range(cfg.num_hidden_layers):
        a = (cfg.compress_ratios[l], int(l in cfg.kv_source_layer_ids), int(l in cfg.index_source_layer_ids),
             int(l in cfg.engram_layer_ids), int(l in cfg.dspark_target_layer_ids))
        archs[mk_kernels.arch_name(a)] = a
    cases = TD.parse_cases(args.cases)
    last = max(start + 37 * (B - 1) + L for B, L, start in cases)
    p = TD.make_params(cfg, rank, world, 8, 1, 6, max(4096, -(-last // 1024) * 1024))
    names = None
    all_ok = True
    if args.head or args.embed:
        part = M.vocab_shard_width(args.head_vocab, world)
        hg = torch.Generator().manual_seed(77)
        w_full = (torch.randn(world * part, cfg.hidden_size, generator=hg) * 0.02).to(torch.bfloat16)
        head_p = [KH.to_dev((1.0 + 0.1 * torch.randn(1, cfg.hidden_size, generator=hg)).float()),
                  KH.to_dev(w_full[rank * part : (rank + 1) * part].contiguous()),
                  KH.to_dev(torch.tensor([[rank * part]], dtype=torch.int32)),
                  mk_host.replica_groups(world, world), (args.head_vocab, 0)]
        e_full = (torch.randn(world * part, cfg.hidden_size, generator=hg) * 0.05).to(torch.bfloat16)
        e_full[args.head_vocab:] = 0
        embed_p = [KH.to_dev(e_full[rank * part : (rank + 1) * part].contiguous()), head_p[2],
                   mk_host.replica_groups(world, world), (int(cfg.hc_mult), 0)]
    for gspec in args.groups.split(";"):
        members = [archs[a] for a in gspec.split(",")]
        kernels = tuple(mk_kernels.layer("decode", a, "tp64") for a in members)
        if names is None:
            body = kernels[0].func
            names = [q.name for q in inspect.signature(body).parameters.values()
                     if q.default is inspect.Parameter.empty]
        ix = {n: names.index(n) for n in ("stream", "pre_mix", "topk_in", "compress_kv_cache", "index_k_cache")}
        for B, L, start in cases:
            tag = f"[rank {rank} group {gspec} B={B} L={L} start={start}]"
            margs = [TD.build_args(cfg, p, B, L, start, world, a) for a in members]
            # what the members share: the nearest kv source's caches, the nearest index source's top-k
            src_caches, ext_topk = {}, {}
            for a, ma in zip(members, margs):
                ratio = a[0]
                if ratio and ratio not in src_caches:
                    src_caches[ratio] = (ma[ix["compress_kv_cache"]], ma[ix["index_k_cache"]])
                    ext_topk[ratio] = ma[ix["topk_in"]] if not a[2] else None
            for a, ma in zip(members, margs):
                if a[0] and not a[1]:
                    ma[ix["compress_kv_cache"]], ma[ix["index_k_cache"]] = src_caches[a[0]]
                if a[1]:
                    src_caches[a[0]] = (ma[ix["compress_kv_cache"]], ma[ix["index_k_cache"]])

            def device_copies():
                # one device tensor per distinct CPU tensor (shared ones stay shared)
                memo = {}
                return [[memo.setdefault(id(x), KH.to_dev(x.clone())) if isinstance(x, torch.Tensor) else x
                         for x in ma] for ma in margs]

            # one by one
            dev = device_copies()
            cur_topk = {}
            outs = []
            taps = []
            emb = None
            if args.embed:
                T = B * L
                ids = torch.randint(0, args.head_vocab, (T, 1), generator=torch.Generator().manual_seed(T),
                                    dtype=torch.int32)
                ids_dev = KH.to_dev(ids)
                emb = KH._entry(head_kernel.embed_expand, 2)(ids_dev, *embed_p)
                want = e_full[ids.reshape(-1).long()].unsqueeze(1).expand(T, int(cfg.hc_mult), cfg.hidden_size)
                one_hot = torch.zeros(T, int(cfg.hc_mult))
                one_hot[:, 0] = 1
                if not (torch.equal(emb[0].cpu(), want) and torch.equal(emb[1].cpu(), one_hot)):
                    all_ok = False
                    print(f"{tag} FAIL standalone embedding differs from the exact lookup", flush=True)
            t0 = time.time()
            for m, (a, k) in enumerate(zip(members, kernels)):
                d = list(dev[m])
                if m:
                    d[ix["stream"]], d[ix["pre_mix"]] = outs[-1][0], outs[-1][1]
                elif emb is not None:
                    d[ix["stream"]], d[ix["pre_mix"]] = emb[0], emb[1]
                if a[0] and not a[2] and a[0] in cur_topk:
                    d[ix["topk_in"]] = cur_topk[a[0]]
                o = KH._entry(k, 2)(*d)
                if a[2]:
                    cur_topk[a[0]] = o[2]
                if a[4]:
                    taps.append(o[3])
                outs.append(o)
            ref = [outs[-1][0].cpu(), outs[-1][1].cpu()] + [t.cpu() for t in taps]
            per_ref = [(o[0].cpu(), o[1].cpu(), o[2].cpu() if a[2] else None) for a, o in zip(members, outs)]
            ref_topk = [t.cpu() for t in cur_topk.values()]
            ref_caches = {id(x): x.cpu() for dm in dev for x in dm if isinstance(x, torch.Tensor)}
            ref_ids = None
            if args.head:
                ref_ids = KH._entry(head_kernel.head_greedy, 2)(outs[-1][0], outs[-1][1], *head_p)
                ref_ids = (ref_ids[0] if isinstance(ref_ids, (tuple, list)) else ref_ids).cpu()
            t_seq = time.time() - t0
            # fused: the same members, one launch
            dev = device_copies()
            pidx, params, plan, returns = {}, [], [], []
            last_idx = {}
            for m, a in enumerate(members):
                srcs = []
                for j, x in enumerate(dev[m]):
                    if m and j == ix["stream"]:
                        srcs.append(("o", m - 1, 0))
                    elif m and j == ix["pre_mix"]:
                        srcs.append(("o", m - 1, 1))
                    elif j == ix["topk_in"] and a[0] and not a[2] and a[0] in last_idx:
                        srcs.append(("o", last_idx[a[0]], 2))
                    else:
                        key = id(x) if isinstance(x, torch.Tensor) else ("c", m, j)
                        if key not in pidx:
                            pidx[key] = len(params)
                            params.append(x)
                        srcs.append(("p", pidx[key]))
                if a[2]:
                    last_idx[a[0]] = m
                plan.append(tuple(srcs))
            nm = len(members)
            fkernels = kernels
            if args.embed:
                # member 0 the embedding: the blocks move up one, block 0 reads its outputs
                shift = lambda src: ("o", src[1] + 1, src[2]) if src[0] == "o" else src  # noqa: E731
                plan = [tuple(shift(x) for x in srcs) for srcs in plan]
                plan[0] = tuple(("o", 0, 0) if j == ix["stream"] else ("o", 0, 1) if j == ix["pre_mix"] else x
                                for j, x in enumerate(plan[0]))
                es = []
                for x in [ids_dev, *embed_p]:
                    key = id(x) if isinstance(x, torch.Tensor) else ("c", x)
                    if key not in pidx:
                        pidx[key] = len(params)
                        params.append(x)
                    es.append(("p", pidx[key]))
                plan.insert(0, tuple(es))
                fkernels = (mk_kernels.embed_expand(),) + kernels
                last_idx = {r: m + 1 for r, m in last_idx.items()}
            eo = 1 if args.embed else 0   # the blocks' member offset
            if args.head:
                hs = [("o", nm - 1 + eo, 0), ("o", nm - 1 + eo, 1)]
                for x in head_p:
                    key = id(x) if isinstance(x, torch.Tensor) else ("c", x)
                    if key not in pidx:
                        pidx[key] = len(params)
                        params.append(x)
                    hs.append(("p", pidx[key]))
                plan.append(tuple(hs))
                fkernels = fkernels + (mk_kernels.head_greedy(),)
            returns = ([("o", nm - 1 + eo, 0), ("o", nm - 1 + eo, 1)]
                       + [("o", m + eo, 3) for m, a in enumerate(members) if a[4]])
            returns += [("o", m, 2) for m in sorted(last_idx.values())]
            for spec in filter(None, args.extra_returns.split(",")):
                mm, kk = (int(v) for v in spec.split(":"))
                if ("o", mm, kk) not in returns:
                    returns.append(("o", mm, kk))
            if args.head:
                returns.append(("o", nm + eo, 0))
            nbase = len(returns)
            if args.per_member:
                for m, a in enumerate(members):
                    for r in [("o", m, 0), ("o", m, 1)] + ([("o", m, 2)] if a[2] else []):
                        if r not in returns:     # (a tensor is returned once)
                            returns.append(r)
            fk = mk_kernels.fused(f"fused_test_g{abs(hash(gspec)) % 10**6}_{B}x{L}{'_embed' if args.embed else ''}"
                                  f"{'_head' if args.head else ''}",
                                  fkernels, tuple(plan),
                                  len(params), tuple(returns))
            t0 = time.time()
            try:
                got = KH._entry(fk, 2)(*params)
            except Exception as exc:  # noqa: BLE001
                all_ok = False
                print(f"{tag} ERROR {type(exc).__name__}: {str(exc).splitlines()[0][:300]}", flush=True)
                continue
            got = [g.cpu() for g in got]
            t_fused = time.time() - t0
            notes = []
            ntap = sum(a[4] for a in members)
            for name, r, g in zip(["out_stream", "ffn_pre"] + [f"tap{i}" for i in range(ntap)] +
                                  [f"topk{i}" for i in range(len(ref_topk))], ref + ref_topk, got):
                if r.shape != g.shape:
                    notes.append(f"{name} shape {tuple(g.shape)} vs {tuple(r.shape)}")
                elif not torch.equal(r, g):
                    err = (g.double() - r.double()).abs()
                    notes.append(f"{name} max_abs {float(err.max()):.4g} ({int((err > 0).sum())} differ)")
            if args.per_member:
                at = {r: i for i, r in enumerate(returns)}
                for m, (a, (rs, rp, rt)) in enumerate(zip(members, per_ref)):
                    gs, gp = got[at[("o", m, 0)]], got[at[("o", m, 1)]]
                    gt = got[at[("o", m, 2)]] if a[2] else None
                    st = "ok" if torch.equal(rs, gs) and torch.equal(rp, gp) and (rt is None or torch.equal(rt, gt)) else "DIFF"
                    dd = float((gs.double() - rs.double()).abs().max())
                    tk = "" if rt is None else f" topk equal={torch.equal(rt, gt)}"
                    print(f"{tag}   member {m} {mk_kernels.arch_name(a)}: {st} stream max_abs {dd:.4g}{tk}", flush=True)
            if args.head:
                gids = got[nbase - 1]
                if not torch.equal(gids, ref_ids):
                    notes.append(f"head ids differ in {int((gids != ref_ids).sum())} of {gids.numel()} rows")
                got = got[: nbase - 1]
            got = got[:nbase]
            got_caches = {id(x): x.cpu() for dm in dev for x in dm if isinstance(x, torch.Tensor)}
            # every input tensor (caches written in place) in the same state, matched by position
            seq_list = list(ref_caches.values())
            fus_list = list(got_caches.values())
            bad = [k for k, (a_, b_) in enumerate(zip(seq_list, fus_list)) if not torch.equal(a_, b_)]
            if bad:
                notes.append(f"{len(bad)} input tensors (caches) end in a different state: "
                             + ", ".join(f"#{k} {tuple(seq_list[k].shape)}" for k in bad[:6]))
            all_ok &= not notes
            if args.time:
                launch = KH._entry(fk, 2)
                pend, ts = [], []
                for _ in range(args.time):
                    t0 = time.perf_counter()
                    o = launch(*params)
                    ts.append(time.perf_counter() - t0)
                    pend.append(o[1])
                    if len(pend) > 4:
                        pend.pop(0).cpu()
                for q in pend:
                    q.cpu()
                t0 = time.perf_counter()
                for _ in range(args.time):
                    launch(*params)[1].cpu()
                t_sync = (time.perf_counter() - t0) / args.time
                ts.sort()
                notes_t = (f"host per launch median {ts[len(ts) // 2] * 1e3:.2f} ms (p10 {ts[len(ts) // 10] * 1e3:.2f}), "
                           f"launch+wait {t_sync * 1e3:.2f} ms")
                print(f"{tag} TIME fused {notes_t}", flush=True)
            print(f"{tag} {'PASS' if not notes else 'FAIL'} ({len(members)} blocks, {len(params)} params; "
                  f"first launch incl. compile: separate {t_seq:.0f}s, fused {t_fused:.0f}s) {'; '.join(notes)}",
                  flush=True)
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
