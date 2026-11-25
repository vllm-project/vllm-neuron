"""Routed-expert load statistics and placement simulation (input: expert_load.sh captures).

For every MoE layer the captures hold each token's 8 routed experts. From
them this script reports how skewed the expert popularity is and how the load
of the 64 EP ranks (3 experts each) would change under other placements:

  * prefill: tokens per rank in each forward pass (the routed-expert kernel
    time follows it); imbalance = busiest rank / mean rank;
  * decode: distinct local experts hit per rank in a step of T tokens (the
    hits kernels read each hit expert once), for T = 1, 4 (one request with
    MTP k=3: 4 consecutive tokens), 8 (8 requests) and 32 (8 requests with MTP);
    reported as the mean over steps and layers of the busiest rank's hits.

Placements: the current one (expert e on rank e // 3), random permutations,
and greedy placements fitted on half of the data and evaluated on the other
half (by request), so the gain does not come from fitting the test data:
  * "balanced": per layer, experts sorted by load go to the least-loaded rank
    with a free slot (longest-processing-time first);
  * "anti-coact": the same greedy, but the cost of a rank is its load plus the
    co-selection of the candidate with the experts already on it (the decode
    hits of a step collide on a rank when co-selected experts share it).

Usage: python expert_load.py <capture dir> [--ranks 64] [--experts-per-rank 3] [--seed 0]
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch


def load(capture_dir):
    """-> prefill [(req_ids, {layer: [T, 8] int})], decode {req_id: {layer: [n_tokens, 8]}}."""
    prefill, decode = [], defaultdict(lambda: defaultdict(list))
    for d in sorted(glob.glob(os.path.join(capture_dir, "dp*", "*", "*_meta.json"))):
        with open(d) as f:
            meta = json.load(f)
        req_ids = meta.get("req_ids") or []
        if not req_ids:  # warmup / dummy forward passes
            continue
        fwd_dir = os.path.dirname(d)
        name = os.path.basename(fwd_dir)
        # Padded prefill rows repeat the last position: keep the increasing prefix.
        pos = meta.get("positions") or []
        n_real = next((i for i in range(1, len(pos)) if pos[i] <= pos[i - 1]), len(pos))
        layers = {}
        for ld in glob.glob(os.path.join(fwd_dir, "manual.moe_route.*")):
            L = int(re.search(r"moe_route\.(\d+)", ld).group(1))
            files = sorted(glob.glob(os.path.join(ld, "rank*.pt")),
                           key=lambda f: int(re.search(r"rank(\d+)", f).group(1)))
            if name.startswith("prefill"):
                # Each rank holds its contiguous 1/64 of the (padded) chunk.
                t = torch.cat([torch.load(f) for f in files]).numpy()
                layers[L] = t[:n_real]
            else:
                layers[L] = torch.load(files[0]).numpy()[: len(req_ids)]  # replicated
        if not layers:
            continue
        if name.startswith("prefill"):
            prefill.append((req_ids, layers))
        else:
            for i, rid in enumerate(req_ids):
                for L, t in layers.items():
                    decode[rid][L].append(t[i])
    decode = {r: {L: np.stack(v) for L, v in ls.items()} for r, ls in decode.items()}
    return prefill, decode


def rank_of(placement, n_experts):
    """placement: [ranks, k] expert ids -> [n_experts] rank."""
    r = np.empty(n_experts, dtype=np.int64)
    for i, row in enumerate(placement):
        r[row] = i
    return r


def prefill_imbalance(chunks, L, rank_map, n_ranks):
    """Mean over chunks of busiest-rank tokens / mean-rank tokens for layer L."""
    out = []
    for _, layers in chunks:
        t = layers[L]
        load = np.bincount(rank_map[t].ravel(), minlength=n_ranks)
        out.append(load.max() / load.mean())
    return float(np.mean(out)) if out else float("nan")


def decode_steps(decode, L, T, mode, rng):
    """Synthetic decode steps of T tokens (lists of [8] expert rows) for layer L."""
    reqs = list(decode)
    steps = []
    if mode == "mtp":  # one request, T consecutive tokens
        for r in reqs:
            x = decode[r][L]
            steps += [x[i : i + T] for i in range(0, len(x) - T + 1, T)]
    else:  # T // per requests in parallel, per consecutive tokens each
        per = 4 if mode == "batch_mtp" else 1
        n = T // per
        pos = {r: 0 for r in reqs}
        while True:
            live = [r for r in reqs if pos[r] + per <= len(decode[r][L])]
            if len(live) < n:
                break
            pick = rng.choice(len(live), n, replace=False)
            rows = []
            for j in pick:
                r = live[j]
                rows.append(decode[r][L][pos[r] : pos[r] + per])
                pos[r] += per
            steps.append(np.concatenate(rows))
    return steps


def decode_busiest(steps, rank_map, n_ranks):
    """Mean over steps of max over ranks of distinct local experts hit."""
    if not steps:
        return float("nan"), float("nan")
    mx, mean = [], []
    for s in steps:
        hit = np.unique(s.ravel())
        h = np.bincount(rank_map[hit], minlength=n_ranks)
        mx.append(h.max())
        mean.append(h.mean())
    return float(np.mean(mx)), float(np.mean(mean))


def greedy(load, coact, n_ranks, k, alpha):
    """Experts by load (desc) to the rank with the lowest load + alpha * co-selection."""
    order = np.argsort(-load)
    ranks = [[] for _ in range(n_ranks)]
    rload = np.zeros(n_ranks)
    for e in order:
        best, bcost = None, None
        for r in range(n_ranks):
            if len(ranks[r]) >= k:
                continue
            c = rload[r] + load[e] + (alpha * sum(coact[e, x] for x in ranks[r]) if alpha else 0.0)
            if bcost is None or c < bcost:
                best, bcost = r, c
        ranks[best].append(e)
        rload[best] += load[e]
    return np.array(ranks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir")
    ap.add_argument("--ranks", type=int, default=64)
    ap.add_argument("--experts-per-rank", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random-placements", type=int, default=20)
    a = ap.parse_args()
    R, K = a.ranks, a.experts_per_rank
    E = R * K
    rng = np.random.default_rng(a.seed)
    prefill, decode = load(a.capture_dir)
    layers = sorted({L for _, ls in prefill for L in ls} | {L for r in decode.values() for L in r})
    n_pre = sum(len(next(iter(ls.values()))) for _, ls in prefill)
    n_dec = sum(len(next(iter(v.values()))) for v in decode.values())
    print(f"{len(prefill)} prefill passes ({n_pre} tokens), {len(decode)} requests with "
          f"{n_dec} decode tokens, layers {layers[0]}-{layers[-1]}")

    # Split by request: fit placements on one half, evaluate on the other.
    req_all = sorted({r for ids, _ in prefill for r in ids} | set(decode))
    perm = rng.permutation(len(req_all))
    train = {req_all[i] for i in perm[: len(perm) // 2]}
    pre_tr = [c for c in prefill if c[0][0] in train]
    pre_te = [c for c in prefill if c[0][0] not in train]
    dec_tr = {r: v for r, v in decode.items() if r in train}
    dec_te = {r: v for r, v in decode.items() if r not in train}

    current = np.arange(E).reshape(R, K)
    T_MODES = [(1, "single"), (4, "mtp"), (8, "batch"), (32, "batch_mtp")]
    rows = defaultdict(list)
    skew = []
    for L in layers:
        # Popularity on all data, per layer.
        cnt = np.zeros(E)
        for _, ls in prefill:
            cnt += np.bincount(ls[L].ravel(), minlength=E)
        for v in decode.values():
            cnt += np.bincount(v[L].ravel(), minlength=E)
        skew.append((cnt.max() / cnt.mean(), np.sort(cnt)[-E // 10 :].sum() / cnt.sum()))
        # Training statistics: load and co-selection (decode tokens + prefill tokens).
        load_tr = np.zeros(E)
        coact = np.zeros((E, E))
        for _, ls in pre_tr:
            load_tr += np.bincount(ls[L].ravel(), minlength=E)
        for v in dec_tr.values():
            x = v[L]
            load_tr += np.bincount(x.ravel(), minlength=E)
            for row in x:
                coact[np.ix_(row, row)] += 1
        np.fill_diagonal(coact, 0)
        placements = {
            "current": current,
            "balanced": greedy(load_tr, coact, R, K, 0.0),
            "anti-coact": greedy(load_tr, coact, R, K, 1.0),
        }
        rand = [rng.permutation(E).reshape(R, K) for _ in range(a.random_placements)]
        steps = {(T, m): decode_steps(dec_te, L, T, m, np.random.default_rng(a.seed + L)) for T, m in T_MODES}
        for name, pl in list(placements.items()) + [("random", None)]:
            pls = rand if pl is None else [pl]
            vals = defaultdict(list)
            for p in pls:
                rm = rank_of(p, E)
                vals["prefill"].append(prefill_imbalance(pre_te, L, rm, R))
                for T, m in T_MODES:
                    mx, mean = decode_busiest(steps[(T, m)], rm, R)
                    vals[f"T{T}"].append(mx)
                    vals[f"T{T}_mean"].append(mean)
            rows[name].append({k: float(np.mean(v)) for k, v in vals.items()})

    sk = np.array(skew)
    print(f"\nexpert popularity per layer: max / mean {np.median(sk[:, 0]):.2f} (median over layers, "
          f"max {sk[:, 0].max():.2f}); top 10% of experts take {100 * np.median(sk[:, 1]):.1f}% of the "
          "selections (10% if uniform)")
    print("\nheld-out half, mean over layers:\n")
    hdr = ["placement", "prefill busiest/mean tokens"] + [f"decode T={T} busiest hits (mean)" for T, _ in T_MODES]
    print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    for name in ("current", "random", "balanced", "anti-coact"):
        df = rows[name]
        cells = [name, f"{np.mean([r['prefill'] for r in df]):.3f}"]
        for T, _ in T_MODES:
            cells.append(f"{np.mean([r[f'T{T}'] for r in df]):.3f} ({np.mean([r[f'T{T}_mean'] for r in df]):.3f})")
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
