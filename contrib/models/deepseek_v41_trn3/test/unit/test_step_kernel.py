# SPDX-License-Identifier: Apache-2.0
"""`kernels/tp64/step_kernel.step_prologue` on one NeuronCore against its definitions:
`mk_host.build_attn_indices` for every index column of the three `(ratio, window)`
combinations and `NgramHashState.hash_spans` (the real method, on a state built here with a
random compressed-token map and history) for the two engram layers' n-gram ids.

Random decode steps: 1-10 spans of 6 rows (some padded: a zero table, slots in block 0, the
first live span's start), contexts 768 / 1280 / 8448 (the compressed reads pinned to the
context as the model pins them), and a previous step's rejection output per span (`v`
valid entries then -1) whose rejected drafts the kernel subtracts from an optimistic start;
`--sync` passes all-zero `prev` and the real starts instead.

    KH_CORE=0 python test_step_kernel.py [--cases 20] [--seed 0] [--sync] [--time]
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kharness import run_nki  # noqa: E402 - sets the environment first

import numpy as np  # noqa: E402
import torch  # noqa: E402

from deepseek_v41 import mk_host  # noqa: E402
from deepseek_v41 import model as M  # noqa: E402
from deepseek_v41.kernels.tp64 import step_kernel as SK  # noqa: E402

BS, L, K, WIN = 32, 6, 5, 128
VOCAB, VC, PAD_TOKEN = 129280, 99092, 2
LAYER_IDS, NGRAM, HEADS, EVOCAB = (1, 14), 4, 8, 16_000_000
ROW_START = (3 * 6_000_000, 3 * 6_000_000 + 11)
FIELDS0 = ("row_pos", "write_slots", "win_slots", "wv")
FIELDSR = FIELDS0 + ("cmp_slots", "clen", "grp_pos", "grp_mslots", "grp_wslots")


def hash_consts():
    """The real `NgramHashState` constants: primes, offsets, multipliers (`__init__`'s)."""
    primes, seen = [], set()
    for _ in LAYER_IDS:
        per = []
        for _ in range(NGRAM - 1):
            sizes, cur = [], EVOCAB - 1
            for _ in range(HEADS):
                cur = M._next_prime(cur, seen)
                seen.add(cur)
                sizes.append(cur)
            per.append(tuple(sizes))
        primes.append(tuple(per))
    flat = [[p for per in layer for p in per] for layer in primes]
    offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
    bound = max(1, (np.iinfo(np.int64).max // VC) // 2)
    mults = [np.random.default_rng(10007 * lid).integers(0, bound, size=(NGRAM,), dtype=np.int64) * 2 + 1
             for lid in LAYER_IDS]
    return torch.tensor(primes), torch.tensor(np.array(offsets)), torch.stack([torch.tensor(m) for m in mults])


def make_state(token_map, history, consts):
    """An `NgramHashState` without a tokenizer: the module's own `hash_spans` on our tables."""
    st = M.NgramHashState.__new__(M.NgramHashState)
    torch.nn.Module.__init__(st)
    st.layer_ids, st.max_ngram_size, st.n_heads = LAYER_IDS, NGRAM, HEADS
    st.pad_id = int(token_map[PAD_TOKEN])
    primes, offsets, mults = consts
    st._host = {"primes": primes, "offsets": offsets, "multipliers": mults, "token_map": token_map}
    st.cache = history.clone()
    return st


class Span:
    """What `hash_spans` reads of a `PagedSpan`."""

    def __init__(self, begin, length, start_pos, table, slots):
        self.begin, self.length, self.start_pos = begin, length, start_pos
        self.host_table, self.host_slots, self.block_size = table, slots, BS


def case(rng, sync):
    ctx = rng.choice((768, 1280, 8448))
    B = rng.choice((1, 2, 4, 8, 10))
    live = rng.randint(1, B)
    W = -(-ctx // BS) + 1
    nslots = (W * B + 8) * BS
    pool = rng.sample(range(1, W * B + 8), live * W)
    starts, tables, slots, opt, prev = [], [], [], [], []
    for b in range(B):
        if b < live:
            s = rng.randint(1, ctx - L - 1) if rng.random() < 0.9 else rng.randint(0, 3)
            t = torch.tensor(pool[b * W : (b + 1) * W], dtype=torch.int64)
            if rng.random() < 0.3:
                t[-1] = -1
            pos = s + torch.arange(L)
            sl = t.clamp_min(0)[pos // BS] * BS + pos % BS
        else:
            s, t, sl = starts[0], torch.zeros(W, dtype=torch.int64), torch.zeros(L, dtype=torch.int64)
        v = K + 1 if sync else rng.randint(1, K + 1)
        row = [rng.randrange(VOCAB) for _ in range(v)] + [-1] * (K + 1 - v)
        if sync:
            row = [0] * (K + 1)
        starts.append(s)
        tables.append(t)
        slots.append(sl)
        prev.append(row)
        opt.append(s + (K + 1 - v))
    read_len = {r: -(-(-(-ctx // r)) // 128) * 128 for r in (1, 2)}
    return dict(ctx=ctx, B=B, live=live, W=W, nslots=nslots, starts=starts, tables=tables, slots=slots,
                opt=opt, prev=prev, read_len=read_len)


def run_case(c, consts, rng_np, timing=False):
    B, W = c["B"], c["W"]
    T = B * L
    G2 = L // 2 + 1
    nj = {r: c["read_len"][r] // (BS // r) for r in (1, 2)}
    Wp = max(W, nj[1], nj[2])
    Hw = SK.H + Wp
    pm = torch.zeros(B, Hw, dtype=torch.int32)
    for b in range(B):
        pm[b, SK.C_START] = c["opt"][b]
        pm[b, SK.C_LIVE] = int(b < c["live"])
        pm[b, SK.C_TBASE] = b * Hw + SK.H
        pm[b, SK.C_BL] = b * L
        pm[b, SK.C_BG2] = b * G2
        pm[b, SK.C_BNJ2] = b * nj[2]
        pm[b, SK.C_BNJ1] = b * nj[1]
        pm[b, SK.H : SK.H + W] = c["tables"][b].clamp_min(0).to(torch.int32)
    prev = torch.tensor(c["prev"], dtype=torch.int32)
    ids = torch.tensor(rng_np.integers(0, VOCAB, size=(T, 1)), dtype=torch.int32)
    token_map = torch.tensor(rng_np.integers(0, VC, size=(VOCAB,)), dtype=torch.int64)
    history = torch.tensor(rng_np.integers(-1, VC, size=(c["nslots"],)), dtype=torch.int64)
    history[rng_np.random(c["nslots"]) < 0.05] = -1
    primes, offsets, mults = consts
    prod = torch.zeros(VC, 2 * 2 * NGRAM, dtype=torch.int32)
    cc = torch.arange(VC, dtype=torch.int64)
    for li in range(2):
        for s in range(NGRAM):
            v = cc * int(mults[li, s])
            prod[:, li * 8 + 2 * s] = (v >> 32).to(torch.int32)
            prod[:, li * 8 + 2 * s + 1] = (v & 0xFFFFFFFF).to(torch.int64).where(
                (v & 0xFFFFFFFF) < 2**31, (v & 0xFFFFFFFF) - 2**32).to(torch.int32)
    pr = primes.reshape(2, -1)                 # [2, 24]
    hcol = torch.zeros(2, SK.HCOLS, dtype=torch.int32)
    hcol[0] = pr.reshape(-1).to(torch.int32)
    hcol[1] = torch.cat([offsets[li] - ROW_START[li] for li in range(2)]).to(torch.int32)
    hinv = (1.0 / pr.reshape(1, -1).to(torch.float64)).to(torch.float32)

    rl1, rl2 = c["read_len"][1], c["read_len"][2]
    nwin = WIN + L - 1
    col = lambda n: torch.zeros(n, 1, dtype=torch.int32)  # noqa: E731
    outs = {}
    outs["a"] = [col(T), col(T), col(B * nwin), col(T)]
    for key, r, rl in (("b", 2, rl2), ("c", 1, rl1)):
        G = L if r == 1 else G2
        outs[key] = [col(T), col(T), col(B * nwin), col(T), col(B * rl), col(T), col(B * G), col(B * G * r), col(B * G)]
    hash0 = torch.zeros(T, 24, dtype=torch.int32)
    hash1 = torch.zeros(T, 24, dtype=torch.int32)
    emb = col(T)
    st_out = col(B)
    hist_dev = history.to(torch.int32).reshape(-1, 1)
    flags = (B, L, K, Wp, BS, WIN, VOCAB, int(token_map[PAD_TOKEN]), 0, 2, 1, rl2, rl1, 0)
    from kharness import _entry, to_dev
    dev = [to_dev(x) for x in (pm.reshape(-1, 1), prev, ids, hist_dev, token_map.to(torch.int32).reshape(-1, 1),
                               prod, hcol, hinv)]
    dout = {k: [to_dev(x) for x in v] for k, v in outs.items()}
    dh0, dh1, demb, dst = to_dev(hash0), to_dev(hash1), to_dev(emb), to_dev(st_out)
    done = _entry(SK.step_prologue, 2)(*dev, *dout["a"], *dout["b"], *dout["c"], dh0, dh1, demb, dst, flags)
    # The kernel writes its buffers in place, which the runtime does not see: a read-back
    # of one is ordered after the launch only through the launch's own output.
    done.cpu()
    if timing:
        import time
        from libtorch_neuronx_lite.compile import native_backend as NB
        args = [*dev, *dout["a"], *dout["b"], *dout["c"], dh0, dh1, demb, dst, flags]
        seen = []
        orig = NB._lite_execute

        def watch(art, ins):
            o = orig(art, ins)
            seen.append((art, list(ins), o))
            return o
        NB._lite_execute = watch
        _entry(SK.step_prologue, 2)(*args).cpu()
        NB._lite_execute = orig
        art, ins, o = seen[-1]
        ex = NB._executor_for(art)
        outs_ = [torch.empty(tuple(x.shape), dtype=x.dtype, device=ins[0].device) for x in art.io_spec.outputs]
        for n in (1, 4, 8, 8):
            t0 = time.perf_counter()
            for _ in range(n):
                ex.execute(ins, outs_)
            outs_[0].cpu()
            dt = (time.perf_counter() - t0) / n
            print(f"  timing: {n} back-to-back direct launches: {dt * 1e6:.0f} us each (B={B})")
        t0 = time.perf_counter()
        ex.execute(ins, outs_)
        t1 = time.perf_counter()
        outs_[0].cpu()
        print(f"  timing: one launch host submit {(t1 - t0) * 1e6:.0f} us, to completion {(time.perf_counter() - t0) * 1e6:.0f} us")
    got = {k: [x.cpu() for x in v] for k, v in dout.items()}
    got_hash = (dh0.cpu(), dh1.cpu())
    got_emb, got_st, got_hist = demb.cpu(), dst.cpu(), dev[3].cpu()

    # ---- references
    spans = [(b * L, L, c["starts"][b]) for b in range(B)]
    sm = torch.cat(c["slots"])
    bad = []
    for key, r in (("a", 0), ("b", 2), ("c", 1)):
        ref = mk_host.build_attn_indices(spans, c["tables"], sm, BS, WIN, c["read_len"].get(r, 0), r, T)
        fields = FIELDS0 if r == 0 else FIELDSR
        for f, g in zip(fields, got[key]):
            x = getattr(ref, f)
            if x.shape != g.shape or not torch.equal(x, g):
                n = int((x.reshape(-1)[: g.numel()] != g.reshape(-1)[: x.numel()]).sum()) if x.numel() == g.numel() else -1
                i = int(torch.nonzero(x.reshape(-1) != g.reshape(-1))[0]) if n > 0 else -1
                bad.append(f"{key}(r{r}).{f}: {n} differ" + (f" (first at {i}: want {int(x.reshape(-1)[i])}, got "
                                                             f"{int(g.reshape(-1)[i])})" if i >= 0 else
                                                             f" shapes {tuple(x.shape)} vs {tuple(g.shape)}"))
    state = make_state(token_map, history, consts)
    paged = [Span(b * L, L, c["starts"][b], c["tables"][b].clamp_min(0), c["slots"][b]) for b in range(B)]
    hashes = state.hash_spans(ids.reshape(-1).to(torch.int64), paged)
    nlive = c["live"] * L
    for li in range(2):
        want = (hashes[0, :, li, :] - ROW_START[li]).to(torch.int32)
        g = got_hash[li]
        d = (want[:nlive] != g[:nlive])
        if d.any():
            i = torch.nonzero(d)[0].tolist()
            bad.append(f"hash{li}: {int(d.sum())} of {d.numel()} live entries differ (first {i}: want "
                       f"{int(want[i[0], i[1]])}, got {int(g[i[0], i[1]])})")
    if not torch.equal(got_emb, ids):
        bad.append("embed_ids")
    if got_st.reshape(-1).tolist() != c["starts"]:
        bad.append(f"starts {got_st.reshape(-1).tolist()} vs {c['starts']}")
    live_slots = torch.cat(c["slots"][: c["live"]])
    if not torch.equal(got_hist.reshape(-1)[live_slots].to(torch.int64), state.cache[live_slots]):
        bad.append("history at the live write slots")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--time", action="store_true", help="also time back-to-back direct launches")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rng_np = np.random.default_rng(a.seed)
    consts = hash_consts()
    nbad = 0
    for i in range(a.cases):
        c = case(rng, a.sync)
        bad = run_case(c, consts, rng_np, a.time)
        tag = f"case {i}: ctx {c['ctx']} B {c['B']} live {c['live']} starts {c['starts'][:3]}"
        if bad:
            nbad += 1
            print(f"{tag}: FAIL\n    " + "\n    ".join(bad), flush=True)
        else:
            print(f"{tag}: ok", flush=True)
    print(f"{a.cases - nbad}/{a.cases} cases equal")
    raise SystemExit(1 if nbad else 0)


if __name__ == "__main__":
    main()
