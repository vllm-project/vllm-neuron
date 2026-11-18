# SPDX-License-Identifier: Apache-2.0
"""`mk_model.attn_indices_np` (numpy, vectorized over the spans) against its definition,
`mk_host.build_attn_indices`, on the CPU: every column equal, for random decode steps (1-10
spans of 1 or 6 rows, the padded spans `_bucket_spans` adds -- a zero table, slots in block
0 --, rows past the spans) and prefill spans, at every compressed ratio (0, 1, 2), short
contexts (window positions before the sequence start) and long ones.

    python test_attn_indices_np.py [--cases 3000]
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch  # noqa: E402

from deepseek_v41 import mk_host, mk_model  # noqa: E402

FIELDS = ("row_pos", "write_slots", "win_slots", "cmp_slots", "twin", "wv", "cbase", "clen",
          "grp_pos", "grp_mslots", "grp_wslots")


def case(rng: random.Random):
    block_size = rng.choice((16, 32, 64))
    ratio = rng.choice((0, 1, 2))
    window = rng.choice((128, 64))
    if rng.random() < 0.8:                      # decode: B spans of L rows
        B, L = rng.randint(1, 10), rng.choice((6, 6, 6, 1))
    else:                                        # prefill: one span
        B, L = 1, rng.randint(1, 700)
    live = rng.randint(1, B)
    starts = [rng.randint(0, 9000) for _ in range(live)]
    max_pos = max(starts) + L + 1
    width = max_pos // block_size + 1 + rng.randint(0, 3)
    pool = rng.sample(range(1, 20000), live * width)
    tables, slots, spans = [], [], []
    for b in range(B):
        if b < live:
            table = torch.tensor(pool[b * width : (b + 1) * width], dtype=torch.int64)
            if rng.random() < 0.3:              # vLLM leaves negatives in unallocated rows
                table[-1] = -1
            start = starts[b]
            pos = start + torch.arange(L)
            s = table.clamp_min(0)[pos // block_size] * block_size + pos % block_size
        else:                                    # a padded span: zero table, block 0
            table = torch.zeros(width, dtype=torch.int64)
            start = starts[0]
            s = torch.zeros(L, dtype=torch.int64)
        tables.append(table)
        slots.append(s)
        spans.append((b * L, L, start))
    clen = (max(starts) + L) // ratio if ratio else 0
    read_len = max(128, -(-clen // 128) * 128) if ratio else 0
    extra = rng.choice((0, 0, 3))               # rows past the spans
    return dict(spans=spans, block_tables=tables, slot_mapping=torch.cat(slots), block_size=block_size,
                window=window, read_len=read_len, ratio=ratio, total_rows=B * L + extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    bad = 0
    for i in range(a.cases):
        c = case(rng)
        ref = mk_host.build_attn_indices(**c)
        got = mk_model.attn_indices_np(**c)
        diffs = []
        for f in FIELDS:
            x, y = getattr(ref, f), getattr(got, f)
            if (x is None) != (y is None):
                diffs.append(f"{f}: None vs {type(y).__name__}")
            elif x is not None and (x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y)):
                diffs.append(f"{f}: {tuple(x.shape)} {x.dtype} vs {tuple(y.shape)} {y.dtype}")
        for f in ("nwin", "n_per_span", "groups_per_span"):
            if getattr(ref, f) != getattr(got, f):
                diffs.append(f"{f}: {getattr(ref, f)} vs {getattr(got, f)}")
        if diffs:
            bad += 1
            if bad <= 5:
                print(f"case {i}: ratio {c['ratio']} spans {c['spans'][:3]}... : {'; '.join(diffs)}")
    print(f"{a.cases - bad}/{a.cases} cases equal")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
