# SPDX-License-Identifier: Apache-2.0
"""`NgramHashState.hash_spans` (every span of a step in one host pass) against `forward`
one span at a time (what `_engram_hashes` ran before), on the CPU.

    python test_ngram_hash_spans.py [--hf /mnt/nvme/hf] [--steps 40]

Sequences with disjoint random block tables are prefilled, then decoded step by step
(6 rows a span, 1..10 spans, plus the padded spans `_bucket_spans` adds, all at block 0
through a zero table). Every live row's hash ids must be equal, and so must the two
states' slot histories at every live slot.
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch  # noqa: E402

from deepseek_v41.config import DeepseekV41Config  # noqa: E402
from deepseek_v41.model import NgramHashState, PagedSpan  # noqa: E402


def span(table, block_size, start, length, begin):
    pos = start + torch.arange(length)
    slots = table[pos // block_size] * block_size + pos % block_size
    return PagedSpan(block_table=table, block_size=block_size, slot_mapping=slots, start_pos=start,
                     length=length, begin=begin, host_table=table, host_slots=slots)


def padded(width, block_size, start, length, begin):
    """A span the bucket added: a zero table and slots in block 0."""
    table = torch.zeros(width, dtype=torch.int64)
    slots = torch.zeros(length, dtype=torch.int64)
    return PagedSpan(block_table=table, block_size=block_size, slot_mapping=slots, start_pos=start,
                     length=length, begin=begin, host_table=table, host_slots=slots)


def reference(state, ids, paged):
    return torch.cat([state(ids[p.begin : p.begin + p.length].reshape(1, p.length), p) for p in paged], dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="/mnt/nvme/hf")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    cfg = DeepseekV41Config()
    tok = AutoTokenizer.from_pretrained(args.hf)
    ref, vec = NgramHashState(cfg, tok), NgramHashState(cfg, tok)
    rng = random.Random(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    block_size, width, nseq = 64, 24, 10
    num_slots = (nseq * width + 1) * block_size
    ref.allocate_state("cpu", num_slots)
    vec.allocate_state("cpu", num_slots)
    blocks = list(range(1, nseq * width + 1))
    rng.shuffle(blocks)
    tables = [torch.tensor(blocks[i * width : (i + 1) * width], dtype=torch.int64) for i in range(nseq)]
    vocab = int(cfg.vocab_size)
    lens = [0] * nseq
    checked = 0

    def run(paged, live_rows):
        nonlocal checked
        T = sum(p.length for p in paged)
        ids = torch.randint(0, vocab, (T,), generator=g)
        want = reference(ref, ids, paged)
        got = vec.hash_spans(ids, paged)
        if got.shape != want.shape:
            raise SystemExit(f"FAIL shape {tuple(got.shape)} vs {tuple(want.shape)}")
        if not torch.equal(got[:, :live_rows], want[:, :live_rows]):
            bad = int((got[:, :live_rows] != want[:, :live_rows]).any(-1).any(-1).sum())
            raise SystemExit(f"FAIL {bad} of {live_rows} live rows differ (spans {[(p.begin, p.length, p.start_pos) for p in paged]})")
        checked += live_rows

    # prefill each sequence (one span a step, as the scheduler runs a prefill)
    for i in range(nseq):
        n = rng.randint(40, 700)
        run([span(tables[i], block_size, 0, n, 0)], n)
        lens[i] = n
    # decode steps: a random subset of sequences, 6 rows each, then the bucket's padding
    for _ in range(args.steps):
        seqs = sorted(rng.sample(range(nseq), rng.randint(1, nseq)))
        paged, begin = [], 0
        for i in seqs:
            paged.append(span(tables[i], block_size, lens[i], 6, begin))
            begin += 6
        live = begin
        bucket = 1 << (len(seqs) - 1).bit_length()
        for _ in range(min(bucket, nseq) - len(seqs)):
            paged.append(padded(width, block_size, paged[0].start_pos, 6, begin))
            begin += 6
        run(paged, live)
        for i in seqs:
            lens[i] += rng.randint(1, 6)       # (the accepted tokens; the rest is rewritten next step)
    live_slots = torch.cat([
        (t[torch.arange(n) // block_size] * block_size + torch.arange(n) % block_size)
        for t, n in zip(tables, lens)
    ])
    if not torch.equal(ref.cache[live_slots], vec.cache[live_slots]):
        raise SystemExit("FAIL the slot histories differ at live slots")
    print(f"PASS {checked} live rows equal ({nseq} sequences, {args.steps} decode steps)")


if __name__ == "__main__":
    main()
