# Cold 85k prefill with HY3_SPARSE_PREFILL=0.25 (commit 66777fb): where the time went

Device profile (`scripts/profile_85k.sh` with `KV_CAP=0.40` -- at 0.49 the
profiled ranks ran out of HBM next to the summaries and the selection's
weight copies -- `HY3_SPARSE_PREFILL=0.25 HY3_SPARSE_DECODE=0.125`, session
c1_cold, rank 0, 21 chunks of 4096 tokens).

Target NEFF per chunk, ms (dense: `results/profile_85k_moe_fp8w`, same kernels):

| chunk (exec) | 0 (0) | 1 (2) | 3 (6) | 6 (12) | 10 (20) | 15 (30) | 20 (40) |
|---|---|---|---|---|---|---|---|
| dense | 426 | 445 | 483 | 554 | | | ~941 (2026-10 profile) |
| sparse 0.25 (66777fb) | 653 | 633 | 671 | 742 | 738 | 740 | 783 |

Chunk 0 has no prior context (nothing to skip) and is 227 ms slower. Its
attention module is 118.6 ms (dense 89.5 ms; the selection's torch ops 9.5 ms,
the summary update 1.5 ms). 188.8 ms (union) are one HLO custom call,
`%custom-call.36311`, that the profiler attributes to `Sampler` (the layer of
the first, deduplicated call site) but whose 22.6k instructions are spread
evenly over the whole chunk in 239 clusters (80 layers x 3): the
`torch.sort` of the 2688 block ids in `sparse_prefill_blocks`, lowered to the
vendored rotational top-k kernel (`rotational_topk_utils.py`).

One core (`neuron:0`, compiled alone, per call incl. ~0.16 ms launch + sync):

| | ms |
|---|---|
| `torch.sort` [2688] | 2.17 |
| `torch.sort` [672] descending | 0.36 |
| `torch.topk` [672], k = 192 | 0.15 |
| `sparse_prefill_blocks`, T = 4096, 2688-block table, 768 kept (66777fb, with the sort) | 2.40 |
| `sparse_prefill_blocks` without the sort (kept groups in top-k order) | 0.27 |

The kept groups need no order (attention over the prior context is
permutation invariant; sparse decode relies on the same), so the fix gathers
the top-k groups' blocks directly. Same blocks kept, ~2.1 ms x 80 layers
less per chunk. End to end: cold 85k prefill 15.80 -> 12.13 s (dense 14.34 s).
