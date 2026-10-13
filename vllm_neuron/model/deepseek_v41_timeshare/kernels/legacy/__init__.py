# SPDX-License-Identifier: Apache-2.0
"""Partition-specific NKI kernels — reference material, not the live path.

These 15 kernels were generated against **single partition traces** during the
device-optimization project that preceded this port (the one recorded at
`OswaldHe/DeepSeek-V4.1-Flash-Trainium`). They are faster than anything in the
parent directory, and none of them can be *called* from `model.py` as written,
because each one is a whole named module at one baked shape: a module-level
`SEQ = 8192` (or `TOK = 128`), `WORLD_SIZE = 4`, 96 experts a rank, a
16-device floorplan, and an `assert` on entry. This port serves whatever bucket
the scheduler picks — 512 rows for a prefill chunk, 6 for a speculative verify
step, 1 for a plain decode step — on one device.

They live in their own package to make that status structural rather than a
claim in a docstring. `kernels/` contains exactly the kernels `model.py`
reaches, and nothing that is present but unreachable.

**They are the starting point for the shape-general ones, and that is how they
have been used.** The disposition is per kernel, so that "is this dead code or
is it the ancestor of the live path" has an answer per file rather than in
aggregate:

    legacy kernel        what it is                  where its arithmetic runs now
    --------------------------------------------------------------------------------
    l2_attn.py           `layers.2.attention`,       sparse_attn.gathered_attn
    l3_attn.py           `layers.3.attention`,       (the attention proper),
    l20_attn.py          `layers.20.attention`,      blockquant.{fp8,bf16}_linear
    l21_attn.py          `layers.21.attention`,      (every projection, lifted
    l24_attn.py          `layers.24.attention` —     from l3_attn's decoders and
                         each the whole module at    GEMM pipeline),
                         SEQ=8192 on 4 ranks, one    index_score.weighted_scores
                         baked floorplan cut each    (the indexer),
                         (sequence-parallel for      compress_pool.softmax_pool
                         l2/l24, head-parallel for   (the KV-compression pooling,
                         l3/l20, both for l21)       lifted from l2_attn:1044-1062),
                                                     rmsnorm.rms_norm (`_norm_rows`)
    moe_short.py         `layers.1.ffn` at 128 and   blockquant.fp4_linear (routed
    moe_long.py          8192 tokens: norm, router,  experts), blockquant.
                         96-of-384 experts a rank,   fp8_linear (shared expert),
                         shared expert, all_reduce   blockquant.bf16_linear
                                                     (router), rmsnorm.rms_norm.
                                                     Their `_swiglu` is NOT yet
                                                     lifted — see `kernels/
                                                     __init__.py`'s "not in a
                                                     kernel" list
    hc_attn_in_short.py  `layers.0.hc_attn_in` at    hc_mixes.kernel (the
    hc_attn_in_long.py   128 and 8192 tokens: the    projection, the norm
                         norm statistic, the mixes   statistic and the
                         projection, the 20-step     20-iteration Sinkhorn),
                         Sinkhorn, and the `hc_pre`  hc_collapse.kernel (the
                         collapse folded in          collapse they fold in)
    engram_short.py      `layers.1.engram` at 128    engram_gate.gated_add (the
    engram_long.py       and 8192 tokens, with the   normalized gate and the
                         `ngram` contraction axis    gated add, parameterized on
                         sharded 4 ways              `dim`/`hc`/rows),
                                                     blockquant.fp8_linear (`wkv`)
    dspark_attn.py       `mtp.0.attention` at        the stages are `Attention` /
    dspark_moe.py        SEQ=5, `mtp.0.ffn` over     `MoE` / `BlockQuantLinear`
                         the 128-expert pool         instances, so they reach
    dspark_markov.py     `mtp.2.markov_head`,        sparse_attn, blockquant and
    dspark_confhead.py   `mtp.2.confidence_head`     rmsnorm by construction —
                                                     the same kernels, at the
                                                     draft block's own shapes

Two kernels in the parent directory have **no** ancestor here, because the
partition project's versions of them were not part of this drop:
`hc_expand.py` and `hc_attn_out.py` (the mHC expand and the `hc_post` mix).
They were written for this port in the same style.

Parameterizing one of these and moving it up a directory is the intended way to
make a heavy op faster, and three have been done that way, so the method is not
hypothetical: `blockquant.py`'s fp8/fp4 decoders and GEMM pipeline are lifted
from `l3_attn.py`; `engram_gate.py` is `engram_short.py`'s normalized gate with
the token count, `dim` and `hc` taken from the operands and the collectives
removed; `hc_mixes.py`'s Sinkhorn is `hc_attn_in_short.py`'s; and
`compress_pool.py` is `l2_attn.py`'s pooling block with the member count taken
off the operand shape. `.autohelix/notes/iter-5.md` records the method and
`iter-13.md` §1 the most recent instance — `nki.simulate(kernel)(...)` checks a
lift against the torch expression it replaces, on the CPU, in seconds.

What a *further* lift would have to solve, since the shapes are the whole
difficulty: these kernels fuse a module end to end, which means they also own
the weight layout (`_dequant_weight` into `private_hbm`), the KV cache layout
(`kvp`, `compress_kv`, `index_k` as kernel-local buffers) and the rank's slice
of the sequence. `engram_short.kernel` is the clearest case —
`layers_1_engram_embed_weight` and its block scales are *kernel operands*,
because on a 16-device floorplan the n-gram table was resident in HBM. Here the
two tables are 94.6 GiB and are read 24 rows per token off an mmap, which is
the engram half of the timeshare, so a kernel that wants the table as an
operand cannot be the shape of this port's engram however its token count is
parameterized. This port's weights arrive one expert at a time off a
`weight_store` mmap and its KV lives in vLLM's paged allocation, addressed by
`slot_mapping` — so lifting `l2_attn` whole would mean giving it the paged
layout, not just a symbolic `SEQ`. The per-op kernels are the shape that fits a
served model; the fused ones are the shape that fits a fixed trace.

Nothing is imported here: `@nki.jit` kernels pull in the NKI frontend, and the
partition-specific ones also pull `nki.collectives`.
"""
