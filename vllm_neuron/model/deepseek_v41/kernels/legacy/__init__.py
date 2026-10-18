# SPDX-License-Identifier: Apache-2.0
"""Partition-specific NKI kernels — reference material, not the live path.

    legacy kernel        what it is                  where its arithmetic runs now
    --------------------------------------------------------------------------------
    l2_attn.py           `layers.2.attention`,       sparse_attn.gathered_attn_idx
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
"""
