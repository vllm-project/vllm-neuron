# SPDX-License-Identifier: Apache-2.0
"""NKI kernels for the DeepSeek-V4.1-Flash timeshare port.

    module                      kernel.entry                 call site (model.py)
    ------------------------------------------------------------------------------
    MLA attention: Q/O low-rank
      and latent KV projections  blockquant.fp8_linear        BlockQuantLinear.forward
                                                              -> nki_linear:673-680
    sparse attention: the slot
      gather, score, softmax
      + weighted sum             sparse_attn.gathered_attn_idx sparse_attn:1795
    indexer score                index_score.weighted_scores  Indexer.forward:2112
    KV compression pooling       compress_pool.softmax_pool   Compressor.forward:1962
    every RMSNorm                rmsnorm.rms_norm             RMSNorm.forward:306
    MoE routed + shared expert
      GEMMs (w1/w2/w3)           blockquant.fp4_linear        _swiglu_expert:2593-2611
                                 blockquant.fp8_linear        -> nki_linear
    MoE router                   blockquant.bf16_linear       _gate_step:2805
    MoE top-6-of-384 selection   nkilib rotational top-k      nki_topk:631
    MHC coefficients + Sinkhorn  hc_mixes.kernel              HyperConnections.mixes:3552
    MHC expand (1 -> 4 copies)   hc_expand.kernel             forward:5077
    MHC collapse (hc_pre)        hc_collapse.kernel           HyperConnections.pre:3575
    MHC sublayer out (hc_post)   hc_attn_out.kernel           HyperConnections.post:3602
    engram gate + gated add      engram_gate.gated_add        Engram.forward:3476
    engram read-out GEMM         blockquant.fp8_linear        Engram.wkv -> nki_linear
    DSpark mtp.0-2 attention,
      FFN, markov and confidence
      heads                      the same kernels as the      DSparkStage.forward:3995,
                                 backbone — the stages are    forward_head:4052,
                                 `Attention` / `MoE` /        DSparkMarkovHead.forward,
                                 `BlockQuantLinear` instances DSparkConfidenceHead.forward
                                 over their own 128-expert
                                 pool, so every GEMM is
                                 `blockquant` and their
                                 attention is `sparse_attn`
"""
