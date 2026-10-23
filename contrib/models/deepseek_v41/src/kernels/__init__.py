# SPDX-License-Identifier: Apache-2.0
"""NKI kernels for the DeepSeek-V4.1-Flash port.

**The megakernels** (the path the model runs, `DEEPSEEK_V41_MEGAKERNEL=1`,
the default): one `@nki.jit` per backbone block and one for the DSpark drafter,
both launched on both physical cores of a logical core (`nl.program_id(0)`
splits the work) and doing their own collectives (`nki.collectives`):

    module                      kernel.entry                 call site
    ------------------------------------------------------------------------------
    one whole `Block` -- engram
      read-out, the DSpark tap,
      hc sublayer-in, MLA
      projections + rope + the
      KV write, the compressor
      and indexer (kv/index
      sources), the index top-k,
      sparse attention over the
      window + compressed slots,
      the grouped O projection
      and its all-reduce, hc
      sublayer-out, hc sublayer-
      in, the 384-expert router,
      the routed + shared
      experts and their
      all-reduce, hc sublayer-out layer_kernel.layer_forward  mk_model.run_layer
                                                              <- model._run_block
    the DSpark drafter -- the
      tap projection, the
      vocab-parallel draft
      embedding, three MTP
      stages (window + draft
      block attention, 128-expert
      MoE), the head, the markov
      bias, the argmax over the
      sharded vocabulary, the
      confidence head             draft_kernel.draft_forward  mk_model.run_draft
                                                              <- model.forward_spec
    shared helpers (row tiles of
      every sublayer, the fp8/fp4
      GEMMs over decoded weight
      rows, gathers/scatters,
      the attention token body,
      the router)                 mk_common, mk_compress       (imported by both)
    the per-step index columns
      (window/compressed slot
      tables, group members)      ../mk_host.py                mk_model.StepIndices

The two megakernels borrow `blockquant`'s GEMM and quantization primitives
(`_gemm`, `_dequant_fp8`, `_dequant_fp4`, `_qdq_fp8`, `_stage_rows`) and
nothing from any library: every other operation below is written out in
`mk_common` / `mk_compress`. Unit tests: `test/unit/test_layer.py`,
`test/unit/test_draft.py` (N ranks through `kharness_mp.py`).

**The per-op kernels** (`DEEPSEEK_V41_MEGAKERNEL=0`, kept for comparison):

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
