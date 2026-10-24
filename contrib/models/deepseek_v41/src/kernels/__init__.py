# SPDX-License-Identifier: Apache-2.0
"""NKI kernels for the DeepSeek-V4.1-Flash port.

**The megakernels** (the path the model runs, `DEEPSEEK_V41_MEGAKERNEL=1`,
the default): one `@nki.jit` per backbone block and one for the DSpark drafter,
both launched on both physical cores of a logical core (`nl.program_id(0)`
splits the work) and doing their own collectives (`nki.collectives`). The
bodies below are traced into **separate entry points** by `mk_kernels`: one per
phase (prefill / decode), per block archetype (`ratio`, kv source, index source,
engram, DSpark tap) and per parallelization scheme (`DEEPSEEK_V41_SCHEME`:
`tp64`, the flat world, or `tp8pp8`, the model-internal pipeline of 8 stages of
8 ranks, whose seams are `pp_kernels`):

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
      all-reduce, hc sublayer-out <scheme>/layer_kernel.layer_forward  mk_model.run_layer
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
      confidence head             <scheme>/draft_kernel.draft_forward  mk_model.run_draft
                                                              <- model.forward_spec
    the fp8 / fp4 GEMMs on the
      tensor engine's MX matmul
      (`nc_matmul_mx`: block-
      scaled weights in the host-
      packed [K/4, N] layout,
      activations quantized on
      chip with `quantize_mx`)    mk_mx                        (imported by both)
    the token-grouped routed
      experts (row lists from
      `nonzero_with_count`, a
      runtime-bounded loop per
      expert over gathered row
      tiles, `dma_compute`
      scatter-adds)               mk_moe                       (imported by both)
    shared helpers (row tiles of
      every sublayer, gathers/
      scatters, the attention
      token bodies -- the keys
      selected on chip through
      tensor indirection into a
      resident transposed slot
      table -- and the router)    mk_common, mk_compress       (imported by both)
    the specialized entry points
      `layer_<phase>_<arch>_<scheme>`,
      `draft_<phase>_<scheme>`,
      `pp_boundary_{even,odd}`    mk_kernels                   (what mk_model runs)
    the tp8pp8 seam: the pairwise
      stage boundary (stream,
      pre_mix, top-k, this step's
      compressed and index-key rows
      into the next stage's cache
      replica), one program for
      both sides                    tp8pp8/pp_kernels            mk_model.run_recv / run_send
    the per-step index columns
      (window/compressed slot
      tables, group members)      ../mk_host.py                mk_model.StepIndices

The two megakernels borrow `blockquant`'s activation quantization (`_qdq_fp8`,
for the values written to the KV caches) and its bf16 `_gemm` (compressor,
indexer keys, the draft head) and nothing from any library: every other
operation is written out in `mk_mx` / `mk_moe` / `mk_common` / `mk_compress`.
Every megakernel module exists once per parallelization scheme (`kernels/tp64/`,
`kernels/tp8pp8/`): the schemes are separate paths and their kernels never share
source, so a change for one cannot alter the other (`mk_kernels.pkg(scheme)`).
Unit tests: `test/unit/test_layer.py`, `test/unit/test_draft.py` (N ranks
through `kharness_mp.py`); device probes of the hardware conventions the
kernels rely on: `test/unit/probe_mx.py`, `probe_dyn.py`, `probe_ti.py`.

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
