# SPDX-License-Identifier: Apache-2.0
"""NKI kernels for the DeepSeek-V4.1-Flash timeshare port.

**Everything in this directory is on the live path except `swiglu.py`**, which
is a measured kernel held back for a stated reason (see the end of this
docstring and `model._swiglu_expert`). `model._nki_entry` imports exactly these
ten:

* `blockquant` — the block-quantized fp8/fp4 GEMMs and the plain bf16 one.
  Every projection, every routed and shared expert, the router, the engram
  read-out and the unembedding go through it.
* `sparse_attn` — the attention score, softmax and weighted sum.
* `index_score` — the indexer's rectified, weight-combined score, which decides
  which compressed positions attention may read.
* `compress_pool` — the KV compressor's learned softmax pooling over the
  `compress_ratio` members of a group.
* `rmsnorm` — every RMSNorm in the model, with a row-count-invariant reduction.
* `engram_gate` — the engram's normalized gate and its gated add.
* `hc_mixes` — the hyper-connection coefficients, including the 20-iteration
  Sinkhorn.
* `hc_expand`, `hc_collapse`, `hc_attn_out` — the hyper-connection expand, the
  collapse, and the sublayer output mix.

All ten take their shapes from their arguments, so they run on whatever bucket
the scheduler produces.

Where the heavy arithmetic runs, module by module
-------------------------------------------------
The map below is the whole answer to "does this module reach a kernel, or a
torch subgraph". Left to right: the architectural module, the kernel that
computes it, the call site, and the kernel under `legacy/` whose arithmetic it
carries (see `legacy/__init__.py` for why that directory is not called
directly). Line numbers are from the revision that added this table.

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

What is deliberately *not* in a kernel, and why that is the line the goal draws
("Torch stays on sampling, glue, reshapes and control flow"):

* **sampling** — `sample_tokens`, the rejection sampler, the greedy argmax.
* **gathers and scatters** — `index_select` into the paged KV, the engram's
  24-row-per-token read off the 94.6 GiB mmap, `index_add` into the MoE's fp32
  accumulator, `slot_mapping` writes. Data movement, not arithmetic;
  `sparse_attn.py`'s module docstring draws the same line for the gathered slot
  list, which is the largest of them.
* **reshapes, dtype casts and masks** — including every `.float()` that widens
  a kernel's bf16 output before the next kernel narrows it again.
* **routing control flow** — `MoE._route_plan` decides *which* experts a step
  touches, on the host, which is what makes the timeshare schedule possible at
  all. The arithmetic it is planning (the gate GEMM, the top-k, the experts) is
  all in kernels.
* **the SwiGLU activation inside an expert** (`_swiglu_expert`: two clamps, a
  `silu` and two multiplies over `[rows, 2304]`, between the w1/w3 and w2
  GEMMs). This is the one piece of expert arithmetic still expressed in torch,
  and it is the one entry in this list that is *not* settled.
  **`swiglu.py` in this directory is the kernel that replaces it.** It is
  `legacy/moe_short._swiglu` parameterized, it was wired in, it passed the
  gate's verifier at four ranks, and it was then taken back out: it differs
  from the torch expression by one bf16 step on about 1 value in 100,000, and
  across 280 expert calls a forward that was enough to flip a near-tied EOS and
  to break the token-for-token agreement between speculation on and speculation
  off. The residual difference is **not** a defect to fix — two equally valid
  fp32 formulations of `silu` disagree in the last bit on 22% of values — so
  what has to change first is that speculation's exactness is empirical rather
  than structural. `_swiglu_expert`'s docstring and
  `.autohelix/notes/iter-13.md` §3b-bis carry the measurements. So `swiglu.py`
  is *not* an unreached kernel that nobody noticed — it is a measured one held
  back by a specific number.

`legacy/` holds the 15 partition-specific kernels the optimization project
produced — faster, but each is a *whole module at one baked sequence length*
(`SEQ = 8192`, 4 ranks, 96 experts a rank, on a 16-device floorplan), so none is
callable from here. They are the reference the ten above were derived from, and
`legacy/__init__.py` holds the kernel-by-kernel disposition.

Nothing is imported eagerly here: `@nki.jit` kernels pull in the NKI frontend,
and the `legacy/` ones also pull `nki.collectives`.
"""
