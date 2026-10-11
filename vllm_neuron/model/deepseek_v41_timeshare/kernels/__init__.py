# SPDX-License-Identifier: Apache-2.0
"""NKI kernels for the DeepSeek-V4.1-Flash timeshare port.

**Everything in this directory is on the live path**, and `model._nki_entry`
imports exactly these eight:

* `blockquant` — the block-quantized fp8/fp4 GEMMs and the plain bf16 one.
  Every projection, every routed and shared expert, the router, the engram
  read-out and the unembedding go through it.
* `sparse_attn` — the attention score, softmax and weighted sum.
* `index_score` — the indexer's rectified, weight-combined score, which decides
  which compressed positions attention may read.
* `engram_gate` — the engram's normalized gate and its gated add.
* `hc_mixes` — the hyper-connection coefficients, including the 20-iteration
  Sinkhorn.
* `hc_expand`, `hc_collapse`, `hc_attn_out` — the hyper-connection expand, the
  collapse, and the sublayer output mix.

All eight take their shapes from their arguments, so they run on whatever
bucket the scheduler produces.

`legacy/` holds the 15 partition-specific kernels the optimization project
produced — faster, but each asserts a baked token count, window length and rank
count on entry, so none is callable from here. They are the reference the eight
above were derived from; `legacy/__init__.py` says which came from which.

Nothing is imported eagerly here: `@nki.jit` kernels pull in the NKI frontend,
and the `legacy/` ones also pull `nki.collectives`, which a single-rank run has
no use for.
"""
