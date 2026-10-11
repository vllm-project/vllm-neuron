# SPDX-License-Identifier: Apache-2.0
"""NKI kernels for the DeepSeek-V4.1-Flash timeshare port.

Two populations live here and they are used differently:

* `blockquant.py`, `sparse_attn.py`, `hc_mixes.py` and the three generalized
  `hc_*` ones — shape-general kernels for all of this model's arithmetic.
  `blockquant` is the block-quantized fp8/fp4 GEMMs and the plain bf16 one, and
  `model.py` routes every projection, every routed and shared expert, the
  router, the engram read-out and the unembedding through them; `sparse_attn` is
  the attention score/softmax/weighted-sum; `hc_mixes` is the hyper-connection
  coefficients including the Sinkhorn. All of them take their shapes from their
  arguments, so they run on whatever the scheduler produces.

* `l2_attn.py`, `moe_short.py`, `engram_long.py`, … — 16 kernels generated
  against single partition traces during the device-optimization project that
  preceded this port. They are faster, but each bakes its token count, window
  length and rank count in as module constants and asserts them on entry
  (`moe_short` is 4-rank, 128-token, 96-expert-per-rank). Using one means
  parameterizing those first; `blockquant.py`'s decoders and GEMM pipeline are
  lifted from `l3_attn.py` so the arithmetic already agrees.

Nothing is imported eagerly here: `@nki.jit` kernels pull in the NKI frontend,
and the partition-specific ones also pull `nki.collectives`, which a single-rank
run has no use for.
"""
