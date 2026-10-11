# SPDX-License-Identifier: Apache-2.0
"""Partition-specific NKI kernels — reference material, not the live path.

These 15 kernels were generated against **single partition traces** during the
device-optimization project that preceded this port (the one recorded at
`OswaldHe/DeepSeek-V4.1-Flash-Trainium`). They are faster than anything in the
parent directory, and none of them can be called from `model.py` as written,
because each one bakes its token count, window length, expert count and rank
count in as module-level constants and asserts them on entry — `moe_short`, for
instance, is 4-rank, 128-token, 96-experts-per-rank, on a 16-device floorplan.
This port serves whatever bucket the scheduler picks, on one device.

They live in their own package to make that status structural rather than a
claim in a docstring. `kernels/` now contains exactly the kernels `model.py`
reaches — `blockquant`, `sparse_attn`, `index_score`, `engram_gate`,
`hc_mixes`, `hc_expand`, `hc_collapse`, `hc_attn_out` — and nothing that is
present but unreachable.

**They are the starting point for the shape-general ones, and that is how they
have been used.** `blockquant.py`'s fp8/fp4 decoders and GEMM pipeline are
lifted from `l3_attn.py`; `engram_gate.py` is `engram_short.py`'s normalized
gate with the token count, `dim` and `hc` taken from the operands and the
collectives removed; `hc_mixes.py`'s Sinkhorn is `hc_attn_in_short.py`'s.
Parameterizing one of these and moving it up a directory is the intended way to
make a heavy op faster, and `.autohelix/notes/iter-5.md` records the method —
`nki.simulate(kernel)(...)` checks a port against the torch expression it
replaces on the CPU in seconds.

Nothing is imported here: `@nki.jit` kernels pull in the NKI frontend, and the
partition-specific ones also pull `nki.collectives`, which a single-rank run
has no use for.
"""
