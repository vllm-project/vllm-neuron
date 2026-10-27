# SPDX-License-Identifier: Apache-2.0
"""The optimized `r1_idx` prefill block megakernel of the flat scheme (`layer_prefill_r1_idx_tp64`,
blocks 24, 28, 32, 36), with its own copies of the helper modules it calls: the optimized r1 prefill kernel (`../r1_prefill`, which carries the r2 prefill kernel's optimizations too)
with this archetype's sections of the generic body. `mk_kernels.layer` routes
(prefill, (1, 0, 1, 0, 0), tp64) here.
"""

from . import blockquant, mk_common, mk_compress, mk_moe, mk_mx, r1_idx_prefill  # noqa: F401
