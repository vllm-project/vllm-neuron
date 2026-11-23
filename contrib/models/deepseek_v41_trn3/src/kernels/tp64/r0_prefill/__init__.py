# SPDX-License-Identifier: Apache-2.0
"""The optimized `r0` prefill block megakernel of the flat scheme (`layer_prefill_r0_tp64`,
block 0), with its own copies of the helper modules it calls: the optimized r1 prefill kernel (`../r1_prefill`, which carries the r2 prefill kernel's optimizations too)
with this archetype's sections of the generic body. `mk_kernels.layer` routes
(prefill, (0, 0, 0, 0, 0), tp64) here.
"""

from . import blockquant, mk_common, mk_compress, mk_moe, mk_mx, r0_prefill  # noqa: F401
