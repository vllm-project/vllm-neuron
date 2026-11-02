# SPDX-License-Identifier: Apache-2.0
"""The optimized `r2_src_idx` decode block megakernel of the flat scheme (`layer_decode_r2_src_idx_tp64`,
blocks 2 and 8), with its own copies of the helper modules it calls: the optimized r2 decode kernel (`../r2_decode`)
with this archetype's sections of the generic body. `mk_kernels.layer` routes
(decode, (2, 1, 1, 0, 0), tp64) here.
"""

from . import blockquant, mk_common, mk_compress, mk_moe, mk_mx, r2_src_idx_decode  # noqa: F401
