# SPDX-License-Identifier: Apache-2.0
"""The optimized `r0_e` decode block megakernel of the flat scheme (`layer_decode_r0_e_tp64`,
block 1), with its own copies of the helper modules it calls: the optimized r2 decode kernel (`../r2_decode`)
with this archetype's sections of the generic body. `mk_kernels.layer` routes
(decode, (0, 0, 0, 1, 0), tp64) here.
"""

from . import blockquant, mk_common, mk_compress, mk_moe, mk_mx, r0_e_decode  # noqa: F401
