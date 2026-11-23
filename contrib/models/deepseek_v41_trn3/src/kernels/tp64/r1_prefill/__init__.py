# SPDX-License-Identifier: Apache-2.0
"""The optimized `r1` prefill block megakernel of the flat scheme (`layer_prefill_r1_tp64`,
blocks 21-23, 25-27, 29-31, 33-35), with its own copies of the helper modules it calls.

Produced in the standalone workspace ~/ds41_r1_prefill (recorded inputs / golden outputs of
the live tp64 server, 64-rank executor, AutoHelix loop; workspace commit de96a9b), starting
from the optimized r2 prefill kernel (`../r2_prefill`; r1 and r2 differ only in the compress
ratio). It is bit-exact with `../layer_kernel.layer_body` specialized to (prefill, r1, tp64)
on the recorded cases and ~4.5x faster. `mk_kernels.layer` routes
(prefill, (1, 0, 0, 0, 0), tp64) here.
"""

from . import blockquant, mk_common, mk_compress, mk_moe, mk_mx, r1_prefill  # noqa: F401
