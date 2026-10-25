# SPDX-License-Identifier: Apache-2.0
"""The optimized `r2` prefill block megakernel of the flat scheme (`layer_prefill_r2_tp64`,
blocks 3-7, 9-13, 15-19), with its own copies of the helper modules it calls.

Produced in the standalone workspace ~/ds41_r2_prefill (recorded inputs / golden outputs of
the live tp64 server, 64-rank executor, AutoHelix loop; workspace commit f5a7e57) from
`../layer_kernel.layer_body` specialized to (prefill, r2, tp64). It is bit-exact with that
body on the recorded cases and ~4-5x faster. The helpers here diverged from `../mk_*.py`,
which every other tp64 kernel keeps using, so they live in this package.
`mk_kernels.layer` routes (prefill, (2, 0, 0, 0, 0), tp64) here.
"""

from . import blockquant, mk_common, mk_compress, mk_moe, mk_mx, r2_prefill  # noqa: F401
