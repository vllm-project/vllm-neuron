# SPDX-License-Identifier: Apache-2.0
"""The optimized `r2` decode block megakernel of the flat scheme (`layer_decode_r2_tp64`, blocks
3-7, 9-13, 15-19), with its own copies of the helper modules it calls.

Produced in the standalone workspace ~/ds41_r2_decode (recorded inputs / golden outputs of the
live tp64 server, 64-rank executor, AutoHelix loop; workspace commit 7850658), starting from the
optimized r2 prefill kernel (`../r2_prefill`). It is bit-exact with `../layer_kernel.layer_body`
specialized to (decode, r2, tp64) on the recorded cases (one and ten 6-row spans at 1k and 8k
context; at bs1 its all-reduces carry only the T real rows (`CC2_T_ROWS_MAX`), whose summation
order can differ from the generic 128-row one by an ulp) and ~2-2.6x faster. `mk_kernels.layer`
routes (decode, (2, 0, 0, 0, 0), tp64) here.
"""

from . import blockquant, mk_common, mk_compress, mk_moe, mk_mx, r2_decode  # noqa: F401
