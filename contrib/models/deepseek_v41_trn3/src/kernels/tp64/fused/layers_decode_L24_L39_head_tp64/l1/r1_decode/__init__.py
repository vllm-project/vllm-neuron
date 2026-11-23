# SPDX-License-Identifier: Apache-2.0
"""The optimized `r1` decode block megakernel of the flat scheme (`layer_decode_r1_tp64`, blocks
21-23, 25-27, 29-31, 33-35), with its own copies of the helper modules it calls.

Produced in the standalone workspace ~/ds41_r1_decode (recorded inputs / golden outputs of the
live tp64 server, 64-rank executor, AutoHelix loop; workspace commit 6fa3d3d), starting from the
optimized r2 prefill kernel (`../r2_prefill`). It is bit-exact with `../layer_kernel.layer_body`
specialized to (decode, r1, tp64) on the recorded cases (one and ten 6-row spans at 1k and 8k
context; at bs1 its all-reduces carry only the T real rows (`CC_SMALL_BYTES`), whose summation
order can differ from the generic 128-row one by an ulp) and ~2.1-2.7x faster.
`mk_kernels.layer` routes (decode, (1, 0, 0, 0, 0), tp64) here.
"""

from . import blockquant, mk_common, mk_compress, mk_moe, mk_mx, r1_decode  # noqa: F401
