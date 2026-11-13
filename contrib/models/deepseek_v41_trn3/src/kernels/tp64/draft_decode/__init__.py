# SPDX-License-Identifier: Apache-2.0
"""The optimized DSpark drafter of the flat scheme's decode steps, with its own copies of the
helper modules it calls:

    draft_seq.draft_body       `draft_decode_tp64`, a step of one sequence (was `../draft_kernel`)
    draft_batched.draft_body   `draft_decode_batched_tp64`, every sequence of a step in one launch
                               (was `../../tp8pp8/draft_kernel`)
    layout.prepare             the decode drafter's weight layout: `main_proj` and the shared
                               expert's gate / up projections column-sharded over the 64 ranks
                               (`mk_model.draft_weights_decode` applies it once)

Produced in the standalone workspace ~/ds41_dspark_draft (recorded inputs / golden outputs of the
live tp64 server, 64-rank executor, AutoHelix loop; workspace commit 645012e). Bit-exact with the
original drafters on the recorded cases (one and ten sequences at 1k and 8k context: drafted
tokens, confidences and window-cache rows) and ~2.6-2.7x faster there. `mk_kernels.draft` /
`draft_batched` route the decode phase here (`DEEPSEEK_V41_OPT_KERNELS`: kind "draft_decode");
the prefill phase keeps the original kernels and weights.
"""

from . import blockquant, draft_batched, draft_seq, layout, mk_common, mk_compress, mk_moe, mk_mx  # noqa: F401
