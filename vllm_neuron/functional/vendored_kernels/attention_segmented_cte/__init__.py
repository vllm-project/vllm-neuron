# SPDX-License-Identifier: Apache-2.0
"""Vendored segmented prefill attention kernel with a fix for head_dim 256.

The nkilib ``attention_segmented_cte`` kernel gives wrong output when head_dim is
more than 128 and one K/V segment has more than ``8192 // num_d_tiles`` keys. For
head_dim 256, this is a Q chunk or a KV segment above 4096 tokens. The kernel
ignores the keys after position 4096 of the segment. This copy fixes the defect.

Provenance: copied from the nkilib package of the consumed image (nki-library
1.0.18316.0a0+8ac2e956):
  - attention_segmented_cte.py   (the ``attention_segmented_cte`` kernel)
  - fused_segmented_attention.py (the fused multi-head implementation)
  - attention_cte.py             (the attention core that both use)

The stable leaf utils (``nkilib.core.utils``) are imported from nkilib directly.

Changes against the source:
  - attention_cte.py: the attention core indexes the K/V tiles that the caller
    preloads in SBUF (``k_cache_sbuf`` / ``v_cache_sbuf``) by the tile position.
    Before, it used the index in the section, so section 1 used the tiles of
    section 0. The partial prior segment path runs into this defect.
  - attention_cte.py: ``_allocate_attention_psum_buffers`` holds the MM1 and MM2
    PSUM allocation. ``_allocate_attention_buffers`` calls it, so the output does
    not change for the shapes that were already correct.
  - fused_segmented_attention.py: ``_run_groups`` runs the Q-group loop once for
    each K/V section of the segment, with new PSUM tiles for each next section.
    Before, it ran only the first section.
  - attention_segmented_cte.py: a revision comment in the kernel function. The
    NKI cache key hashes only the source of that function. Change the revision
    when you change a helper, or a warm cache gives the old kernel.
  - All files: the relative ``..utils`` imports are ``nkilib.core.utils``
    imports, and ruff formats the files.

For the shapes that were correct before (for example head_dim 128, or head_dim
256 with a segment of 4096 or less), the output is bitwise equal to the source
kernel in the NKI simulator. The SBUF usage does not change.

SYNC: when the consumed nkilib image has the fix, delete this subpackage. Then
import ``attention_segmented_cte`` from ``nkilib.core.attention`` again in
``functional/attention/attention_segmented_cte.py``.
"""

from .attention_segmented_cte import attention_segmented_cte

__all__ = ["attention_segmented_cte"]
