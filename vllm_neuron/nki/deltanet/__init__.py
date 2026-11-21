# SPDX-License-Identifier: Apache-2.0
"""DeltaNet NKI kernels for the Qwen3.5 linear attention layers (Qwen3.6/3.8-27B)."""

from .fused import (
    deltanet_fused_chunked_fwd,
    deltanet_fused_chunked_fwd_multihead,
    _make_lower_mask,
    _make_lower_mask_diag,
    _make_identity,
)
from .recurrent import (
    deltanet_recurrent_step_batched,
)

__all__ = [
    "deltanet_fused_chunked_fwd",
    "deltanet_fused_chunked_fwd_multihead",
    "deltanet_recurrent_step_batched",
    "_make_lower_mask",
    "_make_lower_mask_diag",
    "_make_identity",
]
