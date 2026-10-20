# SPDX-License-Identifier: Apache-2.0
"""In-tree MLP NKI kernels."""

# Export only the kernel entry point. The torch reference
# (qwen3_32b_mxfp8_mlp_tkg_torch_ref) is test-only: it imports nkilib at call
# time, which the wheel does not ship. The kernel test imports it directly from
# the kernel module, so it does not need to be a public API here.
from vllm_neuron._kernels.mlp.qwen3_32b_mxfp8_mlp_tkg import (
    qwen3_32b_mxfp8_mlp_tkg,
)

__all__ = [
    "qwen3_32b_mxfp8_mlp_tkg",
]
