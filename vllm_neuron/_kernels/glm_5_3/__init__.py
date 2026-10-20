# SPDX-License-Identifier: Apache-2.0
"""In-tree GLM-5.3 NKI kernels."""

from vllm_neuron._kernels.glm_5_3.rmsnorm_residual_prefill import (
    rmsnorm_residual_prefill,
)

__all__ = ["rmsnorm_residual_prefill"]
