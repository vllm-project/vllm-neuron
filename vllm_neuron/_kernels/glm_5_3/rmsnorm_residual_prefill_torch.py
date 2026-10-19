# SPDX-License-Identifier: Apache-2.0
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch reference for the GLM-5.3 RMSNorm residual prefill kernel."""

from typing import Dict, Optional

import torch


def rmsnorm_residual_prefill_torch_ref(
    hidden: torch.Tensor,
    gamma: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    hidden_actual: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Compute the Torch reference for :func:`rmsnorm_residual_prefill`.

    The output keys match the ``output_tensor_descriptor`` that
    ``UnitTestFramework`` requires. The function flattens a ``[B, S, H]`` input
    to ``[B*S, H]`` to match the kernel output.

    Args:
        hidden: Input hidden states with shape ``[T, H]`` or ``[B, S, H]``.
        gamma: Scale weights with shape ``[1, H]``.
        residual: Optional residual to add before normalization.
        eps: Epsilon for numerical stability.
        hidden_actual: True hidden width when ``H`` includes padding.

    Returns:
        A dictionary with the normalized ``"out"`` tensor. The dictionary also
        contains ``"residual_out"`` when ``residual`` is present.

    Example:
        >>> hidden = torch.ones((2, 4), dtype=torch.bfloat16)
        >>> gamma = torch.ones((1, 4), dtype=torch.bfloat16)
        >>> result = rmsnorm_residual_prefill_torch_ref(hidden, gamma)
        >>> tuple(result["out"].shape)
        (2, 4)
    """
    x = hidden.float() if residual is None else hidden.float() + residual.float()
    n = x.shape[-1] if hidden_actual is None else hidden_actual
    rms = torch.sqrt((x * x).sum(dim=-1, keepdim=True) / n + eps)
    out = {"out": ((x / rms) * gamma.float()).reshape(-1, x.shape[-1])}
    if residual is not None:
        out["residual_out"] = x.reshape(-1, x.shape[-1])
    return out
