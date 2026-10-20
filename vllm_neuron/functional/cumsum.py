# SPDX-License-Identifier: Apache-2.0

import logging

import torch
from torch import Tensor

import nki
from nkilib.core.cumsum.cumsum import cumsum as nkilib_cumsum

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron import envs
from vllm_neuron.utils.hardware_config import get_logical_nc_config
from vllm_neuron.utils.neuron_utils import can_run_kernel

cumsum_jit = nki.jit()(nkilib_cumsum)

logger = logging.getLogger(__name__)


def cumsum(
    tensor: Tensor,
    dim: int = -1,
) -> Tensor:
    """
    Compute cumulative sum along the specified dimension.

    This function uses an NKI kernel on Neuron devices for optimized
    performance, and falls back to a matmul-based implementation on CPU.
    The matmul-based approach has been proven to be more performant than
    torch.cumsum.

    The kernel launches at the process LNC degree; a hardcoded grid of 2 is
    rejected at LNC=1 with ``NCC_ENKI003``.

    Args:
        tensor: Input tensor to compute cumsum on. Must be 2D.
        dim: Dimension along which to compute cumsum. Currently only
            supports dim=-1 (last dimension). Defaults to -1.

    Returns:
        Tensor with cumulative sum along the specified dimension.
        Same shape and dtype as input.

    Raises:
        ValueError: If tensor is not 2D, or if dim is not -1.

    Example:
        >>> import torch
        >>> from vllm_neuron.functional.cumsum import cumsum
        >>>
        >>> # Simple cumsum
        >>> probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        >>> cdf = cumsum(probs, dim=-1)
        >>> # Returns: [[0.1, 0.3, 0.6, 1.0]]
    """
    if tensor.ndim != 2:
        raise ValueError(
            f"cumsum only supports 2D tensors, got {tensor.ndim}D tensor with shape {tensor.shape}"
        )

    if dim != -1 and dim != tensor.ndim - 1:
        raise ValueError(f"cumsum only supports dim=-1, got dim={dim}")

    if _can_use_nki_cumsum(tensor, dim):
        cumsum_nki = wrap_nki(cumsum_jit)
        # Grid must match the runtime lnc so the kernel's program sharding matches
        # the cores the NEFF runs on; a mismatch leaves output shards unwritten.
        grid = 1 if envs.VLLM_NEURON_CPU_MODE else get_logical_nc_config()
        return cumsum_nki[grid](x=tensor, axis=dim)
    else:
        return _cumsum_matmul(tensor)


def _cumsum_matmul(tensor: Tensor) -> Tensor:
    """
    Compute cumsum using matmul with an upper triangular matrix.

    This approach has been proven to be more performant than torch.cumsum.
    Used as the CPU fallback implementation.
    """
    size = tensor.shape[-1]
    triu = torch.triu(torch.ones(size, size, dtype=tensor.dtype, device=tensor.device))
    return tensor @ triu


def _can_use_nki_cumsum(tensor: Tensor, dim: int) -> bool:
    """
    Check if we can use the NKI cumsum kernel.

    Requirements:
    - Simulator available (in CPU mode) or on Neuron device
    - dim must be the last dimension (NKI kernel only supports this)
    """
    if not can_run_kernel(tensor):
        return False

    # NKI kernel only works on last dimension
    if dim != -1 and dim != tensor.ndim - 1:
        return False

    return True
