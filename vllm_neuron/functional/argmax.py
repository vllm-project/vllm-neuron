# SPDX-License-Identifier: Apache-2.0
"""
Distributed argmax kernel for tensor-parallel inference.

This module provides a distributed argmax operation that works across
sharded tensors in a tensor-parallel setting.
"""

import logging
import math
from typing import Optional

import torch
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed._functional_collectives import all_gather_tensor
import nki
import nki.language as nl
from nkilib.core.max.cascaded_max import cascaded_max
from vllm_neuron import envs
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.hardware_config import get_logical_nc_config
from vllm_neuron.utils.neuron_utils import can_run_kernel

cascaded_max_jit = nki.jit()(cascaded_max)

logger = logging.getLogger(__name__)


def argmax(
    tensor: Tensor,
    dim: int,
    gather_dim: int,
    keepdim: bool = False,
    process_group: Optional[ProcessGroup] = None,
) -> Tensor:
    """
    Performs distributed argmax on sharded tensors using 2-step algorithm.

    This function implements a distributed argmax operation for tensor-parallel
    inference where tensors are sharded across multiple devices. It uses a
    2-step approach:

    1. **Local argmax**: Each rank computes argmax on its local shard using
       NKI cascaded_max kernel (when conditions are met) or torch.max fallback
    2. **Global argmax**: Results are gathered and global argmax is computed

    **Sharding Layout**:
    The input tensor is assumed to be uniformly sharded along `gather_dim` across
    all ranks in the process group. For example, with TP=2 and vocab_size=256:
    - Rank 0: logits[:, 0:128]   (first 128 vocab tokens)
    - Rank 1: logits[:, 128:256] (last 128 vocab tokens)

    **When to Use**:

    This function is intended for distributed argmax. It will fall back to torch.argmax when process_group is not provided.

    **Kernel**:
    In step 1 we compute the local maxs using either a nki kernel or torch.max.
    NKI cascaded_max kernel is used when: dim=-1, 2D/3D tensor, size>=128, and
    not in CPU mode. Otherwise falls back to torch.max.

    Args:
        tensor: Input tensor to perform argmax on. Must be uniformly sharded
            along `gather_dim` across all ranks in `process_group`.
        dim: Dimension along which to find argmax.
        gather_dim: Dimension the tensor is sharded on. When `dim == gather_dim`,
            indices are corrected for sharding offset.
        keepdim: Whether to keep the reduced dimension. Defaults to False.
        process_group: Process group for distributed operations.
            Must be provided for distributed execution.

    Returns:
        Tensor with global argmax indices across all shards. All ranks return
        the same result.

    Example:
        >>> import torch
        >>> import torch.distributed as dist
        >>> from vllm_neuron.functional.argmax import argmax
        >>>
        >>> # Tensor-parallel inference with TP=2, vocab_size=256
        >>> # Each rank has logits of shape (batch=1, vocab_per_rank=128)
        >>> logits = torch.randn(1, 128)  # Local shard on this rank
        >>> pg = dist.new_group([0, 1])
        >>>
        >>> # Compute global argmax across both ranks
        >>> token_id = argmax(logits, dim=1, gather_dim=1, process_group=pg)
        >>> # Returns shape (1,) with global token index in range [0, 256)
        >>> # Rank 0 indices are in [0, 128), Rank 1 indices are in [128, 256)
    """
    if process_group is None:
        raise ValueError("process_group must be provided for distributed argmax")

    # Normalize negative dims. torch 2.13's functional all_gather_tensor
    # (_maybe_view_chunk_cat) miscomputes the view shape for negative
    # gather_dim (shape[gather_dim + 1:] wraps around and re-appends the whole
    # shape), crashing Dynamo fake-tensor tracing. dim must move with
    # gather_dim so the `gather_dim == dim` sharding-offset checks below stay
    # consistent.
    dim = dim % tensor.dim()
    gather_dim = gather_dim % tensor.dim()

    tp_degree = torch.distributed.get_world_size(group=process_group)

    # Preserve the original torch path when the NKI kernel cannot run.
    if tp_degree == 1:
        if not _can_use_nki_max(tensor, dim):
            return torch.argmax(tensor, dim=dim, keepdim=keepdim)

        _, local_index = _compute_local_max(tensor.to(torch.float32), dim)
        # Read the index through a gather on the size-1 reduced dim. The gather
        # is a no-op, but without it the compiler fails at batch size 2 or more
        # with NCC_IDDT901: a DRAM-to-DRAM copy changes the kernel uint32 index
        # to int64. The TP > 1 path below has the same gather, so it compiles.
        local_index = torch.gather(local_index, dim, torch.zeros_like(local_index))
        if keepdim:
            return local_index
        return local_index.squeeze(dim)

    # TODO: Remove this upcast - NKI kernel and functional tests work with bfloat16, but E2E models show accuracy regression without it
    tensor = tensor.to(torch.float32)
    # Find local max values and indices
    local_value, local_index = _compute_local_max(tensor, dim)

    # Gather results from all ranks
    global_values = all_gather_tensor(local_value, gather_dim, group=process_group)
    global_indices = all_gather_tensor(local_index, gather_dim, group=process_group)

    # Correct indices for sharding offset when gather_dim == dim
    if gather_dim == dim:
        global_indices = _apply_sharding_offset(
            global_indices, dim, tensor.shape[gather_dim], tp_degree
        )

    # Find global argmax and extract final indices
    global_argmax = torch.argmax(global_values, dim=dim, keepdim=True)
    # Keep the indirect load in range. Neuron needs an explicit bound for the
    # runtime index.
    global_argmax = global_argmax.clamp(min=0, max=global_indices.shape[dim] - 1)
    final_indices = torch.gather(global_indices, dim, global_argmax)

    if not keepdim:
        return final_indices.squeeze(dim)
    return final_indices


def _compute_local_max(tensor: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    """Compute local max using the NKI kernel when possible, otherwise torch.max."""
    if _can_use_nki_max(tensor, dim):
        is_3d = len(tensor.shape) == 3
        input_tensor = tensor.squeeze(0) if is_3d else tensor

        cascaded_max_nki = wrap_nki(cascaded_max_jit)
        # Grid must match the runtime lnc so the kernel's program sharding matches
        # the cores the NEFF runs on; a mismatch leaves output shards unwritten.
        grid = 1 if envs.VLLM_NEURON_CPU_MODE else get_logical_nc_config()
        value, index = cascaded_max_nki[grid](input_tensor)
        # The NKI kernel returns uint32 indices. torch has no promotion rule for
        # uint32, so the dtype after a later add is backend-dependent: the XLA
        # uint32, so the dtype after a later add is backend-dependent.
        # The XLA backend gives int64; the native backend keeps uint32. gloo all_gather
        # also rejects uint32. Cast here to make the dtype the same as the
        # torch.max fallback below.
        index = index.to(torch.int64)
        # Restore dimension if squeezed
        if is_3d:
            value = value.unsqueeze(0)
            index = index.unsqueeze(0)
        return value, index
    else:
        return torch.max(tensor, dim=dim, keepdim=True)


def _can_use_nki_max(tensor: Tensor, dim: int) -> bool:
    """
    Check if we can use the NKI max kernel.

    Requirements:
    - NKI kernels can run (hardware or CPU simulator)
    - dim must be the last dimension (NKI kernel only supports this)
    - Tensor must be 2D or 3D with shape[0] == 1
    - Reduction dimension must have at least 128 elements
    - Partition dimension (shape[-2]) must be <= nl.tile_size.pmax * grid
    - The stage free size of the kernel must be <= _NKI_MAX_STAGE_FREE_SIZE
    """
    if not can_run_kernel(tensor):
        return False

    shape = tensor.shape
    num_dims = len(shape)

    # NKI kernel only works on last dimension
    if dim != num_dims - 1 and dim != -1:
        return False

    # Check tensor dimensionality
    if not (num_dims == 2 or (num_dims == 3 and shape[0] == 1)):
        return False

    # Check minimum reduction size
    if shape[dim] < 128:
        return False

    # Check maximum partition size. The kernel shards the partition dim (BxS)
    # across `grid` programs. Each program gets a maximum of pmax rows.
    grid = 1 if envs.VLLM_NEURON_CPU_MODE else get_logical_nc_config()
    if shape[-2] > nl.tile_size.pmax * grid:
        return False

    # Check the stage free size. The kernel folds the reduction dim into
    # n_stages partition stages. A large BxS gives fewer stages, so each stage
    # holds more elements. nc_find_index8 accepts a maximum of
    # _NKI_MAX_STAGE_FREE_SIZE elements per partition. This copies the stage
    # math of nkilib cascaded_max (CascadedMaxConfig).
    if _nki_max_stage_free_size(shape[-2], shape[-1], grid) > _NKI_MAX_STAGE_FREE_SIZE:
        return False

    return True


# Maximum free elements per partition that nisa.nc_find_index8 accepts.
_NKI_MAX_STAGE_FREE_SIZE = 16384


def _nki_max_stage_free_size(bxs: int, vocab_size: int, grid: int) -> int:
    """Return the per-partition free size that cascaded_max uses for one stage.

    Args:
        bxs: Partition dimension of the input (rows).
        vocab_size: Reduction dimension of the input.
        grid: Launch grid (LNC degree).

    Returns:
        The number of reduction elements in one partition of one stage.

    Example:
        >>> _nki_max_stage_free_size(48, 75968, 2)
        15194
        >>> _nki_max_stage_free_size(64, 75968, 2)
        18992
    """
    pmax = int(nl.tile_size.pmax)
    num_programs = 1 if bxs == 1 else grid
    per_program_bxs = math.ceil(bxs / num_programs)
    max_n_stages = pmax // per_program_bxs
    n_stages = max(1, min(max_n_stages, vocab_size // pmax))
    return math.ceil(vocab_size / n_stages)


def _apply_sharding_offset(
    indices: Tensor, dim: int, shard_size: int, tp_degree: int
) -> Tensor:
    """Apply offset to indices to account for tensor sharding."""
    offset_shape = [1] * len(indices.shape)
    offset_shape[dim] = tp_degree

    offset = torch.arange(0, shard_size * tp_degree, shard_size, device=indices.device)
    offset = offset.view(offset_shape)

    return indices + offset
