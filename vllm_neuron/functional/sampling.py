# SPDX-License-Identifier: Apache-2.0
"""
Sampling functions for token generation.

This module provides Neuron-compatible sampling operations that avoid
unsupported ops like torch.softmax, torch.multinomial, and torch.cumsum.
"""

import torch
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed._functional_collectives import all_gather_tensor

from vllm_neuron.functional.argmax import argmax as distributed_argmax
from vllm_neuron.functional.cumsum import cumsum

# Epsilon for temperature comparison (matches vLLM)
_SAMPLING_EPS = 1e-5


def sample(
    logits: Tensor,
    temperature: int | float | Tensor = 1.0,
    top_k: int | Tensor = -1,
    top_p: float | Tensor = 1.0,
    deterministic: bool = False,
    all_greedy: bool = False,
    tp_group: ProcessGroup | None = None,
    max_top_k: int = 256,
    all2all_group: ProcessGroup | None = None,
    gather_group: ProcessGroup | None = None,
    tp_rank: Tensor | None = None,
    capture_topk: bool = False,
) -> Tensor:
    """
    Sample tokens from logits with per-request parameters.

    Supports batched inference where each request in the batch can have
    different sampling parameters. All operations are Neuron-compatible.

    When process_group is provided, the function expects the logits to be sharded along vocab dim and distributed among the process group.
    A distributed version of sampling will be run if process_group is provided.

    Args:
        logits: Logits tensor of shape [batch_size, vocab_size].
        temperature: Temperature for softmax scaling. Can be a scalar or
            tensor of shape [batch_size]. Zero means greedy sampling.
            Default: 1.0.
        top_k: Number of top tokens to consider. Can be a scalar or tensor
            of shape [batch_size]. Values <= 0 disable top-k filtering
            (vLLM uses -1, but 0 works identically). Default: 0.
        top_p: Cumulative probability threshold for nucleus sampling. Can be
            a scalar or tensor of shape [batch_size]. Must be in (0, 1].
            Set to 1.0 to consider all tokens. Default: 1.0.
        deterministic: If True, use a fixed random value (0.5) for multinomial
            sampling to produce reproducible results. Only affects non-greedy
            sampling. Default: False.
        all_greedy: If True, skip multinomial and return argmax directly.
            Caller should set this when all temperatures < _SAMPLING_EPS.
            Default: False.
        process_group: Optional process group for distributed argmax in
            tensor-parallel inference. When provided, logits MUST be sharded along vocab dimension (dim=-1) across ranks in the group
            Default: None.
        max_top_k: Maximum number of top tokens to consider across all requests
            in the batch. Should be max(256, max(top_k values)). Computed on CPU
            to avoid graph breaks. Default: 256.
        all2all_group: Process group for batch↔vocab redistribution. When provided,
            enables data-parallel sampling with batch sharding. Requires gather_group.
            Default: None.
        gather_group: Process group for gathering topk results. Required when
            all2all_group is provided. Default: None.
        tp_rank: TP rank as a scalar tensor (int32). Passed to distributed
            topk to avoid baking rank as a compilation constant. Default: None.

    Returns:
        Tensor of sampled token indices with shape [batch_size].

    Example:
        >>> import torch
        >>> from vllm_neuron.functional import sample
        >>> # Single request with greedy sampling (temperature=0)
        >>> logits = torch.tensor([[1.0, 2.0, 3.0]])
        >>> token = sample(logits, temperature=0.0, all_greedy=True)
        >>> # Batched requests with per-request parameters
        >>> logits = torch.randn(4, 1000)
        >>> temp = torch.tensor([0.0, 0.8, 0.9, 1.0])  # first is greedy
        >>> top_k = torch.tensor([0, 50, 40, 0], dtype=torch.int32)
        >>> tokens = sample(logits, temperature=temp, top_k=top_k)
    """
    batch_size = logits.shape[0]

    # Normalize scalar/tensor params to [batch_size] tensors
    temperature = _to_tensor(temperature, batch_size, logits)
    top_k = _to_tensor(top_k, batch_size, logits, dtype=torch.int32)
    top_p = _to_tensor(top_p, batch_size, logits)

    # Reshape for broadcasting: [batch_size, 1]
    top_k = top_k.unsqueeze(-1)
    top_p = top_p.unsqueeze(-1)

    # Fast path: all greedy (skip top-k filtering)
    if all_greedy:
        # Cast to int32 so that sampled token ids can be fed into embedding layer
        # (NKI kernels may return uint32 indices)
        return _argmax_sample(logits, tp_group).to(torch.int32)

    # Derive dp_degree from all2all_group if provided
    if all2all_group is not None:
        import torch.distributed as dist

        dp_degree = dist.get_world_size(all2all_group)
    else:
        dp_degree = 1

    # Top-k filtering (uses batch-sharded topk if DP enabled)
    logits, sorted_indices = _top_k_filter(
        logits,
        top_k,
        tp_group,
        max_top_k,
        dp_degree,
        all2all_group,
        gather_group,
        tp_rank,
        capture_topk,
    )

    # If using DP, slice remaining sampling params to match sharded batch
    if all2all_group is not None:
        sharded_batch_size = logits.shape[0]
        top_p = top_p[:sharded_batch_size, :]
        temperature = temperature[:sharded_batch_size]

    # Run the probability math (temperature, softmax, top-p, multinomial) in fp32.
    # top-k itself is a pure selection and runs in the native (bf16) dtype so the
    # bf16-only GpSIMD kernel can be selected; the fp32 promotion that used to live
    # inside topk() (legacy NKIv1 requirement) is applied HERE instead, so the
    # sampling distribution keeps its fp32 precision independent of the top-k dtype.
    logits = logits.to(torch.float32)

    # Temperature scaling (vLLM pattern: replace < eps with 1.0 to avoid div by zero)
    eps = torch.full_like(temperature, _SAMPLING_EPS)
    is_greedy = temperature < eps
    temp_safe = torch.where(is_greedy, torch.ones_like(temperature), temperature)
    logits = logits / temp_safe.unsqueeze(-1)

    # Softmax (manual implementation to avoid isfinite check on Neuron)
    exp_logits = torch.exp(logits - torch.max(logits, dim=-1, keepdim=True)[0])
    probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True)

    # Top-p filtering (uses sorted indices from top-k)
    probs = _top_p_filter(probs, top_p)

    # Sample: greedy for temp < eps, multinomial otherwise (vLLM pattern)
    greedy_sampled = torch.argmax(probs, dim=-1)
    random_sampled = _multinomial(probs, deterministic)
    sampled_idx = torch.where(is_greedy, greedy_sampled, random_sampled)
    # Keep the indirect load in range. Neuron needs an explicit bound for the
    # runtime index.
    sampled_idx = sampled_idx.clamp(min=0, max=sorted_indices.shape[-1] - 1)

    # Map back to original vocab indices
    result = (
        torch.gather(sorted_indices, -1, sampled_idx.unsqueeze(-1))
        .squeeze(-1)
        .to(torch.int32)
    )

    # Final gather if using DP (restore full batch on all ranks)
    if all2all_group is not None:
        result = all_gather_tensor(result, 0, group=all2all_group)

    return result


def _to_tensor(
    val: int | float | Tensor,
    size: int,
    ref: Tensor,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Convert scalar or singleton tensor to a tensor of given size.

    Uses ``ref.new_full`` so the result inherits ``ref``'s device and fakeness
    (important under dynamo tracing with FakeTensors). When ``dtype`` is None,
    the result inherits ``ref``'s dtype as well.
    """
    if isinstance(val, Tensor):
        if val.numel() == 1:
            return val.expand(size).contiguous()
        return val
    if dtype is None:
        return ref.new_full((size,), val)
    return ref.new_full((size,), val, dtype=dtype)


def _top_k_filter(
    logits: Tensor,
    k: Tensor,
    process_group: ProcessGroup | None,
    max_top_k: int,
    dp_degree: int,
    all2all_group: ProcessGroup | None,
    gather_group: ProcessGroup | None,
    rank: Tensor | None = None,
    capture_topk: bool = False,
) -> tuple[Tensor, Tensor]:
    """Apply top-k filtering with per-row k values.

    Uses batch-sharded topk when DP groups provided and batch divisible by dp_degree.
    The distributed/batch-sharded top-k uses the NKI kernel where it can run (see
    ``_select_topk_method``), else ``torch.topk``.

    Returns:
        Tuple of (filtered_logits, sorted_indices) where filtered_logits has values
        outside top-k set to -inf, and sorted_indices are global vocab indices.
    """
    vocab_size = logits.shape[-1]
    active_topk = min(max_top_k, vocab_size)

    # Use batch-sharded topk if DP enabled AND batch is divisible by DP degree
    if all2all_group is not None and logits.shape[0] % dp_degree == 0:
        from vllm_neuron.functional.topk import batch_sharded_topk

        sorted_logits, sorted_indices = batch_sharded_topk(
            logits,
            active_topk,
            dim=-1,
            dp_degree=dp_degree,
            all2all_group=all2all_group,
            gather_group=gather_group,
            rank=rank,
        )
        # After batch sharding, slice k to match sharded batch size
        k = k[: sorted_logits.shape[0], :]
    # Use distributed topk when process_group is provided (works on both HW and CPU)
    elif process_group is not None:
        from vllm_neuron.functional.topk import topk as neuron_topk

        sorted_logits, sorted_indices = neuron_topk(
            logits,
            active_topk,
            dim=-1,
            gather_dim=-1,
            process_group=process_group,
            rank=rank,
        )
    else:
        sorted_logits, sorted_indices = torch.topk(logits, active_topk, dim=-1)

    # Debug/validation hook: capture the top-k outputs AND the pre-top-k input
    # logits for offline comparison. No-op unless a tensor_capture config is
    # active; the guard folds to a constant at trace time when capture_topk=False.
    #
    # Capturing the INPUT logits (the per-rank vocab shard feeding this call) is
    # what lets a validator recompute torch.topk over the SAME logits the kernel
    # saw and compare in-process -- isolating ONLY the top-k selection. Comparing
    # two separate full-model runs toggled by VLLM_NEURON_DISABLE_NKI_KERNELS does
    # NOT isolate top-k: that switch disables every NKI kernel (attention, MoE,
    # rmsnorm, ...), so the two runs produce different upstream logits and their
    # top-k legitimately differs. Capture the input so the test can be sound.
    if capture_topk:
        from vllm_neuron.accuracy.tensor_capture import capture_tensor

        capture_tensor("topk.input_logits", logits)
        capture_tensor("topk.values", sorted_logits)
        capture_tensor("topk.indices", sorted_indices.to(torch.int32))

    # Per-row threshold: k-th largest value, clamped to active_topk
    # (handles case where k > active_topk gracefully)
    row_k_idx = torch.clamp(k - 1, min=0, max=active_topk - 1).long()
    thresholds = sorted_logits.gather(-1, row_k_idx)

    # Mask tokens below threshold (only where k > 0)
    mask = (sorted_logits < thresholds) & (k > 0)
    sorted_logits = sorted_logits.masked_fill_(mask, -3000.0)

    return sorted_logits, sorted_indices


def _top_p_filter(probs: Tensor, p: Tensor) -> Tensor:
    """Apply nucleus (top-p) filtering with per-row p values using pre-sorted indices."""
    # probs are already sorted from _top_k_filter
    sorted_probs = probs
    vocab_size = sorted_probs.shape[-1]

    # Cumsum via matmul with upper triangular matrix (Neuron-compatible)
    triu = torch.triu(probs.new_ones((vocab_size, vocab_size)))
    cumsum_sorted = sorted_probs @ triu

    # Mask where cumsum exceeds per-row p threshold
    sorted_mask = cumsum_sorted > p
    sorted_mask[..., 0] = False  # Always keep at least one token
    sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)

    # Renormalize and return sorted probs
    return sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)


def _multinomial(probs: Tensor, deterministic: bool = False) -> Tensor:
    """Sample from probability distribution using inverse transform sampling."""
    cdf = cumsum(probs, dim=-1)
    # Normalize CDF so final value is exactly 1.0. Cumsum of probs should sum to 1,
    # but floating-point accumulation error may cause slight deviation (e.g., 0.9999997).
    # Dividing by the last value ensures proper [0, 1] range for inverse transform sampling.
    cdf = cdf / cdf[..., -1:]
    cdf[..., -1:] = 1.0  # Avoid OOB from accumulation error

    # Inverse transform sampling
    if deterministic:
        # Fixed random value for reproducibility
        rand = probs.new_full(probs.shape[:-1], 0.5)
    else:
        rand = torch.rand(probs.shape[:-1], device=probs.device, dtype=probs.dtype)
    return torch.sum(rand.unsqueeze(-1) > cdf, dim=-1)


def _argmax_sample(logits: Tensor, process_group: ProcessGroup | None = None) -> Tensor:
    """Greedy sampling using argmax with optional distributed support."""
    if _can_use_distributed_argmax(process_group):
        return distributed_argmax(
            tensor=logits,
            dim=-1,
            gather_dim=-1,
            keepdim=False,
            process_group=process_group,
        )
    return torch.argmax(logits, dim=-1)


def _can_use_distributed_argmax(process_group: ProcessGroup | None) -> bool:
    """Check if distributed argmax kernel can be used."""
    return process_group is not None
